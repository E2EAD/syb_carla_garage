import torch
from torch import nn
import numpy as np
from config import GlobalConfig
import json
import torch.nn.functional as F
from torch import vmap
from utils import print_data_info

class TrajectoryExpert(nn.Module):
    def __init__(self, input_dim=256, output_dim=2):
        super(TrajectoryExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ScoreExpert(nn.Module):
    def __init__(self, input_dim=2560):  # Changed: 10 * 256 = 2560
        super(ScoreExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class PlanningTrajectoryDecoder(nn.Module):
    """
    Planning Transformer Decoder using trajectory anchors as queries, integrated MoE
    Now with time-aware anchor embedding similar to the first code
    
    Input Shapes:
    - encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings from encoder
    
    Output Shapes (predict method):
    - selected_trajectories: (batch_size, 20) - [x1,y1,x2,y2,...,x10,y10] coordinates, using nearest anchor to predict
    - all_trajectories: (num_anchors, batch_size, 20) - All predicted trajectories
    - scores: (num_anchors, batch_size) - Confidence scores for each trajectory
    
    Note: 
    - Each trajectory represents 10 future waypoints
    - Coordinates are in ego-vehicle frame (meters)
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Args:
            cfg: Configuration object containing model parameters
            num_anchors: Number of trajectory anchors to use (default: 10)
        """
        super().__init__()
        self.cfg = cfg
        
        # Create diverse trajectory anchors (num_anchors, 20)
        self._create_anchors(cfg.prior_traj_path)
        
        # ========== MODIFIED: Time encoding for waypoints ==========
        # Time encoding: maps each waypoint to 256-dim vector
        self.time_encoding = nn.Sequential(
            nn.Linear(2, 64),  # Each waypoint has (x,y)
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256)
        )
        
        # Positional encoding for time steps
        self.pos_encoding = nn.Parameter(torch.zeros(1, 10, 256))
        
        # Anchor embedding layer (now operates on (10, 256) shaped trajectory)
        self.anchor_embed = nn.Sequential(
            nn.Linear(256, cfg.tf_de_dim),
            nn.LayerNorm(cfg.tf_de_dim),
            nn.ReLU(),
            nn.Dropout(cfg.tf_de_dropout)
        )
        self.pos_drop = nn.Dropout(cfg.tf_de_dropout)
        # ========== END MODIFIED ==========
        
        # Transformer decoder
        tf_layer = nn.TransformerDecoderLayer(
            d_model=cfg.tf_de_dim, 
            nhead=cfg.tf_de_heads,
            batch_first=False
        )
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=cfg.tf_de_layers)
        
        # Initialize experts (trajectory experts and score experts)
        self.trajectory_experts = nn.ModuleList()
        self.score_experts = nn.ModuleList()
        self._build_experts(self.num_anchors)
        
        self._init_weights()

    def _build_experts(self, num_experts):
        """构建指定数量的 trajectory experts 和 score experts"""
        # ========== MODIFIED: ScoreExpert input dimension ==========
        # Trajectory expert processes per-time-step features (256 dim)
        # Score expert processes all time steps concatenated (10 * 256 = 2560 dim)
        self.trajectory_experts = nn.ModuleList([
            TrajectoryExpert(self.cfg.tf_de_dim, 2)
            for _ in range(num_experts)
        ])
        
        self.score_experts = nn.ModuleList([
            ScoreExpert(self.cfg.tf_de_dim * 10)  # 10 time steps concatenated
            for _ in range(num_experts)
        ])
        # ========== END MODIFIED ==========
        
        self.num_anchors = num_experts
    
    def _create_anchors(self, anchor_path=None):
        """
        Creates diverse trajectory anchors from file or generates default ones.
        Anchors are registered as non-trainable buffers.
        """
        if anchor_path is None:
            print('no anchor path, generate default anchors.')
            return

        else:
            with open(anchor_path, 'r') as f:
                data = json.load(f)

            # Extract cluster IDs and mu values
            cluster_ids = []
            mu_list = []
            for entry in data:
                cluster_ids.append(entry['cluster_id'])
                mu_list.append(entry['mu'][:20])  # First 20 elements for 10 waypoints
                # break  # for debug
            
            anchors_tensor = torch.tensor(mu_list, dtype=torch.float32)
            self.num_anchors = anchors_tensor.size(0)
            print(f'read anchors from {anchor_path}')

        # Register buffers
        self.register_buffer('anchors', anchors_tensor)
        # Store cluster IDs as a tensor for easy device management
        self.register_buffer('cluster_ids', torch.tensor(cluster_ids, dtype=torch.long))
        
        print(f'anchors shape: {self.anchors.shape}')
        print(f'cluster_ids: {self.cluster_ids.tolist()}')
        print(f"Created {self.num_anchors} anchor trajectories")

    def _init_weights(self):
        """Initialize network weights"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # ========== MODIFIED: Initialize position encoding ==========
        # Initialize position encoding
        nn.init.normal_(self.pos_encoding, std=0.01)
        # ========== END MODIFIED ==========
        
        # Expert initialization 
        for expert in self.trajectory_experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Score expert initialization
        for expert in self.score_experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ========== MODIFIED: New method for processing anchors with time encoding ==========
    def _process_anchors_with_time_encoding(self, anchors_expanded):
        """
        Process anchors with time encoding to get (num_anchors, batch_size, 10, tf_de_dim) shaped queries.
        
        Args:
            anchors_expanded: (num_anchors, batch_size, 20) shaped anchors
            
        Returns:
            anchor_queries: (num_anchors, batch_size, 10, tf_de_dim) shaped queries
        """
        num_anchors, batch_size, _ = anchors_expanded.shape
        
        # Reshape anchors to (num_anchors * batch_size, 10, 2)
        anchors_reshaped = anchors_expanded.reshape(-1, 10, 2)
        
        # Apply time encoding to each waypoint
        # (num_anchors * batch_size, 10, 2) -> (num_anchors * batch_size, 10, 256)
        time_encoded = self.time_encoding(anchors_reshaped)
        
        # Add positional encoding for time steps
        time_encoded = time_encoded + self.pos_encoding
        
        # Apply anchor embedding to each time step
        # Reshape for linear layer: flatten time dimension
        time_encoded_flat = time_encoded.reshape(-1, 256)
        anchor_embedded = self.anchor_embed(time_encoded_flat)
        anchor_embedded = anchor_embedded.reshape(num_anchors * batch_size, 10, -1)
        
        # # Reshape back to (num_anchors, batch_size, 10, tf_de_dim)
        # anchor_queries = anchor_embedded.reshape(num_anchors, batch_size, 10, -1)

        # Reshape back to (num_anchors, batch_size, 10, tf_de_dim)
        anchor_queries = anchor_embedded.reshape(num_anchors, batch_size, 10, -1)
        
        return anchor_queries
    # ========== END MODIFIED ==========

    def forward(self, encoder_out):
        """
        Forward pass for training (simplified for inference focus)
        
        Args:
            encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings
            
        Returns:
            pred_trajectories: (num_anchors, batch_size, 20) - Predicted trajectories
            scores: (num_anchors, batch_size) - Confidence scores
        """
        batch_size = encoder_out.size(0)
        seq_len = encoder_out.size(1)
        
        # Transpose for transformer: (seq_len, batch_size, dim)
        memory = encoder_out.permute(1, 0, 2)
        
        # ========== MODIFIED: Prepare anchor queries with time encoding ==========
        # Prepare anchor queries: (num_anchors, batch_size, 20) -> (num_anchors, batch_size, 10, tf_de_dim)
        anchors_expanded = self.anchors.unsqueeze(1).expand(-1, batch_size, -1)
        anchor_queries = self._process_anchors_with_time_encoding(anchors_expanded)
        
        # Process each time step through transformer decoder
        num_anchors, batch_size, num_timesteps, dim = anchor_queries.shape
        
        # Reshape queries for transformer: (num_timesteps, num_anchors * batch_size, dim)
        queries_reshaped = anchor_queries.permute(2, 0, 1, 3).reshape(num_timesteps, num_anchors * batch_size, dim)
        # queries_reshaped = anchor_queries.permute(2, 0, 1, 3).reshape(num_timesteps*num_anchors, batch_size, dim)

        # Expand memory for all anchors
        memory_expanded = memory.repeat(1, num_anchors, 1)
        # memory_expanded = memory
        
        # Apply transformer decoder
        decoder_out = self.tf_decoder(  # (num_timesteps*num_anchors, batch_size, dim)
            tgt=queries_reshaped,
            memory=memory_expanded
        )
        
        # Reshape decoder output back: (num_timesteps, num_anchors, batch_size, dim)
        decoder_out_reshaped = decoder_out.reshape(num_timesteps, num_anchors, batch_size, dim)
        
        # Transpose to (num_anchors, batch_size, num_timesteps, dim)
        decoder_out_transposed = decoder_out_reshaped.permute(1, 2, 0, 3)
        
        # Prepare data for trajectory experts (per time-step processing)
        # Reshape to (num_anchors * batch_size * num_timesteps, dim)
        decoder_out_flat = decoder_out_transposed.reshape(-1, dim)
        # ========== END MODIFIED ==========
        
        # Process trajectory experts with chunking
        chunk_size = 200  # Adjust based on GPU memory
        traj_offsets = []
        
        for start_idx in range(0, num_anchors, chunk_size):
            end_idx = min(start_idx + chunk_size, num_anchors)
            # ========== MODIFIED: Chunk input for trajectory experts ==========
            # Each expert processes data for all time steps of its anchor
            chunk_input = decoder_out_flat[start_idx * batch_size * num_timesteps : end_idx * batch_size * num_timesteps]
            
            chunk_offsets = []
            for local_idx, expert in enumerate(self.trajectory_experts[start_idx:end_idx]):
                data_start = local_idx * batch_size * num_timesteps
                data_end = (local_idx + 1) * batch_size * num_timesteps
                expert_data = chunk_input[data_start:data_end]
                
                # ========== MODIFIED: Process each time step independently ==========
                # Each trajectory expert processes (batch_size * num_timesteps, dim) and outputs offsets
                # Then we reshape to get per-anchor offsets
                # print_data_info(expert_data)  # torch.Size([20, 256])
                offset = expert(expert_data)
                # print_data_info(offset)  # torch.Size([20, 2])
                offset = offset.reshape(batch_size, num_timesteps, 2)
                offset = offset.reshape(batch_size, -1)  # (batch_size, 20)
                chunk_offsets.append(offset)
            
            chunk_offsets = torch.stack(chunk_offsets, dim=0)
            traj_offsets.append(chunk_offsets)

        traj_offsets = torch.cat(traj_offsets, dim=0)
        pred_trajectories = anchors_expanded + traj_offsets

        # ========== MODIFIED: Process score experts ==========
        # Score experts process all time steps concatenated
        # Prepare input: (num_anchors, batch_size, num_timesteps * dim)
        # decoder_for_score = decoder_out_transposed.reshape(num_anchors, batch_size, -1)
        # decoder_for_score = decoder_out_transposed.reshape(num_anchors, batch_size, -1)
        # decoder_for_score_flat = decoder_for_score.reshape(-1, num_timesteps * dim)

        decoder_for_score = memory_expanded.permute(1, 0, 2)
        # print_data_info(decoder_for_score)
        decoder_for_score_flat = decoder_for_score.reshape(-1, seq_len * dim)
        # print_data_info(decoder_for_score_flat)

        scores = []
        
        for start_idx in range(0, num_anchors, chunk_size):
            end_idx = min(start_idx + chunk_size, num_anchors)
            chunk_input = decoder_for_score_flat[start_idx * batch_size : end_idx * batch_size]

            chunk_scores = []
            for local_idx, expert in enumerate(self.score_experts[start_idx:end_idx]):
                data_start = local_idx * batch_size
                data_end = (local_idx + 1) * batch_size
                expert_data = chunk_input[data_start:data_end]
                score = expert(expert_data)
                chunk_scores.append(score)

            chunk_scores = torch.stack(chunk_scores, dim=0)
            scores.append(chunk_scores)

        scores = torch.cat(scores, dim=0).squeeze(-1)  # (num_anchors, batch_size)
        # ========== END MODIFIED ==========
        
        return pred_trajectories, scores
    
    def load_state_dict_with_resize(self, state_dict: dict, strict: bool = True):
        """
        Load with cluster_id based matching: load both trajectory and score experts
        """
        device = next(self.parameters()).device
        
        # Filter out anchor parameters
        local_state = {}
        trajectory_expert_params = {}
        score_expert_params = {}
        
        for key, value in state_dict.items():
            if key.endswith('.anchors') or 'anchors' in key:
                print(f"⚠️ Skipping anchor data: {key} (shape: {value.shape})")
                continue
            if key.endswith('.cluster_ids') or 'cluster_ids' in key:
                print(f"⚠️ Skipping cluster_ids data: {key}")
                continue
            if key.startswith('trajectory_experts.') or '.trajectory_experts.' in key:
                trajectory_expert_params[key] = value
            elif key.startswith('score_experts.') or '.score_experts.' in key:
                score_expert_params[key] = value
            elif key.startswith('score_head.') or '.score_head.' in key:
                # Convert old score_head to new score_experts format
                score_expert_params[key.replace('score_head', 'score_experts.0')] = value
                print(f"🔄 Converting old score_head parameter: {key}")
            else:
                local_state[key] = value
        
        print(f"Found {len(trajectory_expert_params)} trajectory expert parameters in checkpoint")
        print(f"Found {len(score_expert_params)} score expert parameters in checkpoint")
        
        # Extract cluster_ids from checkpoint if available
        ckpt_cluster_ids = None
        for key, value in state_dict.items():
            if key.endswith('.cluster_ids') or 'cluster_ids' in key:
                ckpt_cluster_ids = value.cpu().numpy().tolist()
                print('Got ckpt_cluster_ids.')
                break
        
        if ckpt_cluster_ids is None:
            print("///// Warning: No cluster_ids found in checkpoint, falling back to index-based matching /////")
            return self._load_fallback(
                trajectory_expert_params, 
                score_expert_params, 
                local_state, 
                strict, 
                device
            )
        
        # Get current cluster_ids
        current_cluster_ids = self.cluster_ids.cpu().numpy().tolist()
        
        print(f"[PlanningTrajectoryDecoder] Loading by cluster_id: "
            f"checkpoint has {len(ckpt_cluster_ids)} clusters, current has {len(current_cluster_ids)} clusters")
        print(f"Checkpoint cluster_ids: {ckpt_cluster_ids}")
        print(f"Current cluster_ids: {current_cluster_ids}")

        # Build mapping from cluster_id to expert index in checkpoint
        ckpt_trajectory_cluster_to_params = self._build_cluster_to_params(
            trajectory_expert_params, ckpt_cluster_ids, 'trajectory_experts'
        )
        
        ckpt_score_cluster_to_params = self._build_cluster_to_params(
            score_expert_params, ckpt_cluster_ids, 'score_experts'
        )

        # Build new trajectory experts based on current cluster_ids
        new_trajectory_experts = nn.ModuleList()
        new_score_experts = nn.ModuleList()
        
        trajectory_loaded_count = 0
        score_loaded_count = 0
        trajectory_new_count = 0
        score_new_count = 0
        
        # Compute average parameters for new experts
        trajectory_avg_state = self._compute_average_expert_state(ckpt_trajectory_cluster_to_params)
        score_avg_state = self._compute_average_expert_state(ckpt_score_cluster_to_params)
        
        for i, current_cluster_id in enumerate(current_cluster_ids):
            # Load/Initialize trajectory expert
            if current_cluster_id in ckpt_trajectory_cluster_to_params:
                expert = TrajectoryExpert(self.cfg.tf_de_dim, 2).to(device)
                expert_sd = ckpt_trajectory_cluster_to_params[current_cluster_id]
                
                try:
                    expert.load_state_dict(expert_sd, strict=True)
                    new_trajectory_experts.append(expert)
                    trajectory_loaded_count += 1
                    print(f"✅ Loaded trajectory expert for cluster_id {current_cluster_id}")
                except Exception as e:
                    print(f"❌ Failed to load trajectory expert for cluster_id {current_cluster_id}: {e}")
                    expert = self._create_trajectory_expert_with_avg(trajectory_avg_state, device)
                    new_trajectory_experts.append(expert)
                    trajectory_new_count += 1
            else:
                expert = self._create_trajectory_expert_with_avg(trajectory_avg_state, device)
                new_trajectory_experts.append(expert)
                trajectory_new_count += 1
                print(f"🆕 Created new trajectory expert for cluster_id {current_cluster_id}")
            
            # Load/Initialize score expert
            if current_cluster_id in ckpt_score_cluster_to_params:
                # ========== MODIFIED: ScoreExpert input dimension ==========
                expert = ScoreExpert(self.cfg.tf_de_dim * 10).to(device)
                expert_sd = ckpt_score_cluster_to_params[current_cluster_id]
                
                try:
                    expert.load_state_dict(expert_sd, strict=True)
                    new_score_experts.append(expert)
                    score_loaded_count += 1
                    print(f"✅ Loaded score expert for cluster_id {current_cluster_id}")
                except Exception as e:
                    print(f"❌ Failed to load score expert for cluster_id {current_cluster_id}: {e}")
                    expert = self._create_score_expert_with_avg(score_avg_state, device)
                    new_score_experts.append(expert)
                    score_new_count += 1
            else:
                # ========== MODIFIED: ScoreExpert input dimension ==========
                expert = ScoreExpert(self.cfg.tf_de_dim * 10).to(device)
                # ========== END MODIFIED ==========
                if score_avg_state:
                    try:
                        expert.load_state_dict(score_avg_state, strict=False)
                    except Exception as e:
                        print(f"Warning: Could not initialize score expert with average: {e}")
                new_score_experts.append(expert)
                score_new_count += 1
                print(f"🆕 Created new score expert for cluster_id {current_cluster_id}")

        self.trajectory_experts = new_trajectory_experts
        self.score_experts = new_score_experts
        self.num_anchors = len(current_cluster_ids)

        print(f"Trajectory expert loading summary: {trajectory_loaded_count} loaded, {trajectory_new_count} created new")
        print(f"Score expert loading summary: {score_loaded_count} loaded, {score_new_count} created new")

        # Load non-expert parameters
        try:
            local_state_device = {k: v.to(device) for k, v in local_state.items()}
            missing_keys, unexpected_keys = super(PlanningTrajectoryDecoder, self).load_state_dict(
                local_state_device, strict=strict
            )
            print(f"Non-expert parameters: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
        except Exception as e:
            print(f"Warning: Failed to load non-expert parameters: {e}")

    def _build_cluster_to_params(self, expert_params, ckpt_cluster_ids, expert_prefix):
        """Build mapping from cluster_id to expert parameters"""
        cluster_to_params = {}
        
        for key, value in expert_params.items():
            # Parse expert index from key
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == expert_prefix.split('.')[-1] and i + 1 < len(parts):
                    try:
                        expert_idx = int(parts[i + 1])
                        if expert_idx < len(ckpt_cluster_ids):
                            cluster_id = ckpt_cluster_ids[expert_idx]
                            
                            # Build parameter name mapping
                            if key.startswith(f'{expert_prefix}.'):
                                param_name = key[len(f'{expert_prefix}.{expert_idx}.'):]
                            else:
                                start_idx = key.find(f'.{expert_prefix}.{expert_idx}.') + 1
                                param_name = key[start_idx + len(f'.{expert_prefix}.{expert_idx}'):]
                            
                            if cluster_id not in cluster_to_params:
                                cluster_to_params[cluster_id] = {}
                            cluster_to_params[cluster_id][param_name] = value
                        break
                    except (ValueError, IndexError):
                        continue
        
        return cluster_to_params

    def _compute_average_expert_state(self, cluster_to_params):
        """Compute average parameters from all checkpoint experts"""
        if not cluster_to_params:
            return {}
        
        avg_state = {}
        param_samples = {}
        
        for cluster_id, expert_sd in cluster_to_params.items():
            for param_name, param_value in expert_sd.items():
                if param_name not in param_samples:
                    param_samples[param_name] = []
                param_samples[param_name].append(param_value)
        
        for param_name, tensors in param_samples.items():
            try:
                stacked = torch.stack(tensors, dim=0)
                avg_state[param_name] = torch.mean(stacked, dim=0)
            except Exception as e:
                print(f"Warning: Could not compute average for {param_name}: {e}")
        
        return avg_state

    def _create_trajectory_expert_with_avg(self, avg_state, device):
        """Create trajectory expert with average parameters"""
        expert = TrajectoryExpert(self.cfg.tf_de_dim, 2).to(device)
        
        if avg_state:
            try:
                expert.load_state_dict(avg_state, strict=False)
            except Exception as e:
                print(f"Warning: Could not initialize trajectory expert with average: {e}")
        
        return expert

    def _create_score_expert_with_avg(self, avg_state, device):
        """Create score expert with average parameters"""
        # ========== MODIFIED: ScoreExpert input dimension ==========
        expert = ScoreExpert(self.cfg.tf_de_dim * 10).to(device)
        # ========== END MODIFIED ==========
        
        if avg_state:
            try:
                expert.load_state_dict(avg_state, strict=False)
            except Exception as e:
                print(f"Warning: Could not initialize score expert with average: {e}")
        
        return expert

    def _load_fallback(self, trajectory_expert_params, score_expert_params, local_state, strict, device):
        """
        Fallback to index-based loading when cluster_ids are not available in checkpoint
        """
        print("Using fallback index-based loading (no cluster_ids found in checkpoint)")
        
        # Parse checkpoint expert indices
        def get_num_experts(expert_params, expert_name):
            expert_indices = set()
            for key in expert_params.keys():
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == expert_name and i + 1 < len(parts):
                        try:
                            idx = int(parts[i + 1])
                            expert_indices.add(idx)
                            break
                        except (ValueError, IndexError):
                            continue
            return max(expert_indices) + 1 if expert_indices else 0
        
        ckpt_trajectory_num_anchors = get_num_experts(trajectory_expert_params, 'trajectory_experts')
        ckpt_score_num_anchors = get_num_experts(score_expert_params, 'score_experts')
        
        current_num_anchors = self.num_anchors

        print(f"[Fallback Loading] Checkpoint has {ckpt_trajectory_num_anchors} trajectory experts, {ckpt_score_num_anchors} score experts")
        print(f"Current model has {current_num_anchors} experts")

        # Build new ModuleLists
        new_trajectory_experts = nn.ModuleList()
        new_score_experts = nn.ModuleList()

        # Step 1: Load existing experts by index (up to min of both)
        num_load_trajectory = min(ckpt_trajectory_num_anchors, current_num_anchors)
        num_load_score = min(ckpt_score_num_anchors, current_num_anchors)

        # Load trajectory experts
        for i in range(num_load_trajectory):
            expert = TrajectoryExpert(self.cfg.tf_de_dim, 2).to(device)
            
            # Extract parameters for expert i
            expert_sd = {}
            for key, value in trajectory_expert_params.items():
                patterns = [f'trajectory_experts.{i}.', f'.trajectory_experts.{i}.']
                if any(pattern in key for pattern in patterns):
                    if key.startswith('trajectory_experts.'):
                        param_name = key[len(f'trajectory_experts.{i}.'):]
                    else:
                        start_idx = key.find(f'.trajectory_experts.{i}.') + 1
                        param_name = key[start_idx + len(f'.trajectory_experts.{i}'):]
                    
                    expert_sd[param_name] = value.to(device)
            
            if expert_sd:
                try:
                    expert.load_state_dict(expert_sd, strict=True)
                    new_trajectory_experts.append(expert)
                    print(f"✅ Successfully loaded trajectory expert {i} by index")
                except Exception as e:
                    print(f"❌ Failed to load trajectory expert {i} by index: {e}")
                    new_trajectory_experts.append(expert)
            else:
                print(f"⚠️ No parameters found for trajectory expert {i} by index")
                new_trajectory_experts.append(expert)

        # Load score experts
        for i in range(num_load_score):
            # ========== MODIFIED: ScoreExpert input dimension ==========
            expert = ScoreExpert(self.cfg.tf_de_dim * 10).to(device)
            # ========== END MODIFIED ==========
            
            # Extract parameters for expert i
            expert_sd = {}
            for key, value in score_expert_params.items():
                patterns = [f'score_experts.{i}.', f'.score_experts.{i}.']
                if any(pattern in key for pattern in patterns):
                    if key.startswith('score_experts.'):
                        param_name = key[len(f'score_experts.{i}.'):]
                    else:
                        start_idx = key.find(f'.score_experts.{i}.') + 1
                        param_name = key[start_idx + len(f'.score_experts.{i}'):]
                    
                    expert_sd[param_name] = value.to(device)
            
            if expert_sd:
                try:
                    expert.load_state_dict(expert_sd, strict=True)
                    new_score_experts.append(expert)
                    print(f"✅ Successfully loaded score expert {i} by index")
                except Exception as e:
                    print(f"❌ Failed to load score expert {i} by index: {e}")
                    new_score_experts.append(expert)
            else:
                print(f"⚠️ No parameters found for score expert {i} by index")
                new_score_experts.append(expert)

        # Step 2: Extend with new experts if needed
        if current_num_anchors > num_load_trajectory:
            print(f"🆕 Extending trajectory experts: adding {current_num_anchors - num_load_trajectory} new experts")
            
            # Compute average parameters from loaded trajectory experts
            trajectory_avg_state = self._compute_average_from_module_list(new_trajectory_experts)
            
            for i in range(num_load_trajectory, current_num_anchors):
                expert = TrajectoryExpert(self.cfg.tf_de_dim, 2).to(device)
                if trajectory_avg_state:
                    try:
                        expert.load_state_dict(trajectory_avg_state, strict=False)
                    except Exception as e:
                        print(f"Warning: Could not initialize trajectory expert {i} with average: {e}")
                new_trajectory_experts.append(expert)

        if current_num_anchors > num_load_score:
            print(f"🆕 Extending score experts: adding {current_num_anchors - num_load_score} new experts")
            
            # Compute average parameters from loaded score experts
            score_avg_state = self._compute_average_from_module_list(new_score_experts)
            
            for i in range(num_load_score, current_num_anchors):
                # ========== MODIFIED: ScoreExpert input dimension ==========
                expert = ScoreExpert(self.cfg.tf_de_dim * 10).to(device)
                # ========== END MODIFIED ==========
                if score_avg_state:
                    try:
                        expert.load_state_dict(score_avg_state, strict=False)
                    except Exception as e:
                        print(f"Warning: Could not initialize score expert {i} with average: {e}")
                new_score_experts.append(expert)

        # Replace experts
        self.trajectory_experts = new_trajectory_experts
        self.score_experts = new_score_experts
        self.num_anchors = current_num_anchors

        # Load non-expert parameters
        try:
            local_state_device = {k: v.to(device) for k, v in local_state.items()}
            missing_keys, unexpected_keys = super(PlanningTrajectoryDecoder, self).load_state_dict(
                local_state_device, strict=strict
            )
            
            if missing_keys:
                print(f"Missing {len(missing_keys)} non-expert keys in fallback mode")
            if unexpected_keys:
                print(f"Unexpected {len(unexpected_keys)} non-expert keys in fallback mode")
                
            print("✅ Successfully loaded non-expert parameters in fallback mode")
                
        except Exception as e:
            print(f"❌ Failed to load non-expert parameters in fallback mode: {e}")

        return

    def _compute_average_from_module_list(self, module_list):
        """Compute average parameters from a ModuleList"""
        if not module_list:
            return {}
        
        avg_state = {}
        param_samples = {}
        
        for expert in module_list:
            for name, param in expert.named_parameters():
                if name not in param_samples:
                    param_samples[name] = []
                param_samples[name].append(param.data.clone())
        
        for param_name, tensors in param_samples.items():
            try:
                stacked = torch.stack(tensors, dim=0)
                avg_state[param_name] = torch.mean(stacked, dim=0)
            except Exception as e:
                print(f"Warning: Could not compute average for {param_name}: {e}")
        
        return avg_state