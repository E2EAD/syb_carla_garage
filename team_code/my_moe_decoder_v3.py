import torch
from torch import nn
import numpy as np
from config import GlobalConfig
import json
import torch.nn.functional as F
from torch import vmap
from utils import print_data_info

class TrajectoryExpert(nn.Module):
    def __init__(self, input_dim=256, output_dim=20+8):
        super(TrajectoryExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class PlanningTrajectoryDecoder(nn.Module):
    """
    Planning Transformer Decoder using trajectory anchors as queries, integrated MoE
    1 traj anchor ~ 1 expert decoder (a mlp)
    
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
        # self._create_anchors()
        
        # Anchor embedding layer
        self.anchor_embed = nn.Linear(self.cfg.expert_out_dim, cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(cfg.tf_de_dropout)
        
        # Transformer decoder
        tf_layer = nn.TransformerDecoderLayer(
            d_model=cfg.tf_de_dim, 
            nhead=cfg.tf_de_heads,
            batch_first=False
        )
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=cfg.tf_de_layers)

        # 初始化专家（先空着，后面 load 时再决定）
        self.experts = nn.ModuleList()
        self._build_experts(self.num_anchors)  # 初始构建
        
        # Score prediction head (confidence for each trajectory)
        # FIXED: Score head with better initialization and structure
        self.score_head = nn.Sequential(
            nn.Linear(cfg.tf_de_dim, cfg.tf_de_dim//2),
            nn.LayerNorm(cfg.tf_de_dim//2),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.tf_de_dim//2, cfg.tf_de_dim//4),
            nn.ReLU(),
            nn.Linear(cfg.tf_de_dim//4, 1)
        )
        
        self._init_weights()

    def _build_experts(self, num_experts):
        """构建指定数量的 experts"""
        experts = nn.ModuleList([
            TrajectoryExpert(self.cfg.tf_de_dim, self.cfg.expert_out_dim)
            for _ in range(num_experts)
        ])
        self.experts = experts
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
                mu_list.append(entry['mu'][:28])  # First 20 elements for 10 waypoints, next 8 for speed
            
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
        
        # Special initialization for score head
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize last layer bias to create initial diversity
        last_linear = self.score_head[-1]
        nn.init.normal_(last_linear.weight, std=0.01)
        nn.init.constant_(last_linear.bias, 0.1)  # Small positive bias

        # Expert offset Head Initialization 
        for expert in self.experts:  # each expert is a TrajectoryExpert module
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

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
        device = encoder_out.device
        # print(f'show the 0th anchor: {self.anchors[0]} ')
        
        # Transpose for transformer: (seq_len, batch_size, dim)
        memory = encoder_out.permute(1, 0, 2)
        
        # Prepare anchor queries: (num_anchors, batch_size, 20) -> (num_anchors, batch_size, dim)
        anchors_expanded = self.anchors.unsqueeze(1).expand(-1, batch_size, -1)
        # print_data_info(anchors_expanded)  #  torch.Size([99, 2, 28])
        query = self.anchor_embed(anchors_expanded)
        query = self.pos_drop(query)
        
        # Transformer decoding: (num_anchors, batch_size, dim)
        decoder_out = self.tf_decoder(
            tgt=query, 
            memory=memory
        )

        # 重塑为批量处理格式
        num_anchors, batch_size, feat_dim = decoder_out.shape
        
        # 重塑为 (num_anchors * batch_size, feat_dim)
        decoder_flat = decoder_out.reshape(-1, feat_dim)

        # 获取当前实际使用的专家（前 num_anchors 个）
        active_experts = self.experts[:num_anchors]  # ← 这是一个 Python list of modules
        chunk_size = 32  # 根据GPU内存调整

        # 分块处理
        all_outputs = []
        for start_idx in range(0, num_anchors, chunk_size):
            end_idx = min(start_idx + chunk_size, num_anchors)
            chunk_input = decoder_flat[start_idx * batch_size : end_idx * batch_size]

            chunk_outputs = []
            for local_idx, expert in enumerate(active_experts[start_idx:end_idx]):
                data_start = local_idx * batch_size
                data_end = (local_idx + 1) * batch_size
                expert_data = chunk_input[data_start:data_end]
                output = expert(expert_data)
                chunk_outputs.append(output)

            chunk_outputs = torch.stack(chunk_outputs, dim=0)
            all_outputs.append(chunk_outputs)

        all_outputs_tensor =  torch.cat(all_outputs, dim=0)  # (99,2,28)
        # print_data_info(all_outputs_tensor)
        
        traj_offsets = all_outputs_tensor[:,:,:20]  # (99,2,20)
        # print_data_info(traj_offsets)

        pred_speed_offsets = all_outputs_tensor[:,:,20:]  # (99,2,8)
        # print_data_info(pred_speed_offsets)
        
        pred_trajectories = anchors_expanded[:,:,:20] + traj_offsets  # (99,2,20)
        # print_data_info(pred_trajectories)

        pred_speeds = anchors_expanded[:,:,20:] + pred_speed_offsets  # (99,2,8)
        # print_data_info(pred_speeds)

        scores = self.score_head(decoder_out).squeeze(-1)  # (num_anchors, batch_size)
        
        return pred_trajectories, scores, pred_speeds
    
    def load_state_dict_with_resize(self, state_dict: dict, strict: bool = True):
        """
        Load with cluster_id based matching: only load experts whose cluster_id exists in current model
        """
        device = next(self.parameters()).device
        
        # Filter out anchor parameters
        local_state = {}
        expert_params = {}
        
        for key, value in state_dict.items():
            # print(key)
            if key.endswith('.anchors') or 'anchors' in key and key not in self.state_dict():
                print(f"⚠️ 跳过 anchor 数据: {key} (shape: {value.shape})")
                continue
            if key.endswith('.cluster_ids') or 'cluster_ids' in key and key not in self.state_dict():
                print(f"⚠️ 跳过 cluster_ids 数据: {key}")
                continue
            if key.startswith('experts.') or '.experts.' in key:
                expert_params[key] = value
            else:
                local_state[key] = value
        
        print(f"Found {len(expert_params)} expert parameters in checkpoint")
        
        if not expert_params:
            print("No expert parameters found in checkpoint")
            # Build experts based on current cluster_ids
            self._build_experts(self.num_anchors)
            self.experts.to(device)
            super(PlanningTrajectoryDecoder, self).load_state_dict(state_dict, strict=strict)
            return

        # Extract cluster_ids from checkpoint if available
        ckpt_cluster_ids = None
        for key, value in state_dict.items():
            if key.endswith('.cluster_ids') or 'cluster_ids' in key:
                ckpt_cluster_ids = value.cpu().numpy().tolist()
                break
        
        if ckpt_cluster_ids is None:
            print("Warning: No cluster_ids found in checkpoint, falling back to index-based matching")
            # Fall back to your existing index-based logic
            return self._load_fallback(expert_params, local_state, strict, device)
        
        # Get current cluster_ids
        current_cluster_ids = self.cluster_ids.cpu().numpy().tolist()
        
        print(f"[PlanningTrajectoryDecoder] Loading by cluster_id: "
            f"checkpoint has {len(ckpt_cluster_ids)} clusters, current has {len(current_cluster_ids)} clusters")
        print(f"Checkpoint cluster_ids: {ckpt_cluster_ids}")
        print(f"Current cluster_ids: {current_cluster_ids}")

        # Build mapping from cluster_id to expert index in checkpoint
        ckpt_cluster_to_idx = {}
        ckpt_cluster_to_params = {}
        
        for key, value in expert_params.items():
            # Parse expert index from key: experts.X.net... or query_traj_decoder.experts.X.net...
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'experts' and i + 1 < len(parts):
                    try:
                        expert_idx = int(parts[i + 1])
                        if expert_idx < len(ckpt_cluster_ids):
                            cluster_id = ckpt_cluster_ids[expert_idx]
                            ckpt_cluster_to_idx[cluster_id] = expert_idx
                            
                            # Build parameter name mapping
                            if key.startswith('experts.'):
                                param_name = key[len(f'experts.{expert_idx}.'):]
                            else:
                                start_idx = key.find(f'.experts.{expert_idx}.') + 1
                                param_name = key[start_idx + len(f'.experts.{expert_idx}'):]
                            
                            if cluster_id not in ckpt_cluster_to_params:
                                ckpt_cluster_to_params[cluster_id] = {}
                            ckpt_cluster_to_params[cluster_id][param_name] = value
                        break
                    except (ValueError, IndexError):
                        continue

        # Build new experts based on current cluster_ids
        new_experts = nn.ModuleList()
        loaded_count = 0
        new_count = 0
        
        # Compute average parameters for new experts
        avg_state = self._compute_average_expert_state(ckpt_cluster_to_params)
        
        for i, current_cluster_id in enumerate(current_cluster_ids):
            if current_cluster_id in ckpt_cluster_to_params:
                # Load existing expert
                expert = TrajectoryExpert(self.cfg.tf_de_dim, self.cfg.expert_out_dim).to(device)
                expert_sd = ckpt_cluster_to_params[current_cluster_id]
                
                try:
                    expert.load_state_dict(expert_sd, strict=True)
                    new_experts.append(expert)
                    loaded_count += 1
                    print(f"✅ Loaded expert for cluster_id {current_cluster_id}")
                except Exception as e:
                    print(f"❌ Failed to load expert for cluster_id {current_cluster_id}: {e}")
                    # Fallback: use average initialization
                    expert = self._create_expert_with_avg(avg_state, device)
                    new_experts.append(expert)
                    new_count += 1
            else:
                # Create new expert with average parameters
                expert = self._create_expert_with_avg(avg_state, device)
                new_experts.append(expert)
                new_count += 1
                print(f"🆕 Created new expert for cluster_id {current_cluster_id}")

        self.experts = new_experts
        self.num_anchors = len(current_cluster_ids)

        print(f"Expert loading summary: {loaded_count} loaded, {new_count} created new")

        # Load non-expert parameters
        try:
            local_state_device = {k: v.to(device) for k, v in local_state.items()}
            missing_keys, unexpected_keys = super(PlanningTrajectoryDecoder, self).load_state_dict(
                local_state_device, strict=strict
            )
            print(f"Non-expert parameters: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
        except Exception as e:
            print(f"Warning: Failed to load non-expert parameters: {e}")

    def _compute_average_expert_state(self, ckpt_cluster_to_params):
        """Compute average parameters from all checkpoint experts"""
        if not ckpt_cluster_to_params:
            return {}
        
        avg_state = {}
        param_samples = {}
        
        for cluster_id, expert_sd in ckpt_cluster_to_params.items():
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

    def _create_expert_with_avg(self, avg_state, device):
        """Create expert with average parameters"""
        expert = TrajectoryExpert(self.cfg.tf_de_dim, self.cfg.expert_out_dim).to(device)
        
        if avg_state:
            try:
                expert.load_state_dict(avg_state, strict=False)
            except Exception as e:
                print(f"Warning: Could not initialize with average: {e}")
        
        return expert

    def _load_fallback(self, expert_params, local_state, strict, device):
        """
        Fallback to index-based loading when cluster_ids are not available in checkpoint
        This replicates the original index-based matching logic
        """
        print("Using fallback index-based loading (no cluster_ids found in checkpoint)")
        
        # Parse checkpoint expert indices
        expert_indices = set()
        for key in expert_params.keys():
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'experts' and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        expert_indices.add(idx)
                        break
                    except (ValueError, IndexError):
                        continue
        
        if not expert_indices:
            print("Warning: Could not parse expert indices from checkpoint in fallback mode")
            ckpt_num_anchors = 0
        else:
            ckpt_num_anchors = max(expert_indices) + 1

        current_num_anchors = self.num_anchors

        print(f"[Fallback Loading] Checkpoint has {ckpt_num_anchors} experts, current model has {current_num_anchors} experts")

        # Build new ModuleList
        new_experts = nn.ModuleList()

        # Step 1: Load existing experts by index (up to min of both)
        num_load = min(ckpt_num_anchors, current_num_anchors)

        for i in range(num_load):
            expert = TrajectoryExpert(self.cfg.tf_de_dim, self.cfg.expert_out_dim)
            expert = expert.to(device)
            
            # Extract parameters for expert i
            expert_sd = {}
            for key, value in expert_params.items():
                patterns = [f'experts.{i}.', f'.experts.{i}.']
                if any(pattern in key for pattern in patterns):
                    if key.startswith('experts.'):
                        param_name = key[len(f'experts.{i}.'):]
                    else:
                        start_idx = key.find(f'.experts.{i}.') + 1
                        param_name = key[start_idx + len(f'.experts.{i}'):]
                    
                    expert_sd[param_name] = value.to(device)
            
            if expert_sd:
                try:
                    expert.load_state_dict(expert_sd, strict=True)
                    new_experts.append(expert)
                    print(f"✅ Successfully loaded expert {i} by index")
                except Exception as e:
                    print(f"❌ Failed to load expert {i} by index: {e}")
                    print(f"Expert state dict keys: {list(expert_sd.keys())}")
                    # Use randomly initialized expert as fallback
                    new_experts.append(expert)
            else:
                print(f"⚠️ No parameters found for expert {i} by index")
                new_experts.append(expert)

        # Step 2: If current model has more experts, extend with averaged parameters
        if current_num_anchors > ckpt_num_anchors:
            print(f"🆕 Extending experts: adding {current_num_anchors - ckpt_num_anchors} new experts by index")

            # Compute average parameters from loaded experts
            avg_state = {}
            if num_load > 0:
                # Collect parameter averages from successfully loaded experts
                for name, param in new_experts[0].named_parameters():
                    try:
                        tensors = []
                        for expert in new_experts[:num_load]:
                            for n, p in expert.named_parameters():
                                if n == name:
                                    tensors.append(p.data.clone())
                                    break
                        if tensors:
                            tensors_stack = torch.stack(tensors, dim=0)
                            avg_state[name] = tensors_stack.mean(dim=0)
                            print(f"Computed average for {name}: shape {avg_state[name].shape}")
                    except Exception as e:
                        print(f"Warning: Failed to compute average for {name}: {e}")
            else:
                avg_state = {}

            # Create new experts with average initialization
            for i in range(ckpt_num_anchors, current_num_anchors):
                new_expert = TrajectoryExpert(self.cfg.tf_de_dim, self.cfg.expert_out_dim)
                new_expert = new_expert.to(device)
                
                if avg_state:
                    # Try to initialize with average parameters
                    try:
                        new_expert.load_state_dict(avg_state, strict=False)
                        print(f"Initialized new expert {i} with average parameters")
                    except Exception as e:
                        print(f"Warning: Could not initialize new expert {i} with average: {e}")
                        # Expert remains with random initialization
                else:
                    print(f"Created new expert {i} with random initialization (no averages available)")
                
                new_experts.append(new_expert)

        # Replace experts
        self.experts = new_experts
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
        

