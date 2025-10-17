import torch
from torch import nn
import numpy as np
from config import GlobalConfig
import json
import torch.nn.functional as F
from torch import vmap
from utils import print_data_info

class TrajectoryExpert(nn.Module):
    def __init__(self, input_dim=256, output_dim=20):
        super(TrajectoryExpert, self).__init__()
        self.net = self.offset_head = nn.Sequential(
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
        self.anchor_embed = nn.Linear(20, cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(cfg.tf_de_dropout)
        
        # Transformer decoder
        tf_layer = nn.TransformerDecoderLayer(
            d_model=cfg.tf_de_dim, 
            nhead=cfg.tf_de_heads,
            batch_first=False
        )
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=cfg.tf_de_layers)
        
        # Offset prediction head (predicts corrections to anchors)
        # self.offset_head = nn.Linear(cfg.tf_de_dim, 20)
        # self.offset_head = nn.Sequential(
        #     nn.Linear(cfg.tf_de_dim, cfg.tf_de_dim//2),
        #     nn.LayerNorm(cfg.tf_de_dim//2),  # Add normalization
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(cfg.tf_de_dim//2, cfg.tf_de_dim//4),
        #     nn.ReLU(),
        #     nn.Linear(cfg.tf_de_dim//4, 20)
        # )
        self.experts = nn.Sequential(*[
            TrajectoryExpert(cfg.tf_de_dim, 20)
            for _ in range(self.num_anchors)
        ])
        
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
    
    def _create_anchors(self, anchor_path=None):
        """
        Creates diverse trajectory anchors from file or generates default ones.
        Anchors are registered as non-trainable buffers.
        """
        if anchor_path is None:
            print('no anchor path, generate default anchors.')
            anchors = []
            t = np.linspace(0, 2.5, 10)  # 10 waypoints over 2.5 seconds

            # 1. Straight trajectories (different speeds)
            for speed in [3.0, 5.0, 7.0]:
                x = t * speed
                y = np.zeros_like(t)
                anchors.append(np.stack([x, y], axis=1).flatten())

            # 2. Left turns (different curvatures)
            for curve in [0.5, 1.0, 1.5]:
                x = t * 5.0
                y = curve * (t ** 2) / 2
                anchors.append(np.stack([x, y], axis=1).flatten())

            # 3. Right turns (different curvatures)
            for curve in [0.5, 1.0, 1.5]:
                x = t * 5.0
                y = -curve * (t ** 2) / 2
                anchors.append(np.stack([x, y], axis=1).flatten())

            anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
            self.num_anchors = anchors_tensor.size(0)

        else:
            with open(anchor_path, 'r') as f:
                data = json.load(f)

            # Extract first 20 elements (x,y for 10 waypoints) from each cluster's 'mu'
            mu_list = [entry['mu'][:20] for entry in data]
            anchors_tensor = torch.tensor(mu_list, dtype=torch.float32)
            self.num_anchors = anchors_tensor.size(0)
            print(f'read anchors from {anchor_path}')

        # ✅ Register buffer correctly: only once, and don't assign self.anchors first!
        self.register_buffer('anchors', anchors_tensor)

        print(f'anchors shape: {self.anchors.shape}')  # e.g., torch.Size([9, 20])
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
        # print_data_info(anchors_expanded)  #  torch.Size([99, 2, 20])
        query = self.anchor_embed(anchors_expanded)
        query = self.pos_drop(query)
        
        # Transformer decoding: (num_anchors, batch_size, dim)
        decoder_out = self.tf_decoder(
            tgt=query, 
            memory=memory
        )

        # # 每个专家都预测轨迹偏移量
        # pred_trajectories = []
        # for i in range(self.num_anchors):
        #     # 获取第i个专家的decoder输出: (batch_size, dim)
        #     expert_input = decoder_out[i]  # shape: (batch_size, cfg.tf_de_dim)
            
        #     # 第i个专家预测轨迹偏移量: (batch_size, 20)
        #     traj_offset = self.experts[i](expert_input)
            
        #     # 对应的anchor: (batch_size, 20)
        #     traj_anchor = anchors_expanded[i]
            
        #     # 最终轨迹 = anchor + 偏移量: (batch_size, 20)
        #     pred_traj = traj_anchor + traj_offset
        #     pred_trajectories.append(pred_traj)
        
        # # 堆叠所有专家的预测结果: (num_anchors, batch_size, 20)
        # pred_trajectories = torch.stack(pred_trajectories, dim=0)
            # 使用vmap进行批量专家处理

        # # 回退到手动循环
        # print('using for loop to calcu traj offset')
        # traj_offsets = torch.stack([
        #     self.experts[i](decoder_out[i]) 
        #     for i in range(self.num_anchors)
        # ], dim=0)
        
        # pred_trajectories = anchors_expanded + traj_offsets  # (num_anchors, batch_size, 20)

        # 重塑为批量处理格式
        num_anchors, batch_size, feat_dim = decoder_out.shape
        
        # 重塑为 (num_anchors * batch_size, feat_dim)
        decoder_flat = decoder_out.reshape(-1, feat_dim)
        
        # 批量处理所有专家（分块避免内存溢出）
        chunk_size = 32  # 根据GPU内存调整
        all_offsets = []
        
        for start_idx in range(0, num_anchors, chunk_size):
            end_idx = min(start_idx + chunk_size, num_anchors)
            chunk_size_current = end_idx - start_idx
            
            # 当前chunk的输入: (chunk_size * batch_size, feat_dim)
            chunk_input = decoder_flat[start_idx * batch_size : end_idx * batch_size]
            
            # 批量处理当前chunk的所有专家
            chunk_offsets = []
            for expert_idx in range(start_idx, end_idx):
                # 当前专家的数据范围
                data_start = (expert_idx - start_idx) * batch_size
                data_end = (expert_idx - start_idx + 1) * batch_size
                expert_data = chunk_input[data_start:data_end]
                
                # 专家前向传播
                offset = self.experts[expert_idx](expert_data)
                chunk_offsets.append(offset)
            
            # 堆叠当前chunk的结果
            chunk_offsets = torch.stack(chunk_offsets, dim=0)  # (chunk_size, batch_size, 20)
            all_offsets.append(chunk_offsets)
        
        # 合并所有chunk的结果
        traj_offsets = torch.cat(all_offsets, dim=0)  # (num_anchors, batch_size, 20)
        
        pred_trajectories = anchors_expanded + traj_offsets

        scores = self.score_head(decoder_out).squeeze(-1)  # (num_anchors, batch_size)
        
        return pred_trajectories, scores

    # def predict(self, encoder_out, top_k=3):
    #     """
    #     Inference method with top-k trajectory selection.
        
    #     Args:
    #         encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings
    #         top_k: Number of trajectories to return (default: 3)
            
    #     Returns:
    #         selected_trajectories: (batch_size, 20) - Best trajectory for each sample
    #         all_trajectories: (num_anchors, batch_size, 20) - All predicted trajectories
    #         scores: (num_anchors, batch_size) - Confidence scores
    #         topk_indices: (top_k, batch_size) - Indices of selected trajectories
    #     """
    #     with torch.no_grad():
    #         pred_trajectories, scores = self.forward(encoder_out)
    #         # scores = torch.sigmoid(scores)  # Now between 0 and 1
    #         scores_probs = F.softmax(scores, dim=0)
    #         batch_size = scores.size(1)
            
    #         # Initialize storage for top-k indices
    #         topk_indices = torch.zeros((top_k, batch_size), dtype=torch.long, device=scores.device)
            
    #         # Process each sample in batch
    #         for b in range(batch_size):
    #             # Get top-k scores for this sample
    #             _, indices = torch.topk(scores_probs[:, b], min(top_k, self.num_anchors))
    #             topk_indices[:, b] = indices
                
    #             # Diversity sampling (avoid similar trajectories)
    #             if top_k > 1:
    #                 unique_indices = [indices[0].item()]
    #                 for i in range(1, len(indices)):
    #                     current_idx = indices[i].item()
    #                     is_unique = True
                        
    #                     # Check similarity with already selected trajectories
    #                     for selected_idx in unique_indices:
    #                         current_traj = pred_trajectories[current_idx, b]
    #                         selected_traj = pred_trajectories[selected_idx, b]
    #                         distance = torch.norm(current_traj - selected_traj)
    #                         if distance < 1.0:  # Threshold for similarity
    #                             is_unique = False
    #                             break
                        
    #                     if is_unique and len(unique_indices) < top_k:
    #                         unique_indices.append(current_idx)
                    
    #                 # Update topk_indices with diverse selections
    #                 if len(unique_indices) < top_k:
    #                     # Fill remaining slots if needed
    #                     remaining = top_k - len(unique_indices)
    #                     for i in range(1, remaining + 1):
    #                         if len(unique_indices) < top_k and i < len(indices):
    #                             unique_indices.append(indices[i].item())
                    
    #                 topk_indices[:len(unique_indices), b] = torch.tensor(unique_indices, device=scores.device)
            
    #         # Get selected trajectories
    #         selected_trajectories = torch.stack([
    #             pred_trajectories[topk_indices[i, b], b] 
    #             for b in range(batch_size) 
    #             for i in range(top_k)
    #         ]).view(top_k, batch_size, 20)
            
    #         # Return the highest scoring trajectory as the primary output
    #         return selected_trajectories[0], pred_trajectories, scores_probs, topk_indices
        

