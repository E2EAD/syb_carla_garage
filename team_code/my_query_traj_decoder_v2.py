import torch
from torch import nn
import numpy as np
from config import GlobalConfig
import json
import torch.nn.functional as F

class PlanningTrajectoryDecoder(nn.Module):
    """
    Planning Transformer Decoder using trajectory anchors as queries.
    
    Input Shapes:
    - encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings from encoder
    
    Output Shapes (predict method):
    - selected_trajectories: (batch_size, 20) - [x1,y1,x2,y2,...,x10,y10] coordinates
    - all_trajectories: (num_anchors, batch_size, 20) - All predicted trajectories
    - scores: (num_anchors, batch_size) - Confidence scores for each trajectory
    
    Note: 
    - Each trajectory represents 10 future waypoints 
    - Coordinates are in ego-vehicle frame (meters)
    - USe CLS token to choose anchor
    """
    
    def __init__(self, cfg: GlobalConfig, num_anchors=53):
        """
        Args:
            cfg: Configuration object containing model parameters
            num_anchors: Number of trajectory anchors to use (default: 10)
        """
        super().__init__()
        self.cfg = cfg
        # self.num_anchors = num_anchors
        
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
        self.offset_head = nn.Sequential(
            nn.Linear(cfg.tf_de_dim, cfg.tf_de_dim//2),
            nn.LayerNorm(cfg.tf_de_dim//2),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.tf_de_dim//2, cfg.tf_de_dim//4),
            nn.ReLU(),
            nn.Linear(cfg.tf_de_dim//4, 20)
        )
        
        # 可学习的CLS token，用于聚合全局信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.tf_de_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 路由网络：从CLS特征生成路由权重
        self.router_net = nn.Sequential(
            nn.Linear(cfg.tf_de_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_anchors)  # 输出每个锚点的路由权重
        )
        
        # 路由温度参数（控制选择的锐利度）
        # self.routing_temperature = nn.Parameter(torch.tensor(1.0))
        
        self._init_weights()
    
    def _create_anchors(self, anchor_path=None):
        """
        Creates diverse trajectory anchors covering common driving maneuvers.
        Goal anchor-based methods focus on predicting a set of feasible goal points that serve as anchors for trajectory generation. [[5]]
        """
        if anchor_path == None:
            print('no anchor path, return.')
            return
        else:
            with open(anchor_path, 'r') as f:
                data = json.load(f)

            # Extract the first 20 elements of 'mu' from each cluster
            mu_list = [entry['mu'][:20] for entry in data]

            # Convert to a PyTorch tensor and register directly
            anchors = torch.tensor(mu_list, dtype=torch.float32)  # ✅ 第一次转换是 OK 的（list → tensor）
            self.register_buffer('anchors', anchors)  # ✅ 直接传 tensor，不要再用 torch.tensor()
            print(f'read anchors from {anchor_path}')
            # print(f'anchors shape: {torch.tensor(anchors, dtype=torch.float32).shape}')
            print(f"Created {len(anchors)} anchor trajectories")
            self.num_anchors = len(anchors)

    def _init_weights(self):
        """Initialize network weights"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Offset Head Initialization 
        for m in self.offset_head.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization since we're using ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.router_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 较小的增益，避免初始权重过大
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 路由温度初始化
        # nn.init.constant_(self.routing_temperature, 1.0)

    def forward(self, encoder_out):
        """
        修改后的forward，加入CLS Token路由机制
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # 1. 准备Transformer输入
        memory = encoder_out.permute(1, 0, 2)  # (seq_len, batch_size, dim)
        
        # 2. 准备查询：CLS token + anchor queries
        # 扩展CLS token到batch维度
        cls_tokens = self.cls_token.expand(1, batch_size, -1)  # (1, batch_size, dim)
        
        # 准备anchor查询
        anchors_expanded = self.anchors.unsqueeze(1).expand(-1, batch_size, -1)
        query = self.anchor_embed(anchors_expanded)
        query = self.pos_drop(query)
        
        # 拼接CLS token和anchor queries
        all_queries = torch.cat([cls_tokens, query], dim=0)  # (1+num_anchors, batch_size, dim)
        
        # 3. Transformer解码
        decoder_out = self.tf_decoder(tgt=all_queries, memory=memory)  # (1+num_anchors, batch_size, dim)
        
        # 4. 分离CLS输出和anchor输出
        cls_features = decoder_out[0]  # (batch_size, dim) - CLS特征
        anchor_features = decoder_out[1:]  # (num_anchors, batch_size, dim) - 锚点特征
        
        # 5. 计算路由权重
        routing_logits = self._compute_routing_weights(cls_features)  # (batch_size, num_anchors)
        # routing_weights = F.softmax(routing_logits / self.routing_temperature, dim=-1)
        # scores = routing_weights.permute(1,0)
        scores = routing_logits.permute(1,0)
        
        # 6. 预测轨迹偏移（使用同一个offset_head，保持现有结构）
        offsets = self.offset_head(anchor_features)  # (num_anchors, batch_size, 20)
        pred_trajectories = anchors_expanded + offsets

        return pred_trajectories, scores

    def _compute_routing_weights(self, cls_features):
        """
        计算路由权重
        cls_features: (batch_size, dim) - CLS token的特征
        返回: (batch_size, num_anchors) - 路由logits
        """
        # 方法1: 直接通过路由网络计算
        router_features = self.router_net(cls_features)  # (batch_size, num_anchors)
        
        # 方法2: 与专家原型计算相似度（更精确）
        # router_features: (batch_size, 128) - 路由网络中间层特征
        # expert_prototypes: (num_anchors, 128) - 专家原型
        # routing_logits = torch.matmul(router_features, self.expert_prototypes.t())
        
        return router_features

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
        

