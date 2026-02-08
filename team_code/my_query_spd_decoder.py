import torch
from torch import nn
import numpy as np
from config import GlobalConfig
import json
import torch.nn.functional as F

class PlanningSpeedDecoder(nn.Module):
    """
    Planning Transformer Decoder using Speed anchors as queries.
    
    Input Shapes:
    - encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings from encoder
    
    Output Shapes (predict method):
    - selected_speeds: (batch_size, 8) - 8d 2-hot encoded speed
    - all_speeds: (num_anchors, batch_size, 8) - All predicted speeds
    - scores: (num_anchors, batch_size) - Confidence scores for each speed
    
    Note: 
    - Each speed represents 10 future waypoints 
    - Coordinates are in ego-vehicle frame (meters)
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Args:
            cfg: Configuration object containing model parameters
            num_anchors: Number of speed anchors to use 
        """
        super().__init__()
        self.cfg = cfg
        
        # Create diverse speed anchors (num_anchors, 8)
        self._create_anchors(cfg.prior_speed_path)
        # self._create_anchors()
        
        # Anchor embedding layer
        self.anchor_embed = nn.Linear(8, cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(cfg.tf_de_dropout)
        
        # Transformer decoder
        tf_layer = nn.TransformerDecoderLayer(
            d_model=cfg.tf_de_dim, 
            nhead=cfg.tf_de_heads,
            batch_first=False
        )
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=cfg.tf_de_layers)
        
        # Offset prediction head (predicts corrections to anchors)
        # self.offset_head = nn.Linear(cfg.tf_de_dim, 8)
        self.offset_head = nn.Sequential(
            nn.Linear(cfg.tf_de_dim, cfg.tf_de_dim//2),
            nn.LayerNorm(cfg.tf_de_dim//2),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.tf_de_dim//2, cfg.tf_de_dim//4),
            nn.ReLU(),
            nn.Linear(cfg.tf_de_dim//4, 8)
        )
        
        # Score prediction head (confidence for each speed)
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
        Creates diverse speed anchors covering common driving maneuvers.
        Goal anchor-based methods focus on predicting a set of feasible goal points that serve as anchors for speed generation. [[5]]
        """
        if anchor_path == None:
            print('no anchor path, return.')
            return
        else:
            with open(anchor_path, 'r') as f:
                data = json.load(f)

            # Extract the first 8 elements of 'mu' from each cluster
            mu_list = [entry['mu'][:8] for entry in data]

            # Convert to a PyTorch tensor and register directly
            anchors = torch.tensor(mu_list, dtype=torch.float32)  # ✅ 第一次转换是 OK 的（list → tensor）
            self.register_buffer('anchors', anchors)  # ✅ 直接传 tensor，不要再用 torch.tensor()
            print(f'read anchors from {anchor_path}')
            # print(f'anchors shape: {torch.tensor(anchors, dtype=torch.float32).shape}')
            print(f"Created {len(anchors)} anchor speeds")
            self.num_anchors = len(anchors)

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

        # Offset Head Initialization 
        for m in self.offset_head.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization since we're using ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, encoder_out):
        """
        Forward pass for training (simplified for inference focus)
        
        Args:
            encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings
            
        Returns:
            pred_speeds: (num_anchors, batch_size, 8) - Predicted speeds
            scores: (num_anchors, batch_size) - Confidence scores
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Transpose for transformer: (seq_len, batch_size, dim)
        memory = encoder_out.permute(1, 0, 2)
        
        # Prepare anchor queries: (num_anchors, batch_size, 8) -> (num_anchors, batch_size, dim)
        anchors_expanded = self.anchors.unsqueeze(1).expand(-1, batch_size, -1)
        query = self.anchor_embed(anchors_expanded)
        query = self.pos_drop(query)
        
        # Transformer decoding: (num_anchors, batch_size, dim)
        decoder_out = self.tf_decoder(
            tgt=query, 
            memory=memory
        )
        
        # Predict speed offsets and scores
        offsets = self.offset_head(decoder_out)  # (num_anchors, batch_size, 8)
        pred_speeds = anchors_expanded + offsets  # (num_anchors, batch_size, 8)
        scores = self.score_head(decoder_out).squeeze(-1)  # (num_anchors, batch_size)
        
        return pred_speeds, scores

    # def predict(self, encoder_out, top_k=3):
    #     """
    #     Inference method with top-k speed selection.
        
    #     Args:
    #         encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings
    #         top_k: Number of speeds to return (default: 3)
            
    #     Returns:
    #         selected_speeds: (batch_size, 8) - Best speed for each sample
    #         all_speeds: (num_anchors, batch_size, 8) - All predicted speeds
    #         scores: (num_anchors, batch_size) - Confidence scores
    #         topk_indices: (top_k, batch_size) - Indices of selected speeds
    #     """
    #     with torch.no_grad():
    #         pred_speeds, scores = self.forward(encoder_out)
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
                
    #             # Diversity sampling (avoid similar speeds)
    #             if top_k > 1:
    #                 unique_indices = [indices[0].item()]
    #                 for i in range(1, len(indices)):
    #                     current_idx = indices[i].item()
    #                     is_unique = True
                        
    #                     # Check similarity with already selected speeds
    #                     for selected_idx in unique_indices:
    #                         current_traj = pred_speeds[current_idx, b]
    #                         selected_traj = pred_speeds[selected_idx, b]
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
            
    #         # Get selected speeds
    #         selected_speeds = torch.stack([
    #             pred_speeds[topk_indices[i, b], b] 
    #             for b in range(batch_size) 
    #             for i in range(top_k)
    #         ]).view(top_k, batch_size, 8)
            
    #         # Return the highest scoring speed as the primary output
    #         return selected_speeds[0], pred_speeds, scores_probs, topk_indices
        

