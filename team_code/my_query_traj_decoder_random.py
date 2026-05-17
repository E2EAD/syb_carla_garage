import torch
from torch import nn
import numpy as np
from config import GlobalConfig
import torch.nn.functional as F

class PlanningTrajectoryDecoder(nn.Module):
    """
    Planning Transformer Decoder using learned random queries.
    
    Input Shapes:
    - encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings from encoder
    
    Output Shapes (forward method):
    - pred_trajectories: (num_queries, batch_size, 20) - Predicted trajectories
    - scores: (num_queries, batch_size) - Confidence scores for each trajectory
    
    Note: 
    - Each trajectory represents 10 future waypoints 
    - Coordinates are in ego-vehicle frame (meters)
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Args:
            cfg: Configuration object containing model parameters
            num_queries: Number of learned trajectory queries to use
        """
        super().__init__()
        self.cfg = cfg
        
        self.num_queries = getattr(cfg, 'random_query_traj_num_queries', getattr(cfg, 'traj_num_queries', 99))
        self.query_traj_tokens = nn.Parameter(torch.empty(self.num_queries, 20))
        
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
        
        # Transformer decoder
        tf_layer = nn.TransformerDecoderLayer(
            d_model=cfg.tf_de_dim, 
            nhead=cfg.tf_de_heads,
            batch_first=False
        )
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=cfg.tf_de_layers)
        
        # Offset prediction head
        self.offset_head = nn.Sequential(
            nn.Linear(cfg.tf_de_dim, cfg.tf_de_dim//2),
            nn.LayerNorm(cfg.tf_de_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.tf_de_dim//2, cfg.tf_de_dim//4),
            nn.ReLU(),
            nn.Linear(cfg.tf_de_dim//4, 2)  # Predict (x,y) offset for each waypoint
        )
        
        # Score prediction head that uses all time steps
        self.score_head = nn.Sequential(
            nn.Linear(cfg.tf_de_dim * 10, cfg.tf_de_dim * 2),  # Input: all 10 time steps concatenated
            nn.LayerNorm(cfg.tf_de_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.tf_de_dim * 2, cfg.tf_de_dim),
            nn.ReLU(),
            nn.Linear(cfg.tf_de_dim, 1)
        )
        
        self._init_weights()
    
    def _create_anchors(self, anchor_path=None):
        return
    
    def _init_weights(self):
        """Initialize network weights"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize position encoding
        nn.init.normal_(self.pos_encoding, std=0.01)
        nn.init.xavier_uniform_(self.query_traj_tokens)
        
        # Special initialization for score head
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Offset Head Initialization
        for m in self.offset_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _process_queries_with_time_encoding(self, queries_expanded):
        """
        Process anchors with time encoding to get (num_anchors, batch_size, 10, 256) shaped queries.
        
        Args:
            queries_expanded: (num_queries, batch_size, 20) shaped query tokens
            
        Returns:
            query_feats: (num_queries, batch_size, 10, 256) shaped queries
        """
        num_queries, batch_size, _ = queries_expanded.shape
        
        # Reshape anchors to (num_anchors * batch_size, 10, 2)
        queries_reshaped = queries_expanded.reshape(-1, 10, 2)
        
        # Apply time encoding to each waypoint
        # (num_anchors * batch_size, 10, 2) -> (num_anchors * batch_size, 10, 256)
        time_encoded = self.time_encoding(queries_reshaped)
        
        # Add positional encoding for time steps
        time_encoded = time_encoded + self.pos_encoding
        
        # Apply anchor embedding to each time step
        # Reshape for linear layer: flatten time dimension
        time_encoded_flat = time_encoded.reshape(-1, 256)
        anchor_embedded = self.anchor_embed(time_encoded_flat)
        anchor_embedded = anchor_embedded.reshape(num_queries * batch_size, 10, -1)
        
        # Reshape back to (num_queries, batch_size, 10, tf_de_dim)
        query_feats = anchor_embedded.reshape(num_queries, batch_size, 10, -1)
        
        return query_feats
    
    def forward(self, encoder_out):
        """
        Forward pass for training.
        
        Args:
            encoder_out: (batch_size, seq_len, dim) - BEV feature embeddings
            
        Returns:
            pred_trajectories: (num_queries, batch_size, 20) - Predicted trajectories
            scores: (num_queries, batch_size) - Confidence scores
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Transpose for transformer: (seq_len, batch_size, dim)
        memory = encoder_out.permute(1, 0, 2)
        
        # Prepare learned random query tokens with time encoding
        queries_expanded = self.query_traj_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        anchor_queries = self._process_queries_with_time_encoding(queries_expanded)
        
        # Process each time step through transformer decoder
        num_anchors, batch_size, num_timesteps, dim = anchor_queries.shape
        
        # Reshape queries for transformer: (num_timesteps, num_anchors * batch_size, dim)
        queries_reshaped = anchor_queries.permute(2, 0, 1, 3).reshape(num_timesteps, num_anchors * batch_size, dim)
        
        # Expand memory for all anchors
        memory_expanded = memory.repeat(1, num_anchors, 1)
        
        # Apply transformer decoder
        decoder_out = self.tf_decoder(
            tgt=queries_reshaped,
            memory=memory_expanded
        )
        
        # Reshape decoder output back: (num_timesteps, num_anchors, batch_size, dim)
        decoder_out_reshaped = decoder_out.reshape(num_timesteps, num_anchors, batch_size, dim)
        
        # Transpose to (num_anchors, batch_size, num_timesteps, dim)
        decoder_out_transposed = decoder_out_reshaped.permute(1, 2, 0, 3)
        
        # Vectorized offset prediction for all time steps
        # Reshape to (num_anchors * batch_size * num_timesteps, dim)
        decoder_out_flat = decoder_out_transposed.reshape(-1, dim)
        
        # Predict offsets for all time steps at once: (num_anchors * batch_size * num_timesteps, 2)
        offsets_flat = self.offset_head(decoder_out_flat)
        
        # Reshape offsets: (num_anchors, batch_size, num_timesteps, 2)
        offsets = offsets_flat.reshape(num_anchors, batch_size, num_timesteps, 2)
        
        # Flatten the last two dimensions: (num_anchors, batch_size, 20)
        offsets_flattened = offsets.reshape(num_anchors, batch_size, -1)
        
        # Add offsets to learned random query tokens.
        pred_trajectories = queries_expanded + offsets_flattened
        
        # Score prediction using all time steps
        # Reshape decoder output for score head: (num_anchors, batch_size, num_timesteps * dim)
        decoder_for_score = decoder_out_transposed.reshape(num_anchors, batch_size, -1)
        
        # Predict scores: (num_anchors, batch_size, 1) -> (num_anchors, batch_size)
        scores = self.score_head(decoder_for_score).squeeze(-1)
        
        return pred_trajectories, scores
