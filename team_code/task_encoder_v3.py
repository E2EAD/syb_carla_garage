import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np
from utils import print_data_info

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    """
    Transformer-based VAE Encoder
    """
    def __init__(self, config, input_dim=256, d_model=512, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1, latent_dim=20):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling and latent projection
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over sequence
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len=11, input_dim=256)
        Returns:
            mu: (batch_size, latent_dim)
            log_var: (batch_size, latent_dim)
        """
        # Input projection
        x = self.input_proj(x)  # (batch_size, 11, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # (batch_size, 11, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 11, d_model)
        
        # Global average pooling over sequence
        x = x.transpose(1, 2)  # (batch_size, d_model, 11)
        x = self.pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Latent projection
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var

class TransformerDecoder(nn.Module):
    """
    Transformer-based VAE Decoder
    """
    def __init__(self, config, output_dim=256, d_model=512, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1, latent_dim=20, seq_len=11):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Latent projection to initial sequence
        self.latent_proj = nn.Linear(latent_dim, d_model * seq_len)
        
        # Learnable initial tokens
        self.decoder_tokens = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, latent_dim)
        Returns:
            output: (batch_size, seq_len=11, output_dim=256)
        """
        batch_size = z.size(0)
        
        # Option 1: Use latent vector to generate initial sequence
        # initial_seq = self.latent_proj(z).view(batch_size, self.seq_len, self.d_model)
        
        # Option 2: Use learnable tokens conditioned on latent vector
        # Expand learnable tokens and condition with latent
        decoder_input = self.decoder_tokens.expand(batch_size, -1, -1)
        latent_condition = self.latent_proj(z).view(batch_size, self.seq_len, self.d_model)
        decoder_input = decoder_input + 0.1 * latent_condition  # Add conditioning
        
        # Add positional encoding
        decoder_input = self.pos_encoder(decoder_input.transpose(0, 1)).transpose(0, 1)
        decoder_input = self.dropout(decoder_input)
        
        # Self-attention decoding (no encoder memory in standard VAE decoder)
        # For auto-regressive generation, we'd use a mask, but for VAE we typically do parallel decoding
        tgt_mask = self._generate_square_subsequent_mask(self.seq_len).to(z.device)
        
        # Transformer decoding
        output = self.transformer_decoder(
            tgt=decoder_input,
            memory=decoder_input,  # Self-attention only
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        
        # Project to output dimension
        output = self.output_proj(output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerTaskEncoder(nn.Module):
    """
    Transformer-based VAE for trajectory task representation learning.
    Replaces MLP with Transformer architecture while maintaining same interface.
    """
    
    def __init__(self, config, input_dim=256, d_model=512, nhead=8, 
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024,
                 dropout=0.1, latent_dim=20, seq_len=11):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Transformer encoder and decoder
        self.encoder = TransformerEncoder(
            config=config,
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            latent_dim=latent_dim
        )
        
        self.decoder = TransformerDecoder(
            config=config,
            output_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            latent_dim=latent_dim,
            seq_len=seq_len
        )
        
        # 保持原有的锚点相关功能
        self.feature_anchors = None
        self.anchor_momentum = 0.9
        self.anchor_initialized = False
        self.anchor_reg_weight = getattr(config, 'anchor_reg_weight', 0.05)
        self.anchor_cluster_ids = None

    # 以下方法保持与原始TaskEncoder类完全相同
    def init_feature_anchors(self, num_anchors, cluster_ids=None):
        """初始化特征锚点"""
        device = next(self.parameters()).device
        self.feature_anchors = nn.Parameter(
            torch.zeros(num_anchors, self.seq_len, self.input_dim, device=device), 
            requires_grad=False
        )
        if cluster_ids is not None:
            self.anchor_cluster_ids = torch.tensor(cluster_ids, dtype=torch.long, device=device)
            self.register_buffer('anchor_cluster_ids_buffer', self.anchor_cluster_ids)
        self.anchor_initialized = True
        print(f"Initialized feature anchors for {num_anchors} prototypes")

    def load_anchor_mu_and_var(self):
        """从JSON文件加载anchor的均值和方差"""
        with open(self.config.prior_traj_path, 'r') as f:
            data = json.load(f)
        mu_list = [entry['mu'][:20] for entry in data]
        anchors_mu = torch.tensor(mu_list, dtype=torch.float32)
        var_list = [entry['var'][:20] for entry in data]
        anchor_var = torch.tensor(var_list, dtype=torch.float32)
        cluster_ids = [entry['cluster_id'] for entry in data]
        self.num_anchors = len(mu_list)
        print(f'got {self.num_anchors} anchors.')
        print(f'Read anchors mu and var from {self.config.prior_traj_path} for sampling')
        return anchors_mu, anchor_var, cluster_ids

    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode input to latent parameters"""
        # x should be (batch_size, 11, 256)
        return self.encoder(x)

    def decode(self, z):
        """Decode latent vector to reconstructed input"""
        return self.decoder(z)

    def forward(self, x, prototype_probs=None, update_anchors=False, prototype_labels=None):
        """
        前向传播，接口与原始TaskEncoder保持一致
        
        Args:
            x: 输入特征，可以是 (batch_size, 11*256) 或 (batch_size, 11, 256)
            prototype_probs: 原型选择概率
            update_anchors: 是否更新特征锚点
            prototype_labels: 原型标签
            
        Returns:
            包含各种输出的字典
        """
        # 处理输入形状
        original_shape = x.shape
        if len(original_shape) == 3:  # (batch_size, 11, 256)
            x_3d = x
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
            x_3d = x.reshape(original_shape[0], self.seq_len, self.input_dim)
        
        # Transformer VAE编码-解码过程
        mu, log_var = self.encode(x_3d)  # 直接使用3D输入
        z = self.reparameterize(mu, log_var)
        reconstructed_3d = self.decode(z)  # 输出是3D
        reconstructed_flat = reconstructed_3d.reshape(original_shape[0], -1)
        
        # 更新特征锚点（如果启用）
        if update_anchors and prototype_labels is not None and not self.config.sample_from_vae:
            self.num_anchors = prototype_probs.size(1)
            self.update_feature_anchors(x_3d, prototype_labels, prototype_probs)
        
        # 计算锚点正则化损失
        anchor_loss = torch.tensor(0.0, device=x.device)
        alignment_metrics = {}
        if prototype_probs is not None and self.anchor_initialized and not self.config.sample_from_vae:
            anchor_loss = self.compute_anchor_regularization_loss(x_3d, prototype_probs)
            alignment_metrics = self.get_anchor_alignment_metrics(x_3d, prototype_probs)
        else:
            anchor_loss = None
            alignment_metrics = None
        
        return {
            'mu': mu,
            'log_var': log_var, 
            'z': z,
            'reconstructed_flat': reconstructed_flat,
            'reconstructed_3d': reconstructed_3d,
            'features_3d': x_3d,
            'anchor_loss': anchor_loss,
            'alignment_metrics': alignment_metrics
        }

    # 以下工具方法保持与原始TaskEncoder完全相同
    def compute_kl_loss(self, mu, log_var, anchor_mu, anchor_var, soft_labels=None, temperature=1.0, focus_threshold=0.01):
        """KL损失计算（保持不变）"""
        batch_size = mu.size(0)
        num_anchors = anchor_mu.size(0)
        
        if soft_labels is not None:
            soft_labels = soft_labels.transpose(0, 1)
            if temperature != 1.0:
                soft_labels = F.softmax(soft_labels / temperature, dim=-1)
            
            if focus_threshold > 0:
                mask = soft_labels > focus_threshold
                masked_soft_labels = soft_labels * mask.float()
                masked_soft_labels = masked_soft_labels / (masked_soft_labels.sum(dim=-1, keepdim=True) + 1e-8)
                effective_soft_labels = masked_soft_labels
            else:
                effective_soft_labels = soft_labels
            
            # 向量化KL计算
            mu_expanded = mu.unsqueeze(1).expand(-1, num_anchors, -1)
            log_var_expanded = log_var.unsqueeze(1).expand(-1, num_anchors, -1)
            anchor_mu_expanded = anchor_mu.unsqueeze(0).expand(batch_size, -1, -1)
            anchor_var_expanded = anchor_var.unsqueeze(0).expand(batch_size, -1, -1)
            
            kl_div = self.kl_gaussian_vectorized(
                mu_expanded, log_var_expanded, anchor_mu_expanded, anchor_var_expanded
            )
            
            weighted_kl = torch.sum(kl_div * effective_soft_labels, dim=1)
            kl_loss = weighted_kl.mean()
        else:
            kl_loss = None
        
        return kl_loss

    def kl_gaussian_vectorized(self, mu_q, log_var_q, mu_p, var_p):
        """向量化KL散度计算（保持不变）"""
        var_q = torch.exp(log_var_q)
        kl = 0.5 * (
            torch.log(var_p + 1e-8) - log_var_q + 
            (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1
        )
        return kl.sum(dim=-1)

    def compute_reconstruction_loss(self, reconstructed, original, loss_type='mse', reduction='mean'):
        """重建损失计算（保持不变）"""
        if loss_type == 'mse':
            recon_loss = F.mse_loss(reconstructed, original, reduction=reduction)
        elif loss_type == 'l1':
            recon_loss = F.l1_loss(reconstructed, original, reduction=reduction)
        elif loss_type == 'smooth_l1':
            recon_loss = F.smooth_l1_loss(reconstructed, original, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        return recon_loss

    def get_latent_representation(self, x, deterministic=False):
        """获取潜在表示（保持不变）"""
        mu, log_var = self.encode(x)
        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, log_var)

    def sample_feat_and_traj(self, sample_num=2):
        """采样特征和轨迹（保持不变）"""
        if not hasattr(self, 'anchors_mu') or not hasattr(self, 'anchor_var'):
            self.anchors_mu, self.anchor_var, _ = self.load_anchor_mu_and_var()
            device = next(self.parameters()).device
            self.anchors_mu = self.anchors_mu.to(device)
            self.anchor_var = self.anchor_var.to(device)
        
        num_anchors = self.anchors_mu.size(0)
        anchor_indices = torch.randint(0, num_anchors, (sample_num,))
        selected_mu = self.anchors_mu[anchor_indices]
        selected_var = self.anchor_var[anchor_indices]
        selected_log_var = torch.log(selected_var + 1e-8)
        z = self.reparameterize(selected_mu, selected_log_var)
        reconstructed_3d = self.decode(z)
        sample_joined_checkpoint_features = reconstructed_3d
        sample_checkpoint_label = selected_mu.reshape(sample_num, 10, 2)
        return sample_joined_checkpoint_features.detach(), sample_checkpoint_label.detach()

    def compute_anchor_regularization_loss(self, features, prototype_probs):
        """锚点正则化损失（保持不变）"""
        if not self.anchor_initialized or self.feature_anchors is None:
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.size(0)
        num_anchors = len(self.feature_anchors)
        
        features_expanded = features.unsqueeze(1).expand(-1, num_anchors, -1, -1)
        anchors_expanded = self.feature_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        distances = F.mse_loss(
            features_expanded, 
            anchors_expanded, 
            reduction='none'
        ).mean(dim=(-1, -2))
        
        mask = prototype_probs > 0.01
        weighted_distances = distances * prototype_probs * mask.float()
        sample_weights = (prototype_probs * mask.float()).sum(dim=1)
        valid_samples_mask = sample_weights > 0
        sample_losses = weighted_distances.sum(dim=1)
        
        if valid_samples_mask.any():
            valid_losses = sample_losses[valid_samples_mask]
            valid_weights = sample_weights[valid_samples_mask]
            anchor_loss = (valid_losses / valid_weights).mean()
        else:
            anchor_loss = torch.tensor(0.0, device=features.device)
        
        return anchor_loss

    def get_anchor_alignment_metrics(self, features, prototype_probs):
        """锚点对齐指标（保持不变）"""
        if not self.anchor_initialized:
            return {}
        
        batch_size = features.size(0)
        num_anchors = len(self.feature_anchors)
        best_proto_indices = torch.argmax(prototype_probs, dim=1)
        batch_indices = torch.arange(batch_size, device=features.device)
        selected_anchors = self.feature_anchors[best_proto_indices]
        distances = F.mse_loss(features, selected_anchors, reduction='none').mean(dim=(-1, -2))
        max_probs = torch.gather(prototype_probs, 1, best_proto_indices.unsqueeze(1)).squeeze(1)
        
        alignment_distances = distances.detach().cpu().numpy()
        max_probs_np = max_probs.detach().cpu().numpy()
        
        metrics = {
            'avg_alignment_distance': np.mean(alignment_distances),
            'avg_max_prob': np.mean(max_probs_np),
            'alignment_std': np.std(alignment_distances)
        }
        return metrics

    def update_feature_anchors(self, features, prototype_labels, prototype_probs=None):
        """更新特征锚点（保持不变）"""
        if not self.anchor_initialized:
            if self.feature_anchors is None:
                self.init_feature_anchors(self.num_anchors)
            return
        
        batch_size = features.size(0)
        with torch.no_grad():
            for proto_idx in range(len(self.feature_anchors)):
                if prototype_probs is not None:
                    proto_weights = prototype_probs[:, proto_idx]
                    mask = proto_weights > 0.01
                else:
                    mask = (prototype_labels == proto_idx)
                    proto_weights = torch.ones(batch_size, device=features.device)
                
                if mask.sum() > 0:
                    proto_features = features[mask]
                    proto_weights_masked = proto_weights[mask]
                    weights_sum = proto_weights_masked.sum()
                    if weights_sum > 0:
                        weighted_features = torch.sum(
                            proto_features * proto_weights_masked.view(-1, 1, 1), 
                            dim=0
                        ) / weights_sum
                        if torch.any(self.feature_anchors[proto_idx] != 0):
                            updated_anchor = (
                                self.anchor_momentum * self.feature_anchors[proto_idx] + 
                                (1 - self.anchor_momentum) * weighted_features
                            )
                            self.feature_anchors.data[proto_idx] = updated_anchor
                        else:
                            self.feature_anchors.data[proto_idx] = weighted_features

    # 保持原有的状态字典加载方法
    def load_state_dict_with_resize(self, state_dict, strict=True):
        """状态字典加载（保持不变）"""
        device = next(self.parameters()).device
        anchor_params = {}
        other_params = {}
        
        for key, value in state_dict.items():
            if 'feature_anchors' in key or 'anchor_cluster_ids' in key:
                anchor_params[key] = value
            else:
                other_params[key] = value
        
        if anchor_params:
            self._load_anchors_by_cluster_id(anchor_params, device)
        else:
            print("No anchor parameters found in checkpoint")
        
        try:
            missing_keys, unexpected_keys = self.load_state_dict(other_params, strict=False)
            if missing_keys:
                print(f"Missing keys in non-anchor parameters: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in non-anchor parameters: {unexpected_keys}")
        except Exception as e:
            print(f"Error loading non-anchor parameters: {e}")

    def _load_anchors_by_cluster_id(self, anchor_params, device):
        """根据聚类ID加载锚点（保持不变）"""
        ckpt_anchors = None
        ckpt_cluster_ids = None
        
        for key, value in anchor_params.items():
            if 'feature_anchors' in key and 'cluster_ids' not in key:
                ckpt_anchors = value
            elif 'anchor_cluster_ids' in key or 'cluster_ids' in key:
                ckpt_cluster_ids = value.cpu().numpy().tolist()
        
        if ckpt_cluster_ids is None:
            print("Warning: No cluster_ids found in checkpoint, using fallback method")
            return self._load_anchors_fallback(anchor_params, device)
        
        current_cluster_ids = self.anchor_cluster_ids.cpu().numpy().tolist() if self.anchor_cluster_ids is not None else []
        print(f"Loading anchors by cluster_id: checkpoint = {ckpt_cluster_ids}, \n current = {current_cluster_ids}")
        
        new_anchors = []
        new_cluster_ids = []
        
        for current_id in current_cluster_ids:
            if current_id in ckpt_cluster_ids:
                ckpt_idx = ckpt_cluster_ids.index(current_id)
                new_anchors.append(ckpt_anchors[ckpt_idx])
                new_cluster_ids.append(current_id)
                print(f"✅ Loaded anchor for cluster_id {current_id}")
            else:
                if ckpt_anchors is not None:
                    avg_anchor = torch.mean(ckpt_anchors, dim=0)
                else:
                    avg_anchor = torch.zeros(self.seq_len, self.input_dim)
                new_anchors.append(avg_anchor)
                new_cluster_ids.append(current_id)
                print(f"🆕 Created new anchor for cluster_id {current_id}")
        
        if new_anchors:
            new_anchors_tensor = torch.stack(new_anchors).to(device)
            self.feature_anchors.data = new_anchors_tensor
            self.anchor_cluster_ids = torch.tensor(new_cluster_ids, dtype=torch.long, device=device)
            if hasattr(self, 'anchor_cluster_ids_buffer'):
                self.anchor_cluster_ids_buffer.data = self.anchor_cluster_ids
            else:
                self.register_buffer('anchor_cluster_ids_buffer', self.anchor_cluster_ids)

    def _load_anchors_fallback(self, anchor_params, device):
        """回退加载方法（保持不变）"""
        ckpt_anchors = None
        for key, value in anchor_params.items():
            if 'feature_anchors' in key:
                ckpt_anchors = value
                break
        
        if ckpt_anchors is None:
            print("No anchor data found in checkpoint")
            return
        
        ckpt_num_anchors = ckpt_anchors.size(0)
        current_num_anchors = self.feature_anchors.size(0) if self.anchor_initialized else 0
        print(f"Fallback loading: checkpoint has {ckpt_num_anchors} anchors, current has {current_num_anchors}")
        
        num_to_load = min(ckpt_num_anchors, current_num_anchors)
        if num_to_load > 0:
            self.feature_anchors.data[:num_to_load] = ckpt_anchors[:num_to_load].to(device)
            print(f"Loaded {num_to_load} anchors by index")
        
        if current_num_anchors > ckpt_num_anchors:
            if ckpt_num_anchors > 0:
                avg_anchor = torch.mean(ckpt_anchors, dim=0)
            else:
                avg_anchor = torch.zeros(self.seq_len, self.input_dim)
            self.feature_anchors.data[ckpt_num_anchors:] = avg_anchor.to(device)
            print(f"Initialized {current_num_anchors - ckpt_num_anchors} new anchors with average")