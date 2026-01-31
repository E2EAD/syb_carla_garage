import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np
from utils import print_data_info


class TaskEncoder(nn.Module):
    """
    VAE Encoder-Decoder for trajectory task representation learning.
    Encodes flattened joined_checkpoint_features into a latent space,
    forces alignment with trajectory anchors, and reconstructs input features.
    """
    
    def __init__(self, config, input_dim=11*256, hidden_dims=[1024, 1024, 1024, 1024], latent_dim=20):
        super().__init__()
        self.config = config

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mu and log_var heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)
        
        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        decoder_dims = [latent_dim] + hidden_dims[::-1]
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.BatchNorm1d(decoder_dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        
        # Final layer to reconstruct input
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)

        # 新增：特征锚点正则化
        self.feature_anchors = None
        self.anchor_momentum = 0.9  # 锚点更新的动量参数
        self.anchor_initialized = False
        
        # 锚点正则化的权重
        self.anchor_reg_weight = getattr(config, 'anchor_reg_weight', 0.05)

        # 新增：存储锚点ID
        self.anchor_cluster_ids = None
        
    def init_feature_anchors(self, num_anchors, cluster_ids=None):
        """
        初始化特征锚点
        
        Args:
            num_anchors: 原型数量
            cluster_ids: 锚点对应的聚类ID列表
        """
        device = next(self.parameters()).device
        
        # 锚点形状: (num_anchors, 11, 256)
        self.feature_anchors = nn.Parameter(
            torch.zeros(num_anchors, 11, 256, device=device), 
            requires_grad=False
        )
        
        # 存储锚点ID
        if cluster_ids is not None:
            self.anchor_cluster_ids = torch.tensor(cluster_ids, dtype=torch.long, device=device)
            self.register_buffer('anchor_cluster_ids_buffer', self.anchor_cluster_ids)
        
        self.anchor_initialized = True
        print(f"Initialized feature anchors for {num_anchors} prototypes")
    
    def load_anchor_mu_and_var(self):
        """从JSON文件加载anchor的均值和方差，同时获取聚类ID"""
        with open(self.config.prior_traj_path, 'r') as f:
            data = json.load(f)

        # Extract first 20 elements (x,y for 10 waypoints) from each cluster's 'mu'
        mu_list = [entry['mu'][:20] for entry in data]
        anchors_mu = torch.tensor(mu_list, dtype=torch.float32)
        var_list = [entry['var'][:20] for entry in data]
        anchor_var = torch.tensor(var_list, dtype=torch.float32)
        
        # 获取聚类ID
        cluster_ids = [entry['cluster_id'] for entry in data]

        self.num_anchors = len(mu_list)
        print(f'got {self.num_anchors} anchors.')
        
        print(f'Read anchors mu and var from {self.config.prior_traj_path} for sampling')
        return anchors_mu, anchor_var, cluster_ids
    
    def load_state_dict_with_resize(self, state_dict, strict=True):
        """
        根据锚点ID匹配加载权重，保留存在的锚点特征，为新增ID创建新锚点
        """
        device = next(self.parameters()).device
        
        # 分离锚点相关参数和其他参数
        anchor_params = {}
        other_params = {}
        
        for key, value in state_dict.items():
            if 'feature_anchors' in key or 'anchor_cluster_ids' in key:
                anchor_params[key] = value
            else:
                other_params[key] = value
        
        # 处理锚点参数
        if anchor_params:
            self._load_anchors_by_cluster_id(anchor_params, device)
        else:
            print("No anchor parameters found in checkpoint")
        
        # 加载其他参数
        try:
            missing_keys, unexpected_keys = self.load_state_dict(other_params, strict=False)
            if missing_keys:
                print(f"Missing keys in non-anchor parameters: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in non-anchor parameters: {unexpected_keys}")
        except Exception as e:
            print(f"Error loading non-anchor parameters: {e}")
    
    def _load_anchors_by_cluster_id(self, anchor_params, device):
        """
        根据聚类ID匹配加载锚点
        """
        # 从检查点中提取锚点数据和聚类ID
        ckpt_anchors = None
        ckpt_cluster_ids = None
        
        for key, value in anchor_params.items():
            if 'feature_anchors' in key and 'cluster_ids' not in key:
                ckpt_anchors = value
            elif 'anchor_cluster_ids' in key or 'cluster_ids' in key:
                ckpt_cluster_ids = value.cpu().numpy().tolist()
        
        # 如果没有聚类ID信息，使用回退方法
        if ckpt_cluster_ids is None:
            print("Warning: No cluster_ids found in checkpoint, using fallback method")
            return self._load_anchors_fallback(anchor_params, device)
        
        # 获取当前模型的聚类ID
        current_cluster_ids = self.anchor_cluster_ids.cpu().numpy().tolist() if self.anchor_cluster_ids is not None else []
        
        print(f"Loading anchors by cluster_id: checkpoint = {ckpt_cluster_ids}, \n current = {current_cluster_ids}")
        
        # 创建新的锚点张量
        new_anchors = []
        new_cluster_ids = []
        
        # 匹配现有的锚点
        for current_id in current_cluster_ids:
            if current_id in ckpt_cluster_ids:
                # 找到对应的索引
                ckpt_idx = ckpt_cluster_ids.index(current_id)
                new_anchors.append(ckpt_anchors[ckpt_idx])
                new_cluster_ids.append(current_id)
                print(f"✅ Loaded anchor for cluster_id {current_id}")
            else:
                # 新增的锚点，使用平均值或零初始化
                if ckpt_anchors is not None:
                    # 使用检查点锚点的平均值
                    avg_anchor = torch.mean(ckpt_anchors, dim=0)
                else:
                    # 使用零初始化
                    avg_anchor = torch.zeros(11, 256)
                
                new_anchors.append(avg_anchor)
                new_cluster_ids.append(current_id)
                print(f"🆕 Created new anchor for cluster_id {current_id}")
        
        # 如果有检查点中的锚点不在当前模型中，可以选择忽略或添加
        # 这里我们选择忽略，只保持当前模型的锚点结构
        
        # 更新锚点参数
        if new_anchors:
            new_anchors_tensor = torch.stack(new_anchors).to(device)
            self.feature_anchors.data = new_anchors_tensor
            self.anchor_cluster_ids = torch.tensor(new_cluster_ids, dtype=torch.long, device=device)
            
            # 更新注册的buffer
            if hasattr(self, 'anchor_cluster_ids_buffer'):
                self.anchor_cluster_ids_buffer.data = self.anchor_cluster_ids
            else:
                self.register_buffer('anchor_cluster_ids_buffer', self.anchor_cluster_ids)
    
    def _load_anchors_fallback(self, anchor_params, device):
        """
        回退方法：当没有聚类ID信息时使用索引匹配
        """
        ckpt_anchors = None
        for key, value in anchor_params.items():
            if 'feature_anchors' in key:
                ckpt_anchors = value
                break
        
        if ckpt_anchors is None:
            print("No anchor data found in checkpoint")
            return
        
        ckpt_num_anchors = ckpt_anchors.size(0)
        current_num_anchors = self.feature_anchors.size(0) if self.feature_initialized else 0
        
        print(f"Fallback loading: checkpoint has {ckpt_num_anchors} anchors, current has {current_num_anchors}")
        
        # 确定要加载的锚点数量
        num_to_load = min(ckpt_num_anchors, current_num_anchors)
        
        # 加载匹配的锚点
        if num_to_load > 0:
            self.feature_anchors.data[:num_to_load] = ckpt_anchors[:num_to_load].to(device)
            print(f"Loaded {num_to_load} anchors by index")
        
        # 如果当前模型有更多锚点，使用平均值初始化
        if current_num_anchors > ckpt_num_anchors:
            if ckpt_num_anchors > 0:
                avg_anchor = torch.mean(ckpt_anchors, dim=0)
            else:
                avg_anchor = torch.zeros(11, 256)
            
            self.feature_anchors.data[ckpt_num_anchors:] = avg_anchor.to(device)
            print(f"Initialized {current_num_anchors - ckpt_num_anchors} new anchors with average")
    
    def update_feature_anchors(self, features, prototype_labels, prototype_probs=None):
        """
        使用指数移动平均更新特征锚点
        
        Args:
            features: 聚合特征 (batch_size, 11, 256)
            prototype_labels: 每个样本最匹配的原型索引 (batch_size,)
            prototype_probs: 原型选择概率 (batch_size, num_anchors)，用于加权更新
        """
        if not self.anchor_initialized:
            if self.feature_anchors is None:
                self.init_feature_anchors(self.num_anchors)
            return
        
        batch_size = features.size(0)
        
        with torch.no_grad():
            # 使用向量化操作更新每个原型的锚点
            for proto_idx in range(len(self.feature_anchors)):
                if prototype_probs is not None:
                    # 使用概率加权
                    proto_weights = prototype_probs[:, proto_idx]  # (B,)
                    mask = proto_weights > 0.01
                else:
                    # 使用硬标签
                    mask = (prototype_labels == proto_idx)
                    proto_weights = torch.ones(batch_size, device=features.device)
                
                if mask.sum() > 0:
                    # 获取属于该原型的特征
                    proto_features = features[mask]  # (num_samples, 11, 256)
                    proto_weights_masked = proto_weights[mask]  # (num_samples,)
                    
                    # 计算加权平均特征
                    weights_sum = proto_weights_masked.sum()
                    if weights_sum > 0:
                        # 向量化加权平均计算
                        weighted_features = torch.sum(
                            proto_features * proto_weights_masked.view(-1, 1, 1), 
                            dim=0
                        ) / weights_sum  # (11, 256)
                        
                        # 指数移动平均更新
                        if torch.any(self.feature_anchors[proto_idx] != 0):
                            updated_anchor = (
                                self.anchor_momentum * self.feature_anchors[proto_idx] + 
                                (1 - self.anchor_momentum) * weighted_features
                            )
                            self.feature_anchors.data[proto_idx] = updated_anchor
                        else:
                            # 第一次初始化，直接赋值
                            self.feature_anchors.data[proto_idx] = weighted_features
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var)
        from N(0,1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def decode(self, z):
        """Decode latent vector to reconstructed input"""
        return self.decoder(z)
    
    def forward(self, x, prototype_probs=None, update_anchors=False, prototype_labels=None):
        """
        增强的前向传播，支持特征锚点正则化
        
        Args:
            x: 输入特征，可以是 (batch_size, 11*256) 或 (batch_size, 11, 256)
            prototype_probs: 原型选择概率 (batch_size, num_anchors)，用于锚点正则化
            update_anchors: 是否更新特征锚点
            prototype_labels: 原型标签 (batch_size,)，用于锚点更新
            
        Returns:
            包含各种输出的字典
        """
        # 保存原始特征形状用于锚点计算
        original_shape = x.shape
        if len(original_shape) == 3:  # (batch_size, 11, 256)
            x_flat = x.reshape(original_shape[0], -1)  # (batch_size, 11*256)
            features_3d = x  # 保存3D特征用于锚点计算
        else:
            x_flat = x
            features_3d = x.reshape(original_shape[0], 11, 256)  # 重塑为3D
        
        # VAE编码-解码过程
        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        reconstructed_flat = self.decode(z)
        reconstructed_3d = reconstructed_flat.reshape(original_shape[0], 11, 256)
        
        # 更新特征锚点（如果启用）
        if update_anchors and prototype_labels is not None:
            self.num_anchors = prototype_probs.size(1)
            self.update_feature_anchors(features_3d, prototype_labels, prototype_probs)
        
        # 计算锚点正则化损失（如果提供原型概率）
        anchor_loss = torch.tensor(0.0, device=x.device)
        alignment_metrics = {}
        if prototype_probs is not None and self.anchor_initialized:
            anchor_loss = self.compute_anchor_regularization_loss(features_3d, prototype_probs)
            alignment_metrics = self.get_anchor_alignment_metrics(features_3d, prototype_probs)
        
        return {
            'mu': mu,
            'log_var': log_var, 
            'z': z,
            'reconstructed_flat': reconstructed_flat,
            'reconstructed_3d': reconstructed_3d,
            'features_3d': features_3d,
            'anchor_loss': anchor_loss,
            'alignment_metrics': alignment_metrics
        }
    
    def compute_kl_loss(self, mu, log_var, anchor_mu, anchor_var, soft_labels=None, temperature=1.0, focus_threshold=0.01):
        """
        严格的原型对齐KL散度 - 修改版
        每个样本的潜变量应该明确靠近其最可能对应的原型
        
        Args:
            mu: Encoded mean (batch_size, 20) - 来自VAE编码器
            log_var: Encoded log variance (batch_size, 20) - 来自VAE编码器  
            anchor_mu: Anchor means (num_anchors, 20) - 目标分布
            anchor_var: Anchor variances (num_anchors, 20) - 目标分布
            soft_labels: Soft assignment probabilities (num_anchors, batch_size)
            temperature: Temperature for softmax weighting
            focus_threshold: 只考虑概率超过此阈值的原型，增强对齐明确性
            
        Returns:
            kl_loss: 严格对齐的KL散度损失
        """
        # batch_size = mu.size(0)
        # num_anchors = anchor_mu.size(0)
        
        # if soft_labels is not None:
        #     # soft_labels: (num_anchors, batch_size) -> (batch_size, num_anchors)
        #     soft_labels = soft_labels.transpose(0, 1)
            
        #     # 应用温度缩放使分布更尖锐
        #     if temperature != 1.0:
        #         soft_labels = F.softmax(soft_labels / temperature, dim=-1)
            
        #     # 增强对齐：对低概率原型进行mask，专注于主要原型
        #     if focus_threshold > 0:
        #         mask = soft_labels > focus_threshold
        #         # 重新归一化mask后的概率，专注于显著原型
        #         masked_soft_labels = soft_labels * mask.float()
        #         masked_soft_labels = masked_soft_labels / (masked_soft_labels.sum(dim=-1, keepdim=True) + 1e-8)
        #         effective_soft_labels = masked_soft_labels
        #     else:
        #         effective_soft_labels = soft_labels
            
        #     total_kl = 0
        #     # 对每个样本单独处理，实现严格对齐
        #     for i in range(batch_size):
        #         sample_kl = 0
        #         valid_prototypes = 0
                
        #         for proto_idx in range(num_anchors):
        #             weight = effective_soft_labels[i, proto_idx]
        #             if weight > 1e-6:  # 只考虑有权重的原型
        #                 # 计算该样本与特定原型的KL散度
        #                 kl = self.kl_gaussian(
        #                     mu[i], log_var[i], 
        #                     anchor_mu[proto_idx], anchor_var[proto_idx]
        #                 )
        #                 sample_kl += weight * kl
        #                 valid_prototypes += 1
                
        #         if valid_prototypes > 0:
        #             total_kl += sample_kl
            
        #     kl_loss = total_kl / batch_size if batch_size > 0 else 0
        # else:
        #     kl_loss = None
        
        # return kl_loss
        
        batch_size = mu.size(0)
        num_anchors = anchor_mu.size(0)
        
        if soft_labels is not None:
            # soft_labels: (num_anchors, batch_size) -> (batch_size, num_anchors)
            soft_labels = soft_labels.transpose(0, 1)
            
            # 应用温度缩放使分布更尖锐
            if temperature != 1.0:
                soft_labels = F.softmax(soft_labels / temperature, dim=-1)
            
            # 增强对齐：对低概率原型进行mask，专注于主要原型
            if focus_threshold > 0:
                mask = soft_labels > focus_threshold
                # 重新归一化mask后的概率，专注于显著原型
                masked_soft_labels = soft_labels * mask.float()
                masked_soft_labels = masked_soft_labels / (masked_soft_labels.sum(dim=-1, keepdim=True) + 1e-8)
                effective_soft_labels = masked_soft_labels
            else:
                effective_soft_labels = soft_labels
            
            # 向量化KL计算
            # 扩展维度用于广播计算
            mu_expanded = mu.unsqueeze(1).expand(-1, num_anchors, -1)  # (B, num_anchors, 20)
            log_var_expanded = log_var.unsqueeze(1).expand(-1, num_anchors, -1)  # (B, num_anchors, 20)
            anchor_mu_expanded = anchor_mu.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_anchors, 20)
            anchor_var_expanded = anchor_var.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_anchors, 20)
            
            # 向量化KL散度计算
            kl_div = self.kl_gaussian_vectorized(
                mu_expanded, log_var_expanded, anchor_mu_expanded, anchor_var_expanded
            )  # (B, num_anchors)
            
            # 加权平均
            weighted_kl = torch.sum(kl_div * effective_soft_labels, dim=1)  # (B,)
            kl_loss = weighted_kl.mean()
        else:
            kl_loss = None
        
        return kl_loss


    def kl_gaussian_vectorized(self, mu_q, log_var_q, mu_p, var_p):
        """
        向量化计算两个高斯分布之间的KL散度
        """
        # 将log_var_q转换为方差
        var_q = torch.exp(log_var_q)
        
        # KL散度公式: KL(q||p) = 0.5 * [log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
        kl = 0.5 * (
            torch.log(var_p + 1e-8) - log_var_q + 
            (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1
        )
        
        # 在潜在维度上求和
        return kl.sum(dim=-1)

    def kl_gaussian(self, mu_q, log_var_q, mu_p, var_p):
        """
        计算两个高斯分布之间的KL散度: KL(q||p)
        
        Args:
            mu_q: q分布的均值 (20,)
            log_var_q: q分布的对数方差 (20,)
            mu_p: p分布的均值 (20,) 
            var_p: p分布的方差 (20,)
            
        Returns:
            kl: KL散度值
        """
        # 将log_var_q转换为方差
        var_q = torch.exp(log_var_q)
        
        # KL散度公式: KL(q||p) = 0.5 * [log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
        kl = 0.5 * (
            torch.log(var_p + 1e-8) - log_var_q + 
            (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1
        )
        
        # 求和得到总KL散度
        return kl.sum()
    
    def compute_reconstruction_loss(self, reconstructed, original, loss_type='mse', reduction='mean'):
        """
        Compute reconstruction loss between original and reconstructed inputs
        
        Args:
            reconstructed: Reconstructed features (batch_size, input_dim)
            original: Original input features (batch_size, input_dim)
            loss_type: Type of loss - 'mse', 'l1', or 'smooth_l1'
            reduction: Reduction method - 'mean', 'sum', or 'none'
            
        Returns:
            recon_loss: Reconstruction loss
        """
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
        """
        Get latent representation without sampling (for inference)
        
        Args:
            x: Input features
            deterministic: If True, return mean instead of sampling
            
        Returns:
            z: Latent representation
        """
        mu, log_var = self.encode(x)
        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, log_var)
        
    def sample_feat_and_traj(self, sample_num = 2):
        """
        Sample features and trajectories from the VAE using anchor distributions.
        
        Steps:
        1. Sample anchors and use their mu as trajectory labels
        2. Reparameterize to get latent z using anchor mu and var  
        3. Decode z to get joined_checkpoint_features
        
        Returns:
            sample_checkpoint_label: Sampled trajectories (sample_num, 10, 2)
            sample_joined_checkpoint_features: Reconstructed features (sample_num, 11, 256)
        """
        # Load anchors if not already loaded
        if not hasattr(self, 'anchors_mu') or not hasattr(self, 'anchor_var'):
            # print('to load the anchor for sampling.')
            self.anchors_mu, self.anchor_var, _ = self.load_anchor_mu_and_var()
            # Move anchors to model's device if not already done
            if self.anchors_mu.device != next(self.parameters()).device:
                device = next(self.parameters()).device
                self.anchors_mu = self.anchors_mu.to(device)
                self.anchor_var = self.anchor_var.to(device)
        
        num_anchors = self.anchors_mu.size(0)
        
        # 1. Randomly sample anchor indices
        anchor_indices = torch.randint(0, num_anchors, (sample_num,))
        
        # Get the mu and var for selected anchors
        selected_mu = self.anchors_mu[anchor_indices]  # (sample_num, 20)
        selected_var = self.anchor_var[anchor_indices]  # (sample_num, 20)
        
        # 2. Reparameterize to get latent z using anchor distribution
        # Convert variance to log_var for reparameterization
        selected_log_var = torch.log(selected_var + 1e-8)
        z = self.reparameterize(selected_mu, selected_log_var)  # (sample_num, 20)
        
        # 3. Decode latent z to get reconstructed features
        reconstructed_features = self.decode(z)  # (sample_num, 11*256)
        
        # Reshape to match expected format
        sample_joined_checkpoint_features = reconstructed_features.reshape(sample_num, 11, 256)
        
        # Reshape trajectory from (sample_num, 20) to (sample_num, 10, 2)
        sample_checkpoint_label = selected_mu.reshape(sample_num, 10, 2)
        return sample_joined_checkpoint_features.detach(), sample_checkpoint_label.detach()  # return detached samples to avoid grad backward
    
    # def load_anchor_mu_and_var(self):
    #     """从JSON文件加载anchor的均值和方差"""
    #     with open(self.config.prior_traj_path, 'r') as f:
    #         data = json.load(f)

    #     # Extract first 20 elements (x,y for 10 waypoints) from each cluster's 'mu'
    #     mu_list = [entry['mu'][:20] for entry in data]
    #     anchors_mu = torch.tensor(mu_list, dtype=torch.float32)
    #     var_list = [entry['var'][:20] for entry in data]
    #     anchor_var = torch.tensor(var_list, dtype=torch.float32)

    #     self.num_anchors = len(mu_list)
    #     print(f'got {self.num_anchors} anchors.')
        
    #     print(f'Read anchors mu and var from {self.config.prior_traj_path} for sampling')
    #     return anchors_mu, anchor_var
    
    def compute_anchor_regularization_loss(self, features, prototype_probs):
        """
        计算特征锚点正则化损失
        
        Args:
            features: 当前聚合特征 (batch_size, 11, 256)
            prototype_probs: 原型选择概率 (batch_size, num_anchors)
            
        Returns:
            anchor_loss: 锚点正则化损失
        """
        # if not self.anchor_initialized or self.feature_anchors is None:
        #     return torch.tensor(0.0, device=features.device)
        
        # batch_size = features.size(0)
        # num_anchors = len(self.feature_anchors)
        
        # total_loss = 0
        # valid_samples = 0

        # # 确保锚点在正确的设备上
        # # anchors = self.feature_anchors.to(features.device)
        
        # for i in range(batch_size):
        #     sample_loss = 0
        #     sample_weight = 0
            
        #     for proto_idx in range(num_anchors):
        #         weight = prototype_probs[i, proto_idx]
        #         if weight > 0.01:  # 只考虑显著的原型
        #             # 计算特征与对应锚点的距离
        #             anchor = self.feature_anchors[proto_idx]  # (11, 256)
        #             feature = features[i]  # (11, 256)
                    
        #             # 使用MSE损失
        #             distance = F.mse_loss(feature, anchor, reduction='mean')
        #             sample_loss += weight * distance
        #             sample_weight += weight
            
        #     if sample_weight > 0:
        #         total_loss += sample_loss / sample_weight  # 归一化
        #         valid_samples += 1
        
        # anchor_loss = total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=features.device)
        # return anchor_loss

        if not self.anchor_initialized or self.feature_anchors is None:
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.size(0)
        num_anchors = len(self.feature_anchors)
        
        # 向量化计算：将特征扩展到 (batch_size, num_anchors, 11, 256)
        features_expanded = features.unsqueeze(1).expand(-1, num_anchors, -1, -1)  # (B, num_anchors, 11, 256)
        anchors_expanded = self.feature_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, num_anchors, 11, 256)
        
        # 向量化MSE计算
        distances = F.mse_loss(
            features_expanded, 
            anchors_expanded, 
            reduction='none'
        ).mean(dim=(-1, -2))  # (B, num_anchors)
        
        # 应用权重掩码
        mask = prototype_probs > 0.01  # (B, num_anchors)
        weighted_distances = distances * prototype_probs * mask.float()  # (B, num_anchors)
        
        # 计算每个样本的有效权重和
        sample_weights = (prototype_probs * mask.float()).sum(dim=1)  # (B,)
        valid_samples_mask = sample_weights > 0  # (B,)
        
        # 计算每个样本的加权损失
        sample_losses = weighted_distances.sum(dim=1)  # (B,)
        
        # 只对有效样本求平均
        if valid_samples_mask.any():
            valid_losses = sample_losses[valid_samples_mask]
            valid_weights = sample_weights[valid_samples_mask]
            anchor_loss = (valid_losses / valid_weights).mean()
        else:
            anchor_loss = torch.tensor(0.0, device=features.device)
        
        return anchor_loss
    
    def get_anchor_alignment_metrics(self, features, prototype_probs):
        """
        计算锚点对齐的评估指标，用于监控训练效果
        
        Args:
            features: 聚合特征 (batch_size, 11, 256)
            prototype_probs: 原型选择概率 (batch_size, num_anchors)
            
        Returns:
            metrics: 包含各种对齐指标的字典
        """
        # if not self.anchor_initialized:
        #     return {}
        
        # batch_size = features.size(0)
        # best_proto_indices = torch.argmax(prototype_probs, dim=1)  # (batch_size,)
        
        # alignment_distances = []
        # max_probs = []

        # # anchors = self.feature_anchors.to(features.device)
        
        # for i in range(batch_size):
        #     best_proto_idx = best_proto_indices[i]
        #     max_prob = prototype_probs[i, best_proto_idx]
            
        #     # 计算与最佳原型锚点的距离
        #     anchor = self.feature_anchors[best_proto_idx]
        #     feature = features[i]
        #     distance = F.mse_loss(feature, anchor, reduction='mean')
            
        #     alignment_distances.append(distance.item())
        #     max_probs.append(max_prob.item())
        
        # metrics = {
        #     'avg_alignment_distance': np.mean(alignment_distances),
        #     'avg_max_prob': np.mean(max_probs),
        #     'alignment_std': np.std(alignment_distances)
        # }
        
        # return metrics

        if not self.anchor_initialized:
            return {}
        
        batch_size = features.size(0)
        num_anchors = len(self.feature_anchors)
        
        # 找到每个样本最可能对应的原型
        best_proto_indices = torch.argmax(prototype_probs, dim=1)  # (B,)
        
        # 向量化计算：为每个样本选择对应的锚点
        # 创建一个索引张量来收集对应的锚点
        batch_indices = torch.arange(batch_size, device=features.device)
        selected_anchors = self.feature_anchors[best_proto_indices]  # (B, 11, 256)
        
        # 向量化MSE计算
        distances = F.mse_loss(
            features, 
            selected_anchors, 
            reduction='none'
        ).mean(dim=(-1, -2))  # (B,)
        
        # 获取每个样本的最大概率
        max_probs = torch.gather(prototype_probs, 1, best_proto_indices.unsqueeze(1)).squeeze(1)  # (B,)
        
        # 计算指标
        alignment_distances = distances.detach().cpu().numpy()
        max_probs_np = max_probs.detach().cpu().numpy()
        
        metrics = {
            'avg_alignment_distance': np.mean(alignment_distances),
            'avg_max_prob': np.mean(max_probs_np),
            'alignment_std': np.std(alignment_distances)
        }
        
        return metrics

