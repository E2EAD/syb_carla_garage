"""
Trajectory Denoiser for autonomous driving trajectory prediction
Based on JiT diffusion model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path

# 从原JiT代码中导入必要的组件
from model_jit import (
    TimestepEmbedder, 
    BottleneckPatchEmbed, 
    JiTBlock,
    modulate,
    RMSNorm,
    scaled_dot_product_attention,
    SwiGLUFFN,
    FinalLayer
)

class TrajectoryDenoiser(nn.Module):
    def __init__(self, config, trajectory_dim=20, speed_dim=8, perception_dim=256):
        super().__init__()
        self.config = config
        # self.num_anchors = num_anchors
        self.trajectory_dim = trajectory_dim
        self.speed_dim = speed_dim
        self.total_dim = trajectory_dim + speed_dim  # 28维

        # Anchor parameters (will be loaded from file)
        self.anchors = None
        self.anchor_vars = None
        self.anchor_path = config.prior_traj_path  # 需要配置anchor文件路径
        # 加载anchor数据
        if hasattr(config, 'prior_traj_path'):
            self.load_anchor_data(config.prior_traj_path)
        
        # 条件编码器（处理感知特征）
        self.perception_encoder = nn.Sequential(
            nn.Linear(perception_dim, perception_dim),
            nn.GELU(),
            nn.Dropout(config.proj_dropout),
            nn.Linear(perception_dim, perception_dim)
        )
        
        # 时间步编码器
        self.t_embedder = TimestepEmbedder(perception_dim)
        
        # 轨迹-速度编码器
        self.trajectory_embed = nn.Linear(self.total_dim, perception_dim)
        
        # Transformer解码器（类似JiT但适配轨迹数据）
        self.blocks = nn.ModuleList([
            JiTBlock(perception_dim, 
                    num_heads=config.num_denoiser_heads,
                    mlp_ratio=config.denoiser_mlp_ratio,
                    attn_drop=config.attn_dropout,
                    proj_drop=config.proj_dropout)
            for _ in range(config.denoiser_depth)
        ])
        
        # 多任务输出头
        self.reconstruction_head = nn.Linear(perception_dim, self.total_dim)  # 重构轨迹+速度
        self.selection_head = nn.Sequential(  # anchor选择概率
            nn.Linear(perception_dim, perception_dim//2),
            nn.GELU(),
            nn.Dropout(config.proj_dropout),
            nn.Linear(perception_dim//2, self.num_anchors)
        )
        
        # 最终归一化层
        self.norm_final = RMSNorm(perception_dim)
        
        # 扩散参数
        self.label_drop_prob = getattr(config, 'label_drop_prob', 0.1)
        self.P_mean = getattr(config, 'P_mean', -1.2)
        self.P_std = getattr(config, 'P_std', 1.2)
        self.t_eps = getattr(config, 't_eps', 1e-3)
        self.noise_scale = getattr(config, 'noise_scale', 1.0)
        
        # 损失权重
        self.velocity_weight = getattr(config, 'velocity_weight', 1.0)
        self.traj_weight = getattr(config, 'traj_weight', 1.0)
        self.speed_weight = getattr(config, 'speed_weight', 1)  
        self.selection_weight = getattr(config, 'selection_weight', 1)
        
        # 初始化权重
        self.initialize_weights()
        

    def initialize_weights(self):
        """权重初始化"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # 零初始化输出层
        nn.init.constant_(self.reconstruction_head.weight, 0)
        nn.init.constant_(self.reconstruction_head.bias, 0)
        
        # 零初始化选择头
        nn.init.constant_(self.selection_head[-1].weight, 0)
        nn.init.constant_(self.selection_head[-1].bias, 0)

    def load_anchor_data(self, prior_traj_path):
        """加载anchor的均值和方差"""
        if not Path(prior_traj_path).exists():
            print(f"Warning: Anchor file {prior_traj_path} not found. Using random initialization.")
            # # 使用随机初始化
            # self.anchors = nn.Parameter(torch.randn(self.num_anchors, self.total_dim))
            # self.anchor_vars = nn.Parameter(torch.ones(self.num_anchors, self.total_dim))
            return
        
        try:
            anchors_mu, anchor_var = self.load_anchor_mu_and_var(prior_traj_path)
            
            # # 确保anchor数量匹配
            # if anchors_mu.shape[0] != self.num_anchors:
            #     print(f"Warning: Expected {self.num_anchors} anchors, got {anchors_mu.shape[0]}. Using first {self.num_anchors}.")
            #     anchors_mu = anchors_mu[:self.num_anchors]
            #     anchor_var = anchor_var[:self.num_anchors]
            
            # 注册为parameter
            self.anchors = nn.Parameter(anchors_mu)
            self.anchor_vars = nn.Parameter(anchor_var)
            
            print(f"Successfully loaded {self.num_anchors} anchors from {prior_traj_path}")
            
        except Exception as e:
            print(f"Error loading anchors from {prior_traj_path}: {e}")
            # 使用随机初始化作为fallback
            self.anchors = nn.Parameter(torch.randn(self.num_anchors, self.total_dim))
            self.anchor_vars = nn.Parameter(torch.ones(self.num_anchors, self.total_dim))

    def load_anchor_mu_and_var(self, prior_traj_path):
        """从JSON文件加载anchor的均值和方差"""
        with open(prior_traj_path, 'r') as f:
            data = json.load(f)

        # Extract first 28 elements (x,y for 10 waypoints) from each cluster's 'mu'
        mu_list = [entry['mu'][:28] for entry in data]
        anchors_mu = torch.tensor(mu_list, dtype=torch.float32)
        var_list = [entry['var'][:28] for entry in data]
        anchor_var = torch.tensor(var_list, dtype=torch.float32)

        self.num_anchors = len(mu_list)
        print(f'got {self.num_anchors} anchors.')
        
        print(f'Read anchors mu and var from {prior_traj_path}')
        return anchors_mu, anchor_var

    def sample_t(self, n: int, device=None):
        """采样时间步 t ~ logit-normal"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, perception_feat, gt_trajectories=None, gt_speeds=None, training=True):
        """
        perception_feat: (B, 11, 256) 感知特征
        gt_trajectories: (B, 10, 2) 真实轨迹（仅训练时需要）
        gt_speeds: (B, 8) 真实速度（仅训练时需要）
        """
        bs = perception_feat.shape[0]
        
        if training:
            return self._forward_training(perception_feat, gt_trajectories, gt_speeds, bs)
        else:
            return self._forward_inference(perception_feat, bs)

    def _forward_training(self, perception_feat, gt_trajectories, gt_speeds, bs):
        """训练阶段 - 采用原版策略"""
        device = perception_feat.device
        
        # 组合真实数据
        gt_combined = torch.cat([
            gt_trajectories.reshape(bs, -1),  # (B, 20)
            gt_speeds  # (B, 8)
        ], dim=1)  # (B, 28)
        
        # 找到最近anchor并采样
        with torch.no_grad():
            closest_anchor_idx = self._find_closest_anchor(gt_combined)
        initial_trajectory = self.sample_from_anchor(closest_anchor_idx, training=True)
        
        # 扩散过程
        t = self.sample_t(bs, device=device).view(bs, 1)
        noise = torch.randn_like(initial_trajectory) * self.noise_scale
        noisy_trajectory = t * initial_trajectory + (1 - t) * noise
        
        # 计算真实速度场（关键步骤）
        v_target = (gt_combined - noisy_trajectory) / (1 - t).clamp_min(self.t_eps)
        
        # 网络预测
        perception_emb = self.perception_encoder(perception_feat.mean(dim=1))
        t_emb = self.t_embedder(t.squeeze(1))
        condition = perception_emb + t_emb
        
        x = self.trajectory_embed(noisy_trajectory).unsqueeze(1)
        for block in self.blocks:
            x = block(x, condition, feat_rope=None)
        
        x = self.norm_final(x)
        reconstruction = self.reconstruction_head(x.squeeze(1))
        selection_logits = self.selection_head(x.squeeze(1))
        
        # 计算预测速度场（关键步骤）
        v_pred = (reconstruction - noisy_trajectory) / (1 - t).clamp_min(self.t_eps)
        
        # 分离输出
        pred_trajectories = reconstruction[:, :self.trajectory_dim].reshape(bs, 10, 2)
        pred_speeds = reconstruction[:, self.trajectory_dim:]
        selection_probs = F.softmax(selection_logits, dim=-1)
        
        return pred_trajectories, pred_speeds, selection_probs, v_pred, v_target
    
    def _forward_inference(self, perception_feat, bs):
        """推理阶段前向传播 - 对每个anchor并行生成"""
        device = perception_feat.device
        all_trajectories = []
        all_speeds = []
        all_scores = []
        
        # 聚合感知特征
        perception_emb = self.perception_encoder(perception_feat.mean(dim=1))  # (B, 256)
        
        # 对每个anchor进行生成
        for anchor_idx in range(self.num_anchors):
            # 从anchor采样初始状态（推理时用均值）
            initial_state = self.sample_from_anchor(
                torch.tensor([anchor_idx] * bs, device=device), 
                training=False
            )  # (B, 28)
            
            # ODE去噪过程（简化版 - 单步预测）
            # 这里使用t=0.5作为示例，实际应该使用完整ODE求解
            t = torch.ones(bs, 1, device=device) * 0.5
            t_emb = self.t_embedder(t.squeeze(1))
            condition = perception_emb + t_emb
            
            # 通过网络
            x = self.trajectory_embed(initial_state).unsqueeze(1)
            for block in self.blocks:
                x = block(x, condition, feat_rope=None)
            
            x = self.norm_final(x)
            reconstruction = self.reconstruction_head(x.squeeze(1))
            selection_logits = self.selection_head(x.squeeze(1))
            
            # 分离输出
            traj = reconstruction[:, :self.trajectory_dim].reshape(bs, 10, 2)  # (B, 10, 2)
            speed = reconstruction[:, self.trajectory_dim:]  # (B, 8)
            score = selection_logits[:, anchor_idx]  # (B,)
            
            all_trajectories.append(traj)
            all_speeds.append(speed)
            all_scores.append(score)
        
        # 重组输出格式匹配原接口
        trajectories = torch.stack(all_trajectories)  # (num_anchors, bs, 10, 2)
        speeds = torch.stack(all_speeds)  # (num_anchors, bs, 8)
        scores = torch.stack(all_scores)  # (num_anchors, bs)
        
        # 计算选择概率
        probs = F.softmax(scores, dim=0)  # (num_anchors, bs)
        
        return trajectories, speeds, probs

    def compute_loss(self, pred_trajectories, pred_speeds, pred_probs, v_pred, v_target, gt_trajectories, gt_speeds):
        """损失计算 - 以速度场损失为主"""
        losses = {}
        bs = gt_trajectories.shape[0]
        
        # 主损失：速度场MSE损失
        velocity_loss = F.mse_loss(v_pred, v_target)
        losses['loss_velocity'] = velocity_loss
        
        # 辅助损失：直接重构损失（L1，更鲁棒）
        traj_recon_loss = F.l1_loss(pred_trajectories, gt_trajectories)
        speed_recon_loss = F.l1_loss(pred_speeds, gt_speeds)
        losses.update({
            'loss_traj_recon': traj_recon_loss,
            'loss_speed_recon': speed_recon_loss
        })
        
        # 选择损失
        with torch.no_grad():
            gt_combined_flat = torch.cat([gt_trajectories.reshape(bs, -1), gt_speeds], dim=1)
            distances = self._compute_anchor_distances(gt_combined_flat)
            closest_anchor = distances.argmin(dim=1)
            selection_labels = F.one_hot(closest_anchor, self.num_anchors).float()
        
        selection_loss = F.binary_cross_entropy(pred_probs, selection_labels)
        losses['loss_selection'] = selection_loss
        
        # total_loss = (velocity_loss * self.velocity_weight + 
        #              traj_recon_loss * self.traj_weight + 
        #              speed_recon_loss * self.speed_weight + 
        #              selection_loss * self.selection_weight)
        # losses['loss_total'] = total_loss
        
        return losses
    
    def _find_closest_anchor(self, gt_combined):
        """找到离真实轨迹最近的anchor索引"""
        distances = self._compute_anchor_distances(gt_combined)
        return distances.argmin(dim=1)

    def _compute_anchor_distances(self, gt_combined):
        """计算到每个anchor的距离"""
        # gt_combined: (B, 28)
        # anchors: (num_anchors, 28)
        
        # 可以加权不同部分（轨迹vs速度）
        trajectory_weight = 1.0
        speed_weight = 0.5
        
        # 计算加权距离
        trajectory_dist = F.pairwise_distance(
            gt_combined[:, :self.trajectory_dim].unsqueeze(1),
            self.anchors[:, :self.trajectory_dim].unsqueeze(0),
            p=1
        )  # (B, num_anchors)
        
        speed_dist = F.pairwise_distance(
            gt_combined[:, self.trajectory_dim:].unsqueeze(1),
            self.anchors[:, self.trajectory_dim:].unsqueeze(0),
            p=1
        )  # (B, num_anchors)
        
        total_dist = trajectory_weight * trajectory_dist + speed_weight * speed_dist
        return total_dist

    def sample_from_anchor(self, anchor_idx, training=True):
        """从指定anchor采样"""
        if training:
            # 训练时：添加anchor方差噪声
            mean = self.anchors[anchor_idx]  # (B, 28) or (28,)
            std = self.anchor_vars[anchor_idx].sqrt()  # (B, 28) or (28,)
            return mean + std * torch.randn_like(mean)
        else:
            # 推理时：直接用均值
            return self.anchors[anchor_idx]