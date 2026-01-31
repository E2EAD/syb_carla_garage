import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json

class SpeedFrontDoorEncoder(nn.Module):
    """
    速度前门编码器 - 处理速度特征和速度anchor的交互
    速度特征: (batch_size, 1, hidden_size) 单个token序列
    速度anchor: (num_anchors, 8) 8维速度向量
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.sf_de_dim if hasattr(config, 'sf_de_dim') else config.tf_de_dim  # 通常256
        
        # 读取速度anchor
        self._create_speed_anchors(self.config.prior_speed_path)
        
        # Anchor投影层：将8维速度anchor投影到hidden_size
        self.anchor_proj = nn.Sequential(
            nn.Linear(8, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )
        
        # 自注意力层（单token的自注意力简化为线性变换，但仍保留结构）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 交叉注意力层：速度特征查询速度anchor
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.norm3 = nn.LayerNorm(self.hidden_size)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(0.1)
        )
        
        # 门控机制
        self.aug_gate = nn.Linear(self.hidden_size, 1)
        self.ori_gate = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def _create_speed_anchors(self, anchor_path=None):
        """
        从文件读取速度anchor
        Args:
            anchor_path: 存储速度anchor聚类结果的JSON文件路径
        """
        if anchor_path is None:
            print('No speed anchor path provided, using random initialization.')
            return
        else:
            with open(anchor_path, 'r') as f:
                data = json.load(f)
            

            mu_list = [entry['mu'][:8] for entry in data]  # 取前8维作为速度

            
            anchors = torch.tensor(mu_list, dtype=torch.float32)
            self.register_buffer('speed_anchors', anchors)
            print(f'Read speed anchors from {anchor_path}, shape: {anchors.shape}')

    def forward(self, speed_feats, speed_mask=None):
        """
        Args:
            speed_feats: (batch_size, 1, hidden_size) - 速度特征，单token序列
            speed_mask: (batch_size, 1) - 速度特征mask（可选）
            
        Returns:
            enhanced_speed: (batch_size, 1, hidden_size) - 增强后的速度特征
            gate_weights: (batch_size, 1) - 门控权重
        """
        batch_size, seq_len, hidden_size = speed_feats.shape
        num_anchors, anchor_dim = self.speed_anchors.shape
        
        # 1. 投影速度anchor到特征空间 (num_anchors, 8) -> (num_anchors, hidden_size)
        # 注意：这里不需要时间扩展，因为速度anchor本身没有时间维度
        global_feats = self.anchor_proj(self.speed_anchors)  # [53, 256]
        
        # 2. 转换输入格式为seq_first
        # 速度特征: (batch_size, 1, 256) -> (1, batch_size, 256)
        local_feats_t = speed_feats.transpose(0, 1)  # [1, batch_size, 256]
        
        # 3. 速度anchor特征扩展以匹配batch维度
        # 将global_feats从[num_anchors, 256]扩展为[1, num_anchors, 256]
        # 然后重复batch_size次，变为[batch_size * num_anchors, 256]（稍后处理）
        
        # 4. 局部特征自注意力（单token的自注意力）
        # 对于单token，自注意力可视为残差连接
        self_attn_out, _ = self.self_attn(
            query=local_feats_t,      # [1, batch_size, 256]
            key=local_feats_t,        # [1, batch_size, 256]
            value=local_feats_t,      # [1, batch_size, 256]
            key_padding_mask=speed_mask  # [batch_size, 1]
        )
        self_attn_out = self.norm1(local_feats_t + self_attn_out)
        
        # 5. 交叉注意力：速度特征查询速度anchor
        # 扩展局部特征以匹配全局特征的维度 [1, batch_size, 256] -> [1, batch_size*num_anchors, 256]
        local_expanded = self_attn_out.repeat(1, 1, num_anchors).view(
            1, batch_size * num_anchors, hidden_size
        )
        
        # 扩展全局特征以匹配局部特征的维度
        # 将global_feats从[num_anchors, 256]扩展为[1, num_anchors, 256]
        # 然后重复batch_size次，变为[1, num_anchors*batch_size, 256]
        global_expanded = global_feats.unsqueeze(0)  # [1, num_anchors, 256]
        global_expanded = global_expanded.repeat(1, batch_size, 1)  # [1, num_anchors*batch_size, 256]
        
        # 交叉注意力
        cross_attn_out, cross_weights = self.cross_attn(
            query=local_expanded,     # [1, batch_size*num_anchors, 256]
            key=global_expanded,      # [1, batch_size*num_anchors, 256]
            value=global_expanded,    # [1, batch_size*num_anchors, 256]
        )
        
        # 6. 恢复原始维度并聚合anchor信息
        # cross_attn_out: [1, batch_size*num_anchors, 256] -> [1, batch_size, num_anchors, 256]
        cross_attn_out = cross_attn_out.view(1, batch_size, num_anchors, hidden_size)
        
        # 聚合所有anchor的信息（平均池化）
        cross_attn_out = torch.mean(cross_attn_out, dim=2)  # [1, batch_size, 256]
        cross_attn_out = self.norm2(self_attn_out + cross_attn_out)
        
        # 7. 前馈网络
        ffn_out = self.ffn(cross_attn_out)  # [1, batch_size, 256]
        ffn_out = self.norm3(cross_attn_out + ffn_out)
        
        # 8. 门控融合
        aug_weight = self.aug_gate(ffn_out)  # [1, batch_size, 1]
        ori_weight = self.ori_gate(local_feats_t)  # [1, batch_size, 1]
        gate_weights = self.sigmoid(aug_weight + ori_weight)  # [1, batch_size, 1]
        
        # 9. 最终输出
        out_feats_t = gate_weights * ffn_out + (1 - gate_weights) * local_feats_t  # [1, batch_size, 256]
        
        # 10. 转换回batch_first格式
        enhanced_speed = out_feats_t.transpose(0, 1)  # [batch_size, 1, 256]
        gate_weights = gate_weights.squeeze(-1).transpose(0, 1)  # [batch_size, 1]
        
        return enhanced_speed, gate_weights


# # ==================== 配置类示例 ====================
# class Config:
#     def __init__(self):
#         self.tf_de_dim = 256  # 轨迹编码器维度
#         self.sf_de_dim = 256  # 速度编码器维度
#         self.speed_anchor_path = "speed_anchors.json"  # 速度anchor文件路径
#         self.prior_traj_path = "traj_anchors.json"  # 轨迹anchor文件路径

# # ==================== 使用示例 ====================
# if __name__ == "__main__":
#     # 创建配置
#     config = Config()
    
#     # 创建速度前门编码器实例
#     speed_encoder = speedFrontDoorEncoder(config)
    
#     # 创建模拟输入
#     batch_size = 4
#     hidden_size = 256
#     speed_feats = torch.randn(batch_size, 1, hidden_size)  # 速度特征
    
#     # 前向传播
#     enhanced_speed, gate_weights = speed_encoder(speed_feats)
    
#     print(f"输入速度特征形状: {speed_feats.shape}")
#     print(f"增强后速度特征形状: {enhanced_speed.shape}")
#     print(f"门控权重形状: {gate_weights.shape}")
#     print(f"门控权重值: {gate_weights.squeeze().detach().numpy()}")