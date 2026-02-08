import torch
import torch.nn as nn
import torch.nn.functional as F
import math
<<<<<<< HEAD
=======
import json
>>>>>>> 292b63d6ceceb7e250022de6871d308bc00b4f72


class TaskEncoder(nn.Module):
    """
    VAE Encoder-Decoder for trajectory task representation learning.
    Encodes flattened joined_checkpoint_features into a latent space,
    forces alignment with trajectory anchors, and reconstructs input features.
    """
    
<<<<<<< HEAD
    def __init__(self, input_dim=11*256, hidden_dims=[11*256*2, 11*256, 11*256/2, 1024, 512, 256], latent_dim=20):
        super().__init__()
        
=======
    def __init__(self, config, input_dim=11*256, hidden_dims=[1024, 1024, 1024, 1024], latent_dim=20):
        super().__init__()
        self.config = config

>>>>>>> 292b63d6ceceb7e250022de6871d308bc00b4f72
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
    
    def forward(self, x):
        """
        Forward pass of the VAE encoder-decoder
        
        Args:
            x: Flattened joined_checkpoint_features of shape (batch_size, 11*256)
            
        Returns:
            mu: Mean of latent distribution (batch_size, 20)
            log_var: Log variance of latent distribution (batch_size, 20)
            z: Sampled latent vector (batch_size, 20)
            reconstructed: Reconstructed input (batch_size, 11*256)
        """
        # Encode input
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode latent vector
        reconstructed = self.decode(z)
        
        return mu, log_var, z, reconstructed
    
    def compute_kl_loss(self, mu, log_var, anchor_mu, anchor_var, soft_labels=None, temperature=1.0):
        """
        Compute KL divergence between encoded distribution and anchor distributions
        
        Args:
            mu: Encoded mean (batch_size, 20) - from VAE (q)
            log_var: Encoded log variance (batch_size, 20) - from VAE (q)  
            anchor_mu: Anchor means (num_anchors, 20) - target distribution (p)
            anchor_var: Anchor variances (num_anchors, 20) - target distribution (p)
            soft_labels: Soft assignment probabilities (num_anchors, batch_size)
            temperature: Temperature for softmax weighting
            
        Returns:
            kl_loss: Weighted KL divergence loss KL(q||p)
        """
        batch_size = mu.size(0)
        num_anchors = anchor_mu.size(0)
        
        # Expand dimensions for broadcasting
        # q distribution: VAE output
        mu_q = mu.unsqueeze(1).expand(-1, num_anchors, -1)  # (batch_size, num_anchors, 20)
        log_var_q = log_var.unsqueeze(1).expand(-1, num_anchors, -1)
        var_q = torch.exp(log_var_q)  # Convert log_var to variance
        
        # p distribution: Anchor distribution (target)
        mu_p = anchor_mu.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_anchors, 20)
        var_p = anchor_var.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_anchors, 20)
        
        # KL divergence formula: KL(q||p) = 0.5 * [log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
        # This measures how much q diverges from p
        kl_div = 0.5 * (
            torch.log(var_p + 1e-8) - log_var_q +  # log(var_p) - log(var_q) = log(var_p/var_q)
            (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1
        )
        
        # Sum over latent dimension (20)
        kl_div = kl_div.sum(dim=-1)  # (batch_size, num_anchors)
        
        if soft_labels is not None:
            # Use soft_labels for weighting
            # soft_labels: (num_anchors, batch_size) -> (batch_size, num_anchors)
            soft_labels = soft_labels.transpose(0, 1)
            
            # Apply temperature scaling
            if temperature != 1.0:
                soft_labels = F.softmax(soft_labels / temperature, dim=-1)
            
            # Weighted average over anchors
            weighted_kl = torch.sum(kl_div * soft_labels, dim=-1)
            kl_loss = weighted_kl.mean()
        else:
            # Use uniform weighting (average over anchors and batch)
            kl_loss = kl_div.mean()
        
        return kl_loss
    
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
<<<<<<< HEAD
=======
        
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
            self.anchors_mu, self.anchor_var = self.load_anchor_mu_and_var()
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
    
    def load_anchor_mu_and_var(self):
        """从JSON文件加载anchor的均值和方差"""
        with open(self.config.prior_traj_path, 'r') as f:
            data = json.load(f)

        # Extract first 20 elements (x,y for 10 waypoints) from each cluster's 'mu'
        mu_list = [entry['mu'][:20] for entry in data]
        anchors_mu = torch.tensor(mu_list, dtype=torch.float32)
        var_list = [entry['var'][:20] for entry in data]
        anchor_var = torch.tensor(var_list, dtype=torch.float32)

        self.num_anchors = len(mu_list)
        print(f'got {self.num_anchors} anchors.')
        
        print(f'Read anchors mu and var from {self.config.prior_traj_path} for sampling')
        return anchors_mu, anchor_var
>>>>>>> 292b63d6ceceb7e250022de6871d308bc00b4f72


# class TaskEncoderConfig:
#     """Configuration for Task Encoder"""
    
#     def __init__(self, 
#                  input_dim=11*256,
#                  hidden_dims=[1024, 512, 256],
#                  latent_dim=20,
#                  kl_weight=0.1,
#                  recon_weight=1.0,
#                  recon_loss_type='mse',
#                  temperature=0.1):
        
#         self.input_dim = input_dim
#         self.hidden_dims = hidden_dims
#         self.latent_dim = latent_dim
#         self.kl_weight = kl_weight
#         self.recon_weight = recon_weight
#         self.recon_loss_type = recon_loss_type
#         self.temperature = temperature


# def test_task_encoder():
#     """Test function for TaskEncoder"""
#     config = TaskEncoderConfig()
#     encoder = TaskEncoder(config.input_dim, config.hidden_dims, config.latent_dim)
    
#     # Test input
#     batch_size = 4
#     test_input = torch.randn(batch_size, config.input_dim)
    
#     # Forward pass
#     mu, log_var, z, reconstructed = encoder(test_input)
    
#     print(f"Input shape: {test_input.shape}")
#     print(f"Mu shape: {mu.shape}")
#     print(f"Log var shape: {log_var.shape}")
#     print(f"Latent z shape: {z.shape}")
#     print(f"Reconstructed shape: {reconstructed.shape}")
    
#     # Test KL loss with anchor_var (not log_var)
#     num_anchors = 5
#     anchor_mu = torch.randn(num_anchors, config.latent_dim)
#     anchor_var = torch.ones(num_anchors, config.latent_dim) * 0.1  # variance, not log variance
#     soft_labels = torch.randn(num_anchors, batch_size)
    
#     kl_loss = encoder.compute_kl_loss(mu, log_var, anchor_mu, anchor_var, soft_labels)
#     print(f"KL Loss: {kl_loss.item()}")
    
#     # Test reconstruction loss
#     recon_loss = encoder.compute_reconstruction_loss(reconstructed, test_input, 'mse')
#     print(f"Reconstruction Loss: {recon_loss.item()}")
    
#     return encoder


# if __name__ == "__main__":
#     test_task_encoder()