import numpy as np
import torch
import torch.nn.functional as F

from oracle_kd import align_anchor_distributions, safe_kl_with_target_probs


def _unwrap_model(model):
  return model.module if hasattr(model, 'module') else model


class ForgettingMonitor:
  """
  Online forward-KL forgetting monitor for a frozen base policy and current
  policy, evaluated on new-task batches.
  """

  def __init__(self, base_model, config, kl_threshold=None, patience=None):
    self.base_model = base_model
    self.config = config
    self.kl_threshold = float(
        kl_threshold if kl_threshold is not None else getattr(config, 'kl_forgetting_threshold', 0.5))
    self.patience = int(patience if patience is not None else getattr(config, 'kl_patience', 3))
    self.kl_history = []
    self.last_check_step = 0

  def _inputs_from_batch(self, data_batch, device):
    lidar_key = 'temporal_lidar' if int(getattr(self.config, 'lidar_seq_len', 1)) > 1 else 'lidar'
    inputs = {
        'rgb': data_batch['rgb'].to(device, dtype=torch.float32),
        'lidar_bev': data_batch[lidar_key].to(device, dtype=torch.float32),
        'target_point': data_batch['target_point'].to(device, dtype=torch.float32),
        'ego_vel': data_batch['speed'].to(device, dtype=torch.float32).unsqueeze(1),
        'command': data_batch['command'].to(device, dtype=torch.float32),
    }
    if bool(getattr(self.config, 'two_tp_input', False)):
      inputs['target_point_next'] = data_batch['target_point_next'].to(device, dtype=torch.float32)
    return inputs

  def estimate_kl(self, current_model, data_batch, device):
    if self.base_model is None:
      return 0.0

    base_model = _unwrap_model(self.base_model)
    active_model = _unwrap_model(current_model)
    was_training = current_model.training
    base_model.eval()
    current_model.eval()

    with torch.no_grad():
      inputs = self._inputs_from_batch(data_batch, device)
      _ = self.base_model(**inputs)
      _ = current_model(**inputs)

      base_tensors = getattr(base_model, 'latest_distill_tensors', None)
      current_tensors = getattr(active_model, 'latest_distill_tensors', None)
      if base_tensors is None or current_tensors is None:
        kl_total = torch.tensor(0.0, device=device)
      else:
        kl_traj = torch.tensor(0.0, device=device)
        if 'traj_logits' in base_tensors and 'traj_logits' in current_tensors:
          base_traj_logits = base_tensors['traj_logits'].transpose(0, 1)
          current_traj_logits = current_tensors['traj_logits'].transpose(0, 1)
          base_traj_probs = F.softmax(base_traj_logits, dim=-1)
          if base_traj_logits.size(1) == current_traj_logits.size(1):
            target_traj_probs = base_traj_probs
          elif 'pred_trajectories' in base_tensors and 'pred_trajectories' in current_tensors:
            current_traj = current_tensors['pred_trajectories'].permute(1, 0, 2, 3)
            base_traj = base_tensors['pred_trajectories'].permute(1, 0, 2, 3)
            target_traj_probs = align_anchor_distributions(current_traj, base_traj, base_traj_probs)
          else:
            target_traj_probs = None

          if target_traj_probs is not None:
            kl_traj = safe_kl_with_target_probs(current_traj_logits, target_traj_probs, temperature=1.0)

        kl_speed = torch.tensor(0.0, device=device)
        if 'speed_logits' in base_tensors and 'speed_logits' in current_tensors:
          kl_speed = safe_kl_with_target_probs(
              current_tensors['speed_logits'],
              F.softmax(base_tensors['speed_logits'], dim=-1),
              temperature=1.0,
          )

        kl_total = 0.6 * kl_traj + 0.4 * kl_speed

    if was_training:
      current_model.train()
    return float(kl_total.item())

  def check_and_act(self, current_model, data_loader, device, global_step):
    monitor_every = int(getattr(self.config, 'monitor_kl_every', 100))
    if monitor_every > 0 and self.last_check_step >= 0 and global_step - self.last_check_step < monitor_every:
      return 'continue', {}
    self.last_check_step = global_step

    if self.base_model is None:
      return 'continue', {'forgetting_kl': 0.0}

    kl_values = []
    max_batches = int(getattr(self.config, 'monitor_kl_batches', 3))
    for batch_idx, data in enumerate(data_loader):
      if batch_idx >= max_batches:
        break
      kl_values.append(self.estimate_kl(current_model, data, device))

    if len(kl_values) == 0:
      return 'continue', {}

    avg_kl = float(np.mean(kl_values))
    self.kl_history.append(avg_kl)

    recent = self.kl_history[-5:]
    if len(recent) >= 2:
      kl_trend = float(np.polyfit(np.arange(len(recent)), np.asarray(recent), 1)[0])
    else:
      kl_trend = 0.0

    metrics = {
        'forgetting_kl': avg_kl,
        'kl_history_mean': float(np.mean(self.kl_history[-10:])),
        'kl_trend': kl_trend,
    }

    if avg_kl > self.kl_threshold:
      if len(self.kl_history) >= self.patience and all(k > self.kl_threshold
                                                       for k in self.kl_history[-self.patience:]):
        return 'early_stop', metrics
      return 'reduce_lr', metrics

    return 'continue', metrics
