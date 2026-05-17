import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle_kd import (
    align_anchor_distributions,
    compute_oracle_speed_probs,
    compute_oracle_traj_probs,
    get_correct_anchor_set,
    get_correct_speed_support,
    renormalize_over_mask,
    safe_kl_with_target_probs,
)


class OracleKDLoss(nn.Module):
  """
  Oracle Distribution KL loss for continual driving policy training.

  The target distribution is the base policy distribution conditioned on the
  ground-truth-correct support. If no frozen base tensors are provided, the
  current student's detached distribution is conditioned on the same support.
  """

  def __init__(self, config, base_model=None):
    super().__init__()
    self.config = config
    self.base_model = base_model
    self.traj_threshold = float(getattr(config, 'oracle_traj_threshold', 2.0))
    self.kd_temperature = float(getattr(config, 'oracle_kd_temperature', 1.0))
    self.min_correct_anchors = int(getattr(config, 'min_correct_anchors', 1))

  @staticmethod
  def _traj_probs_key(tensors):
    if tensors is None:
      return None
    if 'pred_traj_probs' in tensors:
      return 'pred_traj_probs'
    if 'traj_probs' in tensors:
      return 'traj_probs'
    return None

  @staticmethod
  def _route_target(gt_data, horizon, device):
    route = gt_data['route'].to(device, dtype=torch.float32)
    return route[:, :horizon]

  @staticmethod
  def _speed_target(gt_data, device):
    target_speed = gt_data['target_speed_twohot'].to(device, dtype=torch.float32)
    target_sum = target_speed.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return target_speed / target_sum

  def forward(self, student_tensors, gt_data, base_tensors=None):
    losses = {}
    if student_tensors is None:
      return losses

    if 'pred_trajectories' in student_tensors and 'traj_logits' in student_tensors:
      student_traj = student_tensors['pred_trajectories'].permute(1, 0, 2, 3)
      gt_route = self._route_target(gt_data, student_traj.size(2), student_traj.device)
      correct_mask = get_correct_anchor_set(
          gt_route,
          student_traj,
          distance_threshold=self.traj_threshold,
          min_correct_anchors=self.min_correct_anchors,
      )
      student_traj_logits = student_tensors['traj_logits'].transpose(0, 1)

      if base_tensors is not None and 'traj_logits' in base_tensors:
        base_traj_logits = base_tensors['traj_logits'].transpose(0, 1).detach()
        if base_traj_logits.size(1) == student_traj_logits.size(1):
          oracle_probs = compute_oracle_traj_probs(base_traj_logits, correct_mask)
        elif 'pred_trajectories' in base_tensors:
          teacher_probs = F.softmax(base_traj_logits / self.kd_temperature, dim=-1).detach()
          teacher_traj = base_tensors['pred_trajectories'].permute(1, 0, 2, 3).detach()
          aligned_probs = align_anchor_distributions(student_traj.detach(), teacher_traj, teacher_probs)
          oracle_probs = renormalize_over_mask(aligned_probs, correct_mask)
        else:
          oracle_probs = compute_oracle_traj_probs(student_traj_logits.detach(), correct_mask)
      else:
        oracle_probs = compute_oracle_traj_probs(student_traj_logits.detach(), correct_mask)

      losses['loss_traj_oracle_kl'] = safe_kl_with_target_probs(
          student_traj_logits,
          oracle_probs,
          self.kd_temperature,
      )

      probs_key = self._traj_probs_key(student_tensors)
      if probs_key is not None:
        best_idx = torch.argmax(student_tensors[probs_key], dim=0)
        batch_idx = torch.arange(best_idx.size(0), device=best_idx.device)
        selected_traj = student_traj[batch_idx, best_idx]
        losses['loss_traj_l1'] = F.l1_loss(selected_traj, gt_route)
        losses['oracle_correct_anchor_rate'] = correct_mask.float().mean().detach()

    if 'speed_logits' in student_tensors:
      student_speed_logits = student_tensors['speed_logits']
      target_speed = self._speed_target(gt_data, student_speed_logits.device)
      correct_indices = get_correct_speed_support(target_speed)

      if base_tensors is not None and 'speed_logits' in base_tensors:
        base_speed_logits = base_tensors['speed_logits'].detach()
      else:
        base_speed_logits = student_speed_logits.detach()

      oracle_speed = compute_oracle_speed_probs(base_speed_logits, correct_indices)
      losses['loss_speed_oracle_kl'] = safe_kl_with_target_probs(
          student_speed_logits,
          oracle_speed,
          self.kd_temperature,
      )

      pred_idx = torch.argmax(student_speed_logits, dim=1)
      target_idx = torch.argmax(target_speed, dim=1)
      losses['speed_acc'] = (pred_idx == target_idx).float().mean().detach()

    return losses
