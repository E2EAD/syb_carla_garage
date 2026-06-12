import torch
import torch.nn.functional as F


def get_correct_anchor_set(gt_trajectory, pred_anchors, distance_threshold=2.0, min_correct_anchors=1):
  """
  Build the correct trajectory-anchor support with mean per-step L1 distance.

  gt_trajectory: (B, T, 2)
  pred_anchors: (B, K, T, 2)
  """
  anchor_l1 = torch.abs(pred_anchors - gt_trajectory.unsqueeze(1)).sum(dim=-1).mean(dim=-1)
  correct_mask = anchor_l1 < float(distance_threshold)

  if min_correct_anchors > 0:
    missing = correct_mask.sum(dim=1) < int(min_correct_anchors)
    if missing.any():
      k = min(int(min_correct_anchors), anchor_l1.size(1))
      nearest = torch.topk(anchor_l1[missing], k=k, dim=1, largest=False).indices
      fallback = torch.zeros_like(correct_mask[missing])
      fallback.scatter_(1, nearest, True)
      correct_mask[missing] = fallback

  return correct_mask


def renormalize_over_mask(probs, correct_mask, eps=1e-8):
  masked_probs = probs * correct_mask.to(dtype=probs.dtype)
  denom = masked_probs.sum(dim=-1, keepdim=True)

  # If an upstream caller passes an empty support, fall back to the original
  # probabilities instead of returning an invalid all-zero target.
  empty = denom <= eps
  if empty.any():
    masked_probs = torch.where(empty, probs, masked_probs)
    denom = masked_probs.sum(dim=-1, keepdim=True)

  return masked_probs / denom.clamp_min(eps)


def compute_oracle_traj_probs(base_logits, correct_mask, eps=1e-8):
  """
  q*(k) = pi_0(k) / sum_{j in C} pi_0(j), for k in C, else 0.
  """
  base_probs = F.softmax(base_logits, dim=-1)
  return renormalize_over_mask(base_probs, correct_mask, eps=eps)


def align_anchor_distributions(student_anchors, teacher_anchors, teacher_probs, eps=1e-8):
  """
  Project teacher trajectory probabilities onto the student's anchor space by
  soft geometric matching.

  student_anchors: (B, Ks, T, 2)
  teacher_anchors: (B, Kt, T, 2)
  teacher_probs: (B, Kt)
  """
  batch_size, _, t_steps, xy_dim = student_anchors.shape
  s_flat = student_anchors.reshape(batch_size, student_anchors.size(1), -1)
  t_flat = teacher_anchors.reshape(batch_size, teacher_anchors.size(1), -1)

  pairwise_l1 = torch.cdist(s_flat, t_flat, p=1) / float(t_steps * xy_dim)
  # match_temp = pairwise_l1.detach().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
  match_temp = 0.5
  assign_s_given_t = F.softmax(-pairwise_l1 / match_temp, dim=1).detach()

  aligned_probs = torch.einsum('bsk,bk->bs', assign_s_given_t, teacher_probs)
  return aligned_probs / aligned_probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def get_correct_speed_support(target_speed_2hot):
  return [torch.where(target_speed_2hot[i] > 1e-6)[0].tolist() for i in range(target_speed_2hot.size(0))]


def compute_oracle_speed_probs(base_speed_logits, correct_indices, eps=1e-8):
  """
  Renormalize base speed probabilities over the two-hot target support.
  """
  batch_size, num_classes = base_speed_logits.shape
  base_probs = F.softmax(base_speed_logits, dim=-1)
  oracle_probs = torch.zeros_like(base_probs)

  for batch_idx in range(batch_size):
    if len(correct_indices[batch_idx]) == 0:
      oracle_probs[batch_idx] = base_probs[batch_idx]
      continue

    correct_mask = torch.zeros(num_classes, device=base_probs.device, dtype=base_probs.dtype)
    correct_mask[correct_indices[batch_idx]] = 1.0
    masked_probs = base_probs[batch_idx] * correct_mask
    oracle_probs[batch_idx] = masked_probs / masked_probs.sum().clamp_min(eps)

  return oracle_probs


def safe_kl_with_target_probs(student_logits, target_probs, temperature=1.0):
  """Forward KL KL(target || student), with a target probability tensor."""
  s_log = F.log_softmax(student_logits / float(temperature), dim=-1)
  t_prob = torch.clamp(target_probs.detach(), min=1e-8)
  t_prob = t_prob / t_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
  return F.kl_div(s_log, t_prob, reduction='batchmean') * (float(temperature)**2)


_safe_kl_with_target_probs = safe_kl_with_target_probs
