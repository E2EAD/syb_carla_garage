import torch.nn.functional as F

from my_model_wTFFdeQtdA3D import LidarCenterNet as BaseLidarCenterNet


class LidarCenterNet(BaseLidarCenterNet):
  """
  Oracle-only loss variant of the QtdA3D model.

  The architecture and forward outputs are inherited unchanged from
  my_model_wTFFdeQtdA3D.LidarCenterNet. Only compute_loss() is overridden so
  legacy controller trajectory/speed supervision is not constructed in the
  model. OracleKDLoss in the train script supplies the trajectory/speed loss.
  """

  def compute_loss(self, pred_wp, pred_target_speed, pred_trajectories, pred_traj_probs, pred_semantic,
                   pred_bev_semantic, pred_depth, pred_bounding_box, pred_wp_1, selected_path, waypoint_label,
                   target_speed_label, checkpoint_label, semantic_label, bev_semantic_label, depth_label,
                   center_heatmap_label, wh_label, yaw_class_label, yaw_res_label, offset_label, velocity_label,
                   brake_target_label, pixel_weight_label, avg_factor_label):
    loss = {}

    if self.config.use_semantic:
      loss_semantic = self.loss_semantic(pred_semantic, semantic_label)
      loss.update({'loss_semantic': loss_semantic})

    if self.config.use_bev_semantic:
      visible_bev_semantic_label = self.valid_bev_pixels.squeeze(1).int() * bev_semantic_label
      visible_bev_semantic_label = (self.valid_bev_pixels.squeeze(1).int() - 1) + visible_bev_semantic_label
      loss_bev_semantic = self.loss_bev_semantic(pred_bev_semantic, visible_bev_semantic_label)
      loss.update({'loss_bev_semantic': loss_bev_semantic})

    if self.config.use_depth:
      loss_depth = F.l1_loss(pred_depth, depth_label)
      loss.update({'loss_depth': loss_depth})

    if self.config.detect_boxes:
      loss_bbox = self.head.loss(pred_bounding_box[0], pred_bounding_box[1], pred_bounding_box[2], pred_bounding_box[3],
                                 pred_bounding_box[4], pred_bounding_box[5], pred_bounding_box[6],
                                 center_heatmap_label, wh_label, yaw_class_label, yaw_res_label, offset_label,
                                 velocity_label, brake_target_label, pixel_weight_label, avg_factor_label)
      loss.update(loss_bbox)

    return loss
