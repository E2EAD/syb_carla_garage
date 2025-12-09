"""
Open-loop evaluation script for trajectory and speed prediction models.
Usage:
python open_loop_eval.py --logdir /path/to/model_folder --data_root /path/to/dataset --output_dir /path/to/results
i.e. :
python ./team_code/open_loop_eval.py --logdir ./log/syb_train_noMoe_4-gw --data_root /media/syb/syb_disk_2/b2d_base_v3/carla_dataset
 --output_dir ./ol_test_result/syb_train_noMoe_4-gw

 add the following text into model folder's config.json:
     "bev_class_names" : [
      "unlabeled",
      "road",
      "sidewalk",
      "lane_markers",
      "lane_markers broken",
      "stop_signs",
      "traffic_light_green",
      "traffic_light_yellow",
      "traffic_light_red",
      "vehicle",
      "walker"
    ],

    "selected_ability_list": [
     "No_Scenario","Give_Way", "Overtaking", "Merging", "Traffic_Sign", "Emergency_Brake"
    ],
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


# import sys
# import os
# # 确保从正确的位置导入
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)
# Import your model and data classes
from my_model_moe_v03tf import LidarCenterNet
from ability_data import Ability_CARLA_Data
from config import GlobalConfig
import transfuser_utils as t_u

import random

from utils import print_data_info


class BEV_mIoU:
    def __init__(self, num_classes, ignore_index=0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred, target):
        """
        Update confusion matrix with batch predictions
        
        Args:
            pred: (B, C, H, W) torch tensor - model predictions (logits)
            target: (B, H, W) torch tensor - ground truth labels with -1 as ignore
        """
        # Convert predictions to class indices
        pred = pred.argmax(dim=1)  # (B, H, W)
        # print_data_info(pred)  # (16,256,256)
        
        # Move to CPU for numpy operations
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Flatten batch dimensions
        pred_flat = pred_np.ravel()
        target_flat = target_np.ravel()
        # print(f'check 1 bev sem pred: {pred_flat[:20]}')
        # print(f'check 1 bev sem gt: {target_flat[:20]}')
        
        # Filter out ignore_index (-1)
        mask = target_flat != self.ignore_index
        pred_valid = pred_flat[mask]
        target_valid = target_flat[mask]

        # print_data_info(target_flat)
        # print_data_info(target_valid)
        
        # Build confusion matrix for this batch
        if len(target_valid) > 0:
            # Also filter out invalid class indices (just in case)
            mask_valid = (target_valid >= 0) & (target_valid < self.num_classes)
            target_valid = target_valid[mask_valid]
            pred_valid = pred_valid[mask_valid]
            
            if len(target_valid) > 0:
                cm = np.bincount(
                    self.num_classes * target_valid.astype(int) + pred_valid.astype(int),
                    minlength=self.num_classes * self.num_classes
                ).reshape(self.num_classes, self.num_classes)
                self.confusion_matrix += cm

    def compute_iou_per_class(self):
        """Compute IoU for each class"""
        iou_per_class = np.zeros(self.num_classes, dtype=np.float32)
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator > 0:
                iou_per_class[i] = tp / denominator
            else:
                iou_per_class[i] = float('nan')  # Class not present
        
        return iou_per_class
    
    def compute_miou(self):
        """Compute mean IoU (excluding classes with NaN)"""
        iou_per_class = self.compute_iou_per_class()
        valid_ious = iou_per_class[~np.isnan(iou_per_class) & (iou_per_class > 0)]
        
        if len(valid_ious) > 0:
            return float(np.mean(valid_ious))
        return 0.0
    
    def get_results(self):
        """Return mIoU and per-class IoU"""
        iou_per_class = self.compute_iou_per_class()
        miou = self.compute_miou()
        
        return {
            'mIoU': miou,
            'IoU_per_class': iou_per_class.tolist(),
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    # def compute_iou_per_class(self):
    #     """Compute IoU for each class"""
    #     iou_per_class = np.zeros(self.num_classes, dtype=np.float32)
        
    #     for i in range(self.num_classes):
    #         tp = self.confusion_matrix[i, i]
    #         fp = self.confusion_matrix[:, i].sum() - tp
    #         fn = self.confusion_matrix[i, :].sum() - tp
            
    #         denominator = tp + fp + fn
    #         if denominator > 0:
    #             iou_per_class[i] = tp / denominator
    #         else:
    #             iou_per_class[i] = float('nan')  # Class not present
        
    #     return iou_per_class
    
    # def compute_miou(self):
    #     """Compute mean IoU (excluding classes with NaN)"""
    #     iou_per_class = self.compute_iou_per_class()
    #     valid_ious = iou_per_class[~np.isnan(iou_per_class)]
        
    #     if len(valid_ious) > 0:
    #         return float(np.mean(valid_ious))
    #     return 0.0
    
    # def get_results(self):
    #     """Return mIoU and per-class IoU"""
    #     iou_per_class = self.compute_iou_per_class()
    #     miou = self.compute_miou()
        
    #     return {
    #         'mIoU': miou,
    #         'IoU_per_class': iou_per_class.tolist(),
    #         'confusion_matrix': self.confusion_matrix.tolist()
    #     }

class OpenLoopEvaluator:
    """Evaluator for open-loop trajectory and speed prediction"""
    
    def __init__(self, config, model_path, data_root, output_dir, device='cuda:0', seed=42):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存随机种子
        self.seed = seed
        
        # 设置随机种子
        self.set_random_seed(seed)
        
        # Initialize model (similar to agent)
        self.model = LidarCenterNet(config)
        
        # Load checkpoint (following agent style)
        self.load_model_checkpoint(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Create dataset
        print(f"Loading dataset from {data_root}")
        self.dataset = Ability_CARLA_Data(
            root=data_root,
            config=config,
            shared_dict=None,
            rank=0,
            validation=False,
            ability_list=config.selected_ability_list
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        # Metrics storage
        self.metrics = {
            'trajectory': defaultdict(list),
            'speed': defaultdict(list),
            'scenario_wise': defaultdict(lambda: defaultdict(list))
        }
        
        # Store predictions for visualization
        self.predictions = []
        self.ground_truths = []

        # Add BEV mIoU calculator initialization
        self.bev_miou_calculator = BEV_mIoU(
            num_classes=config.num_bev_semantic_classes,  # for unlabeled is ignored
            ignore_index=0
        )
        
        # Add to metrics storage
        self.metrics['bev_semantic'] = defaultdict(list)

    def set_random_seed(self, seed):
        """设置所有相关的随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 确保cudnn的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"Set random seed to: {seed}")
        
    def load_model_checkpoint(self, model_path):
        """Load model checkpoint following the agent's loading style"""
        print(f"Loading model from: {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # This is a full checkpoint with optimizer state etc.
            state_dict = state_dict['model_state_dict']
        
        # Convert SyncBatchNorm if needed (similar to agent)
        if self.config.sync_batch_norm:
            print("Converting SyncBatchNorm...")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Load state dict with strict=False to handle mismatches
        self.model.load_state_dict(state_dict, strict=False)
        
        # Try to compile if configured (similar to agent)
        if hasattr(self.config, 'compile') and self.config.compile:
            try:
                self.model = torch.compile(self.model, mode=self.config.compile_mode)
                print("Model compiled for evaluation")
            except Exception as e:
                print(f"Could not compile model: {e}")
        
        print("Model loaded successfully")
        
        # Print missing and unexpected keys
        print("\nLoading summary:")
        model_state = self.model.state_dict()
        missing_keys = []
        unexpected_keys = []
        
        for key in state_dict.keys():
            if key not in model_state:
                unexpected_keys.append(key)
        
        for key in model_state.keys():
            if key not in state_dict:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Missing keys in checkpoint ({len(missing_keys)}):")
            for key in missing_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
        
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")
    
    def run_evaluation(self):
        """Run full evaluation on the dataset"""
        print(f"Starting evaluation on {len(self.dataset)} samples")
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                try:
                    self.process_batch(data, batch_idx)
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Compute final metrics
        self.compute_metrics()
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.visualize_results()
        
        return self.metrics
    
    def process_batch(self, data, batch_idx):
        """Process a single batch of data"""
        # Move data to device
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        
        if self.config.lidar_seq_len > 1:
            lidar = data['temporal_lidar'].to(self.device, dtype=torch.float32)
        else:
            lidar = data['lidar'].to(self.device, dtype=torch.float32)
        
        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)
        command = data['command'].to(self.device, dtype=torch.float32)
        
        # Ground truth labels
        gt_checkpoints = data['route'][:, :self.config.predict_checkpoint_len].to(self.device, dtype=torch.float32)
        
        if self.config.use_twohot_target_speeds:
            gt_speed_twohot = data['target_speed_twohot'].to(self.device, dtype=torch.float32)
            gt_speed = torch.argmax(gt_speed_twohot, dim=1)
        else:
            gt_speed = data['target_speed'].to(self.device, dtype=torch.long)
        
        # Get scenario/ability information if available
        scenario = None
        if 'ability' in data:
            scenario = data['ability'][0] if isinstance(data['ability'], list) else data['ability']

        # Get BEV ground truth if available
        bev_gt = None
        if 'bev_semantic' in data:
            bev_gt = data['bev_semantic'].to(self.device, dtype=torch.long)
        
        # Model forward pass
        pred_wp, pred_target_speed, pred_trajectories, pred_traj_probs, task_latent_mu, task_latent_log_var, reconstructed_features, task_anchor_loss, task_alignment_metrics, sample_checkpoint_label, sample_pred_trajectories, sample_pred_traj_probs,\
        pred_semantic, pred_bev_semantic, pred_depth, \
      pred_bounding_box, attention_weights, pred_wp_1, selected_path = self.model(
            rgb=rgb,
            lidar_bev=lidar,
            target_point=target_point,
            ego_vel=ego_vel,
            command=command
        )
        
        # Convert predictions
        # print(f'pred_target_speed shape: {pred_target_speed.shape}')
        batch_size = pred_target_speed.shape[0] if pred_target_speed is not None else 1
        
        # # Handle different prediction types
        if pred_trajectories is not None and pred_traj_probs is not None:
            # Select best trajectory (top-1)
            best_anchor_indices = torch.argmax(pred_traj_probs, dim=0)
            batch_indices = torch.arange(batch_size, device=pred_trajectories.device)
            pred_checkpoint = pred_trajectories[best_anchor_indices, batch_indices]
        else:
            # Fallback: use waypoints if available
            if pred_wp is not None:
                pred_checkpoint = pred_wp
            else:
                # Create dummy predictions
                pred_checkpoint = torch.zeros_like(gt_checkpoints)
        
        # Get speed prediction
        if pred_target_speed is not None:
            speed_probs = F.softmax(pred_target_speed, dim=1)
            pred_speed_class = torch.argmax(speed_probs, dim=1)
        else:
            speed_probs = None
            pred_speed_class = None
        
        # Store for later analysis
        for i in range(batch_size):
            self.predictions.append({
                'checkpoints': pred_checkpoint[i].cpu().numpy(),
                'speed_class': pred_speed_class[i].item() if pred_speed_class is not None else None,
                'speed_probs': speed_probs[i].cpu().numpy() if speed_probs is not None else None,
                # 'trajectory_probs': pred_traj_probs[:, i].cpu().numpy() if pred_traj_probs is not None else None,
                # Add BEV semantic prediction
                'bev_semantic': pred_bev_semantic[i].argmax(dim=0).cpu().numpy() if pred_bev_semantic is not None else None
            })
            
            self.ground_truths.append({
                'checkpoints': gt_checkpoints[i].cpu().numpy(),
                'speed_class': gt_speed[i].item(),
                'scenario': scenario,
                # Add BEV ground truth
                'bev_semantic': bev_gt[i].cpu().numpy() if bev_gt is not None else None
            })
        
        # Compute batch metrics
        self.compute_batch_metrics(
            pred_checkpoint, 
            pred_speed_class, 
            speed_probs,
            gt_checkpoints, 
            gt_speed,
            scenario
        )

        # Update BEV mIoU calculator
        if bev_gt is not None and pred_bev_semantic is not None:
            self.bev_miou_calculator.update(pred_bev_semantic, bev_gt)
    
    def compute_batch_metrics(self, pred_checkpoints, pred_speed, speed_probs, 
                            gt_checkpoints, gt_speed, scenario):
        """Compute metrics for a single batch"""
        batch_size = pred_checkpoints.shape[0]
        
        for i in range(batch_size):
            pred_cp = pred_checkpoints[i].cpu().numpy()  # (10, 2)
            gt_cp = gt_checkpoints[i].cpu().numpy()      # (10, 2)
            
            # 1. Trajectory Metrics
            # Average Displacement Error (ADE)
            pointwise_distances = np.linalg.norm(pred_cp - gt_cp, axis=1)
            ade = np.mean(pointwise_distances)
            
            # Final Displacement Error (FDE)
            fde = pointwise_distances[-1]
            
            # Average Lateral Error (perpendicular to direction of travel)
            lateral_errors = np.abs(pred_cp[:, 1] - gt_cp[:, 1])
            avg_lateral_error = np.mean(lateral_errors)
            
            # Average Longitudinal Error (along direction of travel)
            longitudinal_errors = np.abs(pred_cp[:, 0] - gt_cp[:, 0])
            avg_longitudinal_error = np.mean(longitudinal_errors)
            
            # Path similarity (Hausdorff distance)
            try:
                hausdorff = max(
                    np.max(np.min(cdist(pred_cp, gt_cp), axis=1)),
                    np.max(np.min(cdist(gt_cp, pred_cp), axis=1))
                )
            except:
                hausdorff = np.max(pointwise_distances)
            
            # Cumulative trajectory error
            cumulative_error = np.sum(pointwise_distances)
            
            # Store trajectory metrics
            self.metrics['trajectory']['ADE'].append(ade)
            self.metrics['trajectory']['FDE'].append(fde)
            self.metrics['trajectory']['Avg_Lateral_Error'].append(avg_lateral_error)
            self.metrics['trajectory']['Avg_Longitudinal_Error'].append(avg_longitudinal_error)
            self.metrics['trajectory']['Hausdorff_Distance'].append(hausdorff)
            self.metrics['trajectory']['Cumulative_Error'].append(cumulative_error)
            self.metrics['trajectory']['Pointwise_Errors'].append(pointwise_distances)
            
            # 2. Speed Classification Metrics
            if pred_speed is not None:
                pred_speed_i = pred_speed[i].item() if isinstance(pred_speed[i], torch.Tensor) else pred_speed[i]
                gt_speed_i = gt_speed[i].item() if isinstance(gt_speed[i], torch.Tensor) else gt_speed[i]
                
                # Speed classification accuracy
                speed_correct = int(pred_speed_i == gt_speed_i)
                self.metrics['speed']['Accuracy'].append(speed_correct)
                
                # Speed class predictions
                self.metrics['speed']['Predicted_Classes'].append(pred_speed_i)
                self.metrics['speed']['True_Classes'].append(gt_speed_i)
                
                # If we have probabilities, compute confidence metrics
                if speed_probs is not None:
                    pred_prob = speed_probs[i, pred_speed_i].item()
                    true_class_prob = speed_probs[i, gt_speed_i].item()
                    
                    self.metrics['speed']['Prediction_Confidence'].append(pred_prob)
                    self.metrics['speed']['True_Class_Probability'].append(true_class_prob)
            
            # 3. Scenario-wise metrics
            if scenario is not None:
                scenario_key = str(scenario)
                self.metrics['scenario_wise'][scenario_key]['ADE'].append(ade)
                self.metrics['scenario_wise'][scenario_key]['FDE'].append(fde)
                if pred_speed is not None:
                    self.metrics['scenario_wise'][scenario_key]['Speed_Accuracy'].append(speed_correct)
    
    def compute_metrics(self):
        """Compute aggregated metrics from all collected data"""
        print("\n" + "="*60)
        print("Computing Final Metrics")
        print("="*60)
        
        # Initialize summary dict
        self.metrics['summary'] = {}
        
        # 1. Trajectory Metrics Summary
        trajectory_metrics = self.metrics['trajectory']
        if trajectory_metrics['ADE']:
            print("\n--- Trajectory Prediction Metrics ---")
            
            # Basic statistics
            for metric_name in ['ADE', 'FDE', 'Avg_Lateral_Error', 'Avg_Longitudinal_Error', 
                              'Hausdorff_Distance', 'Cumulative_Error']:
                values = trajectory_metrics[metric_name]
                if values:
                    self.metrics['summary'][f'{metric_name}_mean'] = np.mean(values)
                    self.metrics['summary'][f'{metric_name}_std'] = np.std(values)
                    self.metrics['summary'][f'{metric_name}_min'] = np.min(values)
                    self.metrics['summary'][f'{metric_name}_max'] = np.max(values)
                    self.metrics['summary'][f'{metric_name}_median'] = np.median(values)
                    
                    print(f"{metric_name}:")
                    print(f"  Mean: {self.metrics['summary'][f'{metric_name}_mean']:.4f}m")
                    print(f"  Std: {self.metrics['summary'][f'{metric_name}_std']:.4f}m")
                    print(f"  Min: {self.metrics['summary'][f'{metric_name}_min']:.4f}m")
                    print(f"  Max: {self.metrics['summary'][f'{metric_name}_max']:.4f}m")
                    print(f"  Median: {self.metrics['summary'][f'{metric_name}_median']:.4f}m")
            
            # Point-wise error analysis
            if trajectory_metrics['Pointwise_Errors']:
                pointwise_errors = np.array(trajectory_metrics['Pointwise_Errors'])  # (n_samples, 10)
                pointwise_means = np.mean(pointwise_errors, axis=0)
                pointwise_stds = np.std(pointwise_errors, axis=0)
                
                self.metrics['summary']['pointwise_means'] = pointwise_means.tolist()
                self.metrics['summary']['pointwise_stds'] = pointwise_stds.tolist()
                
                print("\nPoint-wise Distance Errors (mean ± std):")
                for i, (mean_val, std_val) in enumerate(zip(pointwise_means, pointwise_stds)):
                    print(f"  Waypoint {i+1}: {mean_val:.4f}m ± {std_val:.4f}m")
        
        # 2. Speed Classification Metrics Summary
        speed_metrics = self.metrics['speed']
        if speed_metrics.get('Accuracy'):
            print("\n--- Speed Classification Metrics ---")
            
            accuracy = np.mean(speed_metrics['Accuracy']) * 100
            self.metrics['summary']['speed_accuracy'] = accuracy
            print(f"Accuracy: {accuracy:.2f}%")
            
            if speed_metrics.get('Predicted_Classes') and speed_metrics.get('True_Classes'):
                # Basic classification metrics
                pred_classes = speed_metrics['Predicted_Classes']
                true_classes = speed_metrics['True_Classes']
                
                # Calculate additional metrics
                f1 = f1_score(true_classes, pred_classes, average='weighted') * 100
                self.metrics['summary']['speed_f1_score'] = f1
                print(f"F1 Score: {f1:.2f}%")
                
                # Confusion matrix
                cm = confusion_matrix(
                    true_classes,
                    pred_classes,
                    labels=range(len(self.config.target_speeds))
                )
                
                self.metrics['summary']['confusion_matrix'] = cm.tolist()
                
                # Calculate precision and recall per class
                tp = np.diag(cm)
                fp = np.sum(cm, axis=0) - tp
                fn = np.sum(cm, axis=1) - tp
                
                precision_per_class = tp / (tp + fp + 1e-8)
                recall_per_class = tp / (tp + fn + 1e-8)
                f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)
                
                print("\nPer-class Metrics:")
                target_names = [f'{speed}m/s' for speed in self.config.target_speeds]
                for idx, (speed_name, prec, rec, f1_val) in enumerate(zip(target_names, 
                                                                         precision_per_class, 
                                                                         recall_per_class, 
                                                                         f1_per_class)):
                    print(f"  {speed_name}:")
                    print(f"    Precision: {prec:.3f}")
                    print(f"    Recall: {rec:.3f}")
                    print(f"    F1-Score: {f1_val:.3f}")
                    print(f"    Support: {cm[idx].sum()}")
                
                # Confidence analysis
                if speed_metrics.get('Prediction_Confidence'):
                    avg_confidence = np.mean(speed_metrics['Prediction_Confidence'])
                    avg_true_prob = np.mean(speed_metrics['True_Class_Probability'])
                    
                    self.metrics['summary']['avg_prediction_confidence'] = avg_confidence
                    self.metrics['summary']['avg_true_class_probability'] = avg_true_prob
                    
                    print(f"\nAverage Prediction Confidence: {avg_confidence:.3f}")
                    print(f"Average True Class Probability: {avg_true_prob:.3f}")
        
        # 3. Scenario-wise Analysis
        if self.metrics['scenario_wise']:
            print("\n--- Scenario-wise Analysis ---")
            
            scenario_results = {}
            for scenario, metrics in self.metrics['scenario_wise'].items():
                if metrics['ADE']:
                    scenario_ade = np.mean(metrics['ADE'])
                    scenario_fde = np.mean(metrics['FDE'])
                    
                    if 'Speed_Accuracy' in metrics:
                        scenario_speed_acc = np.mean(metrics['Speed_Accuracy']) * 100
                    else:
                        scenario_speed_acc = None
                    
                    scenario_results[scenario] = {
                        'ADE': scenario_ade,
                        'FDE': scenario_fde,
                        'Speed_Accuracy': scenario_speed_acc,
                        'Num_Samples': len(metrics['ADE'])
                    }
                    
                    print(f"\nScenario: {scenario}")
                    print(f"  Num Samples: {len(metrics['ADE'])}")
                    print(f"  ADE: {scenario_ade:.4f}m")
                    print(f"  FDE: {scenario_fde:.4f}m")
                    if scenario_speed_acc is not None:
                        print(f"  Speed Accuracy: {scenario_speed_acc:.2f}%")
            
            self.metrics['summary']['scenario_results'] = scenario_results
        
        # 4. Overall Score (weighted combination)
        if 'ADE_mean' in self.metrics['summary'] and 'speed_accuracy' in self.metrics['summary']:
            # Normalize ADE (assuming typical range 0-10m)
            normalized_ade = min(1.0, self.metrics['summary']['ADE_mean'] / 10.0)
            speed_acc_normalized = self.metrics['summary']['speed_accuracy'] / 100.0
            
            # Weighted score (adjust weights as needed)
            trajectory_weight = 0.7
            speed_weight = 0.3
            
            overall_score = (1 - normalized_ade) * trajectory_weight + speed_acc_normalized * speed_weight
            self.metrics['summary']['overall_score'] = overall_score
            
            print("\n" + "="*60)
            print(f"Overall Performance Score: {overall_score:.4f}")
            print("="*60)

        # 5. BEV Semantic Segmentation Metrics
        print("\n--- BEV Semantic Segmentation Metrics ---")
        
        bev_results = self.bev_miou_calculator.get_results()
        
        self.metrics['summary']['bev_miou'] = bev_results['mIoU']
        self.metrics['summary']['bev_iou_per_class'] = bev_results['IoU_per_class']
        
        print(f"BEV mIoU: {bev_results['mIoU']:.4f}")
        
        # Print per-class IoU if you have class names
        print("\nBEV Per-class IoU:")
        for i, iou in enumerate(bev_results['IoU_per_class']):
            if not np.isnan(iou) and iou>0:
                print(f"  Class {i}: {iou:.4f}")
        if hasattr(self.config, 'bev_class_names') and self.config.bev_class_names:
            print("\nBEV Per-class IoU:")
            for i, (class_name, iou) in enumerate(zip(self.config.bev_class_names, bev_results['IoU_per_class'])):
                if not np.isnan(iou) and iou>0:
                    print(f"  {class_name}: {iou:.4f}")
        else:
            print("\nBEV Per-class IoU:")
            for i, iou in enumerate(bev_results['IoU_per_class']):
                if not np.isnan(iou) and iou>0:
                    print(f"  Class {i}: {iou:.4f}")
    
    def save_results(self):
        """Save evaluation results to files"""
        # Save metrics as JSON
        metrics_file = self.output_dir / "evaluation_metrics.json"
        
        # Helper function to convert numpy types
        def numpy_to_python(obj):
            if isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [numpy_to_python(item) for item in obj]
            elif isinstance(obj, defaultdict):
                return {k: numpy_to_python(v) for k, v in obj.items()}
            else:
                return obj
        
        # Convert all numpy types in metrics
        serializable_metrics = numpy_to_python(self.metrics)
        
        # with open(metrics_file, 'w') as f:
        #     json.dump(serializable_metrics, f, indent=2)
        
        # print(f"\nMetrics saved to: {metrics_file}")
        
        # Save summary as CSV
        if 'summary' in self.metrics:
            summary_data = {}
            for key, value in self.metrics['summary'].items():
                if not isinstance(value, (list, dict)):  # Only simple values for CSV
                    if isinstance(value, (np.float32, np.float64)):
                        summary_data[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        summary_data[key] = int(value)
                    else:
                        summary_data[key] = value
            
            if summary_data:
                summary_df = pd.DataFrame([summary_data])
                summary_file = self.output_dir / "summary.csv"
                summary_df.to_csv(summary_file, index=False)
                print(f"Summary saved to: {summary_file}")

        # Add BEV semantic results to the serialized metrics
        if hasattr(self, 'bev_miou_calculator'):
            bev_results = self.bev_miou_calculator.get_results()
            serializable_metrics['bev_semantic'] = bev_results
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_file}")
    
    def visualize_results(self):
        """Generate visualization plots"""
        print("\nGenerating visualizations...")
        
        # Create figures directory
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Trajectory Error Distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # ADE distribution
        if self.metrics['trajectory']['ADE']:
            axes[0, 0].hist(self.metrics['trajectory']['ADE'], bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_xlabel('ADE (m)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Average Displacement Error Distribution')
            axes[0, 0].axvline(np.mean(self.metrics['trajectory']['ADE']), color='r', 
                              linestyle='--', label=f'Mean: {np.mean(self.metrics["trajectory"]["ADE"]):.3f}m')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # FDE distribution
        if self.metrics['trajectory']['FDE']:
            axes[0, 1].hist(self.metrics['trajectory']['FDE'], bins=50, alpha=0.7, color='lightcoral')
            axes[0, 1].set_xlabel('FDE (m)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Final Displacement Error Distribution')
            axes[0, 1].axvline(np.mean(self.metrics['trajectory']['FDE']), color='r',
                              linestyle='--', label=f'Mean: {np.mean(self.metrics["trajectory"]["FDE"]):.3f}m')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Lateral vs Longitudinal error
        if (self.metrics['trajectory'].get('Avg_Lateral_Error') and 
            self.metrics['trajectory'].get('Avg_Longitudinal_Error')):
            axes[0, 2].scatter(self.metrics['trajectory']['Avg_Lateral_Error'], 
                              self.metrics['trajectory']['Avg_Longitudinal_Error'], 
                              alpha=0.5)
            axes[0, 2].set_xlabel('Average Lateral Error (m)')
            axes[0, 2].set_ylabel('Average Longitudinal Error (m)')
            axes[0, 2].set_title('Lateral vs Longitudinal Error')
            axes[0, 2].axhline(0, color='gray', linestyle='--', alpha=0.3)
            axes[0, 2].axvline(0, color='gray', linestyle='--', alpha=0.3)
            axes[0, 2].grid(True, alpha=0.3)
        
        # Point-wise error progression
        if 'pointwise_means' in self.metrics['summary']:
            waypoint_indices = range(1, len(self.metrics['summary']['pointwise_means']) + 1)
            means = self.metrics['summary']['pointwise_means']
            stds = self.metrics['summary']['pointwise_stds']
            
            axes[1, 0].errorbar(waypoint_indices, means, yerr=stds, 
                               fmt='o-', capsize=5, capthick=2)
            axes[1, 0].set_xlabel('Waypoint Index')
            axes[1, 0].set_ylabel('Distance Error (m)')
            axes[1, 0].set_title('Error Progression Along Trajectory')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Speed confusion matrix
        if 'confusion_matrix' in self.metrics['summary']:
            cm = np.array(self.metrics['summary']['confusion_matrix'])
            target_names = [f'{round(speed,2)}m/s' for speed in self.config.target_speeds]
            
            im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1, 1].set_title('Speed Classification Confusion Matrix')
            
            # Add labels
            tick_marks = np.arange(len(target_names))
            axes[1, 1].set_xticks(tick_marks)
            axes[1, 1].set_xticklabels(target_names, rotation=45)
            axes[1, 1].set_yticks(tick_marks)
            axes[1, 1].set_yticklabels(target_names)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black")
            
            plt.colorbar(im, ax=axes[1, 1])
        
        # Empty plot for sample trajectory info
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, 'Sample trajectories\n(saved separately)',
                       ha='center', va='center', fontsize=12)
        
        plt.suptitle('Traj&Speed Evaluation Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(figures_dir / "evaluation_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Generate sample trajectory plots
        if len(self.predictions) > 0:
            self.plot_sample_trajectories()

        # Add BEV semantic visualization
        self.plot_bev_semantic_samples()    
        
        print(f"Visualizations saved to: {figures_dir}")

    def plot_bev_semantic_samples(self, num_samples=5):
        """Plot sample BEV semantic predictions vs ground truth"""
        if not hasattr(self, 'bev_miou_calculator'):
            return
        
        # Find samples that have BEV predictions and ground truth
        bev_samples = []
        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            if pred.get('bev_semantic') is not None and gt.get('bev_semantic') is not None:
                bev_samples.append((i, pred, gt))
        
        if not bev_samples:
            return
        
        num_to_plot = min(num_samples, len(bev_samples))
        # indices = np.random.choice(len(bev_samples), num_to_plot, replace=False)
        # 使用固定种子生成"伪随机"索引
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(len(bev_samples), num_to_plot, replace=False)
        
        fig, axes = plt.subplots(num_to_plot, 3, figsize=(15, 5*num_to_plot))
        
        # If only one sample, axes is 1D
        if num_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        for plot_idx, sample_idx in enumerate(indices):
            idx, pred, gt = bev_samples[sample_idx]
            
            pred_bev = pred['bev_semantic']
            gt_bev = gt['bev_semantic']
            
            # Calculate sample IoU
            mask = gt_bev != -1
            if mask.any():
                pred_valid = pred_bev[mask]
                gt_valid = gt_bev[mask]
                
                # Compute IoU for this sample
                iou_per_class_sample = []
                for class_idx in range(self.bev_miou_calculator.num_classes):
                    tp = ((pred_valid == class_idx) & (gt_valid == class_idx)).sum()
                    fp = ((pred_valid == class_idx) & (gt_valid != class_idx)).sum()
                    fn = ((pred_valid != class_idx) & (gt_valid == class_idx)).sum()
                    
                    if tp + fp + fn > 0:
                        iou_per_class_sample.append(tp / (tp + fp + fn))
                
                sample_miou = np.mean(iou_per_class_sample) if iou_per_class_sample else 0
            
            # Plot ground truth
            axes[plot_idx, 0].imshow(gt_bev, cmap='tab20c', vmin=0, vmax=self.bev_miou_calculator.num_classes-1)
            axes[plot_idx, 0].set_title(f'Sample {idx} - Ground Truth')
            axes[plot_idx, 0].axis('off')
            
            # Plot prediction
            axes[plot_idx, 1].imshow(pred_bev, cmap='tab20c', vmin=0, vmax=self.bev_miou_calculator.num_classes-1)
            axes[plot_idx, 1].set_title(f'Sample {idx} - Prediction')
            axes[plot_idx, 1].axis('off')
            
            # Plot error map
            error = np.zeros_like(gt_bev, dtype=bool)
            if mask.any():
                error[mask] = pred_bev[mask] != gt_bev[mask]
            
            axes[plot_idx, 2].imshow(error, cmap='Reds')
            axes[plot_idx, 2].set_title(f'Sample {idx} - Errors (mIoU: {sample_miou:.3f})')
            axes[plot_idx, 2].axis('off')
        
        # plt.suptitle('BEV Semantic Segmentation Samples', fontsize=16)
        plt.tight_layout()
        
        figures_dir = self.output_dir / "figures"
        plt.savefig(figures_dir / "bev_semantic_samples.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_sample_trajectories(self, num_samples=20):
        """Plot sample trajectories with predictions vs ground truth"""
        num_to_plot = min(num_samples, len(self.predictions))
        # indices = np.random.choice(len(self.predictions), num_to_plot, replace=False)
        # 使用固定种子生成"伪随机"索引
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(len(self.predictions), num_to_plot, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, plot_idx in enumerate(indices):
            if idx >= len(axes):
                break
                
            pred = self.predictions[plot_idx]
            gt = self.ground_truths[plot_idx]
            
            # Plot ground truth trajectory
            gt_points = gt['checkpoints']
            axes[idx].plot(gt_points[:, 0], gt_points[:, 1], 'g-', linewidth=2, label='Ground Truth')
            axes[idx].scatter(gt_points[:, 0], gt_points[:, 1], c='g', s=30, marker='o')
            
            # Plot predicted trajectory
            pred_points = pred['checkpoints']
            axes[idx].plot(pred_points[:, 0], pred_points[:, 1], 'b--', linewidth=2, label='Predicted')
            axes[idx].scatter(pred_points[:, 0], pred_points[:, 1], c='b', s=30, marker='s')
            
            # Plot start point
            axes[idx].scatter(0, 0, c='r', s=100, marker='*', label='Start')
            
            # Add error text
            ade = np.mean(np.linalg.norm(pred_points - gt_points, axis=1))
            fde = np.linalg.norm(pred_points[-1] - gt_points[-1])
            
            axes[idx].text(0.05, 0.95, f'ADE: {ade:.2f}m\nFDE: {fde:.2f}m',
                          transform=axes[idx].transAxes, fontsize=9,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add speed info if available
            if 'speed_class' in pred and pred['speed_class'] is not None:
                pred_speed = self.config.target_speeds[pred['speed_class']]
                gt_speed = self.config.target_speeds[gt['speed_class']]
                
                speed_text = f'Spd: P{pred_speed}m/s, T{gt_speed}m/s'
                if pred_speed == gt_speed:
                    speed_color = 'green'
                else:
                    speed_color = 'red'
                
                axes[idx].text(0.05, 0.05, speed_text,
                             transform=axes[idx].transAxes, fontsize=8,
                             verticalalignment='bottom', color=speed_color,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            axes[idx].set_aspect('equal', adjustable='box')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlabel('X (m)')
            axes[idx].set_ylabel('Y (m)')
            
            if idx == 0:
                axes[idx].legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'Sample Trajectory Predictions (n={num_to_plot})', fontsize=16)
        plt.tight_layout()
        
        figures_dir = self.output_dir / "figures"
        plt.savefig(figures_dir / "sample_trajectories.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Open-loop evaluation of trajectory prediction model')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Directory containing model checkpoints and config')
    parser.add_argument('--model_file', type=str, default=None,
                       help='Specific model file to load (e.g., model_0100.pth). If None, loads latest.')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of validation dataset')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run evaluation on')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.logdir) / 'config.json'  # using the config file in the /log
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'rt', encoding='utf-8') as f:
        json_config = f.read()
    
    # Initialize config
    loaded_config = json.loads(json_config)
    config = GlobalConfig()
    config.__dict__.update(loaded_config)
    
    # Update config with evaluation settings
    config.batch_size = args.batch_size
    
    # Find model file
    if args.model_file:
        model_path = Path(args.logdir) / args.model_file
    else:
        # Find latest model file (following agent pattern but loading just one)
        model_files = list(Path(args.logdir).glob('model_*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {args.logdir}")
        
        # Extract epoch numbers and find latest
        def extract_epoch(f):
            try:
                return int(''.join(filter(str.isdigit, f.stem)))
            except:
                return -1
        
        model_files.sort(key=extract_epoch, reverse=True)
        model_path = model_files[0]
    
    print(f"Loading model from: {model_path}")
    print(f"Using config from: {config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Create evaluator
    evaluator = OpenLoopEvaluator(
        config=config,
        model_path=model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    metrics = evaluator.run_evaluation()
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    if 'summary' in metrics:
        if 'overall_score' in metrics['summary']:
            print(f"\nOverall Score: {metrics['summary']['overall_score']:.4f}")
        
        if 'ADE_mean' in metrics['summary']:
            print(f"Trajectory ADE: {metrics['summary']['ADE_mean']:.4f} ± {metrics['summary']['ADE_std']:.4f}m")
            print(f"Trajectory FDE: {metrics['summary']['FDE_mean']:.4f} ± {metrics['summary']['FDE_std']:.4f}m")
        
        if 'speed_accuracy' in metrics['summary']:
            print(f"Speed Accuracy: {metrics['summary']['speed_accuracy']:.2f}%")
            if 'speed_f1_score' in metrics['summary']:
                print(f"Speed F1 Score: {metrics['summary']['speed_f1_score']:.2f}%")

        if 'bev_miou' in metrics['summary']:
            print(f"\nBEV Semantic mIoU: {metrics['summary']['bev_miou']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()