'''
Training script for training transFuser and related models.
Usage:
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d
train.py --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --id exp_000 --cpu_cores 8
'''

import argparse
import copy
import json
import os
import pathlib
import datetime
import random
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
from diskcache import Cache
import torchmetrics

from config import GlobalConfig
from my_model_qtd import LidarCenterNet
# from data import CARLA_Data
from ability_data import Ability_CARLA_Data
from plant import PlanT
from forgetting_monitor_v2 import ForgettingMonitor
from oracle_kd_loss import OracleKDLoss

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)

# On some systems it is necessary to increase the limit on open file descriptors.
try:
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except (ModuleNotFoundError, ImportError) as e:
  print(e)

# def load_checkpoint_with_anchor_fix(model, checkpoint_path, device, strict=False):
#     """加载检查点，自动处理锚点不匹配问题"""
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model_state_dict = model.state_dict()
#     checkpoint_state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model_state_dict', checkpoint)
    
#     # 过滤掉不匹配的键
#     filtered_state_dict = {}
#     for key, value in checkpoint_state_dict.items():
#         if key in model_state_dict:
#             if model_state_dict[key].shape == value.shape:
#                 filtered_state_dict[key] = value
#             else:
#                 print(f"跳过不匹配的参数: {key} | 检查点形状: {value.shape} | 模型形状: {model_state_dict[key].shape}")
#         else:
#             print(f"跳过不存在的参数: {key}")
    
#     # 加载过滤后的状态字典
#     model.load_state_dict(filtered_state_dict, strict=False)
#     return model

def load_checkpoint_ignore_anchors(model, checkpoint_path, device, strict=False):
    """加载检查点，只跳过anchor数据本身，保留相关网络参数"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model_state_dict', checkpoint)
    
    # 过滤状态字典
    filtered_state_dict = {}
    
    for key, value in checkpoint_state_dict.items():
        # 只跳过anchor数据本身，不跳过anchor相关的网络参数
        if key.endswith('.anchors') and 'anchors' in key:
            print(f"跳过anchor数据: {key} | 检查点形状: {value.shape} | 模型形状: {model_state_dict[key].shape}")
            continue
            
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"跳过不匹配的参数: {key} | 检查点形状: {value.shape} | 模型形状: {model_state_dict[key].shape}")
        else:
            print(f"跳过不存在的参数: {key}")
    
    # 加载过滤后的状态字典
    model.load_state_dict(filtered_state_dict, strict=False)
    return model


def _select_top1_trajectory(pred_trajectories, pred_traj_probs):
  if pred_trajectories is None or pred_traj_probs is None:
    return None
  best_anchor_indices = torch.argmax(pred_traj_probs, dim=0)
  batch_indices = torch.arange(pred_traj_probs.size(1), device=pred_trajectories.device)
  return pred_trajectories[best_anchor_indices, batch_indices]


def unwrap_model(model):
  while hasattr(model, '_orig_mod'):
    model = model._orig_mod
  while hasattr(model, 'module'):
    model = model.module
  return model


def _compute_speed_acc_from_logits(speed_logits, target_speed):
  if speed_logits is None:
    return None
  if target_speed.dtype in (torch.float16, torch.float32, torch.float64):
    target_idx = torch.argmax(target_speed, dim=1)
  else:
    target_idx = target_speed
  pred_idx = torch.argmax(speed_logits, dim=1)
  return (pred_idx == target_idx).float()


def _compute_traj_score(pred_trajectories, pred_traj_probs, checkpoint, config):
  top1_traj = _select_top1_trajectory(pred_trajectories, pred_traj_probs)
  if top1_traj is None:
    return None
  with torch.no_grad():
    traj_l1 = torch.norm(top1_traj - checkpoint, dim=-1).sum(dim=-1)  # (B,)
    traj_score = 1.0 - (traj_l1 / float(config.a3d_traj_threshold))
    traj_score = torch.clamp(traj_score, min=0.0, max=1.0)
  return traj_score


def load_config_from_training_dir(config_dir):
  config_path = os.path.join(config_dir, 'config.json')
  if not os.path.isfile(config_path):
    raise FileNotFoundError(f'Cannot find reference config.json at: {config_path}')
  with open(config_path, 'rt', encoding='utf-8') as f:
    json_config = f.read()
  loaded_config = jsonpickle.decode(json_config)
  ref_config = GlobalConfig()
  ref_config.__dict__.update(loaded_config.__dict__)
  ref_config.use_traj_front_door_encoder = False
  ref_config.use_prior_fuseFeat = False
  return ref_config


def load_ref_model_from_dir(ref_dir, device):
  if not os.path.isdir(ref_dir):
    raise ValueError(f'--oracle_ref_file must be a directory containing config.json and model_*.pth, got: {ref_dir}')
  ref_config = load_config_from_training_dir(ref_dir)

  ckpt_files = []
  for file in os.listdir(ref_dir):
    if file.endswith('.pth') and file.startswith('model'):
      ckpt_files.append(file)
  if not ckpt_files:
    raise FileNotFoundError(f'No model_*.pth found in reference dir: {ref_dir}')
  ckpt_files = sorted(ckpt_files)
  ref_ckpt = os.path.join(ref_dir, ckpt_files[-1])

  ref_model = LidarCenterNet(ref_config)
  ref_model.cuda(device=device)
  ref_model = load_checkpoint_ignore_anchors(ref_model, ref_ckpt, device)
  ref_model.eval()
  ref_model.requires_grad_(False)
  return ref_model, ref_ckpt


def _safe_kl(student_logits, teacher_logits, temperature):
  s_log = F.log_softmax(student_logits / temperature, dim=-1)
  t_prob = F.softmax(teacher_logits / temperature, dim=-1)
  return F.kl_div(s_log, t_prob, reduction='batchmean') * (temperature**2)


def _safe_kl_with_target_probs(student_logits, target_probs, temperature):
  """KL with externally prepared target probs on the same support as student_logits."""
  s_log = F.log_softmax(student_logits / temperature, dim=-1)
  t_prob = torch.clamp(target_probs, min=1e-8)
  t_prob = t_prob / t_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
  return F.kl_div(s_log, t_prob, reduction='batchmean') * (temperature**2)


def _align_teacher_traj_prob_to_student(student_traj, teacher_traj, teacher_prob):
  """
  Align teacher trajectory distribution to student anchors using soft geometry matching.
  student_traj: (B, Ks, T, 2), teacher_traj: (B, Kt, T, 2), teacher_prob: (B, Kt)
  """
  bsz, _, t_steps, xy_dim = student_traj.shape
  s_flat = student_traj.reshape(bsz, student_traj.size(1), -1)
  t_flat = teacher_traj.reshape(bsz, teacher_traj.size(1), -1)

  # (B, Ks, Kt): average per-point L1 between every student-teacher trajectory pair.
  pairwise_l1 = torch.cdist(s_flat, t_flat, p=1) / float(t_steps * xy_dim)

  # Batch-adaptive temperature keeps matching stable across scenes/speeds.
  match_temp = pairwise_l1.detach().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
  assign_s_given_t = F.softmax(-pairwise_l1 / match_temp, dim=1)
  assign_s_given_t = assign_s_given_t.detach()

  teacher_prob_on_student = torch.einsum('bsk,bk->bs', assign_s_given_t, teacher_prob)
  teacher_prob_on_student = torch.clamp(teacher_prob_on_student, min=1e-8)
  teacher_prob_on_student = teacher_prob_on_student / teacher_prob_on_student.sum(dim=-1,
                                                                                   keepdim=True).clamp_min(1e-8)

  # For each teacher anchor, expected matched distance on student anchors.
  expected_teacher_to_student_l1 = torch.sum(assign_s_given_t * pairwise_l1, dim=1)
  return teacher_prob_on_student, expected_teacher_to_student_l1


def build_ability_monitor_loaders(config, shared_dict, rank):
  if rank != 0 or not bool(getattr(config, 'use_ability_open_loop_monitor', 0)):
    return None

  abilities = []
  for ability in getattr(config, 'selected_ability_list', []):
    if ability == 'No_Scenario' or ability in abilities:
      continue
    abilities.append(ability)

  if not abilities:
    print('Ability open-loop monitor enabled but no non-No_Scenario abilities were selected.', flush=True)
    return None

  split = str(getattr(config, 'ability_monitor_split', 'train')).lower()
  validation = split in ('val', 'valid', 'validation')
  root_override = str(getattr(config, 'ability_monitor_root', '')).strip()
  root = root_override if root_override else (config.mini_dataset_root if validation else config.mini_dataset_root)

  monitor_config = copy.copy(config)
  if bool(getattr(config, 'ability_monitor_disable_augment', 1)):
    monitor_config.augment = False
    monitor_config.augment_percentage = 0.0
    monitor_config.use_color_aug = False
    monitor_config.use_cutout = False
  monitor_config.use_semantic = False
  monitor_config.use_bev_semantic = False
  monitor_config.use_depth = False
  monitor_config.detect_boxes = False
  monitor_config.use_traj_front_door_encoder = False
  monitor_config.use_prior_fuseFeat = False

  batch_size = max(1, int(getattr(config, 'ability_monitor_batch_size', 2)))
  num_workers = max(0, int(getattr(config, 'ability_monitor_num_workers', 0)))
  seed = getattr(config, 'seed', None)
  seed = 0 if seed is None else int(seed)

  loaders = {}
  for ability_idx, ability in enumerate(abilities):
    dataset = Ability_CARLA_Data(root=root,
                                 config=monitor_config,
                                 estimate_class_distributions=False,
                                 estimate_sem_distribution=False,
                                 shared_dict=None,
                                 rank=rank,
                                 validation=validation,
                                 ability=ability)
    if len(dataset) == 0:
      print(f'Ability open-loop monitor skipped {ability}: dataset is empty.', flush=True)
      continue

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 1729 + ability_idx)
    loaders[ability] = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  worker_init_fn=seed_worker if num_workers > 0 else None,
                                  generator=generator,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  drop_last=False)

  if loaders:
    print(f'Ability open-loop monitor enabled for abilities: {list(loaders.keys())}', flush=True)
  else:
    print('Ability open-loop monitor enabled but no monitor loaders were created.', flush=True)
  return loaders

@record  # Records error and tracebacks in case of failure
def main():
  torch.cuda.empty_cache()

  # Loads the default values for the argparse so we have only one default
  config = GlobalConfig()
  # print(config.forcast_time)
  # print(config.tf_de_dim)

  parser = argparse.ArgumentParser()
  parser.add_argument('--id', type=str, default=config.id, help='Unique experiment identifier.')
  parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of train epochs.')
  parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=config.batch_size,
                      help='Batch size for one GPU. When training with multiple GPUs the effective'
                      ' batch size will be batch_size*num_gpus')
  parser.add_argument('--logdir', type=str, required=True, help='Directory to log data and models to.')
  parser.add_argument('--load_file',
                      type=str,
                      default=config.load_file,
                      help='Model to load for initialization.'
                      'Expects the full path with ending /path/to/model.pth '
                      'Optimizer files are expected to exist in the same directory')
  parser.add_argument('--setting',
                      type=str,
                      default=config.setting,
                      help='What training setting to use. Options: '
                      'all: Train on all towns no validation data. '
                      '13_withheld: Do not train on Town 13. '
                      '12_only: Only trains with data from Town 12 '
                      'Withheld data is used for validation')
  parser.add_argument('--root_dir', type=str, required=True, nargs='+', help='Root directory of your training data')
  parser.add_argument('--schedule_reduce_epoch_01',
                      type=int,
                      default=config.schedule_reduce_epoch_01,
                      help='Epoch at which to reduce the lr by a factor of 10 the first '
                      'time. Only used with --schedule 1')
  parser.add_argument('--schedule_reduce_epoch_02',
                      type=int,
                      default=config.schedule_reduce_epoch_02,
                      help='Epoch at which to reduce the lr by a factor of 10 the second '
                      'time. Only used with --schedule 1')
  parser.add_argument('--backbone',
                      type=str,
                      default=config.backbone,
                      help='Which fusion backbone to use. Options: transFuser, aim, bev_encoder')
  parser.add_argument(
      '--image_architecture',
      type=str,
      default=config.image_architecture,
      help='Which architecture to use for the image branch. resnet34, regnety_032, hf-hub:apple/mobileclip_s0_timm etc.'
      'All options of the TIMM lib can be used but some might need adjustments to the backbone.')
  parser.add_argument('--lidar_architecture',
                      type=str,
                      default=config.lidar_architecture,
                      help='Which architecture to use for the lidar branch. Tested: resnet34, regnety_032.'
                      'Has the special video option video_resnet18 and video_swin_tiny.')
  parser.add_argument('--use_velocity',
                      type=int,
                      default=config.use_velocity,
                      help='Whether to use the velocity input. Expected values are 0:False, 1:True')
  parser.add_argument('--n_layer',
                      type=int,
                      default=config.n_layer,
                      help='Number of transformer layers used in the transfuser')
  parser.add_argument('--val_every', type=int, default=config.val_every, help='At which epoch frequency to validate.')
  parser.add_argument('--sync_batch_norm',
                      type=int,
                      default=config.sync_batch_norm,
                      help='0: Compute batch norm for each GPU independently, 1: Synchronize batch norms across GPUs.')
  parser.add_argument('--zero_redundancy_optimizer',
                      type=int,
                      default=config.zero_redundancy_optimizer,
                      help='0: Normal AdamW Optimizer, 1: Use zero-redundancy Optimizer to reduce memory footprint.')
  parser.add_argument('--use_disk_cache',
                      type=int,
                      default=config.use_disk_cache,
                      help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH '
                      'environment variable. Useful if the dataset is stored on shared slow filesystem and can be '
                      'temporarily stored on faster SSD storage on the compute node.')
  parser.add_argument('--lidar_seq_len',
                      type=int,
                      default=config.lidar_seq_len,
                      help='How many temporal frames in the LiDAR to use. 1 equals single timestep.')
  parser.add_argument('--realign_lidar',
                      type=int,
                      default=int(config.realign_lidar),
                      help='Whether to realign the temporal LiDAR frames, to all lie in the same coordinate frame.')
  parser.add_argument('--use_ground_plane',
                      type=int,
                      default=int(config.use_ground_plane),
                      help='Whether to use the ground plane of the LiDAR. Only affects methods using the LiDAR.')
  parser.add_argument('--use_controller_input_prediction',
                      type=int,
                      default=int(config.use_controller_input_prediction),
                      help='Whether to classify target speeds and regress a path as output representation.')
  parser.add_argument('--use_wp_gru',
                      type=int,
                      default=int(config.use_wp_gru),
                      help='Whether to predict the waypoint output representation.')
  parser.add_argument('--pred_len', type=int, default=config.pred_len, help='Number of waypoints the model predicts')
  parser.add_argument('--estimate_class_distributions',
                      type=int,
                      default=int(config.estimate_class_distributions),
                      help='# Whether to estimate the weights to re-balance CE loss, or use the config default.')
  parser.add_argument('--use_focal_loss',
                      type=int,
                      default=int(config.use_focal_loss),
                      help='# Whether to use focal loss instead of cross entropy for target speed classification.')
  parser.add_argument('--use_cosine_schedule',
                      type=int,
                      default=int(config.use_cosine_schedule),
                      help='Whether to use a cyclic cosine learning rate schedule instead of the linear one.')
  parser.add_argument('--augment',
                      type=int,
                      default=int(config.augment),
                      help='# Whether to use rotation and translation augmentation')
  parser.add_argument('--use_plant',
                      type=int,
                      default=int(config.use_plant),
                      help='If true trains a privileged PlanT model, otherwise a sensorimotor agent like TF++')
  parser.add_argument('--learn_origin',
                      type=int,
                      default=int(config.learn_origin),
                      help='Whether to learn the origin of the waypoints or use 0/0')
  parser.add_argument('--local_rank',
                      type=int,
                      default=int(config.local_rank),
                      help='Local rank for launch with torch.launch. Default = -999 means not used.')
  parser.add_argument('--train_sampling_rate',
                      type=int,
                      default=int(config.train_sampling_rate),
                      help='Rate at which the dataset is sub-sampled during training.'
                      'Should be an odd number ideally ending with 1 or 5, because of the LiDAR sweeps alternating '
                      'every frame')
  parser.add_argument('--use_amp',
                      type=int,
                      default=int(config.use_amp),
                      help='Currently amp produces inf gradients. DO NOT USE!.'
                      'Whether to use automatic mixed precision with fp16 during training.')
  parser.add_argument('--use_grad_clip',
                      type=int,
                      default=int(config.use_grad_clip),
                      help='Whether to clip the gradients during training.')
  parser.add_argument('--use_color_aug',
                      type=int,
                      default=int(config.use_color_aug),
                      help='Whether to use color augmentation on the images.')
  parser.add_argument('--use_semantic',
                      type=int,
                      default=int(config.use_semantic),
                      help='Whether to use semantic segmentation as auxiliary loss')
  parser.add_argument('--use_depth',
                      type=int,
                      default=int(config.use_depth),
                      help='Whether to use depth prediction as auxiliary loss for training.')
  parser.add_argument('--detect_boxes',
                      type=int,
                      default=int(config.detect_boxes),
                      help='Whether to use the bounding box auxiliary task.')
  parser.add_argument('--use_bev_semantic',
                      type=int,
                      default=int(config.use_bev_semantic),
                      help='Whether to use bev semantic segmentation as auxiliary loss for training.')
  parser.add_argument('--estimate_semantic_distribution',
                      type=int,
                      default=int(config.estimate_semantic_distribution),
                      help='Whether to estimate the weights to rebalance the semantic segmentation loss by class.'
                      'This is extremely slow.')
  parser.add_argument('--use_discrete_command',
                      type=int,
                      default=int(config.use_discrete_command),
                      help='Whether the discrete command is an input for the model.')
  parser.add_argument('--gru_hidden_size',
                      type=int,
                      default=int(config.gru_hidden_size),
                      help='Number of features used in the hidden size of the GRUs')
  parser.add_argument('--use_cutout',
                      type=int,
                      default=int(config.use_cutout),
                      help='Whether to use the cutout data augmentation technique.')
  parser.add_argument('--add_features',
                      type=int,
                      default=int(config.add_features),
                      help='Whether to add (or concatenate) the features at the end of the backbone.')
  parser.add_argument('--freeze_backbone',
                      type=int,
                      default=int(config.freeze_backbone),
                      help='Freezes the encoder and auxiliary heads. Should be used when loading a already trained '
                      'model. Can be used for fine-tuning or multi-stage training.')
  parser.add_argument('--learn_multi_task_weights',
                      type=int,
                      default=int(config.learn_multi_task_weights),
                      help='Whether to learn the multi-task weights according to https://arxiv.org/abs/1705.07115.')
  parser.add_argument('--transformer_decoder_join',
                      type=int,
                      default=int(config.transformer_decoder_join),
                      help='Whether to use a transformer decoder instead of global average pool + MLP for planning.')
  parser.add_argument('--bev_down_sample_factor',
                      type=int,
                      default=int(config.bev_down_sample_factor),
                      help='Factor (int) by which the bev auxiliary tasks are down-sampled.')
  parser.add_argument('--perspective_downsample_factor',
                      type=int,
                      default=int(config.perspective_downsample_factor),
                      help='Factor (int) by which the perspective auxiliary tasks are down-sampled.')
  parser.add_argument('--gru_input_size',
                      type=int,
                      default=int(config.gru_input_size),
                      help='Number of channels in the InterFuser GRU input and Transformer decoder.'
                      'Must be divisible by number of heads (8)')
  parser.add_argument('--num_repetitions',
                      type=int,
                      default=int(config.num_repetitions),
                      help='Our dataset consists of x repetitions of the same routes. '
                      'This specifies how many repetitions we will train with. Max 3, Min 1.')
  parser.add_argument('--bev_grid_height_downsample_factor',
                      type=int,
                      default=int(config.bev_grid_height_downsample_factor),
                      help='Ratio by which the height size of the voxel grid in BEV decoder are larger than width '
                      'and depth. Value should be >= 1. Larger values uses less gpu memory. '
                      'Only relevant for the bev_encoder backbone.')
  parser.add_argument('--wp_dilation',
                      type=int,
                      default=int(config.wp_dilation),
                      help='Factor by which the wp are dilated compared to full CARLA 20 FPS')
  parser.add_argument('--use_tp',
                      type=int,
                      default=int(config.use_tp),
                      help='Whether to use the target point as input to the network.')
  parser.add_argument('--continue_epoch',
                      type=int,
                      default=int(config.continue_epoch),
                      help='Whether to continue the training from the loaded epoch or from 0.')
  parser.add_argument('--max_height_lidar',
                      type=float,
                      default=float(config.max_height_lidar),
                      help='Points higher than this threshold are removed from the LiDAR.')
  parser.add_argument('--smooth_route',
                      type=int,
                      default=int(config.smooth_route),
                      help='Whether to smooth the route points with linear interpolation.')
  parser.add_argument('--use_speed_weights',
                      type=int,
                      default=int(config.use_speed_weights),
                      help='Whether to weight target speed classes.')
  parser.add_argument('--max_num_bbs',
                      type=int,
                      default=int(config.max_num_bbs),
                      help='Maximum number of bounding boxes our system can detect.')
  parser.add_argument('--use_optim_groups',
                      type=int,
                      default=int(config.use_optim_groups),
                      help='Whether to use optimizer groups to exclude some parameters from weight decay')
  parser.add_argument('--weight_decay',
                      type=float,
                      default=float(config.weight_decay),
                      help='Weight decay coefficient used during training')
  parser.add_argument('--use_label_smoothing',
                      type=int,
                      default=int(config.use_label_smoothing),
                      help='Whether to use label smoothing in the classification losses. '
                      'Not working as intended when combined with use_speed_weights.')
  parser.add_argument('--cpu_cores',
                      type=int,
                      required=True,
                      help='How many cpu cores are available on the machine.'
                      'The code will spawn a thread for each cpu.')
  parser.add_argument('--tp_attention',
                      type=int,
                      default=int(config.tp_attention),
                      help='Adds a TP at the TF decoder and computes it with attention visualization. '
                      'Only compatible with transformer decoder.')
  parser.add_argument('--multi_wp_output',
                      type=int,
                      default=int(config.multi_wp_output),
                      help='Predict 2 WP outputs and select between them. '
                      'Only compatible with use_wp=1, transformer_decoder_join=1')
  parser.add_argument('--seed',
                      type=int,
                      default=None,
                      help='Specify a torch seed for reproducible model initialization. '
                      'Training will still not be reproducible because of non-deterministic PyTorch operations.')
  parser.add_argument('--crop_image',
                      type=int,
                      default=int(config.crop_image),
                      help='Whether to crop the image to the dimensions specified in the config.')
  parser.add_argument('--input_path_to_target_speed_network',
                      type=int,
                      default=int(config.input_path_to_target_speed_network),
                      help='Whether to input the predicted checkpoints to the target speed network.')
  parser.add_argument('--predict_checkpoint_len',
                      type=int,
                      default=int(config.predict_checkpoint_len),
                      help='Number of checkpoints to be predicted by the GRU in the disentangled output '
                      'representation.')
  parser.add_argument('--max_x',
                      type=int,
                      default=int(config.max_x),
                      help='BEV range in front of the vehicle in meters')
  parser.add_argument('--crop_bev_height_only_from_behind',
                      type=int,
                      default=int(config.crop_bev_height_only_from_behind),
                      help='If true, cuts BEV off behind the vehicle. If False, cuts off front and back symetrically')
  parser.add_argument('--lidar_resolution_height',
                      type=int,
                      default=int(config.lidar_resolution_height),
                      help='Height of the lidar bev frame (change together with max_x)')
  parser.add_argument('--dataset_cache_name',
                      type=str,
                      default='dataset_cache',
                      help='Name of the temporary folder for dataet caching. Important to use this when running '
                      'multiple trainings with different input data')
  parser.add_argument('--cosine_t0',
                      type=int,
                      default=int(config.cosine_t0),
                      help='Multiplier for the cosine learning rate schedule.')
  parser.add_argument('--compile',
                      type=int,
                      default=int(config.compile),
                      help='Whether to use torch.compile on the model. Not necessarily faster with TF.')
  parser.add_argument('--compile_mode',
                      type=str,
                      default=str(config.compile_mode),
                      help='compile mode for torch compile')
  parser.add_argument('--use_forgetting_monitor',
                      type=int,
                      default=int(getattr(config, 'use_forgetting_monitor', 0)),
                      help='Monitor KL against a frozen reference policy from --oracle_ref_file.')
  parser.add_argument('--oracle_ref_file',
                      type=str,
                      default=str(getattr(config, 'oracle_ref_file', '')),
                      help='Frozen reference model directory containing config.json and model_*.pth.')
  parser.add_argument('--monitor_kl_every',
                      type=int,
                      default=int(getattr(config, 'monitor_kl_every', 100)),
                      help='Training steps between forgetting KL checks.')
  parser.add_argument('--monitor_kl_batches',
                      type=int,
                      default=int(getattr(config, 'monitor_kl_batches', 3)),
                      help='Number of batches used for loader-based KL estimates.')
  parser.add_argument('--kl_forgetting_threshold',
                      type=float,
                      default=float(getattr(config, 'kl_forgetting_threshold', 0.5)),
                      help='KL threshold that triggers monitor actions.')
  parser.add_argument('--kl_patience',
                      type=int,
                      default=int(getattr(config, 'kl_patience', 3)),
                      help='Consecutive KL-threshold exceedances before early stopping.')
  parser.add_argument('--monitor_lr_decay',
                      type=float,
                      default=float(getattr(config, 'monitor_lr_decay', 0.1)),
                      help='Learning-rate multiplier used when forgetting monitor requests reduce_lr.')
  parser.add_argument('--use_a3d',
                      type=int,
                      default=int(config.use_a3d),
                      help='Enable Advantage-Anchored Adaptive Distillation (A3D) against the frozen reference model.')
  parser.add_argument('--a3d_ref_file',
                      type=str,
                      default=str(config.a3d_ref_file),
                      help='Teacher model directory containing config.json and model_*.pth for A3D.')
  parser.add_argument('--a3d_traj_beta',
                      type=float,
                      default=float(getattr(config, 'a3d_traj_beta', config.a3d_beta)))
  parser.add_argument('--a3d_speed_beta',
                      type=float,
                      default=float(getattr(config, 'a3d_speed_beta', config.a3d_beta)))
  parser.add_argument('--a3d_traj_tau',
                      type=float,
                      default=float(getattr(config, 'a3d_traj_tau', config.a3d_tau)))
  parser.add_argument('--a3d_speed_tau',
                      type=float,
                      default=float(getattr(config, 'a3d_speed_tau', config.a3d_tau)))
  parser.add_argument('--a3d_traj_lambda_max',
                      type=float,
                      default=float(getattr(config, 'a3d_traj_lambda_max', config.a3d_lambda_max)))
  parser.add_argument('--a3d_speed_lambda_max',
                      type=float,
                      default=float(getattr(config, 'a3d_speed_lambda_max', config.a3d_lambda_max)))
  parser.add_argument('--a3d_traj_lambda_ema',
                      type=float,
                      default=float(getattr(config, 'a3d_traj_lambda_ema', config.a3d_lambda_ema)))
  parser.add_argument('--a3d_speed_lambda_ema',
                      type=float,
                      default=float(getattr(config, 'a3d_speed_lambda_ema', config.a3d_lambda_ema)))
  parser.add_argument('--a3d_beta', type=float, default=float(config.a3d_beta))
  parser.add_argument('--a3d_tau', type=float, default=float(config.a3d_tau))
  parser.add_argument('--a3d_w_traj', type=float, default=float(config.a3d_w_traj))
  parser.add_argument('--a3d_w_speed', type=float, default=float(config.a3d_w_speed))
  parser.add_argument('--a3d_traj_threshold', type=float, default=float(config.a3d_traj_threshold))
  parser.add_argument('--a3d_lambda_max', type=float, default=float(config.a3d_lambda_max))
  parser.add_argument('--a3d_lambda_ema', type=float, default=float(config.a3d_lambda_ema))
  parser.add_argument('--a3d_kd_temperature', type=float, default=float(config.a3d_kd_temperature))
  parser.add_argument('--a3d_traj_kd_weight', type=float, default=float(config.a3d_traj_kd_weight))
  parser.add_argument('--a3d_speed_kd_weight', type=float, default=float(config.a3d_speed_kd_weight))
  parser.add_argument('--a3d_offset_kd_weight', type=float, default=float(config.a3d_offset_kd_weight))
  parser.add_argument('--use_lwf',
                      type=int,
                      default=int(getattr(config, 'use_lwf', 0)),
                      help='Enable Learning without Forgetting (LwF): fixed-lambda softened-KL distillation against the frozen reference model on top of GT losses.')
  parser.add_argument('--lwf_ref_file',
                      type=str,
                      default=str(getattr(config, 'lwf_ref_file', '')),
                      help='LwF teacher model directory containing config.json and model_*.pth. Falls back to --a3d_ref_file when empty.')
  parser.add_argument('--lwf_temperature',
                      type=float,
                      default=float(getattr(config, 'lwf_temperature', 2.0)),
                      help='Softening temperature for LwF distillation (paper: T=2).')
  parser.add_argument('--lwf_lambda',
                      type=float,
                      default=float(getattr(config, 'lwf_lambda', 1.0)),
                      help='Fixed LwF balance weight (paper: lambda_o=1).')
  parser.add_argument('--lwf_traj_weight',
                      type=float,
                      default=float(getattr(config, 'lwf_traj_weight', 1.0)),
                      help='Relative weight of the trajectory LwF KL before lambda.')
  parser.add_argument('--lwf_speed_weight',
                      type=float,
                      default=float(getattr(config, 'lwf_speed_weight', 1.0)),
                      help='Relative weight of the speed LwF KL before lambda.')
  parser.add_argument('--lwf_offset_kd_weight',
                      type=float,
                      default=float(getattr(config, 'lwf_offset_kd_weight', 0.0)),
                      help='Optional geometry offset term for LwF (0 = faithful LwF, KL only).')
  parser.add_argument('--use_oracle_kd',
                      type=int,
                      default=int(getattr(config, 'use_oracle_kd', 0)),
                      help='Enable Oracle Distribution KL training (oracleized GT supervision).')
  parser.add_argument('--oracle_traj_threshold',
                      type=float,
                      default=float(getattr(config, 'oracle_traj_threshold', 2.0)),
                      help='Distance threshold defining the GT-correct anchor support for oracle KD.')
  parser.add_argument('--oracle_kd_traj_weight',
                      type=float,
                      default=float(getattr(config, 'oracle_kd_traj_weight', 0.3)),
                      help='Loss weight for trajectory oracle KL before global normalization.')
  parser.add_argument('--oracle_kd_speed_weight',
                      type=float,
                      default=float(getattr(config, 'oracle_kd_speed_weight', 0.2)),
                      help='Loss weight for speed oracle KL before global normalization.')
  parser.add_argument('--oracle_kd_traj_l1_weight',
                      type=float,
                      default=float(getattr(config, 'oracle_kd_traj_l1_weight', 0.0)),
                      help='Optional extra weight for top-1 trajectory L1 from Oracle KD.')
  parser.add_argument('--oracle_kd_temperature',
                      type=float,
                      default=float(getattr(config, 'oracle_kd_temperature', 1.0)),
                      help='Temperature for Oracle KL.')
  parser.add_argument('--min_correct_anchors',
                      type=int,
                      default=int(getattr(config, 'min_correct_anchors', 1)),
                      help='Minimum correct anchors per sample; nearest anchors are used as fallback.')
  parser.add_argument('--use_ability_open_loop_monitor',
                      type=int,
                      default=int(getattr(config, 'use_ability_open_loop_monitor', 0)),
                      help='Enable rank-0 TensorBoard monitor for per-ability open-loop trajectory/speed metrics.')
  parser.add_argument('--ability_monitor_every',
                      type=int,
                      default=int(getattr(config, 'ability_monitor_every', 500)),
                      help='Training steps between per-ability open-loop monitor runs.')
  parser.add_argument('--ability_monitor_num_samples',
                      type=int,
                      default=int(getattr(config, 'ability_monitor_num_samples', 8)),
                      help='Maximum samples evaluated per ability at each monitor run.')
  parser.add_argument('--ability_monitor_batch_size',
                      type=int,
                      default=int(getattr(config, 'ability_monitor_batch_size', 2)),
                      help='Batch size for the per-ability open-loop monitor.')
  parser.add_argument('--ability_monitor_num_workers',
                      type=int,
                      default=int(getattr(config, 'ability_monitor_num_workers', 0)),
                      help='DataLoader workers for the per-ability open-loop monitor.')
  parser.add_argument('--ability_monitor_split',
                      type=str,
                      default=str(getattr(config, 'ability_monitor_split', 'train')),
                      choices=['train', 'val', 'valid', 'validation'],
                      help='Dataset split used by the per-ability open-loop monitor.')
  parser.add_argument('--ability_monitor_root',
                      type=str,
                      default=str(getattr(config, 'ability_monitor_root', '')),
                      help='Optional dataset root override for the per-ability open-loop monitor.')
  parser.add_argument('--ability_monitor_disable_augment',
                      type=int,
                      default=int(getattr(config, 'ability_monitor_disable_augment', 1)),
                      help='Disable dataset augmentation for monitor samples.')

  args = parser.parse_args()
  args.logdir = os.path.join(args.logdir, args.id)

  if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

  if bool(args.use_disk_cache):
    # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
    # During training, we cache the dataset on the fast storage of the local compute nodes.
    # Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
    # same folder (so they can share the cache).
    tmp_folder = str(os.environ.get('SCRATCH', '/tmp'))
    tmp_folder = tmp_folder + '/' + args.dataset_cache_name
    print('Tmp folder for dataset cache: ', tmp_folder)
    shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
  else:
    shared_dict = None

  # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
  rank = int(os.environ['RANK'])  # Rank across all processes
  if args.local_rank == -999:  # For backwards compatibility
    local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
  else:
    local_rank = int(args.local_rank)
  world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
  print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')

  device = torch.device(f'cuda:{local_rank}')

  torch.distributed.init_process_group(backend='nccl',
                                       init_method='env://',
                                       world_size=world_size,
                                       rank=rank,
                                       timeout=datetime.timedelta(minutes=15))

  ngpus_per_node = torch.cuda.device_count()
  ncpus_per_node = args.cpu_cores
  num_workers = int(ncpus_per_node / ngpus_per_node)
  print('Rank:', rank, 'Device:', device, 'Num GPUs on node:', ngpus_per_node, 'Num CPUs on node:', ncpus_per_node,
        'Num workers:', num_workers)
  torch.cuda.device(device)
  # We want the highest performance
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.allow_tf32 = True

  # Configure config. Converts all arguments into config attributes
  config.initialize(**vars(args))
  config.use_traj_front_door_encoder = False
  config.use_prior_fuseFeat = False

  config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0))
  # Before normalizing we need to set the losses we don't use to 0
  if config.use_plant:
    config.detailed_loss_weights['loss_semantic'] = 0.0
    config.detailed_loss_weights['loss_bev_semantic'] = 0.0
    config.detailed_loss_weights['loss_depth'] = 0.0
    config.detailed_loss_weights['loss_center_heatmap'] = 0.0
    config.detailed_loss_weights['loss_wh'] = 0.0
    config.detailed_loss_weights['loss_offset'] = 0.0
    config.detailed_loss_weights['loss_yaw_class'] = 0.0
    config.detailed_loss_weights['loss_yaw_res'] = 0.0
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0
  else:
    config.detailed_loss_weights['loss_forcast'] = 0.0

  if not config.use_controller_input_prediction:
    config.detailed_loss_weights['loss_target_speed'] = 0.0
    config.detailed_loss_weights['loss_checkpoint'] = 0.0

  # ---------------- A3D / Oracle continual distillation loss wiring ----------------
  # Distillation methods are mutually exclusive baselines.
  if (bool(getattr(config, 'use_lwf', 0)) and
      (bool(config.use_a3d) or bool(config.use_oracle_kd))):
    raise ValueError('LwF, A3D and Oracle KD are mutually exclusive. Enable only one.')

  # LwF: fixed-lambda softened-KL distillation on top of GT supervision (GT stays ON).
  for lwf_key in ('loss_lwf_traj_kd', 'loss_lwf_speed_kd', 'loss_lwf_offset_kd'):
    if bool(getattr(config, 'use_lwf', 0)):
      if lwf_key == 'loss_lwf_offset_kd' and float(getattr(config, 'lwf_offset_kd_weight', 0.0)) <= 0.0:
        config.detailed_loss_weights[lwf_key] = 0.0  # offset term disabled by default (faithful LwF)
      else:
        config.detailed_loss_weights[lwf_key] = 1.0
    else:
      config.detailed_loss_weights[lwf_key] = 0.0

  if bool(config.use_oracle_kd):
    # Oracleized GT: replace direct GT traj/speed supervision with oracle-conditioned
    # distillation targets (teacher distribution renormalized over the GT-correct support).
    config.detailed_loss_weights['loss_target_speed'] = 0.0
    config.detailed_loss_weights['loss_traj_kl_div'] = 0.0
    config.detailed_loss_weights['loss_weighted_regression'] = 0.0
    config.detailed_loss_weights['loss_best_trajectory'] = 0.0
    config.detailed_loss_weights['loss_speed_kl_div'] = 0.0
    config.detailed_loss_weights['loss_traj_bce'] = 0.0
    config.detailed_loss_weights['sample_loss_weighted_regression'] = 0.0
    config.detailed_loss_weights['sample_loss_best_trajectory'] = 0.0
    config.detailed_loss_weights['sample_loss_kl_div'] = 0.0

  # Plain A3D: teacher-gated KD kept on top of GT supervision (QtdA3dv2 recipe).
  for a3d_key in ('loss_a3d_total', 'loss_a3d_traj_kd', 'loss_a3d_speed_kd', 'loss_a3d_offset_kd'):
    if bool(config.use_a3d) and not bool(config.use_oracle_kd):
      config.detailed_loss_weights[a3d_key] = 1.0
    else:
      config.detailed_loss_weights[a3d_key] = 0.0

  # Oracle-only additive distillation (A3D off, oracle on).
  for oracle_key in ('loss_traj_oracle_kl', 'loss_speed_oracle_kl', 'loss_traj_l1'):
    if bool(config.use_oracle_kd) and not bool(config.use_a3d):
      if oracle_key == 'loss_traj_oracle_kl':
        config.detailed_loss_weights[oracle_key] = float(config.oracle_kd_traj_weight)
      elif oracle_key == 'loss_speed_oracle_kl':
        config.detailed_loss_weights[oracle_key] = float(config.oracle_kd_speed_weight)
      elif oracle_key == 'loss_traj_l1':
        config.detailed_loss_weights[oracle_key] = float(config.oracle_kd_traj_l1_weight)
    else:
      config.detailed_loss_weights[oracle_key] = 0.0

  # Oracleized A3D: the A3D gate blends teacher-KD <-> oracle-KD.
  for blend_key in ('loss_a3d_oracle_traj', 'loss_a3d_oracle_speed'):
    if (bool(config.use_a3d) and bool(config.use_oracle_kd) and
        (not config.use_plant) and bool(config.use_controller_input_prediction)):
      if blend_key.endswith('traj'):
        config.detailed_loss_weights[blend_key] = float(config.oracle_kd_traj_weight)
      else:
        config.detailed_loss_weights[blend_key] = float(config.oracle_kd_speed_weight)
    else:
      config.detailed_loss_weights[blend_key] = 0.0

  if not config.use_wp_gru:
    config.detailed_loss_weights['loss_wp'] = 0.0

  if not config.use_semantic:
    config.detailed_loss_weights['loss_semantic'] = 0.0

  if not config.use_bev_semantic:
    config.detailed_loss_weights['loss_bev_semantic'] = 0.0

  if not config.use_depth:
    config.detailed_loss_weights['loss_depth'] = 0.0

  if not config.detect_boxes:
    config.detailed_loss_weights['loss_center_heatmap'] = 0.0
    config.detailed_loss_weights['loss_wh'] = 0.0
    config.detailed_loss_weights['loss_offset'] = 0.0
    config.detailed_loss_weights['loss_yaw_class'] = 0.0
    config.detailed_loss_weights['loss_yaw_res'] = 0.0
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0

  # Not possible to predicted in a principled way from a single frame
  if config.lidar_seq_len == 1 and config.seq_len == 1:
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0

  if config.freeze_backbone:
    config.detailed_loss_weights['loss_semantic'] = 0.0
    config.detailed_loss_weights['loss_bev_semantic'] = 0.0
    config.detailed_loss_weights['loss_depth'] = 0.0
    config.detailed_loss_weights['loss_center_heatmap'] = 0.0
    config.detailed_loss_weights['loss_wh'] = 0.0
    config.detailed_loss_weights['loss_offset'] = 0.0
    config.detailed_loss_weights['loss_yaw_class'] = 0.0
    config.detailed_loss_weights['loss_yaw_res'] = 0.0
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0

  if config.multi_wp_output:
    config.detailed_loss_weights['loss_selection'] = 1.0

  if args.learn_multi_task_weights:
    for k in config.detailed_loss_weights:
      if config.detailed_loss_weights[k] > 0.0:
        config.detailed_loss_weights[k] = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
      else:
        # These losses we don't train
        config.detailed_loss_weights[k] = None
    # Convert to pytorch dictionary for proper parameter handling
    config.detailed_loss_weights = torch.nn.ParameterDict(config.detailed_loss_weights)
  else:
    # Normalize loss weights.
    factor = 1.0 / sum(config.detailed_loss_weights.values())
    for k in config.detailed_loss_weights:
      config.detailed_loss_weights[k] = config.detailed_loss_weights[k] * factor

  # Data, configures config. Create before the model
  train_set = Ability_CARLA_Data(root=config.mini_dataset_root,
                         config=config,
                         estimate_class_distributions=config.estimate_class_distributions,
                         estimate_sem_distribution=config.estimate_semantic_distribution,
                         shared_dict=shared_dict,
                         rank=rank,
                         validation=False,
                         ability=config.selected_ability)  # 'No_Scenario','Give_Way', 'Overtaking', 'Merging', 'Traffic_Sign', 'Emergency_Brake'

  if args.setting != 'all':
    val_set = Ability_CARLA_Data(root=config.mini_dataset_root, config=config, shared_dict=shared_dict, rank=rank, validation=True, ability_list=config.selected_ability_list)
  else:
    val_set = None

  if rank == 0:
    print('Target speed weights: ', config.target_speed_weights, flush=True)
    print('Angle weights: ', config.angle_weights, flush=True)
    print(f'Seed: {args.seed}', flush=True)
    print(f'config.crop_image: {config.crop_image}', flush=True)
    print(f'config.cropped_height: {config.cropped_height}', flush=True)
    print(f'config.cropped_width: {config.cropped_width}', flush=True)

  # Create model and optimizers
  if config.use_plant:
    model = PlanT(config)
  else:
    model = LidarCenterNet(config)

  # Register loss weights as parameters of the model if we learn them
  if args.learn_multi_task_weights:
    for k in config.detailed_loss_weights:
      if config.detailed_loss_weights[k] is not None:
        model.register_parameter(name='weight_' + k, param=config.detailed_loss_weights[k])
  model.cuda(device=device)

  start_epoch = 0  # Epoch to continue training from
#   if not args.load_file is None:
#     # Load checkpoint
#     print('=============load=================')
#     # Add +1 because the epoch before that was already trained
#     load_name = str(pathlib.Path(args.load_file).stem)
#     if args.continue_epoch:
#       start_epoch = int(''.join(filter(str.isdigit, load_name))) + 1
#     model.load_state_dict(torch.load(args.load_file, map_location=device), strict=False)

  if not args.load_file is None:
    # Load checkpoint
    print('=============load=================')
    # Add +1 because the epoch before that was already trained
    load_name = str(pathlib.Path(args.load_file).stem)
    if args.continue_epoch:
      start_epoch = int(''.join(filter(str.isdigit, load_name))) + 1
    
    # 使用修复后的加载方式

    # 先尝试正常加载
    # model.load_state_dict(torch.load(args.load_file, map_location=device), strict=False)
    model = load_checkpoint_ignore_anchors(model, args.load_file, device)
    print("anchor加载成功")


  if config.freeze_backbone:
    model.backbone.requires_grad_(False)

    if config.detect_boxes:
      model.head.requires_grad_(False)

    if config.use_semantic:
      model.semantic_decoder.requires_grad_(False)

    if config.use_bev_semantic:
      model.bev_semantic_decoder.requires_grad_(False)

    if config.use_depth:
      model.depth_decoder.requires_grad_(False)

  # Synchronizing the Batch Norms increases the Batch size with which they are compute by *num_gpus
  if bool(args.sync_batch_norm):
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
  find_unused_parameters = False
  if config.use_plant:
    find_unused_parameters = True
  model = torch.nn.parallel.DistributedDataParallel(model,
                                                    device_ids=None,
                                                    output_device=None,
                                                    broadcast_buffers=False,
                                                    find_unused_parameters=find_unused_parameters)
                                                    # find_unused_parameters=True)

  if config.use_optim_groups:
    params = model.module.create_optimizer_groups(config.weight_decay)
  else:
    params = model.parameters()

  if args.compile:
    model = torch.compile(model, mode=args.compile_mode)
    print('Compiled model')

  # One frozen reference model serves LwF/A3D teacher, oracle-KD base and forgetting monitor.
  ref_model = None
  if bool(getattr(config, 'use_lwf', 0)):
    ref_dir = str(getattr(config, 'lwf_ref_file', '')).strip()
    if ref_dir == '':
      ref_dir = str(config.a3d_ref_file)
  elif bool(config.use_a3d):
    ref_dir = config.a3d_ref_file
  else:
    ref_dir = str(getattr(config, 'oracle_ref_file', '')).strip()
  needs_ref = (
      bool(config.use_a3d) or
      bool(getattr(config, 'use_lwf', 0)) or
      bool(getattr(config, 'use_forgetting_monitor', 0)) or
      (bool(getattr(config, 'use_oracle_kd', 0)) and ref_dir != '')
  )
  if needs_ref and ref_dir != '':
    ref_model, ref_ckpt = load_ref_model_from_dir(ref_dir, device)
    if rank == 0:
      print(f'Loading frozen reference model dir: {ref_dir}', flush=True)
      print(f'Reference checkpoint selected: {ref_ckpt}', flush=True)
  elif bool(config.use_a3d):
    raise ValueError('A3D is enabled but --a3d_ref_file is empty.')
  elif bool(getattr(config, 'use_lwf', 0)):
    raise ValueError('LwF is enabled but --lwf_ref_file (or --a3d_ref_file) is empty.')
  elif bool(getattr(config, 'use_forgetting_monitor', 0)):
    raise ValueError('Forgetting monitor is enabled but --oracle_ref_file is empty.')

  if bool(args.zero_redundancy_optimizer):
    # Saves GPU memory during DDP training
    optimizer = ZeroRedundancyOptimizer(params, optimizer_class=optim.AdamW, lr=args.lr, amsgrad=True)
  else:
    optimizer = optim.AdamW(params, lr=args.lr, amsgrad=True)

  if not args.load_file is None and not config.freeze_backbone and args.continue_epoch:
    optimizer.load_state_dict(torch.load(args.load_file.replace('model_', 'optimizer_'), map_location=device))

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  num_params = sum(np.prod(p.size()) for p in model_parameters)
  if rank == 0:
    print('Total trainable parameters: ', num_params)

  g_cuda = torch.Generator(device='cpu')
  g_cuda.manual_seed(torch.initial_seed())

  sampler_train = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                  shuffle=True,
                                                                  num_replicas=world_size,
                                                                  rank=rank,
                                                                  drop_last=True)
  dataloader_train = DataLoader(train_set,
                                sampler=sampler_train,
                                batch_size=args.batch_size,
                                worker_init_fn=seed_worker,
                                generator=g_cuda,
                                num_workers=num_workers,
                                pin_memory=False,
                                drop_last=True)

  if args.setting != 'all':
    sampler_val = torch.utils.data.distributed.DistributedSampler(val_set,
                                                                  shuffle=True,
                                                                  num_replicas=world_size,
                                                                  rank=rank,
                                                                  drop_last=True)
    dataloader_val = DataLoader(val_set,
                                sampler=sampler_val,
                                batch_size=args.batch_size,
                                worker_init_fn=seed_worker,
                                generator=g_cuda,
                                num_workers=num_workers,
                                pin_memory=False,
                                drop_last=True)
  else:
    sampler_val, dataloader_val = None, None

  ability_monitor_loaders = build_ability_monitor_loaders(config, shared_dict, rank)

  # Create logdir
  if ((not os.path.isdir(args.logdir)) and (rank == 0)):
    print('Created dir:', args.logdir, rank)
    os.makedirs(args.logdir, exist_ok=True)

  # We only need one process to log the losses
  if rank == 0:
    writer = SummaryWriter(log_dir=args.logdir)
    # Log args
    with open(os.path.join(args.logdir, 'args.txt'), 'w', encoding='utf-8') as f:
      json.dump(args.__dict__, f, indent=2)

    json_config = jsonpickle.encode(config)
    with open(os.path.join(args.logdir, 'config.json'), 'w') as f2:
      f2.write(json_config)
  else:
    writer = None

  if config.use_cosine_schedule:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=config.cosine_t0,
                                                                     T_mult=config.cosine_t_mult)
  else:
    milestones = [args.schedule_reduce_epoch_01, args.schedule_reduce_epoch_02]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=config.multi_step_lr_decay)
  scaler = torch.amp.GradScaler(device, enabled=bool(config.use_amp))
  if not args.load_file is None and not config.freeze_backbone:
    if args.continue_epoch:
      scheduler.load_state_dict(torch.load(args.load_file.replace('model_', 'scheduler_'), map_location=device))
      scaler.load_state_dict(torch.load(args.load_file.replace('model_', 'scaler_'), map_location=device))

  trainer = Engine(model=model,
                   ref_model=ref_model,
                   optimizer=optimizer,
                   dataloader_train=dataloader_train,
                   dataloader_val=dataloader_val,
                   ability_monitor_loaders=ability_monitor_loaders,
                   args=args,
                   config=config,
                   writer=writer,
                   device=device,
                   rank=rank,
                   world_size=world_size,
                   cur_epoch=start_epoch,
                   scheduler=scheduler,
                   scaler=scaler)

  for epoch in range(trainer.cur_epoch, args.epochs):
    print(f'Epoch {epoch}, learning rate: ', scheduler.get_last_lr())
    # Update the seed depending on the epoch so that the distributed
    # sampler will use different shuffles across different epochs
    sampler_train.set_epoch(epoch)

    trainer.train()
    torch.cuda.empty_cache()

    if (not trainer.should_stop) and ((args.setting != 'all') and (epoch % args.val_every == 0)):
      trainer.validate()
      torch.cuda.empty_cache()

    if (not trainer.should_stop) and (not config.use_cosine_schedule):
      scheduler.step()

    if bool(args.zero_redundancy_optimizer):
      # To save the whole optimizer we need to gather it on GPU 0.
      optimizer.consolidate_state_dict(0)
    if rank == 0:
      trainer.save()

    if trainer.should_stop:
      if rank == 0:
        print('Stopping early because forgetting monitor requested early_stop.', flush=True)
      break

    trainer.cur_epoch += 1


class Engine(object):
  """
    Engine that runs training.
    """

  def __init__(self,
               model,
               ref_model,
               optimizer,
               dataloader_train,
               dataloader_val,
               ability_monitor_loaders,
               args,
               config,
               writer,
               device,
               scheduler,
               scaler,
               rank=0,
               world_size=1,
               cur_epoch=0):
    self.cur_epoch = cur_epoch
    self.bestval_epoch = cur_epoch
    self.train_loss = []
    self.val_loss = []
    self.bestval = 1e10
    self.model = model
    self.ref_model = ref_model
    self.optimizer = optimizer
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.ability_monitor_loaders = ability_monitor_loaders
    self.args = args
    self.config = config
    self.writer = writer
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.step = 0
    self.vis_save_path = self.args.logdir + r'/visualizations'
    self.scheduler = scheduler
    self.iters_per_epoch = len(self.dataloader_train)
    self.scaler = scaler
    self.should_stop = False

    if self.config.debug:
      pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

    self.detailed_loss_weights = config.detailed_loss_weights
    self.oracle_loss_fn = OracleKDLoss(config, base_model=ref_model) if (
        bool(getattr(config, 'use_oracle_kd', 0)) and not config.use_plant) else None
    self.forgetting_monitor = ForgettingMonitor(ref_model, config) if (
        bool(getattr(config, 'use_forgetting_monitor', 0)) and ref_model is not None and not config.use_plant) else None
    if self.rank == 0:
      print(f'A3D enabled: {bool(getattr(config, "use_a3d", 0)) and ref_model is not None}', flush=True)
      print(f'LwF enabled: {bool(getattr(config, "use_lwf", 0)) and ref_model is not None}', flush=True)
      print(f'Oracle KD enabled: {self.oracle_loss_fn is not None}', flush=True)
      print(f'Forgetting monitor enabled: {self.forgetting_monitor is not None}', flush=True)
      print(f'Ability open-loop monitor enabled: {bool(self.ability_monitor_loaders)}', flush=True)
    self.a3d_lambda_ema = 0.0
    self.a3d_traj_lambda_ema = 0.0
    self.a3d_speed_lambda_ema = 0.0
    self.a3d_monitor_tags = {
        'a3d_traj_advantage': 'traj_a_rel',
        'a3d_traj_ref_score': 'traj_ref_acc',
        'a3d_traj_active_score': 'traj_active_score',
        'a3d_traj_lambda_raw': 'traj_lambda_raw',
        'a3d_traj_lambda_ema': 'traj_lambda_ema',
        'a3d_traj_lambda_kd': 'traj_lambda_kd',
        'a3d_speed_advantage': 'speed_a_rel',
        'a3d_speed_ref_score': 'speed_ref_acc',
        'a3d_speed_active_score': 'speed_active_score',
        'a3d_speed_lambda_raw': 'speed_lambda_raw',
        'a3d_speed_lambda_ema': 'speed_lambda_ema',
        'a3d_speed_lambda_kd': 'speed_lambda_kd',
        'a3d_lambda': 'lambda_kd_avg',
    }
    self.oracle_monitor_tags = {
        'loss_a3d_oracle_traj': 'a3d_oracle_traj',
        'loss_a3d_oracle_speed': 'a3d_oracle_speed',
        'speed_acc': 'speed_acc',
        'oracle_correct_anchor_rate': 'correct_anchor_rate',
    }

  def _apply_forgetting_lr_decay(self, decay_factor):
    decay_factor = float(decay_factor)
    if decay_factor <= 0.0:
      return

    for param_group in self.optimizer.param_groups:
      param_group['lr'] *= decay_factor

    if hasattr(self.scheduler, 'base_lrs'):
      self.scheduler.base_lrs = [lr * decay_factor for lr in self.scheduler.base_lrs]
    if hasattr(self.scheduler, '_last_lr'):
      self.scheduler._last_lr = [lr * decay_factor for lr in self.scheduler._last_lr]

  def _trim_monitor_batch(self, data, max_items):
    if max_items <= 0:
      return data

    trimmed = {}
    for key, value in data.items():
      if torch.is_tensor(value) and value.size(0) > max_items:
        trimmed[key] = value[:max_items]
      else:
        trimmed[key] = value
    return trimmed

  def _build_monitor_traj_soft_labels(self, pred_trajectories, checkpoint):
    with torch.no_grad():
      distances = torch.norm(pred_trajectories - checkpoint.unsqueeze(0), dim=-1).sum(dim=-1)
      distances = torch.clamp(distances, min=1e-6, max=100.0)
      neg_distances = -distances / float(self.config.temperature)
      neg_distances = neg_distances - neg_distances.max(dim=0, keepdim=True)[0]
      soft_labels = F.softmax(neg_distances, dim=0)
      soft_labels = torch.clamp(soft_labels, min=1e-8, max=1.0)
      soft_labels = soft_labels / soft_labels.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return soft_labels

  def _compute_ability_open_loop_metrics(self, data):
    if self.config.use_plant:
      return {}, 0

    rgb = data['rgb'].to(self.device, dtype=torch.float32)
    if self.config.lidar_seq_len > 1:
      lidar = data['temporal_lidar'].to(self.device, dtype=torch.float32)
    else:
      lidar = data['lidar'].to(self.device, dtype=torch.float32)

    target_point = data['target_point'].to(self.device, dtype=torch.float32)
    target_point_next = None
    if self.config.two_tp_input:
      target_point_next = data['target_point_next'].to(self.device, dtype=torch.float32)
    command = data['command'].to(self.device, dtype=torch.float32)
    ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)
    batch_size = int(rgb.size(0))

    base_model = unwrap_model(self.model)
    _ = base_model(rgb=rgb,
                   lidar_bev=lidar,
                   target_point=target_point,
                   ego_vel=ego_vel,
                   command=command,
                   target_point_next=target_point_next)

    student_tensors = getattr(base_model, 'latest_distill_tensors', None)
    if student_tensors is None:
      return {}, batch_size

    metrics = {}
    pred_trajectories = student_tensors.get('pred_trajectories')
    pred_traj_probs = student_tensors.get('pred_traj_probs', student_tensors.get('traj_probs'))
    traj_logits = student_tensors.get('traj_logits')
    if pred_trajectories is not None and pred_traj_probs is not None:
      traj_steps = min(int(pred_trajectories.size(2)), int(data['route'].size(1)))
      pred_trajectories = pred_trajectories[:, :, :traj_steps]
      checkpoint = data['route'][:, :traj_steps].to(self.device, dtype=torch.float32)
      top1_traj = _select_top1_trajectory(pred_trajectories, pred_traj_probs)

      if top1_traj is not None:
        traj_error = torch.norm(top1_traj - checkpoint, dim=-1)
        metrics['traj_ade'] = float(traj_error.mean().item())
        metrics['traj_fde'] = float(traj_error[:, -1].mean().item())

      soft_labels = self._build_monitor_traj_soft_labels(pred_trajectories, checkpoint)
      if (traj_logits is not None and traj_logits.size(0) == soft_labels.size(0) and
          traj_logits.size(1) == soft_labels.size(1)):
        log_pred_traj = F.log_softmax(traj_logits.transpose(0, 1), dim=-1)
      else:
        pred_prob = pred_traj_probs.transpose(0, 1).clamp_min(1e-8)
        pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_pred_traj = torch.log(pred_prob)
      metrics['traj_kl'] = float(F.kl_div(log_pred_traj,
                                          soft_labels.transpose(0, 1),
                                          reduction='batchmean',
                                          log_target=False).item())

    speed_logits = student_tensors.get('speed_logits')
    target_speed = data.get('target_speed_twohot')
    if speed_logits is not None and target_speed is not None:
      target_speed = target_speed.to(self.device, dtype=torch.float32)
      if target_speed.size(1) == speed_logits.size(1):
        target_speed = target_speed / target_speed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pred_speed_idx = torch.argmax(speed_logits, dim=1)
        target_speed_idx = torch.argmax(target_speed, dim=1)
        metrics['speed_acc'] = float((pred_speed_idx == target_speed_idx).float().mean().item())
        metrics['speed_kl'] = float(F.kl_div(F.log_softmax(speed_logits, dim=-1),
                                             target_speed,
                                             reduction='batchmean',
                                             log_target=False).item())

    return metrics, batch_size

  @torch.inference_mode()
  def run_ability_open_loop_monitor(self):
    if self.rank != 0 or self.writer is None:
      return
    if not bool(getattr(self.config, 'use_ability_open_loop_monitor', 0)):
      return
    if not self.ability_monitor_loaders:
      return

    monitor_every = int(getattr(self.config, 'ability_monitor_every', 500))
    if monitor_every <= 0 or self.step <= 0 or (self.step % monitor_every) != 0:
      return

    max_samples = max(1, int(getattr(self.config, 'ability_monitor_num_samples', 8)))
    was_training = self.model.training
    self.model.eval()
    print(f'Running ability open-loop monitor... step={self.step}', flush=True)

    try:
      for ability, loader in self.ability_monitor_loaders.items():
        metric_sums = defaultdict(float)
        num_samples = 0

        for data in loader:
          remaining = max_samples - num_samples
          if remaining <= 0:
            break

          data = self._trim_monitor_batch(data, remaining)
          metrics, batch_size = self._compute_ability_open_loop_metrics(data)
          if batch_size <= 0:
            continue

          for key, value in metrics.items():
            metric_sums[key] += float(value) * batch_size
          num_samples += batch_size

          if num_samples >= max_samples:
            break

        if num_samples <= 0:
          continue

        for key, value in metric_sums.items():
          self.writer.add_scalar(f'ability_open_loop/{ability}/{key}', value / float(num_samples), self.step)
        self.writer.add_scalar(f'ability_open_loop/{ability}/num_samples', num_samples, self.step)
      self.writer.flush()
    finally:
      if was_training:
        self.model.train()

  def load_data_compute_loss(self, data, validation=False):
    # Validation = True will compute additional metrics not used for optimization
    # Load data used in both methods
    future_bounding_box_label = None
    if self.config.detect_boxes or self.config.use_plant:
      bounding_box_label = data['bounding_boxes'].to(self.device, dtype=torch.float32)
      if not self.config.use_plant:
        bb_center_heatmap = data['center_heatmap'].to(self.device, dtype=torch.float32)
        bb_wh = data['wh'].to(self.device, dtype=torch.float32)
        bb_yaw_class = data['yaw_class'].to(self.device, dtype=torch.long)
        bb_yaw_res = data['yaw_res'].to(self.device, dtype=torch.float32)
        bb_offset = data['offset'].to(self.device, dtype=torch.float32)
        bb_velocity = data['velocity'].to(self.device, dtype=torch.float32)
        bb_brake_target = data['brake_target'].to(self.device, dtype=torch.long)
        bb_pixel_weight = data['pixel_weight'].to(self.device, dtype=torch.float32)
        bb_avg_factor = data['avg_factor'].to(self.device, dtype=torch.float32)
      else:
        future_bounding_box_label = data['future_bounding_boxes'].to(self.device, dtype=torch.long)
    else:
      bounding_box_label = None
      bb_center_heatmap = None
      bb_wh = None
      bb_yaw_class = None
      bb_yaw_res = None
      bb_offset = None
      bb_velocity = None
      bb_brake_target = None
      bb_pixel_weight = None
      bb_avg_factor = None

    if self.config.use_wp_gru:
      ego_waypoint = data['ego_waypoints'].to(self.device, dtype=torch.float32)
    else:
      ego_waypoint = None

    target_point = data['target_point'].to(self.device, dtype=torch.float32)
    target_point_next = data['target_point_next'].to(self.device, dtype=torch.float32)
    command = data['command'].to(self.device, dtype=torch.float32)

    ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)

    if self.config.use_twohot_target_speeds:  # 1
      target_speed = data['target_speed_twohot'].to(self.device, dtype=torch.float32)
    else:
      target_speed = data['target_speed'].to(self.device, dtype=torch.long)

    # Load model specific data and execute model
    if self.config.use_plant:  # 0
      checkpoint = data['route'][:, :self.config.num_route_points].to(self.device, dtype=torch.float32)
      light_hazard = data['light'].to(self.device, dtype=torch.int32).unsqueeze(1)
      stop_hazard = data['stop_sign'].to(self.device, dtype=torch.int32).unsqueeze(1)
      junction = data['junction'].to(self.device, dtype=torch.int32).unsqueeze(1)
      route = data['route'][:, :self.config.num_route_points].to(self.device, dtype=torch.float32)

      pred_wp, pred_target_speed, \
      pred_checkpoint, pred_future_bounding_box = self.model(bounding_boxes=bounding_box_label,
                                    route=route,
                                    target_point=target_point,
                                    light_hazard=light_hazard,
                                    stop_hazard=stop_hazard,
                                    junction=junction,
                                    velocity=ego_vel)
    elif self.args.backbone in ('transFuser', 'aim', 'bev_encoder'):
      checkpoint = data['route'][:, :self.config.predict_checkpoint_len].to(self.device, dtype=torch.float32)
      rgb = data['rgb'].to(self.device, dtype=torch.float32)
      if self.config.use_semantic:
        semantic_label = data['semantic'].to(self.device, dtype=torch.long)
      else:
        semantic_label = None
      if self.config.use_bev_semantic:
        bev_semantic_label = data['bev_semantic'].to(self.device, dtype=torch.long)
      else:
        bev_semantic_label = None
      if self.config.use_depth:
        depth_label = data['depth'].to(self.device, dtype=torch.float32)
      else:
        depth_label = None
      if self.config.lidar_seq_len > 1:
        lidar = data['temporal_lidar'].to(self.device, dtype=torch.float32)
      else:
        lidar = data['lidar'].to(self.device, dtype=torch.float32)

      pred_wp,\
      pred_target_speed,\
      pred_trajectories, \
      pred_traj_probs, \
      pred_semantic, \
      pred_bev_semantic, \
      pred_depth, \
      pred_bounding_box, _, \
      pred_wp_1, \
      selected_path = self.model(rgb=rgb,
                          lidar_bev=lidar,
                          target_point=target_point,
                          ego_vel=ego_vel,
                          command=command,
                          target_point_next=target_point_next if self.config.two_tp_input else None,)
    else:
      raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

    compute_loss = self.model.module.compute_loss
    visualize_model = self.model.module.visualize_model

    if self.config.use_plant:  # 0
      losses = compute_loss(pred_wp=pred_wp,
                            pred_target_speed=pred_target_speed,
                            pred_checkpoint=pred_checkpoint,
                            pred_future_bounding_box=pred_future_bounding_box,
                            waypoint_label=ego_waypoint,
                            target_speed_label=target_speed,
                            checkpoint_label=checkpoint,
                            future_bounding_box_label=future_bounding_box_label)
    else:
      losses = compute_loss(pred_wp=pred_wp,
                            pred_target_speed=pred_target_speed,
                            pred_trajectories = pred_trajectories, 
                            pred_traj_probs = pred_traj_probs,
                            pred_semantic=pred_semantic,
                            pred_bev_semantic=pred_bev_semantic,
                            pred_depth=pred_depth,
                            pred_bounding_box=pred_bounding_box,
                            waypoint_label=ego_waypoint,
                            target_speed_label=target_speed,
                            checkpoint_label=checkpoint,
                            semantic_label=semantic_label,
                            bev_semantic_label=bev_semantic_label,
                            depth_label=depth_label,
                            center_heatmap_label=bb_center_heatmap,
                            wh_label=bb_wh,
                            yaw_class_label=bb_yaw_class,
                            yaw_res_label=bb_yaw_res,
                            offset_label=bb_offset,
                            velocity_label=bb_velocity,
                            brake_target_label=bb_brake_target,
                            pixel_weight_label=bb_pixel_weight,
                            avg_factor_label=bb_avg_factor,
                            pred_wp_1=pred_wp_1,
                            selected_path=selected_path)

    # Compute metrics for logging
    metrics = {}

    # ---------------- A3D / Oracle continual distillation (train only) ----------------
    teacher_tensors_for_oracle = None
    a3d_traj_ref_loss = None
    a3d_speed_ref_loss = None
    a3d_traj_lambda = None
    a3d_speed_lambda = None
    base_model = unwrap_model(self.model)

    # A3D: adaptive teacher-gated KD against the frozen reference model.
    if bool(self.config.use_a3d) and (self.ref_model is not None) and (not validation) and (not self.config.use_plant):
      with torch.no_grad():
        _ = self.ref_model(rgb=rgb,
                           lidar_bev=lidar,
                           target_point=target_point,
                           ego_vel=ego_vel,
                           command=command,
                           target_point_next=target_point_next if self.config.two_tp_input else None)
        teacher_tensors = self.ref_model.latest_distill_tensors
        teacher_tensors_for_oracle = teacher_tensors

      student_tensors = base_model.latest_distill_tensors
      if teacher_tensors is not None and student_tensors is not None:
        t_traj_score = _compute_traj_score(
            teacher_tensors.get('pred_trajectories'),
            teacher_tensors.get('traj_probs', teacher_tensors.get('pred_traj_probs')),
            checkpoint,
            self.config,
        )
        s_traj_score = _compute_traj_score(
            student_tensors.get('pred_trajectories'),
            student_tensors.get('traj_probs', student_tensors.get('pred_traj_probs')),
            checkpoint,
            self.config,
        )
        t_speed_acc = _compute_speed_acc_from_logits(teacher_tensors.get('speed_logits'), target_speed)
        s_speed_acc = _compute_speed_acc_from_logits(student_tensors.get('speed_logits'), target_speed)

        if t_traj_score is not None and s_traj_score is not None and t_speed_acc is not None and s_speed_acc is not None:
          traj_a_rel = torch.mean(t_traj_score - s_traj_score).detach()
          traj_ref_acc = torch.mean(t_traj_score).detach()
          traj_beta = float(getattr(self.config, 'a3d_traj_beta', self.config.a3d_beta))
          traj_tau = float(getattr(self.config, 'a3d_traj_tau', self.config.a3d_tau))
          traj_lambda_decay = float(getattr(self.config, 'a3d_traj_lambda_ema', self.config.a3d_lambda_ema))
          traj_lambda_cap = float(getattr(self.config, 'a3d_traj_lambda_max', self.config.a3d_lambda_max))

          traj_lambda_raw = torch.sigmoid(traj_beta * traj_a_rel).item()
          if float(traj_ref_acc.item()) < traj_tau:
            traj_lambda_raw = 0.0
          self.a3d_traj_lambda_ema = (traj_lambda_decay * self.a3d_traj_lambda_ema +
                                      (1.0 - traj_lambda_decay) * traj_lambda_raw)
          traj_lambda_kd = min(traj_lambda_cap, self.a3d_traj_lambda_ema)

          speed_a_rel = torch.mean(t_speed_acc - s_speed_acc).detach()
          speed_ref_acc = torch.mean(t_speed_acc).detach()
          speed_beta = float(getattr(self.config, 'a3d_speed_beta', self.config.a3d_beta))
          speed_tau = float(getattr(self.config, 'a3d_speed_tau', self.config.a3d_tau))
          speed_lambda_decay = float(getattr(self.config, 'a3d_speed_lambda_ema', self.config.a3d_lambda_ema))
          speed_lambda_cap = float(getattr(self.config, 'a3d_speed_lambda_max', self.config.a3d_lambda_max))

          speed_lambda_raw = torch.sigmoid(speed_beta * speed_a_rel).item()
          if float(speed_ref_acc.item()) < speed_tau:
            speed_lambda_raw = 0.0
          self.a3d_speed_lambda_ema = (speed_lambda_decay * self.a3d_speed_lambda_ema +
                                       (1.0 - speed_lambda_decay) * speed_lambda_raw)
          speed_lambda_kd = min(speed_lambda_cap, self.a3d_speed_lambda_ema)

          t_kd = self.config.a3d_kd_temperature
          s_traj_logits = student_tensors['traj_logits'].transpose(0, 1)
          t_traj_logits = teacher_tensors['traj_logits'].transpose(0, 1).detach()
          t_traj_prob = F.softmax(t_traj_logits / t_kd, dim=-1).detach()

          s_speed_logits = student_tensors['speed_logits']
          t_speed_logits = teacher_tensors['speed_logits'].detach()
          loss_speed_kd = _safe_kl(s_speed_logits, t_speed_logits, t_kd)

          # Align teacher trajectory distribution to student anchors for dynamic prototype sets.
          s_traj = student_tensors['pred_trajectories'].permute(1, 0, 2, 3)
          t_traj = teacher_tensors['pred_trajectories'].permute(1, 0, 2, 3).detach()
          t_prob_on_s, matched_t2s_l1 = _align_teacher_traj_prob_to_student(s_traj, t_traj, t_traj_prob)

          # KL on aligned support (student anchors).
          loss_traj_kd = _safe_kl_with_target_probs(s_traj_logits, t_prob_on_s, t_kd)

          # Teacher-weighted expected geometry mismatch under soft matching.
          loss_offset = torch.mean(torch.sum(matched_t2s_l1 * t_traj_prob, dim=-1))

          a3d_traj_ref_loss = loss_traj_kd + self.config.a3d_offset_kd_weight * loss_offset
          a3d_speed_ref_loss = loss_speed_kd
          a3d_traj_lambda = torch.tensor(traj_lambda_kd, device=self.device, dtype=loss_traj_kd.dtype)
          a3d_speed_lambda = torch.tensor(speed_lambda_kd, device=self.device, dtype=loss_speed_kd.dtype)

          avg_lambda_kd = 0.5 * (traj_lambda_kd + speed_lambda_kd)
          self.a3d_lambda_ema = avg_lambda_kd
          losses['a3d_lambda'] = torch.tensor(avg_lambda_kd, device=self.device, dtype=loss_traj_kd.dtype)

          losses['a3d_traj_advantage'] = traj_a_rel
          losses['a3d_traj_ref_score'] = traj_ref_acc
          losses['a3d_traj_active_score'] = torch.mean(s_traj_score).detach()
          losses['a3d_traj_lambda_raw'] = torch.tensor(traj_lambda_raw, device=self.device, dtype=loss_traj_kd.dtype)
          losses['a3d_traj_lambda_ema'] = torch.tensor(self.a3d_traj_lambda_ema,
                                                       device=self.device,
                                                       dtype=loss_traj_kd.dtype)
          losses['a3d_traj_lambda_kd'] = torch.tensor(traj_lambda_kd, device=self.device, dtype=loss_traj_kd.dtype)

          losses['a3d_speed_advantage'] = speed_a_rel
          losses['a3d_speed_ref_score'] = speed_ref_acc
          losses['a3d_speed_active_score'] = torch.mean(s_speed_acc).detach()
          losses['a3d_speed_lambda_raw'] = torch.tensor(speed_lambda_raw, device=self.device, dtype=loss_speed_kd.dtype)
          losses['a3d_speed_lambda_ema'] = torch.tensor(self.a3d_speed_lambda_ema,
                                                        device=self.device,
                                                        dtype=loss_speed_kd.dtype)
          losses['a3d_speed_lambda_kd'] = torch.tensor(speed_lambda_kd, device=self.device, dtype=loss_speed_kd.dtype)

          if not bool(getattr(self.config, 'use_oracle_kd', 0)):
            # Plain A3D (oracle off): gated teacher-KD added on top of GT losses.
            traj_kd_combined = (
                self.config.a3d_traj_kd_weight * loss_traj_kd +
                self.config.a3d_offset_kd_weight * loss_offset)
            losses['loss_a3d_total'] = (
                torch.tensor(traj_lambda_kd, device=self.device, dtype=loss_traj_kd.dtype) * traj_kd_combined +
                torch.tensor(speed_lambda_kd, device=self.device, dtype=loss_speed_kd.dtype) *
                (self.config.a3d_speed_kd_weight * loss_speed_kd))
            losses['loss_a3d_traj_kd'] = torch.tensor(traj_lambda_kd,
                                                      device=self.device,
                                                      dtype=loss_traj_kd.dtype) * loss_traj_kd
            losses['loss_a3d_speed_kd'] = torch.tensor(speed_lambda_kd,
                                                       device=self.device,
                                                       dtype=loss_speed_kd.dtype) * loss_speed_kd
            losses['loss_a3d_offset_kd'] = torch.tensor(traj_lambda_kd,
                                                        device=self.device,
                                                        dtype=loss_offset.dtype) * loss_offset

    # LwF (Learning without Forgetting): fixed-lambda softened-KL distillation against
    # the frozen previous-task model, added on top of the GT losses (Li & Hoiem 2017).
    if (bool(getattr(self.config, 'use_lwf', 0)) and (self.ref_model is not None) and
        (not validation) and (not self.config.use_plant)):
      with torch.no_grad():
        _ = self.ref_model(rgb=rgb,
                           lidar_bev=lidar,
                           target_point=target_point,
                           ego_vel=ego_vel,
                           command=command,
                           target_point_next=target_point_next if self.config.two_tp_input else None)
        teacher_tensors = self.ref_model.latest_distill_tensors

      student_tensors = base_model.latest_distill_tensors
      if teacher_tensors is not None and student_tensors is not None:
        t_kd = float(getattr(self.config, 'lwf_temperature', 2.0))
        lwf_lambda = float(getattr(self.config, 'lwf_lambda', 1.0))
        lwf_traj_w = float(getattr(self.config, 'lwf_traj_weight', 1.0))
        lwf_speed_w = float(getattr(self.config, 'lwf_speed_weight', 1.0))

        s_traj_logits = student_tensors['traj_logits'].transpose(0, 1)
        t_traj_logits = teacher_tensors['traj_logits'].transpose(0, 1).detach()
        t_traj_prob = F.softmax(t_traj_logits / t_kd, dim=-1).detach()

        s_speed_logits = student_tensors['speed_logits']
        t_speed_logits = teacher_tensors['speed_logits'].detach()
        loss_speed_kd = _safe_kl(s_speed_logits, t_speed_logits, t_kd)

        # Align teacher trajectory distribution to student anchors (infrastructure).
        s_traj = student_tensors['pred_trajectories'].permute(1, 0, 2, 3)
        t_traj = teacher_tensors['pred_trajectories'].permute(1, 0, 2, 3).detach()
        t_prob_on_s, matched_t2s_l1 = _align_teacher_traj_prob_to_student(s_traj, t_traj, t_traj_prob)
        loss_traj_kd = _safe_kl_with_target_probs(s_traj_logits, t_prob_on_s, t_kd)

        losses['loss_lwf_traj_kd'] = (torch.tensor(lwf_lambda * lwf_traj_w, device=self.device,
                                                   dtype=loss_traj_kd.dtype) * loss_traj_kd)
        losses['loss_lwf_speed_kd'] = (torch.tensor(lwf_lambda * lwf_speed_w, device=self.device,
                                                    dtype=loss_speed_kd.dtype) * loss_speed_kd)
        if float(getattr(self.config, 'lwf_offset_kd_weight', 0.0)) > 0.0:
          loss_offset = torch.mean(torch.sum(matched_t2s_l1 * t_traj_prob, dim=-1))
          losses['loss_lwf_offset_kd'] = (torch.tensor(lwf_lambda * self.config.lwf_offset_kd_weight,
                                                       device=self.device,
                                                       dtype=loss_offset.dtype) * loss_offset)

    # Oracle KD: teacher distribution renormalized over the GT-correct support.
    if self.oracle_loss_fn is not None and (not validation) and (not self.config.use_plant):
      base_tensors = None
      if self.ref_model is not None:
        if teacher_tensors_for_oracle is None:
          with torch.no_grad():
            _ = self.ref_model(rgb=rgb,
                               lidar_bev=lidar,
                               target_point=target_point,
                               ego_vel=ego_vel,
                               command=command,
                               target_point_next=target_point_next if self.config.two_tp_input else None)
            teacher_tensors_for_oracle = self.ref_model.latest_distill_tensors
        base_tensors = teacher_tensors_for_oracle

      oracle_components, oracle_metrics = self.oracle_loss_fn.compute_components(
          student_tensors=base_model.latest_distill_tensors,
          gt_data=data,
          base_tensors=base_tensors,
      )
      metrics.update({key: float(value.detach().item()) for key, value in oracle_metrics.items()})

      if bool(self.config.use_a3d) and (self.ref_model is not None):
        # Oracleized A3D: the gate lambda blends teacher-KD <-> oracle-KD.
        if a3d_traj_ref_loss is not None and a3d_traj_lambda is not None and 'traj_oracle_kl' in oracle_components:
          losses['loss_a3d_oracle_traj'] = (
              a3d_traj_lambda * a3d_traj_ref_loss +
              (1.0 - a3d_traj_lambda) * oracle_components['traj_oracle_kl']
          )

        if a3d_speed_ref_loss is not None and a3d_speed_lambda is not None and 'speed_oracle_kl' in oracle_components:
          losses['loss_a3d_oracle_speed'] = (
              a3d_speed_lambda * a3d_speed_ref_loss +
              (1.0 - a3d_speed_lambda) * oracle_components['speed_oracle_kl']
          )
      else:
        # Oracle-only additive distillation (A3D off).
        if 'traj_oracle_kl' in oracle_components:
          losses['loss_traj_oracle_kl'] = oracle_components['traj_oracle_kl']
        if 'speed_oracle_kl' in oracle_components:
          losses['loss_speed_oracle_kl'] = oracle_components['speed_oracle_kl']
        if float(getattr(self.config, 'oracle_kd_traj_l1_weight', 0.0)) > 0.0:
          student_tensors = base_model.latest_distill_tensors
          if student_tensors is not None and 'pred_trajectories' in student_tensors:
            student_traj = student_tensors['pred_trajectories'].permute(1, 0, 2, 3)
            gt_route = data['route'][:, :student_traj.size(2)].to(self.device, dtype=torch.float32)
            probs_key = 'traj_probs' if 'traj_probs' in student_tensors else 'pred_traj_probs'
            best_idx = torch.argmax(student_tensors[probs_key], dim=0)
            batch_idx = torch.arange(best_idx.size(0), device=best_idx.device)
            losses['loss_traj_l1'] = F.l1_loss(student_traj[batch_idx, best_idx], gt_route)

    if validation:
      if self.config.use_semantic and not self.config.use_plant:
        ss_miou = torchmetrics.functional.jaccard_index(pred_semantic,
                                                        semantic_label,
                                                        task='multiclass',
                                                        num_classes=self.config.num_semantic_classes).item()
        metrics['semantic_miou'] = ss_miou
      if self.config.use_bev_semantic and not self.config.use_plant:
        valid_bev_pixels = self.model.module.valid_bev_pixels

        visible_bev_semantic_label = valid_bev_pixels.squeeze(1).int() * bev_semantic_label
        # Set 0 class to ignore index -1
        visible_bev_semantic_label = (valid_bev_pixels.squeeze(1).int() - 1) + visible_bev_semantic_label

        bev_ss_miou = torchmetrics.functional.jaccard_index(pred_bev_semantic,
                                                            visible_bev_semantic_label,
                                                            task='multiclass',
                                                            ignore_index=-1,
                                                            num_classes=self.config.num_bev_semantic_classes).item()
        metrics['bev_semantic_miou'] = bev_ss_miou

    self.step += 1
    # Debug visualizations
    if self.config.debug and (self.step % self.config.train_debug_save_freq == 0) and \
        (self.vis_save_path is not None) and not self.config.use_plant:
      with torch.no_grad():
        if self.config.detect_boxes:
          pred_bounding_box = self.model.module.convert_features_to_bb_metric(pred_bounding_box)
        else:
          pred_bounding_box = None

        visualize_model(self.vis_save_path,
                        self.step,
                        rgb,
                        lidar,
                        target_point,
                        pred_wp,
                        target_point_next=target_point_next if self.config.two_tp_input else None,
                        pred_semantic=pred_semantic,
                        pred_bev_semantic=pred_bev_semantic,
                        pred_depth=pred_depth,
                        # pred_checkpoint=pred_checkpoint,
                        pred_trajectories = pred_trajectories, 
                        pred_traj_probs = pred_traj_probs,
                        pred_speed=F.softmax(pred_target_speed, dim=1) if pred_target_speed is not None else None,
                        pred_bb=pred_bounding_box,
                        gt_wp=ego_waypoint,
                        gt_bbs=bounding_box_label,
                        gt_checkpoints=checkpoint,
                        gt_bev_semantic=bev_semantic_label,
                        gt_speed=ego_vel)

    return losses, metrics

  def train(self):
    self.model.train()

    num_batches = 0
    loss_epoch = 0.0
    detailed_losses_epoch = {key: 0.0 for key in self.detailed_loss_weights}
    a3d_monitor_sum = {key: 0.0 for key in self.a3d_monitor_tags}
    a3d_monitor_count = {key: 0 for key in self.a3d_monitor_tags}
    oracle_monitor_sum = {key: 0.0 for key in self.oracle_monitor_tags}
    oracle_monitor_count = {key: 0 for key in self.oracle_monitor_tags}
    self.optimizer.zero_grad(set_to_none=False)

    # Train loop
    for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):

      with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=bool(self.config.use_amp)):
        losses, metrics = self.load_data_compute_loss(data, validation=False)
        loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        active_loss_keys = []

        for key, value in losses.items():
          if key not in self.detailed_loss_weights:
            continue
          if self.detailed_loss_weights[key] is None:
            continue
          active_loss_keys.append(key)
          if self.config.learn_multi_task_weights:
            precision = torch.exp(-self.detailed_loss_weights[key])
            loss += precision * value + self.detailed_loss_weights[key]
            detailed_losses_epoch[key] += float(precision * value + self.detailed_loss_weights[key])
          else:
            loss += self.detailed_loss_weights[key] * value
            detailed_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

        for loss_key in self.a3d_monitor_tags:
          if loss_key in losses:
            scalar_value = float(losses[loss_key].detach().item())
            a3d_monitor_sum[loss_key] += scalar_value
            a3d_monitor_count[loss_key] += 1

        for loss_key in self.oracle_monitor_tags:
          if loss_key in losses:
            scalar_value = float(losses[loss_key].detach().item())
          elif loss_key in metrics:
            scalar_value = float(metrics[loss_key])
          else:
            continue
          oracle_monitor_sum[loss_key] += scalar_value
          oracle_monitor_count[loss_key] += 1

      if not loss.requires_grad:
        raise RuntimeError(
            'No differentiable training loss was produced. '
            f'Active weighted loss keys: {active_loss_keys}; '
            f'produced loss keys: {list(losses.keys())}; '
            f'freeze_backbone={int(bool(getattr(self.config, "freeze_backbone", 0)))}.'
        )

      self.scaler.scale(loss).backward()

      if self.config.use_grad_clip:
        # Unscales the gradients of optimizers assigned params in-place
        self.scaler.unscale_(self.optimizer)
        # Since the gradients of optimizers assigned params are now unscaled, we can clip as usual.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=int(self.config.grad_clip_max_norm),
                                       error_if_nonfinite=True)

      self.scaler.step(self.optimizer)
      self.scaler.update()
      self.optimizer.zero_grad(set_to_none=True)

      num_batches += 1
      loss_epoch += float(loss.item())

      if self.config.use_cosine_schedule:
        self.scheduler.step(self.cur_epoch + i / self.iters_per_epoch)

      if self.forgetting_monitor is not None:
        action, kl_metrics = self.forgetting_monitor.check_batch_and_act(
            self.model,
            data,
            self.device,
            self.step,
        )
        if kl_metrics and self.rank == 0:
          print(f'Running forgetting monitor... step={self.step}', flush=True)
        if self.rank == 0 and self.writer is not None:
          for key, value in kl_metrics.items():
            self.writer.add_scalar(f'monitor/{key}', value, self.step)
          if action != 'continue':
            self.writer.add_text('monitor/action', action, self.step)
        if action == 'reduce_lr':
          self._apply_forgetting_lr_decay(getattr(self.config, 'monitor_lr_decay', 0.1))
          if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('monitor/lr_after_decay', self.optimizer.param_groups[0]['lr'], self.step)
        elif action == 'early_stop':
          self.should_stop = True
          break

      self.run_ability_open_loop_monitor()

    self.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')
    if self.rank == 0 and self.writer is not None:
      for loss_key, short_tag in self.a3d_monitor_tags.items():
        if a3d_monitor_count[loss_key] > 0:
          avg_value = a3d_monitor_sum[loss_key] / float(a3d_monitor_count[loss_key])
          self.writer.add_scalar(f'epoch/a3d_{short_tag}', avg_value, self.cur_epoch)
      for loss_key, short_tag in self.oracle_monitor_tags.items():
        if oracle_monitor_count[loss_key] > 0:
          avg_value = oracle_monitor_sum[loss_key] / float(oracle_monitor_count[loss_key])
          self.writer.add_scalar(f'epoch/oracle_{short_tag}', avg_value, self.cur_epoch)

  @torch.inference_mode()
  def validate(self):
    self.model.eval()

    num_batches = 0
    loss_epoch = 0.0
    detailed_val_losses_epoch = defaultdict(float)

    # Evaluation loop loop
    for data in tqdm(self.dataloader_val, disable=self.rank != 0):
      losses, metrics = self.load_data_compute_loss(data, validation=True)

      loss = torch.zeros(1, dtype=torch.float32, device=self.device)

      for key, value in losses.items():
        if key not in self.detailed_loss_weights:
          continue
        if self.detailed_loss_weights[key] is None:
          continue
        if self.config.learn_multi_task_weights:
          precision = torch.exp(-self.detailed_loss_weights[key])
          loss += precision * value + self.detailed_loss_weights[key]
          # We log the unweighted validation loss for comparability
          detailed_val_losses_epoch[key] += float(value)
        else:
          loss += self.detailed_loss_weights[key] * value
          detailed_val_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

      for key, value in metrics.items():
        detailed_val_losses_epoch[key] += float(value)

      num_batches += 1
      loss_epoch += float(loss.item())

      del losses
      del metrics

    self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')

  def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
    # Collecting the losses from all GPUs has led to issues.
    # I simply log the loss from GPU 0 for now they should be similar.
    if self.rank == 0:
      self.writer.add_scalar(prefix + 'loss_total', loss_epoch / num_batches, self.cur_epoch)

      for key, value in detailed_losses_epoch.items():
        self.writer.add_scalar(prefix + key, value / num_batches, self.cur_epoch)

  def save(self):

    model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch:04d}.pth')
    optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch:04d}.pth')
    scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch:04d}.pth')
    scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch:04d}.pth')

    # The parallel weights are named differently with the module.
    # We remove that, so that we can load the model with the same code.
    torch.save(self.model.module.state_dict(), model_file)

    torch.save(self.optimizer.state_dict(), optimizer_file)
    torch.save(self.scaler.state_dict(), scaler_file)
    torch.save(self.scheduler.state_dict(), scheduler_file)

    # Remove last epochs files to avoid accumulating storage
    if self.cur_epoch > 0:
      keep_prev_epoch = False
      if self.config.use_cosine_schedule:
        # We want to keep the model files that correspond to a minimum in the SGDR learning rate schedule
        # (skipping the first two)
        keep_epochs = [
            sum([self.config.cosine_t0 * self.config.cosine_t_mult**i for i in range(n)]) for n in range(3, 10)
        ]  # == [7, 15, 31, 63, 127, 255, 511]
        if self.cur_epoch in keep_epochs:
          # so we keep the models from the end of the epochs number 6, 14, 30
          # (epoch number 6 means it's the 7th epoch because we count from)
          keep_prev_epoch = True
      last_model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch - 1:04d}.pth')
      last_optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch - 1:04d}.pth')
      last_scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch - 1:04d}.pth')
      last_scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch - 1:04d}.pth')
      if not keep_prev_epoch:
        if os.path.isfile(last_model_file):
          os.remove(last_model_file)
        if os.path.isfile(last_optimizer_file):
          os.remove(last_optimizer_file)
        if os.path.isfile(last_scaler_file):
          os.remove(last_scaler_file)
        if os.path.isfile(last_scheduler_file):
          os.remove(last_scheduler_file)


# We need to seed the workers individually otherwise random processes in the
# dataloader return the same values across workers!
def seed_worker(worker_id):  # pylint: disable=locally-disabled, unused-argument
  worker_seed = (torch.initial_seed()) % 2**32  # this is different across workers, but not gpus when setting args.seed
  rank = int(os.environ['RANK'])
  # if args.seed is not None, torch.initial_seed is the same across different gpus, so we need to combine it with the
  # rank to get different rng seeds on different gpus. multiply with 1000 because the last digit is already
  # incremented across workers
  worker_seed = worker_seed + rank * 1000
  torch.manual_seed(worker_seed)
  np.random.seed(worker_seed)
  random.seed(worker_seed)
  print(
      f'Rank: {rank}, Worker id: {worker_id}, torch.inital_seed(): {torch.initial_seed()}, worker_seed: {worker_seed}')


if __name__ == '__main__':
  # Select how the threads in the data loader are spawned
  available_start_methods = mp.get_all_start_methods()
  if 'fork' in available_start_methods:
    mp.set_start_method('fork')
  # Available on all OS.
  elif 'spawn' in available_start_methods:
    mp.set_start_method('spawn')
  elif 'forkserver' in available_start_methods:
    mp.set_start_method('forkserver')
  print('Start method of multiprocessing:', mp.get_start_method())

  main()
