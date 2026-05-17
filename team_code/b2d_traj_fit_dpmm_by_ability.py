'''
run this code under ~/dpmm_model/model (after properly setting b2d_train).
'''

# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import bnpy
import sys
import json
import os
import carla # 如果不直接使用CARLA的Transform，可以考虑用numpy/scipy实现变换
import torch.utils.data
import numpy as np
import torchvision.transforms
from PIL import Image, ImageDraw # ImageDraw 用于绘图
from loguru import logger
import math
import pickle
from torch.utils.data import Dataset, DataLoader
import yaml
import random
import time
import resource

# 获取当前脚本的上两级目录（LEGION/my_code）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # parent_dir=my_code
sys.path.append(parent_dir)
# from dataset.b2d_1000_dataset import ScenarioDataset
# from dataset.b2d_ability_dataset import AbilityDataset
from team_code.ability_data import Ability_CARLA_Data
from my_dpmm_model import BNPModel

from utils import weighted_kl_divergence, collect_samples_for_tsne, collect_samples_for_tsne_v2, visualize_tsne, \
plot_losses, reset_optimizer, convert_tensor_to_list, purge_invalid_values, cluster_and_evaluate, collect_samples_for_cluster_eval, print_data_info, \
combine_skill_dataloaders
import torch.optim as optim
from team_code.config import GlobalConfig
from plot_cluster_traj import visualize_all_waypoints

from datetime import datetime


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size() if torch.is_tensor(tensor) else 0


def batch_tensor_nbytes(batch):
    return sum(tensor_nbytes(v) for v in batch.values() if torch.is_tensor(v))


def process_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def gpu_memory_stats_mb():
    if not torch.cuda.is_available():
        return {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'max_allocated_mb': 0.0, 'max_reserved_mb': 0.0}
    return {
        'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
        'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
        'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'max_reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def build_traj_dpmm_input(batch, config):
    source = config.get('traj_fit_source', 'route')
    if source == 'target_speed_twohot':
        return batch['target_speed_twohot'].detach()
    if source == 'ego_waypoints':
        batch_size = batch['ego_waypoints'].size(0)
        return batch['ego_waypoints'].detach().reshape(batch_size, -1)
    batch_size = batch['route'].size(0)
    return batch['route'][:, :config.get('predict_checkpoint_len', 10)].detach().reshape(batch_size, -1)


def sample_replay(dpmm, current_count, config):
    if len(dpmm.components) == 0:
        return None, 0
    replay_ratio = config.get('replay_sample_ratio', None)
    if replay_ratio is None:
        K = len(dpmm.components)
        new_data_ratio = 1 / (1 + K)
        num_to_sample = int((1 - new_data_ratio) * current_count / new_data_ratio)
    else:
        num_to_sample = int(current_count * replay_ratio)
    max_replay = config.get('max_replay_samples_per_update', None)
    if max_replay is not None:
        num_to_sample = min(num_to_sample, max_replay)
    if num_to_sample <= 0:
        return None, 0
    return dpmm.sample_all(num_samples=num_to_sample), num_to_sample


def train_dpmm(dpmm, config, skill_dataloaders):
    """dpmm学习主训练循环: buffer fills -> DPMM self-sampling + buffer update."""
    print(f"Starting dpmm learning at {datetime.now()}")

    buffer_size = config.get('traj_buffer_size', config.get('dpmm_buffer_size', 4096))
    overhead_log_path = os.path.join('dpmm_results', exp_time, 'overhead_log', 'traj_dpmm_overhead.jsonl')
    summary_path = os.path.join('dpmm_results', exp_time, 'overhead_log', 'traj_dpmm_summary.json')
    update_records = []
    total_raw_batch_bytes = 0
    total_buffer_bytes = 0
    total_batches = 0
    update_id = 0
    skill_id = 0

    for skill, dataloaders in skill_dataloaders.items():
        print(f"\n{'*' * 15} Training Skill {skill} {'*' * 15} ")

        for task_id, dataloader in enumerate(dataloaders):
            for epoch in range(config["epochs_per_task"]):
                print(f"\nEpoch {epoch + 1}/{config['epochs_per_task']}")
                traj_buffer = []
                buffered_samples = 0

                for batch_idx, batch in enumerate(dataloader):
                    raw_batch_bytes = batch_tensor_nbytes(batch)
                    total_raw_batch_bytes += raw_batch_bytes
                    total_batches += 1
                    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                    current_traj = build_traj_dpmm_input(batch, config)
                    traj_buffer.append(current_traj)
                    buffered_samples += current_traj.size(0)
                    total_buffer_bytes += tensor_nbytes(current_traj)

                    while buffered_samples >= buffer_size:
                        buffered_data = torch.cat(traj_buffer, dim=0)
                        update_data = buffered_data[:buffer_size]
                        remaining_data = buffered_data[buffer_size:]
                        traj_buffer = [remaining_data] if remaining_data.numel() > 0 else []
                        buffered_samples = remaining_data.size(0) if remaining_data.numel() > 0 else 0

                        print(f"Updating DPMM at batch {batch_idx} with buffer_size={len(update_data)}...")
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        memory_before = process_memory_mb()
                        gpu_before = gpu_memory_stats_mb()
                        start_time = time.perf_counter()

                        sampled_traj, num_to_sample = sample_replay(dpmm, len(update_data), config)
                        if sampled_traj is not None:
                            z_samples = torch.cat((sampled_traj, update_data), dim=0)
                        else:
                            z_samples = update_data
                        z_samples = purge_invalid_values(z_samples, "z_samples")
                        dpmm.fit(z_samples)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start_time
                        memory_after = process_memory_mb()
                        gpu_after = gpu_memory_stats_mb()

                        tracked_clusters = sorted([{'cluster_id': data['id'], 'mu': data['mu'], 'var': data['var']} for data in dpmm.current_clusters.values()], key=lambda x: x['cluster_id'])
                        tracked_clusters = convert_tensor_to_list(tracked_clusters)
                        tracked_clusters_path = os.path.join('dpmm_results/' + exp_time + '/track_cluster_log', str(skill_id) + '-' + str(task_id) + '-' + str(epoch) + '-' + str(batch_idx) + "-tracked_clusters.json")
                        with open(tracked_clusters_path, 'w') as f:
                            json.dump(tracked_clusters, f, indent=4)
                        print(f"Saved tracked_cluster to {tracked_clusters_path}")
                        components = sorted(dpmm.components, key=lambda x: x['k'])
                        components_path = os.path.join('dpmm_results/' + exp_time + '/component_log', str(skill_id) + '-' + str(task_id) + '-' + str(epoch) + '-' + str(batch_idx) + "-components.json")
                        with open(components_path, 'w') as f:
                            json.dump(components, f, indent=4)
                        print(f"Saved component to {components_path}")

                        record = {
                            'update_id': update_id,
                            'skill': skill,
                            'skill_id': skill_id,
                            'task_id': task_id,
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'buffer_samples': int(len(update_data)),
                            'replay_samples': int(num_to_sample),
                            'fit_samples': int(len(z_samples)),
                            'feature_dim': int(update_data.reshape(update_data.size(0), -1).size(1)),
                            'update_time_sec': elapsed,
                            'process_memory_before_mb': memory_before,
                            'process_memory_after_mb': memory_after,
                            'process_memory_delta_mb': memory_after - memory_before,
                            'gpu_before': gpu_before,
                            'gpu_after': gpu_after,
                            'buffer_memory_mb': tensor_nbytes(update_data) / (1024 ** 2),
                            'raw_batch_memory_mb': raw_batch_bytes / (1024 ** 2),
                            'raw_vs_buffer_batch_saving_ratio': 1.0 - (tensor_nbytes(current_traj) / raw_batch_bytes) if raw_batch_bytes > 0 else None,
                            'num_clusters': len(tracked_clusters),
                            'tracked_clusters_path': tracked_clusters_path,
                            'components_path': components_path,
                        }
                        update_records.append(record)
                        append_jsonl(overhead_log_path, record)
                        update_id += 1

        skill_id = skill_id + 1

    print(f"dpmm learning completed at {datetime.now()}")
    dpmm.save_model(dpmm_save_dir)
    visualize_all_waypoints(date_dir=os.path.join('./dpmm_results', exp_time))

    summary = {
        'num_updates': len(update_records),
        'buffer_size': buffer_size,
        'total_batches_seen': total_batches,
        'total_buffer_storage_mb': total_buffer_bytes / (1024 ** 2),
        'total_raw_batch_storage_mb': total_raw_batch_bytes / (1024 ** 2),
        'storage_saving_mb': (total_raw_batch_bytes - total_buffer_bytes) / (1024 ** 2),
        'storage_saving_ratio': 1.0 - (total_buffer_bytes / total_raw_batch_bytes) if total_raw_batch_bytes > 0 else None,
        'avg_update_time_sec': float(np.mean([r['update_time_sec'] for r in update_records])) if update_records else 0.0,
        'max_update_time_sec': float(np.max([r['update_time_sec'] for r in update_records])) if update_records else 0.0,
        'avg_process_memory_delta_mb': float(np.mean([r['process_memory_delta_mb'] for r in update_records])) if update_records else 0.0,
        'max_gpu_allocated_mb': float(np.max([r['gpu_after']['max_allocated_mb'] for r in update_records])) if update_records else 0.0,
        'overhead_log_path': overhead_log_path,
    }
    save_json(summary_path, summary)
    print(f"Saved DPMM overhead summary to {summary_path}")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    exp_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(f'exp start at {exp_time}')

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current Working Directory:", os.getcwd())
    dpmm_save_dir = os.path.join(script_dir, 'dpmm_results', exp_time, 'dpmm_model')

    # 获取当前脚本所在目录
    # project_root = "../../"
    # scenario_dirs = [os.path.join(project_root, "b2d_1000", d) for d in
    #                  os.listdir(os.path.join(project_root, "b2d_1000"))]
    # data_root = "../../b2d_1000_train"
    # data_root = "../../b2d_143_train"
    data_root = "/share/home/u19666033/syb/pdm_dataset" #"/media/syb/syb_disk_2/b2d_base_v3/carla_dataset"

    # for d in scenario_dirs:
    #     print(d)
    #     # 加载scen_skill_desc_list（需替换为实际路径）
    # with open('../text_enco/scen_skill_desc_list.pkl', 'rb') as f:
    #     scen_skill_desc_list = pickle.load(f)
        # print(scen_skill_desc_list)

    config = GlobalConfig()

    # rank = int(os.environ['RANK'])  # Rank across all processes
    # if config.local_rank == -999:  # For backwards compatibility
    #     local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
    # else:
    #     local_rank = int(config.local_rank)
    #     world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
    # print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')

    # device = torch.device(f'cuda:{local_rank}')

    # assign scenario dataloaders to skill(ability) datalaoders and check
    # skill_dataloaders = {'Emergency_Brake':[], 'Traffic_Sign':[], 'Merging':[], 'Overtaking':[], 'Give_Way':[]}
    # skill_dataloaders = {'Give_Way':[], 'Overtaking':[], 'Merging':[], 'Traffic_Sign':[], 'Emergency_Brake':[], 'No_Scenario': []}
    # skill_dataloaders = { 'No_Scenario': [],'Give_Way':[], 'Overtaking':[], 'Merging':[], 'Traffic_Sign':[], 'Emergency_Brake':[]}
    # skill_dataloaders = { 'No_Scenario': [],'Overtaking':[], 'Give_Way':[],  'Traffic_Sign':[], 'Merging':[],'Emergency_Brake':[]}
    skill_dataloaders = {'Give_Way':[]}

    for k in skill_dataloaders.keys():
        ability = k
        print(ability)

        dataset = Ability_CARLA_Data(root=data_root,
                         config=config,
                         estimate_class_distributions=config.estimate_class_distributions,
                         estimate_sem_distribution=config.estimate_semantic_distribution,
                         shared_dict=None,
                        #  rank=rank,
                         validation=False,
                         ability=ability)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )

        skill_dataloaders[k].append(dataloader)

    print(skill_dataloaders)

    # 初始化参数配置
    dpmm_config = {
        # "dpmm_update_freq": 10,  # DPMM更新间隔
        # "dpmm_update_every_epoch": True,  # DPMM更新间隔
        "dpmm_update_per_epoch": 1,  # DPMM更新间隔
        "traj_buffer_size": 10000,
        "traj_fit_source": "route",  # route, ego_waypoints, target_speed_twohot
        "predict_checkpoint_len": config.predict_checkpoint_len,
        "replay_sample_ratio": None,
        "max_replay_samples_per_update": 5000,
        "epochs_per_task": 1,  # 每个任务训练轮数 5 may be enough
        # "batch_size": 16,  # 批次大小
        # "learning_rate": 1e-4,  # 学习率
        "latent_dim": 8,  # 潜在空间维度
        "save_dir": "dpmm_results",  # 保存路径
        "gamma0": 5,  # DPMM初始参数
        "num_lap": 1000,
        "sF": 1e-5,
        # "new_task_data_ratio": 0.5,
        "w_kl_beta": 1,
        "hist_frame_nums":5,
        "future_frame_nums":5,
    }
    # 创建保存目录（如果不存在）
    os.makedirs('dpmm_results/'+exp_time, exist_ok=True)
    os.makedirs('dpmm_results/'+exp_time+'/component_log', exist_ok=True)
    os.makedirs('dpmm_results/'+exp_time+'/track_cluster_log', exist_ok=True)
    os.makedirs('dpmm_results/'+exp_time+'/ckpt', exist_ok=True)
    os.makedirs('dpmm_results/'+exp_time+'/overhead_log', exist_ok=True)
    # os.makedirs(os.path.join('./dpmm_results',exp_time,'eval_clustering'), exist_ok=True)
    # 保存 config 到 JSON 文件
    dpmm_config_path = os.path.join('dpmm_results/'+exp_time, "config.json")
    with open(dpmm_config_path, 'w') as f:
        json.dump(dpmm_config, f, indent=4)
    print(f"Saved config to {dpmm_config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpmm = BNPModel(save_dir=dpmm_save_dir, gamma0=dpmm_config["gamma0"], num_lap=dpmm_config["num_lap"], sF=dpmm_config["sF"])  # DPMM模型

    train_dpmm(dpmm, dpmm_config, skill_dataloaders)
