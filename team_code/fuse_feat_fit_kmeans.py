import argparse
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from ability_data import Ability_CARLA_Data
from config import GlobalConfig
from my_model_qtd import LidarCenterNet

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)


DEFAULT_ABILITIES = ['Emergency_Brake']


class FeatureExtractor:
    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = device
        self.model = self.load_model(model_path)
        self.feature_buffer = []

    def load_model(self, model_path):
        model = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        print(f'Loaded model from {model_path}')
        return model

    def extract_features(self, dataloader, max_num_batches=None):
        self.feature_buffer = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_num_batches is not None and batch_idx >= max_num_batches:
                    break
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=bool(self.config.use_amp)):
                    features = self._forward_pass(batch)
                if features is not None:
                    self.feature_buffer.append(features.detach().cpu())

        if not self.feature_buffer:
            return None
        all_features = torch.cat(self.feature_buffer, dim=0)
        print(f'Extracted {len(all_features)} feature samples with shape {tuple(all_features.shape)}')
        return all_features

    def _forward_pass(self, data):
        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        command = data['command'].to(self.device, dtype=torch.float32)
        ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        if self.config.lidar_seq_len > 1:
            lidar = data['temporal_lidar'].to(self.device, dtype=torch.float32)
        else:
            lidar = data['lidar'].to(self.device, dtype=torch.float32)

        self.model(
            rgb=rgb,
            lidar_bev=lidar,
            target_point=target_point,
            ego_vel=ego_vel,
            command=command,
        )
        if hasattr(self.model, 'last_joined_features'):
            return self.model.last_joined_features.detach()
        print('get no joined_checkpoint_features')
        return None


def purge_invalid_values(array, name):
    valid_mask = np.isfinite(array).all(axis=1)
    if valid_mask.all():
        return array
    print(f'Removed {(~valid_mask).sum()} invalid samples from {name}')
    return array[valid_mask]


def flatten_features(features):
    if torch.is_tensor(features):
        features = features.detach().cpu().float().numpy()
    return features.reshape(features.shape[0], -1)


def clusters_from_kmeans(features, original_shape, n_clusters, random_state):
    n_clusters = min(n_clusters, len(features))
    if n_clusters <= 0:
        raise ValueError('No valid features for KMeans')

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(features)

    tracked_clusters = []
    for cluster_id in range(n_clusters):
        members = features[labels == cluster_id]
        if len(members) == 0:
            continue
        tracked_clusters.append({
            'cluster_id': int(cluster_id),
            'mu': kmeans.cluster_centers_[cluster_id].astype(float).tolist(),
            'var': (members.var(axis=0) + 1e-6).astype(float).tolist(),
            'size': int(len(members)),
        })

    metadata = {
        'num_clusters': len(tracked_clusters),
        'feature_shape': list(original_shape),
        'flattened_dim': int(features.shape[1]),
        'n_samples': int(features.shape[0]),
        'method': 'kmeans',
    }
    return tracked_clusters, labels, metadata


def save_pickle(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def plot_tsne(features, labels, centers, output_path, random_state, max_points):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(features) < 3:
        print('Too few samples for t-SNE, skipping')
        return

    rng = np.random.default_rng(random_state)
    if len(features) > max_points:
        indices = rng.choice(len(features), size=max_points, replace=False)
        plot_features = features[indices]
        plot_labels = labels[indices]
    else:
        plot_features = features
        plot_labels = labels

    tsne_input = np.concatenate([plot_features, centers], axis=0)
    perplexity = min(30, max(2, (len(tsne_input) - 1) // 3))
    embedded = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto', perplexity=perplexity).fit_transform(tsne_input)
    sample_xy = embedded[:len(plot_features)]
    center_xy = embedded[len(plot_features):]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(sample_xy[:, 0], sample_xy[:, 1], c=plot_labels, s=8, cmap='tab20', alpha=0.55)
    plt.scatter(center_xy[:, 0], center_xy[:, 1], c=np.arange(len(centers)), s=120, cmap='tab20', marker='X', edgecolors='black')
    plt.colorbar(scatter, label='cluster_id')
    plt.title('KMeans fuse feature clusters t-SNE')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_latest_copy(source_path, latest_path):
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, latest_path)


def load_training_config(model_folder):
    config = GlobalConfig()
    config_path = os.path.join(model_folder, 'config.json')
    if not os.path.exists(config_path):
        print(f'No config.json found at {config_path}; using GlobalConfig defaults')
        return config

    with open(config_path, 'rt', encoding='utf-8') as f:
        loaded_config = jsonpickle.decode(f.read())
    config.__dict__.update(loaded_config.__dict__)
    return config


def main():
    parser = argparse.ArgumentParser(description='Fit KMeans anchors over joined fuse features.')
    parser.add_argument('--dataset-root', default='/share/home/u19666033/syb/pdm_dataset')
    parser.add_argument('--model-folder', default='./log/syb_TFFdeQtd_2stg')
    parser.add_argument('--model-name', default='model_0030.pth')
    parser.add_argument('--abilities', nargs='+', default=DEFAULT_ABILITIES)
    parser.add_argument('--n-clusters', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-num-batches', type=int, default=None)
    parser.add_argument('--tsne-max-points', type=int, default=5000)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()

    exp_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root) if args.output_root is not None else script_dir / 'kmeans_results'
    exp_dir = output_root / exp_time
    cluster_dir = exp_dir / 'tracked_clusters'
    tsne_dir = exp_dir / 'cluster_tsne_plots'
    cluster_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), exp_dir / 'fuse_feat_kmeans_config.json')

    config = load_training_config(args.model_folder)
    config.use_prior_fuseFeat = False
    config.use_random_query_tokens = False

    dataset = Ability_CARLA_Data(
        root=args.dataset_root,
        config=config,
        shared_dict=None,
        rank=0,
        validation=False,
        ability_list=args.abilities,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(args.model_folder, args.model_name)
    feature_extractor = FeatureExtractor(config, model_path, device)
    features = feature_extractor.extract_features(dataloader, max_num_batches=args.max_num_batches)
    if features is None:
        print('No features extracted. Exiting.')
        return

    original_shape = tuple(features.shape)
    flat_features = purge_invalid_values(flatten_features(features), 'fuse_features')
    tracked_clusters, labels, metadata = clusters_from_kmeans(flat_features, original_shape, args.n_clusters, args.random_state)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'all_abilities-{timestamp}'
    pickle_path = cluster_dir / f'{base_name}-tracked_clusters.pkl'
    metadata_path = cluster_dir / f'{base_name}-metadata.json'
    save_data = {
        'tracked_clusters': tracked_clusters,
        'metadata': metadata,
    }
    save_pickle(save_data, pickle_path)
    save_json({
        'metadata': metadata,
        'cluster_summary': {
            'num_clusters': len(tracked_clusters),
            'cluster_ids': [cluster['cluster_id'] for cluster in tracked_clusters],
            'cluster_sizes': [cluster['size'] for cluster in tracked_clusters],
        }
    }, metadata_path)
    plot_tsne(flat_features, labels, np.array([c['mu'] for c in tracked_clusters]), tsne_dir / f'{base_name}-tsne.png', args.random_state, args.tsne_max_points)
    write_latest_copy(pickle_path, output_root / 'latest' / 'tracked_clusters' / 'all_abilities-tracked_clusters.pkl')

    print(f'Saved KMeans fuse feature clusters to {pickle_path}')
    print(f'KMeans fuse feature clustering completed: {exp_dir}')


if __name__ == '__main__':
    main()
