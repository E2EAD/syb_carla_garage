import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from team_code.ability_data import Ability_CARLA_Data
from team_code.config import GlobalConfig
from plot_cluster_traj import visualize_all_waypoints


DEFAULT_ABILITIES = ['Emergency_Brake']


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def purge_invalid_values(array, name):
    valid_mask = np.isfinite(array).all(axis=1)
    if valid_mask.all():
        return array
    print(f'Removed {(~valid_mask).sum()} invalid samples from {name}')
    return array[valid_mask]


def collect_trajectory_samples(data_root, config, abilities, batch_size, num_workers, max_samples_per_ability):
    samples_by_ability = {}

    for ability in abilities:
        print(f'Collecting trajectory samples for {ability}')
        dataset = Ability_CARLA_Data(
            root=data_root,
            config=config,
            estimate_class_distributions=config.estimate_class_distributions,
            estimate_sem_distribution=config.estimate_semantic_distribution,
            shared_dict=None,
            validation=False,
            ability=ability,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

        ability_samples = []
        for batch in dataloader:
            route = batch['route'][:, :config.predict_checkpoint_len].detach().cpu().float()
            flat_route = route.reshape(route.size(0), -1).numpy()
            ability_samples.append(flat_route)

            if max_samples_per_ability is not None:
                current_count = sum(len(x) for x in ability_samples)
                if current_count >= max_samples_per_ability:
                    break

        if not ability_samples:
            print(f'No samples found for {ability}, skipping')
            continue

        ability_array = np.concatenate(ability_samples, axis=0)
        if max_samples_per_ability is not None:
            ability_array = ability_array[:max_samples_per_ability]
        ability_array = purge_invalid_values(ability_array, ability)
        samples_by_ability[ability] = ability_array
        print(f'{ability}: {ability_array.shape}')

    return samples_by_ability


def clusters_from_kmeans(samples, n_clusters, random_state):
    n_clusters = min(n_clusters, len(samples))
    if n_clusters <= 0:
        raise ValueError('No valid samples for KMeans')

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(samples)

    clusters = []
    components = []
    for cluster_id in range(n_clusters):
        members = samples[labels == cluster_id]
        if len(members) == 0:
            continue
        var = members.var(axis=0) + 1e-6
        mu = kmeans.cluster_centers_[cluster_id]
        cluster = {
            'cluster_id': int(cluster_id),
            'mu': mu.astype(float).tolist(),
            'var': var.astype(float).tolist(),
            'size': int(len(members)),
        }
        clusters.append(cluster)
        components.append({
            'k': int(cluster_id),
            'mu': cluster['mu'],
            'var': cluster['var'],
            'size': int(len(members)),
        })

    return clusters, components, labels


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def plot_tsne(samples, labels, centers, output_path, random_state, max_points):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(samples) < 3:
        print('Too few samples for t-SNE, skipping')
        return

    rng = np.random.default_rng(random_state)
    if len(samples) > max_points:
        indices = rng.choice(len(samples), size=max_points, replace=False)
        plot_samples = samples[indices]
        plot_labels = labels[indices]
    else:
        plot_samples = samples
        plot_labels = labels

    tsne_input = np.concatenate([plot_samples, centers], axis=0)
    perplexity = min(30, max(2, (len(tsne_input) - 1) // 3))
    embedded = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto', perplexity=perplexity).fit_transform(tsne_input)
    sample_xy = embedded[:len(plot_samples)]
    center_xy = embedded[len(plot_samples):]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(sample_xy[:, 0], sample_xy[:, 1], c=plot_labels, s=8, cmap='tab20', alpha=0.55)
    plt.scatter(center_xy[:, 0], center_xy[:, 1], c=np.arange(len(centers)), s=120, cmap='tab20', marker='X', edgecolors='black')
    plt.colorbar(scatter, label='cluster_id')
    plt.title('KMeans trajectory clusters t-SNE')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_latest_copy(source_path, latest_path):
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, latest_path)


def main():
    parser = argparse.ArgumentParser(description='Fit KMeans trajectory anchors by ability.')
    parser.add_argument('--data-root', default='/share/home/u19666033/syb/pdm_dataset')
    parser.add_argument('--abilities', nargs='+', default=DEFAULT_ABILITIES)
    parser.add_argument('--n-clusters', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples-per-ability', type=int, default=None)
    parser.add_argument('--tsne-max-points', type=int, default=5000)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()

    exp_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_root = Path(args.output_root) if args.output_root is not None else Path(current_dir) / 'kmeans_results'
    exp_dir = output_root / exp_time
    track_dir = exp_dir / 'track_cluster_log'
    component_dir = exp_dir / 'component_log'
    tsne_dir = exp_dir / 'cluster_tsne_plots'
    for path in [track_dir, component_dir, tsne_dir]:
        path.mkdir(parents=True, exist_ok=True)

    config = GlobalConfig()
    save_json(vars(args), exp_dir / 'config.json')

    samples_by_ability = collect_trajectory_samples(
        args.data_root,
        config,
        args.abilities,
        args.batch_size,
        args.num_workers,
        args.max_samples_per_ability,
    )

    all_samples = []
    for skill_id, ability in enumerate(args.abilities):
        samples = samples_by_ability.get(ability)
        if samples is None or len(samples) == 0:
            continue
        clusters, components, labels = clusters_from_kmeans(samples, args.n_clusters, args.random_state)
        base_name = f'{skill_id}-0-0-{len(samples)}'
        cluster_path = track_dir / f'{base_name}-tracked_clusters.json'
        component_path = component_dir / f'{base_name}-components.json'
        save_json(clusters, cluster_path)
        save_json(components, component_path)
        plot_tsne(samples, labels, np.array([c['mu'] for c in clusters]), tsne_dir / f'{base_name}-{ability}-tsne.png', args.random_state, args.tsne_max_points)
        print(f'Saved {ability} trajectory clusters to {cluster_path}')
        all_samples.append(samples)

    if all_samples:
        merged_samples = np.concatenate(all_samples, axis=0)
        clusters, components, labels = clusters_from_kmeans(merged_samples, args.n_clusters, args.random_state)
        cluster_path = track_dir / 'all_abilities-tracked_clusters.json'
        component_path = component_dir / 'all_abilities-components.json'
        save_json(clusters, cluster_path)
        save_json(components, component_path)
        plot_tsne(merged_samples, labels, np.array([c['mu'] for c in clusters]), tsne_dir / 'all_abilities-tsne.png', args.random_state, args.tsne_max_points)
        write_latest_copy(cluster_path, output_root / 'latest' / 'track_cluster_log' / 'all_abilities-tracked_clusters.json')
        print(f'Saved merged trajectory clusters to {cluster_path}')

    visualize_all_waypoints(date_dir=str(exp_dir))
    print(f'KMeans trajectory clustering completed: {exp_dir}')


if __name__ == '__main__':
    main()
