import torch
import numpy as np
from pathlib import Path
import pickle
import json

class FeatureAnalysis:
    """Analyze extracted features and clustering results"""
    
    @staticmethod
    def load_tracked_clusters(cluster_path):
        """Load tracked clusters from saved file"""
        cluster_path = Path(cluster_path)
        
        if cluster_path.suffix == '.pkl':
            with open(cluster_path, 'rb') as f:
                return pickle.load(f)
        elif cluster_path.suffix == '.pt':
            return torch.load(cluster_path)
        else:
            raise ValueError(f"Unsupported file format: {cluster_path.suffix}")
    
    @staticmethod
    def analyze_cluster_evolution(cluster_dir):
        """Analyze how clusters evolve over training"""
        cluster_dir = Path(cluster_dir)
        cluster_files = sorted(cluster_dir.glob("*-tracked_clusters.pkl"))
        
        evolution_data = []
        
        for cluster_file in cluster_files:
            data = FeatureAnalysis.load_tracked_clusters(cluster_file)
            metadata = data['metadata']
            
            evolution_data.append({
                'file': cluster_file.name,
                'epoch': metadata['epoch'],
                'iteration': metadata['iteration'],
                'num_clusters': metadata['num_clusters'],
                'timestamp': metadata['timestamp']
            })
        
        return evolution_data
    
    @staticmethod
    def visualize_cluster_features(features, clusters, save_path=None):
        """Visualize feature clusters (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import seaborn as sns
            
            # Reduce dimensionality for visualization
            if features.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features.numpy())
            else:
                features_2d = features.numpy()
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            if clusters:
                # Color by cluster
                cluster_ids = [cluster['cluster_id'] for cluster in clusters]
                # You might need to assign each feature to a cluster
                # This is a simplified version - adjust based on your data
                scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=cluster_ids[:len(features_2d)], 
                                    cmap='tab10', alpha=0.6)
                plt.colorbar(scatter, label='Cluster ID')
            else:
                plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
            
            plt.title('Feature Cluster Visualization')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib or sklearn not available for visualization")


def create_feature_extraction_config():
    """Create configuration template for feature extraction"""
    config = {
        "model_path": "/path/to/your/model.pth",
        "dataset_root": "/path/to/your/dataset",
        "abilities": ["Emergency_Brake", "Give_Way", "Overtaking", "Merging", "Traffic_Sign"],
        "output_dir": "./dpmm_feature_results",
        "dpmm_config": {
            "gamma0": 5.0,
            "num_lap": 1000,
            "sF": 1e-5,
            "dpmm_update_per_epoch": 4
        },
        "training": {
            "epochs": 1,
            "iterations_per_epoch": 50,
            "batch_size": 8,
            # "num_batches_per_ability": 20
        }
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    config = create_feature_extraction_config()
    
    # Save config
    with open("feature_extraction_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created feature extraction configuration template")


'''
from feature_extraction_utils import FeatureAnalysis
evolution = FeatureAnalysis.analyze_cluster_evolution("./dpmm_feature_results/tracked_clusters")

dpmm_feature_results/
├── tracked_clusters/
│   ├── carla_emergency_brake-epoch00-iter000-20240115_143022-tracked_clusters.pkl
│   ├── carla_emergency_brake-epoch00-iter000-20240115_143022-tracked_clusters.pt
│   └── carla_emergency_brake-epoch00-iter000-20240115_143022-metadata.json
├── extracted_features/
│   └── carla_emergency_brake_features.pt
└── final_dpmm_model/
    ├── bnp_model_state.pth
    └── info_dict.json
'''