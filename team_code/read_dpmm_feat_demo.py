import pickle
from utils import print_data_info
import torch

def load_pickle_clusters(pickle_path):
    """Load tracked clusters from pickle file"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Access the data
    tracked_clusters = data['tracked_clusters']
    cluster_history = data['cluster_history']
    components = data['components']
    metadata = data['metadata']
    
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Epoch: {metadata['epoch']}, Iteration: {metadata['iteration']}")
    print(f"Number of clusters: {len(tracked_clusters)}")
    
    return data

pkl_path ='./dpmm_feature/noMoe_demo/Traffic_Sign/tracked_clusters/Traffic_Sign-epoch00-iter001-20251208_172531-tracked_clusters.pkl'
data = load_pickle_clusters(pkl_path)
feat = [
    torch.tensor(c['mu'], dtype=torch.float32) 
    for c in sorted(data['tracked_clusters'], key=lambda x: x['cluster_id'])
    ]
feat = torch.stack(feat)
print_data_info(feat)
print(feat[0,:20])