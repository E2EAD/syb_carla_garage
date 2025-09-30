# Import necessary modules
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger

# Add your project path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import your configuration and utility modules
from team_code.config import GlobalConfig
from b2d_v2_ability_dataset import AbilityDatasetV2, create_ability_datasets

def main():
    # Load configuration
    config = GlobalConfig()
    
    # Set ability list (must match your scenario skills)
    config.ability_list = ['GiveWay', 'Overtaking', 'Merging', 'TrafficSign', 'EmergencyBrake']
    
    # Set path to scenario-to-ability mapping file
    config.scenario_skill_file = "../text_enco/scen_skill_desc_list.pkl"
    
    # Set dataset root directories
    root_dirs = ["../../b2d_base_v2"]
    
    # Define towns for training and validation
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10', 'Town11', 'Town13']
    val_towns = []
    
    # Create all ability datasets
    logger.info("Creating ability datasets...")
    ability_datasets = create_ability_datasets(
        root_dirs=root_dirs,
        config=config,
        train_towns=train_towns,
        val_towns=val_towns,
        val_ratio=0.2
    )
    
    # Example 1: Training on a single ability (Overtaking)
    logger.info("\n" + "="*50)
    logger.info("Example 1: Training on Overtaking ability")
    logger.info("="*50)
    
    overtaking_train = ability_datasets['Overtaking']['train']
    overtaking_val = ability_datasets['Overtaking']['val']
    
    logger.info(f"Overtaking train dataset size: {len(overtaking_train)}")
    logger.info(f"Overtaking validation dataset size: {len(overtaking_val)}")
    
    # Create dataloader
    overtaking_train_loader = DataLoader(
        overtaking_train,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get a batch of data
    batch = next(iter(overtaking_train_loader))
    
    # Print data structure
    logger.info("\nData structure for Overtaking dataset:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}")
        else:
            logger.info(f"  {key}: {type(value)}")
    
    # Example 2: Training on all abilities combined
    logger.info("\n" + "="*50)
    logger.info("Example 2: Training on all abilities combined")
    logger.info("="*50)
    
    combined_train = ability_datasets['combined']['train']
    combined_val = ability_datasets['combined']['val']
    
    logger.info(f"Combined train dataset size: {len(combined_train)}")
    logger.info(f"Combined validation dataset size: {len(combined_val)}")
    
    # Create dataloader
    combined_train_loader = DataLoader(
        combined_train,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get a batch of data
    batch = next(iter(combined_train_loader))
    
    # Print data structure
    logger.info("\nData structure for combined dataset:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}")
        else:
            logger.info(f"  {key}: {type(value)}")
    
    # Example 3: Verify ability_one_hot is correctly set
    logger.info("\n" + "="*50)
    logger.info("Example 3: Verify ability_one_hot encoding")
    logger.info("="*50)
    
    for ability in config.ability_list:
        dataset = ability_datasets[ability]['train']
        if len(dataset) > 0:
            sample = dataset[0]
            ability_one_hot = sample['ability_one_hot'].numpy()
            
            logger.info(f"\n{ability} dataset sample:")
            logger.info(f"  ability_one_hot: {ability_one_hot}")
            logger.info(f"  Expected: {ability} should be 1.0 at index {config.ability_list.index(ability)}")
    
    # Example 4: Train on each ability one by one
    logger.info("\n" + "="*50)
    logger.info("Example 4: Training on each ability one by one")
    logger.info("="*50)
    
    for ability in config.ability_list:
        logger.info(f"\nTraining on {ability} ability...")
        train_dataset = ability_datasets[ability]['train']
        val_dataset = ability_datasets[ability]['val']
        
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        
        # Here you would initialize your model and training loop
        # For demonstration, we'll just create a dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        
        # Get one batch to verify
        batch = next(iter(train_loader))
        ability_one_hot = batch['ability_one_hot'][0].numpy()
        logger.info(f"  Sample ability_one_hot: {ability_one_hot}")
        logger.info(f"  Should have 1.0 at index {config.ability_list.index(ability)}")

if __name__ == "__main__":
    main()
