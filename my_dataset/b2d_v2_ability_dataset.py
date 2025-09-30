"""Code that loads the ability-based dataset for training with the same format as CARLA_Data."""
import os
import gzip
import ujson
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import re
import random
import pickle
import sys
import transfuser_utils as t_u
import gaussian_target as g_t
from sklearn.utils.class_weight import compute_class_weight
from center_net import angle2class
from imgaug import augmenters as ia
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple

class AbilityDatasetV2(Dataset):
    """Ability-based dataset that returns data in the exact same format as CARLA_Data."""
    
    def __init__(self, 
                 root: List[str], 
                 config: Any,
                 is_train: bool = True,
                 town_list: Optional[List[str]] = None,
                 val_ratio: float = 0.2,
                 shared_dict: Optional[Dict] = None,
                 rank: int = 0,
                 validation: bool = False,
                 ability: Optional[str] = None):
        """
        Dataset that loads scenarios based on a specific ability/skill with the same output format as CARLA_Data.
        
        Args:
            root: List of root directories containing scenario folders
            config: Configuration object (same as CARLA_Data)
            is_train: Whether this is training data
            town_list: List of towns to include (e.g., ['Town01', 'Town02'])
            val_ratio: Ratio of data to use for validation
            shared_dict: Shared dictionary for caching (same as CARLA_Data)
            rank: Process rank for distributed training
            validation: Whether this is validation data
            ability: Specific ability to filter for (None for all abilities)
        """
        super(AbilityDatasetV2, self).__init__()
        self.config = config
        self.validation = validation
        self.is_train = is_train
        assert config.img_seq_len == 1, "This implementation assumes img_seq_len == 1"
        
        # Set ability filter
        self.ability = ability
        self.ability_list = config.ability_list  # ['GiveWay', 'Overtaking', ...]
        
        # Set up data caching
        self.data_cache = shared_dict
        self.rank = rank
        
        # Initialize storage for paths
        self.lidars = []
        self.boxes = []
        self.future_boxes = []
        self.measurements = []
        self.sample_start = []
        self.temporal_lidars = []
        self.temporal_measurements = []
        self.images = []
        self.images_augmented = []
        self.semantics = []
        self.semantics_augmented = []
        self.bev_semantics = []
        self.bev_semantics_augmented = []
        self.depth = []
        self.depth_augmented = []
        
        # Image augmentation
        self.image_augmenter_func = self._create_image_augmenter()
        self.lidar_augmenter_func = self._create_lidar_augmenter()
        
        # Load scenario-to-ability mapping
        self._load_ability_mapping()
        
        # Process the dataset
        self._process_dataset(root, town_list, val_ratio)
        
        # Convert lists to numpy arrays for efficient indexing
        self._convert_to_numpy_arrays()
        
        logger.info(f"Loaded {len(self.lidars)} samples for ability-based dataset")
        if self.ability:
            logger.info(f"Filtering for ability: {self.ability}")
    
    def _load_ability_mapping(self):
        """Load the scenario-to-ability mapping from the config file."""
        try:
            with open(self.config.scenario_skill_file, 'rb') as f:
                self.scen_skill_desc_list = pickle.load(f)
            
            # Create a mapping from scenario name to abilities
            self.scenario_to_abilities = {}
            for item in self.scen_skill_desc_list:
                self.scenario_to_abilities[item['scen_name']] = item['skill']
                
            logger.info(f"Loaded scenario-to-ability mapping for {len(self.scenario_to_abilities)} scenarios")
        except Exception as e:
            logger.error(f"Error loading scenario-to-ability mapping: {e}")
            self.scenario_to_abilities = {}
    
    def _create_image_augmenter(self):
        """Create image augmentation function."""
        if self.config.use_color_aug:
            return ia.Sequential([
                ia.Sometimes(self.config.color_aug_prob, 
                    ia.SomeOf([
                        ia.WithColorspace(
                            to_colorspace="HSV",
                            from_colorspace="RGB",
                            children=ia.OneOf([
                                ia.WithChannels(0, ia.Add((-10, 10))),
                                ia.WithChannels(1, ia.Add((-10, 10))),
                                ia.WithChannels(2, ia.Add((-20, 20)))
                            ])
                        ),
                        ia.AddToHueAndSaturation((-10, 10)),
                        ia.Multiply((0.9, 1.1)),
                        ia.Gamma((0.9, 1.1)),
                        ia.GaussianBlur(sigma=(0.0, 0.5)),
                        ia.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
                    ])
                ),
                ia.Fliplr(0)  # No horizontal flipping (would mess up driving)
            ])
        return None
    
    def _create_lidar_augmenter(self):
        """Create LiDAR augmentation function."""
        if self.config.lidar_aug_prob > 0:
            return ia.Sequential([
                ia.Sometimes(self.config.lidar_aug_prob, 
                    ia.SomeOf([
                        ia.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
                    ])
                )
            ])
        return None
    
    def _process_dataset(self, root: List[str], town_list: Optional[List[str]], val_ratio: float):
        """Process the dataset and populate the path lists."""
        total_routes = 0
        trainable_routes = 0
        skipped_routes = 0
        
        # Load scenario-to-ability mapping
        try:
            with open(self.config.scenario_skill_file, 'rb') as f:
                scen_skill_desc_list = pickle.load(f)
            
            # Create a mapping from scenario name to abilities
            scenario_to_abilities = {}
            for item in scen_skill_desc_list:
                scenario_to_abilities[item['scen_name']] = item['skill']
        except Exception as e:
            logger.error(f"Error loading scenario-to-ability mapping: {e}")
            scenario_to_abilities = {}
        
        # Process each root directory
        for sub_root in tqdm(root, file=sys.stdout, disable=self.rank != 0):
            # Get all town directories
            towns = [d for d in os.listdir(sub_root) 
                    if os.path.isdir(os.path.join(sub_root, d))]
            
            for town in towns:
                # Skip if town not in specified list
                if town_list and town not in town_list:
                    continue
                
                town_path = os.path.join(sub_root, town)
                scenarios = [d for d in os.listdir(town_path)
                            if os.path.isdir(os.path.join(town_path, d))]
                
                for scenario in scenarios:
                    # Get abilities for this scenario
                    abilities = scenario_to_abilities.get(scenario, [])
                    
                    # Skip if this scenario doesn't have our requested ability
                    if self.ability and self.ability not in abilities:
                        continue
                    
                    # Process routes within this scenario
                    scenario_path = os.path.join(town_path, scenario)
                    route_dirs = [d for d in os.listdir(scenario_path)
                                if os.path.isdir(os.path.join(scenario_path, d))
                                and re.match(r'Route\d+_Rep\d+', d)]
                    
                    for route_dir in route_dirs:
                        route_path = os.path.join(scenario_path, route_dir)
                        measurements_dir = os.path.join(route_path, 'measurements')
                        
                        # Skip if measurements directory doesn't exist
                        if not os.path.exists(measurements_dir):
                            continue
                        
                        total_routes += 1
                        
                        # Get all frames in this route
                        try:
                            frame_files = [f for f in os.listdir(measurements_dir) 
                                         if f.endswith('.json.gz')]
                            frame_files.sort()
                            
                            # Only keep frames where we have enough future frames
                            valid_frames = []
                            for i in range(len(frame_files) - self.config.seq_len - self.config.pred_len + 1):
                                valid_frames.append(i)
                            
                            # Split into train/val if needed
                            is_val = self.validation or (not self.is_train and int(town[4:]) in self.config.val_towns)
                            
                            if is_val:
                                # For validation, take a portion of the frames
                                split_idx = int(len(valid_frames) * (1 - val_ratio))
                                valid_frames = valid_frames[split_idx:]
                            else:
                                # For training, take most of the frames
                                split_idx = int(len(valid_frames) * (1 - val_ratio))
                                valid_frames = valid_frames[:split_idx]
                            
                            # Add valid frames to our dataset
                            for seq in valid_frames:
                                # Load image paths
                                if not self.config.use_plant:
                                    self.images.append(os.path.join(route_path, 'rgb', f'{seq:04d}.jpg'))
                                    self.images_augmented.append(os.path.join(route_path, 'rgb_augmented', f'{seq:04d}.jpg'))
                                if self.config.use_semantic:
                                    self.semantics.append(os.path.join(route_path, 'semantics', f'{seq:04d}.png'))
                                    self.semantics_augmented.append(os.path.join(route_path, 'semantics_augmented', f'{seq:04d}.png'))
                                if self.config.use_bev_semantic:
                                    self.bev_semantics.append(os.path.join(route_path, 'bev_semantics', f'{seq:04d}.png'))
                                    self.bev_semantics_augmented.append(os.path.join(route_path, 'bev_semantics_augmented', f'{seq:04d}.png'))
                                if self.config.use_depth:
                                    self.depth.append(os.path.join(route_path, 'depth', f'{seq:04d}.png'))
                                    self.depth_augmented.append(os.path.join(route_path, 'depth_augmented', f'{seq:04d}.png'))
                                
                                # Store measurement path
                                self.measurements.append((route_path, seq))
                                
                                # Store sample start index
                                self.sample_start.append(seq)
                                
                                trainable_routes += 1
                        except Exception as e:
                            logger.warning(f"Error processing route {route_path}: {e}")
                            skipped_routes += 1
                            continue
        
        logger.info(f"Total routes: {total_routes}, Skipped routes: {skipped_routes}, Trainable routes: {trainable_routes}")
    
    def _convert_to_numpy_arrays(self):
        """Convert stored paths to numpy arrays for efficient indexing."""
        # Convert to numpy arrays for efficient indexing
        self.lidars = np.array(self.lidars).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.future_boxes = np.array(self.future_boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        self.sample_start = np.array(self.sample_start)
        self.temporal_lidars = np.array(self.temporal_lidars).astype(np.string_)
        self.temporal_measurements = np.array(self.temporal_measurements).astype(np.string_)
        self.images = np.array(self.images).astype(np.string_)
        self.images_augmented = np.array(self.images_augmented).astype(np.string_)
        self.semantics = np.array(self.semantics).astype(np.string_)
        self.semantics_augmented = np.array(self.semantics_augmented).astype(np.string_)
        self.bev_semantics = np.array(self.bev_semantics).astype(np.string_)
        self.bev_semantics_augmented = np.array(self.bev_semantics_augmented).astype(np.string_)
        self.depth = np.array(self.depth).astype(np.string_)
        self.depth_augmented = np.array(self.depth_augmented).astype(np.string_)
    
    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.measurements)
    
    def __getitem__(self, index):
        """Returns the item at index idx in the same format as CARLA_Data."""
        # Disable threading because the data loader will already split in processes.
        cv2.setNumThreads(0)
        data = {}
        
        # Get the route path and starting frame
        route_path, sample_start = self.measurements[index]
        sample_start = int(sample_start)
        
        # Randomly select augmentation parameters
        aug_translation = 0.0
        aug_rotation = 0.0
        if self.config.augment and self.is_train:
            aug_translation = random.uniform(-self.config.aug_max_translation, self.config.aug_max_translation)
            aug_rotation = random.uniform(-self.config.aug_max_rotation, self.config.aug_max_rotation)
        
        # Load measurements
        loaded_measurements = []
        for i in range(self.config.seq_len):
            measurement_file = os.path.join(route_path, 'measurements', f'{sample_start + i:04d}.json.gz')
            if self.data_cache is not None and measurement_file in self.data_cache:
                measurements_i = self.data_cache[measurement_file]
            else:
                try:
                    with gzip.open(measurement_file, 'rt', encoding='utf-8') as f:
                        measurements_i = ujson.load(f)
                    if self.data_cache is not None:
                        self.data_cache[measurement_file] = measurements_i
                except Exception as e:
                    logger.error(f"Error loading measurement {measurement_file}: {e}")
                    # Return a dummy item or handle error appropriately
                    return self.__getitem__((index + 1) % len(self))
            loaded_measurements.append(measurements_i)
        
        # Get current measurement (last in sequence)
        current_measurement = loaded_measurements[self.config.seq_len - 1]
        
        # Load image data
        if not self.config.use_plant:
            try:
                # Load RGB image
                img_path = str(self.images[index], encoding='utf-8')
                loaded_image = cv2.imread(img_path)
                if loaded_image is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                
                # Apply color augmentation if needed
                if self.config.use_color_aug and self.is_train:
                    processed_image = self.image_augmenter_func(image=loaded_image)
                else:
                    processed_image = loaded_image
                
                # Resize and crop if needed
                if self.config.image_resolution != [loaded_image.shape[1], loaded_image.shape[0]]:
                    processed_image = cv2.resize(processed_image, 
                                              (self.config.image_resolution[0], self.config.image_resolution[1]))
                
                # The transpose changes the image into pytorch (C,H,W) format
                data['rgb'] = np.transpose(processed_image, (2, 0, 1))
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                # Fallback: create a dummy image
                dummy_img = np.zeros((self.config.image_resolution[1], 
                                    self.config.image_resolution[0], 3), dtype=np.uint8)
                data['rgb'] = np.transpose(dummy_img, (2, 0, 1))
        
        # Load BEV semantic data
        if self.config.use_bev_semantic:
            try:
                bev_semantic_path = str(self.bev_semantics[index], encoding='utf-8')
                loaded_bev_semantic = cv2.imread(bev_semantic_path, cv2.IMREAD_UNCHANGED)
                if loaded_bev_semantic is None:
                    raise FileNotFoundError(f"BEV semantic not found: {bev_semantic_path}")
                
                # Convert using converter (from config)
                bev_semantics_i = self.config.bev_converter[loaded_bev_semantic]
                data['bev_semantic'] = bev_semantics_i
            except Exception as e:
                logger.error(f"Error loading BEV semantic {bev_semantic_path}: {e}")
                # Fallback: create dummy BEV semantic
                dummy_bev = np.zeros((self.config.lidar_resolution_height, 
                                    self.config.lidar_resolution_width), dtype=np.uint8)
                data['bev_semantic'] = self.config.bev_converter[dummy_bev]
        
        # Load depth data
        if self.config.use_depth:
            try:
                depth_path = str(self.depth[index], encoding='utf-8')
                loaded_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if loaded_depth is None:
                    raise FileNotFoundError(f"Depth not found: {depth_path}")
                
                # We saved the data in 8 bit and now convert back to float
                depth_i = loaded_depth.astype(np.float32) / 255.0
                
                # Resize depth
                data['depth'] = cv2.resize(
                    depth_i,
                    dsize=(
                        depth_i.shape[1] // self.config.perspective_downsample_factor,
                        depth_i.shape[0] // self.config.perspective_downsample_factor
                    ),
                    interpolation=cv2.INTER_LINEAR
                )
            except Exception as e:
                logger.error(f"Error loading depth {depth_path}: {e}")
                # Fallback: create dummy depth
                dummy_depth = np.zeros((
                    self.config.image_resolution[1] // self.config.perspective_downsample_factor,
                    self.config.image_resolution[0] // self.config.perspective_downsample_factor
                ), dtype=np.float32)
                data['depth'] = dummy_depth
        
        # Extract measurement data
        data['stop_sign'] = current_measurement['stop_sign_hazard']
        data['junction'] = current_measurement['junction']
        data['speed'] = current_measurement['speed']
        data['theta'] = current_measurement['theta']
        data['command'] = t_u.command_to_one_hot(current_measurement['command'])
        data['next_command'] = t_u.command_to_one_hot(current_measurement['next_command'])
        
        # Extract route information
        route = np.array(current_measurement['route'])
        if len(route) < self.config.num_route_points:
            num_missing = self.config.num_route_points - len(route)
            # Fill the empty spots by repeating the last point
            if len(route) > 0:
                last_point = route[-1]
                route = np.vstack([route, np.tile(last_point, (num_missing, 1))])
            else:
                route = np.zeros((self.config.num_route_points, 2))
        data['route'] = route[:self.config.num_route_points]
        
        # Extract target speed and brake
        data['target_speed'] = current_measurement['target_speed']
        data['brake'] = current_measurement['brake']
        
        # Add ability one-hot encoding
        ability_one_hot = np.zeros(len(self.ability_list), dtype=np.float32)
        
        # Determine the ability for this sample
        scenario_name = os.path.basename(os.path.dirname(route_path))
        if scenario_name in self.scenario_to_abilities:
            abilities = self.scenario_to_abilities[scenario_name]
            for ability in abilities:
                if ability in self.ability_list:
                    ability_idx = self.ability_list.index(ability)
                    ability_one_hot[ability_idx] = 1.0
        
        data['ability_one_hot'] = torch.from_numpy(ability_one_hot).float()
        
        # Add waypoints if using wp_gru
        if self.config.use_wp_gru:
            waypoints = self._get_waypoints(loaded_measurements[self.config.seq_len - 1:], 
                                          aug_translation, aug_rotation)
            data['ego_waypoints'] = np.array(waypoints)
        
        # Add centerline targets if using centerline
        if self.config.use_centerline:
            try:
                bounding_boxes = self._load_bounding_boxes(route_path, sample_start)
                target_result, avg_factor = self._compute_centerline_targets(bounding_boxes)
                
                data['center_heatmap_target'] = target_result['center_heatmap_target']
                data['wh'] = target_result['wh_target']
                data['offset'] = target_result['offset_target']
                data['velocity'] = target_result['velocity_target']
                data['brake_target'] = target_result['brake_target']
                data['pixel_weight'] = target_result['pixel_weight']
                data['avg_factor'] = avg_factor
            except Exception as e:
                logger.error(f"Error computing centerline targets: {e}")
        
        return data
    
    def _get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
        """Transform waypoints to be origin at ego_matrix."""
        origin = measurements[0]
        origin_matrix = np.array(origin['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]
        waypoints = []
        
        for index in range(self.config.seq_len, len(measurements)):
            waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
            # Transform waypoint to ego coordinates
            waypoint_ego = np.linalg.inv(origin_rotation) @ (waypoint - origin_translation)
            waypoints.append(waypoint_ego[:2, 0])
        
        # Data augmentation
        waypoints_aug = []
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], 
                                   [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])
        translation = np.array([[0.0], [y_augmentation]])
        
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))
        
        return waypoints_aug
    
    def _load_bounding_boxes(self, route_path, seq):
        """Load bounding boxes for the sequence."""
        bounding_boxes = []
        
        for i in range(self.config.seq_len, self.config.seq_len + self.config.pred_len):
            box_path = os.path.join(route_path, 'boxes', f'{seq + i:04d}.pkl')
            try:
                with open(box_path, 'rb') as f:
                    boxes = pickle.load(f)
                bounding_boxes.append(boxes)
            except Exception as e:
                logger.error(f"Error loading bounding boxes {box_path}: {e}")
                bounding_boxes.append([])
        
        return bounding_boxes
    
    def _compute_centerline_targets(self, bounding_boxes):
        """Compute centerline targets from bounding boxes."""
        num_classes = self.config.num_bb_classes
        img_h = self.config.lidar_resolution_height
        img_w = self.config.lidar_resolution_width
        feat_h = img_h // self.config.down_ratio
        feat_w = img_w // self.config.down_ratio
        
        height_ratio = float(feat_h / img_h)
        center_heatmap_target = np.zeros([num_classes, feat_h, feat_w], dtype=np.float32)
        wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
        offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
        yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
        yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
        pixel_weight = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        
        # Process each bounding box
        for i, boxes in enumerate(bounding_boxes):
            for box in boxes:
                # Only process relevant classes
                if box['class'] not in ['car', 'walker']:
                    continue
                
                # Get box parameters
                x, y = box['position'][0], box['position'][1]
                w, h = box['extent'][0] * 2, box['extent'][1] * 2  # Convert from half-extent to full extent
                yaw = box['yaw']
                speed = box.get('speed', 0.0)
                brake = box.get('brake', 0.0)
                
                # Convert to feature space
                x_feat = x * height_ratio
                y_feat = y * height_ratio
                w_feat = w * height_ratio
                h_feat = h * height_ratio
                
                # Compute center point
                ctx = x_feat / self.config.down_ratio
                cty = y_feat / self.config.down_ratio
                
                # Check if within image bounds
                if (ctx < 0 or ctx >= feat_w or 
                    cty < 0 or cty >= feat_h):
                    continue
                
                # Compute integer center point
                ctx_int, cty_int = int(ctx), int(cty)
                
                # Compute radius for heatmap
                radius = max(0, int(g_t.gaussian_radius((h_feat, w_feat), self.config.gaussian_iou)))
                
                # Draw heatmap
                g_t.draw_umich_gaussian(center_heatmap_target[0], (ctx_int, cty_int), radius)
                
                # Store size
                wh_target[0, cty_int, ctx_int] = w_feat
                wh_target[1, cty_int, ctx_int] = h_feat
                
                # Store offset
                offset_target[0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[1, cty_int, ctx_int] = cty - cty_int
                
                # Store yaw (convert to class and residual)
                yaw_class, yaw_res = angle2class(yaw, self.config.num_dir_bins)
                yaw_class_target[0, cty_int, ctx_int] = yaw_class
                yaw_res_target[0, cty_int, ctx_int] = yaw_res
                
                # Store velocity and brake
                velocity_target[0, cty_int, ctx_int] = speed
                brake_target[0, cty_int, ctx_int] = int(round(brake))
                
                # Set pixel weight to 1 for pixels with bounding boxes
                pixel_weight[0, cty_int, ctx_int] = 1.0
        
        # Compute average factor for loss normalization
        avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
        
        target_result = {
            'center_heatmap_target': center_heatmap_target,
            'wh_target': wh_target,
            'offset_target': offset_target,
            'yaw_class_target': yaw_class_target,
            'yaw_res_target': yaw_res_target,
            'velocity_target': velocity_target,
            'brake_target': brake_target,
            'pixel_weight': pixel_weight
        }
        
        return target_result, avg_factor

def create_ability_datasets(root_dirs: List[str], 
                           config: Any,
                           train_towns: Optional[List[str]] = None,
                           val_towns: Optional[List[str]] = None,
                           val_ratio: float = 0.2) -> Dict[str, Dict[str, Dataset]]:
    """
    Create datasets for all abilities, split into train and validation.
    
    Args:
        root_dirs: List of root directories containing scenario folders
        config: Configuration object
        train_towns: List of towns to use for training
        val_towns: List of towns to use for validation
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Dictionary of datasets: {'ability_name': {'train': train_dataset, 'val': val_dataset}}
        Plus a combined dataset: {'combined': {'train': combined_train, 'val': combined_val}}
    """
    # Create datasets for individual abilities
    ability_datasets = {}
    
    for ability in config.ability_list:
        # Create training dataset for this ability
        train_dataset = AbilityDatasetV2(
            root=root_dirs,
            config=config,
            is_train=True,
            town_list=train_towns,
            val_ratio=val_ratio,
            validation=False,
            ability=ability
        )
        
        # Create validation dataset for this ability
        val_dataset = AbilityDatasetV2(
            root=root_dirs,
            config=config,
            is_train=False,
            town_list=val_towns,
            val_ratio=val_ratio,
            validation=True,
            ability=ability
        )
        
        ability_datasets[ability] = {
            'train': train_dataset,
            'val': val_dataset
        }
    
    # Create combined dataset (all abilities)
    combined_train = AbilityDatasetV2(
        root=root_dirs,
        config=config,
        is_train=True,
        town_list=train_towns,
        val_ratio=val_ratio,
        validation=False,
        ability=None
    )
    
    combined_val = AbilityDatasetV2(
        root=root_dirs,
        config=config,
        is_train=False,
        town_list=val_towns,
        val_ratio=val_ratio,
        validation=True,
        ability=None
    )
    
    ability_datasets['combined'] = {
        'train': combined_train,
        'val': combined_val
    }
    
    return ability_datasets