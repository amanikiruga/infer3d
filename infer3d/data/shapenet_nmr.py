import json 
import glob
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .readers import readCamerasFromTxt
from infer3d.utils.general import PILtoTorch, matrix_to_quaternion
from infer3d.utils.graphics import getWorld2View2, getProjectionMatrix, getView2World

from .shared import SharedDataset
from infer3d import config
SHAPENET_NMR_ROOT = config.SHAPENET_NMR_ROOT
NMR_TRAIN_TEST_SPLIT_PATH = config.NMR_SPLITS_JSON
CLASS_INFO_JSON_PATH = config.NMR_CLASS_INFO_JSON

assert SHAPENET_NMR_ROOT is not None, "Update the location of the SRN Shapenet Dataset (infer3d/config.py)"

class ShapenetNMR(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train", deterministic_test_idxs = None, 
                 override_example_ids = None, override_overall_view = None, 
                 additional_overall_views=None, save_splits = False, only_poses= False, 
                 override_num_input_images=None, is_return_view_idx = False, 
                 particular_class_list = [], randomly_subsample = True, override_root = None):
        super().__init__()
        if override_root is not None: 
            global SHAPENET_NMR_ROOT
            SHAPENET_NMR_ROOT = override_root
            print("Overriding SHAPENET_NMR_ROOT to", SHAPENET_NMR_ROOT)
        self.cfg = cfg
        self.is_return_view_idx = is_return_view_idx
        self.num_input_images = override_num_input_images or cfg.data.input_images
        self.only_poses = only_poses
        self.additional_overall_views = additional_overall_views or []
        self.randomly_subsample = randomly_subsample
        self.deterministic_test_idxs = deterministic_test_idxs
        self.is_val_in_dist = (dataset_name == "val_in_dist")
        self.is_val_ood = (dataset_name == "val_ood")
        self.dataset_name = dataset_name
        if dataset_name == "vis" or dataset_name == "val_ood" or dataset_name == "val_in_dist":
            self.dataset_name = "test"
            
        # self.base_path = os.path.join(SHAPENET_MODIFIED_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
        #                                                                            cfg.data.category,
        #                                                                            self.dataset_name))
        # self.base_path = os.path.join(SHAPENET_MODIFIED_ROOT, "02958343") # use if using /nobackup/nvme1/ShapeNetCore.v2.modified/
        self.base_path = SHAPENET_NMR_ROOT

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp
                
        overall_view = cfg.data.overall_view
        if override_overall_view is not None:
            overall_view = override_overall_view

        if cfg.data.overall_view_two: 
            self.additional_overall_views.append(cfg.data.overall_view_two)
            print("WARNING: using two overall views", cfg.data.overall_view, cfg.data.overall_view_two) 
        

        if override_example_ids is not None:
            # When specific example IDs are provided, construct paths directly instead of globbing
            self.intrins = []
            for example_id in override_example_ids:
                intrins_path = os.path.join(self.base_path, example_id, overall_view, "intrinsics.txt")
                if os.path.exists(intrins_path):
                    self.intrins.append(intrins_path)
                else: 
                    raise ValueError(f"Intrinsics path {intrins_path} does not exist. base_path: {self.base_path}, example_id: {example_id}, overall_view: {overall_view}")
            print(len(self.intrins), "intrinsics found for override_example_ids", override_example_ids)
        else:
            with open(config.nmr_intrins_json(overall_view), "r") as f:
                self.intrins = json.load(f)
            print(len(self.intrins), "intrinsics found for overall_view", overall_view)
            
            # If override_root is provided, update all paths to use the new base_path
            if override_root is not None:
                updated_intrins = []
                skipped_count = 0
                for old_path in self.intrins:
                    # Extract example_id and view from old path
                    # Path format: /old/base/path/example_id/view/intrinsics.txt
                    parts = old_path.split('/')
                    # Find the intrinsics.txt part and work backwards
                    if 'intrinsics.txt' in old_path:
                        # Get example_id (2 levels up from intrinsics.txt)
                        example_id = os.path.basename(os.path.dirname(os.path.dirname(old_path)))
                        # Get view (1 level up from intrinsics.txt)
                        view = os.path.basename(os.path.dirname(old_path))
                        # Construct new path
                        new_path = os.path.join(self.base_path, example_id, view, "intrinsics.txt")
                        # Only include paths that actually exist
                        if os.path.exists(new_path):
                            updated_intrins.append(new_path)
                        else:
                            skipped_count += 1
                    else:
                        if os.path.exists(old_path):
                            updated_intrins.append(old_path)
                        else:
                            skipped_count += 1
                self.intrins = updated_intrins
                print(f"Updated {len(self.intrins)} intrinsics paths to use new base_path: {self.base_path}")
                if skipped_count > 0:
                    print(f"Skipped {skipped_count} intrinsics paths that don't exist at new base_path")

        # self.intrins = [x for x in self.intrins if len(glob.glob(os.path.join(os.path.dirname(x), "rgb", "*"))) == 24 and len(glob.glob(os.path.join(os.path.dirname(x), "pose", "*"))) == 24]
        # self.intrins = [x for x in self.intrins if len(glob.glob(os.path.join(os.path.dirname(x), "rgb", "*"))) > 0 and len(glob.glob(os.path.join(os.path.dirname(x), "pose", "*"))) > 0]
        
        # Filter to ensure consistent number of views across all objects
        # if len(self.intrins) > 0:
        #     # Get view counts for all objects
        #     view_counts = [len(glob.glob(os.path.join(os.path.dirname(x), "rgb", "*"))) for x in self.intrins]
        #     # Find the most common view count
        #     from collections import Counter
        #     most_common_view_count = Counter(view_counts).most_common(1)[0][0]
        #     # print also how many objects have the most common view count
        #     print(f"Filtering objects to have consistent view count: {most_common_view_count} with {len([x for x in view_counts if x == most_common_view_count])} objects")
        #     # Keep only objects with the most common view count
        #     print("before filtering for consistent view count", len(self.intrins))
        #     self.intrins = [x for x in self.intrins if len(glob.glob(os.path.join(os.path.dirname(x), "rgb", "*"))) >= most_common_view_count]
        #     print("after filtering for consistent view count", len(self.intrins))
        
        # Split the data into train and test sets
        # optionally filter by class list 
        if len(particular_class_list) > 0:
            print("filtering by class list", particular_class_list)
            with open(CLASS_INFO_JSON_PATH, "r") as f:
                class_info = json.load(f)
            all_obj_ids = [obj_id.strip() for class_id in particular_class_list for obj_id in class_info[class_id]]
            self.intrins = [x for x in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(x))).strip() in all_obj_ids]
            print("after filtering by class list", len(self.intrins))
        
        # When override_example_ids is provided, skip train/test split filtering
        # because we explicitly want to use those specific examples
        if override_example_ids is None:
            with open(NMR_TRAIN_TEST_SPLIT_PATH, "r") as f:
                object_splits = json.load(f)
            
            self.train_intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(intrin_path))).strip() in object_splits["train"]]
            self.test_intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(intrin_path))).strip() in object_splits["test"]]
            
            if self.dataset_name == "train":
                self.intrins = self.train_intrins
            elif self.is_val_ood or self.is_val_in_dist:
                # shuffle test_intrins and take the first cfg.data.val_size
                # seed 
                np.random.seed(42)
                np.random.shuffle(self.test_intrins)
                if cfg.data.val_size is not None:
                    self.intrins = self.test_intrins[:cfg.data.val_size]
                else:
                    self.intrins = self.test_intrins
                    cfg.data.val_size = len(self.test_intrins)
                # print example ids of self.intrins
                print("example ids of val_ood or val_in_dist dataset", [os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) for intrin_path in self.intrins])
                assert len(self.intrins) > 0, "size of val_ood or val_in_dist dataset is 0"
            else: 
                np.random.seed(42)
                np.random.shuffle(self.test_intrins)
                if cfg.data.val_size is not None:
                    self.intrins = self.test_intrins[:cfg.data.val_size]
                else:
                    self.intrins = self.test_intrins
                    cfg.data.val_size = len(self.test_intrins)
        else:
            # When override_example_ids is provided, use intrins as-is without split filtering
            if self.dataset_name == "train":
                self.train_intrins = self.intrins
            else:
                self.test_intrins = self.intrins
        if override_example_ids is not None and len(self.intrins) > 0:
            # Only filter if we used the glob approach (when override_example_ids was None initially)
            # If override_example_ids was provided from the start, self.intrins already contains only the specified IDs
            if any(os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) not in override_example_ids for intrin_path in self.intrins):
                print("length before overriding example ids", len(self.intrins))
                self.intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) in override_example_ids]
                print("overriding example ids", override_example_ids, "length of intrins:", len(self.intrins), "self.intrins:", self.intrins)
            else:
                print("override_example_ids already applied during initial path construction, length of intrins:", len(self.intrins))
            
            

        print("length of intrinsics after filtering", len(self.intrins))
        if cfg.data.subset != -1:
            self.intrins = self.intrins[:cfg.data.subset]

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj
   
            
        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.num_input_images == 1:
            # self.test_input_idxs = [64]
            if deterministic_test_idxs is not None:
                assert len(deterministic_test_idxs) == 1
                self.test_input_idxs = torch.tensor(deterministic_test_idxs)
                print("using deterministic test idxs:", self.test_input_idxs)
            else: 
                self.test_input_idxs = torch.randint(0, 24, (len(self.intrins),))
                # print("Chosen random testing input idxs", self.test_input_idxs)
        else:
            # self.test_input_idxs = [64, 128]
            self.test_input_idxs = torch.linspace(0, 47, self.num_input_images).long()
            print(f"self.num_input_images {self.num_input_images} is not 1")
            print("using linspace created test idxs:", self.test_input_idxs)
            

    def __len__(self):
        return len(self.intrins)

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        dir_path = os.path.dirname(intrin_path)
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        if self.only_poses:
            rgb_paths = [""] * len(pose_paths)  # dummy paths
        else: 
            rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
            
        if self.is_return_view_idx:
            view_idxs = [int(os.path.splitext(os.path.basename(pose_path))[0]) for pose_path in pose_paths]
            
        assert len(rgb_paths) == len(pose_paths)
        
        
        if self.randomly_subsample:
            subsample_factor = len(self.additional_overall_views) + 1
            num_samples = len(rgb_paths) // subsample_factor
            if num_samples == 0:
                raise ValueError(f"Subsample factor {subsample_factor} is too large for number of images {len(rgb_paths)}")
            if self.dataset_name == "train":
                # For training, randomly subsample and then slice to imgs_per_obj
                indices = sorted(random.sample(range(len(rgb_paths)), min(num_samples, self.imgs_per_obj)))
            else:
                indices = sorted(random.sample(range(len(rgb_paths)), num_samples))
            rgb_paths = [rgb_paths[i] for i in indices]
            pose_paths = [pose_paths[i] for i in indices]
        
            if self.is_return_view_idx:
                view_idxs = [view_idxs[i] for i in indices]
                
        for additional_view in self.additional_overall_views:
            additional_dir_path = os.path.join(os.path.dirname(dir_path), additional_view)
            rgb_paths_view = sorted(glob.glob(os.path.join(additional_dir_path, "rgb", "*")))
            pose_paths_view = sorted(glob.glob(os.path.join(additional_dir_path, "pose", "*")))
            assert len(rgb_paths_view) == len(pose_paths_view) and len(rgb_paths_view) == 24, \
                f"View paths length mismatch for view '{additional_view}: len(pose_paths_view): {len(rgb_paths_view)} !=  len(pose_paths_view): {len(pose_paths_view)} != 24'"
            
            if self.is_return_view_idx:
                view_idxs_view = [int(os.path.splitext(os.path.basename(pose_path))[0]) for pose_path in pose_paths_view]
            
            if self.randomly_subsample:
                num_samples = len(rgb_paths_view) // subsample_factor
                
                if num_samples == 0:
                    raise ValueError(f"Subsample factor {subsample_factor} is too large for number of images {len(rgb_paths_view)}")
                if self.dataset_name == "train":
                    # For training, randomly subsample and then slice to imgs_per_obj
                    indices = sorted(random.sample(range(len(rgb_paths_view)), min(num_samples, self.imgs_per_obj)))
                else:
                    indices = sorted(random.sample(range(len(rgb_paths_view)), num_samples))
                rgb_paths_view = [rgb_paths_view[i] for i in indices]
                pose_paths_view = [pose_paths_view[i] for i in indices]
                
                if self.is_return_view_idx:
                    view_idxs_view = [view_idxs_view[i] for i in indices]
                        
            rgb_paths += rgb_paths_view
            pose_paths += pose_paths_view
            if self.is_return_view_idx:
                view_idxs += view_idxs_view
        
        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}
            self.all_view_idxs = {}

        # Skip caching for training dataset since we load different subsets each time
        if self.dataset_name == "train":
            # Don't cache, process data directly
            cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))], no_imgs=self.only_poses)
            
            current_rgbs = []
            current_world_view_transforms = []
            current_view_to_world_transforms = []
            current_full_proj_transforms = []
            current_camera_centers = []
            
            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T
                if self.only_poses: 
                    current_rgbs.append(torch.empty(0)) 
                else: 
                    current_rgbs.append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                current_world_view_transforms.append(world_view_transform)
                current_view_to_world_transforms.append(view_world_transform)
                current_full_proj_transforms.append(full_proj_transform)
                current_camera_centers.append(camera_center)
            
            # Store in temporary variables instead of caching
            self.current_world_view_transforms = torch.stack(current_world_view_transforms)
            self.current_view_to_world_transforms = torch.stack(current_view_to_world_transforms)
            self.current_full_proj_transforms = torch.stack(current_full_proj_transforms)
            self.current_camera_centers = torch.stack(current_camera_centers)
            self.current_rgbs = torch.stack(current_rgbs)
            if self.is_return_view_idx:
                self.current_view_idxs = torch.tensor(view_idxs)
        else:
            # Cache for non-training datasets
            if example_id not in self.all_rgbs.keys():
                self.all_rgbs[example_id] = []
                self.all_world_view_transforms[example_id] = []
                self.all_full_proj_transforms[example_id] = []
                self.all_camera_centers[example_id] = []
                self.all_view_to_world_transforms[example_id] = []
                if self.is_return_view_idx:
                    self.all_view_idxs[example_id] = view_idxs

                cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))], no_imgs=self.only_poses)

                for cam_info in cam_infos:
                    R = cam_info.R
                    T = cam_info.T
                    if self.only_poses: 
                        self.all_rgbs[example_id].append(torch.empty(0)) 
                    else: 
                        self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                                (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                    view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]

                    self.all_world_view_transforms[example_id].append(world_view_transform)
                    self.all_view_to_world_transforms[example_id].append(view_world_transform)
                    self.all_full_proj_transforms[example_id].append(full_proj_transform)
                    self.all_camera_centers[example_id].append(camera_center)
                    
                self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
                self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
                self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
                self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
                self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])
                if self.is_return_view_idx:
                    self.all_view_idxs[example_id] = torch.tensor(self.all_view_idxs[example_id])

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]

        example_id = os.path.basename(os.path.dirname(os.path.dirname(intrin_path)))
        # print("intrin_path", intrin_path)
        
        if self.dataset_name == "test":
            split_to_print = "val_ood" if self.is_val_ood else "val_in_dist" if self.is_val_in_dist else "test"
            # print("example_id", example_id, "split", split_to_print, "overall_view", self.cfg.data.overall_view)

        self.load_example_id(example_id, intrin_path)
        if self.dataset_name == "train":
            # For training, use current data length
            current_data_length = len(self.current_rgbs)
            if self.num_input_images == 1: 
                frame_idxs = torch.randperm(current_data_length)[:self.imgs_per_obj]
                frame_idxs = torch.cat([frame_idxs[:self.num_input_images], frame_idxs], dim=0)
            else: 
                input_idxs = self.test_input_idxs
                assert len(input_idxs) == self.num_input_images, f"input_idxs: {input_idxs} should be of length {self.num_input_images}"
                # deterministic with linspace depending on the number of images
                frame_idxs = torch.randperm(current_data_length)[:self.imgs_per_obj]
                frame_idxs = torch.cat([input_idxs, frame_idxs], dim=0)
                frame_idxs = frame_idxs.type(torch.int)

        else:
            # print("testing, input_idxs:", self.test_input_idxs)
            input_idxs = self.test_input_idxs
            
            # frame_idxs = torch.cat([torch.tensor(input_idxs), 
            #                         torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 
            if self.deterministic_test_idxs is not None:
                frame_idxs = torch.cat([input_idxs, 
                                        torch.tensor([i for i in range(len(self.all_rgbs[example_id])) if i not in input_idxs])], dim=0)
            else: 
                frame_idxs = torch.cat([input_idxs[index:index+1], 
                                        torch.tensor([i for i in range(len(self.all_rgbs[example_id])) if i not in input_idxs])], dim=0)
        try:         
            # Use different data sources based on dataset type
            if self.dataset_name == "train":
                # Use current data (not cached)
                # print("example_id", example_id, "len of current_rgbs", len(self.current_rgbs))
                # print("frame_idxs", frame_idxs) 
                images_and_camera_poses = {
                    "gt_images": self.current_rgbs[frame_idxs].clone(),
                    "world_view_transforms": self.current_world_view_transforms[frame_idxs],
                    # Important: keep absolute copies truly absolute (avoid aliasing the same tensor)
                    "world_view_transforms_absolute": self.current_world_view_transforms[frame_idxs].clone(),
                    "view_to_world_transforms": self.current_view_to_world_transforms[frame_idxs],
                    "view_to_world_transforms_absolute": self.current_view_to_world_transforms[frame_idxs].clone(),
                    "full_proj_transforms": self.current_full_proj_transforms[frame_idxs],
                    "full_proj_transforms_absolute": self.current_full_proj_transforms[frame_idxs].clone(),
                    "camera_centers": self.current_camera_centers[frame_idxs], 
                    "camera_centers_absolute": self.current_camera_centers[frame_idxs].clone(),
                    "example_id": example_id,
                }
                if self.is_return_view_idx:
                    print("is_return_view_idx is True")
                    images_and_camera_poses["view_idxs"] = self.current_view_idxs[frame_idxs]
            else:
                # Use cached data for test/val
                # print("example_id", example_id, "len of self.all_rgbs[example_id]", len(self.all_rgbs[example_id]))
                # print("frame_idxs", frame_idxs) 
                images_and_camera_poses = {
                    "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
                    "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
                    # Important: keep absolute copies truly absolute (avoid aliasing the same tensor)
                    "world_view_transforms_absolute": self.all_world_view_transforms[example_id][frame_idxs].clone(),
                    "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
                    "view_to_world_transforms_absolute": self.all_view_to_world_transforms[example_id][frame_idxs].clone(),
                    "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
                    "full_proj_transforms_absolute": self.all_full_proj_transforms[example_id][frame_idxs].clone(),
                    "camera_centers": self.all_camera_centers[example_id][frame_idxs], 
                    "camera_centers_absolute": self.all_camera_centers[example_id][frame_idxs].clone(),
                    "example_id": example_id,
                }
                if self.is_return_view_idx:
                    print("is_return_view_idx is True")
                    images_and_camera_poses["view_idxs"] = self.all_view_idxs[example_id][frame_idxs]

            images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
            images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
        except Exception as e: 
            print("error in getting item")
            print("frame_idxs:", frame_idxs)
            raise e
        # self.all_rgbs = {}
        # self.all_world_view_transforms = {}
        # self.all_full_proj_transforms = {}
        # self.all_camera_centers = {}
        # self.all_view_to_world_transforms = {}
        return images_and_camera_poses