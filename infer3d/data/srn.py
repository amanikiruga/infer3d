import glob
import os

import csv 
import numpy as np
import torch
from torch.utils.data import Dataset
import os 

from .readers import readCamerasFromTxt
from ..utils.general import PILtoTorch, matrix_to_quaternion
from ..utils.graphics import getWorld2View2, getProjectionMatrix, getView2World

from .shared import SharedDataset

SHAPENET_DATASET_ROOT = os.getenv("SRN_CARS_ROOT") # os.getenv("SRN_CHAIRS_ROOT")
assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

def load_override_from_csv(csv_path):
    override_first_train_img_map = {}

    # Open the CSV file and read its contents
    with open(csv_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV and build the dictionary
        for row in csv_reader:
            object_id = row["object_id"]
            override_first_train_img_map[object_id] = {
                "img_id": row["highest_img_id"],
                "lowest_img": row["lowest_img_id"],
                "highest_psnr": float(row["highest_psnr"]),
                "lowest_psnr": float(row["lowest_psnr"]),
            }
    
    return override_first_train_img_map

class SRNDataset(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train", override_image_paths = None, override_object_ids = None, override_first_train_img_path=None, override_test_idx=None, data_category=None):
        super().__init__()
        self.cfg = cfg


        if override_first_train_img_path is not None: 
            self.override_first_train_img_map = load_override_from_csv(override_first_train_img_path)
            print(f"Overriding first train image with {override_first_train_img_path}")

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"

        # self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/{}_{}".format((data_category or cfg.data.category),
        #                                                                            (data_category or cfg.data.category),
        #                                                                            self.dataset_name))
        
        if "all" not in dataset_name:
            self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "{}_{}".format((data_category or cfg.data.category),
                                                                                    self.dataset_name))
            print("intrins_search_path: ", self.base_path)
            self.intrins = sorted(
                glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
            )
        elif dataset_name == "all":
            self.base_path = SHAPENET_DATASET_ROOT
            self.intrins = sorted(
                glob.glob(os.path.join(self.base_path, f"{(data_category or cfg.data.category)}_train", "*", "intrinsics.txt"))
            )
            
            self.intrins += sorted(
                glob.glob(os.path.join(self.base_path, f"{(data_category or cfg.data.category)}_test", "*", "intrinsics.txt"))
            )
        else:
            self.base_path = SHAPENET_DATASET_ROOT
            split_name = dataset_name.split("_")[1] # eg. all_train = train
            self.intrins = sorted(
                glob.glob(os.path.join(self.base_path, f"{(data_category or cfg.data.category)}_{split_name}", "*", "intrinsics.txt"))
            )
            
        if cfg.data.is_override_image_ids and override_image_paths is not None:
            cur_seq_ids = [os.path.basename(os.path.dirname(intrin_path)) for intrin_path in self.intrins]
            override_seq_ids = [os.path.basename(os.path.dirname(os.path.dirname(image_id_path))) for image_id_path in override_image_paths]   
            self.intrins = [intrin_path for intrin_path, seq_id in zip(self.intrins, cur_seq_ids) if seq_id in override_seq_ids]
            override_image_ids = [int(os.path.basename(image_id_path).split(".")[0]) for image_id_path in override_image_paths]
            self.override_image_ids = dict(zip(override_seq_ids, override_image_ids))
        
        else: 
            self.override_image_ids = None


        if override_object_ids is not None:
            self.intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(intrin_path)) in override_object_ids]
            print(f"Overriding object ids with {override_object_ids}")
        
        print(len(self.intrins))
        assert len(self.intrins) > 0, f"No SRN cars intrinsics found for {dataset_name} dataset name and {(data_category or cfg.data.category)} category"
        if cfg.data.subset != -1:
            self.intrins = self.intrins[:cfg.data.subset]

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.cfg.data.input_images == 1:
            if override_test_idx is not None: 
                self.test_input_idxs = [override_test_idx]
            else: 
                self.test_input_idxs = [64]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [64, 128]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.intrins)

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        assert len(rgb_paths) == len(pose_paths)

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []

            cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))])

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)
                # this is AI camera center
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

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id
    
    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        
            

        self.load_example_id(example_id, intrin_path)
        if self.cfg.data.is_override_image_ids and self.override_image_ids is not None:
            print("Override image id is on for dataloader.")
            override_image_id = self.override_image_ids[example_id]
            assert override_image_id is not None, "Override image id is not found for example_id: {}".format(example_id)

            if self.dataset_name == "train":
                frame_idxs = torch.randperm(len(self.all_rgbs[example_id]))
                frame_idxs = torch.tensor([x for x in frame_idxs if x != override_image_id])[:self.imgs_per_obj-1]
                frame_idxs = torch.cat([torch.tensor([override_image_id]), frame_idxs], dim=0)
            
            else:
                input_idxs = [override_image_id]
                frame_idxs = torch.cat([torch.tensor(input_idxs),
                                        torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0)
                
        elif self.dataset_name == "train" and self.cfg.data.is_deterministic:
            # print("Deterministic mode is on for datalaoder.")
            frame_idxs = torch.arange(0, len(self.all_rgbs[example_id]))[:self.imgs_per_obj]

            frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs[self.cfg.data.input_images:]], dim=0)
        elif self.dataset_name == "train":
            frame_idxs = torch.randperm(
                    len(self.all_rgbs[example_id])
                    )[:self.imgs_per_obj]
            frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)
        elif "all" in self.dataset_name: 
            frame_idxs = torch.arange(0, len(self.all_rgbs[example_id]))
        elif self.dataset_name == "test" and hasattr(self, "override_first_train_img_map"): 
            if example_id not in self.override_first_train_img_map.keys():
                raise ValueError(f"Override first train image is not found for example_id: {example_id}")
                print(self.override_first_train_img_map.keys())
            print("Override first train image is on for dataloader.")
            override_image_id = self.override_first_train_img_map[example_id]["lowest_img"]
            override_image_id = int(override_image_id.replace(".png", ""))
            input_idxs = [override_image_id]
            
            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 
        else: 
            input_idxs = self.test_input_idxs
            
            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 

        images_and_camera_poses = {
            "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
            "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
            "world_view_transforms_absolute": self.all_world_view_transforms[example_id][frame_idxs].clone(),
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
            "view_to_world_transforms_absolute": self.all_view_to_world_transforms[example_id][frame_idxs].clone(),
            "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
            "full_proj_transforms_absolute": self.all_full_proj_transforms[example_id][frame_idxs].clone(),
            "camera_centers": self.all_camera_centers[example_id][frame_idxs],
            "camera_centers_absolute": self.all_camera_centers[example_id][frame_idxs].clone(),
            "example_id": example_id
        }

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses