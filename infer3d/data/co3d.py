import glob
import os

from einops import repeat

import numpy as np
import torch


from .bad_sequences import (
    NAN_SEQUENCES, 
    NO_FG_COND_FRAME_SEQ, 
    LARGE_FOCAL_FRAME_SEQ,
    EXCLUDE_SEQUENCE,
    CAMERAS_CLOSE_SEQUENCE,
    CAMERAS_FAR_AWAY_SEQUENCE,
    LOW_QUALITY_SEQUENCE
    )

from .shared import SharedDataset

from .readers import readCamerasFromNpy
from infer3d.utils.general import matrix_to_quaternion
from infer3d.utils.graphics import getWorld2View2, getProjectionMatrix, getView2World, fov2focal

from infer3d import config
CO3D_DATASET_ROOT = config.CO3D_DATASET_ROOT          # set via infer3d/config.py or $CO3D_DATASET_ROOT
CO3D_DATASET_ROOT_HQ = config.CO3D_DATASET_ROOT_HQ
assert CO3D_DATASET_ROOT is not None, "Update the location of the CO3D Dataset (infer3d/config.py)"

class CO3DDataset(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train", use_hq=False):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.use_hq = use_hq
        if dataset_name == "vis":
            self.dataset_name = "test"

        if "train_deterministic" in self.cfg.data and self.cfg.data.train_deterministic and dataset_name == "train":
            print("train_deterministic is on for dataloader.")
        if "load_hq_version" in self.cfg.data and self.cfg.data.load_hq_version:
            print("Also loading HQ version of CO3D dataset")
            
        # assumes cfg.data.category ends with an "s", for example hydrantS, which
        # is not included in the dataset name 

        if use_hq:
            assert "load_hq_version" not in self.cfg.data, (
                "Don't pass both use_hq and cfg.data.load_hq_version: "
                "the latter is meant to load both 128 and 1080 versions."
            )
            dataset_root = CO3D_DATASET_ROOT_HQ
            print(f"Using HQ version of CO3D dataset from {dataset_root}")
        else:
            dataset_root = CO3D_DATASET_ROOT

        self.base_path = os.path.join(dataset_root, 
                                      "co3d_{}_for_gs".format(cfg.data.category[:-1]), 
                                      self.dataset_name)
        # self.base_path = os.path.join(CO3D_DATASET_ROOT, 
        #                               "co3d_car_for_gs", 
        #                               self.dataset_name)

        frame_order_files = sorted(
            glob.glob(os.path.join(self.base_path, "*", "frame_order.txt"))
        )
        self.frame_order_files = []
        try: 
            exclude_sequences = NO_FG_COND_FRAME_SEQ[cfg.data.category[:-1]] + \
                LARGE_FOCAL_FRAME_SEQ[cfg.data.category[:-1]] + \
                NAN_SEQUENCES[cfg.data.category[:-1]] + \
                EXCLUDE_SEQUENCE[cfg.data.category[:-1]] + \
                CAMERAS_CLOSE_SEQUENCE[cfg.data.category[:-1]] + \
                CAMERAS_FAR_AWAY_SEQUENCE[cfg.data.category[:-1]] + \
                LOW_QUALITY_SEQUENCE[cfg.data.category[:-1]]
        except: 
            exclude_sequences = []

        self.read_cameras()
        # Check that the sequence was included in the preprocessed sequences
        for frame_order_file in frame_order_files:
            if os.path.basename(os.path.dirname(frame_order_file)) not in exclude_sequences:
                if os.path.basename(os.path.dirname(frame_order_file)) not in self.Ts.keys():
                    print(frame_order_file)
                else:
                    self.frame_order_files.append(frame_order_file)

        if cfg.data.subset != -1:
            self.frame_order_files = self.frame_order_files[:cfg.data.subset]

        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        if self.cfg.data.input_images == 1:
            self.test_input_idxs = [0]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [0, 30]
        else:
            raise NotImplementedError

        self.init_ray_dirs()

    def __len__(self):
        return len(self.frame_order_files)

    def init_ray_dirs(self):
        """
        Initialises an image of ray directions, unscaled by focal lengths.
        """
        x = torch.linspace(-63.5, 63.5, 128) 
        y = torch.linspace(63.5, -63.5, 128)
        if self.cfg.model.inverted_x:
            x = -x
        if self.cfg.model.inverted_y:
            y = -y
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
        ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)
        self.ray_dirs = ray_dirs

    def get_origin_distance(self, camera_to_world):
        # outputs the origin_distances. This helps resolve depth 
        # ambiguity in single-view depth estimation. Follows PixelNeRF
        camera_center_to_origin = - camera_to_world[3, :3]
        camera_z_vector = camera_to_world[2, :3]
        origin_distances = torch.dot(camera_center_to_origin, camera_z_vector).unsqueeze(0)
        origin_distances = repeat(origin_distances, 'c -> c h w', 
                                h=self.cfg.data.training_resolution, w=self.cfg.data.training_resolution)

        return origin_distances

    def read_cameras(self):
        self.Ts = np.load(os.path.join(self.base_path, "camera_Ts.npz"))
        self.Rs = np.load(os.path.join(self.base_path, "camera_Rs.npz"))

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0, load_hq_version=False):
        """
        Reads an example from storage if it has not been read already.
        """
        
        
        dir_path = os.path.dirname(intrin_path)
        focals_folder_path = os.path.join(self.base_path,
                                          os.path.basename(dir_path))

        rgb_path = os.path.join(dir_path, "images_fg.npy")

        dir_path_hq = dir_path.replace(CO3D_DATASET_ROOT, CO3D_DATASET_ROOT_HQ)
        assert dir_path_hq != dir_path or self.use_hq, "dir_path_hq is the same as dir_path"
        rgb_path_hq = os.path.join(dir_path_hq, "images_fg.npy")

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_rgbs_hq = {}
            self.all_origin_distances = {}
            self.all_ray_embeddings = {}

            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}
            self.all_focals_pixels = {}

        if example_id not in self.all_rgbs.keys():
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []
            self.all_focals_pixels[example_id] = []

            self.all_ray_embeddings[example_id] = []
            self.all_origin_distances[example_id] = []

            images = np.load(rgb_path)
            self.all_rgbs[example_id] = torch.from_numpy(images)
            
            if load_hq_version:
                images_hq = np.load(rgb_path_hq)
                assert images_hq.shape[0] == images.shape[0], f"images_hq and images have number of images, hq version {images_hq.shape} != {images.shape}"
                self.all_rgbs_hq[example_id] = torch.from_numpy(images_hq)

            # print("Loaded example with {} frames".format(len(images)))
            # print("Loading focals from {}".format(focals_folder_path))
            w2c_Ts_rmo = self.Ts[example_id]
            w2c_Rs_rmo = self.Rs[example_id]

            # Read cameras, convert into our camera convention and compute full projection matrices
            cam_infos = readCamerasFromNpy(dir_path, 
                                           w2c_Rs_rmo=w2c_Rs_rmo, 
                                           w2c_Ts_rmo=w2c_Ts_rmo,
                                           focals_folder_path=focals_folder_path)

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                projection_matrix = getProjectionMatrix(
                        znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
                        fovX=cam_info.FovX, 
                        fovY=cam_info.FovY
                    ).transpose(0,1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)
                self.all_focals_pixels[example_id].append(torch.tensor([fov2focal(cam_info.FovX, 128),
                                                                        fov2focal(cam_info.FovY, 128)]))

                ray_dirs = self.ray_dirs.clone()[0]
                ray_dirs[:2, ...] = ray_dirs[:2, ...] / self.all_focals_pixels[example_id][-1].unsqueeze(1).unsqueeze(2)
                self.all_ray_embeddings[example_id].append(ray_dirs)

                self.all_origin_distances[example_id].append(
                    self.get_origin_distance(self.all_view_to_world_transforms[example_id][-1]))

            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_focals_pixels[example_id] = torch.stack(self.all_focals_pixels[example_id])
            self.all_origin_distances[example_id] = torch.stack(self.all_origin_distances[example_id])
            self.all_ray_embeddings[example_id] = torch.stack(self.all_ray_embeddings[example_id])
    
    def find_index_for_sequence_prefix(self, short_id: str):
        """
        Given a short CO3D id like '429_60420', find the dataset index whose
        full example_id starts with this prefix (e.g. '429_60420_117369').

        Returns the unique matching index, or raises a ValueError with details.
        """
        matches = []
        for idx, frame_order_file in enumerate(self.frame_order_files):
            example_id = os.path.basename(os.path.dirname(frame_order_file))
            if example_id == short_id or example_id.startswith(short_id + "_"):
                matches.append((idx, example_id))

        if len(matches) == 0:
            raise ValueError(
                f"No example_id starting with '{short_id}' found in CO3DDataset.\n"
                f"Hint: available example_ids (first 10): "
                f"{[os.path.basename(os.path.dirname(f)) for f in self.frame_order_files[:10]]}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple example_ids start with '{short_id}': {matches}. "
                f"Please disambiguate."
            )
        return matches[0][0], matches[0][1]

    def find_index_for_ply_path(self, ply_path: str):
        """
        Given a FastGS .ply path like:

          <root>/co3d_hydrant_for_gs/test/hydrant_429_60420/point_cloud/iteration_30000/point_cloud.ply

        extract the short id '429_60420' and return the matching dataset index
        whose example_id starts with '429_60420' (e.g. '429_60420_117369').
        """
        # Go up three levels: point_cloud.ply -> iteration_xxx -> point_cloud -> sequence_dir
        seq_dir = os.path.dirname(os.path.dirname(os.path.dirname(ply_path)))
        seq_name = os.path.basename(seq_dir)  # e.g. 'hydrant_429_60420'

        parts = seq_name.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Cannot parse sequence name '{seq_name}' from ply path '{ply_path}'. "
                f"Expected something like 'hydrant_429_60420'."
            )

        short_id = "_".join(parts[-2:])  # -> '429_60420'
        return self.find_index_for_sequence_prefix(short_id)


    def get_example_id(self, index):
        intrin_path = self.frame_order_files[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.frame_order_files[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        if "load_hq_version" in self.cfg.data and self.cfg.data.load_hq_version:
            load_hq_version = True
        else:
            load_hq_version = False
        self.load_example_id(example_id, intrin_path, load_hq_version=load_hq_version)
        if self.dataset_name == "train":
            if "train_deterministic" in self.cfg.data and self.cfg.data.train_deterministic:
                frame_idxs = torch.arange(0, len(self.all_rgbs[example_id]))[:self.imgs_per_obj]
                frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)
            else: 
                frame_idxs = torch.randperm(
                        len(self.all_rgbs[example_id])
                        )[:self.imgs_per_obj]
                frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)
        else:
            input_idxs = self.test_input_idxs
            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(len(self.all_rgbs[example_id])) if i not in input_idxs])], dim=0) 

        images_and_camera_poses = {
            "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
            "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
            "world_view_transforms_absolute": self.all_world_view_transforms[example_id][frame_idxs],
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
            "view_to_world_transforms_absolute": self.all_view_to_world_transforms[example_id][frame_idxs],
            "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
            "full_proj_transforms_absolute": self.all_full_proj_transforms[example_id][frame_idxs],
            "camera_centers": self.all_camera_centers[example_id][frame_idxs],
            "camera_centers_absolute": self.all_camera_centers[example_id][frame_idxs],
            "focals_pixels": self.all_focals_pixels[example_id][frame_idxs].clone(),
            "origin_distances": self.all_origin_distances[example_id][frame_idxs],
            "ray_embeddings": self.all_ray_embeddings[example_id][frame_idxs],
            "example_id": example_id
        }

        if "load_hq_version" in self.cfg.data and self.cfg.data.load_hq_version:
            images_and_camera_poses["gt_images_hq"] = self.all_rgbs_hq[example_id][frame_idxs].clone()

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)

        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
        images_and_camera_poses["source_cv2wT_quat_absolute"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms_absolute"])


        # Check that data does not have NaN values
        for k, v in images_and_camera_poses.items():
            if isinstance(v, torch.Tensor):
                assert torch.all(torch.logical_not(v.isnan())), "Found a nan value"

        # clear the entire dictionary by deleting everything in it
        if "clear_cache" in self.cfg.data and self.cfg.data.clear_cache:
            del self.all_rgbs[example_id]
            del self.all_rgbs_hq[example_id]
            del self.all_world_view_transforms[example_id]
            del self.all_view_to_world_transforms[example_id]
            del self.all_full_proj_transforms[example_id]
            del self.all_camera_centers[example_id]
            del self.all_focals_pixels[example_id]
            del self.all_origin_distances[example_id]
            del self.all_ray_embeddings[example_id]
            

        return images_and_camera_poses