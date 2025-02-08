import os.path as osp
from glob import glob

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

from typing import NamedTuple
class CameraInfo(NamedTuple):
    c2w: np.array
    image_name: str

class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False,
                 scale_factor=0, crop=0, depth_scale=1000.0, max_depth=10, **kwargs) -> None:
        self.crop = crop
        self.depth_scale = depth_scale
        self.data_path = data_path
        self.scale_factor = scale_factor
        self.use_gt = use_gt
        num_imgs = len(glob(osp.join(data_path, 'train/ours_30000/gt/*.png')))
        self.max_depth = max_depth
        self.gt_dir = "/root/autodl-tmp/Proj/3Dv_Reconstruction/GS-Reconstruction/Data/ScanNetpp/" +  data_path.split('/')[-1]

        # self.K = self.load_intrinsic()
        self.depth_files = [
            osp.join(data_path, 'train/ours_30000/renders_expected_depth/{0:05d}.npz'.format(i)) for i in range(num_imgs)]
        self.image_files = [
            osp.join(data_path, 'train/ours_30000/gt/{0:05d}.png'.format(i)) for i in range(num_imgs)]
        # self.pose_files = [
        #     osp.join(data_path, 'train/ours_30000/pose/{0:05d}.txt'.format(i)) for i in range(num_imgs)]
        
        # read camera extrinsics and intrinsics from gs dataset
        cameras_extrinsic_file = os.path.join(self.gt_dir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(self.gt_dir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        # loading intrinsics
        focal_length_x = cam_intrinsics[1].params[0]
        focal_length_y = cam_intrinsics[1].params[1]
        (height, width, _)= ( self.load_image(0) ).shape
        self.K = np.array([[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])

        # loading extrinsics
        cam_infos_unsorted = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]

            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            cam_info = CameraInfo(c2w = np.linalg.inv(getWorld2View2(R, T)), image_name = extr.name.split(".")[0])
            cam_infos_unsorted.append(cam_info)

        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

        # train_test_lists in gs data
        import json
        train_test_lists = json.load(open(os.path.join(self.gt_dir, "train_test_lists.json")))
        train_cam_ids = [i.split(".")[0] for i in train_test_lists["train"]]
        train_cam_infos = [c for c in cam_infos if c.image_name in train_cam_ids]

        self.pose_lis = [c.c2w for c in train_cam_infos]

        self.num_imgs = num_imgs

    def load_intrinsic(self):
        self.K = np.loadtxt(
            osp.join(self.data_path, 'intrinsic/intrinsic_depth.txt'))[:3, :3]
        if self.scale_factor > 0:
            scale = 2 ** self.scale_factor
            self.K = self.K / scale
            self.K[2, 2] = 1
        if self.crop > 0:
            self.K[0, 2] = self.K[0, 2] - self.crop
            self.K[1, 2] = self.K[1, 2] - self.crop
        return self.K

    def load_depth(self, index):
        depth = np.load(self.depth_files[index], -1)['arr_0'] / self.depth_scale
        depth[depth > self.max_depth] = 0
        if self.scale_factor > 0:
            skip = 2 ** self.scale_factor
            depth = depth[::skip, ::skip]
        if self.crop > 0:
            depth = depth[self.crop:-self.crop, self.crop:-self.crop]
        return depth

    def get_init_pose(self, init_frame=None):
        # return np.loadtxt(self.pose_files[init_frame])
        return self.pose_lis[init_frame]

    def load_image(self, index):
        img = cv2.imread(self.image_files[index], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.scale_factor > 0:
            factor = 2 ** self.scale_factor
            size = (640 // factor, 480 // factor)
            img = cv2.resize(img, size, cv2.INTER_AREA)
        if self.crop > 0:
            img = img[self.crop:-self.crop, self.crop:-self.crop]
        return img / 255.0

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, index):
        img = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()
        # pose = np.loadtxt(self.pose_files[index]) if self.use_gt else None
        pose = self.pose_lis[index] if self.use_gt else None
        return index, img, depth, self.K, pose


if __name__ == '__main__':
    import sys

    loader = DataLoader(sys.argv[1], 1)
    for data in loader:
        index, img, depth = data
        print(index, img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(1)
