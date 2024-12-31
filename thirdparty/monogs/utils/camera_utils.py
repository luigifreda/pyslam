import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2

import os
import sys
try: 
    from slam_utils import image_gradient, image_gradient_mask
except:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, "."))
    from .slam_utils import image_gradient, image_gradient_mask

import torch.nn.functional as F

#import json
import ujson as json


class CameraViewpoint(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(CameraViewpoint, self).__init__()
        self.uid = uid
        self.device = device

        self.T = torch.eye(4, device=device).to(torch.float32)
        self.T_gt = gt_T.to(device=device).to(torch.float32).clone() if gt_T is not None else None
        
        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return CameraViewpoint(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return CameraViewpoint(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return self.T.transpose(0, 1).to(device=self.device)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform #TODO: Need to invert for high order SHs by inverse_t(self.world_view_transform).

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            size = 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            I = img_grad_intensity.unsqueeze(0)
            I_unf = F.unfold(I, size, stride=size)
            median_patch, _ = torch.median(I_unf, dim=1,keepdim=True)
            mask = (I_unf > (median_patch * multiplier)).float()
            I_f = F.fold(mask, I.shape[-2:],size,stride=size).squeeze(0)
            self.grad_mask = I_f
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        gt_image = self.original_image.cuda()
        _, h, w = self.original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        self.rgb_pixel_mask = rgb_pixel_mask * self.grad_mask
        self.rgb_pixel_mask_mapping = rgb_pixel_mask
        
        if self.depth is not None:
            self.gt_depth = torch.from_numpy(self.depth).to(
            dtype=torch.float32, device=self.device
        )[None]

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
        
        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None

    def to_json(self):
        """
        Serializes the object to a JSON string.
        """
        data = {
            "uid": self.uid,
            "T": self.T.tolist(),
            "T_gt": self.T_gt.tolist() if self.T_gt is not None else None,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "FoVx": self.FoVx,
            "FoVy": self.FoVy,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "cam_rot_delta": self.cam_rot_delta.detach().cpu().tolist(),
            "cam_trans_delta": self.cam_trans_delta.detach().cpu().tolist(),
            "exposure_a": self.exposure_a.detach().cpu().tolist(),
            "exposure_b": self.exposure_b.detach().cpu().tolist(),
            "projection_matrix": self.projection_matrix.tolist(),
        }
        return json.dumps(data)

    @staticmethod
    def from_json(json_str, device="cuda:0"):
        """
        Creates an instance of the class from a JSON string.
        """
        data = json.loads(json_str)
        obj = CameraViewpoint(
            uid=data["uid"],
            color=None,  # Placeholder; provide actual tensor when loading
            depth=None,  # Placeholder; provide actual tensor when loading
            gt_T=torch.tensor(data["T_gt"], device=device) if data["T_gt"] is not None else None,
            projection_matrix=torch.tensor(data["projection_matrix"], device=device),
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            fovx=data["FoVx"],
            fovy=data["FoVy"],
            image_height=data["image_height"],
            image_width=data["image_width"],
            device=device,
        )
        obj.cam_rot_delta.data = torch.tensor(data["cam_rot_delta"], device=device)
        obj.cam_trans_delta.data = torch.tensor(data["cam_trans_delta"], device=device)
        obj.exposure_a.data = torch.tensor(data["exposure_a"], device=device)
        obj.exposure_b.data = torch.tensor(data["exposure_b"], device=device)
        obj.T = torch.tensor(data["T"], device=device)
        return obj

    def save(self, path):
        """
        Saves the object to a file.
        """
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path, device="cuda:0"):
        """
        Loads the object from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            json_str = f.read()
        return CameraViewpoint.from_json(json_str, device=device)


class CameraMsg():
    def __init__(self, Camera):
        self.uid = Camera.uid
        self.T = Camera.T
        self.T_gt = Camera.T_gt