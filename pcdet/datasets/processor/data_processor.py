from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():  # 用于封装不同版本的体素生成器
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:  # 尝试导入不同版本的体素生成器
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:  # 根据不同版本的体素生成器初始化
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):  # 点云数据作为输入
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points  # 体素，体素坐标，体素中点的数量


class DataProcessor(object):  # 点云数据的预处理流程
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range  # 点云范围
        self.training = training  # 训练模式
        self.num_point_features = num_point_features  # 特征数
        self.mode = 'train' if training else 'test'
        # self.grid_size = self.voxel_size = None     # 初始化网格大小和体素大小  
        self.voxel_sizes = None  # 初始化网格大小和体素大小  
        self.grid_sizes = []  
        self.data_processor_queue = []  # 数据处理队列

        # self.voxel_generator = None    # 体素生成器 
        self.voxel_generators = []  # 体素生成器

        for cur_cfg in processor_configs:  # 遍历传入处理器配置
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):  # 遮蔽超出指定范围的点和边界框
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:  # 移除多余的点
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)  # 基于有效位置确定掩码
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:  # 移除多余的框
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1),
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):  # 随机打乱点云中点的
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']  # 获得点云
            shuffle_idx = np.random.permutation(points.shape[0])  # 随机排序的索引数组
            points = points[shuffle_idx]  # 重新排列点云
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):  # 计算网格大小的占位方法
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)

        return data_dict

    def double_flip(self, points): 
        # y flip
        points_yflip = points.copy()  # 复制原始点云 
        points_yflip[:, 1] = -points_yflip[:, 1] 

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):  # 点云数据转换为体素表示
        if data_dict is None:  
            for voxel_size in config.VOXEL_SIZES:
                # grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)     
                grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(voxel_size)
                # self.grid_sizes = np.round(grid_size).astype(np.int64)   
                self.grid_sizes.append(np.round(grid_size).astype(np.int64))
            # self.grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            # self.voxel_size = config.VOXEL_SIZE 
            self.voxel_sizes = config.VOXEL_SIZES   

            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generators is None or len(self.voxel_generators) == 0:  # 检查是否生成了体素生成器  
            # self.voxel_generator = VoxelGeneratorWrapper(
            #     vsize_xyz=config.VOXEL_SIZE,
            #     coors_range_xyz=self.point_cloud_range,
            #     num_point_features=self.num_point_features,
            #     max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
            #     max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            # )
            for voxel_size in self.voxel_sizes:
                self.voxel_generators.append(VoxelGeneratorWrapper(
                    vsize_xyz=voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=self.num_point_features,
                    max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                ))

        points = data_dict['points']
        data_dict['voxels'] = []  
        data_dict['voxel_coords'] = []
        data_dict['voxel_num_points'] = []
        for voxel_generator in self.voxel_generators: 
            voxel_output = voxel_generator.generate(points)  # 生成体素  
            voxels, coordinates, num_points = voxel_output  # 体素本身，坐标以及每个体素的点数

            if not data_dict['use_lead_xyz']:  # 通常用于不需要位置信息的场景
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

            if config.get('DOUBLE_FLIP', False):  # 如果启用了DOUBLE_FLIP则对翻转后的点云数据体素化
                voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
                points_yflip, points_xflip, points_xyflip = self.double_flip(points)
                points_list = [points_yflip, points_xflip, points_xyflip]
                keys = ['yflip', 'xflip', 'xyflip']
                for i, key in enumerate(keys):
                    voxel_output = voxel_generator.generate(points_list[i]) 
                    voxels, coordinates, num_points = voxel_output

                    if not data_dict['use_lead_xyz']:
                        voxels = voxels[..., 3:]
                    voxels_list.append(voxels)
                    voxel_coords_list.append(coordinates)
                    voxel_num_points_list.append(num_points)

                data_dict['voxels'].append(voxels_list)  # lyt
                data_dict['voxel_coords'].append(voxel_coords_list)
                data_dict['voxel_num_points'].append(voxel_num_points_list)
            else:
                data_dict['voxels'].append(voxels)  # lyt
                data_dict['voxel_coords'].append(coordinates)
                data_dict['voxel_num_points'].append(num_points)
        return data_dict

    def sample_points(self, data_dict=None, config=None):  # 数据采样  按照配置指点数量的点，基于点的深度进行有偏采样
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]  # 从配置中或许要采样的点数
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):  # 配置点数少于实际点数则需要采样
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0  # 近点
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]  # 远点
            choice = []
            if num_points > len(far_idxs_choice):  # 如果需要的点数大于远点的数量，则从近点补足
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):  # 计算网格大小，用于设置体素的尺寸和数量
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):  # 对深度图进行下采样
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(  # 局部均值下采样
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def image_normalize(self, data_dict=None, config=None):  # 标准化图像数据
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict

    def image_calibrate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
