import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int64)  
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in  


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F  

        spatial_features_ori = data_dict['multi_scale_2d_features'] 
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None

        li = list(spatial_features_ori.keys())
        li.reverse() 

        for idx in li:  
            spatial_features = spatial_features_ori[idx]
            x_conv4 = spatial_features['x_conv4']
            x_conv5 = spatial_features['x_conv5']

            ups = [self.deblocks[0](x_conv4)]

            x = self.blocks[1](x_conv5)
            ups.append(self.deblocks[1](x))

            x = torch.cat(ups, dim=1)
            x = self.blocks[0](x)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                print("=========one vovel========")

        data_dict['spatial_features_2d'] = fused_features

        return data_dict

class BaseBEVBackboneV1LYT(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in 

    def forward(self, spatial_features):  
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        # data_dict['spatial_features_2d'] = x

        return x

class BaseBEVBackboneV1LYTV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in 

    def forward(self, spatial_features): 
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        import torch.nn.functional as F  
        min_h = min([up.shape[2] for up in ups])
        min_w = min([up.shape[3] for up in ups])
        ups = [F.interpolate(up, size=(min_h, min_w), mode='bilinear', align_corners=False) for up in ups]
        x = torch.cat(ups, dim=1)

        x = self.blocks[0](x)

        # data_dict['spatial_features_2d'] = x

        return x

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            padding: int = 1,
            downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out

class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class BaseBEVBackboneV1LYTDown11(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.singleBase = nn.ModuleList()
        self.voxel_sizes = self.model_cfg.VOXEL_SIZES 
        for i in range(len(self.voxel_sizes)):
            self.singleBase.append(BaseBEVBackboneV1LYT(model_cfg))

        c_in = sum(num_upsample_filters)

        self.num_bev_features = c_in

        channels = num_filters[0]
        # self.down_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        # self.lateral_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F

        spatial_features_ori = data_dict['multi_scale_2d_features'] 
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        for idx in li: 
            module = self.singleBase[idx]

            spatial_features = spatial_features_ori[idx]  
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                # data_dict['spatial_features_2d'].append(x)
                # 上采样低分辨率特征图
                # print(x.size()[2], x.size()[3])
                down_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                         align_corners=False)  
                down_1 = self.donw_conv(down_0)
                # data_dict['spatial_features_2d_ups'].append(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))     
                combined_feature = torch.cat([x, down_1], dim=1)
                fused_features = self.fusion_conv(combined_feature)

        data_dict['spatial_features_2d'] = fused_features

        return data_dict

class BaseBEVBackboneV1LYTUP11(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.singleBase = nn.ModuleList()
        self.voxel_sizes = self.model_cfg.VOXEL_SIZES  
        for i in range(len(self.voxel_sizes)):
            self.singleBase.append(BaseBEVBackboneV1LYT(model_cfg))

        c_in = sum(num_upsample_filters)

        self.num_bev_features = c_in 

        channels = num_filters[0]

        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F 

        spatial_features_ori= data_dict['multi_scale_2d_features'] 
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        for idx in li :  
            module = self.singleBase[idx]

            spatial_features = spatial_features_ori[idx]  
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                up_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                       align_corners=False) 
                up_1 = self.up_conv(up_0)
                # data_dict['spatial_features_2d_ups'].append(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))     
                combined_feature = torch.cat([x, up_1], dim=1)
                fused_features = self.fusion_conv(combined_feature)

        data_dict['spatial_features_2d'] = fused_features

        return data_dict

class BaseBEVBackboneV1LYTDownAtten(nn.Module): 
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.singleBase = nn.ModuleList()
        self.voxel_sizes = self.model_cfg.VOXEL_SIZES  
        for i in range(len(self.voxel_sizes)):
            self.singleBase.append(BaseBEVBackboneV1LYT(model_cfg))

        c_in = sum(num_upsample_filters)

        self.num_bev_features = c_in  # 特征汇总

        channels = num_filters[0]
        # self.down_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        # self.lateral_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.down_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.Sigmoid()  
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F 
        
        spatial_features_ori = data_dict['multi_scale_2d_features'] 
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        for idx in li: 
            module = self.singleBase[idx]

            spatial_features = spatial_features_ori[idx]  
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                down_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                         align_corners=False) 
                down_1 = self.down_conv(down_0)
                # data_dict['spatial_features_2d_ups'].append(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))     
                combined_feature = torch.cat([x, down_1], dim=1)

                attention_weights = self.attention(combined_feature)
                attention_feature = combined_feature * attention_weights
                fused_features = self.fusion_conv(attention_feature)

        data_dict['spatial_features_2d'] = fused_features

        return data_dict


class BaseBEVBackboneV1LYTUPAtten(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.singleBase = nn.ModuleList()
        self.voxel_sizes = self.model_cfg.VOXEL_SIZES  
        for i in range(len(self.voxel_sizes)):
            self.singleBase.append(BaseBEVBackboneV1LYT(model_cfg))

        c_in = sum(num_upsample_filters)


        self.num_bev_features = c_in  
        
        channels = num_filters[0]

        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.Sigmoid() 
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F 

        spatial_features_ori = data_dict['multi_scale_2d_features']  
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        for idx in li: 
            module = self.singleBase[idx] 

            spatial_features = spatial_features_ori[idx]
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                up_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                       align_corners=False)  
                up_1 = self.up_conv(up_0)
                # data_dict['spatial_features_2d_ups'].append(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))     

                combined_feature = torch.cat([x, up_1], dim=1)
                attention_weights = self.attention(combined_feature)
                attended_feature = combined_feature * attention_weights
                fused_features = self.fusion_conv(attended_feature)

        data_dict['spatial_features_2d'] = fused_features

        return data_dict

class BaseBEVBackboneV1LYTDownAttenAddRes(nn.Module): 
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.singleBase = nn.ModuleList()
        self.voxel_sizes = self.model_cfg.VOXEL_SIZES   
        for i in range(len(self.voxel_sizes)):
            self.singleBase.append(BaseBEVBackboneV1LYT(model_cfg))

        c_in = sum(num_upsample_filters)


        self.num_bev_features = c_in 
        channels = num_filters[0]

        self.down_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.Sigmoid() 
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F 

        spatial_features_ori = data_dict['multi_scale_2d_features'] 
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        for idx in li: 
            module = self.singleBase[idx]

            spatial_features = spatial_features_ori[idx]  
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                down_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                         align_corners=False)  
                down_1 = self.down_conv(down_0)
                # data_dict['spatial_features_2d_ups'].append(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))     

                combined_feature = torch.cat([x, down_1], dim=1)

                attention_weights = self.attention(combined_feature)
                attention_feature = combined_feature * attention_weights

                fused_features = attention_feature + combined_feature   

        data_dict['spatial_features_2d'] = fused_features

        return data_dict


class BaseBEVBackboneV1LYTUPAttenInteg(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.neck = BaseBEVBackboneV1LYT(model_cfg)

        c_in = sum(num_upsample_filters)


        self.num_bev_features = c_in  
        channels = num_filters[0]

        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),  
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.Sigmoid() 
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F 

        spatial_features_ori = data_dict['multi_scale_2d_features']  
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        module = self.neck

        for idx in li: 

            spatial_features = spatial_features_ori[idx] 
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                up_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                       align_corners=False) 
                up_1 = self.up_conv(up_0)
                # data_dict['spatial_features_2d_ups'].append(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))     

                combined_feature = torch.cat([x, up_1], dim=1)

                attention_weights = self.attention(combined_feature)
                attended_feature = combined_feature * attention_weights
                fused_features = self.fusion_conv(attended_feature)

        data_dict['spatial_features_2d'] = fused_features

        return data_dict

class BaseBEVBackboneV1LYTUPAttenV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.singleBase = nn.ModuleList()
        self.voxel_sizes = self.model_cfg.VOXEL_SIZES  
        for i in range(len(self.voxel_sizes)):
            self.singleBase.append(BaseBEVBackboneV1LYTV2(model_cfg))

        c_in = sum(num_upsample_filters)


        self.num_bev_features = c_in 

        channels = num_filters[0]

        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.Sigmoid() 
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        import torch.nn.functional as F 

        spatial_features_ori = data_dict['multi_scale_2d_features']  
        data_dict['spatial_features_2d'] = []
        # data_dict['spatial_features_2d_ups'] = []
        fused_features = None
        
        li = list(spatial_features_ori.keys())
        li.reverse()

        for idx in li:  
            module = self.singleBase[idx]  

            spatial_features = spatial_features_ori[idx] 
            x = module(spatial_features)

            # data_dict['spatial_features_2d'].append(x)

            if (fused_features == None):
                fused_features = x
            else:
                up_0 = F.interpolate(fused_features, size=(x.size()[2], x.size()[3]), mode='bilinear',
                                       align_corners=False)  
                up_1 = self.up_conv(up_0)

                combined_feature = torch.cat([x, up_1], dim=1)
                attention_weights = self.attention(combined_feature)  
                attended_feature = combined_feature * attention_weights
                fused_features = self.fusion_conv(attended_feature)

        data_dict['spatial_features_2d'] = fused_features

        return data_dict
