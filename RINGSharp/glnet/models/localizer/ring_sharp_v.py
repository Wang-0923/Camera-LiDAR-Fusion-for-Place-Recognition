import copy
import os
import sys
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic
from glnet.utils.params import ModelParams
from glnet.config.config import *
from glnet.models.backbones_2d.base_lss_fpn import BaseLSSFPN
from glnet.models.backbones_2d.bev_depth_head import BEVDepthHead
from glnet.models.backbones_2d.unet import conv_block_unet, conv_last_block_unet, AdaptationBlock
from glnet.models.localizer.ring_sharp_vl import run_ring_sharp_downstream
from glnet.utils.common_utils import _ex

EPS = 1e-4

from glnet.models.aggregation.GeM import GeM
from glnet.models.aggregation.NetVLADLoupe import NetVLADLoupe
from glnet.models.localizer.spec_head import SpecGlobalDescriptorHead


class RINGSharpV(nn.Module):
    '''Modified from `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        yaw_mode (int: 0 or 1): 0 for applying convs before sinogram,
        1 for applying convs after sinogram.
            Default: True.
    '''

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, model_params: ModelParams, backbone_conf=backbone_conf):
        super(RINGSharpV, self).__init__()

        backbone_conf = copy.deepcopy(backbone_conf)
        
        self.yaw_mode = 1 # 1
        self.trans_mode = 0 # 0

        if model_params.dataset_type == 'nclt':
            backbone_conf['final_dim'] = (224, 384)
        elif model_params.dataset_type == 'oxford':
            backbone_conf['final_dim'] = (320, 640)

        self.visual_feature_dim = backbone_conf['output_channels']
        self.feature_dim = self.visual_feature_dim
        self.output_dim = model_params.output_dim  
        self.descriptor_from_spec = model_params.descriptor_from_spec
        self.global_descriptor_dim = model_params.global_descriptor_dim
        self.use_submap = model_params.use_submap
        self.use_pretrained_model = model_params.use_pretrained_model
        
        self.theta = model_params.theta
        self.radius = model_params.radius
        self.coordinates = model_params.coordinates
        self.use_normalize = model_params.normalize
        self.aggregation = model_params.aggregation
        self.confidence = model_params.confidence
        
        if not self.use_pretrained_model:
            backbone_conf['img_backbone_conf'] = copy.deepcopy(backbone_conf['img_backbone_conf'])
            backbone_conf['img_backbone_conf'].pop('init_cfg', None)

        self.encoder = BaseLSSFPN(**backbone_conf)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer('visual_reliability_sobel_x', sobel_x, persistent=False)
        self.register_buffer('visual_reliability_sobel_y', sobel_y, persistent=False)
        self.visual_reliability_tau = 0.01
        self.visual_reliability_eps = 1e-6
        self.visual_reliability_window = 5
        
        if self.confidence:
            self.confidence_layer = AdaptationBlock(self.visual_feature_dim, 1)

        self.image_meta_path = model_params.image_meta_path
        self.occ_conv_yaw = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.occ_conv_trans = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.pooling = NetVLADLoupe(feature_size=self.feature_dim, cluster_size=64,
        #                             output_dim=self.output_dim, gating=True, add_batch_norm=True)
        if self.descriptor_from_spec:
            self.spec_descriptor_head = SpecGlobalDescriptorHead(output_dim=self.global_descriptor_dim)


    def _resolve_image_meta(self, batch, num_views):
        image_meta = batch.get('image_meta')
        if image_meta is None:
            if self.image_meta_path is None:
                raise ValueError('image_meta_path is required when batch does not provide image_meta')
            with open(_ex(self.image_meta_path), 'rb') as handle:
                image_meta = pickle.load(handle)

        k_list = [np.asarray(k, dtype=np.float32) for k in image_meta['K'][:num_views]]
        t_list = [np.asarray(t, dtype=np.float32) for t in image_meta['T'][:num_views]]
        if len(k_list) != num_views or len(t_list) != num_views:
            raise ValueError(f'Expected image meta for {num_views} views, got K={len(k_list)} and T={len(t_list)}')

        return {'K': k_list, 'T': t_list}

    def _build_mats_dict(self, image_meta, batch_size, num_views, device):
        intrins = torch.from_numpy(np.stack(image_meta['K'], axis=0)).float()
        pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0).to(device)
        cams_T_body = torch.from_numpy(np.stack(image_meta['T'], axis=0)).unsqueeze(0).float().to(device)

        pix_T_cams = pix_T_cams.repeat(batch_size, 1, 1, 1)
        cams_T_body = cams_T_body.repeat(batch_size, 1, 1, 1)
        body_T_cams = geom.safe_inverse(cams_T_body.reshape(batch_size * num_views, 4, 4)).reshape(
            batch_size, num_views, 4, 4
        )

        eye4 = torch.eye(4, dtype=torch.float32, device=device)
        ida_mats = eye4.view(1, 1, 1, 4, 4).repeat(batch_size, 1, num_views, 1, 1)
        bda_mat = eye4.view(1, 4, 4).repeat(batch_size, 1, 1)

        return {
            'sensor2ego_mats': body_T_cams.view(batch_size, 1, num_views, 4, 4),
            'intrin_mats': pix_T_cams.view(batch_size, 1, num_views, 4, 4),
            'ida_mats': ida_mats,
            'bda_mat': bda_mat,
        }

    def _compute_image_reliability(self, image):
        b, n, c, _, _ = image.shape
        image = image.reshape(b * n, c, image.shape[-2], image.shape[-1]).float()
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        sobel_x = self.visual_reliability_sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.visual_reliability_sobel_y.to(device=gray.device, dtype=gray.dtype)
        ix = F.conv2d(gray, sobel_x, padding=1)
        iy = F.conv2d(gray, sobel_y, padding=1)
        k = self.visual_reliability_window
        pad = k // 2
        a = F.avg_pool2d(ix * ix, kernel_size=k, stride=1, padding=pad)
        c_mat = F.avg_pool2d(iy * iy, kernel_size=k, stride=1, padding=pad)
        b_mat = F.avg_pool2d(ix * iy, kernel_size=k, stride=1, padding=pad)
        delta = torch.sqrt((a - c_mat).pow(2) + 4.0 * b_mat.pow(2) + self.visual_reliability_eps)
        lambda_min = 0.5 * (a + c_mat - delta)
        lambda_min = lambda_min.clamp_min(0.0)
        reliability = lambda_min / (lambda_min + self.visual_reliability_tau + self.visual_reliability_eps)
        return reliability.reshape(b, n, 1, reliability.shape[-2], reliability.shape[-1]).clamp(0.0, 1.0)

    def extract_vision_bev(self, batch, return_reliability=False):
        im = batch['img']
        if im.ndim != 5:
            raise ValueError(f'Expected batch[\"img\"] with shape (B, S, C, H, W), got {tuple(im.shape)}')

        batch_size, num_views, channels, height, width = im.shape
        if channels != 3:
            raise ValueError(f'Expected RGB images with 3 channels, got {channels}')

        image_meta = self._resolve_image_meta(batch, num_views)
        mats_dict = self._build_mats_dict(image_meta, batch_size, num_views, im.device)
        x = im.unsqueeze(1).float()

        if return_reliability:
            image_reliability = self._compute_image_reliability(im)
            bev, depth_pred, visual_reliability_bev = self.encoder(
                x,
                mats_dict,
                timestamps=None,
                is_return_depth=True,
                image_reliability=image_reliability,
                is_return_reliability=True,
            )
        else:
            bev, depth_pred = self.encoder(
                x,
                mats_dict,
                timestamps=None,
                is_return_depth=True,
            )

        if self.confidence:
            bev_confidence = self.confidence_layer(bev).sigmoid()
            bev = bev * bev_confidence
        else:
            bev_confidence = None

        output = {'bev': bev, 'depth': depth_pred, 'confidence': bev_confidence}
        if return_reliability:
            if visual_reliability_bev.shape[-2:] != bev.shape[-2:]:
                raise ValueError(
                    'Projected visual reliability BEV shape does not match visual BEV: '
                    f'{tuple(visual_reliability_bev.shape)} vs {tuple(bev.shape)}'
                )
            output['visual_reliability_bev'] = visual_reliability_bev.to(device=bev.device, dtype=bev.dtype)
        return output

    def extract_bev_features(self, batch):
        return self.extract_vision_bev(batch)

    def forward_bev_downstream(self, vision_bev, depth_pred=None, bev_confidence=None, extra_outputs=None):
        return run_ring_sharp_downstream(
            self,
            vision_bev,
            depth_pred=depth_pred,
            bev_confidence=bev_confidence,
            yaw_module=self.occ_conv_yaw,
            trans_module=self.occ_conv_trans,
            extra_outputs=extra_outputs,
        )

    def forward_downstream_from_bev(self, bev, depth_pred=None, bev_confidence=None, extra_outputs=None):
        return self.forward_bev_downstream(bev, depth_pred, bev_confidence, extra_outputs)

    def forward(self, batch):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        '''
        bev_outputs = self.extract_vision_bev(batch)
        return self.forward_bev_downstream(
            bev_outputs['bev'],
            depth_pred=bev_outputs['depth'],
            bev_confidence=bev_outputs['confidence'],
        )
    

    def print_info(self):
        print('Model class: RING#-V')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()])
        print('Encoder parameters: {}'.format(n_params))
