import os
import sys
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from mmcv.ops import Voxelization
import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic
from glnet.utils.params import ModelParams
from glnet.utils.data_utils.point_clouds import generate_bev, generate_bev_feats
from glnet.config.config import *
from glnet.models.backbones_2d.unet import last_conv_block, UNet, Autoencoder, AdaptationBlock

EPS = 1e-4

from glnet.models.aggregation.GeM import GeM
from glnet.models.aggregation.NetVLADLoupe import NetVLADLoupe
from glnet.models.backbones_2d.steerable_cnn import SteerableCNN
from glnet.models.localizer.ring_sharp_vl import run_ring_sharp_downstream
from glnet.models.localizer.spec_head import SpecGlobalDescriptorHead


class RINGSharpL(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(RINGSharpL, self).__init__()
        
        self.yaw_mode = 1
        self.trans_mode = 0

        self.params = model_params
        self.dataset = model_params.dataset_type
        self.use_bev = model_params.use_bev
        self.point_encoder = model_params.point_encoder
        self.bev_encoder = model_params.bev_encoder
        self.feature_dim = model_params.feature_dim
        self.output_dim = model_params.output_dim
        self.descriptor_from_spec = model_params.descriptor_from_spec
        self.global_descriptor_dim = model_params.global_descriptor_dim
        self.use_submap = model_params.use_submap

        self.theta = model_params.theta
        self.radius = model_params.radius
        self.coordinates = model_params.coordinates
        self.use_normalize = model_params.normalize
        self.aggregation = model_params.aggregation
        self.confidence = model_params.confidence
        
        if self.dataset == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                       pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        self.X = pc_bev_conf['x_grid']
        self.Y = pc_bev_conf['y_grid']
        self.Z = pc_bev_conf['z_grid']
        
        self.encoder = SteerableCNN(self.Z, self.feature_dim, bn=False)

        if self.confidence:
            self.confidence_layer = AdaptationBlock(self.feature_dim, 1)
        
        if self.use_bev and self.encoder == nn.Identity():
            self.feature_dim = self.Z
        if self.bev_encoder == 'unet':
            self.encoder_yaw = UNet(self.feature_dim, bn=False, is_circular=False)
        elif self.bev_encoder == 'autoencoder':
            self.encoder_yaw = Autoencoder(self.feature_dim, bn=False, is_circular=False)
        else:
            mid_channels = model_params.feature_dim
            self.encoder_yaw = last_conv_block(self.feature_dim, mid_channels, bn=False)

        if self.descriptor_from_spec:
            self.spec_descriptor_head = SpecGlobalDescriptorHead(output_dim=self.global_descriptor_dim)
            
    def extract_lidar_bev(self, batch):
        pc = batch['pc']
        if pc.ndim != 4:
            raise ValueError(f'Expected batch["pc"] with shape (B, C, H, W), got {tuple(pc.shape)}')

        bev = self.encoder(pc)

        if self.confidence:
            bev_confidence = self.confidence_layer(bev).sigmoid()
            bev = bev * bev_confidence
        else:
            bev_confidence = None

        return {'bev': bev, 'confidence': bev_confidence}

    def extract_bev_features(self, batch):
        return self.extract_lidar_bev(batch)

    def forward_bev_downstream(self, lidar_bev, bev_confidence=None, extra_outputs=None):
        return run_ring_sharp_downstream(
            self,
            lidar_bev,
            bev_confidence=bev_confidence,
            yaw_module=self.encoder_yaw,
            trans_module=None,
            extra_outputs=extra_outputs,
        )
        
    def forward(self, batch):
        bev_outputs = self.extract_lidar_bev(batch)
        bev = bev_outputs['bev']
        bev_confidence = bev_outputs['confidence']
        return self.forward_bev_downstream(bev, bev_confidence=bev_confidence)
    
    
    def print_info(self):
        print('Model class: RING#-L')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
