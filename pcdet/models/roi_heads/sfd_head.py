import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
import spconv.pytorch as spconv
import numpy as np
import torch.nn.functional as F
from ...utils import box_coder_utils, common_utils, loss_utils, box_utils
import time
import cv2
from ...ops.iou3d import oriented_iou_loss
from ...ops.iou3d_nms import iou3d_nms_utils

class SFDHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # RoI Grid Pool
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps, 
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

        block = self.post_act_block
        c0 = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES_RAW
        c1 = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES
        
        if 1==1:
            shared_fc_list_valid = []
            pre_channel_valid = GRID_SIZE * GRID_SIZE * GRID_SIZE * c1
            for k in range(0, self.model_cfg.SHARED_FC_VALID.__len__()):
                shared_fc_list_valid.extend([
                    nn.Linear(pre_channel_valid, self.model_cfg.SHARED_FC_VALID[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC_VALID[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel_valid = self.model_cfg.SHARED_FC_VALID[k]

                if k != self.model_cfg.SHARED_FC_VALID.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list_valid.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layer_valid = nn.Sequential(*shared_fc_list_valid)   

            cls_fc_list_valid = []
            cls_pre_channel_valid = pre_channel_valid
            for k in range(0, self.model_cfg.CLS_FC_VALID.__len__()):
                cls_fc_list_valid.extend([
                    nn.Linear(cls_pre_channel_valid, self.model_cfg.CLS_FC_VALID[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC_VALID[k]),
                    nn.ReLU()
                ])
                cls_pre_channel_valid = self.model_cfg.CLS_FC_VALID[k]

                if k != self.model_cfg.CLS_FC_VALID.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list_valid.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.cls_fc_layers_valid = nn.Sequential(*cls_fc_list_valid)
            self.cls_pred_layer_valid = nn.Linear(cls_pre_channel_valid, self.num_class, bias=True)

            reg_fc_list_valid = []
            reg_pre_channel_valid = pre_channel_valid
            for k in range(0, self.model_cfg.REG_FC_VALID.__len__()):
                reg_fc_list_valid.extend([
                    nn.Linear(reg_pre_channel_valid, self.model_cfg.REG_FC_VALID[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC_VALID[k]),
                    nn.ReLU()
                ])
                reg_pre_channel_valid = self.model_cfg.REG_FC_VALID[k]

                if k != self.model_cfg.REG_FC_VALID.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list_valid.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.reg_fc_layers_valid = nn.Sequential(*reg_fc_list_valid)
            self.reg_pred_layer_valid = nn.Linear(reg_pre_channel_valid, self.model_cfg.AUXILIARY_CODE_SIZE * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_

        # valid head
        for module_list in [self.shared_fc_layer_valid, self.cls_fc_layers_valid, self.reg_fc_layers_valid]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer_valid.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer_valid.bias, 0)
        nn.init.normal_(self.reg_pred_layer_valid.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer_valid.bias, 0)

    def roi_grid_pool(self, batch_dict):
  
        # Args:
        #     batch_dict:
        #         batch_size:
        #         rois: (B, num_rois, 7 + C)
        #         point_coords: (num_points, 4)  [bs_idx, x, y, z]
        #         point_features: (num_points, C)
        #         point_cls_scores: (N1 + N2 + N3 + ..., 1)
        #         point_part_offset: (N1 + N2 + N3 + ..., 3)
        # Returns:
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        # (BxN, 6x6x6, 3)
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  
        # (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # (B, Nx6x6x6, 3) compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        # (B, Nx6x6x6, 1) compute the batch index of grid points
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
            
        # (B) [Nx6x6x6, Nx6x6x6, ..., Nx6x6x6]
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        #  grouper --> query range --> 
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # [18196, 4] / [8372, 4]
            cur_coords = cur_sp_tensors.indices 

            # [18196, 3] / [8372, 3]
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            ) 

            # 18196 / 8372
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            # [1, 11, 400, 352] / [1, 5, 200, 176]
            v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

            # [1, 27648, 4]
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()

            # [27648, 64]
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            # [BxN, 6x6x6, 64]
            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  
            pooled_features_list.append(pooled_features)
        
        # [BxN, 6x6x6, 128]
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    @staticmethod
    def fake_sparse_idx(sparse_idx, batch_size_rcnn):
        print('Warning: Sparse_Idx_Shape(%s) \r' % (str(sparse_idx.shape)), end='', flush=True)
        # at most one sample is non-empty, then fake the first voxels of each sample(BN needs at least
        # two values each channel) as non-empty for the below calculation
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    # SQD的代码
    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        # RoI Grid Pool
        x_valid = self.roi_grid_pool(batch_dict)
        x_valid  = x_valid.transpose(1,2)

        # RoI Point Pool
        B, N, _ = batch_dict['rois'].shape

        x_valid = x_valid.reshape(x_valid.size(0), -1)
        shared_features_valid = self.shared_fc_layer_valid(x_valid)
        rcnn_cls_valid = self.cls_pred_layer_valid(self.cls_fc_layers_valid(shared_features_valid))
        rcnn_reg_valid = self.reg_pred_layer_valid(self.reg_fc_layers_valid(shared_features_valid))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls_valid, box_preds=rcnn_reg_valid
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False

        else:

            targets_dict['rcnn_cls_valid'] = rcnn_cls_valid
            targets_dict['rcnn_reg_valid'] = rcnn_reg_valid
            self.forward_ret_dict = targets_dict

        return batch_dict

    
    def get_loss(self, tb_dict=None):
        # voxel_rcnn loss
        tb_dict = {} if tb_dict is None else tb_dict
    
        # valid---------------------------------------------------------------------------------------------------------
        rcnn_loss = 0
        rcnn_loss_reg, reg_tb_dict, iou_target = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
    
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict, iou_target)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)
    
        tb_dict['rcnn_loss'] = rcnn_loss.item()
    
        return rcnn_loss, tb_dict
    
    
    def get_box_cls_layer_loss(self, forward_ret_dict, iou_target):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls_valid']
        # rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_valid  = forward_ret_dict['rcnn_cls_labels'].view(-1)
        iou_target = iou_target.view(-1)
    
        rcnn_cls_labels = iou_target
    
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_valid >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_valid >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError
    
        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict
    
    
    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg_valid']
        # rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]
    
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
    
        tb_dict = {}
    
        if True:
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
    
            dt_boxes = self.box_coder.decode_torch(
                rcnn_reg, rois_anchor
            )
            gt_boxes = gt_boxes3d_ct.view(rcnn_batch_size, code_size)
    
            if loss_cfgs.REG_LOSS == 'iou':
                iou3d = oriented_iou_loss.cal_iou_3d(dt_boxes.unsqueeze(0),  gt_boxes.unsqueeze(0))
                iou_loss = 1. - iou3d
            elif loss_cfgs.REG_LOSS == 'giou':
                iou_loss, iou3d = oriented_iou_loss.cal_giou_3d(dt_boxes.unsqueeze(0), gt_boxes.unsqueeze(0))
            elif loss_cfgs.REG_LOSS == 'diou':
                iou_loss, iou3d = oriented_iou_loss.cal_diou_3d(dt_boxes.unsqueeze(0), gt_boxes.unsqueeze(0))
            iou_loss = iou_loss.squeeze(0)
            rcnn_loss_reg = iou_loss
    
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(dt_boxes, gt_boxes)
            iou3d_index = range(0,len(iou3d))
            iou_target = iou3d[iou3d_index, iou3d_index].clone().detach()
    
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
    
            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
    
                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)
    
                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz
    
                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']
    
                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError
    
        return rcnn_loss_reg, tb_dict, iou_target