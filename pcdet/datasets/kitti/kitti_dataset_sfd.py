from collections import defaultdict

import copy
import numpy as np
import os
import pickle
import torch
from PIL import Image
from skimage import io

from pcdet.datasets.dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti, transform_utils
from pcdet.utils.transforms import Compose, BottomCrop

class KittiDatasetSFD(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.training = training

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        x_trans_cfg = self.dataset_cfg.get('X_TRANS', None)
        if x_trans_cfg is not None:
            self.x_trans = X_TRANS(x_trans_cfg, rot_num=self.rot_num)
        else:
            self.x_trans = None

        self.input_discard_rate = self.dataset_cfg.get('INPUT_DISCARD_RATE', 0.8)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_lidar_pseudo(self, idx):
        # lidar_pseudo_file = self.root_split_path / 'depth_pseudo_rgbseguv_twise' / ('%s.bin' % idx)
        lidar_pseudo_file = self.root_split_path / 'depth_pseudo_rgbseguv_dense_kbnet' / ('%s.bin' % idx)
        
        # lidar_pseudo_file = self.root_split_path / self.dataset_cfg.DATA_PSEUDOLiDAR_PATH / ('%s.bin' % idx)

        assert lidar_pseudo_file.exists()
        num_ponit_features_pseudo = self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES_PSEUDO
        point_pseudo = np.fromfile(str(lidar_pseudo_file), dtype=np.float32).reshape(-1, num_ponit_features_pseudo)    
        
        range_mask = (point_pseudo[:,2] < 1.7) & (point_pseudo[:,2] > -1.7)
        point_pseudo = point_pseudo[range_mask]

        return  point_pseudo
        
    def get_sparse_depth(self, calib, image, raw_points):
        img_height, img_width, _ = image.shape
        # imgfov_pc_velo, pts_2d, fov_inds, pts_depth = get_lidar_in_image_fov(
        #     raw_points[:, 0:3], calib, 0, 0, img_width, img_height, True)

        pts_2d, pts_depth = calib.lidar_to_img(raw_points)
        fov_inds = (pts_2d[:, 0] < img_width) & (pts_2d[:, 0] >= 0) & \
                   (pts_2d[:, 1] < img_height) & (pts_2d[:, 1] >= 0)
        fov_inds = fov_inds & (raw_points[:, 0] > 2.0)
        imgfov_pc_velo = raw_points[fov_inds, :]


        pts_depth = pts_depth[fov_inds]

        imgfov_pts_2d = pts_2d[fov_inds, :].astype(np.int32) # lidar投影到图像的坐标

        img_depth = np.zeros(image.shape[:-1], np.float32)
        for i in range(imgfov_pts_2d.shape[0]):
            img_depth[tuple(imgfov_pts_2d[i])[::-1]] = pts_depth[i]
        return img_depth

    def get_dense_depth(self, idx, data_format='HW'):
        '''
        Loads a depth map from a 16-bit PNG file

        Arg(s):
            path : str
                path to 16-bit PNG file
            data_format : str
                HW, CHW, HWC
        Returns:
            numpy[float32] : depth map
        '''

        # Loads depth map from 16-bit PNG file
        z = np.array(Image.open(self.root_split_path / 'depth_dense_kbnet' / ('%s.png' % idx)), dtype=np.float32)

        # Assert 16-bit (not 8-bit) depth map
        z = z / 256.0
        z[z <= 0] = 0.0

        transform = Compose([
            BottomCrop((256, 1216)),
        ])

        z = transform(z)

        if data_format == 'HW':
            pass
        elif data_format == 'CHW':
            z = np.expand_dims(z, axis=0)
        elif data_format == 'HWC':
            z = np.expand_dims(z, axis=-1)
        else:
            raise ValueError('Unsupported data format: {}'.format(data_format))

        return z

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()

        image = io.imread(img_file)
        image = transform_utils.img_transform(image)
        image = image.astype(np.float32)
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        database_save_path_pseudo = Path(self.root_path) / ('gt_database_pseudo_seguv' if split == 'train' else ('gt_database_%s_pseudo_seguv' % split))

        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s_sfd_seguv.pkl' % split)

        # database_save_path = Path(self.root_path) / ('gt_database_online_test')
        # database_save_path_pseudo = Path(self.root_path) / ('gt_database_pseudo_seguv_online_test')

        # db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s_sfd_seguv_online_test.pkl')


        database_save_path.mkdir(parents=True, exist_ok=True)
        database_save_path_pseudo.mkdir(parents=True, exist_ok=True)


        all_db_infos = {}

        # with open(info_path, 'rb') as f:
        #     infos = pickle.load(f)

        with open('/moyujian/moyujian/Data/yujmo/KITTI/kitti_sfd_seguv_twise/kitti_infos_trainval.pkl', 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']

            points = self.get_lidar(sample_idx)
            points_pseudo = self.get_lidar_pseudo(sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']
            num_obj = gt_boxes.shape[0]

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            point_indices_pseudo = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points_pseudo[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)

                filepath = database_save_path / filename
                filepath_pseudo = database_save_path_pseudo / filename

                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                gt_points_pseudo = points_pseudo[point_indices_pseudo[i] > 0]
                gt_points_pseudo[:, :3] -= gt_boxes[i, :3]

                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                with open(filepath_pseudo, 'w') as f:
                    gt_points_pseudo.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))                             # gt_database/xxxxx.bin
                    db_path_pseudo = str(filepath_pseudo.relative_to(self.root_path))  # gt_database_pseudo/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'path_pseudo': db_path_pseudo, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0], 'num_points_in_gt_pseudo': gt_points_pseudo.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict
        
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def get_frame_id(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        return sample_idx

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)       
        
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        image = self.get_image(sample_idx)
        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)
        
        dense_depth = self.get_dense_depth(sample_idx)
        sparse_depth = self.get_sparse_depth(calib, image, points[:, :3])
        
        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
            'image': image,
            'dense_depth': dense_depth,
            'image_shape': img_shape,
            'sparse_depth': sparse_depth,
        }
        
        input_dict.update({
            'mm': np.ones(shape=(1, 1))
        })
        
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            if (self.dataset_cfg.get('USE_VAN', None) is True) and (self.training is True):
                gt_names = np.array(['Car' if gt_names[i]=='Van' else gt_names[i] for i in range(len(gt_names))])

            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane


        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape
        return data_dict

    def process_data_sqd_sample(self, image_shape_ori, cu, cv, fu, fv, tx, ty, P2, R0, V2C, image, raw_points,
                                dense_depth):
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_height, img_width, _ = image.shape

        P2 = torch.from_numpy(P2).to(device, non_blocking=True)
        R0 = torch.from_numpy(R0).to(device, non_blocking=True)
        V2C = torch.from_numpy(V2C).to(device, non_blocking=True)
        image = torch.from_numpy(image).to(device, non_blocking=True)
        dense_depth = torch.from_numpy(dense_depth).to(device, non_blocking=True)
        raw_points = torch.from_numpy(raw_points).to(device, non_blocking=True)

        #-------------稀疏LiDAR点云投影到2D 前视图，得到稀疏深度图-------------------------------------------------------------
        # 添加齐次坐标
        pts_lidar_hom = torch.cat((raw_points[:, :3],
                                   torch.ones(raw_points.shape[0], 1, dtype=torch.float32,device=P2.device)), dim=1)
        # pts_lidar_hom = np.hstack((raw_points, np.ones((raw_points.shape[0], 1), dtype=np.float32)))

        # 计算矩形坐标
        pts_rect = torch.mm(pts_lidar_hom, torch.mm(V2C.t(), R0.t()))
        # pts_rect = np.dot(pts_lidar_hom, np.dot(V2C.T, R0.T))

        # 再次添加齐次坐标
        pts_rect_hom = torch.cat((pts_rect, torch.ones(pts_rect.shape[0], 1, dtype=torch.float32, device=P2.device)), dim=1)
        # pts_rect_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)))

        # 计算2D齐次坐标
        pts_2d_hom = torch.mm(pts_rect_hom, P2.t())

        # 转换为2D坐标
        pts_2d = (pts_2d_hom[:, 0:2].t() / pts_2d_hom[:, 2]).t()

        # 获取2D稀疏深度图像中的深度值
        pts_depth = pts_2d_hom[:, 2] - P2.t()[3, 2]

        # 根据LiDAR点云投影到2D稀疏深度图像的坐标，以及raw_points的深度信息，
        fov_inds = (pts_2d[:, 0] < img_width) & (pts_2d[:, 0] >= 0) & \
                   (pts_2d[:, 1] < img_height) & (pts_2d[:, 1] >= 0)
        fov_inds = fov_inds & (raw_points[:, 0] > 0.0)

        # raw_points = raw_points[fov_inds, :]

        pts_depth = pts_depth[fov_inds]
        # 数据由float转int32

        imgfov_pts_2d = pts_2d[fov_inds].to(torch.int32)
        # imgfov_pts_2d 坐标[w, h] [1216, 256]->[1215, 255]
        # imgfov_pts_2d = pts_2d[fov_inds, :].astype(np.int32) # lidar投影到图像的坐标

        sparse_depth_incides_d = torch.floor(pts_depth / 5)
        sparse_depth_incides_w = torch.floor(imgfov_pts_2d[:,0] / 76)

        # 组合索引并获取唯一值
        sparse_depth_coords = torch.stack((sparse_depth_incides_w, sparse_depth_incides_d), dim=1)
        sparse_depth_unique_coords, sparse_indices, sparse_counts = torch.unique(sparse_depth_coords, dim=0,
                                                                                 return_inverse=True, return_counts=True, sorted=True)

        sparse_depth_occ_mask = sparse_depth_unique_coords[:, 1] * (1216 / 76) + sparse_depth_unique_coords[:, 0]


        _h, _w = dense_depth.shape
        _d = dense_depth.flatten().unsqueeze(1)

        _tv_h_coords, _tv_w_coords = torch.meshgrid(torch.arange(_h, device=P2.device),
                                                  torch.arange(_w, device=P2.device))
        _tv_h_coords = _tv_h_coords.flatten().unsqueeze(1)
        _tv_w_coords = _tv_w_coords.flatten().unsqueeze(1)

        dense_depth_tv = torch.cat((
            _d,
            _tv_w_coords,
            _tv_h_coords,
            image.view(-1, 3)
        ), dim=1)

        dense_depth_tv_coords = torch.cat((
            torch.floor(_tv_w_coords / 76),
            torch.floor(_d / 5)
        ), dim=1)

        dense_depth_occ_mask = torch.floor(_d / 5) * (1216 / 76) + torch.floor(_tv_w_coords / 76)

        dense_in_sparse_mask = torch.isin(dense_depth_occ_mask, sparse_depth_occ_mask)

        ds1_dense_depth_tv = dense_depth_tv[dense_in_sparse_mask.squeeze()]
        ds1_dense_depth_tv_coords = dense_depth_tv_coords[dense_in_sparse_mask.squeeze()]

        dense_depth_unique_coords, dense_inverse_indices, dense_counts = torch.unique(ds1_dense_depth_tv_coords[:, :2], dim=0, return_inverse=True,
                                                   return_counts=True, sorted=True)
        sparse_in_dense_mask = torch.isin(sparse_depth_occ_mask, dense_depth_occ_mask)

        if self.training:
            assert (sparse_depth_unique_coords[sparse_in_dense_mask] == dense_depth_unique_coords).all()
            assert (dense_depth_unique_coords[dense_inverse_indices] == ds1_dense_depth_tv_coords[:, :2]).all()

        sparse_weight = sparse_counts[sparse_in_dense_mask].to(dtype=torch.float)
        dense_weight = dense_counts.to(dtype=torch.float)

        sparse_weight_weights = sparse_weight[dense_inverse_indices]

        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
        # 生成比较多的伪点云，使用一阶段粗糙的RoIs来crop伪点云，refine粗糙的框
        pesudo_points_weights = dense_counts[dense_inverse_indices]
        weights = torch.rand(ds1_dense_depth_tv_coords.shape[0])
        max_weight = weights.max() + 1

        weights[sparse_weight_weights < 20] = max_weight

        ds2_dense_depth_mask = weights > 0.4

        pseudo_coords = ds1_dense_depth_tv[ds2_dense_depth_mask]

        n_points = pseudo_coords.shape[0]
        depth_rect = pseudo_coords[:, 0]
        u = pseudo_coords[:, 1] + int(round((image_shape_ori[1] - img_width) / 2.))
        v = pseudo_coords[:, 2] + image_shape_ori[0] - img_height

        x = ((u - cu) * depth_rect) / fu + tx
        y = ((v - cv) * depth_rect) / fv + ty

        pts_rect = torch.stack((x, y, depth_rect), dim=1)
        pts_rect_hom = torch.cat((pts_rect, torch.ones(n_points, 1, device=P2.device, dtype=torch.float32)), dim=1)

        R0_ext = torch.cat((R0, torch.zeros(3, 1, device=P2.device, dtype=torch.float32)), dim=1)
        R0_ext = torch.cat((R0_ext, torch.zeros(1, 4, device=P2.device, dtype=torch.float32)), dim=0)
        R0_ext[3, 3] = 1

        V2C_ext = torch.cat((V2C, torch.zeros(1, 4, device=P2.device, dtype=torch.float32)), dim=0)
        V2C_ext[3, 3] = 1

        pseudo_points = torch.mm(pts_rect_hom, torch.inverse(torch.mm(R0_ext, V2C_ext).t()))[:, :3]

        pc_velo_intensity_rgb_uv = torch.cat(
            [
                pseudo_points,
                pseudo_coords[:, 3:6],
                torch.full((n_points, 1), 0, device=P2.device, dtype=torch.float32),  # mask
                pseudo_coords[:, 1:3]
            ],
            dim=1
        )
        range_mask = (pc_velo_intensity_rgb_uv[:,2] < 1.7) & (pc_velo_intensity_rgb_uv[:,2] > -1.7)
        pc_velo_intensity_rgb_uv = pc_velo_intensity_rgb_uv[range_mask]
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————#



        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
        # 按照占用栅格的方式，来采样少部分的伪点云，来补充LiDAR点云，输入到Backbone 3D中，产生每个Voxel的特征
        weights_for_voxel = torch.rand(ds1_dense_depth_tv_coords.shape[0])

        weights_for_voxel[sparse_weight_weights < 10] = max_weight
        weights_for_voxel[sparse_weight_weights < 3] = 0
        pseudo_points_for_voxel_mask = weights_for_voxel > 0.9

        pseudo_coords_for_voxel = ds1_dense_depth_tv[pseudo_points_for_voxel_mask]

        n_points = pseudo_coords_for_voxel.shape[0]
        depth_rect = pseudo_coords_for_voxel[:, 0]
        u = pseudo_coords_for_voxel[:, 1] + int(round((image_shape_ori[1] - img_width) / 2.))
        v = pseudo_coords_for_voxel[:, 2] + image_shape_ori[0] - img_height

        x = ((u - cu) * depth_rect) / fu + tx
        y = ((v - cv) * depth_rect) / fv + ty

        pts_rect = torch.stack((x, y, depth_rect), dim=1)
        pts_rect_hom = torch.cat((pts_rect, torch.ones(n_points, 1, device=P2.device, dtype=torch.float32)), dim=1)

        R0_ext = torch.cat((R0, torch.zeros(3, 1, device=P2.device, dtype=torch.float32)), dim=1)
        R0_ext = torch.cat((R0_ext, torch.zeros(1, 4, device=P2.device, dtype=torch.float32)), dim=0)
        R0_ext[3, 3] = 1

        V2C_ext = torch.cat((V2C, torch.zeros(1, 4, device=P2.device, dtype=torch.float32)), dim=0)
        V2C_ext[3, 3] = 1

        pseudo_points_for_voxel = torch.mm(pts_rect_hom, torch.inverse(torch.mm(R0_ext, V2C_ext).t()))[:, :3]

        # 创建最终张量
        pseudo_points_for_voxel = torch.cat(
            [
                pseudo_points_for_voxel,
                torch.full((n_points, 1), 0, device=P2.device, dtype=torch.float32),  # 常数强度值
            ],
            dim=1
        )

        range_mask = (pseudo_points_for_voxel[:,2] < 1.7) & (pseudo_points_for_voxel[:,2] > -1.7)
        pseudo_points_for_voxel = pseudo_points_for_voxel[range_mask]

        pseudo_points_for_voxel = pseudo_points_for_voxel.cpu().numpy()
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

        return pc_velo_intensity_rgb_uv, pseudo_points_for_voxel

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        
        calib = data_dict['calib']
        P2 = calib.P2
        R0 = calib.R0
        V2C = calib.V2C
        cu = calib.cu
        cv = calib.cv
        fu = calib.fu
        fv = calib.fv
        tx = calib.tx
        ty = calib.ty

        image = data_dict['image']
        raw_points = data_dict['points']
        dense_depth = data_dict['dense_depth']
        image_shape_ori = data_dict['image_shape']

        pc_velo_intensity_rgb_uv, pseudo_points_for_voxel = self.process_data_sqd_sample(image_shape_ori, cu, cv, fu, fv, tx, ty, P2, R0, V2C, image, raw_points, dense_depth)

        pseudo_points_intensity = np.zeros((pseudo_points_for_voxel.shape[0], 1)) + 0.5
        pseudo_points_flag = np.ones((pseudo_points_for_voxel.shape[0], 1))
        pseudo_points_rgb = np.zeros((pseudo_points_for_voxel.shape[0], 3))
        new_pseudo_points = np.hstack((pseudo_points_for_voxel[:, 0:3],
                                       pseudo_points_intensity,
                                       pseudo_points_rgb,
                                       pseudo_points_flag
                                       ))
        raw_points[:,3] *= 10
        raw_points_rgb = np.zeros((raw_points.shape[0], 3))
        raw_points_flag = np.ones((raw_points.shape[0], 1)) * 2

        new_lidar_points = np.hstack((raw_points,
                                      raw_points_rgb,
                                      raw_points_flag
                                      ))

        final_points = np.concatenate((new_pseudo_points,
                                       new_lidar_points))
        data_dict['points'] = final_points

        
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
        else:
            if self.x_trans is not None:
                data_dict = self.x_trans.input_transform(
                    data_dict={
                        **data_dict,
                    }
                )    

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes


        data_dict = self.point_feature_encoder.forward(data_dict)
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        try:
            data_dict.pop('valid_noise')
        except KeyError:
            pass
        
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['name_img_dt','bbox_img_dt','score_img_dt']:
                    max_gt = max([len(x) for x in val])
                    batch_img_info = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_img_info[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_img_info
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDatasetSFD(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)

    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2]), Loader=yaml.FullLoader))

        ROOT_DIR = Path('data/KITTI/')
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR
        )
