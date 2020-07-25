import numpy as np
import torch

from mmdet3d.datasets.pipelines import MultiScaleFlipAug3D


def test_multi_scale_flip_aug_3D():
    np.random.seed(0)
    transforms = [{
        'type': 'GlobalRotScaleTrans',
        'rot_range': [-0.1, 0.1],
        'scale_ratio_range': [0.9, 1.1],
        'translation_std': [0, 0, 0]
    }, {
        'type': 'RandomFlip3D',
        'sync_2d': False,
        'flip_ratio_bev_horizontal': 0.5
    }, {
        'type': 'IndoorPointSample',
        'num_points': 5
    }, {
        'type':
        'DefaultFormatBundle3D',
        'class_names': ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                        'dresser', 'night_stand', 'bookshelf', 'bathtub'),
        'with_label':
        False
    }, {
        'type': 'Collect3D',
        'keys': ['points']
    }]
    img_scale = (1333, 800)
    pts_scale_ratio = 1
    multi_scale_flip_aug_3D = MultiScaleFlipAug3D(transforms, img_scale,
                                                  pts_scale_ratio)
    pts_file_name = 'tests/data/sunrgbd/points/000001.bin'
    sample_idx = 4
    file_name = 'tests/data/sunrgbd/points/000001.bin'
    bbox3d_fields = []
    points = np.array([[0.20397437, 1.4267826, -1.0503972, 0.16195858],
                       [-2.2095256, 3.3159535, -0.7706928, 0.4416629],
                       [1.5090443, 3.2764456, -1.1913797, 0.02097607],
                       [-1.373904, 3.8711405, 0.8524302, 2.064786],
                       [-1.8139812, 3.538856, -1.0056694, 0.20668638]])
    results = dict(
        points=points,
        pts_file_name=pts_file_name,
        sample_idx=sample_idx,
        file_name=file_name,
        bbox3d_fields=bbox3d_fields)
    results = multi_scale_flip_aug_3D(results)
    expected_points = torch.tensor(
        [[-2.2095, 3.3160, -0.7707, 0.4417], [-1.3739, 3.8711, 0.8524, 2.0648],
         [-1.8140, 3.5389, -1.0057, 0.2067], [0.2040, 1.4268, -1.0504, 0.1620],
         [1.5090, 3.2764, -1.1914, 0.0210]],
        dtype=torch.float64)
    assert torch.allclose(
        results['points'][0]._data, expected_points, atol=1e-4)
