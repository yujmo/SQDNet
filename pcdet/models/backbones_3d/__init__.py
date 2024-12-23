from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
# from .spconv_backbone_origin import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2
# from .spconv_backbone_origin import VoxelBackBone4x
# from .spconv_backbone_origin import VirConvL8x
from .spconv_backbone import VoxelBackBone8x


__all__ = {
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    # 'VoxelBackBone4x': VoxelBackBone4x,
    'VoxelBackBone8x': VoxelBackBone8x,
    # 'VoxelResBackBone8x': VoxelResBackBone8x,
    # 'VirConvL8x': VirConvL8x,
}