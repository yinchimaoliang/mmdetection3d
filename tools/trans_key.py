import torch
from collections import OrderedDict


class TransKey():

    def __init__(self,
                 ori_ckpt_path='work_dirs/pp/epoch_19.pth',
                 target_ckpt_path='work_dirs/pp/epoch_1.pth',
                 save_ckpt_path='work_dirs/pp/generated.pth'):
        ori_ckpt = torch.load(ori_ckpt_path)
        target_ckpt = torch.load(target_ckpt_path)
        self.ori_model_state = ori_ckpt['state_dict']
        self.target_model_state = target_ckpt['state_dict']
        self.save_ckpt_path = save_ckpt_path

    def trans_key(self):
        pts_voxel_encoder = list()
        pts_backbone = list()
        pts_neck = list()
        pts_bbox_head = list()
        new_state_dict = OrderedDict()
        for key in list(self.target_model_state):
            if key.startswith('pts_voxel_encoder'):
                pts_voxel_encoder.append(key)
            if key.startswith('pts_backbone'):
                pts_backbone.append(key)
            if key.startswith('pts_neck'):
                pts_neck.append(key)
            if key.startswith('pts_bbox_head'):
                pts_bbox_head.append(key)

        for key, value in self.ori_model_state.items():
            if key.startswith('reader.pfn_layers'):
                new_key = key.replace('reader.pfn_layers',
                                      'pts_voxel_encoder.pfn_layers')
            elif key.startswith('neck.blocks'):
                key_splits = key.split('.')
                key_splits[0] = 'pts_backbone'
                key_splits[3] = f'{int(key_splits[3]) - 1}'
                new_key = '.'.join(key_splits)
            elif key.startswith('neck.deblocks'):
                new_key = key.replace('neck', 'pts_neck')
            elif key.startswith('bbox_head'):
                new_key = key.replace('bbox_head', 'pts_bbox_head')
            if self.target_model_state[new_key].shape != value.shape:
                print(new_key)
                raise KeyError
            new_state_dict[new_key] = value
        new_checkpoint = dict(state_dict=new_state_dict, meta=[])
        torch.save(new_checkpoint, self.save_ckpt_path)

    def main_func(self):
        self.trans_key()


if __name__ == '__main__':
    trans_key = TransKey()
    trans_key.main_func()
