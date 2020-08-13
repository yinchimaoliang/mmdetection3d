import torch


class TransKey():

    def __init__(self,
                 ori_ckpt_path='work_dirs/pp/epoch_19.pth',
                 new_ckpt_path='work_dirs/pp/epoch_1.pth'):
        self.ori_ckpt = torch.load(ori_ckpt_path)
        self.new_ckpt = torch.load(new_ckpt_path)
        for key in self.ori_ckpt['state_dict'].keys():
            print(key)


if __name__ == '__main__':
    trans_key = TransKey()
