import os.path as osp
import torch
import torch.utils.data as data
import datasets.util as util


class Vid4_Dataset(data.Dataset):
    """
    Vid4 test dataset
    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(Vid4_Dataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LR_root = opt['dataroot_GT'], opt['dataroot_LR']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LR': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LR, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4']:
            subfolders_LR = util.glob_file_list(self.LR_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            for subfolder_LR, subfolder_GT in zip(subfolders_LR, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LR = util.glob_file_list(subfolder_LR)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LR)
                assert max_idx == len(
                    img_paths_GT), 'Different number of images in LR and GT folders'
                self.data_info['path_LR'].extend(img_paths_LR)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_LR[subfolder_name] = util.read_img_seq(img_paths_LR)
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
        else:
            raise ValueError(
                'Not support video test dataset.')

    def __getitem__(self, index):
        # path_LR = self.data_info['path_LR'][index]
        # path_GT = self.data_info['path_GT'][index]
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            imgs_LR = self.imgs_LR[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]
        else:
            pass  # TODO

        return {
            'LRs': imgs_LR,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
