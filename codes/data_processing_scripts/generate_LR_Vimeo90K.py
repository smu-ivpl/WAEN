""" Generate LR for Vimeo90K """

import os
import sys
import cv2
import numpy as np
import os.path as osp
import glob

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datasets.util import imresize_np
except ImportError:
    pass


def generate_LR_Vimeo90K():
    ### set parameters
    up_scale = 4
    mod_scale = 4

    ### set data dir
    sourcedir = '../../datasets/Vimeo90K/vimeo_septuplet/sequences'
    saveLRpath = '../../datasets/Vimeo90K/modvimeo_septuplet_matlabLRx4/sequences'
    txt_file = '../../datasets/Vimeo90K/vimeo_septuplet/sep_trainlist.txt'

    ### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        all_img_list.extend(glob.glob(osp.join(sourcedir, folder, sub_folder, '*')))
    all_img_list = sorted(all_img_list)
    num_files = len(all_img_list)

    ### prepare data with augementation
    for i in range(num_files):
        filename = all_img_list[i]
        print('No.{} -- Processing {}'.format(i, filename))

        ### read image
        image = cv2.imread(filename)

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))

        ### modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]

        ### LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)

        folder = filename.split('/')[7]
        sub_folder = filename.split('/')[8]
        name = filename.split('/')[9]

        if not os.path.isdir(osp.join(saveLRpath, folder)):
            os.mkdir(osp.join(saveLRpath, folder))

        if not os.path.isdir(osp.join(saveLRpath, folder, sub_folder)):
            os.mkdir(osp.join(saveLRpath, folder, sub_folder))

        cv2.imwrite(osp.join(saveLRpath, folder, sub_folder, name), image_LR)

    print('Finish LR generation')


if __name__ == "__main__":
    generate_LR_Vimeo90K()
