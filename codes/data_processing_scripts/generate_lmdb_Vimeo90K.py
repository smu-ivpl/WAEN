""" Generate lmdb files for Vimeo90K """

import sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import lmdb
import cv2

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import utils.util as util

def main():
    mode = 'GT'  # GT | LR | flow

    if mode == 'GT':
        img_folder = '../../datasets/Vimeo90K/vimeo_septuplet/sequences'
        lmdb_save_path = '../../datasets/Vimeo90K/vimeo90k_train_GT.lmdb'
        txt_file = '../../datasets/Vimeo90K/vimeo_septuplet/sep_trainlist.txt'
    elif mode == 'LR':
        img_folder = '../../datasets/Vimeo90K/modvimeo_septuplet_matlabLRx4/sequences'
        lmdb_save_path = '../../datasets/Vimeo90K/vimeo90k_train_LR7frames.lmdb'
        txt_file = '../../datasets/Vimeo90K/vimeo_septuplet/sep_trainlist.txt'
    elif mode == 'flow':
        img_folder = '../../datasets/Vimeo90K/vimeo_septuplet/sequences_flowx4'
        lmdb_save_path = '../../datasets/Vimeo90K/vimeo90k_train_flowx4.lmdb'
        txt_file = '../../datasets/Vimeo90K/vimeo_septuplet/sep_trainlist.txt'

    generate_lmdb_Vimeo90K(mode, img_folder, lmdb_save_path, txt_file)

def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)

def generate_lmdb_Vimeo90K(mode, img_folder, lmdb_save_path, txt_file):
    """Create lmdb for the Vimeo90K dataset, each image with a fixed size
    GT: [3, 256, 448]
        Now only need the 4th frame, e.g., 00001_0001_4
    LR: [3, 64, 112]
        1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001

    flow: downsampled flow: [3, 360, 320], keys: 00001_0001_4_[p3, p2, p1, n1, n2, n3]
        Each flow is calculated with GT images by PWCNet and then downsampled by 1/4
        Flow map is quantized by mmcv and saved in png format
    """
    ### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 50000  # After BATCH images, lmdb commits, if read_all_imgs = False

    ### set resolution (default scale : x4)
    if mode == 'GT':
        H_dst, W_dst = 256, 448
    elif mode == 'LR':
        H_dst, W_dst = 64, 112
    elif mode == 'flow':
        H_dst, W_dst = 128, 112
    else:
        raise ValueError('Wrong dataset mode: {}'.format(mode))

    n_thread = 40

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    ### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        all_img_list.extend(glob.glob(osp.join(img_folder, folder, sub_folder, '*')))
        if mode == 'flow':
            for j in range(1, 4):
                keys.append('{}_{}_4_n{}'.format(folder, sub_folder, j))
                keys.append('{}_{}_4_p{}'.format(folder, sub_folder, j))
        else:
            for j in range(7):
                keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'GT':  # only read the 4th frame for the GT mode
        print('Only keep the 4th frame.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    if read_all_imgs:
        ### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            """get the image data and update pbar"""
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    ### write data to lmdb
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    txn = env.begin(write=True)
    pbar = util.ProgressBar(len(all_img_list))
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    ### create meta information
    meta_info = {}
    if mode == 'GT':
        meta_info['name'] = 'Vimeo90K_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo90K_train_LR'
    elif mode == 'flow':
        meta_info['name'] = 'Vimeo90K_train_flowx4'
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    key_set = set()
    for key in keys:
        if mode == 'flow':
            a, b, _, _ = key.split('_')
        else:
            a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = list(key_set)
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

if __name__ == "__main__":
    main()
