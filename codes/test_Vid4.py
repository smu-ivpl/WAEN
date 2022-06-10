""" Test for Vid4 """

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import datasets.util as data_util
import models.archs.WAEN_P_arch as WAEN_P_arch
import models.archs.WAEN_S_arch as WAEN_S_arch

def main():
    ### settings
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data_mode = 'Vid4'
    model_mode = 'S' # P (Parallel structure) | S (Serial structure)
    padding = 'new_info'
    save_imgs = False
    calc_onlyY = True
    N_in = 7

    ### pretrained model path
    if model_mode == 'P':
        model_path = '../pretrained_models/WAEN_P.pth'
    elif model_mode == 'S':
        model_path = '../pretrained_models/WAEN_S.pth'

    ### model arch
    if model_mode == 'P':
        model = WAEN_P_arch.WAEN_P()
    elif model_mode == 'S':
        model = WAEN_S_arch.WAEN_S()

    ### dataset
    test_dataset_folder = '../datasets/Vid4/BIx4'
    GT_dataset_folder = '../datasets/Vid4/GT'

    ### save settings
    save_folder = '../experiments/test_results/results_{}/{}'.format(data_mode, model_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    ### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model: {} - {}'.format(model_mode, model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Calculate only Y: {}'.format(calc_onlyY))

    ### load model
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_ssim_l = [], []
    subfolder_name_l = []

    subfolder_l = sorted(filter(os.path.isdir, glob.glob(osp.join(test_dataset_folder, '*'))))
    subfolder_GT_l = sorted(filter(os.path.isdir, glob.glob(osp.join(GT_dataset_folder, '*'))))

    ### for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)

        if save_imgs:
            util.mkdirs(save_subfolder)

        ### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_psnr, avg_ssim, N_data = 0, 0, 0

        ### process each image
        for img_idx, img_path in enumerate(img_path_l):
            if img_idx > 1 and img_idx < max_idx - 2: # Without the first two frames and the last two frames
                img_name = osp.splitext(osp.basename(img_path))[0]
                select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
                imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

                output = util.single_forward(model, imgs_in)
                output = util.tensor2img(output.squeeze(0))

                if save_imgs:
                    cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

                ### calculate PSNR and SSIM
                output = output / 255.
                GT = np.copy(img_GT_l[img_idx])

                if calc_onlyY:  # bgr2y
                    GT = data_util.bgr2ycbcr(GT, only_y=True)
                    output = data_util.bgr2ycbcr(output, only_y=True)

                crt_psnr = util.calculate_psnr(output * 255, GT * 255)
                crt_ssim = util.calculate_ssim(output * 255, GT * 255)
                logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))
                logger.info('{:3d} - {:25} \tSSIM: {:.6f}'.format(img_idx + 1, img_name, crt_ssim))

                avg_psnr += crt_psnr
                avg_ssim += crt_ssim
                N_data += 1

        avg_psnr = avg_psnr / N_data
        avg_psnr_l.append(avg_psnr)

        avg_ssim = avg_ssim / N_data
        avg_ssim_l.append(avg_ssim)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '.format(subfolder_name, avg_psnr, N_data))
        logger.info('Folder {} - Average SSIM: {:.6f} dB for {} frames; '.format(subfolder_name, avg_ssim, N_data))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, ssim in zip(subfolder_name_l, avg_psnr_l, avg_ssim_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '.format(subfolder_name, psnr))
        logger.info('Folder {} - Average SSIM: {:.6f}.'.format(subfolder_name, ssim))

    logger.info('################ Final Results ################')
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '.format(sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l)))
    logger.info('Total Average SSIM: {:.6f} for {} clips.'.format(sum(avg_ssim_l) / len(avg_ssim_l), len(subfolder_l)))

if __name__ == '__main__':
    main()