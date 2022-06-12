import os
import os.path as osp
import glob
import sys
import cv2
import numpy as np

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datasets.util import imresize_np
except ImportError:
    pass


def generate_LR_bic():
    # set parameters
    mod_scale = 4
    
    # set data dir
    sourcedir = '../../datasets/Vid4/Source'
    savedir = '../../datasets/Vid4'

    saveHRpath = os.path.join(savedir, 'GT')
    saveLRpath = os.path.join(savedir, 'BIx' + str(mod_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'GT')):
        os.mkdir(os.path.join(savedir, 'GT'))
    if not os.path.isdir(os.path.join(savedir, 'BIx' + str(mod_scale))):
        os.mkdir(os.path.join(savedir, 'BIx' + str(mod_scale)))

    subfolder_l = sorted(filter(os.path.isdir, glob.glob(osp.join(sourcedir, '*'))))

    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)

        if not os.path.isdir(os.path.join(savedir, 'GT', subfolder_name)):
            os.mkdir(os.path.join(savedir, 'GT', subfolder_name))
        if not os.path.isdir(os.path.join(savedir, 'BIx' + str(mod_scale), subfolder_name)):
            os.mkdir(os.path.join(savedir, 'BIx' + str(mod_scale), subfolder_name))

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        num_files = len(img_path_l)

        # prepare data with augementation
        for i in range(num_files):
            img_path = img_path_l[i]
            img_name = osp.basename(img_path)
            print('No.{} -- Processing {}'.format(i, img_path))
            # read image
            image = cv2.imread(img_path)

            width = int(np.floor(image.shape[1] / mod_scale))
            height = int(np.floor(image.shape[0] / mod_scale))
            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
            else:
                image_HR = image[0:mod_scale * height, 0:mod_scale * width]
            # LR
            image_LR = imresize_np(image_HR, 1 / mod_scale, True)

            cv2.imwrite(os.path.join(saveHRpath, subfolder_name, img_name), image_HR)
            cv2.imwrite(os.path.join(saveLRpath, subfolder_name, img_name), image_LR)


if __name__ == "__main__":
    generate_LR_bic()
