import os
import glob
import numpy as np
from PIL import Image


def average_imgs(*paths):
    for p, path in enumerate(paths):
        img = np.array(Image.open(path), dtype='float')
        if p == 0:
            img_sum = img
        else:
            img_sum += img
    return img_sum / len(paths)


def preprocess_video(dirs):
    for _dir in dirs:
        if os.path.isfile(_dir):
            continue
        if _dir[-3:] == '_gt':
            continue           
        new_dir = _dir.replace('UCSD_Anomaly_Dataset.v1p2', 'UCSD_processed')
        os.makedirs(new_dir, exist_ok=True)
        dir_files = sorted(glob.glob(_dir + '/*.tif'))
        print('{} >>> {}'.format(_dir, new_dir))
        for t in range(5, len(dir_files)):
            I_t = average_imgs(dir_files[t], dir_files[t-1])
            I_tm2 = average_imgs(dir_files[t-2], dir_files[t-3])
            I_tm4 = average_imgs(dir_files[t-4], dir_files[t-5])
            I = np.stack([I_t, I_tm2, I_tm4], axis=0)
            img = Image.fromarray(I.transpose(1, 2, 0).astype(np.uint8))
            save_path = os.path.join(new_dir, '{:03d}.png'.format(t))
            img.save(save_path)
            
            
if __name__ == '__main__':
    ped1_train_dirs = sorted(glob.glob('data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*'))
    ped2_train_dirs = sorted(glob.glob('data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/*'))
    ped1_test_dirs = sorted(glob.glob('data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/*'))
    ped2_test_dirs = sorted(glob.glob('data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/*'))

    preprocess_video(ped1_train_dirs)
    preprocess_video(ped2_train_dirs)
    preprocess_video(ped1_test_dirs)
    preprocess_video(ped2_test_dirs)
