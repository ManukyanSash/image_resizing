import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.transform import resize

class DataLoader():
    def __init__(self, dataset_name, img_res=(224, 224)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        
        path = glob('./%s/*' % (self.dataset_name))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:

            img_path2 = os.path.join('original_good_images', img_path.split('/')[-1])

            img = self.imread(img_path)
            #print(img.shape)
            img_good = self.imread(img_path2)

            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = resize(img, self.img_res)
            img_hr = resize(img_good, self.img_res)
            img_lr = resize(img, (low_h, low_w))
    
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def imread(self, path):
        return np.array(Image.open(path).convert('RGB')).astype(np.float32)
