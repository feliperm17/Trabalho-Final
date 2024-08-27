from skimage.feature import local_binary_pattern
from typing import Literal
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import os
#import cv2 

def lbp(image : np.ndarray,
        P : int = 8,
        R : int = 2,
        method : Literal['default', 'ror', 'uniform', 'nri_uniform', 'var'] = 'nri_uniform'):

    assert isinstance(image, np.ndarray) and len(image.shape) == 2
    desc = local_binary_pattern(image, P, R, method=method)
    n_bins = int(desc.max() + 1)
    hist, _ = np.histogram(desc, density=True, bins=n_bins, range=(0, n_bins))

    return hist

dir = './SoybeanSeeds/Broken'
lbp_hist = []
for i in range(51, 950):
    img_path = os.path.join(dir, f'{i}.jpg')
    gray_img = imread(img_path, as_gray=1)
    gray_img_u = img_as_ubyte(gray_img)
    hist : np.ndarray[float] = np.ndarray(1)
    hist = lbp(gray_img_u,
            P=8,
            R=2,
            method='nri_uniform')
    hist = hist.astype("float")
    hist /= (hist.sum()+ 1e-6)
    hist2 : np.ndarray[float] = np.ndarray(1)
    hist = np.insert(hist, 0, 1)

    lbp_hist.append(hist)


dir = './SoybeanSeeds/Immature'
for i in range(51, 1001):
    img_path = os.path.join(dir, f'{i}.jpg')
    gray_img = imread(img_path, as_gray=1)
    gray_img_u = img_as_ubyte(gray_img)
    hist : np.ndarray[float] = np.ndarray(1)
    hist = lbp(gray_img_u,
            P=8,
            R=2,
            method='nri_uniform')
    hist = hist.astype("float")
    hist /= (hist.sum()+ 1e-6)
    hist2 : np.ndarray[float] = np.ndarray(1)
    hist = np.insert(hist, 0, 2)

    lbp_hist.append(hist)


dir = './SoybeanSeeds/Skin-damaged'
for i in range(51, 1001):
    img_path = os.path.join(dir, f'{i}.jpg')
    gray_img = imread(img_path, as_gray=1)
    gray_img_u = img_as_ubyte(gray_img)
    hist : np.ndarray[float] = np.ndarray(1)
    hist = lbp(gray_img_u,
            P=8,
            R=2,
            method='nri_uniform')
    hist = hist.astype("float")
    hist /= (hist.sum()+ 1e-6)
    hist2 : np.ndarray[float] = np.ndarray(1)
    hist = np.insert(hist, 0, 3)

    lbp_hist.append(hist)

dir = './SoybeanSeeds/Spotted'

for i in range(51, 1001):
    img_path = os.path.join(dir, f'{i}.jpg')
    gray_img = imread(img_path, as_gray=1)
    gray_img_u = img_as_ubyte(gray_img)
    hist : np.ndarray[float] = np.ndarray(1)
    hist = lbp(gray_img_u,
            P=8,
            R=2,
            method='nri_uniform')
    hist = hist.astype("float")
    hist /= (hist.sum()+ 1e-6)
    hist2 : np.ndarray[float] = np.ndarray(1)
    hist = np.insert(hist, 0, 4)

    lbp_hist.append(hist)

dir = './SoybeanSeeds/Intact'
for i in range(51, 1001):
    img_path = os.path.join(dir, f'{i}.jpg')
    gray_img = imread(img_path, as_gray=1)
    gray_img_u = img_as_ubyte(gray_img)
    hist : np.ndarray[float] = np.ndarray(1)
    hist = lbp(gray_img_u,
            P=8,
            R=2,
            method='nri_uniform')
    hist = hist.astype("float")
    hist /= (hist.sum()+ 1e-6)
    hist2 : np.ndarray[float] = np.ndarray(1)
    hist = np.insert(hist, 0, 5)

    lbp_hist.append(hist)


df_lbp_hist_b = pd.DataFrame(lbp_hist)

df_lbp_hist_b.to_csv('./features.csv', index=False)


print(df_lbp_hist_b)


#colored_img = cv2.imread('./SoybeanSeeds/Broken/1.jpg')
#cv2.imshow(colored_img)
