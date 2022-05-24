import cv2
import numpy as np
import pandas as pd


def read_img(path):
    return cv2.imread(path)


def diff_cmp(img1, img2):
    diff_loc = img1[:, :, 0] == img2[:, :, 0]
    print(f'different location are: {diff_loc}')

    diff_cnt = np.sum(diff_loc == False)
    print(f'different numbers is: {diff_cnt}')


def diff_locate(diff):
    x, y = [], []
    print(diff.shape)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i][j] == False:
                x.append(i)
                y.append(j)
    return x, y


def to_excel(path, diff_x, diff_y):
    df = pd.DataFrame({'x': diff_x, 'y': diff_y})
    df.to_excel(path)
    return


if __name__ == '__main__':
    PATH_MATLAB = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/rectL_Y.png"
    PATH_CPP = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/rectL_cpp1.png"
    OUT_PATH = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/diff.xlsx"

    img1, img2 = read_img(PATH_MATLAB), read_img(PATH_CPP)

    diff_cmp(img1, img2)

    diff = img1[:, :, 0] - img2[:, :, 0]

    x, y = diff_locate(diff)

    to_excel(OUT_PATH, x, y)
