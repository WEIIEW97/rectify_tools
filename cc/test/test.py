import cv2
import numpy as np
import pandas as pd


def read_img(path):
    return cv2.imread(path)


def diff_cmp(img1, img2):
    print(f'img1 size: {img1.shape}')
    print(f'img2 size: {img2.shape}')

    diff_loc = img1[:, :, 0] == img2[:, :, 0]
    print(f'different location are: {diff_loc}')

    diff_cnt = np.sum(diff_loc == False)
    print(f'different numbers is: {diff_cnt}')

    diff_val = img1[:, :, 0].astype(
        np.double) - img2[:, :, 0].astype(np.double)
    print(
        f'max diff value: {diff_val.max()}, min diff value: {diff_val.min()}, avg diff value: {np.mean(diff_val)}')


def diff_locate(diff):
    return np.where(diff == False)


def diff_to_excel(path, diff):
    x = [val for val in diff[0]]
    y = [val for val in diff[1]]
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_excel(path)
    return


def img_to_csv(path, img):
    df = pd.DataFrame(img)
    df.to_csv(path, index=False, index_label=False, encoding='utf-8')
    return


if __name__ == '__main__':
    PATH_MATLAB = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/rectL_Y.png"
    PATH_CPP = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/rectL_cpp1.png"
    OUT_PATH = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/diff.xlsx"
    OUT_PATH1 = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/img_mat.csv"
    OUT_PATH2 = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/img_cpp.csv"
    OUT_PATH3 = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/diff.csv"

    img1, img2 = read_img(PATH_MATLAB), read_img(PATH_CPP)

    diff_cmp(img1, img2)

   # img_to_csv(OUT_PATH1, img1[:, :, 0])
   # img_to_csv(OUT_PATH2, img2[:, :, 0])
   # img_to_csv(OUT_PATH3, img1[:, :, 0].astype(np.double) - img2[:, :, 0].astype(np.double))
