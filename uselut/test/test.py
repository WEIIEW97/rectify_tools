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

    diff_val = img1[:, :, 0].astype(np.double) - img2[:, :, 0].astype(np.double)
    print(f'max diff value: {diff_val.max()}, min diff value: {diff_val.min()}, avg diff value: {np.mean(diff_val)}')


def diff_locate(diff):
    return np.where(diff == False)

def diff_to_excel(path, diff):
    x = [val for val in diff[0]]
    y = [val for val in diff[1]]
    df = df.DataFrame({'x':x, 'y':y})
    df.to_excel(path)
    return

def img_to_csv(path, img):
    df = pd.DataFrame(img)
    df.to_csv(path, index=False, index_label=False, encoding='utf-8')
    return

if __name__ == '__main__':
    PATH_MATLAB = "D:/rectify_tools/cpp/data/case3/output/rectR_Y.png"
    PATH_CPP = "D:/rectify_tools/cpp/data/case3/output/rectR_cpp.png"
    OUT_PATH = "D:/rectify_tools/cpp/data/case3/output/diff.xlsx"
    CSV_PATH1 = "D:/rectify_tools/cpp/data/case3/output/cpp.csv"
    CSV_PATH2 = "D:/rectify_tools/cpp/data/case3/output/matlab.csv"
    CSV_PATH3 = "D:/rectify_tools/cpp/data/case3/output/diff.csv"
    
    img1, img2 = read_img(PATH_MATLAB), read_img(PATH_CPP)

    diff_cmp(img1, img2)

    # img_to_csv(CSV_PATH2, img1[:, :, 0])
    # img_to_csv(CSV_PATH1, img2[:, :, 0])
    # img_to_csv(CSV_PATH3, img1[:, :, 0] - img2[:, :, 0])


