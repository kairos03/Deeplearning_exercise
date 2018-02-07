import requests
from io import open as iopen
import pandas as pd
import scipy.ndimage
import scipy.misc
import os
import matplotlib.pyplot as plt

origin_dir = '../0.data_set/gogh_origin/'
resize_dir = '../0.data_set/gogh_resize/'
resize64_dir = '../0.data_set/gogh_resize64/'
resize64gray_dir = '../0.data_set/gogh_resize64gray/'
r_128_gray_dir = '../0.data_set/gogh_r_128_gray/'


def read_url_from_csv(file_path):

    csv = pd.read_csv(file_path)
    csv = csv[(csv['Artist'] == 'Vincent van Gogh')]
    csv = csv['ImageURL'].as_matrix()

    return csv


def download_image(path, logging=False):
    # set save_dir and file_name

    file_name = path.split('/')[-1]

    # if already downloaded pass
    img_list = os.listdir(origin_dir)
    if file_name in img_list:
        return False

    # img download and save to file
    img = requests.get(path)
    if img.status_code == requests.codes.ok:
        with iopen(origin_dir + file_name, 'wb') as file:
            file.write(img.content)
            if logging:
                print('Download Done : {}'.format(file_name))
    else:
        return False


def resize_all_image(source_dir, result_dir, size, mode='RGB'):
    # image list
    img_list = os.listdir(source_dir)

    for fname in img_list:
        img = scipy.ndimage.imread(origin_dir + fname, mode=mode)
        img = scipy.misc.imresize(img, size)
        scipy.misc.imsave(result_dir + fname, img)


def main():
    pass
    # download
    # for url in read_url_from_csv('./vgdb_2016.csv'):
    #     download_image(url, True)
    resize_all_image(origin_dir, r_128_gray_dir, (128, 128), mode='L')


if __name__ == '__main__':
    main()
