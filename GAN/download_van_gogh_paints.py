import requests
from io import open as iopen
import pandas as pd
import os


def read_url_from_csv(file_path):

    csv = pd.read_csv(file_path)
    csv = csv[(csv['Artist'] == 'Vincent van Gogh')]
    csv = csv['ImageURL'].as_matrix()

    return csv


def download_image(path, logging=False):
    # set save_dir and file_name
    save_dir = './gogh/'
    file_name = path.split('/')[-1]

    # if already downloaded pass
    img_list = os.listdir(save_dir)
    if file_name in img_list:
        return False

    # img download and save to file
    img = requests.get(path)
    if img.status_code == requests.codes.ok:
        with iopen(save_dir + file_name, 'wb') as file:
            file.write(img.content)
            if logging:
                print('Download Done : {}'.format(file_name))
    else:
        return False


def main():
    for url in read_url_from_csv('./vgdb_2016.csv'):
        download_image(url, True)


if __name__ == '__main__':
    main()
