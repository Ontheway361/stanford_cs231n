"""
Created on 2018/11/22
@author: lujie
"""

import os
from time import time
from datetime import datetime
from IPython import embed

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = '%s/dataset' % ROOT_PATH
CONF_PATH = '%s/config' % ROOT_PATH

def get_now():
    return datetime.fromtimestamp(time()).strftime('%Y%m%d%H%M%S')

def get_latest(file_path):
    """
    latest file timestamp, eg. 20161229154431
    :param file_path: the path to search for the latest file timestamp
    """
    files = [f.split('@')[-1] for f in get_files(file_path)]
    latest_time = max(files)

    return latest_time


def get_files(file_path):
    """
    list of filenames
    :param file_path: the path to search for all direct files
    :return:
    """
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path,f))]

    return files


def create_dir(path, path_type='file'):
    """
    if folder not exists, create one
    :param path: the dir/file path to check if exists
    :param path_type: dir or file
    :return:
    """
    if (path_type == 'file'):
        folder_path = os.path.dirname(path)
    elif (path_type == 'dir'):
        folder_path = path
    else:
        raise(ValueError, 'invalid path_type for utils.path.create_dir()')

    # check if folder exists, if not, create one
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def save_file(df, file):
    """
    save file
    :param df: save data
    :param file: file name
    :return: None
    """
    create_dir(file)
    df.to_csv(file, index=False)    # add the info the tail


if __name__ == '__main__':

    print(ROOT_PATH)
    embed()
