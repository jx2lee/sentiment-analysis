import logging
import numpy as np
import os
import pickle
import random


def set_logger(logger_name):
    ret_logger = logging.getLogger(logger_name)
    ret_logger.setLevel(logging.INFO)
    formatter_ = logging.Formatter('[%(asctime)s][%(process)d][%(filename)s][%(funcName)s:%(lineno)d][%('
                                  'levelname)s] %(message)s')
    stream_handler_ = logging.StreamHandler()
    stream_handler_.setFormatter(formatter_)
    ret_logger.addHandler(stream_handler_)

    return ret_logger


def folder_check(dir_path, dir_name):
    if not os.path.exists(dir_path + '/' + dir_name):
        os.makedirs(dir_path + '/' + dir_name)
        util_logger.info(f'Created directory. {dir_path + "/" + dir_name}')
    else:
        util_logger.info(f'Existed directory. {dir_path + "/" + dir_name}')
    return None


def file_to_list(file_dir):
    pos_files = []
    neg_files = []

    for path, dirs, files in os.walk(file_dir):
        for i in range(len(files)):
            if files[i].find('pos') != -1:
                pos_files.append(files[i])
            else:
                neg_files.append(files[i])

    pos_reviews = []
    neg_reviews = []

    for file in pos_files:
        with open(file_dir + "/" + file, 'r', encoding='utf-8') as f:
            reviews = f.readlines()
        for i in range(len(reviews)):
            pos_reviews.append(reviews[i].replace('  \n', ''))

    for file in neg_files:
        with open(file_dir + "/" + file, 'r', encoding='utf-8') as f:
            reviews = f.readlines()
        for i in range(len(reviews)):
            neg_reviews.append(reviews[i].replace('  \n', ''))

    neg_reviews = random.sample(neg_reviews, 45000)
    pos_reviews = random.sample(pos_reviews, 45000)

    util_logger.info(
        f"Created Lists. neg_reviews's length: {len(neg_reviews)}\t pos_reviews's length: {len(pos_reviews)}")
    return neg_reviews + pos_reviews


def load_pkl(pkl_path: str, pkl_name: str):
    with open(pkl_path + pkl_name, 'rb') as p:
        res = pickle.load(p)

    util_logger.info(f'Loaded pkl. {pkl_path}{pkl_name}')
    return res


def save_pkl(pkl_path: str, pkl_name: str, save_object):
    with open(pkl_path + '/' + pkl_name, 'wb') as p:
        pickle.dump(save_object, p)
    util_logger.info(f'Saved pkl. {pkl_path}/{pkl_name}')
    return None


def save_npz(npz_path: str, npz_name: str):
    np.savez(npz_path + '/' + npz_name)
    util_logger.info(f'Saved npz. {npz_path}/{npz_name}')
    return None


util_logger = set_logger('utils')