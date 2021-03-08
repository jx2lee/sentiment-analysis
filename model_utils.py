from utils import set_logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def array_to_df(x_arr, y_arr):
    def trans_target(array):
        result = 1
        if array[0] == 1:
            result = 0
        return result

    seq = []
    seq_len = []
    sentiment = []

    for i in range(len(x_arr)):
        seq.append(x_arr[i])
        seq_len.append(len(x_arr[i]))
        sentiment.append(trans_target(y_arr[i]))

    dic = {'seq': seq, 'seqLength': seq_len, 'sentiment': sentiment}
    df_res = pd.DataFrame(dic, columns=['seq', 'seqLength', 'sentiment'])
    model_logger.info(f"Created dataFrame.\tShape: {df_res.shape}")
    return df_res


def load_data(np_path, x_name, y_name):
    x = np.load(np_path + "/" + x_name, allow_pickle=True)['arr_0']
    y = np.load(np_path + "/" + y_name, allow_pickle=True)['arr_0']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


class Data:
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()
        self.cursor = 0

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def next_batch(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()
        # res = self.df.ix[self.cursor:self.cursor + n - 1]
        res = self.df.loc[self.cursor:self.cursor + n - 1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        max_len = max(res['seqLength'])
        x = np.zeros([n, max_len], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['seqLength'].values[i]] = res['seq'].values[i]

        return x, res['sentiment'], res['seqLength']


model_logger = set_logger('model_')