import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class StandardScaler:
    """将数据进行标准化和恢复成真实数据"""
    def __init__(self):
        self.x_mean = 0.0
        self.x_std = 1.0

    def fit_x(self, data):
        self.x_mean = data.mean(0)
        self.x_std = data.std(0)

    def fit_y(self, data):
        self.y_mean = data.mean(0)
        self.y_std = data.std(0)

    def transform_x(self, data):
        """数据标准化"""
        return (data - self.x_mean) / self.x_std

    def transform_y(self, data):
        """数据标准化"""
        return (data - self.y_mean) / self.y_std

    def inverse_transform_y(self, data):
        """将数据恢复成真实数据"""
        return (data * self.y_std) + self.y_mean

    def transform_time(self, data):
        """时间放缩在[-0.5,0.5]区间"""
        data[:, 0] = (data[:, 0] - 1) / 11.0 - 0.5  # Month
        data[:, 1] = (data[:, 1] - 1) / 30.0 - 0.5  # Day
        data[:, 2] = data[:, 2] / 23.0 - 0.5  # Hour
        return data

    def inverse_transform_time(self, data):
        """将时间恢复成真实时间"""
        data[:, 0] = (data[:, 0] + 0.5) * 11 + 1  # Month
        data[:, 1] = (data[:, 1] + 0.5) * 30 + 1  # Day
        data[:, 2] = (data[:, 2] + 0.5) * 23  # Hour
        return np.floor(data + 0.5).astype(int)  # 四舍五入转成int


class Dataset_Weather(Dataset):
    def __init__(self, data_path, dataset_type, feature, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        type_map = {"train": 0, "vali": 1, "test": 2}
        self.set_type = type_map[dataset_type]
        self.feature = feature

        self.data_path = data_path
        self.__read_data()

    def __read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.data_path)
        # df.columns: [Year, Month, Day, Hour, Po, P, U, Ff, Td, T]

        num_train = int(len(df) * 0.8)
        num_vali = int(len(df) * 0.1)
        num_test = len(df) - num_train - num_vali
        # border1s, border2s: 分别是训练集、验证集、测试集的数据下标的起止范围
        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_x = df[["Po", "P", "U", "Ff", "Td", "T"]].values if self.feature == "MS" else df[["T"]].values
        data_y = df[["T"]].values
        data_time = df[["Month", "Day", "Hour"]].values

        self.data_x = data_x[border1:border2].astype(np.float32)
        self.data_y = data_y[border1:border2].astype(np.float32)
        self.data_time = data_time[border1:border2].astype(np.float32)

        # 数据标准化
        train_data_x = data_x[border1s[0] : border2s[0]]
        self.scaler.fit_x(train_data_x)
        self.data_x = self.scaler.transform_x(self.data_x)

        train_data_y = data_y[border1s[0] : border2s[0]]
        self.scaler.fit_y(train_data_y)
        self.data_y = self.scaler.transform_y(self.data_y)

        self.data_time = self.scaler.transform_time(self.data_time)

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end - 1
        y_end = y_begin + 1 + self.pred_len

        seq_x = self.data_x[x_begin:x_end]
        seq_y = self.data_y[y_begin:y_end]
        seq_x_time = self.data_time[x_begin:x_end]
        seq_y_time = self.data_time[y_begin:y_end]
        return seq_x, seq_y, seq_x_time, seq_y_time

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform_y(self, data):
        """将标准化后的数据恢复成真实数据"""
        return self.scaler.inverse_transform_y(data)

    def inverse_transform_time(self, data):
        """将标准化后的时间恢复成真实时间"""
        return self.scaler.inverse_transform_time(data)
