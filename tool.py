import numpy as np
import torch
import os


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


class EarlyStopping:
    def __init__(self):
        self.patience = 3
        self.counter = 0
        self.early_stop = False
        self.min_vali_loss = np.Inf

    def __call__(self, vali_loss, model, path):
        if vali_loss < self.min_vali_loss:
            self.save_checkpoint(vali_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, vali_loss, model, checkpoint_path):
        print(f"Validation loss decreased ({self.min_vali_loss:.6f} --> {vali_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), checkpoint_path)
        self.min_vali_loss = vali_loss
