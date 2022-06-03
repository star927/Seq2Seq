import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from data_loader import Dataset_Weather
from model import EncoderDecoder
import matplotlib
from matplotlib.ticker import MaxNLocator
import os

matplotlib.use("agg")
from matplotlib import pyplot as plt


class Exp_Seq2Seq:
    def __init__(self, args):
        super(Exp_Seq2Seq, self).__init__()
        self.args = args
        self.exp_info = {}
        self.model = EncoderDecoder(self.args)
        self.model.to(self.args.device)

    def _get_data(self, dataset_type):
        shuffle_flag = True if dataset_type == "train" or dataset_type == "pred" else False
        batch_size = 1 if dataset_type == "pred" else self.args.batch_size
        if dataset_type == "pred":  #  预测的时候，从测试集里取数据
            dataset_type = "test"
        dataset = Dataset_Weather(
            data_path=self.args.data_path,
            dataset_type=dataset_type,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            feature=self.args.feature,
        )
        print(dataset_type, len(dataset))
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=True,
        )

        return dataset, data_loader

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(vali_loader):
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_time, batch_y_time)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(dataset_type="train")
        vali_data, vali_loader = self._get_data(dataset_type="vali")
        test_data, test_loader = self._get_data(dataset_type="test")

        time_start = time.time()

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.HuberLoss()

        min_vali_loss = 1e10
        bad_train = 0
        total_iter_count = 0
        train_steps = len(train_loader)
        actual_train_epochs = self.args.max_train_epochs
        for epoch in range(self.args.max_train_epochs):
            train_loss = []

            self.model.train()
            time_epoch_start = time.time()
            for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(train_loader):
                total_iter_count += 1

                # batch_x: (batch_size, seq_len, 变量的个数)
                # batch_x_time: (batch_size, seq_len, 时间特征的维度数) 时间特征的维度数，如月、日、小时
                # batch_y: (batch_size, pred_len, 变量的个数)
                # batch_y_time: (batch_size, pred_len, 时间特征的维度数)
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_time, batch_y_time)
                loss = criterion(pred, true)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_start) / total_iter_count
                    left_time = speed * ((self.args.max_train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - time_epoch_start))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            # 每个epoch训练完，计算在验证集上的loss，记录迄今为止最小的loss，即变量min_vali_loss
            # 如果当前验证集上的loss比min_vali_loss大，则bad_train加1，bad_train=3则停止训练
            if vali_loss < min_vali_loss:
                min_vali_loss = vali_loss
                bad_train = 0
            else:
                bad_train += 1
                if bad_train == 3:
                    actual_train_epochs = epoch + 1
                    break

        train_cost_time = time.time() - time_start
        print("Train, cost time: {}".format(train_cost_time))
        return actual_train_epochs, train_cost_time

    def test(self):
        test_data, test_loader = self._get_data(dataset_type="test")
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(test_loader):
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_time, batch_y_time)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        # batch_num, batch_size, pred_len, num_var = preds.shape  # 预测单变量, num_var=1

        # 在训练的时候，所有数据进行了标准化
        # 现将标准化后的数据恢复成实际数据，再计算mae和mse
        preds = test_data.inverse_transform_y(preds.reshape(-1, 1))
        trues = test_data.inverse_transform_y(trues.reshape(-1, 1))
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        print("mse:{}, mae:{}".format(mse, mae))

        return mse, mae

    def predict(self):
        test_data, test_loader = self._get_data(dataset_type="pred")
        self.model.eval()
        preds = []
        trues = []
        times = []
        # 在测试集中随机选取6个数据样本
        for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(test_loader):
            # batch_y_time: (batch_size, 1 + pred_len, 时间特征的维度数)
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_time, batch_y_time)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            times.append(batch_y_time[:, 1:, :].cpu().numpy())
            if i == 5:
                break

        # np.array(preds): (batch_num, batch_size, pred_len, num_var), batch_num=6, batch_size=1, num_var=1
        # np.array(times): (batch_num, batch_size, pred_len, num_var), num_var=3包括月、日、小时
        preds = test_data.inverse_transform_y(np.array(preds).reshape(-1, 1)).reshape(6, -1)
        trues = test_data.inverse_transform_y(np.array(trues).reshape(-1, 1)).reshape(6, -1)
        times = test_data.inverse_transform_time(np.array(times).reshape(-1, 3)).reshape(6, -1, 3)
        mae = np.mean(np.abs(preds - trues))

        plt.figure(figsize=(15, 9))
        for i in range(6):
            month, day, hour = times[i, :, 0], times[i, :, 1], times[i, :, 2]
            date = []
            for j in range(len(month)):
                date.append(str(month[j]) + "-" + str(day[j]) + "-" + str(hour[j]) + ":00")
            plt.subplot(2, 3, i + 1)
            plt.plot(date, trues[i], label="true")
            plt.plot(date, preds[i], label="pred")
            plt.legend()
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_locator(MaxNLocator(8))  # x轴最多画8个刻度

        img_title = "{0} seq_len:{1} pred_len:{2} mae:{3:.6f}".format(
            self.args.cell, self.args.seq_len, self.args.pred_len, mae
        )
        plt.suptitle(img_title)
        if not os.path.exists("./img"):
            os.mkdir("./img")
        img_path = "./img/{0} seq_len({1}) pred_len({2}) mae({3:.6f}).png".format(
            self.args.cell, self.args.seq_len, self.args.pred_len, mae
        )
        plt.tight_layout()
        plt.savefig(img_path)

    def _process_one_batch(self, batch_x, batch_y, batch_x_time, batch_y_time):
        # batch_x: (batch_size, seq_len, 变量的个数)
        # batch_x_time: (batch_size, seq_len, 时间特征的维度数) 时间特征的维度数，如月、日、小时
        # batch_y: (batch_size, 1 + pred_len, 变量的个数)
        # batch_y_time: (batch_size, 1 + pred_len, 时间特征的维度数)

        batch_x = batch_x.float().to(self.args.device)
        batch_y = batch_y.float().to(self.args.device)
        batch_x_time = batch_x_time.float().to(self.args.device)
        batch_y_time = batch_y_time.float().to(self.args.device)

        pred = self.model(batch_x, batch_y, batch_x_time, batch_y_time)
        true = batch_y[:, -self.args.pred_len :, :]
        return pred, true
