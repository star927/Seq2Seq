import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, cell, enc_in, num_hidden, num_layer, dropout=0.0):
        super(Encoder, self).__init__()
        assert cell == "GRU" or cell == "LSTM"
        self.cell = cell
        if self.cell == "GRU":
            self.rnn = nn.GRU(enc_in, num_hidden, num_layer, dropout=dropout)
        elif self.cell == "LSTM":
            self.rnn = nn.LSTM(enc_in, num_hidden, num_layer, dropout=dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, enc_in)
        # 在循环神经网络模型中，第一个轴对应于时间步
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, enc_in)
        output, state = self.rnn(x)
        # output: (seq_len, batch_size, num_hidden)
        # 对于GRU, state: (num_layer, batch_size, num_hidden)
        # 对于LSTM, state = (H, C)
        # H: (num_layer, batch_size, num_hidden)
        # C: (num_layer, batch_size, num_hidden)
        return output, state


class Decoder(nn.Module):
    def __init__(self, cell, dec_in, num_hidden, num_layer, dec_out, dropout=0.0):
        super(Decoder, self).__init__()
        assert cell == "GRU" or cell == "LSTM"
        self.cell = cell
        if self.cell == "GRU":
            self.rnn = nn.GRU(dec_in + num_hidden, num_hidden, num_layer, dropout=dropout)
        elif self.cell == "LSTM":
            self.rnn = nn.LSTM(dec_in + num_hidden, num_hidden, num_layer, dropout=dropout)
        self.dense = nn.Linear(num_hidden, dec_out)

    def init_state(self, state):
        # 如果是LSTM, state = (H, C)
        self.enc_hidden_state = state if self.cell == "GRU" else state[0]
        # (num_layer, batch_size, num_hidden)
        self.enc_hidden_state = self.enc_hidden_state[-1]  # Decoder最后一层的最后的隐状态
        # (batch_size, num_hidden)

    def forward(self, x, state):
        # 训练 x: (batch_size, pred_len, dec_in)
        # 预测 x: (batch_size, 1, dec_in)
        x = x.permute(1, 0, 2)
        # 广播context，使其具有与x相同长度的时间步
        context = self.enc_hidden_state.repeat(x.shape[0], 1, 1)
        x_and_context = torch.cat((x, context), dim=2)
        output, state = self.rnn(x_and_context, state)
        # output: (pred_len或1, batch_size, num_hidden)
        # state: (num_layer, batch_size, num_hidden)
        output = self.dense(output).permute(1, 0, 2)
        # output: (batch_size, pred_len或1, dec_out)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.encoder = Encoder(
            self.args.cell,
            self.args.enc_in,
            self.args.num_hidden,
            self.args.num_layer,
            self.args.dropout,
        )
        self.decoder = Decoder(
            self.args.cell,
            self.args.dec_in,
            self.args.num_hidden,
            self.args.num_layer,
            self.args.dec_out,
            self.args.dropout,
        )

    def forward(self, batch_x, batch_y, batch_x_time, batch_y_time):
        # batch_x: (batch_size, seq_len, 变量的个数)
        # batch_x_time: (batch_size, seq_len, 时间特征的维度数) 时间特征的维度数，如月、日、小时
        # batch_y: (batch_size, 1 + pred_len, 变量的个数)
        # batch_x_time: (batch_size, 1 + pred_len, 时间特征的维度数)

        x_enc = torch.cat([batch_x, batch_x_time], dim=2)
        enc_out, state = self.encoder(x_enc)
        self.decoder.init_state(state)
        if self.training:  # 训练模式，Decoder输入的是准确的数据
            x_dec = torch.cat([batch_y[:, :-1, :], batch_y_time[:, :-1, :]], dim=2)
            dec_out, state = self.decoder(x_dec, state)
            return dec_out

        # eval模式
        out = []
        x_dec = batch_y[:, [0], :]
        for i in range(self.args.pred_len):
            x_dec = torch.cat([x_dec, batch_y_time[:, [i], :]], dim=2)
            dec_cell_out, state = self.decoder(x_dec, state)
            out.append(dec_cell_out)
            x_dec = dec_cell_out
        dec_out = torch.cat(out, dim=1)
        return dec_out
