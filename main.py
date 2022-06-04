import argparse
import pandas as pd
import torch
import os
from exp import Exp_Seq2Seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq, 时间序列预测")

    parser.add_argument("--cell", type=str, default="GRU", help="可选项：[GRU, LSTM]")
    parser.add_argument("--feature", type=str, default="MS", help="可选项：[MS, S]，MS：多变量预测单变量，S：单变量预测单变量")
    parser.add_argument("--seq_len", type=int, default=40, help="编码器输入长度")
    parser.add_argument("--pred_len", type=int, default=8, help="解码器预测长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_hidden", type=int, default=64, help="隐藏层的节点个数")
    parser.add_argument("--num_layer", type=int, default=2, help="隐藏层的层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout值")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="学习率")
    parser.add_argument("--data_path", type=str, default="./data/Weather_WH.csv", help="数据文件的路径")
    parser.add_argument("--max_train_epochs", type=int, default=10, help="最大训练的epoch数")
    parser.add_argument("--exp_num", type=int, default=1, help="实验次数")
    parser.add_argument("--predict", action="store_true", default=False, help="是否进行预测")

    args = parser.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Weather_WH: Month, Day, Hour, Po, P, U, Ff, Td, T
    data_info = {"S": [4, 4, 1], "MS": [9, 4, 1]}
    # 编码器输入变量的个数, 解码器输入变量的个数, 解码器预测变量的个数
    args.enc_in, args.dec_in, args.dec_out = data_info[args.feature]

    for i in range(args.exp_num):
        exp = Exp_Seq2Seq(args)

        setting = "{}_ft{}_sl{}_pl{}_bs{}_nh{}_nl{}_do{}_lr{}_{}".format(
            args.cell,
            args.feature,
            args.seq_len,
            args.pred_len,
            args.batch_size,
            args.num_hidden,
            args.num_layer,
            args.dropout,
            args.learning_rate,
            i,
        )

        checkpoint_folder = "./checkpoint"
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        checkpoint_path = os.path.join(checkpoint_folder, f"{setting}.pth")  # checkpoint_path对应最小的验证集loss

        print(">>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>")
        actual_train_epochs, train_cost_time = exp.train(checkpoint_path)

        print(">>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        mse, mae = exp.test()

        # 存储实验结果
        path_result = "./result_seq2seq.csv"
        open(path_result, "a").close()
        result = vars(args)
        result["actual_train_epochs"] = actual_train_epochs
        result["train_cost_time"] = train_cost_time
        result["test_mse"] = mse
        result["test_mae"] = mae
        try:
            df = pd.read_csv(path_result)
            pd.concat([df, pd.DataFrame([result])]).to_csv(path_result, index=False)
        except:
            pd.DataFrame([result]).to_csv(path_result, index=False)

        # 在测试集中随机选取6个样本进行预测，预测结果的图片存放在img文件夹下
        if args.predict:
            exp.predict()

        torch.cuda.empty_cache()
