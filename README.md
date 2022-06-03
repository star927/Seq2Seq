# Sequence to Sequence

用GRU或LSTM实现的Seq2Seq模型

## 程序运行环境

- Pytorch

## 运行说明

| 参数名 | 参数说明 |
| ------ | -------- |
| cell | Seq2Seq中的结构单元，可选项：[GRU, LSTM]；默认值是GRU |
| feature | 可选项：[MS, S]，MS：多变量预测单变量，S：单变量预测单变量；默认是MS |
| seq_len | 编码器输入长度；默认值是40 |
| pred_len | 解码器预测长度；默认值是8 |
| batch_size | 批量大小；默认值32 |
| num_hidden | 隐藏层的节点个数；默认值是64 |
| num_layer | 隐藏层的层数；默认值是2 |
| dropout | Dropout值；默认值是0.1 |
| learning_rate | 学习率；默认值是0.005 |
| data_path | 数据文件的路径；默认值是`./data/Weather_WH.csv` |
| max_train_epochs | 最大训练的epoch数；默认值是10 |
| exp_num | 实验次数；默认值是1 |
| predict | 是否进行预测，如果有该参数，则在测试集中随机选取6个样本进行预测，并绘制图像，图片存放在img文件夹中；默认值是`False` |

运行示例：

```
python -u main.py --cell LSTM --feature S --seq_len 96 --pred_len 24 --exp_num 5 --predict
```

程序的运行结果会存放在文件`result_seq2seq.csv`中

## 数据集

数据存放在文件`data/Weather_WH.csv`中，数据表中各列具体含义如下表，其中有6个气象相关的变量，预测的目标变量是T。数据之间的时间间隔是3个小时，包含从`2015-4-1-2:00`到`2021-12-31-23:00`的19736条数据。

| 列名  | 说明                                              |
| ----- | ------------------------------------------------- |
| Year  | 年                                                |
| Month | 月                                                |
| Day   | 日                                                |
| Hour  | 小时                                              |
| Po    | 气象站水平的大气压(毫米汞柱)                      |
| P     | 平均海平面的大气压(毫米汞柱)                      |
| U     | 地面2米处的相对湿度                               |
| Ff    | 观测前10分钟内地面高度10~12米处的平均风速(米每秒) |
| Td    | 地面高度2米处的露点温度(摄氏度)                   |
| T     | 地面以上2米处的大气温度(摄氏度)                   |

数据来源：<https://rp5.ru/武汉市(机场)历史天气_>