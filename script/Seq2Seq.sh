python -u main.py --cell LSTM --feature MS --seq_len 160 --pred_len 40 --num_hidden 256 --num_layer 2 --exp_num 5
python -u main.py --cell GRU --feature MS --seq_len 160 --pred_len 40 --num_hidden 256 --num_layer 2 --exp_num 5

python -u main.py --cell LSTM --feature MS --seq_len 96 --pred_len 24 --num_hidden 256 --num_layer 2 --exp_num 5
python -u main.py --cell GRU --feature MS --seq_len 96 --pred_len 24 --num_hidden 256 --num_layer 2 --exp_num 5

python -u main.py --cell LSTM --feature MS --seq_len 40 --pred_len 8 --num_hidden 64 --num_layer 2 --exp_num 5
python -u main.py --cell GRU --feature MS --seq_len 40 --pred_len 8 --num_hidden 64 --num_layer 2 --exp_num 5


python -u main.py --cell LSTM --feature S --seq_len 160 --pred_len 40 --num_hidden 256 --num_layer 2 --exp_num 5
python -u main.py --cell GRU --feature S --seq_len 160 --pred_len 40 --num_hidden 256 --num_layer 2 --exp_num 5

python -u main.py --cell LSTM --feature S --seq_len 96 --pred_len 24 --num_hidden 256 --num_layer 2 --exp_num 5
python -u main.py --cell GRU --feature S --seq_len 96 --pred_len 24 --num_hidden 256 --num_layer 2 --exp_num 5

python -u main.py --cell LSTM --feature S --seq_len 40 --pred_len 8 --num_hidden 64 --num_layer 2 --exp_num 5
python -u main.py --cell GRU --feature S --seq_len 40 --pred_len 8 --num_hidden 64 --num_layer 2 --exp_num 5
