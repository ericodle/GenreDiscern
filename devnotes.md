# 1 set up env
source env/bin/activate

# 2 extract mfccs
python3 src/MFCC_extraction.py /home/eo/Documents/genres_original/ ./output/ gtzan_mfcc

# 3 train
python3 src/train_xlstm.py ./output/gtzan_mfcc.json xLSTM ./output 0.001


acess tensorboard
tensorboard --logdir=output/

running the full grid hyperparameter search (256 runs)
python3 src/grid_search_xlstm.py

How to execute a single specific train run:
python3 src/train_xlstm.py ./output/gtzan_mfcc.json xLSTM ./output 0.001 --batch_size 64 --hidden_size 128 --num_layers 2 --dropout 0.2 --optimizer adam --init xavier --class_weight auto