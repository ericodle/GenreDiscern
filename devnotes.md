# 1 set up env
source env/bin/activate

# 2 extract mfccs
python src/MFCC_extraction.py /home/eo/Documents/genres_original/ ./output/ gtzan_mfcc

# 3 train
python3 src/train_xlstm.py ./output/gtzan_mfcc.json xLSTM ./output 0.0001

