# 1
set up env

# 2
python src/MFCC_extraction.py /home/eo/Documents/genres_original/ ./output/ gtzan_mfcc

# 3
python3 src/train_model.py ./output/gtzan_mfcc.json LSTM ./output 0.001

