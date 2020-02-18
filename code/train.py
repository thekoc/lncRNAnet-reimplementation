#!/usr/bin/python3
#keras
from keras.utils.np_utils import to_categorical 
from keras.callbacks.callbacks import ModelCheckpoint

#etc
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np

from preprocessing import load_bucket_dataset, load_dataset
from model import build_rnn
    
def train():
    checkpoint = ModelCheckpoint('data/model/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=5)

    X_buckets, ORF_buckets, y_buckets = load_bucket_dataset('data/train/49000-train.h5')
    X_val, ORF_val, y_val = load_dataset('data/train/49000-validation.h5')
    history = []
    model = build_rnn()
    bucket_num = len(X_buckets)
    epochs = 200
    for epoch in range(epochs):
        print('===Epoch', epoch)
        for i in range(bucket_num):
            if len(y_buckets[i]) > 0:
                model.fit([X_buckets[i], ORF_buckets[i]], y_buckets[i], callbacks=[checkpoint], verbose=1)
        loss, acc, sensibility, specificity = model.evaluate([X_val, ORF_val], y_val)
        print('epoch: {}, acc: {}'.format(epoch, acc))
        history.append({'loss': loss, 'acc': acc})
    
    import pickle
    with open('history.pickle', 'wb') as f:
        pickle.dump(history, f)



if __name__ == "__main__":
    train()