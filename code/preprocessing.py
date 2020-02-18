from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical   
import numpy as np
import pandas as pd
import h5py
from Bio import SeqIO

from model import build_stop

CHARS = 'ACGT'
CHAR_NUM = 4

def batch_mask_one_hot(sequences, num_classes):
    sequences=(np.arange(num_classes + 1) == sequences[:, :, None]).astype(dtype='float32') #one_hot
    return np.delete(sequences, 0, axis=-1)

def mask_one_hot(sequence, num_classes):
    """This one-hot encoder will keep value 0 as [0, 0, 0...]"""
    return batch_mask_one_hot(sequence.reshape((1,) + sequence.shape), num_classes)[0]


def encode_sequence(str_sequence):
    int_array = np.empty(len(str_sequence))
    for i, c in enumerate(str_sequence):
        index = CHARS.index(c) + 1
        int_array[i] = index
    return int_array

def load_and_preprocess_fa(filename, limit=21000, maxlen=None):
    """Load a fasta file and encode it. Discard the sequences longer than `maxlen` and those have chars rather than `chars`.
    
    Arguments:
        filename {str} -- filename
    
    Returns:
        list -- A int array list of the sequences, starts from 1 ([[1, 2, 3, 4], [2, 4, 6, 7]....])
    """

    results = []
    number = 0
    with open(filename) as f:
        for s in SeqIO.parse(f,'fasta'):
            if limit is not None and number >= limit:
                break
            try:
                if maxlen is not None:
                    if len(s) > maxlen:
                        continue
                encoded = encode_sequence(s)
                results.append(encoded)
                number += 1
            except ValueError:
                pass
    return results

def find_ORF(seq, model):
    orflen=0
    o_s=0
    o_e=0
    length=len(seq)
    model_input = mask_one_hot(seq, num_classes=CHAR_NUM)
    for frame in range(3):
        tseq=model.predict(model_input[None, frame:])[:,:(length-frame)//3]
        tseq=np.argmax(tseq,axis=-1)-1
        sseq=np.append(-1,np.where(tseq==1)[1])
        sseq=np.append(sseq,tseq.shape[1])
        lseq=np.diff(sseq)-1
        flenp=np.argmax(lseq)
        flen=lseq[flenp]
        n_s=frame+3*sseq[flenp]+3
        n_e=frame+3*sseq[flenp+1]
        
        if flen>orflen or ((orflen==flen) and n_s<o_s):
            orflen=flen
            o_s=n_s
            o_e=n_e
    
    orf = np.ones(o_e - o_s)
    orf = np.concatenate([2*np.ones(o_s),orf,2*np.ones(length-o_e)])
                
    return orf


def batch_find_ORF(seqs, model):
    """
    Arguments:
        seqs {list} -- [[1, 2, 3, 4], [2, 3, 4, 1], ...]
    
    Returns:
        list -- orfs of the same length
    """
    batches=[]
    for seq in seqs:
        orf = find_ORF(seq, model)
        batches.append(orf)
    
    batches=np.array(batches)
    return batches


def arg_split(X, step=500, maxlen=100000):
    """Split the arguments of an array
    
    Arguments:
        X {np.ndarray} -- X
    
    Keyword Arguments:
        step {int} -- step (default: {500})
        maxlen {int} -- maxlen (default: {200})
    
    Returns:
        list -- [(upper_bound, bucket_indices1), (upper_bound, bucket_indices2)...]
    """

    data_set = [(upper_bound, []) for upper_bound in range(step, maxlen, step)]
    for i, x in enumerate(X):
        assert len(x) < maxlen
        for upper_bound, bucket_data in data_set:
            if len(x) <= upper_bound:
                bucket_data.append(i)
                break
    return data_set

def generate_h5file(positive_files, negtive_files, outputname, bucket_step=500, bucket_num=200, limit=None, ratios=None):
    model = build_stop()
    X_all = np.empty(0)
    ORF_all = np.empty(0)
    y_all = np.empty(0)
    fdict = {0: negtive_files, 1: positive_files}
    for label in fdict:
        files = fdict[label]
        for filename in files:
            print('loading fasta file', filename)
            seqs = load_and_preprocess_fa(filename, limit=limit / 2 if limit else None, maxlen=3000)
            print('loaded fasta file', filename)
            print('finding ORF of file', filename)
            orfs = batch_find_ORF(seqs, model)
            l = len(seqs)
            X_all = np.concatenate([X_all, seqs])
            ORF_all = np.concatenate([ORF_all, orfs])
            y_all = np.concatenate([y_all, label * np.ones(l)])

    print('sequences loaded')

    indices_all = np.arange(len(y_all))
    length = len(y_all)
    np.random.shuffle(indices_all)
    indices_all = indices_all[:limit]

    def saveh5(filename, X, ORF, y):
        dt = h5py.vlen_dtype(np.dtype('int32'))
        with h5py.File(filename, 'w') as h5file:
            h5file.create_dataset('X', dtype=dt, data=X)
            h5file.create_dataset('ORF', dtype=dt, data=ORF)
            h5file.create_dataset('y', data=y)

    if ratios is not None:
        test_length = int(length * ratios[0])
        val_length = int(length * ratios[1])
        train_length = int(length * ratios[2])

        test_indices = indices_all[:test_length]
        validation_indices = indices_all[test_length:test_length + val_length]
        train_indices = indices_all[test_length+val_length:test_length+val_length+train_length]
        saveh5('{}-test.h5'.format(outputname), X_all[test_indices], ORF_all[test_indices], y_all[test_indices])
        saveh5('{}-validation.h5'.format(outputname), X_all[validation_indices], ORF_all[validation_indices], y_all[validation_indices])
        saveh5('{}-train.h5'.format(outputname), X_all[train_indices], ORF_all[train_indices], y_all[train_indices])
    else:
        saveh5(outputname, X_all[indices_all], ORF_all[indices_all], y_all[indices_all])


def prepad_onehot(X, num_classes, maxlen):
    return batch_mask_one_hot(pad_sequences(X, maxlen), num_classes)

def load_bucket_dataset(h5filename, limit=None):
    """Load dataset
    
    Arguments:
        h5filename {str} -- filename
    
    Returns:
        tuple -- X_buckets, ORF_buckets, y_buckets
    """
    with h5py.File(h5filename, mode='r') as h5f:
        X = np.array(h5f['X'])
        ORF = np.array(h5f['ORF'])
        y = np.array(h5f['y'])

        if limit is not None:
            np.random.seed(0)
            indices = np.random.choice(len(y), limit, replace=False)
            X = X[indices]
            ORF = ORF[indices]
            y = y[indices]
        
        X_buckets = []
        ORF_buckets = []
        y_buckets = []
        
        bucket_indices = arg_split(X, 500, 5000)
        for upper_bound, indices in bucket_indices:
            X_bucket = prepad_onehot(X[indices], CHAR_NUM, upper_bound)
            ORF_bucket = prepad_onehot(ORF[indices], 2, upper_bound)
            y_bucket = to_categorical(y[indices], num_classes=2)

            X_buckets.append(X_bucket)
            ORF_buckets.append(ORF_bucket)
            y_buckets.append(y_bucket)
        
        return X_buckets, ORF_buckets, y_buckets

def load_dataset(h5filename, limit=None, maxlen=3000):
    with h5py.File(h5filename, mode='r') as h5f:
        X = np.array(h5f['X'])
        ORF = np.array(h5f['ORF'])
        y = np.array(h5f['y'])
    # X = np.array([to_categorical(x - 1, CHAR_NUM) for x in X])
    # ORF = np.array([to_categorical(orf - 1, 2) for orf in ORF])
    X = prepad_onehot(X, CHAR_NUM, maxlen)
    ORF = prepad_onehot(ORF, 2, maxlen)
    y = to_categorical(y, num_classes=2)
    if limit is not None:
        X = X[:limit]
        ORF = ORF[:limit]
        y = y[:limit]
    return X, ORF, y

def load_dataset_concatnate_X(h5filename, limit=None, maxlen=3000):
    with h5py.File(h5filename, mode='r') as h5f:
        X = np.array(h5f['X'])
        ORF = np.array(h5f['ORF'])
        y = np.array(h5f['y'])
    X = pad_sequences(X, maxlen)
    ORF = pad_sequences(ORF, maxlen)
    CONCATNATED = X * ORF
    CONCATNATED = batch_mask_one_hot(CONCATNATED, CHAR_NUM * 2)
    
    return CONCATNATED, y

def load_svm_dataset(h5filename, limit=None, maxlen=3000):
    with h5py.File(h5filename, mode='r') as h5f:
        X = np.array(h5f['X'])
        ORF = np.array(h5f['ORF'])
        y = np.array(h5f['y'])
    if limit is not None:
        X = X[:limit]
        ORF = ORF[:limit]
        y = y[:limit]
    X = pad_sequences(X, maxlen)
    ORF = pad_sequences(ORF, maxlen)
    CONCATNATED = X * ORF
    CONCATNATED = batch_mask_one_hot(CONCATNATED, CHAR_NUM * 2)
    sample_num = len(CONCATNATED)
    CONCATNATED = CONCATNATED.reshape(sample_num, CONCATNATED.shape[1] * CONCATNATED.shape[2])
    
    return CONCATNATED, y * 2 - 1

if __name__ == "__main__":
    # posf = ['data/train/gencode.v25.lncRNA_transcripts.fa']
    # negf = ['data/train/gencode.v25.pc_transcripts.fa']
    # generate_h5file(posf, negf, 'data/train/49000')

    posf = ['data/train/gencode.vM9.lncRNA_transcripts.fa']
    negf = ['data/train/gencode.vM9.pc_transcripts.fa']
    generate_h5file(posf, negf, 'data/test/mouse.h5', limit=3500)