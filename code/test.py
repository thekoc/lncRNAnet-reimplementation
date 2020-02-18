from model import build_rnn
import pickle
import numpy as np
from preprocessing import load_bucket_dataset, load_svm_dataset
def test_rnn(h5filename):
    X_human_bucket, ORFs_human_bucket, y_human_bucket = load_bucket_dataset(h5filename)

    def rnn_input_generator():
        for i in range(len(X_human_bucket)):
            if len(y_human_bucket) == 0:
                continue
            seqs = X_human_bucket[i]
            orfs = ORFs_human_bucket[i]
            y = y_human_bucket[i]
            yield ([seqs, orfs], y)


    bucket_number = sum([1 for b in y_human_bucket if len(b) > 0])
    gru = build_rnn()
    gru.load_weights('data/model/model-80.h5')
    print(gru.evaluate_generator(rnn_input_generator(), steps=bucket_number, verbose=1))

def test_svm(h5filename):
    def sensibility(y_true, y_pred):
        y_true = (y_true + 1) / 2
        y_pred = (y_pred + 1) / 2
        tp = np.sum(y_true * y_pred)
        return tp / np.sum(y_pred)

    def specificity(y_true, y_pred):
        y_true = -y_true
        y_pred = -y_pred
        return sensibility(y_true, y_pred)
    
    def accuracy(y_true, y_pred):
        return (np.sum(y_true == y_pred) / len(y_true))
    
    def metrics(y_true, y_pred):
        return [metric(y_true, y_pred) for metric in (accuracy, sensibility, specificity)]
        
    X_human, y_human = load_svm_dataset(h5filename)
    with open('svm.pickle', 'rb') as f:
        svm = pickle.load(f)
    y_pred = svm.predict(X_human)
    print('accuracy: {}, sensibility: {}, specificity: {}'.format(*metrics(y_human, y_pred)))

def main():
    # test_rnn('data/test/mouse.h5')
    test_svm('data/test/mouse.h5')


if __name__ == "__main__":
    main()