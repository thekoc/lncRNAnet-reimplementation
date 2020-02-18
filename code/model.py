
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, Input, Activation, Masking, Lambda, TimeDistributed, Flatten, Dot, Concatenate

def sensitivity(y_true, y_pred):
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    true_negatives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def build_stop():
    stop_model = load_model('./data/model/stopfinder_singleframe.h5')
    return stop_model

char_num = 4
outD = 2
dropout = 0.5
hidden = 100
orfref = 10000
def build_rnn():
    
    RNN=GRU
    
    rnn_input=Input(shape=(None,char_num))
    orf_input=Input(shape=(None, 2))
    
    #orf
    orf_size=Lambda(lambda x: K.expand_dims(K.sum(x,axis=-2)[:,0]/orfref,axis=-1), output_shape=lambda  s: (s[0],1))(orf_input)#.repeat(1)
    orf_ratio=Lambda(lambda x: K.sum(x,axis=-1),output_shape=lambda s: (s[0],s[1]))(rnn_input)
    orf_ratio=Lambda(lambda x: orfref/(K.sum(x,axis=-1,keepdims=True)+1),output_shape=lambda s: (s[0],1))(orf_ratio)
    orf_ratio=Dot(-1)([orf_size,orf_ratio])
    
    orf_in=RNN(hidden,return_sequences=True)(orf_input)
    rnn_in=RNN(hidden,return_sequences=True)(rnn_input)
        
    rnn_in=Concatenate(name='cat1')([orf_in,rnn_in])
    rnn_in=RNN(hidden,return_sequences=False)(rnn_in)
    rnn_in=Dropout(dropout)(rnn_in)
        
    rnn_in=Concatenate(name='cat2')([rnn_in,orf_size,orf_ratio])
    rnn_out=Dense(outD)(rnn_in)
    rnn_act=Activation('softmax')(rnn_out)
    
    model=Model(inputs=[rnn_input,orf_input],outputs=rnn_act)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', sensitivity, specificity])
    
    return model