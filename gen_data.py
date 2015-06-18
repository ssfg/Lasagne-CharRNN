#/bin/python2

import numpy as np
import theano


def str2matrix(input_str):
    out = np.zeros((len(input_str), 256))
    for i in range(len(input_str)):
        out[i, ord(input_str[i])] = 1

    return out

def gen_data(corpus, idx,  batch_size, seq_length):
    corpus_size = len(corpus)
    
    num_blocks = corpus_size // (batch_size*seq_length +1) - 2

    sub_corp = corpus[idx%num_blocks*batch_size*seq_length:(idx%num_blocks+1)*batch_size*seq_length+1]

    X = np.zeros((batch_size, seq_length, 256))
    Y = np.zeros((batch_size, seq_length, 256))
    for i in range(batch_size):
        x_str = sub_corp[i*seq_length:(i+1)*seq_length]
        y_str = sub_corp[i*seq_length+1:(i+1)*seq_length+1]
        X[i,:,:] = str2matrix(x_str)
        Y[i,:,:] = str2matrix(y_str)

    return (X.astype(theano.config.floatX), Y.astype(theano.config.floatX))
        
if __name__ == '__main__':
    train_f= open('chatlog.txt', 'r')
    f_data = train_f.read()
    gen_data(f_data, 0, 3, 150)


