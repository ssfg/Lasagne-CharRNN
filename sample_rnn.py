#/bin/python2

import numpy as np
import theano
from theano import tensor as T
import lasagne
from nnio import *
from train_char_rnn import build_network
from gen_data import gen_data, str2matrix
import sys
MAX_LENGTH = 200
LEARNING_RATE = 0.001
GRAD_CLIP = 100
N_ITERATIONS = 1000
N_HIDDEN = 150
N_FEAT_DIM = 256
CHECK_FREQUENCY = 50
SAMPLE_LENGTH = 100


def main(fname):
    l_out = build_network(1, MAX_LENGTH)
    read_model_data(l_out, fname)
    
    in_val = T.tensor3('input')
    network_output = lasagne.layers.get_output(l_out, in_val)

    train_f = open('chatlog.txt','r')
    f_data = train_f.read()
    startidx = 4700
    seed_str = f_data[startidx:startidx+MAX_LENGTH]

    instr = np.zeros((1, MAX_LENGTH, N_FEAT_DIM))
    
    instr =instr.astype(theano.config.floatX)
    instr[0,:,:] = str2matrix(seed_str)
    sample = theano.function([in_val], network_output)
    sys.stdout.write(seed_str)
    sys.stdout.write('||')
    
    for i in range(SAMPLE_LENGTH):
        out_vals = sample(instr)
        last = out_vals[0,-1,:]
        sys.stdout.write(chr(last.argmax(axis=0)))
        sys.stdout.flush()
        instr2 = np.zeros((1, MAX_LENGTH, N_FEAT_DIM))
        instr2[:,0:-1,:]= instr[:,1:,:]
        instr2[:,-1,last.argmax(axis=0)] = 1
        instr2 = instr2.astype(theano.config.floatX)
        instr = instr2

if __name__ == '__main__':
    main(sys.argv[1])
