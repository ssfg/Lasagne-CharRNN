#/bin/python2

import numpy as np
import theano
from theano import tensor as T
from gen_data import gen_data, str2matrix
import lasagne
from nnio import *
from lasagne.objectives import categorical_crossentropy

N_BATCH = 200 
MAX_LENGTH = 60
LEARNING_RATE = 0.01
GRAD_CLIP = 100
N_ITERATIONS = 1000000
N_HIDDEN = 512
N_FEAT_DIM = 256 # for 256 ascii characters
CHECK_FREQUENCY = 50
CHECKPOINT_FREQUENCY = 1000

def build_network(batch_size, seq_length = MAX_LENGTH):
    # input layer
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_length, N_FEAT_DIM))
    # 3 LSTM layers
    l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, backwards=False, learn_init = True, peepholes=True)
    l_forward2 = lasagne.layers.LSTMLayer(l_forward, N_HIDDEN, backwards=False, learn_init = True, peepholes=True)
    l_forward3 = lasagne.layers.LSTMLayer(l_forward2, N_HIDDEN, backwards=False, learn_init = True, peepholes=True)
    # reshape by collapsing batch and seq_length dimensions into a single dim for non-recurrent layers
    l_reshape = lasagne.layers.ReshapeLayer(
        l_forward3, (batch_size*seq_length, N_HIDDEN))
    # Our output layer is a simple dense connection, with 1 output unit
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=N_FEAT_DIM, nonlinearity=lasagne.nonlinearities.softmax)
#    l_softmax = lasagne.layers.NonlinearityLayer(l_recurrent_out, nonlinearity=lasagne.nonlinearities.softmax)
    # Now, reshape the output back to the input format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_length, N_FEAT_DIM))

    return l_out


def main():
    print "Building network ..."
    l_out = build_network(N_BATCH)
    read_model_data(l_out, 'lstm_iter_60000')
    print "Done building network"

    target_values = T.tensor3('target_output')
    input_values = T.tensor3('input')

    network_output = lasagne.layers.get_output(l_out, input_values)

    # categorical crossentropy loss because it's the proper way
    cost = T.mean(categorical_crossentropy(T.reshape(network_output, (N_BATCH*MAX_LENGTH, N_FEAT_DIM)) , T.reshape(target_values, (N_BATCH*MAX_LENGTH, N_FEAT_DIM))))
    all_params = lasagne.layers.get_all_params(l_out)
    print "Computing updates..."
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    print "Compiling functions..."

    train = theano.function(
        [input_values, target_values], cost, updates=updates)
    compute_cost = theano.function([input_values, target_values], cost)
    train_f = open('chatlog.txt','r')
    f_data = train_f.read()
    print "Training ..."
    try:
        for n in xrange(N_ITERATIONS):
            X, Y = gen_data(f_data, n, N_BATCH, MAX_LENGTH)
            train(X, Y)
            if not n % CHECK_FREQUENCY:
                cost_val = compute_cost(X, Y)
                print "Iteration {} training cost = {}".format(n, cost_val)
            if n % CHECKPOINT_FREQUENCY == 0 and n > 0:
                print "Saving checkpoint..."
                fname = "lstm_iter_%d" % (n)
                write_model_data(l_out, fname)
        
    except KeyboardInterrupt:
        pass    

    
if __name__ == '__main__':
    main()
