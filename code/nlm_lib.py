#!/usr/bin/env python

"""
"""

debug = True

import sys
import re
import codecs

import cPickle
import random
import numpy as np
import scipy.sparse as sp
import theano
import theano.tensor as T
import theano.sparse as S
reload(sys)
sys.setdefaultencoding("utf-8")

def save_matrix(ouf, matrix, header, is_transpose=0):
  if is_transpose == 1:
    matrix = np.transpose(matrix)
  num_rows = len(matrix)
  num_cols = 1
  if matrix.ndim>1:
    num_cols = len(matrix[0, :])
  sys.stderr.write('Output matrix %d x %d %s\n' % (num_rows, num_cols, header))
  ouf.write('%s' % header)
  for ii in xrange(num_rows):
    if num_cols>1:
      for jj in xrange(num_cols):
        ouf.write('%g ' % matrix[ii, jj])
      ouf.write('\n')
    else:
      ouf.write('%g\n' % matrix[ii])

def shared_dataset(data_xy):
  data_x, data_y = data_xy
  shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
  shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
  return shared_x, shared_y 


#####################
### Softmax Layer ###
#####################
class SoftmaxLayer(object):
  """
  Adapt from this tutorial http://deeplearning.net/tutorial/logreg.html
  """
  def __init__(self, input, W, b, self_norm_coeff, is_test):
    # shared variables
    self.W = theano.shared(value=W, name='W', borrow=True)
    self.b = theano.shared(value=b, name='b', borrow=True)

    # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
    # this softmax version is better than the above, see http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
    # https://github.com/Theano/Theano/issues/1563
    x = T.dot(input, self.W) + self.b
    if is_test==1 and self_norm_coeff>0: # test with self-norm models, easy!
      self.log_p_y_given_x = x
    else:
      x_max = T.max(x, axis=1, keepdims=True) # take max for numerical stability
      if is_test == 1:
        # KNOWN PROBLEM: currently, I'm running into memory problem here, which means I can't properly test with normal models (without self-norm)
        self.log_p_y_given_x = x - T.log(T.sum(T.exp(x - x_max), axis=1, keepdims=True)) - x_max
      else:
        self.log_norm = T.log(T.sum(T.exp(x - x_max), axis=1, keepdims=True)) + x_max
        self.log_p_y_given_x = x - self.log_norm 

    # params
    self.params = [self.W, self.b]

  def nll(self, y):
    """
    Mean negative log-lilelihood
    """
    return -T.mean(self.log_p_y_given_x[T.arange(y.shape[0]), y])

  def sum_ll(self, y):
    """
    Sum log-lilelihood
    """
    return T.sum(self.log_p_y_given_x[T.arange(y.shape[0]), y])

  def ind_ll(self, y):
    """
    Individual log-lilelihood
    """
    return self.log_p_y_given_x[T.arange(y.shape[0]), y]

  
   
####################
### Hidden Layer ###
####################
class HiddenLayer(object):
  def __init__(self, rng, input, n_in, n_out, activation, W_values, b_values, dropout):
    """
    rng: random number generator
    """
    #self.input= input


    # W
    self.W = theano.shared(value=W_values, name='W', borrow=True)
    
    # b
    self.b = theano.shared(value=b_values, name='b', borrow=True)
 
    # output
    self.output = activation(T.dot(input, self.W) + self.b)

    if dropout<1.0: # dropout
      # follow: https://github.com/mdenil/dropout/blob/master/mlp.py
      srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
      self.output = self.output * T.cast(srng.binomial(n=1, p=dropout, size=self.output.shape), theano.config.floatX) / dropout


    # params
    self.params = [self.W, self.b]


####################
### Linear Layer ###
####################
class LinearLayer(object):
  def __init__(self, rng, input, emb_dim, context_size, in_vocab_size, linear_W_emb):
    #self.input= input

    # shared variables
    self.W_emb = theano.shared(value=linear_W_emb, name='W_emb', borrow=True)
 
    # stack vectors
    input = T.cast(input, 'int32')
    self.output = self.W_emb[input.flatten()].reshape((input.shape[0], input.shape[1]* emb_dim)) # self.W_emb.shape[1]

    # params
    self.params = [self.W_emb]

def init_model(rng, emb_dim, in_vocab_size, context_size, hidden_layers, n_out, init_range, activation):
  sys.stderr.write('# Init model\n')
  linear_W_emb = np.asarray(rng.uniform(low=-init_range, high=init_range, size=(in_vocab_size, emb_dim)), dtype=theano.config.floatX)

  # hidden
  hidden_in = context_size*emb_dim
  hidden_Ws = []
  hidden_bs = []
  for ii in xrange(len(hidden_layers)):
    hidden_out = hidden_layers[ii] 
    sys.stderr.write('# Hidden layer %d: in=%d, out=%d\n' % (ii, hidden_in, hidden_out))
    hidden_W = np.asarray(rng.uniform(low=-init_range, high=init_range, size=(hidden_in, hidden_out)), dtype=theano.config.floatX)
    #if activation == 'relu':
    #elif activation == 'tanh':
    #  hidden_W = np.asarray(rng.uniform(low=-np.sqrt(6. / (hidden_in + hidden_out)), high=np.sqrt(6. / (hidden_in + hidden_out)), size=(hidden_in, hidden_out)), dtype=theano.config.floatX)
    #else:
    #  sys.err.write('! Unknown activation %s\n' % activation)
    #  sys.exit(1)
    #if activation == theano.tensor.nnet.sigmoid:
    #  hidden_W *= 4
    hidden_b = np.zeros((hidden_out,), dtype=theano.config.floatX)

    hidden_Ws.append(hidden_W)
    hidden_bs.append(hidden_b)
    hidden_in = hidden_out

  
  # softmax
  softmax_in = hidden_out
  softmax_out = n_out
  #softmax_W = np.zeros((softmax_in, softmax_out), dtype=theano.config.floatX)
  softmax_W = np.asarray(rng.uniform(low=-init_range, high=init_range, size=(softmax_in, softmax_out)), dtype=theano.config.floatX)
  softmax_b = np.zeros((softmax_out,), dtype=theano.config.floatX)

  ngram_size = context_size + 1
  return (ngram_size, linear_W_emb, hidden_Ws, hidden_bs, softmax_W, softmax_b)

def save_model(file_name, classifier):
  #sys.stderr.write('  save model to %s\n' % (file_name)) 
  save_file = open(file_name, 'wb')
  cPickle.dump(classifier.ngram_size, save_file, -1) # ngram size
  cPickle.dump(classifier.num_hidden_layers, save_file, -1) # num hidden layers
  cPickle.dump(classifier.linearLayer.W_emb.get_value(), save_file, -1) # embeddings
  
  # hidden layers
  for ii in xrange(classifier.num_hidden_layers):
    cPickle.dump(classifier.hidden_layers[ii].W.get_value(), save_file, -1)
    cPickle.dump(classifier.hidden_layers[ii].b.get_value(), save_file, -1)
  
  # softmax
  cPickle.dump(classifier.softmaxLayer.W.get_value(), save_file, -1)
  cPickle.dump(classifier.softmaxLayer.b.get_value(), save_file, -1)
  save_file.close()

def save_model_config(file_name, config_map):
  #sys.stderr.write('  save model config to %s\n' % (file_name)) 
  f = open(file_name, 'w')
  for key in config_map:
    f.write('%s=%s\n' % (key, str(config_map[key])))
  f.close()

def print_matrix_stat(W, label):
  if W.ndim==1:
    num_rows = W.shape[0]
    num_cols = 1
  else:
    (num_rows, num_cols) = W.shape
    
  sys.stderr.write('%s [%d, %d]: min=%g, max=%g, avg=%g\n' % (label, num_rows, num_cols, W.min(), W.max(), W.mean()))
  #print(W)

def load_model(model_file):
  sys.stderr.write('# Loading model from %s ...\n' % model_file)
  f = file(model_file, 'rb')
  ngram_size = cPickle.load(f)
  num_hidden_layers = cPickle.load(f)
  sys.stderr.write('  ngram_size=%d\n' % ngram_size)
  sys.stderr.write('  num_hidden_layers=%d\n' % num_hidden_layers)
  
  linear_W_emb = cPickle.load(f)
  print_matrix_stat(linear_W_emb, '  W_emb')

  hidden_Ws = []
  hidden_bs = []
  for ii in xrange(num_hidden_layers):
    hidden_W = cPickle.load(f)
    hidden_b = cPickle.load(f)
    hidden_Ws.append(hidden_W)
    hidden_bs.append(hidden_b)
    print_matrix_stat(hidden_W, '  hidden_W_' + str(ii))
    print_matrix_stat(hidden_b, '  hidden_b_' + str(ii))

  softmax_W = cPickle.load(f)
  softmax_b = cPickle.load(f)
  print_matrix_stat(softmax_W, '  softmax_W ')
  print_matrix_stat(softmax_b, '  softmax_b ')
  f.close()

  return (ngram_size, linear_W_emb, hidden_Ws, hidden_bs, softmax_W, softmax_b)

def load_model_config(file_name):
  sys.stderr.write('# Loading model config from %s ...\n' % file_name)
  config_map = {}
  f = open(file_name, 'r')
  for line in f:
    tokens = re.split('=', line.strip())
    assert(len(tokens)==2)
    config_map[tokens[0]] = tokens[1]
  f.close()
  return config_map 

class NLM(object):
  def __init__(self, rng, input, model_params, self_norm_coeff, activation, dropout, is_test):
    (self.ngram_size, linear_W_emb, hidden_Ws, hidden_bs, softmax_W, softmax_b) = model_params

    (in_vocab_size, emb_dim) = linear_W_emb.shape
    (softmax_in, softmax_out) = softmax_W.shape
    context_size = self.ngram_size-1
    
    self.emb_dim = emb_dim
    self.in_vocab_size = in_vocab_size
    
    # linear embeding layer
    sys.stderr.write('# linear layer: in_vocab_size=%d, emb_dim=%d, context_size=%d\n' % (in_vocab_size, emb_dim, context_size))
    self.linearLayer = LinearLayer(rng, input, emb_dim, context_size, in_vocab_size, linear_W_emb)

    # hidden layers
    self.hidden_layers = []
    cur_hidden_in = emb_dim*context_size
    self.num_hidden_layers = len(hidden_Ws)
    sys.stderr.write('# hidden layers=%d\n' % self.num_hidden_layers)
    hidden_params = []
    prev_layer = self.linearLayer
    for ii in xrange(self.num_hidden_layers):
      hidden_W = hidden_Ws[ii]
      hidden_b = hidden_bs[ii]
      (hidden_in, hidden_out) = hidden_W.shape
      assert cur_hidden_in==hidden_in, '! hidden layer %d: cur_hidden_in %d != hidden_in %d\n' % (ii+1, cur_hidden_in, hidden_in)

      sys.stderr.write('  hidden layer %d: hidden_in=%d, hidden_out=%d\n' % (ii+1, hidden_in, hidden_out))
      hidden_layer = HiddenLayer(rng, prev_layer.output, hidden_in, hidden_out, activation, hidden_W, hidden_b, dropout)
      self.hidden_layers.append(hidden_layer)
      hidden_params = hidden_params + hidden_layer.params

      cur_hidden_in = hidden_out
      prev_layer = hidden_layer

    # softmax
    assert cur_hidden_in==softmax_in, '! softmax layer: cur_hidden_in %d != softmax_in %d\n' % (ii+1, cur_hidden_in, softmax_in)
    sys.stderr.write('# softmax layer: softmax_in=%d, softmax_out=%d\n' % (softmax_in, softmax_out))
    self.softmaxLayer = SoftmaxLayer(self.hidden_layers[self.num_hidden_layers-1].output, softmax_W, softmax_b, self_norm_coeff, is_test)
    
    # L1
    #self.L1 = abs(self.hidden_layer.W).sum() + abs(self.softmaxLayer.W).sum()

    # L2
    #self.L2 = (self.hidden_layer.W ** 2).sum() + (self.softmaxLayer.W ** 2).sum()

    # nll
    self.nll = self.softmaxLayer.nll

    # sum_ll
    self.sum_ll = self.softmaxLayer.sum_ll

    # sum_ll
    if is_test==1:
      self.ind_ll = self.softmaxLayer.ind_ll


    if is_test==0 and self_norm_coeff > 0:
      self.mean_abs_log_norm = T.mean(T.abs_(self.softmaxLayer.log_norm)) # to observe how much we compressed log |Z(x)|
      self.mean_square_log_norm = T.mean(self.softmaxLayer.log_norm ** 2) # for cost function (log Z(x))^2

    # params
    self.params = self.linearLayer.params + hidden_params + self.softmaxLayer.params

def rectifier(x):
  return x*(x>0)

def leaky_rect(x):
  return x*(x>0) + 0.01*x*(x<0)

def build_nlm_model(rng, model_params, self_norm_coeff, act_func, dropout, is_test=0):
  """
  Adapt from this tutorial http://deeplearning.net/tutorial/mlp.html 
  """
  # symbolic variables
  x = T.matrix('x')
  y = T.ivector('y') # GPU stores values in float32, so now we have to convert to int32
  lr = T.scalar('lr')

  # classsifier
  if act_func == 'tanh':
    sys.stderr.write('# act_func=tanh\n')
    activation = T.tanh 
  elif act_func == 'relu':
    sys.stderr.write('# act_func=rectifier\n')
    activation = rectifier
  elif act_func == 'leakyrelu':
    sys.stderr.write('# act_func=leaky rectifier\n')
    activation = leaky_rect 
  else:
    sys.stderr.write('! Unknown activation function %s, not tanh or relu\n' % (act_func))
    sys.exit(1)
  
  sys.stderr.write('# self_norm_coeff=%f\n' % self_norm_coeff)

  classifier = NLM(rng, x, model_params, self_norm_coeff, activation, dropout, is_test)

  if is_test==1:
    return (classifier, x, y)

  # cost
  cost = classifier.nll(y)
  if self_norm_coeff > 0:
    cost = cost + self_norm_coeff * classifier.mean_square_log_norm
    mean_abs_log_norm = classifier.mean_abs_log_norm

  # grad
  gparams = []
  #clip_range = 0.1
  grad_norm = 0.0
  for param in classifier.params:
    gparam = T.grad(cost, param)
    grad_norm += (gparam ** 2).sum()
    #gparam = T.clip(T.grad(cost, param), -clip_range, clip_range) # clip gradients 
    gparams.append(gparam)
  grad_norm = T.sqrt(grad_norm)

  # grad norm is small overall
  #max_grad_norm = 5 
  #if T.gt(grad_norm, max_grad_norm):
  #  lr = lr * max_grad_norm / grad_norm

  # update
  updates = []
  for param, gparam in zip(classifier.params, gparams):
    updates.append((param, param - lr * gparam))
  
  if self_norm_coeff > 0:
    return (classifier, x, y, lr, cost, grad_norm, mean_abs_log_norm, updates)
  else:
    return (classifier, x, y, lr, cost, grad_norm, updates)


