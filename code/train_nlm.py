#!/usr/bin/env python

"""
"""

usage = 'To train NLMs using Theano'

import cPickle
import gzip
import os
import sys
import time
import re
import codecs
import argparse
import datetime

import numpy as np
import theano
import theano.tensor as T

# our libs
import text_lib
import nlm_lib

reload(sys)
sys.setdefaultencoding("utf-8")

def gpu_check():
  vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
  iters = 1000

  rng = np.random.RandomState(22)
  x = theano.shared(np.asarray(rng.rand(vlen), theano.config.floatX))
  f = theano.function([], T.exp(x))
  t0 = time.time()
  for i in xrange(iters):
      r = f()
  t1 = time.time()
  print '# GPU check'
  print '  function: ' + str(f.maker.fgraph.toposort())
  print '  Looping %d times took' % iters, t1 - t0, 'seconds'
  print '  Result is', r
  if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
      print '  Used the cpu'
  else:
      print '  Used the gpu'

def print_dict(dict):
  for key in dict.keys():
    print('  ' + str(key) + '=' + str(dict[key]))

def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """
  
  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('train_file', metavar='train_file', type=str, help='train file') 
  parser.add_argument('valid_file', metavar='valid_file', type=str, help='valid file') 
  parser.add_argument('test_file', metavar='test_file', type=str, help='test file') 
  parser.add_argument('ngram_size', metavar='ngram_size', type=int, help='tgt ngram size') 
  parser.add_argument('vocab_size', metavar='vocab_size', type=int, help='vocab size') 
  parser.add_argument('out_prefix', metavar='out_prefix', type=str, help='output prefix') 

  # optional arguments
  parser.add_argument('--model_file', dest='model_file', type=str, default='', help='load model from a file (default=\'\')')
  parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=128, help='embedding dimension (default=128)')
  parser.add_argument('--hidden_layers', dest='hidden_layers', type=str, default='512', help='hidden layers, e.g. 512-512 (default=512)')
  parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.1, help='learning rate (default=0.1)')
  parser.add_argument('--init_range', dest='init_range', type=float, default=0.01, help='init range (default=0.01)')
  parser.add_argument('--chunk', dest='chunk', type=int, default=2000, help='each time consider batch_size*chunk ngrams (default=2000)')
  parser.add_argument('--log_freq', dest='log_freq', type=int, default=1000, help='log freq (default=1000)')
  parser.add_argument('--option', dest='opt', type=int, default=0, help='option: 0 -- predict last word, 1 -- predict middle word (default=0)')
  parser.add_argument('--min_sent_len', dest='min_sent_len', type=int, default=3, help='sentences with length < min_sent_len will be skipped (default=3)')
  parser.add_argument('--act_func', dest='act_func', type=str, default='relu', help='non-linear function: \'tanh\' or \'relu\' (default=\'relu\')')
  parser.add_argument('--self_norm_coeff', dest='self_norm_coeff', type=float, default=0, help='self normalization coefficient (default: 0, i.e. disabled)') 
  parser.add_argument('--finetune', dest='finetune', type=int, default=1, help='after training for this number of epoches, start halving learning rate(default: 1)') 
  parser.add_argument('--epoch', dest='epoch', type=int, default=4, help='Number of training epochs (default=4)')
  parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=1, help='start epoch (default=1)')
  parser.add_argument('--start_iter', dest='start_iter', type=int, default=0, help='start iter (default=0)')
  
  parser.add_argument('--dropout', dest='dropout', type=float, default=1, help='prob of keeping a unit (default: 1.0, i.e. no dropout)') 
  #parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed (default=-1, i.e., based on clock time)')
  #parser.add_argument('--dropout_opt', dest='dropout_opt', type=int, default=0, help='1 -- not dropout input layer (default: 0, if dropout, dropout input layer as well)') 
  
  # joint model
  parser.add_argument('--joint', dest='is_joint', action='store_true', default=False, help='to enable training joint model, we assume there exists files train_file.src_lang|tgt_lang|align; similarly, for valid_file and test_file (default=False)')
  parser.add_argument('--src_window', dest='src_window', type=int, default=5, help='src window for joint model (default=5)')
  parser.add_argument('--src_lang', dest='src_lang', type=str, default='', help='src lang (default=\'\')')
  parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='', help='tgt_lang (default=\'\')')
  
  args = parser.parse_args()
  return args

def check_dir(out_file):
  dir_name = os.path.dirname(out_file)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def eval(model, num_ngrams, batch_size, num_batches):
  """
  Return average negative log-likelihood
  """
  loss = 0.0
  for i in xrange(num_batches):
    start_id = i*batch_size
    end_id = (i+1)*batch_size if i<(num_batches-1) else num_ngrams
    loss -= model(start_id, end_id) # model returns sum log likelihood
  loss /= num_ngrams
  perp = np.exp(loss)
  return (loss, perp)

def get_datetime():
  return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def train(train_file, model_file, start_config_map, is_joint, data, model, batch_size, opt, min_sent_len, log_freq, self_norm_coeff, finetune_epoch, chunk_size, n_epochs, is_shuffle=True):
  valid_freq = log_freq * 10
  if self_norm_coeff > 0:
    (classifier, x, y, lr, cost, grad_norm, mean_abs_log_norm, updates) = model
  else:
    (classifier, x, y, lr, cost, grad_norm, updates) = model
  default_learning_rate = float(start_config_map['lr'])

  # config file
  config_file = model_file + '.config'
  cur_model_file = model_file + '.cur'
  cur_config_file = cur_model_file + '.config'
  config_map = {}
  if is_joint==True:
    (src_lang, src_vocab_map, src_words, src_vocab_size, tgt_lang, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, valid_set_x, valid_set_y, test_set_x, test_set_y) = data
    src_f = codecs.open(train_file + '.' + src_lang, 'r', 'utf-8')
    tgt_f = codecs.open(train_file + '.' + tgt_lang, 'r', 'utf-8')
    align_f = codecs.open(train_file + '.align', 'r', 'utf-8')
    config_map['src_window'] = src_window
    config_map['src_vocab_size'] = src_vocab_size
    config_map['tgt_vocab_size'] = tgt_vocab_size
  else:
    (vocab_map, words, vocab_size, ngram_size, valid_set_x, valid_set_y, test_set_x, test_set_y) = data
    config_map['vocab_size'] = vocab_size
    f = codecs.open(train_file, 'r', 'utf-8')

  config_map['ngram_size'] = ngram_size
  config_map['self_norm_coeff'] = self_norm_coeff
  config_map['lr'] = default_learning_rate 
  config_map['model_file'] = model_file
  config_map['cur_model_file'] = cur_model_file
  config_map['config_file'] = config_file
  config_map['cur_config_file'] = cur_config_file

  # get some training data first
  if is_joint==True:
    (data_x, data_y) = text_lib.get_joint_ngrams(src_f, tgt_f, align_f, src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, chunk_size, shuffle=is_shuffle)
  else:
    (data_x, data_y) = text_lib.get_ngrams(f, vocab_map, words, vocab_size, ngram_size, opt, num_read_lines=chunk_size, shuffle=is_shuffle)
  train_set_x, train_set_y = nlm_lib.shared_dataset([data_x, data_y])
  int_train_set_y = T.cast(train_set_y, 'int32') 

  num_valid_ngrams = valid_set_x.get_value(borrow=True).shape[0]  
  num_valid_batches = (num_valid_ngrams-1)/ batch_size + 1
  num_test_ngrams = test_set_x.get_value(borrow=True).shape[0]
  num_test_batches = (num_test_ngrams-1)/ batch_size + 1
  sys.stderr.write('# Valid: num ngrams=%d, num batches=%d\n' % (num_valid_ngrams, num_valid_batches)) 
  sys.stderr.write('# Test: num ngrams=%d, num batches=%d\n' % (num_test_ngrams, num_test_batches))     
  
  sys.stderr.write('# Batch size=%d\n' % (batch_size)) 
  sys.stderr.write('# Chunk size=%d\n' % (chunk_size)) 
  sys.stderr.write('# Learning rate=%f\n' % (config_map['lr'])) 
  sys.stderr.write('# Theano configs: device=%s, floatX=%s\n' % (theano.config.device, theano.config.floatX))

  sys.stderr.write('# Compiling function %s ...\n' % get_datetime())
  start_index = T.iscalar()
  end_index = T.iscalar()
  learning_rate = T.scalar(dtype=theano.config.floatX)
  
  # train
  if config_map['self_norm_coeff']>0:
    train_outputs = [cost, grad_norm, mean_abs_log_norm]
  else:
    train_outputs = [cost, grad_norm]

  train_model = theano.function(inputs=[start_index, end_index, learning_rate], outputs=train_outputs, updates=updates,
      givens={
        x: train_set_x[start_index:end_index],
        y: int_train_set_y[start_index:end_index],
        lr: learning_rate})

  # test
  test_model = theano.function(inputs=[start_index, end_index], outputs=classifier.sum_ll(y), 
      givens={
        x: test_set_x[start_index:end_index],
        y: test_set_y[start_index:end_index]})
  
  # valid
  valid_model = theano.function(inputs=[start_index, end_index], outputs=classifier.sum_ll(y), 
      givens={
        x: valid_set_x[start_index:end_index],
        y: valid_set_y[start_index:end_index]})
  
  ### RESUME TRAINING ###
  iter = 0
  epoch = int(start_config_map['epoch']) if 'epoch' in start_config_map else 1
  start_iter = int(start_config_map['iter']) if 'iter' in start_config_map else 0 
  config_map['best_valid_perp'] = float(start_config_map['best_valid_perp']) if 'best_valid_perp' in start_config_map else np.inf
  config_map['test_score'] = float(start_config_map['test_score']) if 'test_score' in start_config_map else 0
  train_batches_epoch = int(start_config_map['train_batches_epoch']) if 'train_batches_epoch' in start_config_map else 0 # num training batches per epoch, we only know after the first epoch
  config_map['train_batches_epoch'] = train_batches_epoch
  if train_batches_epoch>0: # already train for more than an epoch
    assert epoch>1
    iter = (epoch-1)*train_batches_epoch
  if epoch>1:
    assert train_batches_epoch>0

  ### TRAIN MODEL ###
  # finetuning
  finetune_fraction = 0.5 # the fraction of an epoch that we halve our learning rate
  assert finetune_epoch >= 1

  best_params = None
  start_time = time.time()
  done_looping = False
  iter_start_time = time.time()
  train_costs = [] # accumulate log_freq costs
  grad_norms = []
  self_norm_abs_values = []
  sys.stderr.write('# Start training model %s ...\n' % get_datetime())
  while (epoch <= n_epochs) and (not done_looping):
    while(True):
      num_train_ngrams = train_set_x.get_value(borrow=True).shape[0]
      num_train_batches = (num_train_ngrams-1) / batch_size + 1
      if epoch==1: train_batches_epoch += num_train_batches

      # train
      for i in xrange(num_train_batches):
        iter += 1
        if iter<start_iter: # skip until we reach the saved point
          continue
        
        start_id = i*batch_size
        end_id = (i+1)*batch_size if i<(num_train_batches-1) else num_train_ngrams
        outputs = train_model(start_id, end_id, config_map['lr'])

        # check for nan/inf and print out debug infos
        if np.isnan(outputs[0]) or np.isinf(outputs[0]):
          sys.stderr.write('! epoch %d, iter %d: nan or inf, bad ... Dummping out training data\n' % (epoch, iter))
          for ngram_id in xrange(start_id, end_id):
            sys.stderr.write('%d: ' % ngram_id)
            if is_joint==True:
              text_lib.print_joint_ngram(data_x[ngram_id], data_y[ngram_id], src_window, src_words, tgt_words, tgt_vocab_size)
            else:
              text_lib.print_ngram(data_x[ngram_id], data_y[ngram_id], words)
          sys.exit(1)
          
        train_costs.append(outputs[0])
        grad_norms.append(outputs[1])
        if config_map['self_norm_coeff']>0: self_norm_abs_values.append(outputs[2])
       
        # finetuning
        if epoch > finetune_epoch and iter % (train_batches_epoch*finetune_fraction) == 0:
          sys.stderr.write('# epoch %d, iter %d: halving learning rate from %g to %g\n' % (epoch, iter, config_map['lr'], config_map['lr']/2))
          config_map['lr'] = config_map['lr']/2

        # logging
        if iter % log_freq == 0:
          iter_end_time = time.time()
          config_map = logTrainInfo(epoch, iter, train_costs, grad_norms, self_norm_abs_values, iter_end_time, iter_start_time, config_map)

          # eval on valid/test data
          if iter % valid_freq == 0:
            config_map = evalValidTest(classifier, epoch, iter, valid_model, num_valid_ngrams, num_valid_batches, test_model, num_test_ngrams, num_test_batches, batch_size, config_map)
            iter_end_time = time.time()

          # update
          iter_start_time = time.time()

      if done_looping == True: break

      # read more data
      if is_joint==True:
        (data_x, data_y) = text_lib.get_joint_ngrams(src_f, tgt_f, align_f, src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, chunk_size, shuffle=is_shuffle)
      else:
        (data_x, data_y) = text_lib.get_ngrams(f, vocab_map, words, vocab_size, ngram_size, opt, num_read_lines=chunk_size, shuffle=is_shuffle)
      if len(data_y)==0: # eof
        break
      train_set_x.set_value(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
      train_set_y.set_value(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
      int_train_set_y = T.cast(train_set_y, 'int32') 
   
    # end an epoch 
    sys.stderr.write('# Done epoch %d, iter %d, %s\n' % (epoch, iter, get_datetime()))

    epoch = epoch + 1
    if epoch==2: # done with the first epoch
      config_map['train_batches_epoch'] = train_batches_epoch
      sys.stderr.write('  Num training batches per epoch = %d\n' % train_batches_epoch)
      
      if log_freq>train_batches_epoch:
        sys.stderr.write('! change log_freq from %d -> %d\n' % (log_freq, train_batches_epoch))
        log_freq = train_batches_epoch
        iter_end_time = time.time()
        config_map = logTrainInfo(epoch, iter, train_costs, grad_norms, self_norm_abs_values, iter_end_time, iter_start_time, config_map)
    config_map = evalValidTest(classifier, epoch, iter, valid_model, num_valid_ngrams, num_valid_batches, test_model, num_test_ngrams, num_test_batches, batch_size, config_map)

    if is_joint==True:
      src_f.close()
      tgt_f.close()
      align_f.close()
      src_f = codecs.open(train_file + '.' + src_lang, 'r', 'utf-8')
      tgt_f = codecs.open(train_file + '.' + tgt_lang, 'r', 'utf-8')
      align_f = codecs.open(train_file + '.align', 'r', 'utf-8')
      (data_x, data_y) = text_lib.get_joint_ngrams(src_f, tgt_f, align_f, src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, chunk_size, shuffle=is_shuffle)
    else:
      f.close() 
      f = codecs.open(train_file, 'r', 'utf-8')
      (data_x, data_y) = text_lib.get_ngrams(f, vocab_map, words, vocab_size, ngram_size, opt, num_read_lines=chunk_size, shuffle=is_shuffle)

    train_set_x.set_value(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    train_set_y.set_value(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
    int_train_set_y = T.cast(train_set_y, 'int32') 
    
  if is_joint==True:
    src_f.close()
    tgt_f.close()
    align_f.close()
  else:
    f.close()
  end_time = time.time() # time.clock()

  sys.stderr.write(('# Optimization complete with best validation score of %f, with test score=%f\n') % (config_map['best_valid_perp'], config_map['test_score']))
  sys.stderr.write(('# The code ran for %.1fs\n' % ((end_time - start_time))))
  sys.stderr.write('# End training model %s\n' % get_datetime())

def logTrainInfo(epoch, iter, train_costs, grad_norms, self_norm_abs_values, iter_end_time, iter_start_time, config_map):
  config_map['train_loss'] = np.mean(train_costs)
  config_map['avg_grad_norm'] = np.mean(grad_norms)
  config_map['speed'] = log_freq*batch_size*0.001/(iter_end_time-iter_start_time)

  if config_map['self_norm_coeff']>0:
    config_map['avg_self_norm_abs'] = np.mean(self_norm_abs_values)
    sys.stderr.write('%d, %d, %g, %.2fK, %s: train_loss=%.2f, grad_norm=%.2f, log|Z|=%.2f\n' % (epoch, iter, config_map['lr'], config_map['speed'], get_datetime(), config_map['train_loss'], config_map['avg_grad_norm'], config_map['avg_self_norm_abs']))
  else:
    sys.stderr.write('%d, %d, %g, %.2fK, %s: train_loss=%.2f, grad_norm=%.2f\n' % (epoch, iter, config_map['lr'], config_map['speed'], get_datetime(), config_map['train_loss'], config_map['avg_grad_norm']))
  return config_map

def evalValidTest(classifier, epoch, iter, valid_model, num_valid_ngrams, num_valid_batches, test_model, num_test_ngrams, num_test_batches, batch_size, config_map):
  # valid
  (valid_loss, valid_perp) = eval(valid_model, num_valid_ngrams, batch_size, num_valid_batches)

  # test
  (test_loss, test_perp) = eval(test_model, num_test_ngrams, batch_size, num_test_batches)
  
  if config_map['self_norm_coeff']>0:
    sys.stderr.write('eval %.2f, %.2f, %d, %d, %g, %.2fK, %s: train_loss=%.2f, grad_norm=%.2f, log|Z|=%.2f, valid_perp=%.2f (%.2f), test_perp=%.2f (%.2f)\n' % (test_perp, config_map['avg_self_norm_abs'], epoch, iter, config_map['lr'], config_map['speed'], get_datetime(), config_map['train_loss'], config_map['avg_grad_norm'], config_map['avg_self_norm_abs'], valid_perp, valid_loss, test_perp, test_loss))
  else:
    sys.stderr.write('eval %.2f, %d, %d, %g, %.2fK, %s: train_loss=%.2f, grad_norm=%.2f, valid_perp=%.2f (%.2f), test_perp=%.2f (%.2f)\n' % (test_perp, epoch, iter, config_map['lr'], config_map['speed'], get_datetime(), config_map['train_loss'], config_map['avg_grad_norm'], valid_perp, valid_loss, test_perp, test_loss))

  config_map['epoch'] = epoch
  config_map['iter'] = iter
  config_map['test_perp'] = test_perp
  nlm_lib.save_model(config_map['cur_model_file'], classifier)
  nlm_lib.save_model_config(config_map['cur_config_file'], config_map)
  if valid_perp < config_map['best_valid_perp']: # found a better model 
    config_map['test_score'] = test_perp 
    config_map['best_valid_perp'] = valid_perp
    sys.stderr.write('  new best valid perp=%f, test_score=%f\n' % (config_map['best_valid_perp'], config_map['test_score']))
    
    # save
    nlm_lib.save_model(config_map['model_file'], classifier)
    nlm_lib.save_model_config(config_map['config_file'], config_map)

  return config_map

if __name__ == '__main__':
  args = process_command_line()
 
  print('# Program arguments: ')
  print_dict(vars(args))
  gpu_check()

  ## args ##
  init_range = args.init_range #0.01
  batch_size = 128 
  emb_dim = args.emb_dim #128
  hidden_layers = [int(x) for x in re.split('-', args.hidden_layers)]
  learning_rate = args.learning_rate
  ngram_size = args.ngram_size 
  log_freq = args.log_freq
  opt = args.opt
  min_sent_len = args.min_sent_len
  chunk_size = batch_size*args.chunk
  model_file = args.model_file
  self_norm_coeff = args.self_norm_coeff
  act_func = args.act_func
  src_lang = args.src_lang
  tgt_lang = args.tgt_lang
  is_joint = args.is_joint
  num_epochs = args.epoch
  finetune_epoch = args.finetune # after training for this number of epoches, start halving learning rate
  dropout = args.dropout

  # check dir
  dir_name = os.path.dirname(args.out_prefix)
  sys.stderr.write('# dir = %s\n' % dir_name)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    sys.stderr.write('  not exists. created.\n')
 
  ## data processing ##
  if is_joint==True: # joint model
    src_window = args.src_window
    assert src_lang!='' and tgt_lang!=''
    sys.stderr.write('## Train joint model with src_lang=%s and tgt_lang=%s\n' % (src_lang, tgt_lang))
 
    # load vocab
    src_train_file = args.train_file + '.' + src_lang
    src_vocab_file = src_train_file + '.vocab.' + str(args.vocab_size)
    tgt_train_file = args.train_file + '.' + tgt_lang
    tgt_vocab_file = tgt_train_file + '.vocab.' + str(args.vocab_size)
    (src_words, src_vocab_map, src_vocab_size) = text_lib.get_vocab(src_train_file, src_vocab_file, -1, args.vocab_size)
    (tgt_words, tgt_vocab_map, tgt_vocab_size) = text_lib.get_vocab(tgt_train_file, tgt_vocab_file, -1, args.vocab_size)

     # load valid/test ngrams
    (valid_set_x, valid_set_y) = text_lib.get_all_joint_ngrams(args.valid_file + '.' + src_lang, args.valid_file + '.' + tgt_lang, args.valid_file + '.align', src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, -1)
    (test_set_x, test_set_y) = text_lib.get_all_joint_ngrams(args.test_file + '.' + src_lang, args.test_file + '.' + tgt_lang, args.test_file + '.align', src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, -1)
  else:
    sys.stderr.write('## Train monolingual model with tgt_lang=%s\n' % (tgt_lang))
    
    # load vocab
    vocab_file = args.train_file + '.vocab.' + str(args.vocab_size)
    (words, vocab_map, vocab_size) = text_lib.get_vocab(args.train_file, vocab_file, -1, args.vocab_size)

     # load valid/test ngrams
    (valid_set_x, valid_set_y) = text_lib.get_all_ngrams(args.valid_file, vocab_map, words, vocab_size, ngram_size, opt)
    (test_set_x, test_set_y) = text_lib.get_all_ngrams(args.test_file, vocab_map, words, vocab_size, ngram_size, opt)

  valid_set_x, valid_set_y = nlm_lib.shared_dataset([valid_set_x, valid_set_y])
  test_set_x, test_set_y = nlm_lib.shared_dataset([test_set_x, test_set_y])
  valid_set_y = T.cast(valid_set_y, 'int32') 
  test_set_y = T.cast(test_set_y, 'int32')

  if is_joint==True:
    data = (src_lang, src_vocab_map, src_words, src_vocab_size, tgt_lang, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, valid_set_x, valid_set_y, test_set_x, test_set_y)
    text_lib.write_vocab(args.out_prefix + '.vocab.' + src_lang, src_words)
    text_lib.write_vocab(args.out_prefix + '.vocab.' + tgt_lang, tgt_words)
    context_size = 2*src_window + ngram_size
    in_vocab_size = src_vocab_size + tgt_vocab_size
    out_vocab_size = tgt_vocab_size
  else:
    data = (vocab_map, words, vocab_size, ngram_size, valid_set_x, valid_set_y, test_set_x, test_set_y)
    text_lib.write_vocab(args.out_prefix + '.vocab', words)
    context_size = ngram_size - 1
    in_vocab_size = vocab_size
    out_vocab_size = vocab_size

  ## build model ##
  rng = np.random.RandomState()
  sys.stderr.write('# rng = %s\n' % str(rng))

  if model_file == '':
    model_file = args.out_prefix + '.model'
     
    is_bootstrap = 0 
  else:
    is_bootstrap = 1
  cur_model_file = model_file + '.cur'

  config_map = {'epoch' : args.start_epoch, 'iter' : args.start_iter, 'hidden_layers' : args.hidden_layers, 'emb_dim': args.emb_dim, 'lr': learning_rate}
  if os.path.isfile(cur_model_file)==True: # cur model file exists
    model_params = nlm_lib.load_model(cur_model_file)
    config_file = cur_model_file + '.config'
    if is_bootstrap==0 and os.path.isfile(config_file)==True:
      config_map = nlm_lib.load_model_config(config_file)
  elif os.path.isfile(model_file)==True: # model file exists
    model_params = nlm_lib.load_model(model_file)
    config_file = model_file + '.config'
    if is_bootstrap==0 and os.path.isfile(config_file)==True:
      config_map = nlm_lib.load_model_config(config_file)
  else: # init from scratch
    sys.stderr.write('! Model file %s does not exist\n' % model_file)
    model_params = nlm_lib.init_model(rng, emb_dim, in_vocab_size, context_size, hidden_layers, out_vocab_size, init_range, act_func)

  sys.stderr.write('# Config_map: %s\n' % str(config_map))

  model = nlm_lib.build_nlm_model(rng, model_params, self_norm_coeff, act_func, dropout)
 
  ## train ##
  out_model_file = args.out_prefix + '.model'
  train(args.train_file, out_model_file, config_map, is_joint, data, model, batch_size, opt, min_sent_len, log_freq, self_norm_coeff, finetune_epoch, chunk_size, num_epochs)
  #if seed == -1:
  #else:
  #  sys.stderr.write('# Fixed random seed = %d\n' % seed)
  #  rng = np.random.RandomState(seed)

