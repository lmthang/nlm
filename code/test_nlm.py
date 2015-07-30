#!/usr/bin/env python

"""
"""

debug = True
usage = 'To test NLMs using Theano'

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

def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """
  
  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('model_file', metavar='model_file', type=str, help='model file') 
  parser.add_argument('vocab_file', metavar='vocab_file', type=str, help='vocab file') 
  parser.add_argument('test_file', metavar='test_file', type=str, help='test file') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output prefix') 

  # optional arguments
  parser.add_argument('--batch', dest='batch_size', type=int, default=2048, help='batch size (default=2048)')
  parser.add_argument('--act_func', dest='act_func', type=str, default='tanh', help='non-linear function: \'tanh\' or \'relu\' (default=\'tanh\')')
  parser.add_argument('--self_norm_coeff', dest='self_norm_coeff', type=float, default=0, help='self normalization coefficient (default: 0, i.e. disabled)') 
  parser.add_argument('--option', dest='opt', type=int, default=0, help='option: 0 -- predict last word, 1 -- predict middle word (default=0)')
  parser.add_argument('--nbest', dest='is_nbest', action='store_true', help='input file is an nbest file')
 
  # joint model
  parser.add_argument('--joint', dest='is_joint', action='store_true', default=False, help='to enable testing joint model. We expect vocab files $vocab_file.$src_lang and $vocab_file.$tgt_lang. We also assume the input file contains the translation history information at the end in the following format: ||| source1 <r> target1 <r> sourcePosition1 <r> alignment1 |R| source2 <r> target2 <r> sourcePosition2 <r> alignment2 ...')
  parser.add_argument('--src_window', dest='src_window', type=int, default=5, help='src window for joint model (default=5)')
  parser.add_argument('--src_lang', dest='src_lang', type=str, default='', help='src lang (default=\'\')')
  parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='', help='tgt_lang (default=\'\')')
  parser.add_argument('--src_file', dest='src_file', type=str, default='', help='src file, source sentences that we translated (default=\'\')')
  parser.add_argument('--align_file', dest='align_file', type=str, default='', help='align file, to score with a joint model and when --nbest is not specified (default=\'\')')
  
  # debug
  parser.add_argument('--debug', dest='is_debug', action='store_true', default=False, help='if enabled, print out detailed ngram scores.')
  
  args = parser.parse_args()
  return args

def check_dir(out_file):
  dir_name = os.path.dirname(out_file)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def load_batch(f, vocab_map, ngram_size, opt, is_nbest, num_lines_read=10000):
  """
  Load data to test target-only models
  """
  all_data_x = []
  all_data_y = []
  starts = []
  ends = [] # exlusive
  ngram_count = 0 
  line_count = 0
  sent_id_map = {} # map sents -> 0 ... num_uniq_sents
  sent_map = {} # map sent id -> uniq sent id
  uniq_sents = [] # for debugging purpose
  num_uniq_sents = 0
  for ii in xrange(num_lines_read):
    line = f.readline().strip()
    if line == '': # eof
      break
    line = line.strip()
    if is_nbest==True: # nbest file
      tokens = re.split(' \|\|\| ', line)
      line = tokens[1]
    assert line != ''

    if line not in sent_map: # new
      sent_map[line] = num_uniq_sents
      uniq_sents.append(line)
      num_uniq_sents += 1

      # get sent ngrams
      (data_x, data_y) = text_lib.get_ngrams_line(line, vocab_map, ngram_size, opt)
      assert len(data_y)>0
      assert len(data_x) == len(data_y)
     
      # append
      starts.append(ngram_count)
      ngram_count += len(data_y)
      ends.append(ngram_count) # exclusive
      all_data_x.extend(data_x)
      all_data_y.extend(data_y)
    
    sent_id_map[line_count] = sent_map[line]
    
    line_count += 1

  assert len(uniq_sents)==num_uniq_sents
  sys.stderr.write('# get batch: line_count=%d, num_uniq_sents=%d, ngram_count=%d\n' % (line_count, num_uniq_sents, ngram_count))
  return (all_data_x, all_data_y, starts, ends, sent_id_map, line_count, num_uniq_sents, ngram_count, uniq_sents)

def process_align(align_info):
  """
  Process alignment from tgt to src, e.g., () (0,1) (2) (4)
  """
  t2s = {}
  if align_info == 'I-I': # identiy translation
    t2s[0] = [0]
  elif align_info != '':
    tokens = re.split(' ', align_info)
    for tgt_pos in xrange(len(tokens)): 
      if tokens[tgt_pos][1:-1] == '': continue
      t2s[tgt_pos] = [int(x) for x in re.split(',', tokens[tgt_pos][1:-1])]
  return t2s 
   
def append_src_info(data_x, data_y, src_tokens, src_window, src_vocab_map, src_unk_index, src_sos_index, src_eos_index, history_str):
  global debug 
  
  # now try to add src ngrams
  phrase_pairs = re.split(' \|R\| ', history_str) #tokens[-1])
  tgt_pos = 0 # pos in a sent
  for phrase_pair in phrase_pairs: # go through each phrase pair
    (src_phrase, tgt_phrase, src_pos, align_info) = re.split(' <r> ', phrase_pair)
    src_pos = int(src_pos)
    t2s = process_align(align_info)
    tgt_phrase_tokens = re.split(' ', tgt_phrase)
    tgt_phrase_len = len(tgt_phrase_tokens) 

    # debug
    #sys.stderr.write('\n# %s\n' % phrase_pair)
    src_phrase_tokens = re.split(' ', src_phrase)
    assert src_phrase_tokens[0] == src_tokens[src_pos]
    
    for tgt_phrase_pos in xrange(len(tgt_phrase_tokens)):
      src_phrase_pos = text_lib.infer_src_pos(tgt_phrase_pos, t2s, tgt_phrase_len) 
      assert src_phrase_pos != -1 # unaligned
      src_ngram = text_lib.get_src_ngram(src_pos + src_phrase_pos, src_tokens, src_window, src_vocab_map, src_unk_index, src_sos_index, src_eos_index, tgt_vocab_size)
      data_x[tgt_pos] = src_ngram + data_x[tgt_pos] 

      # debug
      #sys.stderr.write('  align \"%s\"  --  \"%s\"\n' % (src_tokens[src_pos+src_phrase_pos], tgt_tokens[tgt_pos]))
      if debug==True:
        text_lib.print_joint_ngram(data_x[tgt_pos], data_y[tgt_pos], src_window, src_words, tgt_words, tgt_vocab_size)
      
      tgt_pos += 1

  # src ngram for tgt </s>
  src_ngram = text_lib.get_src_ngram(len(src_tokens)-1, src_tokens, src_window, src_vocab_map, src_unk_index, src_sos_index, src_eos_index, tgt_vocab_size)
  data_x[tgt_pos] = src_ngram + data_x[tgt_pos]

  # debug
  if debug==True:
    text_lib.print_joint_ngram(data_x[tgt_pos], data_y[tgt_pos], src_window, src_words, tgt_words, tgt_vocab_size)

  return data_x

def load_joint_batch(f, src_sents, align_sents, src_vocab_map, src_words, tgt_vocab_map, tgt_words, src_window, tgt_ngram_size, opt, is_nbest, num_lines_read=10000, sos='<s>', eos='</s>', unk='<unk>'):
  """
  Load data to test joint models
  """
  global debug 
  all_data_x = []
  all_data_y = []
  starts = []
  ends = [] # exlusive
  ngram_count = 0 
  line_count = 0
  sent_id_map = {} # map sents -> 0 ... num_uniq_sents
  sent_map = {} # map sent id -> uniq sent id
  uniq_sents = [] # for debugging purpose
  src_uniq_sents = [] # for debugging purpose
  num_uniq_sents = 0
  
  src_unk_index = src_vocab_map[unk]
  src_sos_index = src_vocab_map[sos]
  src_eos_index = src_vocab_map[eos]

  if is_nbest==0:
    tgt_unk_index = tgt_vocab_map[unk]
    tgt_sos_index = tgt_vocab_map[sos]
    tgt_eos_index = tgt_vocab_map[eos]

    predict_pos = (tgt_ngram_size-1) # predict last word
    if opt==1: # predict middle word
      predict_pos = (tgt_ngram_size-1)/2
    start_ngram = [sos for i in xrange(predict_pos)]
    end_ngram = [eos for i in xrange(tgt_ngram_size-predict_pos)]
    assert len(start_ngram) == predict_pos
    assert len(start_ngram) + len(end_ngram) == tgt_ngram_size


  for ii in xrange(num_lines_read):
    line = f.readline().strip()
    # eof 
    if line == '': break 
   
    if is_nbest:
      tokens = re.split(' \|\|\| ', line)
      if tokens[1].strip() == '':
        sys.stderr.write('\n! Empty translation %s\n' % line)
      line = tokens[1].strip() # tgt translation
      line_id = int(tokens[0])
    else:
      line_id = line_count

    src_line = src_sents[line_id]
  
   
    if line not in sent_map: # new
      sent_map[line] = num_uniq_sents
      uniq_sents.append(line)
      src_uniq_sents.append(src_line)
      num_uniq_sents += 1

      if line != '':
        if is_nbest: # nbest
          # debug
          if debug==True:
            sys.stderr.write('src: %s\n' % src_line)
            sys.stderr.write('tgt: %s\n' % line)
          
          src_tokens = re.split(' ', src_line) # look up src sent from id
          tgt_tokens = re.split(' ', line)

           # get tgt ngrams
          (data_x, data_y) = text_lib.get_ngrams_line(line, tgt_vocab_map, tgt_ngram_size, opt)

          # add src ngrams
          append_src_info(data_x, data_y, src_tokens, src_window, src_vocab_map, src_unk_index, src_sos_index, src_eos_index, tokens[-1])
         
          line_ngram_count = len(data_y)
        else: # non-nbest
          align_line = align_sents[line_id]
          (data_x, data_y, line_ngram_count) = text_lib.get_line_joint_ngrams(src_line, line, align_line, tgt_ngram_size, src_window, src_words, src_vocab_map, src_vocab_size, tgt_words, tgt_vocab_map, tgt_vocab_size, unk, sos, eos, start_ngram, end_ngram, predict_pos, src_unk_index, src_sos_index, src_eos_index, tgt_unk_index, tgt_sos_index, tgt_eos_index, 1, debug)
        
        if debug==True:
          sys.stderr.write('  line_ngram_count=%d\n' % line_ngram_count)

        all_data_x.extend(data_x)
        all_data_y.extend(data_y)
      else:
        line_ngram_count = 0

      if line_ngram_count==0:
        if is_nbest:
          sys.stderr.write('! empty line %d:\n  src=%s\n  tgt=%s\n' % (line_count, src_line, line))
        else:
          sys.stderr.write('! empty line %d:\n  src=%s\n  tgt=%s\n  align=%s\n' % (line_count, src_line, line, align_line))

      # append
      starts.append(ngram_count)
      ngram_count += line_ngram_count
      ends.append(ngram_count) # exclusive

    else:
      if is_nbest==0:
        sys.stderr.write('! Duplicated sent: %s\n' % line)

    
    sent_id_map[line_count] = sent_map[line]
    line_count += 1

    # debug
    if debug==True:
      debug = False


  sys.stderr.write('# get batch: line_count=%d, num_uniq_sents=%d, ngram_count=%d\n' % (line_count, num_uniq_sents, ngram_count))
  return (all_data_x, all_data_y, starts, ends, sent_id_map, line_count, num_uniq_sents, ngram_count, uniq_sents, src_uniq_sents)

def test(test_file, model, data, is_joint, out_file, opt, self_norm_coeff, is_nbest, batch_size, is_debug):
  (classifier, x, y) = model
  ngram_size = classifier.ngram_size
  chunk_size = 10000 # number of sentences processed each time
  f = codecs.open(test_file, 'r', 'utf-8')
  sys.stderr.write('# ngram_size=%d\n' % ngram_size)

  if is_joint==True:
    sys.stderr.write('# src_window=%d\n' % src_window)
    tgt_ngram_size = ngram_size - 2*src_window - 1
    assert tgt_ngram_size > 1

    (src_sents, align_sents, src_words, src_vocab_map, src_vocab_size, tgt_words, tgt_vocab_map, tgt_vocab_size) = data
    assert (src_vocab_size + tgt_vocab_size) == classifier.in_vocab_size
    
    (data_x, data_y, starts, ends, sent_id_map, num_lines, num_uniq_sents, num_total_ngrams, uniq_sents, src_uniq_sents) = load_joint_batch(f, src_sents, align_sents, src_vocab_map, src_words, tgt_vocab_map, tgt_words, src_window, tgt_ngram_size, opt, is_nbest, chunk_size)
  else:
    (words, vocab_map, vocab_size) = data 
    assert(vocab_size == classifier.in_vocab_size)
    (data_x, data_y, starts, ends, sent_id_map, num_lines, num_uniq_sents, num_total_ngrams, uniq_sents) = load_batch(f, vocab_map, ngram_size, opt, is_nbest, chunk_size)
  
  ngram_count = batch_size if len(data_y)>batch_size else len(data_y) # num of ngrams we have processed
  test_set_x, test_set_y = nlm_lib.shared_dataset([data_x[0:batch_size], data_y[0:batch_size]])
  int_test_set_y = T.cast(test_set_y, 'int32') 

  sys.stderr.write('# Theano configs: device=%s, floatX=%s\n' % (theano.config.device, theano.config.floatX))
  sys.stderr.write('# Compiling function ...\n')
  
  # test
  test_model = theano.function(inputs=[], outputs=classifier.ind_ll(y), #[classifier.ind_ll(y), classifier.sum_ll(y)], 
      givens={x: test_set_x, y: int_test_set_y})
  
  sys.stderr.write('# Start testing %s ...\n' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  start_time = time.time()

  sent_id = 0
  ouf = codecs.open(out_file, 'w', 'utf-8')
  all_scores = np.asarray([])
  total_score = 0.0
  total_num_ngram = 0
  start_time = time.time()
  while(True):
    # test
    all_scores = np.append(all_scores, test_model())
    if ngram_count == num_total_ngrams: # score all ngrams, write scores
      score_map = {}
      count_map = {}

      end_time = time.time()
      sys.stderr.write('  %d ngrams, %.2fs, %.2fK wps\n' % (ngram_count, (end_time-start_time), ngram_count*0.001/(end_time-start_time)))
      start_time = time.time()

      # compute scores for unique sents
      for ii in xrange(num_uniq_sents):
        if starts[ii] == ends[ii]: # empty line
          score_map[ii] = -1000
          sys.stderr.write('! empty line %d\n' % ii)
        else:
          scores = all_scores[starts[ii]:ends[ii]]
          score_map[ii] = scores.sum()
        count_map[ii] = ends[ii]-starts[ii]

      # populate scores for all
      for ii in xrange(num_lines):
        map_id = sent_id_map[ii]
        sent_score = score_map[map_id]  
        ouf.write('%g\n' % sent_score)
        
        total_score += sent_score
        total_num_ngram += count_map[map_id]

        # debug info
        if is_debug:
          if is_joint:
            sys.stderr.write('# %s ||| %s\t%g\n' % (src_uniq_sents[map_id], uniq_sents[map_id], sent_score))
          else:
            sys.stderr.write('# %s\t%g\n' % (uniq_sents[map_id], sent_score))
          for jj in xrange(starts[map_id], ends[map_id]):
            sys.stderr.write('%g ' % all_scores[jj])
            if is_joint:
              text_lib.print_joint_ngram(data_x[jj], data_y[jj], src_window, src_words, tgt_words, tgt_vocab_size)
            else:  
              text_lib.print_ngram(data_x[jj], data_y[jj], words)
          for jj in xrange(starts[map_id], ends[map_id]):
            debug_ngram = data_x[jj]
            debug_ngram.append(data_y[jj])
            sys.stderr.write('%s\n' % (' '.join([str(x) for x in debug_ngram])))

      sent_id += num_lines 

      #if sent_id % 1000 == 0:
        #sys.stderr.write(' (%d) ' % sent_id)
        #ouf.flush()

      # read more data
      if is_joint == True:
        (data_x, data_y, starts, ends, sent_id_map, num_lines, num_uniq_sents, num_total_ngrams, uniq_sents, src_uniq_sents) = load_joint_batch(f, src_sents, align_sents, src_vocab_map, src_words, tgt_vocab_map, tgt_words, src_window, tgt_ngram_size, opt, is_nbest, chunk_size)
      else:
        (data_x, data_y, starts, ends, sent_id_map, num_lines, num_uniq_sents, num_total_ngrams, uniq_sents) = load_batch(f, vocab_map, ngram_size, opt, chunk_size)
     
      # eof 
      if num_lines==0: break
      
      # reset
      ngram_count = 0
      all_scores = []

    # get next batch of ngrams
    next_ngram_count = ngram_count + batch_size
    if next_ngram_count > num_total_ngrams:
      next_ngram_count = num_total_ngrams
    
    test_set_x.set_value(np.asarray(data_x[ngram_count:next_ngram_count], dtype=theano.config.floatX), borrow=True)
    test_set_y.set_value(np.asarray(data_y[ngram_count:next_ngram_count], dtype=theano.config.floatX), borrow=True)
    int_test_set_y = T.cast(int_test_set_y, 'int32') 
    ngram_count = next_ngram_count
  f.close()

  sys.stderr.write('# Total num ngrams = %d, total_score = %f, perplexity = %f\n' % (total_num_ngram, total_score, np.exp(-total_score/total_num_ngram)))
  end_time = time.time() # time.clock()
  sys.stderr.write(('# The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs\n' % ((end_time - start_time))))
  sys.stderr.write('# End testing model %s\n' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
  args = process_command_line()
 
  vocab_file = args.vocab_file
  model_file = args.model_file
  test_file = args.test_file
  out_file = args.out_file
  self_norm_coeff = args.self_norm_coeff
  opt = args.opt
  is_nbest = args.is_nbest
  batch_size = args.batch_size

  #L1_reg = 0.0 #args.L1_reg 
  #L2_reg = 0.0 # args.L2_reg
  act_func = args.act_func
  
  # joint model
  is_joint = args.is_joint
  if is_joint==True:
    src_window = args.src_window
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_file = args.src_file
    align_file = args.align_file
    assert src_lang!='' and tgt_lang!=''
    assert src_window>0
    assert src_file!=''
    assert is_nbest!=0 or align_file !=''
    sys.stderr.write('# Train joint model with src_lang=%s and tgt_lang=%s, src_window=%d, src_file=%s\n' % (src_lang, tgt_lang, src_window, src_file))
 
    # load vocab
    src_vocab_file = vocab_file + '.' + src_lang
    tgt_vocab_file = vocab_file + '.' + tgt_lang
    (src_words, src_vocab_map, src_vocab_size) = text_lib.load_vocab(src_vocab_file)
    (tgt_words, tgt_vocab_map, tgt_vocab_size) = text_lib.load_vocab(tgt_vocab_file)

    # load src_file
    src_sents = text_lib.load_text(src_file)

    if is_nbest==0: # non-nbest
      align_sents = text_lib.load_text(align_file)
    else: # nbest file
      align_sents = []

    data = (src_sents, align_sents, src_words, src_vocab_map, src_vocab_size, tgt_words, tgt_vocab_map, tgt_vocab_size)
  else:
    # load vocab
    (words, vocab_map, vocab_size) = text_lib.load_vocab(vocab_file)
    data = (words, vocab_map, vocab_size)

  # load model
  rng = np.random.RandomState(1234)
  model_params = nlm_lib.load_model(model_file)
  is_test = 1
  dropout = 1.0
  model = nlm_lib.build_nlm_model(rng, model_params, self_norm_coeff, act_func, dropout, is_test) # , L1_reg, L2_reg
  
  # test 
  test(test_file, model, data, is_joint, out_file, opt, self_norm_coeff, is_nbest, batch_size, args.is_debug)

