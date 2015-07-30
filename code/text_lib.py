"""
"""

debug = True

import os
import sys
import re
import codecs

import random

def aggregate_alignments(align_line):
  align_tokens = re.split('\s+', align_line.strip())
  s2t = {}
  t2s = {}
  # process alignments
  for align_token in align_tokens:
    if align_token=='': continue
    (src_pos, tgt_pos) = re.split('\-', align_token)
    src_pos = int(src_pos)
    tgt_pos = int(tgt_pos)
    if src_pos not in s2t: s2t[src_pos] = []
    s2t[src_pos].append(tgt_pos)

    if tgt_pos not in t2s: t2s[tgt_pos] = []
    t2s[tgt_pos].append(src_pos)
  
  return (s2t, t2s)

def look_up(token, vocab_map, unk_index):
  if token in vocab_map:
    id = vocab_map[token]
  else:
    id = unk_index
  return id

###################
### I/O & vocab ###
###################
def load_text(in_file):
  f = codecs.open(in_file, 'r', 'utf-8')
  lines = []
  for line in f:
    lines.append(line.strip())
  f.close()
  return lines

def get_vocab(corpus_file, vocab_file, freq, vocab_size, unk='<unk>'):
  if os.path.isfile(vocab_file): # vocab_file exist
    (words, vocab_map, vocab_size) = load_vocab(vocab_file)
  else:
    (words, vocab_map, freq_map, vocab_size, num_train_words, num_lines) = load_vocab_from_corpus(corpus_file, freq, vocab_size, unk)
    write_vocab(vocab_file, words)
  return (words, vocab_map, vocab_size)


def add_word_to_vocab(word, words, vocab_map, vocab_size):
  if word not in vocab_map:
    words.append(word)
    vocab_map[word] = vocab_size
    vocab_size += 1
    #sys.stderr.write('  add %s\n' % word)
  return (words, vocab_map, vocab_size)

def annotate_unk(tokens, vocab_map, eos, max_num_unks=-1, default_unk='<unk>'):
  sent_unk_map = {} # make sure same words are mapped to the same unk in each sent
  num_unk_types=0
  num_unk_tokens = 0
  unk_tokens = list(tokens)
  for pos in xrange(len(tokens)):
    token = tokens[pos]
       
    if token not in vocab_map:
      if token in sent_unk_map:
        unk = sent_unk_map[token]
      else: # generate a new unk
        if max_num_unks>0 and num_unk_types>=max_num_unks: # this sent has lots of unk, use <unk>
          unk = default_unk 
        else:
          unk = '<unk' + str(num_unk_types) + '>'
          num_unk_types += 1

        sent_unk_map[token] = unk

      unk_tokens[pos] = unk
      num_unk_tokens += 1

  unk_tokens.append(eos)
  return (unk_tokens, num_unk_tokens, num_unk_types)

def to_id(tokens, vocab_map, offset=0, unk='<unk>'):
  return [str(vocab_map[token]+offset) if token in vocab_map else str(vocab_map[unk]+offset) for token in tokens]
    
def to_text(indices, words, offset=0):
  return [words[int(index)-offset] for index in indices]
  
def write_vocab(out_file, words, freqs=[]):
  f = codecs.open(out_file, 'w', 'utf-8')
  sys.stderr.write('# Output vocab to %s ...\n' % out_file)
  vocab_size = 0
  for word in words:
    #f.write('%s %d\n' % (word, vocab_size))
    if len(freqs)==0:
      f.write('%s\n' % word)
    else:
      f.write('%s %d\n' % (word, freqs[vocab_size]))
    vocab_size += 1
  f.close()
  sys.stderr.write('  num words = %d\n' % vocab_size)

def load_vocab(in_file, sos='<s>', eos='</s>', unk='<unk>'):
  sys.stderr.write('# Loading vocab file %s ...\n' % in_file) 
  vocab_inf = codecs.open(in_file, 'r', 'utf-8')
  words = []
  vocab_map = {}
  vocab_size = 0
  for line in vocab_inf:
    tokens = re.split('\s+', line.strip())
    word = tokens[0]
    words.append(word)
    vocab_map[word] = vocab_size
    vocab_size += 1
  
  # add sos, eos, unk 
  for word in [sos, eos, unk]:
    (words, vocab_map, vocab_size) = add_word_to_vocab(word, words, vocab_map, vocab_size)
  vocab_inf.close()
  sys.stderr.write('  num words = %d\n' % vocab_size)
  return (words, vocab_map, vocab_size)

def load_vocab_from_corpus(in_file, freq, max_vocab_size, unk='<unk>'):
  f = codecs.open(in_file, 'r', 'utf-8')
  sys.stderr.write('# Loading vocab from %s ... ' % in_file)
  
  words = []
  vocab_map = {}
  freq_map = {}
  vocab_size = 0
  num_train_words = 0
  num_lines = 0 
  for line in f:
    tokens = re.split('\s+', line.strip())
    num_train_words += len(tokens)
    for token in tokens:
      if token not in vocab_map:
        words.append(token)
        vocab_map[token] = vocab_size
        freq_map[token] = 0
        vocab_size += 1
      freq_map[token] += 1

    num_lines += 1
    if num_lines % 100000 == 0:
      sys.stderr.write(' (%d) ' % num_lines)
      #break
  f.close()
  sys.stderr.write('\n  vocab_size=%d, num_train_words=%d, num_lines=%d\n' % (vocab_size, num_train_words, num_lines))
  
  if freq>0 or max_vocab_size>0:
    (words, vocab_map, freq_map, vocab_size) = update_vocab(words, vocab_map, freq_map, freq, max_vocab_size, unk=unk)
  return (words, vocab_map, freq_map, vocab_size, num_train_words, num_lines)

def update_vocab(words, vocab_map, freq_map, freq, max_vocab_size, sos='<s>', eos='</s>', unk='<unk>'):
  """
  Filter out rare words (<freq) or keep the top vocab_size frequent words
  """
 
  new_words = [unk, sos, eos]
  new_vocab_map = {unk:0, sos:1, eos:2}
  new_freq_map = {unk:0, sos:0, eos:0}
  vocab_size = 3
  if freq>0:
    for word in words:
      if freq_map[word] < freq: # rare
        new_freq_map[unk] += freq_map[word]
      else:
        new_words.append(word)
        new_vocab_map[word] = vocab_size
        new_freq_map[word] = freq_map[word]
        vocab_size += 1
    sys.stderr.write('  convert rare words (freq<%d) to %s: new vocab size=%d, unk freq=%d\n' % (freq, unk, vocab_size, new_freq_map[unk]))
  else:
    assert(max_vocab_size>0)
    sorted_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
    for (word, freq) in sorted_items:
      new_words.append(word)
      new_vocab_map[word] = vocab_size
      new_freq_map[word] = freq
      vocab_size += 1
      if vocab_size == max_vocab_size:
        break
    sys.stderr.write('  update vocab: new vocab size=%d\n' % (vocab_size))
  return (new_words, new_vocab_map, new_freq_map, vocab_size)

#################
### NLM data ###
#################
def get_ngrams_line(line, vocab_map, ngram_size, opt, sos='<s>', eos='</s>', unk='<unk>'):
  predict_pos = (ngram_size-1) # predict last word
  if opt==1: # predict middle word
    predict_pos = (ngram_size-1)/2

  x = [] # training examples
  y = [] # labels
  tokens = re.split('\s+', line.strip())
  if len(tokens)==0:
    return (x, y)
  
  start_ngram = [sos for i in xrange(predict_pos)]
  end_ngram = [eos for i in xrange(ngram_size-predict_pos)]
  tokens = start_ngram + tokens + end_ngram
  unk_index = vocab_map[unk]
  
  # start (ngram_size-1)
  ngram = []
  num_ngrams = 0 
  for pos in xrange(ngram_size-1):
    token = tokens[pos]
    if token in vocab_map:
      id = vocab_map[token]
    else:
      id = unk_index
    ngram.append(id)    
 
  # continue
  for pos in xrange(ngram_size-1, len(tokens)):
    # predicted word 
    token = tokens[pos]
    if token in vocab_map:
      id = vocab_map[token]
    else:
      id = unk_index
    ngram.append(id)

    # extract data/label
    example = list(ngram)
    del example[predict_pos]
    x.append(example) 
    y.append(ngram[predict_pos]) 
      
    num_ngrams += 1

    # remove prev word 
    ngram.pop(0)
    
  return (x, y)

def print_ngram(x, y, words):
  context = [words[j] for j in x]
  sys.stderr.write('%s -> %s\n' % (' '.join(context), words[y]))
 

def get_ngrams(f, vocab_map, words, vocab_size, ngram_size, opt, num_read_lines=-1, shuffle=False, sos='<s>', eos='</s>', unk='<unk>'):
  """
  Extract ngrams for training purpose for a fixed number of sentences
  """
  x = [] # training examples
  y = [] # labels
  global debug 
  unk_index = vocab_map[unk]
  num_ngrams = 0 
  num_lines = 0
  
  predict_pos = (ngram_size-1) # predict last word
  if opt==1: # predict middle word
    predict_pos = (ngram_size-1)/2
  
  start_ngram = [sos for i in xrange(predict_pos)]
  end_ngram = [eos for i in xrange(ngram_size-predict_pos)]

  for line in f:
    tokens = re.split('\s+', line.strip())
    if len(tokens)==0: continue 
    tokens = start_ngram + tokens + end_ngram
    
    # start (ngram_size-1)
    ngram = []
    for pos in xrange(ngram_size-1):
      token = tokens[pos]
      if token in vocab_map:
        id = vocab_map[token]
      else:
        id = unk_index
      ngram.append(id)    
   
    # continue
    for pos in xrange(ngram_size-1, len(tokens)):
      # predicted word 
      token = tokens[pos]
      if token in vocab_map:
        id = vocab_map[token]
      else:
        id = unk_index
      ngram.append(id)

      # extract data/label
      example = list(ngram)
      del example[predict_pos]
      x.append(example) 
      y.append(ngram[predict_pos]) 
        
      num_ngrams += 1

      # remove prev word 
      ngram.pop(0)
    
    # debug
    if debug==True:
      sys.stderr.write('  Line: %s' % line)
      for i in xrange(num_ngrams):
        context = [words[j] for j in x[i]]
        sys.stderr.write('  %s -> %s\n' % (' '.join(context), words[y[i]]))
      debug=False

    num_lines += 1
    if num_lines == num_read_lines:
      break
  
  if shuffle==True:
    indices = range(len(y))
    random.shuffle(indices)
    new_x = []
    new_y = []
    
    for i in indices:
      new_x.append(x[i])
      new_y.append(y[i])
    return (new_x, new_y)
  else:
    return (x, y)

def get_all_ngrams(in_file, vocab_map, words, vocab_size, ngram_size, opt, num_read_lines=-1):
  f = codecs.open(in_file, 'r', 'utf-8')
  sys.stderr.write('# Loading ngrams from %s ...\n' % in_file)
  (x, y) = get_ngrams(f, vocab_map, words, vocab_size, ngram_size, opt, num_read_lines)
  f.close()
  return (x, y) 

def get_src_ngram(src_pos, src_tokens, src_window, src_vocab_map, src_unk_index, src_sos_index, src_eos_index, tgt_vocab_size):
  src_ngram = []
  src_len = len(src_tokens)

  # left
  for ii in xrange(src_window):
    if (src_pos - src_window + ii)<0: # sos
      src_id = src_sos_index  
    else:
      src_id = look_up(src_tokens[src_pos - src_window + ii], src_vocab_map, src_unk_index)
    src_ngram.append(src_id + tgt_vocab_size)
  
  # current word
  src_id = look_up(src_tokens[src_pos], src_vocab_map, src_unk_index)
  src_ngram.append(src_id + tgt_vocab_size)

  # right
  for ii in xrange(src_window):
    if (src_pos+ii+1)>=src_len: # eos
      src_id = src_eos_index  
    else:
      src_id = look_up(src_tokens[src_pos+ii+1], src_vocab_map, src_unk_index)
    src_ngram.append(src_id + tgt_vocab_size)
  
  return src_ngram
  
def get_src_pos(tgt_pos, t2s):
  """
  Get aligned src pos by average if there're multiple alignments. Return -1 if no alignment.
  """
  if tgt_pos in t2s:
    src_pos = 0
    for src_aligned_pos in t2s[tgt_pos]:
      src_pos += src_aligned_pos
    return int(src_pos/len(t2s[tgt_pos]))
  else:
    return -1

def infer_src_pos(tgt_pos, t2s, tgt_len):
  """
  Infer src aligned pos. Try to look around if there's no direct alignment
  """
  src_pos = get_src_pos(tgt_pos, t2s)
  if src_pos==-1: # unaligned word, try to search alignments around
    k = 1
    while (tgt_pos-k)>=0 or (tgt_pos+k)<tgt_len:
      if(tgt_pos-k)>=0: # left
        src_pos = get_src_pos(tgt_pos-k, t2s)
      if src_pos==-1 and (tgt_pos+k)<tgt_len: # right
        src_pos = get_src_pos(tgt_pos+k, t2s)
      if src_pos != -1: break
      k += 1
      #if k>=3: break
  return src_pos

def print_joint_ngram(x, y, src_window, src_words, tgt_words, tgt_vocab_size):
  src_ngram_size = 2*src_window + 1
  src_context = [src_words[x[j]-tgt_vocab_size] for j in xrange(src_ngram_size)]
  context = [tgt_words[x[j]] for j in xrange(src_ngram_size, len(x))]
  sys.stderr.write('  %s ||| %s -> %s\n' % (' '.join(src_context).encode('utf-8'), ' '.join(context).encode('utf-8'), tgt_words[y].encode('utf-8')))

def get_line_joint_ngrams(src_line, tgt_line, align_line, tgt_ngram_size, src_window, src_words, src_vocab_map, src_vocab_size, tgt_words, tgt_vocab_map, tgt_vocab_size, unk, sos, eos, start_ngram, end_ngram, predict_pos, src_unk_index, src_sos_index, src_eos_index, tgt_unk_index, tgt_sos_index, tgt_eos_index, min_sent_len, debug):
  x = [] # training examples
  y = [] # labels
  num_ngrams = 0

  src_tokens = re.split('\s+', src_line)
  tgt_tokens = re.split('\s+', tgt_line)
  tgt_orig_len = len(tgt_tokens)
  if len(src_tokens)<min_sent_len or len(tgt_tokens)<=min_sent_len: 
    return (x, y, num_ngrams)

  # alignment
  (s2t, t2s) = aggregate_alignments(align_line)
  if len(t2s)==0:
    return (x, y, num_ngrams)

  if debug==True:
    sys.stderr.write('  src: %s\n' % src_line.encode('utf-8'))
    sys.stderr.write('  tgt: %s\n' % tgt_line.encode('utf-8'))
    sys.stderr.write('  align: %s\n' % align_line)
    sys.stderr.write('  s2t: %s\n' % str(s2t))
    sys.stderr.write('  t2s: %s\n' % str(t2s))
    sys.stderr.write('  tgt_orig_len: %d\n' % tgt_orig_len)
  tgt_tokens = start_ngram + tgt_tokens + end_ngram
  
  # start (tgt_ngram_size-1)
  ngram = []
  for pos in xrange(tgt_ngram_size-1):
    ngram.append(look_up(tgt_tokens[pos], tgt_vocab_map, tgt_unk_index))
       
  # continue
  for pos in xrange(tgt_ngram_size-1, len(tgt_tokens)): 
    ngram.append(look_up(tgt_tokens[pos], tgt_vocab_map, tgt_unk_index))

    # get src ngram
    tgt_pos = pos - predict_pos # predict_pos = len(start_ngram)
    src_pos = infer_src_pos(tgt_pos, t2s, tgt_orig_len)
    if src_pos==-1: continue

    ## src part
    src_tgt_ngram = get_src_ngram(src_pos, src_tokens, src_window, src_vocab_map, src_unk_index, src_sos_index, src_eos_index, tgt_vocab_size)
          
    ## tgt part
    for ii in xrange(tgt_ngram_size):
      if ii!=predict_pos: src_tgt_ngram.append(ngram[ii])
    x.append(src_tgt_ngram) 
    y.append(ngram[predict_pos]) 
    num_ngrams += 1

    if debug==True:
      #sys.stderr.write('  %s (%d) --  %s (%d)\n' % (tgt_tokens[pos], tgt_pos, src_tokens[src_pos], src_pos))
      print_joint_ngram(x[-1], y[-1], src_window, src_words, tgt_words, tgt_vocab_size)

    # remove prev word 
    ngram.pop(0)
  
  return (x, y, num_ngrams)

def get_joint_ngrams(src_f, tgt_f, align_f, src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, tgt_ngram_size, src_window, opt, min_sent_len, num_read_lines, shuffle=False, sos='<s>', eos='</s>', unk='<unk>'):
  x = [] # training examples
  y = [] # labels
  global debug 
  num_ngrams = 0 
  num_lines = 0

  src_unk_index = src_vocab_map[unk]
  tgt_unk_index = tgt_vocab_map[unk]
  src_sos_index = src_vocab_map[sos]
  tgt_sos_index = tgt_vocab_map[sos]
  src_eos_index = src_vocab_map[eos]
  tgt_eos_index = tgt_vocab_map[eos]
  
  predict_pos = (tgt_ngram_size-1) # predict last word
  if opt==1: # predict middle word
    predict_pos = (tgt_ngram_size-1)/2
  start_ngram = [sos for i in xrange(predict_pos)]
  end_ngram = [eos for i in xrange(tgt_ngram_size-predict_pos)]
  assert len(start_ngram) == predict_pos
  assert len(start_ngram) + len(end_ngram) == tgt_ngram_size

  for align_line in align_f:
    src_line = src_f.readline().strip()
    tgt_line = tgt_f.readline().strip()
    align_line = align_line.strip()
    
    (line_x, line_y, line_num_ngrams) = get_line_joint_ngrams(src_line, tgt_line, align_line, tgt_ngram_size, src_window, src_words, src_vocab_map, src_vocab_size, tgt_words, tgt_vocab_map, tgt_vocab_size, unk, sos, eos, start_ngram, end_ngram, predict_pos, src_unk_index, src_sos_index, src_eos_index, tgt_unk_index, tgt_sos_index, tgt_eos_index, min_sent_len, debug)
    x.extend(line_x)
    y.extend(line_y)
    num_ngrams = num_ngrams + line_num_ngrams
    
    # debug
    if debug==True:
      debug=False
      #sys.exit(1)

    num_lines += 1
    if num_lines == num_read_lines:
      break
 
  #sys.stderr.write('  get_joint_ngrams: count=%d\n' % num_ngrams)
  if shuffle==True:
    indices = range(len(y))
    random.shuffle(indices)
    new_x = []
    new_y = []
    
    for i in indices:
      new_x.append(x[i])
      new_y.append(y[i])
    return (new_x, new_y)
  else:
    return (x, y)

def get_all_joint_ngrams(src_file, tgt_file, align_file, src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, num_read_lines):
  src_f = codecs.open(src_file, 'r', 'utf-8')
  tgt_f = codecs.open(tgt_file, 'r', 'utf-8')
  align_f = codecs.open(align_file, 'r', 'utf-8')
  sys.stderr.write('# Loading ngrams from %s %s %s ...\n' % (src_file, tgt_file, align_file))
  sys.stderr.write('  src_vocab_size=%d, tgt_vocab_size=%d, ngram_size=%d, src_window=%d\n' % (src_vocab_size, tgt_vocab_size, ngram_size, src_window))
  (x, y) = get_joint_ngrams(src_f, tgt_f, align_f, src_vocab_map, src_words, src_vocab_size, tgt_vocab_map, tgt_words, tgt_vocab_size, ngram_size, src_window, opt, min_sent_len, num_read_lines)
  src_f.close()
  tgt_f.close()
  align_f.close()
  #sys.stderr.write('  num ngrams extracted=%d\n' % (len(y)))
 
  return (x, y) 

 
