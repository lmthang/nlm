Training / Testing Neural Language Models (NLMs) using Python/Theano
Thang Luong @ 2014, 2015 <lmthang@stanford.edu>

This codebase allows for training feed-forward NLMs, both monolingual (normal) models and bilingual (joint) models that condition on the source text as well. The joint NLM is in the context of machine translation (MT) and replicates the model proposed in the BBN's paper http://acl2014.org/acl2014/P14-1/pdf/P14-1129.pdf with several differences. 

For more details about this code, please refer to our paper:
Deep Neural Language Models for Machine Translation
Minh-Thang Luong Michael Kayser Christopher D. Manning
http://www.aclweb.org/anthology/K/K15/K15-1031.pdf

Feature highlights:
(a) train both monolingual (normal) and bilingual (joint) NLM models.
(b) have self-normalization feature.
(c) include all the preprocessing steps (build vocab, convert text form into integer format, and extract ngrams to train).
(d) resume training from a saved model.
(e) test trained NLMs to produce sentence probabilities.
(f) have dropout (we haven't tested this feature thoroughly and weren't able to achieve gains).

Files & Directories:
(a) README.txt     - this file
(b) code/           - directory contains all the code files, e.g. train_nlm.py and test_nlm.py.
(c) data/: contains files (train|tune|test).(en|zh|align) where -.align contains alignments for a pair of sentences per line. Each line is a series of pairs Chinese positions - English positions.

Main code:
(a) Train normal NLMs: train_nlm.py [options] train_data tune_data test_data ngram_size vocab_size out_prefix
./code/train_nlm.py --act_func tanh --learning_rate 0.1 --emb_dim 16 --hidden_layers 64 --log_freq 10 ./data/train.en ./data/tune.en ./data/test.en 11 1000 ./output/toy

After running the above command, you should get back: the model with the best valid perplexity (./output/toy.model), the most recent model (./output/toy.model.cur), and an vocab file (./output/toy.vocab). Each model goes with a config file with detailed training information.

To train more than one hidden layers, change 64 into 64-64 (2 layers) or 32-64-128 (3 layers), etc.
To use GPUs, append the following text to the beginning of a running comand: THEANO_FLAGS='device=gpu0' .

(b) Train joint NLMs: add the following options  --joint --src_lang <str> --tgt_lang <str>
THEANO_FLAGS='device=gpu0' python ./code/train_nlm.py --act_func tanh --learning_rate 0.1 --emb_dim 16 --hidden_layers 64 --joint --src_lang zh --tgt_lang en ./data/train ./data/tune ./data/test 5 1000 ./output/toy_joint

After running the above command, you should get back model files similar to (a), and two vocab files (./output/toy_joint.vocab.en and ./output/toy_joint.vocab.zh).

(c) Train self-norm models:
THEANO_FLAGS='device=gpu0' python ./code/train_nlm.py --self_norm_coeff 0.1 --act_func tanh --learning_rate 0.1 --emb_dim 16 --hidden_layers 64 --joint --src_lang zh --tgt_lang en ./data/train ./data/tune ./data/test 5 1000 ./output/toy_joint_self

(c) Test NLMs: test_nlm.py [options] model_file vocab_file test_file out_file
THEANO_FLAGS='device=gpu0' python ./code/test_nlm.py --self_norm_coeff 0.1 --act_func tanh --joint --src_lang zh --tgt_lang en --src_file ./data/test.zh --align_file ./data/test.align ./output/toy_joint_self.model ./output/toy_joint_self.vocab ./data/test.en ./output/toy_scores.txt 

Note that: the test_nlm.py code will try to output a perplexity score as well. For self-norm model, to get a correct perplexity, remove the option --self_norm_coeff 0.1
