import nltk
import numpy as np
from nltk.corpus import brown
from nltk import sent_tokenize
from nltk.tokenize import TweetTokenizer, WhitespaceTokenizer
from nltk.util import ngrams
from collections import Counter
from sklearn.model_selection import train_test_split
import math


def flat_lst(lst):
    flatten_lst = []
    for lst in lst:
        for l in lst:
            flatten_lst.append(l)
    return flatten_lst


def drop_least_common(train_corpus: list, threshold, UNK='*UNK*'):
    oov = []
    flatten_corpus = flat_lst(train_corpus)
    vocab = Counter(flatten_corpus)
    c = 0
    for k, v in vocab.items():
        if v < threshold:
            oov.append(k)
            c += v

    # for k in oov:
    #     del vocab[k]
    unk_tokenizer = [UNK if word in oov else word for word in flatten_corpus]

    '''
    produce the unigram Counter()

    vocab[UNK] = c
    '''
    return unk_tokenizer, oov


def text_tokenizer(flatten_corpus):

    sentences = sent_tokenize(' '.join(flatten_corpus))
    whitespace_wt = WhitespaceTokenizer()
    sentences_tokenized = []
    for sent in sentences:
        sent_tok = whitespace_wt.tokenize(sent)
        sentences_tokenized.append(sent_tok)
    return sentences_tokenized


def replace_with_unk_test_dev(corpus: list, oov: list, UNK='*UNK*'):

    flatten_corpus = flat_lst(corpus)
    unk_tokenizer = [UNK if word in oov else word for word in flatten_corpus]

    return unk_tokenizer


def bigram_accuracy(alpha, data, unigram_counter, bigram_counter, END='*end*'):

    sum_prob = 0
    bigram_cnt = 0
    vocab_size = len(unigram_counter)

    for sent in data:
        sent = sent + [END]
        # Iterate over the bigrams of the sentence
        for idx in range(1, len(sent)):
            bigram_prob = (bigram_counter[(sent[idx - 1], sent[idx])] + alpha) / (
                unigram_counter[(sent[idx - 1],)] + alpha * vocab_size)
            sum_prob += math.log2(bigram_prob)
            bigram_cnt += 1

        CE = -sum_prob / bigram_cnt
        perpl = math.pow(2, CE)

    return {
        'CE': '{0:.3f}'.format(CE),
        'P': '{0:.3f}'.format(perpl),
        'alpha': alpha
    }


def trigram_accuracy(alpha, data, unigram_counter, bigram_counter, trigram_counter, END='*end*'):

    sum_prob = 0
    trigram_cnt = 0
    vocab_size = len(unigram_counter)

    for sent in data:
        sent = sent + [END] + [END]
        # Iterate over the bigrams of the sentence
        for idx in range(2, len(sent) - 1):
            trigram_prob = (trigram_counter[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha) / (
                bigram_counter[(sent[idx - 2], sent[idx - 1])] + alpha * vocab_size)
            sum_prob += math.log2(trigram_prob)
            trigram_cnt += 1

        CE = -sum_prob / trigram_cnt
        perpl = math.pow(2, CE)

    return {
        'CE': '{0:.3f}'.format(CE),
        'P': '{0:.3f}'.format(perpl),
        'alpha': alpha
    }


def alpha_hp(fun, validation_tokenized, unigram_counter, bigram_counter, trigram_counter=None):
    hyp_alpha = np.linspace(0.01, 0.1, 10)
    min_ce = 100
    best_alpha = 0
    if fun.__name__ == 'bigram_accuracy':
        for h in hyp_alpha:
            if float(fun(h, validation_tokenized, unigram_counter, bigram_counter)['CE']) < min_ce:
                min_ce = float(fun(h, validation_tokenized,
                               unigram_counter, bigram_counter)['CE'])
                best_alpha = float(
                    fun(h, validation_tokenized, unigram_counter, bigram_counter)['alpha'])

    if fun.__name__ == 'trigram_accuracy':
        for h in hyp_alpha:
            if float(fun(h, validation_tokenized, unigram_counter, bigram_counter, trigram_counter)['CE']) < min_ce:
                min_ce = float(fun(h, validation_tokenized,
                               unigram_counter, bigram_counter, trigram_counter)['CE'])
                best_alpha = float(
                    fun(h, validation_tokenized, unigram_counter, bigram_counter, trigram_counter)['alpha'])

    return best_alpha


def kneser_ney_trigram(data, discount, unigram_counter, bigram_counter, trigram_counter, END='*end*'):

    sum_prob = 0
    trigram_cnt = 0
    vocab_size = len(unigram_counter)

    def D(w1, w2, w3): return max(trigram_counter[(w1, w2, w3)] - discount, 0) / (
        bigram_counter[(w1, w2)] + 0.01 * vocab_size)

    for sent in data:
        sent = sent + [END] + [END]
        # Iterate over the bigrams of the sentence
        for idx in range(2, len(sent) - 1):
            trigram_prob = D(sent[idx - 2], sent[idx - 1], sent[idx]) + 0.01 / (
                unigram_counter[(sent[idx])] + 0.01 * vocab_size)
            sum_prob += math.log2(trigram_prob)
            trigram_cnt += 1

        CE = -sum_prob / trigram_cnt
        perpl = math.pow(2, CE)

    return {
        'CE': '{0:.3f}'.format(CE),
        'P': '{0:.3f}'.format(perpl),
        'discount': discount
    }
