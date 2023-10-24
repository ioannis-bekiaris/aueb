from utils import *
import pickle
import math

from constants import *


nltk.download('brown')
nltk.download('punkt')

corpus_sents = brown.sents(categories=brown.categories()[:4])
# corpus_sents = brown.sents(categories=brown.categories())  # whole brown corpus.
train_corpus_sents, test_set = train_test_split(corpus_sents, test_size=0.3)
test_corpus_sents, dev_corpus_sents = train_test_split(test_set, test_size=0.4)


fltc, oov = drop_least_common(train_corpus_sents, threshold=THRESHOLD)

unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

fltc = text_tokenizer(fltc)

for sent in fltc:
    unigram_counter.update([gram for gram in ngrams(sent, 1, pad_left=True, pad_right=True,
                                                    left_pad_symbol=START, right_pad_symbol=END)])
    bigram_counter.update([gram for gram in ngrams(sent, 2, pad_left=True, pad_right=True,
                                                   left_pad_symbol=START, right_pad_symbol=END)])
    trigram_counter.update([gram for gram in ngrams(sent, 3, pad_left=True, pad_right=True,
                                                    left_pad_symbol=START, right_pad_symbol=END)])

validation_tokens = replace_with_unk_test_dev(dev_corpus_sents, oov=oov)
validation_tokenized = text_tokenizer(validation_tokens)

test_tokens = replace_with_unk_test_dev(test_corpus_sents, oov=oov)
test_tokenized = text_tokenizer(test_tokens)

opt_alpha_bi = alpha_hp(bigram_accuracy, validation_tokenized,
                        unigram_counter, bigram_counter, None)
opt_alpha_tri = alpha_hp(trigram_accuracy, validation_tokenized,
                         unigram_counter, bigram_counter, trigram_counter)


''' 

##### LOG THE METRICS OF OUR MODEL #####

print(bigram_accuracy(opt_alpha_bi, validation_tokenized, unigram_counter, bigram_counter))
print(trigram_accuracy(opt_alpha_tri, validation_tokenized, unigram_counter, bigram_counter, trigram_counter))

print(bigram_accuracy(opt_alpha_bi, test_tokenized, unigram_counter, bigram_counter))
print(trigram_accuracy(opt_alpha_tri, test_tokenized, unigram_counter, bigram_counter, trigram_counter))

print(kneser_ney_trigram(test_tokenized,discount = 0.75, , unigram_counter, bigram_counter, trigram_counter))

'''

del unigram_counter[('*UNK*',)]
unigram_counter[(START,)] = 1
unigram_counter[(END,)] = 1


lst = []
for k in bigram_counter.keys():
    if '*UNK*' in k:
        lst.append(k)

for k in lst:
    del bigram_counter[k]

    # trigram
lst = []
for k in trigram_counter.keys():
    if '*UNK*' in k:
        lst.append(k)

for k in lst:
    del trigram_counter[k]

del lst

with open('unigram_counter.json', 'wb+') as f:
    pickle.dump(unigram_counter, f)
f.close()
with open('bigram_counter.json', 'wb+') as f:
    pickle.dump(bigram_counter, f)
f.close()
with open('trigram_counter.json', 'wb+') as f:
    pickle.dump(trigram_counter, f)
f.close()

with open('opt_alpha_bi.pkl', 'wb') as a:
    pickle.dump(opt_alpha_bi, a)
a.close()

with open('opt_alpha_tri.pkl', 'wb') as a:
    pickle.dump(opt_alpha_tri, a)
a.close()
