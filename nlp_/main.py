import math
from time import sleep
import pickle
from constants import *

if __name__ == '__main__':

    with open('unigram_counter.json', 'rb') as f:
        unigram_counter = pickle.load(f)
        f.close()
    with open('bigram_counter.json', 'rb') as f:
        bigram_counter = pickle.load(f)
        f.close()

    with open('trigram_counter.json', 'rb') as f:
        trigram_counter = pickle.load(f)
        f.close()

    with open('opt_alpha_bi.pkl', 'rb') as f:
        opt_alpha_bi = pickle.load(f)
        f.close()
    with open('opt_alpha_tri.pkl', 'rb') as f:
        opt_alpha_tri = pickle.load(f)
        f.close()

    vocab_size = len(unigram_counter)

    input = 'i am not'
    # input = 'playing is bad'
    sentence = input.split(' ')
    # sentence = [START] + sentence
    sentence = [START] + [START] + sentence

    flag = True

    while flag:
        sleep(1)
        print(sentence)
        prob = 0

        # for idx in range(1, len(sentence)):
        #     bigram_prob = (bigram_counter[(sentence[idx - 1], sentence[idx])] + opt_alpha_bi) / (
        #         unigram_counter[(sentence[idx - 1],)] + opt_alpha_bi * vocab_size)
        #     prob += math.log2(bigram_prob)

        # next_token = []
        # for token in unigram_counter.keys():
        #     next_token.append([math.log2((bigram_counter[(sentence[-1], token[0])] + opt_alpha_bi) / (
        #         unigram_counter[(sentence[-1],)] + opt_alpha_bi * vocab_size)), token])

        for idx in range(2, len(sentence) - 1):
            trigram_prob = (trigram_counter[(sentence[idx - 2], sentence[idx - 1], sentence[idx])] + opt_alpha_tri) / (
                bigram_counter[(sentence[idx - 2], sentence[idx - 1])] + opt_alpha_tri * vocab_size)
            prob += math.log2(trigram_prob)

        next_token = []
        for token in unigram_counter.keys():
            next_token.append([math.log2((trigram_counter[(sentence[-2], sentence[-1], token[0])] + opt_alpha_tri) /
                                         (bigram_counter[(sentence[-2], sentence[-1])] + opt_alpha_tri * vocab_size)), token])

        for estimation in next_token:
            estimation[0] += prob

            # next_token = [(math.pow(2, t[0]), t[1]) for t in next_token]

        suggestion = max(next_token, key=lambda x: x[0])
        sentence += [suggestion[1][0]]

        if sentence[-1] == END:
            flag = False
            print(sentence)
