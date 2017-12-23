import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import numpy as np 
import random, pickle 
from collections import Counter 

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

# If you get a MemoryError, then that means you ran out of either VRAM or RAM
# (depending on whether you use GPU or not) 

# Lemmatizer parses out the stem, and makes it an actual word 
# Tense matters, but we are not going to matter about that for right now 

# this is used to pre process the data we will train and test our network on 

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # Counter returns a dictionary-like oject with word:count 
    
    l2 = []
    # l2 is the final lexicon

    for w in w_counts: 
        if 1000 > w_counts[w] > 50: 
            l2.append(w)
    
    # we do not want super common words like "and", "the" etc... but we do not
    # want super rare words either! 
    # 
    # we want the lexicon to be relatively small enough to be workable, and
    # efficient. you can tiker with the upper and lower bounds, but these do not
    # matter _that_ much apparently
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    '''
    feature_set = [
    [[0 1 0 1 ...], [0 1]],
    ...[],
    ]
    has a list (word bag? lexicon thing), and [x, y] indicating whether it is
    negnegative or positive example 
    '''

    feature_set = []
    
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1 
                    # or = 1, depends... 

            features = list(features)
            feature_set.append([features, classification])
    return feature_set 

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    random.shuffle(features)
    # important to train it properly lol

    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])    
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y


    
if __name__=='__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt',
            'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

