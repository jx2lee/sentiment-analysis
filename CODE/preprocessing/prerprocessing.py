
# coding: utf-8

# In[1]:


import os
import random
import re
import pickle
import numpy as np
from hangul_utils import split_syllables
from tqdm import tqdm
from keras.preprocessing import sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from konlpy.tag import Mecab


# ### _Load data_

# In[2]:


PWD = './../data/rawData/'
FILES_neg = []
FILES_pos = []

for path, dirs, files in os.walk(PWD):
    for i in range(len(files)):
        if files[i].find('pos') != -1:
            FILES_pos.append(files[i])
        else:
            FILES_neg.append(files[i])
            
def fileTolist(file_name):
    
    '''
    
    load raw text to list
    and return list about moview reviews
    
    '''
    
    allReviews = []
    for i in range(len(file_name)):
        with open(PWD+file_name[i], 'r', encoding='utf-8') as f:
                reviews = f.readlines()
        for j in range(len(reviews)):
            allReviews.append(reviews[j].replace('  \n', ''))
    
    return allReviews


# ### _split function_

# In[3]:


def textToPhon(text):
    '''
    
    text divided into phoneme
    and return text
    
    '''

    sp_text = []
    hangul = re.compile('[^\u3131-\u3163\uac00-\ud7a3]+')
    for split in text:
        review = hangul.sub('', split_syllables(split))
        if len(review)!=0:
            sp_text.append(review)
    
    return sp_text

def textToMorp(text):
    
    '''
    
    text divided into morpheme
    and return text
    
    '''
    
    mc = Mecab()
    sp_text = []
    for i in range(len(text)):
        sp_text.append(mc.morphs(text[i]))
        
    return sp_text

def textToWord(text):
    
    '''
    
    text divided into word
    and return text
    
    '''
    
    sp_text = []
    for i in range(len(text)):
        sp_text.append(text[i].split())
    
    return sp_text


# ### _Make dictionary_

# In[4]:


def makeDict_phon(text, path):
    
    '''
    
    make dictionary for movie revews
    
    '''
    
    unqPhon = []
    
    print('make phon-dictionary...')
    
    for i in tqdm(range(len(text))):
        for phon in (text)[i]:
            if phon not in unqPhon:
                unqPhon.append(phon)        

    print('# of unique Phoneme : {}\nexample : {}'.format(len(unqPhon), unqPhon))
    
    
    phon_label = {ch : i+1 for i, ch in enumerate(unqPhon)}
    label_phon = {i+1 : ch for i, ch in enumerate(unqPhon)}
    
    
    #save dictionary
    with open(path+'dictionary_phon.pkl', 'wb') as p:
        pickle.dump(label_phon, p)
       
    return unqPhon, phon_label


def makeDict_morp(text, path):
    
    '''
    
    make dictionary for movie revews
    
    '''
    
    unqMorp = []
    
    print('make morp-dictionary...')
    
    for i in tqdm(range(len(text))):
        for morp in (text)[i]:
            if morp not in unqMorp:
                unqMorp.append(morp)
    
    print('# of unique Morpheme : {}\nexample : {}'.format(len(unqMorp), random.sample(unqMorp, 10)))
    
    Morp = []

    for i in tqdm(range(len(text))):
        for j in range(len(text[i])):
            Morp.append(text[i][j])

    newMorp = []
    for i in range(len(Morp)):
        hangul = re.compile('[-=.#/?:^~!$}0-9]')
        Morp[i] = hangul.sub('', Morp[i])
        if len(Morp[i])!=0:
            newMorp.append(Morp[i])

    morps_count={}
    for morp in newMorp:
        if morp in morps_count:
            morps_count[morp] += 1
        else:
            morps_count[morp] = 1
    sorted_morps = sorted([(k,v) for k,v in morps_count.items()],
                           key=lambda morp_count: -morp_count[1])[:10000]
    print(sorted_morps)

    label_morp = {i+1 : ch[0] for i, ch in enumerate(sorted_morps)}
    morp_label = {y:x for x,y in label_morp.items()}
    #save dictionary
    with open(path+'dictionary_morp.pkl', 'wb') as p:
        pickle.dump(label_morp, p)
        
    return unqMorp, morp_label


def makeDict_word(text, path):
    
    '''
    
    make dictionary for movie reviews
    
    '''
    
    unqWord = []
    
    print('make word-dictionary...')
    
    for i in tqdm(range(len(text))):
        for word in (text)[i]:
            if word not in unqWord:
                unqWord.append(word)        

    print('# of unique Word : {}\nexample : {}'.format(len(unqWord), unqWord))
    
    Word = []

    for i in tqdm(range(len(text))):
        for j in range(len(text[i])):
            Word.append(text[i][j])

    newWord = []
    for i in range(len(Word)):
        hangul = re.compile('[-=.#/?:^~!$}0-9]')
        Word[i] = hangul.sub('', Word[i])
        if len(Word[i])!=0:
            newWord.append(Word[i])

    words_count={}
    for word in newWord:
        if word in words_count:
            words_count[word] += 1
        else:
            words_count[word] = 1
    sorted_words = sorted([(k,v) for k,v in words_count.items()], 
                           key=lambda word_count: -word_count[1])[:40000]
    print(sorted_words)
    
    label_word = {i+1 : ch[0] for i, ch in enumerate(sorted_words)}
    word_label = {y:x for x,y in label_word.items()}
    
    #save dictionary
    with open(path+'dictionary_word.pkl', 'wb') as p:
        pickle.dump(label_word, p)
       
    return unqWord, word_label


# ### _Make array_

# ###### phoneme

# In[11]:


def phonToArray(neg, pos, phon_label, np_path):
    
    '''
    NO ONE_HOT
    
    make array using text and phon label
    and return array
    
    '''
    
    #make array
    negPhonArray = np.asarray([[phon_label[w] for w in sent if w in phon_label.keys()] for sent in neg])
    posPhonArray = np.asarray([[phon_label[w] for w in sent if w in phon_label.keys()] for sent in pos])

    #make X, y
    X = np.concatenate((negPhonArray, posPhonArray), axis = 0)
    
    #2-dimension y
    y_neg = [[1,0] for _ in range(45000)]
    y_pos = [[0,1] for _ in range(45000)]
    y = np.asarray(y_neg+y_pos)
        
    #shuffle and save
    np.random.seed(618);np.random.shuffle(X)
    np.random.seed(618);np.random.shuffle(y)
    
    np.savez(np_path+'X_phon.npz', X)
    np.savez(np_path+'y_phon.npz', y)
    
    return X, y



def phonToArray_oneHot(neg, pos, phon_label):
    
    '''
    
    ONE_HOT & PADDING
    
    make array using text and phon label
    and return array
    
    '''
    
    #make array
    negPhonArray = np.asarray([[phon_label[w] for w in sent if w in phon_label.keys()] for sent in neg])
    posPhonArray = np.asarray([[phon_label[w] for w in sent if w in phon_label.keys()] for sent in pos])

    #make X-ONE_HOT, y
    X = np.concatenate((negPhonArray, posPhonArray), axis = 0)
    
    #confirm max length for X
    maxlen = []
    for i in range(len(X)):
        maxlen.append(len(X[i]))
    
    X = sequence.pad_sequences(X, maxlen=max(maxlen))
    ohe = OneHotEncoder(52)
    newX = []
    
    print('set one-hot vector for phoneme...')
    for i in tqdm(range(len(X))):
            newX.append(ohe.fit_transform(np.reshape(X[i], (-1, 1))).toarray())
    
    newX = np.asarray(newX)
    
    print('change blank to 0...')
    for i in tqdm(range(len(newX))):
        for j in range(len(newX[i])):
            if newX[i][j][0] == 1:
                newX[i][j][0] = 0

    y_neg = [[1,0] for _ in range(45000)]
    y_pos = [[0,1] for _ in range(45000)]
    y = np.asarray(y_neg+y_pos)

    np.random.seed(618);np.random.shuffle(newX)
    np.random.seed(618);np.random.shuffle(y)
    
    X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.3)
    
    np.savez('./data/phon_npz/Xtrain_phon_oneHot.npz', X_train)
    np.savez('./data/phon_npz/Xtest_phon_oneHot.npz', X_test)
    np.savez('./data/phon_npz/ytrain_phon_oneHot.npz', y_train)
    np.savez('./data/phon_npz/ytest_phon_oneHot.npz', y_test)

    return X_train, X_test, y_train, y_test


# ###### morpheme

# In[6]:


def morpToArray(neg, pos, morp_label, np_path):
    
    '''
    
    make array using text and morp label
    and return array
    and no padding
    
    '''
    
    #make array
    negMorpArray = np.asarray([[morp_label[w] for w in sent if w in morp_label.keys()] for sent in neg])
    posMorpArray = np.asarray([[morp_label[w] for w in sent if w in morp_label.keys()] for sent in pos])

    #make X, y
    X = np.concatenate((negMorpArray, posMorpArray), axis = 0)

    y_neg = [[1,0] for _ in range(45000)]
    y_pos = [[0,1] for _ in range(45000)]
    y = np.asarray(y_neg+y_pos)

    #shuffle and save
    np.random.seed(618);np.random.shuffle(X)
    np.random.seed(618);np.random.shuffle(y)
    
    np.savez(np_path+'X_morp.npz', X)
    np.savez(np_path+'y_morp.npz', y)
    
    return X, y


# ###### word

# In[7]:


def wordToArray(neg, pos, word_label, np_path):
    
    '''
    
    make array using text and word label
    and return array
    and no padding
    
    '''
    
    #make array
    negWordArray = np.asarray([[word_label[w] for w in sent if w in word_label.keys()] for sent in neg])
    posWordArray = np.asarray([[word_label[w] for w in sent if w in word_label.keys()] for sent in pos])

    #make X, y
    X = np.concatenate((negWordArray, posWordArray), axis = 0)

    y_neg = [[1,0] for _ in range(45000)]
    y_pos = [[0,1] for _ in range(45000)]
    y = np.asarray(y_neg+y_pos)

    #shuffle and save
    np.random.seed(618);np.random.shuffle(X)
    np.random.seed(618);np.random.shuffle(y)
    
    np.savez(np_path+'X_word.npz', X)
    np.savez(np_path+'y_word.npz', y)
    
    return X, y


# # RUN

# In[8]:


neg = fileTolist(FILES_neg)
pos = fileTolist(FILES_pos)

random_neg = random.sample(neg, 45000)
random_pos = random.sample(pos, 45000)

dict_path = '../data/dict/'

phon_path = '../data/dataNp/phon/'
morp_path = '../data/dataNp/morp/'
word_path = '../data/dataNp/word/'


# In[12]:


#phoneme
neg_phon = textToPhon(random_neg)
pos_phon = textToPhon(random_pos)

unqPhon, phon_label = makeDict_phon(neg_phon+pos_phon, dict_path)
X, y = phonToArray(neg_phon, pos_phon, phon_label, phon_path)

print('X shape : {}\n'.format(X.shape, X[0]))
print('y shape : {}\n'.format(y.shape))


# In[13]:


#morpheme
neg_morph = textToMorp(random_neg)
pos_morph = textToMorp(random_pos)

unqMorp, morp_label = makeDict_morp(neg_morph+pos_morph, dict_path)

X, y = morpToArray(neg_morph, pos_morph, morp_label, morp_path)
print('X shape : {}\nexample : {}\n'.format(X.shape, X[0]))
print('y shape : {}\n'.format(y.shape))

print('preprocess for morpheme clear..!')


# In[14]:


neg_word = textToWord(random_neg)
pos_word = textToWord(random_pos)

unqWord, word_label = makeDict_word(neg_word+pos_word, dict_path)

X, y = wordToArray(neg_word, pos_word, word_label, word_path)
print('X shape : {}\nexample : {}\n'.format(X.shape, X[0]))
print('y shape : {}\n'.format(y.shape))

print('preprocess for morpheme clear..!')

