from hangul_utils import split_syllables
from konlpy.tag import Mecab
from tqdm import tqdm
from utils import file_to_list, folder_check, save_pkl, save_npz, set_logger
import numpy as np
import random
import re
import argparse


def text_to_phoneme(text, save_dir):
    logger = set_logger('phoneme-process')
    sp_text = []
    hangul = re.compile('[^\u3131-\u3163\uac00-\ud7a3]+')
    for split in text:
        review = hangul.sub(' ', split_syllables(split))
        if len(review) != 0:
            sp_text.append(review)

    unq_phoneme = []
    logger.info('Set Dictionary.')

    for line in tqdm(sp_text):
        for phoneme in line:
            if phoneme not in unq_phoneme:
                unq_phoneme.append(phoneme)

    logger.info('# of unique Phoneme : {}\nexample : {}'.format(len(unq_phoneme), unq_phoneme[:50]))

    phoneme_label = {ch: i + 1 for i, ch in enumerate(unq_phoneme)}
    label_phoneme = {i + 1: ch for i, ch in enumerate(unq_phoneme)}

    x = np.asarray([[phoneme_label[w] for w in sent if w in phoneme_label.keys()] for sent in sp_text])
    y_neg = [[1, 0] for _ in range(45000)]
    y_pos = [[0, 1] for _ in range(45000)]
    y = np.asarray(y_neg + y_pos)

    np.random.seed(618);
    np.random.shuffle(x)
    np.random.seed(618);
    np.random.shuffle(y)

    # Check Folder
    folder_check(dir_path=save_dir, dir_name='npz')
    folder_check(dir_path=save_dir, dir_name='dictionary')

    # Save Array & Dictionary
    save_npz(npz_path=save_dir + '/npz', npz_name='x_phoneme.npz', arr=x)
    save_npz(npz_path=save_dir + '/npz', npz_name='y_phoneme.npz', arr=y)
    save_pkl(pkl_path=save_dir + '/dictionary', pkl_name='dictionary_phoneme.pkl', save_object=label_phoneme)

    return None


def text_to_morpheme(text, save_dir):
    logger = set_logger('morpheme-process')
    mc = Mecab()
    sp_text = []
    for line in text:
        sp_text.append(mc.morphs(line))

    unq_morpheme = []

    logger.info('Set Dictionary.')

    for line in tqdm(sp_text):
        for morpheme in line:
            if morpheme not in unq_morpheme:
                unq_morpheme.append(morpheme)

    logger.info('# of unique Morpheme : {}\texample : {}'.format(len(unq_morpheme), random.sample(unq_morpheme, 50)))

    all_morphemes = []
    hangul = re.compile('[-=.#/?:^~!$}0-9]')

    for line in tqdm(sp_text):
        for morpheme in line:
            morpheme = hangul.sub('', morpheme)
            if morpheme:
                all_morphemes.append(morpheme)

    morpheme_count = {}
    for morpheme in all_morphemes:
        if morpheme in morpheme_count:
            morpheme_count[morpheme] += 1
        else:
            morpheme_count[morpheme] = 1

    sorted_morpheme = sorted([(k, v) for k, v in morpheme_count.items()],
                             key=lambda morpheme_count: -morpheme_count[1])[:50000]

    label_morpheme = {i + 1: ch[0] for i, ch in enumerate(sorted_morpheme)}
    morpheme_label = {y: x for x, y in label_morpheme.items()}

    x = np.asarray([[morpheme_label[w] for w in sent if w in morpheme_label.keys()] for sent in sp_text])

    y_neg = [[1, 0] for _ in range(45000)]
    y_pos = [[0, 1] for _ in range(45000)]
    y = np.asarray(y_neg + y_pos)

    np.random.seed(618);
    np.random.shuffle(x)
    np.random.seed(618);
    np.random.shuffle(y)

    # Check Folder
    folder_check(dir_path=save_dir, dir_name='npz')
    folder_check(dir_path=save_dir, dir_name='dictionary')

    # Save Array & Dictionary
    save_npz(npz_path=save_dir + '/npz', npz_name='x_morpheme.npz', arr=x)
    save_npz(npz_path=save_dir + '/npz', npz_name='y_morpheme.npz', arr=y)
    save_pkl(pkl_path=save_dir + '/dictionary', pkl_name='dictionary_morpheme.pkl', save_object=label_morpheme)

    return None


def text_to_word(text, save_dir):
    logger = set_logger('word-process')
    sp_text = []
    for i in range(len(text)):
        sp_text.append(text[i].split())

    unq_word = []

    logger.info('Set Dictionary.')

    for line in tqdm(sp_text):
        for word in line:
            if word not in unq_word:
                unq_word.append(word)

    logger.info('# of unique Word : {}\texample : {}'.format(len(unq_word), random.sample(unq_word, 50)))

    all_words = []
    hangul = re.compile('[-=.#/?:^~!$}0-9]')

    for line in tqdm(sp_text):
        for word in line:
            word = hangul.sub('', word)
            if word:
                all_words.append(word)

    word_count = {}
    for word in all_words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    sorted_words = sorted([(k, v) for k, v in word_count.items()],
                          key=lambda word_count: -word_count[1])[:50000]

    label_word = {i + 1: ch[0] for i, ch in enumerate(sorted_words)}
    word_label = {y: x for x, y in label_word.items()}

    x = np.asarray([[word_label[w] for w in sent if w in word_label.keys()] for sent in sp_text])

    y_neg = [[1, 0] for _ in range(45000)]
    y_pos = [[0, 1] for _ in range(45000)]
    y = np.asarray(y_neg + y_pos)

    np.random.seed(618);
    np.random.shuffle(x)
    np.random.seed(618);
    np.random.shuffle(y)

    # Check Folder
    folder_check(dir_path=save_dir, dir_name='npz')
    folder_check(dir_path=save_dir, dir_name='dictionary')

    # Save Array & Dictionary
    save_npz(npz_path=save_dir + '/npz', npz_name='x_word.npz', arr=x)
    save_npz(npz_path=save_dir + '/npz', npz_name='y_word.npz', arr=y)
    save_pkl(pkl_path=save_dir + '/dictionary', pkl_name='dictionary_word.pkl', save_object=label_word)

    return None


def main():
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--token-type', type=str, required=True,
                        help='set token type (phoneme, morpheme, word')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='set output directory. ex) output (not included slash "/")')
    args = parser.parse_args()

    inputs = file_to_list('input/raw')
    if args.token_type == 'phoneme':
        text_to_phoneme(text=inputs, save_dir=args.save_dir)
    elif args.token_type == 'morpheme':
        text_to_morpheme(text=inputs, save_dir=args.save_dir)
    elif args.token_type == 'word':
        text_to_word(text=inputs, save_dir=args.save_dir)


main()