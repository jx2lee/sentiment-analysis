from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import set_logger, folder_check
import fire
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf


def random_batch(data, size):
    random_inputs = []
    random_labels = []

    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels


def trans_target(array):
    result = 1
    if array[0] == 1:
        result = 0

    return result


def array_to_df(x_arr, y_arr):
    seq = []
    seq_len = []
    sentiment = []

    for i in range(len(x_arr)):
        seq.append(x_arr[i])
        seq_len.append(len(x_arr[i]))
        sentiment.append(trans_target(y_arr[i]))

    dic = {'seq': seq, 'seqLength': seq_len, 'sentiment': sentiment}
    res = pd.DataFrame(dic, columns=['seq', 'seqLength', 'sentiment'])
    return res


def preprocess_input(path: str, x_nm: str, y_nm: str, dict_nm: str):
    x = np.load(path + '/npz/' + x_nm, allow_pickle=True)['arr_0']
    y = np.load(path + '/npz/' + y_nm, allow_pickle=True)['arr_0']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    inputs_df = pd.concat([array_to_df(x_train, y_train), array_to_df(x_test, y_test)], ignore_index=True)
    with open(path + '/dictionary/' + dict_nm, 'rb') as p:
        dictionary = {y: x for x, y in pickle.load(p).items()}

    skip_grams = []
    for i in tqdm(range(len(inputs_df['seq']))):
        for j in range(1, len(inputs_df['seq'][i]) - 1):
            target = inputs_df['seq'][i][j]
            context = [inputs_df['seq'][i][j - 1], inputs_df['seq'][i][j + 1]]
            for w in context:
                skip_grams.append([target, w])

    return skip_grams, dictionary


def run_word2vec(path: str, input: list, dictionary: dict, checkpoint_nm:str, embedding_nm):
    logger = set_logger('run-word2vec')
    folder_check(dir_path=path, dir_name='checkpoints')

    # model
    training_epoch = 1000
    learning_rate = 0.1
    batch_size = 500
    embedding_size = 99

    num_sampled = 100
    voc_size = len(dictionary) + 1

    x = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    y = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.Variable(tf.random.uniform([voc_size, embedding_size], -1.0, 1.0))
    selected_embed = tf.nn.embedding_lookup(embeddings, x)

    nce_weights = tf.Variable(tf.random.uniform([voc_size, embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([voc_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, y, selected_embed, num_sampled, voc_size))

    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        saver = tf.compat.v1.train.Saver()
        for step in range(1, training_epoch + 1):
            batch_inputs, batch_labels = random_batch(input, batch_size)
            _, loss_val = sess.run([train_op, loss],
                                   feed_dict={x: batch_inputs,
                                              y: batch_labels})
            if step % 100 == 0:
                logger.info(f"step: {step}\tloss_val:{loss_val}")
                saver.save(sess, path + checkpoint_nm, global_step=step)
        trained_embeddings = embeddings.eval()

    # save matrix
    folder_check(path, 'matrix')
    with open(path + embedding_nm, 'wb') as p:
        pickle.dump(trained_embeddings, p)
        logger.info(f'Saved matrix.\tpath:{path}\t{path + embedding_nm}')

    return None


def main(token_type):
    logger = set_logger('word2vec')
    inputs, dictionary = preprocess_input(path='output', x_nm='x_' + token_type + '.npz',
                                          y_nm='y_' + token_type + '.npz',
                                          dict_nm='dictionary_'+token_type+'.pkl')
    logger.info(f'Finished inputs.\tlength {len(inputs)}')
    run_word2vec(path='output', input=inputs, dictionary=dictionary,
                 checkpoint_nm=f'/checkpoints/{token_type}/word2vec_{token_type}.ckpt',
                 embedding_nm='/matrix/' + token_type + '.pkl')
    logger.info(f'Finished Word2vec {token_type}')


if __name__ == '__main__':
    fire.Fire(main)