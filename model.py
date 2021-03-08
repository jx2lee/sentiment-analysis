import os
import pickle
import timeit

import pandas as pd
import tensorflow as tf

from utils import folder_check, save_pkl


def build_phoneme_model(model_type, dict_size, state_size, batch_size=100, num_classes=2):
    reset_tf_graph()

    x = tf.compat.v1.placeholder(tf.int32, [batch_size, None])  # [batch_size, num_steps]
    seq_len = tf.compat.v1.placeholder(tf.int32, [batch_size])
    y = tf.compat.v1.placeholder(tf.int32, [batch_size])
    keep_prob = tf.compat.v1.placeholder(1.0, [])

    # Embedding-layer
    embeddings = tf.compat.v1.get_variable('embedding_matrix', [dict_size, state_size], trainable=False)
    lstm_inputs = tf.nn.embedding_lookup(embeddings, x)

    # Model Layer
    hidden_layer_size = [128, 128, 128]
    if model_type == "rnn":
        hidden_layers = [tf.contrib.rnn.BasicRNNCell(size) for size in hidden_layer_size]
    elif model_type == "lstm":
        hidden_layers = [tf.contrib.rnn.BasicLSTMCell(size) for size in hidden_layer_size]
    elif model_type == "gru":
        hidden_layers = [tf.contrib.rnn.GRUCell(size) for size in hidden_layer_size]

    drops = [tf.contrib.rnn.DropoutWrapper(hidden_layer, output_keep_prob=keep_prob) for hidden_layer in hidden_layers]
    cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)

    init_state = cell.zero_state(batch_size, tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, sequence_length=seq_len, initial_state=init_state)

    # Extract last output
    idx = tf.range(batch_size) * tf.shape(lstm_outputs)[1] + (seq_len - 1)
    last_lstm_output = tf.gather(tf.reshape(lstm_outputs, [-1, state_size]), idx)

    # Softmax-layer
    with tf.compat.v1.variable_scope('softmax'):
        W = tf.compat.v1.get_variable('W', [state_size, num_classes])
        b = tf.compat.v1.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(last_lstm_output, W) + b
    predictions = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'y': y,
        'seq_len': seq_len,
        'dropout_rate': keep_prob,
        'loss': loss,
        'train_step': train_step,
        'predictions': predictions,
        'accuracy': accuracy
    }


def train_phoneme_model(graph, train_set, test_set,
                        checkpoint_path, save_df_name, save_time_name,
                        batch_size=100, num_epochs=100):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()  # model save

        folder_check(dir_path=checkpoint_path, dir_name="checkpoints")

        step, accuracy, losses = 0, 0, 0
        tr_losses, te_losses = [], []
        tr_acc, te_acc = [], []
        current_epoch = 0

        start_time = timeit.default_timer()  # check learning time
        learning_time = []  # learning time(list)

        while current_epoch < num_epochs:
            step += 1
            batch = train_set.next_batch(batch_size)
            feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seq_len']: batch[2], graph['dropout_rate']: 0.5}
            accuracy_, loss_, _ = sess.run([graph['accuracy'], graph['loss'], graph['train_step']], feed_dict=feed)
            accuracy += accuracy_
            losses += loss_

            if step % 315 == 0:
                print('current_epoch : {}\t step : {}'.format(current_epoch, step))

            if train_set.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(losses / step)
                tr_acc.append(accuracy / step)

                step, accuracy, losses = 0, 0, 0
                # model save
                saver.save(sess, checkpoint_path + '/model.ckpt', global_step=current_epoch)

                # eval test set
                te_epoch = test_set.epochs
                while test_set.epochs == te_epoch:
                    step += 1
                    batch = test_set.next_batch(batch_size)
                    feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seq_len']: batch[2]}
                    accuracy_, loss_ = sess.run([graph['accuracy'], graph['loss']], feed_dict=feed)
                    accuracy += accuracy_
                    losses += loss_

                te_losses.append(losses / step)
                te_acc.append(accuracy / step)
                step, accuracy, losses = 0, 0, 0
                print("Accuracy after epoch", current_epoch, " - tr_losses : ", tr_losses[-1], "- te_losses : ",
                      te_losses[-1])

    # time save
    learning_time.append(timeit.default_timer() - start_time)

    # result.df
    result_dic = {'tr_losses': tr_losses, 'tr_acc': tr_acc, 'te_losses': te_losses, 'te_acc': te_acc}
    result_df = pd.DataFrame(result_dic, columns=['tr_losses', 'tr_acc', 'te_losses', 'te_acc'])

    # Save result
    # with open(checkpoint_path + pkl_name, 'wb') as p:
    #     pickle.dump(result_df, p)
    # with open(checkpoint_path + learn_time, 'wb') as p:
    #     pickle.dump(learning_time, p)
    save_pkl(pkl_path=checkpoint_path, pkl_name=save_df_name, save_object=result_df)
    save_pkl(pkl_path=checkpoint_path, pkl_name=save_time_name, save_object=learning_time)

    # return tr_losses, tr_acc, te_losses, te_acc


def reset_tf_graph():
    tf.compat.v1.reset_default_graph()