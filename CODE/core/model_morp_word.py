import tensorflow as tf
import numpy as np
import os, timeit
import pickle
import pandas as pd
import pickle

from random import randint

def Reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


#####################################################################################################

def Build_LSTMgraph(
    state_size,
    batch_size,
    num_classes,
    lstm_sizes,
    keepProb_,
    matrix_name):

    Reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(0.5)

    # Embedding-layer
    with open('./data/embedding_matrix/'+matrix_name, 'rb') as p:
        embeddings = pickle.load(p)
    
    lstm_inputs = tf.nn.embedding_lookup(embeddings, x)

    # LSTM
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_sizes)
    drops = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keepProb_)
    

    init_state = drops.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(drops, lstm_inputs, 
                                                  sequence_length=seqlen,
                                                  initial_state=init_state)
    
    
    # Extract last output
    idx = tf.range(batch_size)*tf.shape(lstm_outputs)[1] + (seqlen - 1)
    last_lstm_output = tf.gather(tf.reshape(lstm_outputs, [-1, state_size]), idx)
    
    # Softmax-layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    logits = tf.matmul(last_lstm_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }



def Build_stackedLSTMgraph(
    state_size,
    batch_size,
    num_classes,
    lstm_sizes,
    keepProb_,
    matrix_name):

    Reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(0.5)

    # Embedding-layer
    with open('./data/embedding_matrix/'+matrix_name, 'rb') as p:
        embeddings = pickle.load(p)
    
    lstm_inputs = tf.nn.embedding_lookup(embeddings, x)

    # LSTM
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keepProb_) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)
    
    init_state = cell.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, sequence_length=seqlen,
                                                 initial_state=init_state)
    
    # Extract last output
    idx = tf.range(batch_size)*tf.shape(lstm_outputs)[1] + (seqlen - 1)
    last_lstm_output = tf.gather(tf.reshape(lstm_outputs, [-1, state_size]), idx)
    
    # Softmax-layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    logits = tf.matmul(last_lstm_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                         labels=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }


def Train_graph(graph, tr, te, batch_size, num_epochs, iterator, checkpoint_path, pkl_name, learn_time):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver() # model save
        
        '''
        Saver and writer
        '''
        
        '''
        saver = tf.train.Saver()
        with tf.name_scope('summary'):
            tf.summary.scalar('Loss', loss)
            tf.summary.scalar('Acc', accuracy)
            tf.summary.histogram('histogram_loss', loss)
        
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(checkpointPath+'/sentimentGraph',
                                        sess.graph)
        '''
        
        '''
        Train start
        '''
        try:
            os.mkdir(checkpoint_path)
        except:
            print('directory existed..!')
        
        step, accuracy, losses = 0, 0, 0
        tr_losses, te_losses = [], []
        tr_acc, te_acc = [], []
        current_epoch = 0
        
        start_time = timeit.default_timer() # check learning time
        learningTime = [] # learning time(list)
        
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seqlen']: batch[2]}
            accuracy_, loss_, _ = sess.run([graph['accuracy'], graph['loss'], graph['ts']], feed_dict=feed)
            accuracy += accuracy_
            losses += loss_
            
            if step % 315 == 0:
                print('current_epoch : {}\t step : {}'.format(current_epoch, step))
            
            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(losses / step)
                tr_acc.append(accuracy / step)
                
                step, accuracy, losses = 0, 0, 0
                
                #model save
                saver.save(sess, checkpoint_path+'/model.ckpt', global_step=current_epoch)
                
                #eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seqlen']: batch[2]}
                    accuracy_, loss_ = sess.run([graph['accuracy'], graph['loss']], feed_dict=feed)
                    accuracy += accuracy_
                    losses += loss_

                te_losses.append(losses / step)
                te_acc.append(accuracy / step)
                step, accuracy, losses = 0,0,0
                print("Accuracy after epoch", current_epoch, " - tr_losses : ", tr_losses[-1], "- te_losses : ", te_losses[-1])
    
    # time save
    learningTime.append(timeit.default_timer() - start_time)
    
    # result.df
    result_dic = {'tr_losses' : tr_losses, 'tr_acc' : tr_acc, 'te_losses' : te_losses, 'te_acc' : te_acc}
    result_df = pd.DataFrame(result_dic, columns=['tr_losses', 'tr_acc', 'te_losses', 'te_acc'])
    
    # save df
    with open(checkpoint_path+pkl_name, 'wb') as p:
        pickle.dump(result_df, p)
    with open(checkpoint_path+learn_time, 'wb') as p:
        pickle.dump(learningTime, p)
    
    return tr_losses, tr_acc, te_losses, te_acc


