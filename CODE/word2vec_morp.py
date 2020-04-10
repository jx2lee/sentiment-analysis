import pickle
from core.prepo import *
from tqdm import tqdm

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels

if __name__=="__main__":
    
    npPath = './data/dataNp/morp/'
    path1 = 'X_morp.npz'
    path2 = 'y_morp.npz'

    X_train, X_test, y_train, y_test = LoadData(npPath, path1, path2)

    train_df = arraytoDf(X_train, y_train)
    test_df = arraytoDf(X_test, y_test)

    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # import dictionary
    with open('./data/dict/dictionary_morp.pkl', 'rb') as p:
        dictMorp = {y:x for x,y in pickle.load(p).items()}

    print(len(dictMorp))

    skip_grams = []

    for i in tqdm(range(len(df['seq']))):
        for j in range(1, len(df['seq'][i]) - 1):

            target = df['seq'][i][j]
            context = [df['seq'][i][j - 1], df['seq'][i][j + 1]]

            for w in context:
                skip_grams.append([target, w])

    print(len(skip_grams))
    
    # model
    training_epoch = 1000
    learning_rate = 0.1
    batch_size = 200
    embedding_size = 99

    num_sampled = 100
    voc_size = len(dictMorp)

    inputs = tf.placeholder(tf.int32, shape=[batch_size])
    labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

    nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([voc_size]))


    loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # train
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('./checkpoints_Word2vec/morpGraph', sess.graph)

        for step in range(1, training_epoch + 1):
            batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

            _, loss_val = sess.run([train_op, loss],
                                   feed_dict={inputs: batch_inputs,
                                              labels: batch_labels})

            if step % 100 == 0:
                print("loss at step ", step, ": ", loss_val)
                saver.save(sess, './checkpoints_Word2vec/morpGraph/word2vec.ckpt', global_step=step)

        trained_embeddings = embeddings.eval()
    
    # save matrix
    with open('./data/embedding_matrix/morp.pkl', 'wb') as p:
        pickle.dump(trained_embeddings, p)