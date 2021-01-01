from core.prepo import *
from core.model_phon import *


if __name__=="__main__":

    #parameters
    batch_size = 100
    features = 51
    numClasses = 2
    unitSize = 128
    epochs = 100
    Units=[128,128,128]
    keepProb = 0.5
    checkpointPath = './checkpoints/checkpoints_phonRNN'
    npPath = './data/dataNp/phon/'
    path1 = 'X_phon.npz'
    path2 = 'y_phon.npz'
    pklName = '/result_RNNphon.pkl'
    runTime = '/learningTime.pkl'

    X_train, X_test, y_train, y_test = LoadData(npPath, path1, path2)

    train_df = arraytoDf(X_train, y_train)
    test_df = arraytoDf(X_test, y_test)
    tr = PaddedData(train_df)
    te = PaddedData(test_df)

    g = Build_stackedRNNgraph(phon_size=features, state_size=unitSize, batch_size=batch_size, num_classes=numClasses, rnn_sizes=Units, keepProb_=keepProb)

    tr_losses, tr_acc, te_losses, te_acc = Train_graph_phon(g, tr, te, batch_size, epochs, PaddedData, checkpointPath, pklName, runTime)
