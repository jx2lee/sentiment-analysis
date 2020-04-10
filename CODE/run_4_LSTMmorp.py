from core.prepo import *
from core.model_morp_word import *



if __name__=="__main__":
    
    # parameters
    batch_size = 100
    numClasses = 2
    unitSize = 128
    epochs = 100
    Units=128
    keepProb = 0.5
    checkpointPath = './checkpoints/checkpoints_morpLSTM'
    npPath = './data/dataNp/morp/'
    path1 = 'X_morp.npz'
    path2 = 'y_morp.npz'
    matrixPath = 'morp.pkl'
    pklName = '/result_LSTMmorp.pkl'
    runTime = '/learningTime.pkl'
    
    # run
    X_train, X_test, y_train, y_test = LoadData(npPath, path1, path2)

    train_df = arraytoDf(X_train, y_train)
    test_df = arraytoDf(X_test, y_test)
    tr = PaddedData(train_df)
    te = PaddedData(test_df)

    g = Build_LSTMgraph(state_size=unitSize,batch_size=batch_size, num_classes=numClasses,
                        lstm_sizes=Units, keepProb_=keepProb, matrix_name=matrixPath)

    tr_losses, tr_acc, te_losses, te_acc = Train_graph(g, tr, te, batch_size, epochs, PaddedData, checkpointPath, pklName, runTime)

