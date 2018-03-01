import numpy as np
import random as rd



def load_data(batch_size,add):
    x_trian = np.load(add+"data/train_data.npy")
    y_train = np.load(add+"data/train_label.npy")

    x_val = np.load(add+"data/test_data.npy")
    y_val = np.load(add+"data/test_label.npy")



    print("====================data  shape=====================")
    print("train_shape {}".format(x_trian.shape))
    print("label_shape {}".format(y_train.shape))
    print("dev_shape {}".format(x_val.shape))
    print("labe_shape {}".format(y_val.shape))
    print("====================================================")


    if x_trian.shape[0]%batch_size!=0:
        train_times = x_trian.shape[0]%batch_size
        while (train_times>0):
            train_index = rd.randint(0, x_trian.shape[0])
            x_trian.tolist().append(x_trian[train_index])
            y_train.tolist().append(y_train[train_index])
            train_times = train_times - 1

    if x_val.shape[0]%batch_size!=0:
        test_times = x_val.shape[0]%batch_size
        while (test_times>0):
            test_index = rd.randint(0, x_val.shape[0])
            x_val.tolist().append(x_val[test_index])
            y_val.tolist().append(y_val[test_index])
            test_times = test_times-1

    train_batch = int(x_trian.shape[0]/batch_size)
    test_batch  = int(x_val.shape[0]/batch_size)
    print(train_batch)
    print(test_batch)

    return train_batch,test_batch,x_trian,y_train,x_val,y_val
