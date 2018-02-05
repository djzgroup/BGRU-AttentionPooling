import  numpy as np

print("loading data")


def load_data(batch_size,add,embeding_size):
    x_trian = np.load(add+"/x_train{}.npy".format(embeding_size))
    y_train = np.load(add+"/y_train{}.npy".format(embeding_size))

    x_val = np.load(add+"/x_val{}.npy".format(embeding_size))
    y_val = np.load(add+"/y_val{}.npy".format(embeding_size))



    print("====================data  shape=====================")
    print("train_shape {}".format(x_trian.shape))
    print("label_shape {}".format(y_train.shape))
    print("dev_shape {}".format(x_val.shape))
    print("labe_shape {}".format(y_val.shape))
    print("====================================================")
    train_batch = int(x_trian.shape[0]/batch_size)
    test_batch  = int(x_val.shape[0]/batch_size)
    print(train_batch)
    print(test_batch)
    return train_batch,test_batch,x_trian,y_train,x_val,y_val
