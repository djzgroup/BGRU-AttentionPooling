import  tensorflow as tf
from BLSTM_Att import Blstm_att
import numpy as np
import dataUtils
import sys
import os
"C:\python\\neural_network\\test_BLSTＭ\data"
#====loading data=======
batch_size = 10
sentence_length = 59
# embed_sieze =300
# HIDDEN_SIZE =300
num_label = 2
n_epochs = 100
batch_dev = 10
num_class = 2
grad_clip = 5
regularizer_rate = 0.00001
train_log_dir ="G:/20158835-hongmb/model"
dev_log_dir ="G:/20158835-hongmb/model"
model_path = "G:/20158835-hongmb/model"

def train(fliter_size,stride,embeding_size):
    embedding = np.load(r"G:\MR\embedding_matrix_{}.npy".format(embeding_size))


    sent = tf.placeholder(tf.int64, [None, sentence_length])
    y = tf.placeholder(tf.float64, [None, num_class])
    seq_len = tf.placeholder(tf.int32, [None])
    dropout_blstm_prob = tf.placeholder(tf.float32, name="dropout")
    dropout_word_prob = tf.placeholder(tf.float32, name="dropout")

    model = Blstm_att(batch_size,sentence_length,embeding_size,embeding_size,num_label,embedding,seq_len,regularizer_rate,dropout_blstm_prob,dropout_word_prob)

    out_trian = model.Blstm_att(sent,fliter_size,stride,True)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.name_scope("cost"):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_trian, labels=y))
        loss_train = cost_train + regularizer_rate * sum(reg_loss)

    with tf.name_scope("acc"):
        Acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out_trian, 1)), tf.float32))




    optimizer = tf.train.AdadeltaOptimizer(1.0)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_train, tvars), grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



    with tf.Session() as sess:
        train_seq_len = np.ones(batch_size) * sentence_length
        sess.run(tf.global_variables_initializer())


        #=========save_the_model======
        checkpoint_dir = os.path.abspath(os.path.join(model_path, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()
        #=============================
        def dev(x_batch, y_batch):
            feed_dict = {
                sent: x_batch,
                y: y_batch,
                seq_len: train_seq_len,
                dropout_blstm_prob: 1.0,
                dropout_word_prob: 1.0
            }
            step,loss, acc = sess.run([global_step,cost_train, Acc], feed_dict)
            return loss, acc

        def train(x_batch,y_batch):
            feed_dict = {
                sent: x_batch,
                y: y_batch,
                seq_len:train_seq_len,
                dropout_blstm_prob:0.5,
                dropout_word_prob:0.5

            }
            _,step,loss,acc = sess.run([train_op,global_step,loss_train,Acc],feed_dict)
            print("Train Step{},loss {:g},acc {:g} ".format(step,loss,acc))
            return loss,acc

        max_acc= 0
        valid=[]
        for i in range(n_epochs):
            training_step,test_npochs,x_train,y_train,x_val,y_val = dataUtils.load_data(batch_size,"G:\MR",embeding_size)
            for i in range(training_step):
                current_step = tf.train.global_step(sess, global_step)
                start = i * batch_size
                end = start + batch_size
                batch_train = x_train[start:end]
                train_label = y_train[start:end]
                train(batch_train,train_label)

                if current_step%500==0:
                    dev_loss = 0
                    dev_acc = 0
                    for i in range(test_npochs):
                        start = i * batch_dev
                        end = start + batch_dev
                        batch_val = x_val[start:end]
                        val_label = y_val[start:end]
                        m, n = dev(batch_val, val_label)
                        dev_loss = m + dev_loss
                        dev_acc = n + dev_acc
                    print("\nValid Step{},loss {:g},acc {:g} \n".format(current_step, dev_loss / test_npochs, dev_acc / test_npochs))

                    if dev_acc / test_npochs > max_acc:
                        max_acc = dev_acc / test_npochs
                        print("the max acc: {:g}".format(max_acc))

                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    valid.append(
                        "Valid Step:{},acc:{:g},loss:{:g}".format(current_step, dev_acc / test_npochs, dev_loss / test_npochs))
                    np.save("MR_{}_.npy".format(embeding_size), valid)





def main(argv):
    train(int(sys.argv[1]),int(sys.argv[1]),int(sys.argv[2]))




if __name__ == '__main__':
    main(sys.argv)
