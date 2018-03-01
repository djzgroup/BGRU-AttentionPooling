import  tensorflow as tf
from BLSTM_Att import Blstm_att
import numpy as np
import dataUtils
import sys
import os
import config


def train(dataset):
    embedding = np.load("")#loading your word embedding


    sent = tf.placeholder(tf.int64, [None, config.sentence_length])
    y = tf.placeholder(tf.float64, [None, config.num_class])
    seq_len = tf.placeholder(tf.int32, [None])
    dropout_blstm_prob = tf.placeholder(tf.float32, name="dropout")
    dropout_word_prob = tf.placeholder(tf.float32, name="dropout")

    model = Blstm_att(config.batch_size,config.sentence_length,config.embeddings_size,config.hidden_size,config.num_label,embedding,seq_len,config.regularizer_rate,dropout_blstm_prob,dropout_word_prob)

    out_trian = model.Blstm_att(sent,config.fliter_size,config.stride,True)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.name_scope("cost"):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_trian, labels=y))
        loss_train = cost_train + config.regularizer_rate * sum(reg_loss)

    with tf.name_scope("acc"):
        Acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out_trian, 1)), tf.float32))




    optimizer = tf.train.AdadeltaOptimizer(1.0)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_train, tvars), config.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



    with tf.Session() as sess:
        train_seq_len = np.ones(config.batch_size) * config.sentence_length
        sess.run(tf.global_variables_initializer())


        #=========save_the_model======
        checkpoint_dir = os.path.abspath(os.path.join(config.model_path, "checkpoints"))
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
        for i in range(config.n_epochs):
            training_step,test_npochs,x_train,y_train,x_val,y_val = dataUtils.load_data(config.batch_size,dataset)
            for i in range(training_step):
                current_step = tf.train.global_step(sess, global_step)
                start = i * config.batch_size
                end = start + config.batch_size
                batch_train = x_train[start:end]
                train_label = y_train[start:end]
                train(batch_train,train_label)

                if current_step%500==0:
                    dev_loss = 0
                    dev_acc = 0
                    for i in range(test_npochs):
                        start = i * config.batch_dev
                        end = start + config.batch_dev
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
                    np.save("Subj.npy", valid)





def main():
    train()




if __name__ == '__main__':
    main()
