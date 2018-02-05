import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import layers

class Blstm_att():
    def __init__(self,batch_size,sentence_length,embed_sieze,HIDDEN_SIZE,num_label,word_embedding,seq_len,regularizer_rate,dropout_blstm_prob,dropout_word_prob):

        #  define your parameter

        self.batch_size = batch_size
        self.embed_sieze = embed_sieze
        self.word_embedding = word_embedding
        self.sentence_length = sentence_length
        self.num_label = num_label
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.dropout_blstm_prob = dropout_blstm_prob
        self.seq_len = seq_len
        self.regularizer_rate = regularizer_rate
        self.dropout_word_prob = dropout_word_prob

    def BLSTM_layer(self,input):
        with tf.variable_scope('LSTM-cell'):
            lstm_cell_fw = tf.nn.rnn_cell.GRUCell(self.HIDDEN_SIZE,kernel_initializer=tf.orthogonal_initializer())
            lstm_cell_bw = tf.nn.rnn_cell.GRUCell(self.HIDDEN_SIZE,kernel_initializer=tf.orthogonal_initializer())
        lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.dropout_blstm_prob)
        lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.dropout_blstm_prob)

        with tf.variable_scope("BLSTM",initializer=tf.orthogonal_initializer()):
                (fw_outputs,bw_outputs),state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw,input,sequence_length=self.seq_len,dtype="float32")
                outputs = tf.add(fw_outputs,bw_outputs)
        return outputs

    def Sentence_attentionlayer(self,input):
        with tf.variable_scope("attention_layer"):
            inputs_act = layers.fully_connected(input, self.HIDDEN_SIZE, activation_fn=tf.nn.tanh)
            Uw = tf.Variable(tf.truncated_normal([self.HIDDEN_SIZE]), name='Uw')
            attention = tf.reduce_sum(tf.multiply(inputs_act, Uw), axis=2, keep_dims=True)
            attention = tf.nn.softmax(attention,dim=1)
            attention = tf.multiply(input,attention)
            return  attention



    def Maxpool(self,input,fliter_size,stride):
        with tf.variable_scope("pool_layer"):
            input = tf.reshape(input,[self.batch_size,self.sentence_length,self.embed_sieze,1])
            pool = tf.nn.max_pool(input, ksize=[1, fliter_size, fliter_size, 1], strides=[1, stride, stride, 1], padding="VALID")
            tf.nn.convolution
            return pool

    def fully_connect(self,input):
        with tf.variable_scope("full_connect_layer"):
            input_size = input.get_shape().as_list()
            nodes = input_size[1] * input_size[2] * input_size[3]
            reshaped = tf.reshape(input,[self.batch_size,nodes])
            output = layers.fully_connected(reshaped , self.num_label , weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer_rate) )
            output = tf.nn.dropout(output, self.dropout_blstm_prob)
            return output

    def embedding_layer(self,input,is_training):
        with tf.name_scope("word_embedding"):
            W = tf.Variable(self.word_embedding, trainable=is_training, name="Word_emb", dtype='float32')
            inputs = tf.nn.embedding_lookup(W, input)
            inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_word_prob)
        return inputs

    def Blstm_att(self,sen,fliter_size,stride,is_training):
        inputs = self.embedding_layer(sen,is_training)
        layer1 = self.BLSTM_layer(inputs)
        layer2 = self.Sentence_attentionlayer(layer1)
        layer3 = self.Maxpool(layer2,fliter_size,stride)
        layer4 = self.fully_connect(layer3)
        return layer4