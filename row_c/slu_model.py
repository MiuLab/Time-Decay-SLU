import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class slu_model(object):
    def __init__(self, max_seq_len, intent_dim, args):
        self.hidden_size = 128
        self.intent_dim = intent_dim # one hot encoding
        self.embedding_dim = 200 # read from glove
        self.total_word = 400001 # total word embedding vectors
        self.max_seq_len = max_seq_len
        self.hist_len = 7
        self.add_variables()
        self.add_placeholders()
        self.add_variables()
        self.build_graph(args)
        self.add_loss()
        self.add_train_op()
        self.init_embedding()
        self.init_model = tf.global_variables_initializer()

    def init_embedding(self):
        self.init_embedding = self.embedding_matrix.assign(self.read_embedding_matrix)

    def add_variables(self):
        self.embedding_matrix = tf.Variable(tf.truncated_normal([self.total_word, self.embedding_dim]), dtype=tf.float32, name="glove_embedding")

    def add_placeholders(self):
        self.history_intent = tf.placeholder(tf.float32, [None, self.hist_len * 2, self.intent_dim])
        self.tourist_input_intent, self.guide_input_intent = tf.split(self.history_intent, num_or_size_splits=2, axis=1)
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        self.current_nl_len = tf.placeholder(tf.int32, [None])
        self.current_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def hist_dense(self, scope, idx, nl_outputs):
        with tf.variable_scope(scope):
            reuse = False
            if idx != 0:
                tf.get_variable_scope().reuse_variables()
                reuse = True
            if scope == 'tourist':
                inputs = tf.unstack(self.tourist_input_intent, axis=1)[idx]
            elif scope == 'guide':
                inputs = tf.unstack(self.guide_input_intent, axis=1)[idx]
            
            return tf.layers.dense(inputs=tf.concat([inputs, nl_outputs], axis=1), units=1, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer, name=scope+"hist_dense", reuse=reuse)

    def nl_biRNN(self, history_summary):
        with tf.variable_scope("nl"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            history_summary = tf.expand_dims(history_summary, axis=1)
            replicate_summary = tf.tile(history_summary, [1, self.max_seq_len, 1]) # [batch_size, self.max_seq_len, self.intent_dim]
            concat_input = tf.concat([inputs, replicate_summary], axis=2) # [batch_size, self.max_seq_len, self.intent_dim+self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, concat_input, sequence_length=self.current_nl_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            return outputs

    def attention(self, args):
        with tf.variable_scope("curent_nl"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=self.current_nl_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            nl_outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            nl_outputs = tf.layers.dense(inputs=nl_outputs, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
       
        if args.level == 'sentence':
            self.unstack_tourist_hist = list()
            self.unstack_guide_hist = list()
            for i in range(self.hist_len):
                self.unstack_tourist_hist.append(self.hist_dense('tourist', i, nl_outputs))
                self.unstack_guide_hist.append(self.hist_dense('guide', i, nl_outputs))
            tourist_weight = tf.nn.softmax(tf.squeeze(tf.stack(self.unstack_tourist_hist, axis=1), axis=2))
            guide_weight = tf.nn.softmax(tf.squeeze(tf.stack(self.unstack_guide_hist, axis=1), axis=2))
            tourist_hist = tf.multiply(tf.expand_dims(tourist_weight, axis=2), self.tourist_input_intent)
            guide_hist = tf.multiply(tf.expand_dims(guide_weight, axis=2), self.guide_input_intent)

        with tf.variable_scope("history_tourist_rnn"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            if args.level == 'sentence':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, tourist_hist, dtype=tf.float32)
            elif args.level == 'role':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.tourist_input_intent, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            tourist_outputs = tf.concat([final_fw, final_bw], axis=1)
        
        with tf.variable_scope("history_guide_rnn"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            if args.level == 'sentence':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, guide_hist, dtype=tf.float32)
            elif args.level == 'role':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.guide_input_intent, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            guide_outputs = tf.concat([final_fw, final_bw], axis=1)
        
        if args.level == 'role':
            tourist_dense = tf.layers.dense(inputs=tf.concat([tourist_outputs, nl_outputs], axis=1), units=1, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer, name="dense_layer")
            guide_dense = tf.layers.dense(inputs=tf.concat([guide_outputs, nl_outputs], axis=1), units=1, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer, name="dense_layer", reuse=True)
            weight = tf.unstack(tf.nn.softmax(tf.concat([tourist_dense, guide_dense], axis=1)), axis=1)
            assert len(weight) == 2
            return tf.add(tf.multiply(tf.expand_dims(weight[0], axis=1), tourist_outputs), tf.multiply(tf.expand_dims(weight[1], axis=1), guide_outputs))
        return tf.add(tourist_outputs, guide_outputs)
    
    def build_graph(self, args):
        concat_output = self.attention(args)
        if args.level == 'sentence':
            history_summary = concat_output
        elif args.level == 'role':
            history_summary = tf.layers.dense(inputs=concat_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        final_output = self.nl_biRNN(history_summary)
        self.output = tf.layers.dense(inputs=final_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        self.intent_output = tf.sigmoid(self.output)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)
