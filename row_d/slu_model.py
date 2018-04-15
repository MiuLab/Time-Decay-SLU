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
        self.combined_weight = tf.Variable([0.33, 0.33, 0.33])
        self.embedding_matrix = tf.Variable(tf.truncated_normal([self.total_word, self.embedding_dim]), dtype=tf.float32, name="glove_embedding")

    def add_placeholders(self):
        self.history_intent = tf.placeholder(tf.float32, [None, self.hist_len * 2, self.intent_dim])
        self.tourist_input_intent, self.guide_input_intent = tf.split(self.history_intent, num_or_size_splits=2, axis=1)
        self.history_distance = tf.placeholder(tf.float32, [None, self.hist_len * 2])
        self.tourist_dist, self.guide_dist = tf.split(self.history_distance, num_or_size_splits=2, axis=1)
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        self.current_nl_len = tf.placeholder(tf.int32, [None])
        self.current_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])

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

        # train two params (a, b) = init(1, 1), 1/(a*t^b)
        self.a = tf.constant(1.0)
        self.b = tf.constant(1.0)
        tourist_b = tf.scalar_mul(self.b, tf.ones_like(self.tourist_dist))
        guide_b = tf.scalar_mul(self.b, tf.ones_like(self.guide_dist))
        tourist_linear_mapping_0 = tf.reciprocal(tf.scalar_mul(self.a, tf.pow(self.tourist_dist, tourist_b)))
        guide_linear_mapping_0 = tf.reciprocal(tf.scalar_mul(self.a, tf.pow(self.guide_dist, guide_b)))

        # k-m*t
        self.m = tf.constant(0.2)
        self.k = tf.constant(0.9)
        tourist_linear_mapping_1 = tf.subtract(tf.scalar_mul(self.k, tf.ones_like(self.tourist_dist, dtype=tf.float32)), tf.scalar_mul(self.m, self.tourist_dist))
        guide_linear_mapping_1 = tf.subtract(tf.scalar_mul(self.k, tf.ones_like(self.guide_dist, dtype=tf.float32)), tf.scalar_mul(self.m, self.guide_dist))
        tourist_linear_mapping_1 = tf.add(tf.fill(tf.shape(tourist_linear_mapping_1), 0.0001), tf.nn.relu(tourist_linear_mapping_1))
        guide_linear_mapping_1 = tf.add(tf.fill(tf.shape(guide_linear_mapping_1), 0.0001), tf.nn.relu(guide_linear_mapping_1))

        # train two params (n, d) = init(5, 3)
        self.d = tf.constant(4.5)
        self.n = tf.constant(2.5)
        tourist_n = tf.scalar_mul(self.n, tf.ones_like(self.tourist_dist))
        guide_n = tf.scalar_mul(self.n, tf.ones_like(self.guide_dist))
        tourist_linear_mapping_2 = tf.reciprocal(tf.add(tf.ones_like(self.tourist_dist), tf.pow(tf.divide(self.tourist_dist, self.d), tourist_n)))
        guide_linear_mapping_2 = tf.reciprocal(tf.add(tf.ones_like(self.guide_dist), tf.pow(tf.divide(self.guide_dist, self.d), guide_n)))

        # merge 3 functions
        combined_weight = tf.scalar_mul(tf.reciprocal(tf.reduce_sum(self.combined_weight)), self.combined_weight)
        combined_weight = tf.unstack(combined_weight)
        tourist_linear_mapping = tf.add(tf.scalar_mul(combined_weight[0], tourist_linear_mapping_0), tf.scalar_mul(combined_weight[1], tourist_linear_mapping_1))
        guide_linear_mapping = tf.add(tf.scalar_mul(combined_weight[0], guide_linear_mapping_0), tf.scalar_mul(combined_weight[1], guide_linear_mapping_1))
        tourist_linear_mapping = tf.add(tourist_linear_mapping, tf.scalar_mul(combined_weight[2], tourist_linear_mapping_2))
        guide_linear_mapping = tf.add(guide_linear_mapping, tf.scalar_mul(combined_weight[2], guide_linear_mapping_2))
        
        if args.attention == 'convex':
            tourist_linear_mapping = tourist_linear_mapping_0
            guide_linear_mapping   = guide_linear_mapping_0
        elif args.attention == 'linear':
            tourist_linear_mapping = tourist_linear_mapping_1
            guide_linear_mapping   = guide_linear_mapping_1
        elif args.attention == 'concave':
            tourist_linear_mapping = tourist_linear_mapping_2
            guide_linear_mapping   = guide_linear_mapping_2


        tourist_mean = tf.expand_dims(tf.reduce_sum(tourist_linear_mapping, axis=1), axis=1)
        guide_mean = tf.expand_dims(tf.reduce_sum(guide_linear_mapping, axis=1), axis=1)
        tourist_mean = tf.matmul(tourist_mean, tf.ones([1, self.hist_len], dtype=tf.float32))
        guide_mean = tf.matmul(guide_mean, tf.ones([1, self.hist_len], dtype=tf.float32))
        tourist_att = tf.divide(tourist_linear_mapping, tourist_mean)
        guide_att = tf.divide(guide_linear_mapping, guide_mean)
        tourist_hist = tf.multiply(self.tourist_input_intent, tf.expand_dims(tourist_att, axis=2))
        guide_hist = tf.multiply(self.guide_input_intent, tf.expand_dims(guide_att, axis=2))
        
        with tf.variable_scope("history_tourist_rnn"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            if args.level == 'role':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.tourist_input_intent, dtype=tf.float32)
            elif args.level == 'sentence':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, tourist_hist, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            tourist_outputs = tf.concat([final_fw, final_bw], axis=1)

        with tf.variable_scope("history_guide_rnn"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            if args.level == 'role':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.guide_input_intent, dtype=tf.float32)
            elif args.level == 'sentence':
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, guide_hist, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            guide_outputs = tf.concat([final_fw, final_bw], axis=1)

        if args.level == 'role':
            role_vector = tf.concat([tf.reduce_max(tourist_linear_mapping, axis=1, keep_dims=True), tf.reduce_max(guide_linear_mapping, axis=1, keep_dims=True)], axis=1)
            role_mean = tf.expand_dims(tf.reduce_sum(role_vector, axis=1), axis=1)
            role_mean = tf.matmul(role_mean, tf.ones([1, 2], dtype=tf.float32))
            normalized_weight = tf.unstack(tf.divide(role_vector, role_mean), axis=1)
            tourist_outputs = tf.multiply(tourist_outputs, tf.expand_dims(normalized_weight[0], axis=1))
            guide_outputs = tf.multiply(guide_outputs, tf.expand_dims(normalized_weight[1], axis=1))
        
        return tf.add(tourist_outputs, guide_outputs)
    
    def build_graph(self, args):
        concat_output = self.attention(args)
        history_summary = tf.layers.dense(inputs=concat_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        final_output = self.nl_biRNN(history_summary)
        self.output = tf.layers.dense(inputs=final_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        self.intent_output = tf.sigmoid(self.output)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)
