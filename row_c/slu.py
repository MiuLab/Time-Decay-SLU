import argparse
import tensorflow as tf
import numpy as np
from slu_preprocess import slu_data
from slu_model import slu_model
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser(description='Choose several settings.')
parser.add_argument('--target', default='ALL', choices=['ALL', 'Guide', 'Tourist'])
parser.add_argument('--level', default='role', choices=['role', 'sentence'])
args = parser.parse_args()

def one_hot(idx, T):
    # intent dim is 26, 5 is act, 21 is uttribute
    if T == 'act':
        ret = np.zeros(5)
        ret[idx] = 1.0
    elif T == 'attribute':
        ret = np.zeros(22)
        ret[idx] = 1.0
    elif T == 'mix':
        ret = np.zeros(27)
        for i in idx:
            ret[i] = 1.0
    return ret

def process_batch(batch_nl, batch_intent, max_seq_len, intent_pad_id, nl_pad_id, total_intent, batch_distance):
    nl_pad_id += 1
    hist_len = 7
    current_nl = list()
    current_nl_len = list()
    ground_truth = list()
    history_nl = list()
    history_nl_len = list()
    history_intent = list()
    history_distance = list()

    #history nl part
    for i in batch_nl:
        temp_nl = list() # following the format t t t, g g g
        temp_nl_len = list()
        assert len(i) == hist_len * 2 + 1
        history = i[:-1]
        for nl in history:
            temp_nl.append(nl+[nl_pad_id for _ in range(max_seq_len-len(nl))])
            temp_nl_len.append(len(nl))
        history_nl.append(temp_nl)
        history_nl_len.append(temp_nl_len)

    # current nl part
    for i in batch_nl:
        assert len(i) == hist_len * 2 + 1
        nl = i[-1]
        current_nl.append(nl+[nl_pad_id for _ in range(max_seq_len-len(nl))])
        current_nl_len.append(len(nl))

    # intent part
    for i in batch_intent:
        temp_history = list() # following the format t t t, g g g
        assert len(i) == hist_len * 2 + 1
        history = i[:-1]
        for tup in history:
            d = [tup[0]] + [attri for attri in tup[1]]
            temp_history.append(one_hot(d, "mix"))
        history_intent.append(temp_history)

    # target part
    for i in batch_intent:
        ground_truth.append(one_hot([i[-1][0]]+[attri for attri in i[-1][1]], "mix"))

    # history distance part, used for distance attention
    for i in batch_distance:
        history_distance.append(i[:-1])

    return history_nl, history_nl_len, current_nl, current_nl_len, history_intent, ground_truth, history_distance

def calculate_score(predict_output, ground_truth, talker=None):
    test_talker = open('../Data/test/talker', 'r').readlines()
    ret_pred_outputs = list()
    ret_ground_truth = list()
    talker_cnt = -1
    for pred, label in zip(predict_output, ground_truth):
        talker_cnt += 1
        if len(test_talker) <= talker_cnt:
            talker_cnt = len(test_talker) - 1
        if test_talker[talker_cnt].strip('\n') != talker and talker != 'ALL':
            continue
        pred_act = pred[:5] # first 5 is act
        pred_attribute = pred[5:] # remaining is attribute
        binary = Binarizer(threshold=0.5)
        act_logit = one_hot(np.argmax(pred_act), "act")
        attribute_logit = binary.fit_transform([pred_attribute])
        if np.sum(attribute_logit) == 0:
            attribute_logit = one_hot(np.argmax(pred_attribute), "attribute")
        label = binary.fit_transform([label])
        ret_pred_outputs = np.append(ret_pred_outputs, np.append(act_logit, attribute_logit))
        ret_ground_truth = np.append(ret_ground_truth, label)
    return ret_pred_outputs, ret_ground_truth

if __name__ == '__main__':
    sess = tf.Session(config=config)
    max_seq_len = 40
    epoch = 30
    batch_size = 256

    data = slu_data()
    total_intent = data.total_intent
    total_word = data.total_word
    model = slu_model(max_seq_len, total_intent, args)
    sess.run(model.init_model)
    # read in the glove embedding matrix
    sess.run(model.init_embedding, feed_dict={model.read_embedding_matrix:data.embedding_matrix})
    test_f1_scores = list()

    # Train
    for cur_epoch in range(epoch):
        pred_outputs = list()
        labels = list()
        total_loss = 0.0
        for cnt in range(50):
            # get the data
            batch_nl, batch_intent, batch_distance = data.get_train_batch(batch_size)
            history_nl, history_nl_len, current_nl, current_nl_len, history_intent, ground_truth, history_distance = \
            process_batch(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent, batch_distance)
            _, intent_output, loss = sess.run([model.train_op, model.intent_output, model.loss],
                    feed_dict={
                        model.labels:ground_truth,
                        model.current_nl:current_nl,
                        model.current_nl_len:current_nl_len,
                        model.history_intent:history_intent,
                        model.dropout_keep_prob:0.75,
                        })
                
            total_loss += loss
            pred, truth = calculate_score(intent_output, ground_truth)
            pred_outputs = np.append(pred_outputs, pred)
            labels = np.append(labels, truth)

        #print "Epoch:", cur_epoch
        #print "training f1 score:", f1_score(pred_outputs, labels, average="binary")
        #print "training loss:", total_loss
        
        # Test
        batch_nl, batch_intent, batch_distance = data.get_test_batch()
        history_nl, history_nl_len, current_nl, current_nl_len, history_intent, ground_truth, history_distance = \
        process_batch(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent, batch_distance)
        # need to avoid OOM, so test only a batch one time
        test_output = None
        for i in range(0, len(batch_nl), batch_size):
            temp_test_output = sess.run([model.intent_output],
                feed_dict={
                    model.labels:ground_truth[i:i+batch_size],
                    model.current_nl:current_nl[i:i+batch_size],
                    model.current_nl_len:current_nl_len[i:i+batch_size],
                    model.history_intent:history_intent[i:i+batch_size],
                    model.dropout_keep_prob:1.0
                    })
            temp_test_output = np.squeeze(temp_test_output)
            if i == 0:
                test_output = temp_test_output
            else:
                test_output = np.vstack((test_output, temp_test_output))
        test_pred, test_label = calculate_score(test_output, ground_truth, args.target)
        f1sc = f1_score(test_pred, test_label, average="binary")
        #print "testing f1 score:", f1sc
        test_f1_scores.append(f1sc)
        
    #print "max test f1 score:", max(test_f1_scores)
    print(max(test_f1_scores))
    sess.close()
