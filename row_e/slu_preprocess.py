import numpy as np
import random
from collections import defaultdict

class slu_data():
    def __init__(self):
        train_nl = open('../../Data/train/seq.in', 'r')
        valid_nl = open('../../Data/valid/seq.in', 'r')
        test_nl = open('../../Data/test/seq.in', 'r')
        train_intent = open('../../Data/train/intent', 'r')
        valid_intent = open('../../Data/valid/intent', 'r')
        test_intent = open('../../Data/test/intent', 'r')
        train_talker = open('../../Data/train/talker', 'r')
        train_info = open('../../Data/train/info', 'r')
        test_info = open('../../Data/test/info', 'r')
        self.train_dist = self.read_info(train_info)
        self.test_dist = self.read_info(test_info)
        self.intent_act_dict = None
        self.intent_attri_dict = None
        self.total_intent = None
        self.total_word = None
        self.train_tourist_indices = list()
        self.train_guide_indices = list()
        self.train_guide_indices = list()
        self.get_talker(train_talker)
        self.train_intent = self.convertintent2id(train_intent)
        self.valid_intent = self.convertintent2id(valid_intent)
        self.test_intent = self.convertintent2id(test_intent)
        glove = open('/home/yuan0925pj/glove/glove.6B.200d.txt', 'r')
        self.word2id = defaultdict()
        self.id2word = defaultdict()
        self.embedding_matrix = None
        self.read_GloVe(glove)
        self.train_data, _ = self.convertnl2id(train_nl)
        self.valid_data, _ = self.convertnl2id(valid_nl)
        self.test_data, self.raw_test_data = self.convertnl2id(test_nl)
        assert len(self.train_data) == len(self.train_intent)
        assert len(self.valid_data) == len(self.valid_intent)
        assert len(self.test_data) == len(self.test_intent)
        #print 'train data size:', len(self.train_data)
        #print 'valid data size:', len(self.valid_data)
        #print 'test data size:', len(self.test_data)
        self.train_batch_indices = [i for i in range(len(self.train_data))]
        self.valid_batch_indices = [i for i in range(len(self.valid_data))]
        self.test_indices = [i for i in range(len(self.test_data))] # no shuffle
    
    def read_info(self, data_file):
        ret_dist = list()
        for line in data_file:
            dist = line.split("***next***")[:-1]
            ret_dist.append(dist)
        return ret_dist

    def get_talker(self, data_file):
        for idx, line in enumerate(data_file):
            talker = line.strip('\n')
            if talker == 'Tourist':
                self.train_tourist_indices.append(idx)
            elif talker == 'Guide':
                self.train_guide_indices.append(idx)
            else:
                print("cannot be here!")
                exit(1)

    def get_train_batch(self, batch_size, role=None):
        """ returns a 3-dim list, where each row is a batch contains histories from tourist and guide"""
        if role == None:
            random.shuffle(self.train_batch_indices)
            batch_indices = self.train_batch_indices[:batch_size]
        elif role == 'Tourist':
            random.shuffle(self.train_tourist_indices)
            batch_indices = self.train_tourist_indices[:batch_size]
        elif role == 'Guide':
            random.shuffle(self.train_guide_indices)
            batch_indices = self.train_guide_indices[:batch_size]

        ret_nl_batch = list()
        ret_intent_batch = list()
        ret_dist_batch = list()
        for batch_idx in batch_indices:
            nl_sentences = self.train_data[batch_idx]
            intent = self.train_intent[batch_idx]
            ret_nl_batch.append(nl_sentences)
            ret_intent_batch.append(intent)
            dist = self.train_dist[batch_idx]
            ret_dist_batch.append(dist)
        return ret_nl_batch, ret_intent_batch, ret_dist_batch

    def get_valid_batch(self, batch_size):
        """ returns a 3-dim list, where each row is a batch contains histories from tourist and guide"""
        random.shuffle(self.valid_batch_indices)
        batch_indices = self.valid_batch_indices[:batch_size]
        ret_nl_batch = list()
        ret_intent_batch = list()
        for batch_idx in batch_indices:
            nl_sentences = self.valid_data[batch_idx]
            intent = self.valid_intent[batch_idx]
            ret_nl_batch.append(nl_sentences)
            ret_intent_batch.append(intent)
        return ret_nl_batch, ret_intent_batch

    def get_test_batch(self):
        """ returns a 3-dim list, where each row is a batch contains histories from tourist and guide"""
        batch_indices = self.test_indices
        ret_nl_batch = list()
        ret_intent_batch = list()
        ret_dist_batch = list()
        for batch_idx in batch_indices:
            nl_sentences = self.test_data[batch_idx]
            intent = self.test_intent[batch_idx]
            ret_nl_batch.append(nl_sentences)
            ret_intent_batch.append(intent)
            dist = self.test_dist[batch_idx]
            ret_dist_batch.append(dist)
        return ret_nl_batch, ret_intent_batch, ret_dist_batch

    def convertintent2id(self, data_file):
        intent_corpus = list()
        for line in data_file:
            temp_intent = line.strip('\n').split('***next***')[:-1]
            temp_intent = list(map(lambda x:x.strip(' ').lstrip(' '), temp_intent))
            intent_corpus.append(temp_intent)
        
        if self.intent_act_dict is None or self.intent_attri_dict is None:
            assert self.intent_act_dict is None and self.intent_attri_dict is None
            # build intent dict
            act_dict = defaultdict()
            attri_dict = defaultdict()
            for intents in intent_corpus:
                for intent in intents:
                    act_attri = intent.split('-')
                    act = act_attri[0] # a string
                    attributes = act_attri[1:] # a list, may contain several attributes
                    if act not in act_dict:
                        act_dict[act] = len(act_dict)
                    for attri in attributes:
                        if attri not in attri_dict:
                            attri_dict[attri] = len(attri_dict)
            self.intent_act_dict = act_dict
            self.intent_attri_dict = attri_dict
            self.total_intent = len(act_dict)+len(attri_dict)
        
        # convert act and attributes to id
        ret_intent = list()
        for intents in intent_corpus:
            temp_list = list()
            for intent in intents:
                act_attri = intent.split('-')
                act = act_attri[0] # a string
                attributes = act_attri[1:] # a list, may contain several attributes
                t = (self.intent_act_dict[act], [self.intent_attri_dict[attri]+len(self.intent_act_dict) for attri in attributes])
                temp_list.append(t)
            ret_intent.append(temp_list)
        return ret_intent

    def read_GloVe(self, glove):
        # read in GloVe dict
        #print "Reading from GloVe..."
        embedding_matrix = list()
        word2id = defaultdict()
        id2word = defaultdict()
        for line in glove:
            splitLine = line.strip('\n').split(' ')
            word = splitLine[0]
            word2id[word] = len(word2id)
            id2word[word2id[word]] = word
            embedding = [float(val) for val in splitLine[1:]]
            embedding_matrix.append(embedding)
        #print "Done.", len(embedding_matrix)," words loaded from GloVe!"
        self.total_word = len(embedding_matrix)
        self.word2id = word2id
        self.id2word = id2word
        self.embedding_matrix = embedding_matrix

    def convertnl2id(self, data_file):
        # nl_corpus is a list, where one row contains all the cleaned history nl strings list
        nl_corpus = list()
        for line in data_file:
            temp_nl = line.strip('\n').split('***next***')[:-1] # temp_nl contains many sentences
            nl = self.clean_nl(temp_nl)
            nl_corpus.append(nl)
        
        # start from idx 1, since 0 is for <unk>
        data = list()
        for nl_sentences in nl_corpus:
            one_training_data = list()
            for sentence in nl_sentences:
                one_utterance = list()
                for word in sentence.split(' '):
                    word_id = None
                    if word not in self.word2id:
                        word_id = len(self.word2id) - 1 # the last word in GloVe is <unk>
                    else:
                        word_id = self.word2id[word]
                    one_utterance.append(word_id)
                one_training_data.append(one_utterance)
            data.append(one_training_data)
        return data, nl_corpus    

    def clean_nl(self, temp_nl):
        ret = list()
        for sentence in temp_nl:
            # remove some puntuation marks
            temp = sentence.replace('~', '').strip(' ')
            # restore abbreviations to their original forms
            if '\'m' in temp:
                temp = temp.replace('\'m', ' am')
            if '\'re' in temp:
                temp = temp.replace('\'re', ' are')
            if '\'ll' in temp:
                temp = temp.replace('\'ll', ' will')
            if '\'s' in temp:
                temp = temp.replace('\'s', ' is')
            if '\'d' in temp:
                temp = temp.replace('\'d', ' would')
            if '\'ve' in temp:
                temp = temp.replace('\'ve', ' have')
            if 'don\'t' in temp:
                temp = temp.replace('don\'t', 'do not')
            if 'doesn\'t' in temp:
                temp = temp.replace('doesn\'t', 'does not')
            if 'hasn\'t' in temp:
                temp = temp.replace('hasn\'t', 'has not')
            if 'haven\'t' in temp:
                temp = temp.replace('daven\'t', 'have not')
            if 'wouldn\'t' in temp:
                temp = temp.replace('wouldn\'t', 'would not')
            # remove uh, um
            if 'uh' in temp:
                temp = temp.replace('uh', '')
            if 'um' in temp:
                temp = temp.replace('um', '')
            if '  ' in temp:
                temp = temp.replace('  ', ' ')
            temp = temp.strip(' ').lstrip(' ')
            ret.append(temp) 

        return ret

if __name__ == '__main__':
    slu_data()
