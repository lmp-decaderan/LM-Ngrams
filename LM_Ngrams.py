"""
本程序是建立Ngram语言模型
"""
#!/usr/bin/python
#import division
from __future__ import division
import os
import re
import nltk
import sys
import getopt 
#import ngrams from nltk to help build ngrams
from nltk.util import ngrams

from collections import defaultdict
import codecs
import math

class LM_Ngrams():
    def __init__(self, model_path = None, traindata = None, traindata_path = None, grams_n = 3, Is_word_seg = True, model_save_path = None):
        
        self.traindata = []
        self.model = defaultdict(int)
        self.grams_n = grams_n
        self.model_save_path = model_save_path
        self.Is_word_seg =  Is_word_seg
        
        if model_path != None:
            self.__model__(model_path)
        if traindata_path != None:
            self.__read_traindata__(traindata_path)
        if traindata != None:
            if isinstance(traindata,list):
                self.traindata += traindata
        self.__train__()
        self.sum_model = sum(list(self.model.values()))
        self.__save_model__()
        
    def __model__(self,model_path):
        if not model_path.endswith(".model"):
            print("model read failure, please input right model path!")
        else:
            with codecs.open(model_path,'r','utf-8') as modelfile:
                print("model reading...")
                for line in modelfile.readlines():
                    tmp = line.strip()
                    tmp = tmp.split("\t")
                    #print(tmp[:-1])
                    tmp_tuple = tuple([int(key) for key in tmp[:-1]])
                    #print(tmp[-1])
                    self.model[tmp_tuple] = int(tmp[-1])
            print('model read finished!')
    def __read_traindata__(self,traindata_path):
        '''read train data from text'''
        with codecs.open(traindata_path,'r','utf-8') as datafile:
            print("train data reading...")
            for line in datafile.readlines():
                self.traindata.append(line.strip())
        print('train data read finished!')

    def __train__(self):
        '''Building model'''
        print("Building model ...")
        for line in self.traindata:
            if self.Is_word_seg == True:
                dictsrc = [1 for i in range(self.grams_n - 1)] + line + [2 for i in range(self.grams_n - 1)]
                for word in ngrams(dictsrc,self.grams_n):
                    self.model[word] += 1
            else:
                #append the beginning and ending shown from lecture notes
                dictsrc = list(line.split(' '))
                #print(dictsrc)
                dictsrc = [1 for i in range(self.grams_n - 1)] + dictsrc + [2 for i in range(self.grams_n - 1)]
	            #build the 4-grams and add it to the dictionary
                #dictsrc = list(dictsrc.split(' '))
                for word in ngrams(dictsrc,self.grams_n):
                    self.model[word] += 1
    
    def __save_model__(self):
        if self.model_save_path == None:
            path = os.path.join(os.path.join(os.getcwd(),'model'),str(self.grams_n)+'grams.model')
        else:
            path = os.path.join(self.model_save_path,str(self.grams_n)+'grams.model')
        with codecs.open(path,'w','utf-8') as writefile:
            for key,vaule in self.model.items():
                output_tmp = ''
                for word_seg in key:
                    output_tmp += str(word_seg)
                    output_tmp += '\t'
                output_tmp += str(vaule)
                output_tmp += '\n'
                writefile.write(output_tmp)
    def score(self,test_data):
        smoothing = False
        smoothing_count = 0
        #dictsrc = list(test_data.split(' '))
        if self.Is_word_seg == True:
            dictsrc = test_data
        else:
            dictsrc = list(test_data.split(' '))
        dictsrc = [1 for i in range(self.grams_n - 1)] + test_data + [2 for i in range(self.grams_n - 1)]
        scoredict = defaultdict(int)
        for word in ngrams(dictsrc,self.grams_n):
            scoredict[word] = self.model[word]
            if scoredict[word] == 0:
                smoothing_count += 1
                smoothing = True
        for key,value in scoredict.items():
            scoredict[key] = value + 1
        score_double = 1.0
        sentence_length = 0
        for key,value in scoredict.items():
            sentence_length += 1
            score_double *= value*1.0/(self.sum_model+smoothing)
        return -math.log10(max(score_double,0.0000000000001))/(1.0*sentence_length)
    def scoretext(self,data_path):
        score_list = []
        with codecs.open(data_path,'r','utf-8') as datafile:
            print("test data score...")
            for line in datafile.readlines():
               score_tmp = self.score(line)
               print("{} is {}!".format(line,score_tmp))
               score_list.append(score_tmp)
        return score_list
    def score_count(self,test_data):
        
        #dictsrc = list(test_data.split(' '))
        if self.Is_word_seg == True:
            dictsrc = test_data
        else:
            dictsrc = list(test_data.split(' '))
        dictsrc = [1 for i in range(self.grams_n - 1)] + test_data + [2 for i in range(self.grams_n - 1)]
        #print(dictsrc)
        scoredict = defaultdict(int)
        no_word_count = 0
        word_count = 0
        for word in ngrams(dictsrc,self.grams_n):
            word_count += 1
            scoredict[word] = self.model[word]
            if scoredict[word] == 0:
                no_word_count += 1
        return no_word_count*1.0/word_count
        
            
