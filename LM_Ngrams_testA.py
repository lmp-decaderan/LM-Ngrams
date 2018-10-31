# -*-coding:utf-8 -*-

import LM_Ngrams as lm_model
import sentencepiece as spm
import os
#import io
import codecs
import sys
import jieba
import linecache


#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')



def jieba_cws(string):
    seg_list = jieba.cut(string.strip().encode('utf8').decode('utf8'))
    return u' '.join(seg_list).encode('utf8')

def traindata_file(traindata_path):
    lm = lm_model.LM_Ngrams(traindata_path = traindata_path)

    return lm

def traindata_list(traindata, Is_word_seg = True):
    lm = lm_model.LM_Ngrams(traindata = traindata, grams_n = 3, Is_word_seg = True)

    return lm

def subword_seg(model_path = None ,data_line = None, data_path = None):
    sp = spm.SentencePieceProcessor()
    if sp.Load(model_path):
        print("subword model load suncess!")
    result = []
    if data_line is not None:
        for line in data_line:
            if line is not None:
                result.append(sp.EncodeAsIds(line))
        print("subword segment is over!")
        return result
    with codecs.open(data_path,'r','utf-8') as datafile:
        count = 0
        for line in datafile.readlines():
            if line is not None:
                result.append(sp.EncodeAsIds(line))
                count += 1
            if count >13000000:
                break
    print("subword segment is over!")
    return result
    

def test_score(model,test_data):
    score_list = []
    if model !=None:
        for line in test_data:
            score_list.append(model.score(line))
    return score_list

def listdir(path):
    result = []
    for file in os.listdir(path): 
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path) and file_path.endswith('.decodes'):
            result.append(file_path)
    return result

if __name__ == '__main__':
    
    #data_path = os.path.join(os.getcwd(),'data/20180827_seg.zh')
    #seg_word_finished = subword_seg(model_path = os.path.join(os.getcwd(),'model/sent_all_65k_zh.model'),data_path = data_path)
    #lm = traindata_list(seg_word_finished, Is_word_seg = True)
    #print(seg_word_finished)
    lm = lm_model.LM_Ngrams(model_path= os.path.join(os.path.join(os.getcwd(),'model'),'2grams.model'), grams_n = 2)
    filelist = listdir(os.path.join(os.getcwd(),'data/result_data'))
    result_dict = {}
    for file in filelist:
        print("{} is processsing...".format(file))
        result_dict_tmp = []
        with codecs.open(file,'r','utf-8') as read_data_file:
            line_cws = []
            for line in read_data_file.readlines():
                if line.strip() is not None:
                    #line1 = "一对丹顶鹤正监视着它们的筑巢领地"
                    line_cws.append(jieba_cws(line.strip()))
            #print(line_cws)
        seg_word_finished_tmp = subword_seg(model_path = os.path.join(os.getcwd(),'model/sent_all_65k_zh.model'),data_line = line_cws)
        for seg_word_line in  seg_word_finished_tmp:
            #print(seg_word_line)
            #with codecs.open(os.path.join(os.getcwd(),'debug/fileinf.txt'),'w','utf-8') as debug_data_file:
            #    debug_data_file.write(" ".join(str(s) for s in seg_word_line))
            result_dict_tmp.append(lm.score_count(seg_word_line))
        result_dict[file] = result_dict_tmp
        print("{} is success...".format(file))
    result_list = [filelist[-1] for i in range(8010)]
    result_list_flag = [1.0 for i in range(8010)]
    for key,values in result_dict.items():
        for value_keys,value_values in enumerate(values):
            #print(value_values)
            if value_values < result_list_flag[value_keys]:
                #print(value_values)
                result_list[value_keys] = key
                result_list_flag[value_keys] = value_values
    
    print(result_list)

    with codecs.open(os.path.join(os.getcwd(),'data/good_ref.zh'),'w','utf-8') as output_data_file:
        for key,value in enumerate(result_list):
            output_data_file.write(linecache.getline(value, key))


    #test_data = '我 喜欢 打 篮球 。'
    #print(lm.score(test_data))


