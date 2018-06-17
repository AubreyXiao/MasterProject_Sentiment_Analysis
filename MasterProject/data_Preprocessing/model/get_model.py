from gensim.models import Word2Vec
import os
import numpy as np


class model():


    #1:get model
    def get_model(modelname):

        model = Word2Vec.load(modelname)
        return model

    #get the word2vec model by index
    def find_model(self,index):

        model_dir = "../HAN_Classifier/word2vec_model"
        modle_txt_dir = "../HAN_Classifier/word2vec_model_txt"

        if(index.endswith(".txt")):
            dir = modle_txt_dir
            modelname = dir + "/"+ index
            print(modelname)
            return  modelname
        else:
            dir = model_dir
            modelname = dir + "/" + index
            word2vec_model = model.get_model(modelname)
            print("get word2vec_model!")
            return word2vec_model

    # def load the embedding
    def load_embedding(self,name):
        model_mani = model()
        modelname = model_mani.find_model(name)
        file = open(modelname, 'r')
        lines = file.readlines()[1:]
        file.close()

        embedding = dict()
        for line in lines:
            splits = line.split()
            word = splits[0]
            value = splits[1:]

            embedding[word] = np.asarray(value, dtype='float32')

        return embedding

