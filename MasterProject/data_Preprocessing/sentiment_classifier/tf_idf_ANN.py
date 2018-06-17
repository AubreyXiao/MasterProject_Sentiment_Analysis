from sklearn.feature_extraction.text import  TfidfVectorizer
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk.data
import numpy as np
import csv

#-----------------initialization-------------------------------------------------------

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
dir = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/sentiment_classifier/kaggle_dataset/"







#-------------------function---------------------------------------

def return_averaged_vector_review(dimension,review,model):

    review_vector = np.zeros((dimension,),dtype="float32")

    vocab = set(model.wv.index2word)

    num_words = 0

    for word in review:
        if(word in vocab):
            num_words += 1
            review_vector = np.add(review_vector,model[word])


    #divide the result
    review_vector = np.divide(review_vector,num_words)

    return review_vector

#gather all the reviews in the traning datasets


def return_total_vector(reviews, model, dimension):

    #should be 25000(train reviews) * 300(dimension)
    matrix = np.zeros((len(reviews),dimension),dtype="float32")

    count = 0

    for review in reviews:

        if (count % 1000 == 0):
            print("Review %d of %d" % (count, len(reviews)))

        matrix[count] = return_averaged_vector_review(dimension,review,model)
        count = count +1

    return matrix


#get all the review
def get_clean_review_lists(review_datasets):
    clean_review_datasets = []
    for review in review_datasets:
        clean_review_datasets.append(clean_text(review))
    return clean_review_datasets



#preprocess the text and split each review to a list of tokens
def clean_text(review):
    #1: remove the tags
    review = BeautifulSoup(review).get_text()
    #2.remove the non alpha
    text = re.sub("[^a-zA-Z]", " ",review)
    #3.remove split the tokens
    lowercase =  text.lower().split()
    #remove the stopwords
    stops = set(stopwords.words("English"))
    words = [w for w in lowercase if not w in stops]

    return words


#split each review into sentences and clean the sentences -> to a list
def transfer_review_to_sentences(review, sent_detector):

    #split the review to a list
    sentences = sent_detector.tokenize(review.strip())

    sentencs_list = []
    for sentence in sentences:
        if(len(sentence)>0):
         words = clean_text(sentence)
         sentencs_list.append(words)

    return sentencs_list

# #given a list of tweet tokens, creates an averaged tweet vector
# def build_review_vector(tokens, dimensionality):
#     vec = np.zeros(dimensionality).reshape((1, dimensionality))
#
#     count = 0
#
#     for word in tokens:
#         if word in tokens:
#             try:
#                 vec += model[word].reshape((1,dimensionality))*tfidf[word]
#                 count += 1.
#             except KeyError:
#                 continue
#
#         if(count%1000==0):
#             print("Review %d of %d" % (count, count))
#
#     if(count!=0):
#         vec /= count
#
#
#     return vec
#
# train_vector = f

#-------------------------main---------------------------------------------------

#------------load the training datasets---------------------------------------------------


train_data = pd.read_csv(dir +"labeledTrainData.tsv",header = 0, delimiter='\t', quoting=3)
unlabeled_data = pd.read_csv(dir + "unlabeledTrainData.tsv",header = 0, delimiter='\t',quoting=3)
test_data = pd.read_csv(dir + "testData.tsv",header=0, delimiter='\t',quoting=3)


##--------------ingegrates all the sentences--------------------------------------
all_labeled_sentences = []
all_unlabeled_sentences = []
all_sentences = []

for review in train_data["review"]:
    all_labeled_sentences += transfer_review_to_sentences(review, sent_detector)

print(len(all_labeled_sentences))

for unlabel_review in unlabeled_data["review"]:
    all_unlabeled_sentences += transfer_review_to_sentences(unlabel_review, sent_detector)

all_sentences = all_unlabeled_sentences + all_labeled_sentences


#---------------------------build-up--tf-idf-matrix--------------------------

print("building up the tf-icf matrix")
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=2)
matrix = vectorizer.fit_transform(all_sentences)

print(matrix.shape)

tfidf = dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf),orient='index')
tfidf.to_csv("tfidf_dic.csv")
tfidf.columns = ['tfidf']
print("vocab size:", len(tfidf))
print(tfidf)


with open("tfidf_dict.csv",'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in tfidf.items():
        writer.writerow([key, value])


#
# #-----------load the model-----------------------------------------------------
#
# modelname = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/300features_40minwords_10window"
# model = Word2Vec.load(modelname)
# dimension = 300
#
