from sklearn.feature_extraction.text import  TfidfVectorizer
import pandas as pd
#import pandas
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk.data
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.cross_validation import train_test_split
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


#--------------------load the pickel-------------------------------------------------------

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


#-------------------function---------------------------------------

# get the average feature vector of each review.
# Remove stops words in each review
# define a 1* 300 dimensional array
# Loop over each word in the review and find the corresponding vector and add the vector to the total( use tf-idf)
# Compute a weighted average where each weight gives the importance of the word with respect to the corpus.
# Divide the result by the number of words to get the average vector of each.

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


#Array :Review的数量 * dimensiona
#构造一个reviewFeaturevector[count ] = feature

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


#plot word cloud
def plot_word_cloud(terms):
    text = terms.index
    text = ' '.join(list(text))

    #lower the max fontsize
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize = (25,25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

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
dir = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/sentiment_classifier/kaggle_dataset/"

train_data = pd.read_csv(dir +"labeledTrainData.tsv",header = 0, delimiter='\t', quoting=3)
unlabeled_data = pd.read_csv(dir + "unlabeledTrainData.tsv",header = 0, delimiter='\t',quoting=3)
test_data = pd.read_csv(dir + "testData.tsv",header=0, delimiter='\t',quoting=3)
test_labels = pd.read_csv(dir + "model_prediction.csv",sep=',',header=0)

# print("---------")
# print(test_labels["id"])
# print(test_labels["sentiment"])
#
#
# print("--------")
#
# print(len(train_data["review"]))
# print(len(unlabeled_data["review"]))
# print(len(test_data["review"]))

#-----split the dataset-----cross validation-----

seed = 2000

x = train_data["review"]
y = train_data["sentiment"]

#-----split the train dataset-------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.01, random_state=seed)


print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),                                                                     (len(x_train[y_train == 0]) / (len(x_train)*1.))*100, (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
# print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),(len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,(len(x_test[y_test == 1]) / (len(x_test)*1.))*100))


#-------see how the sentiment analysis results of textblob-----

# tbresult =  [TextBlob(i).sentiment.polarity for i in x_validation]
# tbprediced = [0 if n<0 else 1 for n in tbresult]
#
# conmat = np.array(confusion_matrix(y_validation,tbprediced,labels=[1,0]))
#
# confusion = pd.DataFrame(conmat, index=['positive','negative'],columns=['predicted_positive','predicted_negative'])
#
# print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbprediced)*100))
# print("-"*80)
# print("Confusion Matrix\n")
# print(confusion)
# print("-"*80)
# print("Classification Report\n")
# print(classification_report(y_validation, tbprediced))
#
#
# tbresult =  [TextBlob(i).sentiment.polarity for i in x_test]
# tbprediced = [0 if n<0 else 1 for n in tbresult]
#
# conmat = np.array(confusion_matrix(y_test,tbprediced,labels=[1,0]))
#
# confusion = pd.DataFrame(conmat, index=['positive','negative'],columns=['predicted_positive','predicted_negative'])
#
# print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_test, tbprediced)*100))
# print("-"*80)
# print("Confusion Matrix\n")
# print(confusion)
# print("-"*80)
# print("Classification Report\n")
# print(classification_report(y_test, tbprediced))



# ##--------------ingegrates all the sentences--------------------------------------
all_labeled_sentences = []
all_unlabeled_sentences = []
all_sentences = []
#
for review in train_data["review"]:
     all_labeled_sentences += transfer_review_to_sentences(review, sent_detector)

print(len(all_labeled_sentences))
#
for unlabel_review in unlabeled_data["review"]:
     all_unlabeled_sentences += transfer_review_to_sentences(unlabel_review, sent_detector)

all_sentences = all_unlabeled_sentences + all_labeled_sentences


#---------------------------build-up--tf-idf-matrix--------------------------

# print("building up the tf-icf matrix")
# vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
# matrix = vectorizer.fit_transform(all_sentences)
#
# print(matrix.shape)
#
# tfidf = dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))
# tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf),orient='index')
# tfidf.columns = ['tfidf']


#-----------load the model-----------------------------------------------------

modelname = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/300features_40minwords_10window"
model = Word2Vec.load(modelname)
dimension = 300


#--------------------------getcleandatastes-------------------------------------
train_reviews = x_train
#validation_reviews = x_validation
test_reviews = x_test



train_cleaned_reviews = get_clean_review_lists(train_reviews)
#validation_cleaned_reviews = get_clean_review_lists(validation_reviews)
test_cleaned_reviews = get_clean_review_lists(test_reviews)

# train_vecs = np.concatenate([build_review_vector(z,dimension) for z in tqdm(map(lambda x:x, train_cleaned_reviews))])

#dimensinon=300
trained_vector = return_total_vector(train_cleaned_reviews, model, dimension)
#validation_vector = return_total_vector(validation_cleaned_reviews, model, dimension)
test_vetor = return_total_vector(test_cleaned_reviews,model,dimension)

# print("trained_vector")
# print(trained_vector)
# print(test_vetor.shape)
#
# print("test_vector")
# print(test_vetor)
# print(test_vetor.shape)

#print(ytrain = np.array(train_data["sentiment"]))
y_train = np.array(y_train)
#y_validation = np.array(y_validation)
y_test = np.array(y_test)

#------------------build up a neural network classifier------------------------
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=300))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(trained_vector,y_train, validation_split=0.02,shuffle=True, epochs=20,batch_size=128,verbose=2)



score = model.evaluate(test_vetor,y_test,batch_size=125,verbose=2)
print(score[1])

#---------using K-fold to evaluate the model----------------------------




























