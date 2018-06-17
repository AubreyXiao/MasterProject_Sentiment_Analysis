import logging
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk import tokenize
from gensim.models import word2vec

from MasterProject.data_Preprocessing.Datasets import get_data

#get the train_data
data = get_data.Datasets()
train_data = data.get_train_data()
unsup_data = data.get_unlabeled_data()

# declare the lists
reviews = []
sentences = []
train_sentences = []
unsup_sentences = []
labels = []
all_sentences = []

#separate and cleaning the dataset
for review in train_data["review"]:
    #append all the reviews in the list"reviews"
    cleaned_review = data.clean_text_to_text(review)
    review_sentences = tokenize.sent_tokenize(cleaned_review)
    train_sentences.append(review_sentences)
    for sentence in review_sentences:
        if(len(sentence)>0):
            tokens = text_to_word_sequence(sentence)
            tokens = [token for token in tokens if token.isalpha()]
            sentences.append(tokens)

for review in unsup_data["review"]:
    cleaned_review = data.clean_text_to_text(review)
    review_sentences = tokenize.sent_tokenize(cleaned_review)
    for sentence in review_sentences:
        if(len(sentence)>0):
            tokens = text_to_word_sequence(sentence)
            tokens = [token for token in tokens if token.isalpha()]
            unsup_sentences.append(tokens)

# print(len(sentences))
# print(len(unsup_sentences))
# print(len(sentences)+len(unsup_sentences))

all_sentences = sentences + unsup_sentences

#set the format
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#create a model
num_features = 300 #dimensionality
min_word_count = 30 #filter frequence>40
num_workers = 8 # threads to run in parrell
context = 10 #contect window size
downsampling = 1e-3

print("Training models!")

model = word2vec.Word2Vec(all_sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

#init_sims wiil make the model memory efficient
model.init_sims(replace=True)

#save the modle
filename = "200features_30minwords_10window"
model.save(filename)

#save the modle in text form
filename1 = '200dim_30min_10windows.txt'
model.wv.save_word2vec_format(filename1,binary=False)
