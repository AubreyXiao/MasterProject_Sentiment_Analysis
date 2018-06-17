import numpy as np
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model
from sklearn.cross_validation import train_test_split

from keras.layers import Dense,Input,Flatten
from keras.layers import Conv1D,MaxPooling1D,Embedding,Merge,Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

#define word2vec model
model = "400dim_30min_10windows.txt"
Model_MAN = get_model.model()

#final parmeters
WORDS_NUM = 100
SEN_NUM = 15
MAX_NB_WORDS = 20000
DIM = 400

# class AttLayer(Layer):
#     def __init__(self, **kwargs):
#         self.init = initializers.get('normal')
#         # self.input_spec = [InputSpec(ndim=3)]
#         super(AttLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#         # self.W = self.init((input_shape[-1],1))
#         self.W = self.init((input_shape[-1],))
#         # self.input_spec = [InputSpec(shape=input_shape)]
#         self.trainable_weights = [self.W]
#         super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
#
#     def call(self, x, mask=None):
#         eij = K.tanh(K.dot(x, self.W))
#
#         ai = K.exp(eij)
#         weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
#
#         weighted_input = x * weights.dimshuffle(0, 1, 'x')
#         return weighted_input.sum(axis=1)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[-1])
#



#get the train_data
data = get_data.Datasets()
train_data = data.get_train_data()

# declare the lists
reviews = []
sentences = []
labels = []


#separate and cleaning the dataset
for review in train_data["review"]:
    #append all the reviews in the list"reviews"
    cleaned_review = data.clean_text_to_text(review)
    reviews.append(cleaned_review)
    #append all the sentenes to the list sentences
    review_sentences = tokenize.sent_tokenize(cleaned_review)
    sentences.append(review_sentences)

#append the label to the list
for sentiment in train_data["sentiment"]:
    labels.append(sentiment)


#sequence the review
#create a tokenizer and limit only dealt with top 20000 words
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(reviews)

#each review = 15(lines) * 100words
texts_matrix  = np.zeros((len(reviews),SEN_NUM,WORDS_NUM),dtype='int32')

print(texts_matrix.shape)
#遍历每个review 中的句子以及每个句子中的词， 用唯一的integer去表示, 生成matrix 来表示所有的review, shape(25000, 15, 100)
for index1, review in enumerate(sentences):
    for index2, sentence in enumerate(review):
        #限制句子的数量<1
        if(index2<SEN_NUM):
            tokens = text_to_word_sequence(sentence)
            count = 0
            for _,w in enumerate(tokens):
                if(count<100 and tokenizer.word_index[w]<MAX_NB_WORDS):
                    texts_matrix[index1,index2,count] = tokenizer.word_index[w]
                    count = count+1


#shuffle the data
word_index = tokenizer.word_index
print('total %s unique tokens' % len(word_index))

labels = to_categorical(np.asarray(labels))
print(labels.shape)

#split the tranning and validatino data
x_train, x_val, y_train, y_val = train_test_split(texts_matrix,labels,test_size=0.2)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


#load the embedding layer

#initial create a embedding matrix to save the weights of the word vector

word2vec_dict = Model_MAN.load_embedding(model)
embedding_matrix = np.random.random((len(word_index)+1, DIM))
count = 0
for word, index in word_index.items():
    vector = word2vec_dict.get(word)
    if(vector is not None):
        count = count + 1
        embedding_matrix[index] = vector


print(count)



#create a embedding layer
wordvec_embedding = Embedding(len(word_index)+1,DIM, weights=[embedding_matrix],input_length=WORDS_NUM, trainable=True)


#create the lstm classifer
sentence_input = Input(shape=(WORDS_NUM,), dtype='int32')
embedded_sequences = wordvec_embedding(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(SEN_NUM,WORDS_NUM), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
preds = Dense(2, activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epoch=10, batch_size=50)






#build up the hierarchical neural network
# #create the sentence input
# input_sen = Input(shape=(WORDS_NUM,),dtype='int32')
# sentence_sequence = wordvec_embedding(input_sen)
# l_lstm = Bidirectional(GRU(100,return_sequences=True))(sentence_sequence)
# l_dense = TimeDistributed(Dense(200))(l_lstm)
# l_att = AttLayer()(l_dense)
# sen_encoder = Model(input_sen,l_att)
#
# #create review imnput
# input_review = Input(shape=(SEN_NUM,WORDS_NUM),dtype='int32')
# review_encoder = TimeDistributed(sen_encoder)(input_review)
# l_lstm_sent = Bidirectional(GRU(100,return_sequences=True))(review_encoder)
# l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
# l_att_sent = AttLayer()(l_dense_sent)
# preds = Dense(2,activation='softmax')(l_att_sent)
# model = Model(input_review,preds)
#
# #compile the model
# model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
#
# #fit the model
# print("fitting the_HAN model")
# print(model.summary())
# model.fit(x_train,y_train,validation_data=(x_val,y_val),nb_epoch=10,batch_size=50)















