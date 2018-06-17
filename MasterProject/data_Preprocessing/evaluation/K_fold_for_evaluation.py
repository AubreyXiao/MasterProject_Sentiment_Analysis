import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model

import os

#-----load the training datasets----

data = get_data.Datasets()

train_data = data.get_train_data()
test_data = data.get_test_data()
unlabeled_data = data.get_unlabeled_data()


#----------load the model-------------
model_oper = get_model.model()
index = "400features_30minwords_10window"

word2vec_model = model_oper.find_model(index)
print(word2vec_model)


dimension = 200


#--------------------------getcleandatastes-------------------------------------

seed = 2000

x_train = train_data["review"]
y_train = train_data["sentiment"]


train_cleaned_reviews = data.get_clean_review_lists(x_train)

#dimensinon=300
x_train = data.return_total_vector(train_cleaned_reviews, word2vec_model, dimension)


#print(ytrain = np.array(train_data["sentiment"]))

y_train = np.array(y_train)

#---------using K-fold to evaluate the model----------------------------

seed1 = 7

#define 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed1)
acc_scores = []

for train, validation in kfold.split(x_train,y_train):
    #interate test model
    #define model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=200))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train[train], y_train[train], epochs=10, batch_size=20, verbose=0)

    #evaluate the model
    scores = model.evaluate(x_train[validation],y_train[validation],verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
    acc_scores.append(scores[1]*100)

#average
print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))




























