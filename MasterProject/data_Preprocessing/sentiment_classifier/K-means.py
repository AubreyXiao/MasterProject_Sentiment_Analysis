import time
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model



#create bag of centroids
def create_centroids(words, word_centroid_map):
    num_centroids = max(word_centroid_map.values())+1

    centroids_matrix = np.zeros(num_centroids, dtype="float32")

    for word in words:
        if(word in word_centroid_map):
            index = word_centroid_map[word]
            centroids_matrix[index] += 1
    return centroids_matrix

#define the model

model1 ="/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/300features_30minwords_5window"
model2 = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/300features_30minwords_10window"
model3 = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/300features_30minwords_20window"
model4 = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/400features_30minwords_10window"
model5 ="/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/300features_40minwords_10window"
model6 = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/100features_30minwords_10window"
model7 = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/HAN_Classifier/200features_30minwords_10window"

#----load the model-------
model = get_model.model()

model = model.get_model(model3)



#-------------start time ------
start = time.time()



#-----set 5words per cluster----
wordvectors = model.wv.syn0
print(wordvectors)
num_clusters = wordvectors.shape[0]/5
print(num_clusters)

num_clusters = int(num_clusters)
print(num_clusters)
#initialize a k-means
kmean_cluster = KMeans(n_clusters=num_clusters)
idx = kmean_cluster.fit_predict(wordvectors)

end = time.time()
elapsed = end - start

print("Time taken for Kmeans clustering:", elapsed, "seconds.")

#create a word /index dict
word_centroid_map = dict(zip(model.wv.index2word, idx))

for cluster in range(0,10):
    print("\nCluster %d" % cluster)
    words=[]
    for key, item in word_centroid_map.items():
        if(item == cluster):
            words.append(key)
    print(words)




#_---------------------get reviews------------------

#---load the datasetets-----
data = get_data.Datasets()

train_data = data.get_train_data()
test_data = data.get_test_data()
unlabeled_data = data.get_unlabeled_data()

#-------get clean datasets--------------------------


cleaned_train_datasets = data.get_clean_review_lists(train_data["review"])
cleaned_test_datastes = data.get_clean_review_lists(test_data["review"])


#-------create train bags of centroids----------------
train_count = 0
train_matrix = np.zeros((train_data["review"].size, num_clusters), dtype="float32")

for word_lists in cleaned_train_datasets:

    train_matrix[train_count] =  create_centroids(word_lists, word_centroid_map)
    train_count += 1


#-------create test bags of centroids----------------

test_count = 0
test_matrix = np.zeros((test_data["review"].size, num_clusters),dtype="float32")

for words in cleaned_test_datastes:
    test_matrix[test_count] = create_centroids(words, word_centroid_map)
    test_count += 1


#------------fit in the randdom forest tree--------------

forest = RandomForestClassifier(n_estimators=100)

#fit the forest
forest = forest.fit(train_matrix, train_data["sentiment"])
result = forest.predict(test_matrix)

#write the output
output = pd.DataFrame(data={"id": test_data["id"], "sentiment":result})
output.to_csv("k-means_sentiment.csv",index=False,quoting=3)






























