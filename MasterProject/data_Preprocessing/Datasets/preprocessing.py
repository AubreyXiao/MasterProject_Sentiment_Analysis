from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from wordcloud import WordCloud
from matplotlib import pyplot as plt


class preprocessing():


    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')



    def return_averaged_vector_review(self,dimension,review,model):

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


    def return_total_vector(self, reviews, model, dimension):

        #should be 25000(train reviews) * 300(dimension)
        matrix = np.zeros((len(reviews),dimension),dtype="float32")

        count = 0

        for review in reviews:

            if (count % 1000 == 0):
                print("Review %d of %d" % (count, len(reviews)))

            matrix[count] = preprocessing.return_averaged_vector_review(dimension,review,model)
            count = count +1

        return matrix


    # #get all the review
    # def get_clean_review_lists(self, review_datasets):
    #     clean_review_datasets = []
    #     for review in review_datasets:
    #         clean_review_datasets.append(preprocessing.clean_text(review))
    #     return clean_review_datasets


    #plot word cloud
    def plot_word_cloud(self, terms):
        text = terms.index
        text = ' '.join(list(text))

        #lower the max fontsize
        wordcloud = WordCloud(max_font_size=40).generate(text)
        plt.figure(figsize = (25,25))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    # #preprocess the text and split each review to a list of tokens
    # def clean_text(self,review):
    #     #1: remove the tags
    #     review = BeautifulSoup(review).get_text()
    #     #2.remove the non alpha
    #     text = re.sub("[^a-zA-Z]", " ",review)
    #     #3.remove split the tokens
    #     lowercase =  text.lower().split()
    #     #4:remove the stopwords
    #     stops = set(stopwords.words("English"))
    #     words = [w for w in lowercase if not w in stops]
    #
    #     return words


    #split each review into sentences and clean the sentences -> to a list
    def transfer_review_to_sentences(self,review, sent_detector):

        #split the review to a list
        sentences = sent_detector.tokenize(review.strip())

        sentencs_list = []
        for sentence in sentences:
            if(len(sentence)>0):
             words = preprocessing.clean_text(sentence)
             sentencs_list.append(words)

        return sentencs_list

