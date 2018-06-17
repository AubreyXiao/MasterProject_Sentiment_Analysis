from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#load the model
def word2vec_model_accuracy(model,questions):
   # model = Word2Vec.load(modelname)
    accuracy  = model.accuracy(questions)

    total_correct  = len(accuracy[-1]['correct'])
    total_incorrect = len(accuracy[-1]['incorrect'])

    total = total_correct + total_incorrect

    percent  = lambda a:a/total*100


    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total,percent(total_correct),percent(total_incorrect)))

#load the file for evaluation
def read_evaluation_file(questions):
    evaluation = open(questions,'r').readlines()
    num_sections = len([l for l in evaluation if l.startswith(':')])
    #":" represent the separation
    num_sentences = len(evaluation)-num_sections

    print('Total evaluation sentences: {}'.format(num_sentences))


#load the goole as an example

#evaluate the given model
def evaluate_model(model,questions_words,questions_phrases):

    #load the questions_phrases.txt
    print("Evaluate the phrases")
    word2vec_model_accuracy(model,questions_phrases)


#loop the model and evaluate it
def load_models_evaluation(model_lists,questions_words,questions_phrases):
    dir = "/Users/xiaoyiwen/Desktop/MasterProject/MasterProject/data_Preprocessing/HAN_Classifier/word2vec_model/"
    for modelname in model_lists:
        print("Model:" + modelname+"\n")
        modelname1 = dir + modelname
        model = Word2Vec.load(modelname1)
        evaluate_model(model,questions_words,questions_phrases)
        print("-----------------------------------------------\n")




#----------------evaluation----------------------
questions_words = 'questions-words.txt'
questions_phrases = 'questions-phrases.txt'
print("words!")
read_evaluation_file(questions_words)
print("phrases!")
read_evaluation_file(questions_phrases)
print("-----------------------------------------------\n")

#test the google model
# google = KeyedVectors.load_word2vec_format('/Users/xiaoyiwen/Desktop/GoogleNews-vectors-negative300.bin', binary = True)
# evaluate_model(google,questions_words,questions_phrases)


# define the model
model_lists = ["200features_30minwords_10window","100features_30minwords_10window","300features_40minwords_10window","400features_30minwords_10window","300features_30minwords_20window","300features_30minwords_10window","300features_30minwords_5window"]

#loop over the word2vec model
load_models_evaluation(model_lists,questions_words,questions_phrases)


















