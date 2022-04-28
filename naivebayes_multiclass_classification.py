import random
import string

import nltk
import sklearn
from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import openpyxl

#For downloading packages
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stop_words.add('one')
stop_words.add('life')

def create_data_set():
    print("this method is for creating a data set")



def setup_docs():
    docs = [] # (label, text)
    with open('train_data.txt', 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split(' ::: ')
            doc = ( parts[2], parts[3].strip() )  # (label, text)
            docs.append(doc)
    return docs


def clean_text(text):
    #remove punctuation
    text = text.translate(str.maketrans('','', string.punctuation))
    #convert to lower case
    text = text.lower()
    return text

def get_tokens(text):
    # get individual words
    tokens = word_tokenize(text)
    #remove common words that are useless
    tokens = [t for t in tokens if not t in stop_words]
    return tokens

def print_frequency_dist(docs):
    tokens = defaultdict(list)

    # lets make a giant list of all the words for each category
    labels = []
    for doc in docs:
        doc_label = doc[0].replace("\n", "")
        labels.append(doc_label)
        doc_text = clean_text(doc[1])

        doc_tokens = get_tokens(doc_text)

        tokens[doc_label].extend(doc_tokens)

    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))
    return labels

def get_splits(docs):
    #scramble docs
    random.shuffle(docs)

    X_train = [] # training documents
    y_train = [] # corresponding training labels

    X_test = [] #test documents
    y_test = [] #corresponding test label

    pivot = int(.80 * len(docs))

    for i in range(0, pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0].replace("\n", ""))

    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0].replace("\n", ""))

    return X_train, X_test, y_train, y_test


def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    print(str(title) + ": " + "Precision: " + str(precision), " --- Recall: " + str(recall), " --- F1 Score: " + str(f1))

def train_classifier(X_train, X_test, y_train, y_test):

    # the object that turns text into vectors
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')

    # create doc-term matrix
    dtm = vectorizer.fit_transform(X_train)

    # train Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

    #evaluate on training data
    evaluate_classifier("Naive Bayes\tTRAIN (80% of Training Data)\t", naive_bayes_classifier, vectorizer, X_train, y_train)

    #evaluate on test data
    evaluate_classifier("Naive Bayes\tTEST (20% of Training Data)\t", naive_bayes_classifier, vectorizer, X_test, y_test)

    #store the classifier
    clf_filename = 'naive_bayes_classifier2.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

    #also store the vectorizer so we can transform new data
    vec_filename = 'count_vectorizer2.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))

def classifyOneDoc(text):
    # load classifier
    clf_filename = 'naive_bayes_classifier2.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    # vectorize the new text
    vec_filename = 'count_vectorizer2.pkl'
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    pred = nb_clf.predict(vectorizer.transform([text]))

    print(pred[0])

def ClassifyMultipleDocs(docs, answer):

    test = answer
    # load classifier
    clf_filename = 'naive_bayes_classifier2.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    # vectorize the new text
    vec_filename = 'count_vectorizer2.pkl'
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    results = []
    yHat = []
    y = []
    for i in range(len(docs)):
        prediction = nb_clf.predict(vectorizer.transform([docs[i]]))
        #print((answer[i], prediction[0], docs[i]))
        #Answer, prediction, text
        yHat.append(prediction[0])
        y.append(answer[i])
        results.append((answer[i], prediction[0], docs[i]))
    return yHat, y, results

#index 1 is the systemoutput
def ProcessSystemoutputText(Set):
    ResultFile = open("results.txt", "w")
    for itemArray in Set:
        ResultFile.write(str(itemArray[2]) + " - " + itemArray[1] +"\n")
    ResultFile.close()

#index 0 is the gold standard
def ProcessGoldStandardText(Set):
    ResultFile = open("goldstandard2.txt", "w")
    for itemArray in Set:
        ResultFile.write(str(itemArray[2]) + " - " + itemArray[0] +"\n")
    ResultFile.close()

if __name__ == '__main__':
    #For Creating the Data Set
    #create_data_set()

    docs = setup_docs()
    print("DCS HAVE BEEN RAN")

    X_train, X_test, y_train, y_test = get_splits(docs)

    #Check the frequency of words for each class and return the labels
    labels = print_frequency_dist(docs)

    #Train the Classifier
    #train_classifier(X_train, X_test, y_train, y_test)

    #deployment in production
    #new doc = ""


    #We store our classifier in the .pkl file so we dont have to retrain our model, and we can just run these classify methods to get classes
    #classifyOneDoc("some text")
    yHat, y, results = ClassifyMultipleDocs(X_test, y_test)


    print("Test is done")
    print("F1 Score is: ", str(sklearn.metrics.f1_score(y_true=y, y_pred=yHat, labels=labels, average='macro')))
    #print results to text file
    #ProcessSystemoutputText(results)


    print("Done")