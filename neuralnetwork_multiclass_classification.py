from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D,GRU,BatchNormalization
from keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,LeakyReLU
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.utils import np_utils
import tensorflow_addons as tfa

#For downloading packages
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
nltk.download('wordnet')

def setup_docs(file_path):
    docs = [] # (label, text)
    with open(file_path, 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split(' ::: ')
            doc = ( parts[2], parts[3].strip() )  # (label, text)
            docs.append(doc)
    return docs

def load_glove(word_index, max_features):
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8"))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            embedding = np.asarray(values[-300:], dtype='float32')
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def clean_data(review):
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    return review

def remove_stop_words(doc):
    review_minus_sw = []
    stop_words = stopwords.words('english')
    doc = doc.split()
    doc = [review_minus_sw.append(word) for word in doc if word not in stop_words]
    doc = ' '.join(review_minus_sw)
    return doc

def lematize(doc):
    lemmatizer = WordNetLemmatizer()
    doc = doc.split()
    doc = [lemmatizer.lemmatize(w) for w in doc]
    doc = ' '.join(doc)
    return doc

#Combines all the data, both train and test!
def process_data_file():
    docs = []
    count = 1
    with open("train_data.txt", 'r', encoding='utf8') as datafile:
        for row in datafile:
            docs.append(row.rstrip())
            count += 1
    with open("test_data_solution.txt", 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split(' ::: ')
            parts[0] = str(count)
            row = ' ::: '.join(parts)
            docs.append(row.rstrip())
            count += 1
    count = 0
    with open('all_data.txt', 'w', encoding='utf8') as f:
        for item in docs:
            count += 1
            f.write("%s\n" % item)


if __name__ == '__main__':
    # process_data_file()
    # train_docs = setup_docs('train_data.txt')
    #
    # train_docs_without_labels = []
    # train_labels = []
    #
    # for i in range(len(train_docs)):
    #     train_docs_without_labels.append(train_docs[i][1])
    #     train_labels.append(train_docs[i][0])
    #
    # test_docs = setup_docs('test_data_solution.txt')
    #
    # test_docs_without_labels = []
    # test_labels = []
    #
    # for i in range(len(test_docs)):
    #     test_docs_without_labels.append(test_docs[i][1])
    #     test_labels.append(test_docs[i][0])
    # #print(docs_without_labels)
    #
    #
    # #print(docs)
    #
    #

    df = pd.read_csv('train_data.txt', delimiter=" ::: ", names=['id','movie_name', 'genre', 'description'])

    #clean the data
    df['description'] = df['description'].apply(clean_data)

    #Remove the stop words
    df['description'] = df['description'].apply(remove_stop_words)

    #now we need to stem/lemmatize the data
    #which means we need to convert the words to their roots
    df['description'] = df['description'].apply(lematize)


    #Creating the bag of words model + TfidfTransformer
    corpus = list(df['description'])
    tfidfVectorizer = TfidfVectorizer(max_features=1000)
    X = tfidfVectorizer.fit_transform(corpus).toarray()
    y = df['description'].values

    feature_names = tfidfVectorizer.get_feature_names()


    #Converting labels to numbers
    unique_labels = df['genre'].unique()

    dict_index_to_labels = {i: unique_labels[i] for i in range(0, len(unique_labels))}
    dict_labels_to_index = {unique_labels[i]: i for i in range(0, len(unique_labels))}
    #
    print("Dictionary of Labels to Index: ", dict_labels_to_index)
    #
    df['genre'] = df['genre'].apply(lambda x: dict_labels_to_index[x])



    #Splitting our data into docs and labels
    docs = df['description']
    labels = df['genre']

    num_classes = 27
    categorical_labels = keras.utils.np_utils.to_categorical(labels, num_classes=num_classes, dtype='float32')


    #80/20 split
    X_train, X_test, y_train, y_test = train_test_split(docs, categorical_labels, test_size=0.20)


    #Apply Tokenizer
    vocab_size = 50000
    oov_token = "<OOV>"
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    #Convert X to sequences
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    #pad the sequences
    max_length = 300
    padding_type = "post"
    trunction_type = "post"
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type,
                                   truncating=trunction_type)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length,
                                  padding=padding_type, truncating=trunction_type)

    word_index = tokenizer.word_index

    # #Use glove pretrained word embeddings
    # embeddings_index = load_glove_model('glove.840B.300d.txt')
    #
    # word_index = tokenizer.word_index
    #
    # #now we need to obtain the embedding for every word in the training set
    # embedding_matrix = np.zeros((len(word_index) + 1, max_length))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector


    # embedding_layer = Embedding(input_dim=len(word_index) + 1,
    #                             output_dim=max_length,
    #                             weights=[embedding_matrix],
    #                             input_length=max_length,
    #                             trainable=False)


    # #Encoding Labels
    # le = LabelEncoder()
    # train_y = le.fit_transform(y_train.values)
    # test_y = le.transform(y_test.values)


    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=max_length,
                                input_length=max_length)




    #Dont need glove
    #Do some augmentation that will randomly augment
    #Make random 1 or 2 words at each time
    #If you can have the glove model as a pretrained model and then tweak the embedding
    #Bidirectional LSTM - DO IT FIRST
    #GET RID OF GLOVE
    model = Sequential([
        embedding_layer,
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        # SpatialDropout1D(0.2),
        # LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        # keras.layers.TimeDistributed(Dense(10), input_shape=(X_train_padded.shape[1:]),
        # keras.layers.Bidirectional(keras.layers.LSTM(8)),
        Dense(10, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #               optimizer=keras.optimizers.adam_v2.Adam(learning_rate=4e-4),
    #               metrics=['accuracy',
    #                        tfa.metrics.F1Score(num_classes=num_classes,average="macro", name="macroF1")])

    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.adam_v2.Adam(learning_rate=4e-4),
                  metrics=['accuracy',
                           tfa.metrics.F1Score(num_classes=num_classes,average="macro", name="macroF1")])

    epochs = 20
    batch_size = 128

    history = model.fit(X_train_padded, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_padded, y_test))

    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print('Testing Accuracy is {} '.format(accuracy * 100))





























    #OLD ------------------------------------------------------------------


    # #clean the data
    # train_docs_without_labels = [clean_data(doc) for doc in train_docs_without_labels]
    # test_docs_without_labels = [clean_data(doc) for doc in test_docs_without_labels]
    #
    # #apply stop words
    # train_docs_without_labels = [remove_stop_words(doc) for doc in train_docs_without_labels]
    # test_docs_without_labels = [remove_stop_words(doc) for doc in test_docs_without_labels]
    #
    # #now we need to stem/lemmatize the data
    # #which means we need to convert the words to their roots
    # train_docs_without_labels = [lematize(doc) for doc in train_docs_without_labels]
    # test_docs_without_labels = [lematize(doc) for doc in test_docs_without_labels]
    #
    #
    # #Need to get a unique count of all words
    # all_docs = train_docs_without_labels + test_docs_without_labels
    # all_docs_single_string = ' '.join(all_docs)
    # all_words_list = all_docs_single_string.split()
    # all_words_set = set(all_words_list)
    # #number_of_unique_words = len(all_words_set)
    #
    # df = pd.read_csv('train_data.txt', delimiter=" ::: ")
    #
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)

    # #find max features
    # largest_paragraph = max(all_docs, key=len)
    # largest_paragraph_list = largest_paragraph.split()
    # largest_paragraph_unique_words = set(largest_paragraph_list)
    # number_of_unique_words = len(largest_paragraph_unique_words)
    #
    #
    # # Tokenize the sentences
    # tokenizer = Tokenizer(num_words=number_of_unique_words)
    # tokenizer.fit_on_texts(all_docs)
    # train_X = tokenizer.texts_to_sequences(train_docs_without_labels)
    # test_X = tokenizer.texts_to_sequences(test_docs_without_labels)
    #
    # train_X_padded = pad_sequences(train_X, maxlen=number_of_unique_words)
    # test_X_padded = pad_sequences(test_X, maxlen=number_of_unique_words)
    #
    #
    # le = LabelEncoder()
    # train_y = le.fit_transform(train_labels)
    # test_y = le.transform(test_labels)
    #
    # embedding_matrix = load_glove(tokenizer.word_index, max_features=number_of_unique_words)
    # #model = run_model(embedding_matrix, 100, 8)
    #
    # embedding_layer = Embedding(684, 300,
    #                             weights=[embedding_matrix],
    #                             input_length=number_of_unique_words,
    #                             trainable=False)
    # model = Sequential([
    #     embedding_layer,
    #     Conv1D(128, 5, activation='relu'),
    #     GlobalMaxPooling1D(),
    #     Dense(10, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # #training the CNN model
    # training_padded = np.array(train_X_padded)
    # training_labels = np.array(train_y)
    # testing_padded = np.array(test_X_padded)
    # testing_labels = np.array(test_y)
    # history = model.fit(training_padded, training_labels, epochs=20, validation_data=(testing_padded, testing_labels))
    #
    # loss, accuracy = model.evaluate(testing_padded, testing_labels)
    # print('Testing Accuracy is {} '.format(accuracy * 100))



