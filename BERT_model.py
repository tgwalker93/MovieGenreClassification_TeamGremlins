
import pandas as pd
import numpy as np
from transformers import TFBertModel,  BertConfig, BertTokenizerFast

# tensorflow.keras packages
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# sklearn packages
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report


all_data = pd.read_csv('all_data.txt', delimiter = " ::: ", header=None, index_col=0,engine='python')
all_data.columns=['Title','Genre','Synopsis']
temp = pd.get_dummies(all_data.Genre)
all_data = pd.concat([all_data, temp], axis=1)

GENRE_CLASS = list(set(all_data.Genre))
GENRE_CLASS

EPOCH = 7
BATCH = 32

data = all_data.copy()

# Select required columns
data = data[['Synopsis','Genre']]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Genre'] = le.fit_transform(data.Genre)
# Remove a row if any of the three remaining columns are missing
data = data.dropna()

# Split into train and test - stratify over Issue
data, data_test = train_test_split(data, test_size = 0.2, stratify = data['Genre'])

y_test = le.inverse_transform(data_test['Genre'])

#######################################
### --------- Setup BERT ---------- ###

# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 200

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)

#######################################
### ------- Build the model ------- ###

# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
Genre = Dense(units=len(data.Genre.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Genre')(pooled_output)

outputs = {'Genre': Genre}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()

#######################################
### ------- Train the model ------- ###

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'Genre': CategoricalCrossentropy(from_logits = True)}
metric = {'Genre': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metric)

# Ready output data for the model
y_Genre = to_categorical(data['Genre'])

# Tokenize the input (takes some time)
x = tokenizer(
    text=data['Synopsis'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# Fit the model
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'Genre': y_Genre},
    validation_split=0.2,
    batch_size=BATCH,
    epochs=EPOCH)


#######################################
### ----- Evaluate the model ------ ###

# Ready test data
test_y_Genre = to_categorical(data_test['Genre'])

test_x = tokenizer(
    text=data_test['Synopsis'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

# Run evaluation
y_predicted_raw = model.predict(x={'input_ids': test_x['input_ids']})

y_predicted = le.inverse_transform([np.argmax(y_predicted_raw['Genre'][index,:], axis=None, out=None) for index in np.arange(len(y_predicted_raw['Genre']))])

print(classification_report(y_test, y_predicted, target_names=GENRE_CLASS))