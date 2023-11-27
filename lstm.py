import nltk
nltk.download('punkt')
pip install langdetect
import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')
pip install keras
pip install tensorflow
#import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Embedding, LSTM, Dense
#from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from langdetect import detect
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/Datasets/yelp_academic_dataset_review.json'
# Load a subset of the dataset (adjust chunk_size and num_samples)
chunk_size = 1000
num_samples = 50000

# Create an empty list to store DataFrames
dataframe = []

# Open the JSON file and process it in chunks with explicit encoding
with open(file_path, 'r', encoding='utf-8') as file:
    for chunk in pd.read_json(file, lines=True, chunksize=chunk_size):
        # Perform sentiment analysis on the 'text' column
        dataframe.append(chunk[['text', 'stars']].copy())
        if sum(map(len, dataframe)) >= num_samples:
            break

# Concatenate the list of DataFrames into the final DataFrame
yelp_subset = pd.concat(dataframe, ignore_index=True)

output_json_file_path = 'yelp_subset.json'
yelp_subset.to_json(output_json_file_path, orient='records', lines=True)

# Display the shape of the loaded subset
print("Shape of Yelp Subset:", yelp_subset.shape)
yelp_subset.head()

Exploratory Data Analysis
# Set seaborn style
sns.set(style="whitegrid")

# 2. Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='stars', data=yelp_subset, palette="viridis")
plt.title('Distribution of Ratings', fontsize=16)
plt.xlabel('Star Ratings', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

#Review Length Distribution
plt.figure(figsize=(10, 6))
yelp_subset['review_length'] = yelp_subset['text'].apply(lambda x: len(word_tokenize(x)))
sns.histplot(yelp_subset['review_length'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Review Lengths', fontsize=16)
plt.xlabel('Review Length', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# insert a column to store language information
yelp_subset['language'] = yelp_subset['text'].apply(lambda x: detect(x) if isinstance(x, str) else None)

# calculate the  word counts for reviews in each language
yelp_subset['word_count'] = yelp_subset['text'].apply(lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0)

#new column for number of reviews for each language
review_counts_by_language = yelp_subset['language'].value_counts().reset_index()

#plot for the number of reviews for each language to check if there are non-english words
plt.figure(figsize=(12, 8))
sns.barplot(x='index', y='language', data=review_counts_by_language)
plt.title('Number of Reviews by Language')
plt.xlabel('Language')
plt.ylabel('Number of Reviews')
plt.show()

Preprocessing

Feature Engineering

#derive sentiment labels from star labels
yelp_subset['sentiment'] = yelp_subset['stars'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))

Text Normalization

#handle imbalane classes with random oversampling
x_train, x_test, y_train, y_test = train_test_split(yelp_subset['text'], yelp_subset['sentiment'], test_size=0.2, random_state=42)

#convert all text to lowercase for uniformity
x_train = x_train.apply(lambda x: x.lower())
x_test = x_test.apply(lambda x: x.lower())

#remove punctuation to reduce noise in the text.
x_train = x_train.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
x_test = x_test.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

#remove common words that may not contribute much to sentiment
stop_words = set(stopwords.words('english'))
x_train = x_train.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
x_test = x_test.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

#reduce words to the base form
lemmatizer = WordNetLemmatizer()
x_train = x_train.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
x_test = x_test.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

#reshape x_train to have two dimensions
x_train_reshape = x_train.values.reshape(-1, 1)

#oversample the minority classes
over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
x_train_over_resampled, y_train_resampled = over_sampler.fit_resample(x_train_reshape, y_train)

#convert oversampled indices to text sequences
x_train_over = x_train_over_resampled.flatten()

Tokenization

#maximum number of words to keep based on word frequency
max_num_words = 3000

#create a tokenizer
tokenizer = Tokenizer(num_words=max_num_words,#replace the  words not in the tokenizer's vocabulary
                       oov_token='<OOV>')

#fit the tokenizer on text
tokenizer.fit_on_texts(x_train_over)

#convert the text to integers sequences
x_train_seq = tokenizer.texts_to_sequences(x_train_over)
x_test_seq = tokenizer.texts_to_sequences(x_test)

#define maximum length of the sequence
max_len = 300

#pad sequences to make sentences to the same length
x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

### Model Building
#converting integer indices representing words into dense vectors of fixed size
embedding_dim = 50


model = Sequential()
model.add(Embedding(input_dim=max_num_words,#size of the vocabulary
                    output_dim=embedding_dim,#size of dense embeddings
                    input_length=max_len))#maximum length of input sequences
#model.add(LSTM(100, return_sequences=True))  # Additional LSTM layer
model.add(LSTM(100, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))#to handle overfitting
model.add(Dense(3, activation='softmax')) # 3 output nodes for positive, negative, and neutral

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
