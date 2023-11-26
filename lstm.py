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


