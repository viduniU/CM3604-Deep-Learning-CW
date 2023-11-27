import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import EarlyStopping
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')
json_file_path='/content/drive/My Drive/Datasets/yelp_academic_dataset_review.json'


#load a subset of the dataset
chunk_size = 5000
num_samples = 20000

#create an empty list to store DataFrames
dataframe = []

#open the JSON file and process the chunks with explicit encoding
with open(json_file_path, 'r', encoding='utf-8') as file:
    for chunk in pd.read_json(file, lines=True, chunksize=chunk_size):
        #perform sentiment analysis on the 'text' column
        dataframe.append(chunk[['text', 'stars']].copy())
        if sum(map(len, dataframe)) >= num_samples:
            break

#concatenate the list of DataFrames into the final DataFrame
yelp_subset = pd.concat(dataframe, ignore_index=True)

output_json_file_path = 'yelp_subset.json'
yelp_subset.to_json(output_json_file_path, orient='records', lines=True)

#display the shape of the loaded subset
print("Shape of Yelp Subset:", yelp_subset.shape)

