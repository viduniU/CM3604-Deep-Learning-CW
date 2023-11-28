import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
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
#derive sentiment labels from star labels
yelp_subset['sentiment'] = yelp_subset['stars'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))
#handle imbalane classes with random oversampling
x_train, x_test, y_train, y_test = train_test_split(yelp_subset['text'], yelp_subset['sentiment'], test_size=0.2, random_state=42)
#create the RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

#resample the training data
x_train_resampled, y_train_resampled = undersampler.fit_resample(yelp_subset['text'].to_frame(), yelp_subset['sentiment'])

# Convert the resampled data back to Series
x_train_resampled = x_train_resampled['text']
y_train_resampled = pd.Series(y_train_resampled)

# tokenize with BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)


#tokenize the resampled training set
train_encodings = tokenizer(list(x_train_resampled), truncation=True, padding=True, max_length=400, return_tensors='pt')

#tokenize the testing set
test_encodings = tokenizer(list(x_test), truncation=True, padding=True, max_length=400, return_tensors='pt')

#convert labels to numerical values
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
y_train_mapped = y_train_resampled.map(label_mapping)
y_test_mapped = y_test.map(label_mapping)

#create DataLoader for the resampled training set
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train_mapped.values))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#create DataLoader for the resampled testing set
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test_mapped.values))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
#load pre-trained BERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)


#define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*3)


