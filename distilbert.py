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

epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


#early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

#define lists to store metrics during training
train_losses = []
train_f1_scores = []
train_precisions = []
train_recalls = []
val_accuracies = []
test_accuracies = []
train_accuracies = []



for epoch in range(epochs):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    for batch_num, batch in enumerate(train_loader):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        #calculate predictions and true labels for metric
        predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
        true_labels.extend(inputs['labels'].cpu().numpy())

        if batch_num % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_num}/{len(train_loader)}, Loss: {loss.item()}')

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    #calculate accuracy,F1 score, precision, and recall on training data
    train_accuracy = accuracy_score(true_labels, predictions)
    f1_train = f1_score(true_labels, predictions, average='weighted')
    precision_train = precision_score(true_labels, predictions, average='weighted')
    recall_train = recall_score(true_labels, predictions, average='weighted')


     #append to lists to retrive matrices after training
    train_accuracies.append(train_accuracy)
    train_f1_scores.append(f1_train)
    train_precisions.append(precision_train)
    train_recalls.append(recall_train)

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}, '
          f'Training Accuracy: {train_accuracy}, F1 Score: {f1_train}, '
          f'Precision: {precision_train}, Recall: {recall_train}')

#save the trained model
model.save_pretrained('BERT_model')

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

#load the model
loaded_model = DistilBertForSequenceClassification.from_pretrained('/content/BERT_model')

#load the corresponding tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

label_mapping_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}

#evaluate the model on the test set

model.eval()
test_predictions = []
test_true_labels = []

for test_batch_num, test_batch in enumerate(test_loader):
    test_inputs = {'input_ids': test_batch[0].to(device),
                   'attention_mask': test_batch[1].to(device),
                   'labels': test_batch[2].to(device)}

    test_outputs = model(**test_inputs)
    test_predictions.extend(test_outputs.logits.argmax(dim=1).cpu().numpy())
    test_true_labels.extend(test_inputs['labels'].cpu().numpy())

#calculate the test_accuracy,test_f1,test_precision,test_recall
test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
test_precision = precision_score(test_true_labels, test_predictions, average='weighted')
test_recall = recall_score(test_true_labels, test_predictions, average='weighted')

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'F1 Score: {test_f1}, Precision: {test_precision}, Recall: {test_recall}')

plt.figure(figsize=(10, 5))

#plot for model loss over epohs
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='skyblue', linestyle='-', marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()




