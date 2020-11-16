import numpy as np 
import pandas as pd 

import wandb 

import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import T5Tokenizer, T5ForConditionalGeneration

#import config 
import engine 
import dataset 



# WandB – Initialize a new run
wandb.init(project="transformers_tutorials_summarization")

# wandb for holding hyperparameters  
config = wandb.config         
config.TRAIN_BATCH_SIZE = 2    
config.VALID_BATCH_SIZE = 2    
config.TRAIN_EPOCHS = 2        
config.VAL_EPOCHS = 1 
config.LEARNING_RATE = 1e-4    
config.SEED = 42              
config.MAX_LEN = 512
config.SUMMARY_LEN = 150 

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(config.SEED) 
np.random.seed(config.SEED) 
torch.backends.cudnn.deterministic = True

# tokenzier for encoding the text
tokenizer = T5Tokenizer.from_pretrained("t5-base")


# Loading dataset  
df = pd.read_csv('/home/hasan/Data Set/News Summary/news_summary.csv', encoding='latin-1')
df = df[['text','ctext']]
df.ctext = 'summarize: ' + df.ctext
print(df.head())


# Creation of Dataset and Dataloader
# Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state = config.SEED)
val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(val_dataset.shape))


# Creating the Training and Validation dataset for further creation of Dataloader
training_set = dataset.CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
val_set = dataset.CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': config.TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
    }

val_params = {
    'batch_size': config.VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
    }

# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)



# Loading Model
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model = model.to(device)

# Optimizer 
optimizer = torch.optim.Adam(params= model.parameters(), lr=config.LEARNING_RATE)

# Log metrics with wandb
wandb.watch(model, log="all")
# Training loop
print('Initiating Fine-Tuning for the model on our dataset')

for epoch in range(config.TRAIN_EPOCHS):
    engine.train(epoch, tokenizer, model, device, training_loader, optimizer)


# Validation loop and saving the resulting file with predictions and acutals in a dataframe.
# Saving the dataframe as predictions.csv
print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
for epoch in range(config.VAL_EPOCHS):
    predictions, actuals = engine.validate(epoch, tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv('/home/hasan/Desktop/Code to keep on Github/Text Summarization/predictions.csv')
    print('Output Files generated for review')


