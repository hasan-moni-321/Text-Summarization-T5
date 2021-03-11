import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import wandb, re

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import T5Tokenizer, T5ForConditionalGeneration

import warnings
warnings.filterwarnings('ignore')

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




# Loading dataset  
df = pd.read_csv('../input/news-summary/news_summary.csv', encoding='latin-1')
df = df[['text','ctext']]
#df.ctext = 'summarize: ' + df.ctext
print("Shape of the dataset is :", df.shape)
print(df.head())

########################################################################
# Checking null and dropping
##########################################################################
df.isnull().sum()
df.dropna(inplace=True)
# Checking null after dropping null 
df.isnull().sum()

#######################################################################
# Some Data Visualization
##########################################################################
for i in range(5):
    print("Headline is: \n",df['text'][i], '\n')
    print("Text is: \n",df['ctext'][i], '\n\n')

# word count for both the columns 
word_count = {'summary_word': [], 'text_word': []}

for headl, txt in zip(df['text'], df['ctext']):
    headli_word = len(headl.split())
    text_word = len(txt.split())
    
    word_count['summary_word'].append(headli_word)
    word_count['text_word'].append(text_word)
    
word_count_df = pd.DataFrame(word_count)
word_count_df.head()


print("minimum summary word is :", word_count_df['summary_word'].min(), '\n',
     "mean summary word is :", word_count_df['summary_word'].mean(), "\n",
     "maximum summary word is :", word_count_df['summary_word'].max())

print("minimum text word is :", word_count_df['text_word'].min(), '\n',
     "mean text word is :", word_count_df['text_word'].mean(), "\n",
     "maximum text word is :", word_count_df['text_word'].max())


# Filtering number of word in summary and text columns 
ind_list = []
for i, (summ, txt) in enumerate(zip(word_count_df['summary_word'], word_count_df['text_word'])):
    if (summ >=50 and summ <= 60) and (txt >= 100 and txt <= 500):
        ind_list.append(i)
print("Total number of rows :", len(ind_list))

# Plotting histogram 
word_count_df.hist()

g = sns.JointGrid(x=word_count_df['summary_word'], y=word_count_df['text_word'], data=word_count_df)
g.plot_joint(sns.kdeplot,
             fill=True, clip=((2200, 6800), (10, 25)),
             thresh=0, levels=100, cmap="rocket")
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)

sns.set_theme(style="whitegrid")
sns.lineplot(data=word_count_df, palette="tab10", linewidth=2.5)
plt.title("Summary word and Text word")

# Violin plot
sns.violinplot(data=word_count_df, palette="Set3", bw=.2, cut=1, linewidth=1)
# pairplot of df
plt.figure(figsize=(10, 10))
sns.pairplot(word_count_df)

########################################################################
# Data Cleaning
###########################################################################
new_df = df[df.index.isin(ind_list)].reset_index(drop=True)
print("Shape of new_df is :", new_df.shape)
new_df.head(3)

# Contraction words 
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


#Removes non-alphabetic characters:
def data_clean(column):
    
    text = column.lower() # lowercase

    # Contraction words Handling
    text = text.split()
    for i in range(len(text)):
        word = text[i]
        if word in contraction_mapping:
            text[i] = contraction_mapping[word]
    text = " ".join(text) 

    # Removing stop-words
    stop_words = stopwords.words('english')
    text = text.split()
    new_text = []
    for word in text:
        if word not in stop_words:
            new_text.append(word)

    text = " ".join(new_text)

    #ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!
    #print(text)
    row=re.sub("(\\t)", ' ', str(text)).lower() #remove escape charecters
    row=re.sub("(\\r)", ' ', str(row)).lower() 
    row=re.sub("(\\n)", ' ', str(row)).lower()

    row=re.sub("(__+)", ' ', str(row)).lower()   #remove _ if it occors more than one time consecutively
    row=re.sub("(--+)", ' ', str(row)).lower()   #remove - if it occors more than one time consecutively
    row=re.sub("(~~+)", ' ', str(row)).lower()   #remove ~ if it occors more than one time consecutively
    row=re.sub("(\+\++)", ' ', str(row)).lower()   #remove + if it occors more than one time consecutively
    row=re.sub("(\.\.+)", ' ', str(row)).lower()   #remove . if it occors more than one time consecutively

    row=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower() #remove <>()|&©ø"',;?~*!

    row=re.sub("(mailto:)", ' ', str(row)).lower() #remove mailto:
    row=re.sub(r"(\\x9\d)", ' ', str(row)).lower() #remove \x9* in text
    row=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower() #replace INC nums to INC_NUM
    row=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower() #replace CM# and CHG# to CM_NUM


    row=re.sub("(\.\s+)", ' ', str(row)).lower() #remove full stop at end of words(not between)
    row=re.sub("(\-\s+)", ' ', str(row)).lower() #remove - at end of words(not between)
    row=re.sub("(\:\s+)", ' ', str(row)).lower() #remove : at end of words(not between)

    row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces

    #Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
        repl_url = url.group(3)
        row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(row))
    except:
        pass #there might be emails with no url in them


    row = re.sub("(\s+)",' ',str(row)).lower() #remove multiple spaces

    row = re.sub(r'[^a-zA-Z0-9. ]','', str(row)).lower() # remove punctuations

    #Should always be last
    row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces

    return row
        
new_df['text'] = new_df['text'].apply(lambda x: data_clean(x))
new_df['ctext'] = new_df['ctext'].apply(lambda x: data_clean(x))
print("Shape after clean the dataset :", new_df.shape)
new_df.head(3)

##########################################################################
# Stemming of the Word
#############################################################################
ps = PorterStemmer()
corpus = []

def porter_stemming(column):
    text = column.lower()
    text = re.sub('[^a-zA-Z]',' ', text) 
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

new_df['text'] = new_df['text'].apply(lambda x: porter_stemming(x))
new_df['ctext'] = new_df['ctext'].apply(lambda x: porter_stemming(x))
print("Shape of new_df is :", new_df.shape)
new_df.head(3)

##################################################################################
# Adding keyword in Text
#####################################################################################
new_df['ctext'] = 'summarize: ' + df.ctext
new_df.head()



################################################################################
# Dividing Dataset
####################################################################################
train_dataset, valid_dataset = train_test_split(new_df, test_size=.2, random_state=42)
print("Shape of train_dataset is :{} Shape of valid_dataset is :{}".format(train_dataset.shape, valid_dataset.shape))


##############################################################
# Defining Tokenizer
################################################################


# Creating the Training and Validation dataset for further creation of Dataloader
training_set = dataset.CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
val_set = dataset.CustomDataset(valid_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)


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


