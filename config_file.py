import torch 
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Defining device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Defining tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# Defining model
model = T5ForConditionalGeneration.from_pretrained("t5-base")
