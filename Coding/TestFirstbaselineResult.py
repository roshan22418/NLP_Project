import shutil
import os
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np 
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")
import matplotlib.pyplot as plt
# Path to the trained model checkpoint
model_checkpoint = "/kaggle/working/bart_finetuned_model"
test_data_load_path="/kaggle/input/testdataset/data-00000-of-00001.arrow"
test_data_write_path="/kaggle/working/test_data.arrow"
shutil.copy(test_data_load_path, test_data_write_path)
test_data=Dataset.from_file(test_data_write_path)
input_texts = [example["input_text"] for example in test_data]
target_texts = [example["target_text"] for example in test_data]
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example list of input texts (replace with your actual dataset)
input_texts = [example["input_text"] for example in test_data]
target_texts = [example["target_text"] for example in test_data]

# Prediction loop
for input_text in input_texts:
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    # Decode generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}\nGenerated Target Text: {generated_text}\n{'-'*80}")