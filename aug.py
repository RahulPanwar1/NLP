import os
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to save DataFrame to CSV
def save_df(data, filename):
    data.to_csv(filename, index=False)

# Function to generate text using GPT-2 model
def get_response(messages):
    input_ids = tokenizer.encode(messages, return_tensors="pt")
    # Generate text with max length of 150 tokens
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, early_stopping=True)
    curr_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return curr_response

# Function to prepare messages from DataFrame
def prepare_messages(data):
    messages = [f"Title: {row['title']}\nText: {row['text']}\n" for idx, row in data.iterrows()]
    return messages

# Function to augment data using GPT-2 model
def augment_forex_data(data, filename, template_type, threshold=10):
    if not os.path.exists(filename):
        # Create an empty DataFrame and save it to the CSV file
        empty_df = pd.DataFrame(columns=data.columns)
        save_df(empty_df, filename)

    augmented_data = pd.read_csv(filename)
    saved_examples = len(augmented_data)

    for i, row in tqdm(data.iterrows(), total=len(data)):
        if i < saved_examples:
            continue
        messages = ''.join(prepare_messages(pd.DataFrame(row).T))
        prompt = templates(template_type).format(text=messages)
        curr_response = get_response(prompt)
        augmented_row = row.copy()
        augmented_row['text'] = curr_response
        augmented_data = augmented_data.append(augmented_row, ignore_index=True)
        if i % threshold == 0:
            save_df(augmented_data, filename)
            augmented_data = pd.DataFrame(columns=data.columns)

    if len(augmented_data) > 0:
        save_df(augmented_data, filename)


def templates(type_):
    templates = {
        "one_paraphrase": "Generate a paraphrase for the following text, preserving the sentiment of the following statement: {text}\nGenerate another paraphrase by changing more words also keeping the sentiment",
        "new_text": "Based on the given text, generate another text with a completely new theme, but be inspired by the original text and keep the sentiment of the old one in the new text. Original text: {text}",
        "new_text_v2": "Based on the given text, generate another text with a completely new theme, but be inspired by the original text and keep a {sentiment} sentiment. Original text: {text}"
    }
    return templates[type_]


# Path to the input CSV file and the augmented CSV file
input_csv_filename = "forex.csv"
augmented_csv_filename = "augmented_forex_data.csv"

# Choose the template type
template_type = "one_paraphrase"

# Read the input CSV file
forex_data = pd.read_csv(input_csv_filename)

# Augment the data using GPT-2 model
augment_forex_data(forex_data, augmented_csv_filename, template_type)
