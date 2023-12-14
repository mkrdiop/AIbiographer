import os
import re
import torch
import fitz  # PyMuPDF
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

class AutobiographyDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = self._tokenize(texts)

    def _tokenize(self, texts):
        tokenized_texts = []
        for text in texts:
            tokenized_text = self.tokenizer.encode(text, add_special_tokens=True)
            tokenized_texts.extend(tokenized_text)
        return tokenized_texts

    def __len__(self):
        return len(self.examples) - self.block_size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples[idx : idx + self.block_size]),
            "labels": torch.tensor(self.examples[idx + 1 : idx + self.block_size + 1]),
        }

def extract_text_from_pdf(pdf_path):
    print("processing %s" % pdf_path)
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

def preprocess_text(text):
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def train_autobiography_generator(texts):
    # Initialize GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Create AutobiographyDataset
    dataset = AutobiographyDataset(texts, tokenizer)

    # Create DataLoader for training
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train the model
    print("training the model")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(3):  # You may need more epochs depending on your dataset
        for batch in train_dataloader:
            inputs, labels = batch["input_ids"], batch["labels"]
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the trained model
    output_dir = "autobiography_generator_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Provide a list of paths to autobiography PDF files on your local machine
    pdf_paths = [
        #"data/Alibaba (Duncan Clark)2016 (Z-Library).pdf",
        #"data/Bill Gates Behind Microsoft, Money, Malaria (Staff Forbes)2015 (Z-Library).pdf",
        "data/Einstein (Walter Isaacson)2007 (Z-Library).pdf",
        #"data/Gandhi An Autobiography The Story of My Experiments With Truth (Mohandas Karamchand (Mahatma) Gandhi etc.)1993 (Z-Library).pdf",
        #"data/Oprah Winfrey A Biography, Second Edition (Helen S. Garson)2011 (Z-Library).pdf",
        #"data/Think Like Zuck The Five Business Secrets of Facebooks Improbably Brilliant CEO Mark Zuckerberg (Ekaterina Walter)2012 (Z-Library).pdf"
        
        # Add more paths as needed
    ]

    # Extract text from PDFs
    autobiography_texts = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths]

    train_autobiography_generator(autobiography_texts)



"""

pdf_paths = [
        "data/Alibaba (Duncan Clark)2016 (Z-Library).pdf",
        "data/Bill Gates Behind Microsoft, Money, Malaria (Staff Forbes)2015 (Z-Library).pdf",
        "data/Einstein (Walter Isaacson)2007 (Z-Library).pdf",
        "data/Gandhi An Autobiography The Story of My Experiments With Truth (Mohandas Karamchand (Mahatma) Gandhi etc.)1993 (Z-Library).pdf",
        "data/Oprah Winfrey A Biography, Second Edition (Helen S. Garson)2011 (Z-Library).pdf",
        "data/Think Like Zuck The Five Business Secrets of Facebooks Improbably Brilliant CEO Mark Zuckerberg (Ekaterina Walter)2012 (Z-Library).pdf"
        
        # Add more paths as needed
    ]
    
"""