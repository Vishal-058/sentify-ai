
import re

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # Strips HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keeps only letters
    text = text.lower().strip()
    return text
