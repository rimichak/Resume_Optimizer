import re
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_from_pdf(file_path):
    text = ''
    try:
       with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                   text += page_text
    except:
        pass
    if len(text.strip()) == 0:
        images = convert_from_path(file_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text

    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    text = ' '.join(text.split())  
    return text

def get_text_embeddings(text,  chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    embeddings_list = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk])
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings_list.append(cls_embedding)
    
    
    avg_embedding = np.mean(np.vstack(embeddings_list), axis=0, keepdims=True)
    return avg_embedding

def calculate_similarity(resume_embeddings, job_embeddings):
    similarity = cosine_similarity(resume_embeddings, job_embeddings)[0][0]
    return similarity

resume_file = "resume.pdf" 
resume_text = extract_text_from_pdf(resume_file)
resume_text = preprocess_text(resume_text)

job_text = input("Enter job description: ")
job_text = preprocess_text(job_text)

resume_embeddings = get_text_embeddings(resume_text)
job_embeddings = get_text_embeddings(job_text)

similarity_score = calculate_similarity(resume_embeddings, job_embeddings)
print("Similarity Score:", similarity_score)