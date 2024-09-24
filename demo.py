import argparse
import os
import json
import time
from tqdm import tqdm
import nltk
from PIL import Image 
import torch
import torch.nn as nn
from torchvision import transforms
from utils import create_exp_dir, Ranker
from data_loader import get_loader
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder, DistilBertEncoder
import wandb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DICT = 'data/captions/dict.{}.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
image_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20240924-180328/image-768.th"
caption_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20240924-180328/cap-768.th"
image_model = torch.load(image_model_path).to(device)
caption_model = torch.load(caption_model_path).to(device)
vocab = Vocabulary()
vocab.load(DICT.format("dress"))

def generate_emb(image_path, caption_texts, img_name, save_embeds):
    # Load a single imageReplace with actual image path
    image = Image.open(image_path)
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor_1 = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor_2 = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor_3 = transform(image).unsqueeze(0)  # Add batch dimension

    input_tensor = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=0).to(device)

    caption_texts = ["is solid black with no sleeves", "is black with straps"]
    
    tokens = nltk.tokenize.word_tokenize(str(caption_texts[0]).lower()) + ['<and>'] + \
            nltk.tokenize.word_tokenize(str(caption_texts[1]).lower())
    

    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))

    caption = torch.LongTensor(caption).unsqueeze(0)

    captions = torch.cat([caption, caption, caption], dim=0).to(device)

    image_tensor = image_model(input_tensor)
    caption_tensor = caption_model(captions, image_tensor)
    #convert to numpy
    caption_tensor = caption_tensor.cpu().detach().numpy()[0]

    # Find 5 similar caption_tensor in save_embeds
    similarities = []
    for key, value in save_embeds.items():
        similarity = cosine_similarity([caption_tensor], [value])
        similarities.append((key, similarity[0][0]))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_similar = similarities[:5]

    print("Top 5 similar captions:")
    root_file = 'data/dress/'
    for item in top_5_similar:
        print(f"Image: {root_file}{item[0]}.jpg, Similarity: {item[1]}")

if __name__ == "__main__":
    img_path = 'B003FGW7MK.jpg'
    caption_texts = ["is solid white with no sleeves", "is black with straps"]
    with open('data/embeddings.json', 'r') as f:
        save_embeds = json.load(f)
    generate_emb(img_path, caption_texts, img_path, save_embeds)

