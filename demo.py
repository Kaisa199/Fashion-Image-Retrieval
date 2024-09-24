import argparse
import os
import time
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

DICT = 'data/captions/dict.{}.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def generate_demo_sample():
    image_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20240924-171910/image-768.th"
    caption_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20240924-171910/cap-768.th"
    image_model = torch.load(image_model_path)
    caption_model = torch.load(caption_model_path)

    # Load a single image
    image_path = "B003FGW7MK.jpg"  # Replace with actual image path
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
    
    vocab = Vocabulary()
    vocab.load(DICT.format("dress"))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    caption = torch.Tensor(caption)

    #expand batch size dim input_tensor
    image_tensor = image_model(input_tensor)
    caption_tensor = caption_model(caption)
    print(caption_tensor.shape)







    

if __name__ == "__main__":
    generate_demo_sample()