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

DICT = 'data/captions/dict.{}.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
image_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20241002-130745/image-768.th"
caption_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20241002-130745/cap-768.th"
image_model = torch.load(image_model_path).to(device)
caption_model = torch.load(caption_model_path).to(device)
vocab = Vocabulary()
vocab.load(DICT.format("dress"))

def generate_emb(image_path, caption_texts, img_name, save_embeds):
    try:
        # Load a single imageReplace with actual image path
        image = Image.open(image_path)
        
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        k = 64  # Number of times to repeat the image tensor
        input_tensor = torch.cat([transform(image).unsqueeze(0) for _ in range(k)], dim=0).to(device)
        
        tokens = nltk.tokenize.word_tokenize(str(caption_texts[0]).lower()) + ['<and>'] + \
                nltk.tokenize.word_tokenize(str(caption_texts[1]).lower())
        

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))

        caption = torch.LongTensor(caption).unsqueeze(0)
        captions = torch.cat([caption] * k, dim=0).to(device)

        image_tensor = image_model(input_tensor)
        caption_tensor = caption_model(captions, image_tensor)
        #convert to numpy
        caption_tensor = caption_tensor.cpu().detach().numpy()
        save_embeds[img_name] = caption_tensor[0].tolist()
        #save json file
    except Exception as e:
        print(e)
        pass






#read cap.dress.test.json
def save_embeddings():
    with open('data/captions/cap.dress.test.json', 'r') as f:
        data = json.load(f)
    num_sample = 0
    save_embeds = {}
    max_sample = 1000
    for item in tqdm(data[:max_sample]) :
        img_name = item['candidate']
        img_path = 'data/dress/' + img_name + '.jpg'
        captions = item['captions']
        generate_emb(img_path, captions, img_name, save_embeds)
    
    #save json file
    with open('data/embeddings.json', 'w') as f:
        json.dump(save_embeds, f)



if __name__ == "__main__":
    # generate_demo_sample()
    save_embeddings()
