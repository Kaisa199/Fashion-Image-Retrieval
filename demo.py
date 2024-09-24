import argparse
import os
import time
from PIL import Image 
import torch
import torch.nn as nn
from torchvision import transforms
from utils import create_exp_dir, Ranker
from data_loader import get_loader
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder, DistilBertEncoder
import wandb


def generate_demo_sample():
    # Load a single image
    image_path = "path/to/your/image.jpg"  # Replace with actual image path
    image = Image.open(image_path)
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension