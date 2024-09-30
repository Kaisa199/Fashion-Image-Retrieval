import argparse
import torch
import json
import os
from PIL import Image 
from torchvision import transforms
from data_loader import get_loader
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder,DistilBertEncoder
from utils import Ranker
import nltk

DICT = 'data/captions/dict.{}.json'
IMAGE_ROOT = 'data/dress/'
CAPT = 'data/captions/cap.{}.{}.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
image_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20240930-134046/image-768.th"
caption_model_path = "/home/kaisa/Desktop/LEARN/Thsy/ir-dev/models/dress-20240930-134046/cap-768.th"
vocab = Vocabulary()
vocab.load(DICT.format("dress"))
SPLIT = 'data/image_splits/split.{}.{}.json'

def evaluate(args, image_model, caption_model, json_path):

    transform_test = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    vocab = Vocabulary()
    vocab.load(DICT.format(args.data_set))
    # Build data loader
    data_loader_test = get_loader(IMAGE_ROOT.format(args.data_set),
                                 CAPT.format(args.data_set, args.data_split),
                                 vocab, transform_test,
                                 args.batch_size, shuffle=False, return_target=False, num_workers=args.num_workers)
    ranker = Ranker(root=IMAGE_ROOT.format(args.data_set), image_split_file=SPLIT.format(args.data_set, args.data_split),
                    transform=transform_test, num_workers=args.num_workers)
    
    data_loader_test = get_loader(IMAGE_ROOT.format(args.data_set),
                    CAPT.format(args.data_set, args.data_split),
                    vocab, transform_test,
                    args.batch_size, shuffle=False, return_target=False, num_workers=args.num_workers)
    
    ranker.update_emb(image_model)
    image_model.eval()
    caption_model.eval()

    output = json.load(open(CAPT.format(args.data_set, args.data_split)))

    index = 0
    for _, candidate_images, captions, lengths, meta_info in data_loader_test:
        candidate_images = candidate_images.to(device)
        candidate_ft = image_model.forward(candidate_images)


        captions = captions.to(device)
        caption_ft = caption_model(captions, candidate_ft)

        rankings = ranker.get_nearest_neighbors(candidate_ft + caption_ft)

        for j in range(rankings.size(0)):
            output[index]['ranking'] = [ranker.data_asin[rankings[j, m].item()] for m in range(rankings.size(1))]
            index += 1

    json.dump(output, open("{}.{}.pred.json".format(args.data_set, args.data_split), 'w'), indent=4)
    print('eval completed. Output file: {}'.format("{}.{}.pred.json".format(args.data_set, args.data_split)))


def evaluate_metrics(json_path, pred_path):
    # Read the JSON files
    with open(json_path, 'r') as f:
        data = json.load(f)
    with open(pred_path, 'r') as f:
        pred = json.load(f)

    #Recall @1, 5, 10
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    for item in pred:
        target = item['target']
        ranks = item['ranking']
        if ranks[0] == target:
            recall_1 += 1
        if target in ranks[:10]:
            recall_5 += 1
        if target in ranks[:50]:
            recall_10 += 1

    recall_1 /= len(data)
    recall_5 /= len(data)
    recall_10 /= len(data)
    print(f"Recall@1: {recall_1:.4f}, Recall@10: {recall_5:.4f}, Recall@50: {recall_10:.4f}")
    



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--data_set', type=str, default='dress')
        parser.add_argument('--data_split', type=str, default='val')
        parser.add_argument('--crop_size', type=int, default=224,
                            help='size for randomly cropping images')
        parser.add_argument('--num_workers', type=int, default=8)
        args = parser.parse_args()

        json_path = "data/captions/cap.dress.val.json"
        image_model = torch.load(image_model_path).to(device)
        caption_model = torch.load(caption_model_path).to(device)
        # evaluate(args, image_model, caption_model, json_path)

        evaluate_metrics("data/captions/cap.dress.val.json", "dress.val.pred.json")