import torch
import clip
from PIL import Image
import cv2
import os
import json
import torch.utils.data as tor
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime

TRAIN_FRAMES_PATH = "/home/ubuntu/train_frames"
TEST_FRAMES_PATH = "/home/ubuntu/test_frames"
TRAIN_BATCH_SIZE = 50 #26957 total train images for baseline; 1000 for kmeans
TEST_BATCH_SIZE = 50 #11949 total test images for baseline; 500 for kmeans
MAX_CAPTION_COUNT = 5000
CONTEXT_LEN = 77
'''
This file is similar to the BaselineModel.py file, except it reduces the number of captions to make CLIP faster.
In the augmented model, we have 1 caption for each video, but in the baseline model we have one caption for each
frame, leading to repeats when multiple frames come from the same video. This file condendes the captions.
'''

class image_caption_dataset(tor.Dataset):
    # pass preprocess function from clip.load()
    def __init__(self, df, preprocess, test_data = False):
        self.images = df["image"]
        self.caption = df["caption"]
        self.preprocess = preprocess
        self.path = TEST_FRAMES_PATH if test_data else TRAIN_FRAMES_PATH

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        filename = self.images[idx]
        vid_id = filename[:-4].split("-")[0]
        img = Image.open(f"{self.path}/{filename}")
        images = self.preprocess(img)
        caption = self.caption[vid_id]
        return images, caption, vid_id

def load_data():
    # Load train captions and videos
    train_captions = json.load(open(f"{TRAIN_FRAMES_PATH}/captions.json"))
    train_images = []
    for file in os.listdir(TRAIN_FRAMES_PATH):
        if (file == 'captions.json'):
            continue
        train_images.append(file)
        #train_captions.append(captions[file[:-4].split("-")[0]])
    for k, v in train_captions.items():
        train_captions[k] = v[:CONTEXT_LEN]

    # Load test captions and videos
    test_captions = json.load(open(f"{TEST_FRAMES_PATH}/captions.json"))
    test_images = []
    for file in os.listdir(TEST_FRAMES_PATH):
        if (file == 'captions.json'):
            continue
        test_images.append(file)
        #test_captions.append(captions[file[:-4].split("-")[0]])
    for k, v in test_captions.items():
        test_captions[k] = v[:CONTEXT_LEN]
    return train_captions, train_images, test_captions, test_images

def test_model(model, preprocess, device, images, captions, batch_size, test_data):
    print(f"Testing model on data with test_data={test_data}")
    df = {'image': images, 'caption': captions}
    dataset = image_caption_dataset(df, preprocess, test_data=test_data)
    dataloader = tor.DataLoader(dataset, batch_size = batch_size)
    print(f"Created dataloader with batch size {batch_size}")

    caption_list = list(captions.values())
    id_to_idx = {k: caption_list.index(v) for k, v in captions.items()}
    caption_chunks = [caption_list[i:i + MAX_CAPTION_COUNT] for i in range(0, len(caption_list), MAX_CAPTION_COUNT)]
    tokenized_chunks = [clip.tokenize(chunk).to(device) for chunk in caption_chunks]
    print(f"Split {len(caption_list)} into {len(tokenized_chunks)} chunks of sizes {[len(x) for x in caption_chunks]}")
    img_to_prob = {}

    correct = 0
    batch_num = 0
    for batch in dataloader:
        with torch.no_grad():
            batch_images, _, vid_ids = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
            device_images = batch_images.to(device)
            #print("Video ids =", vid_ids)
            #print("indices =", [id_to_idx[vid_ids[i]] for i in range(len(batch_images))])
            full_logits = []
            for chunk in tokenized_chunks:
                logits_per_chunk, _ = model(device_images, chunk)
                full_logits.append(logits_per_chunk)
            concat_logits = torch.cat(full_logits, dim=1)
            probs = concat_logits.softmax(dim=-1).cpu().numpy()

            start_idx = batch_num * batch_size
            correct += np.sum(np.max(probs, axis=1) == [probs[i, id_to_idx[vid_ids[i]]] for i in range(len(batch_images))])
            print(f"{correct} correct at batch {batch_num} (time={datetime.now()})")
            batch_num += 1
            for x in range(batch_size):
                img_to_prob[images[x]] = probs[x].tolist()
    
    return correct, img_to_prob, id_to_idx

#@profile
def load_and_test_model(train_captions, train_images, test_captions, test_images, model_path):
    #Load trained model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {model_path}")
    
    #Calculate test accuracy
    correct, img_to_prob, img_to_idx = test_model(model, preprocess, device, test_images, test_captions, TEST_BATCH_SIZE, True)
    temp = {"probs": img_to_prob, "idx": img_to_idx}
    with open("test_probs_baseline.json", "w") as f:
        f.write(json.dumps(temp))
    print(f"TEST SET: {correct} out of {len(test_images)} for an accuracy of {correct / len(test_images)}")
    
    #Calculate train accuracy
    correct, img_to_prob, img_to_idx = test_model(model, preprocess, device, train_images, train_captions, TRAIN_BATCH_SIZE, False)
    temp = {"probs": img_to_prob, "idx": img_to_idx}
    with open("train_probs_baseline.json", "w") as f:
        f.write(json.dumps(temp))
    print(f"TRAIN SET: {correct} out of {len(train_images)} for an accuracy of {correct / len(train_images)}")
    

model_path = "/home/ubuntu/saved_models/baseline_model_FULL.pt"

load_start = time.time()
train_captions, train_images, test_captions, test_images = load_data()
load_end = time.time()
print(f"Loading data took {load_end - load_start} seconds")

#train_and_save_model(train_captions, train_images, model_path)
train_end = time.time()
print(f"Training and saving the model took {train_end - load_end} seconds")

load_and_test_model(train_captions, train_images, test_captions, test_images, model_path)
test_end = time.time()
print(f"Loading and testing the model took {test_end - train_end} seconds")
