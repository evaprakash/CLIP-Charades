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

TRAIN_FRAMES_PATH = "/home/ubuntu/grid_train_frames"#"/home/ubuntu/train_frames"
TEST_FRAMES_PATH = "/home/ubuntu/grid_test_frames"#"/home/ubuntu/test_frames"
TRAIN_BATCH_SIZE = 50 #26957 total train images for baseline; 1000 for kmeans
TEST_BATCH_SIZE = 50 #11949 total test images for baseline; 500 for kmeans
MAX_CAPTION_COUNT = 5000
CONTEXT_LEN = 77

# Class that takes in data and serves it to pytorch when requested
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
        img = Image.open(f"{self.path}/{self.images[idx]}")
        images = self.preprocess(img)
        caption = self.caption[idx]
        return images, caption

#Loads training and testing captions + videos
def load_data():
    # Load train captions and videos
    captions = json.load(open(f"{TRAIN_FRAMES_PATH}/captions.json"))
    train_captions = []
    train_images = []
    for file in os.listdir(TRAIN_FRAMES_PATH):
        if (file == 'captions.json'):
            continue
        train_images.append(file)
        train_captions.append(captions[file[:-4].split("-")[0]])

    # Load test captions and videos
    captions = json.load(open(f"{TEST_FRAMES_PATH}/captions.json"))
    test_captions = []
    test_images = []
    for file in os.listdir(TEST_FRAMES_PATH):
        if (file == 'captions.json'):
            continue
        test_images.append(file)
        test_captions.append(captions[file[:-4].split("-")[0]])
    return train_captions, train_images, test_captions, test_images


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# Fine-tune CLIP model, saving it when the train accuracy improves
def train_and_save_model(train_captions, train_images, model_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    print("Using device:", device)
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    print("Model loaded")

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    
    # Create custom PyTorch dataset containing images and corresponding captions
    df = {'image': train_images, 'caption': train_captions}
    print(len(train_images))
    dataset = image_caption_dataset(df, preprocess)
    train_dataloader = tor.DataLoader(dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)


    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper

    EPOCH = 150
    best_acc = 0
    for epoch in range(EPOCH):
      print(f"Starting epoch: {epoch} at {datetime.now()}")
      b = 0
      for batch in train_dataloader:
          if b % 10 == 0:
              print(f"Starting batch: {b} at {datetime.now()}")
          optimizer.zero_grad()

          list_image, list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
          images = list_image.to(device)
          texts = clip.tokenize(list_txt).to(device)
        
          logits_per_image, logits_per_text = model(images, texts)

          ground_truth = torch.arange(TRAIN_BATCH_SIZE, dtype=torch.long, device=device)

          total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
          total_loss.backward()
          if device == "cpu":
             optimizer.step()
          else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
          b += 1
      correct, img_to_prob = test_model(model, preprocess, device, train_images, train_captions, TRAIN_BATCH_SIZE, False)
      if correct > best_acc:
          best_acc = correct
          print(f"best accuracy so far: {best_acc}")
          torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': total_loss,
                  }, model_path) #just change to your preferred folder/filename
    #Save trained model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, model_path) #just change to your preferred folder/filename

# Test the model by running it on a data set and comparing predictions to ground truth
def test_model(model, preprocess, device, images, captions, batch_size, test_data):
    print(f"Testing model on data with test_data={test_data}")
    captions = [c[:CONTEXT_LEN] for c in captions]
    df = {'image': images, 'caption': captions}
    dataset = image_caption_dataset(df, preprocess, test_data=test_data)
    dataloader = tor.DataLoader(dataset, batch_size = batch_size)
    print(f"Created dataloader with batch size {batch_size}")

    caption_chunks = [captions[i:i + MAX_CAPTION_COUNT] for i in range(0, len(captions), MAX_CAPTION_COUNT)]
    tokenized_chunks = [clip.tokenize(chunk).to(device) for chunk in caption_chunks]
    print(f"Split {len(captions)} into {len(tokenized_chunks)} chunks of sizes {[len(x) for x in caption_chunks]}")
    img_to_prob = {}

    correct = 0
    batch_num = 0
    for batch in dataloader:
        with torch.no_grad():
            batch_images, _ = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
            device_images = batch_images.to(device)

            full_logits = []
            for chunk in tokenized_chunks:
                logits_per_chunk, _ = model(device_images, chunk)
                full_logits.append(logits_per_chunk)
            concat_logits = torch.cat(full_logits, dim=1)
            probs = concat_logits.softmax(dim=-1).cpu().numpy()

            start_idx = batch_num * batch_size
            correct += np.sum(np.max(probs, axis=1) == [probs[i, start_idx + i] for i in range(len(batch_images))])
            print(f"{correct} out of {batch_size} in batch {batch_num} (time={datetime.now()})")
            batch_num += 1
            for x in range(len(batch_images)):
                img_to_prob[images[start_idx + x]] = probs[x].tolist()
                if np.max(probs[x]) == probs[x, start_idx + x]:
                    print("Correct: ", images[start_idx + x])
    return correct, img_to_prob

#Test model on train and test set and report accuracies
def load_and_test_model(train_captions, train_images, test_captions, test_images, model_path):
    #Load trained model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    checkpoint = torch.load(model_path)

    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
    #checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
    #checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
    #checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {model_path}")

    #Calculate test accuracy
    correct, img_to_prob = test_model(model, preprocess, device, test_images, test_captions, TEST_BATCH_SIZE, True)
    print(f"TEST SET: {correct} out of {len(test_images)} for an accuracy of {correct / len(test_images)}")
    with open("test_probs.json", "w") as f:
        f.write(json.dumps(img_to_prob))
    
    #Calculate train accuracy
    correct, img_to_prob = test_model(model, preprocess, device, train_images, train_captions, TRAIN_BATCH_SIZE, False)
    print(f"TRAIN SET: {correct} out of {len(train_images)} for an accuracy of {correct / len(train_images)}")
    with open("train_probs.json", "w") as f:
        f.write(json.dumps(img_to_prob))
    
model_path = "/home/ubuntu/saved_models/kMeans_150_epoch.pt"

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
