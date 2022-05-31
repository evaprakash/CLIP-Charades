import json
import pickle
import cv2
import os
import csv
import numpy as np

N = 10
DATA_PATH = '../data/'
SAVE_PATH = '../extra_train_frames/'
CAPTIONS_FILE = 'Charades_v1_train.csv'
V_IDS_PATH = "extra_train_video_ids.txt"

# Function to extract frames
def frame_capture(v_id):
    # Path to video file
    vidObj = cv2.VideoCapture(DATA_PATH + '%s.mp4' % v_id)

    # Used as counter variable
    count = 0

    # Checks whether frames were extracted
    success, image = vidObj.read()
    
    while success:

        if count % N == 0:
    
            frame_url = '%s-%s.png' % (v_id, count)

            # Saves the frames with frame-count
            cv2.imwrite(SAVE_PATH + frame_url, image)

        count += 1
        
        # vidObj object calls read
        # function to extract frames
        success, image = vidObj.read()

    #print("Done ", v_id)

v_ids = []
v_ids_file = open(V_IDS_PATH, "r")
for v_id in v_ids_file:
    v_ids.append(v_id.split('\n')[0])

captions = {}

#Consolidate captions for V videos
with open(CAPTIONS_FILE, 'r') as file:
    reader = csv.reader(file)
    i = 0
    for row in reader:
        if (i == 0):
            i += 1
            continue
        if (row[0] in v_ids):
            captions[row[0]] = row[6]
            i += 1

#with open(SAVE_PATH + 'captions.json', 'w') as f:
#    json.dump(captions, f)

#Create frame images and image data
i = 0
for v_id in captions:
    frame_capture(v_id)
    if (i % 50 == 0):
        print("Done ", i, " videos")
    i += 1
