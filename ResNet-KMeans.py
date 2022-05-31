import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import PIL
from PIL import Image
from datetime import datetime

DIMS = 224
GRID_SIZE = 3

def create_resnet():
    #Create ResNet50-based model to output image embeddings
    resnet = ResNet50(include_top=False)
    input_ = resnet.layers[0].input
    output = resnet.layers[-2].output
    emb_model = Model(inputs=input_, outputs=output)
    return emb_model

def kmeans(emb_model, imgs, n_clusters=9, batch_size=4):
    #Run K-Means algorithm on image embeddings
    embs = emb_model.predict(imgs, batch_size=batch_size).reshape((len(imgs), -1))
    n = len(imgs)
    pca = PCA(n_components=n)
    pca.fit(embs)
    embs = pca.transform(embs)
    if (n < n_clusters):
        n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)
    centroids = kmeans.cluster_centers_
    return embs, centroids, n_clusters

for data_type in ["train", "test"]:
    print(f"Running with data_type = {data_type}")
    V_IDS_PATH = f"/home/ubuntu/CLIP-Charades/{data_type}_video_ids.txt" 
    IMAGES_PATH = f"/home/ubuntu/{data_type}_frames/"
    SAVE_PATH = f"/home/ubuntu/grid_{data_type}_frames/" 
    v = open(V_IDS_PATH, "r")
    
    print(f"Loading video ids at {datetime.now()} from {V_IDS_PATH}")
    Load video ids
    v_ids = {}
    for v_id in v:
        v_ids[v_id.split("\n")[0]] = []
    v.close()
    
    print(f"Finished loading {len(v_ids)} video ids at {datetime.now()}")
    
    print(f"Matching frame images at {IMAGES_PATH} to video ids (time={datetime.now()})")
    #Match frame images to video ids
    for file in os.listdir(IMAGES_PATH):
        if (file == 'captions.json'):
            continue
        v_ids[file.split("-")[0]].append(IMAGES_PATH + file)
    
    #Build ResNet50 embedding model
    emb_model = create_resnet()
    
    print(f"Loading and preprocessing frames for each video at {datetime.now()}")
    i = 0
    for v_id in v_ids:
        if i % 10 == 0:
            print(f"Processing {data_type} video {i} at {datetime.now()}")
        i += 1    
        #Load and preprocess frames for each video
        frames = v_ids[v_id]
        raw_imgs = []
        imgs = []
        for frame in frames:
            img = tf.keras.preprocessing.image.load_img(frame, target_size=(DIMS, DIMS))
            raw_imgs.append(img)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            imgs.append(img)
    
        np_imgs = np.zeros((len(imgs), DIMS, DIMS, 3))
        for n in range(len(imgs)):
            np_imgs[n] = imgs[n]
        imgs = np_imgs

        try:
            #Run K-means
            embs, centroids, n_clusters = kmeans(emb_model, imgs, n_clusters=9, batch_size=4)

            #Find frames closest to each centroid
            img_centroid_idxs = []
            for centroid in centroids:
                img_centroid_idxs.append(np.argmin([np.linalg.norm(emb-centroid) for emb in embs]))
            img_centroids = [raw_imgs[idx] for idx in img_centroid_idxs]

            #Build and save 3x3 grid of centroid key frames (or 2x2 if not enough frames for 3x3)
            if (n_clusters == 9):
                grid = Image.new('RGB', size=(DIMS*3, DIMS*3))
                grid_w, grid_h = grid.size
                grid.paste(img_centroids[0], box=(0, 0))
                grid.paste(img_centroids[1], box=(224, 0))
                grid.paste(img_centroids[2], box=(448, 0))
                grid.paste(img_centroids[3], box=(0, 224))
                grid.paste(img_centroids[4], box=(224, 224))
                grid.paste(img_centroids[5], box=(448, 224))
                grid.paste(img_centroids[6], box=(0, 448))
                grid.paste(img_centroids[7], box=(224, 448))
                grid.paste(img_centroids[8], box=(448, 448))
            else:
                grid = Image.new('RGB', size=(DIMS*2, DIMS*2))
                grid_w, grid_h = grid.size
                grid.paste(img_centroids[0], box=(0, 0))
                grid.paste(img_centroids[1], box=(224, 0))
                grid.paste(img_centroids[2], box=(0, 224))
                grid.paste(img_centroids[3], box=(224, 224))
            grid.save(SAVE_PATH + "/" + v_id + ".png", "png")
        except Exception as e:
            print(f"Error on image {v_id}: {e}")
