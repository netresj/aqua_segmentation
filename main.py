# -*- coding:utf-8 -*-

import datetime, os
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_float
from sklearn.metrics import roc_auc_score
import pandas as pd
import tensorflow.keras.applications.resnet as resnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
import argparse
import glob


def preprocess(args):
    # read csv
    csv_path = glob.glob(f"{args.input_path}/**/{args.csv_name}")[0]
    df = pd.read_csv(csv_path)

    # split train and test data
    images = list(df["ImageId"].unique())
    train_size = int(len(images) * 0.8)
    train_images = images[:train_size]
    test_images = images[train_size:]
    with open(f"{args.preprocessed_data_path}/train.txt", "w") as f:
        for image_name in train_images:
            f.write(f"{image_name}\n")
    with open(f"{args.preprocessed_data_path}/test.txt", "w") as f:
        for image_name in test_images:
            f.write(f"{image_name}\n")    

    # create mask data
    for image_name in images:
        mask = np.zeros(256 * 1600)
        for rle in df[df["ImageId"]==image_name]["EncodedPixels"]:
            rle = rle.split(" ")
            start_px = [int(sp) for sp in rle[0::2]]
            length = [int(le) for le in rle[1::2]]
            for sp, le in zip(start_px, length):
                mask[sp-1: sp+le-1] = 1
        mask = np.reshape(mask, (256, 1600))
        np.save(f"{args.preprocessed_data_path}/{image_name}.npy", mask)        

def train(args):
    # prepare data
    with open(f"{args.preprocessed_data_path}/train.txt") as f:
        train_images = f.read().splitlines()
    with open(f"{args.preprocessed_data_path}/test.txt") as f:
        test_images = f.read().splitlines()
    X_train = np.array([
        imread(glob.glob(f"{args.input_path}/**/{filename}")[0])
        for filename in train_images
    ])
    X_train = X_train / 255
    X_test = np.array([
        imread(glob.glob(f"{args.input_path}/**/{filename}")[0])
        for filename in test_images
    ])
    X_test = X_test / 255
    y_train = np.array([
        np.load(f"{args.preprocessed_data_path}/{filename}.npy")
        for filename in train_images
    ])
    y_train = np.array([
        np.load(f"{args.preprocessed_data_path}/{filename}.npy")
        for filename in test_images
    ])

    # prepare model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(256, 1600, 3)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
    model.add(Dropout(0.5))
    model.add(Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(1, (1, 1), activation="sigmoid", padding="same"))

    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) \
                / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy',
        metrics=[AUC(), dice_coef]
        )

    # callback setting
    mc = ModelCheckpoint(
        filepath=f"{args.output_path}/model_weight.h5",
        save_best_only=True,
        save_weights_only=False
    )
    tb = TensorBoard(
        log_dir=args.log_path,
        histogram_freq=1
    )
    callbacks = [mc, tb]

    # fit
    model.fit(X_train, Y_train,
              epochs=20,
              batch_size=128,
              validation_data=(X_test, y_test), 
              callbacks=callbacks)

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='aqualium demo file')
    parser.add_argument('running_parten', type=str, help='both | train | preprocess')
    parser.add_argument('--input_path', default='/kqi/input/training')
    parser.add_argument('--output_path', default='/kqi/output/demo')
    parser.add_argument('--preprocessed_data_path', default='/kqi/output/preprocessed_data')
    parser.add_argument('--log_path', default='/kqi/output')
    parser.add_argument('--csv_name', default='train.csv')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.preprocessed_data_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    if args.running_parten == "both":
        preprocess(args)
        train(args)
    elif args.running_parten == "preprocess":
        preprocess(args)
    elif args.running_parten == "train":
        train(args)