import os
import numpy as np
import cv2
from numpy.lib.function_base import average
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from train import iou
from sklearn.metrics import jaccard_score
from PIL import Image, ImageDraw, ImageFont
from data import load_data, tf_dataset

def read_imagew(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_maskw(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == "__main__":
    ## Dataset

    path = "Covidd"
    batch_size = 8
    (train_x, train_y), (valid_x, valid_y)  ,(test_x, test_y) = load_data(path)
    
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("files/model.h5")


    lr = 1e-4
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)


    #model.evaluate(test_dataset, steps=test_steps)

    meanscore = 0
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_imagew(x)
       
        

        y = read_maskw(y)
        
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        y_bin  = y_pred
        y_pred = np.float32(y_pred)*255

        

        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0


        X = np.uint8(np.ones((256,256,3)))
        
        X[:,:,0] = x[:,:,0]*255
        X[:,:,1] = x[:,:,0]*255
        X[:,:,2] = x[:,:,0]*255
        
        y_pred = y_pred.astype('float64')
        

        A = np.uint8(np.ones((256,256,3)))
        A[:,:,0] = y_pred[:,:,0]
        A[:,:,1] = y_pred[:,:,0]
        A[:,:,2] = y_pred[:,:,0]

        x = X
        
       
        
        orimage = cv2.bitwise_and(x,A)
        


        a = mask_parse(y).flatten()
        c = mask_parse(y_bin).flatten()
        a = a/255
        a = a.astype('int8')

        score = jaccard_score(a,c,zero_division={0.0,1.0})
        score = round(score,2)
        meanscore = meanscore + score
        mean = meanscore/(i+1)
        meanr = round(mean,4)
        
        img = Image.new('L', (256, 256), color = (0))

        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf",size=25)
        d.text((0,0), 'Jaccard Score :'+str(score)  , fill=(255),font=font)
        d.text((0,50), 'mean :'+str(meanr) , fill=(255) , font=font)


        img = np.expand_dims(img, axis=-1)

        S = np.uint8(np.ones((256,256,3)))
        S[:,:,0] = img[:,:,0]
        S[:,:,1] = img[:,:,0]
        S[:,:,2] = img[:,:,0]

        
        


        print(type(img),img.shape,orimage.shape)

        all_images = [
            x, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred), white_line,
            orimage,white_line,
            S
        ]
        

       

       
      
   
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)
