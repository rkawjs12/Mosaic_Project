import os
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import mtcnn
import numpy as np
from numpy import save
import sys

# global variable for non-face-image
nonface_array = np.zeros((160,160,3))

# extract a single face from a given photograph
# 학습하는 과정에선 하나의 이미지에는 학습 되고자 하는 한명의 얼굴이 1개만 있다고 가정하는게 프로젝트의 usecase에 맞아, 하나의 이미지에 얼굴이 여러개 있는 경우를 생각하지 않았다.
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face

    if len(results) == 0:
        print(filename + " this image doesn't have face")
        return  np.zeros((160,160,3)) 

    else :
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        
        # 얼굴이 검출되지 않았을 경우
        if (face == nonface_array).all():
            continue
        # 얼굴이 검출된 경우    
        else:
            faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X = list()
    y = list()
    
    subdir_list = list()
    
    # enumerate folders, on per class
    for subdir in listdir(directory):        
        subdir_list.append(subdir)
        
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        # 폴더에 이미지가 없는 경우 다음 이미지로 넘어간다.
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y), subdir_list, y

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    
    return yhat[0]

# get_embedding된 것이 하나도 없고, 처음으로 학습을 진행하는 경우로, 이미지 100개를 가진 폴더가 몇개 존재하며 폴더의 이름은 y값으로 될 것이다.
def trainY_and_embedding():
    model_path = "facenet_keras_weight_module/facenet_keras.h5"
    image_folder_path = '/home/joker_92s/Mosaic_Project/image/'
    
    trainX, trainY, subdir_list, y= load_dataset(image_folder_path)
    
    save('trainY',trainY)
    
    # print(trainX.shape, trainY.shape)
    newtrainX = list()
    
    # load the facenet model
    model = load_model(model_path)
    
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newtrainX.append(embedding)
    
    save('/home/joker_92s/Mosaic_Project/face_embedding/face_embedding', newtrainX)
    
    print(len(newtrainY))
    
    subdir_index_list = list()
    
    others_list = ['others' for _ in range(len(trainY))]
   
    # trainY의 구분점을 찾기위한 과정 + trainY에서 others 변수 만드는 과정
    mod = sys.modules[__name__]
    
    for subdir in subdir_list:
        subdir_index_list.append(y.index(subdir))
    
    for i, subdir in enumerate(subdir_list):
        if i == len(subdir_index_list ) - 1: #subdir_index_list의 마지막 index일 경우, range(subdir_index_list[i] : len(trainT))
                for index in range(subdir_index_list[i], len(trainY)):
                    others_list[index] = subdir
                    
                save('/home/joker_92s/Mosaic_Project/trainY/trainY_{}'.format(subdir),asarray(others_list))
                
                others_list = ['others' for _ in range(len(trainY))]

        else: 
            for index in range(subdir_index_list[i], subdir_index_list[i + 1]):
                others_list[index] = subdir     
                
            save('/home/joker_92s/Mosaic_Project/trainY/trainY_{}'.format(subdir),asarray(others_list))
            others_list = ['others' for _ in range(len(trainY))]