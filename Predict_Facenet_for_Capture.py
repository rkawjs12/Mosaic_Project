from os import listdir
from os.path import isdir
from PIL import Image, ImageFilter
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import mtcnn
import numpy as np
import sys
import os

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import pickle
import cv2

# create the detector, using default weights
detector = MTCNN()

# get the face embedding for one face
# 이 함수는 자른 얼굴을 face embedding 시켜주는 함수이다.

def get_embedding(model, face_list):
    # face_list는 자른 얼굴을 가진 list이다.
    face_embedding_list = list()
    #face_embedding_list는 자른 얼굴을 face_embedding 시킨 결과이다.

    for i in range(len(face_list)):
        # scale pixel values]
        face_pixels = face_list[i].astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)    
        face_embedding_list.append(yhat[0])
    face_embedding_list = asarray(face_embedding_list)

    return face_embedding_list

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):

    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)

    # detect faces in the image
    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face
    face_list =[]
    # face_array는 자른 얼굴을 저장할 list로, 크기는 (160,160,3)
    coordinate_of_face_list =[]
    # coordinate_of_face_list는 자른 얼굴이 가지는 좌표를 나타내는 list로, 크기는 (n,5)이며, n은 detection된 얼굴의 수를 의미한다. 값은 x1, y1, x2, y2, 모자이크 유무(mosaic)를 의미한다.

    mosaic = 0

    #############################################################
    # 여기는 얼굴이 검색되지 않으면 그냥 원본 이미지를 face_list, coordinate_of_face에 넣는다.
    if len(results) == 0:
        
        face = pixels[0:image.size[1], 0:image.size[0]]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_list.append(asarray(image))
        coordinate_of_face_list.append([0,0,image.size[0],image.size[1],0])

        print(filename + " this image doesn't have face")
        return face_list, coordinate_of_face_list
    #############################################################

    else:
        for i in range(len(results)):
            x1, y1, width, height = results[i]['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_list.append(asarray(image))
            coordinate_of_face_list.append([x1,y1,x2,y2,mosaic])
            # face_list는 mtcnn을 통해 검출된 얼굴 array 형식으로 저장한 list이다. train_SVM_and_encodernate_of_face_list는 자른 얼굴의 좌표, face_list는 내가 모자이크 안하고자 하는 얼굴,
        return  face_list, coordinate_of_face_list


def svmPredict(face_embedding_list, SVM_file, out_encoder_file):
    # face_embedding_list는 잘린 얼굴을 face_embedding시켜 list에 저장한 것, coordinate_of_face_list는 자른 얼굴의 좌표, face_list는 내가 모자이크 안하고자 하는 얼굴,

    SVM_predict_result = []
    
    for i in range(len(face_embedding_list)):
        sample = expand_dims(face_embedding_list[i], axis = 0)

        yhat_class = SVM_file.predict(sample)
        #yhat_face_name = out_encoder_file.inverse_transform([yhat_class])
        yhat_prob = SVM_file.predict_proba(sample)*100

        class_index = yhat_class[0]
        predict_names = out_encoder_file.inverse_transform(yhat_class)
        class_probability = yhat_prob[0,class_index]
        SVM_predict_result.append([predict_names, class_probability])  
        print(SVM_predict_result[i])  

    return SVM_predict_result
    #SVM_predict_result에는 가장 확률에 해당하는 이름과 확률이다.

def load_SVM_out_encoder(dont_want_mosaic_facelist):
    # dont_want_mosaic_facelist를 통해서 out_encoder, SVM을 load하는 과정
    # 이 함수는 start_mosaic_function에서 1번만 호출된다.
    
    SVM_path = "/home/joker_92s/Mosaic_Project/SVM/"
    out_encoder_path = "/home/joker_92s/Mosaic_Project/out_encoder/"
    
    SVM_path_list = os.listdir(SVM_path)
    SVM_path_list.sort()
    
    out_encoder_path_list = os.listdir(out_encoder_path)
    out_encoder_path_list.sort()

    index_list_for_load_out_encoder = []
    index_list_for_load_SVM = []

    for dont_want_mosaic_face in dont_want_mosaic_facelist:
        index_list_for_load_SVM.append(SVM_path_list.index("SVM_" + dont_want_mosaic_face + ".pkl"))
        index_list_for_load_out_encoder.append(out_encoder_path_list.index("out_encoder_" + dont_want_mosaic_face + ".pkl"))
        
        
    SVM_list = []
    out_encoder_list = []    
    
    for index in index_list_for_load_SVM:
        SVM_list.append(pickle.load(open(SVM_path + SVM_path_list[index],'rb')))
    
    for index in index_list_for_load_out_encoder:
        out_encoder_list.append(pickle.load(open(out_encoder_path + out_encoder_path_list[index],'rb')))

    return SVM_list, out_encoder_list


def start_mosaic_function(imagefile , dont_want_mosaic_facelist, threshold):
# filename는 이미지를 의미하고, dont_want_mosaic_facelist는 모자이크를 하지 말아야하는 사ace_list.append(asarray(image))람들 list를 의미한다.
    
    SVM_list, out_encoder_list = load_SVM_out_encoder(dont_want_mosaic_facelist)
    model = load_model('/home/joker_92s/Mosaic_Project/facenet_keras_weight_module/facenet_keras.h5')
    
    # 여기서부턴 통신을 통해 이미지를 반복적으로 받고 보내는 과정이 필요하다.
    face_list, coordinate_of_face_list = extract_face(imagefile, required_size = (160,160))
    face_embedding_list = get_embedding(model,face_list)
    
    for j in range(len(dont_want_mosaic_facelist)):
        SVM_predict_result = svmPredict(face_embedding_list, SVM_list[j], out_encoder_list[j])
        
        for i, result in enumerate(SVM_predict_result):
            if (result[0][0] in dont_want_mosaic_facelist) and (result[1] >= threshold):
                coordinate_of_face_list[i][4] = 1
    image = Image.open(imagefile)

    for [x1,y1,x2,y2,mosaic] in coordinate_of_face_list:
        if mosaic == 0:
            cropped_image = image.crop((x1,y1,x2,y2))
            blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius = 10))
            image.paste(blurred_image,(x1,y1,x2,y2))

    #image.show()
    image.save('/home/joker_92s/Mosaic_Project/ITZY_result.jpg')
    return np.asarray(image)


dont_want_mosaic_facelist = ["유승우", "ITZY Chaeryeong"]
start_mosaic_function("/home/joker_92s/Mosaic_Project/ITZY_for_test.jpg", dont_want_mosaic_facelist, 80.00)
