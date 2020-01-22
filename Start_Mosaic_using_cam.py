import cv2
import  os
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import Predict_Facenet_for_Capture
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import numpy as np

def start_mosaic_using_cam(dont_want_mosaic_facelist, threshold):
# 모자이크를 원하지 않는 사람의 얼굴을 매개변수로 전해준다.
# 다음 load_SVM_out_encoder를 통해 모자이크를 원하지 않는 사람의 SVM과 out_encoder를 가져온다.
    
    # 만약 없다면 없다고 해야한다.
    for i in dont_want_mosaic_facelist:
        index = i
        i = "SVM_" + index + ".pkl"
        j = "out_encoder_" + index + ".pkl"
        
        if i not in os.listdir("/home/joker_92s/Mosaic_Project/SVM/"):
            print("There is not {}".format(i))
            return
        if j not in os.listdir("/home/joker_92s/Mosaic_Project/out_encoder/"):
            print("There is not {}".format(j))
            return

    # 정상적으로 다 있는 경우
    SVM_list, out_encoder_list = Predict_Facenet_for_Capture.load_SVM_out_encoder(dont_want_mosaic_facelist)

    # 다음 얼굴을 embedding 시켜주는 model을 load한다.        
    model = load_model('/home/joker_92s/Mosaic_Project/facenet_keras_weight_module/facenet_keras.h5')

    cap = cv2.VideoCapture(0)
    cap.set(3, 1000)
    cap.set(4, 1000)

    while (True):
        # cam을 통해 capture한다 
        success, frame = cap.read()

        if success:
            face_list,coordinate_of_face_list, no_face = Predict_Facenet_for_Capture.extract_face(frame)

            if no_face == 0:
                # 얼굴이 하나도 검출되지 않은 경우를 확인한다.

                face_embedding_list = Predict_Facenet_for_Capture.get_embedding(model,face_list)
        
                for j in range(len(dont_want_mosaic_facelist)):
                    SVM_predict_result = Predict_Facenet_for_Capture.svmPredict(face_embedding_list, SVM_list[j], out_encoder_list[j])
                    
                    for i, result in enumerate(SVM_predict_result):
                        if (result[0][0] in dont_want_mosaic_facelist) and (result[1] >= threshold):
                            coordinate_of_face_list[i][4] = 1
                
                frame = Image.fromarray(frame)

                for [x1,y1,x2,y2,mosaic] in coordinate_of_face_list:
                    if mosaic == 0:
                        cropped_image = frame.crop((x1,y1,x2,y2))
                        blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius = 10))
                        frame.paste(blurred_image,(x1,y1,x2,y2))
            
            cv2.imshow("Camera Windwo", np.asarray(frame))

        else:
            print("캠에 연결하지 못했습니다.")
            return

        key = cv2.waitKey(1) & 0xFF
        if (key == 27):
            break
    
    cap.release()
    cv2.destroyAllWindows() 


start_mosaic_using_cam(["유승우","ITZY Yuna"], 99.99999999)