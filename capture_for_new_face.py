import cv2
import os
from PIL import Image,ImageFont,ImageDraw
from mtcnn.mtcnn import MTCNN
from numpy import asarray
import numpy as np


def draw_text_in_image( str, frame2, font = '/home/joker_92s/.local/share/fonts/SDMiSaeng.ttf'):
    font_path = "/home/joker_92s/.local/share/fonts/yg-jalnan.ttf"
    font = ImageFont.truetype(font_path, 20)    
    img_pil = Image.fromarray(frame2)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0,0), str, font = font, fill = (0, 255, 0, 0))
    frame2 = np.array(img_pil)

    return frame2

def start_capture_for_new_face(new_face_name):
    # new_face_name은 추가하고자 하는 사용자의 이름으로, 방송플랫폼에선 고유 아이디와 동일해야한다. 절대 중복이 있으면 안된다.

    # MTCNN에 대한 부분에서  detect_faces()를 공부해야한다.
    # 객체로 만들었을 경우 여러 이미지를 .detect_faces()를 했을 때, 남아있는지 아니면 한장하고 다른 이미지를 하면 그냥 새로한 이미지에 대한 부분의 정보만 가지고 있는지 확인해야한다.
    
    image_path = '/home/joker_92s/Mosaic_Project/image2/'

    detector = MTCNN()    
    i = 0
    cap =  cv2.VideoCapture(0)
    
    while i < 100:
        # 100장의 이미지를 찍는다. 각 이미지는 1명의 얼굴만 있어야 한다. 동일한 얼굴이여야 한다.
        success, frame = cap.read()

        if success:

            results = detector.detect_faces(frame)
            frame2 = frame.copy()
                
            if len(results) == 0:
                # 얼굴이 없는 경우는 다시 찍어야 한다.
                str = "No Face"
                frame2 = draw_text_in_image(str,frame2)

            elif len(results) >= 2:
                # 얼굴이 2개 이상일 경우 다시 찍어야 한다.
                str = "두개 이상의 얼굴이 화면에 있습니다."
                frame2 = draw_text_in_image(str,frame2)
                
                for j in range(len(results)):
                    # detection된 얼굴에 영역표시
                    x1, y1, width, height = results[j]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    cv2.rectangle(frame2, (x1,y1),(x2,y2), (0,255,0), 3)
                    cv2.waitKey(10)

            elif len(results) == 1:
                # 얼굴이 하나일 경우
                
                str = "본인의 얼굴만 화면에 있으면, Enter를 눌러 이미지를 저장해주세요."
                frame2 = draw_text_in_image(str,frame2)

                for j in range(len(results)):
                    # detection된 얼굴에 영역표시
                    x1, y1, width, height = results[j]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    cv2.rectangle(frame2, (x1,y1),(x2,y2), (0,255,0), 3)

                 # Enter를 누르면 이미지 저장
                key = cv2.waitKey(1) & 0xFF
                if (key == 13):
                    if os.path.isdir(image_path + "{}".format(new_face_name)):
                        # 폴더가 있다면
                        cv2.imwrite(image_path+ "{}/{}_{}.jpg".format(new_face_name, new_face_name, i), frame)
                        i = i + 1

                    else :
                        # 폴더가 없다면
                        os.mkdir(image_path + "{}".format(new_face_name))
                        cv2.imwrite(image_path+ "{}/{}_{}.jpg".format(new_face_name, new_face_name, i), frame)
                        i = i + 1
                 
                elif (key == 27):
                    break

            # 프레임 출력
            cv2.imshow("Camera Window", frame2)
        
        else :
            print("카메라를 인식할 수 없습니다.")
        
        key = cv2.waitKey(1) & 0xFF
        if (key == 27):
            break


    cap.release()
    cv2.destroyAllWindows() 

start_capture_for_new_face("유승우")


