import Predict_Facenet
from PIL import Image
import cv2 

cap = cv2.VideoCapture(0)
# 0: default camera
# cap = cv2.VideoCapture("test.mp4") #동영상 파일에서 읽기


dont_want_mosaic_facelist = ["ITZY Yuna", "ITZY Yeji"]
threshold = 99.9999999

while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    frame = Image.fromarray(frame, 'RGB')

    Predict_Facenet.start_mosaic_function(frame, dont_want_mosaic_facelist, threshold)

    # print(frame.shape) == (480, 640, 3)
    print(type(frame))

    # 여기에 얼굴을 classification하고 mosaic을 진행하는 함수를 넣는다.
    # 결과를 frame에 넣어 밑에 문장을 실행한다.
    
    if success:
        # 프레임 출력
        cv2.imshow("Camera Window", frame)

        # ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if (key == 27):
            break

cap.release()
cv2.destroyAllWindows()

# 코드를 뜯어서 다시 조립할 필요가 있다. 
# 그리고 cap.isOpened() 형식에 맞게 변형하는 과정이 필요하다.
