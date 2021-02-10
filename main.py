#importando bibliotecas
import cv2
import numpy as np
from imutils import face_utils
import dlib
from time import sleep

#iniciando o detctor facial
detector = dlib.get_frontal_face_detector()
#preditor de landmarks (68)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Iniciando WebCam
cap=cv2.VideoCapture(0)

while cap.isOpened():
    #leitura e redimensionamento do frame
    ret, frame= cap.read()
    frame=cv2.resize(frame, (500,500), interpolation=cv2.INTER_AREA)
    #Detecção da face(retangulo delimitador)
    rects = detector(frame, 1)
    #interando sobre cada retangulo
    for rect in rects:
        #obtendo o shape e convertando para uma lista contendo a posicção (x,y) de cada landmark
        shape=predictor(frame, rect)
        cords=face_utils.shape_to_np(shape)
        print(cords)
        #desenhando pontos landmarks
        for c in cords:
            cv2.circle(frame,(c[0],c[1]),1,(0,255,0),1)

    cv2.imshow('frame', frame)
    key= cv2.waitKey(1)& 0xFF
    if key== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()