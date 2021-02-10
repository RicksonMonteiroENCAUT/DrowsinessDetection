import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import cv2
import matplotlib.pyplot as plt
import time
#Constantes
ALARME_SONORO="buzina.wav"
EYE_THRESHOLD= 0.28
NUM_FRAMES_CONSEC=10
CONTADOR=0
ALARME_LIGADO=False

def disparar_alarme(path=ALARME_SONORO):
    """

    :param path: arquivo de audio
    :return: None
    """
    playsound.playsound(ALARME_SONORO)
    return None
def calcular_ratio_olho(eye):
    #calcula a distancia euclidiana entre os landmarks na vertical
    A= dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2], eye[4])
    #Calcula a distancia horizontal entre os landmarks do olho
    C=dist.euclidean(eye[0], eye[3])

    ear=(A+B)/(2.0*C)

    return ear

#carregar o dlib para face detector
detector= dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#pegar índices do previsor, para olhos esquerdos e direito
(lStart, lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#Inicializar vídeo
cap=cv2.VideoCapture(0)
time.sleep(2.0)
while cap.isOpened():
    ret, frame= cap.read()
    frame=cv2.resize(frame, (800,800), interpolation=cv2.INTER_AREA)
    #encontrar retângulos delimitadores dos rostos encontrados
    rects=detector(frame,0)

    for rect in rects:
        shape=predictor(frame, rect)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=calcular_ratio_olho(leftEye)
        rightEAR=calcular_ratio_olho(rightEye)

        #ratio médio
        ear=(leftEAR+rightEAR)/2.0

        #criando cotorno com os landmarks
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Olho está fechando?
        if ear <=EYE_THRESHOLD:
            CONTADOR+=1
            #dentro dos critérios, soar o alarme
            if CONTADOR>=NUM_FRAMES_CONSEC:
                #ligar alarme
                if not ALARME_LIGADO:
                    ALARME_LIGADO=True
                    t=Thread(target=disparar_alarme())
                    t.daemon=True
                    t.start()
                cv2.putText(frame, "[ALERTA] FADIGA!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        else:
            CONTADOR=0
            ALARME_LIGADO=False

            #desenhar a proporção de abertura dos olhos
            cv2.putText(frame, "{:.2f}".format(ear),(300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

