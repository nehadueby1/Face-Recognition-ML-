#For collecting our dataset
import cv2
import numpy as np



#for deteting we use face haar case classifier
face_classifier= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#funtion for face extraction
def face_extactor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_classifier.detectMultiScale(gray,1.1,4)
    if face is():
        return None

    for (x,y,w,h) in face:
        cropped_face=img[y:y+h,x:x+h]
    return cropped_face

#opening video capture
cap=cv2.VideoCapture(0)
count=0

#aplling while condition
while True:
    ret,frame=cap.read()
    if face_extactor(frame) is not None:
        count+=1
        face=cv2.resize(face_extactor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        #here the path where we save or collect our samples
        file_path="C:/Users/Neha Dubey/PycharmProject/ML facercognition project/face sample/sample"+str(count)+'.jpg'
        cv2.imwrite(file_path,face)


        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Take Sample',face)

    else:
        print("face not found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("samples are collected!")
print("thank you")