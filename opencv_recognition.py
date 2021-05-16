# mporting the libraries
import cv2
import face_recognition
import os
import numpy as np
import pafy

# initializing properties
path = "faces"
images = []
classNames = []

# finding out in which directory we are in
myList = os.listdir(path)

# filling out the values in images and className
# as we know we are in we are in our project folder
for cl in myList:
    # Reading the images in the faces folder which is our dataset by the way
    curImg = cv2.imread(f'{path}/{cl}')

    # Appending the images in images list
    images.append(curImg)

    # Appending the classNames in className list
    classNames.append(os.path.splitext(cl)[0])

# This function is used for finding the encoding the faces  which help in recognizing the faces


def findEnodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# We save all the encodings in the encodeListKnown Variable
encodeListKnown = findEnodings(images)
print("Encoding Complete")

# here we Put the url of the youtube video in which we want to recognize the faces
url = "https://www.youtube.com/watch?v=2lPkvteqKig"

video = pafy.new(url)
best = video.getbest(preftype="mp4")

cap = cv2.VideoCapture()
cap.open(best.url)

# IF we want to load a video from local storage then
#cap = cv2.VideoCapture("You video name")

# IF you want to read a image then just
#cap = cv2.imread("Image name")

while True:
    # we are  reading the frame of video and storing it in img variable
    success, img = cap.read()
    # we are resizing the image as it saves some memory and make it efficient
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Here are are finding the face location and face encoding of the given video's frame
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        # Here we get the faces match and face description in list
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Here we find the index of  minmum distance
        matchIndex = np.argmin(faceDis)

        # Now we find the index in matches and draw the circle around it and put a name on it
        if(matches[matchIndex]):
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Here it will show the respective image or video
    cv2.imshow('Webcam', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
