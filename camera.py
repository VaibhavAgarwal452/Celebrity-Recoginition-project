import cv2
import face_recognition
import os
import numpy as np
import pafy


class VideoCamera(object):

    # It is constructor in which we assiging some properties to it
    def __init__(self, url):
        self.url = url

        self.abc = pafy.new(url)
        self.best = self.abc.getbest(preftype="mp4")
        self.video = cv2.VideoCapture()
        self.video.open(self.best.url)
        self.images, self.classNames = self.find_images_and_classes()
        self.encodeListKnown = self.findEnodings(self.images)
        print("Encoding Complete")

    def __del__(self):
        self.video.release()

    # This method is for finding the images and their names and store  it in images and classNames Variable
    def find_images_and_classes(self):
        self.path = "faces"
        self.images = []
        self.classNames = []

        self.myList = os.listdir(self.path)

        for cl in self.myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])

        return self.images, self.classNames
    # This method is used for finding the encoding of the face

    def findEnodings(self, images):
        self.encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.encodeList.append(encode)
        return self.encodeList

    # This method is for getting the image frame by frame and finding the person's face
    def get_frame(self):

        ret, img = self.video.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(
                self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                self.encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if(matches[matchIndex]):
                name = self.classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


# import cv2


# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         ret, frame = self.video.read()
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()
