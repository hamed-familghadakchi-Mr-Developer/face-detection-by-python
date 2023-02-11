# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(Hamed):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {Hamed}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import mediapipe as mp
import cv2
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5 ):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self,img,draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        print(self.results)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id,detection)
                #  mpDraw.draw_detection(img,detection)
                # print(id,detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)

                bboxC = detection.location_data.relative_bounding_box

                ih, iw, ic = img.shape  # hight, width, channel

                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id,bbox,detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)

                    cv2.putText(img, f' {int(detection.score[0] * 100)} %', (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        return img, bboxs

    def fancyDraw(self, img,bbox, l = 30, t = 5):
        x , y, w, h = bbox
        x1,y1 = x+w, y+h

        cv2.rectangle(img, bbox, (0, 255, 0), 1)

        #TOP LEFT X,Y
        cv2.line(img, (x,y), (x+l, y), (0, 255, 0), t)
        cv2.line(img, (x,y), (x, y+l), (0, 255, 0), t)
        #TOP RIGHT X1,Y
        cv2.line(img, (x1,y), (x1-l, y), (0, 255, 0), t)
        cv2.line(img, (x1,y), (x1, y+l), (0, 255, 0), t)
        #BOTTOM LEFT X,Y1
        cv2.line(img, (x,y1), (x+l, y1), (0, 255, 0), t)
        cv2.line(img, (x,y1), (x, y1-l), (0, 255, 0), t)
        #BOTTOM RIGHT X1,Y1
        cv2.line(img, (x1,y1), (x1-l, y1), (0, 255, 0), t)
        cv2.line(img, (x1,y1), (x1, y1-l), (0, 255, 0), t)


        return img
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceDetector()


    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__" :
    main()

