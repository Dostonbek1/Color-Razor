import numpy as np
from tkinter import *
import cv2
import tkinter.filedialog as filer
from PIL import Image, ImageTk

class ColorRazor:
    
    def __init__(self):
        self.file = ''
        self.grayImg = ''
        self.blurImg = ''
        self.dilateImg = ''
        self.erodeImg = ''
        self.skinDetectImg = ''
        self.cannyEdgeImg = ''
        self.faceDetectImg = ''
        self.eyeDetectImg = ''
        self.imgLabel = Label()

    def buildUI(self, root):
        self.ImgFrame = Frame(root, width=650, height=500, background="bisque")
        self.ImgFrame.pack()
        self.ImgFrame.place(x=20, y=80)

        image = Image.open("logo2.png")
        photo = ImageTk.PhotoImage(image)
        label = Label(image=photo)
        label.image = photo
        label.pack()
        label.place(x=320, y=0)   

        choose_file = Button(root, text="Choose File", command=self.chooseFile, height=2, width=18, bg='green', fg='white', font=20)
        choose_file.pack()
        choose_file.place(x=700, y=80)

        gray = Button(root, text="Gray", command=self.gray, height=1, width=9, bg='gray', fg='white', font=20)
        gray.pack()
        gray.place(x=700, y=135)

        blur = Button(root, text="Blur", command=self.blur, height=1, width=9, bg='lightgray', fg='white', font=20)
        blur.pack()
        blur.place(x=700, y=170)

        dilate = Button(root, text="Dilate", command=self.dilate, height=1, width=9, bg='red', fg='white', font=20)
        dilate.pack()
        dilate.place(x=700, y=205)

        erode = Button(root, text="Erode", command=self.erode, height=1, width=9, bg='red', fg='white', font=20)
        erode.pack()
        erode.place(x=700, y=240)

        skinDetect = Button(root, text="Skin Detect", command=self.skinDetect, height=1, width=9, bg='orange', fg='white', font=10)
        skinDetect.pack()
        skinDetect.place(x=700, y=275)

        cannyEdge = Button(root, text="Canny Edge", command=self.cannyEdge, height=1, width=9, bg='blue', fg='white', font=10)
        cannyEdge.pack()
        cannyEdge.place(x=700, y=310)

        faceDetect = Button(root, text="Face Detect", command=self.faceDetect, height=1, width=9, bg='lightblue', fg='black', font=10)
        faceDetect.pack()
        faceDetect.place(x=700, y=345)

        liveFaceDetect = Button(root, text="Live Face", command=self.liveFaceDetect, height=1, width=9, bg='lightblue', fg='black', font=10)
        liveFaceDetect.pack()
        liveFaceDetect.place(x=700, y=380)

        eyeDetect = Button(root, text="Eye Detect", command=self.eyeDetect, height=1, width=9, bg='purple', fg='white', font=10)
        eyeDetect.pack()
        eyeDetect.place(x=700, y=415)

        liveEyeDetect = Button(root, text="Live Eye", command=self.liveEyeDetect, height=1, width=9, bg='purple', fg='white', font=10)
        liveEyeDetect.pack()
        liveEyeDetect.place(x=700, y=450)

    def chooseFile(self):
        global test
        self.file = filer.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
        self.showImg()

    def showImg(self):
        img = cv2.imread(self.file)
        self.imageDisplay(img)

    def imageDisplay(self, image):
        b,g,r = cv2.split(image)
        img = cv2.merge((r,g,b))

        im = Image.fromarray(img)
        im = im.resize((650, 500))
        imgtk = ImageTk.PhotoImage(image=im) 

        self.imgLabel.pack_forget()
        self.imgLabel = Label(self.ImgFrame, image=imgtk, height=500, width=650)
        self.imgLabel.photo = imgtk
        self.imgLabel.pack()

    def gray(self):
        color = cv2.imread(self.file, 1)
        self.grayImg = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("Gray Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Gray Image", self.grayImg)

        graySave = Button(root, text="Save", command=test.graySave, height=1, width=7, bg='green', fg='white', font=20)
        graySave.pack()
        graySave.place(x=795, y=135)

    def graySave(self):
        cv2.imwrite("images/Gray.jpg", self.grayImg)
    
    def blur(self):
        print("Blur")
        image = cv2.imread(self.file, 1)
        self.blurImg = cv2.GaussianBlur(image,(5,55),0)
        self.imageDisplay(self.blurImg)

        blurSave = Button(root, text="Save", command=test.blurSave, height=1, width=7, bg='green', fg='white', font=20)
        blurSave.pack()
        blurSave.place(x=795, y=170)

    def blurSave(self):
        cv2.imwrite("images/Blur.jpg", self.blurImg)

    def dilate(self):
        image = cv2.imread(self.file, 1)
        kernel = np.ones((5, 5), 'uint8')
        self.dilateImg = cv2.dilate(image, kernel, iterations=1)
        self.imageDisplay(self.dilateImg)

        dilateSave = Button(root, text="Save", command=test.dilateSave, height=1, width=7, bg='green', fg='white', font=20)
        dilateSave.pack()
        dilateSave.place(x=795, y=205)

    def dilateSave(self):
        cv2.imwrite("images/Dilate.jpg", self.dilateImg)

    def erode(self):
        image = cv2.imread(self.file, 1)
        kernel = np.ones((5, 5), 'uint8')
        self.erodeImg = cv2.erode(image,kernel,iterations=1)
        self.imageDisplay(self.erodeImg)

        erodeSave = Button(root, text="Save", command=test.erodeSave, height=1, width=7, bg='green', fg='white', font=20)
        erodeSave.pack()
        erodeSave.place(x=795, y=240)

    def erodeSave(self):
        cv2.imwrite("images/Erode.jpg", self.erodeImg)

    def skinDetect(self):
        image = cv2.imread(self.file, 1)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]

        ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)
        ret, max_hue = cv2.threshold(h,15,255, cv2.THRESH_BINARY_INV)
        self.skinDetectImg = cv2.bitwise_and(min_sat,max_hue)
        cv2.imshow("Skin Detection", self.skinDetectImg)
        # self.imageDisplay(self.skinDetectImg)

        skinDetectSave = Button(root, text="Save", command=self.skinDetectSave, height=1, width=7, bg='green', fg='white', font=20)
        skinDetectSave.pack()
        skinDetectSave.place(x=795, y=275)

    def skinDetectSave(self):
        cv2.imwrite("images/Skin_Detect.jpg", self.skinDetectImg)

    def cannyEdge(self):
        image = cv2.imread(self.file, 1)
        self.cannyEdgeImg = cv2.Canny(image, 100, 70)
        cv2.imshow("Canny Edges", self.cannyEdgeImg)
        # self.imageDisplay(self.cannyEdgeImg)

        skinDetectSave = Button(root, text="Save", command=self.cannyEdgeSave, height=1, width=7, bg='green', fg='white', font=20)
        skinDetectSave.pack()
        skinDetectSave.place(x=795, y=310)

    def cannyEdgeSave(self):
        cv2.imwrite("images/Canny_Edges.jpg", self.cannyEdgeImg)

    def faceDetect(self):
        self.faceDetectImg = cv2.imread(self.file, 1)
        gray = cv2.cvtColor(self.faceDetectImg, cv2.COLOR_BGR2GRAY)
        path = "data/haarcascade_frontalface_default.xml"

        face_cascade = cv2.CascadeClassifier(path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))

        for (x, y, w, h) in faces:
            cv2.rectangle(self.faceDetectImg, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.imshow("Face Detect", self.faceDetectImg)
        self.imageDisplay(self.faceDetectImg)

        faceDetectSave = Button(root, text="Save", command=self.faceDetectSave, height=1, width=7, bg='green', fg='white', font=20)
        faceDetectSave.pack()
        faceDetectSave.place(x=795, y=345)

    def faceDetectSave(self):
        cv2.imwrite("images/Face_Detect.jpg", self.faceDetectImg)

    def liveFaceDetect(self):
        cap = cv2.VideoCapture(0)
        path = "data/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(path)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0,0), fx=0.9, fy=0.9)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imshow("Face Detect", frame)
            # self.imageDisplay(frame)

            ch = cv2.waitKey(10)
            if ch & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def eyeDetect(self):
        self.eyeDetectImg = cv2.imread(self.file, 1)
        gray = cv2.cvtColor(self.eyeDetectImg, cv2.COLOR_BGR2GRAY)
        path = "data/haarcascade_eye.xml"

        eye_cascade = cv2.CascadeClassifier(path)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=5, minSize=(10,10))

        for (x, y, w, h) in eyes:
            xc = (x + x+w)/2
            yc = (y + y+h)/2
            radius = w/2
            cv2.circle(self.eyeDetectImg, (int(xc),int(yc)), int(radius), (255,0,0), 2)
        # cv2.imshow("Eye Detect", self.eyeDetectImg)
        self.imageDisplay(self.eyeDetectImg)

        eyeDetectSave = Button(root, text="Save", command=self.eyeDetectSave, height=1, width=7, bg='green', fg='white', font=20)
        eyeDetectSave.pack()
        eyeDetectSave.place(x=790, y=415)

    def eyeDetectSave(self):
        cv2.imwrite("images/Eye_Detect.jpg", self.eyeDetectImg)

    def liveEyeDetect(self):
        cap = cv2.VideoCapture(0)
        path = "data/haarcascade_eye.xml"
        face_cascade = cv2.CascadeClassifier(path)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0,0), fx=0.9, fy=0.9)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=10, minSize=(10,10))

            for (x, y, w, h) in eyes:
                xc = (x + x+w)/2
                yc = (y + y+h)/2
                radius = w/2
                cv2.circle(frame, (int(xc),int(yc)), int(radius), (255,0,0), 2)
            cv2.imshow("Live Eye Detect", frame)
            # self.imageDisplay(frame)

            ch = cv2.waitKey(10)
            if ch & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    global test, root
    root = Tk()
    root.geometry('900x700+300+50')
    root.resizable(0, 0)
    root.configure(background='white')
    root.title("Color Razor")

    test = ColorRazor()
    test.buildUI(root)

    root.mainloop()

if __name__ == '__main__':
    main()
