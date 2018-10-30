import numpy as np
from tkinter import *
import cv2
import tkinter.filedialog as filer
from PIL import Image, ImageTk

class Colorize:
    def __init__(self):
        self.file = ''
        self.grayImg = ''
        self.blurImg = ''
        self.dilateImg = ''
        self.erodeImg = ''

    def gray(self):
        color = cv2.imread(self.file, 1)
        self.grayImg = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("Gray Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Gray Image", self.grayImg)

        graySave = Button(root, text="Save", command=test.graySave, height=2, width=7, bg='green', fg='white', font=20)
        graySave.pack()
        graySave.place(x=790, y=160)

    def graySave(self):
        cv2.imwrite("grayImg.jpg", self.grayImg)
    
    def blur(self):
        print("Blur")
        image = cv2.imread(self.file, 1)
        self.blurImg = cv2.GaussianBlur(image,(5,55),0)
        cv2.namedWindow("Blur", cv2.WINDOW_NORMAL)
        cv2.imshow("Blur",self.blurImg)

        blurSave = Button(root, text="Save", command=test.blurSave, height=2, width=7, bg='green', fg='white', font=20)
        blurSave.pack()
        blurSave.place(x=790, y=220)

    def blurSave(self):
        cv2.imwrite("blurImg.jpg", self.blurImg)

    def dilate(self):
        image = cv2.imread(self.file, 1)
        kernel = np.ones((5, 5), 'uint8')
        self.dilateImg = cv2.dilate(image, kernel, iterations=1)
        cv2.namedWindow("Dilate", cv2.WINDOW_NORMAL)
        cv2.imshow("Dilate",self.dilateImg)

        dilateSave = Button(root, text="Save", command=test.dilateSave, height=2, width=7, bg='green', fg='white', font=20)
        dilateSave.pack()
        dilateSave.place(x=790, y=280)

    def dilateSave(self):
        cv2.imwrite("dilateImg.jpg", self.dilateImg)

    def erode(self):
        image = cv2.imread(self.file, 1)
        kernel = np.ones((5, 5), 'uint8')
        self.erodeImg = cv2.erode(image,kernel,iterations=1)
        cv2.namedWindow("Erode", cv2.WINDOW_NORMAL)
        cv2.imshow("Erode",self.erodeImg)

        erodeSave = Button(root, text="Save", command=test.erodeSave, height=2, width=7, bg='green', fg='white', font=20)
        erodeSave.pack()
        erodeSave.place(x=790, y=340)

    def erodeSave(self):
        cv2.imwrite("erodeImg.jpg", self.erodeImg)

def chooseFile():
    global test
    test.file = filer.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
    showImg()

def showImg():
    img = cv2.imread(test.file)
    cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image",img)


def main():
    global test, root
    root = Tk()
    root.geometry('900x700+300+50')
    root.resizable(0, 0)

    test = Colorize()

    choose_file = Button(root, text="Choose File", command=chooseFile, height=3, width=17, bg='green', fg='white', font=20)
    choose_file.pack()
    choose_file.place(x=700, y=80)

    gray = Button(root, text="Gray", command=test.gray, height=2, width=8, bg='gray', fg='white', font=20)
    gray.pack()
    gray.place(x=700, y=160)

    blur = Button(root, text="Blur", command=test.blur, height=2, width=8, bg='lightgray', fg='white', font=20)
    blur.pack()
    blur.place(x=700, y=220)

    dilate = Button(root, text="Dilate", command=test.dilate, height=2, width=8, bg='red', fg='white', font=20)
    dilate.pack()
    dilate.place(x=700, y=280)

    erode = Button(root, text="Erode", command=test.erode, height=2, width=8, bg='red', fg='white', font=20)
    erode.pack()
    erode.place(x=700, y=340)

    

    



    root.mainloop()

if __name__ == '__main__':
    main()
