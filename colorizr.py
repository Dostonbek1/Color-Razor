import numpy as np
from tkinter import *
import cv2
import tkinter.filedialog as filer

class Colorize:
    def __init__(self):
        self.file = ''

    def colorize(self):
        color = cv2.imread(self.file, 1)

        # gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        # cv2.imwrite("gray.jpg", gray)

        b = color[:, :, 0]
        g = color[:, :, 1]
        r = color[:, :, 2]

        rgba = cv2.merge((b, g, r, g))
        cv2.imwrite("rgba.png", rgba)


def chooseFile():
    global test
    test.file = filer.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    print(test.file)
    test.colorize()

def main():
    global test
    root = Tk()
    root.geometry('800x500+300+50')
    root.resizable(0, 0)

    test = Colorize()

    choose_file = Button(root, text="Choose File", command=chooseFile)
    choose_file.pack()



    root.mainloop()

if __name__ == '__main__':
    main()
