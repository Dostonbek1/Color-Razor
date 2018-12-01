from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np

image_name = 'face.jpg'

image = cv2.imread(image_name)

#Rearrang the color channel
b,g,r = cv2.split(image)
img = cv2.merge((r,g,b))

# A root window for displaying objects
root = Tk()  

# Convert the Image object into a TkPhoto object
im = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=im) 

# Put it in the display window
Label(root, image=imgtk).pack() 

root.mainloop() # Start the GUI