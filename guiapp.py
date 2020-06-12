from tkinter import *
from PIL import Image
from PIL import ImageTk
import math
import random
import xlrd
import re
import openpyxl
import pandas as pd
import numpy as np

metadatapd = pd.read_excel('FINAL.xlsx')
metadata = np.array(metadatapd)

global img

def nextportrait():
	i = round(random.uniform(0, 1)*5000)
	im = "Images/" + str(metadata[i, 1])

	global img3
	img3 = ImageTk.PhotoImage(Image.open(im))
	canvas.itemconfig(img2, image = img3)

def reconstruct():
	global outimg3

	inputimg = img3

	z = []
	z.append(var1.get())
	z.append(var2.get())
	z.append(var3.get())
	z.append(var4.get())
	z.append(var5.get())
	z.append(var6.get())
	z.append(var7.get())
	z.append(var8.get())
	z.append(var9.get())
	z.append(var10.get())

	# use inputimg as input img

	# DO STUFF WITH MODELS
	
	# set reconstruction image to variable outimg3

	canvasout.itemconfig(outimg2, image = outimg3)


def randomz():
	scale1.set(round(random.uniform(0, 1)*10.0)/10)
	scale2.set(round(random.uniform(0, 1)*10.0)/10)
	scale3.set(round(random.uniform(0, 1)*10.0)/10)
	scale4.set(round(random.uniform(0, 1)*10.0)/10)
	scale5.set(round(random.uniform(0, 1)*10.0)/10)
	scale6.set(round(random.uniform(0, 1)*10.0)/10)
	scale7.set(round(random.uniform(0, 1)*10.0)/10)
	scale8.set(round(random.uniform(0, 1)*10.0)/10)
	scale9.set(round(random.uniform(0, 1)*10.0)/10)
	scale10.set(round(random.uniform(0, 1)*10.0)/10)

root = Tk()

var1 = DoubleVar()
scale1 = Scale(root, variable = var1, to=1, resolution=0.1, orient="horizontal")
var2 = DoubleVar()
scale2 = Scale(root, variable = var2, to=1, resolution=0.1, orient="horizontal")
var3 = DoubleVar()
scale3 = Scale(root, variable = var3, to=1, resolution=0.1, orient="horizontal")
var4 = DoubleVar()
scale4 = Scale(root, variable = var4, to=1, resolution=0.1, orient="horizontal")
var5 = DoubleVar()
scale5 = Scale(root, variable = var5, to=1, resolution=0.1, orient="horizontal")
var6 = DoubleVar()
scale6 = Scale(root, variable = var6, to=1, resolution=0.1, orient="horizontal")
var7 = DoubleVar()
scale7 = Scale(root, variable = var7, to=1, resolution=0.1, orient="horizontal")
var8 = DoubleVar()
scale8 = Scale(root, variable = var8, to=1, resolution=0.1, orient="horizontal")
var9 = DoubleVar()
scale9 = Scale(root, variable = var9, to=1, resolution=0.1, orient="horizontal")
var10 = DoubleVar()
scale10 = Scale(root, variable = var10, to=1, resolution=0.1, orient="horizontal")


# INPUT IMAGE
canvas = Canvas(root, width = 350, height = 400)  
canvas.pack(side='left')
global img2
img = ImageTk.PhotoImage(Image.open("Images/0000000.jpeg"))
img2 = canvas.create_image((175, 200), image=img)

# OUTPUT IMAGE
canvasout = Canvas(root, width = 350, height = 400)  
canvasout.pack(side='right')
global outimg2
outimg = ImageTk.PhotoImage(Image.open("Images/0000001.jpeg"))
outimg2 = canvasout.create_image((175, 200), image=outimg)

# Z SLIDERS
labelz = Label(root)
labelz.config(text = "Z")
labelz.pack()

scale1.pack()
scale2.pack()
scale3.pack()
scale4.pack()
scale5.pack()
scale6.pack()
scale7.pack()
scale8.pack()
scale9.pack()
scale10.pack()

button = Button(root, text="Random Z", command=randomz)
button.pack()

button1 = Button(root, text="Next Portrait", command=nextportrait)
button1.pack()

button2 = Button(root, text="Reconstruct", command=reconstruct)
button2.pack()



root.mainloop()