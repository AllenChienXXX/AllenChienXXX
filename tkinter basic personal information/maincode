import tkinter as tk
from tkinter import font
import numpy as np
import pandas as pd
from tkinter import messagebox
import csv
window = tk.Tk()
window.title('Basic personal information')
window.geometry('1000x500')
windowlabel = tk.Label(window,text='Basic personal information',font=('Algerian',40)).pack()
name = tk.Entry(window)
name.pack()
height = tk.Entry(window)
height.pack()
weight = tk.Entry(window)
weight.pack()
age = tk.Entry(window)
age.pack()
Edu = tk.Entry(window)
Edu.pack()
instructions = tk.Label(window,text='You must enter all datas in order to append new data',font=('Arial',15)).place(x=300,y=200)
addname = tk.Label(window,text='name',font=('Arial',10)).place(x=580,y=60)
addheight = tk.Label(window,text='height',font=('Arial',10)).place(x=580,y=80)
addweight = tk.Label(window,text='weight',font=('Arial',10)).place(x=580,y=100)
addage = tk.Label(window,text='age',font=('Arial',10)).place(x=580,y=120)
addEdu = tk.Label(window,text='Education',font=('Arial',10)).place(x=580,y=140)
#data.csv(get data and return)
def addcolumndata():
    data = pd.read_csv('data.csv',header=None)
    indexnumber = data.shape
    Name = name.get()
    Height=height.get()
    Weight=weight.get()
    Age=age.get()
    Education=Edu.get()
    addtext = pd.DataFrame({0:Name,1:Height,2:Weight,3:Age,4:Education},index=[indexnumber[0]])
    data = data.append(addtext,ignore_index=True)    
    data.to_csv('data.csv',index=False,header=False)
def return_data():
    newwindow = tk.Toplevel(window)
    with open("data.csv", newline = "") as file:
        reader = csv.reader(file)
        r = 0
        for col in reader:
            c = 0
            for row in col:
                label = tk.Label(newwindow, width = 10, height = 2, \
                                text = row, relief = tk.RIDGE)
                label.grid(row = r, column = c)
                c += 1
            r += 1




buttonadddata = tk.Button(window,text='add data',width=20,height=1,command=addcolumndata).place(x=430,y=170)

buttondata = tk.Button(window,text='current data',width = 20,height=2,command=return_data).place(x=500,y=300)
window.mainloop()
