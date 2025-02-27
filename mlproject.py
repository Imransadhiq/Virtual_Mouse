import tkinter as tk
from tkinter import *
from tkinter import ttk
import subprocess

win = tk.Tk()
win.title("Virtual Mouse")
win.geometry("300x450")  # Increased height to fit all buttons

bg = PhotoImage(file='images.png')
label17 = Label(win, image=bg)
label17.pack(pady=10)

def hand():
    subprocess.run(["python", "virtualmouse.py"])

def eye():
    subprocess.run(["python", "eye.py"])

def gesture_recognition():
    subprocess.run(["python", "test.py"])

def sign_language():
    subprocess.run(["python", "main.py"])

button = ttk.Button(win, text="Virtual Mouse Using Hand", command=hand)
button2 = ttk.Button(win, text="Virtual Mouse Using Eye", command=eye)
button3 = ttk.Button(win, text="Gesture Recognition", command=gesture_recognition)
button4 = ttk.Button(win, text="Human Recognition", command=sign_language)

button.pack(pady=10)
button2.pack(pady=10)
button3.pack(pady=10)
button4.pack(pady=10)

win.mainloop()
