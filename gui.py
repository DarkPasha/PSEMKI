import py_compile
import tkinter as tk
import random

from numpy import var




root = tk.Tk()

x = ""



def guiprinten():
    entry_label = tk.Label(root, text = x)
    entry_label.grid(row = 0, column = 0)



guiprinten


root.mainloop() 



