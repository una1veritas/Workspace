'''
Created on 2020/10/18

@author: sin
'''
#from tkinter import *
from tkinter import ttk, Tk
import random

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        
        # Frame as widget container
        self.frame1 = ttk.Frame(master)
        self.frame1.grid()
        
        random.random()
        a = [random.randrange(24) for i in range(24)]
        self.setup_labels(a)
        
    def setup_labels(self, myarray):
        # Image
        #pencil_image = PhotoImage(file='pencil.png')
        
        for i in range(len(myarray)):
            # Label
            label1 = ttk.Label(self.frame1, text = str(myarray[i]), padding = (5,5) )
            label1.grid(row=0, column=i)
        

    def print_contents(self, event):
        print("Hi. The current entry content is:",
              self.contents.get())

root = Tk()
root.title('Label Example')
myapp = App(root)
myapp.mainloop()


