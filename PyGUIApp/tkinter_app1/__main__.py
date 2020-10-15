'''
Created on 2020/10/11

@author: sin
'''
from tkinter import *

class App(Frame): 

    def __init__(self, master=None): 
        self.master = master 
        self.x = 1
        self.y = 0
  
        self.canvas = Canvas(master) 
        self.rectangle = self.canvas.create_rectangle(5, 5, 25, 25, fill="black") 
        self.canvas.pack() 
        
        self.iterate() 
        
    def iterate(self): 
        self.canvas.move(self.rectangle, self.x, self.y)   
        self.canvas.after(100, self.iterate) 
      
    # for motion in negative x direction 
    def left(self, event): 
        print(event.keysym) 
        self.x = -5
        self.y = 0
      
    # for motion in positive x direction 
    def right(self, event): 
        print(event.keysym) 
        self.x = 5
        self.y = 0
      
    # for motion in positive y direction 
    def up(self, event): 
        print(event.keysym) 
        self.x = 0
        self.y = -5
      
    # for motion in negative y direction 
    def down(self, event): 
        print(event.keysym) 
        self.x = 0
        self.y = 5

  
if __name__ == "__main__": 
  
    # object of class Tk, resposible for creating 
    # a tkinter toplevel window 
    root = Tk() 
    app = App(master=root)
  
    # This will bind arrow keys to the tkinter 
    # toplevel which will navigate the image or drawing 
    app.master.bind("<KeyPress-Left>", app.left) 
    app.master.bind("<KeyPress-Right>", app.right) 
    app.master.bind("<KeyPress-Up>", app.up) 
    app.master.bind("<KeyPress-Down>", app.down) 
      
    # Infnite loop breaks only by interrupt 
    mainloop()
    
