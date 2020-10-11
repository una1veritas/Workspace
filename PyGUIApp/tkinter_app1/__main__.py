'''
Created on 2020/10/11

@author: sin
'''
from tkinter import *
from tkinter.ttk import * 

class GFG: 

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
    master = Tk() 
    gfg = GFG(master) 
  
    # This will bind arrow keys to the tkinter 
    # toplevel which will navigate the image or drawing 
    master.bind("<KeyPress-Left>", lambda e: gfg.left(e)) 
    master.bind("<KeyPress-Right>", lambda e: gfg.right(e)) 
    master.bind("<KeyPress-Up>", lambda e: gfg.up(e)) 
    master.bind("<KeyPress-Down>", lambda e: gfg.down(e)) 
      
    # Infnite loop breaks only by interrupt 
    mainloop()
    
