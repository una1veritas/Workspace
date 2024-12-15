'''
Created on 2024/12/15

@author: sin
'''
import math
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import numpy
from idlelib import sidebar

class Shape:
    def patches(self):
        return self.patches()

class Dot(Shape):
    def __init__(self, center = None, size = 2):
        if center == None or len(center) != 2:
            raise ValueError('center is None. Dot must have the place coordinates.')
        self.diameter = size
        self.center = list(center)
    
    def patches(self):
        return patches.Circle(xy=self.center, radius=min(1,self.diameter/2), facecolor='k', edgecolor='k')
        
class Circle(Shape):
    def __init__(self, radius=None, center = (0,0), boundingbox = None ):
        if boundingbox == None :
            self.radius = radius
            self.center = center
        else:
            if len(boundingbox) == 2 :
                x0,y0 = boundingbox[0][0], boundingbox[0][1]
                x1,y1 = boundingbox[1][0], boundingbox[1][1]
            elif len(boundingbox) == 4 :
                x0,y0, x1, y1 = boundingbox
            else:
                raise ValueError('boundingbox of the circle must specify square by coordinates pair.')
            self.radius = abs(x1 - x0)
            if self.radius != abs(y1 - y0) :
                raise ValueError('boundingbox of the circle must be specified by a square.')
            self.center = ((x1+x0)/2, (y1+y0)/2)
    def radius(self):
        return self.radius
    
    def diameter(self):
        return 2*self.radius
    
    def patches(self):
        return patches.Circle(xy=self.center, radius=self.radius, fill=False, edgecolor='k')

class Triangle(Shape):
    def __init__(self, sides = None, center = (0,0) ):
        if isinstance(sides, (list, tuple,set) ) :
            if len(sides) != 3 :
                raise ValueError('specify the triangle by lengths of 3 sides.')
            self.sides = list(sides)
            self.center = center
        else:
            raise ValueError('specify the triangle by lengths of 3 sides.')
    
    def coordinates(self):
        '''list of coordinates of the corner, from left of the bottom (sides[0]),
         in counter-clockwise'''
        pts = list()
        bottom = self.sides[0]
        x0, y0 = self.center
        cosq = ((self.sides[1]**2) + (self.sides[0]**2) - (self.sides[2]**2))/(2*self.sides[1]*self.sides[0])
        pts.append( (x0 - bottom/2, y0) )
        pts.append( (x0 + bottom/2, y0) )
        '''top corner'''
        x2 = pts[0][0] + cosq * self.sides[1]
        y2 = pts[0][1] + math.sqrt(self.sides[1]**2 - (cosq * self.sides[1])**2)
        pts.append( (x2,y2) )
        return pts
        
    def incenter(self):
        pts = self.coordinates()
        c = self.sides[0]+self.sides[1]+self.sides[2]
        return ( (pts[2][0]*self.sides[0]+pts[0][0]*self.sides[1]+pts[1][0]*self.sides[2])/c, 
                 (pts[2][1]*self.sides[0]+pts[0][1]*self.sides[1]+pts[1][1]*self.sides[2])/c )
        
    def inradius(self):
        pts = self.coordinates()
        c = self.sides[0]+self.sides[1]+self.sides[2]
        ic = ( (pts[2][0]*self.sides[0]+pts[0][0]*self.sides[1]+pts[1][0]*self.sides[2])/c, 
                 (pts[2][1]*self.sides[0]+pts[0][1]*self.sides[1]+pts[1][1]*self.sides[2])/c )
        return abs(pts[0][1] - ic[1])
        
    def patches(self):
        pts = self.coordinates()
        return patches.Polygon(pts, closed=True, edgecolor='k', fill=False, facecolor='w', linewidth=1)
    
if __name__ == '__main__':
    print("Hi.")
    fig = pyplot.figure()
    axes = pyplot.axes()
    
    tri = Triangle([2,4,4])
    ic = tri.incenter()
    point = Dot(ic,0.02)
    r = tri.inradius()
    circ = Circle(radius=r, center=ic)
    axes.add_patch(circ.patches())
    axes.add_patch(point.patches())
    axes.add_patch(tri.patches())
    print(r)
    
    pyplot.axis('scaled')
    axes.set_aspect('equal')
    pyplot.show()
    