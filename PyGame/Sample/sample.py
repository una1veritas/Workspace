# -*- coding:utf-8 -*-
import sys
import pygame
from pygame.locals import *
import random

class tsp2D:
    def __init__(self, nsize, cities=None, area=(500,500)):
        if not cities :
            w = int(area[0]*0.9)
            h = int(area[1]*0.9)
            wm = int(area[0]*0.05)
            hm = int(area[1]*0.05)
            self.cities = [ (random.randrange(w) + wm, random.randrange(h) + hm) 
                           for i in range(nsize)]
        else:
            self.cities = cities
        self.tour = [i for i in range(0,nsize)]
    
    def visit(self):
        return [self.cities[self.tour[i]] for i in range(0, len(self.cities))]

    def __str__(self):
        return 'tsp2D('+str(self.cities)+')'

pygame.init()
fonts_list = pygame.font.get_fonts()
screen = pygame.display.set_mode([768,512])
pygame.display.set_caption("PyGame sample")
sprite = pygame.image.load("./Sample/red.png")
if 'optima' in fonts_list :
    font_typ1 = pygame.font.SysFont("optima", 18)
else:
    font_typ1 = pygame.font.SysFont("helvetica", 18)

def main(tsp=None):
    while True:
        screen.fill([190,190,190])
        #
        text1 = font_typ1.render(str(pygame.time.get_ticks()), True, (255,255,255))
        screen.blit(text1, (80,32))

        for evt in pygame.event.get():
            if evt.type == QUIT :
                pygame.quit()
                sys.exit()

        for p in t.cities:
            pygame.draw.circle(screen, (0,0,0), p, 3)
        pygame.draw.lines(screen, (255,255,255), True, t.visit())

        screen.blit(sprite,(20,50))
        pygame.display.update()
        pygame.time.wait(40)

if __name__ == "__main__" :
    w, h = pygame.display.get_surface().get_size()
    t = tsp2D(32, area=(w,h))
    main(t)


