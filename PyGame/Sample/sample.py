# -*- coding:utf-8 -*-
import sys
import pygame
from pygame.locals import *
import random
import math

class tsp2D:
    def __init__(self, nsize, cities=None, area=(500,500)):
        if not cities :
            w = int(area[0]*0.9)
            h = int(area[1]*0.9)
            wm = int(area[0]*0.05)
            hm = int(area[1]*0.05)
            tcities = set()
        else:
            tcities = set(cities)
        while len(tcities) < nsize :
            tcities.add( (int(random.randrange(w)/4)*4 + wm, int(random.randrange(h)/4)*4 + hm) )
        self.cities = list(tcities)
        self.setTour(self.randomTour())
    
    def size(self):
        return len(self.cities)
    
    def randomTour(self):
        return random.sample(range(self.size()), self.size())
    
    def setTour(self, a_tour):
        self.tour = a_tour
    
    def tourPoints(self):
        return [self.cities[self.tour[i]] for i in range(self.size())]
    
    def opt2tour(self, findex, tindex):
        if findex > tindex :
            tx = findex
            findex = tindex
            tindex = tx
        rev = self.tour[findex:tindex]
        rev.reverse()
        a_tour = self.tour[:findex] + rev + self.tour[tindex:]
        return a_tour
    
    def tourDistance(self, a_tour = None):
        if not a_tour:
            t = self.tour
        else:
            t = a_tour
        dsum = 0
        for i in range(self.size()):
            p0 = self.cities[t[i]]
            p1 = self.cities[t[(i+1) % self.size()]]
            dsum += math.sqrt( (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 )
        return dsum

    def __str__(self):
        return 'tsp2D('+str(self.cities)+')'

pygame.init()
fonts_list = pygame.font.get_fonts()
screen = pygame.display.set_mode([768,512])
pygame.display.set_caption("PyGame sample")
#sprite = pygame.image.load("./Sample/red.png")
if 'optima' in fonts_list :
    font_typ1 = pygame.font.SysFont("optima", 18)
    font_typ2 = pygame.font.SysFont("optima", 10)
else:
    font_typ1 = pygame.font.SysFont("helvetica", 18)
    font_typ2 = pygame.font.SysFont("helvetica", 10)

def main(tsp=None):
    iter = [0, 0]
    base = [0, 0]
    frozen = False
    improved = False
    wstart = pygame.time.get_ticks()
    wstopped = 0
    while True:
        screen.fill([190,190,190])
        #
        for i in range(tsp.size()):
            pt = tsp.cities[i]
            pygame.draw.circle(screen, (0,0,0), pt, 4)
            ptlabel = font_typ2.render(str(i), True, (91,63,127) )
            screen.blit(ptlabel, (pt[0], pt[1] + 4))

        swatch = pygame.time.get_ticks()
        if not frozen :
            while iter[0] < tsp.size() :
                if  pygame.time.get_ticks() - swatch > 50 :
                    break
                ix = (iter[0]+base[0]) % tsp.size()
                iy = (iter[1]+base[1]) % tsp.size()
                #print(iter, [ix, iy])
                a_tour = tsp.opt2tour(ix, iy)
                if tsp.tourDistance(a_tour) < tsp.tourDistance() :
                    tsp.setTour(a_tour)
                    improved = True
                    break
                #
                iter[1] += 1
                if not iter[1] < tsp.size() :
                    iter[0] += 1
                    iter[1] = 0
            else:
                frozen = True
            #print("----", frozen)
            #print(iter, [ix, iy])
            if improved :
                base = [ix, iy]
                iter = [0, 0]
                improved = False
                #print('====')
                #print(base, iter)
                #print('====')
                
        if frozen :
            if wstopped == 0 :
                 wstopped = pygame.time.get_ticks() - wstart
            textcolor = (63,63,255)
            text1 = font_typ1.render('{0:>8.2f} stopped after {1} ms.'.format(tsp.tourDistance(), wstopped), True, textcolor)
        else:
            textcolor = (255,255,255)
            text1 = font_typ1.render('{:>8.2f}'.format(tsp.tourDistance()), True, textcolor)
        screen.blit(text1, (32,16))
        pygame.draw.lines(screen, (0,0,0), True, tsp.tourPoints())
        
        #screen.blit(sprite,(20,50))
        pygame.display.update()
        pygame.time.wait(max(0,50 - pygame.time.get_ticks() - swatch))
        #
        for evt in pygame.event.get():
            if evt.type == QUIT :
                pygame.quit()
                sys.exit()


if __name__ == "__main__" :
    w, h = pygame.display.get_surface().get_size()
    t = tsp2D(160, area=(w,h))
    main(t)


