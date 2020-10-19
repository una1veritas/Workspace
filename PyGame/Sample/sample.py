# -*- coding:utf-8 -*-
import sys
import pygame
from pygame.locals import *

pygame.init()
fonts_list = pygame.font.get_fonts()
screen = pygame.display.set_mode([768,512])
pygame.display.set_caption("PyGame sample")
sprite = pygame.image.load("./Sample/red.png")
if 'optima' in fonts_list :
    font_typ1 = pygame.font.SysFont("optima", 18)
else:
    font_typ1 = pygame.font.SysFont("helvetica", 18)

def main():
    while True:
        screen.fill([127,127,127])
        #
        text1 = font_typ1.render(str(pygame.time.get_ticks()), True, (255,255,255))
        screen.blit(text1, (80,32))

        for evt in pygame.event.get():
            if evt.type == QUIT :
                pygame.quit()
                sys.exit()

        pygame.draw.circle(screen, (255,0,0), (320,240), 24)
        pygame.draw.line(screen, (255,255,255), (0,0), (640,480))

        screen.blit(sprite,(20,50))
        pygame.display.update()
        pygame.time.wait(40)

if __name__ == "__main__" :
    main()


