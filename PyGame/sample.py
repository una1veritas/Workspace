# -*- coding:utf-8 -*-
import sys
import pygame
from pygame.locals import *

pygame.init()
screen = pygame.display.set_mode([400,300])
pygame.display.set_caption("PyGame sample")
sprite = pygame.image.load("./red.png")

def main():
    while True:
        screen.fill([0,0,0])

        for evt in pygame.event.get():
            if evt.type == QUIT :
                pygame.quit()
                sys.exit()

        screen.blit(sprite,(20,50))
        pygame.display.update()

if __name__ == "__main__" :
    main()


