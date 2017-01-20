#Makefile
# Created on: 2017/01/19
#     Author: sin

CC = gcc 
CCFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

#
LIBOBJS = bitio.o c-huff.o

all: enhuff dehuff

enhuff:  e-huff.o $(LIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o enhuff e-huff.o $(LIBOBJS)
	
dehuff: d-huff.o $(LIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o dehuff d-huff.o $(LIBOBJS)


clean: 
	\rm *.o #*#
	\rm dejones enjones
	

