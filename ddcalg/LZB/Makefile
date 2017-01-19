#Makefile
# Created on: 2017/01/19
#     Author: sin

CC = gcc 
CCFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

all: enlzb delzb

enlzb:  e-lzb.o tree.o bin_code.o bitio.o
	$(CC) $(CCFLAGS) $(INCLUDES) -o enlzb e-lzb.o tree.o bin_code.o bitio.o
	
delzb: d-lzb.o bin_code.o bitio.o
	$(CC) $(CCFLAGS) $(INCLUDES) -o delzb d-lzb.o bin_code.o bitio.o


clean: 
	\rm *.o #*#
	\rm delzb enlzb
	

