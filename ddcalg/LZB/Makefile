#Makefile
# Created on: 2017/01/19
#     Author: sin

CC = gcc 
CCFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

#
LZBLIBOBJS = tree.o bin_code.o bitio.o swatch.o

all: enlzb delzb

enlzb:  e-lzb.o $(LZBLIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o enlzb e-lzb.o $(LZBLIBOBJS)

delzb: d-lzb.o $(LZBLIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o delzb d-lzb.o $(LZBLIBOBJS)


clean: 
	\rm *.o delzb enlzb
	\rm *.dSYM
