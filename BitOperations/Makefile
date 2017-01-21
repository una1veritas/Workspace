#Makefile
# Created on: 2017/01/19
#     Author: sin

CC = gcc 
CCFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

#
LIBOBJS = nlz_routines.o nlz_test.o

all: nlz 

nlz: nlz_routines.o nlz_test.o
	$(CC) $(CCFLAGS) $(INCLUDES) -o nlz nlz_test.o nlz_test.o
	
popc:  numofbits_routines.o numofbits_test.o
	$(CC) $(CCFLAGS) $(INCLUDES) -o popc numofbits_routines.o numofbits_test.o


clean: 
	\rm *.o #*#
	\rm nlz popc
	

