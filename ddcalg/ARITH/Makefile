#Makefile
# Created on: 2017/01/19
#     Author: sin

CC = gcc 
CCFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

#
JONESLIBOBJS = bitio.o c-ari.o

all: enjones dejones

enjones:  e-ari.o $(JONESLIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o enjones e-ari.o $(JONESLIBOBJS)
	
dejones: d-ari.o $(JONESLIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o dejones d-ari.o $(JONESLIBOBJS)


clean: 
	\rm *.o #*#
	\rm dejones enjones
	

