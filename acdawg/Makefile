#Makefile
# Created on: 2017/01/19
#     Author: sin

CXX = g++ -std=c++11 
CXXFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) $(INCLUDES) -c $<

#
LIBOBJS = 

all: acdawg ac

acdawg:  acdawg.o $(LIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o acdawg acdawg.o $(LIBOBJS)
	
ac: ac.o $(LIBOBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o ac ac.o $(LIBOBJS)


clean: 
	\rm *.o acdawg ac #*#
