#Makefile
# Created on: 2017/01/19
#     Author: sin

CXX = g++ -std=c++11 
CXXFLAGS = -Wall -O0 -g3
INCLUDES = -I./

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

#
LIBOBJS = 

all: ac 

acdawg:  acdawg.o $(LIBOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o acdawg acdawg.o $(LIBOBJS)
	
ac: ac.o $(LIBOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ac ac.o $(LIBOBJS)


clean: 
	\rm *.o acdawg ac #*#
