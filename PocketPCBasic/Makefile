# Makefile 
CC	= gcc -std=c99
CFLAGS = -Wall -g
CXX = g++ -std=c++11
CXXFLAGS = -Wall -g
#
INCDIRS = -I./

.PHONY: clean

%.o:%.c
	$(CC) $(CCFLAGS) $(INCDIRS) -c $<

%.o:%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $<

#
INCLUDES = 

OBJS = reader.o

all: reader.exe

$(OBJS): $(INCLUDES)

reader.exe: $(OBJS)
	$(CXX) $(CCFLAGS) $(INCDIRS) -o reader.exe $(OBJS)


clean: 
	rm -rf *.o a.out *.exe *.gch *.dSYM *~
