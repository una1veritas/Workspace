
CC	= cc
C++	= c++
CFLAGS = -Wall -g
C++FLAGS = -Wall -g

TMSRCS = ./src/tm.cpp
TMOBJS = tm.o

.SUFFIXES: .o .cc .c++ .c

.c.o:
	$(CC) -c $(CFLAGS) $<

.cpp.o:
	$(C++) -c $(C++FLAGS) $<

all: tm

tm: $(TMOBJS)
	$(C++) $(TMOBJS) -o tm

tm.o:

clean   :
	rm -f core *.o *~
