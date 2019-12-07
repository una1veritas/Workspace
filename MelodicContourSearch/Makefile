#PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CXX = g++ -Wall
CC  = gcc -Wall 

ifeq ($(BUILD_MODE),run)
	CFLAGS += -O2
	CXXFLAGS += -std=c++11
else
	CFLAGS += -g
	CXXFLAGS += -std=c++11
endif


OBJS = main.o
INCLUDES = SMFEvent.h SMFStream.h

all:	mcsearch.exe

mcsearch.exe:	$(OBJS)
	$(CXX) -o $@ $^

#%.o:	$(PROJECT_ROOT)%.cpp
%.o:	%.cpp $(INCLUDES)
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
%.o:	%.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -fr mcsearch.exe $(OBJS)
