#PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CXX = g++ -Wall
CC  = gcc -Wall 

ifeq ($(BUILD_MODE),run)
	CFLAGS += -O2
	CXXFLAGS += -std=c++1z
else
	CFLAGS += -g #-DKMP_DEBUG
	CXXFLAGS += -std=c++1z
endif


OBJS = main.o 
INCLUDES = libsmf/SMFEvent.h libsmf/SMFStream.h stringmatching.h

all: dirlister_test.exe
#all:	melodysearch.exe

dirlister_test.exe:	dirlister_test.o dirlister.o
	$(CXX) -o $@ dirlister_test.o dirlister.o

dirlister_test.o:	dirlister_test.cpp dirlister.h dirlister_unix.h dirlister_win.h
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

pmmtest.exe:	$(OBJS) 
	$(CXX) -o $@ $^

melodysearch.exe:	$(OBJS) 
	$(CXX) -o $@ $^

mana.exe: mana.cpp libsmf/SMFStream.h libsmf/SMFEvent.h DirectoryLister.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) -o mana.exe mana.cpp

#%.o:	$(PROJECT_ROOT)%.cpp
%.o:	%.cpp $(INCLUDES)
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
%.o:	%.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -fr *.exe $(OBJS)
