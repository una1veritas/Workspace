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


#OBJS = main.o
OBJS = dirlister.o
INCLUDES = dlister.h libsmf/SMFEvent.h libsmf/SMFStream.h kmp.h DirectoryLister.h stringmatching.h manamatching.h

all: dirlister.exe
#all:	mcsearch.exe

dirlister.exe:	$(OBJS) 
	$(CXX) -o $@ $^

pmmtest.exe:	$(OBJS) 
	$(CXX) -o $@ $^

mcsearch.exe:	$(OBJS) 
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
	rm -fr mcsearch.exe $(OBJS)
