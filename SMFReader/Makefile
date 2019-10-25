#PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CXX = g++ -Wall -Ilibsmf

OBJS = main.o

CFLAGS += -g

all:	SMFReader.exe

SMFReader.exe:	$(OBJS)
	$(CXX) -o $@ $^

#%.o:	$(PROJECT_ROOT)%.cpp
%.o:	%.cpp libsmf/SMFEvent.h libsmf/SMFStream.h
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
%.o:	%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -fr SMFReader.exe $(OBJS)
