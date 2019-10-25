#Makefile
# Created on: Sep 6, 2019
#     Author: sin

CXX = g++ -Wall

CFLAGS += -g
CXXFLAGS = 

OBJS = main.o
TARGET = ncurses-sample.exe

INCLUDES = -Incurses
LIBS = -lncurses

all: $(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) -o $@ $^ $(LIBS)

#%.o:	$(PROJECT_ROOT)%.cpp
%.o:	%.cpp 
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(INCLUDES) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
%.o:	%.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -fr $(TARGET) $(OBJS)
