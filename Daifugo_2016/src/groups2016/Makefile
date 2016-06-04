CC	= gcc
CXX	= g++
CFLAGS = -Wall -O0 -g3
CXXFLAGS = -Wall -O0 -g3
INCLUDES = -I./ -I../ 

HEADERS = ../Card.h ../CardSet.h ../Dealer.h ../Player.h ../GameStatus.h 

GROUPSOBJS = Group1.o Group2.o Group3.o Group4.o Group5.o Group6.o Group7.o

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

libgroups.a:  $(GROUPSOBJS)
	ar r libgroups.a $(GROUPSOBJS)

clean:
	\rm -f $(GROUPSOBJS) libgroups.a
