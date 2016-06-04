CC	= gcc
CXX	= g++
CFLAGS = -Wall -O0 -g3
CXXFLAGS = -Wall -O0 -g3
INCLUDES = -I./ -I../ 

HEADERS = ../Card.h ../CardSet.h ../Dealer.h ../Player.h ../GameStatus.h 

GROUPSOBJS = GroupB.o GroupC.o GroupH.o 


%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

libgroups.a:  $(GROUPSOBJS)
	ar r libgroups2014.a $(GROUPSOBJS)

clean:
	\rm -f $(GROUPSOBJS) libgroups.a
