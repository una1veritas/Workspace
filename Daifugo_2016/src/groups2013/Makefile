CC	= gcc
CXX	= g++
CFLAGS = -Wall -O0 -g3
CXXFLAGS = -Wall -O0 -g3
INCLUDES = -I./ -I../ 

HEADERS = ../Card.h ../CardSet.h ../Dealer.h ../Player.h ../GameStatus.h 

GROUPSOBJS = GroupA.o GroupB.o GroupC.o GroupD.o GroupE.o GroupF.o GroupG.o GroupH.o 

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

libgroups.a:  $(GROUPSOBJS)
	ar r libgroups.a $(GROUPSOBJS)

clean:
	\rm -f $(GROUPSOBJS) libgroups.a
