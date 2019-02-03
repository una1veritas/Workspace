CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

INCLUDES = Bit64Ring.h
OBJS =		Bit64Ring.o

LIBS =

TARGET =	Bit64Ring

%.o: %.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $<

$(TARGET):	$(OBJS) 
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
