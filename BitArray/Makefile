CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		BitArray.o

LIBS =

TARGET =	BitArray

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
