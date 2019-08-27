#PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OBJS = SMFReader.o

CFLAGS += -g

all:	SMFReader

SMFReader:	$(OBJS)
	$(CXX) -o $@ $^

#%.o:	$(PROJECT_ROOT)%.cpp
%.o:	%.cpp SMFReader.h
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
%.o:	%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -fr SMFReader $(OBJS)
