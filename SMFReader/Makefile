#PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OBJS = demo_main.o

CFLAGS += -g

all:	demo.exe

demo.exe:	$(OBJS)
	$(CXX) -o $@ $^

#%.o:	$(PROJECT_ROOT)%.cpp
%.o:	%.cpp SMFReader.h SMFStream.h
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
%.o:	%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -fr demo.exe $(OBJS)
