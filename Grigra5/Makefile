CPPOPTS = -O3 -g
OBJS = main layout

.PHONY : all
all : layout.exe
	echo success!

layout.exe : ${OBJS:=.o}
	g++ ${CPPOPTS} $^ -o $@

%.o : src/%.cpp
	g++ -c ${CPPOPTS} $^ -o $@

.PHONY : clean
clean:
	\rm -f *.exe *.o
	
