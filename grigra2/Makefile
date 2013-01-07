CPPOPTS = -O3 -g

all : layout layout2 genset ongrid setinfo
	echo success!

layout : src/layout.cpp
	g++  ${CPPOPTS} $^ -o $@

layout2 : src/layout2.cpp
	g++ $^ -o $@ -lpthread

genset : src/genset.cpp
	g++ ${CPPOPTS} $^ -o $@

ongrid : src/ongrid.cpp
	g++ ${CPPOPTS} $^ -o $@

setinfo : src/setinfo.cpp
	g++ ${CPPOPTS} $^ -o $@

clean:
	\rm *.exe *.o
	
