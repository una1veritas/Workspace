CPPOPTS = -O3 -g

all : layout.exe genset.exe ongrid.exe setinfo.exe
	echo success!

layout.exe : src/layout.cpp
	g++ ${CPPOPTS} $^ -o $@

layout2.exe : src/layout2.cpp
	g++ ${CPPOPTS} $^ -o $@

genset.exe : src/genset.cpp
	g++ ${CPPOPTS} $^ -o $@

ongrid.exe : src/ongrid.cpp
	g++ ${CPPOPTS} $^ -o $@

setinfo.exe : src/setinfo.cpp
	g++ ${CPPOPTS} $^ -o $@

clean:
	\rm *.exe *.o
	
