
CXX = g++ 
CXXFLAGS = -Wall -O2 -g

INCLUDES = -I./src
SRC = ./src

%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

all: benchmarks.exe

benchmarks.exe: sieve.o pi.o main.o
	 g++ $(CXXFLAGS)2 -o benchmarks.exe main.o sieve.o pi.o

clean: 
	 benchmarks.exe sieve.o pi.o main.o
