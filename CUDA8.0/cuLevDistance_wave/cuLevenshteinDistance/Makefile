
.SUFFIXES: .cpp .h .cu .o

DEBUG=0
CXXFLAGS=-Wall -g
NVCCFLAGS=-gencode arch=compute_30,code=sm_30 -Xptxas -v

INCLUDES=-I.
LIBRARY=-lcuda -lcudart

culevdist.exe: main.cu
	nvcc $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" $(INCLUDES) $< -o $@

.cu.o: $<
	nvcc $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" $(INCLUDES) $< -o $@

.cpp.o: $<
	g++ -c $(CXXFLAGS) $< -o $@
