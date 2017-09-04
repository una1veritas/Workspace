CC=acc

solver: solver.o  TravelingSalesman.h ../Point.h ../Generic.h
	${CC} -o solver solver.o 

solver.o: TravelingSalesman.h ../Point.h ../Generic.h

bestfactor: bestfactor.o 
	${CC} -o bestfactor bestfactor.o 

bestfactor.o: TravelingSalesman.h ../Point.h ../Generic.h

