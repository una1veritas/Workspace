CC = CC

CLASSHEADERS = Generic.h String.h Set.h Mapping.h

test : test.o
	$(CC) -o test test.o String.o Set.o Bag.o Mapping.o $(LIB)

String.o: String.h Generic.h
Set.o: Set.h Generic.h
Mapping.o: Set.h Generic.h Mapping.h

test.o: String.o Set.o Bag.o Mapping.o

otest: otest.h
	$(CC) -o otest otest.c $(LIB)

clean : 
	-/bin/rm *.o *~
