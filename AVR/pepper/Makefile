###############################################################################
# Makefile for ginger & sugar
###############################################################################

TARGET = ALL
version := ?1.0.0.0,pepper,20090621*
export version

all : pepper45/pepper45.hex pepper85/pepper85.hex

pepper45/pepper45.hex: gio.c gio.h main.c 
	cd pepper45 && $(MAKE)

pepper85/pepper85.hex: gio.c gio.h main.c 
	cd pepper85 && $(MAKE)

copy: pepper45/pepper45.hex pepper85/pepper85.hex
	cp pepper45/pepper45.hex HEX/pepper45.hex
	cp pepper85/pepper85.hex HEX/pepper85.hex

clean:
	( cd pepper45 ; make clean)
	( cd pepper85 ; make clean)
