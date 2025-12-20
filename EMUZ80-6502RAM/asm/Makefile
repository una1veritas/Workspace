# Adjust the run address to match the .org in the source code
all: ehbasic.bin basic2.bin wozmon.bin

ehbasic.mon: ehbasic.bin
	bintomon -v -l 0xc000 ehbasic.bin > ehbasic.mon

ehbasic.bin: ehbasic.o
	cl65 -C none.cfg -m ehbasic.map -o ehbasic.bin ./ehbasic.o
	ld65 -C ~/share/cc65/cfg/none.cfg -vm -m ehbasic.map -o ehbasic.bin ehbasic.o

ehbasic.o: basic.asm min_mon.asm
	ca65 -g -l min_mon.lst --feature labels_without_colons -o ehbasic.o min_mon.asm

rom_ehbasic.c: ehbasic.bin
	srec_cat ehbasic.bin -binary -o rom_ehbasic.c -C-Array rom_ehbasic

basic2.o: basic2.asm mos.asm
	ca65 -g -l basic2.lst --feature labels_without_colons -o basic2.o basic2.asm

basic2.bin: basic2.o
	cl65 -C none.cfg -m basic2.map -o basic2.bin basic2.o

rom_basic2: 
	srec_cat basic2.bin -binary -o rom_basic2.c -C-Array rom_basic2 

wozmon.bin: wozmon.o
	cl65 -C none.cfg -m wozmon.map -o wozmon.bin wozmon.o

wozmon.o: wozmon.asm
	ca65 -g -l wozmon.lst --feature labels_without_colons -o wozmon.o wozmon.asm

rom_wozmon.c: wozmon.bin
	srec_cat wozmon.bin -binary -o rom_wozmon.c -C-Array rom_wozmon

clean:
	$(RM) *.o *.lst *.map *.bin *.run *.mon *.c1p

distclean: clean
