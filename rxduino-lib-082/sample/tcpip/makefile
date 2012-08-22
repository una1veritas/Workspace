TARGET = tkdnip
CCPATH = /usr/local/tkdn-20110720/rx-elf/bin/
AS  = $(CCPATH)rx-elf-as 
CC  = $(CCPATH)rx-elf-gcc -Wall -g -I../../include/tkdnhal -I../../include/lib/ $(CFLAGS) -O2
CPP = $(CCPATH)rx-elf-gcc -Wall -g -I../../include/tkdnhal -I../../include/lib/ $(CFLAGS) -O2
LNK = $(CCPATH)rx-elf-ld
CNV = $(CCPATH)rx-elf-objcopy --srec-forceS3 --srec-len 32 -O srec
DMP = $(CCPATH)rx-elf-objdump
OBJS = ../../lib/start.o ../../common/lowlevel.o ../../common/intvect.o $(TARGET).o arp.o ip.o icmp.o ../../lib/libtkdnhal.a 
LFLAGS = -L../../lib/gcc -L../../lib -lrxduino -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -ltkdnhal -nostartfiles 
MAKEFILE = makefile
RAMSCRIPT=../../common/rx62n_ram_standalone.ld
ROMSCRIPT=../../common/rx62n_rom_standalone.ld
make = make --no-print-directory

all:	rom

lib:	hal rxduino libs 

hal: 
	cd ../../core/tkdnhal && $(make)
rxduino: 
	cd ../../core/rxduino && $(make)
libs:
	cd ../../lib && $(make)

ram:	$(OBJS) $(MAKEFILE) $(RAMSCRIPT)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(RAMSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

rom:	$(OBJS) $(MAKEFILE) $(ROMSCRIPT)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(ROMSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

list:	$(TARGET).elf
	$(DMP) -S $(TARGET).elf > $(TARGET).lst
	$(DMP) -s $(TARGET).elf > $(TARGET).txt

$(TARGET).elf:	$(OBJS) $(MAKEFILE)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(LDSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

start.o:	start.asm
	$(AS) -o start.o start.asm

.c.o:
	$(CC) -c $< -o $@

.cpp.o:
	$(CPP) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(TARGET).out
	rm -f $(TARGET).elf
	rm -f $(TARGET).lst
	rm -f $(TARGET).map
	rm -f $(TARGET).mot

libclean:
	cd core/tkdnhal && $(make) clean
	cd core/rxduino && $(make) clean
	cd lib && $(make) clean
	cd sample && $(make) clean

