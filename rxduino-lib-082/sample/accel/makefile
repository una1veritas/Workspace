TARGET = accel
CCPATH = /usr/local/tkdn-20110720/rx-elf/bin/
AS  = $(CCPATH)rx-elf-as 
CC  = $(CCPATH)rx-elf-gcc -Wall -g -I../../include/tkdnhal $(CFLAGS) -I../../include/lib
CPP = $(CCPATH)rx-elf-gcc -Wall -g -I../../include/tkdnhal $(CFLAGS) -I../../include/rxduino  -I../../include/lib
LNK = $(CCPATH)rx-elf-ld
CNV = $(CCPATH)rx-elf-objcopy --srec-forceS3 --srec-len 32 -O srec
DMP = $(CCPATH)rx-elf-objdump
OBJS = ../../lib/start.o $(TARGET).o ../../common/intvect.o
LFLAGS = -L../../lib/gcc -L../../lib -lspi -lrxduino -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -ltkdnhal -nostartfiles 
RAMSCRIPT=../../common/rx62n_ram_standalone.ld
ROMSCRIPT=../../common/rx62n_rom_standalone.ld
MAKEFILE = makefile

all:	rom

ram:	$(OBJS) $(MAKEFILE) $(RAMSCRIPT)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(RAMSCRIPT) $(LFLAGS) -o $(TARGET).elf 

rom:	$(OBJS) $(MAKEFILE) $(ROMSCRIPT)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(ROMSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

list:	$(TARGET).elf
	$(DMP) -S $(TARGET).elf > $(TARGET).lst
	$(DMP) -s $(TARGET).elf > $(TARGET).txt

$(TARGET).elf:	$(OBJS) $(MAKEFILE)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(LDSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

.c.o:
	$(CC) -c $< -o $@

.cpp.o:
	$(CPP) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(TARGET).elf
	rm -f $(TARGET).lst
	rm -f $(TARGET).map
	rm -f $(TARGET).mot

