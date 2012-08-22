TARGET = nahimon
CCPATH = /usr/local/tkdn-20110720/rx-elf/bin/
AS  = $(CCPATH)rx-elf-as 
CC  = $(CCPATH)rx-elf-gcc -Wall -g -I../../include/tkdnhal -I../../include/lib $(CFLAGS) -O2
CPP = $(CCPATH)rx-elf-gcc -Wall -g -I../../include/tkdnhal -I../../include/lib $(CFLAGS) -O2
LNK = $(CCPATH)rx-elf-ld
CNV = $(CCPATH)rx-elf-objcopy --srec-forceS3 --srec-len 32 -O srec
CNVBIN = $(CCPATH)rx-elf-objcopy -O binary
DMP = $(CCPATH)rx-elf-objdump
OBJS = main.o fatfs-test.o ../../common/intvect.o ../../common/lowlevel.o libmisc.o command.o usbshow.o
#OBJS = ../../lib/start.o $(TARGET).o 
LFLAGS = -L../../lib/gcc -L../../lib -lfatfs -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -ltkdnhal -nostartfiles
SDRAMSCRIPT= rx62n_sdram_standalone.ld
RAMSCRIPT=rx62n_ram_standalone.ld
ROMSCRIPT=rx62n_rom_standalone.ld
GADGETSCRIPT=../../common/gr_sakura_test_Release_auto.gsi
STARTUP=start.o
GSTARTUP=
MAKEFILE = makefile

all:	rom

ram:	$(OBJS) $(STARTUP) $(MAKEFILE) $(RAMSCRIPT)
	$(LNK) -Map $(TARGET).map $(STARTUP) $(OBJS) -T $(RAMSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

rom:	$(OBJS) $(STARTUP) $(MAKEFILE) $(ROMSCRIPT)
	$(LNK) -Map $(TARGET).map $(STARTUP) $(OBJS) -T $(ROMSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

gadget:	$(OBJS) $(GSTARTUP) $(MAKEFILE) $(ROMSCRIPT)
	$(LNK) -Map $(TARGET).map $(GSTARTUP) $(OBJS) -T $(GADGETSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot
	$(CNVBIN) $(TARGET).elf  $(TARGET).bin

sdram:	$(OBJS) $(MAKEFILE) $(ROMSCRIPT)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(SDRAMSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

list:	$(TARGET).elf
	$(DMP) -S $(TARGET).elf > $(TARGET).lst
	$(DMP) -s $(TARGET).elf > $(TARGET).txt

$(TARGET).elf:	$(OBJS) $(MAKEFILE)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(LDSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

.s.o:
	$(AS) -c $< -o $@

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

