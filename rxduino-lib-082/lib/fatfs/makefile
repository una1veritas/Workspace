TARGET = libfatfs.$(TARGET_LIB)
CCPATH = /usr/local/tkdn-20110720/rx-elf/bin/
AS  = $(CCPATH)rx-elf-as 
AR  = $(CCPATH)rx-elf-ar 
RANLIB  = $(CCPATH)rx-elf-ranlib 
CC  = $(CCPATH)rx-elf-gcc -Wall -I../../include/tkdnhal -I../../ -O2
CPP = $(CCPATH)rx-elf-gcc -Wall 
OBJS = mmc_rspi.o ff.o
MAKEFILE = makefile
include ../../core/boardlib.mk

all:	$(TARGET).a $(MAKEFILE)
	cp $(TARGET).a ../../lib/tgtbrd
	cp ff.h ../../include/lib/
	cp ffconf.h ../../include/lib/
	cp integer.h ../../include/lib/

$(TARGET).a:	$(OBJS) $(MAKEFILE)
	rm -f $(TARGET).a
	$(AR) cru $(TARGET).a $(OBJS)
	$(RANLIB) $(TARGET).a

.c.o: $(MAKEFILE)
	$(CC) -c -DTARGET_BOARD=$(TARGET_BOARD) $<

.cpp.o: $(MAKEFILE)
	$(CPP) -c -DTARGET_BOARD=$(TARGET_BOARD) $<

clean:
	rm -f *.o
	rm -f *.a
