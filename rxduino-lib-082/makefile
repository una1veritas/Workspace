###################################################################################################
# RXduinoライブラリ Makefile
# (C) Copyright 2011-2012 TokushuDenshiKairo Inc.
# 
# 使い方
#  make boardlib         各種ボード用のライブラリ(特電HAL、RXduino、外部ペリフェラル)を作る
#  make libclean         ライブラリをクリアする
#  make samples          全てのサンプルプログラムをビルドする
#  make rom              userapp.cppをROM実行用にビルドする
#  make ram              userapp.cppをRAM実行用にビルドする
#  make list             生成されたelfファイルを逆アセンブルする
#  make use_raxino       RaXino用のユーザアプリをビルドできるように環境をセットする
#  make use_rx62n0_sdram 究極RX62Nボード初期版(SDRAM使用)のアプリをビルドできるように。
#  make use_rx62n0_mmc   究極RX62Nボード初期版(SDカード使用)のアプリをビルドできるように。
#  make use_rx62n        究極RX62NボードRevA版のアプリをビルドできるように環境をセットする
#  make use_rxmega       RXMEGA用のアプリをビルドできるように環境をセットする
# 
# ユーザアプリを作る際には、お使いのボードに合わせて
#   make use_raxino
# や
#   make use_rx62na
# といったコマンドを一度実行します。
# その後、make ram または make rom で、userapp.cppがコンパイルされ、アプリが生成されます。
# 
###################################################################################################

TARGET = userapp
CCPATH = /usr/local/rx-elf/bin/
AS  = $(CCPATH)rx-elf-as 
CC  = $(CCPATH)rx-elf-gcc -Wall -g -I./include/tkdnhal -I./include/lib/ 
CPP = $(CCPATH)rx-elf-gcc -Wall -g -I./include/tkdnhal -I./include/rxduino -I./include/lib/ 
LNK = $(CCPATH)rx-elf-ld
CNV = $(CCPATH)rx-elf-objcopy --srec-forceS3 --srec-len 32 -O srec
CNVBIN = $(CCPATH)rx-elf-objcopy -O binary
DMP = $(CCPATH)rx-elf-objdump
OBJS = common/intvect.o common/lowlevel.o $(TARGET).o 
#LFLAGS = -Llib/gcc -Llib -lfatfs -lliquidcrystal -lnokialcd -loled -lrxduino -ltkdnhal -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -nostartfiles
RXDUINO_STANDARD_LIBS = -lspi -lservo -lliquidcrystal -lsdmmc -leeprom
LFLAGS = -Llib/gcc -Llib $(RXDUINO_STANDARD_LIBS) -lrxduino -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -ltkdnhal -nostartfiles
MAKEFILE = makefile
RAMSCRIPT=common/rx62n_ram_standalone.ld
ROMSCRIPT=common/rx62n_rom_standalone.ld
GADGETSCRIPT=common/gr_sakura_test_Release_auto.gsi
STARTUP=lib/start.o
GSTARTUP=
make = make --no-print-directory 

all:	rom

#boardlib: 
#	$(make) liballclean
#	$(make) lib_raxino
#	$(make) libclean
#	$(make) lib_ult62n0_sdram
#	$(make) libclean
#	$(make) lib_ult62n0_mmc
#	$(make) libclean
#	$(make) lib_ult62n
#	$(make) libclean
#	$(make) lib_rxmega

use_raxino:
	$(make) uselib "TARGET_LIB=raxino"
	cp include/tkdnhal/brd_raxino.h include/tkdnhal/boarddef.h
	rm -f common/intvect.o
use_ult62n0_sdram:
	$(make) uselib "TARGET_LIB=ult62n0_sdram"
	cp include/tkdnhal/brd_ult62n0_sdram.h include/tkdnhal/boarddef.h
	rm -f common/intvect.o
use_ult62n0_mmc:
	$(make) uselib "TARGET_LIB=ult62n0_mmc"
	cp include/tkdnhal/brd_ult62n0_mmc.h include/tkdnhal/boarddef.h
	rm -f common/intvect.o
use_ult62n:
	$(make) uselib "TARGET_LIB=ult62n"
	cp include/tkdnhal/brd_ult62n.h include/tkdnhal/boarddef.h
	rm -f common/intvect.o
use_rxmega:
	$(make) uselib "TARGET_LIB=rxmega"
	cp include/tkdnhal/brd_rxmega.h include/tkdnhal/boarddef.h
	rm -f common/intvect.o

#lib_raxino:
#	$(make) lib "TARGET_LIB=raxino"
#	$(make) use_raxino
#lib_ult62n0_sdram:
#	$(make) lib "TARGET_LIB=ult62n0_sdram"
#	$(make) use_ult62n0_sdram
#lib_ult62n0_mmc:
#	$(make) lib "TARGET_LIB=ult62n0_mmc"
#	$(make) use_ult62n0_mmc
#lib_ult62n:
#	$(make) lib "TARGET_LIB=ult62n"
#	$(make) use_ult62n
#lib_rxmega:
#	$(make) lib "TARGET_LIB=rxmega"
#	$(make) use_rxmega
#
#lib:	hal rxduino libs 
#
#hal: 
#	cd core/tkdnhal && $(make)
#rxduino: 
#	cd core/rxduino && $(make)
#libs:
#	cd lib && $(make)
samples:
	cd sample && $(make)

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

list:	$(TARGET).elf
	$(DMP) -S $(TARGET).elf > $(TARGET).lst
	$(DMP) -s $(TARGET).elf > $(TARGET).txt

$(TARGET).elf:	$(OBJS) $(MAKEFILE)
	$(LNK) -Map $(TARGET).map $(OBJS) -T $(LDSCRIPT) $(LFLAGS) -o $(TARGET).elf 
	$(CNV) $(TARGET).elf  $(TARGET).mot

start.o:	start.asm
	$(AS) -o start.o start.asm

.c.o:
	$(CC) -Wall -O2 -c $< -o $@

.cpp.o:
	$(CPP) -Wall -O2 -c $< -o $@

clean: 
	rm -f *.o
	rm -f $(TARGET).out
	rm -f $(TARGET).elf
	rm -f $(TARGET).lst
	rm -f $(TARGET).map
	rm -f $(TARGET).mot

uselib:
	cd lib && $(make) uselib

#libclean:
#	cd core/tkdnhal && $(make) clean
#	cd core/rxduino && $(make) clean
#	rm -f common/*.o
#	cd lib && $(make) clean
#
#liballclean:
#	cd core/tkdnhal && $(make) clean
#	cd core/rxduino && $(make) clean
#	cd lib && $(make) allclean
