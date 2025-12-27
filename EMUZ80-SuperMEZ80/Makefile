-include .config

PROGPORT ?= /dev/tty.usbmodem1444301
CONSPORT ?= /dev/cu.usbserial-144440

BOARD ?= SUPERMEZ80_SPI
#BOARD ?= SUPERMEZ80_CPM
#BOARD ?= EMUZ80_57Q
#BOARD ?= Z8S180_57Q

PIC ?= 18F47Q43
#PIC ?= 18F47Q83
#PIC ?= 18F47Q84
#PIC ?= 18F57Q43

Z180_CLK_MULT ?= 2
Z180_NO_DMA ?= 0
Z180_UART ?= 0

TEST_REPEAT ?= 10

PJ_DIR ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

ifeq ($(origin XC8), undefined)
  ifneq (,$(wildcard /Applications/microchip/xc8/v2.40/bin/xc8))
    XC8 = /Applications/microchip/xc8/v2.40/bin/xc8
  else
    ifneq (,$(wildcard /opt/microchip/xc8/v2.36/bin/xc8))
      XC8 = /opt/microchip/xc8/v2.36/bin/xc8
    else
      $(error Missing XC8 complier. Please install XC8)
    endif
  endif
endif
XC8_OPTS ?= --chip=$(PIC) --std=c99
#XC8 ?= /Applications/microchip/xc8/v2.40/bin/xc8-cc
#XC8_OPTS ?= -mcpu=$(PIC) -std=c99

PP3_DIR ?= $(PJ_DIR)/tools/a-p-prog/sw
PP3_OPTS ?= -c $(PROGPORT) -s 100 -v 2 -r 30 -t $(PIC)

FATFS_DIR ?= $(PJ_DIR)/FatFs
DRIVERS_DIR ?= $(PJ_DIR)/drivers
SRC_DIR ?= $(PJ_DIR)/src
BUILD_DIR ?= $(PJ_DIR)/$(shell echo build/$(BOARD).$(PIC) | tr A-Z a-z)
CPM2_DIR ?= $(PJ_DIR)/cpm2
MODEM_XFER_DIR ?= $(PJ_DIR)/libraries/modem_xfer/src
HEXFILE ?= $(shell echo $(BOARD)-$(PIC).hex | tr A-Z a-z)

ASM_DIR ?= $(PJ_DIR)/tools/zasm
ASM ?= $(ASM_DIR)/zasm
ASM_OPTS ?= --opcodes --bin --target=ram --reqcolon

FATFS_SRCS ?= $(FATFS_DIR)/source/ff.c
DISK_SRCS ?= \
    $(DRIVERS_DIR)/diskio.c $(DRIVERS_DIR)/utils.c $(DRIVERS_DIR)/util_memalloc.c
MODEM_XFER_SRC ?= \
     $(MODEM_XFER_DIR)/modem_xfer.c \
     $(MODEM_XFER_DIR)/ymodem_send.c \
     $(MODEM_XFER_DIR)/ymodem.c
SRCS ?= $(SRC_DIR)/supermez80.c $(SRC_DIR)/disas.c $(SRC_DIR)/disas_z80.c $(SRC_DIR)/memory.c \
    $(SRC_DIR)/monitor.c \
    $(SRC_DIR)/monitor_fs.c \
    $(SRC_DIR)/modem.c \
    $(SRC_DIR)/timer.c \
    $(SRC_DIR)/io.c \
    $(SRC_DIR)/io_aux.c \
    $(SRC_DIR)/board.c

#
# board dependent stuff
#
ifeq ($(BOARD),SUPERMEZ80_SPI)
    SRCS += $(SRC_DIR)/boards/supermez80_spi.c
    SRCS += $(SRC_DIR)/boards/supermez80_spi_ioexp.c
    PIC_IOBASE = 0
endif
ifeq ($(BOARD),SUPERMEZ80_CPM)
    SRCS += $(SRC_DIR)/boards/supermez80_cpm.c
    PIC_IOBASE = 0
endif
ifeq ($(BOARD),EMUZ80_57Q)
    SRCS += $(SRC_DIR)/boards/emuz80_57q.c
    PIC_IOBASE = 0
endif
ifeq ($(BOARD),Z8S180_57Q)
    SRCS += $(SRC_DIR)/boards/z8s180_57q.c
    PIC_IOBASE = 128
endif

CONFIG_SUPERMEZ80_CPM_MMU=1
#CONFIG_CPM_MMU_EXERCISE=1
#CONFIG_NO_MEMORY_CHECK=1
#CONFIG_NO_MON_BREAKPOINT=1
#CONFIG_NO_MON_STEP=1
CONFIG_Z80_CLK_HZ=$(Z80_CLK_HZ)
CONFIG_PIC_IOBASE=$(PIC_IOBASE)
NO_MONITOR ?= 0
ifneq ($(NO_MONITOR), 0)
    CONFIG_NO_MONITOR=1
endif
#CONFIG_AUX_FILE=1

INCS ?=-I$(SRC_DIR) -I$(DRIVERS_DIR) -I$(FATFS_DIR)/source -I$(BUILD_DIR) -I$(MODEM_XFER_DIR)

HDRS ?= $(SRC_DIR)/supermez80.h $(SRC_DIR)/picconfig.h \
        $(DRIVERS_DIR)/SPI.c $(DRIVERS_DIR)/SPI.h $(DRIVERS_DIR)/SDCard.h \
        $(DRIVERS_DIR)/mcp23s08.h \
        $(MODEM_XFER_DIR)/modem_xfer.h \
        $(SRC_DIR)/disas.h $(SRC_DIR)/disas_z80.h \
        $(BUILD_DIR)/ipl.inc $(BUILD_DIR)/trampoline.inc $(BUILD_DIR)/mmu_exercise.inc \
        $(BUILD_DIR)/trampoline_cleanup.inc \
        $(BUILD_DIR)/trampoline_nmi.inc \
        $(BUILD_DIR)/dma_helper.inc \
        $(BUILD_DIR)/dummy.inc \
        $(BUILD_DIR)/z8s180_57q_ipl.inc \
        $(BUILD_DIR)/config.h \
        $(DRIVERS_DIR)/pic18f47q43_spi.c \
        $(DRIVERS_DIR)/SDCard.c \
        $(DRIVERS_DIR)/mcp23s08.c \
        $(SRC_DIR)/boards/emuz80_common.c

ASM_HDRS = $(BUILD_DIR)/supermez80_asm.inc \
	$(BUILD_DIR)/config_asm.inc \

all: $(BUILD_DIR)/$(HEXFILE) \
    $(BUILD_DIR)/CPMDISKS.PIO/drivea.dsk \
    $(BUILD_DIR)/CPMDISKS.180/drivea.dsk

$(BUILD_DIR)/$(HEXFILE): $(SRCS) $(FATFS_SRCS) $(DISK_SRCS) $(MODEM_XFER_SRC) $(HDRS) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        $(XC8) $(XC8_OPTS) $(INCS) $(SRCS) $(FATFS_SRCS) $(DISK_SRCS) $(MODEM_XFER_SRC) && \
        mv supermez80.hex $(HEXFILE)

$(BUILD_DIR)/%.inc: $(SRC_DIR)/%.z80 $(ASM) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        cp $< . && \
        $(ASM) $(ASM_OPTS) -l $*.lst -o $*.bin $*.z80 && \
        cat $*.bin | xxd -i > $@

$(BUILD_DIR)/%.inc: $(SRC_DIR)/boards/%.z80 $(ASM) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        cp $< . && \
        $(ASM) $(ASM_OPTS) -l $*.lst -o $*.bin $*.z80 && \
        cat $*.bin | xxd -i > $@

$(BUILD_DIR)/boot.bin: $(CPM2_DIR)/boot.asm $(BUILD_DIR) $(ASM) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        cp $< . && \
        $(ASM) $(ASM_OPTS) -l boot.lst -o boot.bin boot.asm
$(BUILD_DIR)/bios.bin: $(CPM2_DIR)/bios.asm $(BUILD_DIR) $(ASM) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        cp $< . && \
        $(ASM) $(ASM_OPTS) -l bios.lst -o bios.bin bios.asm

$(BUILD_DIR)/CPMDISKS.PIO/drivea.dsk: $(BUILD_DIR)/boot.bin $(BUILD_DIR)/bios.bin
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
	mkdir -p CPMDISKS.PIO; \
	dd if=$(CPM2_DIR)/z80pack-cpm2-1.dsk of=$@ bs=128; \
	dd if=boot.bin of=$@ bs=128 seek=0  count=1 conv=notrunc; \
	dd if=bios.bin of=$@ bs=128 seek=45 count=6 conv=notrunc

$(BUILD_DIR)/boot_z180.bin: $(CPM2_DIR)/boot_z180.asm $(BUILD_DIR) $(ASM) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        cp $< . && \
        $(ASM) $(ASM_OPTS) -o $@ boot_z180.asm
$(BUILD_DIR)/bios_z180.bin: $(CPM2_DIR)/bios_z180.asm $(BUILD_DIR) $(ASM) $(ASM_HDRS)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
        cp $< . && \
        $(ASM) $(ASM_OPTS) -o $@ bios_z180.asm

$(BUILD_DIR)/CPMDISKS.180/drivea.dsk: $(BUILD_DIR)/boot_z180.bin $(BUILD_DIR)/bios_z180.bin
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
	mkdir -p CPMDISKS.180; \
	dd if=$(CPM2_DIR)/z80pack-cpm2-1.dsk of=$@ bs=128; \
	dd if=boot_z180.bin of=$@ bs=128 seek=0  count=1 conv=notrunc; \
	dd if=bios_z180.bin of=$@ bs=128 seek=45 count=6 conv=notrunc

$(BUILD_DIR)/supermez80_asm.inc: $(SRC_DIR)/supermez80_asm.inc
	mkdir -p $(BUILD_DIR)
	cp -p $(SRC_DIR)/supermez80_asm.inc $(BUILD_DIR)/supermez80_asm.inc

$(BUILD_DIR)/config.h:
	mkdir -p $(BUILD_DIR)
	rm -f $@
	if [ "$(CONFIG_SUPERMEZ80_CPM_MMU)" != "" ]; then \
	    echo "#define SUPERMEZ80_CPM_MMU $(CONFIG_SUPERMEZ80_CPM_MMU)" >> $@; \
	fi
	if [ "$(CONFIG_CPM_MMU_EXERCISE)" != "" ]; then \
	    echo "#define CPM_MMU_EXERCISE $(CONFIG_CPM_MMU_EXERCISE)" >> $@; \
	fi
	if [ "$(CONFIG_NO_MEMORY_CHECK)" != "" ]; then \
	    echo "#define NO_MEMORY_CHECK $(CONFIG_NO_MEMORY_CHECK)" >> $@; \
	fi
	if [ "$(CONFIG_NO_MON_BREAKPOINT)" != "" ]; then \
	    echo "#define NO_MON_BREAKPOINT $(CONFIG_NO_MON_BREAKPOINT)" >> $@; \
	fi
	if [ "$(CONFIG_NO_MON_STEP)" != "" ]; then \
	    echo "#define NO_MON_STEP $(CONFIG_NO_MON_STEP)" >> $@; \
	fi
	if [ "$(CONFIG_Z80_CLK_HZ)" != "" ]; then \
	    echo "#define Z80_CLK_HZ $(CONFIG_Z80_CLK_HZ)" >> $@; \
	fi
	if [ "$(CONFIG_PIC_IOBASE)" != "" ]; then \
	    echo "#define PIC_IOBASE $(CONFIG_PIC_IOBASE)" >> $@; \
	fi
	if [ "$(CONFIG_NO_MONITOR)" != "" ]; then \
	    echo "#define NO_MONITOR $(CONFIG_NO_MONITOR)" >> $@; \
	fi
	if [ "$(CONFIG_AUX_FILE)" != "" ]; then \
	    echo "#define AUX_FILE $(CONFIG_AUX_FILE)" >> $@; \
	fi

$(BUILD_DIR)/config_asm.inc: Makefile
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
	rm -f $@
	echo "PIC_IOBASE	EQU	$(PIC_IOBASE)" >> $@; \
	if [ "$(Z180_UART)" != 0 ]; then \
	    echo "UART_180	EQU	1" >> $@; \
	    echo "UART_PIC	EQU	0" >> $@; \
	else \
	    echo "UART_180	EQU	0" >> $@; \
	    echo "UART_PIC	EQU	1" >> $@; \
	fi
	if [ "$(Z180_NO_DMA)" != 0 ]; then \
	    echo "BYTE_RW	EQU	1" >> $@; \
	else \
	    echo "BYTE_RW	EQU	0" >> $@; \
	fi
	if [ "$(Z180_CLK_MULT)" == 0 ]; then \
	    echo "CLOCK_0	EQU	1" >> $@; \
	else \
	    echo "CLOCK_0	EQU	0" >> $@; \
	fi
	if [ "$(Z180_CLK_MULT)" == 1 ]; then \
	    echo "CLOCK_1	EQU	1" >> $@; \
	else \
	    echo "CLOCK_1	EQU	0" >> $@; \
	fi
	if [ "$(Z180_CLK_MULT)" == 2 ]; then \
	    echo "CLOCK_2	EQU	1" >> $@; \
	else \
	    echo "CLOCK_2	EQU	0" >> $@; \
	fi
	@echo ===================================
	@cat $@
	@echo ===================================

upload: $(BUILD_DIR)/$(HEXFILE) $(PP3_DIR)/pp3
	cd $(PP3_DIR) && ./pp3 $(PP3_OPTS) $(BUILD_DIR)/$(HEXFILE)

dist:: build_all
	cd build/; \
	for build in *.*; do \
	    mkdir -p $(PJ_DIR)/dist/$${build}; \
	    cp -rp $${build}/CPMDISKS* $(PJ_DIR)/dist/$${build}; \
	    cp -p $${build}/*.h $(PJ_DIR)/dist/$${build}; \
	    cp -p $${build}/*.inc $(PJ_DIR)/dist/$${build}; \
	    cp -p $${build}/*.hex $(PJ_DIR)/dist/$${build}; \
	done

test::
	cd test && PORT=$(CONSPORT) ./test.sh

test_repeat::
	cd test && for i in $$(seq $(TEST_REPEAT)); do \
          PORT=$(CONSPORT) ./test.sh || exit 1; \
        done

test_time::
	cd test && PORT=$(CONSPORT) ./measure_time.sh

test_monitor::
	cd test && PORT=$(CONSPORT) ./monitor.sh

build_all::
	make BOARD=SUPERMEZ80_SPI PIC=18F47Q43
	make BOARD=SUPERMEZ80_CPM PIC=18F47Q43
	make BOARD=EMUZ80_57Q     PIC=18F57Q43
	make BOARD=Z8S180_57Q     PIC=18F57Q43
	ls -l build/*.*/*.hex

clean::
	rm -rf $(PJ_DIR)/build/*.*

$(ASM):
	cd $(ASM_DIR) && make

$(PP3_DIR)/pp3:
	cd $(PP3_DIR) && make

realclean:: clean
	cd $(ASM_DIR) && make clean
	cd $(PP3_DIR) && make clean
