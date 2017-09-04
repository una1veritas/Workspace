###############################################################################
# Makefile for pepper
###############################################################################

## General Flags
PROJECT = pepper45
MCU = attiny45
TARGET = $(PROJECT).elf
CC = avr-gcc

## Options common to compile, link and assembly rules
COMMON = -mmcu=$(MCU) -DF_CPU=16500000 

## Compile options common for all C compilation units.
CFLAGS = $(COMMON)
CFLAGS += -Wall -Os -fsigned-char -DPEPPER45 -DFIRMWARE_VERSION=\"$(version)\"

## Assembly specific flags
ASMFLAGS = $(COMMON)
ASMFLAGS += -x assembler-with-cpp -Wa,

## Linker flags
LDFLAGS = $(COMMON)
LDFLAGS +=  -Wl,-Map=$(PROJECT).map


## Intel Hex file production flags
HEX_FLASH_FLAGS = -R .eeprom


## Include Directories
INCLUDES = -I. -I"../usbdrv"

## Objects that must be built in order to link
OBJECTS = usbdrv.o usbdrvasm.o gio.o main.o

## Objects explicitly added by the user
LINKONLYOBJECTS = 

## Build
all: $(TARGET) $(PROJECT).hex $(PROJECT).lss size

## Compile
usbdrv.o: ../usbdrv/usbdrv.c
	$(CC) $(INCLUDES) $(CFLAGS) -c  $<

usbdrvasm.o: ../usbdrv/usbdrvasm.S
	$(CC) $(INCLUDES) $(ASMFLAGS) -c  $<

gio.o: ../gio.c ../gio.h
	$(CC) $(INCLUDES) $(CFLAGS) -c  $<

main.o: ../main.c ../gio.h
	$(CC) $(INCLUDES) $(CFLAGS) -c  $<

##Link
$(TARGET): $(OBJECTS)
	 $(CC) $(LDFLAGS) $(OBJECTS) $(LINKONLYOBJECTS) $(LIBDIRS) $(LIBS) -o $(TARGET)

%.hex: $(TARGET)
	avr-objcopy -O ihex $(HEX_FLASH_FLAGS)  $< $@

%.lss: $(TARGET)
	avr-objdump -h -S $< > $@

size: ${TARGET}
	@echo
	@avr-size -C --mcu=${MCU} ${TARGET}

## Clean target
.PHONY: clean
clean:
	-rm -rf $(OBJECTS) $(PROJECT).*

flash: $(PROJECT).hex
	avrdude -c usbtiny -p $(MCU) -U flash:w:$(PROJECT).hex

fuse: 
	avrdude -c usbtiny -p $(MCU) -U hfuse:w:0x5d:m -U lfuse:w:0xe1:m
