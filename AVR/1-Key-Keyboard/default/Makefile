###############################################################################
# Makefile for the project 1-Key-Keyboard
###############################################################################

## General Flags
PROJECT = 1-Key-Keyboard
MCU = attiny45
TARGET = 1-Key-Keyboard.elf
CC = avr-gcc

CPP = avr-g++

## Options common to compile, link and assembly rules
COMMON = -mmcu=$(MCU)

## Compile options common for all C compilation units.
CFLAGS = $(COMMON)
CFLAGS += -Wall -gdwarf-2 -std=gnu99                             -DF_CPU=16500000UL -Os -funsigned-char -funsigned-bitfields -fpack-struct -fshort-enums
CFLAGS += -MD -MP -MT $(*F).o -MF dep/$(@F).d 

## Assembly specific flags
ASMFLAGS = $(COMMON)
ASMFLAGS += $(CFLAGS)
ASMFLAGS += -x assembler-with-cpp -Wa,-gdwarf2

## Linker flags
LDFLAGS = $(COMMON)
LDFLAGS +=  -Wl,-Map=1-Key-Keyboard.map


## Intel Hex file production flags
HEX_FLASH_FLAGS = -R .eeprom -R .fuse -R .lock -R .signature

HEX_EEPROM_FLAGS = -j .eeprom
HEX_EEPROM_FLAGS += --set-section-flags=.eeprom="alloc,load"
HEX_EEPROM_FLAGS += --change-section-lma .eeprom=0 --no-change-warnings


## Include Directories
INCLUDES = -I"F:\Development\AVR\1-Key-Keyboard\usbdrv" -I"F:\Development\AVR\1-Key-Keyboard\." 

## Objects that must be built in order to link
OBJECTS = main.o oddebug.o usbdrv.o usbdrvasm.o 

## Objects explicitly added by the user
LINKONLYOBJECTS = 

## Build
all: $(TARGET) 1-Key-Keyboard.hex 1-Key-Keyboard.eep 1-Key-Keyboard.lss size

## Compile
main.o: ../main.c
	$(CC) $(INCLUDES) $(CFLAGS) -c  $<

oddebug.o: ../usbdrv/oddebug.c
	$(CC) $(INCLUDES) $(CFLAGS) -c  $<

usbdrv.o: ../usbdrv/usbdrv.c
	$(CC) $(INCLUDES) $(CFLAGS) -c  $<

usbdrvasm.o: ../usbdrv/usbdrvasm.S
	$(CC) $(INCLUDES) $(ASMFLAGS) -c  $<

##Link
$(TARGET): $(OBJECTS)
	 $(CC) $(LDFLAGS) $(OBJECTS) $(LINKONLYOBJECTS) $(LIBDIRS) $(LIBS) -o $(TARGET)

%.hex: $(TARGET)
	avr-objcopy -O ihex $(HEX_FLASH_FLAGS)  $< $@

%.eep: $(TARGET)
	-avr-objcopy $(HEX_EEPROM_FLAGS) -O ihex $< $@ || exit 0

%.lss: $(TARGET)
	avr-objdump -h -S $< > $@

size: ${TARGET}
	@echo
	@avr-size -C --mcu=${MCU} ${TARGET}

## Clean target
.PHONY: clean
clean:
	-rm -rf $(OBJECTS) 1-Key-Keyboard.elf dep/* 1-Key-Keyboard.hex 1-Key-Keyboard.eep 1-Key-Keyboard.lss 1-Key-Keyboard.map


## Other dependencies
-include $(shell mkdir dep 2>/dev/null) $(wildcard dep/*)

