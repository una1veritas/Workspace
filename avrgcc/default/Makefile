###############################################################################
# Makefile for the project avrgcc_bystudio
###############################################################################

## General Flags
PROJECT = avrgcc_bystudio
MCU = atmega328p
TARGET = avrgcc_bystudio.elf
CC = 

CPP = avr-g++

## Options common to compile, link and assembly rules
COMMON = -mmcu=$(MCU)

## Compile options common for all C compilation units.
CFLAGS = $(COMMON)
CFLAGS += -Wall -gdwarf-2 -std=gnu99     -DF_CPU=16000000UL -Os -funsigned-char -funsigned-bitfields -fpack-struct -fshort-enums
CFLAGS += -MD -MP -MT $(*F).o -MF dep/$(@F).d 

## Assembly specific flags
ASMFLAGS = $(COMMON)
ASMFLAGS += $(CFLAGS)
ASMFLAGS += -x assembler-with-cpp -Wa,-gdwarf2

## Linker flags
LDFLAGS = $(COMMON)
LDFLAGS +=  -Wl,-Map=avrgcc_bystudio.map


## Intel Hex file production flags
HEX_FLASH_FLAGS = -R .eeprom -R .fuse -R .lock -R .signature

HEX_EEPROM_FLAGS = -j .eeprom
HEX_EEPROM_FLAGS += --set-section-flags=.eeprom="alloc,load"
HEX_EEPROM_FLAGS += --change-section-lma .eeprom=0 --no-change-warnings


## Include Directories
INCLUDES = -I"Y:\Documents\Eclipse\Workspace\avrgcc\arduino\cores\arduino" -I"Y:\Documents\Eclipse\Workspace\avrgcc\arduino\variants\standard" 

## Library Directories
LIBDIRS = -L"Y:\Documents\Eclipse\Workspace\avrgcc\arduino" 

## Libraries
LIBS = -larduinocore 

## Objects that must be built in order to link
OBJECTS = main.o 

## Objects explicitly added by the user
LINKONLYOBJECTS = 

## Build
all: $(TARGET) avrgcc_bystudio.hex avrgcc_bystudio.eep avrgcc_bystudio.lss## Compile
main.o: ../main.cpp
	$(CPP) $(INCLUDES) $(CFLAGS) -c  $<

##Link
$(TARGET): $(OBJECTS)
	 $(CPP) $(LDFLAGS) $(OBJECTS) $(LINKONLYOBJECTS) $(LIBDIRS) $(LIBS) -o $(TARGET)

%.hex: $(TARGET)
	avr-objcopy -O ihex $(HEX_FLASH_FLAGS)  $< $@

%.eep: $(TARGET)
	-avr-objcopy $(HEX_EEPROM_FLAGS) -O ihex $< $@ || exit 0

%.lss: $(TARGET)
	avr-objdump -h -S $< > $@

## Clean target
.PHONY: clean
clean:
	-rm -rf $(OBJECTS) avrgcc_bystudio.elf dep/* avrgcc_bystudio.hex avrgcc_bystudio.eep avrgcc_bystudio.lss avrgcc_bystudio.map


## Other dependencies
-include $(shell mkdir dep 2>NUL) $(wildcard dep/*)

