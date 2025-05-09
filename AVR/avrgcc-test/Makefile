# Name: Makefile
# Author: <insert your name here>
# Copyright: <insert your copyright message here>
# License: <insert your license reference here>

# This is a prototype Makefile. Modify it according to your needs.
# You should at least check the settings for
# DEVICE ....... The AVR device you compile for
# CLOCK ........ Target AVR clock rate in Hertz
# OBJECTS ...... The object files created from your source files. This list is
#                usually the same as the list of source files with suffix ".o".
# PROGRAMMER ... Options to avrdude which define the hardware you use for
#                uploading to the AVR and the interface where this hardware
#                is connected.
# FUSES ........ Parameters for avrdude to flash the fuses appropriately.

OBJECTS = main.o

DEVICE     = atmega1284p
CLOCK      = 16000000L
PROG_UNO = -carduino -b115200
PROG_DUE = -carduino -b57600
FUSES      = -U hfuse:w:0xdb:m -U lfuse:w:0xff:m 
#-U efuse:w:0x05m
# ATMega8 fuse bits (fuse bits for other devices are different!):
# Example for 8 MHz internal oscillator
# Fuse high byte:
# 0xd9 = 1 1 0 1   1 0 0 1 <-- BOOTRST (boot reset vector at 0x0000)
#        ^ ^ ^ ^   ^ ^ ^------ BOOTSZ0
#        | | | |   | +-------- BOOTSZ1
#        | | | |   +---------- EESAVE (set to 0 to preserve EEPROM over chip erase)
#        | | | +-------------- CKOPT (clock option, depends on oscillator type)
#        | | +---------------- SPIEN (if set to 1, serial programming is disabled)
#        | +------------------ WDTON (if set to 0, watchdog is always on)
#        +-------------------- RSTDISBL (if set to 0, RESET pin is disabled)
# Fuse low byte:
# 0x24 = 0 0 1 0   0 1 0 0
#        ^ ^ \ /   \--+--/
#        | |  |       +------- CKSEL 3..0 (8M internal RC)
#        | |  +--------------- SUT 1..0 (slowly rising power)
#        | +------------------ BODEN (if 0, brown-out detector is enabled)
#        +-------------------- BODLEVEL (if 0: 4V, if 1: 2.7V)

# Example for 12 MHz external crystal:
# Fuse high byte:
# 0xc9 = 1 1 0 0   1 0 0 1 <-- BOOTRST (boot reset vector at 0x0000)
#        ^ ^ ^ ^   ^ ^ ^------ BOOTSZ0
#        | | | |   | +-------- BOOTSZ1
#        | | | |   +---------- EESAVE (set to 0 to preserve EEPROM over chip erase)
#        | | | +-------------- CKOPT (clock option, depends on oscillator type)
#        | | +---------------- SPIEN (if set to 1, serial programming is disabled)
#        | +------------------ WDTON (if set to 0, watchdog is always on)
#        +-------------------- RSTDISBL (if set to 0, RESET pin is disabled)
# Fuse low byte:
# 0x9f = 1 0 0 1   1 1 1 1
#        ^ ^ \ /   \--+--/
#        | |  |       +------- CKSEL 3..0 (external >8M crystal)
#        | |  +--------------- SUT 1..0 (crystal osc, BOD enabled)
#        | +------------------ BODEN (if 0, brown-out detector is enabled)
#        +-------------------- BODLEVEL (if 0: 4V, if 1: 2.7V)


# Tune the lines below only if you know what you are doing:

ARDUINOINCDIR  = ./arduino/cores/arduino
#ARDUINOVARIANT = ./arduino/variants/standard
ARDUINOVARIANT = ./arduino/variants/quaranta
ARDUINOCORELIBDIR = ../Arduino_core/Release
ARDUINOCORELIB = Arduino_core

#AVRDUDE = avrdude $(PROG_DUE) -P$(COMPORT) -p $(DEVICE) 
CCCOMPILE  = avr-gcc -Wall -Os -DF_CPU=$(CLOCK) -mmcu=$(DEVICE) -I$(ARDUINOINCDIR) -I$(ARDUINOVARIANT)
CPPCOMPILE = avr-g++ -Wall -Os -DF_CPU=$(CLOCK) -mmcu=$(DEVICE) -I$(ARDUINOINCDIR) -I$(ARDUINOVARIANT)

.SUFFIXES: .hex .elf .cpp

HEXFILE    = main.hex

# symbolic targets:
all:	$(OBJECTS) $(HEXFILE)
	@echo "Build finished."

alltest:
	@echo $(SOURCES)
	@echo $(OBJECTS)
	
	
.c.o:
	$(CCCOMPILE) -c $< -o $@

.cpp.o:
	$(CPPCOMPILE) -c $< -o $@

.S.o:
	$(CCCOMPILE) -x assembler-with-cpp -c $< -o $@
# "-x assembler-with-cpp" should not be necessary since this is the default
# file type for the .S (with capital S) extension. However, upper case
# characters are not always preserved on Windows. To ensure WinAVR
# compatibility define the file type manually.

.c.s:
	$(CCCOMPILE) -S $< -o $@

winupload.due.%:	all
	avrdude $(PROG_DUE) -P\\\\.\\$* -p $(DEVICE) -U flash:w:main.hex:i

winupload.uno.%:	all
	avrdude $(PROG_UNO) -P\\\\.\\$* -p $(DEVICE) -U flash:w:main.hex:i

macupload.due.%:	all
	avrdude $(PROG_DUE) -P/dev/$* -p $(DEVICE) -U flash:w:main.hex:i

macupload.uno.%:	all
	avrdude $(PROG_UNO) -P/dev/$* -p $(DEVICE) -U flash:w:main.hex:i

#fuse:
#	$(AVRDUDE) $(FUSES)

# Xcode uses the Makefile targets "", "clean" and "install"
install: flash fuse

# if you use a bootloader, change the command below appropriately:
load: all
	bootloadHID main.hex

clean:
	rm -f $(HEXFILE:.hex=.elf) $(OBJECTS) $(CPPOBJECTS)
	rm -f *~


# file targets:
main.elf: $(OBJECTS)
	$(CCCOMPILE) -o main.elf $(OBJECTS) $(CPPOBJECTS) -L$(ARDUINOCORELIBDIR) -l$(ARDUINOCORELIB)

main.hex: main.elf
	rm -f main.hex
	avr-objcopy -j .text -j .data -O ihex main.elf main.hex
# If you have an EEPROM section, you must also create a hex file for the
# EEPROM and add it to the "flash" target.

# Targets for code debugging and analysis:
disasm:	$(HEXFILE:.hex=.elf)
	avr-objdump -d $(HEXFILE:.hex=.elf)

cpp:
	$(CCCOMPILE) -E $(SOURCES)
