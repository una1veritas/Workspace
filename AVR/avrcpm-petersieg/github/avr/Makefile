# AVR-ASM Makefile, derived from the WinAVR template (which
# is public domain), believed to be neutral to any flavor of "make"
# (GNU make, BSD make, SysV make)

#MCU = atmega8
MCU = atmega328P
F_CPU = 20000000
#BAUD = 19200
BAUD = 57600
#BAUD = 115200

DRAM_8BIT = 1

TARGET = avrcpm
ASRC0  = avrcpm.asm 

ASRC0 += config.inc macros.inc init.asm dram-refresh.asm timer.asm utils.asm 
ASRC0 += mmc.asm mmc-old.asm virt_ports.asm
ASRC0 += dsk_cpm.asm dsk_fat16.asm dsk_fsys.asm dsk_mgr.asm dsk_ram.asm 
ASRC0 += 8080int-orig.asm 8080int.asm 8080int-jmp.asm 8080int-t3.asm 8080int-t3-jmp.asm Z80int-jmp.asm

ifneq ($(DRAM_8BIT),0)
  ASRC0 += dram-8bit.inc dram-8bit.asm sw-uart.asm 
else
  ASRC0 += dram-4bit.inc dram-4bit.asm hw-uart.asm 
endif

#ASRC := $(ASRC0) svnrev.inc
ASRC := $(ASRC0) svnrev.inc

# Place -D or -U options here
CDEFS = -DF_CPU=$(F_CPU) -DBAUD=$(BAUD) -D$(MCU) -DDRAM_8BIT=$(DRAM_8BIT)

ASPATH = C:/Programme/Atmel/AVR\ Tools/AvrAssembler2
DEFS = $(ASPATH)/Appnotes

ifeq "$(OS)" "Windows_NT"
  PLATFORM=win32
else
  PLATFORM=Linux
endif

WINE =
ifeq ($(PLATFORM),Linux)
  WINE = wine
endif

AS = $(WINE) $(ASPATH)/avrasm2.exe
ASFLAGS = -I $(DEFS) $(CDEFS)

# Programming support using avrdude. Settings and variables.

AVRDUDE_PROGRAMMER = dragon_isp
AVRDUDE_PORT = usb

AVRDUDE_WRITE_FLASH = -U flash:w:$(TARGET).hex
AVRDUDE_WRITE_EEPROM = -U eeprom:w:$(TARGET).eep


# Uncomment the following if you want avrdude's erase cycle counter.
# Note that this counter needs to be initialized first using -Yn,
# see avrdude manual.
#AVRDUDE_ERASE_COUNTER = -y

# Uncomment the following if you do /not/ wish a verification to be
# performed after programming the device.
#AVRDUDE_NO_VERIFY = -V

# Increase verbosity level.  Please use this when submitting bug
# reports about avrdude. See <http://savannah.nongnu.org/projects/avrdude> 
# to submit bug reports.
#AVRDUDE_VERBOSE = -v -v

AVRDUDE_BASIC = -p $(MCU) -P $(AVRDUDE_PORT) -c $(AVRDUDE_PROGRAMMER)
AVRDUDE_FLAGS = $(AVRDUDE_BASIC) $(AVRDUDE_NO_VERIFY) $(AVRDUDE_VERBOSE) $(AVRDUDE_ERASE_COUNTER)

AVRDUDE = avrdude
REMOVE = rm -f
MV = mv -f


# Define all listing files.
#LST = $(ASRC:.asm=.lst)

# Combine all necessary flags and optional flags.
# Add target processor to flags.
ALL_ASFLAGS =  $(ASFLAGS)

.PHONY:	all hex eep lst map program flash eeprom tags clean

# Default target.
all: hex lst

hex: $(TARGET).hex $(ASRC)
eep: $(TARGET).eep $(ASRC)
lst: $(TARGET).lst $(ASRC)
map: $(TARGET).map $(ASRC)


# Program the device.  
program: $(TARGET).hex $(TARGET).eep
	$(AVRDUDE) $(AVRDUDE_FLAGS) $(AVRDUDE_WRITE_FLASH) $(AVRDUDE_WRITE_EEPROM)

flash: $(TARGET).hex
	$(AVRDUDE) $(AVRDUDE_FLAGS) $(AVRDUDE_WRITE_FLASH)

eeprom: $(TARGET).hex $(TARGET).eep
	$(AVRDUDE) $(AVRDUDE_FLAGS) $(AVRDUDE_WRITE_EEPROM)


$(TARGET).hex: $(ASRC)
$(TARGET).eep: $(ASRC)
$(TARGET).lst: $(ASRC)
$(TARGET).map: $(ASRC)

#.SUFFIXES: .hex .eep .lst
.SUFFIXES:

%.hex: %.asm
	$(AS) $(ALL_ASFLAGS) -fI -o $@ $<

%.lst: %.asm
	@$(AS) $(ALL_ASFLAGS) -v0 -f- -l $@ $<

%.map: %.asm
	$(AS) $(ALL_ASFLAGS) -v0 -f- -m $@ $<

tags: $(SRC) $(ASRC)
	ctags $(SRC) $(ASRC)

svnrev.inc: $(ASRC0)
	@svnrev -osvnrev.inc $^

# Target: clean project.
clean:
	$(REMOVE) $(TARGET).hex $(TARGET).eep $(TARGET).obj $(TARGET).map $(TARGET).lst

