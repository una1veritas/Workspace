TOOLDIR = /Applications/Arduino.app/Contents/Resources/Java/hardware/tools/avr
ASMINCDIR = ../include/AVR000

DEVICE = atmega328p
CLOCK      = 16000000

PORT = /dev/cu.usbserial-A6008d1U
PROGRAMMER = -cstk500v1
PROG_UNO = -carduino -b115200
PROG_DUE = -carduino -b57600
FUSES      = -U hfuse:w:0xdb:m -U lfuse:w:0xff:m 

AVRDUDE = $(TOOLDIR)/bin/avrdude -C $(TOOLDIR)/etc/avrdude.conf -p $(DEVICE) $(PROGRAMMER) -P $(PORT) -b 57600 -D

#$ARDUINO_AVR_DIR/bin/avrdude -C$ARDUINO_AVR_DIR/etc/avrdude.conf -patmega328p -cstk500v1 -P/dev/cu.usbserial-A6008d1U -b 57600 -D -U flash:w:main.hex:i

ASM = avra -fI -I $(ASMINCDIR)

#
SOURCES = main.asm

#
#
#

all:	main.hex

flash:	all
	$(AVRDUDE) -U flash:w:main.hex:i

# file targets:
main.hex: $(SOURCES)
	rm -f *.hex
	$(ASM) $(SOURCES)

winupload.due.%:	all
	avrdude $(PROG_DUE) -P$* -p $(DEVICE) -U flash:w:main.hex:i

winupload.uno.%:	all
	avrdude $(PROG_UNO) -P$* -p $(DEVICE) -U flash:w:main.hex:i

macupload.due.%:	all
	avrdude $(PROG_DUE) -P/dev/cu.usbserial-$* -p $(DEVICE) -U flash:w:main.hex:i

macupload.uno.%:	all
	avrdude $(PROG_UNO) -P/dev/cu.usbmodem$* -p $(DEVICE) -U flash:w:main.hex:i


clean:
	rm -f *.o *.hex *.cof
