DEVICE = atmega328p
AVRDUDE_DEVICE = m328p
DEVICE ?= atmega168
AVRDUDE_DEVICE ?= m168

CFLAGS=-g -Wall -mcall-prologues -mmcu=$(DEVICE) -Os -I../../../
CPP=avr-g++
CC=avr-gcc
OBJ2HEX=avr-objcopy 
LDFLAGS=-Wl,-gc-sections -lpololu_$(DEVICE) -Wl,-relax

#PORT ?= /dev/ttyUSB0
PORT = usb
AVRDUDE=avrdude
TARGET=test

all: $(TARGET).hex

clean:
	rm -f *.o *.hex *.obj *.hex

%.hex: %.obj
	$(OBJ2HEX) -R .eeprom -O ihex $< $@

%.obj: %.o
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

program: $(TARGET).hex
	$(AVRDUDE) -p $(AVRDUDE_DEVICE) -c tinyisp -P $(PORT) -U flash:w:$(TARGET).hex
