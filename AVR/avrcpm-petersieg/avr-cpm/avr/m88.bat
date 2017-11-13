avrdude -c usbtiny -p atmega88 -U lfuse:w:0xf7:m -U hfuse:w:0xdf:m -U flash:w:z80.hex

