avrdude -c usbtiny -p m88 -e -U flash:w:z80.hex
avrdude -c usbtiny -p m88 -U lfuse:w:0xf7:m -U hfuse:w:0xdf:m
