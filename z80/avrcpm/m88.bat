avrdude -c usbtiny -p m88 -e -U flash:w:M88-38400-8N1-30MHz-z80.hex
avrdude -c usbtiny -p m88 -U lfuse:w:0xf7:m -U hfuse:w:0xdf:m
