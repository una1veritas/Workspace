avrdude -c usbtiny -p m168 -e -U flash:w:avrcpm.hex
avrdude -c usbtiny -p m168 -U lfuse:w:0xf7:m -U hfuse:w:0xdf:m
