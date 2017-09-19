avrdude -c usbtiny -p m328p -e -U flash:w:avrcpm.hex
avrdude -c usbtiny -p m328p -U lfuse:w:0xf7:m -U hfuse:w:0xdf:m
