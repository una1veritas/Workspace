rem z80asm ipl.asm -o ipl.bin -lipl.lst
rem z80asm bios.asm -o bios.bin -lbios.lst
tniasm ipl.asm ipl.bin
tniasm bios.asm bios.bin
