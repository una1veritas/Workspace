@ECHO OFF
"C:\Program Files\Atmel\AVR Tools\AvrAssembler2\avrasm2.exe" -S "E:\Development\AVR\Selfprog\labels.tmp" -fI -W+ie -C V2 -o "E:\Development\AVR\Selfprog\Selfprog.hex" -d "E:\Development\AVR\Selfprog\Selfprog.obj" -e "E:\Development\AVR\Selfprog\Selfprog.eep" -m "E:\Development\AVR\Selfprog\Selfprog.map" "E:\Development\AVR\Selfprog\Selfprog.asm"
