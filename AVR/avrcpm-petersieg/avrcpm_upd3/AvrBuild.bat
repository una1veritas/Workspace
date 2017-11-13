@ECHO OFF
"C:\Programme\Atmel\AVR Tools\AvrAssembler2\avrasm2.exe" -S "C:\DATEN\AVR CPM\avr\labels.tmp" -fI -W+ie -o "C:\DATEN\AVR CPM\avr\avrcpm.hex" -d "C:\DATEN\AVR CPM\avr\avrcpm.obj" -e "C:\DATEN\AVR CPM\avr\avrcpm.eep" -m "C:\DATEN\AVR CPM\avr\avrcpm.map" "C:\DATEN\AVR CPM\avr\avrcpm.asm"
