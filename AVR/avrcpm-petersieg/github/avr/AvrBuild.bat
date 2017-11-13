@ECHO OFF
"C:\misc\avrcpm\avr\avrasm2.exe" -S "C:\misc\avrcpm\avr\labels.tmp" -fI -W+ie -o "C:\misc\avrcpm\avr\avrcpm.hex" -d "C:\misc\avrcpm\avr\avrcpm.obj" -e "C:\misc\avrcpm\avr\avrcpm.eep" -m "C:\misc\avrcpm\avr\avrcpm.map" "C:\misc\avrcpm\avr\avrcpm.asm"
