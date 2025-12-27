#!/usr/bin/expect -f

source common.tcl
open_cpm2

send -break
expect "MON>"

send "status\r"
expect "MON>"

send "cont\r\r"
expect "A>"

send "dir reset.com\r"
expect "A>"

send "sdir\r"
expect "RESET    COM"
sleep 1

send -break
expect "MON>"
send "status\r"
expect "MON>"
send "cont\r\r"
expect "A>"

send "\r"
expect "A>"

send -break
expect "MON>"
send "write 1000h,'Hello, world! \\r\\n'\r"
expect "MON>"
send "dump 1000h,16\r"
expect "001000: 48 65 6c 6c 6f 2c 20 77 6f 72 6c 64 21 20 0d 0a Hello, world! .."
expect "MON>"

send "step\r"
expect "MON>"
send "step\r"
expect "MON>"
send "help\r"
expect "    Write byte or string to the target memory"
expect "MON>"
sleep 0.1

send "write 1000h,3Eh\r"
expect "MON>"
send "write ,'H'\r"
expect "MON>"
send "write ,D3h\r"
expect "MON>"
send "write ,01h\r"
expect "MON>"
sleep 0.1

send "write ,3Eh\r"
expect "MON>"
send "write ,'i'\r"
expect "MON>"
send "write ,D3h\r"
expect "MON>"
send "write ,01h\r"
expect "MON>"
sleep 0.1

send "write ,3Eh\r"
expect "MON>"
send "write ,'!'\r"
expect "MON>"
send "write ,D3h\r"
expect "MON>"
send "write ,01h\r"
expect "MON>"
sleep 0.1

send "write ,3Eh\r"
expect "MON>"
send "write ,'\\r'\r"
expect "MON>"
send "write ,D3h\r"
expect "MON>"
send "write ,01h\r"
expect "MON>"
sleep 0.1

send "write ,00h\r"
expect "MON>"
send "write ,3Eh\r"
expect "MON>"
send "write ,'\\n'\r"
expect "MON>"
send "write ,D3h\r"
expect "MON>"
send "write ,01h\r"
expect "MON>"
sleep 0.1

send "write ,76h\r"
expect "MON>"
sleep 0.1

send "dump 1000h,10h\r"
expect "001000: 3e 48 d3 01 3e 69 d3 01 3e 21 d3 01 3e 0d d3 01 >H..>i..>!..>..."
expect "MON>"
sleep 0.1

send "disas 1000h,16h\r"
expect "001015 76          HALT"
expect "MON>"
sleep 0.1

send "jump 1000h\r"
expect "Hi!"
sleep 0.1
send -break
expect "MON>"
sleep 0.1

send "breakpoint 1010h\r"
expect "MON>"
send "breakpoint\r"
expect "Breakpoint is 1010"
expect "MON>"
sleep 0.1

send "jump 1000h\r"
expect "Break at 1010"
expect "MON>"
sleep 0.1

send "step\r"
expect "001011 3E 0A       LD A,0AH"
expect "MON>"
sleep 0.1

send "step\r"
expect "001013 D3 01       OUT (01H),A"
expect "MON>"
sleep 0.1

send "step\r"
expect "001015 76          HALT"
expect "MON>"
sleep 0.1

puts "\n\nOK\n"
