#!/usr/bin/expect -f

source common.tcl
open_cpm2

# create test file
send "pip hello.txt=CON:\r"
sleep 2
expect "\r"
send "Hello, world!\r\x1a"
expect "A>"
sleep 1

# check test file contents
send "type hello.txt\r"
expect "Hello, world!\r"
expect "A>"
sleep 1

send "dump hello.txt\r"
expect "0000 48 65 6C 6C 6F 2C 20 77 6F 72 6C 64 21 0D 1A 1A"
expect "0010 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "0020 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "0030 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "0040 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "0050 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "0060 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "0070 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A 1A"
expect "A>"

send "sdir hello.txt\r"
expect "HELLO    TXT     1k      1 Dir RW      "
expect "A>"

# update test file
send "pip hello.txt=CON:\r"
sleep 2
expect "\r"
send "Good bye, world!\r\x1a"
expect "A>"

# check test file contents
send "type hello.txt\r"
expect "Good bye, world!\r"
expect "A>"

# delete test file
send "era hello.txt\r"
expect "A>"
send "sdir hello.txt\r"
expect "File Not Found."
expect "A>"

puts "\n\nOK\n"
