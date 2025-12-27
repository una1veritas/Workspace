#!/usr/bin/expect -f

proc line {} {
    send_user "=========================\n"
}

proc show { msg time } {
    set tmp [lindex [split $time " +"] 0 ]
    send_user "$msg [ format %7.3f [expr $tmp / 1000000.0 ] ]\n"
}

source common.tcl
open_cpm3

set boot_time [ time {
expect "CP/M boot-loader for Z80-Simulator"
expect "LDRBIOS3 V1.2 for Z80SIM, Copyright 1989-2007 by Udo Munk"
expect "CP/M V3.0 Loader"
expect "Copyright (C) 1998, Caldera Inc."
expect "BANKED BIOS3 V1.6-HD, Copyright 1989-2015 by Udo Munk"
expect "A>vt100dyn"
expect "(C) Alexandre MONTARON - 2015 - VT100DYN"
expect "RSX loaded and initialized."
expect "A>DEVICE CONSOLE"
expect "A>"
} ]

send_user "\n\n"
line
show "    BOOT TIME: " $boot_time
line

send "\r"
expect "A>"
send "J:\r"
expect "J>"

send "\r"
expect "J>"
send "ERA HELLO.COM\r"
expect "J>"
send "C -V HELLO.C\r"
set timeout 300
set compile_time [ time {
expect "J>"
} ]

set timeout $default_timeout
send "J:HELLO\r"
expect "Hello, world!"
expect "J>"

send_user "\n\n"
line
show " COMPILE TIME: " $compile_time
line
send_user "\n"

send "\r"
expect "J>"
send "MBASIC\r"
expect "Ok"
send "LOAD \"ASCIIART.BAS\"\r"
expect "Ok"
send "RUN\r"
set timeout 300
set asciiart_time [ time {
expect "Ok"
} ]

set timeout $default_timeout
send "SYSTEM\r"
expect "J>"

send_user "\n\n"
show "ASCIIART TIME: " $asciiart_time
send_user "\n"

line
show "     BOOT TIME: " $boot_time
show "  COMPILE TIME: " $compile_time
show " ASCIIART TIME: " $asciiart_time
line
send_user "\n"
