#! /bin/sh

FN=$1

wine avrasm2.exe -S labels.tmp -fI -W+ie -o $FN.hex -d $FN.obj \
	-e $FN.eep -m $FN.map -l $FN.lst $FN.asm 2> /dev/null
