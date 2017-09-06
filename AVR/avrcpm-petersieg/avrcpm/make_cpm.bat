dd conv=sync bs=128  count=1 if=ipl.bin > cpm.bin
dd conv=sync bs=128 count=44 if=CPM.SYS >> cpm.bin
dd conv=sync bs=128  count=6 if=bios.bin >> cpm.bin
