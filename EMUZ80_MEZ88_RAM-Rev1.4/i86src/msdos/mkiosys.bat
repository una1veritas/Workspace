asl -L MEZ88IO.ASM
p2bin MEZ88IO.p
powershell .\mkiosys.ps1
copy MSDOS.SYS ..\msdos_bin\.
copy IO.SYS ..\msdos_bin\.
copy MSDOS.SYS ..\..\DISKS\DOSDISKS\.
copy IO.SYS ..\..\DISKS\DOSDISKS\.
