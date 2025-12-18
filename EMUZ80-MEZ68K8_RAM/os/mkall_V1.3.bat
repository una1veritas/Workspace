msdos mcc68k m68cbios.c m68kbios.src null.lst
del null.lst
sed "s/\tSECTION/*\tSECTION/g" m68kbios.src > aaa
sed "s/\tXREF/*\tXREF/g" aaa > bbb
sed "s/\tOPT/*\tOPT/g" bbb > m68kbios.src
del aaa
del bbb
copy *.src m68kbios.asm
del m68kbios.src
msdos asm68k -l -o m68kbios.obj m68kbios.asm > m68kbios.lst
msdos lnk68k -m -o m68kbios.s m68kbios.obj > m68kbios.map
sed "/^S5/d" m68kbios.s > aaa
del m68kbios.s
ren aaa m68kbios.s
asw -L unimon.asm
p2hex -s +5 unimon.p unimon.s
sed -n "1,14p" unimon.s > vector.s
sed "1,14d" unimon.s > uni_body.s
powershell .\mkcbios.ps1
Mot2Bin.exe cbios.s aaa.bin
Bin2Mot.exe -L1D00 -I6200 aaa.bin aaa.s
Mot2Bin.exe aaa.s CBIOS_V1.3.BIN
del aaa.bin
copy CBIOS_V1.3.BIN ..\DISKS\CPMDISKS\CBIOS.BIN
copy vector.s V1.3\.
del *.s
powershell .\mkall_V1.3.ps1
mot2bin.exe aaa.s aaa.bin
Bin2Mot.exe -L5600 -I0 aaa.bin aaa.s
Mot2Bin.exe aaa.s CPM68K_V1.3.BIN
copy CPM68K_V1.3.BIN ..\DISKS\CPMDISKS\CPM68K.BIN
copy CPM68K_V1.3.BIN V1.3\.
del aaa.*
del CPM68K_V1.3.BIN
copy ..\DISKS\DRIVE_V1.3\*.DSK ..\DISKS\CPMDISKS\.
