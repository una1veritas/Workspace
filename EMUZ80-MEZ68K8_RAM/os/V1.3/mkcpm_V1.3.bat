powershell .\mkcpm_V1.3.ps1
mot2bin.exe aaa.s aaa.bin
Bin2Mot.exe -L5600 -I0 aaa.bin aaa.s
Mot2Bin.exe aaa.s CPM68K_V1.3.BIN
copy CPM68K_V1.3.BIN ..\..\DISKS\CPMDISKS\CPM68K.BIN
del aaa.*
