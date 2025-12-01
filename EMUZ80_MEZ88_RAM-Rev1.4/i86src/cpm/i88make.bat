asw -L cbios88.asm
p2bin cbios88.p cbios_code.bin -segment code
p2bin cbios88.p cbios_data.bin -segment data
powershell ./cbios.ps1
copy cbios.bin ..\cpm_bin\CBIOS.BIN
copy ccp_bdos.bin ..\cpm_bin\CCP_BDOS.BIN
copy cbios.bin ..\..\DISKS\CPMDISKS\CBIOS.BIN
copy ccp_bdos.bin ..\..\DISKS\CPMDISKS\CCP_BDOS.BIN
