asw -L cbios88.asm
p2bin cbios88.p cbios_code.bin -segment code
p2bin cbios88.p cbios_data.bin -segment data
powershell ./cbios.ps1
