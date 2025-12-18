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
