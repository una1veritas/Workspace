ASM = /Users/sin/bin/asm6809


all: assist09-orig.s19 assist09-6850.s19

assist09-orig.s19: assist09-orig.asm
#	as9 assist09-orig.asm -l s19 now
	asm6809 -e -s19 -l assist09-orig.lst -o assist09-orig.s19 assist09-orig.asm

assist09-6522.s19: assist09-6522.asm
	as9 assist09-6522.asm -l s19 now

assist09-6850.s19: assist09-6850.asm
#	as9 assist09-6850.asm -l s19 now
	asm6809 -e -s19 -l assist09-6850.lst -o assist09-6850.s19 assist09-6850.asm

clean:
	$(RM) *.s19 *.lst

distclean: clean
