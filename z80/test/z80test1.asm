	ORG $0000
start:
	LD HL,$0
loop:
	LD A,(HL)
	INC HL
	JR loop

	HALT
var_l:	
	db $0
var_h:
	db $0
	