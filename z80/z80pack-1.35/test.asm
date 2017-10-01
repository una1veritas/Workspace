	ORG 0000h
init:
	LD A, $0e
	LD (tmp),A
loop:
	LD A,(tmp)
	out ($01),A
	JP loop
stop:
	HALT

tmp:
	db $0E
const_mess:
	dm "Hello.",13,0
	
	END
