	ORG 0000h
init:
	LD A,(count)
loop:
	DEC A
	LD (count),A
	JP NZ,loop
stop:
	HALT
count:	
	db 5
	
	END
