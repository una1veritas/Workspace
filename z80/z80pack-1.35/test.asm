	ORG 0000h
init:
	LD A,6
	LD (var_x),A
loop:
	LD A,(var_x)
	INC A
	OUT ($02),A
	LD (var_x),A
	IN A,($01)
	LD (var_y),A
	AND A
	JP NZ,loop
stop:
	HALT
var_x:
	db 0
var_y:
	db 0

	END
	