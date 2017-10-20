	ORG 0000h
init:
	LD A, 0
	LD HL, $aa55
loop_entry:
	LD (result),A
	INC A
	OUT ($aa),A
	JP loop_entry
	

result:
	db 0