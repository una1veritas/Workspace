	ORG $0000
start:	
	LD HL,welcome_msg
	call print_msg

echo_nack_init:
	
echo_back_loop:
echo_back_ifempty:	
	IN A,($00)
	CP $FF
	JR NZ,echo_back_ifempty
	IN A,($01)
	LD B,A
	CP $1A
	JR Z,stop
	OUT ($02),A
	LD A,'('
	OUT ($02),A
	LD A,B
	CALL print_hex_loop
	LD A,')'
	OUT ($02),A
	JR echo_back_loop
stop:	
	HALT

print_hex:
	LD B,A
print_hex_loop:
	SRL A
	SRL A
	SRL A
	SRL A
	LD HL,nibble2hex_tbl
	ADD L
	LD L,A
	LD A,(HL)
	OUT ($02),A
	LD A,B
	AND $0f
	LD HL,nibble2hex_tbl
	ADD L
	LD L,A
	LD A,(HL)
	OUT ($02),A
	RET
nibble2hex_tbl:
	dm '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'
	
print_msg:
	LD C,$02
print_msg_loop:
	LD A,(HL)
	AND A
	RET Z
	OUTI
	JR print_msg_loop

welcome_msg:
	dm "Hi, I'm Z80!",10,13,0
tmp:
	db $17
