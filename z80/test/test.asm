COUT 	.equ	$02
	
	org $0000
init:
	ld hl, buff
	ld sp, hl
	jp progstart

	defs $31,0
	org $0038
rst38h:
	ret

	defs $c7, 0
	org $0100
buff:
	
	org $1000
progstart:
	ld hl, str_msg
	call printmsg

progend:
	halt

printmsg:
	ld a,(hl)
	and a
	ret z
	out (COUT),a
	inc hl
	jr printmsg
	
str_msg:
	db "Hi, there!",13,10,0
	