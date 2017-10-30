
start:  org $1000

	ld hl, str_msg
loop_entry:
	ld a,(hl)
	ld (var_a), a
	and a
	jr z, stop
	out ($01), a
	inc hl
	jr NZ, loop_entry

stop:
	halt

var_a:	
	db $0

str_msg:
	dm "Hello, friends!",13,10,0
	
	end
	