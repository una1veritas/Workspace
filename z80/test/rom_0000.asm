
boot:	org $0000
	ld SP, $1000
	jp $1000

	ds $32
rst38h:	org $0038
	pop HL
	jr $0000

