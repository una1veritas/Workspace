	org $0000
boot:	
	ld SP, $1000
	jp $1000

	ds $32

	org $0038
rst38h:	
	pop HL
	jr $0000

