;
; Example program for the Pico Z80
; Blink a LED for a bit
;
; Udo Munk, April 2024
;


	ORG	0000H		; starting at memory location 0

LED	EQU	0		; LED port

	DI			; disable interrupts
	LD	SP,STACK	; setup stack in upper memory

	LD	B,20		; blink LED 20 times
L1:	LD	A,1		; switch LED on
	OUT	(LED),A
	LD	HL,0		; wait a bit
L2:	DEC	HL
	LD	A,H
	OR	L
	JR	NZ,L2
	LD	A,0		; switch LED off
	OUT	(LED),A
	LD	HL,0		; wait a bit
L3:	DEC	HL
	LD	A,H
	OR	L
	JR	NZ,L3
	DJNZ	L1		; again if not done
	HALT			; done, halt CPU

	DS	20		; space for the stack
STACK:

	END			; of program
