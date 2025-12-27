	.z180
;	CP/M 2.2 boot-loader for Z80-Simulator
;
;	Copyright (C) 1988-2007 by Udo Munk
;
	ORG	0		;mem base of boot
;
MSIZE	EQU	64		;mem size in kbytes
;
BIAS	EQU	(MSIZE-20)*1024	;offset from 20k system
CCP	EQU	3400H+BIAS	;base of the ccp
BIOS	EQU	CCP+1600H	;base of the bios
BIOSL	EQU	0300H		;length of the bios
BOOT	EQU	BIOS
SIZE	EQU	BIOS+BIOSL-CCP	;size of cp/m system
SECTS	EQU	SIZE/128	;# of sectors to load
;
	.include "supermez80_asm.inc"
DRIVE   EQU	FDCD
TRACK   EQU	FDCT
SECTOR  EQU	FDCS
;
	JP	COLD
;
ERRMSG:	DEFB	"BOOT: error booting"
	DEFB	13,10,0
;
;	begin the load operation
;
COLD:	LD	BC,2		;b=track 0, c=sector 2
	LD	D,SECTS		;d=# sectors to load
	LD	HL,CCP		;base transfer address
	XOR	A		;select drive A
	OUT	(DRIVE),A
;
;	load the next sector
;
LSECT:	LD	A,B		;set track
	OUT	(TRACK),A
	LD	A,C		;set sector
	OUT	(SECTOR),A
	LD	E,128		;128 bytes per sector

	if BYTE_RW

	LD	A,D_READ	;read sector (non-DMA)
	OUT	(FDCOP),A
	IN	A,(FDCST)	;get status of fdc
	OR	A		;read successful ?
	JP	NZ,RD_ERR	;jump, if error
;
;	read data to the destination address
;
LBYTE:
	IN	A,(FDCDAT)
	LD	(HL),A
	INC	HL
	DEC	E
	JP	NZ,LBYTE
				;go to next sector if load is incomplete
	else

	LD	A,H
	OUT	(DMAH),A
	LD	A,L
	OUT	(DMAL),A
	LD	A,D_DMA_READ	;read sector (DMA)
	OUT	(FDCOP),A
	IN	A,(FDCST)	;get status of fdc
	OR	A		;read successful ?
	JP	NZ,RD_ERR	;jump, if error

	; calc next DMA address
	LD	A,D		; save sectors
	LD	D,0
	ADD	HL,DE		; HL=HL+128
	LD	D,A		; restore sectors
	endif

	DEC	D		;sects=sects-1
	JP	Z,BOOT		;head for the bios
;
;	more sectors to load
;
	INC	C		;sector = sector + 1
	LD	A,C
	CP	27		;last sector of track ?
	JP	C,LSECT		;no, go read another
;
;	end of track, increment to next track
;
	LD	C,1		;sector = 1
	INC	B		;track = track + 1
	JP	LSECT		;for another group
;
RD_ERR:
	LD	HL,ERRMSG	;no, print error

	if UART_180
STROUT:
	ld	a,(hl)
	or	a
	jp	z,STOP
	ld	e, a
ST1:
	IN0	A,(UARTCR_180)
	BIT	1,A
	JP	Z,ST1
	ld	a, e
	OUT0	(UART_TX),A
	inc	hl
	JP	STROUT
	endif

	if	UART_PIC
STROUT:	LD	A,(HL)
	OR	A
	JP	Z,STOP
	OUT	(CONDAT),A
	INC	HL
	JP	STROUT
	endif

STOP:	DI
	HALT			;and halt cpu

	END			;of boot loader
