	page 0
	CPU	8086

	title	'Customized Basic I/O System'

;*********************************************
;*                                           *
;* This Customized BIOS adapts CP/M-86 to    *
;* the following hardware configuration      *
;*     Processor:                            *
;*     Brand:                                *
;*     Controller:                           *
;*                                           *
;*                                           *
;*     Programmer: Akihito Honda             *
;*     Revisions : 1.0                       *
;*     Date : 2023.12.221                    *
;*            2025.08.10                     *
;*               modified for MEZ88_RAM      *
;*                                           *
;*********************************************

cr		equ 0dh ;carriage return
lf		equ 0ah ;line feed

UNIMON = 1
UNI_SEG		equ	07f50h
UNI_OFF		equ	07f30h	; stask 7f300-7f3ff, work 7f400-7f4ff, code 7f500

bdos_int	equ	224 ;reserved BDOS interrupt

;---------------------------------------------
;|                                           |
bios_code	equ 2500h
ccp_offset	equ 0000h
bdos_ofst	equ 0B06h ;BDOS entry point
;|                                           |
;---------------------------------------------

	ASSUME	CS:CODE, DS:DATA, SS:DATA, ES:NOTHING

	SEGMENT	CODE
;	cseg
	org	ccp_offset
ccp:
	org	bios_code

;*********************************************
;*                                           *
;* BIOS Jump Vector for Individual Routines  *
;*                                           *
;*********************************************

	jmp	INIT		;Enter from BOOT ROM or LOADER
	jmp	WBOOT		;Arrive here from BDOS call 0  
	jmp	CONST		;return console keyboard status
	jmp	CONIN		;return console keyboard char
	jmp	CONOUT  	;write char to console device
	jmp	LISTOUT		;write character to list device
	jmp	PUNCH		;write character to punch device
	jmp	READER  	;return char from reader device 
	jmp	HOME		;move to trk 00 on cur sel drive
	jmp	SELDSK  	;select disk for next rd/write
	jmp	SETTRK  	;set track for next rd/write
	jmp	SETSEC  	;set sector for next rd/write
	jmp	SETDMA  	;set offset for user buff (DMA)
	jmp	READ		;read a 128 byte sector
	jmp	WRITE		;write a 128 byte sector
	jmp	LISTST  	;return list status 
	jmp	SECTRAN 	;xlate logical->physical sector 
	jmp	SETDMAB 	;set seg base for buff (DMA)
	jmp	GETSEGT 	;return offset of Mem Desc Table
	jmp	GETIOBF		;return I/O map byte (IOBYTE)
	jmp	SETIOBF		;set I/O map byte (IOBYTE) 

	; for make near jump table
n_jmp	equ	20
	db	128 - n_jmp*3 dup(90h)	; nop

;*********************************************
;*                                           *
;* INIT Entry Point, Differs for LDBIOS and  *
;* BIOS, according to "Loader_Bios" value    *
;*                                           *
;*********************************************

INIT:	;print signon message and initialize hardware
	mov	ax,cs		;we entered with a JMPF so use
	mov	ss,ax		;CS: as the initial value of SS:,
	mov	ds,ax		;DS:,
	mov	es,ax		;and ES:
	;use local stack during initialization
	mov	sp,stkbase
	cld			;set forward direction

;---------------------------------------------
;|                                           |
	; This is a BIOS for the CPM.SYS file.
	; Setup all interrupt vectors in low
	; memory to address trap

	push	ds		;save the DS register
	mov	[IOBYTE],0	;clear IOBYTE
	mov	ax,0
	mov	ds,ax
	mov	es,ax 		;set ES and DS to zero

	if UNIMON = 0
	;setup interrupt 0 to address trap routine
	mov	[int0_offset],int_trap
	mov	[int0_segment],CS
	mov	di,4
	mov	si,0		;then propagate
	mov	cx,510		;trap vector to

;	rep movs ax,ax	;all 256 interrupts
	rep movsw		;all 256 interrupts
	endif

	;BDOS offset to proper interrupt
	mov	[bdos_offset],bdos_ofst
	mov	[bdos_segment],CS
	pop	ds		;restore the DS register

;	(additional CP/M-86 initialization)
;|                                           |
;---------------------------------------------

	mov	bx,signon
	call	pmsg		;print signon message
	mov	cl,0		;default to dr A: on coldstart
	jmp	ccp		;jump to cold start entry of CCP

WBOOT:	jmp	ccp+6		;direct entry to CCP at command level

;---------------------------------------------
;|                                           |
int_trap:
	cli			;block interrupts
	mov	ax,cs
	mov	ds,ax		;get our data segment
	mov	bx,int_trp
	call	pmsg
	hlt			;hardstop

;*********************************************
;*                                           *
;*   CP/M Character I/O Interface Routines   *
;*                                           *
;*********************************************

;  ---- request command to PIC
; UREQ_COM = 1 ; CONIN  : return char in UNI_CHR
;          = 2 ; CONOUT : UNI_CHR = output char
;          = 3 ; CONST  : return status in UNI_CHR
;                       : ( 0: no key, 1 : key exist )
;          = 4 ; STROUT : string address = (PTRSAV, PTRSAV_SEG)
;          = 5 ; DISK READ
;          = 6 ; DISK WRITE
;          = 0 ; request is done( return this flag from PIC )
;                return status is in UNI_CHR;
;
REQ_CONIN	equ	1
REQ_CONOUT	equ	2
REQ_CONST	equ	3
REQ_STROUT	equ	4
REQ_DREAD	equ	5
REQ_DWRITE	equ	6
DUMMY_PORT	equ	0f000h

; ------------------ Request Command 
UREQ_COM	equ	0	; location 00	:CONIN/CONOUT request command
; ------------------ CONSOLE I/O
UNI_CHR		equ	1	; location 01	:charcter (CONIN/CONOUT) or counts(STROUT) or CONST
STR_OFF		equ	2	; location 02	:string offset
STR_SEG		equ	4	; location 04	:string segment
; ------------------ CBIOS Command
CREQ_COM	equ	6	; location 06	:CBIOS request command
; ------------------ CBIOS CONSOLE I/O
CBI_CHR		equ	7	; location 07	:charcter (CONIN/CONOUT) or counts(STROUT) or status
disk_drive	equ	8	; location 06	:
disk_track	equ	9	; location 07	:
disk_sector	equ	10	; location 08	:
data_dmal	equ	12	; location 0A	:
data_dmah	equ	14	; location 0C	:

CONST:
	push	es
	push	dx
	mov	dx, UNI_SEG
	mov	es, dx
	mov	byte ptr es:[CREQ_COM], REQ_CONST	; set CONST request
	jmp	CON_REQ

CONIN:
	push	es
	push	dx
	mov	dx, UNI_SEG
	mov	es, dx
	mov	byte ptr es:[CREQ_COM], REQ_CONIN	; set CONIN request
CON_REQ:
	mov	dx, DUMMY_PORT
	in	al, dx				; invoke PIC F/W
	pop	dx
wait_conin:
	mov	al, es:[CREQ_COM]
	or	al, al
	jnz	wait_conin

	mov	al, es:[CBI_CHR]		; get char or status
	and	al, al
	pop	es
	ret

CONOUT:
	push	es
	push	dx
	mov	dx, UNI_SEG
	mov	es, dx
	mov	byte ptr es:[CREQ_COM], REQ_CONOUT	; set CONOUT request
	mov	es:[CBI_CHR], cl			; set char
	jmp	CON_REQ


LISTOUT:		;list device output
	xor	al, al
	ret

LISTST:			;poll list status
	xor	al, al
	ret

PUNCH:		;write punch device
	xor	al, al
	ret

READER:
	xor	al, al
	ret

GETIOBF:
	mov	al,[IOBYTE]
	ret

SETIOBF:
	mov	[IOBYTE],cl	;set iobyte
return:
	ret			;iobyte not implemented

pmsg:
	push	es
	push	cx
	push	dx
	push	bx

	mov	dx, UNI_SEG
	mov	es, dx

	xor	cl, cl		; clear count
msg_lp:
	mov	al, [bx]
	or	al, al
	jz	msg_cend
	inc	cl
	inc	bx
	jmp	msg_lp

msg_cend:
	pop	bx
	mov	byte ptr es:[CREQ_COM], REQ_STROUT	; set command
	mov	es:[CBI_CHR], cl			; set string bytyes
	mov	es:[data_dmal], bx			; set string offset
	mov	es:[data_dmah], ds			; set string segment

	mov	dx, DUMMY_PORT
	in	al, dx					; invoke PIC F/W

wait_stout:
	mov	al, es:[CREQ_COM]
	or	al, al
	jnz	wait_stout
	pop	dx
	pop	cx
	pop	es
	ret

;*********************************************
;*                                           *
;*          Disk Input/Output Routines       *
;*                                           *
;*********************************************

; input DI : address
; CF = 0 : byte
; CF = 1 : word

set_param:
	push	es
	push	cx
	mov	cx, UNI_SEG
	mov	es, cx
	jc	wrt_w
	mov	es:[di], al
wrt_rt:
	pop	cx
	pop	es
	ret
	
wrt_w:
	mov	es:[di], ax
	jmp	wrt_rt

SELDSK:	;select disk given by register CL
;	mov	[disk],cl	;save disk number
	mov	al,cl		;save disk number

;	cmp	al, 4
	cmp	al, 2
	jc	SELFD		; FD: Drive A and B

	mov	bx,HDB1		;dph harddisk 1
;	cmp	al, 8
	cmp	al, 2
	jz	SELHD		; Drive C

	mov	bx,HDB2		;dph harddisk 2
;	cmp	al, 9
	cmp	al, 3		; Drive D
	jz	SELHD

	mov	bx,0000h	;ready for error return
	ret
SELFD:
	mov	ch,0		;double(n)
	mov	bx,cx		;bx = n
	mov	cl,4		;ready for *16
	shl	bx,cl		;n = n * 16
	mov	cx,dpbase
	add	bx,cx		;dpbase + n * 16
SELHD:
	push	di
	mov	di, disk_drive
	or	al, al		; clear CARRY
	call	set_param
	pop	di
	ret
	

HOME:	;move selected disk to home position (Track 0)
	xor	al, al
	jmp	set_home

SETTRK: ;set track address given by CX
	mov	al, cl

set_home:
	push	di
	mov	di, disk_track
	or	al, al		; clear CARRY
	call	set_param
	pop	di
	ret

SETSEC: ;set sector number given by cx
	mov	ax, cx
	push	di
	mov	di, disk_sector
	stc
	call	set_param
	or	ax, ax
	pop	di
	ret

SECTRAN: ;translate sector CX using table at [DX]
	or	dx, dx
	jz	no_skew
	mov	bx,cx
	add	bx,dx		;add sector to tran table address
	mov	bl,[bx]		;get logical sector
	mov	bh,0
	ret

no_skew:
	mov	bx, cx
	inc	bx		; ;sector no. start with 1
	ret

SETDMA: ;set DMA offset given by CX
	mov	ax, cx
	push	di
	mov	di, data_dmal
	stc
	call	set_param
	or	ax, ax
	pop	di
	ret

SETDMAB: ;set DMA segment given by CX
	mov	ax, cx
	push	di
	mov	di, data_dmah
	stc
	call	set_param
	or	ax, ax
	pop	di
	ret
;
GETSEGT:  ;return address of physical memory table
	mov	bx,segtable
	ret

;*********************************************
;*                                           *
;*  All disk I/O parameters are setup:       *
;*     DISK     is disk number      (SELDSK) *
;*     TRK      is track number     (SETTRK) *
;*     SECT     is sector number    (SETSEC) *
;*     DMA_ADR  is the DMA offset   (SETDMA) *
;*     DMA_SEG  is the DMA segment  (SETDMAB)*
;*  READ reads the selected sector to the DMA*
;*  address, and WRITE writes the data from  *
;*  the DMA address to the selected sector   *
;*  (return 00 if successful,  01 if perm err)*
;*                                           *
;*********************************************

READ:
	push	es
	push	dx
	mov	ax, UNI_SEG
	mov	es, ax

	mov	byte ptr es:[CREQ_COM], REQ_DREAD	; set disk read request

READ_W:
	mov	dx, DUMMY_PORT
	in	al, dx					; invoke PIC F/W
wait_dskrd:
	mov	al, es:[CREQ_COM]
	or	al, al
	jnz	wait_dskrd

	mov	al, es:[CBI_CHR]		; get status
	and	al, al
	jz	rw_ok

	;error
	mov	al, 1
	or	al, al
rw_ok:
	pop	dx
	pop	es
	ret

WRITE:
	push	es
	push	dx
	mov	ax, UNI_SEG
	mov	es, ax
	mov	byte ptr es:[CREQ_COM], REQ_DWRITE	; set disk read request
	jmp	READ_W

;*********************************************
;*                                           *
;*               Data Areas                  *
;*                                           *
;*********************************************
data_offset	equ $

	SEGMENT	DATA
	org	data_offset	;contiguous with code segment

IOBYTE	db	0

;---------------------------------------------
;|                                           |
signon	db	cr,lf
	db	"CP/M-86 BIOS V2.0 Generated!",cr,lf
	db	"MEZ88_RAM edition."
	db	cr,lf,0
;|                                           |
;---------------------------------------------

int_trp	db	cr,lf
	db	'Interrupt Trap Halt'
	db	cr,lf

;	System Memory Segment Table

segtable	db	1	;
	dw tpa_seg	;1st seg starts after BIOS
	dw tpa_len	;and extends

;	include singles.lib ;read in disk definitions

;---------- 4 DISKS --------------------
dpbase	equ	$		;Base of Disk Parameter Blocks
dpe0	dw	xlt0,0000h	;Translate Table
	dw	0000h,0000h	;Scratch Area
	dw	dirbuf,dpb0	;Dir Buff, Parm Block
	dw	csv0,alv0	;Check, Alloc Vectors

dpe1	dw	xlt1,0000h	;Translate Table
	dw	0000h,0000h	;Scratch Area
	dw	dirbuf,dpb1	;Dir Buff, Parm Block
	dw	csv1,alv1	;Check, Alloc Vectors

dpe2	dw	xlt2,0000h	;Translate Table
	dw	0000h,0000h	;Scratch Area
	dw	dirbuf,dpb2	;Dir Buff, Parm Block
	dw	csv2,alv2	;Check, Alloc Vectors

dpe3	dw	xlt3,0000h	;Translate Table
	dw	0000h,0000h	;Scratch Area
	dw	dirbuf,dpb3	;Dir Buff, Parm Block
	dw	csv3,alv3	;Check, Alloc Vectors

;	        DISKDEF 0,1,26,6,1024,243,64,64,2
;
;	 1944:	128 Byte Record Capacity
;	  243:	Kilobyte Drive  Capacity
;	   64:	32 Byte Directory Entries
;	   64:	Checked Directory Entries
;	  128:	Records / Extent
;	    8:	Records / Block
;	   26:	Sectors / Track
;	    2:	Reserved  Tracks
;	    6:	Sector Skew Factor
;
dpb0	equ	$		;Disk Parameter Block
	dw	26		;Sectors Per Track
	db	3		;Block Shift
	db	7		;Block Mask
	db	0		;Extnt Mask
	dw	242		;Disk Size - 1
	dw	63		;Directory Max
	db	192		;Alloc0
	db	0		;Alloc1
	dw	16		;Check Size
	dw	2		;Offset

xlt0	equ	$		;Translate Table
	db	1,7,13,19
	db	25,5,11,17
	db	23,3,9,15
	db	21,2,8,14
	db	20,26,6,12
	db	18,24,4,10
	db	16,22
als0	equ	31		;Allocation Vector Size
css0	equ	16		;Check Vector Size

;	        DISKDEF 1,0
;
;	Disk 1 - 3  are the same as Disk 0
;
dpb1	equ	dpb0		;Equivalent Parameters
dpb2	equ	dpb0		;Equivalent Parameters
dpb3	equ	dpb0		;Equivalent Parameters
als1	equ	als0		;Same Allocation Vector Size
als2	equ	als0		;Same Allocation Vector Size
als3	equ	als0		;Same Allocation Vector Size
css1	equ	css0		;Same Checksum Vector Size
css2	equ	css0		;Same Checksum Vector Size
css3	equ	css0		;Same Checksum Vector Size
xlt1	equ	xlt0		;Same Translate Table
xlt2	equ	xlt0		;Same Translate Table
xlt3	equ	xlt0		;Same Translate Table
;	        ENDEF
;
;	fixed data tables for 4MB harddisks
;
;	disk parameter header
HDB1:	DW	0000H,0000H
	DW	0000H,0000H
	DW	dirbuf,HDBLK
	DW	CHKHD1,ALLHD1
HDB2:	DW	0000H,0000H
	DW	0000H,0000H
	DW	dirbuf,HDBLK
	DW	CHKHD2,ALLHD2
;
;       disk parameter block for harddisk
;
;HDBLK:	DW	32		;SEC PER TRACK
;	DB	4		;BLOCK SHIFT
;	DB	15		;BLOCK MASK
;	DB	0		;EXTNT MASK
;	DW	2047		;DISK SIZE-1
;	DW	255		;DIRECTORY MAX
;	DB	240		;ALLOC0
;	DB	0		;ALLOC1
;	DW	0		;CHECK SIZE
;	DW	0		;OFFSET

HDBLK:  DW    128		;sectors per track
	DB    4			;block shift factor
	DB    15		;block mask
	DB    0			;extent mask
	DW    2039		;disk size-1
	DW    1023		;directory max
	DB    255		;alloc 0
	DB    255		;alloc 1
	DW    0			;check size
	DW    0			;track offset

alshd1	equ	255		;Allocation Vector Size
;alshd1	equ	32		;Allocation Vector Size
csshd1	equ	0		;Check Vector Size
alshd2	equ	alshd1		;Allocation Vector Size
csshd2	equ	csshd1		;Check Vector Size

;
;	Uninitialized Scratch Memory Follows:
;
begdat	equ	$		;Start of Scratch Area
dirbuf	ds	128		;Directory Buffer
alv0	ds	als0		;Alloc Vector
csv0	ds	css0		;Check Vector
alv1	ds	als1		;Alloc Vector
csv1	ds	css1		;Check Vector
alv2	ds	als2		;Alloc Vector
csv2	ds	css2		;Check Vector
alv3	ds	als3		;Alloc Vector
csv3	ds	css3		;Check Vector
ALLHD1:	DS	alshd1		;allocation vector harddisk 1
ALLHD2:	DS	alshd2		;allocation vector harddisk 2
CHKHD1:	equ	$		;check vector harddisk 1 (0)
CHKHD2:	equ	$		;check vector harddisk 2 (0)
enddat	equ	$		;End of Scratch Area

datsiz	equ	enddat - begdat	;Size of Scratch Area

	db	0		;Marks End of Module

loc_stk	dw  32 dup(?)		;local stack for initialization

stkbase	equ	$
lastoff	equ	$
	db 0	;fill last address for GENCMD

tpa_seg	equ (lastoff+0400h+15) / 16

	if UNIMON = 1
tpa_len	equ UNI_OFF - tpa_seg
	else
tpa_len	equ 8000h - tpa_seg
	endif

;*********************************************
;*                                           *
;*          Dummy Data Section               *
;*                                           *
;*********************************************
	SEGMENT	DATA
	org 	0	;(interrupt vectors)

int0_offset	dw	?
int0_segment	dw	?
;	pad to system call vector
	ds	4*(bdos_int-1)

bdos_offset	dw	?
bdos_segment	dw	?
	END
