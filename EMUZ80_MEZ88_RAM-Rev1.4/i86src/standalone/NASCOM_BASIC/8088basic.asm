;	8088 NASCOM BASIC
;		Converted source code from 8080/Z80 to 8086
;	Assembler: Macro Assembler 1.42
;
;
;	SBCV20 NASCOM BASIC
;	Assembler: asm86.com/asm86.cmd
;

	CPU	8086
	ASSUME	CS:CODE, DS:DATA, SS:DATA, ES:DATA

IP_OFF	EQU	0h	; program top
CS_SEG	equ	40h

EMROM	EQU	200h	; Start address in emulation mode

RAMTOP	EQU	TB_WORK ; BASIC Work space
WRKSPC	EQU	RAMTOP+45H  ; BASIC Work space
;
;	dseg
	SEGMENT	DATA
	ORG	0400h-20
VGETCH	DS	4	; CALLN 251
VKBHIT	DS	4	; CALLN 252
VPUTCH	DS	4	; CALLN 253
VBRKEM	DS	4	; BRKEM 254
INTREQ	DS	4	; External interrupt

	SEGMENT	CODE
	ORG	IP_OFF
;
;	Start
start:
	xor	ax, ax
	mov	ds, ax
	mov	ax, CS_SEG	; Set com model
	mov	es, ax		; Overlay ds with cs
	mov	ss, ax		; Overlay ss with cs
	mov	sp, SYSSTK	; Set stack
;
;	Vector 251-255 setup

	mov	WORD PTR [VGETCH], ngetch
	mov	[VGETCH+2], ax
	mov	WORD PTR [VKBHIT], nkbhit
	mov	[VKBHIT+2], ax
	mov	WORD PTR [VPUTCH], nputch
	mov	[VPUTCH+2], ax
	mov	WORD PTR [VBRKEM], EMROM
	mov	[VBRKEM+2], ax
	mov	ds, ax

	; save initial data to work space
	mov	si, INITAB
	mov	di, WRKSPC
	MOV	cx,INITBE-INITAB+3	; Bytes to copy
	cld
	rep movsb

	JMP	PROG_CODE

;
; return to universal monitor
;
ret_mon:
	mov	ah, 0
	int	21h		; goto monitor

; get a charactor : AH = 7
; return AL : charactor
getch:

	mov	ah, 7
	int	21h
	mov	ah, 0
	ret

; put a charactor : AH = 2
; al -> console
putch:
	push	dx
	mov	ah, 2
	mov	dl, al
	int	21h		; system call
	pop	dx
	ret
;
;	put string
puts:
	cld			; set DF for SI increment
ptst1:
	lodsb			; get data to AL and SI++
	cmp	al,00h		; check tail
	jz	ptext		; if tail, return
	call	putch		; display a charactor
	jmp	ptst1		; loop until tail
ptext:
	ret

;	CALLN wrapping
ngetch:
	call	getch		; Get a char
	iret

nkbhit:
	call	CHKCHR
	iret

nputch:
	call	putch		; Put a char
	iret

;;
;;	Returned native mode
;	mov	si,offset natv	;8088 message
;	call	puts		;Out it
;	jmps	$		;Stop
;;


;       MS-BASIC START UP ROUTINE
;       TARGET: SBC8080
;       ASSEMBLER: ARCPIT XZ80.EXE
;
	ORG	EMROM
;
;       START BASIC
PROG_CODE:
	JMP	COLD

; check key status : AH = 0BH
; OUTPUT : AL : 0     ( key is not exist )
;             : 0FFH  ( key is exist )
CHKCHR:
	mov	ah, 0bh
	int	21h
	mov	ah, 0
	and	al, 1
	RET

;
;==================================================================================
; The updates to the original BASIC within this file are copyright Grant Searle
;
; You have permission to use this for NON COMMERCIAL USE ONLY
; If you wish to use it elsewhere, please include an acknowledgement to myself.
;
; http://searle.hostei.com/grant/index.html
;
; eMail: home.micros01@btinternet.com
;
; If the above don't work, please perform an Internet search to see if I have
; updated the web page hosting service.
;
;==================================================================================
;
; NASCOM ROM BASIC Ver 4.7, (C) 1978 Microsoft
; Scanned from source published in 80-BUS NEWS from Vol 2, Issue 3
; (May-June 1983) to Vol 3, Issue 3 (May-June 1984)
; Adapted for the freeware Zilog Macro Assembler 2.10 to produce
; the original ROM code (checksum A934H). PA
;
; GENERAL EQUATES
;
CTRLC		EQU	03H	; Control "C"
CTRLG		EQU	07H	; Control "G"
BKSP		EQU	08H	; Back space
LF		EQU	0AH	; Line feed
CLRSCRN		EQU	0CH	; Clear screen
CR		EQU	0DH	; Carriage return
CTRLO		EQU	0FH	; Control "O"
CTRLQ		EQU	11H	; Control "Q"
CTRLR		EQU	12H	; Control "R"
CTRLS		EQU	13H	; Control "S"
CTRLU		EQU	15H	; Control "U"
ESC		EQU	1BH	; Escape
DEL		EQU	7FH	; Delete
;
; BASIC WORK SPACE LOCATIONS
;
BUFFER		EQU	WRKSPC+61H  +7; Input buffer
STACK		EQU	WRKSPC+66H  +7; Initial stack
CURPOS		EQU	WRKSPC+0ABH +7; Character position on line
LCRFLG		EQU	WRKSPC+0ACH +7; Locate/Create flag
TYPE		EQU	WRKSPC+0ADH +7; Data type flag
DATFLG		EQU	WRKSPC+0AEH +7; Literal statement flag
LSTRAM		EQU	WRKSPC+0AFH +7; Last available RAM
TMSTPT		EQU	WRKSPC+0B1H +7; Temporary string pointer
TMSTPL		EQU	WRKSPC+0B3H +7; Temporary string pool
TMPSTR		EQU	WRKSPC+0BFH +7; Temporary string
STRBOT		EQU	WRKSPC+0C3H +7; Bottom of string space
CUROPR		EQU	WRKSPC+0C5H +7; Current operator in EVAL
LOOPST		EQU	WRKSPC+0C7H +7; First statement of loop
DATLIN		EQU	WRKSPC+0C9H +7; Line of current DATA item
FORFLG		EQU	WRKSPC+0CBH +7; "FOR" loop flag
LSTBIN		EQU	WRKSPC+0CCH +7; Last byte entered
READFG		EQU	WRKSPC+0CDH +7; Read/Input flag
BRKLIN		EQU	WRKSPC+0CEH +7; Line of break
NXTOPR		EQU	WRKSPC+0D0H +7; Next operator in EVAL
ERRLIN		EQU	WRKSPC+0D2H +7; Line of error
CONTAD		EQU	WRKSPC+0D4H +7; Where to CONTinue
PROGND		EQU	WRKSPC+0D6H +7; End of program
VAREND		EQU	WRKSPC+0D8H +7; End of variables
ARREND		EQU	WRKSPC+0DAH +7; End of arrays
NXTDAT		EQU	WRKSPC+0DCH +7; Next data item
FNRGNM		EQU	WRKSPC+0DEH +7; Name of FN argument
FNARG		EQU	WRKSPC+0E0H +7; FN argument value
FPREG		EQU	WRKSPC+0E4H +7; Floating point register
FPEXP		EQU	FPREG+3       ; Floating point exponent
SGNRES		EQU	WRKSPC+0E8H +7; Sign of result
PBUFF		EQU	WRKSPC+0E9H +7; Number print buffer
MULVAL		EQU	WRKSPC+0F6H +7; Multiplier
PROGST		EQU	WRKSPC+0F9H +7; Start of program text area
STLOOK		EQU	WRKSPC+15DH +7; Start of memory test
;
; BASIC ERROR CODE VALUES
;
NF	EQU	00H	; NEXT without FOR
SN	EQU	02H	; Syntax error
RG	EQU	04H	; RETURN without GOSUB
OD	EQU	06H	; Out of DATA
FC	EQU	08H	; Function call error
OV	EQU	0AH	; Overflow
OM	EQU	0CH	; Out of memory
UL	EQU	0EH	; Undefined line number
BS	EQU	10H	; Bad subscript
RD	EQU	12H	; Re-DIMensioned array
DZ	EQU	14H	; Division by zero (/0)
ID	EQU	16H	; Illegal direct
TM	EQU	18H	; Type miss-match
OS	EQU	1AH	; Out of string space
LS	EQU	1CH	; String too long
ST	EQU	1EH	; String formula too complex
CN	EQU	20H	; Can't CONTinue
UF	EQU	22H	; UnDEFined FN function
MO	EQU	24H	; Missing operand
HX	EQU	26H	; HEX error
BN	EQU	28H	; BIN error
;
COLD:
	JMP	STARTB			; Jump for cold start jump
WARM:
	JMP	WARMST			; Jump for warm start jump
STARTB:
	JMP	CSTART			; Jump to initialise
;
	DW	DEINT			; Get integer -32768 to 32767
	DW	ABPASS			; Return integer in AB
;
CSTART:
	MOV	BX,WRKSPC		; Start of workspace RAM
	MOV	SP,BX			; Set up a temporary stack
	JMP	INITST			; Go to initialise
;
INIT:
	; restore initial data from work space
	mov	di, INITAB
	mov	si, WRKSPC
	MOV	cx,INITBE-INITAB+3	; Bytes to copy
	cld
	rep movsb

	MOV	SP,BX			; Temporary stack
	CALL	CLREG			; Clear registers and stack
	CALL	PRCRLF			; Output CRLF
	MOV	[BUFFER+72+1],AL	; Mark end of buffe
	MOV	[PROGST],AL  		; Initialise program area
MSIZE:
	MOV	BX,STLOOK		; Point to start of RAM
MLOOP:
	LAHF
	INC	BX			; Next byte
	SAHF
	MOV	AL,BH			; Above address FFFF ?
	OR	AL,BL
	JZ	SETTOP			; Yes - 64K RAM
	MOV	AL,[BX]			; Get contents
	MOV	CH,AL			; Save it
	NOT	AL			; Flip all bits
	MOV	[BX],AL			; Put it back
	CMP	AL,[BX]			; RAM there if same
	MOV	[BX],CH			; Restore old contents
	JZ	MLOOP			; If RAM - test next byte
;
SETTOP:
	LAHF
	DEC	BX			; Back one byte
	SAHF
	MOV	DX,STLOOK-1		; See if enough RAM
	CALL	CPDEHL			; Compare DE with HL
	JC	NEMEM			; If not enough RAM
	MOV	DX,0-50			; 50 Bytes string space
	MOV	[LSTRAM],BX		; Save last available RAM
	ADD	BX,DX			; Allocate string space
	MOV	[STRSPC],BX		; Save string space
	CALL	CLRPTR			; Clear program area
	MOV	BX,[STRSPC]		; Get end of memory
	MOV	DX,0-17			; Offset for free bytes
	ADD	BX,DX			; Adjust HL
	MOV	DX,PROGST		; Start of program text
	MOV	AL,BL			; Get LSB
	SUB	AL,DL			; Adjust it
	MOV	BL,AL			; Re-save
	MOV	AL,BH			; Get MSB
	SBB	AL,DH			; Adjust it
	MOV	BH,AL			; Re-save
	PUSH	BX			; Save bytes free
	MOV	BX,SIGNON		; Sign-on message
	CALL	PRS			; Output string
	POP	BX			; Get bytes free back
	CALL	PRNTHL			; Output amount of free memory
	MOV	BX,BFREE		; " Bytes free" message
	CALL	PRS			; Output string
WARMST:
	MOV	SP,STACK		; Temporary stack
BRKRET:
	CALL	CLREG			; Clear registers and s
	JMP	PRNTOK			; Go to get command lin
;
NEMEM:
	MOV	BX,MEMMSG		; Memory size not enough
	CALL	PRS			; Print it
XXXXX:
	JMP	XXXXX			; Stop
;
BFREE:
	DB	" Bytes free",CR,LF,0,0
;
SIGNON:
	DB	"INTEL8080 Based x86 BASIC Ver 4.7b",CR,LF
	DB	"Copyright ",40,"C",41
	DB	" 1978 by Microsoft",CR,LF,0,0
;
MEMMSG:
	DB	"Memory size not enough",CR,LF
	DB	"The system is stopped.",CR,LF,0,0
;
; FUNCTION ADDRESS TABLE
;
FNCTAB:
	DW	SGN
	DW	INT
	DW	ABS
	DW	USR
	DW	FRE
	DW	INP
	DW	POS
	DW	SQR
	DW	RND
	DW	LOG
	DW	EXP
	DW	COS
	DW	SIN
	DW	TAN
	DW	ATN
	DW	PEEK
	DW	DEEK
	DW	POINT
	DW	LEN
	DW	STR
	DW	VAL
	DW	ASC
	DW	CHR
	DW	HEX
	DW	BIN
	DW	LEFT
	DW	RIGHT
	DW	MID
;
; RESERVED WORD LIST
;
WORDS:
	DB	0C5H,"ND"
	DB	0C6H,"OR"
	DB	0CEH,"EXT"
	DB	0C4H,"ATA"
	DB	0C9H,"NPUT"
	DB	0C4H,"IM"
	DB	0D2H,"EAD"
	DB	0CCH,"ET"
	DB	0C7H,"OTO"
	DB	0D2H,"UN"
	DB	0C9H,"F"
	DB	0D2H,"ESTORE"

	DB	0C7H,"OSUB"
	DB	0D2H,"ETURN"
	DB	0D2H,"EM"
	DB	0D3H,"TOP"
	DB	0CFH,"UT"
	DB	0CFH,"N"
	DB	0CEH,"ULL"
	DB	0D7H,"AIT"
	DB	0C4H,"EF"
	DB	0D0H,"OKE"
	DB	0C4H,"OKE"
	DB	0D3H,"CREEN"
	DB	0CCH,"INES"
	DB	0C3H,"LS"
	DB	0D7H,"IDTH"
	DB	0CDH,"ONITOR"

	DB	0D3H,"ET"
	DB	0D2H,"ESET"
	DB	0D0H,"RINT"
	DB	0C3H,"ONT"
	DB	0CCH,"IST"
	DB	0C3H,"LEAR"
	DB	0C3H,"LOAD"
	DB	0C3H,"SAVE"
	DB	0CEH,"EW"
;
	DB	0D4H,"AB("
	DB	0D4H,"O"
	DB	0C6H,"N"
	DB	0D3H,"PC("
	DB	0D4H,"HEN"
	DB	0CEH,"OT"
	DB	0D3H,"TEP"
;
	DB	0ABH
	DB	0ADH
	DB	0AAH
	DB	0AFH
	DB	0DEH
	DB	0C1H,"ND"
	DB	0CFH,"R"
	DB	0BEH
	DB	0BDH
	DB	0BCH
;
	DB	0D3H,"GN"
	DB	0C9H,"NT"
	DB	0C1H,"BS"
	DB	0D5H,"SR"
	DB	0C6H,"RE"
	DB	0C9H,"NP"
	DB	0D0H,"OS"
	DB	0D3H,"QR"
	DB	0D2H,"ND"
	DB	0CCH,"OG"
	DB	0C5H,"XP"
	DB	0C3H,"OS"
	DB	0D3H,"IN"
	DB	0D4H,"AN"
	DB	0C1H,"TN"
	DB	0D0H,"EEK"
	DB	0C4H,"EEK"
	DB	0D0H,"OINT"
	DB	0CCH,"EN"
	DB	0D3H,"TR$"
	DB	0D6H,"AL"
	DB	0C1H,"SC"
	DB	0C3H,"HR$"
	DB	0C8H,"EX$"
	DB	0C2H,"IN$"
	DB	0CCH,"EFT$"
	DB	0D2H,"IGHT$"
	DB	0CDH,"ID$"
	DB	80H         ; End of list marker
;
; KEYWORD ADDRESS TABLE
;
WORDTB:
	DW	PEND
	DW	FOR
	DW	NEXT
	DW	DATA
	DW	INPUT
	DW	DIM
	DW	READ
	DW	LET
	DW	GOTO
	DW	RUN
	DW	IF
	DW	RESTOR
	DW	GOSUB
	DW	RETURN
	DW	REM
	DW	STOP
	DW	POUT
	DW	ON
	DW	NULL
	DW	WAIT
	DW	DEF
	DW	POKE
	DW	DOKE
	DW	REM
	DW	LINES
	DW	CLS
	DW	WIDTH
	DW	MONITR
	DW	PSET
	DW	RESET
	DW	PRINT
	DW	CONT
	DW	LIST
	DW	CLEAR
	DW	REM
	DW	REM
	DW	NEW
;
; RESERVED WORD TOKEN VALUES
;
ZEND       EQU      080H        ; END
ZFOR       EQU      081H        ; FOR
ZDATA      EQU      083H        ; DATA
ZGOTO      EQU      088H        ; GOTO
ZGOSUB     EQU      08CH        ; GOSUB
ZREM       EQU      08EH        ; REM
ZPRINT     EQU      09EH        ; PRINT
ZNEW       EQU      0A4H        ; NEW
;
ZTAB       EQU      0A5H        ; TAB
ZTO        EQU      0A6H        ; TO
ZFN        EQU      0A7H        ; FN
ZSPC       EQU      0A8H        ; SPC
ZTHEN      EQU      0A9H        ; THEN
ZNOT       EQU      0AAH        ; NOT
ZSTEP      EQU      0ABH        ; STEP
;
ZPLUS      EQU      0ACH        ; +
ZMINUS     EQU      0ADH        ; -
ZTIMES     EQU      0AEH        ; *
ZDIV       EQU      0AFH        ; /
ZOR        EQU      0B2H        ; OR
ZGTR       EQU      0B3H        ; >
ZEQUAL     EQU      0B4H        ; M
ZLTH       EQU      0B5H        ; <
ZSGN       EQU      0B6H        ; SGN
ZPOINT     EQU      0C7H        ; POINT
ZLEFT      EQU      0CDH +2     ; LEFT$
;
; ARITHMETIC PRECEDENCE TABLE
;
PRITAB:
	DB	79H         ; Precedence value
	DW	PADD        ; FPREG = <last> + FPREG
;
	DB	79H         ; Precedence value
	DW	PSUB        ; FPREG = <last> - FPREG
;
	DB	7CH         ; Precedence value
	DW	MULT        ; PPREG = <last> * FPREG
;
	DB	7CH         ; Precedence value
	DW	DIV         ; FPREG = <last> / FPREG
;
	DB	7FH         ; Precedence value
	DW	POWER       ; FPREG = <last> ^ FPREG
;
	DB	50H         ; Precedence value
	DW	PAND        ; FPREG = <last> AND FPREG
;
	DB	46H         ; Precedence value
	DW	POR         ; FPREG = <last> OR FPREG
;
; BASIC ERROR CODE LIST
;
ERRORS:
	DB	"NF"        ; NEXT without FOR
	DB	"SN"        ; Syntax error
	DB	"RG"        ; RETURN without GOSUB
	DB	"OD"        ; Out of DATA
	DB	"FC"        ; Illegal function call
	DB	"OV"        ; Overflow error
	DB	"OM"        ; Out of memory
	DB	"UL"        ; Undefined line
	DB	"BS"        ; Bad subscript
	DB	"DD"        ; Re-DIMensioned array
	DB	"/0"        ; Division by zero
	DB	"ID"        ; Illegal direct
	DB	"TM"        ; Type mis-match
	DB	"OS"        ; Out of string space
	DB	"LS"        ; String too long
	DB	"ST"        ; String formula too co
	DB	"CN"        ; Can't CONTinue
	DB	"UF"        ; Undefined FN function
	DB	"MO"        ; Missing operand
	DB	"HX"        ; HEX error
	DB	"BN"        ; BIN error
;
; INITIALISATION TABLE --------------------------------
;
INITAB:
	JMP	WARMST			; Warm start jump
USR:	JMP	FCERR			; "USR (X)" jump (Set to Error)
OUTSUB:	OUT	0,AL			; "OUT p,n" skeleton
	RET
OTPORT	equ	OUTSUB+1

DIVSUP:	SUB	AL,0			; Division support routine
DIV1	equ	DIVSUP+1

	MOV	BL,AL
	MOV	AL,BH

div2_op:
DIV2	equ	div2_op+1
	SBB	AL,0
	MOV	BH,AL
	MOV	AL,CH

div3_op:
DIV3	equ	div3_op+1
	SBB	AL,0
	MOV	CH,AL

div4_op:
DIV4	equ	div4_op+1
	MOV	AL,0
	RET

SEED:	DB	0,0,0			; Random number seed ta
	DB	035H,04AH,0CAH,099H	;-2.65145E+07
	DB	039H,01CH,076H,098H	; 1.61291E+07
	DB	022H,095H,0B3H,098H	;-1.17691E+07
	DB	00AH,0DDH,047H,098H	; 1.30983E+07
	DB	053H,0D1H,099H,099H	;-2-01612E+07
	DB	00AH,01AH,09FH,098H	;-1.04269E+07
	DB	065H,0BCH,0CDH,098H	;-1.34831E+07
	DB	0D6H,077H,03EH,098H	; 1.24825E+07
LSTRND:	DB	052H,0C7H,04FH,080H	; Last random n

INPSUB:	IN	AL,0			; INP (x) skeleton
INPORT	equ	INPSUB+1
	RET

NULLS:	DB	1			; POS (x) number (1)
LWIDTH:	DB	255			; Terminal width (255)
COMMAN:	DB	28			; Width for commas (3 colums)
NULFLG:	DB	0			; No nulls after input bytes
CTLOFG:	DB	0			; Output enabled (^O off)
LINESC:	DW	20			; Initial lines counter
LINESN:	DW	20			; Initial lines number
CHKSUM:	DW	0			; Array load/save check sum
NMIFLG:	DB	0			; Break not by NMI
BRKFLG:	DB	0			; Break flag
RINPUT:	JMP	TTYLIN			; Input reflection (set to TTY)
;POINT:	JMP	0000H			; POINT reflection unused
;PSET:	JMP	0000H			; SET reflection
;RESET:	JMP	0000H			; RESET reflection
POINT:	ret				; POINT reflection unused
	nop
	nop
PSET:	ret				; SET reflection unused
	nop
	nop
RESET:	ret				; RESET reflection unused
	nop
	nop
STRSPC:	DW	STLOOK			; Temp string space
LINEAT:	DW	-2			; Current line number (old)
BASTXT:	DW	PROGST+1		; Start of program text
INITBE:					; END OF INITIALISATION TABLE
;
; END OF INITIALISATION TABLE -------------------------
;
ERRMSG:
	DB	" Error",0

INMSG:
	DB	" in ",0
ZERBYT	EQU      $-1			; A zero byte
OKMSG:
	DB	"Ok",CR,LF,0,0
BRKMSG:
	DB	"Break",0
;
BAKSTK:
	MOV	BX,4			; Look for "FOR" block with
	ADD	BX,SP			; same index as specified
LOKFOR:
	MOV	AL,[BX]			; Get block ID
;	LAHF
	INC	BX			; Point to index address
;	SAHF
	CMP	AL,ZFOR			; Is it a "FOR" token
	JZ	LOKFOR1
	RET				; No - exit
LOKFOR1:
	MOV	CL,[BX]			; BC = Address of "FOR" index
;	LAHF
	INC	BX
;	SAHF
	MOV	CH,[BX]
;	LAHF
	INC	BX			; Point to sign of STEP
;	SAHF
	PUSH	BX			; Save pointer to sign
	MOV	BX,CX			; HL = address of "FOR" index
	MOV	AL,DH			; See if an index was specified
	OR	AL,DL			; DE = 0 if no index specified
	XCHG	BX,DX			; Specified index into HL
	JZ	INDFND			; Skip if no index given
	XCHG	BX,DX			; Index back into DE
	CALL	CPDEHL			; Compare index with one given
INDFND:
	MOV	CX,16-3			; Offset to next block
	POP	BX			; Restore pointer to sign
	JNZ	INDFND1
	RET				; Return if block found
INDFND1:
	ADD	BX,CX			; Point to next block
	JMP	LOKFOR			; Keep on looking
;
MOVUP:
	CALL	ENFMEM			; See if enough memory
MOVSTR:
	PUSH	CX			; Save end of source
	MOV	BP,SP
	XCHG	[BP],BX			; Swap source and dest" end
	POP	CX			; Get end of destination
MOVLP:
	CALL	CPDEHL			; See if list moved
	MOV	AL,[BX]			; Get byte
	XCHG	BX,CX
	MOV	[BX],AL			; Move it
	XCHG	BX,CX
	JNZ	MOVLP1
	RET				; Exit if all done
MOVLP1:
;	LAHF
	DEC	CX			; Next byte to move to
;	SAHF
;	LAHF
	DEC	BX			; Next byte to move
;	SAHF
	JMP	MOVLP			; Loop until all bytes moved
;
CHKSTK:
	PUSH	BX			; Save code string address
	MOV	BX,[ARREND]		; Lowest free memory
	MOV	CH,0			; BC = Number of levels to test
	ADD	BX,CX			; 2 Bytes for each level
	ADD	BX,CX
	JMP	ENFMEM1			; Skip "PUSH HL"
ENFMEM:
	PUSH	BX			; Save code string address
ENFMEM1:
	MOV	AL,0D0H			; LOW -48; 48 Bytes minimum RAM
	SUB	AL,BL
	MOV	BL,AL
	MOV	AL,0FFH			; HIGH (-48); 48 Bytes minimum RAM
	SBB	AL,BH
	JC	OMERR			; Not enough - ?OM Erro
	MOV	BH,AL
	ADD	BX,SP			; Test if stack is overflowed
	POP	BX			; Restore code string address
	JNC	OMERR
	RET				; Return if enough memory
OMERR:
	MOV	DL,OM			; ?OM Error
	JMP	ERROR
;
DATSNR:
	MOV	BX,[DATLIN]		; Get line of current DATA item
	MOV	[LINEAT],BX		; Save as current line
SNERR:
	MOV	DL,SN			; ?SN Error
	JMP	ERROR
DZERR:
	MOV	DL,DZ			; ?/0 Error
	JMP	ERROR
NFERR:
	MOV	DL,NF			; ?NF Error
	JMP	ERROR
DDERR:
	MOV	DL,RD			; ?DD Error
	JMP	ERROR
UFERR:
	MOV	DL,UF			; ?UF Error
	JMP	ERROR
OVERR:
	MOV	DL,OV			; ?OV Error
	JMP	ERROR
TMERR:
	MOV	DL,TM			; ?TM Error
;
ERROR:
	CALL	CLREG			; Clear registers and stack
	MOV	[CTLOFG],AL		; Enable output (A is 0)
	CALL	STTLIN			; Start new line
	MOV	BX,ERRORS		; Point to error codes
	MOV	DH,AL			; D = 0 (A is 0)
	MOV	AL,'?'
	CALL	OUTC			; Output '?'
	ADD	BX,DX			; Offset to correct error code
	MOV	AL,[BX]			; First character
	CALL	OUTC			; Output it
	CALL	GETCHR			; Get next character
	CALL	OUTC			; Output it
	MOV	BX,ERRMSG		; "Error" message
ERRIN:
	CALL	PRS			; Output message
	MOV	BX,[LINEAT]		; Get line of error
	MOV	DX,-2			; Cold start error if -2
	CALL	CPDEHL			; See if cold start error
	JNZ	ERRIN1			; Cold start error - Restart
	JMP	CSTART
ERRIN1:
	MOV	AL,BH			; Was it a direct error
	AND	AL,BL			; Line = -1 if direct error
	INC	AL
	JZ	PRNTOK
	CALL	LINEIN			; No - output line of error
	JMP	PRNTOK			; Skip "POP BC"
POPNOK:
	POP	CX			; Drop address in input buffer
;
PRNTOK:
	XOR	AL,AL			; Output "Ok" and get command
	MOV	[CTLOFG],AL		; Enable output
	CALL	STTLIN			; Start new line
	MOV	BX,OKMSG		; "Ok" message
	CALL	PRS			; Output "Ok"
GETCMD:
	MOV	BX,-1			; Flag direct mode
	MOV	[LINEAT],BX		; Save as current line
	CALL	GETLIN			; Get an input line
	JC	GETCMD			; Get line again if break
	CALL	GETCHR			; Get first character
	INC	AL			; Test if end of line
	DEC	AL			; Without affecting Carry
	JZ	GETCMD			; Nothing entered - Get another
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save Carry status
	XCHG	AH,AL
	CALL	ATOH			; Get line number into DE
	PUSH	DX			; Save line number
	CALL	CRUNCH			; Tokenise rest of line
	MOV	CH,AL			; Length of tokenised line
	POP	DX			; Restore line number
	POP	AX			; Restore Carry
	XCHG	AH,AL
	SAHF
	JC	GETCMD1
	JMP	EXCUTE			; No line number - Direct mode
GETCMD1:
	PUSH	DX			; Save line number
	PUSH	CX			; Save length of tokenised line
	XOR	AL,AL
	MOV	[LSTBIN],AL		; Clear last byte input
	CALL	GETCHR			; Get next character
	OR	AL,AL			; Set flags
	LAHF
	XCHG	AH,AL
	PUSH	AX			; And save them
	XCHG	AH,AL
	CALL	SRCHLN			; Search for line numbe
	JC	LINFND			; Jump if line found
	POP	AX			; Get status
	XCHG	AH,AL
	SAHF
	LAHF
	XCHG	AH,AL
	PUSH	AX			; And re-save
	XCHG	AH,AL
	JNZ	GETCMD2
	JMP	ULERR			; Nothing after number - Error
GETCMD2:
	OR	AL,AL			; Clear Carry
LINFND:
	PUSH	CX			; Save address of line in prog
	JNC	INEWLN			; Line not found - Inseer new
	XCHG	BX,DX			; Next line address in DE
	MOV	BX,[PROGND]		; End of program
SFTPRG:
	XCHG	BX,DX
	MOV	AL,[BX]			; Shift rest of program down
	XCHG	BX,DX
	XCHG	BX,CX
	MOV	[BX],AL
	XCHG	BX,CX
	LAHF
	INC	CX			; Next destination
	INC	DX			; Next source
	SAHF
	CALL	CPDEHL			; All done?
	JNZ	SFTPRG			; More to do
	MOV	BX,CX			; HL - New end of program
	MOV	[PROGND],BX		; Update end of program
;
INEWLN:
	POP	DX			; Get address of line,
	POP	AX			; Get status
	XCHG	AH,AL
	SAHF
	JZ	SETPTR			; No text - Set up pointers
	MOV	BX,[PROGND]		; Get end of program
	MOV	BP,SP
	XCHG	[BP],BX			; Get length of input line
	POP	CX			; End of program to BC
	ADD	BX,CX			; Find new end
	PUSH	BX			; Save new end
	CALL	MOVUP			; Make space for line
	POP	BX			; Restore new end
	MOV	[PROGND],BX		; Update end of program pointer
	XCHG	BX,DX			; Get line to move up in HL
	MOV	[BX],BH			; Save MSB
	POP	DX			; Get new line number
	LAHF
	INC	BX			; Skip pointer
	INC	BX
	MOV	[BX],DL			; Save LSB of line numb
	INC	BX
	MOV	[BX],DH			; Save MSB of line numb
	INC	BX			; To first byte in line
	SAHF
	MOV	DX,BUFFER		; Copy buffer to program
MOVBUF:
	XCHG	BX,DX
	MOV	AL,[BX]			; Get source
	XCHG	BX,DX
	MOV	[BX],AL			; Save destinations
	INC	BX			; Next source
	INC	DX			; Next destination
	OR	AL,AL			; Done?
	JNZ	MOVBUF			; No - Repeat
SETPTR:
	CALL	RUNFST			; Set line pointers
	LAHF
	INC	BX			; To LSB of pointer
	SAHF
	XCHG	BX,DX			; Address to DE
PTRLP:
	MOV	BX,DX			; Address to HL
	MOV	AL,[BX]			; Get LSB of pointer
	INC	BX			; To MSB of pointer
	OR	AL,[BX]			; Compare with MSB poiner
	JNZ	PTRLP1
	JMP	GETCMD			; Get command line if end
PTRLP1:
	INC	BX			; To LSB of line number
	INC	BX			; Skip line number
	INC	BX			; Point to first byte i
	XOR	AL,AL			; Looking for 00 byte
FNDEND:
	CMP	AL,[BX]			; Found end of line?
	LAHF
	INC	BX			; Move to next byte
	SAHF
	JNZ	FNDEND			; No - Keep looking
	XCHG	BX,DX			; Next line address to HL
	MOV	[BX],DL			; Save LSB of pointer
	INC	BX
	MOV	[BX],DH			; Save MSB of pointer
	JMP	PTRLP			; Do next line
;
SRCHLN:
	MOV	BX,[BASTXT]		; Start of program text
SRCHLP:
	MOV	CX,BX			; BC = Address to look at
	MOV	AL,[BX]			; Get address of next line
;	LAHF
	INC	BX
;	SAHF
	OR	AL,[BX]			; End of program found?
	LAHF
	DEC	BX
	SAHF
	JNZ	SRCHLP1
	RET				; Yes - Line not found
SRCHLP1:
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	AL,[BX]			; Get LSB of line number
;	LAHF
	INC	BX
;	SAHF
	MOV	BH,[BX]			; Get MSB of line number
	MOV	BL,AL
	CALL	CPDEHL			; Compare with line in DE
	MOV	BX,CX			; HL = Start of this line
	MOV	AL,[BX]			; Get LSB of next line address
	LAHF
	INC	BX
	SAHF
	MOV	BH,[BX]			; Get MSB of next line address
	MOV	BL,AL			; Next line to HL
	CMC
	JNZ	SRCHLP2
	RET				; Lines found - Exit
SRCHLP2:
	CMC
	JC	SRCHLP3
	RET				; Line not found,at line after
SRCHLP3:
	JMP	SRCHLP			; Keep looking
;
NEW:
	JZ	CLRPTR
	RET				; Return if any more on line
CLRPTR:
	MOV	BX,[BASTXT]		; Point to start of program
	XOR	AL,AL			; Set program area to empty
	MOV	[BX],AL			; Save LSB = 00
	LAHF
	INC	BX
	SAHF
	MOV	[BX],AL			; Save MSB = 00
	LAHF
	INC	BX
	SAHF
	MOV	[PROGND],BX		; Set program end
;
RUNFST:
	MOV	BX,[BASTXT]		; Clear all variables
	LAHF
	DEC	BX
	SAHF
;
INTVAR:
	MOV	[BRKLIN],BX		; Initialise RUN variables
	MOV	BX,[LSTRAM]		; Get end of RAM
	MOV	[STRBOT],BX		; Clear string space
	XOR	AL,AL
	CALL	RESTOR			; Reset DATA pointers
	MOV	BX,[PROGND]		; Get end of program
	MOV	[VAREND],BX		; Clear variables
	MOV	[ARREND],BX		; Clear arrays
;
CLREG:
	POP	CX			; Save return address
	MOV	BX,[STRSPC]		; Get end of working RAM
	MOV	SP,BX			; Set stack
	MOV	BX,TMSTPL		; Temporary string pool
	MOV	[TMSTPT],BX		; Reset temporary string ptr
	XOR	AL,AL			; A = 00
	MOV	BL,AL			; HL = 0000
	MOV	BH,AL
	MOV	[CONTAD],BX		; No CONTinue
	MOV	[FORFLG],AL		; Clear FOR flag
	MOV	[FNRGNM],BX		 ; Clear FN argument
	PUSH	BX			; HL = 0000
	PUSH	CX			; Put back return
DOAGN:
	MOV	BX,[BRKLIN]		; Get address of code t
	RET				; Return to execution d
;
PROMPT:
	MOV	AL,'?'			; '?'
	CALL	OUTC			; Output character
	MOV	AL,' '			; Space
	CALL	OUTC			; Output character
;	JMP	RINPUT			; Get input line
	JMP	GETLIN			; Get input line
;
CRUNCH:
	XOR	AL,AL			; Tokenise line @ HL to BUFFER
	MOV	[DATFLG],AL		; Reset literal flag
	MOV	CL,2+3			; 2 byte number and 3 nulls
	MOV	DX,BUFFER		; Start of input buffer
CRNCLP:
	MOV	AL,[BX]			; Get byte
	CMP	AL,' '			; Is it a space?
	JNZ	CRNCLP1			; Yes - Copy direct
	JMP	MOVDIR
CRNCLP1:
	MOV	CH,AL			; Save character
	CMP	AL,'"'			; Is it a quote?
	JNZ	CRNCLP2			; Yes - Copy literal string
	JMP	CPYLIT
CRNCLP2:
	OR	AL,AL			; Is it end of buffer?
	JNZ	CRNCLP3			; Yes - End buffer
	JMP	ENDBUF
CRNCLP3:
	MOV	AL,[DATFLG]		; Get data type
	OR	AL,AL			; Literal?
	MOV	AL,[BX]			; Get byte to copy
	JZ	CRNCLP4			; Literal - Copy direct
	JMP	MOVDIR
CRNCLP4:
	CMP	AL,'?'			; Is it '?' short for PRING
	MOV	AL,ZPRINT		; "PRINT" token
	JNZ	CRNCLP5			; Yes - replace it
	JMP	MOVDIR
CRNCLP5:
	MOV	AL,[BX]			; Get byte again
	CMP	AL,'0'			; Is it less than '0'
	JC	FNDWRD			; Yes - Look for reserved words
	CMP	AL,60			; ";"+1; Is it "0123456789:;" ?
	JNC	FNDWRD			; Yes - copy it direct
	JMP	MOVDIR
FNDWRD:
	PUSH	DX			; Look for reserved words
	MOV	DX,WORDS-1		; Point to table
	PUSH	CX			; Save count
	MOV	CX,RETNAD		; Where to return to
	PUSH	CX			; Save return address
	MOV	CH,ZEND-1		; First token value -1
	MOV	AL,[BX]			; Get byte
	CMP	AL,'a'			; Less than 'a' ?
	JC	SEARCH			; Yes - search for words
	CMP	AL,'z'+1		; Greater than 'z' ?
	JNC	SEARCH			; Yes - search for words
	AND	AL,01011111B		; Force upper case
	MOV	[BX],AL			; Replace byte
SEARCH:
	MOV	CL,[BX]			; Search for a word
	XCHG	BX,DX
GETNXT:
	LAHF
	INC	BX
	SAHF				; Get next reserved word
	OR	AL,[BX]			; Start of word?
	JNS	GETNXT			; No - move on
	INC	CH			; Increment token value
	MOV	AL,[BX]			; Get byte from table
	AND	AL,01111111B		; Strip bit 7
	JNZ	GETNXT1
	RET				; Return if end of list
GETNXT1:
	CMP	AL,CL			; Same character as in buffer?
	JNZ	GETNXT			; No - get next word
	XCHG	BX,DX
	PUSH	BX			; Save start of word
;
NXTBYT:
	LAHF
	INC	DX			; Look through rest of word
	SAHF
	XCHG	BX,DX
	MOV	AL,[BX]			; Get byte from table
	XCHG	BX,DX
	OR	AL,AL			; End of word ?
	JS	MATCH			; Yes - Match found
	MOV	CL,AL			; Save it
	MOV	AL,CH			; Get token value
	CMP	AL,ZGOTO		; Is it "GOTO" token ?
	JNZ	NOSPC			; No - Don't allow spaces
	CALL	GETCHR			; Get next character
	LAHF
	DEC	BX			; Cancel increment fromGETCHR
	SAHF
NOSPC:
;	LAHF
	INC	BX
;	SAHF				; Next byte
	MOV	AL,[BX]			; Get byte
	CMP	AL,'a'			; Less than 'a' ?
	JC	NOCHNG			; Yes - don't change
	AND	AL,01011111B		; Make upper case
NOCHNG:
	CMP	AL,CL			; Same as in buffer ?
	JZ	NXTBYT			; Yes - keep testing
	POP	BX			; Get back start of word
	JMP	SEARCH			; Look at next word
;
MATCH:
	MOV	CL,CH			; Word found - Save token value
	POP	AX			; Throw away return
	XCHG	AH,AL
	SAHF
	XCHG	BX,DX
	RET				; Return to "RETNAD"
RETNAD:
	XCHG	BX,DX			; Get address in string
	MOV	AL,CL			; Get token value
	POP	CX			; Restore buffer length
	POP	DX			; Get destination address
MOVDIR:
	LAHF
	INC	BX			; Next source in buffer
	SAHF
	XCHG	BX,DX
	MOV	[BX],AL			; Put byte in buffer
	XCHG	BX,DX
	LAHF
	INC	DX			; Move up buffer
	SAHF
	INC	CL			; Increment length of buffer
	SUB	AL,':'			; End of statement?
	JZ	SETLIT			; Jump if multi-stateme line
	CMP	AL,ZDATA-3AH		; Is it DATA statement ?
	JNZ	TSTREM			; No - see if REM
SETLIT:
	MOV	[DATFLG],AL		; Set literal flag
TSTREM:
	SUB	AL,ZREM-3AH		; Is it REM?
	JZ	TSTREM1			; No - Leave flag
	JMP	CRNCLP
TSTREM1:
	MOV	CH,AL			; Copy rest of buffer
NXTCHR:
	MOV	AL,[BX]			; Get byte
	OR	AL,AL			; End of line ?
	JZ	ENDBUF			; Yes - Terminate buffe
	CMP	AL,CH			; End of statement ?
	JZ	MOVDIR			; Yes - Get next one
CPYLIT:
	LAHF
	INC	BX			; Move up source string
	SAHF
	XCHG	BX,DX
	MOV	[BX],AL			; Save in destination
	XCHG	BX,DX
	INC	CL			; Increment length
	LAHF
	INC	DX			; Move up destination
	SAHF
	JMP	NXTCHR			; Repeat
;
ENDBUF:
	MOV	BX,BUFFER-1		; Point to start of buffer
	XCHG	BX,DX
	MOV	[BX],AL			; Mark end of buffer (A = 00)
	XCHG	BX,DX
;	LAHF
	INC	DX
;	SAHF
	XCHG	BX,DX
	MOV	[BX],AL			; A = 00
	XCHG	BX,DX
;	LAHF
	INC	DX
;	SAHF
	XCHG	BX,DX
	MOV	[BX],AL			; A = 00
	XCHG	BX,DX
	RET
;
DODEL:
	MOV	AL,[NULFLG]		; Get null flag status
	OR	AL,AL			; Is it zero?
	MOV	AL,0			; Zero A - Leave flags
	MOV	[NULFLG],AL		; Zero null flag
	JNZ	ECHDEL			; Set - Echo it
	DEC	CH			; Decrement length
	JZ	GETLIN			; Get line again if empty
	CALL	OUTC			; Output null character
	JMP	ECHDEL1			; Skip "DEC B"
ECHDEL:
	DEC	CH			; Count bytes in buffer
ECHDEL1:
	LAHF
	DEC	BX			; Back space buffer
	SAHF
	JZ	OTKLN			; No buffer - Try again
	MOV	AL,[BX]			; Get deleted byte
	CALL	OUTC			; Echo it
	JMP	MORINP			; Get more input
;
DELCHR:
	DEC	CH			; Count bytes in buffer
	LAHF
	DEC	BX			; Back space buffer
	SAHF
	CALL	OUTC			; Output character in A
	JNZ	MORINP			; Not end - Get more
OTKLN:
	CALL	OUTC			; Output character in A
KILIN:
	CALL	PRCRLF			; Output CRLF
	JMP	TTYLIN			; Get line again
;
GETLIN:
TTYLIN:
	MOV	BX,BUFFER		; Get a line by charact
	MOV	CH,1			; Set buffer as empty
	XOR	AL,AL
	MOV	[NULFLG],AL		; Clear null flag
MORINP:
	CALL	CLOTST			; Get character and test ^O
	MOV	CL,AL			; Save character in C
	CMP	AL,DEL			; Delete character?
	JZ	DODEL			; Yes - Process it
	MOV	AL,[NULFLG]		; Get null flag
	OR	AL,AL			; Test null flag status
	JZ	PROCES			; Reset - Process character
	MOV	AL,0			; Set a null
	CALL	OUTC			; Output null
	XOR	AL,AL			; Clear A
	MOV	[NULFLG],AL		; Reset null flag
PROCES:
	MOV	AL,CL			; Get character
	CMP	AL,CTRLG		; Bell?
	JZ	PUTCTL			; Yes - Save it
	CMP	AL,CTRLC		; Is it control "C"?
	JNZ	PROCES1
	CALL	PRCRLF			; Yes - Output CRLF
PROCES1:
	STC				; Flag break
	JNZ	PROCES2
	RET				; Return if control "C"
PROCES2:
	CMP	AL,CR			; Is it enter?
	JNZ	PROCES3			; Yes - Terminate input
	JMP	ENDINP
PROCES3:
	CMP	AL,CTRLU		; Is it control "U"?
	JZ	KILIN			; Yes - Get another line
	CMP	AL,'@'			; Is it "kill line"?
	JZ	OTKLN			; Yes - Kill line
	CMP	AL,'_'			; Is it delete?
	JZ	DELCHR			; Yes - Delete character
	CMP	AL,BKSP			; Is it backspace?
	JZ	DELCHR			; Yes - Delete character
	CMP	AL,CTRLR		; Is it control "R"?
	JNZ	PUTBUF			; No - Put in buffer
	PUSH	CX			; Save buffer length
	PUSH	DX			; Save DE
	PUSH	BX			; Save buffer address
	MOV	BYTE PTR [BX],0		; Mark end of buffer
	CALL	OUTNCR			; Output and do CRLF
	MOV	BX,BUFFER		; Point to buffer start
	CALL	PRS			; Output buffer
	POP	BX			; Restore buffer address
	POP	DX			; Restore DE
	POP	CX			; Restore buffer length
	JMP	MORINP			; Get another character
;
PUTBUF:
	CMP	AL,' '			; Is it a control code?
	JC	MORINP			; Yes - Ignore
PUTCTL:
	MOV	AL,CH			; Get number of bytes in buffer
	CMP	AL,72+1			; Test for line overflow
	MOV	AL,CTRLG		; Set a bell
	JNC	OUTNBS			; Ring bell if buffer full
	MOV	AL,CL			; Get character
	MOV	[BX],CL			; Save in buffer
	MOV	[LSTBIN],AL		; Save last input byte
	LAHF
	INC	BX			; Move up buffer
	SAHF
	INC	CH			; Increment length
OUTIT:
	CALL	OUTC			; Output the character entered
	JMP	MORINP			; Get another character
;
OUTNBS:
	CALL	OUTC			; Output bell and back over it
	MOV	AL,BKSP			; Set back space
	JMP	OUTIT			; Output it and get more
;
; Z or Overflow
CPDEHL:
	MOV	AL,BH			; Get H
	SUB	AL,DH			; Compare with D
	JZ	CPDEHL1
	RET				; Different - Exit
CPDEHL1:
	MOV	AL,BL			; Get L
	SUB	AL,DL			; Compare with E
	RET				; Return status
;
CHKSYN:
	MOV	AL,[BX]			; Check syntax of character
	MOV	BP,SP
	XCHG	[BP],BX			; Address of test byte
	CMP	AL,[BX]			; Same as in code string?
	LAHF
	INC	BX			; Return address
	SAHF
	MOV	BP,SP
	XCHG	[BP],BX			; Put it back
	JNZ	CHKSYN1			; Yes - Get next character
	JMP	GETCHR
CHKSYN1:
	JMP	SNERR			; Different - ?SN Error
;
OUTC:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save character
	XCHG	AH,AL
	MOV	AL,[CTLOFG]		; Get control "O" flag
	OR	AL,AL			; Is it set?
	JZ	OUTC1			; Yes - don't output
	JMP	POPAF
OUTC1:
	POP	AX			; Restore character
	XCHG	AH,AL
	SAHF
	PUSH	CX			; Save buffer length
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save character
	XCHG	AH,AL
	CMP	AL,' '			; Is it a control code?
	JC	DINPOS			; Yes - Don't INC POS(X)
	MOV	AL,[LWIDTH]		; Get line width
	MOV	CH,AL			; To B
	MOV	AL,[CURPOS]		; Get cursor position
	INC	CH			; Width 255?
	JZ	INCLEN			; Yes - No width limit
	DEC	CH			; Restore width
	CMP	AL,CH			; At end of line?
	JNZ	INCLEN
	CALL	PRCRLF			; Yes - output CRLF
INCLEN:
	INC	AL			; Move on one character
	MOV	[CURPOS],AL		; Save new position
DINPOS:
	POP	AX			; Restore character
	XCHG	AH,AL
	SAHF
	POP	CX			; Restore buffer length
	CALL	putch			; Send it
	RET
;
CLOTST:
	CALL	getch			; Get input character
	AND	AL,01111111B		; Strip bit 7
	CMP	AL,CTRLO		; Is it control "O"?
	JZ	CLOTST1
	RET				; No don't flip flag
CLOTST1:
	MOV	AL,[CTLOFG]		; Get flag
	NOT	AL			; Flip it
	MOV	[CTLOFG],AL		; Put it back
	XOR	AL,AL			; Null character
	RET
;
LIST:
	CALL	ATOH			; ASCII number to DE
	JZ	LIST1
	RET				; Return if anything extra
LIST1:
	POP	CX			; Rubbish - Not needed
	CALL	SRCHLN			; Search for line number in DE
	PUSH	CX			; Save address of line
	CALL	SETLIN			; Set up lines counter
LISTLP:
	POP	BX			; Restore address of line
	MOV	CL,[BX]			; Get LSB of next line
;	LAHF
	INC	BX
;	SAHF
	MOV	CH,[BX]			; Get MSB of next line
;	LAHF
	INC	BX
;	SAHF
	MOV	AL,CH			; BC = 0 (End of program)?
	OR	AL,CL
	JNZ	LISTLP1			; Yes - Go to command mode
	JMP	PRNTOK
LISTLP1:
	CALL	COUNT			; Count lines
	CALL	TSTBRK			; Test for break key
	PUSH	CX			; Save address of next line
	CALL	PRCRLF			; Output CRLF
	MOV	DL,[BX]			; Get LSB of line numbe
	LAHF
	INC	BX
	SAHF
	MOV	DH,[BX]			; Get MSB of line number
	LAHF
	INC	BX
	SAHF
	PUSH	BX			; Save address of line start
	XCHG	BX,DX			; Line number to HL
	CALL	PRNTHL			; Output line number in decimal
	MOV	AL,' '			; Space after line number
	POP	BX			; Restore start of line address
LSTLP2:
	CALL	OUTC			; Output character in A
LSTLP3:
	MOV	AL,[BX]			; Get next byte in line
	OR	AL,AL			; End of line?
	LAHF
	INC	BX			; To next byte in line
	SAHF
	JZ	LISTLP			; Yes - get next line
	JNS	LSTLP2			; No token - output it
	SUB	AL,ZEND-1		; Find and output word
	MOV	CL,AL			; Token offset+1 to C
	MOV	DX,WORDS		; Reserved word list
FNDTOK:
	XCHG	BX,DX
	MOV	AL,[BX]			; Get character in list
	XCHG	BX,DX
;	LAHF
	INC	DX			; Move on to next
;	SAHF
	OR	AL,AL			; Is it start of word?
	JNS	FNDTOK			; No - Keep looking for word
	DEC	CL			; Count words
	JNZ	FNDTOK			; Not there - keep look
OUTWRD:
	AND	AL,01111111B		; Strip bit 7
	CALL	OUTC			; Output first character
	XCHG	BX,DX
	MOV	AL,[BX]			; Get next character
	XCHG	BX,DX
	LAHF
	INC	DX			; Move on to next
	SAHF
	OR	AL,AL			; Is it end of word?
	JNS	OUTWRD			; No - output the rest
	JMP	LSTLP3			; Next byte in line
;
SETLIN:
	PUSH	BX			; Set up LINES counter
	MOV	BX,[LINESN]		; Get LINES number
	MOV	[LINESC],BX		; Save in LINES counter
	POP	BX
	RET
;
COUNT:
	PUSH	BX			; Save code string address
	PUSH	DX
	MOV	BX,[LINESC]		; Get LINES counter
	MOV	DX,-1
	ADC	BX,DX			; Decrement
;
	MOV	[LINESC],BX		; Put it back
	POP	DX
	POP	BX			; Restore code string address
	JS	COUNT1
	RET				; Return if more lines to go
COUNT1:
	PUSH	BX			; Save code string address
	MOV	BX,[LINESN]		; Get LINES number
	MOV	[LINESC],BX		; Reset LINES counter
	CALL	getch			; Get input character
	CMP	AL,CTRLC		; Is it control "C"?
	JZ	RSLNBK			; Yes - Reset LINES an break
	POP	BX			; Restore code string address
	JMP	COUNT			; Keep on counting
;
RSLNBK:
	MOV	BX,[LINESN]		; Get LINES number
	MOV	[LINESC],BX		; Reset LINES counter
	JMP	BRKRET			; Go and output "Break"
;
FOR:
	MOV	AL,64H			; Flag "FOR" assignment
	MOV	[FORFLG],AL		; Save "FOR" flag
	CALL	LET			; Set up initial index
	POP	CX			; Drop RETurn address
	PUSH	BX			; Save code string address
	CALL	DATA			; Get next statement address
	MOV	[LOOPST],BX		; Save it for start of loop
	MOV	BX,2			; Offset for "FOR" block
	ADD	BX,SP			; Point to it
FORSLP:
	CALL	LOKFOR			; Look for existing "FOR" block
	POP	DX			; Get code string address
	JNZ	FORFND			; No nesting found
	ADD	BX,CX			; Move into "FOR" block
	PUSH	DX			; Save code string address
;	LAHF
	DEC	BX
;	SAHF
	MOV	DH,[BX]			; Get MSB of loop statement
;	LAHF
	DEC	BX
;	SAHF
	MOV	DL,[BX]			; Get LSB of loop statement
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	PUSH	BX			; Save block address
	MOV	BX,[LOOPST]		; Get address of loop statement
	CALL	CPDEHL			; Compare the FOR loops
	POP	BX			; Restore block address
	JNZ	FORSLP			; Different FORs - Find another
	POP	DX			; Restore code string address
	MOV	SP,BX			; Remove all nested loops
;
FORFND:
	XCHG	BX,DX			; Code string address to HL
	MOV	CL,8
	CALL	CHKSTK			; Check for 8 levels of stack
	PUSH	BX			; Save code string address
	MOV	BX,[LOOPST]		; Get first statement of loop
	MOV	BP,SP
	XCHG	[BP],BX			; Save and restore code string
	PUSH	BX			; Re-save code string address
	MOV	BX,[LINEAT]		; Get current line number
	MOV	BP,SP
	XCHG	[BP],BX			; Save and restore code string
	CALL	TSTNUM			; Make sure it's a number
	CALL	CHKSYN			; Make sure "TO" is next
	DB	ZTO			; "TO" token
	CALL	GETNUM			; Get "TO" expression value
	PUSH	BX			; Save code string address
	CALL	BCDEFP			; Move "TO" value to BCDE
	POP	BX			; Restore code string address
	PUSH	CX			; Save "TO" value in block
	PUSH	DX
	MOV	CX,8100H		; BCDE - 1 (default STE
	MOV	DH,CL			; C=0
	MOV	DL,DH			; D=0
	MOV	AL,[BX]			; Get next byte in code string
	CMP	AL,ZSTEP		; See if "STEP" is stated
	MOV	AL,1			; Sign of step = 1
	JNZ	SAVSTP			; No STEP given - Default to 1
	CALL	GETCHR			; Jump over "STEP" token
	CALL	GETNUM			; Get step value
	PUSH	BX			; Save code string address
	CALL	BCDEFP			; Move STEP to BCDE
	CALL	TSTSGN			; Test sign of FPREG
	POP	BX			; Restore code string address
SAVSTP:
	PUSH	CX			; Save the STEP value in block
	PUSH	DX
	LAHF
	XCHG	AH,AL			; Save sign of STEP
	PUSH	AX
	XCHG	AH,AL
	INC	SP			; Don't save flags
	PUSH	BX			; Save code string address
	MOV	BX,[BRKLIN]		; Get address of index variable
	MOV	BP,SP
	XCHG	[BP],BX			; Save and restore code string
PUTFID:
	MOV	CH,ZFOR			; "FOR" block marker
	PUSH	CX			; Save it
	INC	SP			; Don't save C
;
RUNCNT:
	CALL	TSTBRK			; Execution driver - Test break
	MOV	[BRKLIN],BX		; Save code address for a key
	MOV	AL,[BX]			; Get next byte in code string
	CMP	AL,':'			; Multi statement line?
	JZ	EXCUTE			; Yes - Execute it
	OR	AL,AL			; End of line?
	JZ	RUNCNT1			; No - Syntax error
	JMP	SNERR
RUNCNT1:
	LAHF
	INC	BX			; Point to address of next line
	SAHF
	MOV	AL,[BX]			; Get LSB of line point
	LAHF
	INC	BX
	SAHF
	OR	AL,[BX]			; Is it zero (End of prog)?
	JNZ	RUNCNT2			; Yes - Terminate execution
	JMP	ENDPRG
RUNCNT2:
	LAHF
	INC	BX			; Point to line number
	SAHF
	MOV	DL,[BX]			; Get LSB of line numbe
	LAHF
	INC	BX
	SAHF
	MOV	DH,[BX]			; Get MSB of line numbe
	XCHG	BX,DX			; Line number to HL
	MOV	[LINEAT],BX		; Save as current line number
	XCHG	BX,DX			; Line number back to DE
EXCUTE:
	CALL	GETCHR			; Get key word
	MOV	DX,RUNCNT		; Where to RETurn to
	PUSH	DX			; Save for RETurn
IFJMP:
	JNZ	ONJMP
	RET				; Go to RUNCNT if end of STMT
ONJMP:
	SUB	AL,ZEND			; Is it a token?
	JNC	ONJMP1
	JMP	LET			; No - try to assign it
ONJMP1:
	CMP	AL,ZNEW+1-ZEND		; END to NEW ?
	JC	ONJMP2
	JMP	SNERR			; Not a key word - ?SN Error
ONJMP2:
	ROL	AL,1			; Double it
	MOV	CL,AL			; BC = Offset into table
	MOV	CH,0
	XCHG	BX,DX			; Save code string address
	MOV	BX,WORDTB		; Keyword address table
	ADD	BX,CX			; Point to routine address
	MOV	CL,[BX]			; Get LSB of routine address
	LAHF
	INC	BX
	SAHF
	MOV	CH,[BX]			; Get MSB of routine address
	PUSH	CX			; Save routine address
	XCHG	BX,DX			; Restore code string address
;
GETCHR:
;	LAHF
	INC	BX			; Point to next character
;	SAHF
	MOV	AL,[BX]			; Get next code string byte
	CMP	AL,':'			; Z if ':'
	JC	GETCHR1
	RET				; NC if > "9"
GETCHR1:
	CMP	AL,' '
	JZ	GETCHR			; Skip over spaces
	CMP	AL,'0'
	CMC				; NC if < '0'
	INC	AL			; Test for zero - Leave carry
	DEC	AL			; Z if Null
	RET
;
RESTOR:
	XCHG	BX,DX			; Save code string address
	MOV	BX,[BASTXT]		; Point to start of program
	JZ	RESTNL			; Just RESTORE - reset pointer
	XCHG	BX,DX			; Restore code string address
	CALL	ATOH			; Get line number to DE
	PUSH	BX			; Save code string address
	CALL	SRCHLN			; Search for line number in DE
	MOV	BX,CX			; HL = Address of line
	POP	DX			; Restore code string address
	JC	RESTNL
	JMP	ULERR			; ?UL Error if not found
RESTNL:
	LAHF
	DEC	BX			; Byte before DATA statement
	SAHF
UPDATA:
	MOV	[NXTDAT],BX		; Update DATA pointer
	XCHG	BX,DX			; Restore code string address
	RET
;

TSTBRK:
	CALL	CHKCHR			; Check input status
	JNZ	TSTBRK1
	RET				; No key, go back
TSTBRK1:
	CALL	getch			; Get the key into A
	CMP	AL,ESC			; Escape key?
	JZ	BRK			; Yes, break
	CMP	AL,CTRLC		; <Ctrl-C>
	JZ	BRK			; Yes, break
	CMP	AL,CTRLS		; Stop scrolling?
	JZ	STALL
	RET				; Other key, ignore
;

STALL:
	CALL	getch			; Wait for key
	CMP	AL,CTRLQ		; Resume scrolling?
	JNZ	STALL1
	RET				; Release the chokehold
STALL1:
	CMP	AL,CTRLC		; Second break?
	JZ	STOP			; Break during hold exit
	JMP	STALL			; Loop until <Ctrl-Q> o
;
BRK:
	MOV	AL,0FFH      		; Set BRKFLG
	MOV	[BRKFLG],AL		; Store it
;

STOP:
	JZ	STOP1
	RET				; Exit if anything else
STOP1:
	OR	AL,11000000B		; Flag "STOP"
	JMP	PEND1
PEND:
	JZ	PEND1
	RET				; Exit if anything else
PEND1:
	MOV	[BRKLIN],BX		; Save point of break
	JMP	INPBRK1			; Skip "OR AL,11111111B"
INPBRK:
	OR	AL,11111111B		; Flag "Break" wanted
INPBRK1:
	POP	CX			; Return not needed and more
ENDPRG:
	MOV	BX,[LINEAT]		; Get current line number
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save STOP / END statu
	XCHG	AH,AL
	MOV	AL,BL			; Is it direct break?
	AND	AL,BH
	INC	AL			; Line is -1 if direct break
	JZ	NOLIN			; Yes - No line number
	MOV	[ERRLIN],BX		; Save line of break
	MOV	BX,[BRKLIN]		; Get point of break
	MOV	[CONTAD],BX		; Save point to CONTinue
NOLIN:
	XOR	AL,AL
	MOV	[CTLOFG],AL		; Enable output
	CALL	STTLIN			; Start a new line
	POP	AX			; Restore STOP / END status
	XCHG	AH,AL
	SAHF
	MOV	BX,BRKMSG		; "Break" message
	JZ	NOLIN1
	JMP	ERRIN			; "in line" wanted?
NOLIN1:
	JMP	PRNTOK			; Go to command mode
;
CONT:
	MOV	BX,[CONTAD]		; Get CONTinue address
	MOV	AL,BH			; Is it zero?
	OR	AL,BL
	MOV	DL,CN			; ?CN Error
	JNZ	CONT1
	JMP	ERROR			; Yes - output "?CN Error"
CONT1:
	XCHG	BX,DX			; Save code string address
	MOV	BX,[ERRLIN]		; Get line of last break
	MOV	[LINEAT],BX		; Set up current line number
	XCHG	BX,DX			; Restore code string address
	RET				; CONTinue where left off
;
NULL:
	CALL	GETINT			; Get integer 0-255
	JZ	NULL1
	RET				; Return if bad value
NULL1:
	MOV	[NULLS],AL		; Set nulls number
	RET
;

ACCSUM:
	PUSH	BX			; Save address in array
	MOV	BX,[CHKSUM]		; Get check sum
	MOV	CH,0			; BC - Value of byte
	MOV	CL,AL
	ADD	BX,CX			; Add byte to check sum
	MOV	[CHKSUM],BX		; Re-save check sum
	POP	BX			; Restore address in array
	RET
;
CHKLTR:
	MOV	AL,[BX]			; Get byte
	CMP	AL,'A'			; < 'A' ?
	JNC	CHKLTR1
	RET         			  ; Carry set if not letter
CHKLTR1:
	CMP	AL,'Z'+1		; > 'Z' ?
	CMC
	RET				; Carry set if not letter
;
FPSINT:
	CALL	GETCHR			; Get next character
POSINT:
	CALL	GETNUM			; Get integer 0 to 32767
DEPINT:
	CALL	TSTSGN			; Test sign of FPREG
	JS	FCERR			; Negative - ?FC Error
DEINT:
	MOV	AL,[FPEXP]		; Get integer value to DE
	CMP	AL,80H+16		; Exponent in range (16 bits)?
	JNC	DEINT1
	JMP	FPINT			; Yes - convert it
DEINT1:
	MOV	CX,9080H		; BCDE = -32768
	MOV	DX,0000
	PUSH	BX			; Save code string address
	CALL	CMPNUM			; Compare FPREG with BCDE
	POP	BX			; Restore code string address
	MOV	DH,CL			; MSB to D
	JNZ	FCERR
	RET				; Return if in range
FCERR:
	MOV	DL,FC			; ?FC Error
	JMP	ERROR			; Output error-
;
ATOH:
	LAHF
	DEC	BX			; ASCII number to DE binary
	SAHF
GETLN:
	MOV	DX,0			; Get number to DE
GTLNLP:
	CALL	GETCHR			; Get next character
	JC	GTLNLP1
	RET				; Exit if not a digit
GTLNLP1:
	PUSH	BX			; Save code string address
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save digit
	XCHG	AH,AL
	MOV	BX,65529/10		; Largest number 65529
	CALL	CPDEHL			; Number in range?
	JNC	GTLNLP2			; No - ?SN Error
	JMP	SNERR
GTLNLP2:
	MOV	BX,DX			; HL = Number
	ADD	BX,DX			; Times 2
	ADD	BX,BX			; Times 4
	ADD	BX,DX			; Times 5
	ADD	BX,BX			; Times 10
	POP	AX			; Restore digit
	XCHG	AH,AL
	SAHF
	SUB	AL,'0'			; Make it 0 to 9
	MOV	DL,AL			; DE = Value of digit
	MOV	DH,0
	ADD	BX,DX			; Add to number
	XCHG	BX,DX			; Number to DE
	POP	BX			; Restore code string address
	JMP	GTLNLP			; Go to next character
;
CLEAR:
	JNZ	CLEAR1
	JMP	INTVAR			; Just "CLEAR" Keep parameters
CLEAR1:
	CALL	POSINT			; Get integer 0 to 32767 to DE
;	LAHF
	DEC	BX			; Cancel increment
;	SAHF
	CALL	GETCHR			; Get next character
	PUSH	BX			; Save code string address
	MOV	BX,[LSTRAM]		; Get end of RAM
	JZ	STORED			; No value given - Use stored
	POP	BX			; Restore code string address
	CALL	CHKSYN			; Check for comma
	DB	','
	PUSH	DX			; Save number
	CALL	POSINT			; Get integer 0 to 32767
;	LAHF
	DEC	BX			; Cancel increment
;	SAHF
	CALL	GETCHR			; Get next character
	JZ	CLEAR2
	JMP	SNERR			; ?SN Error if more on line
CLEAR2:
	MOV	BP,SP
	XCHG	[BP],BX			; Save code string address
	XCHG	BX,DX			; Number to DE
STORED:
	MOV	AL,BL			; Get LSB of new RAM top
	SUB	AL,DL			; Subtract LSB of string space
	MOV	DL,AL			; Save LSB
	MOV	AL,BH			; Get MSB of new RAM top
	SBB	AL,DH			; Subtract MSB of string space
	MOV	DH,AL			; Save MSB
	JNC	STORED1
	JMP	OMERR			; ?OM Error if not enough mem
STORED1:
	PUSH	BX			; Save RAM top
	MOV	BX,[PROGND]		; Get program end
	MOV	CX,40			; 40 Bytes minimum working RAM
	ADD	BX,CX			; Get lowest address
	CALL	CPDEHL			; Enough memory?
	JC	STORED2
	JMP	OMERR			; No - ?OM Error
STORED2:
	XCHG	BX,DX			; RAM top to HL
	MOV	[STRSPC],BX		; Set new string space
	POP	BX			; End of memory to use
	MOV	[LSTRAM],BX		; Set new top of RAM
	POP	BX			; Restore code string address
	JMP	INTVAR			; Initialise variables
;
RUN:
	JNZ	RUN1
	JMP	RUNFST			; RUN from start if just RUN
RUN1:
	CALL	INTVAR			; Initialise variables
	MOV	CX,RUNCNT		; Execution driver loop
	JMP	RUNLIN			; RUN from line number
;
GOSUB:
	MOV	CL,3			; 3 Levels of stack needed
	CALL	CHKSTK			; Check for 3 levels of stack
	POP	CX			; Get return address
	PUSH	BX			; Save code string for RETURN
	PUSH	BX			; And for GOSUB routine
	MOV	BX,[LINEAT]		; Get current line
	MOV	BP,SP
	XCHG	[BP],BX			; Into stack - Code string out
	MOV	AL,ZGOSUB		; "GOSUB" token
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save token
	XCHG	AH,AL
	INC	SP			; Don't save flags
;
RUNLIN:
	PUSH	CX			; Save return address
GOTO:
	CALL	ATOH			; ASCII number to DE binary
	CALL	REM			; Get end of line
	PUSH	BX			; Save end of line
	MOV	BX,[LINEAT]		; Get current line
	CALL	CPDEHL			; Line after current?
	POP	BX			; Restore end of line
	LAHF
	INC	BX			; Start of next line
	SAHF
	JNC	GOTO1
	CALL	SRCHLP			; Line is after current line
GOTO1:
	JC	GOTO2
	CALL	SRCHLN			; Line is before current line
GOTO2:
	MOV	BX,CX			; Set up code string address
	LAHF
	DEC	BX			; Incremented after
	SAHF
	JNC	ULERR
	RET				; Line found
ULERR:
	MOV	DL,UL			; ?UL Error
	JMP	ERROR			; Output error message
;
RETURN:
	JZ	RETURN1
	RET				; Return if not just RETURN
RETURN1:
	MOV	DH,-1			; Flag "GOSUB" search
	CALL	BAKSTK			; Look "GOSUB" block
	MOV	SP,BX			; Kill all FORs in subroutine
	CMP	AL,ZGOSUB		; Test for "GOSUB" token
	MOV	DL,RG			; ?RG Error
	JZ	RETURN2
	JMP	ERROR			; Error if no "GOSUB" found
RETURN2:
	POP	BX			; Get RETURN line number
	MOV	[LINEAT],BX		; Save as current
	LAHF
	INC	BX			; Was it from direct statement?
	SAHF
	MOV	AL,BH
	OR	AL,BL			; Return to line
	JNZ	RETLIN			; No - Return to line
	MOV	AL,[LSTBIN]		; Any INPUT in subroutine?
	OR	AL,AL			; If so buffer is corrupted
	JZ	RETLIN
	JMP	POPNOK			; Yes - Go to command mode
RETLIN:
	MOV	BX,RUNCNT		; Execution driver loop
	MOV	BP,SP
	XCHG	[BP],BX			; Into stack - Code string out
	JMP	DATA			; Skip "POP BX"
NXTDTA:
	POP	BX			; Restore code string address
;
DATA:
	MOV	CL,':'			; ":" End of statemen
	JMP	REM1
REM:
	MOV	CL,0			; 00 End of statemen
REM1:
	MOV	CH,0
NXTSTL:
	MOV	AL,CL			; Statement and byte
	MOV	CL,CH
	MOV	CH,AL			; Statement end byte
NXTSTT:
	MOV	AL,[BX]			; Get byte
	OR	AL,AL			; End of line?
	JNZ	NXTSTT1
	RET				; Yes - Exit
NXTSTT1:
	CMP	AL,CH			; End of statement?
	JNZ	NXTSTT2
	RET				; Yes - Exit
NXTSTT2:
	LAHF
	INC	BX
	SAHF				; Next byte
	CMP	AL,'"'			; Literal string?
	JZ	NXTSTL			; Yes - Look for another '"'
	JMP	NXTSTT			; Keep looking
;
LET:
	CALL	GETVAR			; Get variable name
	CALL	CHKSYN			; Make sure "=" follows
	DB	ZEQUAL			; "=" token
	PUSH	DX			; Save address of variable
	MOV	AL,[TYPE]		; Get data type
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save type
	XCHG	AH,AL
	CALL	EVAL			; Evaluate expression
	POP	AX			; Restore type
	XCHG	AH,AL
	SAHF
	MOV	BP,SP
	XCHG	[BP],BX			; Save code - Get var addr
	MOV	[BRKLIN],BX		; Save address of variable
	RCR	AL,1			; Adjust type
	CALL	CHKTYP			; Check types are the same
	JZ	LETNUM			; Numeric - Move value
LETSTR:
	PUSH	BX			; Save address of string var
	MOV	BX,[FPREG]		; Pointer to string entry
	PUSH	BX			; Save it on stack
;	LAHF
	INC	BX			; Skip over length
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	DL,[BX]			; LSB of string address
;	LAHF
	INC	BX
;	SAHF
	MOV	DH,[BX]			; MSB of string address
	MOV	BX,[BASTXT]		; Point to start of program
	CALL	CPDEHL			; Is string before program?
	JNC	CRESTR			; Yes - Create string entry
	MOV	BX,[STRSPC]		; Point to string space
	CALL	CPDEHL			; Is string literal in program?
	POP	DX			; Restore address of string
	JNC	MVSTPT			; Yes - Set up pointer
	MOV	BX,TMPSTR		; Temporary string pool
	CALL	CPDEHL			; Is string in temporary pool?
	JNC	MVSTPT			; No - Set up pointer
	JMP	CRESTR1			; Skip "POP DX"
CRESTR:
	POP	DX			; Restore address of string
CRESTR1:
	CALL	BAKTMP			; Back to last tmp-str entry
	XCHG	BX,DX			; Address of string entry
	CALL	SAVSTR			; Save string in string area
MVSTPT:
	CALL	BAKTMP			; Back to last tmp-str entry
	POP	BX			; Get string pointer
	CALL	DETHL4			; Move string pointer to var
	POP	BX			; Restore code string adress
	RET
;
LETNUM:
	PUSH	BX			; Save address of variable
	CALL	FPTHL			; Move value to variable
	POP	DX			; Restore address of variable
	POP	BX			; Restore code string address
	RET
;
ON:
	CALL	GETINT			; Get integer 0-255
	MOV	AL,[BX]			; Get "GOTO" or "GOSUB" token
	MOV	CH,AL			; Save in B
	CMP	AL,ZGOSUB		; "GOSUB" token?
	JZ	ONGO			; Yes - Find line numbe
	CALL	CHKSYN			; Make sure it's "GOTO"
	DB	ZGOTO			; "GOTO" token
	LAHF
	DEC	BX			; Cancel increment
	SAHF
ONGO:
	MOV	CL,DL			; Integer of branch value
ONGOLP:
	DEC	CL			; Count branches
	MOV	AL,CH			; Get "GOTO" or "GOSUB" token
	JNZ	ONGOLP1
	JMP	ONJMP			; Go to that line if right one
ONGOLP1:
	CALL	GETLN			; Get line number to DE
	CMP	AL,','			; Another line number?
	JZ	ONGOLP
	RET				; No - Drop through
	JMP	ONGOLP			; Yes - loop
;
IF:
	CALL	EVAL			; Evaluate expression
	MOV	AL,[BX]			; Get token
	CMP	AL,ZGOTO		; "GOTO" token?
	JZ	IFGO			; Yes - Get line
	CALL	CHKSYN			; Make sure it's "THEN"
	DB	ZTHEN			; "THEN" token
	LAHF
	DEC	BX			; Cancel increment
	SAHF
IFGO:
	CALL	TSTNUM			; Make sure it's numeric
	CALL	TSTSGN			; Test state of expression
	JNZ	IFGO1
	JMP	REM			; False - Drop through
IFGO1:
	CALL	GETCHR			; Get next character
	JNC	IFGO2
	JMP	GOTO			; Number - GOTO that line
IFGO2:
	JMP	IFJMP			; Otherwise do statemen
;
MRPRNT:
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
PRINT:
	JZ	PRCRLF			; CRLF if just PRINT
PRNTLP:
	JNZ	PRNTLP1
	RET				; End of list - Exit
PRNTLP1:
	CMP	AL,ZTAB			; "TAB(" token?
	JNZ	PRNTLP2
	JMP	DOTAB			; Yes - Do TAB routine
PRNTLP2:
	CMP	AL,ZSPC			; "SPC(" token?
	JNZ	PRNTLP3
	JMP	DOTAB			; Yes - Do SPC routine
PRNTLP3:
	PUSH	BX			; Save code string address
	CMP	AL,','			; Comma?
	JNZ	PRNTLP4
	JMP	DOCOM			; Yes - Move to next zone
PRNTLP4:
	CMP	AL,';'			; Semi-colon?
	JNZ	PRNTLP5
	JMP	NEXITM			; Do semi-colon routine
PRNTLP5:
	POP	CX			; Code string address to BC
	CALL	EVAL			; Evaluate expression
	PUSH	BX			; Save code string address
	MOV	AL,[TYPE]		; Get variable type
	OR	AL,AL			; Is it a string variable?
	JNZ	PRNTST			; Yes - Output string contents
	CALL	NUMASC			; Convert number to text
	CALL	CRTST			; Create temporary strig
	MOV	BYTE PTR [BX],' '	; Followed by a space
	MOV	BX,[FPREG]		; Get length of output
	INC	BYTE PTR [BX]		; Plus 1 for the space
	MOV	BX,[FPREG]		; < Not needed >
	MOV	AL,[LWIDTH]		; Get width of line
	MOV	CH,AL			; To B
	INC	CH			; Width 255 (No limit)?
	JZ	PRNTNB			; Yes - Output number string
	INC	CH			; Adjust it
	MOV	AL,[CURPOS]		; Get cursor position
	ADD	AL,[BX]			; Add length of string
	DEC	AL			; Adjust it
	CMP	AL,CH			; Will output fit on this line?
	JC	PRNTNB
	CALL	PRCRLF			; No - CRLF first
PRNTNB:
	CALL	PRS1			; Output string at (HL)
	XOR	AL,AL			; Skip CALL by setting address
PRNTST:
	JZ	PRNTST1
	CALL	PRS1			; Output string at (HL)
PRNTST1:
	POP	BX			; Restore code string address
	JMP	MRPRNT			; See if more to PRINT
;
STTLIN:
	MOV	AL,[CURPOS]		; Make sure on new line
	OR	AL,AL			; Already at start?
	JNZ	PRCRLF			; Start a new line
	RET				; Yes - Do nothing
;	JMP	PRCRLF			; Start a new line
;
ENDINP:
	MOV	BYTE PTR [BX],0		; Mark end of buffer
	MOV	BX,BUFFER-1		; Point to buffer
PRCRLF:
	MOV	AL,CR			; Load a CR
	CALL	OUTC			; Output character
	MOV	AL,LF			; Load a LF
	CALL	OUTC			; Output character
DONULL:
	XOR	AL,AL			; Set to position 0
	MOV	[CURPOS],AL		; Store it
	MOV	AL,[NULLS]		; Get number of nulls
NULLP:
	DEC	AL			; Count them
	JNZ	NULLP1
	RET				; Return if done
NULLP1:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save count
	XCHG	AH,AL
	XOR	AL,AL			; Load a null
	CALL	OUTC			; Output it
	POP	AX			; Restore count
	XCHG	AH,AL
	SAHF
	JMP	NULLP			; Keep counting
;
DOCOM:
	MOV	AL,[COMMAN]		; Get comma width
	MOV	CH,AL			; Save in B
	MOV	AL,[CURPOS]		; Get current position
	CMP	AL,CH			; Within the limit?
	JC	DOCOM1
	CALL	PRCRLF			; No - output CRLF
DOCOM1:
	JNC	NEXITM			; Get next item
ZONELP:
	SUB	AL,14			; Next zone of 14 characters
	JNC	ZONELP			; Repeat if more zones
	NOT	AL			; Number of spaces to output
	JMP	ASPCS			; Output them
;
DOTAB:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save token
	XCHG	AH,AL
	CALL	FNDNUM			; Evaluate expression
	CALL	CHKSYN			; Make sure ")" follows
	DB	")"
	LAHF
	DEC	BX			; Back space on to ")"
	SAHF
	POP	AX			; Restore token
	XCHG	AH,AL
	SAHF
	SUB	AL,ZSPC			; Was it "SPC(" ?
	PUSH	BX			; Save code string address
	JZ	DOSPC			; Yes - Do 'E' spaces
	MOV	AL,[CURPOS]		; Get current position
DOSPC:
	NOT	AL			; Number of spaces to print to
	ADD	AL,DL			; Total number to print
	JNC	NEXITM			; TAB < Current POS(X)
ASPCS:
	INC	AL			; Output A spaces
	MOV	CH,AL			; Save number to print
	MOV	AL,' '			; Space
SPCLP:
	CALL	OUTC			; Output character in A
	DEC	CH			; Count them
	JNZ	SPCLP			; Repeat if more
NEXITM:
	POP	BX			; Restore code string address
	CALL	GETCHR			; Get next character
	JMP	PRNTLP			; More to print
;
REDO:
	DB	"?Redo from start",CR,LF,0

;
BADINP:
	MOV	AL,[READFG]		; READ or INPUT?
	OR	AL,AL
	JZ	BADINP1			; READ - ?SN Error
	JMP	DATSNR
BADINP1:
	POP	CX			; Throw away code string addr
	MOV	BX,REDO			; "Redo from start" message
	CALL	PRS			; Output string
	JMP	DOAGN			; Do last INPUT again
;
INPUT:
	CALL	IDTEST			; Test for illegal direct
	MOV	AL,[BX]			; Get character after "INPUT"
	CMP	AL,'"'			; Is there a prompt string?
	MOV	AL,0			; Clear A and leave flags
	MOV	[CTLOFG],AL		; Enable output
	JNZ	NOPMPT			; No prompt - get input
	CALL	QTSTR			; Get string terminated by '"'
	CALL	CHKSYN			; Check for ';' after prompt
	DB	';'
	PUSH	BX			; Save code string address
	CALL	PRS1			; Output prompt string
	JMP	NOPMPT1			; Skip "PUSH BX"
NOPMPT:
	PUSH	BX			; Save code string addr
NOPMPT1:
	CALL	PROMPT			; Get input with "? " prompt
	POP	CX			; Restore code string address
	JNC	NOPMPT2
	JMP	INPBRK			; Break pressed - Exit
NOPMPT2:
	LAHF
	INC	BX			; Next byte
	SAHF
	MOV	AL,[BX]			; Get it
	OR	AL,AL			; End of line?
	LAHF
	DEC	BX			; Back again
	SAHF
	PUSH	CX			; Re-save code string address
	JNZ	NOPMPT3			; Yes - Find next DATA stmt
	JMP	NXTDTA
NOPMPT3:
	MOV	BYTE PTR [BX],','	; Store comma as separator
	JMP	NXTITM			; Get next item
;
READ:
	PUSH	BX			; Save code string address
	MOV	BX,[NXTDAT]		; Next DATA statement
	OR	AL,0AFH			; Flag "READ"
	JMP	NXTITM1
NXTITM:
	XOR	AL,AL			; Flag "INPUT"
NXTITM1:
	MOV	[READFG],AL		; Save "READ"/"INPUT" flag
	MOV	BP,SP
	XCHG	[BP],BX			; Get code str' , Save pointer
	JMP	GTVLUS			; Get values
;
NEDMOR:
	CALL	CHKSYN			; Check for comma between items
	DB	','
GTVLUS:
	CALL	GETVAR			; Get variable name
	MOV	BP,SP
	XCHG	[BP],BX			; Save code str" , Get pointer
	PUSH	DX			; Save variable address
	MOV	AL,[BX]			; Get next "INPUT"/"DATA" byte
	CMP	AL,','			; Comma?
	JZ	ANTVLU			; Yes - Get another value
	MOV	AL,[READFG]		; Is it READ?
	OR	AL,AL
	JZ	GTVLUS1			; Yes - Find next DATA stmt
	JMP	FDTLP
GTVLUS1:
	MOV	AL,'?'			; More INPUT needed
	CALL	OUTC			; Output character
	CALL	PROMPT			; Get INPUT with prompt
	POP	DX			; Variable address
	POP	CX			; Code string address
	JNC	GTVLUS2
	JMP	INPBRK			; Break pressed
GTVLUS2:
	LAHF
	INC	BX			; Point to next DATA byte
	SAHF
	MOV	AL,[BX]			; Get byte
	OR	AL,AL			; Is it zero (No input) ?
	LAHF
	DEC	BX			; Back space INPUT pointer
	SAHF
	PUSH	CX			; Save code string address
	JNZ	GTVLUS3			; Find end of buffer
	JMP	NXTDTA
GTVLUS3:
	PUSH	DX			; Save variable address
ANTVLU:
	MOV	AL,[TYPE]		; Check data type
	OR	AL,AL			; Is it numeric?
	JZ	INPBIN			; Yes - Convert to binary
	CALL	GETCHR			; Get next character
	MOV	DH,AL			; Save input character
	MOV	CH,AL			; Again
	CMP	AL,'"'			; Start of literal sting?
	JZ	STRENT			; Yes - Create string entry
	MOV	AL,[READFG]		; "READ" or "INPUT" ?
	OR	AL,AL
	MOV	DH,AL			; Save 00 if "INPUT"
	JZ	ITMSEP			; "INPUT" - End with 00
	MOV	DH,':'			; "DATA" - End with 00 or ":"
ITMSEP:
	MOV	CH,','			; Item separator
	LAHF
	DEC	BX			; Back space for DTSTR
	SAHF
STRENT:
	CALL	DTSTR			; Get string terminated by D
	XCHG	BX,DX			; String address to DE
	MOV	BX,LTSTND		; Where to go after LETSTR
	MOV	BP,SP
	XCHG	[BP],BX			; Save HL , get input ppinter
	PUSH	DX			; Save address of string
	JMP	LETSTR			; Assign string to variable
;
INPBIN:
	CALL	GETCHR			; Get next character
	CALL	ASCTFP			; Convert ASCII to FP number
	MOV	BP,SP
	XCHG	[BP],BX			; Save input ptr, Get var addr
	CALL	FPTHL			; Move FPREG to variable
	POP	BX			; Restore input pointer
LTSTND:
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
	JZ	MORDT			; End of line - More needed?
	CMP	AL,','			; Another value?
	JZ	MORDT			; No - Bad input
	JMP	BADINP
MORDT:
	MOV	BP,SP
	XCHG	[BP],BX			; Get code string address
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
	JZ	MORDT1			; More needed - Get it
	JMP	NEDMOR
MORDT1:
	POP	DX			; Restore DATA pointer
	MOV	AL,[READFG]		; "READ" or "INPUT" ?
	OR	AL,AL
	XCHG	BX,DX			; DATA pointer to HL
	JZ	MORDT2
	JMP	UPDATA			; Update DATA pointer if "READ"
MORDT2:
	PUSH	DX			; Save code string address
	OR	AL,[BX]			; More input given?
	MOV	BX,EXTIG		; "?Extra ignored" message
	JZ	MORDT3
	CALL	PRS			; Output string if extra given
MORDT3:
	POP	BX			; Restore code string address
	RET
;
EXTIG:
	DB	"?Extra ignored",CR,LF,0


;
FDTLP:
	CALL	DATA			; Get next statement
	OR	AL,AL			; End of line?
	JNZ	FANDT			; No - See if DATA statement
	LAHF
	INC	BX
	SAHF
	MOV	AL,[BX]			; End of program?
	LAHF
	INC	BX
	SAHF
	OR	AL,[BX]			; 00 00 Ends program
	MOV	DL,OD			; ?OD Error
	JNZ	FDTLP1
	JMP	ERROR			; Yes - Out of DATA
FDTLP1:
	LAHF
	INC	BX
	SAHF
	MOV	DL,[BX]			; LSB of line number
	LAHF
	INC	BX
	SAHF
	MOV	DH,[BX]			; MSB of line number
	XCHG	BX,DX
	MOV	[DATLIN],BX		; Set line of current DATA item
	XCHG	BX,DX
FANDT:
	CALL	GETCHR			; Get next character
	CMP	AL,ZDATA		; "DATA" token
	JNZ	FDTLP			; No "DATA" - Keep looking
	JMP	ANTVLU			; Found - Convert input
;
NEXT:
	MOV	DX,0			; In case no index given
NEXT1:
	JZ	NEXT2
	CALL	GETVAR			; Get index address
NEXT2:
	MOV	[BRKLIN],BX		; Save code string address
	CALL	BAKSTK			; Look for "FOR" block
	JZ	NEXT3
	JMP	NFERR			; No "FOR" - ?NF Error
NEXT3:
	MOV	SP,BX			; Clear nested loops
	PUSH	DX			; Save index address
	MOV	AL,[BX]			; Get sign of STEP
	LAHF
	INC	BX
	SAHF
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save sign of STEP
	XCHG	AH,AL
	PUSH	DX			; Save index address
	CALL	PHLTFP			; Move index value to FPREG
	MOV	BP,SP
	XCHG	[BP],BX			; Save address of TO value
	PUSH	BX			; Save address of index
	CALL	ADDPHL			; Add STEP to index valiable
	POP	BX			; Restore address of index
	CALL	FPTHL			; Move value to index variable
	POP	BX			; Restore address of TO value
	CALL	LOADFP			; Move TO value to BCDE
	PUSH	BX			; Save address of line of FOR
	CALL	CMPNUM			; Compare index with TO value
	POP	BX			; Restore address of line num
	POP	CX			; Address of sign of STEP
	SUB	AL,CH			; Compare with expected sign
	CALL	LOADFP			; BC = Loop stmt,DE = Line num
	JZ	KILFOR			; Loop finished - Terminal it
	XCHG	BX,DX			; Loop statement line number
	MOV	[LINEAT],BX		; Set loop line number
	MOV	BX,CX			; Set code string to loop
	JMP	PUTFID			; Put back "FOR" and continue
;
KILFOR:
	MOV	SP,BX			; Remove "FOR" block
	MOV	BX,[BRKLIN]		; Code string after "NE XT"
	MOV	AL,[BX]			; Get next byte in code string
	CMP	AL,','			; More NEXTs ?
	JZ	KILFOR1			; No - Do next statemen
	JMP	RUNCNT
KILFOR1:
	CALL	GETCHR			; Position to index nam
	CALL	NEXT1			; Re-enter NEXT routine
; < will not RETurn to here , Exit to RUNCNT or Loop >
;
GETNUM:
	CALL	EVAL			; Get a numeric expression
TSTNUM:
	OR	AL,AL			; Clear carry (numeric)
	JMP	CHKTYP
TSTSTR:
	STC				; Set carry (string)
CHKTYP:
	MOV	AL,[TYPE]		; Check types match
	ADC	AL,AL			; Expected + actual
	OR	AL,AL			; Clear carry , set parity
	JPO	CHKTYP1			; RET PE
	RET				; Even parity - Types match
CHKTYP1:
	JMP	TMERR			; Different types - Error
;
OPNPAR:
	CALL	CHKSYN			; Make sure "(" follows
	DB	"("
EVAL:
	LAHF
	DEC	BX
	SAHF				; Evaluate expression & save
	MOV	DH,0			; Precedence value
EVAL1:
	PUSH	DX			; Save precedence
	MOV	CL,1
	CALL	CHKSTK			; Check for 1 level of stack
	CALL	OPRND			; Get next expression value
EVAL2:
	MOV	[NXTOPR],BX		; Save address of next operator
EVAL3:
	MOV	BX,[NXTOPR]		; Restore address of next opt
	POP	CX			; Precedence value and operator
	MOV	AL,CH			; Get precedence value
	CMP	AL,78H			; "AND" or "OR" ?
	JC	EVAL4
	CALL	TSTNUM			; No - Make sure it's a number
EVAL4:
	MOV	AL,[BX]			; Get next operator / function
	MOV	DH,0			; Clear Last relation
RLTLP:
	SUB	AL,ZGTR			; ">" Token
	JC	FOPRND			; + - * / ^ AND OR - Test it
	CMP	AL,ZLTH+1-ZGTR		; < = >
	JNC	FOPRND			; Function - Call it
	CMP	AL,ZEQUAL-ZGTR		; "="
	RCL	AL,1			; <- Test for legal
	XOR	AL,DH			; <- combinations of < = >
	CMP	AL,DH			; <- by combining last token
	MOV	DH,AL			; <- with current one
	JNC	RLTLP1
	JMP	SNERR			; Error if "<<' '==" or ">>"
RLTLP1:
	MOV	[CUROPR],BX		; Save address of current token
	CALL	GETCHR			; Get next character
	JMP	RLTLP			; Treat the two as one
;
FOPRND:
	MOV	AL,DH			; < = > found ?
	OR	AL,AL
	JZ	FOPRND1
	JMP	TSTRED			; Yes - Test for reduction
FOPRND1:
	MOV	AL,[BX]			; Get operator token
	MOV	[CUROPR],BX		; Save operator address
	SUB	AL,ZPLUS		; Operator or function?
	JNC	FOPRND2
	RET				; Neither - Exit
FOPRND2:
	CMP	AL,ZOR+1-ZPLUS		; Is it + - * / ^ AND OR ?
	JC	FOPRND3
	RET				; No - Exit
FOPRND3:
	MOV	DL,AL			; Coded operator
	MOV	AL,[TYPE]		; Get data type
	DEC	AL			; FF = numeric , 00 = string
	OR	AL,DL			; Combine with coded operator
	MOV	AL,DL			; Get coded operator
	JNZ	FOPRND4			; String concatenation
	JMP	CONCAT
FOPRND4:
	ROL	AL,1			; Times 2
	ADD	AL,DL			; Times 3
	MOV	DL,AL			; To DE (D is 0)
	MOV	BX,PRITAB		; Precedence table
	ADD	BX,DX			; To the operator concerned
	MOV	AL,CH			; Last operator precedence
	MOV	DH,[BX]			; Get evaluation precedence
	CMP	AL,DH			; Compare with eval precedence
	JC	FOPRND5
	RET				; Exit if higher precedence
FOPRND5:
	LAHF
	INC	BX			; Point to routine addr
	SAHF
	CALL	TSTNUM			; Make sure it's a number
;
STKTHS:
	PUSH	CX			; Save last precedence token
	MOV	CX,EVAL3		; Where to go on prec' break
	PUSH	CX			; Save on stack for return
	MOV	CH,DL			; Save operator
	MOV	CL,DH			; Save precedence
	CALL	STAKFP			; Move value to stack
	MOV	DL,CH			; Restore operator
	MOV	DH,CL			; Restore precedence
	MOV	CL,[BX]			; Get LSB of routine address
	LAHF
	INC	BX
	SAHF
	MOV	CH,[BX]			; Get MSB of routine address
	LAHF
	INC	BX
	SAHF
	PUSH	CX			; Save routine address
	MOV	BX,[CUROPR]		; Address of current operator
	JMP	EVAL1			; Loop until prec' break
;
OPRND:
	XOR	AL,AL			; Get operand routine
	MOV	[TYPE],AL		; Set numeric expected
	CALL	GETCHR			; Get next character
	MOV	DL,MO			; ?MO Error
	JNZ	OPRND1
	JMP	ERROR			; No operand - Error
OPRND1:
	JNC	OPRND2
	JMP	ASCTFP			; Number - Get value
OPRND2:
	CALL	CHKLTR			; See if a letter
	JNC	CONVAR			; Letter - Find variable
	CMP	AL,'&'			; &H = HEX, &B = BINARY
	JNZ	NOTAMP
	CALL	GETCHR			; Get next character
	CMP	AL,'H'			; Hex number indicated?
	JNZ	OPRND3
	JMP	HEXTFP			; Convert Hex to FPREG
OPRND3:
	CMP	AL,'B'			; Binary number indicat
	JNZ	OPRND4
	JMP	BINTFP			; Convert Bin to FPREG
OPRND4:
	MOV	DL,SN			; If neither then a ?SN
	JNZ	NOTAMP
	JMP	ERROR
NOTAMP:
	CMP	AL,ZPLUS		; '+' Token ?
	JZ	OPRND			; Yes - Look for operand
	CMP	AL,'.'			; '.' ?
	JNZ	NOTAMP1			; Yes - Create FP number
	JMP	ASCTFP
NOTAMP1:
	CMP	AL,ZMINUS		; '-' Token ?
	JZ	MINUS			; Yes - Do minus
	CMP	AL,'"'			; Literal string ?
	JNZ	NOTAMP2
	JMP	QTSTR			; Get string terminated
NOTAMP2:
	CMP	AL,ZNOT			; "NOT" Token ?
	JNZ	NOTAMP3
	JMP	EVNOT			; Yes - Eval NOT expres
NOTAMP3:
	CMP	AL,ZFN			; "FN" Token ?
	JNZ	NOTAMP4
	JMP	DOFN			; Yes - Do FN routine
NOTAMP4:
	SUB	AL,ZSGN			; Is it a function?
	JNC	FNOFST			; Yes - Evaluate functi
EVLPAR:
	CALL	OPNPAR			; Evaluate expression i
	CALL	CHKSYN			; Make sure ")" follows
	DB	")"
	RET
;
MINUS:
	MOV	DH,7DH			; '-' precedence
	CALL	EVAL1			; Evaluate until prec' break
	MOV	BX,[NXTOPR]		; Get next operator address
	PUSH	BX			; Save next operator address
	CALL	INVSGN			; Negate value
RETNUM:
	CALL	TSTNUM			; Make sure it's a number
	POP	BX			; Restore next operator address
	RET
;
CONVAR:
	CALL	GETVAR			; Get variable address to DE
FRMEVL:
	PUSH	BX			; Save code string address
	XCHG	BX,DX			; Variable address to HL
	MOV	[FPREG],BX		; Save address of variable
	MOV	AL,[TYPE]		; Get type
	OR	AL,AL			; Numeric?
	JNZ	FRMEVL1
	CALL	PHLTFP			; Yes - Move contents to FPREG
FRMEVL1:
	POP	BX			; Restore code string address
	RET
;
FNOFST:
	MOV	CH,0			; Get address of function
	ROL	AL,1			; Double function offset
	MOV	CL,AL			; BC = Offset in function table
	PUSH	CX			; Save adjusted token value
	CALL	GETCHR			; Get next character
	MOV	AL,CL			; Get adjusted token value
	CMP	AL,2*(ZLEFT-ZSGN)-1	; Adj' LEFT$,RIGHT$ or MID$ ?
	JC	FNVAL			; No - Do function
	CALL	OPNPAR			; Evaluate expression  (X,...
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
	CALL	TSTSTR			; Make sure it's a string
	XCHG	BX,DX			; Save code string address
	MOV	BX,[FPREG]		; Get address of string
	MOV	BP,SP
	XCHG	[BP],BX			; Save address of string
	PUSH	BX			; Save adjusted token value
	XCHG	BX,DX			; Restore code string address
	CALL	GETINT			; Get integer 0-255
	XCHG	BX,DX			; Save code string address
	MOV	BP,SP
	XCHG	[BP],BX			; Save integer,HL = adj' token
	JMP	GOFUNC			; Jump to string function
;
FNVAL:
	CALL	EVLPAR			; Evaluate expression
	MOV	BP,SP
	XCHG	[BP],BX			; HL = Adjusted token value
	MOV	DX,RETNUM		; Return number from function
	PUSH	DX			; Save on stack
GOFUNC:
	MOV	CX,FNCTAB		; Function routine address
	ADD	BX,CX			; Point to right address
	MOV	CL,[BX]			; Get LSB of address
	LAHF
	INC	BX
	SAHF
	MOV	BH,[BX]			; Get MSB of address
	MOV	BL,CL			; Address to HL
	PUSH	BX
	RET
;	JMP	[BX]			; Jump to function
;
SGNEXP:
	DEC	DH			; Dec to flag negative exponent
	CMP	AL,ZMINUS		; '-' token ?
	JNZ	SGNEXP1
	RET				; Yes - Return
SGNEXP1:
	CMP	AL,'-'			; '-' ASCII ?
	JNZ	SGNEXP2
	RET				; Yes - Return
SGNEXP2:
	INC	DH			; nc to flag positive exponent
	CMP	AL,'+'			; '+' ASCII ?
	JNZ	SGNEXP3
	RET				; Yes - Return
SGNEXP3:
	CMP	AL,ZPLUS		; '+' token ?
	JNZ	SGNEXP4
	RET				; Yes - Return
SGNEXP4:
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	RET				; Return "NZ"
;
POR:
	OR	AL,0AFH			; Flag "OR"
	JMP	PAND1
PAND:
	XOR	AL,AL			; Flag "AND"
PAND1:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save "AND" / "OR" flag
	XCHG	AH,AL
	CALL	TSTNUM			; Make sure it's a number
	CALL	DEINT			; Get integer -32768 to 32767
	POP	AX			; Restore "AND" / "OR" flag
	XCHG	AH,AL
	SAHF
	XCHG	BX,DX			; <- Get last
	POP	CX			; <- value
	MOV	BP,SP
	XCHG	[BP],BX			; <- from
	XCHG	BX,DX			; <- stack
	CALL	FPBCDE			; Move last value to FPREG
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save "AND" / "OR" flag
	XCHG	AH,AL
	CALL	DEINT			; Get integer -32768 to 32767
	POP	AX			; Restore "AND" / "OR" flag
	XCHG	AH,AL
	SAHF
	POP	CX			; Get value
	MOV	AL,CL			; Get LSB
	MOV	BX,ACPASS		; Address of save AC as current
	JNZ	POR1			; Jump if OR
	AND	AL,DL			; "AND" LSBs
	MOV	CL,AL			; Save LSB
	MOV	AL,CH			; Get MBS
	AND	AL,DH			; "AND" MSBs
	PUSH	BX
	RET
;	JMP	[BX]			; Save AC as current (ACPASS)
;
POR1:
	OR	AL,DL			; "OR" LSBs
	MOV	CL,AL			; Save LSB
	MOV	AL,CH			; Get MSB
	OR	AL,DH			; "OR" MSBs
	PUSH	BX
	RET
;	JMP	[BX]			; Save AC as current (ACPASS)
;
TSTRED:
	MOV	BX,CMPLOG		; Logical compare routi
	MOV	AL,[TYPE]		; Get data type
	RCR	AL,1			; Carry set = string
	MOV	AL,DH			; Get last precedence value
	RCL	AL,1			; Times 2 plus carry
	MOV	DL,AL			; To E
	MOV	DH,64H			; Relational precedence
	MOV	AL,CH			; Get current precedence
	CMP	AL,DH			; Compare with last
	JC	TSTRED1
	RET				; Eval if last was rel' or log'
TSTRED1:
	JMP	STKTHS			; Stack this one and get next
;
CMPLOG:
	DW	CMPLG1			; Compare two values / strings
CMPLG1:
	MOV	AL,CL			; Get data type
	OR	AL,AL
	RCR	AL,1
	POP	CX			; Get last expression to BCDE
	POP	DX
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save status
	XCHG	AH,AL
	CALL	CHKTYP			; Check that types match
	MOV	BX,CMPRES		; Result to comparison
	PUSH	BX			; Save for RETurn
	JNZ	CMPLG2
	JMP	CMPNUM			; Compare values if numeric
CMPLG2:
	XOR	AL,AL			; Compare two strings
	MOV	[TYPE],AL		; Set type to numeric
	PUSH	DX			; Save string name
	CALL	GSTRCU			; Get current string
	MOV	AL,[BX]			; Get length of string
	LAHF
	INC	BX
	SAHF
	LAHF
	INC	BX
	SAHF
	MOV	CL,[BX]			; Get LSB of address
	LAHF
	INC	BX
	SAHF
	MOV	CH,[BX]			; Get MSB of address
	POP	DX			; Restore string name
	PUSH	CX			; Save address of string
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save length of string
	XCHG	AH,AL
	CALL	GSTRDE			; Get second string
	CALL	LOADFP			; Get address of second string
	POP	AX			; Restore length of string 1
	XCHG	AH,AL
	SAHF
	MOV	DH,AL			; Length to D
	POP	BX			; Restore address of string 1
CMPSTR:
	MOV	AL,DL			; Bytes of string 2 to do
	OR	AL,DH			; Bytes of string 1 to do
	JNZ	CMPSTR1
	RET				; Exit if all bytes compared
CMPSTR1:
	MOV	AL,DH			; Get bytes of string 1 to do
	SUB	AL,1
	JNC	CMPSTR2
	RET				; Exit if end of string 1
CMPSTR2:
	XOR	AL,AL
	CMP	AL,DL			; Bytes of string 2 to do
	INC	AL
	JC	CMPSTR3
	RET				; Exit if end of string 2
CMPSTR3:
	DEC	DH			; Count bytes in string 1
	DEC	DL			; Count bytes in string 2
	XCHG	BX,CX
	MOV	AL,[BX]			; Byte in string 2
	XCHG	BX,CX
	CMP	AL,[BX]			; Compare to byte in string 1
	LAHF
	INC	BX			; Move up string 1
	SAHF
	LAHF
	INC	CX			; Move up string 2
	SAHF
	JZ	CMPSTR			; Same - Try next bytes
	CMC				; Flag difference (">" or "<")
	JMP	FLGDIF			; "<" gives -1 , ">" given +1
;
CMPRES:
	INC	AL			; Increment current value
	ADC	AL,AL			; Double plus carry
	POP	CX			; Get other value
	AND	AL,CH			; Combine them
	ADD	AL,-1			; Carry set if different
	SBB	AL,AL			; 00 - Equal , FF - Different
	JMP	FLGREL			; Set current value & continue
;
EVNOT:
	MOV	DH,5AH			; Precedence value for "NOT"
	CALL	EVAL1			; Eval until precedence break
	CALL	TSTNUM			; Make sure it's a number
	CALL	DEINT			; Get integer -32768 - 32767
	MOV	AL,DL			; Get LSB
	NOT	AL			; Invert LSB
	MOV	CL,AL			; Save "NOT" of LSB
	MOV	AL,DH			; Get MSB
	NOT	AL			; Invert MSB
	CALL	ACPASS			; Save AC as current
	POP	CX			; Clean up stack
	JMP	EVAL3			; Continue evaluation
;
DIMRET:
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
	JNZ	DIMRET1
	RET				; End of DIM statement
DIMRET1:
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
DIM:
	MOV	CX,DIMRET		; Return to "DIMRET"
	PUSH	CX			; Save on stack
	OR	AL,0AFH			; Flag "Create" variable
	JMP	GETVAR1
GETVAR:
	XOR	AL,AL			; Find variable address to DE
GETVAR1:
	MOV	[LCRFLG],AL		; Set locate / create flag
	MOV	CH,[BX]			; Get First byte of name
GTFNAM:
	CALL	CHKLTR			; See if a letter
	JNC	GTFNAM1
	JMP	SNERR			; ?SN Error if not a letter
GTFNAM1:
	XOR	AL,AL
	MOV	CL,AL			; Clear second byte of name
	MOV	[TYPE],AL		; Set type to numeric
	CALL	GETCHR			; Get next character
	JC	SVNAM2			; Numeric - Save in name
	CALL	CHKLTR			; See if a letter
	JC	CHARTY			; Not a letter - Check type
SVNAM2:
	MOV	CL,AL			; Save second byte of name
ENDNAM:
	CALL	GETCHR			; Get next character
	JC	ENDNAM			; Numeric - Get another
	CALL	CHKLTR			; See if a letter
	JNC	ENDNAM			; Letter - Get another
CHARTY:
	SUB	AL,'$'			; String variable?
	JNZ	NOTSTR			; No - Numeric variable
	INC	AL			; A = 1 (string type)
	MOV	[TYPE],AL		; Set type to string
	ROR	AL,1			; A = 80H , Flag for string
	ADD	AL,CL			; 2nd byte of name has bit 7 on
	MOV	CL,AL			; Resave second byte on name
	CALL	GETCHR			; Get next character
NOTSTR:
	MOV	AL,[FORFLG]		; Array name needed ?
	DEC	AL
	JNZ	NOTSTR1			; Yes - Get array name
	JMP	ARLDSV
NOTSTR1:
	JNS	NSCFOR			; No array with "FOR" or "FN"
	MOV	AL,[BX]			; Get byte again
	SUB	AL,'('			; Subscripted variable?
	JNZ	NSCFOR			; Yes - Sort out subscript
	JMP	SBSCPT
;
NSCFOR:
	XOR	AL,AL			; Simple variable
	MOV	[FORFLG],AL		; Clear "FOR" flag
	PUSH	BX			; Save code string address
	MOV	DH,CH			; DE = Variable name to find
	MOV	DL,CL
	MOV	BX,[FNRGNM]		; FN argument name
	CALL	CPDEHL			; Is it the FN argument?
	MOV	DX,FNARG		; Point to argument value
	JNZ	NSCFOR1
	JMP	POPHRT			; Yes - Return FN argument value
NSCFOR1:
	MOV	BX,[VAREND]		; End of variables
	XCHG	BX,DX			; Address of end of search
	MOV	BX,[PROGND]		; Start of variables address
FNDVAR:
	CALL	CPDEHL			; End of variable list table?
	JZ	CFEVAL			; Yes - Called from EVAL?
	MOV	AL,CL			; Get second byte of name
	SUB	AL,[BX]			; Compare with name in list
	LAHF
	INC	BX			; Move on to first byte
	SAHF
	JNZ	FNTHR			; Different - Find another
	MOV	AL,CH			; Get first byte of name
	SUB	AL,[BX]			; Compare with name in list
FNTHR:
	LAHF
	INC	BX			; Move on to LSB of value
	SAHF
	JZ	RETADR			; Found - Return address
	LAHF
	INC	BX			; <- Skip
	SAHF
	LAHF
	INC	BX			; <- over
	SAHF
	LAHF
	INC	BX			; <- F.P.
	SAHF
	LAHF
	INC	BX			; <- value
	SAHF
	JMP	FNDVAR			; Keep looking
;
CFEVAL:
	POP	BX			; Restore code string address
	MOV	BP,SP
	XCHG	[BP],BX			; Get return address
	PUSH	DX			; Save address of variable
	MOV	DX,FRMEVL		; Return address in EVAL
	CALL	CPDEHL			; Called from EVAL ?
	POP	DX			; Restore address of variable
	JZ	RETNUL			; Yes - Return null variable
	MOV	BP,SP
	XCHG	[BP],BX			; Put back return
	PUSH	BX			; Save code string address
	PUSH	CX			; Save variable name
	MOV	CX,6			; 2 byte name plus 4 by data
	MOV	BX,[ARREND]		; End of arrays
	PUSH	BX			; Save end of arrays
	ADD	BX,CX			; Move up 6 bytes
	POP	CX			; Source address in BC
	PUSH	BX			; Save new end address
	CALL	MOVUP			; Move arrays up
	POP	BX			; Restore new end address
	MOV	[ARREND],BX		; Set new end address
	MOV	BX,CX			; End of variables to HL
	MOV	[VAREND],BX		; Set new end address
;
ZEROLP:
	LAHF
	DEC	BX			; Back through to zero variable
	SAHF
	MOV	BYTE PTR [BX],0		; Zero byte in variable
	CALL	CPDEHL			; Done them all?
	JNZ	ZEROLP			; No - Keep on going
	POP	DX			; Get variable name
	MOV	[BX],DL			; Store second character
	LAHF
	INC	BX
	SAHF
	MOV	[BX],DH			; Store first character
	LAHF
	INC	BX
	SAHF
RETADR:
	XCHG	BX,DX			; Address of variable in DE
	POP	BX			; Restore code string address
	RET
;
RETNUL:
	MOV	[FPEXP],AL		; Set result to zero
	MOV	BX,ZERBYT		; Also set a null string
	MOV	[FPREG],BX		; Save for EVAL
	POP	BX			; Restore code string address
	RET
;
SBSCPT:
	PUSH	BX			; Save code string address
	MOV	BX,[LCRFLG]		; Locate/Create and Type
	MOV	BP,SP
	XCHG	[BP],BX			; Save and get code string
	MOV	DH,AL			; Zero number of dimensions
SCPTLP:
	PUSH	DX			; Save number of dimensions
	PUSH	CX			; Save array name
	CALL	FPSINT			; Get subscript (0-32767)
	POP	CX			; Restore array name
	POP	AX			; Get number of dimensions
	XCHG	AH,AL
	SAHF
	XCHG	BX,DX
	MOV	BP,SP
	XCHG	[BP],BX			; Save subscript value
	PUSH	BX			; Save LCRFLG and TYPE
	XCHG	BX,DX
	INC	AL			; Count dimensions
	MOV	DH,AL			; Save in D
	MOV	AL,[BX]			; Get next byte in code string
	CMP	AL,','			; Comma (more to come)?
	JZ	SCPTLP			; Yes - More subscripts
	CALL	CHKSYN			; Make sure ")" follows
	DB	")"
	MOV	[NXTOPR],BX		; Save code string address
	POP	BX			; Get LCRFLG and TYPE
	MOV	[LCRFLG],BX		; Restore Locate/create & type
	MOV	DL,0			; Flag not CSAVE* or CLOAD*
	PUSH	DX			; Save number of dimensions (D)
	JMP	ARLDSV1			; Skip "PUSH HL" and "PUSH AF"
;
ARLDSV:
	PUSH	BX			; Save code string address
	LAHF
	XCHG	AH,AL
	PUSH	AX			; A = 00 , Flags set = Z,N
	XCHG	AH,AL
ARLDSV1:
	MOV	BX,[VAREND]		; Start of arrays
	JMP	FNDARY1			; Skip "ADD HL,DE"
FNDARY:
	ADD	BX,DX			; Move to next array start
FNDARY1:
	XCHG	BX,DX
	MOV	BX,[ARREND]		; End of arrays
	XCHG	BX,DX			; Current array pointer
	CALL	CPDEHL			; End of arrays found?
	JZ	CREARY			; Yes - Create array
	MOV	AL,[BX]			; Get second byte of name
	CMP	AL,CL			; Compare with name given
	LAHF
	INC	BX			; Move on
	SAHF
	JNZ	NXTARY			; Different - Find next array
	MOV	AL,[BX]			; Get first byte of name
	CMP	AL,CH			; Compare with name given
NXTARY:
	LAHF
	INC	BX			; Move on
	SAHF
	MOV	DL,[BX]			; Get LSB of next array address
	LAHF
	INC	BX
	SAHF
	MOV	DH,[BX]			; Get MSB of next array address
	LAHF
	INC	BX
	SAHF
	JNZ	FNDARY			; Not found - Keep looking
	MOV	AL,[LCRFLG]		; Found Locate or Creat it?
	OR	AL,AL
	JZ	NXTARY1			; Create - ?DD Error
	JMP	DDERR
NXTARY1:
	POP	AX			; Locate - Get number of dimensions
	XCHG	AH,AL
	SAHF
	MOV	CX,BX			; BC Points to array dimensions
	JNZ	NXTARY2			; Jump if array load/save
	JMP	POPHRT
NXTARY2:
	SUB	AL,[BX]			; Same number of dimensions?
	JNZ	BSERR			; Yes - Find element
	JMP	FINDEL
BSERR:
	MOV	DL,BS			; ?BS Error
	JMP	ERROR			; Output error
;
CREARY:
	MOV	DX,4			; 4 Bytes per entry
	POP	AX			; Array to save or 0 dimensions?
	XCHG	AH,AL
	SAHF
	JNZ	CREARY1
	JMP	FCERR			; Yes - ?FC Error
CREARY1:
	MOV	[BX],CL			; Save second byte of n
	LAHF
	INC	BX
	SAHF
	MOV	[BX],CH			; Save first byte of name
	LAHF
	INC	BX
	SAHF
	MOV	CL,AL			; Number of dimensions to C
	CALL	CHKSTK			; Check if enough memory
	LAHF
	INC	BX			; Point to number of dimensions
	SAHF
	LAHF
	INC	BX
	SAHF
	MOV	[CUROPR],BX		; Save address of pointer
	MOV	[BX],CL			; Set number of dimensions
	LAHF
	INC	BX
	SAHF
	MOV	AL,[LCRFLG]		; Locate of Create?
	RCL	AL,1			; Carry set = Create
	MOV	AL,CL			; Get number of dimensions
CRARLP:
	MOV	CX,10+1			; Default dimension size 10
	JNC	DEFSIZ			; Locate - Set default size
	POP	CX			; Get specified dimension size
	LAHF
	INC	CX			; Include zero element
	SAHF
DEFSIZ:
	MOV	[BX],CL			; Save LSB of dimension size
	LAHF
	INC	BX
	SAHF
	MOV	[BX],CH			; Save MSB of dimension size
	LAHF
	INC	BX
	SAHF
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save num' of dimensions an status
	XCHG	AH,AL
	PUSH	BX			; Save address of dimensions size
	CALL	MLDEBC			; Multiply DE by BC to find
	XCHG	BX,DX			; amount of mem needed (to DE)
	POP	BX			; Restore address of dimension
	POP	AX			; Restore number of dimensions
	XCHG	AH,AL
	SAHF
	DEC	AL			; Count them
	JNZ	CRARLP			; Do next dimension if more
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save locate/create flag
	XCHG	AH,AL
	MOV	CX,DX			; memory needed
	XCHG	BX,DX
	ADD	BX,DX			; Add bytes to array start
	JNC	DEFSIZ1
	JMP	OMERR			; Too big - Error
DEFSIZ1:
	CALL	ENFMEM			; See if enough memory
	MOV	[ARREND],BX		; Save new end of array
;
ZERARY:
	LAHF
	DEC	BX			; Back through array data
	SAHF
	MOV	BYTE PTR [BX],0		; Set array element to zero
	CALL	CPDEHL			; All elements zeroed?
	JNZ	ZERARY			; No - Keep on going
	LAHF
	INC	CX			; Number of bytes + 1
	SAHF
	MOV	DH,AL			; A=0
	MOV	BX,[CUROPR]		; Get address of array
	MOV	DL,[BX]			; Number of dimensions
	XCHG	BX,DX			; To HL
	ADD	BX,BX			; Two bytes per dimension size
	ADD	BX,CX			; Add number of bytes
	XCHG	BX,DX			; Bytes needed to DE
	LAHF
	DEC	BX
	SAHF
	LAHF
	DEC	BX
	SAHF
	MOV	[BX],DL			; Save LSB of bytes needed
	LAHF
	INC	BX
	SAHF
	MOV	[BX],DH			; Save MSB of bytes needed
	LAHF
	INC	BX
	SAHF
	POP	AX			; Locate / Create?
	XCHG	AH,AL
	SAHF
	JC	ENDDIM			; A is 0 , End if create
FINDEL:
	MOV	CH,AL			; Find array element
	MOV	CL,AL
	MOV	AL,[BX]			; Number of dimensions
	LAHF
	INC	BX
	SAHF
	JMP	FNDELP1			; Skip "POP BX"
FNDELP:
	POP	BX			; Address of next dim' size
FNDELP1:
	MOV	DL,[BX]			; Get LSB of dim'n size
	LAHF
	INC	BX
	SAHF
	MOV	DH,[BX]			; Get MSB of dim'n size
	LAHF
	INC	BX
	SAHF
	MOV	BP,SP
	XCHG	[BP],BX			; Save address - Get index
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save number of dimensions
	XCHG	AH,AL
	CALL	CPDEHL			; Dimension too large?
	JC	FNDELP2
	JMP	BSERR			; Yes - ?BS Error
FNDELP2:
	PUSH	BX			; Save index
	CALL	MLDEBC			; Multiply previous by size
	POP	DX			; Index supplied to DE
	ADD	BX,DX			; Add index to pointer
	POP	AX			; Number of dimensions
	XCHG	AH,AL
	SAHF
	DEC	AL			; Count them
	MOV	CX,BX			; pointer
	JNZ	FNDELP			; More - Keep going
	ADD	BX,BX			; 4 Bytes per element
	ADD	BX,BX
	POP	CX			; Start of array
	ADD	BX,CX			; Point to element
	XCHG	BX,DX			; Address of element to DE
ENDDIM:
	MOV	BX,[NXTOPR]		; Got code string address
	RET
;
FRE:
	MOV	BX,[ARREND]		; Start of free memory
	XCHG	BX,DX			; To DE
	MOV	BX,0			; End of free memory
	ADD	BX,SP			; Current stack value
	MOV	AL,[TYPE]		; Dummy argument type
	OR	AL,AL
	JZ	FRENUM			; Numeric - Free variable space
	CALL	GSTRCU			; Current string to pool
	CALL	GARBGE			; Garbage collection
	MOV	BX,[STRSPC]		; Bottom of string space in use
	XCHG	BX,DX			; To DE
	MOV	BX,[STRBOT]		; Bottom of string space
FRENUM:
	MOV	AL,BL			; Get LSB of end
	SUB	AL,DL			; Subtract LSB of beginning
	MOV	CL,AL			; Save difference if C
	MOV	AL,BH			; Get MSB of end
	SBB	AL,DH			; Subtract MSB of beginning
ACPASS:
	MOV	CH,CL			; Return integer AC
ABPASS:
	MOV	DH,CH			; Return integer AB
	MOV	DL,0
	MOV	BX,TYPE			; Point to type
	MOV	[BX],DL			; Set type to numeric
	MOV	CH,80H+16		; 16 bit integer
	JMP	RETINT			; Return the integr
;
POS:
	MOV	AL,[CURPOS]		; Get cursor position
PASSA:
	MOV	CH,AL			; Put A into AB
	XOR	AL,AL			; Zero A
	JMP	ABPASS			; Return integer AB
;
DEF:
	CALL	CHEKFN			; Get "FN" and name
	CALL	IDTEST			; Test for illegal direct
	MOV	CX,DATA			; To get next statement
	PUSH	CX			; Save address for RETurn
	PUSH	DX			; Save address of function ptr
	CALL	CHKSYN			; Make sure "(" follows
	DB	"("
	CALL	GETVAR			; Get argument variable name
	PUSH	BX			; Save code string address
	XCHG	BX,DX			; Argument address to HL
	LAHF
	DEC	BX
	SAHF
	MOV	DH,[BX]			; Get first byte of arg name
	LAHF
	DEC	BX
	SAHF
	MOV	DL,[BX]			; Get second byte of ar
	POP	BX			; Restore code string address
	CALL	TSTNUM			; Make sure numeric argument
	CALL	CHKSYN			; Make sure ")" follows
	DB	")"
	CALL	CHKSYN			; Make sure "=" follows
	DB	ZEQUAL			; "=" token
	MOV	CX,BX			; Code string address to BC
	MOV	BP,SP
	XCHG	[BP],BX			; Save code str , Get FN ptr
	MOV	[BX],CL			; Save LSB of FN code string
	LAHF
	INC	BX
	SAHF
	MOV	[BX],CH			; Save MSB of FN code string
	JMP	SVSTAD			; Save address and do function
;
DOFN:
	CALL	CHEKFN			; Make sure FN follows
	PUSH	DX			; Save function pointer address
	CALL	EVLPAR			; Evaluate expression in "()"
	CALL	TSTNUM			; Make sure numeric result
	MOV	BP,SP
	XCHG	[BP],BX			; Save code str , Get FN ptr
	MOV	DL,[BX]			; Get LSB of FN code string
	LAHF
	INC	BX
	SAHF
	MOV	DH,[BX]			; Get MSB of FN code string
	LAHF
	INC	BX
	SAHF
	MOV	AL,DH			; And function DEFined?
	OR	AL,DL
	JNZ	DOFN1
	JMP	UFERR			; No - ?UF Error
DOFN1:
	MOV	AL,[BX]			; Get LSB of argument address
	LAHF
	INC	BX
	SAHF
	MOV	BH,[BX]			; Get MSB of argument address
	MOV	BL,AL			; HL = Arg variable address
	PUSH	BX			; Save it
	MOV	BX,[FNRGNM]		; Get old argument name
	MOV	BP,SP
	XCHG	[BP],BX			; Save old , Get new
	MOV	[FNRGNM],BX		; Set new argument name
	MOV	BX,[FNARG+2]		; Get LSB,NLSB of old arg value
	PUSH	BX			; Save it
	MOV	BX,[FNARG]		; Get MSB,EXP of old arg value
	PUSH	BX			; Save it
	MOV	BX,FNARG		; HL = Value of argument
	PUSH	DX			; Save FN code string address
	CALL	FPTHL			; Move FPREG to argument
	POP	BX			; Get FN code string address
	CALL	GETNUM			; Get value from function
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
	JZ	DOFN2
	JMP	SNERR			; Bad character in FN - Error
DOFN2:
	POP	BX			; Get MSB,EXP of old arg
	MOV	[FNARG],BX		; Restore it
	POP	BX			; Get LSB,NLSB of old arg
	MOV	[FNARG+2],BX		; Restore it
	POP	BX			; Get name of old arg
	MOV	[FNRGNM],BX		; Restore it
	POP	BX			; Restore code string address
	RET
;
IDTEST:
	PUSH	BX			; Save code string address
	MOV	BX,[LINEAT]		; Get current line number
	LAHF
	INC	BX			; -1 means direct statement
	SAHF
	MOV	AL,BH
	OR	AL,BL
	POP	BX			; Restore code string address
	JZ	IDTEST1
	RET				; Return if in program
IDTEST1:
	MOV	DL,ID			; ?ID Error
	JMP	ERROR
;
CHEKFN:
	CALL	CHKSYN			; Make sure FN follows
	DB	ZFN			; "FN" token
	MOV	AL,80H
	MOV	[FORFLG],AL		; Flag FN name to find
	OR	AL,[BX]			; FN name has bit 7 set
	MOV	CH,AL			; in first byte of name
	CALL	GTFNAM			; Get FN name
	JMP	TSTNUM			; Make sure numeric function
;
STR:
	CALL	TSTNUM			; Make sure it's a number
	CALL	NUMASC			; Turn number into text
STR1:
	CALL	CRTST			; Create string entry for it
	CALL	GSTRCU			; Current string to pool
	MOV	CX,TOPOOL		; Save in string pool
	PUSH	CX			; Save address on stack
;
SAVSTR:
	MOV	AL,[BX]			; Get string length
	LAHF
	INC	BX
	SAHF
	LAHF
	INC	BX
	SAHF
	PUSH	BX			; Save pointer to string
	CALL	TESTR			; See if enough string space
	POP	BX			; Restore pointer to string
	MOV	CL,[BX]			; Get LSB of address
	LAHF
	INC	BX
	SAHF
	MOV	CH,[BX]			; Get MSB of address
	CALL	CRTMST			; Create string entry
	PUSH	BX			; Save pointer to MSB of addr
	MOV	BL,AL			; Length of string
	CALL	TOSTRA			; Move to string area
	POP	DX			; Restore pointer to MSB
	RET
;
MKTMST:
	CALL	TESTR			; See if enough string space
CRTMST:
	MOV	BX,TMPSTR		; Temporary string
	PUSH	BX			; Save it
	MOV	[BX],AL			; Save length of string
	LAHF
	INC	BX
	SAHF
SVSTAD:
	LAHF
	INC	BX
	SAHF
	MOV	[BX],DL			; Save LSB of address
	LAHF
	INC	BX
	SAHF
	MOV	[BX],DH			; Save MSB of address
	POP	BX			; Restore pointer
	RET
;
CRTST:
	LAHF
	DEC	BX			; DEC - INCed after
	SAHF
QTSTR:
	MOV	CH,'"'			; Terminating quote
	MOV	DH,CH			; Quote to D
DTSTR:
	PUSH	BX			; Save start
	MOV	CL,-1			; Set counter to -1
QTSTLP:
	LAHF
	INC	BX			; Move on
	SAHF
	MOV	AL,[BX]			; Get byte
	INC	CL			; Count bytes
	OR	AL,AL			; End of line?
	JZ	CRTSTE			; Yes - Create string entry
	CMP	AL,DH			; Terminator D found?
	JZ	CRTSTE			; Yes - Create string entry
	CMP	AL,CH			; Terminator B found?
	JNZ	QTSTLP			; No - Keep looking
CRTSTE:
	CMP	AL,'"'			; End with '"'?
	JNZ	CRTSTE1
	CALL	GETCHR			; Yes - Get next charac
CRTSTE1:
	MOV	BP,SP
	XCHG	[BP],BX			; Starting quote
	LAHF
	INC	BX			; First byte of string
	SAHF
	XCHG	BX,DX			; To DE
	MOV	AL,CL			; Get length
	CALL	CRTMST			; Create string entry
TSTOPL:
	MOV	DX,TMPSTR		; Temporary string
	MOV	BX,[TMSTPT]		; Temporary string pool pointer
	MOV	[FPREG],BX		; Save address of string ptr
	MOV	AL,1
	MOV	[TYPE],AL		; Set type to string
	CALL	DETHL4			; Move string to pool
	CALL	CPDEHL			; Out of string pool?
	MOV	[TMSTPT],BX		; Save new pointer
	POP	BX			; Restore code string address
	MOV	AL,[BX]			; Get next code byte
	JZ	TSTOPL1
	RET				; Return if pool OK
TSTOPL1:
	MOV	DL,ST			; ?ST Error
	JMP	ERROR			; String pool overflow
;
PRNUMS:
	LAHF
	INC	BX			; Skip leading space
	SAHF
PRS:
	CALL	CRTST			; Create string entry for it
PRS1:
	CALL	GSTRCU			; Current string to pool
	CALL	LOADFP			; Move string block to BCDE
	INC	DL			; Length + 1
PRSLP:
	DEC	DL			; Count characters
	JNZ	PRSLP1
	RET				; End of string
PRSLP1:
	XCHG	BX,CX
	MOV	AL,[BX]			; Get byte to output
	XCHG	BX,CX
	CALL	OUTC			; Output character in A
	CMP	AL,CR			; Return?
	JNZ	PRSLP2
	CALL	DONULL			; Yes - Do nulls
PRSLP2:
	LAHF
	INC	CX			; Next byte in string
	SAHF
	JMP	PRSLP			; More characters to output
;
TESTR:
	OR	AL,AL			; Test if enough room
	JMP	GRBDON1			; No garbage collection done
GRBDON:
	POP	AX			; Garbage collection done
	XCHG	AH,AL
	SAHF
GRBDON1:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save status
	XCHG	AH,AL
	MOV	BX,[STRSPC]		; Bottom of string space in use
	XCHG	BX,DX			; To DE
	MOV	BX,[STRBOT]		; Bottom of string area
	NOT	AL			; Negate length (Top down)
	MOV	CL,AL			; -Length to BC
	MOV	CH,-1			; BC = -ve length of string
	ADD	BX,CX			; Add to bottom of space in use
;	LAHF
	INC	BX			; Plus one for 2's complement
;	SAHF
	CALL	CPDEHL			; Below string RAM area?
	JC	TESTOS			; Tidy up if not done else err
	MOV	[STRBOT],BX		; Save new bottom of area
	LAHF
	INC	BX			; Point to first byte of string
	SAHF
	XCHG	BX,DX			; Address to DE
POPAF:
	POP	AX			; Throw away status push
	XCHG	AH,AL
	SAHF
	RET
;
TESTOS:
	POP	AX			; Garbage collect been done?
	XCHG	AH,AL
	SAHF
	MOV	DL,OS			; ?OS Error
	JNZ	TESTOS1
	JMP	ERROR			; Yes - Not enough strig space
TESTOS1:
	CMP	AL,AL			; Flag garbage collect done
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save status
	XCHG	AH,AL
	MOV	CX,GRBDON		; Garbage collection done
	PUSH	CX			; Save for RETurn
GARBGE:
	MOV	BX,[LSTRAM]		; Get end of RAM pointer
GARBLP:
	MOV	[STRBOT],BX		; Reset string pointer
	MOV	BX,0
	PUSH	BX			; Flag no string found
	MOV	BX,[STRSPC]		; Get bottom of string space
	PUSH	BX			; Save bottom of string space
	MOV	BX,TMSTPL		; Temporary string pool
GRBLP:
	XCHG	BX,DX
	MOV	BX,[TMSTPT]		; Temporary string pool pointer
	XCHG	BX,DX
	CALL	CPDEHL			; Temporary string pool done?
	MOV	CX,GRBLP		; Loop until string pool done
	JNZ	STPOOL			; No - See if in string area
	MOV	BX,[PROGND]		; Start of simple variables
SMPVAR:
	XCHG	BX,DX
	MOV	BX,[VAREND]		; End of simple variables
	XCHG	BX,DX
	CALL	CPDEHL			; All simple strings done?
	JZ	ARRLP			; Yes - Do string arrays
	MOV	AL,[BX]			; Get type of variable
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	OR	AL,AL			; "S" flag set if strings done?
	CALL	STRADD			; See if string in strig area
	JMP	SMPVAR			; Loop until simple ones done
;
GNXARY:
	POP	CX			; Scrap address of this array
ARRLP:
	XCHG	BX,DX
	MOV	BX,[ARREND]		; End of string arrays
	XCHG	BX,DX
	CALL	CPDEHL			; All string arrays done?
	JZ	SCNEND			; Yes - Move string if found
	CALL	LOADFP			; Get array name to BCDE
	MOV	AL,DL			; Get type of array
	PUSH	BX			; Save address of num of dimensions
	ADD	BX,CX			; Start of next array
	OR	AL,AL			; Test type of array
	JNS	GNXARY			; Numeric array - Ignore it
	MOV	[CUROPR],BX		; Save address of next array
	POP	BX			; Get address of num of dimensions
	MOV	CL,[BX]			; BC = Number of dimensions
	MOV	CH,0
	ADD	BX,CX			; Two bytes per dimension size
	ADD	BX,CX
	LAHF
	INC	BX			; Plus one for number of dimensions
	SAHF
GRBARY:
	XCHG	BX,DX
	MOV	BX,[CUROPR]		; Get address of next array
	XCHG	BX,DX
	CALL	CPDEHL			; Is this array finished?
	JZ	ARRLP			; Yes - Get next one
	MOV	CX,GRBARY		; Loop until array all done
STPOOL:
	PUSH	CX			; Save return address
	OR	AL,80H			; Flag string type
STRADD:
	MOV	AL,[BX]			; Get string length
	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	DL,[BX]			; Get LSB of string address
;	LAHF
	INC	BX
;	SAHF
	MOV	DH,[BX]			; Get MSB of string address
;	LAHF
	INC	BX
	SAHF
	JS	STRADD1
	RET				; Not a string - Return
STRADD1:
	OR	AL,AL			; Set flags on string length
	JNZ	STRADD2
	RET				; Null string - Return
STRADD2:
	MOV	CX,BX			; Save variable pointer
	MOV	BX,[STRBOT]		; Bottom of new area
	CALL	CPDEHL			; String been done?
	MOV	BX,CX			; Restore variable pointer
	JNC	STRADD3
	RET				; String done - Ignore
STRADD3:
	POP	BX			; Return address
	MOV	BP,SP
	XCHG	[BP],BX			; Lowest available string area
	CALL	CPDEHL			; String within string area?
	MOV	BP,SP
	XCHG	[BP],BX			; Lowest available string area
	PUSH	BX			; Re-save return address
	MOV	BX,CX			; Restore variable pointer
	JC	STRADD4
	RET				; Outside string area - Ignore
STRADD4:
	POP	CX			; Get return , Throw 2 away
	POP	AX
;	XCHG	AH,AL
;	SAHF
	POP	AX
	XCHG	AH,AL
	SAHF
	PUSH	BX			; Save variable pointer
	PUSH	DX			; Save address of current
	PUSH	CX			; Put back return address
	RET				; Go to it
;
SCNEND:
	POP	DX			; Addresses of strings
	POP	BX			;
	MOV	AL,BL			; HL = 0 if no more to do
	OR	AL,BH
	JNZ	SCNEND1
	RET				; No more to do - Return
SCNEND1:
	LAHF
	DEC	BX
	SAHF
	MOV	CH,[BX]			; MSB of address of string
	LAHF
	DEC	BX
	SAHF
	MOV	CL,[BX]			; LSB of address of string
	PUSH	BX			; Save variable address
	LAHF
	DEC	BX
	SAHF
	LAHF
	DEC	BX
	SAHF
	MOV	BL,[BX]			; HL = Length of string
	MOV	BH,0
	ADD	BX,CX			; Address of end of string
	MOV	DH,CH			; String address to DE
	MOV	DL,CL
	LAHF
	DEC	BX			; Last byte in string
	SAHF
	MOV	CX,BX			; Address to BC
	MOV	BX,[STRBOT]		; Current bottom of string area
	CALL	MOVSTR			; Move string to new address
	POP	BX			; Restore variable address
	MOV	[BX],CL			; Save new LSB of address
	LAHF
	INC	BX
	SAHF
	MOV	[BX],CH			; Save new MSB of address
	MOV	BX,CX			; Next string area+1 to HL
	LAHF
	DEC	BX
	SAHF				; Next string area address
	JMP	GARBLP			; Look for more strings
;
CONCAT:
	PUSH	CX			; Save prec' opr & code string
	PUSH	BX			;
	MOV	BX,[FPREG]		; Get first string
	MOV	BP,SP
	XCHG	[BP],BX			; Save first string
	CALL	OPRND			; Get second string
	MOV	BP,SP
	XCHG	[BP],BX			; Restore first string
	CALL	TSTSTR			; Make sure it's a string
	MOV	AL,[BX]			; Get length of second string
	PUSH	BX			; Save first string
	MOV	BX,[FPREG]		; Get second string
	PUSH	BX			; Save second string
	ADD	AL,[BX]			; Add length of second string
	MOV	DL,LS			; ?LS Error
	JNC	CONCAT1			; String too long - Error
	JMP	ERROR
CONCAT1:
	CALL	MKTMST			; Make temporary string
	POP	DX			; Get second string to DE
	CALL	GSTRDE			; Move to string pool if needed
	MOV	BP,SP
	XCHG	[BP],BX			; Get first string
	CALL	GSTRHL			; Move to string pool if needed
	PUSH	BX			; Save first string
	MOV	BX,[TMPSTR+2]		; Temporary string address
	XCHG	BX,DX			; To DE
	CALL	SSTSA			; First string to string area
	CALL	SSTSA			; Second string to strig area
	MOV	BX,EVAL2		; Return to evaluation loop
	MOV	BP,SP
	XCHG	[BP],BX			; Save return,get code string
	PUSH	BX			; Save code string address
	JMP	TSTOPL			; To temporary string to pool
;
SSTSA:
	POP	BX			; Return address
	MOV	BP,SP
	XCHG	[BP],BX			; Get string block,save return
	MOV	AL,[BX]			; Get length of string
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	CL,[BX]			; Get LSB of string address
;	LAHF
	INC	BX
;	SAHF
	MOV	CH,[BX]			; Get MSB of string address
	MOV	BL,AL			; Length to L
TOSTRA:
	INC	BL			; INC - DECed after
TSALP:
	DEC	BL			; Count bytes moved
	JNZ	TSALP1
	RET				; End of string - Return
TSALP1:
	XCHG	BX,CX
	MOV	AL,[BX]
	XCHG	BX,CX			; Get source
	XCHG	BX,DX
	MOV	[BX],AL
	XCHG	BX,DX			; Save destination
;	LAHF
	INC	CX			; Next source
;	SAHF
;	LAHF
	INC	DX			; Next destination
;	SAHF
	JMP	TSALP			; Loop until string moved
;
GETSTR:
	CALL	TSTSTR			; Make sure it's a strig
GSTRCU:
	MOV	BX,[FPREG]		; Get current string
GSTRHL:
	XCHG	BX,DX			; Save DE
GSTRDE:
	CALL	BAKTMP			; Was it last tmp-str?
	XCHG	BX,DX			; Restore DE
	JZ	GSTRDE1
	RET				; No - Return
GSTRDE1:
	PUSH	DX			; Save string
	MOV	DH,CH			; String block address to DE
	MOV	DL,CL
;	LAHF
	DEC	DX			; Point to length
;	SAHF
	MOV	CL,[BX]			; Get string length
	MOV	BX,[STRBOT]		; Current bottom of string area
	CALL	CPDEHL			; Last one in string area?
	JNZ	POPHL			; No - Return
	MOV	CH,AL			; Clear B (A=0)
	ADD	BX,CX			; Remove string from string area
	MOV	[STRBOT],BX		; Save new bottom of string area
POPHL:
	POP	BX			; Restore string
	RET
;
BAKTMP:
	MOV	BX,[TMSTPT]		; Get temporary string pool top
;	LAHF
	DEC	BX			; Back
;	SAHF
	MOV	CH,[BX]			; Get MSB of address
;	LAHF
	DEC	BX			; Back
;	SAHF
	MOV	CL,[BX]			; Get LSB of address
;	LAHF
	DEC	BX			; Back
;	SAHF
;	LAHF
	DEC	BX			; Back
;	SAHF
	CALL	CPDEHL			; String last in string pool?
	JZ	BAKTMP1
	RET				; Yes - Leave it
BAKTMP1:
	MOV	[TMSTPT],BX		; Save new string pool top
	RET
;
LEN:
	MOV	CX,PASSA		; To return integer A
	PUSH	CX			; Save address
GETLEN:
	CALL	GETSTR			; Get string and its length
	XOR	AL,AL
	MOV	DH,AL			; Clear D
	MOV	[TYPE],AL		; Set type to numeric
	MOV	AL,[BX]			; Get length of string
	OR	AL,AL			; Set status flags
	RET
;
ASC:
	MOV	CX,PASSA		; To return integer A
	PUSH	CX			; Save address
GTFLNM:
	CALL	GETLEN			; Get length of string
	JNZ	GTFLNM1			; Null string - Error
	JMP	FCERR
GTFLNM1:
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	DL,[BX]			; Get LSB of address
;	LAHF
	INC	BX
;	SAHF
	MOV	DH,[BX]			; Get MSB of address
	XCHG	BX,DX
	MOV	AL,[BX]
	XCHG	BX,DX			; Get first byte of string
	RET
;
CHR:
	MOV	AL,1			; One character string
	CALL	MKTMST			; Make a temporary string
	CALL	MAKINT			; Make it integer A
	MOV	BX,[TMPSTR+2]		; Get address of string
	MOV	[BX],DL			; Save character
TOPOOL:
	POP	CX			; Clean up stack
	JMP	TSTOPL			; Temporary string to pool
;
LEFT:
	CALL	LFRGNM			; Get number and ending ")"
	XOR	AL,AL			; Start at first byte in string
RIGHT1:
	MOV	BP,SP
	XCHG	[BP],BX			; Save code string,Get string
	MOV	CL,AL			; Starting position in string
MID1:
	PUSH	BX			; Save string block address
	MOV	AL,[BX]			; Get length of string
	CMP	AL,CH			; Compare with number given
	JC	ALLFOL			; All following bytes required
	MOV	AL,CH			; Get new length
	JMP	ALLFOL1			; Skip "LD C,0"
ALLFOL:
	MOV	CL,0			; First byte of string
ALLFOL1:
	PUSH	CX			; Save position in string
	CALL	TESTR			; See if enough string space
	POP	CX			; Get position in string
	POP	BX			; Restore string block address
	PUSH	BX			; And re-save it
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	CH,[BX]			; Get LSB of address
;	LAHF
	INC	BX
;	SAHF
	MOV	BH,[BX]			; Get MSB of address
	MOV	BL,CH			; HL = address of string
	MOV	CH,0			; BC = starting address
	ADD	BX,CX			; Point to that byte
	MOV	CX,BX			; BC = source string
	CALL	CRTMST			; Create a string entry
	MOV	BL,AL			; Length of new string
	CALL	TOSTRA			; Move string to string
	POP	DX			; Clear stack
	CALL	GSTRDE			; Move to string pool if needed
	JMP	TSTOPL			; Temporary string to pool
;
RIGHT:
	CALL	LFRGNM			; Get number and ending ")"
	POP	DX			; Get string length
	PUSH	DX			; And re-save
	XCHG	BX,DX
	MOV	AL,[BX]
	XCHG	BX,DX			; Get length
	SUB	AL,CH			; Move back N bytes
	JMP	RIGHT1			; Go and get sub-string
;
MID:
	XCHG	BX,DX			; Get code string addre
	MOV	AL,[BX]			; Get next byte ',' or ")"
	CALL	MIDNUM			; Get number supplied
	INC	CH			; Is it character zero?
	DEC	CH
	JNZ	MID2			; Yes - Error
	JMP	FCERR
MID2:
	PUSH	CX			; Save starting position
	MOV	DL,255				; All of string
	CMP	AL,')'			; Any length given?
	JZ	RSTSTR			; No - Rest of string
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
	CALL	GETINT			; Get integer 0-255
RSTSTR:
	CALL	CHKSYN			; Make sure ")" follows
	DB	")"
	POP	AX			; Restore starting posion
	XCHG	AH,AL
;	SAHF
	MOV	BP,SP
	XCHG	[BP],BX			; Get string,8ave code string
	MOV	CX,MID1			; Continuation of MID$ routine
	PUSH	CX			; Save for return
	DEC	AL			; Starting position-1
	CMP	AL,[BX]			; Compare with length
	MOV	CH,0			; Zero bytes length
	JC	RSTSTR1
	RET				; Null string if start past end
RSTSTR1:
	MOV	CL,AL			; Save starting position -1
	MOV	AL,[BX]			; Get length of string
	SUB	AL,CL			; Subtract start
	CMP	AL,DL			; Enough string for it?
	MOV	CH,AL			; Save maximum length available
	JNC	RSTSTR2
	RET				; Truncate string if needed
RSTSTR2:
	MOV	CH,DL			; Set specified length
	RET				; Go and create string
;
VAL:
	CALL	GETLEN			; Get length of string
	JNZ	VAL0
	JMP	RESZER			; Result zero
VAL0:
	MOV	DL,AL			; Save length
;	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
;	SAHF
	MOV	AL,[BX]			; Get LSB of address
;	LAHF
	INC	BX
;	SAHF
	MOV	BH,[BX]			; Get MSB of address
	MOV	BL,AL			; HL = String address
	PUSH	BX			; Save string address
	ADD	BX,DX
	MOV	CH,[BX]			; Get end of string+1 byte
	MOV	[BX],DH			; Zero it to terminate
	MOV	BP,SP
	XCHG	[BP],BX			; Save string end,get start
	PUSH	CX			; Save end+1 byte
	MOV	AL,[BX]			; Get starting byte
	CMP	AL,'$'			; Hex number indicated?
	JNZ	VAL1
	CALL	HEXTFP			; Convert Hex to FPREG
	JMP	VAL3
VAL1:
	CMP	AL,'%'			; Binary number indicated?
	JNZ	VAL2
	CALL	BINTFP			; Convert Bin to FPREG
	JMP	VAL3
VAL2:
	CALL	ASCTFP			; Convert ASCII string to FP
VAL3:
	POP	CX			; Restore end+1 byte
	POP	BX			; Restore end+1 address
	MOV	[BX],CH			; Put back original byte
	RET
;
LFRGNM:
	XCHG	BX,DX			; Code string address to HL
	CALL	CHKSYN			; Make sure ")" follows
	DB	")"
MIDNUM:
	POP	CX			; Get return address
	POP	DX			; Get number supplied
	PUSH	CX			; Re-save return address
	MOV	CH,DL			; Number to B
	RET
;
INP:
	CALL	MAKINT			; Make it integer A
	MOV	[INPORT],AL		; Set input port
	CALL	INPSUB			; Get input from port
	JMP	PASSA			; Return integer A
;
POUT:
	CALL	SETIO			; Set up port number
	JMP	OUTSUB			; Output data and return
;
WAIT:
	CALL	SETIO			; Set up port number
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save AND mask
	XCHG	AH,AL
	MOV	DL,0			; Assume zero if none given
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
	JZ	NOXOR			; No XOR byte given
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
	CALL	GETINT			; Get integer 0-255 to XOR with
NOXOR:
	POP	CX			; Restore AND mask
WAITLP:
	CALL	INPSUB			; Get input
	XOR	AL,DL			; Flip selected bits
	AND	AL,CH			; Result non-zero?
	JZ	WAITLP			; No = keep waiting
	RET
;
SETIO:
	CALL	GETINT			; Get integer 0-255
	MOV	[INPORT],AL		; Set input port
	MOV	[OTPORT],AL		; Set output port
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
	JMP	GETINT			; Get integer 0-255 and return
;
FNDNUM:
	CALL	GETCHR			; Get next character
GETINT:
	CALL	GETNUM			; Get a number from 0 to 255
MAKINT:
	CALL	DEPINT			; Make sure value 0 - 255
	MOV	AL,DH			; Get MSB of number
	OR	AL,AL			; Zero?
	JZ	MAKINT1			; No - Error
	JMP	FCERR
MAKINT1:
;	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
;	SAHF
	CALL	GETCHR			; Get next character
	MOV	AL,DL			; Get number to A
	RET
;
PEEK:
	CALL	DEINT			; Get memory address
	XCHG	BX,DX
	MOV	AL,[BX]			; Get byte in memory
	XCHG	BX,DX
	JMP	PASSA			; Return integer A
;
POKE:
	CALL	GETNUM			; Get memory address
	CALL	DEINT			; Get integer -32768 to 32767
	PUSH	DX			; Save memory address
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
	CALL	GETINT			; Get integer 0-255
	POP	DX			; Restore memory address
	XCHG	BX,DX
	MOV	[BX],AL
	XCHG	BX,DX			; Load it into memory
	RET
;
ROUND:
	MOV	BX,HALF			; Add 0.5 to FPREG
ADDPHL:
	CALL	LOADFP			; Load FP at (HL) to BCDE
	JMP	FPADD			; Add BCDE to FPREG
;
SUBPHL:
	CALL	LOADFP			; FPREG = -FPREG + number at HL
	JMP	SUBCDE			; Skip "POP CX" and "POP DX"
PSUB:
	POP	CX			; Get FP number from stack
	POP	DX
SUBCDE:
	CALL	INVSGN			; Negate FPREG
FPADD:
	MOV	AL,CH			; Get FP exponent
	OR	AL,AL			; Is number zero?
	JNZ	FPADD1
	RET				; Yes - Nothing to add
FPADD1:
	MOV	AL,[FPEXP]		; Get FPREG exponent
	OR	AL,AL           	; Is this number zero?
	JNZ	FPADD2			; Yes - Move BCDE to FPREQ
	JMP	FPBCDE
FPADD2:
	SUB	AL,CH			; BCDE number larger?
	JNC	NOSWAP			; No - Don't swap them
	NOT	AL			; Two's complement
	INC	AL			; FP exponent
	XCHG	BX,DX
	CALL	STAKFP			; Put FPREG on stack
	XCHG	BX,DX
	CALL	FPBCDE			; Move BCDE to FPREG
	POP	CX			; Restore number from stack
	POP	DX
NOSWAP:
	CMP	AL,24+1			; Second number insignificant?
	JC	NOSWAP1
	RET				; Yes - First number is result
NOSWAP1:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save number of bits to scale
	XCHG	AH,AL
	CALL	SIGNS			; Set MSBs & sign of result
	MOV	BH,AL			; Save sign of result
	POP	AX			; Restore scaling factor
	XCHG	AH,AL
	SAHF
	CALL	SCALE			; Scale BCDE to same exponent
	OR	AL,BH			; Result to be positive?
	MOV	BX,FPREG		; Point to FPREG
	JNS	MINCDE			; No - Subtract FPREG from CDE
	CALL	PLUCDE			; Add FPREG to CDE
	JC	NOSWAP2			; No overflow - Round it up
	JMP	RONDUP
NOSWAP2:
	LAHF
	INC	BX			; Point to exponent
	SAHF
	INC	BYTE PTR [BX]		; Increment it
	JNZ	NOSWAP3			; Number overflowed - Error
	JMP	OVERR
NOSWAP3:
	MOV	BL,1			; 1 bit to shift right
	CALL	SHRT1			; Shift result right
	JMP	RONDUP			; Round it up
;
MINCDE:
	XOR	AL,AL			; Clear A and carry
	SUB	AL,CH			; Negate exponent
	MOV	CH,AL			; Re-save exponent
	MOV	AL,[BX]			; Get LSB of FPREG
	SBB	AL,DL			; Subtract LSB of BCDE
	MOV	DL,AL			; Save LSB of BCDE
	LAHF
	INC	BX
	SAHF
	MOV	AL,[BX]			; Get NMSB of FPREG
	SBB	AL,DH			; Subtract NMSB of BCDE
	MOV	DH,AL			; Save NMSB of BCDE
	LAHF
	INC	BX
	SAHF
	MOV	AL,[BX]			; Get MSB of FPREG
	SBB	AL,CL			; Subtract MSB of BCDE
	MOV	CL,AL			; Save MSB of BCDE
CONPOS:
	JNC	BNORM
	CALL	COMPL			; Overflow - Make it positive
;
BNORM:
	MOV	BL,CH			; L = Exponent
	MOV	BH,DL			; H = LSB
	XOR	AL,AL
BNRMLP:
	MOV	CH,AL			; Save bit count
	MOV	AL,CL			; Get MSB
	OR	AL,AL			; Is it zero?
	JNZ	PNORM			; No - Do it bit at a time
	MOV	CL,DH			; MSB = NMSB
	MOV	DH,BH			; NMSB= LSB
	MOV	BH,BL			; LSB = VLSB
	MOV	BL,AL			; VLSB= 0
	MOV	AL,CH			; Get exponent
	SUB	AL,8			; Count 8 bits
	CMP	AL,-24-8		; Was number zero?
	JNZ	BNRMLP			; No - Keep normalising
RESZER:
	XOR	AL,AL			; Result is zero
SAVEXP:
	MOV	[FPEXP],AL		; Save result as zero
	RET
;
NORMAL:
	DEC	CH			; Count bits
	ADD	BX,BX			; Shift HL left
	MOV	AL,DH			; Get NMSB
	RCL	AL,1			; Shift left with last bit
	MOV	DH,AL			; Save NMSB
	MOV	AL,CL			; Get MSB
	ADC	AL,AL			; Shift left with last bit
	MOV	CL,AL			; Save MSB
PNORM:
	JNS	NORMAL			; Not done - Keep going
	MOV	AL,CH			; Number of bits shifted
	MOV	DL,BH			; Save HL in EB
	MOV	CH,BL
	OR	AL,AL			; Any shifting done?
	JZ	RONDUP			; No - Round it up
	MOV	BX,FPEXP		; Point to exponent
	ADD	AL,[BX]			; Add shifted bits
	MOV	[BX],AL			; Re-save exponent
	JNC	RESZER			; Underflow - Result is zero
	JNZ	RONDUP
	RET				; Result is zero
RONDUP:
	MOV	AL,CH			; Get VLSB of number
RONDB:
	MOV	BX,FPEXP		; Point to exponent
	OR	AL,AL			; Any rounding?
	JNS	RONDB1
	CALL	FPROND			; Yes - Round number up
RONDB1:
	MOV	CH,[BX]			; B = Exponent
	LAHF
	INC	BX
	SAHF
	MOV	AL,[BX]			; Get sign of result
	AND	AL,10000000B		; Only bit 7 needed
	XOR	AL,CL			; Set correct sign
	MOV	CL,AL			; Save correct sign in number
	JMP	FPBCDE			; Move BCDE to FPREG
;
FPROND:
	INC	DL			; Round LSB
	JZ	FPROND1
	RET				; Return if ok
FPROND1:
	INC	DH			; Round NMSB
	JZ	FPROND2
	RET				; Return if ok
FPROND2:
	INC	CL			; Round MSB
	JZ	FPROND3
	RET				; Return if ok
FPROND3:
	MOV	CL,80H			; Set normal value
	INC	BYTE PTR [BX]		; Increment exponent
	JZ	FPROND4
	RET				; Return if ok
FPROND4:
	JMP	OVERR			; Overflow error
;
PLUCDE:
	MOV	AL,[BX]			; Get LSB of FPREG
	ADD	AL,DL			; Add LSB of BCDE
	MOV	DL,AL			; Save LSB of BCDE
	LAHF
	INC	BX
	SAHF
	MOV	AL,[BX]			; Get NMSB of FPREG
	ADC	AL,DH			; Add NMSB of BCDE
	MOV	DH,AL			; Save NMSB of BCDE
	LAHF
	INC	BX
	SAHF
	MOV	AL,[BX]			; Get MSB of FPREG
	ADC	AL,CL			; Add MSB of BCDE
	MOV	CL,AL			; Save MSB of BCDE
	RET
;
COMPL:
	MOV	BX,SGNRES		; Sign of result
	MOV	AL,[BX]			; Get sign of result
	NOT	AL			; Negate it
	MOV	[BX],AL			; Put it back
	XOR	AL,AL
	MOV	BL,AL			; Set L to zero
	SUB	AL,CH			; Negate exponent,set carry
	MOV	CH,AL			; Re-save exponent
	MOV	AL,BL			; Load zero
	SBB	AL,DL			; Negate LSB
	MOV	DL,AL			; Re-save LSB
	MOV	AL,BL			; Load zero
	SBB	AL,DH			; Negate NMSB
	MOV	DH,AL			; Re-save NMSB
	MOV	AL,BL			; Load zero
	SBB	AL,CL			; Negate MSB
	MOV	CL,AL			; Re-save MSB
	RET
;
SCALE:
	MOV	CH,0			; Clear underflow
SCALLP:
	SUB	AL,8			; 8 bits (a whole byte)?
	JC	SHRITE			; No - Shift right A bits
	MOV	CH,DL			; <- Shift
	MOV	DL,DH			; <- right
	MOV	DH,CL			; <- eight
	MOV	CL,0			; <- bits
	JMP	SCALLP			; More bits to shift
;
SHRITE:
	ADD	AL,8+1			; Adjust count
	MOV	BL,AL			; Save bits to shift
SHRLP:
	XOR	AL,AL			; Flag for all done
	DEC	BL			; All shifting done?
	JNZ	SHRLP1
	RET				; Yes - Return
SHRLP1:
	MOV	AL,CL			; Get MSB
SHRT1:
	RCR	AL,1			; Shift it right
	MOV	CL,AL			; Re-save
	MOV	AL,DH			; Get NMSB
	RCR	AL,1			; Shift right with last bit
	MOV	DH,AL			; Re-save it
	MOV	AL,DL			; Get LSB
	RCR	AL,1			; Shift right with last bit
	MOV	DL,AL			; Re-save it
	MOV	AL,CH			; Get underflow
	RCR	AL,1			; Shift right with last bit
	MOV	CH,AL			; Re-save underflow
	JMP	SHRLP			; More bits to do
;
UNITY:
	DB	000H,000H,000H,081H 	; 1.00000
;
LOGTAB:
	DB	3           		; Table used by LOG
	DB	0AAH,056H,019H,080H	; 0.59898
	DB	0F1H,022H,076H,080H	; 0.96147
	DB	045H,0AAH,038H,082H	; 2.88539
;
LOG:
	CALL	TSTSGN			; Test sign of value
	OR	AL,AL
	JPO	LOG1			; ?FC Error if <= zero	JP PE,FCERR
	JMP	FCERR
LOG1:
	MOV	BX,FPEXP		; Point to exponent
	MOV	AL,[BX]			; Get exponent
	MOV	CX,8035H		; BCDE = SQR(1/2)
	MOV	DX,04F3H
	SUB	AL,CH			; Scale value to be < 1
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save scale factor
	XCHG	AH,AL
	MOV	[BX],CH			; Save new exponent
	PUSH	DX			; Save SQR(1/2)
	PUSH	CX
	CALL	FPADD			; Add SQR(1/2) to value
	POP	CX			; Restore SQR(1/2)
	POP	DX
	INC	CH			; Make it SQR(2)
	CALL	DVBCDE			; Divide by SQR(2)
	MOV	BX,UNITY		; Point to 1.
	CALL	SUBPHL			; Subtract FPREG from 1
	MOV	BX,LOGTAB		; Coefficient table
	CALL	SUMSER			; Evaluate sum of series
	MOV	CX,8080H		; BCDE = -0.5
	MOV	DX,0000H
	CALL	FPADD			; Subtract 0.5 from FPREG
	POP	AX			; Restore scale factor
	XCHG	AH,AL
	SAHF
	CALL	RSCALE			; Re-scale number
MULLN2:
	MOV	CX,8031H		; BCDE = Ln(2)
	MOV	DX,7218H
	JMP	FPMULT			; Skip "POP CX" and "POP DX"
;
MULT:
	POP	CX			; Get number from stack
	POP	DX
FPMULT:
	CALL	TSTSGN			; Test sign of FPREG
	JNZ	FPMULT1
	RET				; Return zero if zero
FPMULT1:
	MOV	BL,0			; Flag add exponents
	CALL	ADDEXP			; Add exponents
	MOV	AL,CL			; Get MSB of multiplier
	MOV	[MULVAL],AL		; Save MSB of multiplier
	XCHG	BX,DX
	MOV	[MULVAL+1],BX		; Save rest of multiplier
	MOV	CX,0			; Partial product (BCDE) = zero
	MOV	DH,CH
	MOV	DL,CH
	MOV	BX,BNORM		; Address of normalise
	PUSH	BX			; Save for return
	MOV	BX,MULT8		; Address of 8 bit multiply
	PUSH	BX			; Save for NMSB,MSB
	PUSH	BX			;
	MOV	BX,FPREG		; Point to number
MULT8:
	MOV	AL,[BX]			; Get LSB of number
;	LAHF
	INC	BX			; Point to NMSB
;	SAHF
	OR	AL,AL           	; Test LSB
	JZ	BYTSFT			; Zero - shift to next byte
	PUSH	BX			; Save address of number
	MOV	BL,8			; 8 bits to multiply by
MUL8LP:
	RCR	AL,1			; Shift LSB right
	MOV	BH,AL			; Save LSB
	MOV	AL,CL			; Get MSB
	JNC	NOMADD			; Bit was zero - Don't add
	PUSH	BX			; Save LSB and count
	MOV	BX,[MULVAL+1]		; Get LSB and NMSB
	ADD	BX,DX			; Add NMSB and LSB
	XCHG	BX,DX			; Leave sum in DE
	POP	BX			; Restore MSB and count
	MOV	AL,[MULVAL]		; Get MSB of multiplier
	ADC	AL,CL			; Add MSB
NOMADD:
	RCR	AL,1			; Shift MSB right
	MOV	CL,AL			; Re-save MSB
	MOV	AL,DH			; Get NMSB
	RCR	AL,1			; Shift NMSB right
	MOV	DH,AL			; Re-save NMSB
	MOV	AL,DL			; Get LSB
	RCR	AL,1			; Shift LSB right
	MOV	DL,AL			; Re-save LSB
	MOV	AL,CH			; Get VLSB
	RCR	AL,1			; Shift VLSB right
	MOV	CH,AL			; Re-save VLSB
	DEC	BL			; Count bits multiplied
	MOV	AL,BH			; Get LSB of multiplier
	JNZ	MUL8LP			; More - Do it
POPHRT:
	POP	BX			; Restore address of number
	RET
;
BYTSFT:
	MOV	CH,DL			; Shift partial product left
	MOV	DL,DH
	MOV	DH,CL
	MOV	CL,AL
	RET
;
DIV10:
	CALL	STAKFP			; Save FPREG on stack
	MOV	CX,8420H		; BCDE = 10.
	MOV	DX,0000H
	CALL	FPBCDE			; Move 10 to FPREG
;
DIV:
	POP	CX			; Get number from stack
	POP	DX
DVBCDE:
	CALL	TSTSGN			; Test sign of FPREG
	JNZ	DVBCDE1
	JMP	DZERR			; Error if division by zero
DVBCDE1:
	MOV	BL,-1			; Flag subtract exponents
	CALL	ADDEXP			; Subtract exponents
	INC	BYTE PTR [BX]		; Add 2 to exponent to adjust
	INC	BYTE PTR [BX]
;	LAHF
	DEC	BX			; Point to MSB
;	SAHF
	MOV	AL,[BX]			; Get MSB of dividend
	MOV	[DIV3],AL		; Save for subtraction
;	LAHF
	DEC	BX
;	SAHF
	MOV	AL,[BX]			; Get NMSB of dividend
	MOV	[DIV2],AL		; Save for subtraction
;	LAHF
	DEC	BX
;	SAHF
	MOV	AL,[BX]			; Get MSB of dividend
	MOV	[DIV1],AL		; Save for subtraction
	MOV	CH,CL			; Get MSB
	XCHG	BX,DX			; NMSB,LSB to HL
	XOR	AL,AL
	MOV	CL,AL			; Clear MSB of quotient
	MOV	DH,AL			; Clear NMSB of quotient
	MOV	DL,AL			; Clear LSB of quotient
	MOV	[DIV4],AL		; Clear overflow count
DIVLP:
	PUSH	BX			; Save divisor
	PUSH	CX
	MOV	AL,BL			; Get LSB of number
	CALL	DIVSUP			; Subt' divisor from dividend
	SBB	AL,0			; Count for overflows
	CMC
	JNC	RESDIV			; Restore divisor if borrow
	MOV	[DIV4],AL		; Re-save overflow count
	POP	AX			; Scrap divisor
;	XCHG	AH,AL
;	SAHF
	POP	AX
	XCHG	AH,AL
	SAHF
	STC				; Set carry to
	JMP	RESDIV1			; Skip "POP CX" and "POP BX"
;
RESDIV:
	POP	CX			; Restore divisor
	POP	BX
RESDIV1:
	MOV	AL,CL			; Get MSB of quotient
	INC	AL
	DEC	AL
	RCR	AL,1			; Bit 0 to bit 7
	JNS	RESDIV2
	JMP	RONDB			; Done - Normalise result
RESDIV2:
	RCL	AL,1			; Restore carry
	MOV	AL,DL			; Get LSB of quotient
	RCL	AL,1			; Double it
	MOV	DL,AL			; Put it back
	MOV	AL,DH			; Get NMSB of quotient
	RCL	AL,1			; Double it
	MOV	DH,AL			; Put it back
	MOV	AL,CL			; Get MSB of quotient
	RCL	AL,1			; Double it
	MOV	CL,AL			; Put it back
	ADD	BX,BX			; Double NMSB,LSB of divisor
	MOV	AL,CH			; Get MSB of divisor
	RCL	AL,1			; Double it
	MOV	CH,AL			; Put it back
	MOV	AL,[DIV4]		; Get VLSB of quotient
	RCL	AL,1			; Double it
	MOV	[DIV4],AL		; Put it back
	MOV	AL,CL			; Get MSB of quotient
	OR	AL,DH			; Merge NMSB
	OR	AL,DL			; Merge LSB
	JNZ	DIVLP			; Not done - Keep dividing
	PUSH	BX			; Save divisor
	MOV	BX,FPEXP		; Point to exponent
	DEC	BYTE PTR [BX]		; Divide by 2
	POP	BX			; Restore divisor
	JNZ	DIVLP			; Ok - Keep going
	JMP	OVERR			; Overflow error
;
ADDEXP:
	MOV	AL,CH			; Get exponent of dividend
	OR	AL,AL			; Test it
	JZ	OVTST3			; Zero - Result zero
	MOV	AL,BL			; Get add/subtract flag
	MOV	BX,FPEXP		; Point to exponent
	XOR	AL,[BX]			; Add or subtract it
	ADD	AL,CH			; Add the other exponent
	MOV	CH,AL			; Save new exponent
	RCR	AL,1			; Test exponent for overflow
	XOR	AL,CH
	MOV	AL,CH			; Get exponent
	JNS	OVTST2			; Positive - Test for overflow
	ADD	AL,80H			; Add excess 128
	MOV	[BX],AL			; Save new exponent
	JNZ	ADDEXP1			; Zero - Result zero
	JMP	POPHRT
ADDEXP1:
	CALL	SIGNS			; Set MSBs and sign of result
	MOV	[BX],AL			; Save new exponent
	LAHF
	DEC	BX			; Point to MSB
	SAHF
	RET
;
OVTST1:
	CALL	TSTSGN			; Test sign of FPREG
	NOT	AL			; Invert sign
	POP	BX			; Clean up stack
OVTST2:
	OR	AL,AL			; Test if new exponent zero
OVTST3:
	POP	BX			; Clear off return addr
	JS	OVTST4			; Result zero
	JMP	RESZER
OVTST4:
	JMP	OVERR			; Overflow error
;
MLSP10:
	CALL	BCDEFP			; Move FPREG to BCDE
	MOV	AL,CH			; Get exponent
	OR	AL,AL			; Is it zero?
	JNZ	MLSP101
	RET				; Yes - Result is zero
MLSP101:
	ADD	AL,2			; Multiply by 4
	JNC	MLSP102
	JMP	OVERR			; Overflow - ?OV Error
MLSP102:
	MOV	CH,AL			; Re-save exponent
	CALL	FPADD			; Add BCDE to FPREG (Time 5)
	MOV	BX,FPEXP		; Point to exponent
	INC	BYTE PTR [BX]		; Double number (Times 10)
	JZ	MLSP103
	RET				; Ok - Return
MLSP103:
	JMP	OVERR			; Overflow error
;
; Z flag
;
TSTSGN:
	MOV	AL,[FPEXP]		; Get sign of FPREG
	OR	AL,AL
	JNZ	TSTSGN1
	RET				; RETurn if number is zero
TSTSGN1:
	MOV	AL,[FPREG+2]		; Get MSB of FPREG
	CMP	AL,02FH
	JMP	RETREL1
RETREL:
	NOT	AL			; Invert sign
RETREL1:
	RCL	AL,1			; Sign bit to carry
FLGDIF:
	SBB	AL,AL			; Carry to all bits of A
	JZ	FLGDIF1
	RET				; Return -1 if negative
FLGDIF1:
	INC	AL			; Bump to +1
	RET				; Positive - Return +1
;
SGN:
	CALL	TSTSGN			; Test sign of FPREG
FLGREL:
	MOV	CH,80H+8		; 8 bit integer in exponent
	MOV	DX,0			; Zero NMSB and LSB
RETINT:
	MOV	BX,FPEXP		; Point to exponent
	MOV	CL,AL			; CDE = MSB,NMSB and LSB
	MOV	[BX],CH			; Save exponent
	MOV	CH,0			; CDE = integer to normalise
;	LAHF
	INC	BX			; Point to sign of result
;	SAHF
	MOV	BYTE PTR [BX],80H	; Set sign of result
	RCL	AL,1			; Carry = sign of integer
	JMP	CONPOS			; Set sign of result
;
ABS:
	CALL	TSTSGN			; Test sign of FPREG
	JS	INVSGN
	RET				; Return if positive
INVSGN:
	MOV	BX,FPREG+2		; Point to MSB
	MOV	AL,[BX]			; Get sign of mantissa
	XOR	AL,80H			; Invert sign of mantissa
	MOV	[BX],AL			; Re-save sign of mantissa
	RET
;
STAKFP:
	XCHG	BX,DX			; Save code string address
	MOV	BX,[FPREG]		; LSB,NLSB of FPREG
	MOV	BP,SP
	XCHG	[BP],BX			; Stack them,get return
	PUSH	BX			; Re-save return
	MOV	BX,[FPREG+2]		; MSB and exponent of FPREG
	MOV	BP,SP
	XCHG	[BP],BX			; Stack them,get return
	PUSH	BX			; Re-save return
	XCHG	BX,DX			; Restore code string address
	RET
;
PHLTFP:
	CALL	LOADFP			; Number at HL to BCDE
FPBCDE:
	XCHG	BX,DX			; Save code string address
	MOV	[FPREG],BX		; Save LSB,NLSB of number
	MOV	BX,CX			; Exponent of number
	MOV	[FPREG+2],BX		; Save MSB and exponent
	XCHG	BX,DX			; Restore code string address
	RET
;
BCDEFP:
	MOV	BX,FPREG		; Point to FPREG
LOADFP:
	MOV	DL,[BX]			; Get LSB of number
	LAHF
	INC	BX
;	SAHF
	MOV	DH,[BX]			; Get NMSB of number
;	LAHF
	INC	BX
;	SAHF
	MOV	CL,[BX]			; Get MSB of number
;	LAHF
	INC	BX
	SAHF
	MOV	CH,[BX]			; Get exponent of number
INCHL:
	LAHF
	INC	BX			; Used for conditional "INC HL"
	SAHF
	RET
;
FPTHL:
	MOV	DX,FPREG		; Point to FPREG
DETHL4:
	MOV	CH,4			; 4 bytes to move
DETHLB:
	XCHG	BX,DX
	MOV	AL,[BX]			; Get source
	XCHG	BX,DX
	MOV	[BX],AL			; Save destination
;	LAHF
	INC	DX			; Next source
;	SAHF
;	LAHF
	INC	BX			; Next destination
;	SAHF
	DEC	CH			; Count bytes
	JNZ	DETHLB			; Loop if more
	RET
;
SIGNS:
	MOV	BX,FPREG+2		; Point to MSB of FPREG
	MOV	AL,[BX]			; Get MSB
	ROL	AL,1			; Old sign to carry
	STC				; Set MSBit
	RCR	AL,1			; Set MSBit of MSB
	MOV	[BX],AL			; Save new MSB
	CMC				; Complement sign
	RCR	AL,1			; Old sign to carry
	LAHF
	INC	BX
;	SAHF
;	LAHF
	INC	BX
	SAHF
	MOV	[BX],AL			; Set sign of result
	MOV	AL,CL			; Get MSB
	ROL	AL,1			; Old sign to carry
	STC				; Set MSBit
	RCR	AL,1			; Set MSBit of MSB
	MOV	CL,AL			; Save MSB
	RCR	AL,1
	XOR	AL,[BX]			; New sign of result
	RET
;
CMPNUM:
	MOV	AL,CH			; Get exponent of numbe
	OR	AL,AL
	JNZ	CMPNUM1			; Zero - Test sign of FPREG
	JMP	TSTSGN
CMPNUM1:
	MOV	BX,RETREL		; Return relation routine
	PUSH	BX			; Save for return
	CALL	TSTSGN			; Test sign of FPREG
	MOV	AL,CL			; Get MSB of number
	JNZ	CMPNUM2
	RET				; FPREG zero - Number's MSB
CMPNUM2:
	MOV	BX,FPREG+2		; MSB of FPREG
	XOR	AL,[BX]			; Combine signs
	MOV	AL,CL			; Get MSB of number
	JNS	CMPNUM3
	RET				; Exit if signs different
CMPNUM3:
	CALL	CMPFP			; Compare FP numbers
	RCR	AL,1			; Get carry to sign
	XOR	AL,CL			; Combine with MSB of number
	RET
;
CMPFP:
	LAHF
	INC	BX			; Point to exponent
	SAHF
	MOV	AL,CH			; Get exponent
	CMP	AL,[BX]			; Compare exponents
	JZ	CMPFP1
	RET				; Different
CMPFP1:
	LAHF
	DEC	BX			; Point to MBS
	SAHF
	MOV	AL,CL			; Get MSB
	CMP	AL,[BX]			; Compare MSBs
	JZ	CMPFP2
	RET				; Different
CMPFP2:
	LAHF
	DEC	BX			; Point to NMSB
	SAHF
	MOV	AL,DH			; Get NMSB
	CMP	AL,[BX]			; Compare NMSBs
	JZ	CMPFP3
	RET				; Different
CMPFP3:
	LAHF
	DEC	BX			; Point to LSB
	SAHF
	MOV	AL,DL			; Get LSB
	SUB	AL,[BX]			; Compare LSBs
	JZ	CMPFP4
	RET				; Different
CMPFP4:
	POP	BX			; Drop RETurn
	POP	BX			; Drop another RETurn
	RET
;
FPINT:
	MOV	CH,AL			; <- Move
	MOV	CL,AL			; <- exponent
	MOV	DH,AL			; <- to all
	MOV	DL,AL			; <- bits
	OR	AL,AL			; Test exponent
	JNZ	FPINT1
	RET				; Zero - Return zero
FPINT1:
	PUSH	BX			; Save pointer to number
	CALL	BCDEFP			; Move FPREG to BCDE
	CALL	SIGNS			; Set MSBs & sign of result
	XOR	AL,[BX]			; Combine with sign of FPREG
	MOV	BH,AL			; Save combined signs
	JNS	FPINT2
	CALL	DCBCDE			; Negative - Decrement BCDE
FPINT2:
	MOV	AL,80H+24		; 24 bits
	SUB	AL,CH			; Bits to shift
	CALL	SCALE			; Shift BCDE
	MOV	AL,BH			; Get combined sign
	RCL	AL,1			; Sign to carry
	JNC	FPINT3
	CALL	FPROND			; Negative - Round number up
FPINT3:
	MOV	CH,0			; Zero exponent
	JNC	FPINT4
	CALL    COMPL			; If negative make positive
FPINT4:
	POP	BX			; Restore pointer to number
	RET
;
DCBCDE:
;	LAHF
	DEC	DX			; Decrement BCDE
;	SAHF
	MOV	AL,DH			; Test LSBs
	AND	AL,DL
	INC	AL
	JZ	DCBCDE1
	RET				; Exit if LSBs not FFFF
DCBCDE1:
	LAHF
	DEC	CX			; Decrement MSBs
	SAHF
	RET
;
INT:
	MOV	BX,FPEXP		; Point to exponent
	MOV	AL,[BX]			; Get exponent
	CMP	AL,80H+24		; Integer accuracy only?
	MOV	AL,[FPREG]		; Get LSB
	JC	INT1
	RET				; Yes - Already integer
INT1:
	MOV	AL,[BX]			; Get exponent
	CALL	FPINT			; F.P to integer
	MOV	BYTE PTR [BX],80H+24	; Save 24 bit integer
	MOV	AL,DL			; Get LSB of number
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save LSB
	XCHG	AH,AL
	MOV	AL,CL			; Get MSB of number
	RCL	AL,1			; Sign to carry
	CALL	CONPOS			; Set sign of result
	POP	AX			; Restore LSB of number
	XCHG	AH,AL
	SAHF
	RET
;
MLDEBC:
	MOV	BX,0			; Clear partial product
	MOV	AL,CH			; Test multiplier
	OR	AL,CL
	JNZ	MLDEBC1
	RET				; Return zero if zero
MLDEBC1:
	MOV	AL,16			; 16 bits
MLDBLP:
	ADD	BX,BX			; Shift P.P left
	JNC	MLDBLP1
	JMP	BSERR			; ?BS Error if overflow
MLDBLP1:
	XCHG	BX,DX
	ADD	BX,BX			; Shift multiplier left
	XCHG	BX,DX
	JNC	NOMLAD			; Bit was zero - No add
	ADD	BX,CX			; Add multiplicand
	JNC	NOMLAD
	JMP	BSERR			; ?BS Error if overflow
NOMLAD:
	DEC	AL			; Count bits
	JNZ	MLDBLP			; More
	RET
;
ASCTFP:
	CMP	AL,'-'			; Negative?
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save it and flags
	XCHG	AH,AL
	JZ	CNVNUM			; Yes - Convert number
	CMP	AL,'+'			; Positive?
	JZ	CNVNUM			; Yes - Convert number
	LAHF
	DEC	BX			; DEC 'cos GETCHR INCs
	SAHF
CNVNUM:
	CALL	RESZER			; Set result to zero
	MOV	CH,AL			; Digits after point counter
	MOV	DH,AL			; Sign of exponent
	MOV	DL,AL			; Exponent of ten
	NOT	AL
	MOV	CL,AL			; Before or after point flag
MANLP:
	CALL	GETCHR			; Get next character
	JC	ADDIG			; Digit - Add to number
	CMP	AL,'.'
	JZ	DPOINT			; '.' - Flag point
	CMP	AL,'E'
	JNZ	CONEXP			; Not 'E' - Scale number
	CALL	GETCHR			; Get next character
	CALL	SGNEXP			; Get sign of exponent
EXPLP:
	CALL	GETCHR			; Get next character
	JC	EDIGIT			; Digit - Add to exponent
	INC	DH			; Is sign negative?
	JNZ	CONEXP			; No - Scale number
	XOR	AL,AL
	SUB	AL,DL			; Negate exponent
	MOV	DL,AL			; And re-save it
	INC	CL			; Flag end of number
DPOINT:
	INC	CL			; Flag point passed
	JZ	MANLP			; Zero - Get another digit
CONEXP:
	PUSH	BX			; Save code string address
	MOV	AL,DL			; Get exponent
	SUB	AL,CH			; Subtract digits after point
SCALMI:
	JS	SCALMI1
	CALL	SCALPL			; Positive - Multiply number
SCALMI1:
	JNS	ENDCON			; Positive - All done
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save number of times to /10
	XCHG	AH,AL
	CALL	DIV10			; Divide by 10
	POP	AX			; Restore count
	XCHG	AH,AL
	SAHF
	INC	AL			; Count divides
;
ENDCON:
	JNZ	SCALMI			; More to do
	POP	DX			; Restore code string address
	POP	AX			; Restore sign of number
	XCHG	AH,AL
	SAHF
	JNZ	ENDCON1
	CALL	INVSGN			; Negative - Negate number
ENDCON1:
	XCHG	BX,DX			; Code string address to HL
	RET
;
SCALPL:
	JNZ	MULTEN
	RET				; Exit if no scaling needed
MULTEN:
;	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save count
	XCHG	AH,AL
	CALL	MLSP10			; Multiply number by 10
	POP	AX			; Restore count
	XCHG	AH,AL
;	SAHF
	DEC	AL			; Count multiplies
	RET
;
ADDIG:
	PUSH	DX			; Save sign of exponent
	MOV	DH,AL			; Save digit
	MOV	AL,CH			; Get digits after point
	ADC	AL,CL			; Add one if after point
	MOV	CH,AL			; Re-save counter
	PUSH	CX			; Save point flags
	PUSH	BX			; Save code string address
	PUSH	DX			; Save digit
	CALL	MLSP10			; Multiply number by 10
	POP	AX			; Restore digit
	XCHG	AH,AL
	SAHF
	SUB	AL,'0'			; Make it absolute
	CALL	RSCALE			; Re-scale number
	POP	BX			; Restore code string address
	POP	CX			; Restore point flags
	POP	DX			; Restore sign of exponent
	JMP	MANLP			; Get another digit
;
RSCALE:
	CALL	STAKFP			; Put number on stack
	CALL	FLGREL			; Digit to add to FPREG
PADD:
	POP	CX			; Restore number
	POP	DX
	JMP	FPADD			; Add BCDE to FPREG and return
;
EDIGIT:
	MOV	AL,DL			; Get digit
	ROL	AL,1			; Times 2
	ROL	AL,1			; Times 4
	ADD	AL,DL			; Times 5
	ROL	AL,1			; Times 10
	ADD	AL,[BX]			; Add next digit
	SUB	AL,'0'			; Make it absolute
	MOV	DL,AL			; Save new digit
	JMP	EXPLP			; Look for another digit
;
LINEIN:
	PUSH	BX			; Save code string address
	MOV	BX,INMSG		; Output " in "
	CALL	PRS			; Output string at HL
	POP	BX			; Restore code string address
PRNTHL:
	XCHG	BX,DX			; Code string address to DE
	XOR	AL,AL
	MOV	CH,80H+24		; 24 bits
	CALL	RETINT			; Return the integer
	MOV	BX,PRNUMS		; Print number string
	PUSH	BX			; Save for return
NUMASC:
	MOV	BX,PBUFF		; Convert number to ASCII
	PUSH	BX			; Save for return
	CALL	TSTSGN			; Test sign of FPREG
	MOV	BYTE PTR [BX],' '	; Space at start
	JNS	SPCFST			; Positive - Space to start
	MOV	BYTE PTR [BX],'-'	; '-' sign at start
SPCFST:
	LAHF
	INC	BX			; First byte of number
	SAHF
	MOV	BYTE PTR [BX],'0'	; '0' if zero
	JNZ	SPCFST1			; Return '0' if zero
	JMP	JSTZER
SPCFST1:
	PUSH	BX			; Save buffer address
	JNS	SPCFST2
	CALL	INVSGN			; Negate FPREG if negative
SPCFST2:
	XOR	AL,AL			; Zero A
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save it
	XCHG	AH,AL
	CALL	RNGTST			; Test number is in range
SIXDIG:
	MOV	CX,9143H		; BCDE - 99999.9
	MOV	DX,4FF8H
	CALL	CMPNUM			; Compare numbers
	OR	AL,AL
	JPO	INRNG			; > 99999.9 - Sort it out	JP PO,INGNG
	POP	AX			; Restore count
	XCHG	AH,AL
	SAHF
	CALL	MULTEN			; Multiply by ten
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Re-save count
	XCHG	AH,AL
	JMP	SIXDIG			; Test it again
;
GTSIXD:
	CALL	DIV10			; Divide by 10
	POP	AX			; Get count
	XCHG	AH,AL
	SAHF
	INC	AL			; Count divides
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Re-save count
	XCHG	AH,AL
	CALL	RNGTST			; Test number is in range
INRNG:
	CALL	ROUND			; Add 0.5 to FPREG
	INC	AL
	CALL	FPINT			; F.P to integer
	CALL	FPBCDE			; Move BCDE to FPREG
	MOV	CX,0306H		; 1E+06 to 1E-03 range
	POP	AX			; Restore count
	XCHG	AH,AL
	SAHF
	ADD	AL,CL			; 6 digits before point
	INC	AL			; Add one
	JS	MAKNUM			; Do it in 'E' form if < 1E-02
	CMP	AL,6+1+1		; More than 999999 ?
	JNC	MAKNUM			; Yes - Do it in 'E' form
	INC	AL			; Adjust for exponent
	MOV	CH,AL			; Exponent of number
	MOV	AL,2			; Make it zero after
;
MAKNUM:
	DEC	AL			; Adjust for digits to do
	DEC	AL
	POP	BX			; Restore buffer addres
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save count
	XCHG	AH,AL
	MOV	DX,POWERS		; Powers of ten
	DEC	CH			; Count digits before point
	JNZ	DIGTXT			; Not zero - Do number
	MOV	BYTE PTR [BX],'.'	; Save point
	LAHF
	INC	BX			; Move on
	SAHF
	MOV	BYTE PTR [BX],'0'	; Save zero
	LAHF
	INC	BX			; Move on
	SAHF
DIGTXT:
	DEC	CH			; Count digits before point
	MOV	BYTE PTR [BX],'.'	; Save point in case
	JNZ	DIGTXT1
	CALL	INCHL			; Last digit - move on
DIGTXT1:
	PUSH	CX			; Save digits before point
	PUSH	BX			; Save buffer address
	PUSH	DX			; Save powers of ten
	CALL	BCDEFP			; Move FPREG to BCDE
	POP	BX			; Powers of ten table
	MOV	CH,'0'-1		; ASCII '0' - 1
TRYAGN:
	INC	CH			; Count subtractions
	MOV	AL,DL			; Get LSB
	SUB	AL,[BX]			; Subtract LSB
	MOV	DL,AL			; Save LSB
	LAHF
	INC	BX
	SAHF
	MOV	AL,DH			; Get NMSB
	SBB	AL,[BX]			; Subtract NMSB
	MOV	DH,AL			; Save NMSB
	LAHF
	INC	BX
	SAHF
	MOV	AL,CL			; Get MSB
	SBB	AL,[BX]			; Subtract MSB
	MOV	CL,AL			; Save MSB
	LAHF
	DEC	BX			; Point back to start
;	SAHF
;	LAHF
	DEC	BX
	SAHF
	JNC	TRYAGN			; No overflow - Try aga
	CALL	PLUCDE			; Restore number
	LAHF
	INC	BX
	SAHF				; Start of next number
	CALL	FPBCDE			; Move BCDE to FPREG
	XCHG	BX,DX			; Save point in table
	POP	BX			; Restore buffer address
	MOV	[BX],CH			; Save digit in buffer
	LAHF
	INC	BX			; And move on
	SAHF
	POP	CX			; Restore digit count
	DEC	CL			; Count digits
	JNZ	DIGTXT			; More - Do them
	DEC	CH			; Any decimal part?
	JZ	DOEBIT			; No - Do 'E' bit
SUPTLZ:
;	LAHF
	DEC	BX			; Move back through buffer
;	SAHF
	MOV	AL,[BX]			; Get character
	CMP	AL,'0'			; '0' character?
	JZ	SUPTLZ			; Yes - Look back for more
	CMP	AL,'.'			; A decimal point?
	JZ	DOEBIT
	CALL	INCHL			; Move back over digit
;
DOEBIT:
	POP	AX			; Get 'E' flag
	XCHG	AH,AL
	SAHF
	JZ	NOENED			; No 'E' needed - End buffer
	MOV	BYTE PTR [BX],'E'	; Put 'E' in buffer
	LAHF
	INC	BX			; And move on
	SAHF
	MOV	BYTE PTR [BX],'+'	; Put '+' in buffer
	JNS	OUTEXP			; Positive - Output exponent
	MOV	BYTE PTR [BX],'-'	; Put '-' in buffer
	NOT	AL			; Negate exponent
	INC	AL
OUTEXP:
	MOV	CH,'0'-1		; ASCII '0' - 1
EXPTEN:
	INC	CH			; Count subtractions
	SUB	AL,10			; Tens digit
	JNC	EXPTEN			; More to do
	ADD	AL,'0'+10		; Restore and make ASCII
	LAHF
	INC	BX
	SAHF				; Move on
	MOV	[BX],CH			; Save MSB of exponent
JSTZER:
	LAHF
	INC	BX
	SAHF
	MOV	[BX],AL			; Save LSB of exponent
	LAHF
	INC	BX
	SAHF
NOENED:
	MOV	[BX],CL			; Mark end of buffer
	POP	BX			; Restore code string address
	RET
;
RNGTST:
	MOV	CX,9474H		; BCDE = 999999.
	MOV	DX,23F7H
	CALL	CMPNUM			; Compare numbers
	OR	AL,AL
	POP	BX			; Return address to HL
	JPE	RNGTST1			; Too big - Divide by ten	JP PO,GTSIND
	JMP	GTSIXD
RNGTST1:
	PUSH	BX
	RET
;	JMP	[BX]			; Otherwise return to caller
;
HALF:
	DB	00H,00H,00H,80H 	; 0.5
;
POWERS:
	DB	0A0H,086H,001H		; 100000
	DB	010H,027H,000H		; 10000
	DB	0E8H,003H,000H		; 1000
	DB	064H,000H,000H		; 100
	DB	00AH,000H,000H		; 10
	DB	001H,000H,000H		; 1
;
NEGAFT:
	MOV	BX,INVSGN		; Negate result
	MOV	BP,SP
	XCHG	[BP],BX			; To be done after call
	PUSH	BX
	RET
;	JMP	[BX]			; Return to caller
;
SQR:
	CALL	STAKFP			; Put value on stack
	MOV	BX,HALF			; Set power to 1/2
	CALL	PHLTFP			; Move 1/2 to FPREG
;
POWER:
	POP	CX			; Get base
	POP	DX
	CALL	TSTSGN			; Test sign of power
	MOV	AL,CH			; Get exponent of base
	JZ	EXP			; Make result 1 if zero
	JNS	POWER1			; Positive base - Ok
	OR	AL,AL			; Zero to negative power?
	JNZ	POWER1			; Yes - ?/0 Error
	JMP	DZERR
POWER1:
	OR	AL,AL			; Base zero?
	JNZ	POWER5			; Yes - Return zero
	JMP	SAVEXP
POWER5:
	PUSH	DX			; Save base
	PUSH	CX
	MOV	AL,CL			; Get MSB of base
	OR	AL,01111111B		; Get sign status
	CALL	BCDEFP			; Move power to BCDE
	JNS	POWER2			; Positive base - Ok
	PUSH	DX			; Save power
	PUSH	CX
	CALL	INT			; Get integer of power
	POP	CX			; Restore power
	POP	DX
	LAHF
	XCHG	AH,AL
	PUSH	AX			; MSB of base
	XCHG	AH,AL
	CALL	CMPNUM			; Power an integer?
	POP	BX			; Restore MSB of base
	MOV	AL,BH			; but don't affect flags
	RCR	AL,1			; Exponent odd or even?
POWER2:
	POP	BX			; Restore MSB and exponent
	MOV	[FPREG+2],BX		; Save base in FPREG
	POP	BX			; LSBs of base
	MOV	[FPREG],BX		; Save in FPREG
	JNC	POWER3
	CALL	NEGAFT			; Odd power - Negate result
POWER3:
	JNZ	POWER4
	CALL	INVSGN			; Negative base - Negate it
POWER4:
	PUSH	DX			; Save power
	PUSH	CX
	CALL	LOG			; Get LOG of base
	POP	CX			; Restore power
	POP	DX
	CALL	FPMULT			; Multiply LOG by power
;
EXP:
	CALL	STAKFP			; Put value on stack
	MOV	CX,08138H		; BCDE = 1/Ln(2)
	MOV	DX,0AA3BH
	CALL	FPMULT			; Multiply value by 1/L(2)
	MOV	AL,[FPEXP]		; Get exponent
	CMP	AL,80H+8		; Is it in range?
	JC	EXP1			; No - Test for overflow
	JMP	OVTST1
EXP1:
	CALL	INT			; Get INT of FPREG
	ADD	AL,80H			; For excess 128
	ADD	AL,2			; Exponent > 126?
	JNC	EXP2			; Yes - Test for overflow
	JMP	OVTST1
EXP2:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save scaling factor
	XCHG	AH,AL
	MOV	BX,UNITY		; Point to 1.
	CALL	ADDPHL			; Add 1 to FPREG
	CALL	MULLN2			; Multiply by LN(2)
	POP	AX			; Restore scaling factor
	XCHG	AH,AL
	SAHF
	POP	CX			; Restore exponent
	POP	DX
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save scaling factor
	XCHG	AH,AL
	CALL	SUBCDE			; Subtract exponent from FPREG
	CALL	INVSGN			; Negate result
	MOV	BX,EXPTAB		; Coefficient table
	CALL	SMSER1			; Sum the series
	MOV	DX,0			; Zero LSBs
	POP	CX			; Scaling factor
	MOV	CL,DH			; Zero MSB
	JMP	FPMULT			; Scale result to correct value
;
EXPTAB:
	DB	8                  	; Table used by EXP
	DB	040H,02EH,094H,074H	; -1/7! (-1/504
	DB	070H,04FH,02EH,077H	;  1/6! ( 1/720
	DB	06EH,002H,088H,07AH	; -1/5! (-1/120
	DB	0E6H,0A0H,02AH,07CH	;  1/4! ( 1/24)
	DB	050H,0AAH,0AAH,07EH	; -1/3! (-1/6)
	DB	0FFH,0FFH,07FH,07FH	;  1/2! ( 1/2)
	DB	000H,000H,080H,081H	; -1/1! (-1/1)
	DB	000H,000H,000H,081H	;  1/0! ( 1/1)
;
SUMSER:
	CALL	STAKFP			; Put FPREG on stack
	MOV	DX,MULT			; Multiply by "X"
	PUSH	DX			; To be done after
	PUSH	BX			; Save address of table
	CALL	BCDEFP			; Move FPREG to BCDE
	CALL	FPMULT			; Square the value
	POP	BX			; Restore address of table
SMSER1:
	CALL	STAKFP			; Put value on stack
	MOV	AL,[BX]			; Get number of coefficients
	LAHF
	INC	BX			; Point to start of table
	SAHF
	CALL	PHLTFP			; Move coefficient to FPREG
	JMP	SUMLP1			; Skip "POP AF"
SUMLP:
	POP	AX			; Restore count
	XCHG	AH,AL
;	SAHF
SUMLP1:
	POP	CX			; Restore number
	POP	DX
	DEC	AL			; Cont coefficients
	JNZ	SUMLP2
	RET				; All done
SUMLP2:
	PUSH	DX			; Save number
	PUSH	CX
;	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save count
	XCHG	AH,AL
	PUSH	BX			; Save address in table
	CALL	FPMULT			; Multiply FPREG by BCD
	POP	BX			; Restore address in table
	CALL	LOADFP			; Number at HL to BCDE
	PUSH	BX			; Save address in table
	CALL	FPADD			; Add coefficient to FPREG
	POP	BX			; Restore address in table
	JMP	SUMLP			; More coefficients
;
RND:
	CALL	TSTSGN			; Test sign of FPREG
	MOV	BX,SEED+2		; Random number seed
	JS	RESEED			; Negative - Re-seed
	MOV	BX,LSTRND		; Last random number
	CALL	PHLTFP			; Move last RND to FPREG
	MOV	BX,SEED+2		; Random number seed
	JNZ	RND0
	RET				; Return if RND(0)
RND0:
	ADD	AL,[BX]			; Add (SEED)+2)
	AND	AL,00000111B		; 0 to 7
	MOV	CH,0
	MOV	[BX],AL			; Re-save seed
;	LAHF
	INC	BX			; Move to coefficient table
;	SAHF
	ADD	AL,AL			; 4 bytes
	ADD	AL,AL			; per entry
	MOV	CL,AL			; BC = Offset into table
	ADD	BX,CX			; Point to coefficient
	CALL	LOADFP			; Coefficient to BCDE
	CALL	FPMULT			; Multiply FPREG by coefficient
	MOV	AL,[SEED+1]		; Get [SEED+1]
	INC	AL			; Add 1
	AND	AL,00000011B		; 0 to 3
	MOV	CH,0
	CMP	AL,1			; Is it zero?
	ADC	AL,CH			; Yes - Make it 1
	MOV	[SEED+1],AL		; Re-save seed
	MOV	BX,RNDTAB-4		; Addition table
	ADD	AL,AL			; 4 bytes
	ADD	AL,AL			; per entry
	MOV	CL,AL			; BC = Offset into table
	ADD	BX,CX			; Point to value
	CALL	ADDPHL			; Add value to FPREG
RND1:
	CALL	BCDEFP			; Move FPREG to BCDE
	MOV	AL,DL			; Get LSB
	MOV	DL,CL			; LSB = MSB
	XOR	AL,01001111B		; Fiddle around
	MOV	CL,AL			; New MSB
	MOV	BYTE PTR [BX],80H	; Set exponent
;	LAHF
	DEC	BX			; Point to MSB
;	SAHF
	MOV	CH,[BX]			; Get MSB
	MOV	BYTE PTR [BX],80H	; Make value -0.5
	MOV	BX,SEED			; Random number seed
	INC	BYTE PTR [BX]		; Count seed
	MOV	AL,[BX]			; Get seed
	SUB	AL,171			; Do it modulo 171
	JNZ	RND2			; Non-zero - Ok
	MOV	[BX],AL			; Zero seed
	INC	CL			; Fillde about
	DEC	DH			; with the
	INC	DL			; number
RND2:
	CALL	BNORM			; Normalise number
	MOV	BX,LSTRND		; Save random number
	JMP	FPTHL			; Move FPREG to last and return
;
RESEED:
	MOV	[BX],AL			; Re-seed random number
;	LAHF
	DEC	BX
;	SAHF
	MOV	[BX],AL
;	LAHF
	DEC	BX
;	SAHF
	MOV	[BX],AL
	JMP	RND1			; Return RND seed
;
RNDTAB:
	DB	068H,0B1H,046H,068H	; Table used by RND
	DB	099H,0E9H,092H,069H
	DB	010H,0D1H,075H,068H
;
COS:
	MOV	BX,HALFPI		; Point to PI/2
	CALL	ADDPHL			; Add it to PPREG
SIN:
	CALL	STAKFP			; Put angle on stack
	MOV	CX,8349H		; BCDE = 2 PI
	MOV	DX,0FDBH
	CALL	FPBCDE			; Move 2 PI to FPREG
	POP	CX			; Restore angle
	POP	DX
	CALL	DVBCDE			; Divide angle by 2 PI
	CALL	STAKFP			; Put it on stack
	CALL	INT			; Get INT of result
	POP	CX			; Restore number
	POP	DX
	CALL	SUBCDE			; Make it 0 <= value < 1
	MOV	BX,QUARTR		; Point to 0.25
	CALL	SUBPHL			; Subtract value from 0.25
	CALL	TSTSGN			; Test sign of value
	STC				; Flag positive
	JNS	SIN1			; Positive - Ok
	CALL	ROUND			; Add 0.5 to value
	CALL	TSTSGN			; Test sign of value
	OR	AL,AL			; Flag negative
SIN1:
	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save sign
	XCHG	AH,AL
	JS	SIN2
	CALL	INVSGN			; Negate value if positive
SIN2:
	MOV	BX,QUARTR		; Point to 0.25
	CALL	ADDPHL			; Add 0.25 to value
	POP	AX			; Restore sign
	XCHG	AH,AL
	SAHF
	JC	SIN3
	CALL	INVSGN			; Negative - Make positive
SIN3:
	MOV	BX,SINTAB		; Coefficient table
	JMP	SUMSER			; Evaluate sum of series
;
HALFPI:
	DB	0DBH,00FH,049H,081H	; 1.5708 (PI/2)
;
QUARTR:
	DB	000H,000H,000H,07FH	; 0.25
;
SINTAB:
	DB	5			; Table used by SIN
	DB	0BAH,0D7H,01EH,086H	; 39.711
	DB	064H,026H,099H,087H	;-76.575
	DB	058H,034H,023H,087H	; 81.602
	DB	0E0H,05DH,0A5H,086H	;-41.342
	DB	0DAH,00FH,049H,083H	; 6.2832
;
TAN:
	CALL	STAKFP			; Put angle on stack
	CALL	SIN			; Get SIN of angle
	POP	CX			; Restore angle
	POP	BX
	CALL	STAKFP			; Save SIN of angle
	XCHG	BX,DX			; BCDE = Angle
	CALL	FPBCDE			; Angle to FPREG
	CALL	COS			; Get COS of angle
	JMP	DIV			; TAN = SIN / COS
;
ATN:
	CALL	TSTSGN			; Test sign of value
	JNS	ATN1
	CALL	NEGAFT			; Negate result after if -ve
ATN1:
	JNS	ATN2
	CALL	INVSGN			; Negate value if -ve
ATN2:
	MOV	AL,[FPEXP]		; Get exponent
	CMP	AL,81H			; Number less than 1?
	JC	ATN3			; Yes - Get arc tangnt
	MOV	CX,8100H		; BCDE = 1
	MOV	DH,CL
	MOV	DL,CL
	CALL	DVBCDE			; Get reciprocal of number
	MOV	BX,SUBPHL		; Sub angle from PI/2
	PUSH	BX			; Save for angle > 1
ATN3:
	MOV	BX,ATNTAB		; Coefficient table
	CALL	SUMSER			; Evaluate sum of series
	MOV	BX,HALFPI		; PI/2 - angle in case > 1
	RET				; Number > 1 - Sub from PI/2
;
ATNTAB:
	DB	9			; Table used by ATN
	DB	04AH,0D7H,03BH,078H	; 1/17
	DB	002H,06EH,084H,07BH	;-1/15
	DB	0FEH,0C1H,02FH,07CH	; 1/13
	DB	074H,031H,09AH,07DH	;-1/11
	DB	084H,03DH,05AH,07DH	; 1/9
	DB	0C8H,07FH,091H,07EH	;-1/7
	DB	0E4H,0BBH,04CH,07EH	; 1/5
	DB	06CH,0AAH,0AAH,07FH	;-1/3
	DB	000H,000H,000H,081H	; 1/1
;

ARET:
	RET				; A RETurn instruction
;
CLS:
	MOV	AL,CLRSCRN		; ASCII Clear screen
	JMP	putch			; Output character
;
WIDTH:
	CALL	GETINT			; Get integer 0-255
	MOV	AL,DL			; Width to A
	MOV	[LWIDTH],AL		; Set width
	RET
;
LINES:
	CALL	GETNUM			; Get a number
	CALL	DEINT			; Get integer -32768 to 32767
	MOV	[LINESC],DX		; Set lines counter
	MOV	[LINESN],DX		; Set lines number
	RET
;
DEEK:
	CALL	DEINT			; Get integer -32768 to 32767
	PUSH	DX			; Save number
	POP	BX			; Number to HL
	MOV	CH,[BX]			; Get LSB of contents
;	LAHF
	INC	BX
;	SAHF
	MOV	AL,[BX]			; Get MSB of contents
	JMP	ABPASS			; Return integer AB
;
DOKE:
	CALL	GETNUM			; Get a number
	CALL	DEINT			; Get integer -32768 to 32767
	PUSH	DX			; Save address
	CALL	CHKSYN			; Make sure ',' follows
	DB	','
	CALL	GETNUM			; Get a number
	CALL	DEINT			; Get integer -32768 to 32767
	MOV	BP,SP
	XCHG	[BP],BX			; Save value,get address
	MOV	[BX],DL			; Save LSB of value
;	LAHF
	INC	BX
;	SAHF
	MOV	[BX],DH			; Save MSB of value
	POP	BX			; Restore code string address
	RET
;
; HEX$(nn) Convert 16 bit number to Hexadecimal string
;
HEX:
	CALL	TSTNUM			; Verify it's a number
	CALL	DEINT			; Get integer -32768 to 32767
	PUSH	CX			; Save contents of BC
	MOV	BX,PBUFF
	MOV	AL,DH			; Get high order into A
	CMP	AL,0
	JZ	HEX2			; Skip output if both high digits are zero
	CALL	BYT2ASC			; Convert D to ASCII
	MOV	AL,CH
	CMP	AL,'0'
	JZ	HEX1			; Don't store high digit if zero
	MOV	[BX],CH			; Store it to PBUFF
;	LAHF
	INC	BX			; Next location
;	SAHF
HEX1:
	MOV	[BX],CL			; Store C to PBUFF+1
;	LAHF
	INC	BX			; Next location
;	SAHF
HEX2:
	MOV	AL,DL			; Get lower byte
	CALL	BYT2ASC			; Convert E to ASCII
	MOV	AL,DH
	CMP	AL,0
	JNZ	HEX3			; If upper byte was not zero then always print lower byte
	MOV	AL,CH
	CMP	AL,'0'			; If high digit of lower byte is zero then don't print
	JZ	HEX4
HEX3:
	MOV	[BX],CH			; to PBUFF+2
	LAHF
	INC	BX			; Next location
	SAHF
HEX4:
	MOV	[BX],CL			; to PBUFF+3
;	LAHF
	INC	BX			; PBUFF+4 to zero
;	SAHF
	XOR	AL,AL			; Terminating character
	MOV	[BX],AL			; Store zero to terminate
;	LAHF
	INC	BX			; Make sure PBUFF is terminated
;	SAHF
	MOV	[BX],AL			; Store the double zero there
	POP	CX			; Get BC back
	MOV	BX,PBUFF		; Reset to start of PBUFF
	JMP	STR1			; Convert the PBUFF to a string and return it
;
BYT2ASC:
	MOV	CH,AL			; Save original value
	AND	AL,0FH			; Strip off upper nybbl
	CMP	AL,0AH			; 0-9?
	JC	ADD30			; If A-F, add 7 more
	ADD	AL,07H			; Bring value up to ASCII A-F
ADD30:
	ADD	AL,'0'			; And make ASCII
	MOV	CL,AL			; Save converted char to C
	MOV	AL,CH			; Retrieve original value
	ROR	AL,1			; and Rotate it right
	ROR	AL,1
	ROR	AL,1
	ROR	AL,1
	AND	AL,0FH			; Mask off upper nybble
	CMP	AL,0AH			; 0-9? < A hex?
	JC	ADD301			; Skip Add 7
	ADD	AL,07H			; Bring it up to ASCII A-F
ADD301:
	ADD	AL,'0'			; And make it full ASCII
	MOV	CH,AL			; Store high order byte
	RET
;
; Convert "&Hnnnn" to FPREG
; Gets a character from (HL) checks for Hexadecimal ASCII numbers "&Hnnnn"
; Char is in A, NC if char is;<=>?@ A-z, CY is set if 0-9
HEXTFP:
	XCHG	BX,DX			; Move code string pointer to DE
	MOV	BX,0000H		; Zero out the value
	CALL	GETHEX			; Check the number for valid hex
	JC	HXERR			; First value wasn't hex, HX error
	JMP	HEXLP1			; Convert first character
HEXLP:
	CALL	GETHEX			; Get second and addtional characters
	JC	HEXIT			; Exit if not a hex character
HEXLP1:
	ADD	BX,BX			; Rotate 4 bits to the left
	ADD	BX,BX
	ADD	BX,BX
	ADD	BX,BX
	OR	AL,BL			; Add in D0-D3 into L
	MOV	BL,AL			; Save new value
	JMP	HEXLP			; And continue until all hex characters are in
;
GETHEX:
;	LAHF
	INC	DX			; Next location
;	SAHF
	XCHG	BX,DX
	MOV	AL,[BX]
	XCHG	BX,DX			; Load character at pointer
	CMP	AL,' '
	JZ	GETHEX			; Skip spaces
	SUB	AL,'0'			; Get absolute value
	JNC	GETHEX1
	RET				; < "0", error
GETHEX1:
	CMP	AL,0AH
	JC	NOSUB7			; Is already in the range 0-9
	SUB	AL,07H			; Reduce to A-F
	CMP	AL,0AH			; Value should be $0A-$0F at this point
	JNC	NOSUB7
	RET				; CY set if was :		; < = > ? @
;
NOSUB7:
	CMP	AL,10H			; > Greater than "F"?
	CMC
	RET				; CY set if it wasn't valid hex
;
HEXIT:
	XCHG	BX,DX			; Value into DE, Code string into HL
	MOV	AL,DH			; Load DE into AC
	MOV	CL,DL			; For prep to
	PUSH	BX
	CALL	ACPASS			; ACPASS to set AC as integer into FPREG
	POP	BX
	RET
;
HXERR:
	MOV	DL,HX			; ?HEX Error
	JMP	ERROR
;
; BIN$(NN) Convert integer to a 1-16 char binary string
BIN:
	CALL	TSTNUM			; Verify it's a number
	CALL	DEINT			; Get integer -32768 to 32767
BIN2:
	PUSH	CX			; Save contents of BC
	MOV	BX,PBUFF
	MOV	CH,17			; One higher than max char count
ZEROSUP:				; Suppress leading zero
	DEC	CH			; Max 16 chars
	MOV	AL,CH
	CMP	AL,01H
	JZ	BITOUT			; Always output at least one character
;	RCL	DL,1
;	RCL	DH,1
	RCL	DX,1
	JNC	ZEROSUP
	JMP	BITOUT2
BITOUT:
;	RCL	DL,1
;	RCL	DH,1
	RCL	DX,1
BITOUT2:
	MOV	AL,'0'			; Char for '0'
	ADC	AL,0			; If carry set then '0' --> '1'
	MOV	[BX],AL
;	LAHF
	INC	BX
;	SAHF
	DEC	CH
	JNZ	BITOUT
	XOR	AL,AL			; Terminating character
	MOV	[BX],AL			; Store zero to terminate
;	LAHF
	INC	BX			; Make sure PBUFF is terminated
;	SAHF
	MOV	[BX],AL			; Store the double zero
	POP	CX
	MOV	BX,PBUFF
	JMP	STR1
;
; Convert "&Bnnnn" to FPREG
; Gets a character from (HL) checks for Binary ASCII numbers "&Bnnnn"
BINTFP:
	XCHG	BX,DX			; Move code string pointer
	MOV	BX,0000H		; Zero out the value
	CALL	CHKBIN			; Check the number for valid bin
	JC	BINERR			; First value wasn't bin, HX error
BINIT:
	SUB	AL,'0'
	ADD	BX,BX			; Rotate HL left
	OR	AL,BL
	MOV	BL,AL
	CALL	CHKBIN			; Get second and addtional characters
	JNC	BINIT			; Process if a bin character
	XCHG	BX,DX			; Value into DE, Code string into HL
	MOV	AL,DH			; Load DE into AC
	MOV	CL,DL			; For prep to
	PUSH	BX
	CALL	ACPASS			; ACPASS to set AC as integer into FPREG
	POP	BX
	RET
;
; Char is in A, NC if char is 0 or 1
CHKBIN:
;	LAHF
	INC	DX
;	SAHF
	XCHG	BX,DX
	MOV	AL,[BX]
	XCHG	BX,DX
	CMP	AL,' '
	JZ	CHKBIN			; Skip spaces
	CMP	AL,'0'			; Set C if < '0'
	JNC	CHKBIN1
	RET
CHKBIN1:
	CMP	AL,'2'
	CMC				; Set C if > '1'
	RET
;
BINERR:
	MOV	DL,BN			; ?BIN Error
	JMP	ERROR
;
JJUMP1:
	JMP	CSTART			; Go and initialise
;
MONITR:
	CLI				; Clear Interrupt flag
;	JMP	0C000H			; Restart (Normally Monitor)
	jmp	ret_mon

INITST:
	MOV	AL,0			; Clear break flag
	MOV	[BRKFLG],AL
	JMP	INIT
;
TSTBIT:
;	LAHF
	XCHG	AH,AL
	PUSH	AX			; Save bit mask
	XCHG	AH,AL
	AND	AL,CH			; Get common bits
	POP	CX			; Restore bit mask
	CMP	AL,CH           	; Same bit set?
	MOV	AL,0			; Return 0 in A
	RET
;
OUTNCR:
	CALL	OUTC			; Output character in A
	JMP	PRCRLF			; Output CRLF

CODE_END:

SYSSTK	equ	((CODE_END+1000h)&0f000h)+200h
TB_WORK		equ	SYSSTK
	END
