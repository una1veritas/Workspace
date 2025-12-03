	page 0
	CPU	8086

; I/O system for Version 2.x of MSDOS.
;
; 2025/09/03 modify for Dos V3.1x
; This BIOS combines with the SYSINIT module in a binary file.
; Created by Akihito Honda.

BIOSIZ  EQU     0DB0h           ;MSDOS.SYS loading location( except INT vector )
				;MSDOS.SYS load location = 0DB0H (IO.SYS size = DA0H)+10H
				;BIOSIZ = MSDOS.SYS in Load location = 0DB0H
				;INT Vector area size =400H.
				;real MSDOS.SYS loading location in MEMORY at 400H + DB0H = 11B0H
BIOSIZS EQU     BIOSIZ/10H      ;MSDOS.SYS loading Paragraphs. DCH
ANSI    EQU     0               ;Ansi switch.

;Additional Information for the MEZ86_RAM.

DUMMY_PORT	equ	0f000h
QSIZE   EQU     100             ;Input queue size.
BIOSSEG EQU     40H             ;I/O system segment.(IO.SYS segment)
MAX_MEM EQU     07F00H          ;Memory size in paragraphs. (508K)
uni_seg	equ	0FF50h		;unimon code segment
sysinit_ss	equ	MAX_MEM	;sysinit initial stack segment
sysinit_sp	equ	100h	; sysinit initial stack offset

UREQ_COM	equ	0000h	; unimon request command
UNI_CHR		equ	0001h	; unimon request charcter
DREQ_COM	equ	0006h	; device request command
PTRSAV_OFF	equ	0008h	; request header address save offset
PTRSAV_SEG	equ	000ah	; request header segment save offset


DRVMAX		equ	4

;Things needed to communicate with SYSINIT

SYSINIT			equ	BIOSSEG+((end_mez+1)/16)	; 0064h (word)
DOSINFO			equ	0003h	; word (MUST BE 0)
CURRENT_DOS_LOCATION	equ	0005h	; word
MSDOS			equ	0007h	; word
ENTRY_POINT		equ	0007h	; word (MUST BE 0)
FINAL_DOS_LOCATION	equ	0009h	; word
DEVICE_LIST		equ	000bh	; double word
MEMORY_SIZE		equ	000fh	; word
DEFAULT_DRIVE		equ	0011h	; byte	default 0(NODRVSET) A=1,B=2,,,,
; do not need set defaults value
BUFFERS			equ	0012h	; byte  default 2
FILES			equ	0014h	; byte  default 8

;;;
;;; BIOS area
;;;

	SEGMENT	CODE

	ASSUME  CS:CODE		;,DS:CODE,ES:CODE,SS:CODE

        ORG     0               ;Starts at an offset of zero.

INIT:   JMP     HWINIT


	dottedstructs on
DPB     STRUC		; BOOT sector definition

JMPNEAR DB      3 DUP (?)       ;Jmp Near xxxx  for boot.
NAMEVER DB      8 DUP (?)       ;Name / Version of OS.

;-------  Start of Drive Parameter Block. (or also known as BPB:Bios Parameter Block)

SECSIZE DW      ?               ;Sector size in bytes.                  (bpb)
ALLOC   DB      ?               ;Number of sectors per alloc. block.    (bpb)
RESSEC  DW      ?               ;Reserved sectors.                      (bpb)
FATS    DB      ?               ;Number of FAT's.                       (bpb)
MAXDIR  DW      ?               ;Number of root directory entries.      (bpb)
SECTORS DW      ?               ;Number of sectors per diskette.        (bpb)
MEDIAID DB      ?               ;Media byte ID.                         (bpb)
FATSEC  DW      ?               ;Number of FAT Sectors.                 (bpb)

;-------  End of Drive Parameter Block.

SECTRK  DW      ?               ;Number of Sectors per track.

DPB     ENDS

; DPB and pointer are initialized
;
INITTAB:
        DW	DSKA.SECSIZE
        DW	DSKB.SECSIZE
        DW	DSKC.SECSIZE
        DW	DSKD.SECSIZE

DSKA DPB
DSKB DPB
DSKC DPB
DSKD DPB


;-----------------------------------------------+
;     DWORD pointer to next device              | 1 word offset.
;         (-1,-1 if last device)                | 1 word segement.
;-----------------------------------------------+
;     Device attribute WORD                     ; 1 word.
;       Bit 15 = 1 for chacter devices.         ;
;                0 for Block devices.           ;
;                                               ;
;       Charcter devices. (Bit 15=1)            ;
;         Bit 0 = 1  current sti device.        ;
;         Bit 1 = 1  current sto device.        ;
;         Bit 2 = 1  current NUL device.        ;
;         Bit 3 = 1  current Clock device.      ;
;                                               ;
;         Bit 13 = 1 for non IBM machines.      ;
;                  0 for IBM machines only.     ;
;         Bit 14 = 1 IOCTL control bit.         ;
;-----------------------------------------------+
;     Device strategy pointer.                  ; 1 word offset.
;-----------------------------------------------+
;     Device interrupt pointer.                 ; 1 word offset.
;-----------------------------------------------+
;     Device name field.                        ; 8 bytes.
;       Character devices are any valid name    ;
;         left justified, in a space filled     ;
;         field.                                ;
;       Block devices contain # of units in     ;
;         the first byte.                       ;
;-----------------------------------------------+

; Device Header

DEVSTART:
CONDEV:                         ;Header for device CON
        DW      AUXDEV,BIOSSEG  ;Link to next device
        DW      8003H           ;Attributes - console input, output device
        DW      STRATEGY        ;Srategy entry point
        DW      CON_INT         ;Interrupt entry point
        DB      "CON     "      ;Device name

AUXDEV:                         ;Header for device AUX
        DW      PRNDEV,BIOSSEG
        DW      8000H
        DW      STRATEGY
        DW      AUX_INT
        DB      "AUX     "

PRNDEV:                         ;Header for device PRN
        DW      TIMDEV,BIOSSEG
        DW      8000H
        DW      STRATEGY
        DW      PRN_INT
        DB      "PRN     "

TIMDEV:                         ;Header for device CLOCK
        DW      DSKDEV,BIOSSEG
        DW      8008H
        DW      STRATEGY
        DW      TIM_INT
        DB      "CLOCK   "

DSKDEV:                         ;Header for disk devices
        DW      -1,-1           ;Last device
;        DW      2000H           ;Is a block device
        DW      2800H           ;Is a block device DOS3.1 support
        DW      STRATEGY
        DW      DSK_INT         ;Interrupt entry point
	DB      DRVMAX          ;Number of Units
        DB      7 DUP (0)


;Define offsets for io data packet

IODAT   STRUC
CMDLEN  DB      ?               ;LENGTH OF THIS COMMAND
UNIT    DB      ?               ;SUB UNIT SPECIFIER
CMD     DB      ?               ;COMMAND CODE
STATUS  DW      ?               ;STATUS
        DB      8 DUP (?)
MEDIA   DB      ?               ;MEDIA DESCRIPTOR
TRANS   DD      ?               ;TRANSFER ADDRESS
COUNT   DW      ?               ;COUNT OF BLOCKS OR CHARACTERS
START   DW      ?               ;FIRST BLOCK TO TRANSFER
IODAT   ENDS

PTRSAV  DD      0               ;Strategy pointer save.
;
; Simplistic Strategy routine for non-multi-Tasking system.
;
;   Currently just saves I/O packet pointers in PTRSAV for
;   later processing by the individual interrupt routines.
;

;STRATP  PROC    FAR

;        PUBLIC  RE_INIT

STRATEGY:
        MOV     WORD PTR CS:[PTRSAV],BX
        MOV     WORD PTR CS:[PTRSAV+2],ES

RE_INIT:
        RETF

;STRATP  ENDP

AUX_INT:
	push	si
	mov	si, 0010h	; set device code
	jmp	ENTRY

PRN_INT:
	push	si
	mov	si, 0020h	; set device code
	jmp	ENTRY

TIM_INT:
	push	si
	mov	si, 0030h	; set device code
	jmp	ENTRY

CON_INT:
	push	si
	mov	si, 0040h	; set device code
	jmp	ENTRY

DSK_INT:
	push	si
	mov	si, 0050h	; set device code

ENTRY:  PUSH    AX              ;Save all nessacary registers.
        PUSH    DX
        PUSH    DS
	pushf
	
	cli
	mov	ax, uni_seg
	mov	ds, ax
	
	mov	ax, si
	mov	[DREQ_COM], al
	mov	byte ptr [UREQ_COM],0	; clear unimon command
	mov	si, word ptr cs:[PTRSAV]
        MOV     [PTRSAV_OFF],si
	mov	ax, word ptr cs:[PTRSAV+2]
        MOV     [PTRSAV_SEG],ax
	mov	ds, ax

	mov	ax,[si+IODAT.STATUS]
	and	ax, 1111111011111111b	; clear done bit
	mov	[si+IODAT.STATUS],ax

	mov	dx, DUMMY_PORT
	in	al, dx			; invoke PIC F/W

wait_done:
	mov	ax,[si+IODAT.STATUS]
	test	ax, 0000000100000000b	; check done bit
	jz	wait_done

	test	ah, 10000000b	; check error
	jnz	err_done
	xor	al, al
err_done:

;EXITP   PROC    FAR             ;Normal exit for device drivers.

	popf
        POP     DS
        POP     DX
        POP     AX
        POP     SI
        RETF                             ;RESTORE REGS AND RETURN
;EXITP   ENDP

	IF      ANSI

;
;ANSI Info and routines. ANSI driver implemented as a finite state automata
;This ANSI driver translates the ANSI standard escape sequences into the
; Zenith Escape sequences used on the Zenith(Heath) Z(H)-19 terminal.
;This is not a full implementation of ANSI, but rather a minimal implementation
; which implements all of the necessary ANSI functions.
;

ESC     EQU     1BH             ;Escape character used in this implementation.
STATE   DW      ST1             ;Current ANSI character state.
PRMPNT  DW      PARMS           ;Current parameter pointer.
PARMS   DB      0,0,0,0,0,0,0   ;Allow for up to eight parameters.
LASTPRM DB      0               ;With this being the eight one.

CMDTABL DB      'A'             ;Cursor up.  "esc","[",#,"A"
        DW      CUU
        DB      'B'             ;Cursor down. "esc","[",#,"B"
        DW      CUD
        DB      'C'             ;Cursor forward. "esc","[",#,"C"
        DW      CUF
        DB      'D'             ;Cursor back. "esc","[",#,"D"
        DW      CUB
        DB      'H'             ;Direct cursor posit. "esc","[",x,y,"H"
        DW      CUP
        DB      'J'             ;Erase. "esc","[",code,"J"
        DW      ED
        DB      'K'             ;Erase in line. "esc","[",code,"K"
        DW      EL
        DB      'f'             ;Direct cursor posit. "esc","[",x,y,"f"
        DW      CUP
        DB      'm'             ;Special video mode. "esc","[",code,"m"
        DW      SGR
        DB      's'             ;Save cursor posit. "esc","[","s"
        DW      PSCP
        DB      'u'             ;Move cursor to saved. "esc","[","u"
        DW      PRCP
        DB      00              ;End of table.

;
; ANSI console output driver.
;

CONOUT: MOV     DI,OFFSET STATE ;Retrieve current ansi state.
        JMP     [DI]            ;Jump to it.

;
; State one (1).
;   Looks for an Escape character.
;

ST1:    CMP     AL,ESC          ;See if this the first character is ESC.
        JNZ     OUTCHR          ;No, treat as regular character output.
        MOV     WORD PTR [DI],OFFSET ST2        ;Yes, setup state two.
        RET

;
; State two (2).
;   Looks for the "[" character.
;

ST2:    CMP     AL,'['          ;See if a valide state two.
        JNZ     OUTCHR          ;No, treat as regular charcter
        MOV     BX,OFFSET PARMS ;Yes, get parameter pointer.
        MOV     WORD PTR [PRMPNT],BX    ;Setup in pointer index.
        MOV     WORD PTR [BX],0 ;Clear first entry.
        MOV     WORD PTR [DI],OFFSET ST3;Setup for state three.
        RET

;
; State three (3).
;   Entered one or more times for parameter passing.
;

ST3:    CMP     AL,';'          ;Look for decimal # seperator.
        JNZ     ST3A            ;No check phase A.
        INC     WORD PTR [PRMPNT]       ;Yes, incr. pointer to next param.
        MOV     AX,OFFSET LASTPRM       ;Check for outside parameter list.
        CMP     [PRMPNT],AX
        JBE     RETST3          ;Yes, proceed with next parameter.
        MOV     [PRMPNT],AX     ;No, treat as extentsion to old.
RETST3: MOV     DI,[PRMPNT]     ;Setup for next parameter.
        MOV     BYTE PTR [DI],0 ;Pre-Initialize it to zero.
        RET

;
; State three A (3A).
;   Check for a ascii digit.
;

ST3A:   CMP     AL,'0'          ;Check for ASCII digit.
        JB      ST3B            ;No, check for seconday command character.
        CMP     AL,'9'          ;Still checking for ASCII digit.
        JA      ST3B            ;No, it must be a secondary.
        SUB     AL,'0'          ;Convert to binary.
        MOV     DI,[PRMPNT]     ;Get the current parameter pointer.
        XCHG    [DI],AL         ;Get existing #.
        MOV     AH,10           ;Scale by 10.
        MUL     AH
        ADD     [DI],AL         ;Add to new digit.
        RET

;
; State three B (3B).
;   Wasn't a ascii digit, so check for secondary command.
;

ST3B:   MOV     [DI],OFFSET ST1         ;Preset STATE to state 1 just in case.
        MOV     DI,OFFSET PARMS-1       ;Get pointer to start of parameters.
        MOV     [PRMPNT],DI             ;Save it in Parameter pointer.
        MOV     DI,OFFSET CMDTABL-3     ;Get start of Secondary command table.

ST3B1:  ADD     DI,3            ;Update Command table pointer.
        CMP     BYTE PTR [DI],0 ;Check for end of table.
        JNZ     ST3B2           ;No, continue processing.
        JMP     OUTCHR          ;Yes, treat as regular character.
ST3B2:  CMP     AL,[DI]         ;Check for valid. command.
        JNZ     ST3B1           ;No, keep checking.
        JMP     [DI+1]          ;Yes, transfer to that secondary command.

;
; Get binary parameter from storage and return a one if = 0
;

GETONE: CALL    GETPARM         ;Get parameter form list.
        OR      AL,AL           ;Verify for non-zero.
        JNZ     GETRET          ;Good, then return to caller.
        INC     AL              ;Bad, make it at least a one.
GETRET: CBW                     ;Sign extend AL.
        MOV     CX,AX           ;Copy of it to CX.
        RET

GETPARM:INC     WORD PTR [PRMPNT]       ;Increment parameter pointer.
GOTPARM:MOV     DI,[PRMPNT]     ;Get parameter pointer.
        MOV     AL,[DI]         ;Get parameter value.
        RET

;
; Send escape, character sequence.
;

OUTESC: MOV     AL,ESC          ;Send escape character.
        CALL    OUTCHR
        MOV     AL,BL           ;Send follow character.
        JMP     OUTCHR

;
; Cursor Positioning routines.
;

CUU:    MOV     BL,'A'          ;Cursor up.
        JMP     SHORT CURPOS
CUD:    MOV     BL,'B'          ;Cursor down.
        JMP     SHORT CURPOS
CUF:    MOV     BL,'C'          ;Cursor forward.
        JMP     SHORT CURPOS
CUB:    MOV     BL,'D'          ;Cursor back.

CURPOS: CALL    GETONE          ;Get number of positions to move into CX.
MOVCUR: CALL    OUTESC          ;Send escape, command characters.
        LOOP    MOVCUR          ;Keep moving until done.
        RET

;
; Direct cursor positioning routine.
;

CUP:    CALL    GETONE          ;Get X position.
        MOV     DX,AX           ;Save in DX.
        CALL    GETONE          ;Get Y position.
        MOV     BL,'Y'
        CALL    OUTESC          ;Send escape, "Y" sequence.
        MOV     AL,DL
        ADD     AL,' '-1        ;Convert binary to Character.
        CALL    OUTCHR          ;Send X posit.
        MOV     AL,CL
        ADD     AL,' '-1        ;Convert binary to Character.
        JMP     OUTCHR          ;Send Y posit.

;
; Erase all/part of screen.
;

ED:     CALL    GETPARM         ;Get trinary command type.
        MOV     BL,'b'
        DEC     AL              ;See if erase from begining of screen.
        JZ      ED1             ;Yes, perform ZDS function.
        MOV     BL,'E'
        DEC     AL              ;See if erase from end of screen.
        JZ      ED1             ;Yes, perform ZDS function.
        MOV     BL,'J'          ;Now we assume erase whole screen.
ED1:    JMP     OUTESC

;
; Erase all/part of a line.
;

EL:     CALL    GETPARM         ;Get trinary command type.
        MOV     BL,'o'
        DEC     AL              ;See if erase from begining of line.
        JZ      EL1             ;Yes, perform ZDS function.
        MOV     BL,'l'
        DEC     AL              ;See if erase whole line.
        JZ      EL1             ;Yes, perform ZDS function.
        MOV     BL,'K'          ;Now we assume erase to end of line.
EL1:    JMP     OUTESC

;
; Special video modes.
;

SGR:    CALL    GETPARM         ;Get trinary command type.
        MOV     BL,'p'
        CMP     AL,7            ;See if enter reverse video mode.
        JZ      SGR2            ;Yes, perform ZDS function.
        MOV     BL,'q'
        OR      AL,AL           ;See if exit reverse video mode.
        JNZ     SGR3            ;No, ignore.
SGR2:   CALL    OUTESC
SGR3:   RET

;
; Save / restore cursor position.
;

PSCP:   MOV     BL,'j'          ;Set save cursor posit. mode.
        JMP     OUTESC

PRCP:   MOV     BL,'k'          ;Restore last cursor save.
        JMP     OUTESC

        ENDIF			; ANSI



;
; Overlayed by MSDOS by SYSINIT.
;

WRKSTK:	DB      100 DUP (0)

REQ_CONOUT	equ	2

HWINIT: mov     BP, sysinit_ss		; MEMORY FF00:0 - FF20:0 : free space(FF30:0 - used unimon)
        MOV     SS,BP
        MOV     SP, sysinit_sp		;Some nice area for stack. (free area in EMU8088/V20)
					;ss:sp=FF00:0100H
        PUSH    CS
        POP     ES
        MOV     AX,SYSINIT
        MOV     DS,AX

	mov	byte ptr [DEFAULT_DRIVE], 3	; set DRIVE C

        MOV     AX,CS				; CS: IO.SYS segment (40H)
        ADD     AX,BIOSIZS			; 0DCH
        MOV     [CURRENT_DOS_LOCATION],AX	; save DCH+40H = 11CH
        MOV     word ptr [MEMORY_SIZE],MAX_MEM		; save FF00H
        MOV     AX,CS
        MOV     [DEVICE_LIST+2],AX	;		; save device headder segment(40H)
        MOV     WORD PTR [DEVICE_LIST],DEVSTART		; save device hedder offset
        ADD     AX,((WRKSTK - INIT)+16) /16
        MOV     [FINAL_DOS_LOCATION],AX			; set MSDOS.SYS entory point(final location)

	mov	si, IOSYS_MSG
conout_loop:
	mov	al, cs:[si]
	cmp	al, 0
	jz	go_sysinit
	call	CONOUT
	inc	si
	jmp	conout_loop

go_sysinit:
        JMPF     SYSINIT:0

CONOUT:
	push	ds
	mov	bx, uni_seg
	mov	ds, bx

	mov	byte ptr [UREQ_COM], REQ_CONOUT	; set CONOUT request
	mov	[UNI_CHR], al

	mov	dx, DUMMY_PORT
	in	al, dx				; invoke PIC F/W
wait_conin:
	mov	al, [UREQ_COM]
	or	al, al
	jnz	wait_conin

	mov	al, [UNI_CHR]		; get char or status
	and	al, al
	pop	ds
	ret

IOSYS_MSG:
	db	13,10,"Initialize MEZ88_RAM IO.SYS",13,10,0

	db	((($+16)/16)*16)-$ dup(0)
end_mez	equ	$-1
        END
