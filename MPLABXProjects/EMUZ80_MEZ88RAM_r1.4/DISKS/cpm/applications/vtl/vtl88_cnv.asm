	page 0
	cpu 8086
;--------------------------------
;
; Very Tiny Language for CP/M-86
; 2024.01.12 modified by A.honda
;
;--------------------------------

BDOS_CALL	equ	224

CODE_OFF	equ	0
VTL_WORK	equ	100h

; file header

	SEGMENT	CODE
	org	0
;GD1
GD1_GF		db	01h	; code
GD1_GL		dw	CODE_END >> 4	; set code paragraph length
GD1_GB		dw	0000h
GD1_GMIN	dw	CODE_END >> 4	; request minimum size
GD1_GMAX	dw	0000h
GD2_GF		db	02h	; data
GD2_GL		dw	data_size >> 4	; data paragraph length
GD2_GB		dw	0000h
GD2_GMIN	dw	0800h		; minimum work paragraph length
GD2_GMAX	dw	0ff0h		; maximun work paragraph length (64K)

		db	128-$ dup(0);

; VTL88 code body

	ASSUME	CS:CODE, DS:DATA, SS:DATA, ES:NOTHING

	SEGMENT	CODE

	ORG	CODE_OFF

	jmp	CSTART
	jmp	WSTART

;-------------------
;
; start up routine
;
;-------------------

; cold start
start_tb:
	mov	cx, [LD0]	; get last offset
	sub	cx, end_data - 1 ; ger work area
	shr	cx, 1
	xor	ax, ax
	mov	di, end_data

mem_clear:
	mov	[di], ax
	inc	di
	inc	di
	loop	mem_clear

	mov	ax, DS
	MOV	SS,AX
	MOV	SP,VTL_STACK

	mov	ax, CODE_OFF
	mov	[s_val], ax	; set initial BASE value for SEED
	xor	ax, ax		; al : st_flg = 0
	jmp	_main

;
; update random seed
;
update_seed:
	push	ax
	push	bx
	mov	bx, [s_val]
	mov	ax, CS:[bx]
	mov	[SEEDX], ax	; update SEED
	inc	bx
	cmp	bx, CODE_END
	jne	w1
	mov	bx, CODE_OFF
w1:
	mov	[s_val], bx	; update base value
	pop	bx
	pop	ax
	ret
;
; warm start
;
_warm_boot:
	MOV	SP,VTL_STACK
	call	update_seed
	mov	al,1	; al : st_flg = 0

	; warm boot!
	
	push	bp
	mov	bp,sp
	sub	sp,14
	jmp	w_boot

;-------------------------------------
;
; Machine depend I/O interface
;
; USE BDOS CALL ( INT 224 )
;
;------------------------------------
_putchr: ; input al
	
	; put a charactor : CL = 2
	; input : DL : charactor
	push	ax
	mov	cl, 6
	mov	dl, al
	int	BDOS_CALL		; system call
	pop	ax
	ret

_c_getch:
	; get a charactor : CL = 6
	; return AL : charactor
	mov	CL, 6
	mov	dl, 0ffh	; input
	int	BDOS_CALL
	or	al, al
	jz	_c_getch
	call	update_seed
	ret

_c_kbhit:
	; check key status : CL = 06H
	; OUTPUT : AL : 0     ( key is not exist )
	;             : 0FFH  ( key is exist )
	mov	cl, 6
	mov	dl, 0feh	; key status
	int	BDOS_CALL
	ret

_getchr:
	call	_c_getch
	call	_putchr		; al : char
	ret

_mach_fin:
	mov	cl, 0
	mov	dl, 0
	int	BDOS_CALL		; Game END : goto CPM86

_srand:
	push	bp	;Entry sequence
	mov	bp,sp

	mov	ax,[bp+4]	; Load Arg1 into AX
	mov	[SEED], ax
	mov	[SEEDX], ax
	
	pop	bp
	ret

_rand:
	push	cx
	push	dx

	mov	ax, [SEEDX]
	or	ax, ax
	jnz	RND3
	mov	ax, 1
RND3:
	mov	dx, ax
	mov	cl, 5
	shl	dx, cl
	xor	ax, dx
	mov	dx, ax
	mov	cl, 3
	shr	dx, cl
	xor	ax, dx
	push	ax

	mov	ax, [SEED]
	or	ax, ax
	jnz	RND4
	mov	ax, 1
RND4:
	mov	[SEEDX], ax
	mov	dx, ax
	shr	dx, 1
	xor	ax, dx
	pop	dx
	xor	ax, dx
	mov	[SEED], ax	; 0 - FFFFH : -32768 ~ 32767

	pop dx
	pop cx
	ret

_breakCheck:

	call	_c_kbhit
	or	al, al
	jz	br2

	call	_c_getch

	cmp	al, 3
	jnz	br2
	jmp	_warm_boot

br2:
	ret	

;---------------------------
;  Very Tiny Language
;
;  T. Nakagawa
;
;  2004/05/23
;  2004/06/26
;
;---------------------------
_main:
	push	bp
	mov	bp,sp
	sub	sp,14

	; init * and &

	mov	cx, [LD0]	; get last offset
	sub	cx, _Lct+255	; ger work area

	mov	[_Lct+88], cx	; *= program space
	mov	word ptr _Lct+80, 264	; &=264

	mov	ax,1458
	push	ax
	call	_srand
	pop	cx

	mov	ax, opmsg
	push	ax
	call	putstr
	pop	cx

w_boot:
	; for (; ; ) {
	mov	ax, okm
	push	ax
	call	putstr
	pop	cx
nmsg_:
	mov	ax, 138			; ax = Lbf + 2
	mov	[bp-4], ax		; ptr : [bp-4]
	push	ax			; push ptr
	call	getln
	pop	cx
	lea	ax,word ptr [bp-2]	; n : [bp-2]
	push	ax			; push &n
	lea	ax,word ptr [bp-4]
	push	ax			; push &ptr
	call	getnm
	pop	cx
	pop	cx
	or	ax,ax
	jnz	L_5			; } else {

	; if (!getnm(&ptr, &n)) {
	mov	bx, 136
	mov	[bp-6], bx		; line : [bp-6] = Lbf (88h)
	mov	word ptr _Lct[bx],0	; _*(u_short *)(Lct+line)=0
	mov	word ptr _Lct+74,0	; Pcc(#) : [_Lct+74] = 0
L_8:
	; for (; ; ) {
	call	_breakCheck
	push	word ptr [bp-4]		; push ptr
	call	ordr
	pop	cx

	mov	bx, [bp-6]		; get line
	mov	ax, [_Lct+74]		; get Pcc
	or	ax, ax			; Pcc = 0?
	je	L_10
	cmp	ax, [bx+_Lct]		; Pcc = line?
	jne	L_9
L_10:
	; if (line == Lbf) {
	cmp	bx ,136			; line = Lbf (136) ?
	je	w_boot			; break; direct mode

	push	bx			; push line
	call	nxtln
	pop	cx
	mov	[bp-6], ax		; update line

	cmp	ax, [_Lct+80]		; line = Bnd ('&')?
	je	w_boot			; break; detect last line
	jmp	L_13

L_9:	; } else {
	mov	bx, [bp-6]		; get line
	mov	ax, [bx+_Lct]		; get branch No.
	inc	ax
	mov	[_Lct+70] ,ax		; Svp('!') = line + 1

	; if (fndln(&line)) break;
	lea	ax, [bp-6]
	push	ax			; push &line
	call	fndln
	pop	cx
	or	ax,ax
	jnz	w_boot			; break;

L_13:
	mov	bx, [bp-6]		; get line
	mov	ax, [bx+_Lct]		; get line No.
	mov	[_Lct+74], ax		; Pcc = ax

	mov	ax, [bp-6]		; ax : get line
	add	ax, 3			; line + 3
	mov	word ptr [bp-4],ax	; save ax to ptr
	jmp	L_8

L_5:
	; if (n == 0) {
	cmp	word ptr [bp-2],0	; n=0?
	jne	L_16

	; for (ptr = Obj; ptr != READW(Bnd); ) {
	mov	word ptr [bp-4],264	; ptr = Obj(264)
	jmp	L_20
L_19:
	call	_breakCheck
	mov	bx, [bp-4]		; get ptr
	push	[bx+_Lct]		; put *ptr
	call	putnm
	pop	cx

	lea	bx, [bp-4]		; get ptr
	add	word ptr [bx], 2	; ptr+=2
	mov	al,0
	push	ax			; push 0
	push	bx			; push ptr
	call	putl
	pop	cx
	pop	cx
	call	crlf
L_20:
	mov	ax, [bp-4]		; get ptr
	cmp	ax, [_Lct+80]		; ptr = Bnd('&')
	jne	L_19

	jmp	w_boot

L_16:	; /* DELETE */
	mov	ax, [bp-2]		; get n
	mov	[_Lct+74], ax		; *Pcc = n

	; if (!fndln(&cur) 
	lea	ax, [bp-14]		; cur : [bp-14]
	push	ax			; push &cur
	call	fndln
	pop	cx
	or	ax,ax			; fndln(&cur) = 0?
	jne	L_22

	; && READW(cur) == n) {
	mov	bx, [bp-14]
	mov	ax, [bx+_Lct]		; get *cur
	cmp	ax, [bp-2]		; *cur = n?
	jne	L_22

	; src = nxtln(cur);
	push	word ptr [bp-14]	; push cur
	call	nxtln
	pop	cx
	mov	[bp-12], ax		; src : [bp-12] = ax

;for (dst = cur; src != *Bnd; *dst++, *src++) ;

	mov	ax, [bp-14]
	mov	[bp-10], ax		; dst = cur
	jmp	L_26

L_25:
	mov	bx, [bp-12]		; bx = src
	mov	al, [bx+_Lct]		; al = *src
	mov	bx, [bp-10]		; bx = dst
	mov	[bx+_Lct], al		; *dst = *src
	inc	word ptr [bp-12]	; src++
	inc	word ptr [bp-10]	; dst++

L_26:
	mov	ax, [bp-12]
	cmp	ax, [_Lct+80]		; src = & ?
	jne	L_25			; loop next

	; WRITEW(Bnd, dst);
	mov	ax, [bp-10]		; get dst
	mov	[_Lct+80] ,ax		; & = dst

L_22:	; /* INSRT */
	; if (READB(ptr) == '\0') continue;
	mov	bx,word ptr [bp-4]
	cmp	byte ptr _Lct[bx],0	; *ptr = 0?
	jne	L_27
L_15:
	jmp	nmsg_

L_27:
; for (m = 3, tmp = ptr; READB(tmp) != '\0'; m++, tmp++) 

	mov	word ptr [bp-6],3	; m=3
	mov	ax, [bp-4]		; get ptr
	mov	word ptr [bp-8],ax	; tmp : [bp-8] = ptr
	jmp	L_31

L_30:
	inc	word ptr [bp-6]		; m++
	inc	word ptr [bp-8]		; tmp++

L_31:
	mov	bx, [bp-8]
	cmp	byte ptr [bx+_Lct],0	; *tmp= 0?
	jne	L_30			; loop next

	; if (READW(Bnd) + m < READW(Lmt)) {
	mov	ax, [_Lct+80]		; get &
	add	ax, [bp-6]		; & + m
	cmp	ax, [_Lct+88]		; & < *?
	jae	L_15			; memory full

	mov	ax, [_Lct+80]		; get &
	mov	[bp-12], ax		; src = &

	mov	ax, [_Lct+80]
	add	ax, [bp-6]
	mov	[_Lct+80] ,ax		; & = &+m

; for (dst = READW(Bnd); src != cur; WRITEB(--dst, READB(--src))) ;
	mov	[bp-10], ax		; dst = &
	jmp	L_36

L_35:
	dec	word ptr [bp-10]	; --dst
	dec	word ptr [bp-12]	; --src
	mov	bx, [bp-12]
	mov	al, [bx+_Lct]		; *src
	mov	bx, [bp-10]
	mov	[bx+_Lct], al		; *dst = *src

L_36:
	mov	ax, [bp-12]
	cmp	ax, [bp-14]		; src != cur?
	jne	L_35			; next loop

	; WRITEW(src, n);
	mov	ax, [bp-2]
	mov	bx, [bp-12]
	mov	[bx+_Lct],ax		; *src = n

	; src += 2;
	add	bx, 2
	mov	[bp-12], bx		; src +=2

L_39: ; while (WRITEB(src++, READB(ptr++)) != '\0') ;
	mov	bx, [bp-4]
	inc	word ptr [bp-4]		; ptr++
	mov	al, [bx+_Lct]		; *ptr

	mov	bx, [bp-12]		; bx = src
	inc	word ptr [bp-12]
	mov	[bx+_Lct], al		; *src = *ptr

	or	al,al			; al = 0?
	jne	L_39			; loop while

	jmp	nmsg_

fndln:
	push	bp
	mov	bp,sp
	push	si
	mov	si, [bp+4]

	mov	word ptr [si], 264
	jmp	L_45

L_44:
	mov	bx, [si]
	mov	ax, [bx+_Lct]
	cmp	ax, [_Lct+74]
	jb	L_46
	xor	ax,ax
	jmp	L_41
L_46:
	push	[si]
	call	nxtln
	pop	cx
	mov	[si], ax
L_45:
	mov	ax, [si]
	cmp	ax, [_Lct+80]
	jne	L_44
	mov	ax, 1

L_41:
	pop	si
	pop	bp
	ret	

nxtln:
	push	bp
	mov	bp,sp
	add	word ptr [bp+4], 2
L_50:
	mov	bx, [bp+4]
	inc	word ptr [bp+4]
	cmp	byte ptr [bx+_Lct], 0
	jne	L_50
	mov	ax, [bp+4]
	pop	bp
	ret	

getln:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si
	xor	si,si

L_55:	; for (p = 0; ; ) {
	call	_getchr
	mov	[bp-1], al
	cmp	al, 8		; BS
	jne	L_56

	or	si,si
	jle	L_55

	dec	si
	jmp	L_55

L_56:
	cmp	byte ptr [bp-1],13	; CR
	jne	L_59
	mov	bx,word ptr [bp+4]
	add	bx,si
	mov	byte ptr _Lct[bx],0
	mov	al,10
	call	_putchr
	jmp	L_52

L_59:
	cmp	byte ptr [bp-1], 21	; 0x15
	je	L_62
	mov	ax,si
	inc	ax
	cmp	ax,74
	jne	L_61

L_62:
	call	crlf
	xor	si,si
	jmp	L_55

L_61:
	cmp	byte ptr [bp-1], 31	; 0x1f
	jbe	L_55

	mov	al, [bp-1]
	mov	bx, si
	add	bx, [bp+4]
	mov	[bx+_Lct], al
	inc	si
	jmp	L_55

L_52:
	pop	si
	mov	sp,bp
	pop	bp
	ret	

getnm:
	push	bp
	mov	bp,sp
	push	si
	push	di
	mov	si, [bp+6]
	mov	di, [bp+4]
	push	[di]
	call	num
	pop	cx
	or	ax, ax
	je	L_66

	mov	word ptr [si], 0
L_70:
	mov	ax, [si]
	mov	dx, 10
	mul	dx
	mov	[si], ax
	mov	bx, [di]
	inc	word ptr [di]

	mov	al, [bx+_Lct]
	mov	ah,0
	sub	ax, 48			; '0'
	add	[si], ax
	push	[di]
	call	num
	pop	cx
	or	ax, ax
	jne	L_70

	mov	ax,1
L_66:
	pop	di
	pop	si
	pop	bp
	ret	

num:
	push	bp
	mov	bp,sp
	mov	bx, [bp+4]
	mov	al, [bx+_Lct]
	cmp	al, '0'
	jb	L_73

	cmp	al, '9'
	ja	L_73

	mov	ax,1
	pop	bp
	ret	

L_73:
	xor	ax,ax
	pop	bp
	ret	

ordr:
	push	bp
	mov	bp,sp
	sub	sp,8
	lea	ax, [bp-2]
	push	ax
	lea	ax, [bp-3]
	push	ax
	lea	ax, [bp+4]
	push	ax
	call	getvr
	add	sp, 6
	inc	word ptr [bp+4]

	mov	bx, [bp+4]
	mov	al, [bx+_Lct]
	cmp	al, 34
	jne	L_75

	inc	word ptr [bp+4]
	push	ax			; al : 34
	lea	ax, [bp+4]
	push	ax
	call	putl
	pop	cx
	pop	cx

	mov	bx, [bp+4]
	cmp	byte ptr [bx+_Lct], 59
	je	L_77
	call	crlf
	jmp	L_77

L_75:
	lea	ax, [bp-6]
	push	ax
	lea	ax, [bp+4]
	push	ax
	call	expr
	pop	cx
	pop	cx

	cmp	byte ptr [bp-3],36
	jne	L_78

	mov	al, [bp-6]
	call	_putchr
	jmp	L_77

L_78:
	sub	byte ptr [bp-3],63
	mov	al, [bp-3]
	or	al, al
	jne	L_80

	push	word ptr [bp-6]
	call	putnm
	pop	cx
	jmp	L_77

L_80:
	mov	ax, [bp-6]
	mov	bx, [bp-2]
	mov	[bx+_Lct], ax
	call	_rand
	mov	[bp-8], ax
	mov	ax, [bp-8]
	mov	[_Lct+82] ,ax

L_77:
	mov	sp,bp
	pop	bp
	ret	

expr:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si
	mov	si,word ptr [bp+4]
	push	word ptr [bp+6]
	push	si
	call	factr
	pop	cx
	pop	cx
	jmp	L_83

L_85:
	push	word ptr [bp+6]
	push	si
	call	term
	pop	cx
	pop	cx
L_83:
	mov	bx,word ptr [si]
	mov	al,byte ptr _Lct[bx]
	mov	byte ptr [bp-1],al
	or	al,al
	je	L_86
	cmp	byte ptr [bp-1],41
	jne	L_85

L_86:
	inc	word ptr [si]
	pop	si
	mov	sp,bp
	pop	bp
	ret	

factr:
	push	bp
	mov	bp,sp
	sub	sp,4
	push	si
	push	di
	mov	di, [bp+6]
	mov	si, [bp+4]
	mov	bx, [si]
	mov	al, [bx+_Lct]
	cmp	al, 0
	jne	L_88

	mov	[di], al	; [di] <- al
	jmp	L_87

L_88:
	push	di
	push	si
	call	getnm
	pop	cx
	pop	cx
	or	ax,ax
	je	L_89
	jmp	L_87

L_89:
	mov	ax, [si]
	inc	word ptr [si]
	mov	bx, ax
	mov	al, [bx+_Lct]

	mov	[bp-1], al
	cmp	al ,63
	jne	L_90

	mov	ax, 136
	mov	[bp-4], ax
	push	ax
	call	getln
	pop	cx
	push	di
	lea	ax, [bp-4]
	push	ax
	call	expr
	pop	cx
	pop	cx
	jmp	L_87

L_90:
	mov	al, [bp-1]
	cmp	al, 36
	jne	L_92
	call	_getchr
	mov	ah,0
	mov	[di], ax
	jmp	L_87

L_92:
	cmp	al ,40
	jne	L_94
	push	di
	push	si
	call	expr
	pop	cx
	pop	cx
	jmp	L_87

L_94:
	dec	word ptr [si]
	lea	ax, [bp-4]
	push	ax
	lea	ax, [bp-1]
	push	ax
	push	si
	call	getvr
	add	sp,6
	mov	bx, [bp-4]
	mov	ax,[bx+_Lct]
	mov	[di], ax

L_87:
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

term:
	push	bp
	mov	bp,sp
	sub	sp,4
	push	si
	mov	si,word ptr [bp+6]
	mov	bx,word ptr [bp+4]
	mov	ax,word ptr [bx]
	inc	word ptr [bx]
	mov	bx,ax
	mov	al,byte ptr _Lct[bx]
	mov	byte ptr [bp-3],al
	lea	ax,word ptr [bp-2]
	push	ax
	push	word ptr [bp+4]
	call	factr
	pop	cx
	pop	cx
	cmp	byte ptr [bp-3],42
	jne	L_97

	mov	ax,word ptr [si]
	mul	word ptr [bp-2]
	mov	word ptr [si],ax
	jmp	L_98

L_97:
	cmp	byte ptr [bp-3],43
	jne	L_99

	mov	ax,word ptr [bp-2]
	add	word ptr [si],ax
	jmp	L_100

L_99:
	cmp	byte ptr [bp-3],45
	jne	L_101

	mov	ax,word ptr [bp-2]
	sub	word ptr [si],ax
	jmp	L_102

L_101:
	cmp	byte ptr [bp-3],47
	jne	L_103

	mov	ax,word ptr [si]
	xor	dx,dx
	div	word ptr [bp-2]
	mov	word ptr _Lct+78,dx
	mov	ax,word ptr [si]
	xor	dx,dx
	div	word ptr [bp-2]
	mov	word ptr [si],ax
	jmp	L_104

L_103:
	cmp	byte ptr [bp-3],61
	jne	L_105

	mov	ax,word ptr [si]
	cmp	ax,word ptr [bp-2]
	jne	L_110
	mov	ax,1
	jmp	L_109

L_110:
	xor	ax,ax
L_109:
	mov	word ptr [si],ax
	jmp	L_106

L_105:
	cmp	byte ptr [bp-3],62
	jne	L_107

	mov	ax,word ptr [si]
	cmp	ax,word ptr [bp-2]
	jb	L_112
	mov	ax,1
	jmp	L_111

L_112:
	xor	ax,ax
L_111:
	mov	word ptr [si],ax
	jmp	L_108

L_107:
	mov	ax,word ptr [si]
	cmp	ax,word ptr [bp-2]
	jae	L_114
	mov	ax,1
	jmp	L_113
L_114:
	xor	ax,ax
L_113:
	mov	word ptr [si],ax
L_108:
L_106:
L_104:
L_102:
L_100:
L_98:
	pop	si
	mov	sp,bp
	pop	bp
	ret	

getvr:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si
	mov	si, [bp+6]

	mov	bx, [bp+4]
	mov	ax, [bx]
	inc	word ptr [bx]
	mov	bx, ax
	mov	al, [bx+_Lct]
	mov	[si], al

	cmp	byte ptr [si], 58
	jne	gv_120

	lea	ax, [bp-2]
	push	ax
	push	word ptr [bp+4]
	call	expr
	pop	cx
	pop	cx

	mov	ax, [bp-2]
	shl	ax, 1
	add	ax, [_Lct+80]
	mov	bx, [bp+8]
	mov	[bx], ax
	jmp	gv_121

gv_120:
	cmp	byte ptr [si], 7fh
	jne	gv_122
	jmp	_mach_fin

gv_122:
	mov	al, [si]
	mov	ah, 0
	and	ax, 63
	inc	ax
	inc	ax
	shl	ax, 1
	mov	bx, [bp+8]
	mov	[bx], ax

gv_121:
	pop	si
	mov	sp,bp
	pop	bp
	ret	

putl:
	push	bp
	mov	bp,sp
	push	si
	mov	si,word ptr [bp+4]
L_121:
	mov	bx,word ptr [si]
	mov	al, [bx+_Lct]
	cmp	al,byte ptr [bp+6]
	je	L_120
	call	_putchr
	inc	word ptr [si]
	jmp	L_121

L_120:
	inc	word ptr [si]
	pop	si
	pop	bp
	ret	

crlf:
	mov	al,13
	call	_putchr
	mov	al,10
	call	_putchr
	ret	

putnm:
	push	bp
	mov	bp,sp
	sub	sp,4
	mov	word ptr [bp-4],135
	mov	bx,word ptr [bp-4]
	mov	byte ptr _Lct[bx],0
L_126:
	mov	ax,word ptr [bp+4]
	mov	bx,10
	xor	dx,dx
	div	bx
	mov	byte ptr [bp-1],dl
	mov	ax,word ptr [bp+4]
	mov	bx,10
	xor	dx,dx
	div	bx
	mov	word ptr [bp+4],ax
	dec	word ptr [bp-4]
	mov	al,byte ptr [bp-1]
	add	al,48
	mov	bx,word ptr [bp-4]
	mov	byte ptr _Lct[bx],al
	cmp	word ptr [bp+4],0
	jne	L_126

	mov	al,0
	push	ax
	lea	ax,word ptr [bp-4]
	push	ax
	call	putl
	pop	cx
	pop	cx
	mov	sp,bp
	pop	bp
	ret	

putstr:
	push	bp
	mov	bp,sp
	jmp	L_129

L_131:
	mov	bx,word ptr [bp+4]
	inc	word ptr [bp+4]
	mov	al, [bx]
	call	_putchr
L_129:
	mov	bx,word ptr [bp+4]
	cmp	byte ptr [bx],0
	jne	L_131

	call	crlf
	pop	bp
	ret	

CSTART:	jmp	start_tb
WSTART:	jmp	_warm_boot

	db	($ & 0FF00H)+100H-$ dup(0FFH)

CODE_END:

	SEGMENT	DATA
	org	0
;
; CP/M-86 Base Page definition
;
BASE_PAGE	equ	0
LC0	equ	BASE_PAGE+0	;00H Last CODE address Low
LC1	equ	BASE_PAGE+1	;01H Last CODE address Middle
LC2	equ	BASE_PAGE+2	;02H Last CODE address High (8080model must be 0)
BC0	equ	BASE_PAGE+3	;03H Base Paragraph address of CODE Low
BC1	equ	BASE_PAGE+4	;04H Base Paragraph address of CODE High
M80	equ	BASE_PAGE+5	;05H 1:(8080model) 0:(other)
LD0	equ	BASE_PAGE+6	;06H LAST DATA address Low
LD1	equ	BASE_PAGE+7	;07H LAST DATA address Middle
LD2	equ	BASE_PAGE+8	;08H LAST DATA address High
BD0	equ	BASE_PAGE+9	;09H Base Paragraph address of DATA Low
BD1	equ	BASE_PAGE+10	;0AH Base Paragraph address of DATA High
; option
LE0	equ	BASE_PAGE+12	;0CH Last EXTRA address Low
LE1	equ	BASE_PAGE+13	;0DH Last EXTRA address Middle
LE2	equ	BASE_PAGE+14	;0EH Last EXTRA address High
BE0	equ	BASE_PAGE+15	;0FH Base Paragraph address of EXTRA Low
BE1	equ	BASE_PAGE+16	;10H Base Paragraph address of EXTRA High
LS0	equ	BASE_PAGE+18	;12H Last STASK address Low
LS1	equ	BASE_PAGE+19	;13H Last STASK address Middle
LS2	equ	BASE_PAGE+20	;14H Last STASK address High
BS0	equ	BASE_PAGE+21	;15H Base Paragraph address of STACK Low
BS1	equ	BASE_PAGE+22	;16H Base Paragraph address of STACK High

	db	256 dup(0)

;	org	VTL_WORK

opmsg:	db	"VTL-C CP/M-86 edition.",0
okm:	db	"\r\nOK",0

end_data	equ	$

SEED:		ds	2
SEEDX:		ds	2
s_val:		ds	2

data_size	equ	($+10h) & 0fff0h

		ds	(($+200h) & 0ff00h) - $	;stack area
VTL_STACK:

_Lct		equ	VTL_STACK	;program area

	end
