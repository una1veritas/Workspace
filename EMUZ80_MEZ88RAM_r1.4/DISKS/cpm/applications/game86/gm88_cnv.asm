	page 0
	cpu 8086
;--------------------------------
;
; GAME Interpreter for CP/M-86
; 2024.01.12 modified by A.honda
;
;--------------------------------

BDOS_CALL	equ	224		; int 0e0h

GM_OFF		equ	0h
BS_PAGE		equ	0	; BASE PAGE
GM88_WORK	equ	100h

	ASSUME	CS:CODE, DS:DATA, SS:DATA, ES:NOTHING

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

; Game-86 code body


	SEGMENT	CODE

	ORG	GM_OFF

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
	sub	cx, end_data + 1 ; ger work area
	shr	cx, 1
	xor	ax, ax
	mov	di, end_data

mem_clear:
	mov	[di], ax
	inc	di
	inc	di
	loop	mem_clear

	; set stack
	MOV	AX, DS
	MOV	SS,AX
	MOV	SP,GM_STACK

	mov	ax, GM_OFF
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
	mov	ax, cs:[bx]
	mov	[SEEDX], ax	; update SEED
	inc	bx
	cmp	bx, CODE_END
	jne	w1
	mov	bx, GM_OFF
w1:
	mov	[s_val], bx	; update base value
	pop	bx
	pop	ax
	ret
;
; warm start
;
_warm_boot:
	MOV	SP,GM_STACK
	call	update_seed
	mov	al,1	; al : st_flg = 0
	jmp	_main

;-------------------------------------
;
; BDOS CALL I/O interface
;
;------------------------------------
_c_putch: ; input al
	
	; put a charactor : CL = 2
	; input : DL : charactor
	push	es
	push	bx
	mov	cl, 6
	mov	dl, al
	int	BDOS_CALL		; system call
	pop	bx
	pop	es
	ret

_c_getch:
	; get a charactor : CL = 6
	; return AL : charactor
	push	es
	push	bx
re_call:
	mov	CL, 6
	mov	dl, 0ffh	; input
	int	BDOS_CALL
	or	al, al
	jz	re_call
	mov	ah, 0
	call	update_seed
	pop	bx
	pop	es
	ret

_c_kbhit:
	; check key status : CL = 06H
	; OUTPUT : AL : 0     ( key is not exist )
	;             : 0FFH  ( key is exist )
	push	es
	push	bx
	mov	cl, 6
	mov	dl, 0feh	; key status
	int	BDOS_CALL
	mov	ah, 0
	pop	bx
	pop	es
	ret

_mach_fin:
	mov	cl, 0
	mov	dl, 0
	int	BDOS_CALL		; Game END : goto CPM86

;-------------------------
; set random seed number.
;-------------------------
_srand:
	push	bp	;Entry sequence
	mov	bp,sp

	mov	ax,[bp+4]	; Load Arg1 into AX
	mov	[SEED], ax
	mov	[SEEDX], ax
	
	pop	bp
	ret

;-------------------------
; get random number
;-------------------------
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
	and	ax, 7fffh	; 0 - 7FFFH : 0 ~ 32767

	pop dx
	pop cx
	ret

;-------------------------
; print strings
;-------------------------
_c_puts:
	push	bp
	mov	bp,sp

	mov	bx, [bp+4]	; set s addr

get_nxtchr:
	mov	al, [bx]	; get char
	or	al, al		; NULL?
	je	L_1
	call	_c_putch
	inc	bx		; s++
	jmp	get_nxtchr

L_1:
	mov	[bp+4], bx	; update s
	pop	bp
	ret	

CSTART:	jmp	start_tb
WSTART:	jmp	_warm_boot

;--------------------------------------------------------------
; GAME Language interpreter ,32bit Takeoka ver.
; by Shozo TAKEOKA (http://www.takeoka.org/~take/ )
;--------------------------------------------------------------

; al : st_flg
_main:
	mov	bp,sp
	sub	sp,2
	or	al, al		; if ( !st_flg ) {
	jnz	L_6

	mov	ax,5678
	push	ax
	call	_srand
	pop	cx

	mov	ax,_text_buf
	mov	word ptr _var+122,ax		; 122 : '=' *2
	; save program space end
	mov	ax, [LD0]
	mov	word ptr _var+84,ax		; 84 : '*' *2
	call	_newText1
	mov	ax, op_msg_
	push	ax
	call	_c_puts
	pop	cx
L_6:
	mov	ax, rdymsg_
	push	ax
	call	_c_puts
	pop	cx
L_9:
	mov	word ptr [_sp],-1
	mov	word ptr _lno,0
	mov	ax,_lin
	push	ax
	call	_c_gets
	pop	cx
	mov	di,ax
	mov	byte ptr _lin[di+1],-128
	mov	word ptr _pc,_lin
	call	_skipBlank
	lea	ax,word ptr [bp-2]
	push	ax
	call	_getNum
	pop	cx
	mov	si,ax
	cmp	word ptr [bp-2],0
	jne	L_10

	call	_exqt
	call	_newline
	mov	ax, rdymsg_
	push	ax
	call	_c_puts
	pop	cx
	jmp	L_11
L_10:
	push	si
	call	_edit
	pop	cx
L_11:
	jmp	L_9

_skipLine:
	push	bp
	mov	bp,sp
	push	si
	mov	si,word ptr [bp+4]
	jmp	L_16
L_15:
	inc	si
L_16:	cmp	byte ptr [si],0
	jne	L_15

	mov	ax,si
	inc	ax
	pop	si
	pop	bp
	ret	

_searchLine:
	push	bp
	mov	bp,sp
	push	si
	push	di

	xor	cx, cx		; f = 0 (no much)
;	mov	si, [_var+58]	; si: TOPP '='
	mov	si, [_var+122]	; 122 : '='*2

sl_loop:
	mov	al, [si]
	test	al, 80h
	jnz	sl_nmch
	
	mov	ah, al		; *sp << 8
	mov	al, [si+1]	; *(p+1)

	cmp	ax, [bp+4]	; cmp l, n (n):[bp+4], (l):ax
	jae	sl_endp

	; p=skipLine(p+2);
	inc	si
	inc	si		; p+2

	push	cx		; save f
	push	si
	call	_skipLine
	mov	si, ax
	pop	ax		; dummy pop

	pop	cx		; restore f
	jmp	sl_loop

sl_endp:
	ja	sl_nmch
	mov	cl, 1		; f=1 (set much flag)
sl_nmch:
	mov	bx, [bp+6]
	mov	[bx], cx	; set *f (0 or 1)
	mov	ax, si

	pop	di
	pop	si
	pop	bp
	ret	

_edit:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si
	push	di
	mov	di,word ptr [bp+4]
	or	di,di
	jne	L_25

;	push	word ptr _var+58
	push	word ptr _var+122	; 122 : '='*2
	call	_dispList
	pop	cx
	jmp	_warm_boot

L_25:
	lea	ax,word ptr [bp-2]
	push	ax
	push	di
	call	_searchLine
	pop	cx
	pop	cx
	mov	si,ax
	mov	bx,word ptr _pc
	cmp	byte ptr [bx],47
	jne	L_26

	push	si
	call	_dispList
	pop	cx
	jmp	_warm_boot

L_26:
;	mov	bx,word ptr _var+12
	mov	bx,word ptr _var+76	; 76 : '&' *2
	cmp	byte ptr [bx],255
	je	L_28

	mov	ax, t_lockm
	push	ax
	call	_er_boot

L_28:
	cmp	word ptr [bp-2],0
	je	L_29

	push	si
	call	_deleteLine
	pop	cx
L_29:
	mov	bx, [_pc]
	cmp	byte ptr [bx],0
	jne	L_30
	xor	ax,ax
	jmp	L_24
L_30:
	push	bx	; bx = pc
	push	si
	push	di
	call	_addLine
	add	sp,6
	xor	ax,ax
L_24:
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

_addLine:
	push	bp
	mov	bp,sp
	push	si
	push	di
	mov	si,word ptr [bp+6]

	push	word ptr [bp+8]
	call	_strlen
	pop	cx
	mov	di,ax
	add	di,3
;	mov	ax,word ptr _var+12
	mov	ax,word ptr _var+76	; 76 : '&' *2
	sub	ax,si
	inc	ax	;ax = (((u_char*)BTMP)-p)+1
	push	ax
	push	si
	mov	ax,si
	add	ax,di	;ax = p+l
	push	ax
	call	_memmove
	add	sp,6
	mov	ax,word ptr [bp+4]
	mov	cl,8
	sar	ax,cl
	mov	byte ptr [si],al

	mov	al,byte ptr [bp+4]
	mov	byte ptr [si+1],al

	push	word ptr [bp+8]
	mov	ax,si
	inc	ax
	inc	ax
	push	ax
	call	_strcpy
	pop	cx
	pop	cx

;	add	word ptr _var+12,di
	add	word ptr _var+76,di	; 76 : '&' *2
	pop	di
	pop	si
	pop	bp
	ret	

_deleteLine:
	push	bp
	mov	bp,sp
	push	si
	push	di
	mov	di,word ptr [bp+4]

	mov	ax,di
	inc	ax
	inc	ax
	push	ax
	call	_strlen
	pop	cx
	mov	si,ax
	add	si,3

;	mov	ax,word ptr _var+12
	mov	ax,word ptr _var+76	; 76 : '&' * 2
	sub	ax,di
	sub	ax,si
	inc	ax
	push	ax

	mov	ax,di
	add	ax,si
	push	ax
	push	di

	call	_memmove
	add	sp,6
;	sub	word ptr _var+12,si
	sub	word ptr _var+76,si	; 76 : '&' *2
	pop	di
	pop	si
	pop	bp
	ret	

_g_decStr:
	push	bp
	mov	bp,sp
	push	si	; buf

;	cx : cnt
	mov	ax,word ptr [bp+6]	; get num
	mov	si,word ptr [bp+4]	; get buf
	xor	cx, cx			; cnt = 0
	mov	bx,10

gdec_1:
	xor	dx,dx		;
	div	bx		; num = num / 10, dx=MOD(num)
	or	dl, '0'		; get '0' to '9' to dl
	mov	[si], dl	; *buf = dl
	inc	si		; buf++
	inc	cx		; cnt++

	or	ax, ax
	jne	gdec_1

	mov	ax, cx		; return cnt

	mov	byte ptr [si], 0	; *buf = NULL
	pop	si
	pop	bp
	ret	

_mk_dStr:
	push	bp
	mov	bp,sp
	sub	sp,12
	push	si
	push	di

	; [bp+4] : d_buf
	; [bp+6] : num( 0 - 32768 )
	; [bp+8] : digit 1 - 5 

	; [bp-12] : s_buf
	; si : s_buf, j
	; di : d_buf
	; cl : sign
	; ch : digit
	; as, bx : num, cnt, i 

	lea	si, [bp-12]	; si = s_buf
	mov	cx, [bp+8]	; cx = digit ( use cl )
	mov	ch, cl		; ch = digit
	mov	bx, [bp+6]	; bx = num
	mov	di, [bp+4]	; di = d_buf

	xor	cl, cl		; cl = sign = 0
	test	bh, 80h		; check MSB
	je	unsignd
	inc	cl		; cl = sign = 1
	neg	bx		; make 2's complement

unsignd:
	mov	[bp-2], cx	; [bp-2] : save sign, digit
	push	bx		; push num
	push	si		; push s_buf
	call	_g_decStr	; return ax : cnt (ah:0 al: cnt)
	pop	si
	pop	bx

	mov	si, ax		; si : j = cnt

	mov	cx, [bp-2]	; ch = digit, cl =sign
	or	cl, cl		; check sign
	jz	nsign
	inc	al		; cnt++

nsign:
	xor	bx, bx		; i=0
_d_loop:
	cmp	ch, al		; digit - cnt
	jle	_d_next
	mov	byte ptr [di+bx], ' '
	inc	bx		; i++
	dec	ch		; digit--
	jmp	_d_loop

_d_next:
	or	cl, cl
	jz	_d_next1
	mov	byte ptr [di+bx], '-'
	inc	bx		; i++

; while(j)
_d_next1:
	or	si, si		; si: j
	je	_d_next2

	; si: j
	mov	al, [bp-12+si-1]	; al <- s_buf[j-1]
	mov	byte ptr [di+bx], al	; d_buf[i] <- al
	inc	bx
	dec	si
	jmp	_d_next1

_d_next2:
	mov	byte ptr [di+bx], 0
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

_g_hexStr:
	push	bp
	mov	bp,sp
	push	si

;	[bp+4] : buf (si)
;	[bp+6] : num
;	[bp+8] : cnt ( 2 or 4 )
;	msk : bx
;	  n : al
;	  i : cx : (use cl)

	mov	si, [bp+4]
	mov	bx, 0f000h
	cmp	word ptr [bp+8], 4	; check cnt == 4
	je	ghs47
	mov	bx, 0f0h

ghs47:
	; i= (cnt-1)*4  (4 or 12)
	mov	ax, [bp+8]	; ax <- cnt ( use al )
	dec	al		; al <- cnt-1
	shl	al, 1		; al : (cnt-1)*2
	shl	al, 1		; al : (cnt-1)*4
	mov	cl, al		; i <- al ( 4 or 12 )

ghs48:
	; n = ((num & msk) >> i);
	mov	ax, [bp+6]	; ax <- num
	and	ax, bx		; ax <- (num & msk)
	shr	ax, cl		; al : n = (msk & num) >> i
	mov	ah, cl		; save i
	mov	cl, 4
	shr	bx, cl		; msk = msk >> 4
	mov	cl, ah		; restore i

	mov	ah, 55
	cmp	al, 9		; check n > 9
	ja	ghs49
	mov	ah, 48
ghs49:
	add	al, ah		; al: get hex char
	mov	[si], al	; save hex char
	inc	si
	sub	cl, 4		; i = i - 4
	jae	ghs48		; check i>=0

	mov	byte ptr [si], 0

	pop	si
	pop	bp
	ret	

_dispLine:
	push	bp
	mov	bp,sp
	sub	sp,8
	push	si

	mov	si, [bp+4]	; si : p

	mov	ah, [si]	; ah : *p << 8
	mov	al, [si+1]	; al : *(p+1)
				; l : ax
	inc	si
	inc	si

	mov	cx,5
	push	cx		; push 5
	push	ax		; push l
	lea	ax,word ptr [bp-8]
	push	ax
	call	_mk_dStr
	pop	ax		; ax : s
	pop	cx		; dummy
	pop	cx		; dummy

	push	ax		; push s
	call	_c_puts
	pop	cx

dli54:
	mov	al, [si]
	or	al, al
	jz	dispLend

	call	_c_putch
	inc	si
	jmp	dli54

dispLend:
	call	_newline

	mov	ax,si
	inc	ax

	pop	si
	mov	sp,bp
	pop	bp
	ret	

_dispList:
	push	bp
	mov	bp,sp
	push	si
	mov	si,word ptr [bp+4]
	jmp	L_52
L_51:
	call	_breakCheck
	push	si
	call	_dispLine
	pop	cx
	mov	si,ax
L_52:
	test	byte ptr [si],128
	je	L_51

	pop	si
	pop	bp
	ret	

_skipBlank:
	mov	bx, [_pc]
L_56:
	mov	al, [bx]
	cmp	al, ' '
	jne	L_57
	inc	bx
	jmp	L_56

L_57:	mov	[_pc], bx	; update pc
	ret	


_skipAlpha:
	mov	bx,word ptr _pc

L_61:
	mov	al,byte ptr [bx]
	cmp	al, 'A'
	jl	no_skip
	cmp	al, 'Z'
	jle	skipA_Z

	cmp	al, 'a'
	jl	no_skip
	cmp	al, 'z'
	jg	no_skip

skipA_Z:
	inc	bx		; pc++
	jmp	L_61

no_skip:
	mov	[_pc], bx	; update pc
	ret	

_exqt:
	call	_skipBlank
	call	_do_cmd
	jmp	_exqt

_topOfLine:
	mov	bx, [_pc]

L_69:
	mov	al, [bx]
	inc	bx

	test	al, 80h
	jz	L_70
	xor	ax,ax
	push	ax
	call	_w_boot
	; no return
L_70:
	mov	ah, al
	mov	al, [bx]
	mov	word ptr _lno, ax
	inc	bx

	cmp	byte ptr [bx], ' '
	je	L_71

	push	bx
	call	_skipLine
	pop	cx
	mov	bx, ax		; get next pc
	jmp	L_69

L_71:
	mov	[_pc], bx	; update pc
	ret	

_breakCheck:
	push	si

	call	_c_kbhit
	or	al,al
	je	L_75

	call	_c_getch
	cbw	
	mov	si,ax

	cmp	si,3
	jne	L_74

	mov	ax, brkmsg_
	push	ax
	call	_w_boot

L_74:
	cmp	si,19
	jne	L_75
	call	_c_getch
L_75:
	pop	si
	ret	

_do_cmd:
	push	bp
	mov	bp,sp
	sub	sp,8
	push	si
	push	di
	call	_breakCheck

	mov	bx, [_pc]
	xor	ax, ax
	mov	al, [bx]		; get c
	mov	si, ax
	inc	bx			; pc++
	mov	[_pc], bx		; update pc
	mov	al, [bx]		; get c1
	xchg	ax, si			; ax=c, si=c1

	mov	cx,8
	push	es
	push	cs
	pop	es
	mov	di,L_106
	cld
	repnz	scasw
	pop	es
	jmp	cs:[di+14]		; ax = c, si = c1

L_106:	dw	0	; NULL
	dw	34	; '"'
	dw	47	; '/'
	dw	63	; '?'
	dw	64	; '@'
	dw	92	; '\'
	dw	93	; ']'
	dw	-1	; dummy

	dw	L_78	; NULL
	dw	L_80	; '"'
	dw	L_81	; '/'
	dw	L_84	; '?'
	dw	L_82	; '@'
	dw	L_85	; '\'
	dw	L_79	; ']'
	dw	L77	; end switch

L_78:	; '\0' NULL
	call	_topOfLine
	mov	ax,1
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_79:	; ']'
	call	_pop
	mov	[_pc], ax
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_80:	; '"'
	call	_do_pr
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_81:	; '/'
	call	_newline
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_82:	; '@'
	cmp	si, 3dh		; si = c1 = '='?
	jne	L113		; go do_do

	; c2= *(pc+1);e=operand();do_until(e,c2); return 0;
	mov	bx,word ptr _pc
	mov	al,byte ptr [bx+1]
	mov	ah,0			; ax = c2
	mov	di, ax			; save c2
	call	_operand		; ax = e
	push	di			; push c2
	push	ax			; push e
	call	_do_until
	pop	cx
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L113:
	call	_do_do
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_84:	; '?'
	push	si		; push c1
	call	_do_prNum
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_85:
	call	_mach_fin
	; no return ( exit GAME86. goto unimon )

L_89: ; '#'
	call	_operand
	push	ax
	call	_do_goto
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_90: ; '!'
	call	_operand
	push	ax
	call	_do_gosub
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_91: ; '$'
	call	_operand	; return al = char
	call	_c_putch
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

; if(c1=='='){
L77:	; ax = c, si = c1
	cmp	si, 3dh		; si = c1 3dh:'='
	jne	L_87		;  -> end switch

; switch(c){
; ax = c, si = c1
	mov	cx, 8
	mov	di, L_109
	push	es
	push	cs
	pop	es
	cld
	repnz	scasw
	pop	es
	jmp	cs:[di+14]

L_109:
	dw	33	; !
	dw	35	; #
	dw	36	; $
	dw	38	; &
	dw	39	; '
	dw	46	; .
	dw	59	; ;
	dw	-1	; dummy

	dw	L_90	; !
	dw	L_89	; #
	dw	L_91	; $
	dw	L_96	; &
	dw	L_94	; '
	dw	L_92	; .
	dw	L_93	; ;
	dw	L_87	; end switch

L_92: ; '.'
	call	_operand
	push	ax
	call	_do_prSpc
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_93: ; ';'
	call	_operand
	push	ax
	call	_do_if
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_94: ; '\''
	call	_operand
	push	ax
	call	_srand
	pop	cx
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_96: ; '&'
	call	_operand
	or	ax, ax
	jne	L_97
	call	_newText
L_97:
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_87:	;  vmode=skipAlpha();
	xchg	ax, si		; si = c
	call	_skipAlpha	; ax : vmode
	mov	di, ax		; di : vmode

; if(vmode==':' || vmode=='(' ){

	cmp	al, ':'
	je	L_100
	cmp	al, '('
	jne	L_99

L_100: ; pc++
	mov	bx, [_pc]
	inc	bx			; pc++

	; off=expr(*pc++);
	mov	al, [bx]
	mov	ah,0
	inc	bx			; pc++
	mov	[_pc], bx		; update pc
	push	ax
	call	_expr
	pop	cx
	mov	[bp-2], ax		; [bp-2] : off

	; if(*(pc-1) !=')') 
	mov	bx, [_pc]
	cmp	byte ptr [bx-1], ')'
	jne	L_101			; error
	; e=operand();
	call	_operand		; ax : e

	; if ( vmode == ':')
	cmp	di, 3ah			; di : vmode = ':'?
	jne	L_102

	; *(((u_char*)VARA(c)+off))=e;
	mov	bx, si			; si : c
;	sub	bx, 20h			; c - ' '
	shl	bx, 1
	mov	bx, [bx+_var]		; VARA(c)
	add	bx, [bp-2]		; +off
	mov	[bx], al		; *(((u_char*)VARA(c)+off))=e : al
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_102: ; if ( vmode == '(' )
	cmp	di,28h			; '('?
	jne	L_97			; return 0

	; *(((u_short*)VARA(c)+off))=e;
	mov	bx, si			; si = c
;	sub	bx, 20h			; c - ' '
	shl	bx, 1
	mov	bx, [bx+_var]		; VARA(c)
	mov	dx, [bp-2]		; off
	shl	dx, 1			; off*2
	add	bx,dx
	mov	[bx], ax		; *(((u_short*)VARA(c)+off))=e : ax
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_101: ; error
	mov	ax, vmiss_
	push	ax
	call	_er_boot
	; no return

L_99: ; e=operand();
	call	_operand		; ax : e

	mov	bx,si			; si : c
;	sub	bx, 20h
	shl	bx,1
	mov	[bx+_var], ax		; VARA(c)=e

	xor	ax, ax
	mov	bx, [_pc]
	mov	al, [bx-1]
	cmp	al, ','
	jne	L197

	mov	al, [bx]
	mov	si, ax			; c= *pc
	inc	bx			; pc++
	mov	[_pc], bx		; update pc
	push	si			; push c
	call	_expr
	pop	cx
	mov	si, ax			; si : e

	push	[_pc]
	call	_push
	pop	cx
	push	si			; push e
	call	_push
	pop	cx
L197:
	xor	ax,ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

_do_until:
	push	bp
	mov	bp,sp
	mov	ax, [bp+4]	; get e
	mov	bx, [bp+6]

;	sub	bx,' '
	shl	bx,1			; bx: var offset
	mov	[bx+_var], ax		; VARA(val)=e;

	mov	bx, [_sp]
	shl	bx,1

	; if(e>stack[sp])
	cmp	ax, [bx+_stack]		; ax = e:([bp+4])
	jle	L_115
	sub	word ptr [_sp], 2
	pop	bp
	ret

L_115: ; repeat
	mov	bx, [_sp]
	dec	bx
	shl	bx,1
	mov	ax, [bx+_stack]
	mov	[_pc], ax
	pop	bp
	ret	

_do_do:
	push	word ptr _pc
	call	_push
	pop	cx

	xor	ax,ax
	push	ax
	call	_push
	pop	cx
	ret	

_do_if:
	push	bp
	mov	bp,sp

	cmp	word ptr [bp+4],0
	jne	L_118

	push	word ptr _pc
	call	_skipLine
	pop	cx
	mov	word ptr _pc,ax

	call	_topOfLine
L_118:
	pop	bp
	ret	

_do_goto:
	push	bp
	mov	bp,sp
	sub	sp,2

	mov	cx, [bp+4]
	cmp	cx, -1
	jne	L_120
	xor	ax, ax
	push	ax
	call	_w_boot
	; no return

L_120:
	lea	ax, [bp-2]
	push	ax
	push	cx		; cx : [bp+4]
	call	_searchLine
	pop	cx		; dummy
	pop	cx		; dummy
	mov	[_pc] ,ax
	call	_topOfLine

	mov	sp,bp
	pop	bp
	ret	

_do_gosub:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si	; p

	lea	ax,word ptr [bp-2]
	push	ax
	push	word ptr [bp+4]
	call	_searchLine	; return ax : p
	pop	cx		; dummy
	pop	cx
	mov	si,ax		; si <- p
	push	[_pc]
	call	_push
	pop	cx		; dummy
	mov	[_pc], si
	call	_topOfLine
	pop	si
	mov	sp,bp
	pop	bp
	ret	

_do_prSpc:
	push	bp
	mov	bp,sp
	push	si
	push	di

	mov	di, [bp+4]
	xor	si, si
L_122:
	cmp	si, di
	jnc	L_123
	mov	ax,32
	call	_c_putch
	inc	si
	jmp	L_122
L_123:
	pop	di
	pop	si
	pop	bp
	ret	

_do_prNum:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si
	push	di

; ax : c1
; si : e

	mov	ax,word ptr [bp+4]	; ax : c1
	cmp	al,40
	jne	dpr137

	inc	word ptr _pc
	push	ax			; push c1
	call	_term
	mov	word ptr [bp-2],ax	; digit

	call	_operand
	mov	si,ax	; si = e

	push	word ptr [bp-2]		; digit
	push	si			; e

	mov	ax, _lky_buf
	push	ax
	call	_mk_dStr
	pop	ax			; form
	pop	cx			; dummy
	pop	cx			; dummy

	push	ax			; form
	call	_c_puts
	pop	cx			; dummy
	pop	ax			; pop c1
	jmp	dpr136			; end

dpr137:
	push	ax		; save c1
	call	_operand
	mov	si,ax		; si = e
	pop	ax		; restore c1
	mov	di, _lky_buf

	cmp	ax,36
	je	dpr140
	cmp	ax,61
	je	dpr141
	cmp	ax,63
	je	dpr139

	mov	ax, uncmd_
	push	ax
	call	_er_boot
	; no return

dpr139:
	mov	ax,4
	push	ax	; 4
	push	si	; e
	push	di	; form
	call	_g_hexStr
	add	sp,6
	jmp	dpr138

dpr140:
	mov	ax,2
	push	ax	; 2
	push	si	; e
	push	di	; form
	call	_g_hexStr
	add	sp,6
	jmp	dpr138

dpr141:
	mov	ax,1
	push	ax	; 1
	push	si	; e
	push	di	; form
	call	_mk_dStr
	add	sp,6

dpr138:
	push	di	; form
	call	_c_puts
	pop	cx

dpr136:
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

_do_pr:
	mov	bx, [_pc]
L_136:
	mov	al, [bx]	; get *pc
	inc	bx
	cmp	al, '"'		; detect string
	jz	L_140
	or	al, al		; NULL?
	jz	L_139
	
	call	_c_putch
	jmp	L_136

L_139:
	dec	bx
L_140:
	mov	[_pc], bx	; update pc
	ret

_pop:
	cmp	word ptr _sp,0
	jge	L_142
	mov	ax, stkunfm_
	push	ax
	call	_er_boot

L_142:
	mov	ax,word ptr _sp
	dec	word ptr _sp
	mov	bx,ax
	shl	bx,1
	mov	ax,word ptr _stack[bx]
	ret	

_push:
	push	bp
	mov	bp,sp

	cmp	word ptr _sp,99
	jl	L_144
	mov	ax, stkovfm_
	push	ax
	call	_er_boot

L_144:
	mov	ax,word ptr [bp+4]
	inc	word ptr _sp
	mov	bx,word ptr _sp
	shl	bx,1
	mov	word ptr _stack[bx],ax
	pop	bp
	ret	

_operand:
	push	si
	mov	si, [_pc]	; si : pc

opr_loop:
	mov	al, [si]	; al : x = *pc
	inc	si
	cmp	al, '='
	je	brk_operand

	test	al, 0dfh	; x & 0xdfh
	jnz	opr_loop

	mov	ax, nooprm_
	push	ax
	call	_errMsg
;	no return

brk_operand:
	mov	al, [si]	; al : x = *pc
	mov	ah, 0
	inc	si
	mov	[_pc], si	; update pc
	push	ax
	call	_expr		; return ax : e
	pop	cx

	pop	si
	ret	

; int expr(c)
_expr:
	push	bp
	mov	bp,sp
	sub	sp,4
	push	si
	push	di

	push	word ptr [bp+4]		; push c
	call	_term
	pop	cx
	mov	si, ax			; si : e

L_154:	; for(;;) {
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	mov	word ptr [bp-4], ax	; [bp-4] : o
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	; ax = o
	mov	cx,12
	mov	di, L_178
	push	es
	push	cs
	pop	es
	cld
	repnz	scasw
	pop	es
	jmp	cs:[di+22]

L_178:
	dw	0	; '\0' : L_156
	dw	32	; ' '  : L_157
	dw	41	; ')'  : L_157
	dw	42	; '*'  : L_171
	dw	43	; '+'  : L_169
	dw	44	; ','  : L_157
	dw	45	; '-'  : L_170
	dw	47	; '/'  : L_172
	dw	60	; '<'  : L_160
	dw	61	; '='  : L_173
	dw	62	; '>'  : L_165
	dw	-1	; goto errMsg
	dw	L_156
	dw	L_157
	dw	L_157
	dw	L_171
	dw	L_169
	dw	L_157
	dw	L_170
	dw	L_172
	dw	L_160
	dw	L_173
	dw	L_165
	dw	L_177	; errMsg

L_156: ; '\0'
	dec	bx		; pc--
	mov	[_pc], bx	; update pc

L_157: ; ' '  ')'  ','
	mov	ax,si		; ax <- si : e
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

L_177:
	mov	al, ' '
	mov	[mm], al
	mov	ax, [bp-4]	; [ bp-4] : o
	mov	ah, '?'
	mov	[mm+1], ax
	mov	[mm+1], ax
	mov	ax, mm
	push	ax
	call	_errMsg
	; no return

L_160:	; '<'
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah, 0		; ax : o1
	mov	[bp-2], ax	; [bp-2] : o1
	inc	bx
	mov	[_pc], bx	; update pc

	; ax = o1
	cmp	al, 61		; '='
	je	L_163

	cmp	al, 62		; '>'
	jne	L_164		; goto default

	; '>'
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term
	pop	cx

	mov	di, ax		; di ; op2

	xor	ax, ax
	cmp	si, di		; cmp e, op2
	je	L_182
	mov	ax,1
L_182:
	mov	si, ax		; e = (e!=op2)
	jmp	L_154		; loop -> for(;;)

L_163: ; '='
	mov	al,byte ptr [bx]
	mov	ah,0
	inc	bx
	mov	[_pc], bx	; update pc

	push	ax
	call	_term
	pop	cx
	mov	di, ax		; di : op2

	mov	ax, 1
	cmp	si, di		; cmp e, op2
	jle	L_184
	xor	ax, ax
L_184:
	mov	si, ax
	jmp	L_154		; loop -> for(;;)

L_164: ; default
	push	word ptr [bp-2]	; push o1
	call	_term		; ax : op2
	pop	cx
	mov	di,ax

	mov	ax, 1
	cmp	si, di		; cmp e, op2
	jl	L_186
	xor	ax, ax
L_186:
	mov	si, ax
	jmp	L_154		; loop -> for(;;)

L_165: ; '>'
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	mov	[bp-2], ax	; [bp-2] : o1
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	; ax = o1
	cmp	al, 61		; '='
	jne	L_168		; goto default

	;'='
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term
	pop	cx
	mov	di, ax		; di ; op2

	mov	ax,1		; true
	cmp	si, di		; cmp e, op2
	jge	L_188
	xor	ax, ax		; false
L_188:
	mov	si,ax
	jmp	L_154		; loop -> for(;;)

L_168: ; default
	push	[bp-2]		; [bp-2] : o1
	call	_term
	pop	cx
	mov	di,ax		; di : op2

	mov	ax,1		; true
	cmp	si, di		; check si > di
	jg	L_190
	xor	ax,ax		; false
L_190:
	mov	si,ax
	jmp	L_154		; loop -> for(;;)


L_169: ; '+'
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term
	pop	cx
	add	si, ax		; e = e + op2
	jmp	L_154

L_170: ; '-'
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term		; ax : op2
	pop	cx

	sub	si, ax
	jmp	L_154

L_171: ; '*'
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term		; ax : op2
	pop	cx

	mul	si		; ax = op2 mul e
	mov	si, ax

	jmp	L_154

L_172: ; '/'
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term		; ax : op2
	pop	cx

	mov	di,ax		; di : op2
	mov	ax, si		; si : e
	cwd	
	idiv	di		; ax = e / op2
;	mov	[_var+10], dx	; % = dx
	mov	[_var+74], dx	; 74 : '%' *2
	mov	si, ax		; si = e / op2
	jmp	L_154

L_173: ; '='
	mov	bx, [_pc]
	mov	al, [bx]	; al = *pc
	mov	ah,0		;
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_term		; ax : op2
	pop	cx
	mov	di,ax

	xor	ax, ax
	cmp	si, di
	jne	L_192
	mov	ax, 1
L_192:
	mov	si,ax
	jmp	L_154

; int term(c)
_term:
	push	bp
	mov	bp,sp
	sub	sp,6
	push	si
	push	di

; [bp+4] : c
; [bp-6] : f
; [bp-4] : vmode
; [bp-2] : ppp

	mov	ax, word ptr [bp+4]	; ax : c
	mov	word ptr [bp-6],0	; f=0
	mov	cx,10
	mov	di, tm227
	push	es
	push	cs
	pop	es
	cld
	repnz	scasw
	pop	es
	jmp	cs:[di+18]

tm227:
	dw	34	; "
	dw	35	; #
	dw	36	; $
	dw	37	; %
	dw	39	; '
	dw	40	; (
	dw	43	; +
	dw	45	; -
	dw	63	; ?
	dw	-1	; dummy
	dw	tm216	; "
	dw	tm212	; #
	dw	tm206	; $
	dw	tm214	; %
	dw	tm213	; '
	dw	tm208	; (
	dw	tm210	; +
	dw	tm211	; -
	dw	tm215	; ?
	dw	tm218	; end switch

tm206: ; $
	lea	ax, [bp-6]
	push	ax		; &f
	call	_getHex		; return ax : e
	pop	bx		; &f
	cmp	word ptr [bx], 0
	jne	tm207		; return e, if (f!=0)
	call	_c_getch	; return al : char
	cbw			; ah = 0

tm207:
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm208: ; (
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax		; *(pc-1)
	call	_expr		; return ax :e
	pop	cx

	mov	bx, [_pc]
	mov	cl, [bx-1]
	cmp	cl, ')'		; *(pc-1) = ')' ?
	jne	tm207e		; return e, if *(pc-1) == ')'
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm207e:
	mov	ax, vmiss1_
	push	ax
	call	_errMsg		; error
	; no return

tm210: ; +
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax
	call	_term		; return ax : e
	pop	cx
	or	ax, ax
	jge	tm207
	neg	ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm211: ; -
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax
	call	_term		; return ax : e
	pop	cx
	neg	ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm212: ; #
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax
	call	_term		; return ax : e
	pop	cx
	neg	ax
	sbb	ax,ax
	inc	ax
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm213: ; '
	call	_rand		; return ax : random number
	push	ax

	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax
	call	_term		; return ax : e
	pop	cx
	mov	bx,ax
	pop	ax		; restore random number
	cwd	
	idiv	bx
	mov	ax,dx		; ax = rand() % term(*pc++)
	inc	ax		; +1
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm214: ; %
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax
	call	_term		; return ax : e
	pop	cx
;	mov	ax, [_var+10]	; get VARA(%)
	mov	ax, [_var+74]	; 74 : '%' *2
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm215: ; ?
	mov	ax, _lky_buf
	push	ax
	call	_c_gets
	pop	cx		; cx : lky_buf

	mov	ax, [_pc]
	mov	[bp-2], ax	; ppp = pc (save pc)
	mov	[_pc], cx	; pc = _lky_buf

	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0		; ax = *p
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	push	ax
	call	_expr		; return ax : e
	pop	cx

	mov	cx, [bp-2]	; (restore pc)
	mov	[_pc], cx	; pc = ppp
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm216: ; "
	mov	bx, [_pc]
	mov	al, [bx]
	mov	ah,0		; e : ax = *p
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	mov	cl, [bx]	; cl = *pc
	inc	bx		; pc++
	mov	[_pc], bx	; update pc
	cmp	cl, 34		; '"'?
	jne	tm216e		; return ax : e
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm216e:
	mov	ax, t_mssm1_
	push	ax
	call	_errMsg		; error
	; no return
	
tm218: ; end switch(c)
	; if(iSnum(c)){
	; ax = c
	cmp	al, '0'
	jl	tm219		; jump, if c < '0'
	cmp	al, '9'
	jg	tm219		; jump, if c > '9'

	dec	word ptr _pc
	lea	ax, [bp-6]
	push	ax		; &f
	call	_getNum		; ax : e
	pop	cx
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm219: ; vmode= skipAlpha();
	call	_skipAlpha	; ax vmode
	mov	si, ax		; save vmode

	cmp	al, ':'
	je	tm221
	cmp	al ,'('
	jne	tm220

tm221:
	; pc++
	mov	bx, [_pc]
	inc	bx		; pc++

	mov	al, [bx]	; al : *pc
	mov	ah,0
	inc	bx		; pc++
	mov	[_pc], bx	; update pc

	push	ax
	call	_expr		; return ax : e
	pop	cx
	mov	di, ax		; di : e

	mov	bx, [_pc]
	mov	al, [bx-1]	; al : *(pc-1)

	cmp	al, ')'
	jne	tm221_err	; jump, if cl <> ')'

	mov	bx, [bp+4]	; bx = c
;	sub	bl, ' '		; bl - ' '
	shl	bx,1		; bx : offset VARA(c)
	mov	bx, [bx+_var]	; bx : VARA(c)

	mov	ax,si		; ax : vmode
	cmp	al,'('
	je	tm225
	cmp	al,':'
	jne	tm220_1

	; return *(((u_char*)VARA(c)+e));
	mov	al, [bx+di]	; di : e, al=[var + e]
	mov	ah,0
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm225:	; return *(((u_short*)VARA(c)+e));
	shl	di, 1
	mov	ax, [bx+di]	; di : e, al=[var + e]
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm220_1:
	mov	ax, bx		; ax : VARA(c)
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm220:
	mov	bx, [bp+4]	; bx = c
;	sub	bl, ' '		; bl - ' '
	shl	bx,1		; bx : offset VARA(c)
	mov	ax, [bx+_var]	; ax : var
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

tm221_err:
	mov	ax, vmiss1_
	push	ax
	call	_errMsg
	; no return

_errMsg:
	push	bp
	mov	bp,sp
	sub	sp,8

	mov	ax, errm_
	push	ax
	call	_c_puts
	pop	cx

	push	word ptr [bp+4]
	call	_c_puts
	pop	cx

	cmp	word ptr _lno,0
	je	errm229

	mov	ax, _in_m
	push	ax
	call	_c_puts
	pop	cx

	mov	ax,1
	push	ax
	push	word ptr _lno
	lea	ax,word ptr [bp-8]
	push	ax
	call	_mk_dStr
	add	sp,6

	lea	ax,word ptr [bp-8]
	push	ax
	call	_c_puts

errm229:
	jmp	_warm_boot

_w_boot:
	push	bp
	mov	bp,sp

	mov	ax, [bp+4]
	cmp	ax ,0
	je	L_223_1
	push	ax
	call	_c_puts
	pop	cx
L_223_1:
	jmp	_warm_boot

_er_boot:
	push	bp
	mov	bp,sp

	mov	ax, errm_
	push	ax
	call	_c_puts
	pop	cx
	mov	ax, [bp+4]
	or	ax, ax
	je	er_223_1
	push	ax
	call	_c_puts
	pop	cx
er_223_1:
	jmp	_warm_boot

_c_toupper:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],122
	jg	L_228
	cmp	byte ptr [bp+4],97
	jl	L_228
	mov	al,byte ptr [bp+4]
	add	al,-32
	jmp	L_227
L_228:
	mov	al,byte ptr [bp+4]
L_227:
	pop	bp
	ret	

_c_isprint:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],32
	jl	L_231
	cmp	byte ptr [bp+4],126
	jg	L_231
	mov	ax,1
	jmp	L_230
L_231:
	xor	ax,ax
L_230:
	pop	bp
	ret	

_c_isspace:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],32
	je	L_235
	cmp	byte ptr [bp+4],13
	jg	L_234
	cmp	byte ptr [bp+4],9
	jl	L_234
L_235:
	mov	ax,1
	jmp	L_233
L_234:
	xor	ax,ax
L_233:
	pop	bp
	ret	

_c_isdigit:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],57
	jg	L_238
	cmp	byte ptr [bp+4],48
	jl	L_238
	mov	ax,1
	jmp	L_237
L_238:
	xor	ax,ax
L_237:
	pop	bp
	ret	

_c_isalpha:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],122
	jg	L_243
	cmp	byte ptr [bp+4],97
	jge	L_242
L_243:
	cmp	byte ptr [bp+4],90
	jg	L_241
	cmp	byte ptr [bp+4],65
	jl	L_241
L_242:
	mov	ax,1
	jmp	L_240
L_241:
	xor	ax,ax
L_240:
	pop	bp
	ret	

_newline:
	mov	ax, newl_m
	push	ax
	call	_c_puts
	pop	cx
	ret

_c_gets:
	push	bp
	mov	bp,sp
	sub	sp,4
	push	si
	mov	si,word ptr [bp+4]
	mov	word ptr [bp-2],0
	jmp	gts270

gts265:
	cmp	byte ptr [bp-3],9
	jne	gts266
	mov	byte ptr [bp-3],32
gts266:
	cmp	byte ptr [bp-3],8
	je	gts268
	cmp	byte ptr [bp-3],127
	jne	gts267
gts268:
	cmp	word ptr [bp-2],0
	jbe	gts267

	dec	word ptr [bp-2]

	mov	ax,8
	call	_c_putch
	mov	ax,32
	call	_c_putch
	mov	ax,8
	call	_c_putch
	jmp	gts270

gts267:
	push	word ptr [bp-3]
	call	_c_isprint
	pop	cx
	or	al,al
	je	gts270
	cmp	word ptr [bp-2],159
	jae	gts270

	mov	al,byte ptr [bp-3]
	mov	bx,word ptr [bp-2]
	mov	byte ptr [bx+si],al
	inc	word ptr [bp-2]
	mov	al,byte ptr [bp-3]
	cbw	
	call	_c_putch
gts270:
	call	_c_getch
	mov	byte ptr [bp-3],al
	cmp	al,13
	jne	gts265

	call	_newline
	mov	bx,word ptr [bp-2]
	mov	byte ptr [bx+si],0
	cmp	word ptr [bp-2],0
	jbe	gts271

gts274:
	dec	word ptr [bp-2]
	mov	bx,word ptr [bp-2]
	push	word ptr [bx+si]
	call	_c_isspace
	pop	cx
	or	al,al
	jne	gts274

	inc	word ptr [bp-2]
	mov	bx,word ptr [bp-2]
	mov	byte ptr [bx+si],0
gts271:
	mov	ax,word ptr [bp-2]
	pop	si
	mov	sp,bp
	pop	bp
	ret	

_memmove:
	push	bp
	mov	bp,sp
	push	si
	push	di

	mov	di, [bp+4]	; di: dest
	mov	si, [bp+6]	; si: src
	mov	cx, [bp+8]	; move counter

	cmp	di, si
	jae	dec_copy

	cld
memm_cp:
	push	ds
	pop	es
	rep	movsb		; while (cx--) {[di++] <- [si++]}
	pop	di		; or while (cx--) {[di--] <- [si--]}
	pop	si
	pop	bp
	ret	

dec_copy:
	add	si, cx
	dec	si
	add	di, cx
	dec	di
	std
	jmp	memm_cp

_strcpy:
	push	bp
	mov	bp,sp
	push	si
	push	di

	xor	ax, ax
	mov	cx, ax		;cl : flg = 0

	mov	si, [bp+6]	; [si] : *pc2
	mov	di, [bp+4]	; [di] : *pc1

stc_loop:
	mov	al, [si]
	or	al, al
	jz	scp_end
	
	cmp	al, 22h		;'"'
	jne	st1
	xor	cl, 1		; flg ~=1
	jmp	stcopy

st1:
	or	cl, cl
	jnz	stcopy		; skip "strings"
	cmp	al, 'a'
	jb	stcopy		; jump if char < 'a'
	cmp	al, 'z'
	ja	stcopy		; jump if char > 'z'
	and	al, 0dfh	; lower to upper
stcopy:
	mov	[di], al	; save char
	inc	di
	inc	si
	jmp	stc_loop

scp_end:
	mov	byte ptr [di], 0	; *p1 = NULL
	pop	di
	pop	si
	pop	bp
	ret

_strlen:
	push	bp
	mov	bp,sp
	push	si

	xor	ax, ax
	mov	si, ax		; num = 0
	mov	bx, [bp+4]

str_lop:
	cmp	al, [bx]
	je	stlen_end
	
	inc	bx
	inc	si
	jmp	str_lop

stlen_end:
	mov	ax, si
	pop	si
	pop	bp
	ret	
	
_getNum:
	push	bp
	mov	bp,sp
	push	si
	push	cx
	
	xor	ax, ax			;  ax : n=0
	mov	cx, ax			;  cx : *f = 0
	mov	si, [_pc]		; [di] : *p

gt_nxtc:
	mov	bl, [si]		; c : bl = *pc
	cmp	bl, '0'			; if c < '0' then return
	jb	L_289
	cmp	bl, '9'			; if c > '9' then return
	ja	L_289

	mov	dx,10
	mul	dx			; ax = n*10

	sub	bl, '0'
	mov	bh, 0
	add	ax, bx
	inc	si
	mov	cl, 1			; *f = 1
	jmp	gt_nxtc

L_289:
	mov	[_pc], si		; pc : next point
	mov	bx, [bp+4]
	mov	[bx], cx		; *f = 0 or 1
	pop	cx
	pop	si
	pop	bp
	ret	

_getHex:
	push	bp
	mov	bp,sp
	push	si
	
	xor	ax, ax
	mov	dx, ax		; ax : n = 0
	mov	cx, ax
	mov	bx, ax		; *f=0

	mov	si, [_pc]	; [si] : *pc

gh_loop:
	mov	dl, [si]	; get char
	cmp	dl, '0'
	jb	hex_end
	cmp	dl, '9'
	jbe	get_dec
	and	dl, 0dfh	; lower to upper
	cmp	dl, 'A'
	jb	hex_end
	cmp	dl, 'F'
	ja	hex_end
	sub	dl, 55		; get digit
calc_no:
	mov	bl, 1		; *f=1
	mov	cl, 4
	shl	ax, cl		; n = n * 16
	add	ax, dx		; n = n + digit
	inc	si		; pc++
	jmp	gh_loop

get_dec:
	sub	dl, '0'		; get digit
	jmp	calc_no

hex_end:
	mov	[_pc], si	; update pc

	mov	cx, bx		;
	mov	bx, [bp+4]	;
	mov	[bx], cx	; *f=cx (0 or 1)

	pop	si
	pop	bp
	ret	

_newText:
;	mov	bx,word ptr _var+12
	mov	bx,word ptr _var+76	; 76 : '&' *2
	cmp	byte ptr [bx],255
	je	L_302
	mov	ax, t_lockm
	push	ax
	call	_er_boot

L_302:
	call	_newText1
	ret	

_newText1:
;	mov	ax,word ptr _var+58
	mov	ax,word ptr _var+122	; 122 : '=' *2
;	mov	word ptr _var+12,ax
	mov	word ptr _var+76,ax	; 76 : '&' *2

;	mov	bx,word ptr _var+12
	mov	bx,word ptr _var+76	; 76 : '&' *2
	mov	byte ptr [bx],255
	ret	

	db	($ & 0FF00H)+100H-$ dup(0FFH)

CODE_END:

	SEGMENT	DATA
	org	0

; CP/M-86 Base Page definition

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

;	org	GM88_WORK

op_msg_:	db	"GAME-86 for CP/M-86 Edition"
newl_m:		db	"\r\n",0
rdymsg_:	db	"\r\n*READY\r\n",0

t_lockm:	db	"1",0
vmiss_:		db	"2",0
uncmd_:		db	"3",0
stkunfm_:	db	"4",0
stkovfm_:	db	"5",0

_in_m:		db	" :in ",0

vmiss1_:	db	" )?",0
nooprm_:
un_oprm_:	db	" ?",0
t_mssm1_:	db	" \"?",0

brkmsg_:	db	"\r\nStop!",0
errm_:		db	"\r\nErr",0

end_data	equ	$
data_size	equ	($+10h) & 0fff0h

; valiable or pointer
SEED:		ds	2
SEEDX:		ds	2
s_val:		ds	2
_pc:		ds	2
_sp:		ds	2
_lno:		ds	2

; buffer
mm:		ds	4
_lin:		ds	160
_lky_buf:	ds	160
_stack:		ds	200-32
_var:		ds	256

		ds	(($+200h) & 0ff00h) - $	;stack area
GM_STACK:

_text_buf	equ	GM_STACK

	end
