	page 0
	cpu 8086

;------------------------------------------
;
;	TOYOSHIKI TinyBASIC V1.0
;	CP/M-86 edition
;
;	2024.01.12 modified by A.honda
;
;-------------------------------------------
;
BDOS_CALL	equ	224

TB_WORK		equ	100h
CODE_OFF	equ	0


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
GD2_GMIN	dw	0400h		; minimum work paragraph length
GD2_GMAX	dw	07ffh		; maximun work paragraph length (32K)

		db	128-$ dup(0);

; ttbasic code body

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
	sub	cx, end_data + 1 ; ger work area
	shr	cx, 1
	xor	ax, ax
	mov	di, end_data

mem_clear:
	mov	[di], ax
	inc	di
	inc	di
	loop	mem_clear

	mov	ax, ds
	MOV	SS,AX
	MOV	SP,TB_STACK

	mov	ax, CODE_OFF
	mov	[s_val], ax	; set initial BASE value for SEED
	call	_main
ret_mon:
	mov	cl, 0
	mov	dl, 0
	int	BDOS_CALL		; Game END : goto CPM86

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
	mov	bx, CODE_OFF
w1:
	mov	[s_val], bx	; update base value
	pop	bx
	pop	ax
	ret
;
; warm start
;
w_start:
	MOV	SP,TB_STACK

	call	update_seed
	mov	ax, ret_mon

	push	ax		; set return address to SP

	push	bp	;Entry sequence
	mov	bp,sp
	JMP	basic_4		; print ">"

;-------------------------------------
;
; Machine depend I/O interface
;
; BDOS cal
; ( INT 224 )
;
;------------------------------------
_c_putch:
	push	bp	;Entry sequence
	mov	bp,sp

	mov	ax,[bp+4]	; Load Arg1 into AX
	
	; put a charactor : CL = 2
	; input : DL : charactor
	push	cx
	mov	cl, 6
	mov	dl, al
	int	BDOS_CALL		; system call
	pop	cx

	pop	bp		; Exit sequence
	ret

_c_getch:
	; get a charactor : CL = 6
	; return AL : charactor
	push	cx
re_call:
	mov	CL, 6
	mov	dl, 0ffh	; input
	int	BDOS_CALL
	or	al, al
	jz	re_call
	call	update_seed
	pop	cx
	ret

_c_kbhit:
	; check key status : CL = 06H
	; OUTPUT : AL : 0     ( key is not exist )
	;             : 0FFH  ( key is exist )
	push	cx
	mov	cl, 6
	mov	dl, 0feh	; key status
	int	BDOS_CALL
	pop	cx
	ret
	

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
	and	ax, 7fffh	; 0 - 7FFFH : 0 ~ 32767

	pop dx
	pop cx
	ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;
;	TOYOSHIKI TinyBASIC V1.0
;	Linux edition
;	(C)2015 Tetsuya Suzuki
;
;	SBCZ8002 edition	2021.06.04
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

_main:
	mov	ax,1234
	push	ax
	call	_srand
	pop	cx
	call	_basic
	xor	ax,ax
	ret	

_newline:
	mov	al,13
	push	ax
	call	_c_putch
	pop	cx
	mov	al,10
	push	ax
	call	_c_putch
	pop	cx
	ret	

;/* Return random number */
_getrnd:
	push	bp
	mov	bp,sp
	call	_rand
	cwd
	idiv	word ptr [bp+4]
	mov	ax,dx
	inc	ax
	pop	bp
	ret	

_sstyle:
	push	bp
	mov	bp,sp
	jmp	sstyle8
sstyle7:
	mov	al, [bp+8]
	mov	ah,0
	mov	bx,ax
	add	bx, [bp+6]
	mov	al, [bx]
	cmp	al, [bp+4]
	jne	sstyle8
	mov	al,1
	jmp	sstyle4
sstyle8:
	mov	al, [bp+8]
	dec	byte ptr [bp+8]
	or	al,al
	jne	sstyle7
	mov	al,0
sstyle4:
	pop	bp
	ret	

_c_toupper:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],122
	jg	c_toupper11
	cmp	byte ptr [bp+4],97
	jl	c_toupper11
	mov	al, [bp+4]
	add	al,-32
	jmp	c_toupper10
c_toupper11:
	mov	al, [bp+4]
c_toupper10:
	pop	bp
	ret	


_c_isprint:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],32
	jl	c_isprint14
	cmp	byte ptr [bp+4],126
	jg	c_isprint14
	mov	ax,1
	jmp	c_isprint13
c_isprint14:
	xor	ax,ax
c_isprint13:
	pop	bp
	ret	

_c_isspace:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],32
	je	c_isspace18
	cmp	byte ptr [bp+4],13
	jg	c_isspace17
	cmp	byte ptr [bp+4],9
	jl	c_isspace17
c_isspace18:
	mov	ax,1
	jmp	c_isspace16
c_isspace17:
	xor	ax,ax
c_isspace16:
	pop	bp
	ret	


_c_isdigit:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],57
	jg	c_isdigit21
	cmp	byte ptr [bp+4],48
	jl	c_isdigit21
	mov	ax,1
	jmp	c_isdigit20
c_isdigit21:
	xor	ax,ax
c_isdigit20:
	pop	bp
	ret	


_c_isalpha:
	push	bp
	mov	bp,sp
	cmp	byte ptr [bp+4],122
	jg	c_isalpha26
	cmp	byte ptr [bp+4],97
	jge	c_isalpha25
c_isalpha26:
	cmp	byte ptr [bp+4],90
	jg	c_isalpha24
	cmp	byte ptr [bp+4],65
	jl	c_isalpha24
c_isalpha25:
	mov	ax,1
	jmp	c_isalpha23
c_isalpha24:
	xor	ax,ax
c_isalpha23:
	pop	bp
	ret	


_c_puts:
	push	bp
	mov	bp,sp
	jmp	c_puts28
c_puts30:
	mov	bx,word ptr [bp+4]
	inc	word ptr [bp+4]
	push	word ptr [bx]
	call	_c_putch
	pop	cx
c_puts28:
	mov	bx,word ptr [bp+4]
	cmp	byte ptr [bx],0
	jne	c_puts30

	pop	bp
	ret	


_c_gets:
	push	bp
	mov	bp,sp
	sub	sp,2
	mov	byte ptr [bp-1],0
	jmp	c_gets32
c_gets34:
	cmp	byte ptr [bp-2],9
	jne	c_gets35
	mov	byte ptr [bp-2],32
c_gets35:
	cmp	byte ptr [bp-2],8
	je	c_gets37
	cmp	byte ptr [bp-2],127
	jne	c_gets36
c_gets37:
	cmp	byte ptr [bp-1],0
	jbe	c_gets36
	dec	byte ptr [bp-1]
	mov	al,8
	push	ax
	call	_c_putch
	pop	cx
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
	mov	al,8
	push	ax
	call	_c_putch
	pop	cx
	jmp	c_gets38
c_gets36:
	push	word ptr [bp-2]
	call	_c_isprint
	pop	cx
	or	al,al
	je	c_gets39
	cmp	byte ptr [bp-1],159
	jae	c_gets39
	mov	al,byte ptr [bp-2]
	mov	dl,byte ptr [bp-1]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _lbuf[bx],al
	inc	byte ptr [bp-1]
	push	word ptr [bp-2]
	call	_c_putch
	pop	cx
c_gets39:
c_gets38:
c_gets32:
	call	_c_getch
	mov	byte ptr [bp-2],al
	cmp	al,13
	jne	c_gets34
c_gets33:
	call	_newline
	mov	al,byte ptr [bp-1]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _lbuf[bx],0
	cmp	byte ptr [bp-1],0
	jbe	c_gets40
	jmp	c_gets41
c_gets43:
c_gets41:
	dec	byte ptr [bp-1]
	mov	al,byte ptr [bp-1]
	mov	ah,0
	mov	bx,ax
	push	word ptr _lbuf[bx]
	call	_c_isspace
	pop	cx
	or	al,al
	jne	c_gets43
c_gets42:
	inc	byte ptr [bp-1]
	mov	al,byte ptr [bp-1]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _lbuf[bx],0
c_gets40:
c_gets31:
	mov	sp,bp
	pop	bp
	ret	


_putnum:
	push	bp
	mov	bp,sp
	sub	sp,2
	cmp	word ptr [bp+4],0
	jge	putnum45
	mov	byte ptr [bp-1],1
	mov	ax,word ptr [bp+4]
	neg	ax
	mov	word ptr [bp+4],ax
	jmp	putnum46
putnum45:
	mov	byte ptr [bp-1],0
putnum46:
;	?debug	L 238
	mov	byte ptr _lbuf+6,0
;	?debug	L 239
	mov	byte ptr [bp-2],6
putnum49:
;	?debug	L 241
	dec	byte ptr [bp-2]
	mov	ax,word ptr [bp+4]
	mov	bx,10
	cwd	
	idiv	bx
	add	dl,48
	mov	al,byte ptr [bp-2]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _lbuf[bx],dl
;	?debug	L 242
	mov	ax,word ptr [bp+4]
	mov	bx,10
	cwd	
	idiv	bx
	mov	word ptr [bp+4],ax
putnum47:
;	?debug	L 243
	cmp	word ptr [bp+4],0
	jg	putnum49
putnum48:
;	?debug	L 245
	cmp	byte ptr [bp-1],0
	je	putnum50
;	?debug	L 246
	dec	byte ptr [bp-2]
	mov	al,byte ptr [bp-2]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _lbuf[bx],45
putnum50:
	jmp	putnum51
putnum53:
;	?debug	L 250
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 251
	dec	word ptr [bp+6]
putnum51:
;	?debug	L 249
	mov	al,byte ptr [bp-2]
	mov	ah,0
	mov	dx,6
	sub	dx,ax
	cmp	dx,word ptr [bp+6]
	jl	putnum53
putnum52:
;	?debug	L 253
	mov	al,byte ptr [bp-2]
	mov	ah,0
	add	ax,_lbuf
	push	ax
	call	_c_puts
	pop	cx
putnum44:
;	?debug	L 254
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 258
_getnum:
	push	bp
	mov	bp,sp
	sub	sp,8
;	?debug	L 264
	mov	byte ptr [bp-2],0
	jmp	putnum55
putnum57:
;	?debug	L 266
	cmp	byte ptr [bp-3],8
	je	putnum59
	cmp	byte ptr [bp-3],127
	jne	putnum58
putnum59:
	cmp	byte ptr [bp-2],0
	jbe	putnum58
;	?debug	L 267
	dec	byte ptr [bp-2]
;	?debug	L 268
	mov	al,8
	push	ax
	call	_c_putch
	pop	cx
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
	mov	al,8
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 269
	jmp	putnum60
putnum58:
;	?debug	L 270
;	?debug	L 271
	cmp	byte ptr [bp-2],0
	jne	putnum63
	cmp	byte ptr [bp-3],43
	je	putnum62
	cmp	byte ptr [bp-3],45
	je	putnum62
putnum63:
	cmp	byte ptr [bp-2],6
	jae	putnum61
	push	word ptr [bp-3]
	call	_c_isdigit
	pop	cx
	or	al,al
	je	putnum61
putnum62:
;	?debug	L 272
	mov	al,byte ptr [bp-3]
	mov	dl,byte ptr [bp-2]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _lbuf[bx],al
	inc	byte ptr [bp-2]
;	?debug	L 273
	push	word ptr [bp-3]
	call	_c_putch
	pop	cx
putnum61:
putnum60:
putnum55:
;	?debug	L 265
	call	_c_getch
	mov	byte ptr [bp-3],al
	cmp	al,13
	je	putnum73
	jmp	putnum57
putnum73:
putnum56:
;	?debug	L 276
	call	_newline
;	?debug	L 277
	mov	al,byte ptr [bp-2]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _lbuf[bx],0
;	?debug	L 279
	mov	al,byte ptr _lbuf
	cbw	
	cmp	ax,43
	je	putnum66
	cmp	ax,45
	je	putnum65
	jmp	putnum67
putnum65:
;	?debug	L 281
	mov	byte ptr [bp-1],1
;	?debug	L 282
	mov	byte ptr [bp-2],1
;	?debug	L 283
	jmp	putnum64
putnum66:
;	?debug	L 285
	mov	byte ptr [bp-1],0
;	?debug	L 286
	mov	byte ptr [bp-2],1
;	?debug	L 287
	jmp	putnum64
putnum67:
;	?debug	L 289
	mov	byte ptr [bp-1],0
;	?debug	L 290
	mov	byte ptr [bp-2],0
;	?debug	L 291
	jmp	putnum64
putnum64:
;	?debug	L 294
	mov	word ptr [bp-8],0
;	?debug	L 295
	mov	word ptr [bp-6],0
	jmp	putnum68
putnum70:
;	?debug	L 297
	mov	ax,word ptr [bp-8]
	mov	dx,10
	mul	dx
	mov	dl,byte ptr [bp-2]
	mov	dh,0
	mov	bx,dx
	push	ax
	mov	al,byte ptr _lbuf[bx]
	cbw	
	pop	dx
	add	dx,ax
	add	dx,-48
	mov	word ptr [bp-6],dx
	inc	byte ptr [bp-2]
;	?debug	L 298
	mov	ax,word ptr [bp-8]
	cmp	ax,word ptr [bp-6]
	jle	putnum71
;	?debug	L 299
	mov	byte ptr _err,2
putnum71:
;	?debug	L 301
	mov	ax,word ptr [bp-6]
	mov	word ptr [bp-8],ax
putnum68:
;	?debug	L 296
	mov	al,byte ptr [bp-2]
	mov	ah,0
	mov	bx,ax
	cmp	byte ptr _lbuf[bx],0
	jne	putnum70
putnum69:
;	?debug	L 303
	cmp	byte ptr [bp-1],0
	je	putnum72
;	?debug	L 304
	mov	ax,word ptr [bp-8]
	neg	ax
	jmp	putnum54
putnum72:
;	?debug	L 305
	mov	ax,word ptr [bp-8]
	jmp	putnum54
putnum54:
;	?debug	L 306
	mov	sp,bp
	pop	bp
	ret	


;	?debug	L 310
_toktoi:
	push	bp
	mov	bp,sp
	sub	sp,10
	push	si
	push	di
;	?debug	L 312
	mov	byte ptr [bp-9],0
;	?debug	L 313
	mov	word ptr [bp-8],0
;	?debug	L 315
	mov	si,_lbuf
	jmp	L_75
L_77:
;	?debug	L 321
	jmp	L_78
L_80:
	inc	si
L_78:
	push	word ptr [si]
	call	_c_isspace
	pop	cx
	or	al,al
	jne	L_80
L_79:
;	?debug	L 324
	mov	byte ptr [bp-10],0
	jmp	L_84
L_83:
;	?debug	L 325
	mov	al,byte ptr [bp-10]
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	ax,word ptr _kwtbl[bx]
	mov	word ptr [bp-8],ax
;	?debug	L 326
	mov	di,si
	jmp	L_85
L_87:
;	?debug	L 330
	inc	word ptr [bp-8]
;	?debug	L 331
	inc	di
L_85:
;	?debug	L 329
	mov	bx,word ptr [bp-8]
	cmp	byte ptr [bx],0
	je	L_88
	push	word ptr [di]
	call	_c_toupper
	pop	cx
	mov	bx,word ptr [bp-8]
	cmp	al,byte ptr [bx]
	je	L_87
L_88:
L_86:
;	?debug	L 334
	mov	bx,word ptr [bp-8]
	cmp	byte ptr [bx],0
	jne	L_89
;	?debug	L 336
	cmp	byte ptr [bp-9],159
	jb	L_90
;	?debug	L 337
	mov	byte ptr _err,4
;	?debug	L 338
	mov	al,0
	jmp	L_74
L_90:
;	?debug	L 342
	mov	al,byte ptr [bp-10]
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	byte ptr [bp-9]
;	?debug	L 343
	mov	si,di
;	?debug	L 344
	jmp	L_81
L_89:
L_82:
	inc	byte ptr [bp-10]
L_84:
	cmp	byte ptr [bp-10],35
	jb	L_83
L_81:
;	?debug	L 349
	cmp	byte ptr [bp-10],8
	je	L_128
	jmp	L_91
L_128:
;	?debug	L 350
	jmp	L_92
L_94:
	inc	si
L_92:
	push	word ptr [si]
	call	_c_isspace
	pop	cx
	or	al,al
	jne	L_94
L_93:
;	?debug	L 351
	mov	di,si
;	?debug	L 352
	mov	byte ptr [bp-10],0
	jmp	L_98
L_97:
L_96:
	inc	byte ptr [bp-10]
L_98:
	mov	bx,di
	inc	di
	cmp	byte ptr [bx],0
	jne	L_97
L_95:
;	?debug	L 353
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	dl,byte ptr [bp-10]
	mov	dh,0
	mov	bx,158
	sub	bx,dx
	cmp	ax,bx
	jl	L_99
;	?debug	L 354
	mov	byte ptr _err,4
;	?debug	L 355
	mov	al,0
	jmp	L_74
L_99:
;	?debug	L 357
	mov	al,byte ptr [bp-10]
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	byte ptr [bp-9]
	jmp	L_100
L_102:
;	?debug	L 359
	mov	al,byte ptr [si]
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	si
	inc	byte ptr [bp-9]
L_100:
;	?debug	L 358
	mov	al,byte ptr [bp-10]
	dec	byte ptr [bp-10]
	or	al,al
	jne	L_102
L_101:
;	?debug	L 361
	jmp	L_76
L_91:
;	?debug	L 364
	mov	bx,word ptr [bp-8]
	cmp	byte ptr [bx],0
	jne	L_103
;	?debug	L 365
	jmp	L_75
L_103:
;	?debug	L 367
	mov	di,si
;	?debug	L 370
	push	word ptr [di]
	call	_c_isdigit
	pop	cx
	or	al,al
	jne	L_129
	jmp	L_104
L_129:
;	?debug	L 371
	mov	word ptr [bp-4],0
;	?debug	L 372
	mov	word ptr [bp-2],0
L_107:
;	?debug	L 374
	mov	ax,word ptr [bp-4]
	mov	dx,10
	mul	dx
	push	ax
	mov	al,byte ptr [di]
	cbw	
	pop	dx
	add	dx,ax
	add	dx,-48
	mov	word ptr [bp-2],dx
	inc	di
;	?debug	L 375
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	jle	L_108
;	?debug	L 376
	mov	byte ptr _err,2
;	?debug	L 377
	mov	al,0
	jmp	L_74
L_108:
;	?debug	L 379
	mov	ax,word ptr [bp-2]
	mov	word ptr [bp-4],ax
L_105:
;	?debug	L 380
	push	word ptr [di]
	call	_c_isdigit
	pop	cx
	or	al,al
	jne	L_107
L_106:
;	?debug	L 382
	cmp	byte ptr [bp-9],157
	jb	L_109
;	?debug	L 383
	mov	byte ptr _err,4
;	?debug	L 384
	mov	al,0
	jmp	L_74
L_109:
;	?debug	L 386
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _ibuf[bx],35
	inc	byte ptr [bp-9]
;	?debug	L 387
	mov	al,byte ptr [bp-4]
	and	al,255
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	byte ptr [bp-9]
;	?debug	L 388
	mov	ax,word ptr [bp-4]
	mov	cl,8
	sar	ax,cl
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	byte ptr [bp-9]
;	?debug	L 389
	mov	si,di
;	?debug	L 390
	jmp	L_110
L_104:
;	?debug	L 394
	cmp	byte ptr [si],34
	je	L_112
	cmp	byte ptr [si],39
	je	L_130
	jmp	L_111
L_130:
L_112:
;	?debug	L 395
	mov	al,byte ptr [si]
	mov	byte ptr [bp-5],al
	inc	si
;	?debug	L 396
	mov	di,si
;	?debug	L 397
	mov	byte ptr [bp-10],0
	jmp	L_116
L_115:
;	?debug	L 398
	inc	di
L_114:
	inc	byte ptr [bp-10]
L_116:
	mov	al,byte ptr [di]
	cmp	al,byte ptr [bp-5]
	je	L_117
	push	word ptr [di]
	call	_c_isprint
	pop	cx
	or	al,al
	jne	L_115
L_117:
L_113:
;	?debug	L 399
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	dl,byte ptr [bp-10]
	mov	dh,0
	mov	bx,159
	sub	bx,dx
	cmp	ax,bx
	jl	L_118
;	?debug	L 400
	mov	byte ptr _err,4
;	?debug	L 401
	mov	al,0
	jmp	L_74
L_118:
;	?debug	L 403
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _ibuf[bx],37
	inc	byte ptr [bp-9]
;	?debug	L 404
	mov	al,byte ptr [bp-10]
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	byte ptr [bp-9]
	jmp	L_119
L_121:
;	?debug	L 406
	mov	al,byte ptr [si]
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	si
	inc	byte ptr [bp-9]
L_119:
;	?debug	L 405
	mov	al,byte ptr [bp-10]
	dec	byte ptr [bp-10]
	or	al,al
	jne	L_121
L_120:
;	?debug	L 408
	mov	al,byte ptr [si]
	cmp	al,byte ptr [bp-5]
	jne	L_122
	inc	si
L_122:
;	?debug	L 409
	jmp	L_123
L_111:
;	?debug	L 413
	push	word ptr [di]
	call	_c_isalpha
	pop	cx
	or	al,al
	je	L_124
;	?debug	L 414
	cmp	byte ptr [bp-9],158
	jb	L_125
;	?debug	L 415
	mov	byte ptr _err,4
;	?debug	L 416
	mov	al,0
	jmp	L_74
L_125:
;	?debug	L 418
	cmp	byte ptr [bp-9],4
	jb	L_126
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	bx,ax
	cmp	byte ptr _ibuf[bx-2],36
	jne	L_126
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	bx,ax
	cmp	byte ptr _ibuf[bx-4],36
	jne	L_126
;	?debug	L 419
	mov	byte ptr _err,20
;	?debug	L 420
	mov	al,0
	jmp	L_74
L_126:
;	?debug	L 422
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _ibuf[bx],36
	inc	byte ptr [bp-9]
;	?debug	L 423
	push	word ptr [di]
	call	_c_toupper
	pop	cx
	add	al,191
	mov	dl,byte ptr [bp-9]
	mov	dh,0
	mov	bx,dx
	mov	byte ptr _ibuf[bx],al
	inc	byte ptr [bp-9]
;	?debug	L 424
	inc	si
;	?debug	L 425
	jmp	L_127
L_124:
;	?debug	L 430
	mov	byte ptr _err,20
;	?debug	L 431
	mov	al,0
	jmp	L_74
L_127:
L_123:
L_110:
L_75:
;	?debug	L 320
	cmp	byte ptr [si],0
	je	L_131
	jmp	L_77
L_131:
L_76:
;	?debug	L 434
	mov	al,byte ptr [bp-9]
	mov	ah,0
	mov	bx,ax
	mov	byte ptr _ibuf[bx],38
	inc	byte ptr [bp-9]
;	?debug	L 435
	mov	al,byte ptr [bp-9]
L_74:
;	?debug	L 436
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 439
_getlineno:
	push	bp
	mov	bp,sp
	push	si
	mov	si,word ptr [bp+4]
;	?debug	L 440
	cmp	byte ptr [si],0
	jne	L_133
;	?debug	L 441
	mov	ax,32767
	jmp	L_132
L_133:
;	?debug	L 442
	mov	al,byte ptr [si+2]
	mov	ah,0
	mov	cl,8
	shl	ax,cl
	mov	dl,byte ptr [si+1]
	mov	dh,0
	or	ax,dx
L_132:
;	?debug	L 443
	pop	si
	pop	bp
	ret	

;	?debug	L 446
_getlp:
	push	bp
	mov	bp,sp
	push	si
;	?debug	L 449
	mov	si,_listbuf
	jmp	L_138
L_137:
;	?debug	L 450
	push	si
	call	_getlineno
	pop	cx
	cmp	ax,word ptr [bp+4]
	jl	L_139
;	?debug	L 451
	jmp	L_135
L_139:
L_136:
	mov	al,byte ptr [si]
	mov	ah,0
	add	si,ax
L_138:
	cmp	byte ptr [si],0
	jne	L_137
L_135:
;	?debug	L 452
	mov	ax,si
L_134:
;	?debug	L 453
	pop	si
	pop	bp
	ret	
;	?debug	L 456
_getsize:
	push	si
;	?debug	L 459
	mov	si,_listbuf
	jmp	L_144
L_143:
L_142:
	mov	al,byte ptr [si]
	mov	ah,0
	add	si,ax
L_144:
	cmp	byte ptr [si],0
	jne	L_143
L_141:
;	?debug	L 460
	mov	ax, [LD0]
	inc	ax

	sub	ax,si
	dec	ax
L_140:
;	?debug	L 461
	pop	si
	ret	

;	?debug	L 465
_inslist:
	push	bp
	mov	bp,sp
	sub	sp,4
	push	si
	push	di
;	?debug	L 470
	call	_getsize
	mov	dl,byte ptr _ibuf
	mov	dh,0
	cmp	ax,dx
	jnc	L_146
;	?debug	L 471
	mov	byte ptr _err,5
	jmp	L_145
L_146:
;	?debug	L 475
	mov	ax,_ibuf
	push	ax
	call	_getlineno
	pop	cx
	push	ax
	call	_getlp
	pop	cx
	mov	word ptr [bp-4],ax
;	?debug	L 477
	push	word ptr [bp-4]
	call	_getlineno
	pop	cx
	push	ax
	mov	ax,_ibuf
	push	ax
	call	_getlineno
	pop	cx
	pop	dx
	cmp	dx,ax
	jne	L_147
;	?debug	L 478
	mov	si,word ptr [bp-4]
;	?debug	L 479
	mov	al,byte ptr [si]
	mov	ah,0
	mov	di,ax
	add	di,si
	jmp	L_148
L_150:
	jmp	L_151
L_153:
;	?debug	L 482
	mov	al,byte ptr [di]
	mov	byte ptr [si],al
	inc	di
	inc	si
L_151:
;	?debug	L 481
	mov	ax,word ptr [bp-2]
	dec	word ptr [bp-2]
	or	ax,ax
	jne	L_153
L_152:
L_148:
;	?debug	L 480
	mov	al,byte ptr [di]
	mov	ah,0
	mov	word ptr [bp-2],ax
	or	ax,ax
	jne	L_150
L_149:
;	?debug	L 485
	mov	byte ptr [si],0
L_147:
;	?debug	L 489
	cmp	byte ptr _ibuf,4
	jne	L_154
	jmp	L_145
L_154:
;	?debug	L 493
	mov	si,word ptr [bp-4]
	jmp	L_158
L_157:
L_156:
	mov	al,byte ptr [si]
	mov	ah,0
	add	si,ax
L_158:
	cmp	byte ptr [si],0
	jne	L_157
L_155:
;	?debug	L 494
	mov	ax,si
	sub	ax,word ptr [bp-4]
	inc	ax
	mov	word ptr [bp-2],ax
;	?debug	L 495
	mov	al,byte ptr _ibuf
	mov	ah,0
	mov	di,ax
	add	di,si
	jmp	L_159
L_161:
;	?debug	L 497
	mov	al,byte ptr [si]
	mov	byte ptr [di],al
	dec	si
	dec	di
L_159:
;	?debug	L 496
	mov	ax,word ptr [bp-2]
	dec	word ptr [bp-2]
	or	ax,ax
	jne	L_161
L_160:
;	?debug	L 500
	mov	al,byte ptr _ibuf
	mov	ah,0
	mov	word ptr [bp-2],ax
;	?debug	L 501
	mov	si,word ptr [bp-4]
;	?debug	L 502
	mov	di,_ibuf
	jmp	L_162
L_164:
;	?debug	L 504
	mov	al,byte ptr [di]
	mov	byte ptr [si],al
	inc	di
	inc	si
L_162:
;	?debug	L 503
	mov	ax,word ptr [bp-2]
	dec	word ptr [bp-2]
	or	ax,ax
	jne	L_164
L_163:
L_145:
;	?debug	L 505
	pop	di
	pop	si
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 508
_putlist:
	push	bp
	mov	bp,sp
	sub	sp,2
	push	si
	mov	si,word ptr [bp+4]
	jmp	L_166
L_168:
;	?debug	L 513
	cmp	byte ptr [si],35
	jae	L_169
;	?debug	L 514
	mov	al,byte ptr [si]
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	push	word ptr _kwtbl[bx]
	call	_c_puts
	pop	cx
;	?debug	L 515
	mov	al,19
	push	ax
	mov	ax,_i_nsa
	push	ax
	push	word ptr [si]
	call	_sstyle
	add	sp,6
	or	al,al
	jne	L_170
;	?debug	L 516
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
L_170:
;	?debug	L 517
	cmp	byte ptr [si],8
	jne	L_171
;	?debug	L 518
	inc	si
;	?debug	L 519
	mov	al,byte ptr [si]
	mov	byte ptr [bp-1],al
	inc	si
	jmp	L_172
L_174:
;	?debug	L 521
	mov	bx,si
	inc	si
	push	word ptr [bx]
	call	_c_putch
	pop	cx
L_172:
;	?debug	L 520
	mov	al,byte ptr [bp-1]
	dec	byte ptr [bp-1]
	or	al,al
	jne	L_174
L_173:
	jmp	L_165
L_171:
;	?debug	L 525
	inc	si
;	?debug	L 526
	jmp	L_175
L_169:
;	?debug	L 530
	cmp	byte ptr [si],35
	jne	L_176
;	?debug	L 531
	inc	si
;	?debug	L 532
	xor	ax,ax
	push	ax
	mov	al,byte ptr [si+1]
	mov	ah,0
	mov	cl,8
	shl	ax,cl
	mov	dl,byte ptr [si]
	mov	dh,0
	or	ax,dx
	push	ax
	call	_putnum
	pop	cx
	pop	cx
;	?debug	L 533
	inc	si
	inc	si
;	?debug	L 534
	mov	al,15
	push	ax
	mov	ax,_i_nsb
	push	ax
	push	word ptr [si]
	call	_sstyle
	add	sp,6
	or	al,al
	jne	L_177
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
L_177:
;	?debug	L 535
	jmp	L_178
L_176:
;	?debug	L 539
	cmp	byte ptr [si],36
	jne	L_179
;	?debug	L 540
	inc	si
;	?debug	L 541
	mov	bx,si
	inc	si
	mov	al,byte ptr [bx]
	add	al,65
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 542
	mov	al,15
	push	ax
	mov	ax,_i_nsb
	push	ax
	push	word ptr [si]
	call	_sstyle
	add	sp,6
	or	al,al
	jne	L_180
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
L_180:
;	?debug	L 543
	jmp	L_181
L_179:
;	?debug	L 547
	cmp	byte ptr [si],37
	jne	L_182
;	?debug	L 550
	mov	byte ptr [bp-2],34
;	?debug	L 551
	inc	si
;	?debug	L 552
	mov	al,byte ptr [si]
	mov	byte ptr [bp-1],al
	jmp	L_186
L_185:
;	?debug	L 553
	mov	al,byte ptr [bp-1]
	mov	ah,0
	mov	bx,ax
	cmp	byte ptr [bx+si],34
	jne	L_187
;	?debug	L 554
	mov	byte ptr [bp-2],39
;	?debug	L 555
	jmp	L_183
L_187:
L_184:
	dec	byte ptr [bp-1]
L_186:
	cmp	byte ptr [bp-1],0
	jne	L_185
L_183:
;	?debug	L 558
	push	word ptr [bp-2]
	call	_c_putch
	pop	cx
;	?debug	L 559
	mov	al,byte ptr [si]
	mov	byte ptr [bp-1],al
	inc	si
	jmp	L_188
L_190:
;	?debug	L 561
	mov	bx,si
	inc	si
	push	word ptr [bx]
	call	_c_putch
	pop	cx
L_188:
;	?debug	L 560
	mov	al,byte ptr [bp-1]
	dec	byte ptr [bp-1]
	or	al,al
	jne	L_190
L_189:
;	?debug	L 563
	push	word ptr [bp-2]
	call	_c_putch
	pop	cx
;	?debug	L 564
	cmp	byte ptr [si],36
	jne	L_191
;	?debug	L 565
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
L_191:
;	?debug	L 566
	jmp	L_192
L_182:
;	?debug	L 570
	mov	byte ptr _err,21
	jmp	L_165
L_192:
L_181:
L_178:
L_175:
L_166:
;	?debug	L 511
	cmp	byte ptr [si],38
	je	L_193
	jmp	L_168
L_193:
L_167:
L_165:
;	?debug	L 574
	pop	si
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 577
_getparam:
	push	bp
	mov	bp,sp
	sub	sp,2
;	?debug	L 580
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],19
	je	L_195
;	?debug	L 581
	mov	byte ptr _err,17
;	?debug	L 582
	xor	ax,ax
	jmp	L_194
L_195:
;	?debug	L 584
	inc	word ptr _cip
;	?debug	L 585
	call	_iexp
	mov	word ptr [bp-2],ax
;	?debug	L 586
	cmp	byte ptr _err,0
	je	L_196
	xor	ax,ax
	jmp	L_194
L_196:
;	?debug	L 588
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],20
	je	L_197
;	?debug	L 589
	mov	byte ptr _err,17
;	?debug	L 590
	xor	ax,ax
	jmp	L_194
L_197:
;	?debug	L 592
	inc	word ptr _cip
;	?debug	L 594
	mov	ax,word ptr [bp-2]
L_194:
;	?debug	L 595
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 598
_ivalue:
	push	bp
	mov	bp,sp
	sub	sp,2
;	?debug	L 601
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	sub	ax,15
	cmp	ax,21
	jbe	L_218
	jmp	L_216
L_218:
	mov	bx,ax
	shl	bx,1
	jmp	word ptr cs:L_217[bx]

L_217:
	dw	L_202
	dw	L_201
	dw	L_216
	dw	L_216
	dw	L_204
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_205
	dw	L_208
	dw	L_210
	dw	L_213
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_216
	dw	L_200
	dw	L_203
L_200:
;	?debug	L 603
	inc	word ptr _cip
;	?debug	L 604
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx+1]
	mov	ah,0
	mov	cl,8
	shl	ax,cl
	mov	bx,word ptr _cip
	mov	dl,byte ptr [bx]
	mov	dh,0
	or	ax,dx
	mov	word ptr [bp-2],ax
;	?debug	L 605
	add	word ptr _cip,2
;	?debug	L 606
	jmp	L_199
L_201:
;	?debug	L 608
	inc	word ptr _cip
;	?debug	L 609
	call	_ivalue
	mov	word ptr [bp-2],ax
;	?debug	L 610
	jmp	L_199
L_202:
;	?debug	L 612
	inc	word ptr _cip
;	?debug	L 613
	call	_ivalue
	xor	dx,dx
	sub	dx,ax
	mov	word ptr [bp-2],dx
;	?debug	L 614
	jmp	L_199
L_203:
;	?debug	L 616
	inc	word ptr _cip
;	?debug	L 617
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	ax,word ptr _var[bx]
	mov	word ptr [bp-2],ax
	inc	word ptr _cip
;	?debug	L 618
	jmp	L_199
L_204:
;	?debug	L 620
	call	_getparam
	mov	word ptr [bp-2],ax
;	?debug	L 621
	jmp	L_199
L_205:
;	?debug	L 623
	inc	word ptr _cip
;	?debug	L 624
	call	_getparam
	mov	word ptr [bp-2],ax
;	?debug	L 625
	cmp	byte ptr _err,0
	je	L_206
;	?debug	L 626
	jmp	L_199
L_206:
;	?debug	L 627
	cmp	word ptr [bp-2],64
	jl	L_207
;	?debug	L 628
	mov	byte ptr _err,3
;	?debug	L 629
	jmp	L_199
L_207:
;	?debug	L 631
	mov	bx,word ptr [bp-2]
	shl	bx,1
	mov	ax,word ptr _arr[bx]
	mov	word ptr [bp-2],ax
;	?debug	L 632
	jmp	L_199
L_208:
;	?debug	L 634
	inc	word ptr _cip
;	?debug	L 635
	call	_getparam
	mov	word ptr [bp-2],ax
;	?debug	L 636
	cmp	byte ptr _err,0
	je	L_209
;	?debug	L 637
	jmp	L_199
L_209:
;	?debug	L 638
	push	word ptr [bp-2]
	call	_getrnd
	pop	cx
	mov	word ptr [bp-2],ax
;	?debug	L 639
	jmp	L_199
L_210:
;	?debug	L 641
	inc	word ptr _cip
;	?debug	L 642
	call	_getparam
	mov	word ptr [bp-2],ax
;	?debug	L 643
	cmp	byte ptr _err,0
	je	L_211
;	?debug	L 644
	jmp	L_199
L_211:
;	?debug	L 645
	cmp	word ptr [bp-2],0
	jge	L_212
;	?debug	L 646
	mov	ax,word ptr [bp-2]
	mov	dx,-1
	mul	dx
	mov	word ptr [bp-2],ax
L_212:
;	?debug	L 647
	jmp	L_199
L_213:
;	?debug	L 649
	inc	word ptr _cip
;	?debug	L 650
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],19
	jne	L_215
	mov	bx,word ptr _cip
	cmp	byte ptr [bx+1],20
	je	L_214
L_215:
;	?debug	L 651
	mov	byte ptr _err,17
;	?debug	L 652
	jmp	L_199
L_214:
;	?debug	L 654
	add	word ptr _cip,2
;	?debug	L 655
	call	_getsize
	mov	word ptr [bp-2],ax
;	?debug	L 656
	jmp	L_199
L_216:
;	?debug	L 659
	mov	byte ptr _err,20
;	?debug	L 660
L_199:
;	?debug	L 662
	mov	ax,word ptr [bp-2]
L_198:
;	?debug	L 663
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 666
_imul:
	push	bp
	mov	bp,sp
	sub	sp,4
;	?debug	L 669
	call	_ivalue
	mov	word ptr [bp-4],ax
;	?debug	L 670
	cmp	byte ptr _err,0
	je	L_220
;	?debug	L 671
	mov	ax,-1
	jmp	L_219
L_220:
	jmp	L_221
L_223:
;	?debug	L 674
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,17
	je	L_225
	cmp	ax,18
	je	L_226
	jmp	L_228
L_225:
;	?debug	L 676
	inc	word ptr _cip
;	?debug	L 677
	call	_ivalue
	mov	word ptr [bp-2],ax
;	?debug	L 678
	mov	ax,word ptr [bp-4]
	mul	word ptr [bp-2]
	mov	word ptr [bp-4],ax
;	?debug	L 679
	jmp	L_224
L_226:
;	?debug	L 681
	inc	word ptr _cip
;	?debug	L 682
	call	_ivalue
	mov	word ptr [bp-2],ax
;	?debug	L 683
	cmp	word ptr [bp-2],0
	jne	L_227
;	?debug	L 684
	mov	byte ptr _err,1
;	?debug	L 685
	mov	ax,-1
	jmp	L_219
L_227:
;	?debug	L 687
	mov	ax,word ptr [bp-4]
	cwd	
	idiv	word ptr [bp-2]
	mov	word ptr [bp-4],ax
;	?debug	L 688
	jmp	L_224
L_228:
;	?debug	L 690
	mov	ax,word ptr [bp-4]
	jmp	L_219
L_224:
L_221:
;	?debug	L 673
	jmp	L_223
L_222:
L_219:
;	?debug	L 692
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 695
_iplus:
	push	bp
	mov	bp,sp
	sub	sp,4
;	?debug	L 698
	call	_imul
	mov	word ptr [bp-4],ax
;	?debug	L 699
	cmp	byte ptr _err,0
	je	L_230
;	?debug	L 700
	mov	ax,-1
	jmp	L_229
L_230:
	jmp	L_231
L_233:
;	?debug	L 703
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,15
	je	L_236
	cmp	ax,16
	je	L_235
	jmp	L_237
L_235:
;	?debug	L 705
	inc	word ptr _cip
;	?debug	L 706
	call	_imul
	mov	word ptr [bp-2],ax
;	?debug	L 707
	mov	ax,word ptr [bp-2]
	add	word ptr [bp-4],ax
;	?debug	L 708
	jmp	L_234
L_236:
;	?debug	L 710
	inc	word ptr _cip
;	?debug	L 711
	call	_imul
	mov	word ptr [bp-2],ax
;	?debug	L 712
	mov	ax,word ptr [bp-2]
	sub	word ptr [bp-4],ax
;	?debug	L 713
	jmp	L_234
L_237:
;	?debug	L 715
	mov	ax,word ptr [bp-4]
	jmp	L_229
L_234:
L_231:
;	?debug	L 702
	jmp	L_233
L_232:
L_229:
;	?debug	L 717
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 720
_iexp:
	push	bp
	mov	bp,sp
	sub	sp,4
;	?debug	L 723
	call	_iplus
	mov	word ptr [bp-4],ax
;	?debug	L 724
	cmp	byte ptr _err,0
	je	L_239
;	?debug	L 725
	mov	ax,-1
	jmp	L_238
L_239:
	jmp	L_240
L_242:
;	?debug	L 729
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	sub	ax,21
	cmp	ax,5
	jbe	L_264
	jmp	L_250
L_264:
	mov	bx,ax
	shl	bx,1
	jmp	word ptr cs:L_251[bx]

L_251:	dw	L_249
	dw	L_245
	dw	L_248
	dw	L_244
	dw	L_247
	dw	L_246

L_244:
;	?debug	L 731
	inc	word ptr _cip
;	?debug	L 732
	call	_iplus
	mov	word ptr [bp-2],ax
;	?debug	L 733
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	jne	L_253
	mov	ax,1
	jmp	L_252
L_253:
	xor	ax,ax
L_252:
	mov	word ptr [bp-4],ax
;	?debug	L 734
	jmp	L_243
L_245:
;	?debug	L 736
	inc	word ptr _cip
;	?debug	L 737
	call	_iplus
	mov	word ptr [bp-2],ax
;	?debug	L 738
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	je	L_255
	mov	ax,1
	jmp	L_254
L_255:
	xor	ax,ax
L_254:
	mov	word ptr [bp-4],ax
;	?debug	L 739
	jmp	L_243
L_246:
;	?debug	L 741
	inc	word ptr _cip
;	?debug	L 742
	call	_iplus
	mov	word ptr [bp-2],ax
;	?debug	L 743
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	jge	L_257
	mov	ax,1
	jmp	L_256
L_257:
	xor	ax,ax
L_256:
	mov	word ptr [bp-4],ax
;	?debug	L 744
	jmp	L_243
L_247:
;	?debug	L 746
	inc	word ptr _cip
;	?debug	L 747
	call	_iplus
	mov	word ptr [bp-2],ax
;	?debug	L 748
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	jg	L_259
	mov	ax,1
	jmp	L_258
L_259:
	xor	ax,ax
L_258:
	mov	word ptr [bp-4],ax
;	?debug	L 749
	jmp	L_243
L_248:
;	?debug	L 751
	inc	word ptr _cip
;	?debug	L 752
	call	_iplus
	mov	word ptr [bp-2],ax
;	?debug	L 753
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	jle	L_261
	mov	ax,1
	jmp	L_260
L_261:
	xor	ax,ax
L_260:
	mov	word ptr [bp-4],ax
;	?debug	L 754
	jmp	L_243
L_249:
;	?debug	L 756
	inc	word ptr _cip
;	?debug	L 757
	call	_iplus
	mov	word ptr [bp-2],ax
;	?debug	L 758
	mov	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-2]
	jl	L_263
	mov	ax,1
	jmp	L_262
L_263:
	xor	ax,ax
L_262:
	mov	word ptr [bp-4],ax
;	?debug	L 759
	jmp	L_243
L_250:
;	?debug	L 761
	mov	ax,word ptr [bp-4]
	jmp	L_238
L_243:
L_240:
;	?debug	L 728
	jmp	L_242
L_241:
L_238:
;	?debug	L 763
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 766
_iprint:
	push	bp
	mov	bp,sp
	sub	sp,6
;	?debug	L 771
	mov	word ptr [bp-4],0
	jmp	L_266
L_268:
;	?debug	L 773
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,22
	je	L_274
	cmp	ax,37
	je	L_270
	jmp	L_276
L_270:
;	?debug	L 775
	inc	word ptr _cip
;	?debug	L 776
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	byte ptr [bp-1],al
	inc	word ptr _cip
	jmp	L_271
L_273:
;	?debug	L 778
	mov	bx,word ptr _cip
	inc	word ptr _cip
	push	word ptr [bx]
	call	_c_putch
	pop	cx
L_271:
;	?debug	L 777
	mov	al,byte ptr [bp-1]
	dec	byte ptr [bp-1]
	or	al,al
	jne	L_273
L_272:
;	?debug	L 779
	jmp	L_269
L_274:
;	?debug	L 781
	inc	word ptr _cip
;	?debug	L 782
	call	_iexp
	mov	word ptr [bp-4],ax
;	?debug	L 783
	cmp	byte ptr _err,0
	je	L_275
	jmp	L_265
L_275:
;	?debug	L 785
	jmp	L_269
L_276:
;	?debug	L 787
	call	_iexp
	mov	word ptr [bp-6],ax
;	?debug	L 788
	cmp	byte ptr _err,0
	je	L_277
	jmp	L_265
L_277:
;	?debug	L 790
	push	word ptr [bp-4]
	push	word ptr [bp-6]
	call	_putnum
	pop	cx
	pop	cx
;	?debug	L 791
L_269:
;	?debug	L 794
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],13
	jne	L_278
;	?debug	L 795
	inc	word ptr _cip
;	?debug	L 796
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],14
	je	L_280
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	jne	L_279
L_280:
	jmp	L_265
L_279:
;	?debug	L 798
	jmp	L_281
L_278:
;	?debug	L 800
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],14
	je	L_282
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	je	L_282
;	?debug	L 801
	mov	byte ptr _err,20
	jmp	L_265
L_282:
L_281:
L_266:
;	?debug	L 772
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],14
	je	L_283
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	je	L_284
	jmp	L_268
L_284:
L_283:
L_267:
;	?debug	L 806
	call	_newline
L_265:
;	?debug	L 807
	mov	sp,bp
	pop	bp
	ret	


;	?debug	L 810
_iinput:
	push	bp
	mov	bp,sp
	sub	sp,6
	jmp	L_286
L_288:
;	?debug	L 817
	mov	byte ptr [bp-1],1
;	?debug	L 819
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],37
	jne	L_289
;	?debug	L 820
	inc	word ptr _cip
;	?debug	L 821
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	byte ptr [bp-2],al
	inc	word ptr _cip
	jmp	L_290
L_292:
;	?debug	L 823
	mov	bx,word ptr _cip
	inc	word ptr _cip
	push	word ptr [bx]
	call	_c_putch
	pop	cx
L_290:
;	?debug	L 822
	mov	al,byte ptr [bp-2]
	dec	byte ptr [bp-2]
	or	al,al
	jne	L_292
L_291:
;	?debug	L 824
	mov	byte ptr [bp-1],0
L_289:
;	?debug	L 827
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,27
	je	L_297
	cmp	ax,36
	je	L_294
	jmp	L_302
L_294:
;	?debug	L 829
	inc	word ptr _cip
;	?debug	L 830
	cmp	byte ptr [bp-1],0
	je	L_295
;	?debug	L 831
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	add	al,65
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 832
	mov	al,58
	push	ax
	call	_c_putch
	pop	cx
L_295:
;	?debug	L 834
	call	_getnum
	mov	word ptr [bp-6],ax
;	?debug	L 835
	cmp	byte ptr _err,0
	je	L_296
	jmp	L_285
L_296:
;	?debug	L 837
	mov	ax,word ptr [bp-6]
	mov	bx,word ptr _cip
	mov	dl,byte ptr [bx]
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _var[bx],ax
	inc	word ptr _cip
;	?debug	L 838
	jmp	L_293
L_297:
;	?debug	L 840
	inc	word ptr _cip
;	?debug	L 841
	call	_getparam
	mov	word ptr [bp-4],ax
;	?debug	L 842
	cmp	byte ptr _err,0
	je	L_298
	jmp	L_285
L_298:
;	?debug	L 844
	cmp	word ptr [bp-4],64
	jl	L_299
;	?debug	L 845
	mov	byte ptr _err,3
	jmp	L_285
L_299:
;	?debug	L 848
	cmp	byte ptr [bp-1],0
	je	L_300
;	?debug	L 849
	mov	ax,attom	;kw_body+521
	push	ax
	call	_c_puts
	pop	cx
;	?debug	L 850
	xor	ax,ax
	push	ax
	push	word ptr [bp-4]
	call	_putnum
	pop	cx
	pop	cx
;	?debug	L 851
	mov	ax,k_colon	;kw_body+524
	push	ax
	call	_c_puts
	pop	cx
L_300:
;	?debug	L 853
	call	_getnum
	mov	word ptr [bp-6],ax
;	?debug	L 854
	cmp	byte ptr _err,0
	je	L_301
	jmp	L_285
L_301:
;	?debug	L 856
	mov	ax,word ptr [bp-6]
	mov	bx,word ptr [bp-4]
	shl	bx,1
	mov	word ptr _arr[bx],ax
;	?debug	L 857
	jmp	L_293
L_302:
;	?debug	L 859
	mov	byte ptr _err,20
	jmp	L_285
L_293:
;	?debug	L 863
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,13
	je	L_304
	cmp	ax,14
	je	L_305
	cmp	ax,38
	je	L_306
	jmp	L_307
L_304:
;	?debug	L 865
	inc	word ptr _cip
;	?debug	L 866
	jmp	L_303
L_305:
L_306:
	jmp	L_285
L_307:
;	?debug	L 871
	mov	byte ptr _err,20
	jmp	L_285
L_303:
L_286:
;	?debug	L 816
	jmp	L_288
L_287:
L_285:
;	?debug	L 875
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 878
_ivar:
	push	bp
	mov	bp,sp
	sub	sp,4
;	?debug	L 882
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	mov	word ptr [bp-2],ax
	inc	word ptr _cip
;	?debug	L 883
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],24
	je	L_309
;	?debug	L 884
	mov	byte ptr _err,18
	jmp	L_308
L_309:
;	?debug	L 887
	inc	word ptr _cip
;	?debug	L 889
	call	_iexp
	mov	word ptr [bp-4],ax
;	?debug	L 890
	cmp	byte ptr _err,0
	je	L_310
	jmp	L_308
L_310:
;	?debug	L 893
	mov	ax,word ptr [bp-4]
	mov	bx,word ptr [bp-2]
	shl	bx,1
	mov	word ptr _var[bx],ax
L_308:
;	?debug	L 894
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 897
_iarray:
	push	bp
	mov	bp,sp
	sub	sp,4
;	?debug	L 901
	call	_getparam
	mov	word ptr [bp-2],ax
;	?debug	L 902
	cmp	byte ptr _err,0
	je	L_312
	jmp	L_311
L_312:
;	?debug	L 905
	cmp	word ptr [bp-2],64
	jl	L_313
;	?debug	L 906
	mov	byte ptr _err,3
	jmp	L_311
L_313:
;	?debug	L 910
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],24
	je	L_314
;	?debug	L 911
	mov	byte ptr _err,18
	jmp	L_311
L_314:
;	?debug	L 914
	inc	word ptr _cip
;	?debug	L 916
	call	_iexp
	mov	word ptr [bp-4],ax
;	?debug	L 917
	cmp	byte ptr _err,0
	je	L_315
	jmp	L_311
L_315:
;	?debug	L 920
	mov	ax,word ptr [bp-4]
	mov	bx,word ptr [bp-2]
	shl	bx,1
	mov	word ptr _arr[bx],ax
L_311:
;	?debug	L 921
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 924
_ilet:
;	?debug	L 925
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,27
	je	L_319
	cmp	ax,36
	je	L_318
	jmp	L_320
L_318:
;	?debug	L 927
	inc	word ptr _cip
;	?debug	L 928
	call	_ivar
;	?debug	L 929
	jmp	L_317
L_319:
;	?debug	L 931
	inc	word ptr _cip
;	?debug	L 932
	call	_iarray
;	?debug	L 933
	jmp	L_317
L_320:
;	?debug	L 935
	mov	byte ptr _err,14
;	?debug	L 936
L_317:
L_316:
;	?debug	L 938
	ret	

;	?debug	L 941
_iexe:
	push	bp
	mov	bp,sp
	sub	sp,10
	push	si
	jmp	L_322
L_324:
;	?debug	L 949
	call	_c_kbhit
	or	al,al
	je	L_325
;	?debug	L 950
	call	_c_getch
	cmp	al,27
	jne	L_326
;	?debug	L 951
	mov	byte ptr _err,22
;	?debug	L 952
	xor	ax,ax
	jmp	L_321
L_326:
L_325:
;	?debug	L 955
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,36
	jbe	L_378
	jmp	L_375
L_378:
	mov	bx,ax
	shl	bx,1
	jmp	word ptr cs:L_377[bx]

L_377:
	dw	L_328
	dw	L_331
	dw	L_335
	dw	L_337
	dw	L_375
	dw	L_375
	dw	L_348
	dw	L_355
	dw	L_358
	dw	L_362
	dw	L_370
	dw	L_369
	dw	L_368
	dw	L_375
	dw	L_371
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_367
	dw	L_375
	dw	L_375
	dw	L_375
	dw	L_372
	dw	L_374
	dw	L_373
	dw	L_375
	dw	L_375
	dw	L_366
L_328:
;	?debug	L 958
	inc	word ptr _cip
;	?debug	L 959
	call	_iexp
	mov	word ptr [bp-10],ax
;	?debug	L 960
	cmp	byte ptr _err,0
	je	L_329
;	?debug	L 961
	jmp	L_327
L_329:
;	?debug	L 962
	push	word ptr [bp-10]
	call	_getlp
	pop	cx
	mov	si,ax
;	?debug	L 963
	push	si
	call	_getlineno
	pop	cx
	cmp	ax,word ptr [bp-10]
	je	L_330
;	?debug	L 964
	mov	byte ptr _err,16
;	?debug	L 965
	jmp	L_327
L_330:
;	?debug	L 968
	mov	word ptr _clp,si
;	?debug	L 969
	mov	ax,word ptr _clp
	add	ax,3
	mov	word ptr _cip,ax
;	?debug	L 970
	jmp	L_327
L_331:
;	?debug	L 973
	inc	word ptr _cip
;	?debug	L 974
	call	_iexp
	mov	word ptr [bp-10],ax
;	?debug	L 975
	cmp	byte ptr _err,0
	je	L_332
;	?debug	L 976
	jmp	L_327
L_332:
;	?debug	L 977
	push	word ptr [bp-10]
	call	_getlp
	pop	cx
	mov	si,ax
;	?debug	L 978
	push	si
	call	_getlineno
	pop	cx
	cmp	ax,word ptr [bp-10]
	je	L_333
;	?debug	L 979
	mov	byte ptr _err,16
;	?debug	L 980
	jmp	L_327
L_333:
;	?debug	L 985
	cmp	byte ptr _gstki,6
	jb	L_334
;	?debug	L 986
	mov	byte ptr _err,6
;	?debug	L 987
	jmp	L_327
L_334:
;	?debug	L 989
	mov	ax,word ptr _clp
	mov	dl,byte ptr _gstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _gstk[bx],ax
	inc	byte ptr _gstki
;	?debug	L 990
	mov	ax,word ptr _cip
	mov	dl,byte ptr _gstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _gstk[bx],ax
	inc	byte ptr _gstki
;	?debug	L 992
	mov	al,byte ptr _lstki
	mov	ah,0
	mov	dl,byte ptr _gstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _gstk[bx],ax
	inc	byte ptr _gstki
;	?debug	L 994
	mov	word ptr _clp,si
;	?debug	L 995
	mov	ax,word ptr _clp
	add	ax,3
	mov	word ptr _cip,ax
;	?debug	L 996
	jmp	L_327
L_335:
;	?debug	L 1000
	cmp	byte ptr _gstki,3
	jae	L_336
;	?debug	L 1001
	mov	byte ptr _err,7
;	?debug	L 1002
	jmp	L_327
L_336:
;	?debug	L 1005
	dec	byte ptr _gstki
	mov	al,byte ptr _gstki
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	al,byte ptr _gstk[bx]
	mov	byte ptr _lstki,al
;	?debug	L 1007
	dec	byte ptr _gstki
	mov	al,byte ptr _gstki
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	ax,word ptr _gstk[bx]
	mov	word ptr _cip,ax
;	?debug	L 1008
	dec	byte ptr _gstki
	mov	al,byte ptr _gstki
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	ax,word ptr _gstk[bx]
	mov	word ptr _clp,ax
;	?debug	L 1009
	jmp	L_327
L_337:
;	?debug	L 1012
	inc	word ptr _cip
;	?debug	L 1014
	mov	bx,word ptr _cip
	inc	word ptr _cip
	cmp	byte ptr [bx],36
	je	L_338
;	?debug	L 1015
	mov	byte ptr _err,12
;	?debug	L 1016
	jmp	L_327
L_338:
;	?debug	L 1019
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	mov	word ptr [bp-8],ax
;	?debug	L 1020
	call	_ivar
;	?debug	L 1021
	cmp	byte ptr _err,0
	je	L_339
;	?debug	L 1022
	jmp	L_327
L_339:
;	?debug	L 1024
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],4
	jne	L_340
;	?debug	L 1025
	inc	word ptr _cip
;	?debug	L 1026
	call	_iexp
	mov	word ptr [bp-6],ax
;	?debug	L 1027
	jmp	L_341
L_340:
;	?debug	L 1029
	mov	byte ptr _err,13
;	?debug	L 1030
	jmp	L_327
L_341:
;	?debug	L 1033
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],5
	jne	L_342
;	?debug	L 1034
	inc	word ptr _cip
;	?debug	L 1035
	call	_iexp
	mov	word ptr [bp-4],ax
;	?debug	L 1036
	jmp	L_343
L_342:
;	?debug	L 1038
	mov	word ptr [bp-4],1
L_343:
;	?debug	L 1041
;	?debug	L 1042
	cmp	word ptr [bp-4],0
	jge	L_346
	mov	ax,-32767
	sub	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-6]
	jg	L_345
L_346:
	cmp	word ptr [bp-4],0
	jle	L_344
	mov	ax,32767
	sub	ax,word ptr [bp-4]
	cmp	ax,word ptr [bp-6]
	jge	L_344
L_345:
;	?debug	L 1043
	mov	byte ptr _err,2
;	?debug	L 1044
	jmp	L_327
L_344:
;	?debug	L 1048
	cmp	byte ptr _lstki,10
	jb	L_347
;	?debug	L 1049
	mov	byte ptr _err,8
;	?debug	L 1050
	jmp	L_327
L_347:
;	?debug	L 1052
	mov	ax,word ptr _clp
	mov	dl,byte ptr _lstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _lstk[bx],ax
	inc	byte ptr _lstki
;	?debug	L 1053
	mov	ax,word ptr _cip
	mov	dl,byte ptr _lstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _lstk[bx],ax
	inc	byte ptr _lstki
;	?debug	L 1055
	mov	ax,word ptr [bp-6]
	mov	dl,byte ptr _lstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _lstk[bx],ax
	inc	byte ptr _lstki
;	?debug	L 1056
	mov	ax,word ptr [bp-4]
	mov	dl,byte ptr _lstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _lstk[bx],ax
	inc	byte ptr _lstki
;	?debug	L 1057
	mov	ax,word ptr [bp-8]
	mov	dl,byte ptr _lstki
	mov	dh,0
	mov	bx,dx
	shl	bx,1
	mov	word ptr _lstk[bx],ax
	inc	byte ptr _lstki
;	?debug	L 1058
	jmp	L_327
L_348:
;	?debug	L 1061
	inc	word ptr _cip
;	?debug	L 1063
	cmp	byte ptr _lstki,5
	jae	L_349
;	?debug	L 1064
	mov	byte ptr _err,9
;	?debug	L 1065
	jmp	L_327
L_349:
;	?debug	L 1068
	mov	al,byte ptr _lstki
	mov	ah,0
	mov	bx,ax
	dec	bx
	shl	bx,1
	mov	ax,word ptr _lstk[bx]
	mov	word ptr [bp-8],ax
;	?debug	L 1069
	mov	bx,word ptr _cip
	inc	word ptr _cip
	cmp	byte ptr [bx],36
	je	L_350
;	?debug	L 1070
	mov	byte ptr _err,10
;	?debug	L 1071
	jmp	L_327
L_350:
;	?debug	L 1073
	mov	bx,word ptr _cip
	inc	word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,word ptr [bp-8]
	je	L_351
;	?debug	L 1074
	mov	byte ptr _err,11
;	?debug	L 1075
	jmp	L_327
L_351:
;	?debug	L 1078
	mov	al,byte ptr _lstki
	mov	ah,0
	mov	bx,ax
	add	bx,-2
	shl	bx,1
	mov	ax,word ptr _lstk[bx]
	mov	word ptr [bp-4],ax
;	?debug	L 1079
	mov	ax,word ptr [bp-4]
	mov	bx,word ptr [bp-8]
	shl	bx,1
	add	word ptr _var[bx],ax
;	?debug	L 1080
	mov	al,byte ptr _lstki
	mov	ah,0
	mov	bx,ax
	add	bx,-3
	shl	bx,1
	mov	ax,word ptr _lstk[bx]
	mov	word ptr [bp-6],ax
;	?debug	L 1083
;	?debug	L 1084
	cmp	word ptr [bp-4],0
	jge	L_354
	mov	bx,word ptr [bp-8]
	shl	bx,1
	mov	ax,word ptr _var[bx]
	cmp	ax,word ptr [bp-6]
	jl	L_353
L_354:
	cmp	word ptr [bp-4],0
	jle	L_352
	mov	bx,word ptr [bp-8]
	shl	bx,1
	mov	ax,word ptr _var[bx]
	cmp	ax,word ptr [bp-6]
	jle	L_352
L_353:
;	?debug	L 1085
	sub	byte ptr _lstki,5
;	?debug	L 1086
	jmp	L_327
L_352:
;	?debug	L 1090
	mov	al,byte ptr _lstki
	mov	ah,0
	mov	bx,ax
	add	bx,-4
	shl	bx,1
	mov	ax,word ptr _lstk[bx]
	mov	word ptr _cip,ax
;	?debug	L 1091
	mov	al,byte ptr _lstki
	mov	ah,0
	mov	bx,ax
	add	bx,-5
	shl	bx,1
	mov	ax,word ptr _lstk[bx]
	mov	word ptr _clp,ax
;	?debug	L 1092
	jmp	L_327
L_355:
;	?debug	L 1095
	inc	word ptr _cip
;	?debug	L 1096
	call	_iexp
	mov	word ptr [bp-2],ax
;	?debug	L 1097
	cmp	byte ptr _err,0
	je	L_356
;	?debug	L 1098
	mov	byte ptr _err,15
;	?debug	L 1099
	jmp	L_327
L_356:
;	?debug	L 1101
	cmp	word ptr [bp-2],0
	je	L_357
;	?debug	L 1102
	jmp	L_327
L_357:
L_358:
	jmp	L_359
L_361:
;	?debug	L 1109
	inc	word ptr _cip
L_359:
;	?debug	L 1108
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	jne	L_361
L_360:
;	?debug	L 1110
	jmp	L_327
L_362:
	jmp	L_363
L_365:
;	?debug	L 1114
	mov	bx,word ptr _clp
	mov	al,byte ptr [bx]
	mov	ah,0
	add	word ptr _clp,ax
L_363:
;	?debug	L 1113
	mov	bx,word ptr _clp
	cmp	byte ptr [bx],0
	jne	L_365
L_364:
;	?debug	L 1115
	mov	ax,word ptr _clp
	jmp	L_321
L_366:
;	?debug	L 1118
	inc	word ptr _cip
;	?debug	L 1119
	call	_ivar
;	?debug	L 1120
	jmp	L_327
L_367:
;	?debug	L 1122
	inc	word ptr _cip
;	?debug	L 1123
	call	_iarray
;	?debug	L 1124
	jmp	L_327
L_368:
;	?debug	L 1126
	inc	word ptr _cip
;	?debug	L 1127
	call	_ilet
;	?debug	L 1128
	jmp	L_327
L_369:
;	?debug	L 1130
	inc	word ptr _cip
;	?debug	L 1131
	call	_iprint
;	?debug	L 1132
	jmp	L_327
L_370:
;	?debug	L 1134
	inc	word ptr _cip
;	?debug	L 1135
	call	_iinput
;	?debug	L 1136
	jmp	L_327
L_371:
;	?debug	L 1139
	inc	word ptr _cip
;	?debug	L 1140
	jmp	L_327
L_372:
L_373:
L_374:
;	?debug	L 1145
	mov	byte ptr _err,19
;	?debug	L 1146
	jmp	L_327
L_375:
;	?debug	L 1149
	mov	byte ptr _err,20
;	?debug	L 1150
L_327:
;	?debug	L 1153
	cmp	byte ptr _err,0
	je	L_376
;	?debug	L 1154
	xor	ax,ax
	jmp	L_321
L_376:
L_322:
;	?debug	L 947
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	je	L_379
	jmp	L_324
L_379:
L_323:
;	?debug	L 1156
	mov	bx,word ptr _clp
	mov	al,byte ptr [bx]
	mov	ah,0
	add	ax,word ptr _clp
L_321:
;	?debug	L 1157
	pop	si
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 1160
_irun:
	push	si
;	?debug	L 1163
	mov	byte ptr _gstki,0
;	?debug	L 1164
	mov	byte ptr _lstki,0
;	?debug	L 1165
	mov	word ptr _clp,_listbuf
	jmp	L_381
L_383:
;	?debug	L 1168
	mov	ax,word ptr _clp
	add	ax,3
	mov	word ptr _cip,ax
;	?debug	L 1169
	call	_iexe
	mov	si,ax
;	?debug	L 1170
	cmp	byte ptr _err,0
	je	L_384
	jmp	L_380
L_384:
;	?debug	L 1172
	mov	word ptr _clp,si
L_381:
;	?debug	L 1167
	mov	bx,word ptr _clp
	cmp	byte ptr [bx],0
	jne	L_383
L_382:
L_380:
;	?debug	L 1174
	pop	si
	ret	

;	?debug	L 1177
_ilist:
	push	bp
	mov	bp,sp
	sub	sp,2
;	?debug	L 1180
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],35
	jne	L_396
	push	word ptr _cip
	call	_getlineno
	pop	cx
	jmp	L_395
L_396:
	xor	ax,ax
L_395:
	mov	word ptr [bp-2],ax
;	?debug	L 1182
	mov	word ptr _clp,_listbuf
	jmp	L_389
L_388:
L_387:
;	?debug	L 1184
	mov	bx,word ptr _clp
	mov	al,byte ptr [bx]
	mov	ah,0
	add	word ptr _clp,ax
L_389:
;	?debug	L 1183
	mov	bx,word ptr _clp
	cmp	byte ptr [bx],0
	je	L_390
	push	word ptr _clp
	call	_getlineno
	pop	cx
	cmp	ax,word ptr [bp-2]
	jl	L_388
L_390:
L_386:
	jmp	L_391
L_393:
;	?debug	L 1187
	xor	ax,ax
	push	ax
	push	word ptr _clp
	call	_getlineno
	pop	cx
	push	ax
	call	_putnum
	pop	cx
	pop	cx
;	?debug	L 1188
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 1189
	mov	ax,word ptr _clp
	add	ax,3
	push	ax
	call	_putlist
	pop	cx
;	?debug	L 1190
	cmp	byte ptr _err,0
	je	L_394
;	?debug	L 1191
	jmp	L_392
L_394:
;	?debug	L 1192
	call	_newline
;	?debug	L 1193
	mov	bx,word ptr _clp
	mov	al,byte ptr [bx]
	mov	ah,0
	add	word ptr _clp,ax
L_391:
;	?debug	L 1186
	mov	bx,word ptr _clp
	cmp	byte ptr [bx],0
	jne	L_393
L_392:
L_385:
;	?debug	L 1195
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 1198
_inew:
	push	bp
	mov	bp,sp
	sub	sp,2
;	?debug	L 1201
	mov	byte ptr [bp-1],0
	jmp	L_401
L_400:
;	?debug	L 1202
	mov	al,byte ptr [bp-1]
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	word ptr _var[bx],0
L_399:
	inc	byte ptr [bp-1]
L_401:
	cmp	byte ptr [bp-1],26
	jb	L_400
L_398:
;	?debug	L 1203
	mov	byte ptr [bp-1],0
	jmp	L_405
L_404:
;	?debug	L 1204
	mov	al,byte ptr [bp-1]
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	mov	word ptr _arr[bx],0
L_403:
	inc	byte ptr [bp-1]
L_405:
	cmp	byte ptr [bp-1],64
	jb	L_404
L_402:
;	?debug	L 1205
	mov	byte ptr _gstki,0
;	?debug	L 1206
	mov	byte ptr _lstki,0
;	?debug	L 1207
	mov	byte ptr _listbuf,0
;	?debug	L 1208
	mov	word ptr _clp,_listbuf
L_397:
;	?debug	L 1209
	mov	sp,bp
	pop	bp
	ret	

;	?debug	L 1212
_icom:
;	?debug	L 1213
	mov	word ptr _cip,_ibuf
;	?debug	L 1214
	mov	bx,word ptr _cip
	mov	al,byte ptr [bx]
	mov	ah,0
	cmp	ax,31
	je	L_411
	cmp	ax,32
	je	L_415
	cmp	ax,33
	je	L_408
	jmp	L_416
L_408:
;	?debug	L 1216
	inc	word ptr _cip
;	?debug	L 1217
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	jne	L_409
;	?debug	L 1218
	call	_inew
	jmp	L_410
L_409:
;	?debug	L 1220
	mov	byte ptr _err,20
L_410:
;	?debug	L 1221
	jmp	L_407
L_411:
;	?debug	L 1223
	inc	word ptr _cip
;	?debug	L 1224
	mov	bx,word ptr _cip
	cmp	byte ptr [bx],38
	je	L_413
	mov	bx,word ptr _cip
	cmp	byte ptr [bx+3],38
	jne	L_412
L_413:
;	?debug	L 1225
	call	_ilist
	jmp	L_414
L_412:
;	?debug	L 1227
	mov	byte ptr _err,20
L_414:
;	?debug	L 1228
	jmp	L_407
L_415:
;	?debug	L 1230
	inc	word ptr _cip
;	?debug	L 1231
	call	_irun
;	?debug	L 1232
	jmp	L_407
L_416:
;	?debug	L 1234
	call	_iexe
;	?debug	L 1235
L_407:
L_406:
;	?debug	L 1237
	ret	


;	?debug	L 1240
_error:
;	?debug	L 1241
	cmp	byte ptr _err,0
	je	L_418
;	?debug	L 1242
	cmp	word ptr _cip,_listbuf
	jb	L_419
	mov	ax, [LD0]
	cmp	ax, [_cip]
	jae	L_419
	mov	bx,word ptr _clp
	cmp	byte ptr [bx],0
	je	L_419
;	?debug	L 1244
	call	_newline
;	?debug	L 1245
	mov	ax,line_m	;kw_body+527
	push	ax
	call	_c_puts
	pop	cx
;	?debug	L 1246
	xor	ax,ax
	push	ax
	push	word ptr _clp
	call	_getlineno
	pop	cx
	push	ax
	call	_putnum
	pop	cx
	pop	cx
;	?debug	L 1247
	mov	al,32
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 1248
	mov	ax,word ptr _clp
	add	ax,3
	push	ax
	call	_putlist
	pop	cx
;	?debug	L 1249
	jmp	L_420
L_419:
;	?debug	L 1252
	call	_newline
;	?debug	L 1253
	mov	ax,you_m	;kw_body+533
	push	ax
	call	_c_puts
	pop	cx
;	?debug	L 1254
	mov	ax,_lbuf
	push	ax
	call	_c_puts
	pop	cx
L_420:
L_418:
;	?debug	L 1258
	call	_newline
;	?debug	L 1259
	mov	al,byte ptr _err
	mov	ah,0
	mov	bx,ax
	shl	bx,1
	push	word ptr _errmsg[bx]
	call	_c_puts
	pop	cx
;	?debug	L 1260
	call	_newline
;	?debug	L 1261
	mov	byte ptr _err,0
L_417:
;	?debug	L 1262
	ret	

;	?debug	L 1268
_basic:
	push	bp
	mov	bp,sp
	sub	sp,2
;	?debug	L 1271
	call	_inew
;	?debug	L 1272
	mov	ax,o_msg
	push	ax
	call	_c_puts
	pop	cx
	call	_newline
;	?debug	L 1273
	mov	ax,edm_1
	push	ax
	call	_c_puts
	pop	cx
;	?debug	L 1274
	mov	ax,edm_2
	push	ax
	call	_c_puts
	pop	cx
	call	_newline
;	?debug	L 1275
	call	_error
	jmp	basic_2
basic_4:
;	?debug	L 1279
	mov	al,62
	push	ax
	call	_c_putch
	pop	cx
;	?debug	L 1280
	call	_c_gets
;	?debug	L 1281
	call	_toktoi
	mov	byte ptr [bp-1],al
;	?debug	L 1282
	cmp	byte ptr _err,0
	je	basic_5
;	?debug	L 1283
	call	_error
;	?debug	L 1284
	jmp	basic_2
basic_5:
;	?debug	L 1287
	cmp	byte ptr _ibuf,34
	jne	basic_6
	jmp	basic_1
basic_6:
;	?debug	L 1291
	cmp	byte ptr _ibuf,35
	jne	basic_7
;	?debug	L 1292
	mov	al,byte ptr [bp-1]
	mov	byte ptr _ibuf,al
;	?debug	L 1293
	call	_inslist
;	?debug	L 1294
	cmp	byte ptr _err,0
	je	basic_8
;	?debug	L 1295
	call	_error
basic_8:
;	?debug	L 1296
	jmp	basic_2
basic_7:
;	?debug	L 1299
	call	_icom
;	?debug	L 1300
	call	_error
basic_2:
;	?debug	L 1278
	jmp	basic_4
basic_1:
;	?debug	L 1302
	mov	sp,bp
	pop	bp
	ret	

CSTART:	jmp	start_tb
WSTART:	jmp	w_start

	db	($ & 0FF00H)+100H-$ dup(0FFH)

CODE_END:

	SEGMENT	DATA
	org	0

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

;	org	TB_WORK

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;enum{
;	I_GOTO, I_GOSUB, I_RETURN,
;	I_FOR, I_TO, I_STEP, I_NEXT,
;	I_IF, I_REM, I_STOP,
;	I_INPUT, I_PRINT, I_LET,
;	I_COMMA, I_SEMI,
;	I_MINUS, I_PLUS, I_MUL, I_DIV, I_OPEN, I_CLOSE,
;	I_GTE, I_SHARP, I_GT, I_EQ, I_LTE, I_LT,
;	I_ARRAY, I_RND, I_ABS, I_SIZE,
;	I_LIST, I_RUN, I_NEW, I_SYSTEM,
;	I_NUM, I_VAR, I_STR,
;	I_EOL
;};
;
;/* Keyword count */
;#define SIZE_KWTBL (sizeof(kwtbl) / sizeof(const char*))
;
;/* List formatting condition */
;/* no space after */
;const unsigned char i_nsa[] = {
;	I_RETURN, I_STOP, I_COMMA,
;	I_MINUS, I_PLUS, I_MUL, I_DIV, I_OPEN, I_CLOSE,
;	I_GTE, I_SHARP, I_GT, I_EQ, I_LTE, I_LT,
;	I_ARRAY, I_RND, I_ABS, I_SIZE
;};
;
;/*no space before (after numeric or variable only) */
;const unsigned char i_nsb[] = {
;	I_MINUS, I_PLUS, I_MUL, I_DIV, I_OPEN, I_CLOSE,
;	I_GTE, I_SHARP, I_GT, I_EQ, I_LTE, I_LT,
;	I_COMMA, I_SEMI, I_EOL
;};
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
_i_nsa:
	db	2
	db	9
	db	13
	db	15
	db	16
	db	17
	db	18
	db	19
	db	20
	db	21
	db	22
	db	23
	db	24
	db	25
	db	26
	db	27
	db	28
	db	29
	db	30
_i_nsb:
	db	15
	db	16
	db	17
	db	18
	db	19
	db	20
	db	21
	db	22
	db	23
	db	24
	db	25
	db	26
	db	13
	db	14
	db	38

_kwtbl:
	dw	kwtbl0		; kw_body
	dw	kwtbl1		; kw_body+5
	dw	kwtbl2		; kw_body+11
	dw	kwtbl3		; kw_body+18
	dw	kwtbl4		; kw_body+22
	dw	kwtbl5		; kw_body+25
	dw	kwtbl6		; kw_body+30
	dw	kwtbl7		; kw_body+35
	dw	kwtbl8		; kw_body+38
	dw	kwtbl9		; kw_body+42
	dw	kwtbl10		; kw_body+47
	dw	kwtbl11		; kw_body+53
	dw	kwtbl12		; kw_body+59
	dw	kwtbl13		; kw_body+63
	dw	kwtbl14		; kw_body+65
	dw	kwtbl15		; kw_body+67
	dw	kwtbl16		; kw_body+69
	dw	kwtbl17		; kw_body+71
	dw	kwtbl18		; kw_body+73
	dw	kwtbl19		; kw_body+75
	dw	kwtbl20		; kw_body+77
	dw	kwtbl21		; kw_body+79
	dw	kwtbl22		; kw_body+82
	dw	kwtbl23		; kw_body+84
	dw	kwtbl24		; kw_body+86
	dw	kwtbl25		; kw_body+88
	dw	kwtbl26		; kw_body+91
	dw	kwtbl27		; kw_body+93
	dw	kwtbl28		; kw_body+95
	dw	kwtbl29		; kw_body+99
	dw	kwtbl30		; kw_body+103
	dw	kwtbl31		; kw_body+108
	dw	kwtbl32		; kw_body+113
	dw	kwtbl33		; kw_body+117
	dw	kwtbl34		; kw_body+121

;kw_body:
kwtbl0:		db	"GOTO",0
kwtbl1:		db	"GOSUB",0
kwtbl2:		db	"RETURN",0
kwtbl3:		db	"FOR",0
kwtbl4:		db	"TO",0
kwtbl5:		db	"STEP",0
kwtbl6:		db	"NEXT",0
kwtbl7:		db	"IF",0
kwtbl8:		db	"REM",0
kwtbl9:		db	"STOP",0
kwtbl10:	db	"INPUT",0
kwtbl11:	db	"PRINT",0
kwtbl12:	db	"LET",0
kwtbl13:	db	",",0
kwtbl14:	db	";",0
kwtbl15:	db	"-",0
kwtbl16:	db	"+",0
kwtbl17:	db	"*",0
kwtbl18:	db	"/",0
kwtbl19:	db	"(",0
kwtbl20:	db	")",0
kwtbl21:	db	">=",0
kwtbl22:	db	"#",0
kwtbl23:	db	">",0
kwtbl24:	db	"=",0
kwtbl25:	db	"<=",0
kwtbl26:	db	"<",0
kwtbl27:	db	"@",0
kwtbl28:	db	"RND",0
kwtbl29:	db	"ABS",0
kwtbl30:	db	"SIZE",0
kwtbl31:	db	"LIST",0
kwtbl32:	db	"RUN",0
kwtbl33:	db	"NEW",0
kwtbl34:	db	"SYSTEM",0

_errmsg:
	dw	er_msg0		; kw_body+128
	dw	er_msg1		; kw_body+131
	dw	er_msg2		; kw_body+148
	dw	er_msg3		; kw_body+157
	dw	er_msg4		; kw_body+180
	dw	er_msg5		; kw_body+198
	dw	er_msg6		; kw_body+208
	dw	er_msg7		; kw_body+230
	dw	er_msg8		; kw_body+253
	dw	er_msg9		; kw_body+273
	dw	er_msg10	; kw_body+290
	dw	er_msg11	; kw_body+311
	dw	er_msg12	; kw_body+329
	dw	er_msg13	; kw_body+350
	dw	er_msg14	; kw_body+365
	dw	er_msg15	; kw_body+386
	dw	er_msg16	; kw_body+407
	dw	er_msg17	; kw_body+429
	dw	er_msg18	; kw_body+449
	dw	er_msg19	; kw_body+462
	dw	er_msg20	; kw_body+478
	dw	er_msg21	; kw_body+491
	dw	er_msg22	; kw_body+506

;errmsg
er_msg0:	db	"OK",0
er_msg1:	db	"Devision by zero",0
er_msg2:	db	"Overflow",0
er_msg3:	db	"Subscript out of range",0
er_msg4:	db	"Icode buffer full",0
er_msg5:	db	"List full",0
er_msg6:	db	"GOSUB too many nested",0
er_msg7:	db	"RETURN stack underflow",0
er_msg8:	db	"FOR too many nested",0
er_msg9:	db	"NEXT without FOR",0
er_msg10:	db	"NEXT without counter",0
er_msg11:	db	"NEXT mismatch FOR",0
er_msg12:	db	"FOR without variable",0
er_msg13:	db	"FOR without TO",0
er_msg14:	db	"LET without variable",0
er_msg15:	db	"IF without condition",0
er_msg16:	db	"Undefined line number",0
er_msg17:	db	"\'(\' or \')\' expected",0
er_msg18:	db	"\'=\' expected",0
er_msg19:	db	"Illegal command",0
er_msg20:	db	"Syntax error",0
er_msg21:	db	"Internal error",0
er_msg22:	db	"Abort by [ESC]",0

attom:	db	"@(",0
k_colon:
	db	"):",0
line_m:	db	"LINE:",0	

you_m:	db	"YOU TYPE: ",0

o_msg:	db	"TOYOSHIKI TINY BASIC"
	db	0
edm_1:	db	"CP/M-86"
	db	0
edm_2:	db	" EDITION"
	db	0

end_data	equ	$
data_size	equ	($+10h) & 0fff0h

SEED:		ds	2
SEEDX:		ds	2
s_val:		ds	2
_ibuf:		ds	160
_clp:		ds	2
_cip:		ds	2
_lbuf:		ds	160
_arr:		ds	128
_err:		ds	1
_gstk:		ds	18
_lstk:		ds	30
_var:		ds	52
_gstki:		ds	1
_lstki:		ds	1

		ds	(($+100h) & 0ff00h) - $	;stack area
TB_STACK:
_listbuf	equ	TB_STACK
	end
