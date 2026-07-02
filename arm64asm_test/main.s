	.build_version macos, 26, 0	sdk_version 26, 5
	.section	__TEXT,__literal16,16byte_literals
	.p2align	4, 0x0                          ; -- Begin function main
lCPI0_0:
	.long	0                               ; 0x0
	.long	1                               ; 0x1
	.long	2                               ; 0x2
	.long	3                               ; 0x3
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32
	sub	sp, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
Lloh0:
	adrp	x8, ___stack_chk_guard@GOTPAGE
Lloh1:
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
Lloh2:
	ldr	x8, [x8]
	stur	x8, [x29, #-40]
	sub	w19, w0, #1
	ubfiz	x9, x19, #2, #32
	add	x8, x9, #15
	and	x8, x8, #0x7fffffff0
Lloh3:
	adrp	x16, ___chkstk_darwin@GOTPAGE
Lloh4:
	ldr	x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
	blr	x16
	mov	x9, sp
	sub	x20, x9, x8
	mov	sp, x20
	add	x21, x1, #8
	cbz	w19, LBB0_3
; %bb.1:
	cmp	w19, #3
	b.hi	LBB0_4
; %bb.2:
	mov	x8, #0                          ; =0x0
	b	LBB0_13
LBB0_3:
	mov	x0, x21
	mov	x1, x20
	mov	x2, x19
	bl	_shell_sort
Lloh5:
	adrp	x0, l_str@PAGE
Lloh6:
	add	x0, x0, l_str@PAGEOFF
	bl	_puts
	b	LBB0_16
LBB0_4:
	adrp	x9, lCPI0_0@PAGE
	cmp	w19, #16
	b.hs	LBB0_6
; %bb.5:
	mov	x8, #0                          ; =0x0
	b	LBB0_10
LBB0_6:
	and	x8, x19, #0xfffffff0
	ldr	q0, [x9, lCPI0_0@PAGEOFF]
	add	x10, x20, #32
	movi.4s	v1, #4
	movi.4s	v2, #8
	movi.4s	v3, #12
	movi.4s	v4, #16
	mov	x11, x8
LBB0_7:                                 ; =>This Inner Loop Header: Depth=1
	add.4s	v5, v0, v1
	add.4s	v6, v0, v2
	add.4s	v7, v0, v3
	stp	q0, q5, [x10, #-32]
	stp	q6, q7, [x10], #64
	add.4s	v0, v0, v4
	subs	x11, x11, #16
	b.ne	LBB0_7
; %bb.8:
	cmp	x8, x19
	b.eq	LBB0_14
; %bb.9:
	tst	x19, #0xc
	b.eq	LBB0_13
LBB0_10:
	mov	x10, x8
	dup.4s	v0, w10
	and	x8, x19, #0xfffffffc
	ldr	q1, [x9, lCPI0_0@PAGEOFF]
	orr.16b	v0, v0, v1
	sub	x9, x10, x8
	add	x10, x20, x10, lsl #2
	movi.4s	v1, #4
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	str	q0, [x10], #16
	add.4s	v0, v0, v1
	adds	x9, x9, #4
	b.ne	LBB0_11
; %bb.12:
	cmp	x8, x19
	b.eq	LBB0_14
LBB0_13:                                ; =>This Inner Loop Header: Depth=1
	str	w8, [x20, x8, lsl #2]
	add	x8, x8, #1
	cmp	x19, x8
	b.ne	LBB0_13
LBB0_14:
	mov	x0, x21
	mov	x1, x20
	mov	x2, x19
	bl	_shell_sort
Lloh7:
	adrp	x0, l_str@PAGE
Lloh8:
	add	x0, x0, l_str@PAGEOFF
	bl	_puts
Lloh9:
	adrp	x22, l_.str.1@PAGE
Lloh10:
	add	x22, x22, l_.str.1@PAGEOFF
LBB0_15:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [x20], #4
	ldr	x8, [x21, x8, lsl #3]
	str	x8, [sp, #-16]!
	mov	x0, x22
	bl	_printf
	add	sp, sp, #16
	subs	x19, x19, #1
	b.ne	LBB0_15
LBB0_16:
	mov	w0, #10                         ; =0xa
	bl	_putchar
	ldur	x8, [x29, #-40]
Lloh11:
	adrp	x9, ___stack_chk_guard@GOTPAGE
Lloh12:
	ldr	x9, [x9, ___stack_chk_guard@GOTPAGEOFF]
Lloh13:
	ldr	x9, [x9]
	cmp	x9, x8
	b.ne	LBB0_18
; %bb.17:
	mov	w0, #0                          ; =0x0
	sub	sp, x29, #32
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
LBB0_18:
	bl	___stack_chk_fail
	.loh AdrpLdrGotLdr	Lloh0, Lloh1, Lloh2
	.loh AdrpAdd	Lloh5, Lloh6
	.loh AdrpAdd	Lloh9, Lloh10
	.loh AdrpAdd	Lloh7, Lloh8
	.loh AdrpLdrGotLdr	Lloh11, Lloh12, Lloh13
	.loh AdrpLdrGot	Lloh3, Lloh4
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str.1:                               ; @.str.1
	.asciz	"%s "

l_str:                                  ; @str
	.asciz	"Result: "

.subsections_via_symbols
