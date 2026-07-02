	.build_version macos, 26, 0	sdk_version 26, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_shell_sort                     ; -- Begin function shell_sort
	.p2align	2
_shell_sort:                            ; @shell_sort
	.cfi_startproc
; %bb.0:
	cmp	w2, #2
	b.lo	LBB0_13				; if w2 < 2
; %bb.1:
	sub	sp, sp, #112
	stp	x28, x27, [sp, #16]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #32]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	mov	x19, x2
	mov	x20, x1
	mov	x21, x0
	mov	x8, x2
	b	LBB0_3
LBB0_2:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #12]                   ; 4-byte Folded Reload
	cmp	w8, #4
	mov	x8, x24
	b.lo	LBB0_12
LBB0_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_7 Depth 2
                                        ;       Child Loop BB0_9 Depth 3
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	lsr	w24, w8, #1
	cmp	w24, w19
	b.hs	LBB0_2
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	neg	w25, w24
	mov	x26, x24
	b	LBB0_7
LBB0_5:                                 ;   in Loop: Header=BB0_7 Depth=2
	mov	x8, x26
LBB0_6:                                 ;   in Loop: Header=BB0_7 Depth=2
	str	w27, [x20, w8, uxtw #2]
	add	x26, x26, #1
	cmp	w19, w26
	b.eq	LBB0_2
LBB0_7:                                 ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_9 Depth 3
	ldr	w27, [x20, x26, lsl #2]
	cmp	x26, x24
	b.lo	LBB0_5
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=2
	ldr	x22, [x21, w27, sxtw #3]
	mov	x28, x26
LBB0_9:                                 ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_7 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	add	w8, w25, w28
	ldr	w23, [x20, w8, uxtw #2]
	ldr	x1, [x21, x23, lsl #3]
	mov	x0, x22
	bl	_strcmp
	tbz	w0, #31, LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=3
	str	w23, [x20, w28, uxtw #2]
	sub	w8, w28, w24
	cmp	w8, w24
	mov	x28, x8
	b.hs	LBB0_9
	b	LBB0_6
LBB0_11:                                ;   in Loop: Header=BB0_7 Depth=2
	mov	x8, x28
	b	LBB0_6
LBB0_12:
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #112
LBB0_13:
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
