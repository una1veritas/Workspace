	.text

	.extern		main
	.extern		_sp_base
#======================================
#      Initialize vectors
#
##     0x00 	 Reset
##     0x04 	 Undefined Instruction
##     0x08 	 Software Interrupt
##     0x0c 	 Abort(prefetch)
##     0x10 	 Abort(data)
##     0x14 	 Reserved
##     0x18 	 IRQ
##     0x1c 	 FIQ
#======================================
	.global _startup

	B	_startup
	nop
	nop
	nop
	nop
	nop
	B	_IRQ_handle
	nop
	.org 0x20	
_startup:
	#IRQ通知を受け取るように設定する
	MRS	R0, cpsr
	eor	R0, R0, #0x80
	MSR	cpsr, R0
	#プロセッサモードをIRQモードへ	
	MRS	r1, CPSR
	BIC	R1, R1, #0x1F
	ORR	R1, R1, #0x12
	MSR	CPSR, R1
	#例外ハンドラ用スタックを設定
	LDR	sp, = _exception_sp_base

	#プロセッサモードをスーパーバイザモードへ戻す
	MRS	r1, CPSR
	BIC	R1, R1, #0x1F
	ORR	R1, R1, #0x13
	MSR	CPSR, R1

	# スタックを設定
	LDR	R13,=_sp_base
.ifdef THUMB
	LDR     R1,=main
	ORR     R1, R1 ,#1
	BX      R1
.else
	BL	main
.endif
	.align 0x4
_IRQ_handle:
	#例外用スタックにレジスタ退避	
	STMDB	sp!,{r0-r12, r14}
	MRS	r0, spsr
	STMDB	sp!, {r0}

	bl	IRQ_func
		
	#例外用スタックからレジスタ復帰	
	LDMIA	sp!, {r0}
	MSR	spsr_cf, r0
	LDMIA	sp!,{r0-r12,r14}
		
	SUBS	PC, R14, #4
