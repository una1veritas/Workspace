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
	#IRQ�ʒm���󂯎��悤�ɐݒ肷��
	MRS	R0, cpsr
	eor	R0, R0, #0x80
	MSR	cpsr, R0
	#�v���Z�b�T���[�h��IRQ���[�h��	
	MRS	r1, CPSR
	BIC	R1, R1, #0x1F
	ORR	R1, R1, #0x12
	MSR	CPSR, R1
	#��O�n���h���p�X�^�b�N��ݒ�
	LDR	sp, = _exception_sp_base

	#�v���Z�b�T���[�h���X�[�p�[�o�C�U���[�h�֖߂�
	MRS	r1, CPSR
	BIC	R1, R1, #0x1F
	ORR	R1, R1, #0x13
	MSR	CPSR, R1

	# �X�^�b�N��ݒ�
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
	#��O�p�X�^�b�N�Ƀ��W�X�^�ޔ�	
	STMDB	sp!,{r0-r12, r14}
	MRS	r0, spsr
	STMDB	sp!, {r0}

	bl	IRQ_func
		
	#��O�p�X�^�b�N���烌�W�X�^���A	
	LDMIA	sp!, {r0}
	MSR	spsr_cf, r0
	LDMIA	sp!,{r0-r12,r14}
		
	SUBS	PC, R14, #4
