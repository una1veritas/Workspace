# ===============================================================
# RX62N�p�̒P���ł킩��₷���X�^�[�g�A�b�v
#              (�Ə����̃A�Z���u�����[�`��)
# (C) Copyright 2011 ����d�q��H�������
# ===============================================================

	.text

# ===============================================================
# �O���Ɍ��J����V���{��
# ===============================================================
	.global _start
#	.global _exit
	.global _PowerON_Reset_PC
	.global __rx_fini
	.global ___cxa_atexit
	.global ___dso_handle

# ===============================================================
# �O���Ō��J���ꂽ�V���{�����Q�Ƃ���
# ===============================================================
	.extern _main
	.extern _Relocatable_Vectors
	.extern _usp_init # �����J�X�N���v�g�Œ�`�����
	.extern _isp_init # �����J�X�N���v�g�Œ�`�����
	.extern _sbss     # �����J�X�N���v�g�Œ�`�����
	.extern _ebss     # �����J�X�N���v�g�Œ�`�����

# ===============================================================
# ���Z�b�g��ŏ��ɌĂяo����郋�[�`��
# ===============================================================

_PowerON_Reset_PC:
_start:

#WDT�̊m�F
	mov.l	#0x8802b, r14	; WDT.RSTCSR.WOVF�̓��e���m�F
	btst	#7,[r14].b  
	bz.b	_set_stack		; �[���Ȃ��(WDT���Z�b�g�ł͂Ȃ�)�A�X�^�b�N�̐ݒ��
	
	mov.l	#4, r14			; 0x00000004�Ԓn�̓��e���m�F
	mov.l	[r14], r4
	mov.l	#0, [r14]		; �L�[���[�h���N���A
	cmp		#0x4e444b54, r4	; "TKDN"�ɂȂ��Ă��Ȃ���΁A
	bne.b	_set_stack		; �X�^�b�N�̐ݒ��

#���[�U���[�`���փW�����v
#���X�^�b�N�̐ݒ�
	mov.l	#8, r14			
	mov.l	[r14], r14		; 8�Ԓn�̓��e��ǂ�
	cmp		#0x7ffffff, r14 ; 
	ble.b	_jmp_to_userprog
	mvtc	#_usp_init, usp
	mvtc	#_isp_init, isp
	bsr.a	_sdram_init     ; SDRAM�̏��������ɂ���Ă���

_jmp_to_userprog:
	mov.l	#8, r14			
	mov.l	[r14], r14		; 8�Ԓn�̓��e��ǂ�
	jmp		r14				; �W�����v�I

#�X�^�b�N�̐ݒ�
_set_stack:
	mvtc	#_usp_init, usp
	mvtc	#_isp_init, isp

#���荞�݃x�N�^�̐ݒ�
	mov.l	#_Relocatable_Vectors, r5
	mvtc	r5,intb

	mov.l	#0x100, r5
	mvtc	r5,fpsw

#I���W�X�^��ݒ肵�A��������荞�݋�����
	mov.l	#0x10000, r5
	mvtc	r5,psw

#PM���W�X�^��ݒ肵�A���[�U���[�h�Ɉڍs����
#	mvfc	psw,r1
#	or		#0x100000, r1
#	push.l	r1

#U���W�X�^���Z�b�g���邽�߂�RTE���߂����s����
#	mvfc	pc,r1
#	add		#0x0a,r1
#	push.l	r1
#	rte
#	nop
#	nop

#�f�[�^�̃R�s�[
	mov	#__datastart, r1
	mov	#__romdatastart, r2
	mov	#__romdatacopysize, r3
	smovf

#BSS�̃N���A
	mov	#__bssstart, r1 ; �]����
	mov	#0, r2          ; �������ރf�[�^
	mov	#__bsssize, r3  ; �������ݒ�
	sstr.l

#�n�[�h�E�F�A�Z�b�g�A�b�v�̌Ăяo��
	bsr.a	_tkdn_hwsetup

#�O���[�o���R���X�g���N�^�̌Ăяo��
	mov		#__init_array_start,r1
	mov		#__init_array_end,r2
	mov		#4, r3
	bsr.a	_rx_run_inilist

#main�փW�����v����
	mov		#0,r1
	mov		#0,r2
	mov		#0,r3
	bsr.a	_main
	bra		_exit

_rx_run_inilist:
next_inilist:
	cmp	r1,r2
	beq.b	done_inilist
	mov.l	[r1],r4
	cmp	#-1, r4
	beq.b	skip_inilist
	cmp	#0, r4
	beq.b	skip_inilist
	pushm	r1-r3
	jsr	r4
	popm	r1-r3
skip_inilist:
	add	r3,r1
	bra.b	next_inilist
done_inilist:
	rts

_exit:
	wait
	bra		_exit

__rx_fini:
	rts

___cxa_atexit:
	.weak   ___cxa_atexit
	rts

	.section .data
	.weak   ___dso_handle
___dso_handle:
	.long	0

	
