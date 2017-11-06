; *** Z80 �T���v���v���O���� ***
; 
; �{�@�͋N������SD�J�[�h���Ǎ��߂Ȃ��Ƃ�AVR��EEPROM��Z80�̃������ɔz�u����
; �X�^�[�g����悤�ɂ��Ă���܂��B
; EEPROM�̃T�C�Y�ȓ��̔C�ӂ�Z80�v���O�����𓮂������Ƃ��ł��܂��B
; ���̃f���v���O�����́A�f�B�X�N�̍đ}���𑣂����b�Z�[�W��\�����A
; IPL���ă��[�h���邾���̒P���Ȃ��̂ł��B
; �R�[�h���C���e��HEX�`���Ő����ł���A�Z���u�����g���Ă��������B
; ZASM http://www.vector.co.jp/soft/dos/prog/se010314.html
; �t�@�C���g���q��.hex����.eep�ɕς���AVR�v���O���}�œo�^���Ă��������B
;

cr:	equ	13
lf:	equ	10
stack:	equ	0100h		;default buffer address

	org	0

	ld	sp,stack
	jr 	start

	ds 	$33, 0

	org 	$0038
rst38h:
	ret

	ds 	$7, 0
	
	org 	$0040
start:
	ld	hl,SIGNON
	call	PRMSG
CONIN:
	in	a,(0)
	cp	0FFh
	jr	nz,CONIN
	in	a,(1)

	ld	a,0	;reload boot loader
	out	(16),a	;track
	out	(18),a	;sector
	out	(20),a	;dmaL
	ld	a,20h
	out	(21),a	;dmaH
	ld	a,1	;read
	out	(22),a
	in	a,(23)
	cp	0
	jp	nz,start
	jp	2000h

PRMSG:
	ld	a,(hl)
	or	a
	ret	z
	out	(2),a
	inc	hl
	jr	PRMSG

SIGNON:
	db	"INSERT SYSTEM DISK AND HIT ANY KEY!"
	db	cr,lf,0


end
