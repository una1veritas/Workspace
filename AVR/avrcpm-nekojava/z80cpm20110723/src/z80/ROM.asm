; *** Z80 サンプルプログラム ***
; 
; 本機は起動時にSDカードが読込めないときAVRのEEPROMをZ80のメモリに配置して
; スタートするようにしてあります。
; EEPROMのサイズ以内の任意のZ80プログラムを動かすことができます。
; このデモプログラムは、ディスクの再挿入を促すメッセージを表示し、
; IPLを再ロードするだけの単純なものです。
; コードをインテルHEX形式で生成できるアセンブラを使ってください。
; ZASM http://www.vector.co.jp/soft/dos/prog/se010314.html
; ファイル拡張子を.hexから.eepに変えてAVRプログラマで登録してください。
;

cr:	equ	13
lf:	equ	10
buff:	equ	0080h		;default buffer address

	org	0

	ld	sp,buff
start:
	ld	hl,SIGNON
	call	PRMSG
CONIN:
	in	a,(0)
	cp	0FFh
	jp	nz,CONIN
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

SIGNON:
	db	"INSERT SYSTEM DISK AND HIT ANY KEY!"
	db	cr,lf,0

PRMSG:
	ld	a,(hl)
	or	a
	ret	z
	out	(2),a
	inc	hl
	jp	PRMSG

end
