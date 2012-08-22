# ===============================================================
# RX62N用の単純でわかりやすいスタートアップ
#              (と少しのアセンブラルーチン)
# (C) Copyright 2011 特殊電子回路株式会社
# ===============================================================

	.text

# ===============================================================
# 外部に公開するシンボル
# ===============================================================
	.global _start
#	.global _exit
	.global _PowerON_Reset_PC
	.global __rx_fini
	.global ___cxa_atexit
	.global ___dso_handle

# ===============================================================
# 外部で公開されたシンボルを参照する
# ===============================================================
	.extern _main
	.extern _Relocatable_Vectors
	.extern _usp_init # リンカスクリプトで定義される
	.extern _isp_init # リンカスクリプトで定義される
	.extern _sbss     # リンカスクリプトで定義される
	.extern _ebss     # リンカスクリプトで定義される

# ===============================================================
# リセット後最初に呼び出されるルーチン
# ===============================================================

_PowerON_Reset_PC:
_start:

#WDTの確認
	mov.l	#0x8802b, r14	; WDT.RSTCSR.WOVFの内容を確認
	btst	#7,[r14].b  
	bz.b	_set_stack		; ゼロならば(WDTリセットではない)、スタックの設定へ
	
	mov.l	#4, r14			; 0x00000004番地の内容を確認
	mov.l	[r14], r4
	mov.l	#0, [r14]		; キーワードをクリア
	cmp		#0x4e444b54, r4	; "TKDN"になっていなければ、
	bne.b	_set_stack		; スタックの設定へ

#ユーザルーチンへジャンプ
#仮スタックの設定
	mov.l	#8, r14			
	mov.l	[r14], r14		; 8番地の内容を読む
	cmp		#0x7ffffff, r14 ; 
	ble.b	_jmp_to_userprog
	mvtc	#_usp_init, usp
	mvtc	#_isp_init, isp
	bsr.a	_sdram_init     ; SDRAMの初期化を先にやっておく

_jmp_to_userprog:
	mov.l	#8, r14			
	mov.l	[r14], r14		; 8番地の内容を読む
	jmp		r14				; ジャンプ！

#スタックの設定
_set_stack:
	mvtc	#_usp_init, usp
	mvtc	#_isp_init, isp

#割り込みベクタの設定
	mov.l	#_Relocatable_Vectors, r5
	mvtc	r5,intb

	mov.l	#0x100, r5
	mvtc	r5,fpsw

#Iレジスタを設定し、割りも割り込み許可する
	mov.l	#0x10000, r5
	mvtc	r5,psw

#PMレジスタを設定し、ユーザモードに移行する
#	mvfc	psw,r1
#	or		#0x100000, r1
#	push.l	r1

#UレジスタをセットするためにRTE命令を実行する
#	mvfc	pc,r1
#	add		#0x0a,r1
#	push.l	r1
#	rte
#	nop
#	nop

#データのコピー
	mov	#__datastart, r1
	mov	#__romdatastart, r2
	mov	#__romdatacopysize, r3
	smovf

#BSSのクリア
	mov	#__bssstart, r1 ; 転送先
	mov	#0, r2          ; 書き込むデータ
	mov	#__bsssize, r3  ; 書き込み長
	sstr.l

#ハードウェアセットアップの呼び出し
	bsr.a	_tkdn_hwsetup

#グローバルコンストラクタの呼び出し
	mov		#__init_array_start,r1
	mov		#__init_array_end,r2
	mov		#4, r3
	bsr.a	_rx_run_inilist

#mainへジャンプする
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

	
