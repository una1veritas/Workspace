*	Vector Addresses
*
trap3:	equ	$8c			*	Trap 3 vector
*
_ccp		equ	$4BC		* V1.3
_bdos		equ	$4CA		* V1.3
START_BIOS	equ	$6200		* V1.3

cpm		equ	$400 		*	Lowest addr of CP/M 
TPA_S		equ	$8000
MEM_END		equ	$80000
TPA_SIZE	equ	MEM_END-TPA_S

	org	START_BIOS

*
*	Global Code addresses
*
	xdef	.init		* at M68KBIOA.S ( this program )
	xdef	.wboot		* at M68KBIOA.S ( this program )
*	xref	.memtab		* memory region table at M68KBIOS.C
*	xref	.biosinit	* at M68KBIOS.C
*	xref	.cbios		* at M68KBIOS.C

.init:
	lea	entry,a0	* set bios call entory vector
	move.l	a0,trap3

*
*	Set TPA Size
**
* struct mrt {
*	uint16_t count;
*	uint32_t tpalow;
*	uint32_t tpalen;
*} memtab;				/* Initialized in M68KBIOA.S	*/
*

	lea	.memtab,a0		*	a0 -> Memory region table
	move.w	#1,(a0)+		*	1 region
	move.l	#TPA_S,(a0)+		*	TPA starts at TPA_S
	move.l	#TPA_SIZE,(a0)+		*	Ends where CP/M begins

	move	#$2000,sr
	jsr	.biosinit
	clr.l	d0
	rts
*
.wboot:	clr.l	d0
	jmp	_ccp
*
entry:	move.l	d2,-(a7)
	move.l	d1,-(a7)
	move.l	d0,-(a7)
	jsr	.cbios
	add.l	#12,a7
	rte

*	Microtec MCC68K Compiler  Version 3.3K 
*	OPT	NOABSPCADD,E,CASE
m68cbios	IDNT	
*	SECTION	9,,C
	XDEF	.wakeup_pic
.wakeup_pic:
	MOVE.L	.pic,A0
	MOVE.B	(A0),D0
_L4:
	MOVE.L	.preq,A0
	TST.B	6(A0)
	BNE.S	_L4
	MOVE.L	.preq,A0
	MOVE.B	7(A0),D1
	MOVEQ	#0,D0
	MOVE.B	D1,D0	*fr
	RTS
*	SECTION	14,,D
* allocations for .wakeup_pic
*	D0	.dummy
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .wakeup_pic
*	CODE SIZE                	   36 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3991 BYTES
*	COMPILE-TIME GLOBAL HEAP 	21105 BYTES
	XDEF	.order_b2l
.order_b2l:
	MOVE.L	D2,-(SP)
	MOVE.W	10(SP),D1
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	ANDI.L	#255,D0
	ASL.L	#8,D0
	MOVE.W	D0,D2
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	ANDI.L	#65280,D0
	ASR.L	#8,D0
	MOVE.W	D0,D0
	MOVEQ	#0,D1
	MOVE.W	D0,D1	*fr
	MOVEQ	#0,D0
	MOVE.W	D2,D0	*fr
	OR.L	D0,D1
	MOVE.W	D1,D0
	MOVEQ	#0,D1
	MOVE.W	D0,D1	*fr
	MOVE.L	D1,D0
	MOVE.L	(SP)+,D2
	RTS
*	SECTION	14,,D
* allocations for .order_b2l
*	D2	.p1
*	D0	.p2
*	10(SP)	.data
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .order_b2l
*	CODE SIZE                	   56 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 6492 BYTES
*	COMPILE-TIME GLOBAL HEAP 	21376 BYTES
	XDEF	.set_dma
.set_dma:
	MOVEM.L	D2/D3/D4,-(SP)
	MOVE.L	16(SP),D1
	MOVE.L	D1,D2
	ANDI.L	#255,D2
	MOVEQ	#24,D0
	LSL.L	D0,D2
	MOVE.L	D1,D4
	ANDI.L	#65280,D4
	LSL.L	#8,D4
	MOVE.L	D1,D3
	ANDI.L	#16711680,D3
	LSR.L	#8,D3
	ANDI.L	#-16777216,D1
	LSR.L	D0,D1
	MOVE.L	.preq,A0
	MOVE.L	D4,D0
	OR.L	D2,D0
	OR.L	D3,D0
	OR.L	D1,D0
	MOVE.L	D0,14(A0)
	MOVEM.L	(SP)+,D2/D3/D4
	RTS
*	SECTION	14,,D
* allocations for .set_dma
*	D2	.p1
*	D4	.p2
*	D3	.p3
*	D1	.p4
*	16(SP)	.dma_adr
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .set_dma
*	CODE SIZE                	   72 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 6388 BYTES
*	COMPILE-TIME GLOBAL HEAP 	21753 BYTES
	XDEF	.con_st
.con_st:
	MOVE.L	.preq,A0
	MOVE.B	#3,6(A0)
	JSR	.wakeup_pic
	MOVE.B	D0,D0
	MOVEQ	#0,D1
	MOVE.B	D0,D1	*fr
	MOVE.L	D1,D0
	RTS
*	SECTION	14,,D
* allocations for .con_st
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .con_st
*	CODE SIZE                	   28 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3538 BYTES
*	COMPILE-TIME GLOBAL HEAP 	21936 BYTES
	XDEF	.con_in
.con_in:
	MOVE.L	.preq,A0
	MOVE.B	#1,6(A0)
	JSR	.wakeup_pic
	MOVE.B	D0,D0
	MOVEQ	#0,D1
	MOVE.B	D0,D1	*fr
	MOVE.L	D1,D0
	RTS
*	SECTION	14,,D
* allocations for .con_in
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .con_in
*	CODE SIZE                	   28 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3538 BYTES
*	COMPILE-TIME GLOBAL HEAP 	22119 BYTES
	XDEF	.con_out
.con_out:
	MOVE.B	7(SP),D0
	MOVE.L	.preq,A0
	MOVE.B	#2,6(A0)
	MOVE.L	.preq,A0
	MOVE.B	D0,7(A0)
	JSR	.wakeup_pic
	RTS
*	SECTION	14,,D
* allocations for .con_out
*	7(SP)	.ch
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .con_out
*	CODE SIZE                	   34 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3206 BYTES
*	COMPILE-TIME GLOBAL HEAP 	22390 BYTES
	XDEF	.read
.read:
	MOVE.L	.preq,A0
	MOVE.B	#5,6(A0)
	JSR	.wakeup_pic
	MOVE.W	D0,D0
	MOVE.W	D0,D1
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	RTS
*	SECTION	14,,D
* allocations for .read
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .read
*	CODE SIZE                	   28 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3576 BYTES
*	COMPILE-TIME GLOBAL HEAP 	22598 BYTES
	XDEF	.write
.write:
	MOVE.B	7(SP),D0
	MOVE.L	.preq,A0
	MOVE.B	#6,6(A0)
	JSR	.wakeup_pic
	MOVE.W	D0,D0
	MOVE.W	D0,D1
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	RTS
*	SECTION	14,,D
* allocations for .write
*	7(SP)	.mode
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .write
*	CODE SIZE                	   32 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3933 BYTES
*	COMPILE-TIME GLOBAL HEAP 	22894 BYTES
	XDEF	.sectran
.sectran:
	MOVE.L	D2,-(SP)
	MOVE.W	10(SP),D2
	MOVE.L	12(SP),D1
	TST.L	D1
	BEQ.S	_L38
	MOVEQ	#0,D0
	MOVE.W	D2,D0	*fr
	ADD.L	D1,D0
	MOVEQ	#0,D1
	MOVE.L	D0,A0
	MOVE.B	(A0),D1	*fr
	BRA.S	_L36
_L38:
	MOVEQ	#0,D0
	MOVE.W	D2,D0	*fr
	ADDQ.L	#1,D0
	MOVE.W	D0,D1
_L36:
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	MOVE.L	(SP)+,D2
	RTS
*	SECTION	14,,D
* allocations for .sectran
*	10(SP)	.s
*	12(SP)	.xp
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .sectran
*	CODE SIZE                	   44 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 6536 BYTES
*	COMPILE-TIME GLOBAL HEAP 	23323 BYTES
	XDEF	.setxvect
.setxvect:
	MOVE.L	D2,-(SP)
	MOVE.W	10(SP),D1
	MOVE.L	12(SP),D2
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	LSL.L	#2,D0
	MOVE.L	D0,A0
	MOVE.L	(A0),D0
	MOVE.L	D2,(A0)
	MOVE.L	(SP)+,D2
	RTS
*	SECTION	14,,D
* allocations for .setxvect
*	D0	.oldval
*	A0	.vloc
*	10(SP)	.vnum
*	12(SP)	.vval
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .setxvect
*	CODE SIZE                	   26 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 4465 BYTES
*	COMPILE-TIME GLOBAL HEAP 	23772 BYTES
	XDEF	.slctdsk
.slctdsk:
	MOVE.B	7(SP),D0
	CMPI.B	#3,D0
	BLS.S	_L48
	MOVEQ	#0,D0
	BRA.S	_L46
_L48:
	MOVE.L	.preq,A0
	MOVE.B	D0,8(A0)
	MOVEQ	#0,D1
	MOVE.B	D0,D1	*fr
	MOVE.W	D1,D0
	MULU	#26,D0
	MOVE.L	#.dphtab,A0
	ADDA.W	D0,A0
	MOVE.L	A0,D0
_L46:
	RTS
*	SECTION	14,,D
* allocations for .slctdsk
*	7(SP)	.dsk
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .slctdsk
*	CODE SIZE                	   46 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 5103 BYTES
*	COMPILE-TIME GLOBAL HEAP 	24117 BYTES
	XDEF	.biosinit
.biosinit:
	MOVE.L	#256,.preq
	MOVE.L	#524288,.pic
	MOVE.L	.preq,A0
	CLR.B	D0
_L54:
	CLR.B	(A0)+
	ADDQ.B	#1,D0
	CMPI.B	#18,D0
	BCS.S	_L54
	RTS
*	SECTION	14,,D
* allocations for .biosinit
*	D0	.c
*	A0	.p
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .biosinit
*	CODE SIZE                	   40 BYTES
*	COMPILE-TIME DYNAMIC HEAP	 3750 BYTES
*	COMPILE-TIME GLOBAL HEAP 	24480 BYTES
	XDEF	.cbios
.cbios:
	MOVEM.L	D2/A2,-(SP)
	MOVE.W	14(SP),D0
	MOVE.L	16(SP),D1
	MOVE.L	20(SP),D2
	MOVE.L	#.preq,A0
	CMPI.W	#22,D0
	BHI	_L60
	ADD.W	D0,D0
	MOVE.W	_L83(PC,D0.W),D0
	JMP	_L84(PC,D0.W)
_L84:
_L83:
	DC.W	_L80-_L84
	DC.W	_L79-_L84
	DC.W	_L78-_L84
	DC.W	_L77-_L84
	DC.W	_L76-_L84
	DC.W	_L75-_L84
	DC.W	_L75-_L84
	DC.W	_L75-_L84
	DC.W	_L74-_L84
	DC.W	_L73-_L84
	DC.W	_L72-_L84
	DC.W	_L71-_L84
	DC.W	_L70-_L84
	DC.W	_L69-_L84
	DC.W	_L68-_L84
	DC.W	_L67-_L84
	DC.W	_L66-_L84
	DC.W	_L60-_L84
	DC.W	_L65-_L84
	DC.W	_L64-_L84
	DC.W	_L63-_L84
	DC.W	_L60-_L84
	DC.W	_L61-_L84
_L80:
	JSR	.biosinit
	BRA	_L60
_L79:
	JSR	.wboot
_L78:
	JSR	.con_st
_L89:
	MOVE.B	D0,D0
	MOVEQ	#0,D1
	MOVE.B	D0,D1	*fr
	BRA	_L59
_L77:
	JSR	.con_in
	BRA.S	_L89
_L76:
	MOVEQ	#0,D0
	MOVE.B	D1,D0	*fr
	MOVE.L	D0,-(SP)
*			STACK OFFSET 4
	JSR	.con_out
_L86:
*			STACK OFFSET 0
	ADDQ.L	#4,SP
	BRA	_L60
_L75:
	BRA	_L90
_L74:
	MOVE.L	(A0),A0
	CLR.W	10(A0)
	BRA	_L60
_L73:
	MOVEQ	#0,D0
	MOVE.B	D1,D0	*fr
	MOVE.L	D0,-(SP)
*			STACK OFFSET 4
	JSR	.slctdsk
*			STACK OFFSET 0
	ADDQ.L	#4,SP
	BRA	_L93
_L72:
	MOVE.L	(A0),A2
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	MOVE.L	D0,-(SP)
*			STACK OFFSET 4
	JSR	.order_b2l
*			STACK OFFSET 0
	ADDQ.L	#4,SP
	MOVE.W	D0,D0
	MOVE.W	D0,10(A2)
	BRA	_L60
_L71:
	MOVE.L	(A0),A2
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	MOVE.L	D0,-(SP)
*			STACK OFFSET 4
	JSR	.order_b2l
*			STACK OFFSET 0
	ADDQ.L	#4,SP
	MOVE.W	D0,D0
	MOVE.W	D0,12(A2)
	BRA.S	_L60
_L70:
	MOVE.L	D1,-(SP)
*			STACK OFFSET 4
	JSR	.set_dma
	BRA.S	_L86
_L69:
	JSR	.read
	BRA.S	_L87
_L68:
	JSR	.write
	BRA.S	_L88
_L67:
	BRA.S	_L91
_L66:
	MOVE.L	D2,-(SP)
*			STACK OFFSET 4
	MOVE.L	D1,-(SP)
*			STACK OFFSET 8
	JSR	.sectran
*			STACK OFFSET 0
	ADDQ.L	#8,SP
_L87:
_L88:
	MOVE.W	D0,D1
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	BRA.S	_L94
_L65:
	MOVE.L	#.memtab,D1
	BRA.S	_L59
_L64:
	MOVEQ	#0,D0
	MOVE.W	.iobyte,D0	*fr
	BRA.S	_L92
_L63:
	MOVE.W	D1,.iobyte
	BRA.S	_L60
_L61:
	MOVE.L	D2,-(SP)
*			STACK OFFSET 4
	MOVEQ	#0,D0
	MOVE.W	D1,D0	*fr
	MOVE.L	D0,-(SP)
*			STACK OFFSET 8
	JSR	.setxvect
*			STACK OFFSET 0
	ADDQ.L	#8,SP
_L92:
_L93:
_L94:
	MOVE.L	D0,D1
	BRA.S	_L59
_L60:
_L90:
_L91:
	MOVEQ	#0,D1
_L59:
	MOVE.L	D1,D0
	MOVEM.L	(SP)+,D2/A2
	RTS
*	SECTION	14,,D
* allocations for .cbios
*	14(SP)	.d0
*	16(SP)	.d1
*	20(SP)	.d2
*	SECTION	9,,C

* COMPILATION SUMMARY FOR .cbios
*	CODE SIZE                	  330 BYTES
*	COMPILE-TIME DYNAMIC HEAP	30891 BYTES
*	COMPILE-TIME GLOBAL HEAP 	25280 BYTES
*	SECTION	14,,D
	XDEF	.dpb2
.dpb2:	DC.B	0
	DC.B	128
	DC.B	4
	DC.B	15
	DCB.B	1,0
	DCB.B	1,0
	DC.B	7
	DC.B	247
	DC.B	3
	DC.B	255
	DCB.B	1,0
	DCB.B	1,0
	DCB.B	2,0
	DCB.B	2,0
	XDEF	.dphtab
.dphtab:	DCB.B	4,0
	DCB.B	2,0
	DCB.B	2,0
	DCB.B	2,0
	DC.L	.dirbuf+0
	DC.L	.dpb2
	DCB.B	4,0
	DC.L	.alv0+0
	DCB.B	4,0
	DCB.B	2,0
	DCB.B	2,0
	DCB.B	2,0
	DC.L	.dirbuf+0
	DC.L	.dpb2
	DCB.B	4,0
	DC.L	.alv1+0
	DCB.B	4,0
	DCB.B	2,0
	DCB.B	2,0
	DCB.B	2,0
	DC.L	.dirbuf+0
	DC.L	.dpb2
	DCB.B	4,0
	DC.L	.alv2+0
	DCB.B	4,0
	DCB.B	2,0
	DCB.B	2,0
	DCB.B	2,0
	DC.L	.dirbuf+0
	DC.L	.dpb2
	DCB.B	4,0
	DC.L	.alv3+0
*	XREF	.wboot
	XDEF	.preq
.preq:	DCB.B	4,0
	XDEF	.pic
.pic:	DCB.B	4,0
	XDEF	.dirbuf
.dirbuf:	DCB.B	128,0
	XDEF	.alv0
.alv0:	DCB.B	256,0
	XDEF	.alv1
.alv1:	DCB.B	256,0
	XDEF	.alv2
.alv2:	DCB.B	256,0
	XDEF	.alv3
.alv3:	DCB.B	256,0
	XDEF	.memtab
.memtab:	DCB.B	10,0
	XDEF	.iobyte
.iobyte:	DCB.B	2,0
* allocations for module
*	common	.preq
*	common	.pic
*	common	.dirbuf
*	common	.alv0
*	common	.alv1
*	common	.alv2
*	common	.alv3
*	common	.dpb2
*	common	.dphtab
*	common	.memtab
*	common	.iobyte
*	SECTION	9,,C

* COMPILATION SUMMARY FOR MODULE
*	CODE SIZE                	  800 BYTES
	END
