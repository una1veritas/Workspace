;    MMC/SD-Card routines
;
;    Copyright (C) 2010 Leo C.
;
;    This file is part of avrcpm.
;
;    avrcpm is free software: you can redistribute it and/or modify it
;    under the terms of the GNU General Public License as published by
;    the Free Software Foundation, either version 3 of the License, or
;    (at your option) any later version.
;
;    avrcpm is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with avrcpm.  If not, see <http://www.gnu.org/licenses/>.
;
;    $Id: mmc.asm 165 2011-04-19 22:45:26Z leo $
;

/* Definitions for MMC/SDC command */
#define CMD0	(0)		/* GO_IDLE_STATE */
#define CMD1	(1)		/* SEND_OP_COND (MMC) */
#define	ACMD41	(0x80+41)	/* SEND_OP_COND (SDC) */
#define CMD8	(8)		/* SEND_IF_COND */
#define CMD9	(9)		/* SEND_CSD */
#define CMD10	(10)		/* SEND_CID */
#define CMD12	(12)		/* STOP_TRANSMISSION */
#define ACMD13	(0x80+13)	/* SD_STATUS (SDC) */
#define CMD16	(16)		/* SET_BLOCKLEN */
#define CMD17	(17)		/* READ_SINGLE_BLOCK */
#define CMD18	(18)		/* READ_MULTIPLE_BLOCK */
#define CMD23	(23)		/* SET_BLOCK_COUNT (MMC) */
#define	ACMD23	(0x80+23)	/* SET_WR_BLK_ERASE_COUNT (SDC) */
#define CMD24	(24)		/* WRITE_BLOCK */
#define CMD25	(25)		/* WRITE_MULTIPLE_BLOCK */
#define CMD55	(55)		/* APP_CMD */
#define CMD58	(58)		/* READ_OCR */

/* Disk Status Bits (DSTATUS) */
#define MMCST_NOINIT	0x01	/* Drive not initialized */
#define MMCST_NODISK	0x02	/* No medium in the drive */
#define MMCST_PROTECT	0x04	/* Write protected */

/* Card type flags (CardType) */
#define CT_MMC		0x01	/* MMC ver 3 */
#define CT_SD1		0x02	/* SD ver 1 */
#define CT_SD2		0x04	/* SD ver 2 */
#define CT_SDC	(CT_SD1|CT_SD2)	/* SD */
#define CT_BLOCK	0x08	/* Block addressing */

#define	RES_OK 		0	/* 0: Successful */
#define	RES_ERROR	1	/* 1: R/W Error */
#define	RES_WRPRT	2	/* 2: Write Protected */
#define	RES_NOTRDY	3	/* 3: Not Ready */
#define	RES_PARERR	4	/* 4: Invalid Parameter */


#define SPI_MODE_0	(0<<CPOL)|(0<<CPHA)
#define SPI_MODE_1	(0<<CPOL)|(1<<CPHA)
#define SPI_MODE_2	(1<<CPOL)|(0<<CPHA)
#define SPI_MODE_3	(1<<CPOL)|(1<<CPHA)
#define SPI_MODE	SPI_MODE_0

;------------------------------------------------
; 
.macro	spi_clkslow
.if MMC_DEBUG > 1
	printstring	"SPI_CLK_SLOW "
.endif
	ldi	temp,SPI_MODE|(1<<SPE)|(1<<MSTR)|(1<<SPR1)|(1<<SPR0)	;clk/128
	out	SPCR,temp
	out	SPSR,_0
.endm

;------------------------------------------------
; 
.macro	spi_clkfast
.if MMC_DEBUG > 1
	printstring	"SPI_CLK_FAST "
.endif
	ldi	temp,SPI_MODE|(1<<SPE)|(1<<MSTR)			;clk/4
	out	SPCR,temp
#if MMC_SPI2X
	ldi	temp,(1<<SPI2X)
	out	SPSR,temp
#else
	out	SPSR,_0
#endif
.endm

;------------------------------------------------
; 
.macro	spi_disable
.if MMC_DEBUG > 1
	printstring	"SPI_DISABLE "
.endif
	out	SPCR,_0
.endm


;------------------------------------------------
;
.macro	spi_waitm
	.set	spiwl_ = PC
	sbism8	SPSR,SPIF
	rjmp	spiwl_
.endm

;------------------------------------------------

	.dseg

mmcStat:
	.byte	1
mmcCardType:
	.byte	1
mmc_ocr:
	.byte	4
	


	.cseg

;------------------------------------------------
; Multiply 32 bit value in yh,yl,xh,xl by 512

mul_yx_512:
	mov	yh,yl
	mov	yl,xh
	mov	xh,xl
	ldi	xl,0
	lsl	xh
	rol	yl
	rol	yh
	ret
		
;------------------------------------------------
spi_rcvr:
	out	SPDR,_255
spi_rcvr_l:
	sbism8 SPSR,SPIF
	rjmp	spi_rcvr_l
	in	temp,SPDR
.if MMC_DEBUG > 2
	push	temp
	printstring "<"
	rcall printhex
	printstring " "
	pop	temp
.endif
	ret

;------------------------------------------------
spi_xmit:
.if MMC_DEBUG > 2
	printstring ">"
	rcall printhex
	printstring " "
.endif
	out	SPDR,temp
					;fall thru
spi_wait:	
	sbism8	SPSR,SPIF
	rjmp	spi_wait
	ret
	
;------------------------------------------------
; Wait for card ready 
;	return  1:OK, 0:Timeout 

mmcWaitReady:
	push	temp2
	ldi	temp2,2			;Wait for ready in timeout of 500ms.
	rcall	spi_rcvr
mmc_wrl:
	sts	delay_timer2,_255
mmc_wrl1:
	rcall	spi_rcvr
	cp	temp,_255
	brne	mmc_wrl2
	ldi	temp,1
	rjmp	mmc_wrbreak

mmc_wrl2:
	lds	temp,delay_timer2
	cpi	temp,0
	brne	mmc_wrl1
	dec	temp2
	brne	mmc_wrl		;tmp is 0 here

mmc_wrbreak:
	pop	temp2	
	tst	temp		;set flags
	ret


;------------------------------------------------
; Deselect the card and release SPI bus
;	return 0

mmcDeselect:
	sbi	P_MMC_CS,mmc_cs		; CS high
	rcall	spi_rcvr
	clr	temp
	ret

;------------------------------------------------
; Select the card and wait for ready 
;	return 255:Successful, 0:Timeout

mmcSelect:
	cbi	P_MMC_CS,mmc_cs		; CS low
	rcall	mmcWaitReady
	breq	mmcDeselect		;return via Deselect
	sbr	temp,255
	ret
	

;------------------------------------------------
; Send a command packet to MMC 
; 	temp2:	Command
;       yh..xl:	Argument

mmcCmd:
	sbrs	temp2,7
	rjmp	mmc_cmddo

; ACMD<n> is the command sequence of CMD55-CMD<n>	
	
	push	yh
	push	yl
	push	xh
	push	xl
	push	temp2
	ldiw	y,0
	movw	x,y
	ldi	temp2,CMD55
	rcall	mmcCmd
	pop	temp2
	pop	xl
	pop	xh
	pop	yl
	pop	yh

	cpi	temp,2
	brlo	mmc_cmddo		; fall thru, if (retval <= 1)
	
	tst	temp
	ret				; else return error

; Select the card and wait for ready

mmc_cmddo:
.if MMC_DEBUG
	sbrc	temp2,7
	rjmp	dmmccmd_nonl
	printnewline

dmmccmd_nonl:
	printstring "mmcCMD: "
	mov	temp,temp2
	cbr	temp,0x80
	rcall	printhex
	printstring " "
	push	temp2
	movw	temp,y
	rcall	printhexw
	movw	temp,x
	rcall	printhexw
	printstring " "
	pop	temp2
.endif
	rcall	mmcDeselect
	rcall	mmcSelect
	brne	mmc_cmd_p

	ldi	temp,0xFF
	rjmp	mmc_cmdexit
	
; Send command packet

mmc_cmd_p:
	mov	temp,temp2
	cbr	temp,0x80
	sbr	temp,0x40
	rcall	spi_xmit
	out	SPDR,yh
	rcall	spi_wait
	out	SPDR,yl
	rcall	spi_wait
	out	SPDR,xh
	rcall	spi_wait
	out	SPDR,xl
	rcall	spi_wait
	
	ldi	temp,0x95		;CRC for CMD0(0)
	cpi	temp2,CMD0
	breq	mmc_cmdxcrc
	ldi	temp,0x87		;CRC for CMD8(0x1AA) 
	cpi	temp2,CMD8
	breq	mmc_cmdxcrc
	ldi	temp,0x01		;Dummy CRC + Stop
mmc_cmdxcrc:
.if MMC_DEBUG
	printstring ".. "
	rcall	printhex
.endif
	rcall	spi_xmit

; Receive command response

	cpi	temp2,CMD12		; Skip a stuff byte when stop reading
	brne	mmc_cmdres
	rcall	spi_rcvr
	
; Wait for a valid response in timeout of 10 attempts

mmc_cmdres:
	ldi	temp,10
	mov	_tmp1,temp
mmc_cmdrl:
	rcall	spi_rcvr
	sbrs	temp,7
	rjmp	mmc_cmdexit
	dec	_tmp1
	brne	mmc_cmdrl

; Return with  response value

mmc_cmdexit:
.if MMC_DEBUG
	printstring " CMDRes: "
	rcall	printhex
	printstring " "
	rcall	uart_wait_empty
.endif
	tst	temp			;set flags
	ret
	
;------------------------------------------------
; Check if 1 sec timeout
;	return Z-Flag set, if timeout

mmc_timeout_1s:
	lds	temp,delay_timer1
	tst	temp
	brne	mmc_ttex
	dec	temp4
	breq	mmc_ttex
	ldi	temp,100
	sts	delay_timer1,temp
mmc_ttex:
	ret


;------------------------------------------------
;       "Public" functions
;------------------------------------------------

;------------------------------------------------
; Initialize MMC/SD card

mmcInit:
.if MMC_DEBUG 
	printnewline
	printstring	"mmcInit "
.endif
	lds	temp,mmcStat		;Set 'NO INIT' status
	sbr	temp,MMCST_NOINIT
	sts	mmcStat,temp

	spi_clkslow
	ldi	temp2,10
mmci_lp:
	rcall	spi_rcvr
	dec	temp2			;80 dummy clocks
	brne	mmci_lp

	ldi	temp3,0			;Card type
	ldi	temp2,CMD0
	ldiw	y,0
	movw	x,y
	rcall	mmcCmd			;Enter Idle state
	cpi	temp,1
	breq	mmci_1
	rjmp	mmci_lend
mmci_1:	
	ldi	temp4,10		;Initialization timeout of 1000 ms.
	ldi	temp,100
	sts	delay_timer1,temp
	ldi	temp2,CMD8
	ldiw	y,0
	ldi	xh,0x01
	ldi	xl,0xAA
	rcall	mmcCmd
	cpi	temp,1			;SDv2?
	brne	mmci_sdv1

; Get trailing return value of R7 response

	ldi	temp2,4
	ldiw	z,mmc_ocr
mmci_v2l1:
	rcall	spi_rcvr
	st	z+,temp
	dec	temp2
	brne	mmci_v2l1
	sbiw	z,4
	ldd	temp,z+2
	cpi	temp,0x01
	ldd	temp,z+3
	cpc	temp,xl			;Reuse 0xAA value in xl
	brne	mmci_sdv1

; The card can work at vdd range of 2.7-3.6V.
; Wait for leaving idle state (ACMD41 with HCS bit).
	
	ldi	temp2,ACMD41
	ldi	yh,0x40
	ldi	yl,0
	ldi	xh,0
	ldi	xl,0
mmci_v2l2:
	rcall	mmcCmd
	breq	mmci_ccc
	rcall	mmc_timeout_1s
	brne	mmci_v2l2
	rjmp	mmci_sdv2end
	
; Check CCS bit in the OCR
mmci_ccc:
	ldi	temp2,CMD58
	ldi	yh,0
	rcall	mmcCmd
	brne	mmci_sdv2end

	ldi	temp2,4
mmci_v2l3:
	rcall	spi_rcvr
	st	z+,temp
	dec	temp2
	brne	mmci_v2l3
	sbiw	z,4
	
	sbr	temp3,CT_SD2
	ldd	temp,z+0
	sbrc	temp,6
	sbr	temp3,CT_BLOCK

mmci_sdv2end:
	rjmp	mmci_lend
	
; SDv1 or MMCv3

mmci_sdv1:
	ldi	temp2,ACMD41
	ldiw	y,0
	movw	x,y
	rcall	mmcCmd	
	cpi	temp,2
	brsh	mmci_mmcv3
	sbr	temp3,CT_SD1		;SDv1
	ldi	temp2,ACMD41
	rjmp	mmci_v1_l
mmci_mmcv3:
	sbr	temp3,CT_MMC		;MMCv3
	ldi	temp2,CMD1

; Wait for leaving idle state
mmci_v1_l:
	rcall	mmcCmd
	breq	mmci_v1_2
	rcall	mmc_timeout_1s
	brne	mmci_v1_l
	rjmp	mmci_lend		;Timeout
	
; Set R/W block length to 512
mmci_v1_2:
	ldi	temp2,CMD16		
	ldiw	x,512
	rcall	mmcCmd
	breq	mmci_lend
	ldi	temp3,0
	
mmci_lend:
	sts	mmcCardType,temp3
	rcall	mmcDeselect
	
; Initialization succeded?

	lds	temp,mmcStat
	tst	temp3
	breq	mmci_lex
	cbr	temp,MMCST_NOINIT	;Yes, clear 'NO INIT' status
	sts	mmcStat,temp
mmci_lex:

.if MMC_DEBUG
	printnewline
	printstring " CT: "
	push	temp
	lds temp,mmcCardType
	rcall	printhex
	pop	temp
	printstring " InitRes: "
	rcall	printhex
	printstring " "
.endif

	spi_disable
	ret
	
	
;--------------------------------------------------------------
; Read sector 
; 	z:	Pointer to the data buffer to store read data
;       yh..xl:	Start sector number (LBA)

mmcReadSect:
.if MMC_DEBUG > 1
	printnewline
	printstring	"mmcRdSect "
.endif
	ldiw	z,hostbuf		;for now

	lds	_tmp0,mmcStat
	ldi	temp,RES_NOTRDY
	sbrc	_tmp0,MMCST_NOINIT
	 ret
	
	spi_clkfast
	lds	temp,mmcCardType
	sbrs	temp,log2(CT_BLOCK)
	 rcall	mul_yx_512		;Convert to byte address  (*512)

	ldi	temp2,CMD17
	rcall	mmcCmd
	ldi	temp2,RES_ERROR
	brne	mmc_rdex
	
; Receive a data packet from MMC

	ldiw	y,512			;Number of bytes to tranfer
	ldi	temp,200		;Wait for data packet in timeout of 200ms.
	sts	delay_timer1,temp
mmc_rcv_wl:
	rcall	spi_rcvr
	cp	temp,_255
	brne	mmc_rcv_start
	lds	_tmp0,delay_timer1
	cp	_tmp0,_0
	brne	mmc_rcv_wl
.if MMC_DEBUG > 1
	printstring	"TIMEOUT "
	rjmp	mmc_rcv_dbg1
.endif

mmc_rcv_start:
.if MMC_DEBUG > 1
	cpi	temp,0xFE		;If not valid data token, 
	breq	mmc_rcv_dbg1
	printstring	"Token: "
	rcall	printhex
	printstring	" "
mmc_rcv_dbg1:
.endif
	cpi	temp,0xFE		;If not valid data token, 
	brne	mmc_rdex
	
	rcall	spi_rcvr		;Shift in first byte.
.if MMC_DEBUG > 3
	printnewline
	rcall	printhex
	printstring	" "
.endif
	out	SPDR,_255		;Start shift in next byte.
mmc_rcv_rl:
	sbiw	yl,1
	breq	mmc_rcv_rle
	st	z+,temp
	spi_waitm
	in	temp,SPDR
.if MMC_DEBUG > 3
	rcall	printhex
	printstring	" "
.endif
	out	SPDR,_255
	rjmp	mmc_rcv_rl

mmc_rcv_rle:
	st	z+,temp			;Store last byte in buffer 
.if MMC_DEBUG > 3
	printnewline
.endif
	rcall	spi_wait		;      while SPI module shifts in crc part1.
	rcall	spi_rcvr		;Read second crc.
	
	ldi	temp2,RES_OK		;Return success
mmc_rdex:
	rcall	mmcDeselect
	spi_disable
	mov	temp,temp2
.if MMC_DEBUG > 1
	printstring	"RdSectRes: "
	rcall	printhex
	printstring	" "
.endif
	ret


;--------------------------------------------------------------
; Read word 
; TODO: Read Word to ZL,ZH at given ZL/ZH Offset
; Need for reading of single FAT16 Entrys without killing the
; Entrys in hostbuffer...
;
; 	in	zh,zl:		Pointer to Word within the Sector to read	
;   in	yh..xl:	Start sector number (LBA)
;	out	zh,zl	: word thats been read

mmcReadWord:
.if MMC_DEBUG > 1
	printnewline
	printstring	"mmcRdWord "
.endif
	lds		_tmp0,mmcStat
	ldi		temp,RES_NOTRDY
	sbrc	_tmp0,MMCST_NOINIT
	ret
	
	spi_clkfast
	lds		temp,mmcCardType
	sbrs	temp,log2(CT_BLOCK)
 	rcall	mul_yx_512		;Convert to byte address  (*512)

	ldi		temp2,CMD17
	rcall	mmcCmd
	ldi		temp2,RES_ERROR
	brne	mmc_rdexw
	
; Receive a data packet from MMC

	ldiw	y,512			;Number of bytes to tranfer
	ldi		temp,200		;Wait for data packet in timeout of 200ms.
	sts		delay_timer1,temp
mmc_rcvw_wl:
	rcall	spi_rcvr
	cp		temp,_255
	brne	mmc_rcvw_start
	lds		temp2,delay_timer1
	cpi		temp2,0
	brne	mmc_rcvw_wl
mmc_rcvw_start:
	cpi		temp,0xFE		;If not valid data token, 
	ldi		temp2,RES_ERROR
	brne	mmc_rdexw
	
	rcall	spi_rcvr		;Shift in first byte.
	out		SPDR,_255		;Start shift in next byte.
mmc_rcvw_rl:
	sbiw	yl,1
	breq	mmc_rcvw_rle
	cp		zl,_0
	cpc		zh,_0
	breq	mmc_rcvw_sto

	sbiw	zl,1
	spi_waitm
	in		temp,SPDR
	out		SPDR,_255
	rjmp	mmc_rcvw_rl

mmc_rcvw_sto:
	mov 	zl,temp
	spi_waitm
	in		temp,SPDR
	out		SPDR,_255
	mov 	zh,temp

mmc_rcvw_rl2:
	sbiw	yl,1
	breq	mmc_rcvw_rle
	spi_waitm
	in		temp,SPDR
	out		SPDR,_255
	rjmp	mmc_rcvw_rl2
mmc_rcvw_rle:
	rcall	spi_wait		;      while SPI module shifts in crc part1.
	rcall	spi_rcvr		;Read second crc.
	
	ldi		temp2,RES_OK		;Return success
mmc_rdexw:
	rcall	mmcDeselect
	spi_disable
	mov	temp,temp2
.if MMC_DEBUG > 1
	printstring	"RdWordRes: "
	rcall	printhex
	printstring	" "
.endif
	ret

;--------------------------------------------------------------
; Write sector 
; 	z:	Pointer to the data to be written
;       yh..xl:	Sector number (LBA)

mmcWriteSect:
.if MMC_DEBUG > 1
	printnewline
	printstring	"mmcWrSect "
.endif
	ldiw	z,hostbuf		;for now
	
	lds	_tmp0,mmcStat
	ldi	temp,RES_NOTRDY
	sbrc	_tmp0,MMCST_NOINIT
	ret
	
	spi_clkfast
	lds	temp,mmcCardType
	sbrs	temp,log2(CT_BLOCK)
	 rcall	mul_yx_512		;Convert to byte address  (*512)

	ldi	temp2,CMD24
	rcall	mmcCmd
	brne	mmc_wrexer
	
; Send a data packet to MMC

.if MMC_DEBUG > 2
;	printnewline
	printstring	"mmcXMIT "
.endif
	rcall	mmcWaitReady
	breq	mmc_wrexer

	ldi	temp,0xFE		;Data token
	out	SPDR,temp
	ldiw	y,512
mmc_x_loop:
	ld	temp,z+
	spi_waitm
	out	SPDR,temp
	sbiw	yl,1
	brne	mmc_x_loop
	
	rcall	spi_wait
	ldi	temp,0xFF		;dummy crc
	rcall	spi_xmit
	rcall	spi_xmit
	rcall	spi_rcvr
.if MMC_DEBUG > 2
	printstring	"XMITRes: "
	rcall	printhex
	printstring	" "
.endif
	andi	temp,0x1F		;If not accepted, return with error
	cpi	temp,0x05
	ldi	temp2,RES_OK		;Return success
	breq	mmc_wrex

mmc_wrexer:
	ldi	temp,RES_ERROR
mmc_wrex:
	rcall	mmcDeselect
	spi_disable
	mov	temp,temp2
.if MMC_DEBUG > 1
	printstring	"WrSectRes: "
	rcall	printhex
	printstring	" "
.endif
	ret

;--------------------------------------------------------------
; vim:set ts=8 noet nowrap

