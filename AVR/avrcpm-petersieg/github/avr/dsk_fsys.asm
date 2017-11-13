;    Filesystem functions for the Interaction with BIOS and Disks
;
;    Copyright (C) 2010 Frank Zoll
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
;    $Id: dsk_fsys.asm 167 2011-04-21 14:56:40Z leo $
;

.equ DSKSEL_DEBUG = 0

; ---------------- Defines for the Filesystem Interface -------

;*****************************************************
;*        Disk-Manager constants                     *
;*****************************************************

; Fields in the parttabl

	.equ	MAXDISKS  = 8			;Max number of Disks (partitions)
	.equ    PARTENTRY_SIZE = 9		;Size of a Partitiontableentry

	.equ	PTAB_TYPE	= 0
	.equ	PTAB_START	= 1
	.equ	PTAB_SIZE	= 5
	.equ	PTAB_SPT	= 7
	.equ	PTAB_BSH	= 8

	.equ	dskType_None	=   0 << 4
	.equ	dskType_CPM	=   1 << 4
	.equ	dskType_FAT	=   2 << 4
;	.equ	dskType_RAM	=   3 << 4
	.equ	dskType_MASK	= 0xf << 4

;*****************************************************
;*         CP/M to host disk constants               *
;*****************************************************
;	.equ	blksize = 1024			;CP/M allocation size
;	.equ	CPMSPT = 26			;

	.equ	HOSTSIZE = 512			;host disk sector size
	.equ	HOSTBLK  = HOSTSIZE/128		;CP/M sects/host buff
	.equ	SECMSK   = HOSTBLK-1		;sector mask
	.equ	SECSHF   = log2(HOSTBLK)	;sector shift

;*****************************************************
;*        BDOS constants on entry to write           *
;*****************************************************
	.equ	WRALL = 0		;write to allocated
	.equ	WRDIR = 1		;write to directory
	.equ	WRUAL = 2		;write to unallocated
	.equ	WRTMSK= 3		;write type mask

	.equ	READ_FUNC  = 7
	.equ	WRITE_FUNC = 6
	.equ	BOOT_FUNC  = 5
	.equ	HOME_FUNC  = 4

;----------------------------------------------- Start of Data Segment

	.dseg

; The following 3 variables are copied from DRAM.
; Don't change order.

biosdrvtbl:	.byte	2	;
biosdirbuf:	.byte	2	;
biosenddat:	.byte	2	;

ndisks:		.byte	1		;Number of CP/M disks

; The following 5 variables are accessed from 8080/z80 via the
; virtual port interface. Don't change order.

biospar_base:
bcbadr:		.byte	2		;adr of BiosControlBlock
seekdsk:	.byte	1		;seek disk number
seektrk:	.byte	2		;seek track number
seeksec:	.byte	2		;seek sector number
dmaadr:		.byte	2		;last dma address

hdrsize:	.byte	1		;Image header size (offset)
cpmspt:		.byte	1		;CP/M sectors per track
secpblk:	.byte	1		;sectors per block (alloc size)
unacnt:		.byte	1		;unalloc rec cnt
unadsk:		.byte	1		;last unalloc disk
unatrk:		.byte	2		;last unalloc track
unasec:		.byte	2		;last unalloc sector

erflag:		.byte	1		;error reporting
wrtype:		.byte	1		;write operation type

hostdsk:	.byte	1		;host disk number
hostlba:	.byte	2		;host sector number (relative to partition start)
hostparttbl:	.byte	PARTENTRY_SIZE*MAXDISKS ;host partition table (type, start sector, sector count)
hostparttbltop:
hostbuf:	.byte	HOSTSIZE 	;host buffer (from/to SD-card)


; ------------------------------- Start of Code Segment
	.cseg

;---------------------------------------------------------------------

.if DSKSEL_DEBUG

dbg_prdrvtbl:
	push	xh
	push	xl
	push	temp3
	push	temp2
	push	temp
	printnewline
	printstring "drvtbl ("
	lds	xl,biosdrvtbl
	lds	xh,biosdrvtbl+1
	movw	temp,x
	rcall	printhexw
	printstring "): "
	ldi	temp3,16
dbg_pcpel:
	rcall	dram_readw_pp
	rcall	printhexw
	printstring " "
	dec	temp3
	brne	dbg_pcpel
	pop	temp
	pop	temp2
	pop	temp3
	pop	xl
	pop	xh
	ret
		
dbg_print_biosd:
	printnewline
	lds	temp,bcbadr
	lds	temp2,bcbadr+1
	rcall	printhexw
	printstring "  "
	lds	temp,biosdrvtbl
	lds	temp2,biosdrvtbl+1
	rcall	printhexw
	printstring "  "
	lds	temp,biosdirbuf
	lds	temp2,biosdirbuf+1
	rcall	printhexw
	printstring "  "
	lds	temp,biosenddat
	lds	temp2,biosenddat+1
	rcall	printhexw
	printstring "  "
	ret
.endif
	
; ====================================================================
; ====================================================================
; Function: Get a Pointer to a Partitiontable entry
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : [w] z			Pointer to the Partitionentry
;              [r] zl			Number of Diskentry to Read
;	       [w] _tmp0		scratch
;	       [w] _tmp1		"
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_getpartentry:
	
	ldi	zh,PARTENTRY_SIZE
	mul	zh,zl
	ldiw	z,hostparttbl
	add	zl,_tmp0
	adc	zh,_tmp1
	ret

; ====================================================================
; ====================================================================
; Function: Virtual Port Interface
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
; Variables  : 
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================
	
dsk_param_getadr:
	ldiw	z,biospar_base
	add	zl,temp3
	adc	zh,_0
	ret

dsk_param_set:
	rcall	dsk_param_getadr
	st	z,temp
	cpi	temp3,bcbadr+1-biospar_base
	breq	SetBCB
	ret

dsk_param_get:
	cpi	temp3,seekdsk-biospar_base
	breq	dskDiskCheck
	rcall	dsk_param_getadr
	ld	temp,z
	ret

SetBCB:
	lds	xl,bcbadr
	mov	xh,temp
	ldiw	z,biosdrvtbl
	ldi	temp3,6
sbcb_l:
	rcall	dram_read_pp
	st	z+,temp
	dec	temp3
	brne	sbcb_l
	
;	rcall	dbg_print_biosd
	rcall	dpb_drvtblclear
;	rcall	dbg_prdrvtbl

	ret

; ====================================================================
; Function: Check if disk exists
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
; Variables  : 
;		return 0,  if selected disk not exist.
;		return !0, if disk exist
; --------------------------------------------------------------------
; Description:
; ====================================================================
dskDiskCheck:
	lds	temp2,ndisks
	lds	temp,seekdsk
.if	DSKSEL_DEBUG
	printnewline
	printstring "DiskCheck: "
	rcall	printhexw
.endif
	cpi	temp,RAMDISKNR
	brsh	dsk_dchrd			;maybe ramdisk

	tst	temp2				;0 disks?
	brne	dsk_dchpart1

; No disks yet, need to init

	rcall	dpb_drvtblclear
.if 0
	ldi	temp2,0x40
	ldi	temp,1
	lcall	clockput
.endif
	rcall	mgr_init_partitions		;disk chanched?
	push	temp
.if 0
	ldi	temp2,0x40
	ldi	temp,2
	lcall	clockput
;	sbrs	temp,7	
;	 rjmp	dsk_dchpart0

	lcall	mgr_prnt_parttbl
	printnewline
.endif

;	rcall	dbg_prdrvtbl
	pop	temp2
dsk_dchpart0:
	cbr	temp2,0x80
	lds	temp,seekdsk
	
; Check if selected disk # is less then # of disks.

dsk_dchpart1:
	cp	temp,temp2
	brsh	dsk_dch_err

.if	DSKSEL_DEBUG
	printnewline
	printstring "Select: "
	rcall	printhex
.endif
	rcall	dpb_drvtbl_entry_get
	or	temp,temp2		;if !0, drive is allready initialized
	brne	dsk_dchend
	lds	temp3,seekdsk
	mov	temp,temp3
	rcall	dpb_biosdph_get
dsk_dchend:

.if	DSKSEL_DEBUG
	push	temp
	lds	temp,seekdsk
	rcall	dpb_drvtbl_entry_get

	printstring ", "
	rcall	printhexw
	pop	temp
.endif	

	ret

dsk_dch_err:
	ldi	temp,0			;error return
	ret	
	
;	Check RAMDISK

dsk_dchrd:
#if RAMDISKCNT
	cpi	temp,RAMDISKNR+RAMDISKCNT
	brsh	dsk_dchrd_err

	ldi	temp,0xff		;return ok
	ret
#endif
dsk_dchrd_err:
	ldi	temp,0			;error return
	ret


; ====================================================================
; Function: Return status of last disk i/o function
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
; Variables  : 
; --------------------------------------------------------------------
; Description:
; ====================================================================
dskErrorRet:
	lds	temp,erflag
	ret

		
; ====================================================================
; ====================================================================


	.dseg
tmpdpb:	.byte	16

	.cseg

str_CPM_Disk:
	.db	10,"<CPM_Disk>",0

;  DPBs for varios fixed formats
;  dpb data starts at 2. byte

dpbdat_avrcpm: 			;(dpb243)
	.db 0x00,0x1A		;sector offset, low(spt)
	.db 0x00,0x03		;high (spt), block shift 
	.db 0x07,0x00		;bock mask, extent mask 
	.db 0xF2,0x00		;disk size - 1,
	.db 0x3F,0x00		;dir max 
	.db 0xC0,0x00		;alloc0, alloc1
	.db 0x10,0x00		;chk size
	.db 0x02,0x00		;offset

dpbdat_myz80: 			;
	.db 0x02,0x80		;sector offset, low(spt)
	.db 0x00,0x05		;high (spt), block shift 
	.db 0x1F,0x01		;bock mask, extent mask 
	.db 0xFF,0x07		;disk size - 1,
	.db 0xFF,0x03		;dir max 
	.db 0xFF,0x00		;alloc0, alloc1
	.db 0x00,0x01		;chk size
	.db 0x00,0x00		;offset

dpbdat_simhd: 			;
	.db 0x00,0x20		;sector offset, low(spt)
	.db 0x00,0x05		;high (spt), block shift 
	.db 0x1F,0x01		;bock mask, extent mask 
	.db 0xF9,0x07		;disk size - 1,
	.db 0xFF,0x03		;dir max 
	.db 0xFF,0x00		;alloc0, alloc1
	.db 0x00,0x01		;chk size
	.db 0x06,0x00		;offset

#if 0
;rd1016	
	.db 0x20,0x00		;spt
	.db 0x04,0x0F		;block shift, bock mask
	.db 0x00,0xFB		;extent mask, low(disk size -1), 
	.db 0x01,0xBF		;high(disk size -1), low(dir max)
	.db 0x00,0xE0		;high(dir max), alloc0
	.db 0x00,0x30		;alloc1, low(chk size)
	.db 0x00,0x02		;high(chk size), low(offset)
	.db 0x00,0x00		;high(offset), fill
;rd9192s
	.db 0x20,0x00		;spt
	.db 0x05,0x1F		;block shift, bock mask
	.db 0x01,0xFD		;extent mask, low(disk size -1), 
	.db 0x07,0xFF		;high(disk size -1), low(dir max)
	.db 0x01,0xF0		;high(dir max), alloc0
	.db 0x00,0x80		;alloc1, low(chk size)
	.db 0x00,0x02		;high(chk size), low(offset)
	.db 0x00,0x00		;high(offset), fill
#endif


; Copy the dpb data from flash memory, pointed to by Z, to temp ram.

dpb_copy_p:
	push	yh
	push	yl
	ldi	temp2,16
	ldiw	y,tmpdpb
cpydpb_pl:
	lpm	temp,z+
	st	y+,temp
	dec	temp2
	brne	cpydpb_pl
	pop	yl
	pop	yh
	ret

; Copy the dpb data, pointed to by Z to temp ram.

dpb_copy:	
	st	y+,temp
	ldi	temp2,15
cpydpb_l:
	ld	temp,z+
	st	y+,temp
	dec	temp2
	brne	cpydpb_l
	ret


; String compare (z, y), one z-string in flash.

strcmp_p:
	lpm	_tmp0,z+
	tst	_tmp0
	breq	strcmp_pex

	ld	temp, y+
	lpm	_tmp0, z+
	sub	temp,_tmp0
	brne	strcmp_pex
	tst	_tmp0
	brne	strcmp_p
strcmp_pex:
	ret

; String compare (x, y, temp2). Max temp2 bytes are compared.

strncmp_p:
	subi	temp2,1
	brcs	strncmp_peq
	ld	temp,y+
	lpm	_tmp0, z+
	sub	temp,_tmp0
	brne	strncmp_pex
	tst	_tmp0
	brne	strncmp_p
strncmp_peq:
	sub	temp,temp
strncmp_pex:
	ret

; ====================================================================
; Function: get drive table entry pointer for drive # in temp
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
; Variables  : 
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================

dpb_drvtbl_entry_p:

	ldsw	x,biosdrvtbl
	lsl	temp			;drive #
	add	xl,temp
	adc	xh,_0
	ret
	

; ====================================================================
; Function: get drive table entry for drive # in temp
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
; Variables  : 
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================

dpb_drvtbl_entry_get:

	rcall	dpb_drvtbl_entry_p
	ljmp	dram_readw_pp		


; ====================================================================
; Function: Clear drive table (entries 0 to 7)
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
; Variables  : 
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================

;

dpb_drvtblclear:
	ldsw	x,biosdrvtbl
	sbiw	x,0
	breq	dpb_drvi_ex
	
dpb_drvi_1:
	ldi	temp3,8
dpb_drvi_lp:
	ldi	temp,0
	ldi	temp2,0
	rcall	dram_writew_pp
	dec	temp3
	brne	dpb_drvi_lp
	
	lds	temp,biosenddat
	lds	temp2,biosenddat+1
	cp	temp,_0
	cpc	temp2,_0
	breq	dpb_drvi_ex

	rcall	heap_init
dpb_drvi_ex:
	clr	temp
	ret

; ====================================================================
; Function: Test disk format: avrcpmhd
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp		drive #
;              
; --------------------------------------------------------------------
; Description: Not implemented yet.
; ====================================================================

dsk_tst_avrcpmhd:
	clr	temp		; Test, return 'not found' for now.
	ret


; ====================================================================
; Function: Test disk format: YAZE
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp		drive #
;              
; --------------------------------------------------------------------
; Description: From the YAZE Doc:
;
; The new disk header occupies the first 128 BYTES of the file and has the
; new format:
;
;          0 -   9    <CPM_Disk>
;         10 -  15    a null-terminated ascii comment (may be empty)
;     new 16          version (0 = yaze-1.06/1.10, 1 = yaze-ag-2.xx)
;         17 -  31    a null-terminated ascii comment (may be empty)
;         32 -  33    sectors per track
;         34          block shift factor
;         35          block mask
;         36          extent mask
;         37 -  38    disk size max
;         39 -  40    directory max
;         41          al0
;         42          al1
;         43 -  44    check size (always zero)
;         45 -  46    track offset
;     new 47          psh (used if version=1 and CP/M 3.1 is running)
;     new 48          phm ( "   "    "   "    "   "   "   "    "    )
;         49 - 127    unused (zeros)
; ====================================================================


dsk_tst_yaze:

	ldiw	y,hostbuf
	ldiw	z,str_CPM_Disk*2
	lpm	temp2,z+		; get length
	rcall	strncmp_p
	brne	dsk_tyze_not
	
	ldiw	z,hostbuf+32
	ldiw	y,tmpdpb
	ldi	temp,1			;1 sector header size
	rcall	dpb_copy	

	ori	temp,0xff
	ret

dsk_tyze_not:
	clr	temp			;Not a YAZE disk image.
	ret
	
; ====================================================================
; Function: Test disk format: MyZ80
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp		drive #
;              
; --------------------------------------------------------------------
; Description:	Test, if first 2 Sectors are filled with 0xE5,
; 		and Size = 8192KB + 256Bytes.
; ====================================================================
dsk_tst_myz80:

	mov	zl,temp3
	rcall	dsk_getpartentry	;get partition entry
	ldd	temp,z+PTAB_SIZE
	ldd	temp2,z+PTAB_SIZE+1	;check, if size is 16385 phys. sectors
	cpi	temp,low(16385)
	ldi	temp,high(16385)
	cpc	temp2,temp
	brne	dsk_tmyz80_not		;wrong size
	
	ldiw	z,hostbuf
	ldi	temp2,0
	
dsk_tmyz80_loop:
	ld	temp,z+
	cpi	temp,0xE5
	brne	dsk_tmyz80_not
	dec	temp2
	brne	dsk_tmyz80_loop

	ori	temp,0xff
	ret

dsk_tmyz80_not:
	clr	temp			;Not a MyZ80 hard disk image.
	ret
	
; ====================================================================
; Function: Test disk format: simhd, simh altair 8800 hard disk format
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp		drive #
;              
; --------------------------------------------------------------------
; Description:	Test, if Size = 8192 KB and 
;		first 6 tracks are filled with 0xE5.
;		Actually, only the first phys. sector is tested, since 
;		the other 47 sectors are not in memory at this time.
; ====================================================================
dsk_tst_simhd:

	mov	zl,temp3
	rcall	dsk_getpartentry	;get partition entry
	ldd	temp,z+PTAB_SIZE
	ldd	temp2,z+PTAB_SIZE+1	;check, if size is 16384 phys. sectors
	cpi	temp,low(16384)
	ldi	temp,high(16384)
	cpc	temp2,temp
	brne	dsk_tsimhd_not		;wrong size
	
	ldiw	y,hostbuf+128-10
	ldiw	z,str_CPM_Disk*2
	lpm	temp2,z+		; get length
	rcall	strncmp_p
	breq	dsk_tsimhd_found
	
	ldiw	z,hostbuf
	ldi	temp2,high(512)
	clr	_tmp0			;low(512)
dsk_tsimhd_loop:
	ld	temp,z+
	cpi	temp,0xE5
	brne	dsk_tsimhd_not
	dec	_tmp0
	brne	dsk_tsimhd_loop
	dec	temp2
	brne	dsk_tsimhd_loop

dsk_tsimhd_found:
	ori	temp,0xff
	ret

dsk_tsimhd_not:
	clr	temp			;Not a simhd hard disk image.
	ret
	
; ====================================================================
; Function: 
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp3		drive #
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================

dsk_format_get:

; Get first sector (512 byte) of current drive into hostbuf.

	push	temp3
	ldi	temp,0			
	ldi	temp2,0			;
	rcall	dsk_readhost_lba
	
; Test for variable format avrcpmhd.

	rcall	dsk_tst_avrcpmhd
	brne	dsk_imgt_done

; Test for YAZE formats.

	rcall	dsk_tst_yaze
	brne	dsk_imgt_done

; Test for simhd format.

	rcall	dsk_tst_simhd
	ldiw	z,dpbdat_simhd*2
	brne	dsk_imgt_fixed

; Test for MyZ80 format.

	rcall	dsk_tst_myz80
	ldiw	z,dpbdat_myz80*2
	brne	dsk_imgt_fixed
	
; No special image found. Use avrcpm.

	ldiw	z,dpbdat_avrcpm*2

dsk_imgt_fixed:
	rcall	dpb_copy_p
	ori	temp,0xff
dsk_imgt_done:
	pop temp3
	ret	

; ====================================================================
; Function: Add CP/M image format data to partition table data
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp3		drive #
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================

dpb_imgdata_get:

; Test for known CP/M formats 

	rcall	dsk_format_get	
	breq	dpb_imgd_err		;no known format detected

;
	mov	zl,temp3
	rcall	dsk_getpartentry	;get partition entry
	ldiw	y,tmpdpb
	
;	std	y+12,_0			;Test: set check size to 0
;	std	y+13,_0
	
	ldd	temp,y+0
	andi	temp,~dskType_MASK
	ldd	temp2,z+PTAB_TYPE
	andi	temp2,dskType_MASK
	or	temp,temp2
	std	z+PTAB_TYPE,temp
	ldd	temp,y+1
	std	z+PTAB_SPT,temp
	ldd	temp,y+2
	tst	temp			;more then 256 sectors per track?
	brne	dsk_imgprp_err		;todo: support 16 bit sector numbers
	ldd	temp2,y+3
	andi	temp2,0x0f
	swap	temp2
	std	z+PTAB_BSH,temp2
	
	ori	temp,255
	ret

dsk_imgprp_err:
	printnewline
	ldi	temp,'A'
	add	temp,temp3
	call	uartputc
	printstring ": Format not supported: Too much sectors per track! "
	printnewline

dpb_imgd_err:
	clr	temp
	ret

; ====================================================================
; Function: 
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : temp		drive #
;              
;			return !0 if ok
;				0 on error
; --------------------------------------------------------------------
; Description: Init CP/M data structures 
;
;         -----------------------------------------------------------------
; DPH:    |  XLT  |       |       |       |DIRBUF |  DPB  |  CSV  |  ALV  |
;         -----------------------------------------------------------------
;offset:      0       2       4       6       8      10      12      14
;
;         -------------------------------------------------------------
; DPB:    |  SPT  |BSH|BLM|EXM|  DSM  |  DRM  |AL0|AL1|  CKS  |  OFF  |
;         -------------------------------------------------------------
;offset:      0                   5       7              11      13    
; ====================================================================

dpb_biosdph_get:
	mov	temp3,temp		;save disk #

	rcall	dsk_format_get	
	brne	dpb_di_0
	rjmp	dpb_di_err

dpb_di_0:

; get mem for DPH 

	ldi	temp, low (16)
	ldi	temp2,high(16)
	rcall	heap_get		;returns ptr to dph mem
	brne	dpb_di_1
	rjmp	dpb_di_err		;
dpb_di_1:
	movw	x,temp
	movw	y,temp			;save dph pointer
	ldi	temp,0
	ldi	temp2,0
	rcall	dram_writew_pp		;XLT
	adiw	x,6
	lds	temp,biosdirbuf
	lds	temp2,biosdirbuf+1
	rcall	dram_writew_pp		;DIRBUF

; get mem for DPB	
	
	ldi	temp, low (15)
	ldi	temp2,high(15)
	rcall	heap_get
	breq	dpb_di_err_p1
	movw	x,temp
	
	ldiw	z,tmpdpb+1		;skip sector offset byte
dpb_dicpl:
	ld	temp,z+
	rcall	dram_write_pp
	cpi	zl,low(tmpdpb + 16)
	brne	dpb_dicpl
	sbiw	x,15
	movw	temp,x
	movw	x,y
	adiw	x,10
	rcall	dram_writew_pp		;DPB
	
; get mem for dir check vector

	lds	temp,tmpdpb+12		;cks
	lds	temp2,tmpdpb+12+1
	cp	temp,_0
	cpc	temp2,_0
	breq	dpb_dicks0
	rcall	heap_get
	breq	dpb_di_err_p1
dpb_dicks0:
	rcall	dram_writew_pp		;CSV

; get mem for alloc vector

	lds	temp,tmpdpb+6		;dsm
	lds	temp2,tmpdpb+6+1
	subi	temp, low (-8)
	sbci	temp2,high(-8)
	lsr	temp2
	ror	temp
	lsr	temp2
	ror	temp
	lsr	temp2
	ror	temp			;(dsm+1+7)/8
	rcall	heap_get
	breq	dpb_di_err_p1
	rcall	dram_writew_pp		;ALV

; success, insert DPH into drvtbl

	mov	temp,temp3
	rcall	dpb_drvtbl_entry_p
	movw	temp,y
	rcall	dram_writew_pp	

	ori	temp,0xff		;return ok
	ret

; error, free mem

dpb_di_err_p1:
	movw	temp,y
	rcall	heap_release
dpb_di_err:
	eor	temp,temp		;return 0 (+ Z-flag)
	ret
	
; ====================================================================
; Function: 
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : 
;
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_setdrvparam:
	ldd	temp2,z+PTAB_TYPE
	andi	temp2,~dskType_MASK	;Lower nibble is image header size
	sts	hdrsize,temp2
	ldd	temp2,z+PTAB_SPT
	sts	cpmspt,temp2
	ldd	temp2,z+PTAB_BSH
	swap	temp2
	andi	temp2,0x0f
	clr	_tmp0
	inc	_tmp0
dsk_sdrvpl:
	lsl	_tmp0
	dec	temp2
	brne	dsk_sdrvpl
	sts	secpblk,_tmp0
	ret
	

; ====================================================================
; Function: Does a Disk interaction
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
dskDoIt:
.if DISK_DEBUG
	push 	temp
	sbrc	temp,READ_FUNC
	rjmp	dskdbgr
	sbrc	temp,WRITE_FUNC
	rjmp	dskdbgw
	rjmp	dskdbge

dskdbgr:
	printnewline
	printstring "Disk read:  "
	rjmp	dskdbg1
dskdbgw:
	printnewline
	printstring "Disk write: "
dskdbg1:
	lds		temp,seekdsk
	subi	temp,-('A')
	rcall	uartputc
	printstring ": track "
	lds 	temp2,seektrk+1
	lds 	temp,seektrk
	rcall 	printhexw
	printstring ", sector "
	lds 	temp,seeksec
	rcall 	printhex
	printstring ", dma-addr "
	lds 	temp2,dmaadr+1
	lds 	temp,dmaadr
	rcall 	printhexw
	pop		temp
	push	temp
	sbrs	temp,WRITE_FUNC
	rjmp	dskdbge
	printstring " wrtype "
	andi	temp,3
	rcall printhex
dskdbge:
	pop temp
.endif
	;See what has to be done.
	sbrc	temp,READ_FUNC
	rjmp	dsk_read
	sbrc	temp,WRITE_FUNC
	rjmp	dsk_write
	sbrc	temp,HOME_FUNC
	rjmp	dsk_home
	sbrc	temp,BOOT_FUNC
	rjmp	dsk_boot

	printstring "DISK I/O: Invalid Function code: "
	rcall	printhex
	rjmp	haltinv

dsk_boot:
	sts	ndisks,_0	;no active partitions
dsk_inval_hostbuf:
	cbi	flags,hostact	;host buffer inactive
	sts	unacnt,_0	;clear unalloc count
	ret

dsk_home:
	sbis	flags,hostwrt	;check for pending write
	cbi	flags,hostact	;clear host active flag
	ret



; ====================================================================
; Function: Does a Disk read operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : in: temp			
; Variables  : [r] seekdsk		Number of Disk to Read
;			   [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_read:
	sts	erflag,_0
	sbi	flags,readop		; Set read operation flag

	;RAM disk?
	lds     zl,seekdsk
#if RAMDISKCNT
	cpi	zl,RAMDISKNR
	brlt	PC+2
	 rjmp	rdsk_read
#endif
	rcall 	dsk_getpartentry	; Get Paritiontableentry
	ld    	temp,z			; Get Partitiontype
	andi	temp,dskType_MASK

; Isn't it a Disk ?
	cpi		temp,dskType_None
	brne	PC+2
	rjmp	dsk_read_err

; It must be a FAT16-Imagefile or CP/M Partition.

	rcall	dsk_setdrvparam		;todo: do this only if needed (disk change)

	sts	unacnt,_0
	sbi	flags,rsflag		;must read data
	ldi	temp,WRUAL		;write type
	sts	wrtype,temp		;treat as unalloc

	rjmp	dsk_rwoper		;to perform the read

dsk_read_err:
	ret

; ====================================================================
; Function: Does a Disk write operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : in: temp		Write type	
; Variables  : [r] seekdsk		Number of Disk to Read
;			   [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_write:
	;write the selected sector
	sts	erflag,_0
	cbi	flags,readop		; not a read operation
	;RAM disk?
	lds     zl,seekdsk
#if RAMDISKCNT
	cpi	zl,RAMDISKNR
	brlt	PC+2
	 rjmp	rdsk_write
#endif
	rcall 	dsk_getpartentry	; Get Paritiontableentry
	ld    	temp2,z			; Get Partitiontype
	andi	temp2,dskType_MASK

; Isn't it a Disk ?
	cpi	temp2,dskType_None
	brne	PC+2
	rjmp	dsk_write_err


; It must be a FAT16-Imagefile or CP/M Partition.

	rcall	dsk_setdrvparam		;todo: do this only if needed (disk change)

	andi	temp,WRTMSK
	sts	wrtype,temp		;save write type

	cpi	temp,WRUAL		;write unallocated?
	brne	dsk_chkuna		;check for unalloc

;	write to unallocated, set parameters
	lds	temp,secpblk		;next unalloc recs (blocksize/128)
	sts	unacnt,temp
	lds	temp,seekdsk		;disk to seek
	sts	unadsk,temp		;unadsk = sekdsk
	lds	temp,seektrk
	sts	unatrk,temp		;unatrk = sectrk
	lds	temp,seektrk+1
	sts	unatrk+1,temp		;unatrk = sectrk
	lds	temp,seeksec
	sts	unasec,temp		;unasec = seksec
;
dsk_chkuna:
	;check for write to unallocated sector
	lds	temp,unacnt		;any unalloc remain?
	tst	temp
	breq	dsk_alloc		;skip if not

;	more unallocated records remain
	dec	temp			;unacnt = unacnt-1
	sts	unacnt,temp
	lds	temp,seekdsk		;same disk?
	lds	temp2,unadsk
	cp	temp,temp2		;seekdsk = unadsk?
	brne	dsk_alloc		;skip if not

;	disks are the same
	lds	temp,unatrk
	lds	temp2,unatrk+1
	lds	temp3,seektrk
	lds	temp4,seektrk+1
	cp	temp,temp3		;seektrk = unatrk?
	cpc	temp2,temp4
	brne	dsk_alloc		;skip if not

;	tracks are the same
	lds	temp,seeksec		;same sector?
	lds	temp2,unasec
	cp	temp,temp2		;seeksec = unasec?
	brne	dsk_alloc		;skip if not

;	match, move to next sector for future ref
	inc	temp2		;unasec = unasec+1
	sts	unasec,temp2
	lds	_tmp0,cpmspt
	cp	temp2,_tmp0		;end of track? (count CP/M sectors)
	brlo	dsk_noovf		;skip if no overflow

;	overflow to next track
	sts	unasec,_0		;unasec = 0
	lds	temp,unatrk
	lds	temp2,unatrk+1
	subi	temp, low(-1)		;unatrk = unatrk+1
	sbci	temp2,high(-1)
	sts	unatrk,temp
	sts	unatrk+1,temp2
;
dsk_noovf:
	cbi	flags,rsflag		;rsflag = 0
	rjmp	dsk_rwoper		;to perform the write
;
dsk_alloc:
	;not an unallocated record, requires pre-read
	sts	unacnt,_0		;unacnt = 0
	sbi	flags,rsflag		;rsflag = 1
	rjmp	dsk_rwoper

dsk_write_err:
	ret

; ====================================================================
; Function: Does a Disk read/write operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seekdsk		Number of Disk to Read
;			   [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_rwoper:
	;enter here to perform the read/write
.if DISK_DEBUG
	printstring ", flags: "
	in	temp,flags
	rcall	printhex
.endif
	;Convert track/sector to an LBA address (in 128byte blocks)

	lds	xl,seeksec		;
	ldi	xh,0			;
	ldi	yl,0			;
	lds	temp,hdrsize		;add image header size
	add	xl,temp			;
	adc	xh,_0			;
	lds	temp3,seektrk		;
	lds	temp4,seektrk+1		;
	lds	temp,cpmspt		;
	mul	temp3,temp		;
	add	xl,r0			;
	adc	xh,r1			;
	mul	temp4,temp		;
	add	xh,r0			;yl:xh:xl := sec + trk * SectorsPerTrack
	adc	yl,r1			;

	mov	temp,xl
	andi	temp,SECMSK		;mask buffer number
	push	temp			;save for later

	;Convert from CP/M LBA blocks to host LBA blocks
	ldi	temp,SECSHF
dsk_sh1:
	lsr	yl
	ror	xh
	ror	xl
	dec	temp
	brne	dsk_sh1
					;todo: yl should be 0 here. 
					;xh:xl = host block to seek
	movw	temp,x
	lds	temp3,seekdsk
	rcall	dsk_rw_hostbuf

	;copy data to or from buffer
	ldiw	z,hostbuf
	ldi	temp,128
	pop	temp2			;get buffer number (which part of hostbuf)
	mul	temp2,temp
	add	zl,r0			;offset in hostbuf
	adc	zh,r1

.if DISK_DEBUG > 2
	movw	temp,r0
	printstring "; host buf adr: "
	rcall 	printhexw
.endif

	lds	xl,dmaadr
	lds	xh,dmaadr+1
	ldi	temp3,128		;length of move
	sbic	flags,readop		;which way?
	rjmp	dsk_rmove		;skip if read

;	mark write operation
	sbi	flags,hostwrt		;hostwrt = 1
dsk_wmove:
	mem_read
	st	z+,temp
	adiw	xl,1
	dec	temp3
	brne 	dsk_wmove
	rjmp	dsk_rwmfin
	
dsk_rmove:
	ld	temp,z+
	mem_write
	adiw	xl,1
	dec	temp3
	brne	dsk_rmove
dsk_rwmfin:
;	data has been moved to/from host buffer
	lds	temp,wrtype		;write type
	cpi	temp,WRDIR		;to directory?
	breq	dsk_wdir
	ret				;no further processing
dsk_wdir:
;	clear host buffer for directory write
	lds	temp,erflag
	tst	temp			;errors?
	breq	dsk_wdir1
	ret				;skip if so
dsk_wdir1:
	rcall	dsk_writehost		;clear host buff
	cbi	flags,hostwrt		;buffer written
	ret

; ====================================================================
; Function: 
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  :	temp2:temp	block to read (lba)
;		temp3		disk #
;              
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_readhost_lba:
#if 0
	printnewline
	printstring "readhst lba"
#endif
	sbi	flags,rsflag		;must read data
	rcall	dsk_rw_hostbuf
	lds	temp,erflag		;returns 0, if ok
	tst	temp
	ret

; ====================================================================
; Function: Get physical disk sector in hostbuf.
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  :	temp2:temp	host block to read/write (lba)
;		temp3		disk #
;              
; --------------------------------------------------------------------
; Description: 
; ====================================================================
dsk_rw_hostbuf:
					;xh:xl = host block to seek
	sts	erflag,_0		;no errors (yet)

;	active host sector?
	in	_tmp0,flags		;host active flag
	sbi	flags,hostact		;always becomes 1
	sbrs	_tmp0,hostact		;was it already?
	rjmp	dsk_filhst		;fill host if not

;	host buffer active, same as seek buffer?
	lds	_tmp0,hostdsk		;same disk?
	cp	temp3,_tmp0		;seekdsk = hostdsk?
	brne	dsk_nomatch

;	same disk, same block?
	lds	_tmp0,hostlba
	lds	_tmp1,hostlba+1
	cp	temp,_tmp0
	cpc	temp2,_tmp1
	breq	dsk_match

dsk_nomatch:
	;proper disk, but not correct sector
	sbis	flags,hostwrt		;host written?
	rjmp	dsk_filhst
	push	temp3
	push	temp2
	push	temp
	rcall	dsk_writehost		;clear host buff
	pop	temp
	pop	temp2
	pop	temp3

dsk_filhst:
	;may have to fill the host buffer
	sts	hostlba,temp
	sts	hostlba+1,temp2
	sts	hostdsk,temp3

	sbic	flags,rsflag		;need to read?
	rcall	dsk_readhost		;yes, if 1
	cbi	flags,hostwrt		;no pending write

dsk_match:
	ret

; ====================================================================
; Function: Does a Disk write operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seekdsk		Number of Disk to Read
;			   [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_writehost:
	lds	zl,hostdsk
	rcall	dsk_getpartentry
	ld	temp,z
	andi	temp,dskType_MASK

#if FAT16_SUPPORT
; Is it a FAT16 Diskimage ?
	cpi	temp,dskType_FAT
	brne	PC+2
	rjmp	fat_writehost
#endif

; Is it a CP/M Partition ?
	cpi	temp,dskType_CPM
	brne	PC+2
	rjmp	cpm_writehost
; Disktype not supported -> Return 
	ret

; ====================================================================
; Function: Does a Disk read operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seekdsk		Number of Disk to Read
;			   [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
dsk_readhost:

#if 0
	printnewline
	printstring "readhost"
	ldiw	z,biosdrvtbl
	rcall	dbg_hexdump_line
	adiw	z,16
	rcall	dbg_hexdump_line
#endif
			
	lds    zl,hostdsk
	rcall  dsk_getpartentry
	ld     temp,z
	andi	temp,dskType_MASK

#if FAT16_SUPPORT
; Is it a FAT16 Diskimage ?
	cpi	temp,dskType_FAT
	brne	PC+2
	rjmp	fat_readhost
#endif

; Is it a CP/M Partition ?
	cpi	temp,dskType_CPM
	brne	PC+2
	rjmp	cpm_readhost
; Disktype not supported -> Return 
	ret


; vim:set ts=8 noet nowrap

