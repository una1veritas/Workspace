;    Various functions for the Interaction with the CPM Filesystem
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
;    $Id: dsk_cpm.asm 162 2011-04-09 21:26:38Z leo $
;

#ifndef CPMDSK_SUPPORT
	#define CPMDSK_SUPPORT 1	
#endif

#if CPMDSK_SUPPORT


;----------------------------------------------- Start of Code Segment
	.cseg

; ====================================================================
; Function: Does a Disk write operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seekdsk		Number of Disk to Read
;			   [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; hostdsk = host disk #,  (partition #)
; hostlba = host block #, relative to partition start 
; Read/Write "hostsize" bytes to/from hostbuf
; --------------------------------------------------------------------
; Description:
; ====================================================================	

cpm_hostparam:
	lds	zl,hostdsk

.if HOSTRW_DEBUG
	mov     temp,zl
	subi	temp,-('A')
	lcall	uartputc
	printstring ": "
.endif

	rcall dsk_getpartentry		; get partition entry

	lds	temp,hostlba		; get sector to access
	lds	temp2,hostlba+1
;	lds	temp3,hostlba+2

.if HOSTRW_DEBUG
	printstring "lba: "
	clr	temp4
	lcall	print_ultoa
.endif

	ldd	xl,z+5			; get disksize
	ldd	xh,z+6
;	ldd	yl,z+7
	
	cp	temp,xl			; check given sector against disksize
	cpc	temp2,xh
;	cpc	temp3,yl
	brcs	cpm_hp1
	
.if HOSTRW_DEBUG
	printstring ", max: "
	push	temp4
	push	temp3
	push	temp2
	push	temp
	movw	temp,x
;	mov	temp3,yl
	clr	temp3
	clr	temp4
	lcall	print_ultoa
	pop	temp
	pop	temp2
	pop	temp3
	pop	temp4
	printstring " "
.endif
	
	clr	temp
	ret

cpm_hp1:
	ldd	xl,z+1			; get startsector of partition
	ldd	xh,z+2
	ldd	yl,z+3
	ldd	yh,z+4

	add	xl,temp			; add offset to startsector
	adc	xh,temp2
;	adc	yl,temp3
	adc	yl,_0
	adc	yh,_0

.if HOSTRW_DEBUG
	printstring ", abs:"
	push	temp4
	push	temp3
	push	temp2
	push	temp
	movw	temp,x
	movw	temp3,y
	lcall	print_ultoa
	pop	temp
	pop	temp2
	pop	temp3
	pop	temp4
	printstring " "
.endif

	ori	temp,255
cpm_hpex:
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

cpm_writehost:
.if HOSTRW_DEBUG
	printnewline
	printstring "host write "
.endif
	rcall	cpm_hostparam
	breq	cpm_rdwr_err
	
	rcall	mmcWriteSect
	tst	temp
	brne	cpm_rdwr_err
	
	rjmp	cpm_rdwr_ok
	

; ====================================================================
; Function: Does a Disk read operation
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seekdsk		Number of Disk to Read
;	       [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================
cpm_readhost:
.if HOSTRW_DEBUG
	printnewline
	printstring "host read  "
.endif

	rcall	cpm_hostparam
	breq	cpm_rdwr_err
	
	rcall	mmcReadSect
	tst	temp
	brne	cpm_rdwr_err

cpm_rdwr_ok:
	sts	erflag,_0
	ret

cpm_rdwr_err:
	sts	erflag,_255
	ret


; ====================================================================
; Function: Add's a CP/M Partition to the Partition table
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  : none
; Variables  : [r] seekdsk		Number of Disk to Read
;	       [r] seeksec		Sector to read
;              [r] seektrk		Track  to read
; --------------------------------------------------------------------
; Description:
; ====================================================================	
cpm_add_partition:
	
	ldi 	temp,dskType_CPM
	std	y+0,temp

	ldd	temp,z+PART_START
	std	y+1,temp
	ldd	temp,z+PART_START+1
	std	y+2,temp
	ldd	temp,z+PART_START+2
	std	y+3,temp
	ldd	temp,z+PART_START+3
	std	y+4,temp
	
	ldd	temp,z+PART_SIZE+2
	ldd	temp2,z+PART_SIZE+3
	or	temp,temp2		;part size larger than 65535 sectors?
	brne	cpm_add_prune

	ldd	temp,z+PART_SIZE
	std	y+5,temp
	ldd	temp,z+PART_SIZE+1
	std	y+6,temp
	rjmp	cpm_add_e

cpm_add_prune:
	std	y+5,_255
	std	y+6,_255

cpm_add_e:
	ret
		
#endif

; vim:set ts=8 noet nowrap

