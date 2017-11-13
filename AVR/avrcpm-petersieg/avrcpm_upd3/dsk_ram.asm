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
;    $Id: dsk_ram.asm 139 2010-10-08 10:50:03Z leo $
;

#ifndef RAMDSK_SUPPORT
	#define RAMDSK_SUPPORT 0	
#endif

#if RAMDSK_SUPPORT

;-------------------------------------- Defines for RAMDISK Structures

;----------------------------------------------- Start of Data Segment

	.dseg

rdskbuf:	.byte	128			; Buffer for RAM-Disk interaktions

; ---------------------------------------------- Start of Code Segment
	.cseg

; ====================================================================
; Function: Calculate an sets the adress of Sector within the RAMDISK
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  :  none
; Variables  :  [r] seeksec		Sector to read
;               [r] seektrk		Track  to read
;				[w] temp3		Number of Bytes per Sector (128)		
; --------------------------------------------------------------------
; Description:
; ====================================================================


rdsk_adr:
	ldi	xl,0
	lds	xh,seeksec
	lds	temp2,seektrk
	
	lsr	xh
	ror	xl				;Col 0..7
	
	mov	 temp,temp2
	andi temp,0x0f
	swap temp
	or	 xh,temp		;Row  0..7
	
	ldiw	z,rdskbuf
	ldi		temp3,128
	DRAM_SETADDR xh, ~0,(1<<ram_ras), ~0,(1<<ram_a8)|(1<<ram_oe)
	cbi	P_RAS,ram_ras

.if DISK_DEBUG > 1
	mov	temp,xh
	rcall	printhex
	printstring " "
	mov	temp,xl
	rcall	printhex
	printstring " "
.endif
	ret

; ====================================================================
; Function: Does a read opperation on a RAMDISK
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  :  none
; Variables  :  [r] seeksec		Sector to read
;               [r] seektrk		Track  to read
;				[r] flags		RW operation Flags
;				[w] erflag		Error Status of the operation
; --------------------------------------------------------------------
; Description:
; ====================================================================


rdsk_read:

.if DISK_DEBUG > 1
	printnewline
	printstring "rd-adr: "
.endif
	rcall	rdsk_adr

rdsk_rdl:
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_oe)), (1<<ram_a8)
	cbi	P_CAS,ram_cas
	cbi	P_A8,ram_a8
	inc	xl
	dram_wait DRAM_WAITSTATES	;
	in	temp,P_DQ-2		; PIN
	sbi	P_CAS,ram_cas

	cbi	P_CAS,ram_cas
	andi	temp,0x0f
	swap	temp
	dram_wait DRAM_WAITSTATES	;
	in	temp2,P_DQ-2		; PIN
	andi	temp2,0x0f
	or	temp,temp2

	sbi	P_OE,ram_oe
	sbi	P_CAS,ram_cas
	dec	temp3
	st	z+,temp
	brne	rdsk_rdl

	sbi	P_RAS,ram_ras
	ldiw	z,rdskbuf
	lds	xl,dmaadr
	lds	xh,dmaadr+1
	ldi	temp3,128	
rdsk_rdstl:
	ld	temp,z+
	mem_write
	adiw	x,1
	dec	temp3
	brne	rdsk_rdstl
	ret
	
; ====================================================================
; Function: Does a write opperation on a RAMDISK
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  :  none
; Variables  :  [r] seeksec		Sector to read
;               [r] seektrk		Track  to read
;				[r] flags		RW operation Flags
;				[w] erflag		Error Status of the operation
; --------------------------------------------------------------------
; Description:
; ====================================================================

rdsk_write:
.if DISK_DEBUG > 1
	printnewline
	printstring "wr-adr: "
.endif	
	lds	xl,dmaadr
	lds	xh,dmaadr+1
	ldiw	z,rdskbuf
	ldi	temp3,128	
rdsk_wrldl:
	mem_read
	st	z+,temp
	adiw	x,1
	dec	temp3
	brne	rdsk_wrldl	

	ldi	temp2,RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out	DDRC,temp2
	rcall	rdsk_adr
rdsk_wrl:
	ld	temp,z+
	mov	temp2,temp
	andi	temp,RAM_DQ_MASK & ~(1<<ram_w)
	ori	temp,(1<<ram_cas)
	out	PORTC,temp
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_a8)),(1<<ram_oe)
	cbi	PORTC,ram_cas
	sbi	PORTD,ram_a8
	sbi	PORTC,ram_cas
	swap	temp2
	andi	temp2,RAM_DQ_MASK & ~(1<<ram_w)
	ori	temp2,(1<<ram_cas)
	out	PORTC,temp2
	cbi	PORTC,ram_cas
	inc	xl
	sbi	PORTC,ram_cas
	dec	temp3
	brne	rdsk_wrl

	sbi	P_RAS,ram_ras
	ldi	temp,~RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out	DDRC,temp
	out	PORTC,temp
	ret


rdsk_add_partition:
	ret


#else

rdsk_read:
	ret
rdsk_write:
	ret
rdsk_add_partition:
	ret

#endif
