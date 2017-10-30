;    Various Management functions for the Interaction with the File-
;    systems
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
;    $Id: dsk_mgr.asm 167 2011-04-21 14:56:40Z leo $
;


; -------------------------- Defines for the disk management Structures

; Partition Table Structures

#define PART_TYPE   4
#define PART_START  8
#define PART_SIZE  12

/*
 *  Partition table id 
 *	(see http://www.win.tue.nl/~aeb/partitions/partition_types-1.html)
 */	
#define PARTID1_FAT16 0x0E
#define PARTID2_FAT16 0x06
#define PARTID_CPM    0x52


; ------------------------------------------------ Start of Code Segment
	.cseg

; ====================================================================
; Function: Scans a Disk for CP/M Partions
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; 
; Registers  : [w] temp	    Number of disk images (raw and fat16) found.
;			    + 0x80 if sd card changes. (not used, doesn't work)
; SREG	:	   Z 	    according to temp
; --------------------------------------------------------------------
; Description:
; This Function scans an SD-Cards Boot-Sector for valid Partitions.
; First all original CP/M Partitions will be usesed as Drives for
; the CPM-System. Wenn all CP/M Partitions are found, a second
; scann will be made. In the second Scan, the first FAT16 Partition
; on the Disk will be used for a detailed analyses. If there
; are any Files like "cpm_x.img" are found, these Files will be
; used as Disks by the CP/M- System. ( x must be in the Range A to D )
; ====================================================================	
mgr_init_partitions:

	sts	ndisks,_0		; Set Number of Disks to 0

; Initialize partition table
	ldiw	y,hostparttbl
	ldi	temp2,PARTENTRY_SIZE*MAXDISKS
mgr_picl:
	st	y+,_0
	dec	temp2
	brne	mgr_picl

; Start mmc Card interaction
	lcall	mmcInit
	andi	temp,MMCST_NOINIT & MMCST_NODISK
	brne	mgr_pierr
	
;Load first sector from MMC (boot sector)
	ldiw	y,0			; Sector 0
	movw	x,y
	lcall	mmcReadSect
	tst	temp
	breq	mgr_check_bootsektor

mgr_pierr:
	clr	temp
	ret

mgr_check_bootsektor:
;Pointer to first table entry
	ldiw	y,hostparttbl
	ldi	temp3,0			;temp3 holds number of found disks (paritions)

;Test, if it has a valid MBR

	lds	temp,hostbuf+510	;MBR signature (0xAA55)  at and of sector?
	lds	temp2,hostbuf+510+1
	ldi	temp4,0xAA
	cpi	temp,0x55		
	cpc	temp2,temp4
	breq	mgr_search

;No MBR, no partition table ...

	inc	temp3			;pretend we have one.
	sts	ndisks,temp3
	ldi	temp,high((1<<16) * 128/512)
	ldi     temp2,dskType_CPM
	std	y+0,temp2
	std	y+1,_0			;start at beginning of card
	std	y+2,_0
	std	y+3,_0
	std	y+4,_0
	std	y+5,_0			;max CP/M 2.2 disk size
	std	y+6,temp		;

	ldi	temp3,0
	rcall	dpb_imgdata_get

	rjmp	mgr_pend

; Search for valid Partitions and ImageFiles 
mgr_search:
	ldiw	z,hostbuf+510-64	;Point to  first byte of partition table
	ldi	temp4,4			;Partition table has 4 entries.

mgr_ploop:

;	Get Partitiontype
	ldd	temp,z+PART_TYPE

;   Test for CP/M Partition
	cpi	temp,PARTID_CPM
	brne	mgr_nextp
	
	rcall	cpm_add_partition
	inc	temp3
	sts	ndisks,temp3
	adiw	y,PARTENTRY_SIZE
	cpi	temp3,MAXDISKS
	breq	mgr_pend
	
mgr_nextp:
	adiw	z,16
	dec	temp4
	brne	mgr_ploop

#if FAT16_SUPPORT

; Test for FAT16 Partition
	ldiw	z,hostbuf+510-64	;Point to  first byte of partition table
	ldi	temp4,4

mgr_ploop2:
;	Get Partitiontype
	ldd	temp,z+PART_TYPE

;   Test for FAT Partition Type 1
	cpi	temp,PARTID1_FAT16
	breq	mgr_fatfound

;   Test for FAT Partition Type 2
	cpi	temp,PARTID2_FAT16
	brne	mgr_nextp2

mgr_fatfound:
	rcall	fat_add_partition
	rcall	fat_scan_partition
	rcall	fat_reset_cache
	rjmp	mgr_pend		;Stop after first FAT16 partition found.

mgr_nextp2:
	adiw	zl,16
	dec	temp4
	brne	mgr_ploop2
#endif

mgr_pend:
	clr	temp3
mgr_imgd_lp:
	lds	temp,ndisks
	cp	temp3,temp
	breq	mgr_pend2
	rcall	dpb_imgdata_get
	inc	temp3
	rjmp	mgr_imgd_lp

mgr_pend2:
	lds	temp,ndisks	;return # of "disks"
	tst	temp
	ret


; ====================================================================
; Function: Print partition table info
; ====================================================================
; Parameters
; --------------------------------------------------------------------
; Registers  :  none
; Variables  :  [r] hostparttbl		Table with Partitioninformations
;               [r] hostparttbltop	Pointer to the Top of the Table
; --------------------------------------------------------------------
; Description:
; ====================================================================

mgr_prnt_parttbl:
	ldiw	z,hostparttbl
	lds	yl,ndisks
	ldi	xh,'A'

pprl:
	ldd	temp ,z+1		;Get partition start
	ldd	temp2,z+2
	ldd	temp3,z+3
	ldd	temp4,z+4

	printnewline

	cp	temp,_0			;If zero ...
	cpc	temp2,_0
	cpc	temp3,_0
	cpc	temp4,_0
	breq	mgr_prnop		;... no partition table at 0

; Partitiontype examining
	ldd 	xl,z+0
	andi	xl,dskType_MASK
; CP/M ?
	cpi	xl,dskType_CPM
	brne 	mgr_prtb_nocpm
	rcall   mgr_prnt_diskname
	rcall	mgr_prnt_table_cpm
	rjmp	mgr_prnt_size

mgr_prtb_nocpm:
#if FAT16_SUPPORT
; FAT16 ?
	cpi	xl,dskType_FAT
	brne 	mgr_prtb_nofat
	rcall   mgr_prnt_diskname
	rcall	mgr_prnt_table_fat
	rjmp	mgr_prnt_size
mgr_prtb_nofat:
#endif
#if 0					/* RAMDISK is not on SD card */
; RAMDISK ?
	cpi	xl,dskType_RAM
	brne 	mgr_prnt_err
	rcall   mgr_prnt_diskname
	rcall	mgr_prnt_table_ram
	rjmp	mgr_prnt_size
#endif
mgr_prnt_err:	
; Entry Error
	rcall	mgr_prnt_table_err
	rjmp	mgr_prnt_size

mgr_prnop:
	rcall   mgr_prnt_diskname
	rcall	mgr_prnt_image

mgr_prnt_size:
	lcall	print_ultoa
	printstring ", size: "

	ldd	temp ,z+5		;Get partition size
	ldd	temp2,z+6
	ldi	temp3,0
	ldi	temp4,0

	lsr	temp4
	ror	temp3
	ror	temp2
	ror	temp
	lcall	print_ultoa
	printstring "KB."

mgr_goto_next_part:	
	adiw	z,PARTENTRY_SIZE
	inc	xh
	dec	yl
	brne	pprl

mgr_pppre:
	ret
	

mgr_prnt_diskname:
	push	temp
	mov	temp,xh
	lcall	uartputc
	ldi	temp,':'
	lcall	uartputc
	pop	temp
	ret

mgr_prnt_table_cpm:
	printstring "CP/M partition at: "
	ret

mgr_prnt_table_fat:
	printstring "FAT16 File-Image at: "
	ret

mgr_prnt_table_ram:
	printstring "Ramdisk at: "
	ret

mgr_prnt_table_err:
	printstring "Unknown Entry at: "
	ret

mgr_prnt_image:
	printstring "Assuming CP/M image at: "
	ret


