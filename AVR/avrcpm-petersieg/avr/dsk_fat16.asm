;    Various functions for the Interaction with the FAT16 Filesystem
;
;    Copyright (C) 2010 Frank Zoll
;    Copyright (C) 2010 Sprite_tm
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
;    $Id: dsk_fat16.asm 165 2011-04-19 22:45:26Z leo $
;

; ============================================================================
; Prelimitary !
; 같같같같같같�
; Size of a Sector is fixed to 512 Bytes by Base - MMC Driver implementation
; The Functions below therefore assume a fixed Size of 512 Bytes per Sector.
; ============================================================================

#if FAT16_SUPPORT


; ############################################################################ 
;                       	Defines for FAT16 Structures
; ############################################################################ 

/*These are the Offsets to the Variables within the Bootsector of a FAT16
  Partition.
 */
;#define FAT16_BSO_SECSIZE	0x0b	; Offset to Sectorsize Word
#define FAT16_BSO_CLUSTSZ   	0x0d    ; Offset to Clustersize Byte
#define FAT16_BSO_RESSECT   	0x0e	; Offset to Number of Reserved Sectors
#define FAT16_BSO_VOLPTR    	0x1c    ; Offset to First VolumeSector
#define FAT16_BSO_SECPERFAT 	0x16    ; Offset to Number of Sectors per Fat
#define FAT16_BSO_NUMFATCP  	0x10	; Offset to Ammount of FAT Copys
#define FAT16_BSO_NUMDIRENT 	0x11	; Offset to Max. Root Dir. Entrys
#define	FAT16_FIRST_IMAGENAME	'A'	; First letter of filename to search 
#define FAT16_LAST_IMAGENAME    'Z'	; Last letter of filename to 
/*
#define FAT16_LAST_IMAGENAME    'A'+MAXDISKS-1	; Last letter of filename to 
						; search 
*/

; ############################################################################ 
; 				Start of Data Segment
; ############################################################################ 
	.dseg

fat_partfound:   .byte   1	; (partition: 0= no found 1=found )
fat_parttbl: 	 .byte	 4	; first fat16 partition entry 
				; only startsector is needed
fat_clustersize: .byte   1	; sectors per cluster
fat_numdirentrys:.byte   2	; Max. num. of entrys within Rootdirektory
fat_ptr2fat:     .byte   4	; pointer to the first fat sector
fat_ptr2dat:     .byte   4	; pointer to the first data sector

/*These variables define a cache that holds the last Cluster and Sector
  thats been searched vor. To save some of the valuabe SRAM- Space these
  variables also are used as temporary variables by the function 
  fat_scan_partition.
  */
fat_last_dsk:	  .byte	 1	; number of disk with entry in cache
fat_log_clust:	  .byte	 2	; last searched logical cluster
fat_clust_offset: .byte	 1	; offset within the cluster
fat_clust_ptr:	  .byte	 4 	; sector of last real cluster

/* This Variable is only needed within the scanning of the directory 
for tempoary variable storage.. todo: optimize away :-) */
fat_temp:	  .byte 3	; for tempoary use

; ############################################################################ 
; 				Start of Code Segment
; ############################################################################ 
	.cseg

; ============================================================================
; Function: Initialize internal FAT-Partition Variables
; ============================================================================
; Parameters
; ----------------------------------------------------------------------------
; Registers  :	none
; Variables  :	[out]	fat_parttabl
; ----------------------------------------------------------------------------
; Description:
; This Routine initializes the internal Variables, that point to the
; first Found FAT16 Partition.
; ============================================================================
fat_init_partitiontable:

	sts 	fat_partfound,_0

	ldiw	y,fat_parttbl
	st	y+,_0
	st	y+,_0
	st	y+,_0
	st	y+,_0
	ret

; ============================================================================
; Function: Resets the Cache
; ============================================================================
; Parameters
; ----------------------------------------------------------------------------
; Registers  :	none
; Variables  :	[out]	fat_log_clust
;		[out]	fat_last_dsk
; ----------------------------------------------------------------------------
; Description:
; This Routine resets the internal Cache- Variables. After reset, the
; next read or write Command will initialize a scan of the FAT of
; the FAT16-Partition for the given sector.
; ============================================================================
fat_reset_cache:
	push	yl
	ldi	yl,0xFF
	sts 	fat_log_clust  ,yl
	sts	fat_log_clust+1,yl
	sts	fat_last_dsk   ,yl
	pop	yl
	ret

; ============================================================================
; Function: Saves FAT16 Partitiondata for later Scanning
; ============================================================================
; Parameters
; ----------------------------------------------------------------------------
; Registers  :	[in]	z		Pointer to the Partitondata
; Variables  :	[out]	fat_partfound	Boolean for "Partition found"
;		[out]	fat_parttbl	Pointer to Partitiontable
; ----------------------------------------------------------------------------
; Description:
; This funktion sets the internal Variables to the Start and Size
; of a given FAT16 Paritition. This Information will be used for a
; later scanning of the Partition. See Function "fat_scan_partition"
; for more information. 
; ============================================================================
fat_add_partition:
	
.if FAT16_DEBUG > 0
	printstring "fat16 part found"
	printnewline
.endif


;   save variables on stack
	push 	yl
	push 	yh

; set fat16 partition found flag
	ldi  	yl,1
	sts 	fat_partfound,yl

;   save data from first fat16 partition
	ldiw	y,fat_parttbl

	ldd	temp,z+PART_START
	st	y+,temp
	ldd	temp,z+PART_START+1
	st	y+,temp
	ldd	temp,z+PART_START+2
	st	y+,temp
	ldd	temp,z+PART_START+3
	st	y+,temp

;   reload variables from stack
	pop	yh
	pop 	yl

	ret

; ============================================================================
; Read and Scann a FAT16 Partition for Imagedatefiles 
; ============================================================================
; Registers	: none
; Variables	: none
; ----------------------------------------------------------------------------
; This Routine reads the Bootblock and scanns it for a Diskimage
; ============================================================================


fat_scan_partition:

.if FAT16_DEBUG > 0
	printstring "fat16 scanning"
	printnewline
.endif

; Check if a FAT16 Partition was realy found
	lds	yl,fat_partfound
	cpi 	yl,1	
	brne	fat_scan_error 


.if FAT16_DEBUG > 0
	printstring "free entrys in ptable ?"
	printnewline
.endif

; Check for free Entrys in Partition table
	lds	yl,ndisks
	cpi	yl,MAXDISKS
	breq	fat_scan_error

.if FAT16_DEBUG > 0
	printstring "read fat bootblock."
	printnewline
.endif

; Scan partition start
	ldiw	z,fat_parttbl	
	ldd	xl,z+0		
	ldd	xh,z+1
	ldd	yl,z+2
	ldd	yh,z+3

; Load first sector from Partition
	rcall	mmcReadSect
	tst	temp
	breq	fat_bootblock_check

; Read error: Block not found
fat_scan_error:
	clr	temp
	ret

fat_bootblock_check:

.if FAT16_DEBUG > 0
	printstring "fat16 bootblock check"
	printnewline
.endif

;   	get sectors per cluster from bootblock
	ldiw	z,hostbuf+FAT16_BSO_CLUSTSZ
	ld	temp,z
	sts     fat_clustersize,temp

.if FAT16_DEBUG > 0
	printstring "Sectors per Cluster "
	rcall printhex
	printnewline
.endif

;   	get num of FAT Tables from bootblock
	ldiw	z,hostbuf+FAT16_BSO_NUMFATCP
	ld	temp,z
	sts     fat_last_dsk,temp		; low byte

.if FAT16_DEBUG > 0
	printstring "Ammount of FAT copies: "
	rcall printhex
	printnewline
.endif

;	get max num of entrys in root direktory from bootblock
	ldiw	z,hostbuf+FAT16_BSO_NUMDIRENT
	ld	temp,z+
	sts     fat_numdirentrys,temp		; low byte
	ld	temp2,z
	sts	fat_numdirentrys+1,temp2	; high byte

.if FAT16_DEBUG > 0
	printstring "Max. entrys in Rootdir.: "
	rcall printhexw
	printnewline
.endif

; 	Print begin of Volume
.if FAT16_DEBUG > 1

	ldiw	z,fat_parttbl	
	ldd	xl,z+0		
	ldd	xh,z+1
	ldd	yl,z+2
	ldd	yh,z+3

	printstring "Begin of Volume at: "
	mov	temp ,yl
	mov	temp2,yh
	rcall	printhexw
	mov	temp ,xl
	mov	temp2,xh
	rcall	printhexw
	printnewline
.endif

;	get num of sectors per FAT-Table from bootblock
	ldiw	z,hostbuf+FAT16_BSO_SECPERFAT
	ld	temp,z+
	sts     fat_log_clust,temp		; low byte
	ld	temp2,z
	sts	fat_log_clust+1,temp2		; high byte

.if FAT16_DEBUG > 0
	printstring "Sectors per FAT__: "
	rcall printhexw
	printnewline
.endif

;	get num of reseved sectors from bootblock
	ldiw	z,hostbuf+FAT16_BSO_RESSECT
	ld	temp,z+
	ld	temp2,z

; 	Calculate begin of FAT within the Volume
	ldiw	z,fat_parttbl	
	ldd	xl,z+0		
	ldd	xh,z+1
	ldd	yl,z+2
	ldd	yh,z+3

	add	xl,temp
	adc 	xh,temp2
	adc	yl,_0
	adc	yh,_0

	sts	fat_ptr2fat  ,xl
	sts	fat_ptr2fat+1,xh
	sts	fat_ptr2fat+2,yl
	sts	fat_ptr2fat+3,yh

.if FAT16_DEBUG > 1
	printstring "Begin of FAT at___: "
	mov	temp ,yl
	mov	temp2,yh
	rcall	printhexw
	mov	temp ,xl
	mov	temp2,xh
	rcall	printhexw
	printnewline
.endif

; Calculate begin of Root- Directory within the Volume
	ldiw	z,fat_ptr2fat
	ldd	xl,z+0
	ldd	xh,z+1
	ldd	yl,z+2
	ldd	yh,z+3

	lds	temp ,fat_log_clust
	lds	temp2,fat_log_clust+1
	lds	temp3,fat_last_dsk

fat_calc_dp_loop:
	cp 	temp3,_0
	breq	fat_calc_dp_lend

	add	xl,temp
	adc	xh,temp2
	adc	yl,_0
	adc	yh,_0

	dec	temp3

	rjmp	fat_calc_dp_loop
fat_calc_dp_lend:

	sts	fat_clust_ptr  ,xl
	sts	fat_clust_ptr+1,xh
	sts	fat_clust_ptr+2,yl
	sts	fat_clust_ptr+3,yh


.if FAT16_DEBUG > 1
	printstring "Begin of DIR at___: "
	mov	temp ,yl
	mov	temp2,yh
	rcall	printhexw
	mov	temp ,xl
	mov	temp2,xh
	rcall	printhexw
	printnewline
.endif

; Calculate begin of DATA Clusters within the Volume
; Num. Dir.Sektors = (Num. of Dir. Entrys * 32) / Bytes per Sektor

; Sectorsize is fixed at 512 Bytes, makes 16 Entrys per Sektor

	lds     zl,fat_numdirentrys		; low byte
	lds	zh,fat_numdirentrys+1		; high byte

;   Num. Direntrys / 16
	lsr	zh
	ror	zl
	lsr	zh
	ror	zl
	lsr	zh
	ror	zl
	lsr	zh
	ror	zl

	lds	xl,fat_clust_ptr
	lds	xh,fat_clust_ptr+1
	lds	yl,fat_clust_ptr+2
	lds	yh,fat_clust_ptr+3

	add	xl,zl
	adc	xh,zh
	adc	yl,_0
	adc	yh,_0

	sts	fat_ptr2dat  ,xl
	sts	fat_ptr2dat+1,xh
	sts	fat_ptr2dat+2,yl
	sts	fat_ptr2dat+3,yh

.if FAT16_DEBUG > 1
	printstring "Begin of Data at__: "
	mov	temp ,yl
	mov	temp2,yh
	rcall	printhexw
	mov	temp ,xl
	mov	temp2,xh
	rcall	printhexw
	printnewline
.endif

; Here Starts the Scann of the Directory for valid image Files.

;	Init Image-Namecounter
	ldi	temp,FAT16_FIRST_IMAGENAME
	sts	fat_last_dsk,temp

fat_scan_for_next_image:

;	Init Offset into Directory-Sectors
	ldi	temp,0
	sts	fat_clust_offset,temp

;	Init counter for number of entry left to scan
	lds	temp,fat_numdirentrys
	sts	fat_log_clust  ,temp

	lds	temp,fat_numdirentrys+1
	sts	fat_log_clust+1,temp

fat_next_sector_loop:
;   Get a Pointer to the first Directory sector
	lds	xl,fat_clust_ptr
	lds	xh,fat_clust_ptr+1
	lds	yl,fat_clust_ptr+2
	lds	yh,fat_clust_ptr+3

;	Add actual offset
	lds	temp,fat_clust_offset
	add	xl,temp
	adc	xh,_0
	adc	yl,_0
	adc	yh,_0

;  Load sector from Directory
	lcall	mmcReadSect
	tst	temp
	breq	fat_look_for_images

; Read error: Block not found
	clr	temp
	ret

; Looks at a read directory block for image entrys
fat_look_for_images:
	
	ldiw	z,hostbuf
	ldi	temp2,0

fat_look_for_loop:	
	ldd 	temp,z+0
	cpi	temp,'C'
	brne	fat_look_not_ok
	
	ldd	temp,z+1
	cpi	temp,'P'
	brne	fat_look_not_ok

	ldd	temp,z+2
	cpi	temp,'M'
	brne	fat_look_not_ok

	ldd	temp,z+3
	cpi	temp,'D'
	brne	fat_look_not_ok

	ldd	temp,z+4
	cpi	temp,'S'
	brne	fat_look_not_ok

	ldd	temp,z+5
	cpi	temp,'K'
	brne	fat_look_not_ok

	ldd	temp,z+6
	cpi	temp,'_'
	brne	fat_look_not_ok

	lds	temp3,fat_last_dsk	; Get actual Diskname (A to Z)
	ldd	temp,z+7
	cp	temp,temp3
	brne	fat_look_not_ok

	ldd	temp,z+8
	cpi	temp,'I'
	brne	fat_look_not_ok

	ldd	temp,z+9
	cpi	temp,'M'
	brne	fat_look_not_ok

	ldd	temp,z+10
	cpi	temp,'G'
	brne	fat_look_not_ok

	sts	fat_temp  ,zl
	sts	fat_temp+1,zh
	sts	fat_temp+2,temp2
	rjmp	fat_store_new_entry

fat_scan_for_more:

	lds	zl   ,fat_temp
	lds	zh   ,fat_temp+1
	lds	temp2,fat_temp+2
fat_look_not_ok:
		
	adiw	z,32

	inc	temp2
	cpi	temp2,16				; max entrys/sector
	breq	fat_scan_next_sector
	rjmp 	fat_look_for_loop

fat_scan_next_sector:

	
	lds	temp3, fat_log_clust
	lds	temp4, fat_log_clust+1

	sub	temp3,temp2
	sbc	temp4,_0

	sts	fat_log_clust,temp3
	sts	fat_log_clust+1,temp4
	
	cp	temp3,_0
	cpc	temp4,_0
	breq	fat_scan_at_end	

	lds	temp,fat_clust_offset
	inc	temp
	sts	fat_clust_offset,temp

	rjmp	fat_next_sector_loop

fat_scan_at_end:

	lds	temp,fat_last_dsk
	inc	temp
	sts	fat_last_dsk,temp

	ldi	temp2,FAT16_LAST_IMAGENAME
	cp	temp,temp2
	brge	fat_scaned_last_disk

	rjmp	fat_scan_for_next_image

fat_scaned_last_disk:

	rjmp	fat_scan_end


;	Create new Partition Entry
fat_store_new_entry:

;   Found a valid image
.if FAT16_DEBUG > 1
	printstring "Found a valid Image! Z="
	mov	temp ,zl
	mov	temp2,zh
	rcall	printhexw
	printnewline
.endif

	ldiw	y,hostparttbl
	lds	temp3,ndisks

fat_look_store_loop:
	tst	temp3
	breq	fat_look_store

	adiw	y,PARTENTRY_SIZE
	dec	temp3
	rjmp	fat_look_store_loop

fat_look_store:
;   Set Type of Partition to FAT16- Fileimage
	ldi 	temp,dskType_FAT
	std	y+0,temp


;   Offset to Startcluster + 2
	ldd	temp,z+0x1A
	std	y+1,temp

	ldd	temp,z+0x1B
	std	y+2,temp	

	ldi	temp,0
	std	y+3,temp
	std	y+4,temp

;   Convert Filesize to ammount of sectors
;   (calc with 512byte/sector)
	ldd	_tmp0,z+0x1C
	ldd	xl,z+0x1D
	ldd	xh,z+0x1E
	ldd	zl,z+0x1F
;	mov	zh,_0

	cpse	_tmp0,_0		;round up
	adiw	x,1
	adc	zl,_0

	lsr	zl
	ror 	xh
	ror 	xl

	adc	xl,_0
	adc	xh,_0
	adc	zl,_0

;   store ammount of sectors in partitiontable	

	tst	zl			;file size larger than 65535 sectors?
	breq	fat_add_noprune
	
	ldi	xl,255
	ldi	xh,255
fat_add_noprune:
	std	y+5,xl
	std	y+6,xh

.if FAT16_DEBUG > 1
; Test finding of the first sector
	
	ldd	xl,z+0x1A
	ldd	xh,z+0x1B

	rcall	fat_gethostsec

	printstring "Begin of Image at: "
	mov	temp ,yl
	mov	temp2,yh
	rcall	printhexw
	mov	temp ,xl
	mov	temp2,xh
	rcall	printhexw
	printnewline

.endif
; Check for another free entry in partition table
	lds	temp,ndisks
	inc	temp
	sts	ndisks,temp
	
	cpi	temp,MAXDISKS
	breq	fat_scan_end	

	rjmp	fat_scan_for_more	

fat_scan_end:
		
	ret


; ============================================================================
; Function: Cluster to HostSector 
; ============================================================================
; Parameters:	[in]	xh,xl		Cluster Number
;		[out]	yh,yl,xh,xl	Sector Number on Disk
; ----------------------------------------------------------------------------
; Registers  : 	
; Variables  : 	[used]	fat_clustersize	Ammount of Sectors per Cluster
;		[changes] temp
; ----------------------------------------------------------------------------
; Description:
; ! Only works with Clustersizes 2,4,8,16,32,64,128 !
; ============================================================================
fat_gethostsec:

;	Get Offset into Data area of Disk
	rcall	fat_clusttosec


;	add begin of data area to offset
	lds	temp,fat_ptr2dat+0
	add	xl,temp
	lds	temp,fat_ptr2dat+1
	adc	xh,temp
	lds	temp,fat_ptr2dat+2
	adc	yl,temp
	lds	temp,fat_ptr2dat+3
	adc	yh,temp
	ret

; ============================================================================
; Function: Cluster to Sector 
; ============================================================================
; Registers:	[in]	xl,xh			Cluster Number
;	 	[out]	xl,xh,yl,yh		Sector Number
; Variables:	[in]	fat_clustersize		Ammount of Sectors per Cluster
;		[out]	temp			=0
; ----------------------------------------------------------------------------
; Description:
; Calculates the logical Sectornumber given the physical ClusterNumber
; and the size of a Cluster un sectors.
;
; ! Only works with Clustersizes 2,4,8,16,32,64,128 !
; ============================================================================
fat_clusttosec:
	clr 	yl
	clr	yh

	ldi	temp,2
	sub	xl,temp		; Substract the 2 reserved clusters
	sbc	xh,_0

	lds	temp,fat_clustersize
	lsr	temp

fat_c2s_loop:
	tst	temp
	breq	fat_c2s_end
	lsr	temp

	lsl	xl
	rol	xh
	rol	yl
	rol	yh
	rjmp	fat_c2s_loop

fat_c2s_end:
	ret

; ====================================================================
; Function: Searches a physical Cluster, given the logical Cluster
; ====================================================================
; Registers:	[in]	xh,xl	logical- Cluster 
;		[out]	yh,yl	physical- Cluster
; Variables:
; --------------------------------------------------------------------
; Description:
; ====================================================================
fat_find_phsy_clust:
	lds	zl,hostdsk
	rcall 	dsk_getpartentry	; get partition entry

;	Get First FAT- Cluster Number of Diskimage
	
	ldd	yl,z+1
	ldd	yh,z+2

.if FAT16_DBG_FAT > 0
	printstring "Search log. Cluster "
	mov	temp,xl
	mov	temp2,xh
	lcall	printhexw
	printnewline
		
	printstring "Search phys. Cluster "
	mov	temp ,yl
	mov	temp2,yh
	lcall	printhexw
	printnewline
.endif

fat_next_phsy_clust:	
	cp	xl,_0
	cpc	xh,_0
	breq	fat_found_phsy_clust
;	Get Next Cluster from Fat

; Trick: 512 Bytes Per Sector equals to 256 FAT- Entrys per Sector
; so given:  yl is the Offset within the FAT Sector
;            yh is the number off se FAT sector to Read 

; 	in	zh,zl:		Pointer to Word within the Sector to read	
;   in	yh..xl:	Start sector number (LBA)
;	out	zh,zl	: word thats been read
	push 	xl
	push 	xh

;   Create FAT Offset Value
	clr	zh
	mov 	zl,yl
	lsl	zl
	rol 	zh
;   Get FAT Start
	mov 	temp,yh
	lds 	xl,fat_ptr2fat
	lds 	xh,fat_ptr2fat+1
	lds 	yl,fat_ptr2fat+2
	lds 	yh,fat_ptr2fat+3
;   Add Cluster offset within sector
	add	xl,temp
	adc	xh,_0
	adc 	yl,_0
	adc 	yh,_0
	lcall 	mmcReadWord

	pop 	xh
	pop 	xl

	mov 	yl,zl
	mov 	yh,zh

;	Check next logical Cluster
	ldi	zl,1
	sub	xl,zl
	sbc	xh,_0
	rjmp	fat_next_phsy_clust
	
; Found the physical cluster
fat_found_phsy_clust:
	
.if FAT16_DBG_FAT > 0
	printstring "Found phys. Cluster at:"
	mov 	temp,yl
	mov 	temp2,yh
	lcall	printhexw
	printnewline
.endif	

	ret

; ============================================================================
; Function: This Routine searches for the Sector within an Imagefile 
; ============================================================================
; Registers:	[out] xl,xh,yl,yh	Pointer to the Sector on the SD-Card
;		[out] temp		Error- Variable (0= No Error)
; Variables:	[in] hostdsk		host disk #,  (partition #)
; 		[in] hostlba		host block #, relative to part.start
;		[in] fat_last_dsk	number of disk with entry in cache
;		[in] fat_log_clust	last searched logical cluster
;		[in] fat_clust_offset	offset within the cluster
;               [in] fat_clust_ptr	sector of last real cluster
; ----------------------------------------------------------------------------
; Description:
; This Routine uses the variables hostdsk and hostlba to find an Sector
; in the Imagefile.
; The CP/M Sector given within "hostlba" are splited to a logical Cluster-
; Number and the Subsector within this logical Cluster.
; logical Cluster Number = hostlba / fat_clustersize
; The logical Cluster Number will be compared to the logical Cluster- Number
; within the Cache. When this Clusters are the same and the DiskID's are
; also the same, then the cached physical Sector will be used.
; When the Clusters or the Disks don't match, a seek for the physical
; Cluster is performed. This seek is done thru an access over the FAT of
; the FAT16 Partition. The Routine starts at the first Cluster of the 
; Imagefile and goes along the linked list of Clusternumber till it reaches
; the searched cluster. The found Clusternumber will be used to calculate
; the Sektor where this Cluster lies on the SD- Card. Both the found physical
; Cluster and the logical Cluster together with the physical Sectornumber
; are stored in the cache.
; The last step done is to add the SubSectorOffset to the found physical
; Sector. This gives the pointer to the Sector to be read and or written.
; ============================================================================

fat_hostparam:
	lds	zl,hostdsk
	rcall	dsk_getpartentry	; get partition entry

fat_hostlend:
	lds	temp ,hostlba
	lds	temp2,hostlba+1
;	lds	temp3,hostlba+2


	ldd	xl,z+5			; get size of disk in sectors
	ldd	xh,z+6
;	ldd	yl,z+7
	
	cp	temp,xl			; check given sector against disksize
	cpc	temp2,xh
;	cpc	temp3,yl
	brcs	fat_hp1
	
	clr	temp
	ret

fat_hp1:
; ################# Get logical Number of Cluster within the imagefile
;	printstring "calc log sector"
; Get logical Sectornumber from temp
	mov 	xl,temp
	mov 	xh,temp2
;	mov 	yl,temp3
	mov 	yl,_0
	mov 	yh,_0
; Divide logical Sectornumber by size of Cluster in sectors
	lds	zl,fat_clustersize
	lsr     zl
fat_search_clst_lp:
	tst	zl
	breq	fat_found_clst

	lsr	yh
	ror	yl
	ror	xh
	ror	xl
	
	lsr	zl

	rjmp	fat_search_clst_lp
		
fat_found_clst:			
; at this point xh and xl are carying the logical cluster number
;	printstring "find subsector"
; ################# Get Subsector within the logical Cluster for later use
	mov	yl,xl
	lds	zl,fat_clustersize
	lsr	zl
fat_search_clst_lp2:
	tst	zl
	breq	fat_found_subsec
	lsl	yl

	lsr	zl
	rjmp	fat_search_clst_lp2		

fat_found_subsec:
	mov	zl,temp
	sub	zl,yl
	sts	fat_clust_offset,zl

; Check against last HOSTDISK searched
	lds	yl,fat_last_dsk
	lds	yh,hostdsk
	cp	yl,yh
	brne	fat_wrong_cache_clst

; Check against last Cluster searched
	lds	yl,fat_log_clust
	lds	yh,fat_log_clust+1

	cp	yl,xl
	brne	fat_wrong_cache_clst
	cp	yh,xh
	brne	fat_wrong_cache_clst

;   Last Cluster = searched Cluster -> get Sectornumber from cache
	lds	xl,fat_clust_ptr
	lds	xh,fat_clust_ptr+1
	lds	yl,fat_clust_ptr+2
	lds	yh,fat_clust_ptr+3

	rjmp	fat_add_offset

;  Cluster is not in cache, so we must search for it
fat_wrong_cache_clst:
	lds	yl,hostdsk
	sts	fat_last_dsk,yl
	sts	fat_log_clust,xl
	sts	fat_log_clust+1,xh

;  Map Logical Cluster-Number to "Physical" Cluster Number using the FAT
	rcall   fat_find_phsy_clust

;  Get StartSector of "physical" Cluster
	mov 	xl,yl
	mov	xh,yh
	rcall   fat_gethostsec

; Found the physical sector
.if FAT16_DBG_FAT > 0
	printstring "Found phys. Sector at:"
	mov	temp,yl
	mov	temp2,yh
	lcall	printhexw
	mov	temp,xl
	mov	temp2,xh
	lcall	printhexw
	printnewline
.endif	

;   Save the found Sector for later use into cache
	sts	fat_clust_ptr  ,xl
	sts	fat_clust_ptr+1,xh
	sts	fat_clust_ptr+2,yl
	sts	fat_clust_ptr+3,yh

;   Add- Subsector to Startsector 
fat_add_offset:
	lds	zl,fat_clust_offset
	add	xl,zl
	adc	xh,_0
	adc	yl,_0
	adc	yh,_0

; Found the physical sector
.if FAT16_DBG_FAT > 0
	printstring "Sector with Offset at:"
	mov	temp,yl
	mov	temp2,yh
	lcall	printhexw
	mov	temp,xl
	mov	temp2,xh
	lcall	printhexw
	printnewline
.endif

	ori	temp,255
fat_hpex:
	ret

; ============================================================================
; Function: Does a Disk write operation
; ============================================================================
; Registers: 	[out]	temp	Error-Variable ( 0= No Error)		
; Variables:	[in]	hostdsk	host disk #,  (partition #)
; 		[in]	hostlba	host block #, relative to part.start
;		[in]	hostbuf Sector to be written
; ----------------------------------------------------------------------------
; Description:
; This Routine writes a Sector to the Imagefile inside an FAT16 Partition.
; ============================================================================

fat_writehost:
.if FAT16_RWDEBUG > 1
	printnewline
	printstring "host write "
.endif
	rcall	fat_hostparam
	breq	fat_rdwr_err
	
	call	mmcWriteSect
	tst	temp
	breq	fat_rdwr_ok
	
	rjmp	fat_rdwr_err		; skip disk change detection code
	
; After a second thought, the following  code doesn't make sense, because
; disk change (change of one or more disk images) can not reliably detected.
; At least with the existing code.



	rcall	mgr_init_partitions
	cbr	temp,0x80
	breq	fat_rdwr_err

	rcall	fat_hostparam
	breq	fat_rdwr_err
	call	mmcWriteSect	; disabled till read is functioning
	tst	temp
	brne	fat_rdwr_err
	rjmp	fat_rdwr_ok

fat_rdwr_ok:
	sts	erflag,_0
	ret

fat_rdwr_err:
	sts	erflag,_255
	ret

; ============================================================================
; Function: Does a Disk read operation
; ============================================================================
; Registers: 	none
; Variables:	[in]	hostdsk		host disk #,  (partition #)
; 		[in]	hostlba		host block #, relative to part.start
;		[out]	hostbuf 	Sector read by this routine
; ----------------------------------------------------------------------------
; Description:
; This Routine reads a Sector from the Imagefile inside an FAT16 Partition. 
; ============================================================================

fat_readhost:
.if FAT16_RWDEBUG > 1
	printnewline
	printstring "host read  "
.endif

	rcall	fat_hostparam
	breq	fat_rdwr_err
	
	
.if FAT16_RWDEBUG > 0
	printstring "Read Image Sector:"
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
	printnewline
.endif
	
	lcall	mmcReadSect
	tst	temp
	breq	fat_rdwr_ok

	rjmp	fat_rdwr_err		; skip disk change detection code
	
	rcall	mgr_init_partitions
	cbr	temp,0x80
	breq	fat_rdwr_err

	rcall	fat_hostparam
	breq	fat_rdwr_err
	lcall	mmcReadSect
	tst	temp
	brne	fat_rdwr_err

#endif

; vim:set ts=8 noet nowrap

