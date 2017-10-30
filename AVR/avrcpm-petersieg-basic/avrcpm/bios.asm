;    CP/M BIOS for avrcpm
;    Copyright (C) 2010 Sprite_tm
;
;    This program is free software: you can redistribute it and/or modify
;    it under the terms of the GNU General Public License as published by
;    the Free Software Foundation, either version 3 of the License, or
;    (at your option) any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see <http://www.gnu.org/licenses/>.

msize:	equ	62	;size of available RAM in k

bias:	equ	(msize-20) * 1024 
ccp:	equ	3400h+bias	;base of cpm ccp
bdos:	equ	ccp+806h	;base of bdos
bios:	equ	ccp+1600h	;base of bios
cdisk:	equ	0004h		;current disk number (0 ... 15)
iobyte:	equ	0003h		;intel iobyte
buff:	equ	0080h		;default buffer address
retry:	equ	3		;max retries on disk i/o before error

cr:	equ	13
lf:	equ	10

READ_FUNC:  equ	7
WRITE_FUNC: equ	6
BOOT_FUNC:  equ 5
HOME_FUNC:  equ 4

	org	bios
nsects:	equ	($-ccp)/128	;warm start sector count
	
	jp boot
wboote:	
	jp wboot
	jp const
	jp conin
	jp conout
	jp list
	jp punch
	jp reader
	jp home
	jp seldsk
	jp settrk
	jp setsec
	jp setdma
	jp read
	jp write
	jp listst
	jp sectran

signon:
	db	cr,lf
	db	msize/10+'0'
	db	msize - (msize/10)*10 + '0'	;modulo doesn't work?
	db	"k cp/m vers 2.2"
msgnl:	db	cr,lf,0

const:
	in a,(0)
	ret

conin:
	in a,(0)
	cp 0ffh
	jp nz,conin

	in a,(1)
	ret

conout:
	ld a,c
	out (2),a
	ret

list:
	ret

listst:
	ld a,0
	ret

punch:
	ret

reader:
	ld a,01Fh
	ret

prmsg:
	ld	a,(hl)
	or	a
	ret	z
	push	hl
	ld	c,a
	call	conout
	pop	hl
	inc	hl
	jp	prmsg
	
prhex:
	ld	a,c
	push	af
	rra	
	rra	
	rra	
	rra	
	call	prhexdigit
	pop	af
	;	fall thru

prhexdigit:
	and	00fh
	cp	10
	jp	c,prd1
	add	a,7
prd1:
	add	a,'0'
	ld	c,a
	jp	conout


boot:
	ld	sp,buff
	ld	hl,signon
	call	prmsg
	
	xor	a
	ld	(bootdsk),a
	ld	a,(dpb)
	ld	(bootspt),a
	
	ld	c,'I'-'A'
	call	seldsk
	ld	a,h
	or	l
	jp	z,boot1

	ld	de,10
	add	hl,de
	ld	e,(hl)
	inc	hl
	ld	d,(hl)		;de = dpb of first ram disk
	
	ld	hl,7
	add	hl,de
	ld	a,(hl)		;get drm
	inc	a
	and	0fch
	rrca			;4 dir entries per sector
	rrca			;Number of sectors to init
	push	af
	
	ld	bc,6
	add	hl,bc
	ld	c,(hl)		;Start track
	push	bc		;Save track
	
; Check, if we have reserved tracks.
	ld	a,c
	or	a
	jp	z,boot0		;Skip if not.

; Save CPM to ram disk.

	ld	a,(de)		;sectors per track
	ld	(bootspt),a
	ld	a,'I'-'A'
	ld	(bootdsk),a
	call	home
	ld	b,nsects
	ld	c,0		;track
	ld	d,1		;sektor (0 based)
	ld	hl,ccp
store1:
	push	bc
	push	de
	push	hl
	ld	c,d
	ld	b,0
	call	setsec
	pop	bc
	push	bc
	call	setdma
	ld	c,0
	call	write
	
	pop	hl
	ld	de,128
	add	hl,de
	pop	de
	pop	bc
	dec	b
	jp	z,boot0
	
	inc	d
	ld	a,(bootspt)
	dec	a
	cp	d		;if sector >= spt then change tracks
	jp	nc,store1
	
	ld	d,0
	inc	c
	push	bc
	push	de
	push	hl
	ld	b,0
	call	settrk	
	pop	hl
	pop	de
	pop	bc
	jp	store1

; Clear directory area of ram disk.

boot0:
	pop	bc
	call	settrk
	pop	af
	ld	d,a		;d = # of sectors
	ld	e,0		;e = sector
	push	de
	ld	hl,dirbuf	;Clear dirbuf
	ld	c,128
	ld	a,0E5h
boot_cl:
	ld	(hl),a
	inc	hl
	dec	c
	jp	nz,boot_cl

	ld	bc,dirbuf
	call	setdma
	pop	de
boot_cl2:
	push	de
	ld	c,e
	ld	b,0
	call	setsec
	ld	c,0
	call	write
	pop	de
	inc	e
	dec	d
	jp	nz,boot_cl2	
	
boot1:	
	xor	a
	ld	(iobyte),a
	ld	(cdisk),a
	jp	gocpm 

wboot:	;re-load CP/M
	ld	sp,buff
	ld	a,1<<BOOT_FUNC	;init (de)blocking
	out	(22),a
	ld	a,(bootdsk)
	ld	c,a
	call	seldsk
	call	home
	ld	b,nsects
	ld	c,0		;track
	ld	d,1		;sektor (0 based)
	ld	hl,ccp
load1:
	push	bc
	push	de
	push	hl
	ld	c,d
	ld	b,0
	call	setsec
	pop	bc
	push	bc
	call	setdma
	call	read
	cp	0		;read error?
	jp	nz,wboot
	
	pop	hl
	ld	de,128
	add	hl,de
	pop	de
	pop	bc
	dec	b
	jp	z,gocpm
	
	inc	d
	ld	a,(bootspt)
	dec	a
	cp	d		;if sector >= spt then change tracks
	jp	nc,load1
	
	ld	d,0
	inc	c
	push	bc
	push	de
	push	hl
	ld	b,0
	call	settrk	
	pop	hl
	pop	de
	pop	bc
	jp	load1
	
gocpm:
	ld	a,0c3h
	ld	(0),a
	ld	hl,wboote
	ld	(1),hl
	ld	(5),a
	ld	hl,bdos
	ld	(6),hl
		
	ld	bc,buff
	call	setdma
	ld	a,(cdisk)
	ld	c,a
	jp	ccp
	
seldsk:
	ld	hl,dphtab
	ld	b,0
	add	hl,bc
	add	hl,bc
	ld	a,(hl)		;get table entry for selected disk
	inc	hl
	ld	h,(hl)
	ld	l,a		
	or	h		;no entry, no disk
	ret	z		;

	ld	a,c
	out	(15),a
	in	a,(15)		;querry, if disk exists
	or	a		;0 = disk is ok
	ret	z
	ld	hl,0		;error return code
	ret

home:
	ld a,1<<HOME_FUNC
	out (22),a
	
	ld bc,0			; same as seek to track 0
settrk:
	ld a,c
	out (16),a
	ld a,b
	out (17),a
	ret

setsec:
	ld a,c
	out (18),a
	ret

setdma:
	ld a,c
	out (20),a
	ld a,b
	out (21),a
	ret

read:
	ld a,1<<READ_FUNC
	out (22),a
	in a,(22)
	and 1
	ret

write:
	ld	a,c
	and	3		;mask write  type
	or	1<<WRITE_FUNC
	out	(22),a
	in a,(22)
	and 1
	ret

sectran:
	;translate sector bc using table at de, res into hl
	ld h,b
	ld l,c
	ld a,d
	or e
	ret z
	ex de,hl
	add hl,bc
	ld l,(hl)
	ld h,0
	ret

dphtab:
	dw	dpe0
	dw	dpe1
	dw	dpe2
	dw	dpe3
	dw	0	
	dw	0	
	dw	0	
	dw	0	
	dw	dperd0
	dw	0	
	dw	0	
	dw	0	
	dw	0	
	dw	0	
	dw	0	
	dw	0	

	

;Disk Parameter Header
dpbase:	
dpe0:	dw 0		;XLT: No sector translation table
	dw 0		;000: Scratchpad
	dw 0		;000: Scratchpad
	dw 0 		;000: Scratchpad
	dw dirbuf 	;DIRBUF: Address of a dirbuff scratchpad
	dw dpb		;DPB: Address of a disk parameter block
	dw chk0		;CSV: Address of scratchpad area for changed disks
	dw all0		;ALV: Address of an allocation info sratchpad
dpe1:	dw 0		;XLT: No sector translation table
	dw 0		;000: Scratchpad
	dw 0		;000: Scratchpad
	dw 0 		;000: Scratchpad
	dw dirbuf 	;DIRBUF: Address of a dirbuff scratchpad
	dw dpb		;DPB: Address of a disk parameter block
	dw chk1		;CSV: Address of scratchpad area for changed disks
	dw all1		;ALV: Address of an allocation info sratchpad
dpe2:	dw 0		;XLT: No sector translation table
	dw 0		;000: Scratchpad
	dw 0		;000: Scratchpad
	dw 0 		;000: Scratchpad
	dw dirbuf 	;DIRBUF: Address of a dirbuff scratchpad
	dw dpb		;DPB: Address of a disk parameter block
	dw chk2		;CSV: Address of scratchpad area for changed disks
	dw all2		;ALV: Address of an allocation info sratchpad
dpe3:	dw 0		;XLT: No sector translation table
	dw 0		;000: Scratchpad
	dw 0		;000: Scratchpad
	dw 0 		;000: Scratchpad
	dw dirbuf 	;DIRBUF: Address of a dirbuff scratchpad
	dw dpb		;DPB: Address of a disk parameter block
	dw chk3		;CSV: Address of scratchpad area for changed disks
	dw all3		;ALV: Address of an allocation info sratchpad

dperd0:	dw 0		;XLT: No sector translation table
	dw 0		;000: Scratchpad
	dw 0		;000: Scratchpad
	dw 0 		;000: Scratchpad
	dw dirbuf 	;DIRBUF: Address of a dirbuff scratchpad
	dw dpbrd	;DPB: Address of a disk parameter block
	dw chkrd0	;CSV: Address of scratchpad area for changed disks
	dw allrd0	;ALV: Address of an allocation info sratchpad


dpb:	dw 26		;SPT: sectors per track
	db 3		;BSH: data allocation block shift factor
	db 7		;BLM: Data Allocation Mask
	db 0		;Extent mask
	dw 242		;DSM: Disk storage capacity
	dw 63		;DRM, no of directory entries
	db 192		;AL0
	db 0		;AL1
	dw 16		;CKS, size of dir check vector
	dw 2		;OFF, no of reserved tracks
	
dpbrd:	dw 32		;SPT: sectors per track
	db 3		;BSH: data allocation block shift factor
	db 7		;BLM: Data Allocation Mask
	db 0		;Extent mask
	dw 55		;DSM: Disk storage capacity
	dw 31		;DRM, no of directory entries
	db 128		;AL0
	db 0		;AL1
	dw 0		;CKS, size of dir check vector
	dw 2		;OFF, no of reserved tracks
	
bootdsk:ds	1
bootspt:ds	1

dirbuf:
	ds 128

chk0:	ds 16
all0:	ds 31
chk1:	ds 16
all1:	ds 31
chk2:	ds 16
all2:	ds 31
chk3:	ds 16
all3:	ds 31
chkrd0:	ds 0
allrd0:	ds 7

;end

