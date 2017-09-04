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


org $4A00+$A800
	jp boot
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

boot:
	xor a
	ld (3),a
	ld (4),a

wboot:
	ld a,'B'
	out (2),a
	ld a,'I'
	out (2),a
	ld a,'O'
	out (2),a
	ld a,'S'
	out (2),a
	ld a,13
	out (2),a
	ld a,10
	out (2),a

	;ToDo: re-load CP/M

	ld hl,firstBytes
	ld de,0
	ld c,100 ;7
fbloop:
	ld a,(hl)
	ld (de),a
	inc hl
	inc de
	dec c
	jp nz,fbloop
	
	ld c,0
	jp $3400+$A800

const:
	in a,(0)
	ret

conin:
	in a,(0)
	cp $ff
	jp nz,conin

	in a,(1)
	ret

conout:
	push af
	ld a,c
	out (2),a
	pop af
	ret

list:
	ret

punch:
	ret

reader:
	ld a,$1F
	ret

home:
	push af
	ld a,0
	out (16),a
	pop af
	ret

seldsk:
	push af
	ld a,c
	cp 0
	jp nz,seldsk_na
	ld hl,dph
	pop af
	ret
seldsk_na:
	ld hl,0
	pop af
	ret

settrk:
	push af
	ld a,c
	out (16),a
	pop af
	ret

setsec:
	push af
	ld a,c
	out (18),a
	pop af
	ret

setdma:
	push af
	ld a,c
	out (20),a
	ld a,b
	out (21),a
	pop af
	ret

read:
	ld a,1
	out (22),a
	ld a,0
	ret

write:
	ld a,2
	out (22),a
	ld a,0
	ret

listst:
	ld a,0
	ret

sectran:
	;translate sector bc using table at de, res into hl
	;not implemented :)
	ld h,b
	ld l,c
	ret


firstBytes:
	jp $4A03+$A800	;JMP WBOOT
	db 0			;IOBYTE
	db 0			;user:drive
	jp $3C06+$A800	;JMP BDOS

;Disk Parameter Header
dph:
	dw trans ;XLT: Address of translation table
	dw 0 ;000: Scratchpad
	dw 0 ;000: Scratchpad
	dw 0 ;000: Scratchpad
	dw dirbuf ;DIRBUF: Address of a dirbuff scratchpad
	dw dpb ;DPB: Address of a disk parameter block
	dw chk ;CSV: Address of scratchpad area for changed disks
	dw all ;ALV: Address of an allocation info sratchpad

dpb:
	dw 26 ;SPT: sectors per track
	db 3 ;BSH: data allocation block shift factor
	db 7 ;BLM: Data Allocation Mask
	db 0 ;Extent mask
	dw 242 ;DSM: Disk storage capacity
	dw 63 ;DRM, no of directory entries
	db 192 ;AL0
	db 0 ;AL1
	dw 16 ;CKS, size of dir check vector
	dw 2 ;OFF, no of reserved tracks

trans:
	db 0,1,2,3,4,5,6,7,8,9
	db 10,11,12,13,14,15,16,17,18,19
	db 19,20,21,22,23,24,25,26

dirbuf:
	ds 128

chk:
	ds 16

all:
	ds 31

end