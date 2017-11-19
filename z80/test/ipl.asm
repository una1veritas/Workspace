;    CP/M IPL for avrcpm
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

CSTATUS:	equ $00
CIN:	equ $01
COUT:	equ $02
	
READ_FUNC_BIT:  equ	7
WRITE_FUNC_BIT: equ	6
BOOT_FUNC_BIT:  equ 5
HOME_FUNC_BIT:  equ 4

	org $0000
boot:	
	ld sp,$0100
	jp $0100

	ds  $32,0
	org $0038
rst38h:	
	ret

	ds  $c7,0

	org $0100
	; IPL for the CP/M-emu in an AVR. Loads CPM from the 'disk' from
	; track 0 sector 2 to track 1 sector 26.
start_ipl:	
	ld hl, IPLMESSAGE
	call print_str

	ld b,49			; 
	ld de,$0001		; track (d) & sector (e) counter, begin with 1
	ld hl,$3400+$A800
	
diskread:
	ld a,d ;track
	out (16),a
	ld a,e ; sector (base-0)
	out (18),a
	ld a,l ;dma L
	out (20),a
	ld a,h ;dma H
	out (21),a
	ld a,1<<READ_FUNC_BIT
	out (22),a

	push 	bc
	ld	bc, 128
	add 	hl,bc 		; increase 128 
	pop 	bc

	inc 	e		; increase sector No.
	ld 	a,e
	cp 	26		; sectors in a track
	jr 	nz, next_sector
	inc 	d		; move to next track
	ld 	e,0		; and the first sector
next_sector
	dec b
	jr nz,diskread

	ld hl, DONEMESSAGE
	call print_str
exit_ipl:	
	jp $4A00+$A800

print_str:
	ld a,(hl)
	and a
	ret z
	out (COUT), a
	inc hl
	jr print_str

stop:
	halt

IPLMESSAGE:	
	dm 13,10,"IPL loading...",13,10,0
DONEMESSAGE:
	dm "done.",13,10,0
	
end.
