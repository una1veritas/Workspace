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

org $2000
	; IPL for the CP/M-emu in an AVR. Loads CPM from the 'disk' from
	; track 0 sector 2 to track 1 sector 26.

	ld sp,$1000
	
	call printipl


	ld b,49
	ld de,$0001
	ld hl,$3400+$A800
loadloop:
	ld a,d ;track
	out (16),a
	ld a,e ; sector (base-0)
	out (18),a
	ld a,l ;dma L
	out (20),a
	ld a,h ;dma H
	out (21),a
	ld a,1
	out (22),a

	push bc
	ld bc,$80
	add hl,bc
	pop bc

	inc e
	ld a,e
	cp 26
	jp nz,noNextTrack

	inc d
	ld e,0

noNextTrack:

	dec b
	jp nz,loadloop

	jp $4A00+$A800

printipl:
	ld a,'i'
	out (2),a
	ld a,'p'
	out (2),a
	ld a,'l'
	out (2),a
	ld a,13
	out (2),a
	ld a,10
	out (2),a
	ret

end