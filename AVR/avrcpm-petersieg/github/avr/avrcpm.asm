;    Z80 emulator with CP/M support. The Z80-specific instructions themselves 
;    actually aren't implemented yet, making this more of an i8080 emulator.
;    This is the main file, glueing all parts together.

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
;    $Id: avrcpm.asm 182 2012-03-19 21:28:26Z leo $
;

.nolist
#define atmega328P
#if defined atmega8
	.include "m8def.inc"
#elif defined atmega168
	.include "m168def.inc"
#elif defined atmega328P
	.include "m328Pdef.inc"
#else                               /* default */
	.include "m88def.inc"
#endif
	.include "config.inc"
	.include "svnrev.inc"
	.include "macros.inc"
#if DRAM_8BIT	/* Implies software uart */
	.include "dram-8bit.inc"
#else
	.include "dram-4bit.inc"
#endif
	.list
	.cseg

	.org 0
	rjmp start		; reset vector
    	
	.org INT_VECTORS_SIZE

	.include "init.asm"
#if DRAM_8BIT	/* Implies software uart */
	.include "sw-uart.asm"
	.include "dram-8bit.asm"
#else	/* 4 bit RAM, hardware uart */
	.include "hw-uart.asm"
	.include "dram-4bit.asm"
#endif
	.include "dram-refresh.asm"
	.include "timer.asm"
	.include "utils.asm"
	.include "heap.asm"
	.include "mmc.asm"
;	.include "mmc-old.asm"

; >>>-------------------------------------- Virtual Devices
	.include "virt_ports.asm"	; Virtual Ports for BIOS
; <<<-------------------------------------- Virtual Devices

; >>>-------------------------------------- File System Management
	.include "dsk_fsys.asm"		; Basic Filesystem definitions
	.include "dsk_mgr.asm"		; Disk- Manager
	.include "dsk_cpm.asm"		; CPM- Disk Interaktion
	.include "dsk_fat16.asm"	; FAT16-DISK Interaktion
	.include "dsk_ram.asm"		; RAM- Disk Interaktion
; <<<-------------------------------------- File System Management
;	.include "8080int-orig.asm"	;Old 8080 interpreter.
;	.include "8080int.asm"		;New 8080 interpreter.
;	.include "8080int-t3.asm"	;Another 8080 interpreter
;	.include "8080int-t3-jmp.asm"	;Can't get enough
;	.include "8080int-jmp.asm"	;
	.include "Z80int-jmp.asm"	;


	.dseg
ramtop:	.byte	0
	
	.cseg

; vim:set ts=8 noet nowrap

