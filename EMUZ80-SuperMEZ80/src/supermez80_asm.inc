;	Common definitions for Z80-Simulator
;
;	Copyright (C) 1988-2007 by Udo Munk
;
	.include "config_asm.inc"
;
;	SuperMEZ80 I/O ports (PIC)
;
CONSTA	EQU	PIC_IOBASE+0		;console status port
CONDAT	EQU	PIC_IOBASE+1		;console data port
PRTSTA	EQU	PIC_IOBASE+2		;printer status port
PRTDAT	EQU	PIC_IOBASE+3		;printer data port
AUXDAT	EQU	PIC_IOBASE+5		;auxiliary data port
FDCDAT	EQU	PIC_IOBASE+8		;fdc-port: # of drive
FDCD	EQU	PIC_IOBASE+10		;fdc-port: # of drive
FDCT	EQU	PIC_IOBASE+11		;fdc-port: # of track
FDCS	EQU	PIC_IOBASE+12		;fdc-port: # of sector
FDCOP	EQU	PIC_IOBASE+13		;fdc-port: command
	D_DMA_READ	EQU	0
	D_DMA_WRITE	EQU	1
	D_READ		EQU	2
	D_WRITE		EQU	3
FDCST	EQU	PIC_IOBASE+14		;fdc-port: status
	D_SUCCESS	EQU	0
	D_ERROR		EQU	1
DMAL	EQU	PIC_IOBASE+15		;dma-port: dma address low
DMAH	EQU	PIC_IOBASE+16		;dma-port: dma address high
MMU_INIT	EQU	PIC_IOBASE+20	;14h MMU initialisation
MMU_BANK_SEL	EQU	PIC_IOBASE+21	;15h MMU bank select
MMU_SEG_SIZE	EQU	PIC_IOBASE+22	;16h MMU select segment size (in pages a 256 bytes)
MMU_WR_PROT	EQU	PIC_IOBASE+23	;17h MMU write protect/unprotect common memory segment
;
MON_CLEANUP	EQU	PIC_IOBASE+26	;1Ah clean up monitor mode
MON_PREPARE	EQU	PIC_IOBASE+27	;1Bh prepare monitor mode
MON_ENTER	EQU	PIC_IOBASE+28	;1Ch enter monitor mode
TGTINV_TRAP	EQU	PIC_IOBASE+29	;1Dh return from target CPU invocation
MON_PREPARE_NMI	EQU	PIC_IOBASE+30	;1Eh prepare monitor mode for NMI
MON_ENTER_NMI	EQU	PIC_IOBASE+31	;1Fh enter monitor mode for NMI
;
;	Z8S180 I/O ports
;
IOBASE		EQU	00h
UART_RX		EQU	IOBASE+08h	; UART DATA REGISTOR
UART_TX		EQU	IOBASE+06h
UARTCR_180	EQU	IOBASE+04h	; UART CONTROL REGISTOR
OMCR_V		EQU	IOBASE+0E0h
