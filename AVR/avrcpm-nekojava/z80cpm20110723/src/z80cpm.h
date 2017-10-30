/*----------------------------------------------------------------------*/
/* CP/M system by Z80CPU and AVR                           2011.6.11    */
/*                                         definitions                  */
/*                                                         neko Java    */
/*----------------------------------------------------------------------*/

#include <avr/io.h>

/* pin assignment ------------------------- */
#define P_RESET	PB0
#define P_WAIT	PB1
#define P_BUSRQ	PB2
#define P_WR	PD3
#define P_RD	PD4
#define P_BUSAK	PD5
#define P_IOREQ	PD6
#define P_CK	PD7

/* pin output ----------------------------- */
#define RESET_H()	PORTB |=  _BV(P_RESET)
#define RESET_L()	PORTB &= ~_BV(P_RESET)
#define WAIT_H()	PORTB |=  _BV(P_WAIT)
#define WAIT_L()	PORTB &= ~_BV(P_WAIT)
#define BUSRQ_H()	PORTB |=  _BV(P_BUSRQ)
#define BUSRQ_L()	PORTB &= ~_BV(P_BUSRQ)
#define WR_H()		PORTD |=  _BV(P_WR)
#define WR_L()		PORTD &= ~_BV(P_WR)
#define RD_H()		PORTD |=  _BV(P_RD)
#define RD_L()		PORTD &= ~_BV(P_RD)
#define CK_H()		PORTD |=  _BV(P_CK)
#define CK_L()		PORTD &= ~_BV(P_CK)

/* pin input ----------------------------- */
#define WR()		(PIND & _BV(P_WR))
#define RD()		(PIND & _BV(P_RD))
#define BUSAK()		(PIND & _BV(P_BUSAK))
#define IOREQ()		(PIND & _BV(P_IOREQ))

/* pin direction control ------------------------------------------------------------ */
#define RW_IN()		PORTD |= _BV(P_WR) | _BV(P_RD); DDRD &= ~(_BV(P_WR) | _BV(P_RD))
#define RW_OUT()	PORTD |= _BV(P_WR) | _BV(P_RD); DDRD |= _BV(P_WR) | _BV(P_RD)
#define D_BUS_IN()	DDRC = 0; PORTC = 0xFF
#define D_BUS_OUT()	DDRC = 0xFF
#define AD_BUS_IN()	DDRA = 0; PORTA = 0xFF
#define AD_BUS_OUT()	DDRA = 0xFF

/* DMA mode or result ------------------------------ */
#define DMA_READ	1
#define DMA_WRITE	2
#define DMA_OK		0
#define DMA_NG		1
#define DMA_WRITE_BACK	3

/* Virtual I/O port assignment [AVR side]-------------------------------------------- */
#define CON_STS		0	//[O] Returns 0xFF if the UART has a byte, 0 otherwise.
#define CON_IN		1	//[O]
#define CON_OUT		2	//[I]
#define TRACK_SEL_L	16	//[I]
#define TRACK_SEL_H	17	//[I]
#define SECTOR_SEL	18	//[I]
#define ADR_L		20	//[I]
#define ADR_H		21	//[I]
#define EXEC_DMA	22	//[I] command 1:read, 2:write.
#define DMA_RS		23	//[O] 0:OK, 1:NG.

/* Disk parameters ------------------------------------------------------------------- */
#define SDC_CLST_SIZE		512	// fixed for SDC.
#define SECT_SIZE		128	// fixed for CP/M.
#define SECT_CNT		26	// CP/M sector size.
#define BLOCK_SIZE		1024	// CP/M block size.
#define CPM_CLST_CNT_PER_BLOCK	(BLOCK_SIZE/SECT_SIZE)
#define SDC_CLST_CNT_PER_BLOCK	(BLOCK_SIZE/SDC_CLST_SIZE)

#define EEPROM_SIZE	2048
