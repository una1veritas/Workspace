/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2017 by Udo Munk
 *
 * This module implements banked memory management for cpmsim
 *
 *      MMU:
 *      ===
 *
 *      +--------+
 * 16KB | common |
 *      +--------+
 *      +--------+  +--------+  ..........  +--------+
 *      |        |  |        |              |        |
 * 48KB |        |  |        |  ..........  |        |
 *      | bank 0 |  | bank 1 |              | bank n |
 *      +--------+  +--------+  ..........  +--------+
 *
 * This is an example for 48KB segments as it was implemented originally.
 * The segment size now can be configured via port 22.
 * If the segment size isn't configured the default is 48 KB as it was
 * before, to maintain compatibility.
 *
 * History:
 * 22-NOV-16 stuff moved to here for further improvements
 * 03-FEB-17 added ROM initialisation
 */

#define MAXSEG 16		/* max. number of memory banks */
#define SEGSIZ 49152		/* default size of one bank = 48 KBytes */

extern void init_memory(void), init_rom(void);

extern BYTE *memory[];
extern int selbnk, maxbnk, segsize, wp_common;

/*
 * memory access for the CPU cores
 */
static inline void memwrt(WORD addr, BYTE data)
{
	if ((addr >= segsize) && (wp_common != 0))
		return;

	if (selbnk == 0) {
		*(memory[0] + addr) = data;
	} else {
		if (addr >= segsize)
			*(memory[0] + addr) = data;
		else
			*(memory[selbnk] + addr) = data;
	}
}

static inline BYTE memrdr(WORD addr)
{
	if (selbnk == 0)
		return(*(memory[0] + addr));

	if (addr >= segsize)
		return(*(memory[0] + addr));
	else
		return(*(memory[selbnk] + addr));
}

/*
 * memory access for DMA devices
 */
static inline void dma_write(WORD addr, BYTE data)
{
	if ((addr >= segsize) && (wp_common != 0))
		return;

	if (selbnk == 0) {
		*(memory[0] + addr) = data;
	} else {
		if (addr >= segsize)
			*(memory[0] + addr) = data;
		else
			*(memory[selbnk] + addr) = data;
	}
}

static inline BYTE dma_read(WORD addr)
{
	if (selbnk == 0)
		return(*(memory[0] + addr));

	if (addr >= segsize)
		return(*(memory[0] + addr));
	else
		return(*(memory[selbnk] + addr));
}

/*
 * return memory base pointer for the simulation frame
 */
static inline BYTE *mem_base(void)
{
	return(memory[0]);
}
