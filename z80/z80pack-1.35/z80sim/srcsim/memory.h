/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 2016-2017 by Udo Munk
 *
 * This module implements memory management for z80sim
 *
 * History:
 * 22-NOV-16 stuff moved to here for further improvements
 * 03-FEB-17 added ROM initialisation
 */

extern void init_memory(void), init_rom(void);
extern BYTE memory[];

/*
 * memory access for the CPU cores
 */
#define memwrt(addr, data) (memory[addr] = data)
#define memrdr(addr) (memory[addr])

/*
 * memory access for DMA devices
 */
#define dma_write(addr, data) (memory[addr] = data)
#define dma_read(addr) (memory[addr])

/*
 * return memory base pointer for the simulation frame
 */
#define mem_base() (&memory[0])
