/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2024 Udo Munk
 * Copyright (C) 2021 David McNaughton
 */

#ifndef SIMFUN_INC
#define SIMFUN_INC

#include "sim.h"
#include "simdefs.h"

/*
 *	The following code can be used as a simport.h file for a machine
 *	running on an OS using the POSIX API. The implementations are
 *	included in simfun.c:
 *
 *	#include "sim.h"
 *	#include "simdefs.h"
 *
 *	extern void sleep_for_us(unsigned long time);
 *	extern void sleep_for_ms(unsigned time);
 *	extern uint64_t get_clock_us(void);
 *	#ifdef WANT_ICE
 *	extern bool get_cmdline(char *buf, int len);
 *	#endif
 */

extern bool load_file(char *fn, WORD start, int size);

#endif /* !SIMFUN_INC */
