/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 2024 Thomas Eberhardt
 */

#ifndef SIMPORT_INC
#define SIMPORT_INC

#include "sim.h"
#include "simdefs.h"

extern void sleep_for_us(unsigned long time);
extern void sleep_for_ms(unsigned time);
extern uint64_t get_clock_us(void);
#ifdef WANT_ICE
extern bool get_cmdline(char *buf, int len);
#endif

#endif /* !SIMPORT_INC */
