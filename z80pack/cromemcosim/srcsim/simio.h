/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 2014-2022 Udo Munk
 */

#ifndef SIMIO_INC
#define SIMIO_INC

#include "sim.h"
#include "simdefs.h"

#include "unix_network.h"

#define IO_DATA_UNUSED	0xff	/* data returned on unused ports */

extern int lpt1, lpt2;

extern net_connector_t ncons[NUMNSOC];

extern in_func_t *const port_in[256];
extern out_func_t *const port_out[256];

extern void init_io(void);
extern void exit_io(void);
extern void reset_io(void);

#ifdef WANT_ICE
extern void ice_go(void);
extern void ice_break(void);
#endif

#endif /* !SIMIO_INC */
