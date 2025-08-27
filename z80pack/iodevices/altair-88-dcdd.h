/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Common I/O devices used by various simulated machines
 *
 * Copyright (C) 2018-2019 by Udo Munk
 *
 * Emulation of the MITS Altair S100 floppy disk controller
 *
 * History:
 * 10-AUG-2018 first version, runs CP/M 1.4 & 2.2 & disk BASIC
 * 02-DEC-2019 use disk names different from Tarbell controller
 */

#ifndef ALTAIR_88_DCDD_INC
#define ALTAIR_88_DCDD_INC

#include "sim.h"
#include "simdefs.h"

extern void altair_dsk_select_out(BYTE data);
extern void altair_dsk_control_out(BYTE data);
extern void altair_dsk_data_out(BYTE data);
extern BYTE altair_dsk_status_in(void);
extern BYTE altair_dsk_sec_in(void);
extern BYTE altair_dsk_data_in(void);
extern void altair_dsk_reset(void);

#endif /* !ALTAIR_88_DCDD_INC */
