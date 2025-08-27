/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Common I/O devices used by various simulated machines
 *
 * Copyright (C) 2019 Mike Douglas
 *
 * Emulation of the Mostek FLP-80 Floppy Disk Controller
 *		(a WD1771 based FDC)
 *
 * History:
 * 15-SEP-2019 (Mike Douglas) created from tarbell_fdc.h
 * 28-SEP-2019 (Udo Munk) use logging
 */

#ifndef MOSTEK_FDC_INC
#define MOSTEK_FDC_INC

#include "sim.h"
#include "simdefs.h"

extern BYTE fdcBoard_stat_in(void), fdcBoard_ctl_in(void);
extern BYTE fdc1771_stat_in(void), fdc1771_track_in(void);
extern BYTE fdc1771_sec_in(void), fdc1771_data_in(void);

extern void fdcBoard_ctl_out(BYTE data);
extern void fdc1771_cmd_out(BYTE data), fdc1771_track_out(BYTE data);
extern void fdc1771_sec_out(BYTE data), fdc1771_data_out(BYTE data);

extern void fdc_reset(void);

#endif /* !MOSTEK_FDC_INC */
