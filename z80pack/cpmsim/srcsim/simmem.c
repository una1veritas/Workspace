/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2018 by Udo Munk
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
 * 21-DEC-2016 moved banked memory implementation to here
 * 03-FEB-2017 added ROM initialization
 * 09-APR-2018 modified MMU write protect port as used by Alan Cox for FUZIX
 */

#include <stdlib.h>

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simmem.h"

#include "log.h"
static const char *TAG = "memory";

BYTE *memory[MAXSEG];		/* MMU with pointers to the banks */
int selbnk;			/* current selected bank */
int maxbnk;			/* number of allocated banks */
int segsize = SEGSIZ;		/* segment size of banks, default 48KB */
int wp_common;			/* write protect/unprotect common segment */

void init_memory(void)
{
	register int i;

	/* allocate the first 64KB bank, so that we have some memory */
	if ((memory[0] = (BYTE *) malloc(65536)) == NULL) {
		LOGE(TAG, "can't allocate memory for bank 0");
		cpu_error = IOERROR;
		cpu_state = ST_STOPPED;
		return;
	}
	maxbnk = 1;
	selbnk = 0;

	/* fill memory content of bank 0 with some initial value */
	if (m_value >= 0) {
		for (i = 0; i < 65536; i++)
			putmem(i, m_value);
	} else {
		for (i = 0; i < 65536; i++)
			putmem(i, (BYTE) (rand() % 256));
	}
}
