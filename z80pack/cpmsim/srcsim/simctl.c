/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2024 by Udo Munk
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simcore.h"
#include "simmem.h"
#include "simio.h"
#include "simctl.h"
#ifdef WANT_ICE
#include "simice.h"
#endif

#include "unix_terminal.h"

#include "log.h"
static const char *TAG = "system";

/*
 *	This function initializes the terminal, loads boot code
 *	and then the Z80 CPU emulation is started.
 */
void mon(void)
{
	/* load boot code into memory */
	if (boot(0))
		exit(EXIT_FAILURE);

	/* empty buffer for teletype */
	fflush(stdout);

#ifdef WANT_ICE
	ice_before_go = set_unix_terminal;
	ice_after_go = reset_unix_terminal;
	atexit(reset_unix_terminal);

	ice_cmd_loop(0);
#else
	/* initialize terminal */
	set_unix_terminal();
	atexit(reset_unix_terminal);

	/* start CPU emulation */
	run_cpu();

	/* reset terminal */
	reset_unix_terminal();

	/* check for CPU emulation errors and report */
	report_cpu_error();
	report_cpu_stats();
#endif
}

/*
 *	boot from a saved core image, a boot file or from
 *	first sector of disk drive A:
 */
int boot(int level)
{
	register int i;
	int fd;
	struct stat sbuf;
	static BYTE buf[128];
	static char fn[MAX_LFN];

	LOG(TAG, "\r\nBooting...\r\n\r\n");

	/* on first boot we can run from core or file */
	if (level == 0) {
		if (l_flag)
			return 0;
		if (x_flag)
			return 0;
	}

	/* else load boot code from disk */

	/* if option -d is used disks are there */
	if (diskdir != NULL) {
		strcpy(fn, diskd);
	} else {
		/* if not first try ./disks */
		if ((stat("./disks", &sbuf) == 0) && S_ISDIR(sbuf.st_mode)) {
			strcpy(fn, "./disks");
		} else {
			/* nope, then DISKSDIR as set in Makefile */
			strcpy(fn, DISKSDIR);
		}
	}

	strcat(fn, "/");
	strcat(fn, disks[0].fn);

	if ((fd = open(fn, O_RDONLY)) == -1) {
		LOGE(TAG, "can't open file %s", fn);
		close(fd);
		return 1;
	}
	if (read(fd, buf, 128) != 128) {
		LOGE(TAG, "can't read file %s", fn);
		close(fd);
		return 1;
	}
	close(fd);

	for (i = 0; i < 128; i++)
		putmem(i, buf[i]);

	return 0;
}
