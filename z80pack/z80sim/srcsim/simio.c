/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2024 by Udo Munk
 *
 * This module of the simulator contains a simple terminal I/O
 * simulation as an example.
 *
 * History:
 * 27-MAY-2024 moved io_in & io_out to simcore
 */

/*
 *	Port 0 input:	reads status of stdin
 *	Port 1 input:	reads the next byte from stdin
 *	Port 160 input:	hardware control lock status
 *	Port 254 input:	mirror of port 255
 *	Port 255 input:	read from front panel switches
 *	Port 1 output:	writes the byte to stdout
 *	Port 160 output: hardware control
 *	Port 254 output: write to front panel switches
 *	Port 255 output: write to front panel lights (dummy)
 *
 *	All the other ports are connected to an I/O-trap handler,
 *	I/O to this ports stops the simulation with an I/O error.
 */

#include <stdio.h>
#include <sys/poll.h>

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simcore.h"
#include "simio.h"

/*
 *	Forward declarations of the I/O functions
 *	for all port addresses.
 */
static BYTE p000_in(void), p001_in(void), p255_in(void);
static BYTE hwctl_in(void);
static void p001_out(BYTE data), p254_out(BYTE data);
static void hwctl_out(BYTE data), fp_out(BYTE data);

static BYTE hwctl_lock = 0xff;	/* lock status hardware control port */
static BYTE sio_last;		/* last byte read from sio */
static BYTE fp_value;		/* port 255 value, can be set with p command */

/*
 *	This array contains function pointers for every input
 *	I/O port (0 - 255), to do the required I/O.
 */
in_func_t *const port_in[256] = {
	[  0] = p000_in,
	[  1] = p001_in,
	[160] = hwctl_in,	/* virtual hardware control */
	[254] = p255_in,	/* mirror of port 255 */
	[255] = p255_in		/* read from front panel switches */
};

/*
 *	This array contains function pointers for every output
 *	I/O port (0 - 255), to do the required I/O.
 */
out_func_t *const port_out[256] = {
	[  1] = p001_out,
	[160] = hwctl_out,	/* virtual hardware control */
	[254] = p254_out,	/* write to front panel switches */
	[255] = fp_out		/* write to front panel lights (dummy) */
};

/*
 *	This function is to initiate the I/O devices.
 *	It will be called from the CPU simulation before
 *	any operation with the CPU is possible.
 *
 *	See the I/O simulation of of the other systems
 *	for more complex examples.
 */
void init_io(void)
{
}

/*
 *	This function is to stop the I/O devices. It is
 *	called from the CPU simulation on exit.
 *
 *	Nothing to do here, see the I/O simulation
 *	of CP/M for a more complex example.
 */
void exit_io(void)
{
}

/*
 *	I/O function port 0 read:
 *	Read status from stdin:
 *	bit 0 = 0, character available for input from stdin
 *	bit 7 = 0, transmitter ready to write character to stdout
 *		   always true
 */
static BYTE p000_in(void)
{
	struct pollfd p[1];
	register BYTE tty_stat = 0x01;

	p[0].fd = fileno(stdin);
	p[0].events = POLLIN;
	p[0].revents = 0;
	poll(p, 1, 0);
	if (p[0].revents & POLLIN)
		tty_stat &= ~1;
	if (p[0].revents & POLLNVAL) {
		cpu_error = IOERROR;
		cpu_state = ST_STOPPED;
	}
	return tty_stat;
}

/*
 *	I/O function port 1 read:
 *	Read next byte from stdin.
 */
static BYTE p001_in(void)
{
	struct pollfd p[1];

	p[0].fd = fileno(stdin);
	p[0].events = POLLIN;
	p[0].revents = 0;
	poll(p, 1, 0);
	if (!(p[0].revents & POLLIN))
		return sio_last; /* someone reads without checking status */
	else {
		sio_last = getchar();
		return sio_last;
	}
}

/*
 *	I/O handler for read hardware control
 *	returns lock status of the port
 */
static BYTE hwctl_in(void)
{
	return hwctl_lock;
}

/*
 *	I/O function port 255 read:
 *	return virtual front panel switches state
 */
static BYTE p255_in(void)
{
	return fp_value;
}

/*
 *	I/O function port 1 write:
 *	Write byte to stdout and flush the output.
 */
static void p001_out(BYTE data)
{
	putchar((int) data & 0x7f); /* strip parity, some software won't */
	fflush(stdout);
}

/*
 *	Port is locked until magic number 0xaa is received!
 *
 *	I/O handler for write hardware control after unlocking:
 *
 *	bit 4 = 1	switch CPU model to 8080
 *	bit 5 = 1	switch CPU model to Z80
 *	bit 6 = 1	reset CPU
 *	bit 7 = 1	halt emulation via I/O
 */
static void hwctl_out(BYTE data)
{
	/* if port is locked do nothing */
	if (hwctl_lock && (data != 0xaa))
		return;

	/* unlock port ? */
	if (hwctl_lock && (data == 0xaa)) {
		hwctl_lock = 0;
		return;
	}

	/* process output to unlocked port */
	/* but first lock port again */
	hwctl_lock = 0xff;

	if (data & 128) {	/* halt system */
		cpu_error = IOHALT;
		cpu_state = ST_STOPPED;
		return;
	}

	if (data & 64) {	/* reset system */
		hwctl_lock = 0xff; /* lock hardware control port */
		reset_cpu();	/* reset CPU */
		return;
	}

#if !defined (EXCLUDE_I8080) && !defined(EXCLUDE_Z80)
	if (data & 32) {	/* switch cpu model to Z80 */
		switch_cpu(Z80);
		return;
	}

	if (data & 16) {	/* switch cpu model to 8080 */
		switch_cpu(I8080);
		return;
	}
#endif
}

/*
 *	This allows to set the virtual front panel switches with ICE p command
 */
static void p254_out(BYTE data)
{
	fp_value = data;
}

/*
 *	Write output to front panel lights (dummy)
 */
static void fp_out(BYTE data)
{
	UNUSED(data);
}
