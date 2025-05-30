/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 2008-2017 by Udo Munk
 *
 * Configuration for an Altair 8800 system
 *
 * History:
 * 20-OCT-08 first version finished
 * 02-MAR-14 source cleanup and improvements
 * 14-MAR-14 added Tarbell SD FDC and printer port
 * 23-MAR-14 enabled interrupts, 10ms timer added to iosim
 * xx-JUN-14 added default CPU define
 * 19-JUL-14 added typedef for signed 16bit
 * 29-APR-15 added Cromemco DAZZLER to the machine
 * 11-AUG-16 implemented memwrt as function to support ROM
 * 01-DEC-16 implemented memrdr to separate memory from CPU
 * 06-DEC-16 implemented status display and stepping for all machine cycles
 * 12-JAN-17 improved configuration and front panel LED timing
 * 26-FEB-17 added Processor Technology VDM-1 to the machine
 * 27-MAR-17 added SIO's connected to UNIX domain sockets
 */

/*
 *	The following defines may be activated, commented or modified
 *	by user for her/his own purpose.
 */
#define CPU_SPEED 2	/* default CPU speed */
#define Z80_UNDOC	/* compile undocumented Z80 instructions */
/*#define WANT_FASTM*/	/* much faster but not accurate Z80 block moves */
/*#define WANT_TIM*/	/* don't count t-states */
/*#define HISIZE  1000*//* no history */
/*#define SBSIZE  10*/	/* no breakpoints */
#define FRONTPANEL	/* emulate a machines frontpanel */
#define BUS_8080	/* emulate 8080 bus status for front panel */

#define HAS_DISKS	/* uses disk images */
#define HAS_CONFIG	/* has configuration files somewhere */

#define NUMNSOC 0	/* number of TCP/IP sockets for SIO connections */
#define NUMUSOC 2	/* number of UNIX sockets for SIO connections */

/*
 *	Default CPU
 */
#define Z80		1
#define I8080		2
#define DEFAULT_CPU	I8080

/*
 *	The following lines of this file should not be modified by user
 */
#define COPYR	"Copyright (C) 1987-2017 by Udo Munk"
#define RELEASE	"1.35"

#define USR_COM	"Altair 8800 Simulation"
#define USR_REL	"1.16"
#define USR_CPR	"Copyright (C) 2008-2017 by Udo Munk"

#define LENCMD		80		/* length of command buffers etc */

#define S_FLAG		128		/* bit definitions of CPU flags */
#define Z_FLAG		64
#define N2_FLAG		32
#define H_FLAG		16
#define N1_FLAG		8
#define P_FLAG		4
#define N_FLAG		2
#define C_FLAG		1

#define CPU_MEMR	128		/* bit definitions for CPU bus status */
#define CPU_INP		64
#define CPU_M1		32
#define CPU_OUT		16
#define CPU_HLTA	8
#define CPU_STACK	4
#define CPU_WO		2
#define CPU_INTA	1

					/* operation of simulated CPU */
#define STOPPED		0		/* stopped */
#define CONTIN_RUN	1		/* continual run */
#define SINGLE_STEP	2		/* single step */
#define RESET		4		/* reset */

					/* error codes */
#define NONE		0		/* no error */
#define OPHALT		1		/* HALT op-code trap */
#define IOTRAPIN	2		/* I/O trap input */
#define IOTRAPOUT	3		/* I/O trap output */
#define IOHALT		4		/* halt system via I/O register */
#define IOERROR		5		/* fatal I/O error */
#define OPTRAP1		6		/* illegal 1 byte op-code trap */
#define OPTRAP2		7		/* illegal 2 byte op-code trap */
#define OPTRAP4		8		/* illegal 4 byte op-code trap */
#define USERINT		9		/* user interrupt */
#define POWEROFF	255		/* CPU off, no error */

typedef unsigned short WORD;		/* 16 bit unsigned */
typedef signed short   SWORD;		/* 16 bit signed */
typedef unsigned char  BYTE;		/* 8 bit unsigned */

#ifdef HISIZE
struct history {			/* structure of a history entry */
	WORD	h_adr;			/* address of execution */
	WORD	h_af;			/* register AF */
	WORD	h_bc;			/* register BC */
	WORD	h_de;			/* register DE */
	WORD	h_hl;			/* register HL */
	WORD	h_ix;			/* register IX */
	WORD	h_iy;			/* register IY */
	WORD	h_sp;			/* register SP */
};
#endif

#ifdef SBSIZE
struct softbreak {			/* structure of a breakpoint */
	WORD	sb_adr;			/* address of breakpoint */
	BYTE	sb_oldopc;		/* op-code at address of breakpoint */
	int	sb_passcount;		/* pass counter of breakpoint */
	int	sb_pass;		/* no. of pass to break */
};
#endif

#ifndef isxdigit
#define isxdigit(c) ((c<='f'&&c>='a')||(c<='F'&&c>='A')||(c<='9'&&c>='0'))
#endif
