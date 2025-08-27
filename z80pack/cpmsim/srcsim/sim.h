/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2025 by Udo Munk
 */

#ifndef SIM_INC
#define SIM_INC

/*
 *	The following defines may be activated, commented or modified
 *	by user for her/his own purpose.
 */
#define DEF_CPU Z80	/* default CPU (Z80 or I8080) */
#define CPU_SPEED 0	/* default CPU speed 0=unlimited */
/*#define ALT_I8080*/	/* use alt. 8080 sim. primarily optimized for size */
/*#define ALT_Z80*/	/* use alt. Z80 sim. primarily optimized for size */
#define UNDOC_INST	/* compile undocumented instrs. (required by ALT_*) */
#ifndef EXCLUDE_Z80
#define FAST_BLOCK	/* much faster but not accurate Z80 block instr. */
#endif

/*#define WANT_ICE*/	/* attach ICE to machine */
#ifdef WANT_ICE
/*#define WANT_TIM*/	/* don't count t-states */
/*#define HISIZE  1000*//* no history */
/*#define SBSIZE  10*/	/* no software breakpoints */
/*#define WANT_HB*/	/* no hardware breakpoint */
#endif

#define HAS_DISKS	/* uses disk images */
/*#define HAS_CONFIG*/	/* has no configuration file */

#define PIPES		/* use named pipes for auxiliary device */
#define NETWORKING	/* TCP/IP networked serial ports */
#define NUMSOC	4	/* number of server sockets */
#define TCPASYNC	/* tcp/ip server can use async I/O */
/*#define CNETDEBUG*/	/* client network protocol debugger */
/*#define SNETDEBUG*/	/* server network protocol debugger */

/*
 *	The following defines may be modified and activated by
 *	user, to print her/his copyright for a simulated system,
 *	which contains the Z80/8080 CPU emulations as a part.
 */
/*
#define USR_COM	"XYZ-System Simulation"
#define USR_REL	"x.y"
#define USR_CPR	"Copyright (C) 20xx by XYZ"
*/

#endif /* !SIM_INC */
