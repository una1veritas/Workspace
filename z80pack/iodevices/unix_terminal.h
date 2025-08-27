/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Common I/O devices used by various simulated machines
 *
 * Copyright (C) 2008-2017 by Udo Munk
 *
 * This module contains initialization and reset functions for
 * the POSIX/BSD line discipline, so that stdin/stdout can be used
 * as terminal for ancient machines.
 *
 * History:
 * 24-SEP-2008 first version finished
 * 16-JAN-2014 discard input at reset
 * 15-APR-2014 added some more c_cc's used on BSD systems
 * 24-FEB-2017 set line discipline only if fd 0 is a tty
 */

#ifndef UNIX_TERMINAL_INC
#define UNIX_TERMINAL_INC

#include <termios.h>

extern struct termios old_term, new_term;

extern void set_unix_terminal(void);
extern void reset_unix_terminal(void);

#endif /* !UNIX_TERMINAL_INC */
