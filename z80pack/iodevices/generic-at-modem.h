/**
 * generic-at-modem.h
 *
 * Emulation of generic 'AT' modem over TCP/IP sockets (telnet)
 *
 * Copyright (C) 2019-2021 by David McNaughton
 *
 * History:
 * 12-SEP-2019	1.0	Initial Release
 * 29-SEP-2019	1.1	Added Answer modes and registers
 * 20-OCT-2019	1.2	Added Telnet handler
 * 23-OCT-2019	1.3	Put Telnet protocol under modem register control
 * 16-JUL-2020	1.4	fix bug/warning detected with gcc 9
 * 17-JUL-2020	1.5	Added/Updated AT$ help, ATE, ATQ, AT&A1 cmds, MODEM.init string
 */

#ifndef GENERIC_AT_MODEM_INC
#define GENERIC_AT_MODEM_INC

extern bool modem_device_alive(int i);
extern int modem_device_poll(int i);
extern int modem_device_get(int i);
extern void modem_device_send(int i, char data);
extern bool modem_device_carrier(int i);
extern void modem_device_init(void);

#define DEV_SIO2B 0

#endif /* !GENERIC_AT_MODEM_INC */
