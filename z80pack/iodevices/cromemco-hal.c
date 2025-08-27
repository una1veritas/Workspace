/*
* cromemco-hal.c
*
* Copyright (C) 2022 by David McNaughton
*
* Cromemco TU-ART hardware abstraction layer
*
* History:
* 9-JUL-2022	1.0	Initial Release
*
*/

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/poll.h>
#include <sys/socket.h>

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simio.h"

#include "unix_terminal.h"
#include "unix_network.h"
#ifdef HAS_NETSERVER
#include "netsrv.h"
#endif
#include "cromemco-hal.h"

/* #define LOG_LOCAL_LEVEL LOG_DEBUG */
#define LOG_LOCAL_LEVEL LOG_WARN
#include "log.h"
static const char *TAG = "HAL";

/* -------------------- NULL device HAL -------------------- */

static bool null_alive(int dev)
{
	UNUSED(dev);

	return true; /* NULL is always alive */
}

#if !defined(HAS_NETSERVER) || !defined(HAS_MODEM)
static bool null_dead(int dev)
{
	UNUSED(dev);

	return false; /* NULL is always dead */
}
#endif

static void null_status(int dev, BYTE *stat)
{
	UNUSED(dev);
	UNUSED(stat);

	return;
}

static int null_in(int dev)
{
	UNUSED(dev);

	return -1;
}

static void null_out(int dev, BYTE data)
{
	UNUSED(dev);
	UNUSED(data);

	return;
}

/* -------------------- WEBTTY HAL -------------------- */

#ifdef HAS_NETSERVER

static bool net_tty_alive(int dev)
{
	if (n_flag) {
		// LOG(TAG, "WEBTTY %d: %d\r\n", dev, net_device_alive(dev));
		/* WEBTTY is only alive if websocket is connected */
		return net_device_alive((net_device_t) dev);
	} else
		return false;
}

static void net_tty_status(int dev, BYTE *stat)
{
	*stat &= (BYTE) (~3);
	if (n_flag) {
		if (net_device_poll((net_device_t) dev))
			*stat |= 2;
		*stat |= 1;
	}
}

static int net_tty_in(int dev)
{
	if (n_flag)
		return net_device_get((net_device_t) dev);
	else
		return -1;
}

static void net_tty_out(int dev, BYTE data)
{
	if (n_flag)
		net_device_send((net_device_t) dev, (char *)&data, 1);
}

#endif /* HAS_NETSERVER */

/* -------------------- STDIO HAL -------------------- */

static bool stdio_alive(int dev)
{
	UNUSED(dev);

	return true; /* STDIO is always alive */
}

static void stdio_status(int dev, BYTE *stat)
{
	struct pollfd p[1];

	UNUSED(dev);

	p[0].fd = fileno(stdin);
	p[0].events = POLLIN;
	p[0].revents = 0;
	poll(p, 1, 0);
	*stat &= (BYTE) (~3);
	if (p[0].revents & POLLIN)
		*stat |= 2;
	if (p[0].revents & POLLNVAL) {
		LOGE(TAG, "can't use terminal, try 'screen simulation ...'");
		exit(EXIT_FAILURE);
		// cpu_error = IOERROR;
		// cpu_state = STOPPED;
	}
	*stat |= 1;
}

static int stdio_in(int dev)
{
	int data;
	struct pollfd p[1];

	UNUSED(dev);

again:
	/* if no input waiting return last */
	p[0].fd = fileno(stdin);
	p[0].events = POLLIN;
	p[0].revents = 0;
	poll(p, 1, 0);
	if (!(p[0].revents & POLLIN))
		return -1;

	if (read(fileno(stdin), &data, 1) == 0) {
		/* try to reopen tty, input redirection exhausted */
		if (freopen("/dev/tty", "r", stdin) == NULL)
			LOGE(TAG, "can't reopen /dev/tty");
		set_unix_terminal();
		goto again;
	}

	return data;
}

static void stdio_out(int dev, BYTE data)
{
	UNUSED(dev);

again:
	if (write(fileno(stdout), (char *) &data, 1) != 1) {
		if (errno == EINTR)
			goto again;
		else {
			LOGE(TAG, "can't write data");
			cpu_error = IOERROR;
			cpu_state = ST_STOPPED;
		}
	}
}

/* -------------------- SOCKET SERVER HAL -------------------- */

static bool scktsrv_alive(int dev)
{
	return ncons[dev].ssc != 0; /* SCKTSRV is alive if there is an open socket */
}

static void scktsrv_status(int dev, BYTE *stat)
{
	struct pollfd p[1];

	/* if socket is connected check for I/O */
	if (ncons[dev].ssc != 0) {
		p[0].fd = ncons[dev].ssc;
		p[0].events = POLLIN;
		p[0].revents = 0;
		poll(p, 1, 0);
		*stat &= (BYTE) (~3);
		if (p[0].revents & POLLHUP) {
			close(ncons[dev].ssc);
			ncons[dev].ssc = 0;
			*stat = 0;
		} else if (p[0].revents & POLLIN)
			*stat |= 2;
		else
			*stat |= 1;
	} else
		*stat = 0;
}

static int scktsrv_in(int dev)
{
	BYTE data, dummy;
	struct pollfd p[1];

	/* if not connected return last */
	if (ncons[dev].ssc == 0)
		return -1;

	/* if no input waiting return last */
	p[0].fd = ncons[dev].ssc;
	p[0].events = POLLIN;
	p[0].revents = 0;
	poll(p, 1, 0);
	if (!(p[0].revents & POLLIN))
		return -1;

	if (read(ncons[dev].ssc, &data, 1) != 1) {
		if ((errno == EAGAIN) || (errno == EINTR)) {
			/* EOF, close socket and return last */
			close(ncons[dev].ssc);
			ncons[dev].ssc = 0;
			return -1;
		} else {
			LOGE(TAG, "can't read tcpsocket %d data", dev);
			cpu_error = IOERROR;
			cpu_state = ST_STOPPED;
			return 0;
		}
	}

	/* process read data */
	/* telnet client sends \r\n or \r\0, drop second character */
	if (ncons[dev].telnet && (data == '\r'))
		if (read(ncons[dev].ssc, &dummy, 1) != 1)
			LOGE(TAG, "can't read tcpsocket %d data", dev);

	return data;
}

static void scktsrv_out(int dev, BYTE data)
{
	/* return if socket not connected */
	if (ncons[dev].ssc == 0)
		return;

again:
	if (write(ncons[dev].ssc, &data, 1) != 1) {
		if (errno == EINTR) {
			goto again;
		} else {
			LOGE(TAG, "can't write socket %d data", dev);
			cpu_error = IOERROR;
			cpu_state = ST_STOPPED;
		}
	}
}

/* -------------------- MODEM HAL -------------------- */

#ifdef HAS_MODEM

#include "generic-at-modem.h"

static bool modem_alive(int dev)
{
	UNUSED(dev);

	return modem_device_alive(0);
}

static void modem_status(int dev, BYTE *stat)
{
	UNUSED(dev);

	*stat &= (BYTE) (~3);
	if (modem_device_poll(0))
		*stat |= 2;
	*stat |= 1;
}

static int modem_in(int dev)
{
	UNUSED(dev);

	return modem_device_get(0);
}

static void modem_out(int dev, BYTE data)
{
	UNUSED(dev);

	modem_device_send(0, (char) data);
}

#endif /* HAS_MODEM */

/* -------------------- HAL port/device mappings -------------------- */

const char *tuart_port_name[MAX_TUART_PORT] =
	{ "TUART0.deviceA", "TUART1.deviceA", "TUART1.deviceB" };

static const hal_device_t devices[] = {
#ifdef HAS_NETSERVER
	{ "WEBTTY", false, DEV_TTY, net_tty_alive, net_tty_status, net_tty_in, net_tty_out },
	{ "WEBTTY2", false, DEV_TTY2, net_tty_alive, net_tty_status, net_tty_in, net_tty_out },
	{ "WEBTTY3", false, DEV_TTY3, net_tty_alive, net_tty_status, net_tty_in, net_tty_out },
	{ "WEBPTR", false, DEV_PTR, net_tty_alive, net_tty_status, net_tty_in, net_tty_out },
#else
	{ "WEBTTY", false, 0, null_dead, null_status, null_in, null_out },
	{ "WEBTTY2", false, 0, null_dead, null_status, null_in, null_out },
	{ "WEBTTY3", false, 0, null_dead, null_status, null_in, null_out },
	{ "WEBPTR", false, 0, null_dead, null_status, null_in, null_out },
#endif
	{ "STDIO", false, 0, stdio_alive, stdio_status, stdio_in, stdio_out },
	{ "SCKTSRV1", false, 0, scktsrv_alive, scktsrv_status, scktsrv_in, scktsrv_out },
	{ "SCKTSRV2", false, 1, scktsrv_alive, scktsrv_status, scktsrv_in, scktsrv_out },
#ifdef HAS_MODEM
	{ "MODEM", false, 0, modem_alive, modem_status, modem_in, modem_out },
#else
	{ "MODEM", false, 0, null_dead, null_status, null_in, null_out },
#endif
	{ "", false, 0, null_alive, null_status, null_in, null_out }
};

hal_device_t tuart[MAX_TUART_PORT][MAX_HAL_DEV];

/* -------------------- HAL utility functions -------------------- */

static void hal_report(void)
{
	int i, j;

	LOG(TAG, "\r\nTU-ART DEVICE MAP:\r\n");
	for (i = 0; i < MAX_TUART_PORT; i++) {
		LOG(TAG, "%s = ", tuart_port_name[i]);
		j = 0;
		while (tuart[i][j].name && j < MAX_HAL_DEV) {
			LOG(TAG, "%s%s", tuart[i][j].name, tuart[i][j].fallthrough ? "+" : " ");
			j++;
		}
		LOG(TAG, "\r\n");
	}
}

static int hal_find_device(char *dev)
{
	int i = 0;
	while (i < MAX_HAL_DEV) {
		if (!strcmp(dev, devices[i].name))
			return i;
		i++;
	}
	return -1;
}

static void hal_init(void)
{
	int i, j, d;
	char *setting;
	char match[80];
	char *dev;

	/**
	 *  Initialize HAL with default configuration, as follows:
	 *
	 *	TUART0.deviceA.device=WEBTTY,STDIO
	 *	TUART1.deviceA.device=SCKTSRV1,WEBTTY2
	 *	TUART1.deviceB.device=SCKTSRV2,WEBTTY3
	 *
	 * Notes:
	 *	- all ports end with NULL and that is always alive
	 *	- the first HAL device in the list that is alive will service the request
	 */
	tuart[0][0] = devices[WEBTTYDEV];
	tuart[0][1] = devices[STDIODEV];
	tuart[0][2] = devices[NULLDEV];

	tuart[1][0] = devices[SCKTSRV1DEV];
	tuart[1][1] = devices[WEBTTY2DEV];
	tuart[1][2] = devices[NULLDEV];

	tuart[2][0] = devices[SCKTSRV2DEV];
	tuart[2][1] = devices[WEBTTY3DEV];
	tuart[2][2] = devices[NULLDEV];

	for (i = 0; i < MAX_TUART_PORT; i++) {
		j = 0;
		strcpy(match, tuart_port_name[i]);
		strcat(match, ".device");

		if ((setting = getenv(match)) != NULL) {
			LOGI(TAG, "%s = %s", match, setting);

			strcpy(match, setting);

			dev = strtok(match, ",\r");
			while (dev) {
				char k = dev[strlen(dev) - 1];
				bool fallthrough = false;

				if (k == '+') {
					dev[strlen(dev) - 1] = 0;
					fallthrough = true;
				}

				d = hal_find_device(dev);
				LOGI(TAG, "\tAdding %s to %s", dev, tuart_port_name[i]);

				if (d >= 0) {
					memcpy(&tuart[i][j], &devices[d], sizeof(hal_device_t));
					tuart[i][j].fallthrough = fallthrough;
					j++;
				}
				dev = strtok(NULL, ",\r");
			}
			memcpy(&tuart[i][j], &devices[NULLDEV], sizeof(hal_device_t));
		}
	}
}

void hal_reset(void)
{
	hal_init();
	hal_report();
}

/* -------------------- HAL - TU-ART interface -------------------- */

void hal_status_in(tuart_port_t dev, BYTE *stat)
{
	int p = 0;
	BYTE s;
	*stat = 0;
next:
	/* Find the first device that is alive */
	while (!tuart[dev][p].alive(tuart[dev][p].device_id))
		p++;

	tuart[dev][p].status(tuart[dev][p].device_id, &s);
	*stat |= s;

	if (tuart[dev][p].fallthrough) {
		p++;
		goto next;
	}
}

int hal_data_in(tuart_port_t dev)
{
	int p = 0;
	int in = 0;

next:
	/* Find the first device that is alive */
	while (!tuart[dev][p].alive(tuart[dev][p].device_id))
		p++;

	in = tuart[dev][p].in(tuart[dev][p].device_id);

	if (in < 0 && tuart[dev][p].fallthrough) {
		p++;
		goto next;
	} else
		return in;
}

void hal_data_out(tuart_port_t dev, BYTE data)
{
	int p = 0;

next:
	/* Find the first device that is alive */
	while (!tuart[dev][p].alive(tuart[dev][p].device_id))
		p++;

	tuart[dev][p].out(tuart[dev][p].device_id, data);

	if (tuart[dev][p].fallthrough) {
		p++;
		goto next;
	}
}

bool hal_alive(tuart_port_t dev)
{
	int p = 0;

	/* Find the first device that is alive */
	while (!tuart[dev][p].alive(tuart[dev][p].device_id))
		p++;

	/* return "alive" (true) when not the NULL device */
	return tuart[dev][p].name ? true : false;
}
