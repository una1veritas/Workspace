/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 2025 by Udo Munk
 *
 * This module contains global variables for the network configuration
 * of Pico W.
 * Not compiled conditionally because the configuration is saved on MicroSD,
 * and I want files with identical format for Pico and Pico W.
 *
 * History:
 * 18-MAY-2025 first implementation
 */

#ifndef NET_VARS_INC
#define NET_VARS_INC

#define WIFI_SSID_LEN	63
#define WIFI_PWD_LEN	33
#define HOST_NAME_MAX	63
#define DEFAULT_NTP	"pool.ntp.org"

extern char wifi_ssid[WIFI_SSID_LEN+1];
extern char wifi_password[WIFI_PWD_LEN+1];
extern char ntp_server[HOST_NAME_MAX+1];
extern int utc_offset;

#endif /* !NET_VARS_INC */
