/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Common I/O devices used by various simulated machines
 *
 * Copyright (C) 2015-2019 by Udo Munk
 * Copyright (C) 2018 David McNaughton
 * Copyright (C) 2025 by Thomas Eberhardt
 *
 * Emulation of a Cromemco DAZZLER S100 board
 *
 * History:
 * 24-APR-2015 first version
 * 25-APR-2015 fixed a few things, good enough for a BETA release now
 * 27-APR-2015 fixed logic bugs with on/off state and thread handling
 * 08-MAY-2015 fixed Xlib multithreading problems
 * 26-AUG-2015 implemented double buffering to prevent flicker
 * 27-AUG-2015 more bug fixes
 * 15-NOV-2016 fixed logic bug, display wasn't always clear after
 *	       the device is switched off
 * 06-DEC-2016 added bus request for the DMA
 * 16-DEC-2016 use DMA function for memory access
 * 26-JAN-2017 optimization
 * 15-JUL-2018 use logging
 * 19-JUL-2018 integrate webfrontend
 * 04-NOV-2019 remove fake DMA bus request
 * 04-JAN-2025 add SDL2 support
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef WANT_SDL
#include <SDL.h>
#else
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simcfg.h"
#include "simmem.h"
#include "simport.h"
#ifdef WANT_SDL
#include "simsdl.h"
#endif

#ifdef HAS_DAZZLER

#ifdef HAS_NETSERVER
#include <string.h>
#include "netsrv.h"
#endif

#if !defined(WANT_SDL) || defined(HAS_NETSERVER)
#include <pthread.h>

/* #define LOG_LOCAL_LEVEL LOG_DEBUG */
#include "log.h"
static const char *TAG = "DAZZLER";
#endif

#include "cromemco-dazzler.h"

/* SDL2/X11 stuff */
#define WSIZE 512
static int size = WSIZE;
#ifdef WANT_SDL
static int dazzler_win_id = -1;
static SDL_Window *window;
static SDL_Renderer *renderer;
static uint8_t colors[16][3] = {
	{ 0x00, 0x00, 0x00 },
	{ 0x80, 0x00, 0x00 },
	{ 0x00, 0x80, 0x00 },
	{ 0x80, 0x80, 0x00 },
	{ 0x00, 0x00, 0x80 },
	{ 0x80, 0x00, 0x80 },
	{ 0x00, 0x80, 0x80 },
	{ 0x80, 0x80, 0x80 },
	{ 0x00, 0x00, 0x00 },
	{ 0xFF, 0x00, 0x00 },
	{ 0x00, 0xFF, 0x00 },
	{ 0xFF, 0xFF, 0x00 },
	{ 0x00, 0x00, 0xFF },
	{ 0xFF, 0x00, 0xFF },
	{ 0x00, 0xFF, 0xFF },
	{ 0xFF, 0xFF, 0xFF }
};
static uint8_t grays[16][3] = {
	{ 0x00, 0x00, 0x00 },
	{ 0x11, 0x11, 0x11 },
	{ 0x22, 0x22, 0x22 },
	{ 0x33, 0x33, 0x33 },
	{ 0x44, 0x44, 0x44 },
	{ 0x55, 0x55, 0x55 },
	{ 0x66, 0x66, 0x66 },
	{ 0x77, 0x77, 0x77 },
	{ 0x88, 0x88, 0x88 },
	{ 0x99, 0x99, 0x99 },
	{ 0xAA, 0xAA, 0xAA },
	{ 0xBB, 0xBB, 0xBB },
	{ 0xCC, 0xCC, 0xCC },
	{ 0xDD, 0xDD, 0xDD },
	{ 0xEE, 0xEE, 0xEE },
	{ 0xFF, 0xFF, 0xFF }
};
#else /* !WANT_SDL */
static Display *display;
static Window window;
static int screen;
static GC gc;
static XWindowAttributes wa;
static Pixmap pixmap;
static Colormap colormap;
static XColor colors[16];
static XColor grays[16];
static char color0[] =  "#000000";
static char color1[] =  "#800000";
static char color2[] =  "#008000";
static char color3[] =  "#808000";
static char color4[] =  "#000080";
static char color5[] =  "#800080";
static char color6[] =  "#008080";
static char color7[] =  "#808080";
static char color8[] =  "#000000";
static char color9[] =  "#FF0000";
static char color10[] = "#00FF00";
static char color11[] = "#FFFF00";
static char color12[] = "#0000FF";
static char color13[] = "#FF00FF";
static char color14[] = "#00FFFF";
static char color15[] = "#FFFFFF";
static char gray0[] =   "#000000";
static char gray1[] =   "#111111";
static char gray2[] =   "#222222";
static char gray3[] =   "#333333";
static char gray4[] =   "#444444";
static char gray5[] =   "#555555";
static char gray6[] =   "#666666";
static char gray7[] =   "#777777";
static char gray8[] =   "#888888";
static char gray9[] =   "#999999";
static char gray10[] =  "#AAAAAA";
static char gray11[] =  "#BBBBBB";
static char gray12[] =  "#CCCCCC";
static char gray13[] =  "#DDDDDD";
static char gray14[] =  "#EEEEEE";
static char gray15[] =  "#FFFFFF";
#endif /* !WANT_SDL */

/* DAZZLER stuff */
static bool state;
static WORD dma_addr;
static BYTE flags = 64;
static BYTE format;

#if !defined(WANT_SDL) || defined(HAS_NETSERVER)
/* UNIX stuff */
static pthread_t thread;
#endif

#ifdef HAS_NETSERVER
static void ws_clear(void);
static BYTE formatBuf = 0;
#endif

/* create the SDL2 or X11 window for DAZZLER display */
static void open_display(void)
{
#ifdef WANT_SDL
	window = SDL_CreateWindow("Cromemco DAzzLER",
				  SDL_WINDOWPOS_UNDEFINED,
				  SDL_WINDOWPOS_UNDEFINED,
				  size, size, 0);
	renderer = SDL_CreateRenderer(window, -1, (SDL_RENDERER_ACCELERATED |
						   SDL_RENDERER_PRESENTVSYNC));
#else /* !WANT_SDL */
	Window rootwindow;
	XSizeHints *size_hints = XAllocSizeHints();
	Atom wm_delete_window;

	display = XOpenDisplay(NULL);
	XLockDisplay(display);
	screen = DefaultScreen(display);
	rootwindow = RootWindow(display, screen);
	XGetWindowAttributes(display, rootwindow, &wa);
	window = XCreateSimpleWindow(display, rootwindow, 0, 0,
				     size, size, 1, 0, 0);
	XStoreName(display, window, "Cromemco DAzzLER");
	size_hints->flags = PSize | PMinSize | PMaxSize;
	size_hints->min_width = size;
	size_hints->min_height = size;
	size_hints->base_width = size;
	size_hints->base_height = size;
	size_hints->max_width = size;
	size_hints->max_height = size;
	XSetWMNormalHints(display, window, size_hints);
	XFree(size_hints);
	wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(display, window, &wm_delete_window, 1);
	colormap = DefaultColormap(display, 0);
	gc = XCreateGC(display, window, 0, NULL);
	XSetFillStyle(display, gc, FillSolid);
	pixmap = XCreatePixmap(display, rootwindow, size, size,
			       wa.depth);

	XParseColor(display, colormap, color0, &colors[0]);
	XAllocColor(display, colormap, &colors[0]);
	XParseColor(display, colormap, color1, &colors[1]);
	XAllocColor(display, colormap, &colors[1]);
	XParseColor(display, colormap, color2, &colors[2]);
	XAllocColor(display, colormap, &colors[2]);
	XParseColor(display, colormap, color3, &colors[3]);
	XAllocColor(display, colormap, &colors[3]);
	XParseColor(display, colormap, color4, &colors[4]);
	XAllocColor(display, colormap, &colors[4]);
	XParseColor(display, colormap, color5, &colors[5]);
	XAllocColor(display, colormap, &colors[5]);
	XParseColor(display, colormap, color6, &colors[6]);
	XAllocColor(display, colormap, &colors[6]);
	XParseColor(display, colormap, color7, &colors[7]);
	XAllocColor(display, colormap, &colors[7]);
	XParseColor(display, colormap, color8, &colors[8]);
	XAllocColor(display, colormap, &colors[8]);
	XParseColor(display, colormap, color9, &colors[9]);
	XAllocColor(display, colormap, &colors[9]);
	XParseColor(display, colormap, color10, &colors[10]);
	XAllocColor(display, colormap, &colors[10]);
	XParseColor(display, colormap, color11, &colors[11]);
	XAllocColor(display, colormap, &colors[11]);
	XParseColor(display, colormap, color12, &colors[12]);
	XAllocColor(display, colormap, &colors[12]);
	XParseColor(display, colormap, color13, &colors[13]);
	XAllocColor(display, colormap, &colors[13]);
	XParseColor(display, colormap, color14, &colors[14]);
	XAllocColor(display, colormap, &colors[14]);
	XParseColor(display, colormap, color15, &colors[15]);
	XAllocColor(display, colormap, &colors[15]);

	XParseColor(display, colormap, gray0, &grays[0]);
	XAllocColor(display, colormap, &grays[0]);
	XParseColor(display, colormap, gray1, &grays[1]);
	XAllocColor(display, colormap, &grays[1]);
	XParseColor(display, colormap, gray2, &grays[2]);
	XAllocColor(display, colormap, &grays[2]);
	XParseColor(display, colormap, gray3, &grays[3]);
	XAllocColor(display, colormap, &grays[3]);
	XParseColor(display, colormap, gray4, &grays[4]);
	XAllocColor(display, colormap, &grays[4]);
	XParseColor(display, colormap, gray5, &grays[5]);
	XAllocColor(display, colormap, &grays[5]);
	XParseColor(display, colormap, gray6, &grays[6]);
	XAllocColor(display, colormap, &grays[6]);
	XParseColor(display, colormap, gray7, &grays[7]);
	XAllocColor(display, colormap, &grays[7]);
	XParseColor(display, colormap, gray8, &grays[8]);
	XAllocColor(display, colormap, &grays[8]);
	XParseColor(display, colormap, gray9, &grays[9]);
	XAllocColor(display, colormap, &grays[9]);
	XParseColor(display, colormap, gray10, &grays[10]);
	XAllocColor(display, colormap, &grays[10]);
	XParseColor(display, colormap, gray11, &grays[11]);
	XAllocColor(display, colormap, &grays[11]);
	XParseColor(display, colormap, gray12, &grays[12]);
	XAllocColor(display, colormap, &grays[12]);
	XParseColor(display, colormap, gray13, &grays[13]);
	XAllocColor(display, colormap, &grays[13]);
	XParseColor(display, colormap, gray14, &grays[14]);
	XAllocColor(display, colormap, &grays[14]);
	XParseColor(display, colormap, gray15, &grays[15]);
	XAllocColor(display, colormap, &grays[15]);

	XMapWindow(display, window);
	XUnlockDisplay(display);
#endif /* !WANT_SDL */
}

/* close the SDL or X11 window for DAZZLER display */
static void close_display(void)
{
#ifdef WANT_SDL
	SDL_DestroyRenderer(renderer);
	renderer = NULL;
	SDL_DestroyWindow(window);
	window = NULL;
#else
	XLockDisplay(display);
	XFreePixmap(display, pixmap);
	XFreeGC(display, gc);
	XUnlockDisplay(display);
	XCloseDisplay(display);
	display = NULL;
#endif
}

#if !defined(WANT_SDL) || defined(HAS_NETSERVER)
static void kill_thread(void)
{
	if (thread != 0) {
		sleep_for_ms(50);
		pthread_cancel(thread);
		pthread_join(thread, NULL);
		thread = 0;
	}
}
#endif

/* switch DAZZLER off from front panel */
void cromemco_dazzler_off(void)
{
	state = false;

#ifdef WANT_SDL
#ifdef HAS_NETSERVER
	if (!n_flag) {
#endif
		if (dazzler_win_id >= 0) {
			simsdl_destroy(dazzler_win_id);
			dazzler_win_id = -1;
		}
#ifdef HAS_NETSERVER
	} else {
		kill_thread();
		ws_clear();
	}
#endif
#else /* !WANT_SDL */
	kill_thread();
	if (display != NULL)
		close_display();
#ifdef HAS_NETSERVER
	if (n_flag)
		ws_clear();
#endif
#endif /* !WANT_SDL */
}

#ifdef WANT_SDL

/* process SDL event */
static void process_event(SDL_Event *event)
{
	UNUSED(event);
}

static inline void set_fg_color(int i)
{
	SDL_SetRenderDrawColor(renderer,
			       colors[i][0], colors[i][1], colors[i][2],
			       SDL_ALPHA_OPAQUE);
}

static inline void set_fg_gray(int i)
{
	SDL_SetRenderDrawColor(renderer,
			       grays[i][0], grays[i][1], grays[i][2],
			       SDL_ALPHA_OPAQUE);
}

static inline void fill_rect(int x, int y, int w, int h)
{
	SDL_Rect r = {x, y, w, h};

	SDL_RenderFillRect(renderer, &r);
}

#else /* !WANT_SDL */

static inline void set_fg_color(int i)
{
	XSetForeground(display, gc, colors[i].pixel);
}

static inline void set_fg_gray(int i)
{
	XSetForeground(display, gc, grays[i].pixel);
}

static inline void fill_rect(int x, int y, int w, int h)
{
	XFillRectangle(display, pixmap, gc, x, y, w, h);
}

#endif /* !WANT_SDL */

/* draw pixels for one frame in hires */
static void draw_hires(void)
{
	int psize, x, y, i;
	WORD addr = dma_addr;

	/* set color or grayscale from lower nibble in graphics format */
	i = format & 0x0f;
	if (format & 16)
		set_fg_color(i);
	else
		set_fg_gray(i);
	if (format & 32) {	/* 2048 bytes memory */
		psize = size / 128;	/* size of one pixel for 128x128 */
		for (y = 0; y < 64; y += 2) {
			for (x = 0; x < 64;) {
				i = dma_read(addr);
				if (i & 1)
					fill_rect(x * psize, y * psize, psize, psize);
				if (i & 2)
					fill_rect((x + 1) * psize, y * psize, psize, psize);
				if (i & 4)
					fill_rect(x * psize, (y + 1) * psize, psize, psize);
				if (i & 8)
					fill_rect((x + 1) * psize, (y + 1) * psize, psize, psize);
				if (i & 16)
					fill_rect((x + 2) * psize, y * psize, psize, psize);
				if (i & 32)
					fill_rect((x + 3) * psize, y * psize, psize, psize);
				if (i & 64)
					fill_rect((x + 2) * psize, (y + 1) * psize, psize, psize);
				if (i & 128)
					fill_rect((x + 3) * psize, (y + 1) * psize, psize, psize);
				x += 4;
				addr++;
			}
		}
		for (y = 0; y < 64; y += 2) {
			for (x = 64; x < 128;) {
				i = dma_read(addr);
				if (i & 1)
					fill_rect(x * psize, y * psize, psize, psize);
				if (i & 2)
					fill_rect((x + 1) * psize, y * psize, psize, psize);
				if (i & 4)
					fill_rect(x * psize, (y + 1) * psize, psize, psize);
				if (i & 8)
					fill_rect((x + 1) * psize, (y + 1) * psize, psize, psize);
				if (i & 16)
					fill_rect((x + 2) * psize, y * psize, psize, psize);
				if (i & 32)
					fill_rect((x + 3) * psize, y * psize, psize, psize);
				if (i & 64)
					fill_rect((x + 2) * psize, (y + 1) * psize, psize, psize);
				if (i & 128)
					fill_rect((x + 3) * psize, (y + 1) * psize, psize, psize);
				x += 4;
				addr++;
			}
		}
		for (y = 64; y < 128; y += 2) {
			for (x = 0; x < 64;) {
				i = dma_read(addr);
				if (i & 1)
					fill_rect(x * psize, y * psize, psize, psize);
				if (i & 2)
					fill_rect((x + 1) * psize, y * psize, psize, psize);
				if (i & 4)
					fill_rect(x * psize, (y + 1) * psize, psize, psize);
				if (i & 8)
					fill_rect((x + 1) * psize, (y + 1) * psize, psize, psize);
				if (i & 16)
					fill_rect((x + 2) * psize, y * psize, psize, psize);
				if (i & 32)
					fill_rect((x + 3) * psize, y * psize, psize, psize);
				if (i & 64)
					fill_rect((x + 2) * psize, (y + 1) * psize, psize, psize);
				if (i & 128)
					fill_rect((x + 3) * psize, (y + 1) * psize, psize, psize);
				x += 4;
				addr++;
			}
		}
		for (y = 64; y < 128; y += 2) {
			for (x = 64; x < 128;) {
				i = dma_read(addr);
				if (i & 1)
					fill_rect(x * psize, y * psize, psize, psize);
				if (i & 2)
					fill_rect((x + 1) * psize, y * psize, psize, psize);
				if (i & 4)
					fill_rect(x * psize, (y + 1) * psize, psize, psize);
				if (i & 8)
					fill_rect((x + 1) * psize, (y + 1) * psize, psize, psize);
				if (i & 16)
					fill_rect((x + 2) * psize, y * psize, psize, psize);
				if (i & 32)
					fill_rect((x + 3) * psize, y * psize, psize, psize);
				if (i & 64)
					fill_rect((x + 2) * psize, (y + 1) * psize, psize, psize);
				if (i & 128)
					fill_rect((x + 3) * psize, (y + 1) * psize, psize, psize);
				x += 4;
				addr++;
			}
		}
	} else {		/* 512 bytes memory */
		psize = size / 64;	/* size of one pixel for 64x64 */
		for (y = 0; y < 64; y += 2) {
			for (x = 0; x < 64;) {
				i = dma_read(addr);
				if (i & 1)
					fill_rect(x * psize, y * psize, psize, psize);
				if (i & 2)
					fill_rect((x + 1) * psize, y * psize, psize, psize);
				if (i & 4)
					fill_rect(x * psize, (y + 1) * psize, psize, psize);
				if (i & 8)
					fill_rect((x + 1) * psize, (y + 1) * psize, psize, psize);
				if (i & 16)
					fill_rect((x + 2) * psize, y * psize, psize, psize);
				if (i & 32)
					fill_rect((x + 3) * psize, y * psize, psize, psize);
				if (i & 64)
					fill_rect((x + 2) * psize, (y + 1) * psize, psize, psize);
				if (i & 128)
					fill_rect((x + 3) * psize, (y + 1) * psize, psize, psize);
				x += 4;
				addr++;
			}
		}
	}
}

/* draw pixels for one frame in lowres */
static void draw_lowres(void)
{
	int psize, x, y, i;
	WORD addr = dma_addr;

	/* get size of DMA memory and draw the pixels */
	if (format & 32) {	/* 2048 bytes memory */
		psize = size / 64;	/* size of one pixel for 64x64 */
		for (y = 0; y < 32; y++) {
			for (x = 0; x < 32;) {
				i = dma_read(addr) & 0x0f;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				i = (dma_read(addr) & 0xf0) >> 4;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				addr++;
			}
		}
		for (y = 0; y < 32; y++) {
			for (x = 32; x < 64;) {
				i = dma_read(addr) & 0x0f;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				i = (dma_read(addr) & 0xf0) >> 4;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				addr++;
			}
		}
		for (y = 32; y < 64; y++) {
			for (x = 0; x < 32;) {
				i = dma_read(addr) & 0x0f;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				i = (dma_read(addr) & 0xf0) >> 4;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				addr++;
			}
		}
		for (y = 32; y < 64; y++) {
			for (x = 32; x < 64;) {
				i = dma_read(addr) & 0x0f;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				i = (dma_read(addr) & 0xf0) >> 4;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				addr++;
			}
		}
	} else {		/* 512 bytes memory */
		psize = size / 32;	/* size of one pixel for 32x32 */
		for (y = 0; y < 32; y++) {
			for (x = 0; x < 32;) {
				i = dma_read(addr) & 0x0f;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				i = (dma_read(addr) & 0xf0) >> 4;
				if (format & 16)
					set_fg_color(i);
				else
					set_fg_gray(i);
				fill_rect(x * psize, y * psize, psize, psize);
				x++;
				addr++;
			}
		}
	}
}

#ifdef HAS_NETSERVER
static uint8_t dblbuf[2048];

static struct {
	uint16_t format;
	uint16_t addr;
	uint16_t len;
	uint8_t buf[2048];
} msg;

static void ws_clear(void)
{
	memset(dblbuf, 0, 2048);

	msg.format = 0;
	msg.addr = 0xFFFF;
	msg.len = 0;
	net_device_send(DEV_DZLR, (char *) &msg, msg.len + 6);
	LOGD(TAG, "Clear the screen.");
}

static void ws_refresh(void)
{
	int len = (format & 32) ? 2048 : 512;
	int addr;
	int i, n, x, la_count;
	bool cont;
	uint8_t val;

	for (i = 0; i < len; i++) {
		addr = i;
		n = 0;
		la_count = 0;
		cont = true;
		while (cont && (i < len)) {
			val = dma_read(dma_addr + i);
			while ((val != dblbuf[i]) && (i < len)) {
				dblbuf[i++] = val;
				msg.buf[n++] = val;
				cont = false;
				val = dma_read(dma_addr + i);
			}
			if (cont)
				break;
			x = 0;
#define LOOKAHEAD 6
			/* look-ahead up to n bytes for next change */
			while ((x < LOOKAHEAD) && !cont && (i < len)) {
				val = dma_read(dma_addr + i++);
				msg.buf[n++] = val;
				la_count++;
				val = dma_read(dma_addr + i);
				if ((i < len) && (val != dblbuf[i])) {
					cont = true;
				}
				x++;
			}
			if (!cont) {
				n -= x;
				la_count -= x;
			}
		}
		if (n || (format != formatBuf)) {
			formatBuf = format;
			msg.format = format;
			msg.addr = addr;
			msg.len = n;
			net_device_send(DEV_DZLR, (char *) &msg, msg.len + 6);
			LOGD(TAG, "BUF update 0x%04X-0x%04X "
			     "len: %d format: 0x%02X l/a: %d",
			     msg.addr, msg.addr + msg.len,
			     msg.len, msg.format, la_count);
		}
	}
}
#endif /* HAS_NETSERVER */

#ifdef WANT_SDL
/* function for updating the display */
static void update_display(bool tick)
{
	UNUSED(tick);

	/* draw one frame dependent on graphics format */
	set_fg_color(0);
	SDL_RenderClear(renderer);
	if (state) {		/* draw frame if on */
		if (format & 64)
			draw_hires();
		else
			draw_lowres();
		SDL_RenderPresent(renderer);

		/* frame done, set frame flag for 4ms */
		flags = 0;
		sleep_for_ms(4);
		flags = 64;
	} else
		SDL_RenderPresent(renderer);
}

static win_funcs_t dazzler_funcs = {
	open_display,
	close_display,
	process_event,
	update_display
};
#endif

#if !defined(WANT_SDL) || defined(HAS_NETSERVER)
/* thread for updating the X11 display or web server */
static void *update_thread(void *arg)
{
	uint64_t t;
	long tleft;

	UNUSED(arg);

	t = get_clock_us();

	while (true) {	/* do forever or until canceled */

		/* draw one frame dependent on graphics format */
		if (state) {		/* draw frame if on */
#ifdef HAS_NETSERVER
			if (!n_flag) {
#endif
#ifndef WANT_SDL
				XLockDisplay(display);
				set_fg_color(0);
				fill_rect(0, 0, size, size);
				if (format & 64)
					draw_hires();
				else
					draw_lowres();
				XCopyArea(display, pixmap, window, gc, 0, 0,
					  size, size, 0, 0);
				XSync(display, True);
				XUnlockDisplay(display);
#endif
#ifdef HAS_NETSERVER
			} else {
				if (net_device_alive(DEV_DZLR)) {
					ws_refresh();
				} else {
					if (msg.format) {
						memset(dblbuf, 0, 2048);
						msg.format = 0;
					}
				}
			}
#endif
		}

		/* frame done, set frame flag for 4ms */
		flags = 0;
		sleep_for_ms(4);
		flags = 64;

		/* sleep rest to 33333us so that we get 30 fps */
		tleft = 33333L - (long) (get_clock_us() - t);
		if (tleft > 0)
			sleep_for_us(tleft);

		t = get_clock_us();
	}

	/* just in case it ever gets here */
	pthread_exit(NULL);
}
#endif /* !WANT_SDL || !HAS_NETSERVER */

void cromemco_dazzler_ctl_out(BYTE data)
{
	/* get DMA address for display memory */
	dma_addr = (data & 0x7f) << 9;

	/* switch DAZZLER on/off */
	if (data & 128) {
#ifdef HAS_NETSERVER
		if (!n_flag) {
#endif
#ifdef WANT_SDL
			if (dazzler_win_id < 0)
				dazzler_win_id = simsdl_create(&dazzler_funcs);
#else
			if (display == NULL)
				open_display();
#endif
#ifdef HAS_NETSERVER
		} else {
			if (!state)
				ws_clear();
		}
#endif
		state = true;
#if defined(WANT_SDL) && defined(HAS_NETSERVER)
		if (n_flag) {
#endif
#if !defined(WANT_SDL) || defined(HAS_NETSERVER)
			if (thread == 0) {
				if (pthread_create(&thread, NULL, update_thread,
						   NULL)) {
					LOGE(TAG, "can't create thread");
					exit(EXIT_FAILURE);
				}
			}
#endif
#if defined(WANT_SDL) && defined(HAS_NETSERVER)
		}
#endif
	} else {
		if (state) {
			state = false;
			sleep_for_ms(50);
#ifdef HAS_NETSERVER
			if (!n_flag) {
#endif
#ifndef WANT_SDL
				XLockDisplay(display);
				XClearWindow(display, window);
				XSync(display, True);
				XUnlockDisplay(display);
#endif
#ifdef HAS_NETSERVER
			} else
				ws_clear();
#endif
		}
	}
}

BYTE cromemco_dazzler_flags_in(void)
{
	BYTE data = 0xff;

#ifdef WANT_SDL
#ifdef HAS_NETSERVER
	if (!n_flag) {
#endif
		if (dazzler_win_id >= 0)
			data = flags;
#ifdef HAS_NETSERVER
	} else {
		if (thread != 0)
			data = flags;
	}
#endif
#else /* !WANT_SDL */
	if (thread != 0)
		data = flags;
#endif /* !WANT_SDL */

	return data;
}

void cromemco_dazzler_format_out(BYTE data)
{
	format = data;
}

#endif /* HAS_DAZZLER */
