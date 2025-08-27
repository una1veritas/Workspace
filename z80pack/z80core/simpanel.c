/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 2015-2019 by Udo Munk
 * Copyright (C) 2025 by Thomas Eberhardt
 */

/*
 *	This module contains an introspection panel to view various
 *	status information of the simulator.
 */

#include <string.h>
#ifdef WANT_SDL
#include <SDL.h>
#else
#include <stdlib.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <pthread.h>
#endif

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simmem.h"
#include "simpanel.h"
#include "simport.h"
#ifdef WANT_SDL
#include "simsdl.h"
#endif

/* #define LOG_LOCAL_LEVEL LOG_DEBUG */
#include "log.h"
static const char *TAG = "panel";

/* panel types */
#define MEMORY_PANEL	0
#define PORTS_PANEL	1

/* 888 RGB colors */
#define C_BLACK		0x00000000
#define C_RED		0x00ff0000
#define C_GREEN		0x0000ff00
#define C_BLUE		0x000000ff
#define C_CYAN		0x0000ffff
#define C_MAGENTA	0x00ff00ff
#define C_YELLOW	0x00ffff00
#define C_WHITE		0x00ffffff

#define C_BUTTER_1	0x00fce94f
#define C_BUTTER_3	0x00c4a000
#define C_ORANGE_1	0x00fcaf3e
#define C_CHOC_1	0x00e9b96e
#define C_CHAM_2	0x0073d216
#define C_CHAM_3	0x004e9a06
#define C_BLUE_1	0x00729fcf
#define C_PLUM_1	0x00e090d7
#define C_RED_2		0x00cc0000
#define C_ALUM_2	0x00d3d7cf
#define C_ALUM_4	0x00888a85
#define C_ALUM_5	0x00555753
#define C_ALUM_6	0x002e3436

#define C_TRANS		0xffffffff

/*
 *	Font type. Depth is ignored and assumed to be 1.
 */
typedef const struct font {
	const uint8_t *bits;
	unsigned depth;
	unsigned width;
	unsigned height;
	unsigned stride;
} font_t;

/* include Terminus bitmap fonts */
#include "fonts/font12.h"
#include "fonts/font14.h"
#include "fonts/font16.h"
#include "fonts/font18.h"
#include "fonts/font20.h"
#include "fonts/font22.h"
#include "fonts/font24.h"
#include "fonts/font28.h"
#include "fonts/font32.h"

/*
 *	Grid type for drawing text with character based coordinates.
 */
typedef struct grid {
	const font_t *font;
	unsigned xoff;
	unsigned yoff;
	unsigned spc;
	unsigned cwidth;
	unsigned cheight;
	unsigned cols;
	unsigned rows;
} grid_t;

/*
 *	Button type
 */
typedef struct button {
	bool enabled;
	bool active;
	bool pressed;
	bool clicked;
	bool hilighted;
	const unsigned x;
	const unsigned y;
	const unsigned width;
	const unsigned height;
	const font_t *font;
	const char *text;
} button_t;

/*
 *	Buttons
 */
#define MEMORY_BUTTON	0
#define PORTS_BUTTON	1
#define STICKY_BUTTON	2

static button_t buttons[] = {
	[ MEMORY_BUTTON ] = {
		false, false, false, false, false,
		500,  1, 60, 19, &font12, "Memory"
	},
	[ PORTS_BUTTON ] = {
		false, false, false, false, false,
		500, 22, 60, 19, &font12, "Ports"
	},
	[ STICKY_BUTTON ] = {
		false, false, false, false, false,
		500, 43, 60, 19, &font12, "Sticky"
	}
};
static const int nbuttons = sizeof(buttons) / sizeof(button_t);

/*
 *	Button events
 */

#define EVENT_PRESS	0
#define EVENT_RELEASE	1
#define EVENT_MOTION	2

static void check_buttons(unsigned x, unsigned y, int event);

/* SDL2/X11 stuff */
static unsigned xsize, ysize;
static uint32_t *pixels;
static int pitch;
static bool shift;
#ifdef WANT_SDL
static int panel_win_id = -1;
static SDL_Window *window;
static SDL_Renderer *renderer;
static SDL_Texture *texture;
#else
static Display *display;
static Visual *visual;
static Window window;
static int screen;
static GC gc;
static XImage *ximage;
static Colormap colormap;
static XEvent event;
#endif

#ifndef WANT_SDL
/* UNIX stuff */
static pthread_t thread;
#endif

/* panel stuff */
static int panel;		/* current panel type */
static WORD mbase;		/* memory panel base address */
static bool sticky;		/* I/O ports panel sticky flag */
static bool showfps;		/* show FPS flag */

/*
 * Create the SDL2 or X11 window for panel display
 */
static void open_display(void)
{
	xsize = 576; /* 72 * 8 */
	ysize = 376;

#ifdef WANT_SDL
	window = SDL_CreateWindow("Z80pack",
				  SDL_WINDOWPOS_UNDEFINED,
				  SDL_WINDOWPOS_UNDEFINED,
				  xsize, ysize, 0);
	if (window == NULL) {
		LOGW(TAG, "can't create window: %s", SDL_GetError());
		return;
	}
	renderer = SDL_CreateRenderer(window, -1, (SDL_RENDERER_ACCELERATED |
						   SDL_RENDERER_PRESENTVSYNC));
	if (renderer == NULL) {
		LOGW(TAG, "can't create renderer: %s", SDL_GetError());
		SDL_DestroyWindow(window);
		window = NULL;
		return;
	}
	texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_XRGB8888,
				    SDL_TEXTUREACCESS_STREAMING, xsize, ysize);
	if (texture == NULL) {
		LOGW(TAG, "can't create texture: %s", SDL_GetError());
		SDL_DestroyRenderer(renderer);
		renderer = NULL;
		SDL_DestroyWindow(window);
		window = NULL;
		return;
	}
#else /* !WANT_SDL */
	Window rootwindow;
	XSetWindowAttributes swa;
	XSizeHints *size_hints = XAllocSizeHints();
	Atom wm_delete_window;
	XVisualInfo vinfo;

	display = XOpenDisplay(NULL);
	if (display == NULL) {
		LOGW(TAG, "can't open display %s", getenv("DISPLAY"));
		return;
	}
	XLockDisplay(display);
	screen = DefaultScreen(display);
	if (!XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo)) {
		LOGW(TAG, "couldn't find a 24-bit TrueColor visual");
		XUnlockDisplay(display);
		XCloseDisplay(display);
		display = NULL;
		return;
	}
	rootwindow = RootWindow(display, vinfo.screen);
	visual = vinfo.visual;
	colormap = XCreateColormap(display, rootwindow, visual, AllocNone);
	swa.border_pixel = 0;
	swa.colormap = colormap;
	swa.event_mask = KeyPressMask | KeyReleaseMask |
			 ButtonPressMask | ButtonReleaseMask |
			 PointerMotionMask;
	window = XCreateWindow(display, rootwindow, 0, 0, xsize, ysize,
			       1, vinfo.depth, InputOutput, visual,
			       CWBorderPixel | CWColormap | CWEventMask, &swa);
	XStoreName(display, window, "Z80pack");
	size_hints->flags = PSize | PMinSize | PMaxSize;
	size_hints->min_width = xsize;
	size_hints->min_height = ysize;
	size_hints->base_width = xsize;
	size_hints->base_height = ysize;
	size_hints->max_width = xsize;
	size_hints->max_height = ysize;
	XSetWMNormalHints(display, window, size_hints);
	XFree(size_hints);
	wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(display, window, &wm_delete_window, 1);
	gc = XCreateGC(display, window, 0, NULL);
	pixels = (uint32_t *) malloc (xsize * ysize * sizeof(uint32_t));
	ximage = XCreateImage(display, visual, vinfo.depth, ZPixmap, 0,
			      (char *) pixels, xsize, ysize, 32, 0);
	/* force little-endian pixels, Xlib will convert if necessary */
	ximage->byte_order = LSBFirst;
	pitch = ximage->bytes_per_line / 4;

	XMapWindow(display, window);
	XUnlockDisplay(display);
#endif /* !WANT_SDL */
}

/*
 * Close the SDL2 or X11 window for panel display
 */
static void close_display(void)
{
#ifdef WANT_SDL
	if (texture != NULL) {
		SDL_DestroyTexture(texture);
		texture = NULL;
	}
	if (renderer != NULL) {
		SDL_DestroyRenderer(renderer);
		renderer = NULL;
	}
	if (window != NULL) {
		SDL_DestroyWindow(window);
		window = NULL;
	}
#else
	if (display != NULL) {
		XLockDisplay(display);
		free(pixels);
		ximage->data = NULL;
		XDestroyImage(ximage);
		XFreeGC(display, gc);
		XDestroyWindow(display, window);
		XFreeColormap(display, colormap);
		XUnlockDisplay(display);
		XCloseDisplay(display);
		display = NULL;
	}
#endif
}

#ifdef WANT_SDL

/*
 * Process a SDL event
 */
static void process_event(SDL_Event *event)
{
	int n;

	switch (event->type) {
	case SDL_MOUSEBUTTONDOWN:
		if (event->window.windowID != SDL_GetWindowID(window))
			break;

		check_buttons(event->button.x, event->button.y, EVENT_PRESS);
		break;

	case SDL_MOUSEBUTTONUP:
		if (event->window.windowID != SDL_GetWindowID(window))
			break;

		check_buttons(event->button.x, event->button.y, EVENT_RELEASE);
		break;

	case SDL_MOUSEMOTION:
		if (event->window.windowID != SDL_GetWindowID(window))
			break;

		check_buttons(event->motion.x, event->motion.y, EVENT_MOTION);
		break;

	case SDL_KEYUP:
		if (event->window.windowID != SDL_GetWindowID(window))
			break;

		switch (event->key.keysym.sym) {
		case SDLK_LSHIFT:
		case SDLK_RSHIFT:
			shift = false;
			break;
		default:
			break;
		}
		break;

	case SDL_KEYDOWN:
		if (event->window.windowID != SDL_GetWindowID(window))
			break;

		switch (event->key.keysym.sym) {
		case SDLK_LSHIFT:
		case SDLK_RSHIFT:
			shift = true;
			break;
		case SDLK_LEFT:
			if (panel == MEMORY_PANEL)
				mbase -= shift ? 0x0010 : 0x0001;
			break;
		case SDLK_RIGHT:
			if (panel == MEMORY_PANEL)
				mbase += shift ? 0x0010 : 0x0001;
			break;
		case SDLK_UP:
			if (panel == MEMORY_PANEL)
				mbase -= shift ? 0x0100 : 0x0010;
			break;
		case SDLK_DOWN:
			if (panel == MEMORY_PANEL)
				mbase += shift ? 0x0100 : 0x0010;
			break;
		case SDLK_PAGEUP:
			if (panel == MEMORY_PANEL)
				mbase -= shift ? 0x1000 : 0x0100;
			break;
		case SDLK_PAGEDOWN:
			if (panel == MEMORY_PANEL)
				mbase += shift ? 0x1000 : 0x0100;
			break;
		case SDLK_HOME:
			if (panel == MEMORY_PANEL)
				mbase &= shift ? 0xff00 : 0xfff0;
			break;
		case SDLK_END:
			if (panel == MEMORY_PANEL) {
				if (shift)
					mbase = (mbase + 0x00ff) & 0xff00;
				else
					mbase = (mbase + 0x000f) & 0xfff0;
			}
			break;
		case SDLK_s:
			showfps = !showfps;
			break;
		default:
			break;
		}
		break;

	case SDL_MOUSEWHEEL:
		if (event->window.windowID != SDL_GetWindowID(window))
			break;

		if (panel == MEMORY_PANEL) {
			if (event->wheel.preciseY < 0)
				n = (int) (event->wheel.preciseY - 0.5);
			else
				n = (int) (event->wheel.preciseY + 0.5);
			if (event->wheel.direction == SDL_MOUSEWHEEL_NORMAL)
				mbase += n * (shift ? 0x0100 : 0x0010);
			else
				mbase -= n * (shift ? 0x0100 : 0x0010);
		}
		break;

	default:
		break;
	}
}

#else /* !WANT_SDL */

/*
 * Process the X11 event queue
 */
static inline void process_events(void)
{
	char buffer[5];
	KeySym key;
	XComposeStatus compose;

	while (XPending(display)) {
		XNextEvent(display, &event);
		switch (event.type) {
		case ButtonPress:
			if (event.xbutton.button < 4)
				check_buttons(event.xbutton.x, event.xbutton.y,
					      EVENT_PRESS);
			else if (panel == MEMORY_PANEL) {
				if (event.xbutton.button == 4)
					mbase += shift ? 0x0100 : 0x0010;
				else if (event.xbutton.button == 5)
					mbase -= shift ? 0x0100 : 0x0010;
			}
			break;

		case ButtonRelease:
			if (event.xbutton.button < 4)
				check_buttons(event.xbutton.x, event.xbutton.y,
					      EVENT_RELEASE);
			break;

		case MotionNotify:
			check_buttons(event.xmotion.x, event.xmotion.y,
				      EVENT_MOTION);
			break;

		case KeyRelease:
			XLookupString(&event.xkey, buffer, sizeof(buffer),
				      &key, &compose);

			switch (key) {
			case XK_Shift_L:
			case XK_Shift_R:
				shift = false;
				break;
			default:
				break;
			}
			break;

		case KeyPress:
			XLookupString(&event.xkey, buffer, sizeof(buffer),
				      &key, &compose);

			switch (key) {
			case XK_Shift_L:
			case XK_Shift_R:
				shift = true;
				break;
			case XK_Left:
				if (panel == MEMORY_PANEL)
					mbase -= shift ? 0x0010 : 0x0001;
				break;
			case XK_Right:
				if (panel == MEMORY_PANEL)
					mbase += shift ? 0x0010 : 0x0001;
				break;
			case XK_Up:
				if (panel == MEMORY_PANEL)
					mbase -= shift ? 0x0100 : 0x0010;
				break;
			case XK_Down:
				if (panel == MEMORY_PANEL)
					mbase += shift ? 0x0100 : 0x0010;
				break;
			case XK_Page_Up:
				if (panel == MEMORY_PANEL)
					mbase -= shift ? 0x1000 : 0x0100;
				break;
			case XK_Page_Down:
				if (panel == MEMORY_PANEL)
					mbase += shift ? 0x1000 : 0x0100;
				break;
			case XK_Home:
				if (panel == MEMORY_PANEL)
					mbase &= shift ? 0xff00 : 0xfff0;
				break;
			case XK_End:
				if (panel == MEMORY_PANEL) {
					if (shift)
						mbase = (mbase + 0x00ff) &
							0xff00;
					else
						mbase = (mbase + 0x000f) &
							0xfff0;
				}
				break;
			case XK_s:
			case XK_S:
				showfps = !showfps;
				break;
			default:
				break;
			}
			break;

		default:
			break;
		}
	}
}

#endif /* !WANT_SDL */

/*
 *	Fill the pixmap with the specified color.
 */
static inline void draw_clear(const uint32_t color)
{
	uint32_t *p = pixels;
	unsigned x, y;

#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
#endif
	for (x = 0; x < xsize; x++)
		*p++ = color;
	for (y = 1; y < ysize; y++) {
		memcpy(p, pixels, pitch * 4);
		p += pitch;
	}
}

/*
 *	Draw a pixel in the specified color.
 */
static inline void draw_pixel(const unsigned x, const unsigned y,
			      const uint32_t color)
{
#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (x >= xsize || y >= ysize) {
		fprintf(stderr, "%s: coord (%d,%d) is outside (0,0)-(%d,%d)\n",
			__func__, x, y, xsize - 1, ysize - 1);
		return;
	}
#endif
	*(pixels + y * pitch + x) = color;
}

/*
 *	Draw a character in the specfied font and colors.
 */
static inline void draw_char(const unsigned x, const unsigned y, const char c,
			     const font_t *font, const uint32_t fgc,
			     const uint32_t bgc)
{
	const unsigned off = (c & 0x7f) * font->width;
	const uint8_t *p0 = font->bits + (off >> 3), *p;
	const uint8_t m0 = 0x80 >> (off & 7);
	uint8_t m;
	uint32_t *q0, *q;
	unsigned i, j;

#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (font == NULL) {
		fprintf(stderr, "%s: font is NULL\n", __func__);
		return;
	}
	if (x >= xsize || y >= ysize || x + font->width > xsize ||
	    y + font->height > ysize) {
		fprintf(stderr, "%s: char '%c' at (%d,%d)-(%d,%d) is "
			"outside (0,0)-(%d,%d)\n", __func__, c, x, y,
			x + font->width - 1, y + font->height - 1,
			xsize - 1, ysize - 1);
		return;
	}
#endif
	q0 = pixels + y * pitch + x;
	for (j = font->height; j > 0; j--) {
		m = m0;
		p = p0;
		q = q0;
		for (i = font->width; i > 0; i--) {
			if (*p & m) {
				if (fgc != C_TRANS)
					*q = fgc;
			} else {
				if (bgc != C_TRANS)
					*q = bgc;
			}
			if ((m >>= 1) == 0) {
				m = 0x80;
				p++;
			}
			q++;
		}
		p0 += font->stride;
		q0 += pitch;
	}
}

/*
 *	Draw a horizontal line in the specified color.
 */
static inline void draw_hline(const unsigned x, const unsigned y, unsigned w,
			      const uint32_t col)
{
	uint32_t *p;

#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (x >= xsize || y >= ysize || x + w > xsize) {
		fprintf(stderr, "%s: line (%d,%d)-(%d,%d) is outside "
			"(0,0)-(%d,%d)\n", __func__, x, y, x + w - 1, y,
			xsize - 1, ysize - 1);
		return;
	}
#endif
	p = pixels + y * pitch + x;
	while (w--)
		*p++ = col;
}

/*
 *	Draw a vertical line in the specified color.
 */
static inline void draw_vline(const unsigned x, const unsigned y, unsigned h,
			      const uint32_t col)
{
	uint32_t *p;

#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (x >= xsize || y >= ysize || y + h > ysize) {
		fprintf(stderr, "%s: line (%d,%d)-(%d,%d) is outside "
			"(0,0)-(%d,%d)\n", __func__, x, y, x, y + h - 1,
			xsize - 1, ysize - 1);
		return;
	}
#endif
	p = pixels + y * pitch + x;
	while (h--) {
		*p = col;
		p += pitch;
	}
}

/*
 *	Setup a text grid defined by font and spacing.
 *	If col < 0 then use the entire pixels texture width.
 *	If row < 0 then use the entire pixels texture height.
 */
static inline void draw_setup_grid(grid_t *grid, const unsigned xoff,
				   const unsigned yoff, const int cols,
				   const int rows, const font_t *font,
				   const unsigned spc)
{
#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (grid == NULL) {
		fprintf(stderr, "%s: grid is NULL\n", __func__);
		return;
	}
	if (font == NULL) {
		fprintf(stderr, "%s: font is NULL\n", __func__);
		return;
	}
	if (cols == 0) {
		fprintf(stderr," %s: number of columns is zero\n", __func__);
		return;
	}
	if (cols >= 0 && (unsigned) cols > (xsize - xoff) / font->width) {
		fprintf(stderr," %s: number of columns %d is too large\n",
			__func__, cols);
		return;
	}
	if (rows == 0) {
		fprintf(stderr," %s: number of rows is zero\n", __func__);
		return;
	}
	if (rows >= 0 && (unsigned) rows > ((ysize - yoff + spc) /
					    (font->height + spc))) {
		fprintf(stderr," %s: number of rows %d is too large\n",
			__func__, rows);
		return;
	}
#endif
	grid->font = font;
	grid->xoff = xoff;
	grid->yoff = yoff;
	grid->spc = spc;
	grid->cwidth = font->width;
	grid->cheight = font->height + spc;
	if (cols < 0)
		grid->cols = (xsize - xoff) / grid->cwidth;
	else
		grid->cols = cols;
	if (rows < 0)
		grid->rows = (ysize - yoff + spc) / grid->cheight;
	else
		grid->rows = rows;
}

/*
 *	Draw a character using grid coordinates in the specified color.
 */
static inline void draw_grid_char(const unsigned x, const unsigned y,
				  const char c, const grid_t *grid,
				  const uint32_t fgc, const uint32_t bgc)
{
#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (grid == NULL) {
		fprintf(stderr, "%s: grid is NULL\n", __func__);
		return;
	}
#endif
	draw_char(x * grid->cwidth + grid->xoff,
		  y * grid->cheight + grid->yoff,
		  c, grid->font, fgc, bgc);
}

/*
 *	Draw a horizontal grid line in the middle of the spacing
 *	above the y grid coordinate specified.
 */
static inline void draw_grid_hline(unsigned x, unsigned y, unsigned w,
				   const grid_t *grid, const uint32_t col)
{
#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (grid == NULL) {
		fprintf(stderr, "%s: grid is NULL\n", __func__);
		return;
	}
#endif
	if (w) {
		x = x * grid->cwidth;
		if (y)
			y = y * grid->cheight - (grid->spc + 1) / 2;
		w = w * grid->cwidth;
		draw_hline(x + grid->xoff, y + grid->yoff, w, col);
	}
}

/*
 *	Draw a vertical grid line in the middle of the x grid coordinate
 *	specified.
 */
static inline void draw_grid_vline(unsigned x, unsigned y, unsigned h,
				   const grid_t *grid, const uint32_t col)
{
#ifdef DRAW_DEBUG
	if (pixels == NULL) {
		fprintf(stderr, "%s: pixels texture is NULL\n", __func__);
		return;
	}
	if (grid == NULL) {
		fprintf(stderr, "%s: grid is NULL\n", __func__);
		return;
	}
#endif
	unsigned hadj = 0;

	if (h) {
		x = x * grid->cwidth + (grid->cwidth + 1) / 2;
		if (y + h < grid->rows)
			hadj += grid->spc / 2 + 1;
		if (y) {
			y = y * grid->cheight - (grid->spc + 1) / 2;
			hadj += (grid->spc + 1) / 2;
		}
		h = h * grid->cheight - grid->spc + hadj;
		draw_vline(x + grid->xoff, y + grid->yoff, h, col);
	}
}

/*
 *	Draw a LED inside a 10x10 circular bracket.
 */
static inline void draw_led(const unsigned x, const unsigned y,
			    const uint32_t col)
{
	int i;

	draw_hline(x + 2, y, 6, C_ALUM_5);
	draw_pixel(x + 1, y + 1, C_ALUM_5);
	draw_pixel(x + 8, y + 1, C_ALUM_5);
	draw_vline(x, y + 2, 6, C_ALUM_5);
	draw_vline(x + 9, y + 2, 6, C_ALUM_5);
	draw_pixel(x + 1, y + 8, C_ALUM_5);
	draw_pixel(x + 8, y + 8, C_ALUM_5);
	draw_hline(x + 2, y + 9, 6, C_ALUM_5);
	for (i = 1; i < 9; i++) {
		if (i == 1 || i == 8)
			draw_hline(x + 2, y + i, 6, col);
		else
			draw_hline(x + 1, y + i, 8, col);
	}
}

/*
 *	Draw CPU registers:
 *
 *	Z80 CPU:
 *
 *	AF 0123 BC 0123 DE 0123 HL 0123 SP 0123 PC 0123
 *	AF'0123 BC'0123 DE'0123 HL'0123 IX 0123 IY 0123
 *	F  SXHPNC       IF 12   I  00   R  00
 *
 *	8080 CPU:
 *
 *	AF 0123 BC 0123 DE 0123 HL 0123 SP 0123 PC 0123
 *	F  SZHPC        IF 1
 */

typedef struct reg {
	uint8_t x;
	uint8_t y;
	enum { RB, RW, RJ, RF, RI, RR } type;
	const char *l;
	union {
		struct {
			const BYTE *p;
		} b;
		struct {
			const WORD *p;
		} w;
		struct {
			const int *p;
		} i;
		struct {
			char c;
			uint8_t m;
		} f;
	};
} reg_t;

#define RXOFF	5	/* x pixel offset of registers text grid */
#define RYOFF	0	/* y pixel offset of registers text grid */
#define RSPC	3	/* vertical text spacing */

#ifndef EXCLUDE_Z80

static const reg_t regs_z80[] = {
	{  4, 0, RB, "AF",   .b.p = &A },
	{  6, 0, RJ, NULL,   .i.p = &F },
	{ 12, 0, RB, "BC",   .b.p = &B },
	{ 14, 0, RB, NULL,   .b.p = &C },
	{ 20, 0, RB, "DE",   .b.p = &D },
	{ 22, 0, RB, NULL,   .b.p = &E },
	{ 28, 0, RB, "HL",   .b.p = &H },
	{ 30, 0, RB, NULL,   .b.p = &L },
	{ 38, 0, RW, "SP",   .w.p = &SP },
	{ 46, 0, RW, "PC",   .w.p = &PC },
	{  4, 1, RB, "AF\'", .b.p = &A_ },
	{  6, 1, RJ, NULL,   .i.p = &F_ },
	{ 12, 1, RB, "BC\'", .b.p = &B_ },
	{ 14, 1, RB, NULL,   .b.p = &C_ },
	{ 20, 1, RB, "DE\'", .b.p = &D_ },
	{ 22, 1, RB, NULL,   .b.p = &E_ },
	{ 28, 1, RB, "HL\'", .b.p = &H_ },
	{ 30, 1, RB, NULL,   .b.p = &L_ },
	{ 38, 1, RW, "IX",   .w.p = &IX },
	{ 46, 1, RW, "IY",   .w.p = &IY },
	{  3, 2, RF, NULL,   .f.c = 'S', .f.m = S_FLAG },
	{  4, 2, RF, "F",    .f.c = 'Z', .f.m = Z_FLAG },
	{  5, 2, RF, NULL,   .f.c = 'H', .f.m = H_FLAG },
	{  6, 2, RF, NULL,   .f.c = 'P', .f.m = P_FLAG },
	{  7, 2, RF, NULL,   .f.c = 'N', .f.m = N_FLAG },
	{  8, 2, RF, NULL,   .f.c = 'C', .f.m = C_FLAG },
	{ 19, 2, RI, "IF",   .f.c = '1', .f.m = 1 },
	{ 20, 2, RI, NULL,   .f.c = '2', .f.m = 2 },
	{ 28, 2, RB, "I",    .b.p = &I },
	{ 36, 2, RR, "R",    .b.p = NULL }
};
static const int num_regs_z80 = sizeof(regs_z80) / sizeof(reg_t);

#endif /* !EXCLUDE_Z80 */

#ifndef EXCLUDE_I8080

static const reg_t regs_8080[] = {
	{  4, 0, RB, "AF", .b.p = &A },
	{  6, 0, RJ, NULL, .i.p = &F },
	{ 12, 0, RB, "BC", .b.p = &B },
	{ 14, 0, RB, NULL, .b.p = &C },
	{ 20, 0, RB, "DE", .b.p = &D },
	{ 22, 0, RB, NULL, .b.p = &E },
	{ 28, 0, RB, "HL", .b.p = &H },
	{ 30, 0, RB, NULL, .b.p = &L },
	{ 38, 0, RW, "SP", .w.p = &SP },
	{ 46, 0, RW, "PC", .w.p = &PC },
	{  3, 1, RF, NULL, .f.c = 'S', .f.m = S_FLAG },
	{  4, 1, RF, "F",  .f.c = 'Z', .f.m = Z_FLAG },
	{  5, 1, RF, NULL, .f.c = 'H', .f.m = H_FLAG },
	{  6, 1, RF, NULL, .f.c = 'P', .f.m = P_FLAG },
	{  7, 1, RF, NULL, .f.c = 'C', .f.m = C_FLAG },
	{ 19, 1, RI, "IF", .f.c = '1', .f.m = 3 }
};
static const int num_regs_8080 = sizeof(regs_8080) / sizeof(reg_t);

#endif /* !EXCLUDE_I8080 */

static void draw_cpu_regs(void)
{
	char c;
	int i, j, n = 0;
	unsigned x;
	WORD w;
	const char *s;
	const reg_t *rp = NULL;
	grid_t grid = { };
	int cpu_type = cpu;

	/* use cpu_type in the rest of this function, since cpu can change */

#ifndef EXCLUDE_Z80
	if (cpu_type == Z80) {
		rp = regs_z80;
		n = num_regs_z80;
	}
#endif
#ifndef EXCLUDE_I8080
	if (cpu_type == I8080) {
		rp = regs_8080;
		n = num_regs_8080;
	}
#endif

	/* setup text grid and draw grid lines */
#ifndef EXCLUDE_Z80
	if (cpu_type == Z80) {
		draw_setup_grid(&grid, RXOFF, RYOFF, 47, 3, &font18, RSPC);

		/* draw vertical grid lines */
		draw_grid_vline(7, 0, 2, &grid, C_ALUM_4);
		draw_grid_vline(15, 0, 3, &grid, C_ALUM_4);
		draw_grid_vline(23, 0, 3, &grid, C_ALUM_4);
		draw_grid_vline(31, 0, 3, &grid, C_ALUM_4);
		draw_grid_vline(39, 0, 2, &grid, C_ALUM_4);
		/* draw horizontal grid lines */
		draw_grid_hline(0, 1, grid.cols, &grid, C_ALUM_4);
		draw_grid_hline(0, 2, grid.cols, &grid, C_ALUM_4);
	}
#endif
#ifndef EXCLUDE_I8080
	if (cpu_type == I8080) {
		draw_setup_grid(&grid, RXOFF, RYOFF, 47, 2, &font18, RSPC);

		/* draw vertical grid lines */
		draw_grid_vline(7, 0, 1, &grid, C_ALUM_4);
		draw_grid_vline(15, 0, 2, &grid, C_ALUM_4);
		draw_grid_vline(23, 0, 1, &grid, C_ALUM_4);
		draw_grid_vline(31, 0, 1, &grid, C_ALUM_4);
		draw_grid_vline(39, 0, 1, &grid, C_ALUM_4);
		/* draw horizontal grid line */
		draw_grid_hline(0, 1, grid.cols, &grid, C_ALUM_4);
	}
#endif
	/* draw register labels & contents */
	for (i = 0; i < n; rp++, i++) {
		if ((s = rp->l) != NULL) {
			x = rp->x - (rp->type == RW ? 6 : 4);
			if (rp->type == RI)
				x++;
			while (*s)
				draw_grid_char(x++, rp->y, *s++, &grid,
					       C_ALUM_2, C_ALUM_6);
		}
		switch (rp->type) {
		case RB: /* byte sized register */
			w = *(rp->b.p);
			j = 2;
			break;
		case RW: /* word sized register */
			w = *(rp->w.p);
			j = 4;
			break;
		case RJ: /* F or F_ integer register */
			w = *(rp->i.p);
			j = 2;
			break;
		case RF: /* flags */
			draw_grid_char(rp->x, rp->y, rp->f.c, &grid,
				       (F & rp->f.m) ? C_CHAM_2 : C_RED_2,
				       C_ALUM_6);
			continue;
		case RI: /* interrupt register */
			draw_grid_char(rp->x, rp->y, rp->f.c, &grid,
				       (IFF & rp->f.m) == rp->f.m ?
				       C_CHAM_2 : C_RED_2, C_ALUM_6);
			continue;
#ifndef EXCLUDE_Z80
		case RR: /* refresh register */
			w = (R_ & 0x80) | (R & 0x7f);
			j = 2;
			break;
#endif
		default:
			continue;
		}
		x = rp->x;
		while (j--) {
			c = w & 0xf;
			c += (c < 10 ? '0' : 'A' - 10);
			draw_grid_char(x--, rp->y, c, &grid, C_CHAM_2,
				       C_ALUM_6);
			w >>= 4;
		}
	}
}

/*
 *	Draw the memory panel:
 *
 *	       0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
 *	0000  00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F  0123456789ABCDEF
 *					...
 *	00F0  00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F  0123456789ABCDEF
 */

#define MXOFF	4
#define MYOFF	68
#define MSPC	1

static void draw_memory_panel(void)
{
	char c, dc;
	int i, j;
	WORD a;
	grid_t grid;

	draw_setup_grid(&grid, MXOFF, MYOFF, 71, 17, &font16, MSPC);

	/* draw vertical grid lines */
	for (i = 0; i < 17; i++)
		draw_grid_vline(5 + i * 3, 0, grid.rows, &grid, C_ALUM_4);

	a = mbase;
	for (i = 0; i < 16; i++) {
		c = ((a & 0xf) + i) & 0xf;
		c += (c < 10 ? '0' : 'A' - 10);
		draw_grid_char(7 + i * 3, 0, c, &grid, C_ALUM_2, C_ALUM_6);
	}
	for (j = 0; j < 16; j++) {
		draw_grid_hline(0, j + 1, grid.cols, &grid, C_ALUM_4);
		c = (a >> 12) & 0xf;
		c += (c < 10 ? '0' : 'A' - 10);
		draw_grid_char(0, j + 1, c, &grid, C_ALUM_2, C_ALUM_6);
		c = (a >> 8) & 0xf;
		c += (c < 10 ? '0' : 'A' - 10);
		draw_grid_char(1, j + 1, c, &grid, C_ALUM_2, C_ALUM_6);
		c = (a >> 4) & 0xf;
		c += (c < 10 ? '0' : 'A' - 10);
		draw_grid_char(2, j + 1, c, &grid, C_ALUM_2, C_ALUM_6);
		c = a & 0xf;
		c += (c < 10 ? '0' : 'A' - 10);
		draw_grid_char(3, j + 1, c, &grid, C_ALUM_2, C_ALUM_6);
		for (i = 0; i < 16; i++) {
			c = (getmem(a) >> 4) & 0xf;
			c += (c < 10 ? '0' : 'A' - 10);
			draw_grid_char(6 + i * 3, j + 1, c, &grid, C_CHAM_2,
				       C_ALUM_6);
			c = getmem(a++) & 0xf;
			c += (c < 10 ? '0' : 'A' - 10);
			draw_grid_char(7 + i * 3, j + 1, c, &grid, C_CHAM_2,
				       C_ALUM_6);
		}
		a -= 16;
		for (i = 0; i < 16; i++) {
			c = getmem(a++);
			dc = c & 0x7f;
			if (dc < 32 || dc == 127)
				dc = '.';
			draw_grid_char(55 + i, j + 1, dc, &grid,
				       (c & 0x80) ? C_PLUM_1 : C_BUTTER_1,
				       C_ALUM_6);
		}
	}
}

/*
 *	Draw the I/O ports panel:
 *
 *	       0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
 *	0000  o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o
 *					...
 *	00F0  o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o
 */

#define IOXOFF	20
#define IOYOFF	68
#define IOSPC	1

static void draw_ports_panel(void)
{
	port_flags_t *p = port_flags;
	char c;
	int i, j;
	unsigned x, y;
	grid_t grid;

	draw_setup_grid(&grid, IOXOFF, IOYOFF, 67, 17, &font16, IOSPC);

	/* draw vertical grid lines */
	for (i = 0; i < 16; i++)
		draw_grid_vline(3 + i * 4, 0, grid.rows, &grid, C_ALUM_4);

	for (i = 0; i < 16; i++) {
		c = i + (i < 10 ? '0' : 'A' - 10);
		draw_grid_char(5 + i * 4, 0, c, &grid, C_ALUM_2, C_ALUM_6);
	}
	for (j = 0; j < 16; j++) {
		draw_grid_hline(0, j + 1, grid.cols, &grid, C_ALUM_4);
		c = j + (j < 10 ? '0' : 'A' - 10);
		draw_grid_char(0, j + 1, c, &grid, C_ALUM_2, C_ALUM_6);
		draw_grid_char(1, j + 1, '0', &grid, C_ALUM_2, C_ALUM_6);
		for (i = 0; i < 16; i++) {
			x = (4 + i * 4) * grid.cwidth + grid.xoff + 1;
			y = (j + 1) * grid.cheight + grid.yoff + 3;
			draw_led(x, y, p->in ? C_CHAM_2 : C_ALUM_6);
			draw_led(x + 13, y, p->out ? C_RED_2 : C_ALUM_6);
			p++;
		}
	}

	/* clear access flags if sticky flag is not set */
	if (!sticky)
		memset(port_flags, 0, sizeof(port_flags));
}

/*
 *	Draw the info line:
 *
 *	Z80pack x.xx		xxxx.xx MHz
 */
static void draw_info(bool tick)
{
	char c;
	int i, f, digit;
	bool onlyz;
	const char *s;
	const font_t *font = &font18;
	const unsigned w = font->width;
	const unsigned n = xsize / w;
	const unsigned x = (xsize - n * w) / 2;
	const unsigned y = ysize - font->height;
	static unsigned count, fps;
	static uint64_t freq;

	/* draw product info */
	s = "Z80pack " RELEASE;
	for (i = 0; *s; i++)
		draw_char(i * w + x, y, *s++, font, C_ORANGE_1, C_ALUM_6);

	/* draw frequency label */
	draw_char((n - 7) * w + x, y, '.', font, C_ORANGE_1, C_ALUM_6);
	draw_char((n - 3) * w + x, y, 'M', font, C_ORANGE_1, C_ALUM_6);
	draw_char((n - 2) * w + x, y, 'H', font, C_ORANGE_1, C_ALUM_6);
	draw_char((n - 1) * w + x, y, 'z', font, C_ORANGE_1, C_ALUM_6);

	/* update fps every second */
	count++;
	if (tick) {
		fps = count;
		count = 0;
	}
	if (showfps) {
		draw_char(30 * w + x, y, fps > 99 ? fps / 100 + '0' : ' ',
			  font, C_ORANGE_1, C_ALUM_6);
		draw_char(31 * w + x, y, fps > 9 ? (fps / 10) % 10 + '0' : ' ',
			  font, C_ORANGE_1, C_ALUM_6);
		draw_char(32 * w + x, y, fps % 10 + '0',
			  font, C_ORANGE_1, C_ALUM_6);
		draw_char(34 * w + x, y, 'f', font, C_ORANGE_1, C_ALUM_6);
		draw_char(35 * w + x, y, 'p', font, C_ORANGE_1, C_ALUM_6);
		draw_char(36 * w + x, y, 's', font, C_ORANGE_1, C_ALUM_6);
	}

	/* update frequency every second */
	if (tick)
		freq = cpu_freq;
	f = (unsigned) (freq / 10000);
	digit = 100000;
	onlyz = true;
	for (i = 0; i < 7; i++) {
		c = '0';
		while (f > digit) {
			f -= digit;
			c++;
		}
		if (onlyz && i < 3 && c == '0')
			c = ' ';
		else
			onlyz = false;
		draw_char((n - 11 + i) * w + x, y, c,
			  font, C_ORANGE_1, C_ALUM_6);
		if (i < 6)
			digit /= 10;
		if (i == 3)
			i++; /* skip decimal point */
	}
}

/*
 *	Draw buttons
 */
static void draw_buttons(void)
{
	int i;
	unsigned x, y;
	button_t *p = buttons;
	uint32_t color;
	const char *s;

	for (i = 0; i < nbuttons; i++) {
		if (p->enabled) {
			color = p->hilighted ? C_ORANGE_1 : C_ALUM_2;
			draw_hline(p->x + 2, p->y, p->width - 4, color);
			draw_pixel(p->x + 1, p->y + 1, color);
			draw_pixel(p->x + p->width - 2, p->y + 1, color);
			draw_vline(p->x, p->y + 2, p->height - 4, color);
			draw_vline(p->x + p->width - 1, p->y + 2,
				   p->height - 4, color);
			draw_pixel(p->x + 1, p->y + p->height - 2, color);
			draw_pixel(p->x + p->width - 2, p->y + p->height - 2,
				   color);
			draw_hline(p->x + 2, p->y + p->height - 1,
				   p->width - 4, color);

			color = C_ALUM_6;
			if (p->active)
				color = C_CHAM_3;
			if (p->pressed)
				color = C_BLUE_1;
			draw_hline(p->x + 2, p->y + 1, p->width - 4, color);
			for (y = p->y + 2; y < p->y + p->height - 2; y++)
				draw_hline(p->x + 1, y, p->width - 2, color);
			draw_hline(p->x + 2, p->y + p->height - 2,
				   p->width - 4, color);

			x = p->x + 1 + (p->width - 2 - strlen(p->text) *
					p->font->width) / 2;
			y = p->y + 1 + (p->height - 2 - p->font->height) / 2;
			for (s = p->text; *s; s++) {
				draw_char(x, y, *s, p->font, C_CHOC_1, C_TRANS);
				x += p->font->width;
			}
		}
		p++;
	}
}

/*
 *	Update button states
 */
static void update_buttons(void)
{
	int i;
	button_t *p = buttons;

	for (i = 0; i < nbuttons; i++) {
		if (p->clicked) {
			switch (i) {
			case MEMORY_BUTTON:
				if (panel != MEMORY_PANEL) {
					buttons[MEMORY_BUTTON].active = true;
					buttons[PORTS_BUTTON].active = false;
					buttons[STICKY_BUTTON].enabled = false;
					panel = MEMORY_PANEL;
				}
				break;
			case PORTS_BUTTON:
				if (panel != PORTS_PANEL) {
					buttons[MEMORY_BUTTON].active = false;
					buttons[PORTS_BUTTON].active = true;
					buttons[STICKY_BUTTON].enabled = true;
					panel = PORTS_PANEL;
				}
				break;
			case STICKY_BUTTON:
				sticky = !sticky;
				buttons[STICKY_BUTTON].active = sticky;
				break;
			default:
				break;
			}
			p->clicked = false;
			break;
		}
		p++;
	}
}

/*
 *	Check for button presses
 */
static void check_buttons(unsigned x, unsigned y, int event)
{
	int i;
	bool inside;
	button_t *p = buttons;

	for (i = 0; i < nbuttons; i++) {
		if (p->enabled) {
			inside = (x >= p->x && x < p->x + p->width &&
				  y >= p->y && y < p->y + p->height);
			switch (event) {
			case EVENT_PRESS:
				if (inside)
					p->pressed = true;
				break;
			case EVENT_RELEASE:
				if (inside && p->pressed)
					p->clicked = true;
				p->pressed = false;
				break;
			case EVENT_MOTION:
				if (inside)
					p->hilighted = true;
				else {
					p->pressed = false;
					p->hilighted = false;
				}
				break;
			default:
				break;
			}
		}
		p++;
	}
}

/*
 *	Refresh the display buffer
 */
static void refresh(bool tick)
{
	draw_clear(C_ALUM_6);

	update_buttons();
	draw_buttons();
	draw_cpu_regs();
	if (panel == MEMORY_PANEL)
		draw_memory_panel();
	else if (panel == PORTS_PANEL)
		draw_ports_panel();
	draw_info(tick);
}

#ifdef WANT_SDL

/* function for updating the display */
static void update_display(bool tick)
{
	SDL_LockTexture(texture, NULL, (void **) &pixels, &pitch);
	pitch /= 4;
	refresh(tick);
	SDL_UnlockTexture(texture);
	SDL_RenderCopy(renderer, texture, NULL, NULL);
	SDL_RenderPresent(renderer);
}

static win_funcs_t panel_funcs = {
	open_display,
	close_display,
	process_event,
	update_display
};

#else /* !WANT_SDL */

static void kill_thread(void)
{
	if (thread != 0) {
		sleep_for_ms(50);
		pthread_cancel(thread);
		pthread_join(thread, NULL);
		thread = 0;
	}
}

/* thread for updating the display */
static void *update_display(void *arg)
{
	uint64_t t1, t2, ttick;
	long tleft;
	bool tick = true;

	UNUSED(arg);

	t1 = get_clock_us();
	ttick = t1 + 1000000;

	while (true) {
		/* lock display, don't cancel thread while locked */
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
		XLockDisplay(display);

		/* process X11 event queue */
		process_events();

		/* update display window */
		refresh(tick);
		XPutImage(display, window, gc, ximage, 0, 0, 0, 0,
			  xsize, ysize);
		XSync(display, False);

		/* unlock display, thread can be canceled again */
		XUnlockDisplay(display);
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

		t2 = get_clock_us();

		/* update seconds tick */
		if ((tick = (t2 >= ttick)))
			ttick = t2 + 1000000;

		/* sleep rest to 16667us so that we get 60 fps */
		tleft = 16667L - (long) (t2 - t1);
		if (tleft > 0)
			sleep_for_us(tleft);

		t1 = get_clock_us();
	}

	/* just in case it ever gets here */
	pthread_exit(NULL);
}

#endif /* !WANT_SDL */

void init_panel(void)
{
#ifdef WANT_SDL
	if (panel_win_id < 0)
		panel_win_id = simsdl_create(&panel_funcs);
#else
	if (display == NULL) {
		open_display();

		if (pthread_create(&thread, NULL, update_display, (void *) NULL)) {
			LOGE(TAG, "can't create thread");
			exit(EXIT_FAILURE);
		}
	}
#endif

	panel = MEMORY_PANEL;
	buttons[MEMORY_BUTTON].enabled = true;
	buttons[MEMORY_BUTTON].active = true;
	buttons[PORTS_BUTTON].enabled = true;
}

void exit_panel(void)
{
#ifdef WANT_SDL
	if (panel_win_id >= 0) {
		simsdl_destroy(panel_win_id);
		panel_win_id = -1;
	}
#else /* !WANT_SDL */
	kill_thread();
	if (display != NULL)
		close_display();
#endif /* !WANT_SDL */
}
