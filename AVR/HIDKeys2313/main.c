/* Name: main.c
 * Project: HID-Test
 * Author: Christian Starkjohann
 * Creation Date: 2006-02-02
 * Tabsize: 4
 * Copyright: (c) 2006 by OBJECTIVE DEVELOPMENT Software GmbH
 * License: GNU GPL v2 (see License.txt) or proprietary (CommercialLicense.txt)
 * This Revision: $Id: main.c 299 2007-03-29 17:07:19Z cs $
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <avr/wdt.h>

#include "usbdrv.h"
#include "oddebug.h"

#include "avr8defs.h"

//#define F_CPU   12000000L    /* evaluation board runs on 4MHz */


/* ----------------------- hardware I/O abstraction ------------------------ */

/* pin assignments:
PB0	Key 1
PB1	Key 2
PB2	Key 3
PB3	Key 4
PB4	Key 5
PB5 Key 6
PB6 Key 7
PB7 Key 8

PD2	USB+ (INT0)
PD3	USB-

PD6	Toggle +8
*/

static void hardwareInit(void)
{
uchar	i, j;

    PORTB = 0xff;   /* activate all pull-ups */
    DDRB = 0;       /* all pins input */
//  PORTC = 0xff;   /* activate all pull-ups */
//  DDRC = 0;       /* all pins input */
//  PORTD = 0xfa;   /* 1111 1010 bin: activate pull-ups except on USB lines */
    PORTD = 0b1110011; // there is no PD7, read-only bit always read as 0
	DDRD  = 0b0001100;    /* 0000 0111 bin: all pins input except USB (-> USB reset) */
	j = 0;
	while(--j) {     /* USB Reset by device only required on Watchdog Reset */
		i = 0;
		while(--i); /* delay >10ms for USB reset */
	}
    DDRD = 0x00; //0x02;    /* 0000 0010 bin: remove USB reset condition */
    /* configure timer 0 for a rate of 12M/(1024 * 256) = 45.78 Hz (~22ms) */
//  TCCR0 = 5;      /* timer 0 prescaler: 1024 */
    TCCR0B = 5;      /* timer 0 prescaler: 1024 */
}

/* ------------------------------------------------------------------------- */

#define NUM_KEYS    16
//#define NUM_KEYS    8

/* The following function returns an index for the first key pressed. It
 * returns 0 if no key is pressed.
 */
static uchar    keyPressed(void) {
	uchar  i, x, result = 0;
	
    x = PINB;
	for (i=0;i<8;i++) {
		if ( TestBit(x,0) == 0 /*(x & 0b01) == 0 */ ) { 
			result = i + 1;
			break;
		}
		x >>= 1;
	}
	if (result == 0)
		return 0;
	x = PIND;
	//
	if ( TestBit(x,6) == 0) 
		result += 8;
    return result;
}

/* ------------------------------------------------------------------------- */
/* ----------------------------- USB interface ----------------------------- */
/* ------------------------------------------------------------------------- */

static uchar    reportBuffer[2];    /* buffer for HID reports */
static uchar    idleRate;           /* in 4 ms units */

PROGMEM char usbHidReportDescriptor[35] = { /* USB report descriptor */
    0x05, 0x01,                    // USAGE_PAGE (Generic Desktop)
    0x09, 0x06,                    // USAGE (Keyboard)
    0xa1, 0x01,                    // COLLECTION (Application)
    0x05, 0x07,                    //   USAGE_PAGE (Keyboard)
    0x19, 0xe0,                    //   USAGE_MINIMUM (Keyboard LeftControl)
    0x29, 0xe7,                    //   USAGE_MAXIMUM (Keyboard Right GUI)
    0x15, 0x00,                    //   LOGICAL_MINIMUM (0)
    0x25, 0x01,                    //   LOGICAL_MAXIMUM (1)
    0x75, 0x01,                    //   REPORT_SIZE (1)
    0x95, 0x08,                    //   REPORT_COUNT (8)
    0x81, 0x02,                    //   INPUT (Data,Var,Abs)
    0x95, 0x01,                    //   REPORT_COUNT (1)
    0x75, 0x08,                    //   REPORT_SIZE (8)
    0x25, 0x65,                    //   LOGICAL_MAXIMUM (101)
    0x19, 0x00,                    //   USAGE_MINIMUM (Reserved (no event indicated))
    0x29, 0x65,                    //   USAGE_MAXIMUM (Keyboard Application)
    0x81, 0x00,                    //   INPUT (Data,Ary,Abs)
    0xc0                           // END_COLLECTION
};
/* We use a simplifed keyboard report descriptor which does not support the
 * boot protocol. We don't allow setting status LEDs and we only allow one
 * simultaneous key press (except modifiers). We can therefore use short
 * 2 byte input reports.
 * The report descriptor has been created with usb.org's "HID Descriptor Tool"
 * which can be downloaded from http://www.usb.org/developers/hidpage/.
 * Redundant entries (such as LOGICAL_MINIMUM and USAGE_PAGE) have been omitted
 * for the second INPUT item.
 */

/* Keyboard usage values, see usb.org's HID-usage-tables document, chapter
 * 10 Keyboard/Keypad Page for more codes.
 */
#define MOD_CONTROL_LEFT    (1<<0)
#define MOD_SHIFT_LEFT      (1<<1)
#define MOD_ALT_LEFT        (1<<2)
#define MOD_GUI_LEFT        (1<<3)
#define MOD_CONTROL_RIGHT   (1<<4)
#define MOD_SHIFT_RIGHT     (1<<5)
#define MOD_ALT_RIGHT       (1<<6)
#define MOD_GUI_RIGHT       (1<<7)

#define KEY_A       4
#define KEY_B       5
#define KEY_C       6
#define KEY_D       7
#define KEY_E       8
#define KEY_F       9
#define KEY_G       10
#define KEY_H       11
#define KEY_I       12
#define KEY_J       13
#define KEY_K       14
#define KEY_L       15
#define KEY_M       16
#define KEY_N       17
#define KEY_O       18
#define KEY_P       19
#define KEY_Q       20
#define KEY_R       21
#define KEY_S       22
#define KEY_T       23
#define KEY_U       24
#define KEY_V       25
#define KEY_W       26
#define KEY_X       27
#define KEY_Y       28
#define KEY_Z       29
#define KEY_1       30
#define KEY_2       31
#define KEY_3       32
#define KEY_4       33
#define KEY_5       34
#define KEY_6       35
#define KEY_7       36
#define KEY_8       37
#define KEY_9       38
#define KEY_0       39
#define KEY_RETURN  40
#define KEY_ESCAPE  41
#define KEY_DELETE  42
#define KEY_TAB     43
#define KEY_SPACEBAR 44

#define KEY_Comma   54
#define KEY_Period  55
#define KEY_Slash   56

#define KEY_F1      58
#define KEY_F2      59
#define KEY_F3      60
#define KEY_F4      61
#define KEY_F5      62
#define KEY_F6      63
#define KEY_F7      64
#define KEY_F8      65
#define KEY_F9      66
#define KEY_F10     67
#define KEY_F11     68
#define KEY_F12     69

#define KEY_RightArrow    79
#define KEY_LeftArrow     80
#define KEY_DownArrow     81
#define KEY_UpArrow       82

//ABDC
static const uchar  keyReport[NUM_KEYS + 1][2] PROGMEM = {
/* none */  {0, 0},                     /* no key pressed */
/*  1 */    {MOD_SHIFT_LEFT, KEY_F5},
/*  2 */    {0, KEY_ESCAPE},
/*  3 */    {0, KEY_RightArrow},
/*  4 */    {0, KEY_LeftArrow},
/*  5 */    {MOD_SHIFT_LEFT, KEY_E},
/*  6 */    {0, KEY_RETURN},
/*  7 */    {0, KEY_LeftArrow},
/*  8 */    {0, KEY_RightArrow},
/*  9 */    {MOD_SHIFT_LEFT, KEY_I},
/* 10 */    {MOD_SHIFT_LEFT, KEY_J},
/* 11 */    {MOD_SHIFT_LEFT, KEY_K},
/* 12 */    {MOD_SHIFT_LEFT, KEY_L},
/* 13 */    {MOD_SHIFT_LEFT, KEY_M},
/* 14 */    {MOD_SHIFT_LEFT, KEY_N},
/* 15 */    {MOD_SHIFT_LEFT, KEY_O},
/* 16 */    {MOD_SHIFT_LEFT, KEY_P}
};

static void buildReport(uchar key)
{
/* This (not so elegant) cast saves us 10 bytes of program memory */
    *(int *)reportBuffer = pgm_read_word(keyReport[key]);
}

uchar	usbFunctionSetup(uchar data[8])
{
usbRequest_t    *rq = (void *)data;

    usbMsgPtr = reportBuffer;
    if((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS){    /* class request type */
        if(rq->bRequest == USBRQ_HID_GET_REPORT){  /* wValue: ReportType (highbyte), ReportID (lowbyte) */
            /* we only have one report type, so don't look at wValue */
            buildReport(keyPressed());
            return sizeof(reportBuffer);
        }else if(rq->bRequest == USBRQ_HID_GET_IDLE){
            usbMsgPtr = &idleRate;
            return 1;
        }else if(rq->bRequest == USBRQ_HID_SET_IDLE){
            idleRate = rq->wValue.bytes[1];
        }
    }else{
        /* no vendor specific requests implemented */
    }
	return 0;
}

/* ------------------------------------------------------------------------- */

int	main(void)
{
uchar   key, lastKey = 0, keyDidChange = 0;
uchar   idleCounter = 0;

	wdt_enable(WDTO_2S);
    hardwareInit();
	odDebugInit();
	usbInit();
	
	sei();
    DBG1(0x00, 0, 0);
	for(;;){	/* main event loop */
		wdt_reset();
		usbPoll();
        key = keyPressed();
        if(lastKey != key){
            lastKey = key;
            keyDidChange = 1;
        } 
        if(TIFR & (1<<TOV0)){   /* 22 ms timer */
            TIFR = 1<<TOV0;
            if(idleRate != 0){
                if(idleCounter > 4){
                    idleCounter -= 5;   /* 22 ms in units of 4 ms */
                }else{
                    idleCounter = idleRate;
                    keyDidChange = 1;
                }
            }
        }
        if(keyDidChange && usbInterruptIsReady()){
            keyDidChange = 0;
            /* use last key and not current key status in order to avoid lost
               changes in key status. */
            buildReport(lastKey);
            usbSetInterrupt(reportBuffer, sizeof(reportBuffer));
        }
	}
	return 0;
}

/* ------------------------------------------------------------------------- */

