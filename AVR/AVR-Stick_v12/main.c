/* Name: main.c
 * Project: Datalogger based on AVR USB driver
 * Original Author: Christian Starkjohann
 * Edited by Ryan Owens (SparkFun Electronics)
 * Creation Date: 2006-04-23
 * Edited 2009-06-30
 * Tabsize: 4
 * Copyright: (c) 2006 by OBJECTIVE DEVELOPMENT Software GmbH
 * License: Proprietary, free under certain conditions. See Documentation.
 * This Revision: $Id: main.c 537 2008-02-28 21:13:01Z cs $
 */

#include <avr/io.h>
#include <avr/wdt.h>
#include <avr/eeprom.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <util/delay.h>

#include "usbdrv.h"
#include "oddebug.h"

/*
Pin assignment:
PB3 = analog input (ADC3)
PB4 = analog input (ADC2) - Can alternately be used as an LED output.
PB1 = LED output

PB0, PB2 = USB data lines
*/

#define WHITE_LED 3
#define YELLOW_LED 1


#define UTIL_BIN4(x)        (uchar)((0##x & 01000)/64 + (0##x & 0100)/16 + (0##x & 010)/4 + (0##x & 1))
#define UTIL_BIN8(hi, lo)   (uchar)(UTIL_BIN4(hi) * 16 + UTIL_BIN4(lo))

#define sbi(var, mask)   ((var) |= (uint8_t)(1 << mask))
#define cbi(var, mask)   ((var) &= (uint8_t)~(1 << mask))

#ifndef NULL
#define NULL    ((void *)0)
#endif

/* ------------------------------------------------------------------------- */

static uchar    reportBuffer[2];    /* buffer for HID reports */
static uchar    idleRate;           /* in 4 ms units */

static uchar    adcPending;
static uchar	currentAdcChannel=2;
static unsigned int		sampleNum=0;

static uchar    valueBuffer[16];
static uchar    *nextDigit;

/* ------------------------------------------------------------------------- */

PROGMEM char usbHidReportDescriptor[USB_CFG_HID_REPORT_DESCRIPTOR_LENGTH] = { /* USB report descriptor */
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
#define KEY_TAB		43
#define KEY_A		4
#define KEY_B		5
#define KEY_C		6
#define KEY_D		7
#define KEY_E		8
#define KEY_F		9

/* ------------------------------------------------------------------------- */

static void buildReport(void)
{
uchar   key = 0;

    if(nextDigit != NULL){
        key = *nextDigit;
    }
    reportBuffer[0] = 0;    /* no modifiers */
    reportBuffer[1] = key;
}

/* ------------------------------------------------------------------------- */
static void changeAdcChannel(void)
{
	if(currentAdcChannel==3){
		currentAdcChannel=2;
		ADMUX = UTIL_BIN8(1001, 0010);	/* Vref=2.56V, measure ADC2 */
    }
	else{
		currentAdcChannel=3;
		ADMUX = UTIL_BIN8(1001, 0011);  /* Vref=2.56V, measure ADC3 */
	}
}

/* ------------------------------------------------------------------------- */
static void evaluateADC(unsigned int value)
{
uchar   digit;
unsigned int	timestamp;	//Keep track of how many readings we've made so we can add it to the report.

    value += value + (value >> 1);  /* value = value * 2.5 for output in mV */
	
    nextDigit = &valueBuffer[sizeof(valueBuffer)];	//The Buffer is constructed 'backwards,' so point to the end of the Value Buffer before adding the values.
    *--nextDigit = 0xff;/* terminate with 0xff */
    *--nextDigit = 0;
    if(currentAdcChannel==2)*--nextDigit = KEY_TAB;	//We'll seperate the two ADC readings with a 'TAB'
	else *--nextDigit = KEY_RETURN;						//After the second ADC reading, a carriage return character will be sent.
    //Convert the ADC reading to ASCII.
	do{
        digit = value % 10;
        value /= 10;
        *--nextDigit = 0;
        if(digit == 0){
            *--nextDigit = KEY_0;
        }else{
            *--nextDigit = KEY_1 - 1 + digit;
        }
    }while(value != 0);
	
	//Prepend each report with a timetamp.
	if(currentAdcChannel==2){
		timestamp=sampleNum++;
		*--nextDigit = KEY_TAB;
		do{
			digit = timestamp % 10;
			timestamp /= 10;
			*--nextDigit = 0;
			if(digit == 0){
				*--nextDigit = KEY_0;
			}else{
				*--nextDigit = KEY_1 - 1 + digit;
			}
		}while(timestamp != 0);
	}

}

/* ------------------------------------------------------------------------- */

static void adcPoll(void)
{
    if(adcPending && !(ADCSRA & (1 << ADSC))){
		adcPending = 0;
		evaluateADC(ADC);
		changeAdcChannel();
    }
}

/* ------------------------------------------------------------------------- */
static void timerPoll(void)
{
static uchar timerCnt;

    if(TIFR & (1 << TOV1)){	//This flag is triggered at 60 hz.
        TIFR = (1 << TOV1); /* clear overflow */
		if(++timerCnt >= 31){		 /* ~ 0.5 second interval */
            timerCnt = 0;
			adcPending = 1;
			ADCSRA |= (1 << ADSC);  /* start next conversion */
		}
	}
}

/* ------------------------------------------------------------------------- */
static void timerInit(void)
{
    TCCR1 = 0x0b;           /* select clock: 16.5M/1k -> overflow rate = 16.5M/256k = 62.94 Hz */
}

/* ------------------------------------------------------------------------- */
static void adcInit(void)
{
    ADMUX = UTIL_BIN8(1001, 0010);	 /* vref = 2.56V, measure ADC2 (PB4) */
	ADCSRA = UTIL_BIN8(1000, 0111); /* enable ADC, not free running, interrupt disable, rate = 1/128 */
}



/* ------------------------------------------------------------------------- */
/* ------------------------ interface to USB driver ------------------------ */
/* ------------------------------------------------------------------------- */

uchar	usbFunctionSetup(uchar data[8])
{
usbRequest_t    *rq = (void *)data;

    usbMsgPtr = reportBuffer;
    if((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS){    /* class request type */
        if(rq->bRequest == USBRQ_HID_GET_REPORT){  /* wValue: ReportType (highbyte), ReportID (lowbyte) */
            /* we only have one report type, so don't look at wValue */
            buildReport();
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
/* ------------------------ Oscillator Calibration ------------------------- */
/* ------------------------------------------------------------------------- */

/* Calibrate the RC oscillator to 8.25 MHz. The core clock of 16.5 MHz is
 * derived from the 66 MHz peripheral clock by dividing. Our timing reference
 * is the Start Of Frame signal (a single SE0 bit) available immediately after
 * a USB RESET. We first do a binary search for the OSCCAL value and then
 * optimize this value with a neighboorhod search.
 * This algorithm may also be used to calibrate the RC oscillator directly to
 * 12 MHz (no PLL involved, can therefore be used on almost ALL AVRs), but this
 * is wide outside the spec for the OSCCAL value and the required precision for
 * the 12 MHz clock! Use the RC oscillator calibrated to 12 MHz for
 * experimental purposes only!
 */
static void calibrateOscillator(void)
{
uchar       step = 128;
uchar       trialValue = 0, optimumValue;
int         x, optimumDev, targetValue = (unsigned)(1499 * (double)F_CPU / 10.5e6 + 0.5);

    /* do a binary search: */
    do{
        OSCCAL = trialValue + step;
        x = usbMeasureFrameLength();    /* proportional to current real frequency */
        if(x < targetValue)             /* frequency still too low */
            trialValue += step;
        step >>= 1;
    }while(step > 0);
    /* We have a precision of +/- 1 for optimum OSCCAL here */
    /* now do a neighborhood search for optimum value */
    optimumValue = trialValue;
    optimumDev = x; /* this is certainly far away from optimum */
    for(OSCCAL = trialValue - 1; OSCCAL <= trialValue + 1; OSCCAL++){
        x = usbMeasureFrameLength() - targetValue;
        if(x < 0)
            x = -x;
        if(x < optimumDev){
            optimumDev = x;
            optimumValue = OSCCAL;
        }
    }
    OSCCAL = optimumValue;
}
/*
Note: This calibration algorithm may try OSCCAL values of up to 192 even if
the optimum value is far below 192. It may therefore exceed the allowed clock
frequency of the CPU in low voltage designs!
You may replace this search algorithm with any other algorithm you like if
you have additional constraints such as a maximum CPU clock.
For version 5.x RC oscillators (those with a split range of 2x128 steps, e.g.
ATTiny25, ATTiny45, ATTiny85), it may be useful to search for the optimum in
both regions.
*/

void    usbEventResetReady(void)
{
    calibrateOscillator();
    eeprom_write_byte(0, OSCCAL);   /* store the calibrated value in EEPROM */
}

/* ------------------------------------------------------------------------- */
/* --------------------------------- main ---------------------------------- */
/* ------------------------------------------------------------------------- */

int main(void)
{
//uchar   i;
unsigned int i;
uchar   calibrationValue;

    calibrationValue = eeprom_read_byte(0); /* calibration value from last time */
    if(calibrationValue != 0xff){
        OSCCAL = calibrationValue;
    }
    //odDebugInit();
	
	//Production Test Routine - Turn on both LEDs and an LED on the SparkFun Pogo Test Bed.
	DDRB |= 1 << WHITE_LED | 1 << YELLOW_LED | 1<<4;   /* output for LED */
	sbi(PORTB, WHITE_LED);
    for(i=0;i<20;i++){  /* 300 ms disconnect */
        _delay_ms(15);
    }
	cbi(PORTB, WHITE_LED);
	
	sbi(PORTB, YELLOW_LED);
    for(i=0;i<20;i++){  /* 300 ms disconnect */
        _delay_ms(15);
    }
	cbi(PORTB, YELLOW_LED);
	
	sbi(PORTB, 4);
    for(i=0;i<20;i++){  /* 300 ms disconnect */
        _delay_ms(15);
    }	
	cbi(PORTB, 4);
	
	DDRB &= ~(1<<4);
	
	//Initialize the USB Connection with the host computer.
    usbDeviceDisconnect();
    for(i=0;i<20;i++){  /* 300 ms disconnect */
        _delay_ms(15);
    }
    usbDeviceConnect();
    
    wdt_enable(WDTO_1S);
    
	timerInit();	//Create a timer that will trigger a flag at a ~60hz rate 
    adcInit();		//Setup the ADC conversions
    usbInit();		//Initialize USB comm.
    sei();
    for(;;){    /* main event loop */
        wdt_reset();
        usbPoll();	//Check to see if it's time to send a USB packet
        if(usbInterruptIsReady() && nextDigit != NULL){ /* we can send another key */
            buildReport();	//Get the next 'key press' to send to the host. 
            usbSetInterrupt(reportBuffer, sizeof(reportBuffer));
            if(*++nextDigit == 0xff)    /* this was terminator character */
                nextDigit = NULL;
        }
        timerPoll();	//Check timer to see if it's time to start another ADC conversion.
        adcPoll();		//If an ADC conversion was started, get the value and switch to the other ADC channel for the next conversion.
    }
    return 0;
}

