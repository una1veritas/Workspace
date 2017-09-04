/*
   gaineravr project:
   -- pepper, tiny gainer --
   main.c
*/

#include <string.h>
#include <avr/io.h>
#include <avr/eeprom.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <avr/wdt.h>
#include <util/delay.h>

#include "usbdrv.h"
#include "oddebug.h"
#include "gio.h"

#if UART_CFG_HAVE_USART
#define HW_CDC_BULK_OUT_SIZE     8
#else
#define HW_CDC_BULK_OUT_SIZE     1
#endif
#define HW_CDC_BULK_IN_SIZE      8
/* Size of bulk transfer packets. The standard demands 8 bytes, but we may
 * be better off with less. Try smaller values if the communication hangs.
 */

enum {
    SEND_ENCAPSULATED_COMMAND = 0,
    GET_ENCAPSULATED_RESPONSE,
    SET_COMM_FEATURE,
    GET_COMM_FEATURE,
    CLEAR_COMM_FEATURE,
    SET_LINE_CODING = 0x20,
    GET_LINE_CODING,
    SET_CONTROL_LINE_STATE,
    SEND_BREAK
};


static PROGMEM char deviceDescrCDC[] = {    /* USB device descriptor */
    18,         /* sizeof(usbDescriptorDevice): length of descriptor in bytes */
    USBDESCR_DEVICE,        /* descriptor type */
    0x01, 0x01,             /* USB version supported */
    0x02,                   /* device class: CDC */
    0,                      /* subclass */
    0,                      /* protocol */
    8,                      /* max packet size */
    USB_CFG_VENDOR_ID,      /* 2 bytes */
    0xe1, 0x05,             /* 2 bytes: shared PID for CDC-ACM devices */
    USB_CFG_DEVICE_VERSION, /* 2 bytes */
    1,                      /* manufacturer string index */
    2,                      /* product string index */
    0,                      /* serial number string index */
    1,                      /* number of configurations */
};

static PROGMEM char configDescrCDC[] = {   /* USB configuration descriptor */
    9,          /* sizeof(usbDescrConfig): length of descriptor in bytes */
    USBDESCR_CONFIG,    /* descriptor type */
    67,
    0,          /* total length of data returned (including inlined descriptors) */
    2,          /* number of interfaces in this configuration */
    1,          /* index of this configuration */
    0,          /* configuration name string index */
#if USB_CFG_IS_SELF_POWERED
    USBATTR_SELFPOWER,  /* attributes */
#else
    USBATTR_BUSPOWER,   /* attributes */
#endif
    USB_CFG_MAX_BUS_POWER/2,            /* max USB current in 2mA units */

    /* interface descriptor follows inline: */
    9,          /* sizeof(usbDescrInterface): length of descriptor in bytes */
    USBDESCR_INTERFACE, /* descriptor type */
    0,          /* index of this interface */
    0,          /* alternate setting for this interface */
    USB_CFG_HAVE_INTRIN_ENDPOINT,   /* endpoints excl 0: number of endpoint descriptors to follow */
    USB_CFG_INTERFACE_CLASS,
    USB_CFG_INTERFACE_SUBCLASS,
    USB_CFG_INTERFACE_PROTOCOL,
    0,          /* string index for interface */

    /* CDC Class-Specific descriptor */
    5,           /* sizeof(usbDescrCDC_HeaderFn): length of descriptor in bytes */
    0x24,        /* descriptor type */
    0,           /* header functional descriptor */
    0x10, 0x01,

    4,           /* sizeof(usbDescrCDC_AcmFn): length of descriptor in bytes */
    0x24,        /* descriptor type */
    2,           /* abstract control management functional descriptor */
    0x02,        /* SET_LINE_CODING,    GET_LINE_CODING, SET_CONTROL_LINE_STATE    */

    5,           /* sizeof(usbDescrCDC_UnionFn): length of descriptor in bytes */
    0x24,        /* descriptor type */
    6,           /* union functional descriptor */
    0,           /* CDC_COMM_INTF_ID */
    1,           /* CDC_DATA_INTF_ID */

    5,           /* sizeof(usbDescrCDC_CallMgtFn): length of descriptor in bytes */
    0x24,        /* descriptor type */
    1,           /* call management functional descriptor */
    3,           /* allow management on data interface, handles call management by itself */
    1,           /* CDC_DATA_INTF_ID */

    /* Endpoint Descriptor */
    7,           /* sizeof(usbDescrEndpoint) */
    USBDESCR_ENDPOINT,  /* descriptor type = endpoint */
    0x83,        /* IN endpoint number 3 */
    0x03,        /* attrib: Interrupt endpoint */
    8, 0,        /* maximum packet size */
    USB_CFG_INTR_POLL_INTERVAL,        /* in ms */

    /* Interface Descriptor  */
    9,           /* sizeof(usbDescrInterface): length of descriptor in bytes */
    USBDESCR_INTERFACE,           /* descriptor type */
    1,           /* index of this interface */
    0,           /* alternate setting for this interface */
    2,           /* endpoints excl 0: number of endpoint descriptors to follow */
    0x0A,        /* Data Interface Class Codes */
    0,
    0,           /* Data Interface Class Protocol Codes */
    0,           /* string index for interface */

    /* Endpoint Descriptor */
    7,           /* sizeof(usbDescrEndpoint) */
    USBDESCR_ENDPOINT,  /* descriptor type = endpoint */
    0x01,        /* OUT endpoint number 1 */
    0x02,        /* attrib: Bulk endpoint */
    HW_CDC_BULK_OUT_SIZE, 0,        /* maximum packet size */
    0,           /* in ms */

    /* Endpoint Descriptor */
    7,           /* sizeof(usbDescrEndpoint) */
    USBDESCR_ENDPOINT,  /* descriptor type = endpoint */
    0x81,        /* IN endpoint number 1 */
    0x02,        /* attrib: Bulk endpoint */
    HW_CDC_BULK_IN_SIZE, 0,        /* maximum packet size */
    0,           /* in ms */
};


uchar usbFunctionDescriptor(usbRequest_t *rq)
{
  if(rq->wValue.bytes[1] == USBDESCR_DEVICE){
    usbMsgPtr = (uchar *)deviceDescrCDC;
    return sizeof(deviceDescrCDC);
  }else{  /* must be config descriptor */
    usbMsgPtr = (uchar *)configDescrCDC;
    return sizeof(configDescrCDC);
  }
}


static uchar    requestType;
static uchar    modeBuffer[7];
static uchar    sendEmptyFrame;
static uchar    intr3Status;    /* used to control interrupt endpoint transmissions */


static uchar buf[8];
static uchar endFlag;

/* ----------------------------- command parser ----------------------------- */

#define toHex(i) (((i) <= 9)?('0' +(i)):((i)+'@'-9))

uchar *ucharToHex2(uchar data, uchar *s) { /* 数字->１６進文字（２桁） */
  uchar d;
  d = data >> 4;
  *s++ = toHex(d);
  d = data & 0x0f;
  *s++ = toHex(d);
  *s = '\0';
  return s;
}

uchar *uintToHex4(uint16_t data, uchar *s) { /* 数字->１６進文字（４桁） */
  uchar d = data >> 8;
  s = ucharToHex2(d, s);
  d = data & 0xff;
  return ucharToHex2(d, s);
}

uchar hexToUchar(uchar s) {	/* １６進文字->数字 */
  if (s >= '0' && s <= '9') {
    return (s - '0');
  }
  if (s >= 'A' && s <= 'F')  {
    return (s - 'A' + 10);
  }
  if (s >= 'a' && s <= 'f') {
    return (s - 'a' + 10);
  }
  return 0xff;			/* エラーの場合0xffを返す */
}

uchar hexToUchar2(uchar *s) {
  return (hexToUchar(*s) << 4) + hexToUchar(*(s+1));
}

uint16_t hexToInt16(uchar *s) {
  return (hexToUchar2(s) << 8) + hexToUchar2(s+2);
}

#define reportError() gioRxBufAppendString("!*")
#define reportVersion() gioRxBufAppendString(FIRMWARE_VERSION)
#define reportEcho(s) gioRxBufAppendString(s)
#define reportEchoChar(c) gioRxBufAppend(c)

#define reportDIN(c) {\
    buf[0] = c;\
    uintToHex4(gioGetDIPortAll(), buf+1);\
    buf[5] = '*';\
    buf[6] = '\0';\
    gioRxBufAppendString(buf); }

#define reportAIN(v) {	\
    ucharToHex2(v, buf);		\
    gioRxBufAppendString(buf); }

/* parse "KONFIGURATION_X" */
uchar parseKonfiguration(uchar *s) {
  uchar c;
  if (strncmp_P(s, PSTR("KONFIGURATION_"),14) != 0) { /* メモリが無いので端折るか？ */
    return 0;
  }
  switch (c = *(s+14)) {
  case '1':
  case '2':
  case '3':
  case '6':
    return (c - '0');
  }
  return 0;
}

void parseInput(uchar *s) {
  uchar c, n;
  switch (*s) {
    /**** reboot */
  case 'Q':
    gioInit(0);			/* initialize */
    break;

    /**** configuration */
  case 'K':			
    if ((c = parseKonfiguration(s)) > 0) {
      gioPortInit(c);
    } else {
      reportError();
      return;
    }
    break;

    /**** degital out HIGH */
  case 'H':			
  case 'L':			
    if (((config == 3)|| (config == 6)) &&
	((c = hexToUchar(*(s+1))) <= 15)) {
      if (c < 4) {
	gioSetDOPort(c, (*s == 'H') ? 1 : 0);
	break;
      }
    }
    reportError();
    return;

#if 0
    /**** degital in all ( we don't have d-in port in pepper board ) */
  case 'R':
    reportDIN('R');
    return;

    /**** degital in all (continuous mode) */
  case 'r':
    pollMode = POLL_DIGITAL;
    endFlag = 0;
    break;

#endif

    /**** exit continunou mode */
  case 'E':
    endFlag = 1;
    return;

  case 'A':			/* analog out (ALL) Axx..xx*/
    gioSetAOPortAll(s+1);
    break;

  case 'a':			/* analog out (channel) anxx*/
    n = hexToUchar(*(s+1));
    c = hexToUchar2(s+2);
    if (n >= gioAOchannel()) {
      reportError();
      return;
    }
    gioSetAOPort(n, c);
    break;

  case 'I':			/* analog in (ALL) I -> ixx..xx */
    pollMode = POLL_ANALOG_ONCE;
    gioStartAd(0);
    return;

  case 'i':			/* analog in (continuous mode) i -> ixx..xx */
    pollMode = POLL_ANALOG_CONT;
    endFlag = 0;
    gioStartAd(0);
    return;

    /**** version number */
  case '?':
    reportVersion();
    return;

  default:
    reportError();
    return;
  }
  /* when the command is ok, we report the results */
  reportEcho(s);
}


/* ----------------------------- USB interface ----------------------------- */


uchar usbFunctionSetup(uchar data[8]) {
  usbRequest_t    *rq = (void *)data;

  if((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS){    /* class request type */
    if( rq->bRequest==GET_LINE_CODING || rq->bRequest==SET_LINE_CODING ){
      requestType = rq->bRequest;
      return 0xff;
      /*    GET_LINE_CODING -> usbFunctionRead()    */
      /*    SET_LINE_CODING -> usbFunctionWrite()    */
    }
    if(rq->bRequest == SET_CONTROL_LINE_STATE){
      /* Report serial state (carrier detect). On several Unix platforms,
       * tty devices can only be opened when carrier detect is set.
       */
      if( intr3Status==0 )
	intr3Status = 2;
    }
    /*  Prepare bulk-in endpoint to respond to early termination   */
    if((rq->bmRequestType & USBRQ_DIR_MASK) == USBRQ_DIR_HOST_TO_DEVICE)
      sendEmptyFrame  = 1;
  }
  return 0;
}

uchar usbFunctionRead( uchar *data, uchar len ) {

  if(requestType == GET_LINE_CODING){
    memcpy( data, modeBuffer, 7 );
    return 7;
  }
  return 0;   /* error -> terminate transfer */
}

uchar usbFunctionWrite( uchar *data, uchar len ) {

  if(requestType == SET_LINE_CODING){
    memcpy( modeBuffer, data, 7 );
    return 1;
  }
  return 1;   /* error -> accept everything until end */
}

void usbFunctionWriteOut( uchar *data, uchar len ) {
  uchar finish = 0;
  /*  usb -> rs232c:  transmit char    */
  for( ; len; len-- ) {
    if (*data == '*') {
      finish = 1;
    }
    gioTxBufAppend(*data++);
  }

  /* ここでTxBufferを解釈する*/
  if (finish == 1) {
    gioTxBufAppend('\0');
    parseInput(tx_buf);
    /* gioRxBufAppendString("\r\n"); */
    uwptr = 0;
  }
  /*  postpone receiving next data    */
  if( gioTxBytesFree()<=8 )
    usbDisableAllRequests();
}

static void hardwareInit(void) {
  uchar    i, j;

  /* activate pull-ups except on USB lines */
  USB_CFG_IOPORT   = (uchar)~((1<<USB_CFG_DMINUS_BIT)|(1<<USB_CFG_DPLUS_BIT));
  /* all pins input except USB (-> USB reset) */
#ifdef USB_CFG_PULLUP_IOPORT    /* use usbDeviceConnect()/usbDeviceDisconnect() if available */
  USBDDR    = 0;    /* we do RESET by deactivating pullup */
  usbDeviceDisconnect();
#else
  USBDDR    = (1<<USB_CFG_DMINUS_BIT)|(1<<USB_CFG_DPLUS_BIT);
#endif

  j = 0;
  while(--j){          /* USB Reset by device only required on Watchdog Reset */
    i = 0;
    while(--i);      /* delay >10ms for USB reset */
  }
#ifdef USB_CFG_PULLUP_IOPORT
  usbDeviceConnect();
#else
  USBDDR    = 0;      /*  remove USB reset condition */
#endif
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
static void calibrateOscillator(void) {
  uchar step = 128;
  uchar trialValue = 0, optimumValue;
  int   x, optimumDev, targetValue = (unsigned)(1499 * (double)F_CPU / 10.5e6 + 0.5);

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

void    usbEventResetReady(void) {
  calibrateOscillator();
  eeprom_write_byte(0, OSCCAL);   /* store the calibrated value in EEPROM */
}

uchar adVals[4];

/* Polling function for gio */
void gioPoll() {
  int16_t val;

#define AD_INTERVAL 3000		/* 適当にインターバルを置く */

  static int cnt = 0;
  /* pollingの必要がない場合 */
  if (pollMode == POLL_NONE) {
    return;
  }

#ifdef USE_DIO
  /* Digital in */
  if (pollMode == POLL_DIGITAL) {
    if (endFlag == 1) {
      pollMode = POLL_NONE;
      reportEcho("E*");
    } else if (cnt++ == 30) {
      reportDIN('r');
      cnt = 0;
    }
    return;
  }

#endif
  /* Analog in */
  if (pollMode == POLL_ANALOG_WAIT) {
    if ((--cnt) == 0) {
	convertingCh = 0;
	/* ADを再スタートする */
	setupCH(convertingCh);
	wait(2);
	ADCStart();
	pollMode = POLL_ANALOG_CONT;
    }
    return;
  }

  if ((val = gioPollAd()) >= 0) { /* 変換終了時のみ */
    /* 値を書き出す */
    if (convertingCh == 3) {
      val =- 200;		/* 200LSBを引くだけにする */
    } else {
      val = val >> 2;		/* 10bit -> 8bit */
    }
    adVals[(uchar)convertingCh] = val;	/* 値を記録する */

#if 0
    if (pollMode == POLL_ANALOG_1CH) {
      if (endFlag == 1) {
	pollMode = POLL_NONE;
	reportEcho("*E*");
      } else {
	reportEcho("*S");
	ADCStart();		/* ADを再スタートする */
      }
      return;
    }
#endif

    if (pollMode == POLL_ANALOG_ONCE) {
      convertingCh++;
      if (convertingCh >= gioADchannel()) { /* 全チャンネル終了 */
	pollMode = POLL_NONE;
	convertingCh = CONVERTING_CH_NONE;
	gioStopAd();
	reportADValues('I');
      } else {
	setupCH(convertingCh);
	wait(2);
	ADCStart();
      }
      return;
    }

    if (pollMode == POLL_ANALOG_CONT) {
      /* 次のチャンネルに設定し、開始する */
      convertingCh++;
      if (convertingCh >= gioADchannel()) { /* 全チャンネル終了 */
	reportADValues('i');
	if(endFlag == 1) {
	  pollMode = POLL_NONE;
	  reportEcho("E*");
	} else {
	  cnt = AD_INTERVAL;
	  pollMode = POLL_ANALOG_WAIT;
	}
      } else {
	setupCH(convertingCh);
	wait(2);
	ADCStart();
      }
      return;
    }
  }
}

void reportADValues(uchar cmd) {
  uchar i;
  reportEchoChar(cmd);
  for (i = 0 ; i < 4 ; i++) {
    reportAIN(adVals[i]);
  }
  reportEchoChar('*');
}

int main(void) {
  uchar   i;
  uchar   calibrationValue;

  calibrationValue = eeprom_read_byte(0); /* calibration value from last time */
  if(calibrationValue != 0xff){
    OSCCAL = calibrationValue;
  }
  odDebugInit();
  usbDeviceDisconnect();
  for(i=0;i<20;i++){  /* 300 ms disconnect */
    _delay_ms(15);
  }

  wdt_enable(WDTO_1S);
  hardwareInit();
  usbInit();
  gioInit(0);

  intr3Status = 0;
  sendEmptyFrame  = 0;

  sei();

  for(;;){    /* main event loop */
    wdt_reset();
    usbPoll();
    gioPoll(); /* (for continuous mode ) */

    /*    gio -> usb:  transmit char        */
    if( usbInterruptIsReady() ) {
      uchar bytesRead, *data;
      
      bytesRead = gioRxBytesAvailable(&data);
      if(bytesRead > 0 || sendEmptyFrame){
	if(bytesRead >= HW_CDC_BULK_IN_SIZE) {
	  bytesRead = HW_CDC_BULK_IN_SIZE;
	  /* send an empty block after last data block to indicate transfer end */
	  sendEmptyFrame  = 1;
	}
	else
	  sendEmptyFrame  = 0;
	usbSetInterrupt(data, bytesRead);
	gioRxDidReadBytes(bytesRead);
      }
    }

    /* We need to report rx and tx carrier after open attempt */
    if(intr3Status != 0 && usbInterruptIsReady3()){
      static uchar serialStateNotification[10] = {0xa1, 0x20, 0, 0, 0, 0, 2, 0, 3, 0};
      
      if(intr3Status == 2){
	usbSetInterrupt3(serialStateNotification, 8);
      }else{
	usbSetInterrupt3(serialStateNotification+8, 2);
      }
      intr3Status--;
    }
  }
  return 0;
}

/* EOF */
