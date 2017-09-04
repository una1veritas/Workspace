/* gio.c */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>   /* needed by usbdrv.h */
#include "oddebug.h"
#include "gio.h"
#include "usbdrv.h"

/* gio buffer */
uchar    rx_buf[RX_SIZE], tx_buf[TX_SIZE];
uchar    urptr, uwptr, iwptr;

uchar config = 0;	/* current configuration */
uchar continuous = 0;	/* continuous mode switch */
uchar pollMode = POLL_NONE; /* polling mode = off */

/* AD変換用 */
char convertingCh = CONVERTING_CH_NONE;	/* 変換中(変換中のチャネル番号） */

uchar smode = 0;

void gioInit(uchar conf) {
  urptr = uwptr = iwptr = 0;
  gioPortInit(conf);
  config = conf;
  continuous = 0;
}

void gioRxBufAppendString(uchar *s) {
  uchar c;
  while ( (c = *s++)!= '\0') {
    gioRxBufAppend(c);
  }
}

/* Portの初期化パラメータ */
typedef struct portConfig_t {
  uchar ddrbIn;			/* DDRB 入力ピンのビット */
  uchar ddrbOut;		/* DDRB 出力ピンのビット */
  uchar ddrbPup;		/* DDRB Pull-upのビット */
} portConfig;

static portConfig portConf[] = {
  {0x39, 0x00, 0x00},			/* Config 0 */
  {0x28, 0x11, 0x00},			/* Config 1 */
  {0x38, 0x01, 0x00},			/* Config 2 */
  {0x28, 0x11, 0x00},			/* Config 3 */
  {0x39, 0x00, 0x00},			/* Config 4 */
  {0x39, 0x00, 0x00},			/* Config 5 */
  {0x00, 0x39, 0x00}			/* Config 6 */
};


void gioPortInit(uchar conf) {
  config = conf;

  /* setup PORTB */
  DDRB &= ~portConf[conf].ddrbIn;
  DDRB |= portConf[conf].ddrbOut;
  PORTB |= portConf[conf].ddrbPup; /* pull-up all input ports */

  /* PWMの設定をする */
  if ((config == 0) || (config == 3) || (config == 6)) {
    TCCR0A = 0x00;		/* Turn off OC0A, OC0B */
  } else {			/* config == 1 or 2 */
    /* OC0Aを初期化 */
    TCCR0A = 0x83;		/* clear OC0A (non inverting mode), unuse OC0B */
    TCCR0B = 0x01;		/* set 0x03 for 1/64 clock , set 0x01 for 1/1 clock */
    OCR0A = 0;			/* for channel 0 */
  }
  if (config == 1) {
    /* OC1Bを使う */
    TCCR1  = 0x01;		/* OC1A disable set 0x07 for CK/64 clock, set 0x01 for 1/1 clock */
    GTCCR  = 0x60;		/* OC1B enable , set 0x00 for disable */
    OCR1B = 0;
  } else {
    TCCR1  = 0x00; 
    GTCCR  = 0x00;
  }
}

#ifdef PEPPER85
void gioControlPWM(uchar portnum, uchar onOff) {
  if (portnum == 0) {
    if ((onOff == ON) && ((TCCR0A & 0x80) == 0x00)) {
      TCCR0A = 0x83;		/* PWM ON */
    } else if ((onOff == OFF) && ((TCCR0A & 0x80) == 0x80)) {
      PORTB &= ~0b00000001;	/* PB0 LOW */
      TCCR0A = 0x03;		/* PWM OFF */
    }
  } else if (portnum == 1) {
    if ((onOff == ON) && ((GTCCR & 0b00110000) == 0b00000000)) {
      GTCCR = 0b01100000;		/* PWM ON */
    } else if ((onOff == OFF) && ((GTCCR & 0b00110000) == 0b00100000)) {
      PORTB &= ~0b00010000;	/* PB4 LOW */
      GTCCR = 0b00000000;		/* PWM OFF */
    }
  }
}
#endif

/* digital output for config 3 and 6 */
void gioSetDOPort(uchar portnum, uchar val) {
  static uchar map[4] = {3, 4, 0, 5};
  if (config == 3) {
    setPortBit(&PORTB, ((portnum == 0) ? 4: 0), val);
  } else {
    setPortBit(&PORTB, map[portnum], val);
  }
}

/* Analog Output with PWM */
void gioSetAOPort(uchar portnum, uchar val) {
  uchar *aoPort = &OCR0A;
  if ((config == 1) && (portnum == 1)) {
    aoPort = &OCR1B;
  }
#ifdef PEPPER85
  gioControlPWM(portnum, (val == 0) ? OFF : ON);
#endif
  *(aoPort) = val;
}

void gioSetAOPortAll(uchar *s) {
  char i;
  uchar maxPort = gioAOchannel();
  uchar val;
  for (i = 0 ; i < maxPort ; i++) {
    val = hexToUchar2(s);
    gioSetAOPort(i, val);
    s += 2;
  }
}

/* setup ad channel */
void setupCH(uchar ch) {
  if (ch == 0) {
    ADMUX = 3;
  } else if (ch == 1) {
    if (config == 2) {
      ADMUX = 2;
    } else {
      ADMUX = 0;
    }
  } else if (ch == 2) {
    if (config == 2) {
      ADMUX = 0;
    } else {
      ADMUX = 0xd;			/* GRAND */
    }
  } else if (ch == 3) {
    ADMUX = 0x8f;	/* TEMP, 1.1V reference */
  }
}


uchar gioADchannel() {
  return 4;			/* 互換性を保つために４チャンネル分返す */
}

uchar gioAOchannel() {
  return ((config == 1) ? 2 :
	  ((config == 2) ? 1 : 0));
}

void wait( unsigned char time) {
  unsigned char j,k;
  for (j = 0; j < time; j++)
    for (k = 0; k < 250; k++)
      ;
}
/* EOF */
