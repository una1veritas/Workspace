/* Name: gio.h
 */

#ifndef __gio_h_included__
#define __gio_h_included__

/* #define USE_DIO */

#ifndef uchar
#define uchar   unsigned char
#endif

#ifndef ulong
#define ulong   unsigned long
#endif

#define	RX_SIZE		32      /* gio -> usb buffer size (must be 2^n ) <=256 , MUST HAVE >32 */
#define	TX_SIZE		32      /* usb -> gio buffer size,  MUST HAVE >32 */

#define	RX_MASK		(RX_SIZE-1)

#define setPortBit(port, bit, val) { \
    if (val == 0) { *(port) &= ~_BV(bit); } else	\
      {*(port) |= _BV(bit); }				\
}

#define getPortBit(port, bit) ((*(port) & _BV(bit)) >> (bit))

/* AD enable, 1/128 clock  */
#define setupADC() ADCSRA = 0x87
/* Aref, 右揃え,CH=ch */
/* 変換開始 */
#define ADCStart() { ADCSRA |= 0x10;  ADCSRA |= 0x40; }
/* 変換停止 */
#define gioStopAd() ADCSRA = 0x00
/* AD変換値を取得する */
#define gioPollAd() (((ADCSRA & 0x10) == 0) ? -1 : ADC)

#define gioStartAd(c) { \
  setupADC();\
  convertingCh = c; \
  setupCH(c);	\
  wait(2);\
  ADCStart(); }

/* USB Communication stuff */
extern uchar    urptr, uwptr, iwptr;
extern uchar    rx_buf[RX_SIZE], tx_buf[TX_SIZE]; 

/* variables */
extern uchar verbose;
extern uchar config;
extern uchar smode;
extern char convertingCh;	/* 変換中(変換中のチャネル番号） */

extern uchar pollMode;
#define POLL_NONE          0
#define POLL_DIGITAL       1
#define POLL_ANALOG_ONCE   2
#define POLL_ANALOG_CONT   3
#define POLL_ANALOG_1CH    4
#define POLL_ANALOG_WAIT   5

#define CONVERTING_CH_NONE (-1)

#define ON 1
#define OFF 0

extern void gioInit(uchar c);
extern void gioRxBufAppendString(uchar *);
extern void gioRxBufAppendString_P(PGM_P *);
extern void gioPortInit(uchar config);
extern void gioSetDOPortAll(uint16_t v);
extern void gioSetDOPort(uchar p, uchar v);
extern uint16_t gioGetDIPortAll(void);
extern uchar gioGetDIPort(uchar p);
extern void gioSetAOPort(uchar p, uchar v);
extern void gioSetAOPortAll(uchar *s);
extern uchar gioGetAIPort(uchar p);
extern void setupCH(uchar ch);
extern uchar gioADchannel();
extern uchar gioAOchannel();
extern uchar gioDIchannel();
extern void reportADValues(uchar cmd);
extern void wait(unsigned char t);

extern uchar hexToUchar2(uchar *s);


/* TxBuffer is used for usb -> gio */
/* TxBufferは単なる一次元の配列 */
static inline void  gioTxBufAppend(uchar c) {
  if (uwptr < TX_SIZE) {
    tx_buf[uwptr++] = c;
  }
}

/* The following function returns the amount of bytes available in the TX
 * buffer before we have an overflow.
 */
static inline uchar gioTxBytesFree(void) {
  return (TX_SIZE - uwptr - 1);
}


/* Rxbuffer is used for gio -> usb */
/* RxBufferはリングバッファ */
static inline void  gioRxBufAppend(uchar c) {
  uchar   iwnxt;
  iwnxt = (iwptr+1) & RX_MASK;
  if( iwnxt==urptr )
    return;         /* buffer overflow */
  
  rx_buf[iwptr] = c;
  iwptr = iwnxt;
}

/* The following function sets *ptr to the current read position and returns
 * the number of bytes which can currently be read from this read position.
 */
static inline uchar gioRxBytesAvailable(uchar **ptr) {
  *ptr = &rx_buf[urptr];
  if(iwptr >= urptr){
    return iwptr - urptr;
  }else{  /* buffer end is between read and write pointer, return continuous range */
    return RX_SIZE - urptr;
  }
}

/* The following function must be called after uartRxBytesAvailable() to
 * remove the bytes from the receiver buffer.
 */
static inline void  gioRxDidReadBytes(uchar numBytes) {
  urptr = (urptr + numBytes) & RX_MASK;
}

static inline void gioRxClear() {
  urptr = uwptr = 0;
}

#endif  /*  __gio_h_included__  */

