/*------------------------------------------------------------------------*/
/* RX62N/RX63N: MMCv3/SDv1/SDv2 (SPIモード) 制御モジュール                   */
/*------------------------------------------------------------------------*/
/*
/  Copyright (C) 2011, ChaN, all right reserved.
/  Copyright (C) 2012, Tokushu Denshi Kairo Inc. all rights reserved.
/
/ * This software is a free software and there is NO WARRANTY.
/ * No restriction on use. You can use, modify and redistribute it for
/   personal, non-profit or commercial products UNDER YOUR RESPONSIBILITY.
/ * Redistributions of source code must retain the above copyright notice.
/
/-------------------------------------------------------------------------*/

#include <stdio.h>
#include "../../include/tkdnhal/tkdn_hal.h"
#ifdef __GNUC__
  #ifdef CPU_IS_RX62N
    #include "../../include/tkdnhal/iodefine_gcc62n.h"
  #endif
  #ifdef CPU_IS_RX63N
    #include "../../include/tkdnhal/iodefine_gcc63n.h"
  #endif
#endif
#ifdef __RENESAS__
  #include "iodefine.h"
#endif

#if (TARGET_BOARD == BOARD_RAXINO) || (TARGET_BOARD == BOARD_NP1055)
  #define RSPI_CH	0	/* 使用するSPIポートの選択: 0:RSPIA-A, 1:RSPIB-A, 10:RSPIA-B, 11:RSPIB-B */	
#elif (TARGET_BOARD == BOARD_RXMEGA)
  #define RSPI_CH	0	/* 使用するSPIポートの選択: 0:RSPIA-A, 1:RSPIB-A, 10:RSPIA-B, 11:RSPIB-B */	
#elif (TARGET_BOARD == BOARD_ULT62N0_MMC) || (TARGET_BOARD == BOARD_ULT62N0_SDRAM) || (TARGET_BOARD == BOARD_ULT62N)
  #define RSPI_CH	1	/* 使用するSPIポートの選択: 0:RSPIA-A, 1:RSPIB-A, 10:RSPIA-B, 11:RSPIB-B */	
#else
  #error Target board is not specified.
#endif

//#define		INS			1//(!PORT0.PORT.BIT.B7)	/* SD_CD(P07)カード検出(真:あり, 偽:なし) */
int INS = 1; // ディスクの挿入状態
#define		WP			0						/* 書き込み禁止(真:禁止, 偽:許可) */

#define PCLK		48000000UL	/* PCLKの周波数[Hz] */
#define CLK_FAST	24000000UL	/* SCLKの周波数[Hz] (動作中) */
#define	CLK_SLOW	400000UL	/* SCLKの周波数[Hz] (初期化中) */

/*------------------------------------------------------------------------*/

#if RSPI_CH == 0	/* RSPIA-A */
#define RSPI	RSPI0
static inline void SPI_ATTACH()
{
#ifdef CPU_IS_RX62N
	PORTC.DDR.BYTE |=  0x80; \
	PORTC.DR.BYTE  |=  0x80; \
	PORTC.DDR.BYTE &= ~0x80; \
	PORTC.ICR.BYTE |= 0x80;		\
	IOPORT.PFGSPI.BYTE = 0x0E;	\
	SYSTEM.MSTPCRB.LONG &= ~(1 << 17); \

//	PORTC.PMR.BIT.B1 = 1; // 周辺機能を選択する
//	MPC.PC1PFS.BIT.PSEL = 0x0d; // 機能選択
//	RSPI0.SPCMD0.BIT.SSLA = 1;
#endif /* CPU_IS_RX62N */
#ifdef CPU_IS_RX63N
	PORTC.PMR.BIT.B7 = 0; // GPIOにする
	PORTC.PODR.BIT.B7 = 1; // 1を出力
	PORTC.PDR.BIT.B7 = 1; // 出力モード
	PORTC.PDR.BIT.B7 = 0; // 入力モード

	SYSTEM.MSTPCRB.LONG &= ~(1 << 17);

	// PC5をRSPCKにする
	PORTC.PMR.BIT.B5 = 1; // 周辺機能を選択する
	MPC.PC5PFS.BIT.PSEL = 0x0d; // 機能選択(RSPCKA)

	// PC6をMOSIAにする
	PORTC.PMR.BIT.B6 = 1; // 周辺機能を選択する
	MPC.PC6PFS.BIT.PSEL = 0x0d; // 機能選択

	// PC7をMISOAにする
	PORTC.PMR.BIT.B7 = 1; // 周辺機能を選択する
	MPC.PC7PFS.BIT.PSEL = 0x0d; // 機能選択

	// PC0をGPIOにする
	PORTC.PMR.BIT.B0 = 0; // GPIOにする
	PORTC.PODR.BIT.B0 = 1; // 1を出力
	PORTC.PDR.BIT.B0 = 1; // 出力モード

//	PORTC.PMR.BIT.B1 = 1; // 周辺機能を選択する
//	MPC.PC1PFS.BIT.PSEL = 0x0d; // 機能選択
//	RSPI0.SPCMD0.BIT.SSLA = 1;
#endif /* CPU_IS_RX63N */
}

#if (TARGET_BOARD == BOARD_NP1055)
static inline void CS_LOW() 	/* CS#(PC1/SSLA0-A)をLにする(RSPI_CH=0のとき) */
{
	PORTC.PODR.BIT.B0 = 0;
}
static inline void CS_HIGH()	/* CS#(PC1/SSLA0-A)をHにする(RSPI_CH=0のとき) */
{
	PORTC.PODR.BIT.B0 = 1;
}
static inline void CS_INIT()	/* CS#,INS#,WP 各端子の初期化(RSPI_CH=1のとき) */\
{
	// PC1をGPIOにする
	PORTC.PMR.BIT.B0 = 0; // GPIOにする
	PORTC.PODR.BIT.B0 = 1; // 1を出力
	PORTC.PDR.BIT.B0 = 1; // 出力モード
}
#elif (TARGET_BOARD == BOARD_RAXINO)
static inline void CS_LOW() 	/* CS#(PC1/SSLA0-A)をLにする(RSPI_CH=0のとき) */
{
	PORTC.DR.BYTE &= ~0x01;
}
static inline void CS_HIGH()	/* CS#(PC1/SSLA0-A)をHにする(RSPI_CH=0のとき) */
{
	PORTC.DR.BYTE |= 0x01;
}
static inline void CS_INIT()	/* CS#,INS#,WP 各端子の初期化(RSPI_CH=1のとき) */\
{
	// PC1をGPIOにする
	PORTC.DR.BYTE |= 0x01;	/* CS#=OUT */
	PORTC.DDR.BYTE |= 0x01;	/* CS#=H */
}
#elif (TARGET_BOARD == BOARD_RXMEGA)
static inline void CS_LOW() 	/* CS#(PC4/SSLA0-A)をLにする(RSPI_CH=0のとき) */
{
	PORTC.DR.BYTE &= ~0x08;
}
static inline void CS_HIGH()	/* CS#(PC4/SSLA0-A)をHにする(RSPI_CH=0のとき) */
{
	PORTC.DR.BYTE |= 0x08;
}
static inline void CS_INIT()	/* CS#,INS#,WP 各端子の初期化(RSPI_CH=1のとき) */\
{
	// PC4をGPIOにする
	PORTC.DR.BYTE |= 0x08;	/* CS#=OUT */
	PORTC.DDR.BYTE |= 0x08;	/* CS#=H */
}
#endif

#elif RSPI_CH == 10	/* RSPIA-B */
#define RSPI	RSPI0
#define	SPI_ATTACH() {			\
	PORTA.ICR.BIT.B7 = 1;		\
	IOPORT.PFGSPI.BYTE = 0x0F;	\
	MSTP_RSPI0 = 0;				\
}

#elif RSPI_CH == 1	/* RSPIB-A */
#define RSPI	RSPI1
#define	SPI_ATTACH() {			 \
	PORT3.DDR.BYTE |=  0x01; \
	PORT3.DR.BYTE  |=  0x01; \
	PORT3.DDR.BYTE &= ~0x01; \
	PORT3.ICR.BYTE |=  0x01; \
	IOPORT.PFHSPI.BYTE = 0x0e;   \
	SYSTEM.MSTPCRB.LONG &= ~(1 << 16); \
}

#if (TARGET_BOARD == BOARD_ULT62N0_MMC) || (TARGET_BOARD == BOARD_ULT62N0_SDRAM) // 初期バージョン
#define	CS_LOW()	{PORT5.DR.BYTE &= ~0x01;}	/* CS#(PC5/SSLA0-A)をLにする(RSPI_CH=1のとき) */
#define	CS_HIGH()	{PORT5.DR.BYTE |=  0x01;}	/* CS#(PC5/SSLA0-A)をHにする(RSPI_CH=1のとき) */
#define CS_INIT()	{		/* CS#,INS#,WP 各端子の初期化(RSPI_CH=1のとき) */\
	PORT3.DDR.BYTE &= ~0x01; \
	PORT5.DR.BYTE  |= 0x01;	/* CS#=OUT */	\
	PORT5.DDR.BYTE |= 0x01;	/* CS#=H */		\
}
#endif

#if (TARGET_BOARD == BOARD_ULT62N) // バージョンA
#define	CS_LOW()	{PORT6.DR.BYTE &= ~0x01;}	/* CS#(PC6/SSLA0-A)をLにする(RSPI_CH=1のとき) */
#define	CS_HIGH()	{PORT6.DR.BYTE |=  0x01;}	/* CS#(PC6/SSLA0-A)をHにする(RSPI_CH=1のとき) */
#define CS_INIT()	{		/* CS#,INS#,WP 各端子の初期化(RSPI_CH=1のとき) */\
	PORT3.DDR.BYTE &= ~0x01; \
	PORT6.DR.BYTE  |= 0x01;	/* CS#=OUT */	\
	PORT6.DDR.BYTE |= 0x01;	/* CS#=H */		\
}
#endif

//	PORT3.DR.BYTE  |=  (1 << 1); 
//	PORT3.DDR.BYTE |=  (1 << 1); 
/* P30_MISO受信バッファ有効 
RSPI1は-A系を使う。SSLB0とSSLB1は無効
※ SPIのメモリは非常に長い間CSを下げっぱなしにするので、
RSPIコントローラの自動操作では対応できない*/

#elif RSPI_CH == 11	/* RSPIB-B */
#define RSPI	RSPI1
#define	SPI_ATTACH() {			\
	PORTE.ICR.BIT.B7 = 1;		\
	IOPORT.PFHSPI.BYTE = 0x0F;	\
	MSTP_RSPI1 = 0;				\
}

#endif

#define FCLK_FAST() {					\
	RSPI.SPCR.BYTE &= ~(1 << 6);				\
	RSPI.SPBR.BYTE = PCLK/2/CLK_FAST-1;	\
	RSPI.SPCR.BYTE |=  (1 << 6);				\
}

//#ifdef __LIT
//#define LDDW(x) revl(x)
//#else
//#define LDDW(x) x
//#endif

unsigned long LDDW(unsigned long x)
{
	return  ((x        & 0xff) << 24) |
	       (((x >> 8)  & 0xff) << 16) |
	       (((x >> 16) & 0xff) << 8 ) |
	       (((x >> 24) & 0xff) << 0 );
}

/*--------------------------------------------------------------------------

   Module Private Functions

---------------------------------------------------------------------------*/

//#include <machine.h>
#include "diskio.h"


/* MMC/SD command */
#define CMD0	(0)			/* GO_IDLE_STATE */
#define CMD1	(1)			/* SEND_OP_COND (MMC) */
#define	ACMD41	(0x80+41)	/* SEND_OP_COND (SDC) */
#define CMD8	(8)			/* SEND_IF_COND */
#define CMD9	(9)			/* SEND_CSD */
#define CMD10	(10)		/* SEND_CID */
#define CMD12	(12)		/* STOP_TRANSMISSION */
#define ACMD13	(0x80+13)	/* SD_STATUS (SDC) */
#define CMD16	(16)		/* SET_BLOCKLEN */
#define CMD17	(17)		/* READ_SINGLE_BLOCK */
#define CMD18	(18)		/* READ_MULTIPLE_BLOCK */
#define CMD23	(23)		/* SET_BLOCK_COUNT (MMC) */
#define	ACMD23	(0x80+23)	/* SET_WR_BLK_ERASE_COUNT (SDC) */
#define CMD24	(24)		/* WRITE_BLOCK */
#define CMD25	(25)		/* WRITE_MULTIPLE_BLOCK */
#define CMD32	(32)		/* ERASE_ER_BLK_START */
#define CMD33	(33)		/* ERASE_ER_BLK_END */
#define CMD38	(38)		/* ERASE */
#define CMD55	(55)		/* APP_CMD */
#define CMD58	(58)		/* READ_OCR */



static volatile
DSTATUS Stat = STA_NOINIT;	/* Physical drive status */

static volatile
WORD Timer1, Timer2;	/* 100Hz decrement timer stopped at zero (disk_timerproc()) */

static
BYTE CardType;			/* Card type flags */



/*-----------------------------------------------------------------------*/
/* Send a byte to MMC  (Platform dependent)                              */
/*-----------------------------------------------------------------------*/

#ifdef CPU_IS_RX63N
static void wait_xmit()
{
/*
	while(ICU.IR[39].BIT.IR == 0) {
		gpio_write_port(PIN_LED0,0);
		gpio_write_port(PIN_LED0,1);
	} //SPRI0が0ならば待ち RX63N
	ICU.IR[39].BIT.IR = 0;
*/
	timer_wait_us(50);
}
#endif

/* Send a byte */
static
void xmit_spi (
	BYTE dat	/* Data to send */
)
{
#ifdef CPU_IS_RX62N
	RSPI.SPDR.LONG = dat;			/* Start transmission */
	while ((RSPI.SPSR.BYTE & 0x80) == 0) ;	/* Wait for end of transfer */
	RSPI.SPDR.LONG;					/* Discard received data */
#endif
#ifdef CPU_IS_RX63N
	volatile unsigned long x = dat;
	RSPI.SPCR.BIT.SPRIE = 1;
	RSPI.SPDR = x;			/* Start transmission */
//	while ((RSPI.SPSR.BYTE & 0x80) == 0) ;	/* Wait for end of transfer RX62N */
	wait_xmit();
	x = RSPI.SPDR;					/* Discard received data */
#endif
}

/* Send multiple byte */
static
void xmit_spi_multi (
	const BYTE *buff,	/* Pointer to the data */
	UINT btx			/* Number of bytes to send (multiple of 4) */
)
{
	const DWORD *lp = (const DWORD*)buff;

	RSPI.SPCMD0.WORD = (RSPI.SPCMD0.WORD & 0xf0ff) | (3 << 8);	/* 32-bit mode */

	do {
#ifdef CPU_IS_RX62N
		RSPI.SPDR.LONG = LDDW(*lp++);	/* Send four bytes */
		while ((RSPI.SPSR.BYTE & 0x80) == 0) ;	/* Wait for end of transfer */
		volatile int x = RSPI.SPDR.LONG;					/* Discard received bytes */
#endif
#ifdef CPU_IS_RX63N
		RSPI.SPCR.BIT.SPRIE = 1;
		RSPI.SPDR = (unsigned long)LDDW(*lp++);	/* Send four bytes */
		wait_xmit();
		volatile unsigned long x = RSPI.SPDR;					/* Discard received bytes */
#endif
	} while (btx -= 4);					/* Repeat until all data sent */

	RSPI.SPCMD0.WORD = (RSPI.SPCMD0.WORD & 0xf0ff) | (7 << 8);	/* 8-bit mode */
}



/*-----------------------------------------------------------------------*/
/* Receive data from MMC  (Platform dependent)                           */
/*-----------------------------------------------------------------------*/

/* Receive a byte */
static
BYTE rcvr_spi (void)
{
#ifdef CPU_IS_RX62N
	RSPI.SPDR.LONG = 0xFF;			/* Send a 0xFF */
	while ((RSPI.SPSR.BYTE & 0x80) == 0) ;	/* Wait for end of transfer */
	return RSPI.SPDR.LONG;			/* Returen received byte */
#endif
#ifdef CPU_IS_RX63N
	unsigned long ul = 0xff;
	RSPI.SPCR.BIT.SPRIE = 1;
	RSPI.SPDR = ul;			/* Send a 0xFF */
	wait_xmit();
	ul = RSPI.SPDR;			/* Returen received byte */
	return ul;
#endif
}

/* Receive multiple byte */
static
void rcvr_spi_multi (
	BYTE *buff,		/* Pointer to data buffer */
	UINT btr		/* Number of bytes to receive (multiple of 4) */
)
{
	DWORD *lp = (DWORD*)buff;


	RSPI.SPCMD0.WORD = (RSPI.SPCMD0.WORD & 0xf0ff) | (3 << 8);	/* 32-bit mode */

	do {
#ifdef CPU_IS_RX62N
		RSPI.SPDR.LONG = 0xFFFFFFFF;	/* Send four 0xFFs */
		while ((RSPI.SPSR.BYTE & 0x80) == 0) ;	/* Wait for end of transfer */
		*lp++ = LDDW(RSPI.SPDR.LONG);	/* Store received bytes */
#endif
#ifdef CPU_IS_RX63N
		RSPI.SPCR.BIT.SPRIE = 1;
		RSPI.SPDR = 0xFFFFFFFF;	/* Send four 0xFFs */
		wait_xmit();
		*lp++ = LDDW(RSPI.SPDR);	/* Store received bytes */
#endif
	} while (btr -= 4);					/* Repeat until all data received */

	RSPI.SPCMD0.WORD = (RSPI.SPCMD0.WORD & 0xf0ff) | (7 << 8);	/* 8-bit mode */
}




/*-----------------------------------------------------------------------*/
/* Wait for card ready                                                   */
/*-----------------------------------------------------------------------*/

static
int wait_ready (	/* 1:Ready, 0:Timeout */
	UINT wt			/* Timeout [ms] */
)
{
	Timer2 = (WORD)wt;

	rcvr_spi();		/* Read a byte (Force enable DO output) */
	do {
		if (rcvr_spi() == 0xFF) return 1;	/* Card goes ready */
		/* This loop takes a time. Insert rot_rdq() here for multitask envilonment. */
	} while (Timer2);	/* Wait until card goes ready or timeout */

	return 0;	/* Timeout occured */
}



/*-----------------------------------------------------------------------*/
/* Deselect card and release SPI                                         */
/*-----------------------------------------------------------------------*/

static
void deselect (void)
{
	CS_HIGH();		/* CS = H */
	rcvr_spi();		/* Dummy clock */
}



/*-----------------------------------------------------------------------*/
/* Select card and wait for ready                                        */
/*-----------------------------------------------------------------------*/

static
int select (void)	/* 1:OK, 0:Timeout */
{
	CS_LOW();		/* CS = H */
	if (!wait_ready(500)) {
		deselect();
		return 0;	/* Failed to select the card due to timeout */
	}
	return 1;	/* OK */
}



/*-----------------------------------------------------------------------*/
/* Control SPI module (Platform dependent)                               */
/*-----------------------------------------------------------------------*/

static
void power_on (void)	/* Enable SPI */
{
	CS_INIT();

	/* Attach RSPI module to I/O pads */
	SPI_ATTACH();

	/* Initialize RSPI module */
	RSPI.SPCR.BYTE = 0;		/* Stop SPI */
	RSPI.SPPCR.BYTE = 0;	/* Fixed idle value, disable loop-back mode */
	RSPI.SPSCR.BYTE = 0;	/* Disable sequence control */
	RSPI.SPDCR.BYTE = 0x20;	/* SPLW=1 */
	RSPI.SPCMD0.WORD = 0x0700;	/* LSBF=0, SPB=7, BRDV=0, CPOL=0, CPHA=0 */
	RSPI.SPBR.BYTE = PCLK / 2 / CLK_SLOW - 1;	/* Bit rate */
#ifdef CPU_IS_RX62N
	RSPI.SPCR.BYTE = 0x49;	/* Start SPI in master mode */
#endif
#ifdef CPU_IS_RX63N
	RSPI.SPCR.BYTE = 0xc9;	/* Start SPI in master mode */
#endif
}


static
void power_off (void)	/* Disable SPI function */
{
	select();				/* Wait for card ready */
	deselect();

	RSPI.SPCR.BYTE = 0;		/* Stop SPI */
}



/*-----------------------------------------------------------------------*/
/* Receive a data packet from the MMC                                    */
/*-----------------------------------------------------------------------*/

static
int rcvr_datablock (	/* 1:OK, 0:Error */
	BYTE *buff,			/* Data buffer */
	UINT btr			/* Data block length (byte) */
)
{
	BYTE token;


	Timer1 = 200;
	do {							/* Wait for DataStart token in timeout of 200ms */
		token = rcvr_spi();
		/* This loop will take a time. Insert rot_rdq() here for multitask envilonment. */
	} while ((token == 0xFF) && Timer1);
	if(token != 0xFE) return 0;		/* Function fails if invalid DataStart token or timeout */

	rcvr_spi_multi(buff, btr);		/* Store trailing data to the buffer */
	rcvr_spi(); rcvr_spi();			/* Discard CRC */

	return 1;						/* Function succeeded */
}



/*-----------------------------------------------------------------------*/
/* Send a data packet to the MMC                                         */
/*-----------------------------------------------------------------------*/

#if _READONLY == 0
static
int xmit_datablock (	/* 1:OK, 0:Failed */
	const BYTE *buff,	/* Ponter to 512 byte data to be sent */
	BYTE token			/* Token */
)
{
	BYTE resp;


	if (!wait_ready(500)) return 0;		/* Wait for card ready */

	xmit_spi(token);					/* Send token */
	if (token != 0xFD) {				/* Send data if token is other than StopTran */
		xmit_spi_multi(buff, 512);		/* Data */
		xmit_spi(0xFF); xmit_spi(0xFF);	/* Dummy CRC */

		resp = rcvr_spi();				/* Receive data resp */
		if ((resp & 0x1F) != 0x05)		/* Function fails if the data packet was not accepted */
			return 0;
	}
	return 1;
}
#endif /* _READONLY */



/*-----------------------------------------------------------------------*/
/* Send a command packet to the MMC                                      */
/*-----------------------------------------------------------------------*/

static
BYTE send_cmd (		/* Return value: R1 resp (bit7==1:Failed to send) */
	BYTE cmd,		/* Command index */
	DWORD arg		/* Argument */
)
{
	BYTE n, res;


	if (cmd & 0x80) {	/* Send a CMD55 prior to ACMD<n> */
		cmd &= 0x7F;
		res = send_cmd(CMD55, 0);
		if (res > 1) return res;
	}

	/* Select card */
	deselect();
	if (!select()) return 0xFF;

	/* Send command packet */
	xmit_spi(0x40 | cmd);				/* Start + command index */
	xmit_spi((BYTE)(arg >> 24));		/* Argument[31..24] */
	xmit_spi((BYTE)(arg >> 16));		/* Argument[23..16] */
	xmit_spi((BYTE)(arg >> 8));			/* Argument[15..8] */
	xmit_spi((BYTE)arg);				/* Argument[7..0] */
	n = 0x01;							/* Dummy CRC + Stop */
	if (cmd == CMD0) n = 0x95;			/* Valid CRC for CMD0(0) */
	if (cmd == CMD8) n = 0x87;			/* Valid CRC for CMD8(0x1AA) */
	xmit_spi(n);

	/* Receive command resp */
	if (cmd == CMD12) rcvr_spi();		/* Diacard following one byte when CMD12 */
	n = 10;								/* Wait for response (10 bytes max) */
	do
		res = rcvr_spi();
	while ((res & 0x80) && --n);

	return res;							/* Return received response */
}



/*--------------------------------------------------------------------------

   Public Functions

---------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------*/
/* Initialize disk drive                                                 */
/*-----------------------------------------------------------------------*/

DSTATUS disk_initialize (
	BYTE drv		/* Physical drive number (0) */
)
{
	BYTE n, cmd, ty, ocr[4];


	if (drv) return STA_NOINIT;			/* Supports only drive 0 */
	if (Stat & STA_NODISK) return Stat;	/* Is card existing in the soket? */

	power_on();							/* Initialize SPI */
	for (n = 10; n; n--) rcvr_spi();	/* Send 80 dummy clocks */

	ty = 0;
	if (send_cmd(CMD0, 0) == 1) {			/* Put the card SPI/Idle state */
		Timer1 = 1000;						/* Initialization timeout = 1 sec */
		if (send_cmd(CMD8, 0x1AA) == 1) {	/* SDv2? */
			for (n = 0; n < 4; n++) ocr[n] = rcvr_spi();		/* Get 32 bit return value of R7 resp */
			if (ocr[2] == 0x01 && ocr[3] == 0xAA) {				/* Is the card supports vcc of 2.7-3.6V? */
				while (Timer1 && send_cmd(ACMD41, 1UL << 30)) ;	/* Wait for end of initialization with ACMD41(HCS) */
				if (Timer1 && send_cmd(CMD58, 0) == 0) {		/* Check CCS bit in the OCR */
					for (n = 0; n < 4; n++) ocr[n] = rcvr_spi();
					ty = (ocr[0] & 0x40) ? CT_SD2 | CT_BLOCK : CT_SD2;	/* Card id SDv2 */
				}
			}
		} else {	/* Not SDv2 card */
			if (send_cmd(ACMD41, 0) <= 1) 	{	/* SDv1 or MMC? */
				ty = CT_SD1; cmd = ACMD41;	/* SDv1 (ACMD41(0)) */
			} else {
				ty = CT_MMC; cmd = CMD1;	/* MMCv3 (CMD1(0)) */
			}
			while (Timer1 && send_cmd(cmd, 0)) ;		/* Wait for end of initialization */
			if (!Timer1 || send_cmd(CMD16, 512) != 0)	/* Set block length: 512 */
				ty = 0;
		}
	}
	CardType = ty;	/* Card type */
	deselect();

	if (ty) {			/* OK */
		FCLK_FAST();			/* Set fast clock */
		Stat &= ~STA_NOINIT;	/* Clear STA_NOINIT flag */
	} else {			/* Failed */
		power_off();
		Stat = STA_NOINIT;
	}

	return Stat;
}



/*-----------------------------------------------------------------------*/
/* Get disk status                                                       */
/*-----------------------------------------------------------------------*/

DSTATUS disk_status (
	BYTE drv		/* Physical drive number (0) */
)
{
	if (drv) return STA_NOINIT;		/* Supports only drive 0 */

	return Stat;	/* Return disk status */
}



/*-----------------------------------------------------------------------*/
/* Read sector(s)                                                        */
/*-----------------------------------------------------------------------*/

DRESULT disk_read (
	BYTE drv,		/* Physical drive number (0) */
	BYTE *buff,		/* Pointer to the data buffer to store read data */
	DWORD sector,	/* Start sector number (LBA) */
	BYTE count		/* Number of sectors to read (1..128) */
)
{
	if (drv || !count) return RES_PARERR;		/* Check parameter */
	if (Stat & STA_NOINIT) return RES_NOTRDY;	/* Check if drive is ready */

	if (!(CardType & CT_BLOCK)) sector *= 512;	/* LBA ot BA conversion (byte addressing cards) */

	if (count == 1) {	/* Single sector read */
		if ((send_cmd(CMD17, sector) == 0)	/* READ_SINGLE_BLOCK */
			&& rcvr_datablock(buff, 512))
			count = 0;
	}
	else {				/* Multiple sector read */
		if (send_cmd(CMD18, sector) == 0) {	/* READ_MULTIPLE_BLOCK */
			do {
				if (!rcvr_datablock(buff, 512)) break;
				buff += 512;
			} while (--count);
			send_cmd(CMD12, 0);				/* STOP_TRANSMISSION */
		}
	}
	deselect();

	return count ? RES_ERROR : RES_OK;	/* Return result */
}



/*-----------------------------------------------------------------------*/
/* Write sector(s)                                                       */
/*-----------------------------------------------------------------------*/

#if _READONLY == 0
DRESULT disk_write (
	BYTE drv,			/* Physical drive number (0) */
	const BYTE *buff,	/* Ponter to the data to write */
	DWORD sector,		/* Start sector number (LBA) */
	BYTE count			/* Number of sectors to write (1..128) */
)
{
	if (drv || !count) return RES_PARERR;		/* Check parameter */
	if (Stat & STA_NOINIT) return RES_NOTRDY;	/* Check drive status */
	if (Stat & STA_PROTECT) return RES_WRPRT;	/* Check write protect */

	if (!(CardType & CT_BLOCK)) sector *= 512;	/* LBA ==> BA conversion (byte addressing cards) */

	if (count == 1) {	/* Single sector write */
		if ((send_cmd(CMD24, sector) == 0)	/* WRITE_BLOCK */
			&& xmit_datablock(buff, 0xFE))
			count = 0;
	}
	else {				/* Multiple sector write */
		if (CardType & CT_SDC) send_cmd(ACMD23, count);
		if (send_cmd(CMD25, sector) == 0) {	/* WRITE_MULTIPLE_BLOCK */
			do {
				if (!xmit_datablock(buff, 0xFC)) break;
				buff += 512;
			} while (--count);
			if (!xmit_datablock(0, 0xFD))	/* STOP_TRAN token */
				count = 1;
		}
	}
	deselect();

	return count ? RES_ERROR : RES_OK;	/* Return result */
}
#endif /* _READONLY == 0 */



/*-----------------------------------------------------------------------*/
/* Miscellaneous drive controls other than data read/write               */
/*-----------------------------------------------------------------------*/

#if _USE_IOCTL != 0
DRESULT disk_ioctl (
	BYTE drv,		/* Physical drive number (0) */
	BYTE ctrl,		/* Control command code */
	void *buff		/* Pointer to the conrtol data */
)
{
	DRESULT res;
	BYTE n, csd[16], *ptr = buff;
	WORD csize;
	DWORD *dp, st, ed;


	if (drv) return RES_PARERR;					/* Check parameter */
	if (Stat & STA_NOINIT) return RES_NOTRDY;	/* Check if drive is ready */

	res = RES_ERROR;

	switch (ctrl) {
	case CTRL_SYNC :		/* Wait for end of internal write process of the drive */
		if (select()) {
			deselect();
			res = RES_OK;
		}
		break;

	case GET_SECTOR_COUNT :	/* Get drive capacity in unit of sector (DWORD) */
		if ((send_cmd(CMD9, 0) == 0) && rcvr_datablock(csd, 16)) {
			if ((csd[0] >> 6) == 1) {	/* SDC ver 2.00 */
				csize = csd[9] + ((WORD)csd[8] << 8) + 1;
				*(DWORD*)buff = (DWORD)csize << 10;
			} else {					/* SDC ver 1.XX or MMC ver 3 */
				n = (csd[5] & 15) + ((csd[10] & 128) >> 7) + ((csd[9] & 3) << 1) + 2;
				csize = (csd[8] >> 6) + ((WORD)csd[7] << 2) + ((WORD)(csd[6] & 3) << 10) + 1;
				*(DWORD*)buff = (DWORD)csize << (n - 9);
			}
			res = RES_OK;
		}
		break;

	case GET_SECTOR_SIZE :	/* Get sector size in unit of byte (WORD) */
		*(WORD*)buff = 512;
		res = RES_OK;
		break;

	case GET_BLOCK_SIZE :	/* Get erase block size in unit of sector (DWORD) */
		if (CardType & CT_SD2) {	/* SDC ver 2.00 */
			if (send_cmd(ACMD13, 0) == 0) {	/* Read SD status */
				rcvr_spi();
				if (rcvr_datablock(csd, 16)) {				/* Read partial block */
					for (n = 64 - 16; n; n--) rcvr_spi();	/* Purge trailing data */
					*(DWORD*)buff = 16UL << (csd[10] >> 4);
					res = RES_OK;
				}
			}
		} else {					/* SDC ver 1.XX or MMC */
			if ((send_cmd(CMD9, 0) == 0) && rcvr_datablock(csd, 16)) {	/* Read CSD */
				if (CardType & CT_SD1) {	/* SDC ver 1.XX */
					*(DWORD*)buff = (((csd[10] & 63) << 1) + ((WORD)(csd[11] & 128) >> 7) + 1) << ((csd[13] >> 6) - 1);
				} else {					/* MMC */
					*(DWORD*)buff = ((WORD)((csd[10] & 124) >> 2) + 1) * (((csd[11] & 3) << 3) + ((csd[11] & 224) >> 5) + 1);
				}
				res = RES_OK;
			}
		}
		break;

	case CTRL_ERASE_SECTOR :	/* Erase a block of sectors (used when _USE_ERASE == 1) */
		if (!(CardType & CT_SDC)) break;				/* Check if the card is SDC */
		if (disk_ioctl(drv, MMC_GET_CSD, csd)) break;	/* Get CSD */
		if (!(csd[0] >> 6) && !(csd[10] & 0x40)) break;	/* Check if sector erase can be applied to the card */
		dp = buff; st = dp[0]; ed = dp[1];				/* Load sector block */
		if (!(CardType & CT_BLOCK)) {
			st *= 512; ed *= 512;
		}
		if (send_cmd(CMD32, st) == 0 && send_cmd(CMD33, ed) == 0 && send_cmd(CMD38, 0) == 0 && wait_ready(30000))	/* Erase sector block */
			res = RES_OK;
		break;

	/* Following command are not used by FatFs module */

	case MMC_GET_TYPE :		/* Get MMC/SDC type (BYTE) */
		*ptr = CardType;
		res = RES_OK;
		break;

	case MMC_GET_CSD :		/* Read CSD (16 bytes) */
		if (send_cmd(CMD9, 0) == 0		/* READ_CSD */
			&& rcvr_datablock(ptr, 16))
			res = RES_OK;
		break;

	case MMC_GET_CID :		/* Read CID (16 bytes) */
		if (send_cmd(CMD10, 0) == 0		/* READ_CID */
			&& rcvr_datablock(ptr, 16))
			res = RES_OK;
		break;

	case MMC_GET_OCR :		/* Read OCR (4 bytes) */
		if (send_cmd(CMD58, 0) == 0) {	/* READ_OCR */
			for (n = 4; n; n--) *ptr++ = rcvr_spi();
			res = RES_OK;
		}
		break;

	case MMC_GET_SDSTAT :	/* Read SD status (64 bytes) */
		if (send_cmd(ACMD13, 0) == 0) {	/* SD_STATUS */
			rcvr_spi();
			if (rcvr_datablock(ptr, 64))
				res = RES_OK;
		}
		break;

	default:
		res = RES_PARERR;
	}

	deselect();

	return res;
}
#endif /* _USE_IOCTL != 0 */


/*-----------------------------------------------------------------------*/
/* Device timer function  (Platform dependent)                           */
/*-----------------------------------------------------------------------*/
/* This function must be called from timer interrupt routine in period
/  of 1 ms to generate card control timing.
*/

void disk_timerproc (void)
{
	WORD n;
	BYTE s;


	n = Timer1;						/* 1kHz decrement timer stopped at 0 */
	if (n) Timer1 = --n;
	n = Timer2;
	if (n) Timer2 = --n;

	s = Stat;
	if (WP)		/* Write protected */
		s |= STA_PROTECT;
	else		/* Write enabled */
		s &= ~STA_PROTECT;
	if (INS)	/* Card is in socket */
		s &= ~STA_NODISK;
	else		/* Socket empty */
		s |= (STA_NODISK | STA_NOINIT);
	Stat = s;
}

void disk_ins (int ins)
{
	INS = ins;
}

