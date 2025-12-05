/*
 * This source code is for executing MS-DOS I/O device driver.
 *
 * Designed by Akihito Honda(Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Target: PIC18F57QXX/PIC18F47QXX
 *  Date. 2024.4.20
 *  2025/09/03 Update for DOS3.1
 */


#include "mez88.h"
#include <stdio.h>
#include <assert.h>

#include "../fatfs/ff.h"
#include "../drivers/utils.h"

#define SECTOR_SIZE      512
#define SECTOR_SZPH      SECTOR_SIZE >> 4

static DPB dsk_t[NUM_DRIVES];

// Device Header
#define CONDEV	0x73
#define AUXDEV	0x85
#define PRNDEV	0x97
#define TIMDEV	0xA9
#define DSKDEV	0xBB

//
// file pointer for each Drive
//
FIL *filep_t[DRVMAX];

// from unimon
#define CONIN_REQ	0x01
#define CONOUT_REQ	0x02
#define CONST_REQ	0x03
#define STROUT_REQ	0x04
#define TERMINATE	0xff

// from IO.SYS
#define AUX_INT		0x10
#define PRN_INT		0x20
#define TIM_INT		0x30
#define CON_INT		0x40
#define DSK_INT		0x50

static int open_cnt;
static uint8_t disk_drive;
static uint16_t disk_sector;
static uint16_t disk_dmal;
static uint16_t disk_dmah;
static uint8_t disk_buf[SECTOR_SIZE];
static uint8_t verify_buf[SECTOR_SIZE];

static void dev_exit(iodat *req_h);
static void dev_tim(iodat *req_h);
static void dev_aux(iodat *req_h);
static void dev_DS1307(iodat *req_h);

// clock device invoke pointer. timer0 or DS1307 RTC module
static void (*clock_device)(iodat *req_h);
// aux device invoke pointer. uart1 or dev_FT200XD USB-UART module
static void (*aux_device)(iodat *req_h);

static const char *setup0 = "Setup timer0 for clock device.\r\n";
static const char *setup2 = "Setup DS1307 RTC module for clock device.\r\n";

void dosio_init(void) {
    disk_drive = 0;
    disk_dmal = 0;
    disk_dmah = 0;
	open_cnt = 0;
}

uint16_t dos_sec_size(void) {return(SECTOR_SIZE);}

void setup_clk_dev(void) {

	// setup_val = 0, 1
	if ( !time_dev ) {
		// setup timer0
		printf("%s", setup0);
		clock_device = dev_tim;
	}
	else {
		// setup RTC module.
		printf("%s", setup2);
		clock_device = dev_DS1307;
	}
}

void setup_dpb(void) {
	int drv, i;
	FIL *filep;
	UINT br;
	dsk_param *dsk_h;
	uint8_t *buf;

	dsk_h = (dsk_param *)&tmp_buf[0][0];
	for( drv=0; drv < NUM_DRIVES; drv++ ) {
		filep = filep_t[drv];
		if ( filep == NULL ) continue;		// skip
		f_rewind(filep);					// top of file
		f_read(filep, (void *)dsk_h, sizeof(dsk_param), &br);	// get boot sec
	
		//copy DPB data to dpb_t[drv]
		buf = (uint8_t *)&dsk_t[drv];
		for( i = 0; i < sizeof(DPB); i++ ) *buf++ = tmp_buf[0][i];
	}
}

void copy_dpb(void) {

	uint32_t addr;
	iosys_head *h;
	int i;

	/* copy Disk Parameter Block to physical memory area */
	h = (iosys_head *)iosys_off;
	for( i=0; i<DRVMAX; i++) {
		addr = get_physical_addr(iosys_seg, (uint16_t)&(h->drive[i]));
		write_sram(addr, (uint8_t *)&dsk_t[i], (unsigned int)sizeof(DPB));
	}
}

/*////////////////////////////////////////////////////////
;
; Universal monitor I/F
;
;	CONIN_REQ	0x01
;	CONOUT_REQ	0x02
;	CONST_REQ	0x03
;	STROUT_REQ	0x04
;	TERMINATE	0xff

////////////////////////////////////////////////////////*/

static void unimon_console(PTRSAV *u_buff) {

	uint8_t *buf;
	uint16_t cnt;

	switch (u_buff->UREQ_COM) {
		// CONIN
		case CONIN_REQ:
			u_buff->UNI_CHR = (uint8_t)getch();
			break;
		// CONOUT
		case CONOUT_REQ:
			putch((char)u_buff->UNI_CHR);		// Write data
			break;
		// CONST
		case CONST_REQ:
			u_buff->UNI_CHR = (uint8_t)(rx_cnt !=0);
			break;
		case STROUT_REQ:
			buf = tmp_buf[0];
			cnt = (uint16_t)u_buff->UNI_CHR;
			// get string
			read_sram(get_physical_addr(u_buff->STR_SEG, u_buff->STR_off), buf, cnt);
			while( cnt ) {
				putch( *buf++);
				cnt--;
			}
			break;
		case TERMINATE:
			nmi_sig = 0;
			terminate = 1;
	}
	u_buff->UREQ_COM = 0;	// clear unimon request
}

#define CON_CERR	3	// Reserved. (Currently returns error)
#define CON_READ	4	// Character read. (Destructive)
#define CON_RDND	5	// Character read. (Non-destructive)
#define CON_FLSH	7	// Character write.
#define CON_WRIT1	8	// Character write.
#define CON_WRIT2	9	// Character write.
#if 0
static void dsk_sec_err(iodat *req_h) {
	req_h->status = (uint16_t)0x8108;			//set error code & done bits
}
#endif
static void dsk_drv_err(iodat *req_h) {
	req_h->status = (uint16_t)0x8101;			//set error code & done bits
}

static void dsk_rd_err(iodat *req_h) {
	req_h->status = (uint16_t)0x810b;			//set error code & done bits
}

static void dsk_wr_err(iodat *req_h) {
	req_h->status = (uint16_t)0x810a;			//set error code & done bits
}

static void dsk_crc_err(iodat *req_h) {
	req_h->status = (uint16_t)0x8104;			//set error code & done bits
}

static void dsk_bpb_err(iodat *req_h) {
	req_h->status = (uint16_t)0x8107;			//set error code & done bits
}

static void dsk_media_err(iodat *req_h) {
	req_h->status = (uint16_t)0x8102;			//set error code & done bits
}
/*
static void dsk_no_media(iodat *req_h) {
	req_h->status = (uint16_t)0x810f;			//set error code & done bits
}
*/
static void command_error(iodat *req_h) {
	req_h->status = (uint16_t)0x8103;			//set error code & done bits
}

static void busy_exit(iodat *req_h) {
	req_h->status = (uint16_t)0x0300;			//set busy code & done bits
}

static void dev_exit(iodat *req_h) {
	req_h->status = (uint16_t)0x0100;			//set done bits
}

//static void not_support(iodat *req_h) {
//	dev_exit(req_h);
//};

static void dev_con(iodat *req_h) {

	uint8_t	*buf;
	uint32_t trans_adr;

	buf = tmp_buf[1];
	trans_adr = get_physical_addr(req_h->trans_seg, req_h->trans_off);
	
	switch(req_h->cmd) {
		case CON_CERR:
			command_error(req_h);
			break;

		case CON_READ:
			*buf = (uint8_t)getch();
			write_sram( trans_adr, buf, 1 );
			dev_exit(req_h);
			break;

		case CON_RDND:
			if (rx_cnt !=0) {
				req_h->media = rx_buf[rx_rp];
				dev_exit(req_h);
			}
			else busy_exit(req_h);
			break;

		case CON_FLSH:
			GIE = 0;                // Disable interrupt
			rx_wp = 0;
			rx_rp = 0;
			rx_cnt = 0;
			GIE = 1;                // Eable interrupt
			dev_exit(req_h);
			break;

		case CON_WRIT1:
		case CON_WRIT2:
			read_sram( trans_adr, buf, 1 );
			putch(*buf);

		default:
			dev_exit(req_h);
	}
};

#define TIM_ERR		3			//Reserved. (Currently returns an error)
#define TIM_RED		4			//Character read. (Destructive)
#define TIM_BUSY	5			//(Not used, returns busy flag.)
#define TIM_WRT1	8			//Character write.
#define TIM_WRT2	9			//Character write with verify.

static void dev_tim(iodat *req_h) {

	uint32_t trans_adr;

	trans_adr = get_physical_addr(req_h->trans_seg, req_h->trans_off);

	switch(req_h->cmd) {
		case TIM_ERR:
			command_error(req_h);
			break;

		case TIM_RED:
			TMR0IE = 0;			// disable timer0 interrupt
			write_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
			TMR0IE = 1;			// Enable timer0 interrupt
			dev_exit(req_h);
			break;

		case TIM_BUSY:
			busy_exit(req_h);
			break;

		case TIM_WRT1:
		case TIM_WRT2:
			TMR0IE = 0;			// disable timer0 interrupt
			read_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
			TMR0IE = 1;			// Enable timer0 interrupt
			dev_exit(req_h);
			break;

		default:
			dev_exit(req_h);
	}
};

/*
* DS1307 format
* rtc[0] : seconds (BCD) 00-59
* rtc[1] : minuts  (BCD) 00-59
* rtc[2] : hours   (BCD) 00-23 (or 1-12 +AM/PM)
* rtc[3] : day     (BCD) week day 01-07
* rtc[4] : date    (BCD) 01-31
* rtc[5] : month   (BCD) 01-12
* rtc[6] : year    (BCD) 00-99 : range : (19)80-(19)99, (20)00-(20)79
*/
static void dev_DS1307(iodat *req_h) {

	uint32_t trans_adr;
	uint8_t rtc[7];
	uint16_t year, month, date;
	
	trans_adr = get_physical_addr(req_h->trans_seg, req_h->trans_off);

	switch(req_h->cmd) {
		case TIM_ERR:
			command_error(req_h);
			break;

		case TIM_RED:
			// get RTC data
			if ( read_I2C(DS1307, 0, 7, &rtc[0]) == 0xFF) {
				command_error(req_h);
				break;
			}
			// convert BCD to bin
			TMR0IE = 0;			// disable timer0 interrupt
			tim_pb.TIM_SECS = cnv_byte(rtc[0]);
			tim_pb.TIM_MINS = cnv_byte(rtc[1]);
			tim_pb.TIM_HRS  = cnv_byte(rtc[2]);

			date  = (uint16_t)cnv_byte(rtc[4]);
			month = (uint16_t)cnv_byte(rtc[5]);
			year  = (uint16_t)cnv_byte(rtc[6]);
			if (year >= 80) year += 1900;
			else year += 2000;

			// convert year, month and date to number of days from 1980
			tim_pb.TIM_DAYS = days_from_1980(year, month, date);

			TMR0IE = 1;			// Enable timer0 interrupt
			// write time date to TPB
			write_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
			dev_exit(req_h);
			break;

		case TIM_BUSY:
			busy_exit(req_h);
			break;

		case TIM_WRT1:
		case TIM_WRT2:
			read_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
			// convert number of days to year, month and date
			cnv_ymd(tim_pb.TIM_DAYS, &year, &month, &date );
			// convert bin to BCD
			rtc[0] = cnv_bcd(tim_pb.TIM_SECS);
			rtc[1] = cnv_bcd(tim_pb.TIM_MINS);
			rtc[2] = cnv_bcd(tim_pb.TIM_HRS);
			rtc[4] = cnv_bcd((uint8_t)date);
			rtc[5] = cnv_bcd((uint8_t)month);
			rtc[6] = cnv_bcd((uint8_t)year);
			// write to RTC
			if (write_I2C(DS1307, 0, 7, &rtc[0] ) == 0xFF) {
				command_error(req_h);
				break;
			}

		default:
			dev_exit(req_h);
	}
}

#define AUX_CERR	3	// Reserved. (Currently returns error)
#define AUX_READ	4	// Character read. (Destructive)
#define AUX_RDND	5	// Character read. (Non-destructive)
#define AUX_FLSH	7	// Character write.
#define AUX_WRIT1	8	// Character write.
#define AUX_WRIT2	9	// Character write.

static void dev_aux(iodat *req_h) {
	uint8_t	*buf;
	uint32_t trans_adr;

	buf = tmp_buf[1];
	trans_adr = get_physical_addr(req_h->trans_seg, req_h->trans_off);
	
	switch(req_h->cmd) {
		case AUX_CERR:
			command_error(req_h);
			break;

		case AUX_READ:
			*buf = (uint8_t)getax();
			write_sram( trans_adr, buf, 1 );
			dev_exit(req_h);
			break;

		case AUX_RDND:
			if (ax_cnt !=0) {
				req_h->media = ax_buf[ax_rp];
				dev_exit(req_h);
			}
			else busy_exit(req_h);
			break;

		case AUX_FLSH:
			U5RXIE = 0;					// disable Rx interruot
			ax_wp = 0;
			ax_rp = 0;
			ax_cnt = 0;
		    U5RXIE = 1;          // Receiver interrupt enable
			dev_exit(req_h);
			break;

		case AUX_WRIT1:
		case AUX_WRIT2:
			read_sram( trans_adr, buf, 1 );
			putax(*buf);
		default:
			dev_exit(req_h);
	}
};

static void dev_prn(iodat *req_h) {
	dsk_media_err(req_h);
};

#define DSK_INIT	0		// Initialize Driver.
#define MEDIAC		1		// Return current media code.
#define GET_BPB		2		// Get Bios Parameter Block.
#define CMDERR		3		// Reserved. (currently returns error)
#define DSK_RED		4		// Block read.
#define CMDBSY		5		// (Not used, return busy flag)
#define DSK_WRT		8		// Block write.
#define DSK_WRV		9		// Block write with verify.
// for DOS3.1
#define DSK_OPN		13		// Disk Open.
#define DSK_CLS		14		// Disk Close
#define DSK_RMV		15		// check Removable media

static void set_dskinit_bpb(CMDP *req_h) {

	iosys_head *h;
	
	h = (iosys_head *)iosys_off;

	req_h->bpb1 = DRVMAX;						/* max drive */
	req_h->bpb3_off = (uint16_t)&h->inittab[0];	/* inittab offset */
	req_h->bpb3_seg = iosys_seg;
}

static void dsk_media_check(MEDIAS *req_h) {
	req_h->medias2 = 1;			// No replacement
}

static int set_bpb(BPB *req_h) {

	iosys_head *h;
	dsk_param *dsk_h;
	uint8_t drv;
	
	drv = req_h->unit;
	if ( drv > DRVMAX ) return -1;

	h = (iosys_head *)iosys_off;
	req_h->bpb3_off = (uint16_t)&h->drive[drv] + sizeof(DPB_HEAD);
	req_h->media = dsk_t[drv].media_id;
	req_h->bpb3_seg = iosys_seg;
	return(0);
}

static int set_drive( iodat *req_h ) {
	uint8_t u;
	
	u = req_h->unit;
	if ( u >= DRVMAX ) return( -1 );
	disk_drive = u;
	return(0);
}

static int setup_drive(iodat *req_h) {
	if ( set_drive( req_h ) ) {
		dsk_drv_err( req_h );
		return( -1 );
	}
	disk_dmal = req_h->trans_off;
	disk_dmah = req_h->trans_seg;

	return( 0 );
}

static int seek_disk(iodat *req_h) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = filep_t[disk_drive];

	if (filep_t[disk_drive] == NULL) return(-1);
	if ((fres = f_lseek(filep, (uint32_t)disk_sector * SECTOR_SIZE)) != FR_OK) {
		printf("f_lseek(): ERROR %d\r\n", fres);
		return(-1);
	}
	return(0);
}

static int read_sector(iodat *req_h, uint8_t *buf, int flg) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = filep_t[disk_drive];
	
	if (seek_disk(req_h)) return(-1);

	// read from the DISK
	if ((fres = f_read(filep, buf, SECTOR_SIZE, &n)) != FR_OK || n != SECTOR_SIZE) {
		printf("f_read(): ERROR res=%d, n=%d\r\n", fres, n);
		return(-1);
	}
	else if (DEBUG_DISK_READ && DEBUG_DISK_VERBOSE && !(debug.disk_mask & (1 << disk_drive))) {
				util_hexdump_sum("buf: ", buf, SECTOR_SIZE);
	}
	else {
		if (flg) {
			// transfer read data to SRAM
			write_sram(get_physical_addr( disk_dmah, disk_dmal ), buf, SECTOR_SIZE);

			#ifdef MEM_DEBUG
			uint32_t addr = get_physical_addr( disk_dmah, disk_dmal );
			printf("f_read(): SRAM address(%08lx),disk_dmah(%04x),disk_dmal(%04x)\r\n", addr, disk_dmah, disk_dmal);
			read_sram(addr, buf, SECTOR_SIZE);
			util_hexdump_sum("RAM: ", buf, SECTOR_SIZE);
			#endif  // MEM_DEBUG
		}
	}
	return(0);
}

static int read_disk(iodat *req_h) {
	uint16_t cnt;

	disk_sector = req_h->start;			//set logical sector No from MSDOS.SYS
	cnt = req_h->count;

	while( cnt ) {
		if (read_sector(req_h, disk_buf, 1)) {
			req_h->count -= cnt;		// set number read sectors
			return(-1);
		}
		--cnt;
		disk_dmah += SECTOR_SZPH;		//paragraph of SECTOR_SIZE
		disk_sector++;
	}
	return(0);
}

static int write_sector(iodat *req_h) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = filep_t[disk_drive];
	
	if (seek_disk(req_h)) return(-1);

	// transfer write data from SRAM to the buffer
	read_sram(get_physical_addr( disk_dmah, disk_dmal ), disk_buf, SECTOR_SIZE);

	if (DEBUG_DISK_WRITE && DEBUG_DISK_VERBOSE && !(debug.disk_mask & (1 << disk_drive))) {
		util_hexdump_sum("buf: ", disk_buf, SECTOR_SIZE);
	}

	// write buffer to the DISK
	if ((fres = f_write(filep, disk_buf, SECTOR_SIZE, &n)) != FR_OK || n != SECTOR_SIZE) {
		printf("f_write(): ERROR res=%d, n=%d\r\n", fres, n);
		return(-1);
	}
	else if ((fres = f_sync(filep)) != FR_OK) {
		printf("f_sync(): ERROR %d\r\n", fres);
		return(-1);
	}
	return(0);
}

static int write_disk(iodat *req_h) {
	uint16_t cnt;

	disk_sector = req_h->start;
	cnt = req_h->count;
	while( cnt ) {
		if (write_sector(req_h)) {
			req_h->count -= cnt;		// set number read sectors
			return(-1);
		}
		--cnt;
		disk_dmah += SECTOR_SZPH;		//paragraph of SECTOR_SIZE
		disk_sector++;
	}
	return(0);
}

static int write_verify(iodat *req_h) {

	uint16_t cnt, i;

	disk_sector = req_h->start;
	cnt = req_h->count;
	while( cnt ) {
		if (write_sector(req_h)) {
			req_h->count -= cnt;		// set number read sectors
			return(-1);
		}
		if (read_sector(req_h, verify_buf, 0)) {
			req_h->count -= cnt;		// set number read sectors
			return(-1);
		}
		for(i=0; i != SECTOR_SIZE; i++) {
			if (disk_buf[i] != verify_buf[i]) {
				req_h->count -= cnt;		// set number read sectors
				return(-1);
			}
		}
		--cnt;
		disk_dmah += SECTOR_SZPH;		//paragraph of SECTOR_SIZE
		disk_sector++;
	}
	return(0);
}

static void dev_dsk(iodat *req_h) {
	uint8_t	*buf;
	uint16_t cnt1, cnt2;
	uint32_t trans_adr;

	buf = (uint8_t *)tmp_buf[1];
	switch(req_h->cmd) {
		case DSK_INIT:		// Initialize Driver.
			set_dskinit_bpb((CMDP *)req_h);
			dev_exit(req_h);
			break;
		case MEDIAC:		// Return current media code.
			dsk_media_check( (MEDIAS *)req_h );
			dev_exit(req_h);
			break;
		case GET_BPB:		// Get Bios Parameter Block.
			if (set_bpb((BPB *)req_h)) dsk_bpb_err(req_h);
			else dev_exit(req_h);
			break;
		case CMDERR:		// Reserved. (currently returns error)
			command_error(req_h);
			break;
		case DSK_RED:		// Block read.
			if ( setup_drive(req_h) ) break;
			if ( read_disk(req_h) ) dsk_rd_err(req_h);
			else dev_exit(req_h);
			break;
		case CMDBSY:		// (Not used, return busy flag)
			busy_exit(req_h);
			break;
		case DSK_WRT:		// Block write.
			if ( setup_drive(req_h) ) break;
			if ( write_disk(req_h) ) dsk_wr_err(req_h);
			else dev_exit(req_h);
			break;
		case DSK_WRV:		// Block write with verify.
			if ( setup_drive(req_h) ) break;
			if ( write_verify(req_h) ) {
				dsk_crc_err(req_h);
				break;
			}
		case DSK_OPN:			// 13: Disk Open.
			open_cnt++;
			dev_exit(req_h);
			break;
		case DSK_CLS:			// 14: Disk Close
			if (!open_cnt) command_error(req_h);
			else dev_exit(req_h);
			break;
		case DSK_RMV:			// 15: check Removable media
			busy_exit(req_h);
			break;
		default:
			dev_exit(req_h);
	}
}

//
// bus master handling
// this fanction is invoked at main() after HOLDA = 1
//
// bioreq_buffadr = top address of unimon
//
//;  ---- unimon request
//; UREQ_COM = 1 ; unimon request CONIN  : return char in UNI_CHR
//;          = 2 ; unimon request CONOUT : UNI_CHR = output char
//;          = 3 ; unimon request CONST  : return status in UNI_CHR
//;                                      : ( 0: no key, 1 : key exist )
//;			 = 4 ; unimon request STROUT
//;          = 0 ; unimon request is done
//;
//;  ---- IO.SYS request
//; UREQ_COM = 10h ; AUX INT
//;	   = 20h ; PRN INT
//;	   = 30h ; TIM INT
//;	   = 40h ; CON INT
//;	   = 50h ; DSK INT
//;
void dos_bus_master_operation(void) {

	uint32_t addr;
	PTRSAV u_buff;
	iodat *req_h;

	// read request from 8088/V20
	read_sram(bioreq_buffadr, (uint8_t *)&u_buff, (unsigned int)sizeof(PTRSAV));

	if ( u_buff.UREQ_COM ) {
		unimon_console(&u_buff);
	}
	else {
		// get DOS request header from SRAM into PIC buffer
		req_h = (iodat *)tmp_buf[0];
		addr = (uint32_t)(u_buff.PTRSAV_SEG)*0x10 + (uint32_t)u_buff.PTRSAV_off;
		read_sram(addr, (uint8_t *)req_h, (unsigned int)sizeof(iodat));
		
		switch ( u_buff.DREQ_COM ) {
			case AUX_INT:
				dev_aux(req_h);
				break;

			case PRN_INT:
				dev_prn(req_h);
				break;

			case TIM_INT:
				// dev_tim(req_h);
				(*clock_device)(req_h);
				break;

			case CON_INT:
				dev_con(req_h);
				break;

			case DSK_INT:
				dev_dsk(req_h);
				break;

			default:
				req_h -> status = (uint16_t)0x8103;		//set error code & done bits
				printf("UNKNOWN DEVICE : CMD(%02x)\r\n", u_buff.UREQ_COM);
		}
		write_sram(addr, (uint8_t *)req_h, (unsigned int)sizeof(iodat));	// save request header
	}
	// write end request to SRAM for 8088/V20
	write_sram(bioreq_buffadr, (uint8_t *)&u_buff, 2);	// 2bytes( UREQ_COM & UNI_CHR )
}
