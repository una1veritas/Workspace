/*
 * This source code is for executing CPM I/O requests.
 *
 * Base source code is maked by @hanyazou
 *  https://twitter.com/hanyazou
 *
 * Redesigned by Akihito Honda(Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Target: PIC18F57QXX/PIC18F47QXX
 *  Date. 2024.4.20
 *  UPDATE 2025.8.10
 */

#include "mez88.h"
#include <stdio.h>
#include <assert.h>

#include "../fatfs/ff.h"
#include "../drivers/utils.h"

#define SECTOR_SIZE      128

// from unimon
#define CONIN_REQ	0x01
#define CONOUT_REQ	0x02
#define CONST_REQ	0x03
#define STROUT_REQ	0x04
#define REQ_DREAD	0x05
#define REQ_DWRITE	0x06
#define READ_TIME	0x07
#define SET_TIME	0x08
#define CONIN_REQ1	0x09
#define CONOUT_REQ1	0x0A
#define STRIN_REQ	0x0B
#define TERMINATE	0xff

drive_t cpm_drives[] = {
    { 26 },
    { 26 },
    { 128 },
    { 128 },
    { 16484 },
};

#define RETURN_TBL 2		/* bytes of return parameter */

// request table
static cpm_hdr req_tbl;
// disk buffer
static uint8_t disk_buf[SECTOR_SIZE];

void cpmio_init(void) {
	req_tbl.UREQ_COM = 0;
	req_tbl.CREQ_COM = 0;
}

static int seek_disk(void) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = cpm_drives[req_tbl.disk_drive].filep;
	uint32_t sector;

	if (cpm_drives[req_tbl.disk_drive].filep == NULL) return(-1);
	sector = req_tbl.disk_track * cpm_drives[req_tbl.disk_drive].sectors + req_tbl.disk_sector - 1;
	
	if ((fres = f_lseek(filep, sector * SECTOR_SIZE)) != FR_OK) {
		printf("f_lseek(): ERROR %d\r\n", fres);
		return(-1);
	}
	return(0);
}

static int write_sector(void) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = cpm_drives[req_tbl.disk_drive].filep;
	uint32_t addr;
	
	if (seek_disk()) return(-1);

	addr = get_physical_addr( req_tbl.data_dmah, req_tbl.data_dmal );

	// transfer write data from SRAM to the buffer
	read_sram(addr, disk_buf, SECTOR_SIZE);

	if (DEBUG_DISK_WRITE && DEBUG_DISK_VERBOSE && !(debug.disk_mask & (1 << req_tbl.disk_drive))) {
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

static int read_sector(void) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = cpm_drives[req_tbl.disk_drive].filep;
	uint32_t addr;
	
	if (seek_disk()) return(-1);

	// read from the DISK
	if ((fres = f_read(filep, disk_buf, SECTOR_SIZE, &n)) != FR_OK || n != SECTOR_SIZE) {
		printf("f_read(): ERROR res=%d, n=%d\r\n", fres, n);
		return(-1);
	}
	else if (DEBUG_DISK_READ && DEBUG_DISK_VERBOSE && !(debug.disk_mask & (1 << req_tbl.disk_drive))) {
		util_hexdump_sum("buf: ", disk_buf, SECTOR_SIZE);
	}
	else {
		// transfer read data to SRAM
		addr = get_physical_addr( req_tbl.data_dmah, req_tbl.data_dmal );
		write_sram(addr, disk_buf, SECTOR_SIZE);

		#ifdef MEM_DEBUG
		printf("f_read(): SRAM address(%08lx),data_dmah(%04x),data_dmal(%04x)\r\n",
			     addr, req_tbl.data_dmah, req_tbl.data_dmal);
		read_sram(addr, disk_buf, SECTOR_SIZE);
		util_hexdump_sum("RAM: ", disk_buf, SECTOR_SIZE);
		#endif  // MEM_DEBUG
	}
	return(0);
}

static int setup_drive(void) {
	req_tbl.CBI_CHR = 0;		/* clear error status */
	if ( req_tbl.disk_drive >= NUM_DRIVES ) return( -1 );
	if ( cpm_drives[req_tbl.disk_drive].sectors == 0 ) return( -1 );	// not support disk
	if ( req_tbl.disk_sector > cpm_drives[req_tbl.disk_drive].sectors ) return( -1 ); // bad sector
	return( 0 );
}

static void dsk_err(void) {
	req_tbl.UNI_CHR = 1;
}

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
static uint8_t rd_time() {

	uint32_t trans_adr;
	uint16_t year, month, date;
	
	trans_adr = get_physical_addr( req_tbl.data_dmah, req_tbl.data_dmal );

	//read TIME
	if (time_dev) { //DS1307
		// get RTC data
		if ( read_I2C(DS1307, 0, 7, &rtc[0]) == 0xFF) return 1;

		// convert BCD to bin
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

		// write time date to TPB
		write_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
	}
	else {			//TIMER0
		TMR0IE = 0;			// disable timer0 interrupt
		write_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
		TMR0IE = 1;			// Enable timer0 interrupt
	}
	return 0;
}

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
static uint8_t wr_time() {

	uint32_t trans_adr;
	uint16_t year, month, date;
	
	trans_adr = get_physical_addr( req_tbl.data_dmah, req_tbl.data_dmal );

	// set time
	if (time_dev) { //DS1307
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
		if (write_I2C(DS1307, 0, 7, &rtc[0] ) == 0xFF) return 1;
	}
	else {			//TIMER0
		TMR0IE = 0;			// disable timer0 interrupt
		read_sram( trans_adr, (uint8_t *)&tim_pb, sizeof(tim_pb) );
		TMR0IE = 1;			// Enable timer0 interrupt
	}
	return 0;
}	

static void unimon_console(void) {

	uint8_t *buf, c;
	uint16_t cnt;

	switch (req_tbl.UREQ_COM) {
		// CONIN
		case CONIN_REQ:
			req_tbl.UNI_CHR = (uint8_t)getch();
			break;
		// CONOUT
		case CONOUT_REQ:
			putch((char)req_tbl.UNI_CHR);		// Write data
			break;
		// CONST
		case CONST_REQ:
			req_tbl.UNI_CHR = (uint8_t)(rx_cnt !=0);
			break;
		case STROUT_REQ:
			buf = tmp_buf[0];
			cnt = (uint16_t)req_tbl.UNI_CHR;
			// get string
			read_sram(get_physical_addr(req_tbl.STR_SEG, req_tbl.STR_off), buf, cnt);
			while( cnt ) {
				putch( *buf++);
				cnt--;
			}
			break;
		case CONIN_REQ1:
			if ( rx_cnt ) req_tbl.UNI_CHR = (uint8_t)getch();
			else req_tbl.UNI_CHR = 0;
			break;
		case CONOUT_REQ1:
			c = req_tbl.UNI_CHR;
			req_tbl.UNI_CHR = out_chr(c);
			break;
		case STRIN_REQ:
			buf = tmp_buf[0];
			cnt = (uint16_t)get_str((char *)buf, req_tbl.UNI_CHR);
			req_tbl.UNI_CHR = (uint8_t)cnt;
			if (cnt) write_sram(get_physical_addr(req_tbl.STR_SEG, req_tbl.STR_off), buf, (unsigned int)cnt);
			break;
		case TERMINATE:
			nmi_sig = 0;
			terminate = 1;
			break;
		default:
			printf("UNKNOWN unimon CMD(%02x)\r\n", req_tbl.UREQ_COM);
	}
	req_tbl.UREQ_COM = 0;	// clear unimon request
}

//
// bus master handling
// this fanction is invoked at main() after HOLDA = 1
//
// bioreq_ubuffadr = top address of unimon
//
void cpm_bus_master_operation(void) {
	uint32_t addr;
	uint8_t *buf, c;
	uint16_t cnt;

	// read request from 8088/V20
	read_sram(bioreq_ubuffadr, (uint8_t *)&req_tbl, (unsigned int)sizeof(cpm_hdr));

	if (req_tbl.UREQ_COM) {
		unimon_console();
		// write end request to SRAM for 8088/V20
		write_sram(bioreq_ubuffadr, (uint8_t *)&req_tbl, RETURN_TBL);	// 2bytes
	}
	else {
		switch (req_tbl.CREQ_COM) {
			// CONIN
			case CONIN_REQ:
				req_tbl.CBI_CHR = (uint8_t)getch();
				break;
			// CONOUT
			case CONOUT_REQ:
				putch((char)req_tbl.CBI_CHR);		// Write data
				break;
			// CONST
			case CONST_REQ:
				req_tbl.CBI_CHR = (rx_cnt !=0) ? 255 : 0;
				break;
			case STROUT_REQ:
				buf = tmp_buf[0];
				cnt = (uint16_t)req_tbl.CBI_CHR;
				// get string
				read_sram(get_physical_addr(req_tbl.data_dmah, req_tbl.data_dmal), buf, cnt);
				while( cnt ) {
					putch( *buf++);
					cnt--;
				}
				break;
			case REQ_DREAD:
				if ( setup_drive() ) {
					dsk_err();
					break;
				}
				if ( read_sector() ) {
					dsk_err();
					break;
				}
				break;
			case REQ_DWRITE:
				if ( setup_drive() ) {
					dsk_err();
					break;
				}
				if ( write_sector() ) {
					dsk_err();
					break;
				}
				break;
			case READ_TIME:
				req_tbl.CBI_CHR = rd_time();
				break;
			case SET_TIME:
				req_tbl.CBI_CHR = wr_time();
				break;
			case CONIN_REQ1:
				if ( rx_cnt ) req_tbl.CBI_CHR = (uint8_t)getch();
				else req_tbl.CBI_CHR = 0;
				break;
			case CONOUT_REQ1:
				c = req_tbl.CBI_CHR;
				req_tbl.CBI_CHR = out_chr(c);
				break;
			case STRIN_REQ:
				buf = tmp_buf[0];
				cnt = (uint16_t)get_str((char *)buf, req_tbl.CBI_CHR);
				req_tbl.CBI_CHR = (uint8_t)cnt;
				if (cnt) write_sram(get_physical_addr(req_tbl.data_dmah, req_tbl.data_dmal), buf, (unsigned int)cnt);
		}
		req_tbl.CREQ_COM = 0;	// clear cbios request
		// write end request to SRAM for 8088/V20
		write_sram(bioreq_cbuffadr, (uint8_t *)&req_tbl.CREQ_COM, RETURN_TBL);	// 2bytes
	}

}

