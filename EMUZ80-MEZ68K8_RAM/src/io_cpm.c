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
 */

#include "../src/mez68k8.h"
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

drive_t cpm_drives[] = {
    { 128 },
    { 128 },
    { 128 },
    { 128 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
    { 0 },
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
		printf("f_lseek(): ERROR %d\n\r", fres);
		return(-1);
	}
	return(0);
}

static int write_sector(void) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = cpm_drives[req_tbl.disk_drive].filep;
	
	if (seek_disk()) return(-1);

	// transfer write data from SRAM to the buffer
	read_sram(req_tbl.data_dma, disk_buf, SECTOR_SIZE);

	if (DEBUG_DISK_WRITE && DEBUG_DISK_VERBOSE && !(debug.disk_mask & (1 << req_tbl.disk_drive))) {
		util_hexdump_sum("buf: ", disk_buf, SECTOR_SIZE);
	}

	// write buffer to the DISK
	if ((fres = f_write(filep, disk_buf, SECTOR_SIZE, &n)) != FR_OK || n != SECTOR_SIZE) {
		printf("f_write(): ERROR res=%d, n=%d\n\r", fres, n);
		return(-1);
	}
	else if ((fres = f_sync(filep)) != FR_OK) {
		printf("f_sync(): ERROR %d\n\r", fres);
		return(-1);
	}
	return(0);
}

static int read_sector(void) {
	unsigned int n;
	FRESULT fres;
	FIL *filep = cpm_drives[req_tbl.disk_drive].filep;
	
	if (seek_disk()) return(-1);

	// read from the DISK
	if ((fres = f_read(filep, disk_buf, SECTOR_SIZE, &n)) != FR_OK || n != SECTOR_SIZE) {
		printf("f_read(): ERROR res=%d, n=%d\n\r", fres, n);
		return(-1);
	}
	else if (DEBUG_DISK_READ && DEBUG_DISK_VERBOSE && !(debug.disk_mask & (1 << req_tbl.disk_drive))) {
		util_hexdump_sum("buf: ", disk_buf, SECTOR_SIZE);
	}
	else {
		// transfer read data to SRAM
		write_sram(req_tbl.data_dma, disk_buf, SECTOR_SIZE);

		#ifdef MEM_DEBUG
		printf("f_read(): SRAM address(%08lx)\n\r", req_tbl.data_dma);
		read_sram(req_tbl.data_dma, disk_buf, SECTOR_SIZE);
		util_hexdump_sum("RAM: ", disk_buf, SECTOR_SIZE);
		#endif  // MEM_DEBUG
	}
	return(0);
}

static int setup_drive(void) {
	req_tbl.CBI_CHR = 0;		/* clear error status */
	if ( req_tbl.disk_drive >= NUM_DRIVES ) return( -1 );
	if ( cpm_drives[req_tbl.disk_drive].sectors == 0 ) return( -1 );	// not support disk
	if ( req_tbl.disk_sector > cpm_drives[req_tbl.disk_drive].sectors ) return( -1 ); // sector no. start with 1
	return( 0 );
}

static void dsk_err(void) {
	req_tbl.UNI_CHR = 1;
}

static void unimon_console(void) {

	uint8_t *buf;
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
			read_sram(req_tbl.STR_adr, buf, cnt);
			while( cnt ) {
				putch( *buf++);
				cnt--;
			}
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
//  ---- request command to PIC
// UREQ_COM = 1 ; CONIN  : return char in UNI_CHR
//          = 2 ; CONOUT : UNI_CHR = output char
//          = 3 ; CONST  : return status in UNI_CHR
//                       : ( 0: no key, 1 : key exist )
//          = 4 ; STROUT : string address = (PTRSAV, PTRSAV_SEG)
//          = 5 ; DISK READ
//          = 6 ; DISK WRITE
//          = 7 ; NMI LED OFF
//          = 0 ; request is done( return this flag from PIC )
//                return status is in UNI_CHR;
//debug
//printf("\r\nCOM(%02x), drive(%02x), track(%04x), sec(%04x), dma(%08lx)\r\n",
//   req_tbl.CREQ_COM, req_tbl.disk_drive, req_tbl.disk_track, req_tbl.disk_sector, req_tbl.data_dma);
//debug

void bus_master_operation(void) {
	uint8_t *buf;
	uint16_t cnt;

	// read request from MC68008
	read_sram(bioreq_ubuffadr, (uint8_t *)&req_tbl, (unsigned int)sizeof(cpm_hdr));

	if (req_tbl.UREQ_COM) {
		unimon_console();
		// write end request to SRAM for MC68008
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
				read_sram(req_tbl.data_dma, buf, cnt);
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
		}
		req_tbl.CREQ_COM = 0;	// clear cbios request
		// write end request to SRAM for MC68008
		write_sram(bioreq_cbuffadr, (uint8_t *)&req_tbl.CREQ_COM, RETURN_TBL);	// 2bytes
	}

}

