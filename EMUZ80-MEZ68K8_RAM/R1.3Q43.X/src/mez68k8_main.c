/*
 * Based on main.c by Tetsuya Suzuki 
 * and emuz80_z80ram.c by Satoshi Okue
 * PIC18F47Q43/PIC18F47Q83/PIC18F47Q84 ROM image uploader
 * and UART emulation firmware.
 * This single source file contains all code.
 *
 * Base source code of this firmware is maked by
 * @hanyazou (https://twitter.com/hanyazou) *
 *
 *  Target: MEZ68K8_RAM
 *  Written by Akihito Honda (Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Date. 2024.5.16
 */

#define INCLUDE_PIC_PRAGMA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "drivers/utils.h"
#include "drivers/utils.h"
#include "fatfs/ff.h"
#include "mez68k8.h"

static FATFS fs;
static DIR fsdir;
static FILINFO fileinfo;
static FIL files[NUM_FILES];
static FIL rom_fl;

uint8_t tmp_buf[2][TMP_BUF_SIZE];
#define BUF_SIZE TMP_BUF_SIZE * 2

debug_t debug = {
    0,  // disk
    0,  // disk_read
    0,  // disk_write
    0,  // disk_verbose
    0,  // disk_mask
};

#define BS	0x08
static char *cpmdir	= "CPMDISKS";
static char *unimon = "UMON_68K.BIN";
static char *basic = "BASIC68K.BIN";
static char *cbios = "CBIOS.BIN";
static char *cpm68k = "CPM68K.BIN";

static int disk_init(void);
static int chk_dsk(void);
static int load_program(uint8_t *buf, uint32_t load_adr);
static int open_dskimg(void);
static void setup_cpm(void);

static char *board_name = "MEZ68K8_RAM Firmware Rev1.3 on EMUZ80";

// main routine
void main(void)
{
	int c;
	int	selection;
	
	sys_init();
	devio_init();
	printf("Board: %s\r\n", board_name);

    mem_init();
    printf("mem_init() passed.\r\n");
    
    setup_sd();

    if (disk_init() < 0) while (1);

	GIE = 1;             // Global interrupt enable

	selection = 1;		// default : unimon68k
	printf("\r\nSelect(unimon68k = 1, basic68k = 2, CP/M-68K = 3) : ");
	while (1) {
		c = (uint8_t)getch();  // Wait for input char
		if ( c == '1' || c == '2' || c == '3' ) {
			putch((char)c);
			putch((char)BS);
			selection = c - (int)'0';
		}
		if ( c == 0x0d || c == 0x0a ) break;
	}
	printf("\n\r");

	c = chk_dsk();

	switch (selection) {
		case 1:		// unimon68k
			fileinfo.fname[0] = 0;		// set No directory
			c = load_program((uint8_t *)unimon, UNIMON_OFF);
			break;
		case 2:		// basic68k
			fileinfo.fname[0] = 0;		// set No directory
			c = load_program((uint8_t *)basic, BASIC_OFF);
			break;
		default:	// CP/M-68K
			if ( !c ) {
				printf("No CPMDISKS directory found.\r\n");
				while(1);
			}
			if ( open_dskimg() < 0 ) {
		        printf("No boot disk.\n\r");
				while(1);
			}
			setup_cpm();
			c = 0;
	}
	if ( c ) {
		printf("Program File Load Error.\r\n");
		while(1);
	}
	
	
	//
    // Start MC68008
    //
	printf("Use NCO1 %2.3fMHz\r\n",NCO1INC * 30.5175781 / 1000000);
    printf("\n\r");
	
    start_M68K();
	board_event_loop();
}

//
// load program from SD card
//
static int load_program(uint8_t *fname, uint32_t load_adr) {
	
	FRESULT		fr;
	void		*rdbuf;
	UINT		btr, br;
	uint16_t	cnt, size;
	uint32_t	adr;

	TCHAR	buf[30];

	rdbuf = (void *)&tmp_buf[0][0];		// program load work area(512byte)
	
	sprintf((char *)buf, "%s", fname);

	fr = f_open(&rom_fl, buf, FA_READ);
	if ( fr != FR_OK ) return((int)fr);

	adr = load_adr;
	cnt = size = (uint16_t)f_size(&rom_fl);				// get file size
	btr = BUF_SIZE;									// default 512byte
	while( cnt ) {
		fr = f_read(&rom_fl, rdbuf, btr, &br);
		if (fr == FR_OK) {
			write_sram(adr, (uint8_t *)rdbuf, (unsigned int)br);
			adr += (uint32_t)br;
			cnt -= (uint16_t)br;
			if (btr > (UINT)cnt) btr = (UINT)cnt;
		}
		else break;
	}
	if (fr == FR_OK) {
		printf("Load %s : Adr = %06lx, Size = %04x\r\n", fname, load_adr, size);
	}
	f_close(&rom_fl);
	return((int)fr);
}

//
// mount SD card
//
static int disk_init(void)
{
    if (f_mount(&fs, "0://", 1) != FR_OK) {
        printf("Failed to mount SD Card.\n\r");
        return -2;
    }

    return 0;
}

//
// check dsk
// 0  : No CPMDISKS directory
// 1  : CPMDISKS directory exist
//
static int chk_dsk(void)
{
    int selection;
    uint8_t c;

    //
    // Select disk image folder
    //
    if (f_opendir(&fsdir, "/")  != FR_OK) {
        printf("Failed to open SD Card.\n\r");
		while(1);
    }

	selection = 0;
	f_rewinddir(&fsdir);
	while (f_readdir(&fsdir, &fileinfo) == FR_OK && fileinfo.fname[0] != 0) {
		if (strcmp(fileinfo.fname, cpmdir) == 0) {
			selection = CPM;
			printf("Detect %s\n\r", fileinfo.fname);
			break;
		}
	}
	f_closedir(&fsdir);
	
	return(selection);
}

//
// Open disk images
//
static int open_dskimg(void) {
	
	int num_files;
    uint16_t drv;
	
	for (drv = num_files = 0; drv < NUM_DRIVES && num_files < NUM_FILES; drv++) {
        char drive_letter = (char)('A' + drv);
        char * const buf = (char *)tmp_buf[0];
        sprintf(buf, "%s/DRIVE%c.DSK", fileinfo.fname, drive_letter);
        if (f_open(&files[num_files], buf, FA_READ|FA_WRITE) == FR_OK) {
        	printf("Image file %s/DRIVE%c.DSK is assigned to drive %c\n\r",
                   fileinfo.fname, drive_letter, drive_letter);
			cpm_drives[drv].filep = &files[num_files];
			if (cpm_drives[0].filep == NULL) return -4;
        	num_files++;
        }
    }
    return 0;
}

static void setup_cpm(void) {
	
	const TCHAR	buf[30];
	int flg;

	cpmio_init();
	printf("\n\r");

	sprintf((char *)buf, "%s/%s", fileinfo.fname, cpm68k);
	flg = load_program((uint8_t *)buf, CPM68K_OFF);
	if (!flg) {
		sprintf((char *)buf, "%s/%s", fileinfo.fname, cbios);
		flg = load_program((uint8_t *)buf, CBIOS_OFF);
	}
	if ( flg ) {
		printf("Program File Load Error.\r\n");
		while(1);
	}
}
