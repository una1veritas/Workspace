/*
 * CP/M-86 and MS-DOS for MEZ86_RAM
 * This firmware is only for PICF47QXX.
 * This firmware can run CPM-86 or MS-DOS on CPU i8086.
 *
 * Based on main.c by Tetsuya Suzuki 
 * and emuz80_z80ram.c by Satoshi Okue
 * PIC18F47Q43/PIC18F47Q83/PIC18F47Q84 ROM image uploader
 * and UART emulation firmware.
 * This single source file contains all code.
 *
 * Base source code of this firmware is maked by
 * @hanyazou (https://twitter.com/hanyazou) *
 *
 *  Target: MEZ88_RAM 512KB
 *  Written by Akihito Honda (Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Date. 2025/8/10
 *  2025/9/3 Update for MS-DOS Ver3.10
 */

#define INCLUDE_PIC_PRAGMA
#include "mez88.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../drivers/utils.h"

static FATFS fs;
static FILINFO fileinfo;
static FIL files[NUM_FILES];
static FIL fl;
static UINT p_size;

uint16_t time_dev;	// 0:Timer0, 1:DS1307
uint8_t	nmi_sig;	// NMI request flag
uint8_t ctlq_ev;
int	terminate;		// cpu terminate flag

uint8_t tmp_buf[2][TMP_BUF_SIZE];

uint16_t int_vec;
uint16_t clk_fs;
static uint16_t unimon_seg;
static uint16_t unimon_off;
static uint16_t apli_seg;
static uint16_t apli_off;
static uint16_t cpm_seg;
static uint16_t cpm_off;
static uint16_t cbios_off;
uint16_t iosys_seg;
uint16_t iosys_off;
static uint16_t exec;

#define BUF_SIZE TMP_BUF_SIZE * 2
#define BS	0x08

debug_t debug = {
    0,  // disk
    0,  // disk_read
    0,  // disk_write
    0,  // disk_verbose
    0,  // disk_mask
};

/* define structure type */
typedef struct {
	const TCHAR *conf;
	uint16_t *val;
} sys_param;

const TCHAR *conf = "MEZ88.CFG";
static char *cpmdir   = "CPMDISKS";
static char *cbios = "CBIOS.BIN";
static char *ccp_bdos = "CCP_BDOS.BIN";

static char *msdosdir	= "DOSDISKS";
static char *io_sys = "IO.SYS";
static char *msdos_sys = "MSDOS.SYS";

static char *basic86 = "BASIC_86.BIN";
static char *ttbasic = "TT_BAS88.BIN";
static char *vtl = "VTL_S88.BIN";
static char *game = "GMI_S88.BIN";
static char *unimon_s = "UMON_S88.BIN";
static uint16_t start_vec[2];

#define num_param 11
static sys_param t_conf[num_param] = {
	{"CLK", &clk_fs},
	{"UNIMON_SEG", &unimon_seg},
	{"UNIMON_OFF", &unimon_off},
	{"APLI_SEG", &apli_seg},
	{"APLI_OFF", &apli_off},
	{"CPM_SEG", &cpm_seg},
	{"CPM_OFF", &cpm_off},
	{"CBIOS_OFF", &cbios_off},
	{"IOSYS_SEG", &iosys_seg},
	{"IOSYS_OFF", &iosys_off},
	{"EXEC", &exec}
};

static char *clkfs_msg[6] = {
	"4.92MHz duty 38.4%",
	"8MHz duty 37.5%",
	"9.14MHz duty 42.9%",
	"10.67MHz duty 50%",
	"12.8MHz duty 60%",
	"16MHz duty 50%"
};

static int load_config(void);
static int disk_init(void);
static int chk_dsk(char *);
static int setup_cpm(void);
static int setup_msdos(void);
static int open_dskimg(int);
static int load_program(uint8_t *, uint32_t);
static void set_tod(void);
static int get_line(char *, int);
static int load_apli(uint8_t *);

static char *board_name = "MEZ88_RAM Firmware Rev1.4";

// main routine
void main(void)
{
	uint16_t c, a;
	uint16_t selection;

	nmi_sig = 0;	// clear NMI request flag
	ctlq_ev = 0;	// clear CTL+Q event
	sys_init();
	reset_clk(5);	// init CLK = 5MHz
	setup_sd();

	printf("Board: %s\n\r", board_name);

	clr_uart_rx_buf();	// clear rx buffer and enable rx interrupt
	uart5_init();
	timer0_init();		// clear timer value
	setup_I2C();
	time_dev = chk_i2cdev();	//0:timer 1:I2C(DS1307)
    mem_init();
    if (disk_init() < 0) while (1);

	if( load_config() < 0) while (1);
	switch (clk_fs) {
		case 8:
		case 9:
		case 10:
			c = clk_fs - 7;
			reset_clk(clk_fs);
			break;
		case 12:
			c = 4;
			reset_clk(clk_fs);
			break;
		case 16:
			c = 5;
			reset_clk(clk_fs);
			break;
		default:
			c = 0;
	}
	printf("CPU clock %s\r\n", clkfs_msg[c]);

	enable_int();
	
sel_list:
	start_vec[0] = 0;
	start_vec[1] = 0;
	selection = 1;
	terminate = 0;
	printf("\n\rSelect:\n\r");
	printf("0:TOD(Time Of Day)\n\r");
	printf("1:Universal Monitor\n\r");
	printf("2:8086 NASCOM BASIC\n\r");
	printf("3:Toyoshiki Tiny Basic\n\r");
	printf("4:VTL-C\n\r");
	printf("5:GAME-C Interpreter\n\r");
	printf("6:CP/M-86\n\r");
	printf("7:MS-DOS\n\r");
	printf("? ");
	while (1) {
		c = (uint16_t)getch();  // Wait for input char
		a = c - (uint16_t)'0';
		if ( a >= 0 && a <= 7 ) {
			putch((char)c);
			putch((char)BS);
			selection = c - (uint16_t)'0';
		}
		if ( c == 0x0d || c == 0x0a ) break;
	}
	printf("\n\r");

	// set default bus master operation function
	bus_master_operation = cpm_bus_master_operation;
	c = 0;
	switch (selection) {
		case 0:		// set Time of day
			set_tod();
			goto sel_list;
		case 1:		// Universal Monitor
			c = (uint16_t)load_program((uint8_t *)unimon_s, get_physical_addr(unimon_seg, unimon_off));
			break;
		case 2:		// 8086 NASCOM BASIC
			c = (uint16_t)load_apli((uint8_t *)basic86);
			break;
		case 3:		// Toyoshiki Tiny Basic
			c = (uint16_t)load_apli((uint8_t *)ttbasic);
			break;
		case 4:		// VTL-C
			c = (uint16_t)load_apli((uint8_t *)vtl);
			break;
		case 5:		// GAME-C Interpreter
			c = (uint16_t)load_apli((uint8_t *)game);
			break;
		case 6:		// CP/M-86
			if (chk_dsk(cpmdir)) goto sel_list;
			if ( open_dskimg(CPM) < 0 ) {
		        printf("No drive A found.\n\r");
				goto sel_list;
			}
			if ( setup_cpm() ) goto sel_list;
			break;
		default:	// MS-DOS
			if (chk_dsk(msdosdir)) goto sel_list;
			if ( open_dskimg(MSDOS) < 0 ) {
		        printf("No drive A found.\n\r");
				goto sel_list;
			}
			setup_dpb();
			if ( setup_msdos() ) goto sel_list;
	}
	if ( c ) {
		printf("Program File Load Error.\r\n");
		goto sel_list;
	}
	
	printf("\n\r");

	//
    // Start i8086
    //
	start_i88();
	board_event_loop();
	goto sel_list;
}

static uint8_t del_space(char *bytes) {
	uint8_t pos = 0;
	uint8_t i = 0;
	char c;
	
	while( (c = bytes[i++]) ) {
		if (c == '\r' || c == '\n' || c == ' ') {
			continue;
		}
		bytes[pos++] = c;
	}
	bytes[pos] = c;		// save NULL code
	return pos;
}

static int load_config(void)
{
	FRESULT	fr;
	char *buf, *a;
	uint16_t cnt, size;
	uint16_t adr;
	int i;
	TCHAR *str;
	
	str = (TCHAR *)&tmp_buf[0][0];
	
	printf("Load %s\r\n",(const char *)conf);
	
	fr = f_open(&fl, conf, FA_READ);
	if ( fr != FR_OK ) {
		printf("%s not found..\r\n", conf);
		return -1;
	}

	while ( f_gets(str, 80, &fl) ) {	// max 80 characters

		// delete space
		del_space(str);

		if (str[0] == ';' || str[0] == 0 ) continue;

		// search keywoard
		for( i=0; i<num_param; i++ ) {
			if ( !strstr(str, t_conf[i].conf )) continue;
			if(str[strlen(t_conf[i].conf)] != '=') continue;

			// get value
			buf = &str[strlen(t_conf[i].conf)+1];
			*(t_conf[i].val) = (uint16_t)strtol((const char *)buf, &a, 0);
		}
	}

	if (!f_eof(&fl)) {
		printf("File read error!\r\n");
		f_close( &fl );
		return -1;
	}
	f_close( &fl );
	if ( exec ) printf("Set debug mode.\r\n");
	printf("CLK = %d\r\n", clk_fs);
	
	return 0;
}

// load standalone program
static int load_apli(uint8_t *flname) {
	int c;
	uint32_t adr32;
	
	if ( !exec ) {
		start_vec[0] = apli_off;
		start_vec[1] = apli_seg;
	}
	adr32 = get_physical_addr(unimon_seg, unimon_off);
	c = load_program((uint8_t *)unimon_s, adr32);
	if ( !c ) {
		c = load_program(flname, get_physical_addr(apli_seg, apli_off));
		if ( !c ) write_sram(adr32, (uint8_t *)start_vec, sizeof(start_vec));
	}
	return c;
}

// check dsk
//
static int chk_dsk(char *dir)
{
    int sig;
    uint8_t c;
	DIR fsdir;

    //
    // Select disk image folder
    //
    if (f_opendir(&fsdir, "/")  != FR_OK) {
        printf("Failed to open SD Card.\n\r");
		return 1;	// error return
    }

	sig = 0;
	f_rewinddir(&fsdir);
	while (f_readdir(&fsdir, &fileinfo) == FR_OK && fileinfo.fname[0] != 0) {
		if (strcmp(fileinfo.fname, dir) == 0) {
			sig = 1;
			printf("Detect %s\n\r", fileinfo.fname);
			break;
		}
	}
	f_closedir(&fsdir);
	
	if ( !sig ) {
		printf("No %s directory found.\r\n", dir);
		return 1;
	}
	return 0;
}

static int setup_msdos(void) {

	const TCHAR	buf[30];
	int flg;
	uint16_t sec_size;
	uint32_t adr32;

	// set dos operation 
	bus_master_operation = dos_bus_master_operation;

	dosio_init();
	setup_clk_dev();
	printf("Setup UART5 for AUX device.\n\r");

	sec_size = dos_sec_size();
	flg = load_program((uint8_t *)unimon_s, get_physical_addr(unimon_seg, unimon_off));
	if (!flg) {
		sprintf((char *)buf, "%s/%s", fileinfo.fname, io_sys);
		adr32 = get_physical_addr(iosys_seg, iosys_off);
		flg = load_program((uint8_t *)buf, adr32);
		if (!flg) {
			adr32 += ((((uint32_t)p_size+seg_bound) / seg_bound)+1)*seg_bound;
			sprintf((char *)buf, "%s/%s", fileinfo.fname, msdos_sys);
			flg = load_program((uint8_t *)buf, adr32);
		}
	}
	if ( flg ) {
		printf("Program File Load Error.\r\n");
		return 1;
	}

	/* copy Disk Parameter Block to physical memory area */
	copy_dpb();

	if ( !exec ) {
		start_vec[0] = iosys_off;
		start_vec[1] = iosys_seg;
	}
	adr32 = get_physical_addr(unimon_seg, unimon_off);
	write_sram(adr32, (uint8_t *)start_vec, sizeof(start_vec));
	return 0;
}


static int setup_cpm(void) {
	
	const TCHAR	buf[30];
	int flg;

	// set cpm operation
//	bus_master_operation = cpm_bus_master_operation;

	cpmio_init();
	printf("\n\r");

	flg = load_program((uint8_t *)unimon_s, get_physical_addr(unimon_seg, unimon_off));
	if (!flg) {
		sprintf((char *)buf, "%s/%s", fileinfo.fname, cbios);
		flg = load_program((uint8_t *)buf, get_physical_addr(cpm_seg, cbios_off));
		if (!flg) {
			sprintf((char *)buf, "%s/%s", fileinfo.fname, ccp_bdos);
			flg = load_program((uint8_t *)buf, get_physical_addr(cpm_seg, cpm_off));
		}
	}
	if ( flg ) {
		printf("Program File Load Error.\r\n");
		return 1;
	}
	if ( !exec ) {
		start_vec[0] = cbios_off;
		start_vec[1] = cpm_seg;
	}
	write_sram(get_physical_addr(unimon_seg, unimon_off), (uint8_t *)start_vec, sizeof(start_vec));

	return 0;
}

//
// load program from SD card
//
static int load_program(uint8_t *fname, uint32_t load_adr) {
	
	FRESULT		fr;
	void		*rdbuf;
	UINT		btr, br, cnt;
	uint32_t	adr;

	TCHAR	buf[30];
//	FIL fl;

	rdbuf = (void *)&tmp_buf[0][0];		// program load work area(512byte)
	
	sprintf((char *)buf, "%s", fname);

	fr = f_open(&fl, buf, FA_READ);
	if ( fr != FR_OK ) return((int)fr);

	adr = load_adr;
	cnt = p_size = (UINT)f_size(&fl);				// get file size
	btr = BUF_SIZE;									// default 512byte
	while( cnt ) {
		fr = f_read(&fl, rdbuf, btr, &br);
		if (fr == FR_OK) {
			write_sram(adr, (uint8_t *)rdbuf, (unsigned int)br);
			adr += (uint32_t)br;
			cnt -= br;
			if (btr > cnt) btr = cnt;
		}
		else break;
	}
	if (fr == FR_OK) {
		printf("Load %s : Adr = %06lx, Size = %04x\r\n", fname, load_adr, p_size);
	}
	f_close(&fl);
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
// Open disk images
//
static int open_dskimg(int os) {
	
	int num_files;
    uint16_t drv;
	
	for (drv = num_files = 0; drv < NUM_DRIVES && num_files < NUM_FILES; drv++) {
        char drive_letter = (char)('A' + drv);
        char * const buf = (char *)tmp_buf[0];
        sprintf(buf, "%s/DRIVE%c.DSK", fileinfo.fname, drive_letter);
        if (f_open(&files[num_files], buf, FA_READ|FA_WRITE) == FR_OK) {
        	printf("Image file %s/DRIVE%c.DSK is assigned to drive %c\n\r",
                   fileinfo.fname, drive_letter, drive_letter);
        	if (os == CPM) {
	        	cpm_drives[drv].filep = &files[num_files];
				if (cpm_drives[0].filep == NULL) return -4;
        	}
        	else {
        		filep_t[drv] = &files[num_files];
				if (filep_t[0] == NULL) return -4;
        	}
        	num_files++;
        }
    }
    return 0;
}

static uint16_t get_dn( uint8_t *str, uint16_t *cnt ) {
	uint16_t n, er;
	uint8_t s;

	er = 0xffff;	// error flag
	n = *cnt = 0;
	while( *str ) {
		s = *str++;
		*cnt += 1;
		if ( s < (uint8_t)'0' || s > (uint8_t)'9' ) break;
		n = n*10+(uint16_t)(s-(uint8_t)'0');
		er = 0;
	}
	if ( er == 0xffff ) {
		n = er;
		return n;
	}

	// skip non value char or detect delimiter(0)
	while ( *str < (uint8_t)'0' || *str > (uint8_t)'9' ) {
		if ( *str == 0 ) break;
		str++;
		*cnt += 1;
	}
	return n;
}

static int get_todval( uint8_t *str, uint16_t *yh, uint16_t *mm, uint16_t *ds ) {
	uint16_t n, cnt;
	
	n = get_dn( str, &cnt );
	if ( n == 0xffff ) return -1;
	*yh = n;
	str += cnt;
	
	n = get_dn( str, &cnt );
	if ( n == 0xffff ) return -1;
	*mm = n;
	str += cnt;
	
	n = get_dn( str, &cnt );
	if ( n == 0xffff ) return -1;
	*ds = n;
	
	return 0;
}

//
// set time of day
//   time_dev=0:timer
//   time_dev=1:I2C(DS1307)
//
// DS1307 format
// rtc[0] : seconds (BCD) 00-59
// rtc[1] : minuts  (BCD) 00-59
// rtc[2] : hours   (BCD) 00-23 (or 1-12 +AM/PM)
// rtc[3] : day     (BCD) week day 01-07
// rtc[4] : date    (BCD) 01-31
// rtc[5] : month   (BCD) 01-12
// rtc[6] : year    (BCD) 00-99 : range : (19)80-(19)99, (20)00-(20)79

static void set_tod(void) {

	uint8_t *s, commit;
	uint16_t year, month, date;
	uint16_t yh, mm, ds;

	commit = 0;
	// Set date
	// convert number of days to year, month and date
	cnv_ymd(tim_pb.TIM_DAYS, &year, &month, &date );
	if (year >= 80) year += 1900;
	else year += 2000;

re_inp1:
	printf("Date %04d/%02d/%02d = ", year, month, date);
	s = &tmp_buf[0][0];
	if ( get_line((char *)s, 80) != 0 ) {
		if ( get_todval( s, &yh, &mm, &ds ) ) {
			printf("\r\n");
			goto re_inp1;
		}
		if (yh < 1980 || yh > 2079 ) {
			printf("\r\n");
			goto re_inp1;
		}
		if ( mm > 12 || mm == 0 ) {
			printf("\r\n");
			goto re_inp1;
		}
		if ( mm == 2 && ds == 29 ) {
			if ( chk_leap(yh) ) goto skip_ds;
			printf("\r\n");
			goto re_inp1;
		}
		if ( ds > mtod[mm-1] || ds == 0) {
			printf("\r\n");
			goto re_inp1;
		}
skip_ds:
		year = yh;
		month = mm;
		date = ds;
		commit = 1;
	}
	// set time
	yh = tim_pb.TIM_HRS;
	mm = tim_pb.TIM_MINS;
	ds = tim_pb.TIM_SECS;
	
re_inp2:
	printf("Time %02d:%02d:%02d = ", tim_pb.TIM_HRS, tim_pb.TIM_MINS, tim_pb.TIM_SECS);
	s = &tmp_buf[0][0];
	if ( get_line((char *)s, 80) != 0 ) {
		if ( get_todval( s, &yh, &mm, &ds ) ) {
			printf("\r\n");
			goto re_inp2;
		}
		if ( yh  > 23 ) {
			printf("\r\n");
			goto re_inp2;
		}
		if ( mm > 59 ) {
			printf("\r\n");
			goto re_inp2;
		}
		if ( ds > 59 ) {
			printf("\r\n");
			goto re_inp2;
		}
		TMR0IE = 0;			// disable timer0 interrupt
		tim_pb.TIM_HRS = (uint8_t)yh;
		tim_pb.TIM_MINS = (uint8_t)mm;
		tim_pb.TIM_SECS = (uint8_t)ds;
		datcnv_tim_rtc();
		TMR0IE = 1;			// Enable timer0 interrupt
		commit = 1;
	}
	if ( commit ) {
		// convert year, month and date to number of days from 1980
		tim_pb.TIM_DAYS = days_from_1980(year, month, date);
		if ( time_dev ) {		// DS1307 ready!
			datcnv_tim_rtc();	// convert time data to DS1307 format
			// write to RTC
			if (write_I2C(DS1307, 0, 7, &rtc[0] ) == 0xFF) {
				printf("DS1307 Wite I2C error!\r\n");
				return;
			}
		}
		printf("\r\nSet Date(%04d/%02d/%02d)\r\n", year, month, date);
		printf("Set Time(%02d:%02d:%02d)\r\n", yh, mm, ds);
	}
}

static int get_line(char *s, int length) {
	char n;
	int c;
	
	for (c=0;;) {
		n = (char)getch();
		if ( n == BS ) {
			if ( c > 0) {
				putch(BS);
				putch(' ');
				putch(BS);
				c--;
				s--;
			}
			continue;
		}
		if ( n == 0x0d || n == 0x0a ) {
			*s = 0x00;
			printf("\r\n");
			return c;
		}
		if ( c <= length-1 ) {
			putch(n);
			if ( n >='a' && n <='z' ) n -= 0x20;		// lower to upper
			*s++ = n;
			c++;
		}
	}
	return c;
}
