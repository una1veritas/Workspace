/*
 * CP/M-86 and MS-DOS for MEZ88_47Q
 * This firmware is only for PICF47QXX.
 * This firmware can run CPM-86 or MS-DOS on CPU i8088/V20.
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
 *  Target: MEZ88_47Q_RAM 512KB Rev1.0
 *  Written by Akihito Honda (Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Date. 2024.4.20
 */

#ifndef __SUPERMEZ80_H__
#define __SUPERMEZ80_H__

#include "picconfig.h"
#include <xc.h>
#include <stdint.h>
#include "fatfs/ff.h"

//
// Configlations
//

#define CPM		1
#define MSDOS	2

#define P64 15.625

#define ENABLE_DISK_DEBUG

//#define NUM_FILES        6
//#define NUM_FILES        2
#define NUM_FILES        4

#define TMP_BUF_SIZE     256

#define MEM_CHECK_UNIT	TMP_BUF_SIZE * 16	// 4 KB
#define MAX_MEM_SIZE	0x00080000			// 512KB
#define bioreq_ubuffadr	0x100				// monitor request IO header address
#define bioreq_cbuffadr	0x106				// function request IO header address

//#define TIMER0_INITC	0x86e8	//Theoretically value
//#define TIMER0_INITCH	0x86
//#define TIMER0_INITCL	0xe8
#define TIMER0_INITC	0x87e1	//Actual value
#define TIMER0_INITCH	0x87
#define TIMER0_INITCL	0xe1
//
// Constant value definitions
//

#define CTL_Q 0x11

#define UNIMON_OFF		0x0000			// MONITOR
#define BASIC_OFF		0x0000
#define CPM68K_OFF		0x0000
#define CBIOS_OFF		0x6200

typedef struct {
	uint8_t  UREQ_COM;		// unimon CONIN/CONOUT request command
	uint8_t  UNI_CHR;		// charcter (CONIN/CONOUT) or number of strings
	uint32_t STR_adr;		// string address
	uint8_t  CREQ_COM;		// unimon CONIN/CONOUT request command
	uint8_t  CBI_CHR;		// charcter (CONIN/CONOUT) or number of strings
	uint8_t	 disk_drive;
	uint8_t	 dummy;			/* alignment word boundary */
	uint16_t disk_track;
	uint16_t disk_sector;
	uint32_t data_dma;
} cpm_hdr;

//
// Type definitions
//

// Address Bus
union address_bus_u {
    uint32_t w;             // 32 bits Address
    struct {
        uint8_t ll;        // Address L low
        uint8_t lh;        // Address L high
        uint8_t hl;        // Address H low
        uint8_t hh;        // Address H high
    };
};

union io_address {
	uint16_t adr;
	struct {
		uint8_t l8;
		uint8_t h8;
	};
};

typedef struct {
    unsigned int sectors;
    FIL *filep;
} drive_t;

typedef struct {
    uint8_t disk;
    uint8_t disk_read;
    uint8_t disk_write;
    uint8_t disk_verbose;
    uint16_t disk_mask;
} debug_t;

typedef struct {
    uint8_t *addr;
    uint16_t offs;
    unsigned int len;
} mem_region_t;

typedef struct {
	uint8_t  cmd_len;		// LENGTH OF THIS COMMAND
	uint8_t  unit;			// SUB UNIT SPECIFIER
	uint8_t  cmd;			// COMMAND CODE
	uint16_t status;		// STATUS
	uint8_t  reserve[8];	// RESERVE
	uint8_t  media;			// MEDIA DESCRIPTOR
	uint16_t trans_off;		// TRANSFER OFFSET
	uint16_t trans_seg;		// TRANSFER SEG
	uint16_t count;			// COUNT OF BLOCKS OR CHARACTERS
	uint16_t start;			// FIRST BLOCK TO TRANSFER
} iodat;

typedef struct {
	uint8_t  cmd_len;		// LENGTH OF THIS COMMAND
	uint8_t  unit;			// SUB UNIT SPECIFIER
	uint8_t  cmd;			// COMMAND CODE
	uint16_t status;		// STATUS
	uint8_t  reserve[8];	// RESERVE
	uint8_t  bpb1;			// number of support drives.
	uint16_t bpb2_off;		// DWORD transfer address.
	uint16_t bpb2_seg;
	uint16_t bpb3_off;		// DWORD pointer to BPB
	uint16_t bpb3_seg;
	uint8_t  bdev_no;		// block device No.
} CMDP;

typedef struct {
	uint8_t  UREQ_COM;		// unimon CONIN/CONOUT request command
	uint8_t  UNI_CHR;		// charcter (CONIN/CONOUT) or number of strings
	uint16_t STR_off;		// unimon string offset
	uint16_t STR_SEG;		// unimon string segment
	uint8_t  DREQ_COM;		// device request command
	uint8_t  DEV_RES;		// reserve
	uint16_t PTRSAV_off;	// request header offset
	uint16_t PTRSAV_SEG;	// request header segment
} PTRSAV;


typedef struct {
	uint8_t  jmp_ner[3];	// Jmp Near xxxx  for boot.
	uint8_t  mane_var[8];	// Name / Version of OS.
} DPB_HEAD;

typedef struct {
	DPB_HEAD reserve;
//-------  Start of Drive Parameter Block.
	uint16_t sec_size;		// Sector size in bytes.                  (dpb)
	uint8_t  alloc;			// Number of sectors per alloc. block.    (dpb)
	uint16_t res_sec;		// Reserved sectors.                      (dpb)
	uint8_t  fats;			// Number of FAT's.                       (dpb)
	uint16_t max_dir;		// Number of root directory entries.      (dpb)
	uint16_t sectors;		// Number of sectors per diskette.        (dpb)
	uint8_t  media_id;		// Media byte ID.                         (dpb)
	uint16_t fat_sec;		// Number of FAT Sectors.                 (dpb)
//-------  End of Drive Parameter Block.
	uint16_t sec_trk;		// Number of Sectors per track.
} DPB;

typedef struct {
	uint8_t  cmd_len;		// LENGTH OF THIS COMMAND
	uint8_t  unit;			// SUB UNIT SPECIFIER
	uint8_t  cmd;			// COMMAND CODE
	uint16_t status;		// STATUS
	uint8_t  reserve[8];	// RESERVE
	uint8_t  medias1;		//Media byte.
	uint8_t  medias2;		//Media status byte flag.
} MEDIAS;

typedef struct {
	uint8_t  cmd_len;		// LENGTH OF THIS COMMAND
	uint8_t  unit;			// SUB UNIT SPECIFIER
	uint8_t  cmd;			// COMMAND CODE
	uint16_t status;		// STATUS
	uint8_t  reserve[8];	// RESERVE
	uint8_t  media;			// MEDIA DESCRIPTOR
	uint16_t bpb2_off;		// DWORD transfer address.
	uint16_t bpb2_seg;
	uint16_t bpb3_off;		// DWORD pointer to BPB
	uint16_t bpb3_seg;
} BPB;

typedef struct {
	uint16_t TIM_DAYS;		//Number of days since 1-01-1980.
	uint8_t  TIM_MINS;		//Minutes.
	uint8_t  TIM_HRS;		//Hours.
	uint8_t  TIM_HSEC;		//Hundreths of a second.
	uint8_t  TIM_SECS;		//Seconds.
} TPB;

#define TIM20240101	16071	// 16071days from 1980

//I2C
//General Call Address
#define GeneralCallAddr	0
#define module_reset	0x06
#define module_flash	0x0e

//DS1307 Slave address << 1 + R/~W
#define DS1307			0b11010000	// support RTC module client address

//FT200XD Slave address << 1 + R/~W
#define FT200XD			0b01000100	// support USB I2C module client address

#define BUS_NOT_FREE	1
#define NACK_DETECT		2
#define NUM_DRIVES		16
//
// Global variables and function prototypes
//

extern uint8_t tmp_buf[2][TMP_BUF_SIZE];
extern debug_t debug;

extern void io_init(void);
extern void devio_init(void);

extern uint16_t chk_i2cdev(void);
extern void setup_clk_aux(void);
extern void cpmio_init(void);
extern void dosio_init(void);
extern drive_t cpm_drives[];
extern drive_t dos_drives[];
extern void mem_init(void);

extern void write_sram(uint32_t addr, uint8_t *buf, unsigned int len);
extern void read_sram(uint32_t addr, uint8_t *buf, unsigned int len);
extern void board_event_loop(void);
extern void bus_master_operation(void);

//
// debug macros
//
#ifdef ENABLE_DISK_DEBUG
#define DEBUG_DISK (debug.disk || debug.disk_read || debug.disk_write || debug.disk_verbose)
#define DEBUG_DISK_READ (debug.disk_read)
#define DEBUG_DISK_WRITE (debug.disk_write)
#define DEBUG_DISK_VERBOSE (debug.disk_verbose)
#else
#define DEBUG_DISK 0
#define DEBUG_READ 0
#define DEBUG_WRITE 0
#define DEBUG_DISK_VERBOSE 0
#endif

extern unsigned char rx_buf[];				//UART Rx ring buffer
extern unsigned int rx_wp, rx_rp, rx_cnt;

// AUX: input buffers
extern unsigned char ax_buf[];				//UART Rx ring buffer
extern unsigned int ax_wp, ax_rp, ax_cnt;
extern TPB tim_pb;					// TIME device parameter block

extern void putax(char c);
extern int getax(void);

extern void timer0_init(void);

extern void sys_init(void);
extern void start_M68K(void);
extern void setup_sd(void);

#endif  // __SUPERMEZ80_H__
