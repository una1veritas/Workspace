/*
 * UART, disk I/O and monitor firmware for SuperMEZ80-SPI
 *
 * Based on main.c by Tetsuya Suzuki and emuz80_z80ram.c by Satoshi Okue
 * Modified by @hanyazou https://twitter.com/hanyazou
 */
/*!
 * PIC18F47Q43/PIC18F47Q83/PIC18F47Q84 ROM image uploader and UART emulation firmware
 * This single source file contains all code
 *
 * Target: EMUZ80 with Z80+RAM
 * Compiler: MPLAB XC8 v2.40
 *
 * Modified by Satoshi Okue https://twitter.com/S_Okue
 * Version 0.1 2022/11/15
 */

/*
    PIC18F47Q43 ROM RAM and UART emulation firmware
    This single source file contains all code

    Target: EMUZ80 - The computer with only Z80 and PIC18F47Q43
    Compiler: MPLAB XC8 v2.36
    Written by Tetsuya Suzuki
*/

#define INCLUDE_PIC_PRAGMA
#include <supermez80.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utils.h>
#include <assert.h>

debug_t debug = {
    0,  // disk
    0,  // disk_read
    0,  // disk_write
    0,  // disk_verbose
    0,  // disk_mask
};

// global variable which is handled by board dependent stuff
int turn_on_io_led = 0;

const uint8_t rom[] = {
// Initial program loader at 0x0000
#ifdef CPM_MMU_EXERCISE
#include "mmu_exercise.inc"
#else
#include "ipl.inc"
#endif
};

void bus_master(int enable);
void sys_init(void);
int disk_init(void);
int menu_select(void);
void start_z80(void);

// main routine
void main(void)
{
    sys_init();
    printf("Board: %s\n\r", board_name());
    if (disk_init() < 0)
        while (1);
    io_init();
    mem_init();
    mon_init();

    U3RXIE = 1;          // Receiver interrupt enable
    GIE = 1;             // Global interrupt enable

    //
    // Transfer ROM image to the SRAM
    //
#if defined(CPM_MMU_EXERCISE)
    dma_write_to_sram(0x00000, rom, sizeof(rom));
#else
    if (board_ipl) {
        dma_write_to_sram(0x00000, board_ipl, board_ipl_size);
    } else {
        dma_write_to_sram(0x00000, rom, sizeof(rom));
    }
    if (menu_select() < 0)
        while (1);
#endif  // !CPM_MMU_EXERCISE

    //
    // Start Z80
    //
    if (NCO1EN) {
        unsigned long khz = ((uint32_t) NCO1INC * 61 / 2) / 1000;
        if (1000 < khz) {
            printf("NCO1: %.2f MHz\n\r", (double)khz / 1000.0);
        } else {
            printf("NCO1: %.2f KHz\n\r", (double)khz);
        }
    }
    printf("\n\r");
    start_z80();

    while(1) {
        // Wait for IO access
        board_wait_io_event();
        io_handle();
        timer_run();
    }
}

void bus_master(int enable)
{
    board_bus_master(enable);
}

static FIL files[NUM_FILES];
static uint16_t file_available = (1 << NUM_FILES) - 1;
FIL *get_file(void)
{
    for (int i = 0; i < NUM_FILES; i++) {
        if (file_available & (1L << i)) {
            file_available &= ~(1L << i);
            // printf("%s: allocate %d, available=%04x\n\r", __func__, i, file_available);
            return &files[i];
        }
    }

    return NULL;
}

void put_file(FIL *file)
{
    for (int i = 0; i < NUM_FILES; i++) {
        if (file == &files[i]) {
            assert(!(file_available & (1L << i)));
            file_available |= (1L << i);
            return ;
        }
    }
}

void sys_init()
{
    static uint8_t memory_pool[512];
    board_init();
    board_sys_init();
    util_memalloc_init(memory_pool, sizeof(memory_pool));
}

int disk_init(void)
{
    static FATFS fs;
    board_disk_init();
    if (f_mount(&fs, "0://", 1) != FR_OK) {
        printf("Failed to mount SD Card.\n\r");
        return -2;
    }

    return 0;
}

int menu_select(void)
{
    int i;
    unsigned int drive;
    DIR fsdir;
    FILINFO fileinfo;

    //
    // Select disk image folder
    //
    if (f_opendir(&fsdir, "/")  != FR_OK) {
        printf("Failed to open SD Card..\n\r");
        return -3;
    }
 restart:
    i = 0;
    int selection = -1;
    int preferred = -1;
    f_rewinddir(&fsdir);
    if (is_board_disk_name_available() && board_disk_name()[0]) {
        printf("Preferred disk image: %s\n\r", board_disk_name());
    }
    while (f_readdir(&fsdir, &fileinfo) == FR_OK && fileinfo.fname[0] != 0) {
        if (strncmp(fileinfo.fname, "CPMDISKS", 8) == 0 ||
            strncmp(fileinfo.fname, "CPMDIS~", 7) == 0) {
            printf("%d: %s\n\r", i, fileinfo.fname);
            if (is_board_disk_name_available() && board_disk_name()[0] &&
                strncmp(fileinfo.fname, "CPMDISKS.", 9) == 0 &&
                strncmp(fileinfo.fname + 9, board_disk_name(), 3) == 0) {
                preferred = i;
            }
            if (strcmp(fileinfo.fname, "CPMDISKS") == 0) {
                selection = i;
            }
            i++;
        }
    }
    if (0 <= preferred) {
        selection = preferred;
    }
    printf("M: Monitor prompt\n\r");
    if (1 < i) {
        if (0 <= selection) {
            printf("Select[%d]: ", selection);
        } else {
            printf("Select: ");
        }
        while (1) {
            uint8_t c = (uint8_t)getch_buffered();  // Wait for input char
            if ('0' <= c && c <= '9' && c - '0' < i) {
                selection = c - '0';
                break;
            }
            if (c == 'm' || c == 'M') {
                printf("M\n\r");
                while (mon_prompt() != MON_CMD_EXIT);
                goto restart;
            }
            if ((c == 0x0d || c == 0x0a) && 0 <= selection)
                break;
        }
        printf("%d\n\r", selection);
        f_rewinddir(&fsdir);
        i = 0;
        while (f_readdir(&fsdir, &fileinfo) == FR_OK && fileinfo.fname[0] != 0) {
            if (strncmp(fileinfo.fname, "CPMDISKS", 8) == 0 ||
                strncmp(fileinfo.fname, "CPMDIS~", 7) == 0) {
                if (selection == i)
                    break;
                i++;
            }
        }
        printf("%s is selected.\n\r", fileinfo.fname);
    } else {
        strcpy(fileinfo.fname, "CPMDISKS");
    }
    f_closedir(&fsdir);

    //
    // Open disk images
    //
    char *buf = util_memalloc(26);
    for (drive = 0; drive < num_drives; drive++) {
        char drive_letter = (char)('A' + drive);
        sprintf(buf, "%s/DRIVE%c.DSK", fileinfo.fname, drive_letter);
        if (f_stat(buf, NULL) != FR_OK) {
            sprintf(buf, "CPMDISKS.CMN/DRIVE%c.DSK", drive_letter);
            if (f_stat(buf, NULL) != FR_OK) {
                continue;
            }
        }
        FIL *filep = get_file();
        if (filep == NULL) {
            printf("Too many files\n\r");
            break;
        }
        if (f_open(filep, buf, FA_READ|FA_WRITE) == FR_OK) {
            printf("Image file %s is assigned to drive %c\n\r", buf, drive_letter);
            drives[drive].filep = filep;
        }
    }
    util_memfree(buf);
    if (drives[0].filep == NULL) {
        printf("No boot disk.\n\r");
        return -4;
    }

    return 0;
}

void start_z80(void)
{
    board_start_z80();
}
