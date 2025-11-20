/*
 * Memory test
 *
 * Base source code is maked by @hanyazou
 *  https://twitter.com/hanyazou
 *
 * Redesigned by Akihito Honda(Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Date. 2025.8.11
 */

#include "mez88.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "../drivers/utils.h"

uint32_t mem_size = 0;

void mem_init()
{
    unsigned int i;
    uint32_t addr;

    // RAM check
    for (i = 0; i < TMP_BUF_SIZE; i += 2) {
        tmp_buf[0][i + 0] = 0xa5;
        tmp_buf[0][i + 1] = 0x5a;
    }
    for (addr = 0; addr < MAX_MEM_SIZE; addr += MEM_CHECK_UNIT) {
        printf("Memory 000000 - %06lXH\r", addr);
        tmp_buf[0][0] = addr & 0xff;
        tmp_buf[0][1] = (addr >>  8) & 0xff;
        tmp_buf[0][2] = (addr >> 16) & 0xff;

    	write_sram(addr, tmp_buf[0], TMP_BUF_SIZE);
        read_sram(addr, tmp_buf[1], TMP_BUF_SIZE);

    	if (memcmp(tmp_buf[0], tmp_buf[1], TMP_BUF_SIZE) != 0) {
            printf("\nMemory error at %06lXH\n\r", addr);
            util_addrdump("WR: ", addr, tmp_buf[0], TMP_BUF_SIZE);
            util_addrdump("RD: ", addr, tmp_buf[1], TMP_BUF_SIZE);
			while(1){};		// stop
            break;
        }
        if (addr == 0) continue;

    	read_sram(0, tmp_buf[1], TMP_BUF_SIZE);
        if (memcmp(tmp_buf[0], tmp_buf[1], TMP_BUF_SIZE) == 0) {
            // if the page at addr is the same as the first page,
			// then addr reachs end of memory
			printf("\nMemory wrap around.\n\r");
        	break;
        }
    }
	mem_size = addr;
	printf("Memory 000000 - %06lXH %d KB OK\r\n", addr-1, (int)(mem_size / 1024));
}
