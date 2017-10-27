/*
 * main.c
 *
 *  Created on: 2017/09/12
 *      Author: sin
 */
#include <stdio.h>
#include <stdlib.h>

#include <avr/io.h>
#include <util/delay.h>

#include "uart.h"
#include "sram.h"


uint8_t crc8(uint8_t crc, uint8_t inbyte) {
    uint8_t j;
    uint8_t mix;

    for (j = 0; j < 8; j++) {
        mix = (crc ^ inbyte) & 0x01;
        crc >>= 1;
        if (mix) crc ^= 0x8C;
        inbyte >>= 1;
    }
    return crc;
}

const uint16_t blksize = 31;

int main(void) {
	uint8_t mem[blksize];

	uart_init(38400);
	printf("\nHello.\n");
	sram_init();

	uint8_t val;
	uint32_t passedaddr = 0;
	uint32_t errcount = 0;

	uart_puts("sample: \n");
	for(uint16_t i = 0; i < blksize; i++) {
		val = rand() & 0xff;
		mem[i] = val;
		printf("%2x ", mem[i]);
	}
	uart_puts("\n");
	uart_puts("bank 0\n");
	uart_puts("writing...\n");
	for(uint32_t addr = 0; addr < 0x20000; addr ++) {
		sram_bank(addr>>16);
		sram_write(addr, mem[addr % blksize]);
		if ( passedaddr < addr+blksize ) {
			printf("%05lx, ", passedaddr);
			passedaddr += 0x1000;
		}
		//_delay_ms(10);
	}
	uart_puts("\nreading...\n");
	uint16_t newerr = 0;
	passedaddr = 0;
	for(uint32_t addr = 0; addr < 0x20000; addr ++) {
		sram_bank(addr>>16);
		val = sram_read(addr);
		if (mem[addr % blksize] != val) {
			errcount++;
		}
		if ( passedaddr < addr+blksize ) {
			printf("%05lx, ", passedaddr);
			passedaddr += 0x1000;
		}
		if ( newerr > 0 ) {
			uart_puts("\n");
			printf("%05lx: ", addr);
			printf("%02x ",mem[addr % blksize]);
			val = sram_read(addr);
			printf("%02x ", val);
			uart_puts("\n");
			newerr = 0;
		}
		//_delay_ms(200);
	}

	uart_puts("\n");
	uart_puts("total errcount = ");
	uart_putnum_u16(errcount, 5);
	uart_puts("\n");

	while(1);
}
