/*
 * main.c
 *
 *  Created on: 2017/09/12
 *      Author: sin
 */
#include <stdio.h>
#include <stdlib.h>

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "types.h"
#include "uart.h"
#include "z80bus.h"
#include "sram.h"

int mem_check(uint32 addr_end) {
	const uint16 blocksize = (1<<13); // 8kb
	const uint16 datasize = 127;
	uint8 data[datasize];
	uint16 blockcount = 0;
	uint32 totalerrs = 0, errs;
	uint32 addr = 0;
	uint8 readout;

	sram_bus_init();
	sram_enable();

	srandom(addr_end^blocksize^datasize);
	for(uint16_t i = 0; i < datasize; i++) {
		data[i] = random() | 0x55;
	}

	for( addr = 0; addr <= addr_end; addr += blocksize) {
		errs = 0;

	    printf("%04lx -- %04lx w", addr, addr+blocksize-1);
	    for(uint16 i = 0; i < blocksize; i++) {
	      sram_write(addr+i, data[i % datasize]);
	    }

	    printf("/r");
	    for(uint16 i = 0; i < blocksize; i++) {
	      readout = sram_read(addr+i);
	      if ( readout != data[i % datasize] ) {
	    	  printf(" error @ %04lx [%02x/%02x]\r\n",addr+i,data[ i % datasize],readout);
	    	  errs++;
	      } else {
	    	  sram_write(addr+i, 0x00);
	      }
	    }
	    if ( errs ) {
	    	totalerrs += errs;
	    } else {
	    	blockcount++;
	    	printf(" ok");
	    }
	    printf("\r\n");
	}
    printf("\r\n");
    printf("Finished %lu bytes with %lu errors.\r\n", blockcount*(uint32_t)blocksize, totalerrs);

    sram_disable();

    return blockcount;
}

void start_OC1A(uint8 presc, uint16 top) {
  const uint8 MODE = 4;

  cli();

  TIMSK1 = 0;

  TCCR1A = 0;
  TCCR1B = 0;
  TCCR1C = 0;
  TCNT1 = 0;
  OCR1A = top - 1;

  TCCR1A |= (1 << COM1A0) | (MODE & 0x3);
  TCCR1B |= ((MODE >> 2 & 0x03) << 3) | ((presc&0x07) << CS10);
  TCCR1C |= (1 << FOC1A);

  sei();
}

void loop();

const uint16 mem_size = 1<<8;
uint8 mem[256] = {
		0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3,
		0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3,
		0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3,
		0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3,
		0xc3, 0x00, 0x00,
};

void mem_load(uint8 * mem, uint16 saddr, uint16 length) {
	sram_enable();

	for(uint16 i = 0; i < length; i++)
		sram_write(saddr+i,mem[i]);

	sram_disable();
}

uint16 addressbus() {
	return (((uint16) PINC)<<8 | PINA);
}

void databus_readmode() {
	DDRF = 0x00;
}

void databus_writemode() {
	DDRF = 0xff;
}

uint8 databus_read(void) {
	return PINF;
}

uint8 databus_write(uint8_t data) {
	return PORTF = data;
}

int main(void) {
	uart_init(57600);
	printf("\r\n\r\nHello there.\r\n");

	printf("start Z80 clock osc.\r\n");
	start_OC1A(5,800);
	_delay_ms(500);
	printf("busreq to z80 ");
	if ( z80_busreq() ) {
		printf("ok.\r\n");
	} else {
		printf("failed! stop.\r\n");
		while(1);
	}
	INPUTMODE(Z80_CLK_PORT, Z80_CLK);

	sram_bus_init();
//	mem_check(0x20000 - 1);
//	mem_load(mem, 0x000, mem_size);
	sram_bus_release();

	z80_bus_init();
	printf("z80 gained bus. reseting Z80.\r\n");
	z80_reset();

	for(;;)
		loop();
}

void loop() {
	uint8 data;
	uint16 addr;

	if ( !z80_m1rd() ) {
		databus_writemode();
		addr = addressbus();
		data = mem[addr & (mem_size-1)];
		databus_write(data);
		while ( !z80_rd());
		databus_readmode();
		printf("M1 R [%04x] %02x\r\n", addr, data );
	} else if ( !z80_rd() ) {
		databus_writemode();
		addr = addressbus();
		data = mem[addr & (mem_size-1)];
		databus_write(data);
		while ( !z80_rd() );
		databus_readmode();
		printf("   R [%04x] %02x\r\n", addr, data );
	} else if ( !z80_wr() ) {
		databus_readmode();
		addr = addressbus();
		data = databus_read();
		mem[addr] = data;
		while (!z80_wr());
		printf("   W [%04x] %02x\r\n", addr, data );
	} else if ( !z80_in() ) {
		databus_writemode();
		addr = addressbus();
		data = 0xff;
		databus_write(data);
		while ( !z80_in());
		databus_readmode();
		printf("   I [%04x] %02x\r\n", addr, data );
	} else if ( !z80_out() ) {
		databus_readmode();
		addr = addressbus();
		data = databus_read();
		while (!z80_out());
		printf("   O [%04x] %02x\r\n", addr, data );
	}
}
