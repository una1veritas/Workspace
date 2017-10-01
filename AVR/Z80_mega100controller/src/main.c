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

#include "types.h"
#include "uart.h"
#include "z80bus.h"
#include "sram.h"

int mem_check(uint32 addr_end) {
	//const char text[] = "If I speak in the tongues of men or of angels, but do not have love, I am only a resounding gong or a clanging cymbal. If I have the gift of prophecy and can fathom all mysteries and all knowledge, and if I have a faith that can move mountains, but do not have love, I am nothing. If I give all I possess to the poor and give over my body to hardship that I may boast, but do not have love, I gain nothing.\n\nLove is patient, love is kind. It does not envy, it does not boast, it is not proud. It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs. Love does not delight in evil but rejoices with the truth. It always protects, always trusts, always hopes, always perseveres.\n\nLove never fails. But where there are prophecies, they will cease; where there are tongues, they will be stilled; where there is knowledge, it will pass away. For we know in part and we prophesy in part, but when completeness comes, what is in part disappears. When I was a child, I talked like a child, I thought like a child, I reasoned like a child. When I became a man, I put the ways of childhood behind me. For now we see only a reflection as in a mirror; then we shall see face to face. Now I know in part; then I shall know fully, even as I am fully known.\n\nAnd now these three remain: faith, hope and love. But the greatest of these is love.\n";
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
	    	  printf(" error @ %04lx [%02x/%02x]\n",addr+i,data[ i % datasize],readout);
	    	  errs++;
	      } else {
	    	  sram_write(addr+i, 0x00);
	      }
	    }
	    if ( errs ) {
	    	totalerrs += errs;
	    } else {
	    	blockcount++;
	    }
	    printf("\n");
	}
    printf("\n");
    printf("Finished %lu bytes with %lu errors.\n", blockcount*(uint32_t)blocksize, totalerrs);

    sram_disable();

    return blockcount;
}

void start_OC1A(uint8 presc, uint16 top) {
  const uint8 MODE = 4;
//  noInterrupts();
  TIMSK1 = 0;

  TCCR1A = 0;
  TCCR1B = 0;
  TCCR1C = 0;
  TCNT1 = 0;
  OCR1A = top - 1;

  TCCR1A |= (1 << COM1A0) | (MODE & 0x3);
  TCCR1B |= ((MODE >> 2 & 0x03) << 3) | ((presc&0x07) << CS10);
  TCCR1C |= (1 << FOC1A);

//  interrupts();
}

void loop();

uint8 mem[] = {
		0x3e, 0x0e, 0x32, 0x0e, 0x00, 0x3a, 0x0e, 0x00,
		0xd3, 0x01, 0xc3, 0x05, 0x00, 0x76, 0x0e, 0x48,
		0x65, 0x6c, 0x6c, 0x6f, 0x2e, 0x0d, 0x00,
};
const uint16 mem_size = sizeof mem;

void mem_load(uint8 * mem, uint16 saddr, uint16 length) {
	sram_enable();

	for(uint16 i = 0; i < length; i++)
		sram_write(saddr+i,mem[i]);

	sram_disable();
}

uint16 bus_address() {
	return (((uint16) PINC)<<8 | PINA);
}
uint8 bus_data() {
	return PINF;
}

int main(void) {
	uart_init(57600);
	printf("\n\nHello there.\n");

	printf("start Z80 clock osc.\n");
	start_OC1A(5,400);

	printf("reseting Z80.\n");
	z80_reset();
	printf("busreq to z80 ");
	if ( z80_busreq() ) {
		printf("ok.\n");
	} else {
		printf("failed! stop.\n");
		while(1);
	}

	sram_bus_init();
	mem_check(0x20000 - 1);
	mem_load(mem, 0x000, mem_size);

	busmode_z80();
	z80_busfree();
	printf("z80 gained bus. reseting Z80.\n");
	z80_reset();

	for(;;)
		loop();
}

void loop() {
	if ( !z80_m1() ) {
		printf("M1 R [%04x] %02x\n", bus_address(), bus_data() );
		while ( !z80_rd());
	} else if ( !z80_rd() ) {
		printf("   R [%04x] %02x\n", bus_address(), bus_data() );
		while ( !z80_rd());
	} else if ( !z80_wr() ) {
		printf("   W [%04x] %02x\n", bus_address(), bus_data() );
		while (!z80_wr());
	} else if ( !z80_in() ) {
		printf("   I [%04x] %02x\n", bus_address(), bus_data() );
		while (!z80_in());
	} else if ( !z80_out() ) {
		printf("   O [%04x] %02x\n", bus_address(), bus_data() );
		while (!z80_out());
	}
}
