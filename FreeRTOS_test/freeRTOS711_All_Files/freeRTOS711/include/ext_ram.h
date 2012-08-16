/*
 * ext_ram.h
 *
 *  Created on: 17/02/2012
 *      Author: Phillip
 */

#ifndef EXT_RAM_H_
#define EXT_RAM_H_


#ifdef __cplusplus
extern "C" {
#endif


#include <stdbool.h>


// AVR include files.
#include <avr/io.h>

#include <FreeRTOS.h>


#if defined(portEXT_RAM) && ( defined(__AVR_ATmega640__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega1281__) || defined(__AVR_ATmega2560__) || defined(__AVR_ATmega2561__) )

/****************************************************************************
  Defines
****************************************************************************/

#if defined(portQUAD_RAM)
	// QUAD RAM Enable (select) pin PD7 must be LOW to enable the external RAM.
#define EXT_RAM_SELECT(x) { DDRD |= _BV(PD7); if (x) PORTD |= _BV(PD7); else PORTD &= ~_BV(PD7); }

// Bits PL7, PL6, PL5 select the bank
// We're assuming with this initialisation we want to have 8 banks of 56kByte (+ 8kByte inbuilt unbanked)
#define EXT_RAM_ADDR19(x) { DDRL |= _BV(PL8); if (x) PORTL |= _BV(PL7); else PORTL &= ~_BV(PL7); }
#define EXT_RAM_ADDR18(x) { DDRL |= _BV(PL6); if (x) PORTL |= _BV(PL6); else PORTL &= ~_BV(PL6); }
#define EXT_RAM_ADDR17(x) { DDRL |= _BV(PL5); if (x) PORTL |= _BV(PL5); else PORTL &= ~_BV(PL5); }

#define EXT_RAM_ADDR16(x) { DDRC |= _BV(PC7); if (x) PORTC |= _BV(PC7); else PORTC &= ~_BV(PC7); }

#elif defined(portMEGA_RAM)
	// MEGA RAM Enable (select) pin PL7 must be HIGH to enable the external RAM.
#define EXT_RAM_SELECT(x) { DDRL |= _BV(PL7); if (x) PORTL |= _BV(PL7); else PORTL &= ~_BV(PL7); }

// Bit PD7 selects the bank BANKSEL
// We're assuming with this initialisation we want to have 2 banks of 56kByte (+ 8kByte inbuilt unbanked)
#define EXT_RAM_ADDR17(x) { DDRD |= _BV(PD7); if (x) PORTD |= _BV(PD7); else PORTD &= ~_BV(PD7); }

#define EXT_RAM_ADDR16(x) { DDRC |= _BV(PC7); if (x) PORTC |= _BV(PC7); else PORTC &= ~_BV(PC7); }

#else
#warning "Bad External RAM Definition"
#endif

/* Pointers to the start extended memory. XRAMEND is a system define */
#define XRAMSTART  0x2200

/****************************************************************************
  Global definitions
****************************************************************************/

// put this C code into .init3 (assembler could go into .init1)
void extRAMinit (void) __attribute__ ((used, naked, section (".init3")));

// This must be called from main() to ensure that this library is included in the build,
// if no other function from this library is being used. i.e. you're not using the banks.
// Would rather not do this, but don't know another method to force the library to be included.
uint8_t extRAMcheck (void);


#if defined(portEXT_RAM_8_BANK)
/****************************************************************************
  Variable definitions
****************************************************************************/

/* State variables used by the heap */
typedef struct {
	char *__malloc_heap_start;
	char *__malloc_heap_end;
	void *__brkval;
	void *__flp;
} heapState;

/* Results of a self-test run */
typedef struct {
		bool succeeded;
		volatile uint8_t *failedAddress;
		uint8_t failedBank;
} extRAMSelfTestResults;


/****************************************************************************
  Global definitions
****************************************************************************/


void extRAMheap8init(bool heapInXmem_);

void setMemoryBank(uint8_t bank_, bool switchHeap_);

extRAMSelfTestResults extRAMselfTest8(void);

#endif

/* these symbols get us to the details of the stack and heap */
#define STACK_POINTER() ((char *)AVR_STACK_POINTER_REG)

/* References to the private heap variables */
// These are system provided variables. The don't need to be further defined.
extern char  __heap_start;
extern char  __heap_end;
extern char *__malloc_heap_start;
extern char *__malloc_heap_end;
extern void *__brkval;
extern void *__flp;


#endif


#ifdef __cplusplus
}
#endif


#endif /* EXT_RAM_H_ */
