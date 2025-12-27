/*
 * UART, disk I/O and monitor firmware for SuperMEZ80
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

#define BOARD_DEPENDENT_SOURCE

#include <supermez80.h>

static void emuz80_common_sys_init()
{
    // System initialize
    OSCFRQ = 0x08;      // 64MHz internal OSC

    // Disable analog function
    ANSELA = 0x00;
    ANSELB = 0x00;
    ANSELC = 0x00;
    ANSELD = 0x00;
    ANSELE0 = 0;
    ANSELE1 = 0;
    ANSELE2 = 0;

    ANSEL(Z80_ADDR_L) = 0x00;
    ANSEL(Z80_DATA) = 0x00;
    #ifdef Z80_ADDR_H
    ANSEL(Z80_ADDR_H) = 0x00;
    #endif

    // RESET output pin
    LAT(Z80_RESET) = 0;         // Reset
    TRIS(Z80_RESET) = 0;        // Set as output

    // UART3 initialize
    // U3BRG = 416;        // 9600bps @ 64MHz
    U3BRG = (_XTAL_FREQ/16)/57600UL+1; //(_XTAL_FREQ/16)/115200+1;
    //U3BRG = (_XTAL_FREQ/16)/230400+1;
    //U3BRG = (_XTAL_FREQ/16)/460800+1;
    U3RXEN = 1;         // Receiver enable
    U3TXEN = 1;         // Transmitter enable

    // UART3 Receiver
    TRISA7 = 1;         // RX set as input
    U3RXPPS = 0x07;     // RA7->UART3:RX3;

    // UART3 Transmitter
    LATA6 = 1;          // Default level
    TRISA6 = 0;         // TX set as output
    RA6PPS = 0x26;      // RA6->UART3:TX3;

    U3ON = 1;           // Serial port enable

    // /BUSREQ output pin
    LAT(Z80_BUSRQ) = 0;         // BUS request
    TRIS(Z80_BUSRQ) = 0;        // Set as output

    #ifdef Z80_NMI
    LAT(Z80_NMI) = 1;           // deactivate NMI
    TRIS(Z80_NMI) = 0;          // Set as output
    #endif

    #ifdef Z80_INT
    LAT(Z80_INT) = 1;           // deactivate INT
    TRIS(Z80_INT) = 0;          // Set as output
    #endif

    // Address bus A7-A0 pin
    LAT(Z80_ADDR_L) = 0x00;
    TRIS(Z80_ADDR_L) = 0x00;    // Set as output

    // Data bus D7-D0 pin
    LAT(Z80_DATA) = 0x00;
    TRIS(Z80_DATA) = 0x00;      // Set as output

    // Z80 clock
    #ifdef Z80_CLK
    #if Z80_CLK_HZ != 0
    PPS(Z80_CLK) = 0x3f;        // asign NCO1
    TRIS(Z80_CLK) = 0;          // NCO output pin
    NCO1INC = Z80_CLK_HZ * 2 / 61;
    NCO1CLK = 0x00;             // Clock source Fosc
    NCO1PFM = 0;                // FDC mode
    NCO1EN = 1;                 // NCO enable
    #else
    // Disable clock output for Z80 (Use external clock for Z80)
    PPS(Z80_CLK) = 0;           // select LATxy
    TRIS(Z80_CLK) = 1;          // set as input
    NCO1EN = 0;                 // NCO disable
    #endif
    #endif

    // TMR0
    // BOARD_TICK_HZ (= 100) periodic interrupts per second
    T0CON1 = 0x56;              // select Fsoc/4, pre scale 1:64 and disable TMR0 sync
    TMR0H = 250;                // load TMR0H
                                // value = (Delay * clk) / Prescalar
                                // = (1ms * 64Mhz / 4) / 64
                                // = 250
    T0CON0 = 0x89;              // enable TMR0 8 bit timer and post scale 1:10 (1ms to 10ms)
    PIR3bits.TMR0IF = 0;        // clear TMR0 interrupt flag
    PIE3bits.TMR0IE = 1;        // enable TMR0 interrupt
}

static void emuz80_common_start_z80(void)
{
    // Address bus A15-A8 pin (A14:/RFSH, A15:/WAIT)
    #ifdef Z80_ADDR_H
    WPU(Z80_ADDR_H) = 0xff;     // Week pull up
    TRIS(Z80_ADDR_H) = 0xff;    // Set as input
    #endif

    // Address bus A7-A0 pin
    WPU(Z80_ADDR_L) = 0xff;     // Week pull up
    TRIS(Z80_ADDR_L) = 0xff;    // Set as input

    // Data bus D7-D0 input pin
    WPU(Z80_DATA) = 0xff;       // Week pull up
    TRIS(Z80_DATA) = 0xff;      // Set as input

    // /IORQ input pin
    WPU(Z80_IOREQ) = 1;         // Week pull up
    TRIS(Z80_IOREQ) = 1;        // Set as input

    // /MREQ input pin
    #ifdef Z80_MEMRQ
    WPU(Z80_MEMRQ) = 1;         // Week pull up
    TRIS(Z80_MEMRQ) = 1;        // Set as input
    #endif

    // /RD input pin
    WPU(Z80_RD) = 1;            // Week pull up
    TRIS(Z80_RD) = 1;           // Set as input

    // /WR input pin
    #ifdef Z80_WR
    WPU(Z80_WR) = 1;            // Week pull up
    TRIS(Z80_WR) = 1;           // Set as input
    #endif

    // /M1 input pin
    #ifdef Z80_M1
    WPU(Z80_M1) = 1;            // Week pull up
    TRIS(Z80_M1) = 1;           // Set as input
    #endif

    // /RFSH input pin
    #ifdef Z80_RFSH
    WPU(Z80_RFSH) = 1;          // Week pull up
    TRIS(Z80_RFSH) = 1;         // Set as input
    #endif

    // /WAIT (RD7) output pin
    LAT(Z80_WAIT) = 1;          // WAIT
    TRIS(Z80_WAIT) = 0;         // Set as output

    // Unlock IVT
    IVTLOCK = 0x55;
    IVTLOCK = 0xAA;
    IVTLOCKbits.IVTLOCKED = 0x00;

    // Default IVT base address
    IVTBASE = 0x000008;

    // Lock IVT
    IVTLOCK = 0x55;
    IVTLOCK = 0xAA;
    IVTLOCKbits.IVTLOCKED = 0x01;
}

static uint32_t emuz80_common_high_addr_mask(void)
{
    return HIGH_ADDR_MASK;
}

static uint16_t emuz80_common_low_addr_mask(void)
{
    return LOW_ADDR_MASK;
}

static void emuz80_common_write_to_sram(uint16_t addr, uint8_t *buf, unsigned int len)
{
    union address_bus_u ab;
    unsigned int i;

    #ifdef SRAM_CE
    LAT(SRAM_CE) = 0;
    #endif

    ab.w = addr;
    #ifdef Z80_ADDR_H
    LAT(Z80_ADDR_H) = ab.h;
    #endif
    LAT(Z80_ADDR_L) = ab.l;
    for(i = 0; i < len; i++) {
        LAT(SRAM_WE) = 0;      // activate /WE
        LAT(Z80_DATA) = ((uint8_t*)buf)[i];
        LAT(SRAM_WE) = 1;      // deactivate /WE
        LAT(Z80_ADDR_L) = ++ab.l;
        if (ab.l == 0) {
            ab.h++;
            #ifdef Z80_ADDR_H
            LAT(Z80_ADDR_H) = ab.h;
            #endif
        }
    }

    #ifdef SRAM_CE
    LAT(SRAM_CE) = 1;
    #endif
}

static void emuz80_common_read_from_sram(uint16_t addr, uint8_t *buf, unsigned int len)
{
    union address_bus_u ab;
    unsigned int i;

    #ifdef SRAM_CE
    LAT(SRAM_CE) = 0;
    #endif

    ab.w = addr;
    #ifdef Z80_ADDR_H
    LAT(Z80_ADDR_H) = ab.h;
    #endif
    LAT(Z80_ADDR_L) = ab.l;
    for(i = 0; i < len; i++) {
        LAT(SRAM_OE) = 0;      // activate /OE
        ((uint8_t*)buf)[i] = PORT(Z80_DATA);
        LAT(SRAM_OE) = 1;      // deactivate /OE
        LAT(Z80_ADDR_L) = ++ab.l;
        if (ab.l == 0) {
            ab.h++;
            #ifdef Z80_ADDR_H
            LAT(Z80_ADDR_H) = ab.h;
            #endif
        }
    }

    #ifdef SRAM_CE
    LAT(SRAM_CE) = 1;
    #endif
}

static int emuz80_common_clock_op_hook(int clocks)
{
    static __uint24 nco1inc;
    static int running = 1;

    switch (clocks) {
    case BOARD_CLOCK_SUSPEND:
        if (!running)
            return 0;
        nco1inc = NCO1INC;
        NCO1INC = 0;
        NCO1INC = 0;
        running = 0;
        if (NCO1OUT)
            NCO1POL = !NCO1POL;
        break;
    case BOARD_CLOCK_RESUME:
        if (running)
            return 0;
        NCO1INC = nco1inc;
        NCO1INC = nco1inc;
        running = 1;
        break;
    case BOARD_CLOCK_GET:
        return NCO1OUT;
        break;
    case BOARD_CLOCK_HIGH:
        if (!NCO1OUT)
            NCO1POL = !NCO1POL;
        break;
    case BOARD_CLOCK_LOW:
        if (NCO1OUT)
            NCO1POL = !NCO1POL;
        break;
    case BOARD_CLOCK_INVERT:
            NCO1POL = !NCO1POL;
        break;
    default:
        if (clocks <= 0)
            return 0;
        while (0 < clocks--) {
            NCO1POL = !NCO1POL;
            NCO1POL = !NCO1POL;
        }
        break;
    }

    return 0;
}

static volatile uint32_t tick_count = 0;
static volatile uint8_t tick = 0;
static void __interrupt(irq(TMR0),high_priority) tmr0_isr(void)
{
    TMR0IF=0;
    tick_count++;
    tick = 1;
    return;
}

static void emuz80_common_tick(uint32_t *time)
{
    GIE = 0;                    // disable interrupt
    *time = tick_count;
    GIE = 1;                    // enable interrupt
}

static uint8_t emuz80_common_addr_l_pins(void) { return PORT(Z80_ADDR_L); }
static void emuz80_common_set_addr_l_pins(uint8_t v) { LAT(Z80_ADDR_L) = v; }
static uint8_t emuz80_common_data_pins(void) { return PORT(Z80_DATA); }
static void emuz80_common_set_data_pins(uint8_t v) { LAT(Z80_DATA) = v; }
static void emuz80_common_set_data_dir(uint8_t v) { TRIS(Z80_DATA) = v; }
static __bit emuz80_common_ioreq_pin(void) { return R(Z80_IOREQ); }
static __bit emuz80_common_memrq_pin(void) { return R(Z80_MEMRQ); }
static __bit emuz80_common_rd_pin(void) { return R(Z80_RD); }
#ifdef Z80_WR
static __bit emuz80_common_wr_pin(void) { return R(Z80_WR); }
#endif
static void emuz80_common_set_busrq_pin(uint8_t v) { LAT(Z80_BUSRQ) = (__bit)(v & 0x01); }
static void emuz80_common_set_reset_pin(uint8_t v) { LAT(Z80_RESET) = (__bit)(v & 0x01); }

static void emuz80_common_set_nmi_pin(uint8_t v) {
    #ifdef Z80_NMI
    LAT(Z80_NMI) = (__bit)(v & 0x01);
    #endif
}

static void emuz80_common_set_int_pin(uint8_t v) {
    #ifdef Z80_INT
    LAT(Z80_INT) = (__bit)(v & 0x01);
    #endif
}

static void emuz80_common_set_mq_pin(uint8_t v) {
    #ifdef Z80_M1
    LAT(Z80_M1) = (__bit)(v & 0x01);
    #endif
}

static void emuz80_common_set_wait_pin(uint8_t v) {
    #ifdef Z80_WAIT
    LAT(Z80_WAIT) = (__bit)(v & 0x01);
    #endif
}

static void emuz80_common_init()
{
    board_sys_init_hook         = emuz80_common_sys_init;
    board_start_z80_hook        = emuz80_common_start_z80;
    board_high_addr_mask_hook   = emuz80_common_high_addr_mask;
    board_low_addr_mask_hook    = emuz80_common_low_addr_mask;
    board_write_to_sram_hook    = emuz80_common_write_to_sram;
    board_read_from_sram_hook   = emuz80_common_read_from_sram;
    #if Z80_CLK_HZ != 0 && defined(Z80_CLK)
    board_clock_op_hook         = emuz80_common_clock_op_hook;
    #endif
    board_tick_hook             = emuz80_common_tick;

    board_addr_l_pins_hook      = emuz80_common_addr_l_pins;
    board_set_addr_l_pins_hook  = emuz80_common_set_addr_l_pins;
    board_data_pins_hook        = emuz80_common_data_pins;
    board_set_data_pins_hook    = emuz80_common_set_data_pins;
    board_set_data_dir_hook     = emuz80_common_set_data_dir;
    board_ioreq_pin_hook        = emuz80_common_ioreq_pin;
    board_memrq_pin_hook        = emuz80_common_memrq_pin;
    board_rd_pin_hook           = emuz80_common_rd_pin;
    #ifdef Z80_WR
    board_wr_pin_hook           = emuz80_common_wr_pin;
    #endif

    board_set_busrq_pin_hook    = emuz80_common_set_busrq_pin;
    board_set_reset_pin_hook    = emuz80_common_set_reset_pin;
    #ifdef Z80_NMI
    board_set_nmi_pin_hook      = emuz80_common_set_nmi_pin;
    #endif
    #ifdef Z80_INT
    board_set_int_pin_hook      = emuz80_common_set_int_pin;
    #endif
    board_set_wait_pin_hook     = emuz80_common_set_wait_pin;
}

static void emuz80_common_wait_for_programmer()
{
    //
    // Give a chance to use PRC (RB6/A6) and PRD (RB7/A7) to PIC programer.
    // It must prevent Z80 from driving A6 and A7 while this period.
    //
    printf("\n\r");
    printf("wait for programmer ...\r");
    __delay_ms(200);
    printf("                       \r");

    printf("\n\r");
}
