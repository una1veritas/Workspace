#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

// Configuration bits should be set in your project settings or a separate config file
// e.g., #pragma config FOSC = ECH, ... (check your specific PIC18F-Q datasheet)

#define _XTAL_FREQ 16000000UL // Define your crystal frequency for delay functions

// --- UART Buffer Configuration ---
#define RX_BUFFER_SIZE 64
volatile uint8_t rx_buffer[RX_BUFFER_SIZE];
volatile uint16_t rx_buffer_head = 0;
volatile uint16_t rx_buffer_tail = 0;

#define TX_BUFFER_SIZE 64
volatile uint8_t tx_buffer[TX_BUFFER_SIZE];
volatile uint16_t tx_buffer_head = 0;
volatile uint16_t tx_buffer_tail = 0;

// --- Function Prototypes ---
void SYSTEM_Initialize(void);
void UART_Initialize(void);
void __interrupt(high_priority) High_Priority_ISR(void);
void UART_WriteChar(uint8_t data);
uint8_t UART_ReadChar(void);
bool UART_RxBufferDataReady(void);
uint16_t UART_RxBufferCount(void);
bool UART_TxBufferFreeSpace(void);

// ----------------------------------------------------------------------
// Main function
// ----------------------------------------------------------------------
void main(void) {
    SYSTEM_Initialize();
    UART_Initialize();

    // Enable global interrupts
    RCONbits.IPEN = 1;      // Enable interrupt priorities
    INTCONbits.GIEH = 1;    // Enable high priority interrupts
    INTCONbits.GIEL = 1;    // Enable low priority interrupts

    // Test data for transmission
    const char *hello_msg = "Hello, PIC18F-Q UART!\r\n";
    for (uint16_t i = 0; hello_msg[i] != '\0'; i++) {
        UART_WriteChar(hello_msg[i]);
    }

    while (1) {
        // Main loop processes received data from the buffer
        if (UART_RxBufferDataReady()) {
            uint8_t received_data = UART_ReadChar();
            // Echo back the received character
            UART_WriteChar(received_data);
        }
        // Other non-blocking tasks can run here
    }
}

// ----------------------------------------------------------------------
// System Initialization
// ----------------------------------------------------------------------
void SYSTEM_Initialize(void) {
    // Set internal oscillator to 16MHz (adjust as needed for your part)
    OSCCONbits.IRCF = 0b111; // 16 MHz
    OSCCONbits.SCS = 0b11;   // Internal oscillator block
    while (OSCCONbits.HFIOFS != 1); // Wait for the oscillator to stabilize

    // Configure pins for UART (e.g., RC6 as TX, RC7 as RX, common on many PIC18F)
    TRISCbits.TRISC6 = 0; // TX pin as output
    TRISCbits.TRISC7 = 1; // RX pin as input

    // Disable analog functions on these pins if necessary
    ANSELCbits.ANSELC6 = 0;
    ANSELCbits.ANSELC7 = 0;
}

// ----------------------------------------------------------------------
// UART Initialization
// ----------------------------------------------------------------------
void UART_Initialize(void) {
    // Configure EUSART for 9600 baud rate @ 16MHz Fosc, high speed baud
    // SPBRG calculation: (Fosc / (16 * Baudrate)) - 1 if BRGH=1
    // (16000000 / (16 * 9600)) - 1 = 104.16. Use 104.
    SPBRG = 104;
    
    // Configure TXSTA
    TXSTAbits.TXEN = 1;  // Transmit enabled
    TXSTAbits.BRGH = 1;  // High Baud Rate Select
    TXSTAbits.SYNC = 0;  // Asynchronous mode
    
    // Configure RCSTA
    RCSTAbits.CREN = 1;  // Continuous Receive Enable
    RCSTAbits.SPEN = 1;  // Serial Port Enable

    // Configure Interrupts
    PIE1bits.RCIE = 1;   // Enable Receive Interrupt
    PIE1bits.TXIE = 0;   // Disable Transmit Interrupt initially (only enable when data is in the TX buffer)
    IPR1bits.RCIP = 1;   // Receive interrupt high priority
    IPR1bits.TXIP = 1;   // Transmit interrupt high priority
}

// ----------------------------------------------------------------------
// High Priority Interrupt Service Routine (ISR)
// ----------------------------------------------------------------------
void __interrupt(high_priority) High_Priority_ISR(void) {
    // Handle Receive Interrupt
    if (PIR1bits.RCIF) {
        if (RCSTAbits.OERR) { // Handle Overrun Error
            RCSTAbits.CREN = 0;
            RCSTAbits.CREN = 1;
        }
        if (UART_RxBufferCount() < RX_BUFFER_SIZE) {
            // Read RCREG to clear the RCIF flag and store the data
            rx_buffer[rx_buffer_head] = RCREG;
            rx_buffer_head = (rx_buffer_head + 1) % RX_BUFFER_SIZE;
        } else {
            // Buffer overrun, data lost. Read RCREG to clear the flag anyway.
            uint8_t dummy = RCREG;
        }
    }

    // Handle Transmit Interrupt
    if (PIR1bits.TXIF && PIE1bits.TXIE) {
        if (tx_buffer_head != tx_buffer_tail) {
            // Write data from buffer to TXREG to start transmission
            TXREG = tx_buffer[tx_buffer_tail];
            tx_buffer_tail = (tx_buffer_tail + 1) % TX_BUFFER_SIZE;
        } else {
            // Buffer empty, disable transmit interrupt
            PIE1bits.TXIE = 0;
        }
    }
}

// ----------------------------------------------------------------------
// UART Buffer Management Functions (accessible from main loop)
// ----------------------------------------------------------------------

// Function to check if data is ready in the RX buffer
bool UART_RxBufferDataReady(void) {
    return (rx_buffer_head != rx_buffer_tail);
}

// Function to get the number of bytes in the RX buffer
uint16_t UART_RxBufferCount(void) {
    if (rx_buffer_head >= rx_buffer_tail) {
        return rx_buffer_head - rx_buffer_tail;
    } else {
        return (RX_BUFFER_SIZE - rx_buffer_tail + rx_buffer_head);
    }
}

// Function to read a character from the RX buffer
uint8_t UART_ReadChar(void) {
    uint8_t data = 0;
    while (!UART_RxBufferDataReady()); // Wait until data is available

    // Disable interrupts temporarily while accessing shared variables (optional but safe)
    INTCONbits.GIEH = 0;
    data = rx_buffer[rx_buffer_tail];
    rx_buffer_tail = (rx_buffer_tail + 1) % RX_BUFFER_SIZE;
    INTCONbits.GIEH = 1;
    
    return data;
}

// Function to check if the TX buffer has free space
bool UART_TxBufferFreeSpace(void) {
    uint16_t next_head = (tx_buffer_head + 1) % TX_BUFFER_SIZE;
    return (next_head != tx_buffer_tail);
}

// Function to write a character to the TX buffer and enable TX interrupts
void UART_WriteChar(uint8_t data) {
    while (!UART_TxBufferFreeSpace()); // Wait if buffer is full

    // Disable interrupts temporarily while accessing shared variables
    INTCONbits.GIEH = 0;
    tx_buffer[tx_buffer_head] = data;
    tx_buffer_head = (tx_buffer_head + 1) % TX_BUFFER_SIZE;
    PIE1bits.TXIE = 1; // Enable TX interrupt to start transmission
    INTCONbits.GIEH = 1;
}
