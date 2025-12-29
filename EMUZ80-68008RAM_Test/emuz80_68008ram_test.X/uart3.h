/**
 * UART3 Generated Driver API Header File
 * 
 * @file uart3.h
 * 
 * @defgroup uart3 UART3
 * 
 * @brief This file contains API prototypes and other data types for the the Universal Asynchronous Receiver and Transmitter (UART) module.
 *
 * @version UART3 Driver Version 3.0.9
*/

#ifndef UART3_H
#define UART3_H


#include <stdbool.h>
#include <stdint.h>

#include <stdio.h>
#include "system.h"

#ifdef __cplusplus  // Provide C++ Compatibility

    extern "C" {

#endif

typedef union {
    struct {
        uint8_t perr : 1;     /**<This is a bit field for Parity Error status*/
        uint8_t ferr : 1;     /**<This is a bit field for Framing Error status*/
        uint8_t oerr : 1;     /**<This is a bit field for Overfrun Error status*/
        uint8_t reserved : 5; /**<Reserved*/
    };
    size_t status;            /**<Group byte for status errors*/
} uart3_status_t;

void UART3_Initialize(void);
void UART3_Deinitialize(void);

void UART3_Enable(void);
void UART3_Disable(void);

/*
void UART3_SendBreakControlEnable(void);
void UART3_SendBreakControlDisable(void);
*/

void UART3_ReceiveInterruptEnable(void);
void UART3_ReceiveInterruptDisable(void);

bool UART3_IsRxReady(void);
bool UART3_IsTxReady(void);
bool UART3_IsTxDone(void);


uint8_t UART3_Read(void);
void UART3_Write(uint8_t txData);

extern void (*UART3_RxInterruptHandler)(void);
void UART3_ReceiveISR(void);

int getch(void);
void putch(char txData);


#ifdef __cplusplus  // Provide C++ Compatibility


    }

#endif

#endif  // UART3_H
