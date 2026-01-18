/* 
 * File:   pic18common.h
 * Author: sin
 *
 * Created on December 29, 2025, 9:28 AM
 */

#ifndef PIC18COMMON_H
#define	PIC18COMMON_H

#ifdef	__cplusplus
extern "C" {
#endif

#define HIGH    1
#define LOW     0
#define INPUT   1
#define OUTPUT  0
#define ENABLE  1
#define DISABLE 0

#define PORT_INPUT      0xff
#define PORT_OUTPUT     0x00
#define PORT_ENABLE     0xff
#define PORT_DISABLE    0x00

#define CCAT(x,y)   _CCAT(x,y)
#define _CCAT(x,y)  x##y

#define pinmode(pin, mode) (CCAT(TRIS, pin) = mode)
// input with weak pull-up, output without w pull-up
#define pinwpu(pin, endis)  ( CCAT(WPU, pin) = endis)
#define pinmodewpu(pin, mode)    ( CCAT(WPU,pin) = mode, CCAT(TRIS,pin) = mode)
#define pinwrite(pin, val) ( CCAT(LAT,pin) = val) 
#define pinread(pin)       ( CCAT(PORT,pin) ) 

#define portmode(port, mode8)       ( CCAT(TRIS,port) = mode8)
#define portmodewpu(port, mode8)    ( CCAT(TRIS,port) = mode8, CCAT(WPU,port) = mode8)
#define portwrite(port, val8)       ( CCAT(LAT,port) = val8)
#define portread(port)              ( CCAT(PORT,port) )

#define MODE(pinport)   ( CCAT(TRIS, pinport) )
#define PIN(pin)        ( CCAT(R, pin) )
#define PORTIN(port)      ( CCAT(PORT, port) )
#define OUT(pinport)    ( CCAT(LAT, pinport) )
#define ANSEL(pinport)  ( CCAT(ANSEL, pinport) )
#define WPU(pinport)    ( CCAT(WPU, pinport) )
#define PPS(pin)        ( CCAT(CCAT(R,pin), PPS) )

#ifdef	__cplusplus
}
#endif

#endif	/* PIC18COMMON_H */

