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
// ANSEL AND WPU
#define ENABLE  1
#define DISABLE 0

#define PORT_INPUT      0xff
#define PORT_OUTPUT     0x00
#define PORT_ENABLE     0xff
#define PORT_DISABLE    0x00

#define MCAT(x,y)   _MCAT(x,y)
#define _MCAT(x,y)  x ## y
    
#define pinmode(pin, mode)          ( MCAT(TRIS, pin) = mode)
// weak pull-up on then input mode (both by setting high)), 
#define pinmodewpu(pin, mode)       ( MCAT(WPU, pin) = mode, MCAT(TRIS, pin) = mode)
#define portmode(port, modebyte)        ( MCAT(TRIS,port) = modebyte)
#define portmodewpu(port, modebyte)     ( MCAT(TRIS,port) = modebyte, MCAT(WPU,port) = modebyte)
#define TRIS(pinport)               ( MCAT(TRIS, pinport) )
#define WPU(pinport)                ( MCAT(WPU, pinport))

#define pinwrite(pin, val)              ( MCAT(LAT,pin) = val ) 
#define portwrite(port, modebyte)       ( MCAT(LAT,port) = modebyte)
#define LAT(pinport)                    ( MCAT(TRIS, pinport) )

#define pinread(pin)                    ( MCAT(R,pin) ) 
#define portread(port)                  ( MCAT(PORT, port) )
#define PORT(port)                      ( MCAT(PORT, port) )

#define pinanalogmode(pin, mode)        ( MCAT(ANSEL,pin) = mode)
#define portanalogmode(port, modebyte)        ( MCAT(ANSEL,pin) = modebyte)
#define ANSEL(pinport, onoff)           (MCAT(ANSEL, pinport) = onoff)

#define PPS(pin)        ( MCAT(MCAT(R,pin),PPS) )

#ifdef	__cplusplus
}
#endif

#endif	/* PIC18COMMON_H */

