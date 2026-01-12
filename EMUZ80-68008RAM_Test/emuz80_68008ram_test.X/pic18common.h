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
#define ON      1
#define OFF     0
#define INPUT   1
#define OUTPUT  0
#define ENABLE  1
#define DISABLE 0

#define PORT_INPUT  0xff
#define PORT_OUTPUT 0x00
#define PORT_ON     0xff
#define PORT_OFF    0x00

#define WPU_ON      1
#define WPU_OFF     0

#define MACROCAT(x,y)   _MACROCAT(x,y)
#define _MACROCAT(x,y)  x ## y
    
#define pinmode(pin, mode)  ( MACROCAT(TRIS, pin) = mode)
// weak pull-up on then input mode (both by setting high)), 
#define pinmodewpu(pin, mode)    ( MACROCAT(WPU, pin) = mode, MACROCAT(TRIS, pin) = mode)
#define portmode(port, modebyte)        ( MACROCAT(TRIS,port) = modebyte)
#define portmodewpu(port, modebyte)     ( MACROCAT(TRIS,port) = modebyte, MACROCAT(WPU,port) = modebyte)
#define TRIS(pinport)           ( MACROCAT(TRIS, pinport) )
#define WPU(pinport)            ( MACROCAT(WPU, pinport))

#define pinwrite(pin, val)              ( MACROCAT(LAT,pin) = val ) 
#define portwrite(port, modebyte)       ( MACROCAT(LAT,port) = modebyte)
#define LAT(pinport)                    ( MACROCAT(TRIS, pinport) )

#define pinread(pin)                    ( MACROCAT(R,pin) ) 
#define portread(port)                  ( MACROCAT(PORT, port) )
#define PORT(port)                      ( MACROCAT(PORT, port) )

#define pinanalogmode(pin, mode)       ( MACROCAT(ANSEL,pin) = val)
#define portanalogmode(port, modebyte)        ( MACROCAT(ANSEL,pin) = modebyte)
#define ANSEL(pinport, onoff)           (MACROCAT(ANSEL, pinport) = onoff)

#ifdef	__cplusplus
}
#endif

#endif	/* PIC18COMMON_H */

