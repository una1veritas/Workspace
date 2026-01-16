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
// weak pull-up on then input mode (both by setting high)), 
// and weak pull-up off then output mode (both by setting low))
#define pinmodewpu(pin, mode)    ( CCAT(WPU,pin) = mode, CCAT(TRIS,pin) = mode)
#define pinwrite(pin, val) ( CCAT(LAT,pin) = val) 
#define pinread(pin)       ( CCAT(PORT,pin) ) 
#define pinanalog(pin, val)        ( CCAT(ANSEL, pin) = val)

#define portmode(port, mode8)       ( CCAT(TRIS,port) = mode8)
#define portmodewpu(port, mode8)    ( CCAT(TRIS,port) = mode8, CCAT(WPU,port) = mode8)
#define portwrite(port, val8)       ( CCAT(LAT,port) = val8)
#define portread(port)       ( CCAT(PORT,port) )
#define portanalog(port, val8)        ( CCAT(ANSEL,port) = val8)

#ifdef	__cplusplus
}
#endif

#endif	/* PIC18COMMON_H */

