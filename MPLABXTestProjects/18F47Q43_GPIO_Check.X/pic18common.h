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
#define PORT_INPUT  0xff
#define PORT_OUTPUT 0x00
    
#define WPUON      1
#define WPUOFF     0
#define PORT_WPUON  0xff
#define PORT_WPUOFF 0x00

#define ANALOG      1
#define DIGITAL     0
#define PORT_DIGITAL    0x00
#define PORT_ANALOG     0xff
#define PORT_WPU_ON     0xff
#define PORT_WPU_OFF    0x00
#define WPU_ON      1
#define WPU_OFF     0

#define pinmode(pin, mode) (TRIS##pin = mode)
#define pinmodewpu(pin)    (TRIS##pin = mode, WPU##pin = mode)
#define pinwrite(pin, val) (LAT##pin = val) 
#define pinread(pin)       (PORT##pin) 
#define pinADmode(pin, val)        (ANSEL##pin = val)
#define portmode(port, mode8)       (TRIS##port = mode8)
#define portmodewpu(port, mode8)    (TRIS##port = mode8, WPU##port = mode8)
#define portwrite(port, val8)       (LAT##port = val8)

#ifdef	__cplusplus
}
#endif

#endif	/* PIC18COMMON_H */

