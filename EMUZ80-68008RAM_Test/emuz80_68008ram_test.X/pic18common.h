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


#define INPUT   1
#define OUTPUT  0
#define PORT_INPUT  0xff
#define PORT_OUTPUT 0x00
    
#define WPUON      1
#define WPUOFF     0
#define PORT_WPUON  0xff
#define PORT_WPUOFF 0x00

#define HIGH    1
#define LOW     0

#define ANALOG      1
#define DIGITAL     0
#define PORT_DIGITAL 0x00

#ifdef	__cplusplus
}
#endif

#endif	/* PIC18COMMON_H */

