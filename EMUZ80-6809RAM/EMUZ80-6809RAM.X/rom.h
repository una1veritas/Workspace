/* 
 * File:   rom.h
 * Author: sin
 *
 * Created on December 17, 2025, 1:06 PM
 */

#ifndef ROM_H
#define	ROM_H

#ifdef	__cplusplus
extern "C" {
#endif

#define ROM_TOP     0xC000		// ROM TOP Address
#define ROM_SIZE    0x4000		// 16K bytes

    extern const unsigned char rom[];

#ifdef	__cplusplus
}
#endif

#endif	/* ROM_H */

