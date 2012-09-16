/**************************************************
 *
 * Macro declarations used for peripheral I/O
 * declarations for ARM IAR Assembler.
 *
 * Copyright 1999-2010 IAR Systems. All rights reserved.
 *
 **************************************************/

#ifndef __IO_MACROS_H
#define __IO_MACROS_H

/***********************************************
 *      Assembler specific macros
 ***********************************************/

#ifdef __IAR_SYSTEMS_ASM__

/***********************************************
 * I/O reg attributes (ignored)
 ***********************************************/
#define __READ_WRITE 0
#define __READ 0
#define __WRITE 0

/***********************************************
 * Define NAME as an I/O reg
 ***********************************************/
#define __IO_REG8(NAME, ADDRESS, ATTRIBUTE)      \
                  NAME DEFINE ADDRESS

#define __IO_REG16(NAME, ADDRESS, ATTRIBUTE)     \
                   NAME DEFINE ADDRESS

#define __IO_REG32(NAME, ADDRESS, ATTRIBUTE)     \
                   NAME DEFINE ADDRESS

/***********************************************
 * Define NAME as an I/O reg
 ***********************************************/
#define __IO_REG8_BIT(NAME, ADDRESS, ATTRIBUTE, BIT_STRUCT)  \
                      NAME DEFINE ADDRESS

#define __IO_REG16_BIT(NAME, ADDRESS, ATTRIBUTE, BIT_STRUCT) \
                       NAME DEFINE ADDRESS

#define __IO_REG32_BIT(NAME, ADDRESS, ATTRIBUTE, BIT_STRUCT) \
                       NAME DEFINE ADDRESS

#endif /* __IAR_SYSTEMS_ASM__ */

#endif /* __IO_MACROS_H */
