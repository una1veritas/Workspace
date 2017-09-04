//==============================================================================// File:           desdefs.h
// Compiler:       IAR Atmel AVR C/EC++ Compiler
// Output Size:    192 bytes
// Created:        02/06/03
//
// Description:    This file contains the settings to configure the bootldr.c
//                 according to the configurations used in the encrypted file
//
//==============================================================================

#define PAGE_SIZE 128
#define MEM_SIZE 14336
#define SIGNATURE 0x08192A3B
#define BUFFER_SIZE 148
#define INITIALVECTOR_HI 0x00112233
#define INITIALVECTOR_LO 0x44556677
#define _3DES
