;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         pia.s
;* Version:      V0.1
;*
;* Language:     Assembly
;*
;* Description:  This is the firmware for the Parallel Interface Agent.
;*               This implementation matches hardware V1.0.1 which cannot itself
;*               create a SCSI bus reset (external SCSI bus resets are handled
;*               by hardware, therefore there is no handling for the RESET
;*               condition).
;*
;*               The PIA firmware was designed for ATmega165 but was written for
;*               compatibility with ATmega169 (because I could not get an
;*               ATmega165 I have tested it on an ATmega169).
;*
;*               Notes:
;*               - If the ENABLE Bit in status register STAT is not set, the
;*                 PIA is not configured by the target and therefore disabled.
;*                 In this state it cannot execute any command except the
;*                 CONFIGURE, ABORT and RECOVER commands.        
;*
;* Copyright:    (C) 2004 by Michael Baeuerle <micha@hilfe-fuer-linux.de>
;* License:      All rights reserved.
;*               Redistribution and use in source and binary forms, with or
;*               without modification, are permitted provided that the
;*               following conditions are met:
;*               - Redistributions of source code must retain the above
;*                 copyright notice, this list of conditions and the following
;*                 disclaimer.
;*               - Redistributions in binary form must reproduce the above
;*                 copyright notice, this list of conditions and the following
;*                 disclaimer in the documentation and/or other materials
;*                 provided with the distribution.
;*               THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
;*               CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
;*               INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
;*               MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
;*               DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
;*               CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
;*               SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
;*               NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
;*               LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
;*               HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
;*               CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
;*               OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
;*               EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
;*
;* Written for:  Assembler:  tavrasm
;*               Platform:   ATmega165
;*               OS:         none
;* Tested with:  Assembler:  tavrasm (Version 1.19)
;*               Platform:   ATmega169
;*               OS:         none
;* Tested with:  Assembler:  tavrasm (Version 1.22)
;*               Platform:   ATmega169
;*               OS:         none
;* Do not work:  -
;* 
;* 
;* Changelog:
;* V0.0          Implement SCSI parallel interface for target mode
;*               2004-09-18  Michael Baeuerle
;*               Added global variable BUS_PHASE
;*               Added interrupt vector table
;*
;*               2004-09-19  Michael Baeuerle
;*               Added interrupt vector for LCD interrupt of ATmega169
;*               Added global variables STAT, CONTROL, DATA and ID
;*               Added global memory DATA_BUFFER
;*               Added local variables TMP, I_TMP, RESULT
;*               Added static local variable STROBE
;*
;*               2004-09-26  Michael Baeuerle
;*               Added constant BUFFERSIZE
;*               Added global variables BSH, BSL
;*               Added local variable PARAMETER
;*               Added CLEAR_BUFFER routine
;*               Added WR_BUFFER routine
;*               Added RD_BUFFER routine
;*
;*               2004-09-27  Michael Baeuerle
;*               Added global variable CONFIG
;*
;*               2004-09-28  Michael Baeuerle
;*               Added global variable ERRORCODE
;*               Removed global variable DATA
;*               Set size of data buffer to 768 bytes
;*               (With 1:2 handshake ratio complete sectors of 512 Bytes can be
;*               transferred without interrupting SCSI transfers and
;*               nevertheless enough space for stack remains)
;*
;*               2004-10-08  Michael Baeuerle
;*               Accelerated data buffer access routines by 1 cycle
;*               Command handling updated
;*
;*               2004-10-10  Michael Baeuerle
;*               Reject unknown commands with errorcode INVALID
;*               Reject all commands except RECOVER if ERROR flag is set
;*
;*               2004-10-16  Michael Baeuerle
;*               Moved data buffer handling code to 'buffer.s'
;*               Command handling updated
;*
;*               2004-10-17  Michael Baeuerle
;*               Command handling updated
;*
;*               2004-10-23  Michael Baeuerle
;*               Command handling updated
;*               Switch on Busy LED if PIA is not configured
;*               Switch on Busy LED if PIA is not in Bus-Free phase
;*
;*               2004-11-13  Michael Baeuerle
;*               Bugfix: Command error handling fixed
;*
;*               2004-11-27  Michael Baeuerle
;*               Bugfix: Command handling fixed
;*
;*               2005-05-28  Michael Baeuerle
;*               Bugfix: RECOVER command handling fixed
;*               BUSFREE command handling added
;*
;* V0.1          Accelerate asynchronous transfers
;*               2005-08-13  Michael Baeuerle
;*               Added local variable PAR
;*
;* 
;* To do:        -
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Definitions
;*
;*******************************************************************************

;Note: We do not use the ATmega169 LCD controller, so ATmega165 will also work
.include "./include/m169def.h"     ;Atmel compliant register and bit names
.include "./include/pia.h"         ;Project specific port and bit names


;*******************************************************************************
;*
;* Global variables
;*
;* => Look at "include/pia.h" for Bit names
;*
;*******************************************************************************

.dseg
.org  0x0100                       ;tavrasm V1.19 don't know extended I/O
BUFFER:
.equ  BUFFERSIZE = 768             ;Data buffer size (Limit: 2^16 - 1 - Stack)
                                   ;(Must be at least 512)
                                   ;(>1Ki probably creates handshake problems!)
                                   ;(Must end on or below address 0xFFFE)
.byte  BUFFERSIZE                  ;FIFO buffer for information transfer phases
				   ; X is the byte count in FIFO
                                   ; Y points to the FIFO read end
                                   ; Z points to the FIFO write end

.def  BSH = r7                     ;Copy of BUFFERSIZE
.def  BSL = r8
.def  LPLH = r9                    ;Data buffer pointer minimum value
.def  LPLL = r10
.def  UPLH = r11                   ;Data buffer pointer maximum value
.def  UPLL = r12
.def  NULL = r13                   ;Zero

.def  ID = r14                     ;Our SCSI address (Bit position coded)
.def  CONFIG = r15                 ;PIA configuration
.def  CONTROL = r16                ;PIA control register (PIA address 0)
.def  STAT = r17                   ;PIA status register (PIA address 0)
.def  BUS_PHASE = r18              ;This indicates the current SCSI bus phase
.def  ERRORCODE = r19              ;Error code
                                   ; (Only valid if ERROR Bit in STAT is set)


;*******************************************************************************
;*
;* Static local variables
;*
;*******************************************************************************

.def  STROBE = r20                 ;PIA interface strobe line states
                                   ; Bit4: /PIA_RD active
                                   ; Bit5: /PIA_WR active


;*******************************************************************************
;*
;* Local variables
;*
;*******************************************************************************

.def  PARAMETER = r2               ;Parameter of function call
.def  I_PARAMETER = r3             ;Parameter for interrupt handlers
.def  INDEX = r4                   ;Loop index
.def  I_SAVE = r5                  ;SREG buffer for interrupt handlers
.def  PAR = r6                     ;Local variable for parity calculation
.def  TMP = r21                    ;General purpose local variable
.def  I_TMP = r22                  ;Local variable for interrupt handlers
.def  RESULT = r23                 ;Result of function call

;LEN is used for DATA phases (R24/R25 can be used with ADIW and SBIW commands)
.def  LENL = r24                   ;Data length (Low byte)
.def  LENH = r25                   ;Data length (High byte)


;*******************************************************************************
;*
;* Interrupt vectors
;*
;*******************************************************************************

.cseg
      .org  0x0000
      jmp  INT_RESET               ;=> Look at "interrupt.s"
      jmp  INT_UNEXPECTED
      jmp  INT_PCINT0
      jmp  INT_PCINT1
      jmp  INT_TIMER2_COMP
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED
      jmp  INT_UNEXPECTED          ;This vector is only present in ATmega169
                                   ; (never executed in ATmega165 => No problem)
      .org  0x002E                 ;First unused location after vector table


;*******************************************************************************
;*
;* External functions
;*
;*******************************************************************************

.include  "./init.s"               ;MCU configuration routine
.include  "./timing.s"             ;Delay routines
.include  "./buffer.s"             ;Data buffer handling
.include  "./interrupt.s"          ;Interrupt handlers
.include  "./busphases.s"          ;SCSI bus phases
.include  "./command.s"            ;PIA command handlers


;*******************************************************************************
;*
;* Main task
;*
;* Wait for commands and execute them (infinite loop)
;*
;*******************************************************************************

MAIN:
      ;Switch LED on if we are busy or not configured
      sbrs  STAT, ENABLE           ;Check whether we are configured
      rjmp  MAIN_BUSY              ;No, switch LED on
      cpi  BUS_PHASE, BUS_FREE     ;Check current SCSI bus phase
      brne  MAIN_BUSY
      sbi  PORTB, LED              ;Switch off Busy LED
      rjmp  MAIN_BUSY_END
   MAIN_BUSY:
      cbi  PORTB, LED              ;Switch on Busy LED
   MAIN_BUSY_END:

      ;Wait for command
      tst  CONTROL
      breq  MAIN

      ;Command received => Check for ERROR flag is set
      sbrs  STAT, ERROR
      rjmp  MAIN_EXEC              ;No => Exec command
                                   ;Yes => Reject all commands except RECOVER
      cpi  CONTROL, RECOVER        ;Check for RECOVER command
      breq  MAIN_EXEC              ;=> RECOVER => Execute command
      ldi  ERRORCODE, INVALID      ;=> Not RECOVER => Reject command
      ldi  RESULT, 0x01
      rjmp  MAIN_CC

      ;Execute command
   MAIN_EXEC:
      cbr  STAT, (1 << COMPLETE) | (1 << ERROR)
      clr  ERRORCODE
      cpi  CONTROL, CONFIGURE      ;Check for CONFIGURE command
      brne  MAIN_C2
      call  CMD_CONFIGURE          ;=> Exec CONFIGURE command
      rjmp  MAIN_CC
   MAIN_C2:
      cpi  CONTROL, RECOVER        ;Check for RECOVER command
      brne  MAIN_C3
      call  CMD_RECOVER            ;=> Exec RECOVER command
      rjmp  MAIN_CC
   MAIN_C3:
      cpi  CONTROL, ACCEPT_SELECTION  ;Check for ACCEPT_SELECTION command
      brne  MAIN_C4
      call  CMD_ACCEPT_SELECTION   ;=> Exec ACCEPT_SELECTION command
      rjmp  MAIN_CC
   MAIN_C4:
      cpi  CONTROL, GET_MESSAGE    ;Check for GET_MESSAGE command
      brne  MAIN_C5
      call  CMD_GET_MESSAGE        ;=> Exec GET_MESSAGE command
      rjmp  MAIN_CC
   MAIN_C5:
      cpi  CONTROL, PUT_MESSAGE    ;Check for PUT_MESSAGE command
      brne  MAIN_C6
      call  CMD_PUT_MESSAGE        ;=> Exec PUT_MESSAGE command
      rjmp  MAIN_CC
   MAIN_C6:
      cpi  CONTROL, GET_COMMAND    ;Check for GET_COMMAND command
      brne  MAIN_C7
      call  CMD_GET_COMMAND        ;=> Exec GET_COMMAND command
      rjmp  MAIN_CC
   MAIN_C7:
      cpi  CONTROL, PUT_STATUS     ;Check for PUT_STATUS command
      brne  MAIN_C8
      call  CMD_PUT_STATUS         ;=> Exec PUT_STATUS command
      rjmp  MAIN_CC
   MAIN_C8:
      cpi  CONTROL, GET_DATA       ;Check for GET_DATA command
      brne  MAIN_C9
      call  CMD_GET_DATA           ;=> Exec GET_DATA command
      rjmp  MAIN_CC
   MAIN_C9:
      cpi  CONTROL, PUT_DATA       ;Check for PUT_DATA command
      brne  MAIN_C10
      call  CMD_PUT_DATA           ;=> Exec PUT_DATA command
      rjmp  MAIN_CC
   MAIN_C10:
      cpi  CONTROL, BUSFREE        ;Check for BUSFREE command
      brne  MAIN_C11
      call  CMD_BUSFREE            ;=> Exec BUSFREE command
      rjmp  MAIN_CC
   MAIN_C11:
      ldi  ERRORCODE, INVALID      ;Error: Unknown command
      ldi  RESULT, 0x01

      ;Command complete
   MAIN_CC:
      ldi  TMP, 0x00               ;Set ERROR bit if necessary
      cpse  RESULT, TMP
      sbr  STAT, (1 << ERROR)
      call  CLEAR_BUFFER           ;Clear data buffer for next command
      sbr  STAT, (1 << COMPLETE)   ;Set COMPLETE bit
      clr  CONTROL                 ;Terminate command
      cbi  PCB, PIA_IRQ            ;Activate PIA IRQ      
      rjmp  MAIN
  

;EOF
