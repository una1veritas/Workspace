;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         pia.h
;*
;* Language:     Assembly
;*
;* Description:  Port and bit name definitions (specific to this project)
;*               Register mapping is NOT specified here!
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
;* Do not work:  -
;* 
;* 
;* Changelog:    2004-09-19  Michael Baeuerle
;*               Added port and bit names
;*
;*               2004-09-25  Michael Baeuerle
;*               Added more port and bit names
;*               Added CONFIGURE command
;*
;*               2004-09-27  Michael Baeuerle
;*               Added bit names for CONFIG register
;*               Added ABORT command
;*               Added RECOVER command
;*
;*               2004-10-09  Michael Baeuerle
;*               Added some command error codes
;*               Added ACCEPT_SELECTION command
;*               Added GET_MESSAGE command
;*
;*               2004-10-16  Michael Baeuerle
;*               Added DATA_LENGTH error code
;*
;*               2004-10-17  Michael Baeuerle
;*               Added GET_COMMAND command
;*               Added PUT_STATUS command
;*
;*               2004-10-17  Michael Baeuerle
;*               Added GET_DATA command
;*               Added PUT_DATA command
;*               Position of FLOW bit in STAT register changed
;*               Added FLOW2 bit name for STAT register
;*
;* 
;* To do:        -
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Externally visible PIA status bits
;*
;* Bit names for STAT register
;* Bits 2-6 generates a PIA interrupt when they become set
;* Bit 7 generates a PIA interrupt when changed
;*
;*******************************************************************************

.equ  ENABLE = 0                   ;PIA is initialized and running
.equ  ERROR = 1                    ;Command was not successful
.equ  COMPLETE = 2                 ;Command complete
;.equ  RESET = 3                    ;RESET condition detected
.equ  ATTENTION = 4                ;ATTENTION condition detected
.equ  SELECT = 5                   ;Somebody has selected us
.equ  FLOW = 6                     ;Data buffer available (Flow control)
.equ  FLOW2 = 7                    ;Additional flow control for large data


;*******************************************************************************
;*
;* PIA commands
;*
;* Values for CONTROL register
;*
;*******************************************************************************

.equ  ABORT = 0x00                 ;Must be zero
.equ  CONFIGURE = 0x01             ;Configure PIA
.equ  RECOVER = 0x02               ;Recover after error (Clear data buffer)
.equ  ACCEPT_SELECTION = 0x03      ;Accept selection from initiator
.equ  GET_MESSAGE = 0x04           ;Transfers message from initiator to target
.equ  PUT_MESSAGE = 0x05           ;Transfers message from target to initiator
.equ  GET_COMMAND = 0x06           ;Transfers command from initiator to target
.equ  PUT_STATUS = 0x07            ;Transfers status from target to initiator
.equ  GET_DATA = 0x08              ;Transfers data from initiator to target
.equ  PUT_DATA = 0x09              ;Transfers data from target to initiator
.equ  BUSFREE = 0x0A               ;Switch to BUS-FREE phase


;*******************************************************************************
;*
;* PIA configuration
;*
;* Values for CONFIG register
;*
;*******************************************************************************

.equ  TARGET = 0                   ;Enable Target mode (Accept selections)


;*******************************************************************************
;*
;* Command error codes
;*
;* Values for ERRORCODE register
;*
;*******************************************************************************

.equ  ABORTED = 0x00               ;Command aborted
.equ  INVALID = 0x01               ;Invalid parameter or command
.equ  SEQUENCE = 0x02              ;Illegal bus phase sequence
.equ  PARITY = 0x03                ;Parity error
.equ  PROTOCOL = 0x04              ;Protocol error
.equ  DATA_LENGTH = 0x05           ;Data length error


;*******************************************************************************
;*
;* SCSI bus phases
;*
;* Values for BUS_PHASE register
;*
;*******************************************************************************

.equ  BUS_FREE = 0x00
.equ  ARBITRATION = 0x01
.equ  SELECTION = 0x02
.equ  RESELECTION = 0x03
.equ  COMMAND = 0x04
.equ  DATA_OUT = 0x05
.equ  DATA_IN = 0x06
.equ  STATUS = 0x07
.equ  MESSAGE_OUT = 0x08
.equ  MESSAGE_IN = 0x09


;*******************************************************************************
;*
;* Port names
;*
;*******************************************************************************

.equ  SDB  = PORTA                 ;SCSI data bus
.equ  D_SDB  = DDRA
.equ  R_SDB  = PINA

.equ  T_SDB  = PORTC

.equ  SCB  = PORTE                 ;SCSI control bus
.equ  D_SCB  = DDRE
.equ  R_SCB  = PINE

.equ  T_SCB  = PORTD

.equ  MISC  = PORTG                ;SCSI miscellaneous pins
.equ  D_MISC  = DDRG
.equ  R_MISC  = PING

.equ  PDB  = PORTF                 ;PIA data bus
.equ  D_PDB  = DDRF
.equ  R_PDB  = PINF

.equ  PCB  = PORTB                 ;PIA control bus
.equ  D_PCB  = DDRB
.equ  R_PCB  = PINB


;*******************************************************************************
;*
;* Pin names
;*
;*******************************************************************************

.equ  EN_PHY = PG0                 ;PHY bus signal pin names
.equ  P = PG1
.equ  RE_P = PG2
.equ  RST = PG3
.equ  RE_RST = PG4

.equ  BSY = PE0
.equ  SEL = PE1
.equ  CD = PE2
.equ  IO = PE3
.equ  ATN = PE4
.equ  MSG = PE5
.equ  REQ = PE6
.equ  ACK = PE7

.equ  PIA_A0 = PB0                 ;PIA control bus pin names
.equ  PIA_ACK = PB3
.equ  PIA_RD = PB4
.equ  PIA_WR = PB5
.equ  PIA_IRQ = PB6
.equ  LED = PB7                    ;Busy-LED pin


;EOF