;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         init.s
;*
;* Language:     Assembly
;*
;* Description:  Configure the MCU after reset
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
;* Changelog:    2004-09-18  Michael Baeuerle
;*               Disable interrupts and init stack pointer
;*               Added PHY interface configuration
;*               Added PIA interface configuration
;*               Mask pin change interrupts and external interrupt0
;*
;*               2004-09-26  Michael Baeuerle
;*               Clear STAT register and set RESET bit
;*
;*               2004-09-27  Michael Baeuerle
;*               Clear data buffer
;*               Finally unmask pin change interrupts for PIA strobe lines
;*               Activate /PIA_IRQ line
;*               Enable interrupts
;*               Clear CONTROL register
;*
;*               2004-09-28  Michael Baeuerle
;*               Init variable NULL
;*
;*               2004-10-09  Michael Baeuerle
;*               Init Timer2 as Selection Abort Timer (192us)
;*
;*               2005-06-21  Michael Baeuerle
;*               Interrupt initialisation modified
;*               
;* 
;* To do:        -
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Init MCU
;*
;* This routine must be called by a RESET interrupt to initialize the MCU
;* Manually jumping to this routine executes a soft reset
;*
;* Stack: -
;* Call : -
;* Macro: -
;* Read : RAMEND [Integer] (Highest SRAM address)
;* Write: TMP [High register] (Local variable)
;*        STAT [High register] (PIA status register)
;*        CONTROL [High register] (PIA command currently in progress)
;*        STROBE [High register] (PIA interface strobe line status buffer)
;*        NULL [Register] (Always available zero)
;* Interrupt latency: Not relevant
;*
;*******************************************************************************

INIT:
      ;Disable interrupts
      cli

      ;Init Stack pointer
      ldi  TMP, high(RAMEND)
      out  SPH, TMP
      ldi  TMP, low(RAMEND)
      out  SPL, TMP

      ;Init PHY interface
      cbi  MISC, EN_PHY            ;Disable PHY
      sbi  D_MISC, EN_PHY
      ldi  TMP, 0x00               ;Switch PHY bus to High-Z
      out  D_SDB, TMP
      out  D_SCB, TMP
      out  SDB, TMP
      out  SCB, TMP
      ldi  TMP, 0b00010101
      out  D_MISC, TMP
      ldi  TMP, 0xFF               ;Switch PHY to input mode
      out  DDRD, TMP
      out  DDRC, TMP
      ldi  TMP, 0x00
      out  T_SCB, TMP
      out  T_SDB, TMP
      cbi  MISC, RE_P
      cbi  MISC, RE_RST

      ;Enable Pull-up for unused PHY interface pin RESET
      ;(This can be removed if RESET line should be handled by PIA)
      sbi  MISC, RST

      ;Init PIA interface
      ldi  TMP, 0x00               ;Switch PIA data bus to High-Z with pull-up
      out  D_PDB, TMP
      ldi  TMP, 0xFF
      out  PDB, TMP
      ldi  TMP, 0b11001000         ;Init control port (SPI pins to slave mode)
      out  D_PCB, TMP
      ldi  TMP, 0b11111111
      out  PCB, TMP

      ;Mask SCSI control bus interrupts
      ldi  TMP, 0x00
      sts  PCMSK0, TMP

      ;Unmask PIA interface interrupts for strobe lines
      ldi  TMP, (1 << PCINT12) | (1 << PCINT13)
      sts  PCMSK1, TMP

      ;Enable pin change interrupts and disable external interrupt 0
      ldi  TMP, (1 << PCIE0) | (1 << PCIE1)
      out  EIMSK, TMP

      ;Disable SPI
      ldi  TMP, 0x00
      out  SPCR, TMP

      ;Init NULL status registers
      ldi  TMP, 0x00
      mov  NULL, TMP

      ;Init PIA status registers
      ldi  STAT, (1 << COMPLETE)   ;ENABLE = 0, COMPLETE = 1

      ;Init PIA control register
      ldi  CONTROL, 0x00           ;No active command

      ;Init PIA interface strobe line status buffer
      ldi  STROBE, 0x00

      ;Clear data buffer
      call  CLEAR_BUFFER

      ;Init Timer2 (Selection Abort Timer)
      ldi  TMP, (1 << WGM21)       ;Set Timer2 to CTC mode, OC2 pin disconnected
      sts  TCCR2A, TMP             ;Timer stopped (Prescaler 1024 is required)
      clr  TMP                     ;Reset Timer2 to 0x00
      sts  TCNT2, TMP
      clr  TMP                     ;Set Timer2 clock source to sync. core clock
      sts  ASSR, TMP
                                   ;SCSI3 specify 200us, up to 250ms should work
;      ldi  TMP, 0x03               ;Set Timer2 compare value (0x03 => 192us)
      ldi  TMP, 0xFF               ;Set Timer2 compare value (0xFF => 16ms)
      sts  OCR2A, TMP
      ldi  TMP, (1 << TOV2) | (1 << OCF2A)  ;Clear Timer2 interrupt flags
      out  TIFR2, TMP
      ldi  TMP, (1 << OCIE2A)      ;Unmask Timer2 compare match interrupt
      sts  TIMSK2, TMP
      
      ;Boot complete, activate PIA interrupt to request configuration
      ;sbr  STAT, (1 << RESET)     ;Activate RESET condition (not supported)
      cbi  PCB, PIA_IRQ

      ;Enable interrupts
      sei

      ;Continue with main task
      jmp  MAIN
      

;EOF
