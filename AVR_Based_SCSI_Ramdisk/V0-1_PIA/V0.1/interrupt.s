;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         interrupt.s
;*
;* Language:     Assembly
;*
;* Description:  Interrupt handlers
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
;*               Added unexpected interrupt handler
;*
;*               2004-09-19  Michael Baeuerle
;*               Added RESET interrupt handler
;*               Added PCINT1 interrupt handler
;*
;*               2004-09-28  Michael Baeuerle
;*               PCINT1: Now directly accesses the data buffer
;*               PCINT1: Send ERRORCODE on read request if ERROR status is set
;*               PCINT1: Preserve SREG status
;*
;*               2004-10-09  Michael Baeuerle
;*               Added PCINT0 interrupt handler
;*               Added INT_TIMER2_COMP interrupt handler for Selection Abort
;*                Timeout
;*
;*               2004-10-10  Michael Baeuerle
;*               INT_PCINT0: Handling for ATTENTION condition added
;*
;*               2004-10-16  Michael Baeuerle
;*               INT_PCINT1: Use 'I_xxx' macros for data buffer access
;*
;*               2004-10-24  Michael Baeuerle
;*               INT_PCINT0: Bugfix: Mask other SCSI IDs to check whether we
;*               are selected
;*               INT_PCINT0: Bugfix: Preserve SREG
;*
;*               2005-05-27  Michael Baeuerle
;*               Bugfix: INT_PCINT1: Write access now reads PDB instead of PCB
;*
;* 
;* To do:        
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Unexpected interrupt handler
;*
;* This routine is never called if things work as intended by me ;-)
;* The MCU will be haltet and the Busy LED flash
;* This handler never returns!
;*
;*******************************************************************************

INT_UNEXPECTED:
      ;Release PIA data bus
      ldi  I_TMP, 0x00
      out  D_PDB, I_TMP
      ldi  I_TMP, 0xFF
      out  PDB, I_TMP

      ;Activate /PIA_ACK to indicate interface busy
      cbi  PCB, PIA_ACK

      ;Toggle LED
      cbi  PORTB, LED              ;Switch LED on
      ;MDELAY  250
      sbi  PORTB, LED              ;Switch LED off
      ;MDELAY  250
      
      ;Continue with main task
      rjmp  INT_UNEXPECTED


;*******************************************************************************
;*
;* RESET interrupt handler
;*
;* Execute soft reset
;* This handler never returns!
;*
;*******************************************************************************

INT_RESET:
      ;Init MCU (this also clears the stack)
      jmp  INIT                    ;=> Look at "init.s"


;*******************************************************************************
;*
;* PCINT0 interrupt handler
;*
;* Pin change on BSY and/or ATN (SCSI control bus)
;* Check for ATTENTION condition.
;* Check for SELECTION phase and whether we are selected.
;*
;* Stack: 2 Bytes
;* Call : -
;* Macro: -
;* Read : BUS_PHASE [High register] (global)
;* Write: I_SAVE register] (local)
;*        I_TMP [High register] (local)
;*        STAT [High register] (global)
;* Runtime: 13 Cycles (Minimum), 48 + 7 Cycles for delay (Worst case)
;*
;*******************************************************************************

INT_PCINT0:
      ;Save SREG
      in  I_SAVE, SREG

      ;Check for ATTENTION condition
      cpi  BUS_PHASE, BUS_FREE     ;Check for Bus-Free phase
      breq  INT_PCINT0_SEL         ;Bus-Free => Ignore ATN

      sbic  R_SCB, ATN             ;Update ATTENTION flag in status register
      rjmp  INT_PCINT0_ATN1
      cbr  STAT, (1 << ATTENTION)
      rjmp  INT_PCINT0_ATN_EXIT
   INT_PCINT0_ATN1:
      sbrs  STAT, ATTENTION        ;Check for ATTENTION flag already set
      cbi  PCB, PIA_IRQ            ;No => Activate PIA IRQ
      sbr  STAT, (1 << ATTENTION)
   INT_PCINT0_ATN_EXIT:

      ;Check for SELECTION phase
   INT_PCINT0_SEL:
      sbic  R_SCB, BSY
      rjmp  INT_PCINT0_EXIT        ;BSY active => Ignore
      sbis  R_SCB, SEL
      rjmp  INT_PCINT0_EXIT        ;SEL inactive (No selection phase) => Ignore
      sbic  R_SCB, IO
      rjmp  INT_PCINT0_EXIT        ;I/O active (Reselection phase) => Ignore

      ;SELECTION phase detected, check whether we are selected
      in  I_TMP, R_SDB
      and  I_TMP, ID
      cpse  I_TMP, ID
      rjmp  INT_PCINT0_EXIT        ;Not our ID on the data bus => Ignore

      ;We are selected, wait one bus settle delay and check again
      BUS_SETTLE_DELAY
      sbic  R_SCB, BSY
      rjmp  INT_PCINT0_EXIT        ;BSY active => Ignore
      sbis  R_SCB, SEL
      rjmp  INT_PCINT0_EXIT        ;SEL inactive (No selection phase) => Ignore
      sbic  R_SCB, IO
      rjmp  INT_PCINT0_EXIT        ;I/O active (Reselection phase) => Ignore
      in  I_TMP, R_SDB            
      and  I_TMP, ID
      cpse  I_TMP, ID
      rjmp  INT_PCINT0_EXIT        ;Not our ID on the data bus => Ignore

      ;We are really selected
      cbi  PCB, PIA_IRQ            ;Activate PIA IRQ (as soon as possible)
      lds  I_TMP, TCCR2A           ;Start Selection Abort Timer
      sbr  I_TMP, (7 << CS20)      ;(Set prescaler to 1024)
      sts  TCCR2A, I_TMP
      ldi  BUS_PHASE, ARBITRATION  ;Set ARBITRATION phase (for sequence check)
      sbr  STAT, (1 << SELECT)     ;Set SELECT flag in status register

      ;Return
   INT_PCINT0_EXIT:
      out  SREG, I_SAVE            ;Restore SREG
      reti


;*******************************************************************************
;*
;* PCINT1 interrupt handler
;*
;* Called after pin change on /PIA_WR or /PIA_RD
;* It is illegal to assert both strobe lines at any time!
;* This routine provides PIA register access
;*
;* Stack: 4 Bytes
;* Call : RD_BUFFER [Function]
;*        WR_BUFFER [Function]
;* Macro: -
;* Read : STAT [High register] (global)
;*        ERRORCODE [High register] (global)
;*        NULL [Register] (global)
;* Write: STROBE [High register] (static local)
;*        CONTROL [Register] (global)
;*        I_PARAMETER [Register] (local)
;*        I_TMP [High register] (local)
;*        I_SAVE register] (local)
;* Runtime: 18 Cyles (Minimum), 28 + 16 Cycles for buffer access (Worst case)
;*
;*******************************************************************************

INT_PCINT1:
      ;Check for strobe lines to be active
      sbis  R_PCB, PIA_WR          ;Check for /PIA_WR
      rjmp  INT_PCINT1_WR
      sbis  R_PCB, PIA_RD          ;Check for /PIA_RD
      rjmp  INT_PCINT1_RD

      ;Both lines released
      out  D_PDB, NULL             ;Release PIA data bus
      ldi  I_TMP, 0xFF             ;Enable Pull-Up resistors
      out  PDB, I_TMP
      sbi  PCB, PIA_ACK            ;Release /PIA_ACK
      ldi  STROBE, 0x00            ;Store current strobe state
      reti

      ;Attention: SREG is not saved up to here but must be preserved!
      ;-------------------------------------------------------------------------

      ;/PIA_WR active
   INT_PCINT1_WR:
      in  I_SAVE, SREG             ;Save SREG
      sbrc  STROBE, PIA_WR         ;Check for stored state
      reti                         ;Glitch detected => Return
      ldi  STROBE, (1 << PIA_WR)   ;Store current state
      sbis  R_PCB, PIA_A0          ;Check for address PIA_A0
      rjmp  INT_PCINT1_CMD
      in  I_PARAMETER, R_PDB       ;Get data
      cbi  PCB, PIA_ACK            ;Activate PIA ACK
      I_WR_BUFFER                  ;Write data to data buffer
      out  SREG, I_SAVE            ;Restore SREG
      reti
   INT_PCINT1_CMD:
      in  CONTROL, R_PDB           ;Get command
      cbi  PCB, PIA_ACK            ;Activate PIA ACK
      out  SREG, I_SAVE            ;Restore SREG
      reti
      
      ;/PIA_RD active
   INT_PCINT1_RD:
      in  I_SAVE, SREG             ;Save SREG
      sbrc  STROBE, PIA_RD         ;Check for stored state
      reti                         ;Glitch detected => Return
      ldi  STROBE, (1 << PIA_RD)   ;Store current state
      ldi  I_TMP, 0xFF             ;Drive PIA data bus
      out  D_PDB, I_TMP
      sbis  R_PCB, PIA_A0          ;Check for address line PIA_A0
      rjmp  INT_PCINT1_STAT
      sbrc  STAT, ERROR            ;Check for ERROR Bit
      rjmp  INT_PCINT1_ERROR       ;=> Don't access data buffer if ERROR active
      I_RD_BUFFER                  ;Read data from data buffer
      out  PDB, I_PARAMETER        ;Put data on PIA bus
      cbi  PCB, PIA_ACK            ;Activate PIA ACK
      out  SREG, I_SAVE            ;Restore SREG
      reti
   INT_PCINT1_ERROR:
      out  PDB, ERRORCODE          ;Put error code on PIA bus
      cbi  PCB, PIA_ACK            ;Activate PIA ACK
      out  SREG, I_SAVE            ;Restore SREG
      reti
   INT_PCINT1_STAT:
      out  PDB, STAT               ;Put status on PIA bus
      sbi  PCB, PIA_IRQ            ;Deactivate PIA IRQ
      cbi  PCB, PIA_ACK            ;Activate PIA ACK
      out  SREG, I_SAVE            ;Restore SREG
      reti


;*******************************************************************************
;*
;* TIMER2_COMP interrupt handler
;*
;* Compare match on Timer2.
;* Timer2 is used for the Selection Abort Timeout
;*
;* Stack: 2 Bytes
;* Call : PHASE_BUS_FREE [Function]
;* Macro: -
;* Read : -
;* Write: I_SAVE register] (local)
;*        I_TMP [High register] (local)
;*        STAT [High register] (global)
;* Runtime: 19 + 28 Cycles for phase change
;*
;*******************************************************************************

INT_TIMER2_COMP:
      ;Save SREG
      in  I_SAVE, SREG

      ;Clear SELECT flag in status register
      cbr  STAT, (1 << SELECT)

      ;Stop Timer2
      lds  I_TMP, TCCR2A
      cbr  I_TMP, (7 << CS20)
      sts  TCCR2A, I_TMP

      ;Clear Timer2
      ldi  I_TMP, 0x00
      sts  TCNT2, I_TMP

      ;Switch to BUS-FREE phase
      call  PHASE_BUS_FREE

      ;Restore SREG and return
      out  SREG, I_SAVE
      reti


;EOF
