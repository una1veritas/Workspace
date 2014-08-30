;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         busphases.s
;*
;* Language:     Assembly
;*
;* Description:  These routines implement the bus phase handling of the SCSI
;*               Parallel Interface (SPI).
;*               All bus phase routines must return 0 on success and 1 if an
;*               an error occurs. ERRORCODE must be updated.
;*               Loops should check for ABORT command from Target.
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
;* Changelog:    2004-09-19  Michael Baeuerle
;*               Added Bus-Free phase handling
;*
;*               2004-10-08  Michael Baeuerle
;*               Bus-Free: Update BUS_PHASE register on success
;*               Added Selection phase handling
;*               Added CHECK_PARITY macro
;*
;*               2004-10-10  Michael Baeuerle
;*               Selection: Check for Selection Abort Timeout after stopping
;*               Timer2
;*               Added Message-Out phase handling
;*
;*               2004-10-16  Michael Baeuerle
;*               Message-Out: Implementation fixed and completed
;*               Added Message-In phase handling
;*
;*               2004-10-17  Michael Baeuerle
;*               Added Command phase handling
;*               Added Status phase handling
;*
;*               2004-10-23  Michael Baeuerle
;*               Added Data-Out phase handling
;*               Added Data-In phase handling
;*
;*               2005-05-28  Michael Baeuerle
;*               Bugfix: CHECK_PARITY macro works now
;*               Bugfix: Selection phase handling fixed
;*
;*               2005-06-06  Michael Baeuerle
;*               CHECK_PARITY now returns data byte in PARAMETER
;*               HANDSHAKE_OUT is now one cycle faster
;*               PHASE_SELECTION is now one cycle faster
;*
;*               2005-06-07  Michael Baeuerle
;*               Bugfix: Glitches on control lines BSY, SEL, REQ and ACK when
;*                they are deasserted because the PHY bus was released before
;*                the PHY is switched to input mode (the PHY bus signals are
;*                seen active when they are not driven) => Now this four pins
;*                are disabled in overlapping mode with disabled interrupts
;*                (driven by PIA and PHY for one PIA core cycle). This is only a
;*                collision if somebody else asserts the SCSI line at this time
;*               Similar problems when asserting these signals fixed
;*               Note: The described glitches on MSG, I/O and C/D are legal
;*
;*               2005-07-11  Michael Baeuerle
;*               PHASE_DATA_OUT speed enhanced
;*               PHASE_DATA_IN speed enhanced
;*
;*               2005-07-14  Michael Baeuerle
;*               HANDSHAKE_OUT now preserves SREG
;*               HANDSHAKE_IN now preserves SREG
;*
;*               2005-07-22  Michael Baeuerle
;*               Macro PHASE_DATA_OUT_FAST added
;*
;*               2005-08-13  Michael Baeuerle
;*               CHECK_PARITY now uses PAR instead of INDEX as local variable
;*
;*               2005-08-15  Michael Baeuerle
;*               Macro PHASE_DATA_IN_FAST added
;*
;* 
;* To do:        -
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Check for odd parity
;*
;* Macro (requires readable P signal on R_MISC:P and bus data on R_SDB).
;*
;* Parity checking is not supported by Hardware V1.0 due to a design bug ;-(
;* => We have to check in software using this macro.
;*
;* Attention:
;* Because some books and internet sources are wrong (and yes, Hardware V1.0 was
;* also wrong because I believed them) here the TRUTH what odd parity really is:
;* THE NUMBER OF LOGICAL ONES ON ALL NINE LINES (INCLUDING THE PARITY LINE!)
;* MUST BE ALWAYS AN ODD NUMBER.
;*
;* The data byte is returned in PARAMETER.
;* Returns 0 if parity is valid and 1 if parity is not valid.
;*
;* Stack: -
;* Call : -
;* Macro: -
;* Read : -
;* Write: RESULT [High register]
;*        PARAMETER [High register] (Data byte)
;*        PAR [Register]
;*        T [Flag]
;* Interrupt latency: -
;*
;*******************************************************************************

.macro  CHECK_PARITY
      clt                          ;Store parity bit in T flag
      sbic  R_MISC, P
      set

      in  PARAMETER, R_SDB         ;Read SCSI data bus

      clr  PAR                     ;Calculate parity
      sbrc  PARAMETER, 0
      inc  PAR
      sbrc  PARAMETER, 1
      inc  PAR
      sbrc  PARAMETER, 2
      inc  PAR
      sbrc  PARAMETER, 3
      inc  PAR
      sbrc  PARAMETER, 4
      inc  PAR
      sbrc  PARAMETER, 5
      inc  PAR
      sbrc  PARAMETER, 6
      inc  PAR
      sbrc  PARAMETER, 7
      inc  PAR
      brtc  CHECK_PARITY_T
      inc  PAR
   CHECK_PARITY_T:

      ldi  RESULT, 0x00            ;Check for odd parity is valid
      sbrs  PAR, 0
      ldi  RESULT, 0x01
.endmacro


;*******************************************************************************
;*
;* REQ/ACK handshake for Initiator output
;*
;* Macro (requires SCSI bus in output mode).
;* Respects ABORT command.
;* SREG is preserved
;* Returns 0 on success and 1 otherwise.
;*
;* Stack: -
;* Call : -
;* Macro: CHECK_PARITY
;* Read : CONTROL [Register]
;* Write: RESULT [High register]
;*        PARAMETER [High register] (Data byte)
;*        ERRORCODE [High register]
;*        TMP [High register]
;* Interrupt latency: 7 core cycles
;*
;*******************************************************************************

.macro  HANDSHAKE_OUT
      ;Assert REQ
      in  TMP, SREG                ;Save SREG
      cli                          ;Disable interrupts
      sbi  SCB, REQ                ;Set PHY line for REQ
      sbi  D_SCB, REQ              ;Drive PHY line for REQ Pin
      sbi  T_SCB, REQ              ;Drive REQ
      out  SREG, TMP               ;Restore interrupt flag

      ;Wait for Initiator to assert ACK
   HANDSHAKE_OUT_WAIT_A:
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  HANDSHAKE_OUT_WAIT_A2
      ldi  ERRORCODE, ABORTED      ;Command aborted
      ldi  RESULT, 0x01
      rjmp  HANDSHAKE_OUT_ERROR
   HANDSHAKE_OUT_WAIT_A2:
      sbis  R_SCB, ACK             ;Read status of ACK
      rjmp  HANDSHAKE_OUT_WAIT_A

      ;Check parity
      CHECK_PARITY
      tst  RESULT
      breq  HANDSHAKE_OUT_PARITY_OK
      ldi  ERRORCODE, PARITY       ;Parity error
      ldi  RESULT, 0x01
      rjmp  HANDSHAKE_OUT_ERROR
   HANDSHAKE_OUT_PARITY_OK:

      ;Data byte already in place => Do nothing

      ;Release REQ
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin
      out  SREG, TMP               ;Restore interrupt flag
      
      ;Wait for Initiator to release ACK
   HANDSHAKE_OUT_WAIT_B:
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  HANDSHAKE_OUT_WAIT_B2
      ldi  ERRORCODE, ABORTED      ;Command aborted
      ldi  RESULT, 0x01
      rjmp  HANDSHAKE_OUT_ERROR
   HANDSHAKE_OUT_WAIT_B2:
      sbic  R_SCB, ACK             ;Read status of ACK
      rjmp  HANDSHAKE_OUT_WAIT_B

      ;Success
      ldi  RESULT, 0x00
      rjmp  HANDSHAKE_OUT_EXIT

   HANDSHAKE_OUT_ERROR:
      ;Release REQ
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin

   HANDSHAKE_OUT_EXIT:
      out  SREG, TMP               ;Restore SREG (to preserve T & I flags)
.endmacro


;*******************************************************************************
;*
;* REQ/ACK handshake for Initiator input
;*
;* Macro (requires SCSI bus in input mode).
;* Respects ABORT command.
;* SREG is preserved
;* Returns 0 on success and 1 otherwise.
;*
;* Stack: -
;* Call : -
;* Macro: -
;* Read : CONTROL [Register]
;* Write: RESULT [High register]
;*        ERRORCODE [High register]
;*        TMP [High register]
;* Interrupt latency: 7 core cycles
;*
;*******************************************************************************

.macro  HANDSHAKE_IN
      ;Assert REQ
      in  TMP, SREG                ;Save SREG
      cli                          ;Disable interrupts
      sbi  SCB, REQ                ;Set PHY line for REQ
      sbi  D_SCB, REQ              ;Drive PHY line for REQ Pin
      sbi  T_SCB, REQ              ;Drive REQ
      out  SREG, TMP               ;Restore interrupt flag

      ;Wait for Initiator to assert ACK
   HANDSHAKE_IN_WAIT_A:
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  HANDSHAKE_IN_WAIT_A2
      ldi  ERRORCODE, ABORTED      ;Command aborted
      ldi  RESULT, 0x01
      rjmp  HANDSHAKE_IN_ERROR
   HANDSHAKE_IN_WAIT_A2:
      sbis  R_SCB, ACK             ;Read status of ACK
      rjmp  HANDSHAKE_IN_WAIT_A

      ;Release REQ
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin
      out  SREG, TMP               ;Restore interrupt flag
      
      ;Wait for Initiator to release ACK
   HANDSHAKE_IN_WAIT_B:
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  HANDSHAKE_IN_WAIT_B2
      ldi  ERRORCODE, ABORTED      ;Command aborted
      ldi  RESULT, 0x01
      rjmp  HANDSHAKE_IN_ERROR
   HANDSHAKE_IN_WAIT_B2:
      sbic  R_SCB, ACK             ;Read status of ACK
      rjmp  HANDSHAKE_IN_WAIT_B

      ;Success
      ldi  RESULT, 0x00
      rjmp  HANDSHAKE_IN_EXIT

   HANDSHAKE_IN_ERROR:
      ;Release REQ
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin

   HANDSHAKE_IN_EXIT:
      out  SREG, TMP               ;Restore SREG (to preserve T & I flags)
.endmacro


;*******************************************************************************
;*
;* Bus-Free phase
;*
;* Can be entered from any other phase.
;* The bus is released immediately.
;* Always return 0.
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;* Read : -
;* Write: RESULT [High register]
;*        TMP [High register]
;*        STAT [High register]
;*        BUS_PHASE [High register]
;* Interrupt latency: 4 core cycles
;*
;*******************************************************************************

PHASE_BUS_FREE:
      ;Release parity line
      cbi  MISC, P                 ;Set PHY line for P incactive
      cbi  D_MISC, P               ;Release PHY line for P
      cbi  MISC, RE_P              ;Switch transceiver for P to input mode

      ;Release data lines
      clr  TMP
      out  SDB, TMP                ;Set PHY lines for SCSI data bus inactive
      out  D_SDB, TMP              ;Release PHY lines for SCSI data bus
      out  T_SDB, TMP              ;Switch transceivers to input mode

      ;Release control lines
      in  RESULT, SREG             ;Disable interrupts
      cli
      out  SCB, TMP                ;Set PHY lines for SCSI control bus inactive
      out  T_SCB, TMP              ;Switch transceivers to input mode
      out  D_SCB, TMP              ;Release PHY lines for SCSI control bus
      out  SREG, RESULT            ;Restore interrupt flag
      
      ;Wait one Bus Settle Delay      
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, BUS_FREE
      cbr  STAT, (1 << ATTENTION)
      ldi  RESULT, 0x00            ;Return OK
      ret      


;*******************************************************************************
;*
;* Selection phase
;*
;* Can be entered only from Arbitration phase.
;* Respects ABORT command.
;* Currently implemented for target mode only!
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: CHECK_PARITY
;* Read : CONTROL [Register]
;* Write: RESULT [High register]
;*        ERRORCODE [High register]
;*        TMP [High register]
;*        PARAMETER [Register] (SCSI address of initiator)
;*        INDEX [Register]
;*        BUS_PHASE [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_SELECTION:
      ;Check for Selection phase to be legal
      cpi  BUS_PHASE, ARBITRATION
      breq  PHASE_SELECTION_CHECK1_END
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_SELECTION_CHECK1_END:

      ;Check parity
      CHECK_PARITY
      tst  RESULT
      breq  PHASE_SELECTION_CHECK2_END
      ldi  ERRORCODE, PARITY       ;Parity error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_SELECTION_CHECK2_END:
      ;Data byte is returned in PARAMETER by CHECK_PARITY

      ;Check for exactly two active Bits on the data bus
      clr  INDEX                   ;Check for 2 active data Bits
      sbrc  PARAMETER, 0
      inc  INDEX
      sbrc  PARAMETER, 1
      inc  INDEX
      sbrc  PARAMETER, 2
      inc  INDEX
      sbrc  PARAMETER, 3
      inc  INDEX
      sbrc  PARAMETER, 4
      inc  INDEX
      sbrc  PARAMETER, 5
      inc  INDEX
      sbrc  PARAMETER, 6
      inc  INDEX
      sbrc  PARAMETER, 7
      inc  INDEX
      ldi  TMP, 0x02
      cp  INDEX, TMP
      breq  PHASE_SELECTION_CHECK3_END
   PHASE_SELECTION_CHECK3_ERROR:
      ldi  ERRORCODE, PROTOCOL     ;Protocol error (Not exact 2 IDs on the bus)
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_SELECTION_CHECK3_END:

      ;Get SCSI ID of initiator
      in  TMP, R_SDB               ;Read SCSI data bus
      mov  PARAMETER, ID           ;Mask our SCSI ID
      com  PARAMETER
      and  TMP, PARAMETER          ;Initiator SCSI ID now in TMP (bit format)
      tst  TMP                     ;Check for Initiator SCSI ID to be illegal
      breq  PHASE_SELECTION_CHECK3_ERROR
      clr  PARAMETER               ;Convert initiator SCSI ID to number format
   PHASE_SELECTION_LOOP:
      sbrc  TMP, 0
      rjmp  PHASE_SELECTION_LOOP_END
      inc  PARAMETER
      lsr  TMP
      rjmp  PHASE_SELECTION_LOOP
   PHASE_SELECTION_LOOP_END:       ;Initiator SCSI ID now in PARAMETER (number)
      ldi  TMP, 0x07               ;Check for Initiator SCSI ID to be legal
      sub  TMP, PARAMETER
      brlo  PHASE_SELECTION_CHECK3_ERROR

      ;Stop Selection Abort Timer and check for timeout
      lds  TMP, TCCR2A             ;Stop Timer2
      cbr  TMP, (7 << CS20)
      sts  TCCR2A, TMP
      clr  TMP                     ;Reset Timer2
      sts  TCNT2, TMP
      cpi  BUS_PHASE, ARBITRATION  ;Check for timeout
      breq  PHASE_SELECTION_CHECK4_END
   PHASE_SELECTION_CHECK4_ERROR:
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error (Timeout)
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_SELECTION_CHECK4_END:

      ;Take over bus control from initiator
      sbis  R_SCB, SEL             ;Verify that initiator is still present
      rjmp  PHASE_SELECTION_CHECK4_ERROR
      sbi  T_SCB, BSY              ;Enable driver for BSY Pin
      sbi  D_SCB, BSY              ;Drive PHY line for BSY
      sbi  SCB, BSY                ;Set PHY line for BSY

      ;Wait for Initiator to release SEL
   PHASE_SELECTION_WAIT:
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  PHASE_SELECTION_WAIT2
      ldi  ERRORCODE, ABORTED      ;Command (who calls us) aborted
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_SELECTION_WAIT2:
      sbic  R_SCB, SEL             ;Read status of SEL
      rjmp  PHASE_SELECTION_WAIT

      ;Update current phase
      ldi  BUS_PHASE, SELECTION
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Message-Out phase
;*
;* Can be entered from any phase except Bus-Free, Arbitration and Reselection.
;* Respects ABORT command.
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;*        HANDSHAKE_OUT
;* Read : STAT [Register]
;*        CONTROL [Register]
;* Write: RESULT [High register]
;*        PARAMETER [Register] (Message byte)
;*        ERRORCODE [High register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_MESSAGE_OUT:
      ;Check for Message-Out phase to be legal
      cpi  BUS_PHASE, BUS_FREE
      breq  PHASE_MESSAGE_OUT_ERR
      cpi  BUS_PHASE, ARBITRATION
      breq  PHASE_MESSAGE_OUT_ERR
      cpi  BUS_PHASE, RESELECTION
      breq  PHASE_MESSAGE_OUT_ERR
      rjmp  PHASE_MESSAGE_OUT_CHECK1_END
   PHASE_MESSAGE_OUT_ERR:
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_MESSAGE_OUT_CHECK1_END:

      ;Check for ATTENTION condition
      sbrc  STAT, ATTENTION
      rjmp  PHASE_MESSAGE_OUT_CHECK2_END
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_MESSAGE_OUT_CHECK2_END:

      ;Release data bus
      cbi  MISC, P                 ;Release PHY line for P
      cbi  D_MISC, P
      clr  TMP                     ;Release PHY lines for data bus
      out  SDB, TMP
      out  D_SDB, TMP
      cbi  MISC, RE_P              ;Release P
      out  T_SDB, TMP              ;Release data bus

      ;Release I/O
      cbi  SCB, IO                 ;Release PHY line for I/O Pin
      cbi  D_SCB, IO
      cbi  T_SCB, IO               ;Release I/O

      ;Assert C/D
      sbi  T_SCB, CD               ;Drive C/D
      sbi  D_SCB, CD               ;Drive PHY line for C/D
      sbi  SCB, CD                 ;Set PHY line for C/D

      ;Assert MSG
      sbi  T_SCB, MSG              ;Drive MSG
      sbi  D_SCB, MSG              ;Drive PHY line for MSG
      sbi  SCB, MSG                ;Set PHY line for MSG

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Transfer data
      HANDSHAKE_OUT
      cpse  RESULT, NULL
      ret                          ;Error (ERRORCODE is already set) => Return

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, MESSAGE_OUT
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Message-In phase
;*
;* Can be entered from any phase except Bus-Free, Arbitration and Selection.
;* Respects ABORT command.
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;*        DATA_RELEASE_DELAY
;*        HANDSHAKE_IN
;* Read : -
;* Write: RESULT [High register]
;*        PARAMETER [Register] (Message byte)
;*        ERRORCODE [High register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_MESSAGE_IN:
      ;Check for Message-In phase to be legal
      cpi  BUS_PHASE, BUS_FREE
      breq  PHASE_MESSAGE_IN_ERR
      cpi  BUS_PHASE, ARBITRATION
      breq  PHASE_MESSAGE_IN_ERR
      cpi  BUS_PHASE, SELECTION
      breq  PHASE_MESSAGE_IN_ERR
      rjmp  PHASE_MESSAGE_IN_CHECK1_END
   PHASE_MESSAGE_IN_ERR:
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_MESSAGE_IN_CHECK1_END:

      ;Assert I/O
      sbi  T_SCB, IO               ;Drive I/O
      sbi  D_SCB, IO               ;Drive PHY line for I/O
      sbi  SCB, IO                 ;Set PHY line for I/O

      ;Wait one Data Release Delay
      DATA_RELEASE_DELAY
      
      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Drive data bus
      cbi  MISC, P                 ;Release PHY line for P
      cbi  D_MISC, P               ;(P is driven by 74280)
      sbi  MISC, RE_P              ;Drive P
      ldi  TMP, 0xFF               ;Drive data bus
      out  T_SDB, TMP
      out  D_SDB, TMP              ;Drive PHY lines for data bus

      ;Assert C/D
      sbi  T_SCB, CD               ;Drive C/D
      sbi  D_SCB, CD               ;Drive PHY line for C/D
      sbi  SCB, CD                 ;Set PHY line for C/D

      ;Assert MSG
      sbi  T_SCB, MSG              ;Drive MSG
      sbi  D_SCB, MSG              ;Drive PHY line for MSG
      sbi  SCB, MSG                ;Set PHY line for MSG

      ;Put message byte on data bus
      out  SDB, PARAMETER          ;Set PHY lines for data bus

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Transfer data
      HANDSHAKE_IN
      cpse  RESULT, NULL
      ret                          ;Error (ERRORCODE is already set) => Return

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, MESSAGE_IN
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Command phase
;*
;* Can be entered only from Command, Message-Out and Message-In phases.
;* Respects ABORT command.
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;*        HANDSHAKE_OUT
;* Read : -
;* Write: RESULT [High register]
;*        PARAMETER [Register] (Command byte)
;*        ERRORCODE [High register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_COMMAND:
      ;Check for Message-Out phase to be legal
      cpi  BUS_PHASE, COMMAND
      breq  PHASE_COMMAND_OK
      cpi  BUS_PHASE, MESSAGE_OUT
      breq  PHASE_COMMAND_OK
      cpi  BUS_PHASE, MESSAGE_IN
      breq  PHASE_COMMAND_OK
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_COMMAND_OK:

      ;Release data bus
      cbi  MISC, P                 ;Release PHY line for P
      cbi  D_MISC, P
      clr  TMP                     ;Release PHY lines for data bus
      out  SDB, TMP
      out  D_SDB, TMP
      cbi  MISC, RE_P              ;Release P
      out  T_SDB, TMP              ;Release data bus

      ;Release I/O
      cbi  SCB, IO                 ;Release PHY line for I/O Pin
      cbi  D_SCB, IO
      cbi  T_SCB, IO               ;Release I/O

      ;Assert C/D
      sbi  T_SCB, CD               ;Drive C/D
      sbi  D_SCB, CD               ;Drive PHY line for C/D
      sbi  SCB, CD                 ;Set PHY line for C/D

      ;Release MSG
      cbi  SCB, MSG                ;Release PHY line for MSG Pin
      cbi  D_SCB, MSG
      cbi  T_SCB, MSG              ;Release MSG

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Transfer data
      HANDSHAKE_OUT
      cpse  RESULT, NULL
      ret                          ;Error (ERRORCODE is already set) => Return

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, COMMAND
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Status phase
;*
;* Can be entered from any phase except Bus-Free, Arbitration, Selection and
;* Reselection.
;* Respects ABORT command.
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;*        DATA_RELEASE_DELAY
;*        HANDSHAKE_IN
;* Read : -
;* Write: RESULT [High register]
;*        PARAMETER [Register] (Status byte)
;*        ERRORCODE [High register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_STATUS:
      ;Check for Status phase to be legal
      cpi  BUS_PHASE, BUS_FREE
      breq  PHASE_STATUS_ERR
      cpi  BUS_PHASE, ARBITRATION
      breq  PHASE_STATUS_ERR
      cpi  BUS_PHASE, SELECTION
      breq  PHASE_STATUS_ERR
      cpi  BUS_PHASE, RESELECTION
      breq  PHASE_STATUS_ERR
      rjmp  PHASE_STATUS_OK
   PHASE_STATUS_ERR:
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_STATUS_OK:

      ;Assert I/O
      sbi  T_SCB, IO               ;Drive I/O
      sbi  D_SCB, IO               ;Drive PHY line for I/O
      sbi  SCB, IO                 ;Set PHY line for I/O

      ;Wait one Data Release Delay
      DATA_RELEASE_DELAY
      
      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Drive data bus
      cbi  MISC, P                 ;Release PHY line for P
      cbi  D_MISC, P               ;(P is driven by 74280)
      sbi  MISC, RE_P              ;Drive P
      ldi  TMP, 0xFF               ;Drive data bus
      out  T_SDB, TMP
      out  D_SDB, TMP              ;Drive PHY lines for data bus

      ;Assert C/D
      sbi  T_SCB, CD               ;Drive C/D
      sbi  D_SCB, CD               ;Drive PHY line for C/D
      sbi  SCB, CD                 ;Set PHY line for C/D

      ;Release MSG
      cbi  SCB, MSG                ;Release PHY line for MSG Pin
      cbi  D_SCB, MSG
      cbi  T_SCB, MSG              ;Release MSG

      ;Put status byte on data bus
      out  SDB, PARAMETER          ;Set PHY lines for data bus

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Transfer data
      HANDSHAKE_IN
      cpse  RESULT, NULL
      ret                          ;Error (ERRORCODE is already set) => Return

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, STATUS
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Asyncronous Data-Out phase
;*
;* Can be entered from any phase except Bus-Free, Arbitration, Selection,
;* Reselection and Status.
;* Respects ABORT command.
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;*        HANDSHAKE_OUT
;* Read : -
;* Write: RESULT [High register]
;*        PARAMETER [Register] (Command byte)
;*        ERRORCODE [High register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_DATA_OUT:
      ;Check for Status phase to be legal
      cpi  BUS_PHASE, DATA_OUT     ;Speed up sequential Data-Out phases
      breq  PHASE_DATA_OUT_OK2
      cpi  BUS_PHASE, BUS_FREE
      breq  PHASE_DATA_OUT_ERR
      cpi  BUS_PHASE, ARBITRATION
      breq  PHASE_DATA_OUT_ERR
      cpi  BUS_PHASE, SELECTION
      breq  PHASE_DATA_OUT_ERR
      cpi  BUS_PHASE, RESELECTION
      breq  PHASE_DATA_OUT_ERR
      cpi  BUS_PHASE, STATUS
      breq  PHASE_DATA_OUT_ERR
      rjmp  PHASE_DATA_OUT_OK
   PHASE_DATA_OUT_ERR:
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_DATA_OUT_OK:

      ;Release data bus
      cbi  MISC, P                 ;Release PHY line for P
      cbi  D_MISC, P
      clr  TMP                     ;Release PHY lines for data bus
      out  SDB, TMP
      out  D_SDB, TMP
      cbi  MISC, RE_P              ;Release P
      out  T_SDB, TMP              ;Release data bus

      ;Release I/O
      cbi  SCB, IO                 ;Release PHY line for I/O Pin
      cbi  D_SCB, IO
      cbi  T_SCB, IO               ;Release I/O

      ;Release C/D
      cbi  SCB, CD                 ;Release PHY line for C/D Pin
      cbi  D_SCB, CD
      cbi  T_SCB, CD               ;Release C/D

      ;Release MSG
      cbi  SCB, MSG                ;Release PHY line for MSG Pin
      cbi  D_SCB, MSG
      cbi  T_SCB, MSG              ;Release MSG

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY
   PHASE_DATA_OUT_OK2:

      ;Transfer data
      HANDSHAKE_OUT
      cpse  RESULT, NULL
      ret                          ;Error (ERRORCODE is already set) => Return

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, DATA_OUT
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Asyncronous Data-In phase
;*
;* Can be entered from any phase except Bus-Free, Arbitration, Selection,
;* Reselection and Status.
;* Respects ABORT command.
;* Returns 0 on success, 1 on error
;*
;* Stack: 2 Bytes (Return address)
;* Call : -
;* Macro: BUS_SETTLE_DELAY
;*        DATA_RELEASE_DELAY
;*        SYSTEM_DESKEW_DELAY
;*        CABLE_SKEW_DELAY
;*        HANDSHAKE_IN
;* Read : -
;* Write: RESULT [High register]
;*        PARAMETER [Register] (Status byte)
;*        ERRORCODE [High register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

PHASE_DATA_IN:
      ;Check for Status phase to be legal
      cpi  BUS_PHASE, DATA_IN      ;Speed up sequential Data-In phases
      breq  PHASE_DATA_IN_OK2
      cpi  BUS_PHASE, BUS_FREE
      breq  PHASE_DATA_IN_ERR
      cpi  BUS_PHASE, ARBITRATION
      breq  PHASE_DATA_IN_ERR
      cpi  BUS_PHASE, SELECTION
      breq  PHASE_DATA_IN_ERR
      cpi  BUS_PHASE, RESELECTION
      breq  PHASE_DATA_IN_ERR
      cpi  BUS_PHASE, STATUS
      breq  PHASE_DATA_IN_ERR
      rjmp  PHASE_DATA_IN_OK
   PHASE_DATA_IN_ERR:
      ldi  ERRORCODE, SEQUENCE     ;Phase sequence error
      ldi  RESULT, 0x01
      ret                          ;=> Return
   PHASE_DATA_IN_OK:

      ;Assert I/O
      sbi  T_SCB, IO               ;Drive I/O
      sbi  D_SCB, IO               ;Drive PHY line for I/O
      sbi  SCB, IO                 ;Set PHY line for I/O

      ;Wait one Data Release Delay
      DATA_RELEASE_DELAY
      
      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Drive data bus
      cbi  MISC, P                 ;Release PHY line for P
      cbi  D_MISC, P               ;(P is driven by 74280)
      sbi  MISC, RE_P              ;Drive P
      ldi  TMP, 0xFF               ;Drive data bus
      out  T_SDB, TMP
      out  D_SDB, TMP              ;Drive PHY lines for data bus

      ;Release C/D
      cbi  SCB, CD                 ;Release PHY line for C/D Pin
      cbi  D_SCB, CD
      cbi  T_SCB, CD               ;Release C/D

      ;Release MSG
      cbi  SCB, MSG                ;Release PHY line for MSG Pin
      cbi  D_SCB, MSG
      cbi  T_SCB, MSG              ;Release MSG

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY
   PHASE_DATA_IN_OK2:

      ;Wait one System Deskew + one Cable Skew Delay
      SYSTEM_DESKEW_DELAY
      CABLE_SKEW_DELAY

      ;Put data byte on data bus
      out  SDB, PARAMETER          ;Set PHY lines for data bus

      ;Transfer data
      HANDSHAKE_IN
      cpse  RESULT, NULL
      ret                          ;Error (ERRORCODE is already set) => Return

      ;Wait one Bus Settle Delay
      BUS_SETTLE_DELAY

      ;Update current phase
      ldi  BUS_PHASE, DATA_IN
      ldi  RESULT, 0x00            ;Return OK
      ret


;*******************************************************************************
;*
;* Fast DATA-OUT phase
;*
;* Macro (requires SCSI bus and REQ in output mode).
;* Respects ABORT command.
;* SREG is preserved.
;* Returns 0 on success and 1 otherwise.
;*
;* Stack: -
;* Call : -
;* Macro: CHECK_PARITY
;* Read : CONTROL [Register]
;* Write: RESULT [High register]
;*        PARAMETER [High register] (Data byte)
;*        ERRORCODE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

.macro  PHASE_DATA_OUT_FAST
      in  TMP, SREG                ;Save SREG

      ;Wait for Initiator to release ACK
   PHASE_DATA_OUT_FAST_WAIT_A:
      sbis  R_SCB, ACK             ;Read status of ACK
      rjmp  PHASE_DATA_OUT_FAST_WAIT_A2
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  PHASE_DATA_OUT_FAST_WAIT_A
      ldi  ERRORCODE, ABORTED      ;Command aborted
      rjmp  PHASE_DATA_OUT_FAST_ERROR
   PHASE_DATA_OUT_FAST_WAIT_A2:

      ;Assert REQ
      sbi  SCB, REQ                ;Set PHY line for REQ

      ;Wait for Initiator to assert ACK
   PHASE_DATA_OUT_FAST_WAIT_B:
      sbic  R_SCB, ACK             ;Read status of ACK
      rjmp  PHASE_DATA_OUT_FAST_WAIT_B2
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  PHASE_DATA_OUT_FAST_WAIT_B
      ldi  ERRORCODE, ABORTED      ;Command aborted
      rjmp  PHASE_DATA_OUT_FAST_ERROR
   PHASE_DATA_OUT_FAST_WAIT_B2:

      ;Check parity
      CHECK_PARITY
      tst  RESULT
      breq  PHASE_DATA_OUT_FAST_PARITY_OK
      ldi  ERRORCODE, PARITY       ;Parity error

   PHASE_DATA_OUT_FAST_ERROR:
      cbi  SCB, REQ                ;Release REQ
      ldi  RESULT, 0x01            ;Return error
      rjmp  PHASE_DATA_OUT_FAST_EXIT

   PHASE_DATA_OUT_FAST_PARITY_OK:
      ;Data byte already in place => Do nothing

      ;Release REQ
      cbi  SCB, REQ                ;Set PHY line for REQ
      
      ;Success
      ldi  RESULT, 0x00

   PHASE_DATA_OUT_FAST_EXIT:
      out  SREG, TMP               ;Restore SREG (to preserve T & I flags)
.endmacro


;*******************************************************************************
;*
;* Fast DATA-IN phase
;*
;* Macro (requires SCSI bus in input and REQ in output mode).
;* Respects ABORT command.
;* SREG is preserved.
;* Returns 0 on success and 1 otherwise.
;*
;* Stack: -
;* Call : -
;* Macro: -
;* Read : CONTROL [Register]
;* Write: RESULT [High register]
;*        PARAMETER [High register] (Data byte)
;*        ERRORCODE [High register]
;*        TMP [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

.macro  PHASE_DATA_IN_FAST
      in  TMP, SREG                ;Save SREG

      ;Wait for Initiator to release ACK
   PHASE_DATA_IN_FAST_WAIT_A:
      sbis  R_SCB, ACK             ;Read status of ACK
      rjmp  PHASE_DATA_IN_FAST_WAIT_A2
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  PHASE_DATA_IN_FAST_WAIT_A
      ldi  ERRORCODE, ABORTED      ;Command aborted
      rjmp  PHASE_DATA_IN_FAST_ERROR
   PHASE_DATA_IN_FAST_WAIT_A2:

      ;Put data byte on data bus
      out  SDB, PARAMETER          ;Set PHY lines for data bus

      ;Assert REQ
      sbi  SCB, REQ                ;Set PHY line for REQ

      ;Wait for Initiator to assert ACK
   PHASE_DATA_IN_FAST_WAIT_B:
      sbic  R_SCB, ACK             ;Read status of ACK
      rjmp  PHASE_DATA_IN_FAST_OK
      cpi  CONTROL, ABORT          ;Check for ABORT command
      brne  PHASE_DATA_IN_FAST_WAIT_B
      ldi  ERRORCODE, ABORTED      ;Command aborted

   PHASE_DATA_IN_FAST_ERROR:
      cbi  SCB, REQ                ;Release REQ
      ldi  RESULT, 0x01            ;Return error
      rjmp  PHASE_DATA_IN_FAST_EXIT

   PHASE_DATA_IN_FAST_OK:
 
      ;Release REQ
      cbi  SCB, REQ                ;Set PHY line for REQ
      
      ;Success
      ldi  RESULT, 0x00

   PHASE_DATA_IN_FAST_EXIT:
      out  SREG, TMP               ;Restore SREG (to preserve T & I flags)
.endmacro


;EOF
