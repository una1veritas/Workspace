;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         command.s
;*
;* Language:     Assembly
;*
;* Description:  These routines implements the PIA commands (available to the
;*               target controller)
;*               If a command returns nonzero status (error), the ERRORCODE
;*               register must be updated before!
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
;*               Added CMD_CONFIGURE command handler
;*
;*               2004-09-26  Michael Baeuerle
;*               Modified command CMD_CONFIGURE to use data buffer
;*
;*               2004-09-27  Michael Baeuerle
;*               Added CMD_RECOVER command handler
;*
;*               2004-10-09  Michael Baeuerle
;*               Added CMD_ACCEPT_SELECTION command handler
;*               RECOVER: Now clears the FLOW flag in STAT
;*               Added CMD_GET_MESSAGE command handler
;*
;*               2004-10-16  Michael Baeuerle
;*               Added CMD_PUT_COMAND command handler
;*
;*               2004-10-17  Michael Baeuerle
;*               Check for ENABLE flag in all commands not allowed until PIA is
;*               configured
;*               Added CMD_PUT_STATUS command handler
;*
;*               2004-10-23  Michael Baeuerle
;*               Added CMD_GET_DATA command handler
;*               Added CMD_PUT_DATA command handler
;*               RECOVER: Now clears the FLOW2 flag in STAT
;*               Added CMD_TOGGLE_FLOW2 macro
;*
;*               2005-05-28  Michael Baeuerle
;*               Added CMD_BUSFREE command handler
;*
;*               2005-06-08  Michael Baeuerle
;*               CMD_PUT_MESSAGE: 256Byte message handling fixed
;*
;*               2005-06-12  Michael Baeuerle
;*               Bugfix: CMD_PUT_DATA: Wait for data size loop fixed
;*
;*               2005-06-19  Michael Baeuerle
;*               Bugfix: CMD_PUT_MESSAGE: After some errors, the stack was
;*                corrupted (data on top of return address was not removed)
;*               Bugfix: CMD_GET_DATA: After some errors, the stack was
;*                corrupted (data on top of return address was not removed)
;*               Bugfix: CMD_PUT_DATA: After some errors, the stack was
;*                corrupted (data on top of return address was not removed)
;*
;*               2005-07-09  Michael Baeuerle
;*               Bugfix: CMD_GET_DATA: Do not touch FLOW2 when IRQ is pending
;*
;*               2005-07-14  Michael Baeuerle
;*               Bugfix: CMD_GET_MESSAGE: Check bytes in date buffer with 16Bit
;*               Bugfix: CMD_PUT_MESSAGE: Check bytes in date buffer with 16Bit
;*               Bugfix: CMD_GET_DATA: Check bytes in date buffer with 16Bit
;*               Bugfix: CMD_PUT_DATA: Check bytes in date buffer with 16Bit
;*               Bugfix: All accesses to X must be atomic - Fixed
;*
;*               2005-07-22  Michael Baeuerle
;*               CMD_GET_DATA: Fast multibyte transfer added
;*
;*               2005-08-13  Michael Baeuerle
;*               CMD_GET_DATA: Bugfix: Fast multibyte transfer must wait until
;*                FLOW2 IRQ before waiting for buffer drain
;*               Because the busphase routines no longer clobber INDEX, the
;*                saving of INDEX on the stack has been removed from all
;*                routines
;*
;*               2005-08-14  Michael Baeuerle
;*               CMD_PUT_DATA: Fast multibyte transfer added
;* 
;*
;* To do:        -
;* 
;*******************************************************************************

;*******************************************************************************
;*
;* Toggle FLOW2
;*
;* This macro toggles the FLOW2 bit in STAT
;*
;*******************************************************************************

.macro  CMD_TOGGLE_FLOW2
      sbrc  STAT, FLOW2
      rjmp  CMD_TOGGLE_FLOW2_1
      sbr  STAT, (1 << FLOW2)
      rjmp  CMD_TOGGLE_FLOW2_2
   CMD_TOGGLE_FLOW2_1:
      cbr  STAT, (1 << FLOW2)
   CMD_TOGGLE_FLOW2_2:
.endmacro


;*******************************************************************************
;*
;* CONFIGURE command
;*
;* This command must be issued first to enable all other commands
;* The PIA will store the committed SCSI address and configuration, enter the
;* BUS-FREE phase, enable the PHY and return OK if no error occur, 1 otherwise.
;*
;* Stack: 4 Bytes
;* Call : PHASE_BUS_FREE [Function]
;*        RD_BUFFER [Function]
;* Macro: -
;* Read : CONTROL [Register]
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        PARAMETER [Register]
;*        ERRORCODE [High register]
;*        STAT [High register]
;*        ID [Register]
;*        BUS_PHASE [High register]
;*        TMP [High register]
;* Interrupt latency: Indirect
;*
;*******************************************************************************

CMD_CONFIGURE:
      ;Wait for SCSI address
      ldi  RESULT, 0x01            ;Prepare for error
      ldi  ERRORCODE, ABORTED
      tst  CONTROL                 ;Check for command abort
      breq  CMD_CONFIGURE_EXIT     ;=> Abort command, return error
      call  RD_BUFFER              ;Check for data available
      tst  RESULT
      brne  CMD_CONFIGURE          ;=> No data, wait

      ;Check for new SCSI address to be valid
      ldi  RESULT, 0x01            ;Prepare for error
      ldi  ERRORCODE, INVALID
      ldi  TMP, 0x08
      cp  PARAMETER, TMP
      brsh  CMD_CONFIGURE_EXIT     ;=> Illegal SCSI ID, return error

      ;Set new SCSI address
      clr  ID                      ;Copy SCSI address to ID (bit position coded)
      inc  ID
   CMD_CONFIGURE_L:
      tst  PARAMETER
      breq  CMD_CONFIGURE_LE
      dec  PARAMETER
      lsl  ID
      rjmp  CMD_CONFIGURE_L
   CMD_CONFIGURE_LE:

      ;Wait for configuration byte
   CMD_CONFIGURE_WAIT:
      ldi  RESULT, 0x01            ;Prepare for error
      ldi  ERRORCODE, ABORTED
      tst  CONTROL                 ;Check for command abort
      breq  CMD_CONFIGURE_EXIT     ;=> Abort command, return error
      call  RD_BUFFER              ;Check for data available
      tst  RESULT
      brne  CMD_CONFIGURE_WAIT     ;=> No data, wait

      ;Set new configuration
      mov  CONFIG, PARAMETER       ;Copy configuration byte to CONFIG register

      ;Switch to BUS-FREE phase
      call  PHASE_BUS_FREE         ;Switch to BUS-FREE phase
      tst  RESULT
      brne  CMD_CONFIGURE_EXIT     ;=> Switch to BUS-FREE failed, return error

      ;Enable transceivers
      sbi  MISC, EN_PHY

      ;Unmask PC interrupt for BSY and ATN pins
      ldi  TMP, (1 << PCINT0) | (1 << PCINT4)
      sts  PCMSK0, TMP 

      ;PIA is now ready
      sbr  STAT, (1 << ENABLE)     ;Set ENABLE flag
      ldi  RESULT, 0x00            ;Return success
   CMD_CONFIGURE_EXIT:
      ret


;*******************************************************************************
;*
;* RECOVER command
;*
;* Calling it after an error forces the data buffer into sane state (main task
;* command handling code clears the data buffer after every command)
;* This command also clears the FLOW and FLOW2 flags in the status register.
;*
;* This command cannot fail.
;*
;* Stack: 2 Bytes
;* Call : -
;* Macro: -
;* Read : -
;* Write: RESULT [High register] (0: OK, 1: Error)
;+        STAT [High register]
;* Interrupt latency: -
;*
;*******************************************************************************

CMD_RECOVER:
      cbr  STAT, (1 << FLOW)       ;Clear FLOW flag
      cbr  STAT, (1 << FLOW2)      ;Clear FLOW flag
      ldi  RESULT, 0x00            ;Return success
      ret


;*******************************************************************************
;*
;* ACCEPT_SELECTION command
;*
;* This command accepts the selection from an inititor.
;*
;* Stack: 4 Bytes
;* Call : PHASE_SELECTION [Function]
;*        WR_BUFFER [Function]
;* Macro: -
;* Read : CONTROL [Register]
;*        XL (Byte count in data buffer)
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        ERRORCODE [High register]
;*        PARAMETER [High register] (SCSI address of initiator)
;*        STAT [High register]
;* Interrupt latency: Indirect
;*
;*******************************************************************************

CMD_ACCEPT_SELECTION:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_ACCEPT_SELECTION_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      ret
   CMD_ACCEPT_SELECTION_OK:

      ;Handle SELECTION phase
      cbr  STAT, (1 << SELECT)     ;Clear SELECT flag
      call  PHASE_SELECTION
      tst  RESULT
      brne  CMD_ACCEPT_SELECTION_EXIT  ;=> SELECTION phase failed, return error

      ;Copy initiator address to data buffer
      call  WR_BUFFER              ;Write initiator address to data buffer
      ;tst  RESULT                 ;Should never fail
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt (Data available)
      
      ;Wait for data buffer to become empty
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01
   CMD_ACCEPT_SELECTION_WAIT:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_ACCEPT_SELECTION_EXIT  ;=> Abort command, return error
      tst  XL
      brne  CMD_ACCEPT_SELECTION_WAIT
      cbr  STAT, (1 << FLOW)       ;Clear FLOW flag
      ldi  RESULT, 0x00            ;Return success
   CMD_ACCEPT_SELECTION_EXIT:
      ret


;*******************************************************************************
;*
;* GET_MESSAGE command
;*
;* This command transfers a message from the initiator to us (the target).
;*
;* Stack: 5 Bytes
;* Call : PHASE_MESSAGE_OUT [Function]
;*        WR_BUFFER [Function]
;* Macro: -
;* Read : CONTROL [Register]
;*        X (Byte count in data buffer)
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        STAT [High register]
;*        ERRORCODE [High register]
;*        PARAMETER [High register]
;*        TMP [High register]
;*        INDEX [Register]
;* Interrupt latency: 4 cycles and Indirect
;*
;*******************************************************************************

CMD_GET_MESSAGE:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_GET_MESSAGE_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      ret
   CMD_GET_MESSAGE_OK:

      ;Get first byte of message
      call  PHASE_MESSAGE_OUT
      tst  RESULT
      brne  CMD_GET_MESSAGE_EXIT   ;=> MESSAGE_OUT phase failed, return error
      call  WR_BUFFER              ;Store first message byte
      ;tst  RESULT                 ;Should never fail

      ;Check for extended message (because 0x01 is in the range 0x00 - 0x1F)
      ldi  TMP, 0x01
      cp  PARAMETER, TMP
      breq  CMD_GET_MESSAGE_EXTENDED

      ;Check for single byte message
      ldi  TMP, 0x80               ;Check for IDENTIFY message
      cp  PARAMETER, TMP
      brsh  CMD_GET_MESSAGE_READY
      ldi  TMP, 0x20               ;0x01 already masked (see above)
      cp  PARAMETER, TMP
      brlo  CMD_GET_MESSAGE_READY

      ;Get second byte of two byte message
      call  PHASE_MESSAGE_OUT
      tst  RESULT
      brne  CMD_GET_MESSAGE_EXIT   ;=> MESSAGE_OUT phase failed, return error
      call  WR_BUFFER              ;Store second message byte
      ;tst  RESULT                 ;Should never fail
      rjmp  CMD_GET_MESSAGE_READY

      ;Handle extended message
   CMD_GET_MESSAGE_EXTENDED:
      call  PHASE_MESSAGE_OUT      ;Get message length
      tst  RESULT
      brne  CMD_GET_MESSAGE_EXIT   ;=> MESSAGE_OUT phase failed, return error
      call  WR_BUFFER              ;Store extended message length
      ;tst  RESULT                 ;Should never fail
      mov  INDEX, PARAMETER        ;Get rest of message
   CMD_GET_MESSAGE_EXTENDED_LOOP:
      call  PHASE_MESSAGE_OUT      ;Get message byte
      tst  RESULT
      brne  CMD_GET_MESSAGE_EXIT   ;=> MESSAGE_OUT phase failed, return error
      call  WR_BUFFER              ;Store message byte
      ;tst  RESULT                 ;Should never fail
      dec  INDEX                   ;Also works for 256 Bytes (using wraparound)
      cpse  INDEX, NULL            ;Check for last message byte
      rjmp  CMD_GET_MESSAGE_EXTENDED_LOOP

      ;Create IRQ to indicate that message is ready to read
   CMD_GET_MESSAGE_READY:
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt (Data available)

      ;Wait for data buffer to become empty
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01
   CMD_GET_MESSAGE_WAIT:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_MESSAGE_EXIT   ;=> Abort command, return error
      cli                          ;Disable interrupts
      cp  XL, NULL                 ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      brne  CMD_GET_MESSAGE_WAIT
      cbr  STAT, (1 << FLOW)       ;Clear FLOW flag
      ldi  RESULT, 0x00            ;Return success
   CMD_GET_MESSAGE_EXIT:
      ret


;*******************************************************************************
;*
;* PUT_MESSAGE command
;*
;* This command transfers a message from us (the target) to an initiator.
;* Message transmission starts when the complete message is in the local buffer.
;*
;* Stack: 6 Bytes
;* Call : PHASE_MESSAGE_IN [Function]
;*        RD_BUFFER [Function]
;* Macro: -
;* Read : CONTROL [Register]
;*        STAT [Register]
;*        X (Byte count in data buffer)
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        ERRORCODE [High register]
;*        PARAMETER [High register]
;*        TMP [High register]
;*        INDEX [Register]
;* Interrupt latency: 4 cycles and Indirect
;*
;*******************************************************************************

CMD_PUT_MESSAGE:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_PUT_MESSAGE_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      ret
   CMD_PUT_MESSAGE_OK:

      ;Wait for data
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01
   CMD_PUT_MESSAGE_WAIT:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_MESSAGE_EXIT2  ;=> Abort command, return error
      cli                          ;Disable interrupts
      cp  XL, NULL                 ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      breq  CMD_PUT_MESSAGE_WAIT

      ;Get first byte of message
      call  RD_BUFFER              ;Read first message byte
      ;tst  RESULT                 ;Should never fail

      ;Check for extended message
      ldi  TMP, 0x01
      cp  PARAMETER, TMP
      breq  CMD_PUT_MESSAGE_EXTENDED

      ;Check for single byte message (0x01 already checked))
      clr  INDEX                   ;Init index
      ldi  TMP, 0x80               ;Check for IDENTIFY message
      cp  PARAMETER, TMP
      brsh  CMD_PUT_MESSAGE_LOOP
      ldi  TMP, 0x20               ;0x01 already masked
      cp  PARAMETER, TMP
      brlo  CMD_PUT_MESSAGE_LOOP

      ;Wait for second byte of two byte message
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01
   CMD_PUT_MESSAGE_WAIT2:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_MESSAGE_EXIT2  ;=> Abort command, return error
      cli                          ;Disable interrupts
      cp  XL, NULL                 ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      breq  CMD_PUT_MESSAGE_WAIT2
      ldi  TMP, 0x01               ;Init index
      mov  INDEX, TMP
      rjmp  CMD_PUT_MESSAGE_LOOP   ;Send message

      ;Jump pad to solve branch overflows
   CMD_PUT_MESSAGE_EXIT2:
      rjmp  CMD_PUT_MESSAGE_EXIT

      ;Wait for extended message
   CMD_PUT_MESSAGE_EXTENDED:
      push  PARAMETER              ;Save message code
      ldi  ERRORCODE, ABORTED      ;Wait for next byte (message length)
      ldi  RESULT, 0x01
   CMD_PUT_MESSAGE_WAIT3:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_MESSAGE_POP_EXIT  ;=> Abort command, return error
      cli                          ;Disable interrupts
      cp  XL, NULL                 ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      breq  CMD_PUT_MESSAGE_WAIT3
      call  RD_BUFFER              ;Read length of extended message
      ;tst  RESULT                 ;Should never fail
      clr  INDEX                   ;Check for 256Byte message
      ldi  TMP, 0x01
      cpse  PARAMETER, INDEX
      ldi  TMP, 0x00
      ldi  ERRORCODE, ABORTED      ;Wait for rest of extended message
      ldi  RESULT, 0x01
   CMD_PUT_MESSAGE_WAIT4:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_MESSAGE_POP_EXIT  ;=> Abort command, return error
      cli                          ;Disable interrupts
      cp  XL, PARAMETER            ;This 16Bit access must be atomic!
      cpc  XH, TMP
      sei                          ;Reenable interrupts
      brlo  CMD_PUT_MESSAGE_WAIT4
      mov  INDEX, PARAMETER        ;Init index (wraparound work for 256Byte)
      dec  INDEX

      ;Send extended message header (0x01 + Length)
      mov  TMP, PARAMETER          ;Send buffered message code on stack first
      pop  PARAMETER               ;Pop message code from stack
      push  TMP                    ;Push length on stack
      call  PHASE_MESSAGE_IN
      tst  RESULT
      brne  CMD_PUT_MESSAGE_POP_EXIT  ;=> MESSAGE_IN phase failed, return error
      pop  PARAMETER               ;Send buffered length on stack now
      call  PHASE_MESSAGE_IN      
      tst  RESULT
      brne  CMD_PUT_MESSAGE_EXIT   ;=> MESSAGE_IN phase failed, return error

      ;Prepare first byte of extended message
      call  RD_BUFFER              ;Read extended message code
      ;tst  RESULT                 ;Should never fail

      ;Send message loop (Sends INDEX + 1 Bytes)
   CMD_PUT_MESSAGE_LOOP:
      call  PHASE_MESSAGE_IN       ;Send byte
      tst  RESULT
      brne  CMD_PUT_MESSAGE_EXIT   ;=> MESSAGE_IN phase failed, return error
      tst  INDEX                   ;Check for last byte of message
      breq  CMD_PUT_MESSAGE_FINISHED
      dec  INDEX
      call  RD_BUFFER              ;Get next byte
      ;tst  RESULT                 ;Should never fail
      rjmp  CMD_PUT_MESSAGE_LOOP
   CMD_PUT_MESSAGE_FINISHED:

      ;Check for oversized message (Buffer must be empty now)
      ldi  ERRORCODE, DATA_LENGTH
      ldi  RESULT, 0x01
      cli                          ;Disable interrupts
      cp  XL, NULL                 ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      brne  CMD_PUT_MESSAGE_EXIT   ;=> Oversized message, return error
      ldi  RESULT, 0x00            ;Return success
      rjmp  CMD_PUT_MESSAGE_EXIT

   CMD_PUT_MESSAGE_POP_EXIT:
      pop  TMP                     ;Kick data from stack

   CMD_PUT_MESSAGE_EXIT:
      ret


;*******************************************************************************
;*
;* GET_COMMAND command
;*
;* This command transfers a CDB from an initiator to us (the target).
;*
;* Stack: 5 Bytes
;* Call : PHASE_COMMAND [Function]
;*        WD_BUFFER [Function]
;* Macro: -
;* Read : CONTROL [Register]
;*        XL (Byte count in data buffer)
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        ERRORCODE [High register]
;*        PARAMETER [High register]
;*        STAT [High register]
;*        TMP [High register]
;*        INDEX [Register]
;* Interrupt latency: Indirect
;*
;*******************************************************************************

CMD_GET_COMMAND:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_GET_COMMAND_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      rjmp  CMD_GET_COMMAND_EXIT
   CMD_GET_COMMAND_OK:

      ;Get first byte of command
      call  PHASE_COMMAND
      tst  RESULT
      brne  CMD_GET_COMMAND_EXIT   ;=> COMMAND phase failed, return error
      call  WR_BUFFER              ;Store first message byte
      ;tst  RESULT                 ;Should never fail

      ;Check for command group
      mov  TMP, PARAMETER
      swap  TMP
      lsr  TMP
      andi  TMP, 0x07
      tst  TMP
      breq  CMD_GET_COMMAND_6
      cpi  TMP, 0x03
      brlo  CMD_GET_COMMAND_10
      cpi  TMP, 0x04
      breq  CMD_GET_COMMAND_16
      cpi  TMP, 0x05
      breq  CMD_GET_COMMAND_12
      ldi  ERRORCODE, PROTOCOL     ;Reserved or unsupported command group
      ldi  RESULT, 0x01            ;=> Return error
      rjmp  CMD_GET_COMMAND_EXIT

      ;Init index with command length
   CMD_GET_COMMAND_6:
      ldi  TMP, 0x06
      rjmp  CMD_GET_COMMAND_READY
   CMD_GET_COMMAND_10:
      ldi  TMP, 0x0A
      rjmp  CMD_GET_COMMAND_READY
   CMD_GET_COMMAND_12:
      ldi  TMP, 0x0C
      rjmp  CMD_GET_COMMAND_READY
   CMD_GET_COMMAND_16:
      ldi  TMP, 0x10
   CMD_GET_COMMAND_READY:
      mov  INDEX, TMP
      dec  INDEX                  ;First byte already read

      ;Get rest of command
   CMD_GET_COMMAND_LOOP:
      call  PHASE_COMMAND          ;Get command byte
      tst  RESULT
      brne  CMD_GET_COMMAND_EXIT   ;=> Command phase failed, return error
      call  WR_BUFFER              ;Store command byte
      ;tst  RESULT                 ;Should never fail
      dec  INDEX
      cpse  INDEX, NULL            ;Check for last command byte
      rjmp  CMD_GET_COMMAND_LOOP

      ;Create IRQ to indicate that message is ready to read
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt (Data available)

      ;Wait for data buffer to become empty
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01
   CMD_GET_COMMAND_WAIT:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_COMMAND_EXIT   ;=> Abort command, return error
      tst  XL
      brne  CMD_GET_COMMAND_WAIT
      cbr  STAT, (1 << FLOW)       ;Clear FLOW flag
      ldi  RESULT, 0x00            ;Return success
   CMD_GET_COMMAND_EXIT:
      ret


;*******************************************************************************
;*
;* PUT_STATUS command
;*
;* This command transfers a status byte from us (the target) to an initiator.
;*
;* Stack: 2 Bytes
;* Call : PHASE_STATUS [Function]
;*        RD_BUFFER [Function]
;* Macro: -
;* Read : STAT [Register]
;*        CONTROL [Register]
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        ERRORCODE [High register]
;*        PARAMETER [High register]
;* Interrupt latency: Indirect
;*
;*******************************************************************************

CMD_PUT_STATUS:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_PUT_STATUS_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      ret
   CMD_PUT_STATUS_OK:

      ;Wait for status byte
   CMD_PUT_STATUS_LOOP:
      ldi  RESULT, 0x01            ;Prepare for error
      ldi  ERRORCODE, ABORTED
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_STATUS_EXIT    ;=> Abort command, return error
      call  RD_BUFFER              ;Check for data available
      tst  RESULT
      brne  CMD_PUT_STATUS_LOOP    ;=> No data, wait

      ;Send status
      call  PHASE_STATUS
      tst  RESULT
      brne  CMD_PUT_STATUS_EXIT    ;=> STATUS phase failed, return error
      ldi  RESULT, 0x00            ;Return success
   CMD_PUT_STATUS_EXIT:
      ret


;*******************************************************************************
;*
;* GET_DATA command
;*
;* This command transfers data from the initiator to us (the target).
;*
;* Stack: 5 Bytes
;* Call : PHASE_DATA_OUT [Function]
;*        RD_BUFFER [Function]
;*        WR_BUFFER [Function]
;* Macro: CMD_TOGGLE_FLOW2
;* Read : CONTROL [Register]
;*        X (Byte count in data buffer)
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        STAT [High register]
;*        ERRORCODE [High register]
;*        PARAMETER [High register]
;*        TMP [High register]
;*        INDEX [Register]
;*        T [Flag]
;*        LEN [16Bit capable register pair]
;* Interrupt latency: 4 cycles and Indirect
;*
;*******************************************************************************

CMD_GET_DATA:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_GET_DATA_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      ret
   CMD_GET_DATA_OK:

      ;Wait for parameter
   CMD_GET_DATA_WAIT1:
      tst  CONTROL                 ;Check for command abort
      brne  CMD_GET_DATA_OK2
      rjmp  CMD_GET_DATA_ABORTED   ;=> Abort command, return error
   CMD_GET_DATA_OK2:
      cli                          ;Disable interrupts
      cpi  XL, 0x02                ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      brlo  CMD_GET_DATA_WAIT1

      ;Copy data length (big endian) to LEN
      call  RD_BUFFER              ;Read data length high byte
      ;tst  RESULT                 ;Should never fail
      mov  LENH, PARAMETER
      call  RD_BUFFER              ;Read data length low byte
      ;tst  RESULT                 ;Should never fail
      mov  LENL, PARAMETER

      ;Init flow control
      cbr  STAT, (1 << FLOW) | (1 << FLOW2)

      ;Check for data length to be equal or larger than 256
      tst  LENH
      brne  CMD_GET_DATA_MULTI
      cpse  LENL, NULL
      rjmp  CMD_GET_DATA_SINGLE

      ;-------------------------------------------------------------------------

      ;Fast mutibyte transfer
   CMD_GET_DATA_MULTI:
      clt                          ;Init half-block flag

      ;Get first byte (Bus phase change and phase sequence check)
      call  PHASE_DATA_OUT         ;Get byte from initiator
      cpse  RESULT, NULL
      rjmp  CMD_GET_DATA_EXIT2     ;=> DATA_OUT phase failed, return error
      call  WR_BUFFER              ;Write byte to data buffer
      ;tst  RESULT                 ;Should never fail
      mov  INDEX, NULL             ;Init byte counter
      dec  INDEX

      ;Drive REQ (continuously for fast data out phases)
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      sbi  D_SCB, REQ              ;Drive PHY line for REQ Pin
      sbi  T_SCB, REQ              ;Drive REQ
      sei                          ;Reenable interrupts

      ;Get half block (256Byte) and write it to buffer
   CMD_GET_DATA_LOOP2:
      PHASE_DATA_OUT_FAST          ;Get byte from initiator
      tst  RESULT
      brne  CMD_GET_DATA_EXIT2     ;=> DATA_OUT phase failed, return error
      call  WR_BUFFER              ;Write byte to data buffer
      ;tst  RESULT                 ;Should never fail
      dec  INDEX
      brne  CMD_GET_DATA_LOOP2     ;Check for last byte of half block

      ;Set FLOW flag if necessary
      sbrc  STAT, FLOW
      rjmp  CMD_GET_DATA_FLOW_END2
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt
   CMD_GET_DATA_FLOW_END2:

      ;Check whether flow control IRQ is required
      ldi  TMP, 0x00               ;Clear FLOW2 toggle event
      brts  CMD_GET_DATA_INT2      ;=> Every 512 Bytes
      set                          ;Set half-block flag
      rjmp  CMD_GET_DATA_CONTINUE2

      ;Toggle FLOW2 flag and generate IRQ every 512 Bytes
   CMD_GET_DATA_INT2:              ;This is executed every 512 Bytes
      clt                          ;Clear half-block flag
      ldi  ERRORCODE, ABORTED      ;Prepare error code
      ldi  RESULT, 0x01            ;Prepare result
   CMD_GET_DATA_INT_WAIT3:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_DATA_EXIT2     ;=> Abort command, return error
      sbis  PCB, PIA_IRQ           ;Check for pending IRQ
      rjmp  CMD_GET_DATA_INT_WAIT3 ;=> Yes, wait until serviced
      CMD_TOGGLE_FLOW2             ;Toggle FLOW2 bit
      ldi  TMP, 0x01               ;Store FLOW2 toggle event
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt

      ;Wait for data buffer have drained below 256 bytes
      ldi  ERRORCODE, ABORTED      ;Prepare error code
      ldi  RESULT, 0x01            ;Prepare result
   CMD_GET_DATA_WAIT:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_DATA_EXIT2     ;=> Abort command, return error
      tst  XH                      ;Check for less than 256 bytes in buffer
      brne  CMD_GET_DATA_WAIT      ;=> No, wait until buffer have drained
   CMD_GET_DATA_CONTINUE2:

      ;Check for last half-block
      dec  LENH                    ;Decrement remaining data length
      breq  CMD_GET_DATA_CHECK     ;=> Last, check remaining data length
      mov  INDEX, NULL             ;Init byte counter for next half-block
      rjmp  CMD_GET_DATA_LOOP2

      ;No more half-blocks to transfer
   CMD_GET_DATA_CHECK:
      tst  LENL                    ;Check for remaining data length
      brne  CMD_GET_DATA_SINGLE    ;=> More data to transfer

      ;No more data to transfer
      ;Release REQ (to terminate fast data out phases)
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin
      sei                          ;Reenable interrupts
      rjmp  CMD_GET_DATA_LAST_IRQ  ;(FLOW2 toggle event in TMP is valid)

   CMD_GET_DATA_EXIT2:
      ;Release REQ (after error in fast DATA_OUT phase)
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin
      sei                          ;Reenable interrupts
      rjmp  CMD_GET_DATA_EXIT

      ;-------------------------------------------------------------------------

      ;Normal single byte transfer
   CMD_GET_DATA_SINGLE:

      ;Init byte counter
      clr  INDEX                   ;Reset byte counter
      clt                          ;Init half-block flag

      ;Get one byte and copy it to data buffer
   CMD_GET_DATA_LOOP:
      call  PHASE_DATA_OUT         ;Get byte from initiator
      tst  RESULT
      brne  CMD_GET_DATA_EXIT      ;=> DATA_OUT phase failed, return error
   CMD_GET_DATA_WR_RETRY:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_DATA_ABORTED   ;=> Abort command, return error
      call  WR_BUFFER              ;Write byte to data buffer
      tst  RESULT
      brne  CMD_GET_DATA_WR_RETRY  ;Retry until success
      inc  INDEX                   ;Increment byte counter

      ;Set FLOW flag if necessary
      sbrc  STAT, FLOW
      rjmp  CMD_GET_DATA_FLOW_END
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt
   CMD_GET_DATA_FLOW_END:

      ;Check whether flow control IRQ is required
      ldi  TMP, 0x00               ;Clear FLOW2 toggle event
      tst  INDEX                   ;Check whether it is time for IRQ
      brne  CMD_GET_DATA_CONTINUE
      brts  CMD_GET_DATA_INT       ;=> Every 512 Bytes
      set                          ;Set half-block flag
      rjmp  CMD_GET_DATA_CONTINUE

      ;Toggle FLOW2 flag and generate IRQ every 512 Bytes
   CMD_GET_DATA_INT:               ;This is executed every 512 Bytes
      clt                          ;Clear half-block flag
   CMD_GET_DATA_INT_WAIT:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_DATA_ABORTED   ;=> Abort command, return error
      sbis  PCB, PIA_IRQ           ;Check for pending IRQ
      rjmp  CMD_GET_DATA_INT_WAIT  ;=> Yes, wait until serviced
      CMD_TOGGLE_FLOW2             ;Toggle FLOW2 bit
      ldi  TMP, 0x01               ;Store FLOW2 toggle event
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt
   CMD_GET_DATA_CONTINUE:

      ;Check for last byte (return to loop if not)
      sbiw  LENL, 0x01             ;Decrement remaining data length
      cp  LENL, NULL               ;Check for last byte
      cpc  LENH, NULL
      brne  CMD_GET_DATA_LOOP

      ;-------------------------------------------------------------------------

      ;Last flow control IRQ (if required)
   CMD_GET_DATA_LAST_IRQ:
      tst  TMP                     ;Check whether additional IRQ is required
      brne  CMD_GET_DATA_INT_EXIT  ;No => Skip
   CMD_GET_DATA_INT_WAIT2:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_DATA_ABORTED   ;=> Abort command, return error
      sbis  PCB, PIA_IRQ           ;Check for pending IRQ
      rjmp  CMD_GET_DATA_INT_WAIT2 ;=> Yes, wait until serviced
      CMD_TOGGLE_FLOW2             ;Toggle FLOW2 bit
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt
   CMD_GET_DATA_INT_EXIT:
      rjmp CMD_GET_DATA_WAIT2

      ;Command aborted
   CMD_GET_DATA_ABORTED:
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01
      rjmp CMD_GET_DATA_EXIT

      ;Wait for data buffer to become empty
   CMD_GET_DATA_WAIT2:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_GET_DATA_ABORTED   ;=> Abort command, return error
      cli                          ;Disable interrupts
      cp   XL, NULL                ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      brne  CMD_GET_DATA_WAIT2
      cbr  STAT, (1 << FLOW) | (1 << FLOW2)  ;Clear FLOW and FLOW2 flags
      ldi  RESULT, 0x00            ;Return success

      ;Exit
   CMD_GET_DATA_EXIT:
      ret


;*******************************************************************************
;*
;* PUT_DATA command
;*
;* This command transfers data from us (the target) to the initiator.
;*
;* Stack: 5 Bytes
;* Call : PHASE_DATA_IN [Function]
;*        RD_BUFFER [Function]
;*        X (Byte count in data buffer)
;* Macro: CMD_TOGGLE_FLOW2
;* Read : CONTROL [Register]
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        STAT [High register]
;*        ERRORCODE [High register]
;*        PARAMETER [High register]
;*        INDEX [Register]
;*        T [Flag]
;*        LEN [16Bit capable register pair]
;* Interrupt latency: 4 cycles and Indirect
;*
;*******************************************************************************

CMD_PUT_DATA:
      ;Check for PIA is enabled
      sbrc  STAT, ENABLE
      rjmp  CMD_PUT_DATA_OK
      ldi  ERRORCODE, INVALID      ;=> PIA not configured, return error
      ldi  RESULT, 0x01
      ret
   CMD_PUT_DATA_OK:

      ;Wait for parameters
   CMD_PUT_DATA_WAIT:
      cpse  CONTROL, NULL          ;Check for command abort
      cpse  NULL, NULL
      rjmp  CMD_PUT_DATA_ABORTED   ;=> Abort command, return error
      cli                          ;Disable interrupts
      cpi  XL, 0x02                ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      brlo  CMD_PUT_DATA_WAIT

      ;Copy data length (big endian) to LEN
      call  RD_BUFFER              ;Read data length high byte
      ;tst  RESULT                 ;Should never fail
      mov  LENH, PARAMETER
      call  RD_BUFFER              ;Read data length low byte
      ;tst  RESULT                 ;Should never fail
      mov  LENL, PARAMETER

      ;Init flow control
      sbr  STAT, (1 << FLOW)
      cbr  STAT, (1 << FLOW2)
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt

      ;Init byte counter
      clr  INDEX                   ;Reset byte counter
      set                          ;Set half-block flag
                                   ;(Allows the target to send 768 bytes)

      ;Check for data length to be equal or larger than 256
      tst  LENH
      brne  CMD_PUT_DATA_MULTI
      cpse  LENL, NULL
      rjmp  CMD_PUT_DATA_SINGLE

      ;-------------------------------------------------------------------------

      ;Fast mutibyte transfer
   CMD_PUT_DATA_MULTI:

      ;Send first byte (Bus phase change and phase sequence check)
   CMD_PUT_DATA_RD_RETRY2:
      cpse  CONTROL, NULL          ;Check for command abort
      cpse  NULL, NULL
      rjmp  CMD_PUT_DATA_ABORTED   ;=> Abort command, return error
      call  RD_BUFFER              ;Read byte from data buffer
      tst  RESULT
      brne  CMD_PUT_DATA_RD_RETRY2 ;Retry until success
      call  PHASE_DATA_IN          ;Send byte to initiator
      cpse  RESULT, NULL
      rjmp  CMD_PUT_DATA_EXIT      ;=> DATA_IN phase failed, return error
      inc  INDEX                   ;Increment byte counter

      ;Drive REQ (continuously for fast DATA_IN phases)
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      sbi  D_SCB, REQ              ;Drive PHY line for REQ Pin
      sbi  T_SCB, REQ              ;Drive REQ
      sei                          ;Reenable interrupts

   CMD_PUT_DATA_LOOP3:

      ;Wait for 256 bytes of data present in buffer 
      ldi  ERRORCODE, ABORTED      ;Prepare error code
      ldi  RESULT, 0x01            ;Prepare result
   CMD_PUT_DATA_WAIT1:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_DATA_EXIT2     ;=> Abort command, return error
      tst  XH                      ;Check for more than 256 bytes in buffer
      breq  CMD_PUT_DATA_WAIT1     ;=> No, wait until buffer have filled

      ;Read half block (256Byte) from buffer and send it
   CMD_PUT_DATA_LOOP2:
      call  RD_BUFFER              ;Read byte from data buffer
      ;tst  RESULT                 ;Should never fail
      PHASE_DATA_IN_FAST           ;Send byte to initiator
      tst  RESULT
      brne  CMD_PUT_DATA_EXIT2     ;=> DATA_IN phase failed, return error
      inc  INDEX
      brne  CMD_PUT_DATA_LOOP2     ;Check for last byte of half block

      ;Set FLOW flag if necessary
      cli                          ;Disable interrupts
      cp  XL, BSL                  ;Check for data buffer is full
      cpc  XH, BSH
      sei                          ;Reenable interrupts
      brsh  CMD_PUT_DATA_CHECK_INT2
      sbrc  STAT, FLOW             ;Check whether FLOW is already set
      rjmp  CMD_PUT_DATA_CHECK_INT2
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt

      ;Check whether flow control IRQ is required
   CMD_PUT_DATA_CHECK_INT2:
      brts  CMD_PUT_DATA_INT2      ;=> Every 512 Bytes
      set                          ;Set half-block flag
      rjmp  CMD_PUT_DATA_CONTINUE3

      ;Toggle FLOW2 flag and generate IRQ every 512 Bytes
   CMD_PUT_DATA_INT2:              ;This is executed every 512 Bytes
      clt                          ;Clear half-block flag
      CMD_TOGGLE_FLOW2             ;Toggle FLOW2 bit
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt
   CMD_PUT_DATA_CONTINUE3:

      ;Check for last half-block
      mov  INDEX, NULL             ;Init byte counter for next half-block
      dec  LENH                    ;Decrement remaining data length
      breq  CMD_PUT_DATA_CHECK     ;=> Last, check remaining data length
      rjmp  CMD_PUT_DATA_LOOP3

      ;No more half blocks to transfer, check for end of data
   CMD_PUT_DATA_CHECK:
      tst  LENL                    ;Check for last byte
      brne  CMD_PUT_DATA_SINGLE

      ;No more data to transfer
      ;Release REQ (to terminate fast data out phases)
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin
      sei                          ;Reenable interrupts
      rjmp  CMD_PUT_DATA_CONTINUE2

   CMD_PUT_DATA_EXIT2:
      ;Release REQ (after error in fast DATA_OUT phase)
      cli                          ;Disable interrupts
      cbi  SCB, REQ                ;Set PHY line for REQ
      cbi  T_SCB, REQ              ;Release REQ
      cbi  D_SCB, REQ              ;Release PHY line for REQ Pin
      sei                          ;Reenable interrupts
      rjmp  CMD_PUT_DATA_EXIT

      ;-------------------------------------------------------------------------

      ;Normal single byte transfer
   CMD_PUT_DATA_SINGLE:

      ;Read one byte from data buffer and send it
   CMD_PUT_DATA_LOOP:
   CMD_PUT_DATA_RD_RETRY:
      tst  CONTROL                 ;Check for command abort
      breq  CMD_PUT_DATA_ABORTED   ;=> Abort command, return error
      call  RD_BUFFER              ;Read byte from data buffer
      tst  RESULT
      brne  CMD_PUT_DATA_RD_RETRY  ;Retry until success
      call  PHASE_DATA_IN          ;Send byte to initiator
      tst  RESULT
      brne  CMD_PUT_DATA_EXIT      ;=> DATA_IN phase failed, return error

      ;Set FLOW flag if necessary
      cli                          ;Disable interrupts
      cp  XL, BSL                  ;Check for data buffer is full
      cpc  XH, BSH
      sei                          ;Reenable interrupts
      brsh  CMD_PUT_DATA_CHECK_INT
      sbrc  STAT, FLOW             ;Check whether FLOW is already set
      rjmp  CMD_PUT_DATA_CHECK_INT
      sbr  STAT, (1 << FLOW)       ;Set FLOW flag
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt

      ;Check whether flow control IRQ is required
   CMD_PUT_DATA_CHECK_INT:
      inc  INDEX                   ;Check whether it is time for IRQ
      tst  INDEX
      brne  CMD_PUT_DATA_CONTINUE
      brts  CMD_PUT_DATA_INT       ;=> Every 512 Bytes
      set                          ;Set half-block flag
      rjmp  CMD_PUT_DATA_CONTINUE

      ;Toggle FLOW2 flag and generate IRQ every 512 Bytes
   CMD_PUT_DATA_INT:               ;This is executed every 512 Bytes
      clt                          ;Clear half-block flag
      CMD_TOGGLE_FLOW2             ;Toggle FLOW2 bit
      cbi  PCB, PIA_IRQ            ;Generate PIA interrupt

      ;Check for last byte (return to loop if not)
   CMD_PUT_DATA_CONTINUE:
      sbiw  LENL, 0x01             ;Decrement remaining data length
      cp  LENL, NULL               ;Check for last byte
      cpc  LENH, NULL
      brne  CMD_PUT_DATA_LOOP

      ;-------------------------------------------------------------------------

   CMD_PUT_DATA_CONTINUE2:
      ;Reset flow control
      cbr  STAT, (1 << FLOW) | (1 << FLOW2)  ;Clear FLOW and FLOW2 flags

      ;Check for correct data length
      ldi  ERRORCODE, DATA_LENGTH
      ldi  RESULT, 0x01
      cli                          ;Disable interrupts
      cp  XL, NULL                 ;This 16Bit access must be atomic!
      cpc  XH, NULL
      sei                          ;Reenable interrupts
      brne  CMD_PUT_DATA_EXIT      ;=> Wrong data length, return error
      ldi  RESULT, 0x00            ;Return success
      rjmp  CMD_PUT_DATA_EXIT

      ;Command aborted
   CMD_PUT_DATA_ABORTED:
      ldi  ERRORCODE, ABORTED
      ldi  RESULT, 0x01

      ;Exit
   CMD_PUT_DATA_EXIT:
      ret


;*******************************************************************************
;*
;* BUSFREE command
;*
;* This command switch to the BUS_FREE phase
;*
;* Stack: 2 Bytes
;* Call : PHASE_BUS_FREE [Function]
;* Macro: -
;* Read : -
;* Write: RESULT [High register] (0: OK, 1: Error)
;*        STAT [High register]
;* Interrupt latency: Indirect
;*
;*******************************************************************************

CMD_BUSFREE:
      ;Switch to BUS-FREE phase
      call  PHASE_BUS_FREE         ;Switch to BUS-FREE phase

      ;Reset flow control
      cbr  STAT, (1 << FLOW) | (1 << FLOW2)  ;Clear FLOW and FLOW2 flags
      ret


;EOF
