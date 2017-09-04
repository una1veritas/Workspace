;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         buffer.s
;*
;* Language:     Assembly
;*
;* Description:  This is the data buffer handling code.
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
;* Changelog:    2004-10-16  Michael Baeuerle
;*               Added CLEAR_BUFFER routine
;*               Added WR_BUFFER routine
;*               Added RD_BUFFER routine
;*               Added I_WR_BUFFER macro
;*               Added I_RD_BUFFER macro
;*
;*               2004-10-23  Michael Baeuerle
;*               I_RD_BUFFER: Clear FLOW Bit in SREG if buffer is empty
;*               CLEAR_BUFFER: Restore former interrupt enable state on exit
;*               I_WR_BUFFER: Clear FLOW Bit in SREG if buffer is full
;*
;*               2004-10-24  Michael Baeuerle
;*               Bugfix: CLEAR_BUFFER: PL was initialized with wrong value
;*               Renamed PL to UPL (Upper Pointer Limit)
;*               Use LPL (Lower Pointer Limit) instead of zero
;*               Bugfix: CLEAR_BUFFER: X was initialized with wrong value
;*               Bugfix: RD_BUFFER: Use Y pointer instead of Z
;*               Bugfix: I_RD_BUFFER: Use Y pointer instead of Z
;*
;*               2005-06-05  Michael Baeuerle
;*               Bugfix: I_WR_BUFFER: FLOW flag must be untouched if not full
;*
;*               2005-07-14  Michael Baeuerle
;*               Bugfix: WR_BUFFER: Write pointer handling fixed
;*               Bugfix: I_WR_BUFFER: Write pointer handling fixed
;*               Bugfix: RD_BUFFER: Read pointer handling fixed
;*               Bugfix: I_RD_BUFFER: Read pointer handling fixed
;*               WR_BUFFER: Runtime decreased
;*               RD_BUFFER: Runtime decreased
;*
;* 
;* To do:        -
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Clear FIFO data buffer
;*
;* Buffer access must be atomic
;*
;* Stack: 2 Bytes
;* Call : -
;* Macro: -
;* Read : NULL [Register]
;* Write: TMP [High register]
;*        BSH, BSL [Registers]
;*        LPLH, LPLL [Registers]
;*        UPLH, UPLL [Registers]
;*        X
;*        Y
;*        Z
;* Interrupt latency: 7 Cycles
;*
;*******************************************************************************

CLEAR_BUFFER:
      ;Init buffer size
      ldi  TMP, low(BUFFERSIZE)    ;BS = BUFFERSIZE
      mov  BSL, TMP
      ldi  TMP, high(BUFFERSIZE)
      mov  BSH, TMP

      ;Init Lower Pointer Limit
      ldi  TMP, low(BUFFER)        ;LPL = BUFFER
      mov  LPLL, TMP
      ldi  TMP, high(BUFFER)      
      mov  LPLH, TMP

      ;Init Upper Pointer Limit
      ldi  TMP, low(BUFFER)        ;UPL = BUFFER + BS - 1
      mov  UPLL, TMP
      ldi  TMP, high(BUFFER)      
      mov  UPLH, TMP
      add  UPLL, BSL
      adc  UPLH, BSH
      ldi  TMP, 0x01
      sub  UPLL, TMP
      sbc  UPLH, NULL

      in  TMP, SREG                ;Disable interrupts
      cli
      clr  XL                      ;Clear buffer
      clr  XH
      mov  YL, LPLL
      mov  YH, LPLH
      mov  ZL, LPLL
      mov  ZH, LPLH
      out  SREG, TMP               ;Restore former interrupt enable state
      ret


;*******************************************************************************
;*
;* Write single byte to FIFO data buffer
;*
;* Buffer access must be atomic
;*
;* Stack: 2 Bytes
;* Call : -
;* Macro: -
;* Read : PARAMETER [Register] (Byte to store)
;*        BSH, BSL [Registers] (Data buffer size)
;*        LPLH, LPLL [Registers] (Data buffer pointer lower limit)
;*        UPLH, UPLL [Registers] (Data buffer pointer upper limit)
;* Write: X (Byte count in data buffer)
;*        Z (Data buffer write pointer)
;*        RESULT [High register] (0 on success, 1 if buffer is full)
;* Interrupt latency: 12, 17 or 18 Cycles
;*
;*******************************************************************************

WR_BUFFER:
      cli
      cp  XL, BSL                  ;Check for data buffer to be full
      cpc  XH, BSH
      brsh  WR_BUFFER_FULL
      ldi  RESULT, 0x00            ;Store data
      st  Z+, PARAMETER
      adiw  XL, 0x01

      cp  UPLL, ZL                 ;Manage ring buffer write pointer
      cpc  UPLH, ZH
      brsh  WR_BUFFER_EXIT
      mov  ZL, LPLL
      mov  ZH, LPLH

   WR_BUFFER_EXIT:
      reti                         ;Save one cycle against sei + ret

   WR_BUFFER_FULL:
      ldi  RESULT, 0x01            ;Buffer full
      rjmp  WR_BUFFER_EXIT


;*******************************************************************************
;*
;* Write single byte to FIFO data buffer
;*
;* This macro is for interrupt handlers only!
;* If the buffer is full, the data byte is silently lost
;*
;* Stack: -
;* Call : -
;* Macro: -
;* Read : I_PARAMETER [Register] (Byte to store)
;*        BSH, BSL [Registers] (Data buffer size)
;*        LPLH, LPLL [Registers] (Data buffer pointer lower limit)
;*        UPLH, UPLL [Registers] (Data buffer pointer upper limit)
;* Write: X (Byte count in data buffer)
;*        Z (Data buffer write pointer)
;* Runtime: 4 Cycles (Buffer full), 16 Cycles (Worst case, buffer not full)
;*
;*******************************************************************************

.macro  I_WR_BUFFER
      cp  XL, BSL                  ;Check for data buffer to be full
      cpc  XH, BSH
      brsh  I_WR_BUFFER_EXIT
      st  Z+, I_PARAMETER
      adiw  XL, 0x01

      cp  UPLL, ZL                 ;Manage ring buffer write pointer
      cpc  UPLH, ZH
      brsh  I_WR_BUFFER_RING
      mov  ZL, LPLL
      mov  ZH, LPLH
   I_WR_BUFFER_RING:

      ;Check for buffer full
      cp  XL, BSL                  ;Check for data buffer to be full again
      cpc  XH, BSH
      brlo  I_WR_BUFFER_EXIT       ;Not full => Don't touch FLOW flag
      cbr  STAT, (1 << FLOW)       ;Full => Clear FLOW flag

   I_WR_BUFFER_EXIT:
.endmacro


;*******************************************************************************
;*
;* Read single byte from FIFO data buffer
;*
;* Buffer access must be atomic
;*
;* Stack: 2 Bytes
;* Call : -
;* Macro: -
;* Read : NULL [Register] (Zero)
;*        LPLH, LPLL [Registers] (Data buffer pointer lower limit)
;*        UPLH, UPLL [Registers] (Data buffer pointer upper limit)
;* Write: PARAMETER [Register] (Byte to read)
;*        X (Byte count in data buffer)
;*        Y (Data buffer read pointer)
;*        RESULT [High register] (0 on success, 1 if buffer is empty)
;* Interrupt latency: 12, 17 or 18 Cycles
;*
;*******************************************************************************

RD_BUFFER:
      cli
      cp  XL, NULL                 ;Check for data buffer to be empty
      cpc  XH, NULL
      breq  RD_BUFFER_EMPTY
      ldi  RESULT, 0x00            ;Read data
      ld  PARAMETER, Y+
      sbiw  XL, 0x01

      cp  UPLL, YL                 ;Manage ring buffer read pointer
      cpc  UPLH, YH
      brsh  RD_BUFFER_EXIT
      mov  YL, LPLL
      mov  YH, LPLH

   RD_BUFFER_EXIT:
      reti                         ;Save one cycle against sei + ret

   RD_BUFFER_EMPTY:
      ldi  RESULT, 0x01            ;Buffer empty
      rjmp  RD_BUFFER_EXIT


;*******************************************************************************
;*
;* Read single byte from FIFO data buffer
;*
;* This macro is for interrupt handlers only!
;* If the data buffer is empty, 0xFF is returned
;*
;* Stack: 2 Bytes
;* Call : -
;* Macro: -
;* Read : NULL [Register] (Zero)
;*        BSH, BSL [Registers] (Data buffer size)
;*        LPLH, LPLL [Registers] (Data buffer pointer lower limit)
;*        UPLH, UPLL [Registers] (Data buffer pointer upper limit)
;* Write: I_PARAMETER [Register] (Byte to read)
;*        I_TMP [High register]
;*        X (Byte count in data buffer)
;*        Y (Data buffer read pointer)
;*        STAT [High register]
;* Runtime: 7 Cycles (Buffer empty), 17 Cycles (Worst case, buffer not empty)
;*
;*******************************************************************************

.macro  I_RD_BUFFER
      cp  XL, NULL                 ;Check for data buffer to be empty
      cpc  XH, NULL
      brne  I_RD_BUFFER_READ

      ;Buffer empty => Return 0xFF
      ldi  I_TMP, 0xFF
      mov  I_PARAMETER, I_TMP
      rjmp  I_RD_BUFFER_EXIT

   I_RD_BUFFER_READ:
      ld  I_PARAMETER, Y+          ;Read data
      sbiw  XL, 0x01

      cp  UPLL, YL                 ;Manage ring buffer read pointer
      cpc  UPLH, YH
      brsh  I_RD_BUFFER_RING
      mov  YL, LPLL
      mov  YH, LPLH
   I_RD_BUFFER_RING:

      ;Check for last byte
      cp  XL, NULL                 ;Check for data buffer to be empty now
      cpc  XH, NULL
      brne  I_RD_BUFFER_EXIT       ;Not empty => Don't touch FLOW flag
      cbr  STAT, (1 << FLOW)       ;Empty => Clear FLOW flag

   I_RD_BUFFER_EXIT:
.endmacro


;EOF
