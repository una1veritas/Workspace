;*******************************************************************************
;*
;* Header
;*  
;* Project:      SCSI controller
;*
;* Module:       PIA
;* File:         timing.s
;*
;* Language:     Assembly
;*
;* Description:  Timings
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
;*               Added macro BUS_SETTLE_DELAY
;* 
;*               2004-10-08  Michael Baeuerle
;*               Added macro CABLE_SKEW_DELAY
;*               Added macro SYSTEM_DESKEW_DELAY
;* 
;*               2004-10-16  Michael Baeuerle
;*               Added macro DATA_RELEASE_DELAY
;*
;* 
;* To do:        -
;* 
;*******************************************************************************


;*******************************************************************************
;*
;* Macros for short timings
;*
;* These macros are only valid for 16MHz core clock! (1 core cycle = 62.5ns)
;*
;*******************************************************************************

.macro  BUS_SETTLE_DELAY           ;Nominal 400ns (437.5ns)
      nop
      nop
      nop
      nop
      nop
      nop
      nop
.endmacro


.macro  CABLE_SKEW_DELAY           ;Nominal 4ns (62.5ns)
      nop
.endmacro


.macro  SYSTEM_DESKEW_DELAY        ;Nominal 45ns (62.5ns)
      nop
.endmacro


.macro  DATA_RELEASE_DELAY         ;Nominal 400ns (437.5ns)
      nop
      nop
      nop
      nop
      nop
      nop
      nop
.endmacro
      

;EOF
