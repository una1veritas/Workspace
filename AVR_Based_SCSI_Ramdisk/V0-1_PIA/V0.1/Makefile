#*******************************************************************************
#*
#* Header
#*
#* Project:      SCSI controller
#*
#* Module:       PIA
#* File:         Makefile
#*
#* Language:     make
#*
#*               Rules: clean, all, gen, hex
#*
#* Description:  Makefile for Parallel Interface Agent firmware
#*
#* Copyright:    (C) 2004 by Michael Baeuerle <micha@hilfe-fuer-linux.de>
#* License:      All rights reserved.
#*               Redistribution and use in source and binary forms, with or
#*               without modification, are permitted provided that the
#*               following conditions are met:
#*               - Redistributions of source code must retain the above
#*                 copyright notice, this list of conditions and the following
#*                 disclaimer.
#*               - Redistributions in binary form must reproduce the above
#*                 copyright notice, this list of conditions and the following
#*                 disclaimer in the documentation and/or other materials
#*                 provided with the distribution.
#*               THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#*               CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#*               INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#*               MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#*               DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#*               CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#*               SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
#*               NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#*               LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#*               HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#*               CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#*               OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#*               EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*
#* Written for:  Interpreter : make
#*               Platform    : all
#*               OS          : UNIX
#* Tested:       Interpreter : GNU make (Version 3.77)
#*               Platform    : IA32-SMP (2x PentiumPro)
#*               OS          : GNU/Linux (Kernel version: 2.2.17)
#* Tested:       Interpreter : GNU make (Version 3.77)
#*               Platform    : IA32-SMP (2x PentiumPro)
#*               OS          : GNU/Linux (Kernel version: 2.2.26)
#* Tested:       Interpreter : GNU make (Version 3.78.1)
#*               Platform    : SPARC32-SMP (2x SuperSPARC)
#*               OS          : GNU/Linux (Kernel version: 2.2.19)
#*
#* Changelog:    2004-09-18  Michael Baeuerle
#*               Created
#*
#*               2004-09-25  Michael Baeuerle
#*               Dependencies updated
#*
#*               2004-10-16  Michael Baeuerle
#*               Added handling for 'buffer.s'
#*
#*
#* To do:        -
#*
#*******************************************************************************


#*******************************************************************************
#*
#* Macro definitions
#*
#*******************************************************************************

# Target names
TARGET          = pia
TARGET1         = init
TARGET2         = interrupt
TARGET3         = busphases
TARGET4         = command
TARGET5         = timing
TARGET6         = buffer

# Assemblers
A1              = tavrasm

# Assembler flags
A1FLAGS         = -v -c -g

# Converters
CONV1           = GENtoHEX

# Converter flags
CONV1FLAGS      =

# Directories
INCLUDE         = ./include
BINDIR          = ./bin


#*******************************************************************************
#*
#* Main rules
#*
#*******************************************************************************

# Complete rebuild
all: hex

# Create target binary
gen: $(BINDIR)/flash.gen

# Convert binary in Atmel GENERIC format to Intel HEX format
hex: $(BINDIR)/flash.gen
	$(CONV1) $(CONV1FLAGS) $(BINDIR)/flash.gen $(BINDIR)/flash.hex

# This forces a complete re-build for "make all"
clean:
	-rm -f $(BINDIR)/*


#*******************************************************************************
#*
#* Sub rules
#*
#*******************************************************************************

$(BINDIR)/flash.gen: $(INCLUDE)/m169def.h $(INCLUDE)/$(TARGET).h \
 $(TARGET).s $(TARGET1).s  $(TARGET2).s $(TARGET3).s $(TARGET4).s $(TARGET5).s \
 $(TARGET6).s
	$(A1) $(A1FLAGS) -e $(BINDIR)/$(TARGET).lst -o $(BINDIR)/flash.gen \
	 $(TARGET).s
	@echo
	@echo Finished.
	@echo


# EOF
