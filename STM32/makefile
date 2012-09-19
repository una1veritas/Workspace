# makefile for application

# DEBUG = 1	# or include DEBUG=1 in the command line to compile under debug mode

# making this makefile works even if cygwin is installed
#SHELL=cmd.exe
SHELL=/bin/csh

# include board specific setting
include makefile.in

export DEBUG
export MESSAGES

TARGET_ARCH = -mcpu=cortex-m3 -mthumb

INCLUDE_DIRS = -I . -I inc \
               -I lib \
               -I lib/CMSIS/CM3/CoreSupport -I lib/CMSIS/CM3/DeviceSupport/ST/STM32F10x \
               -I lib/STM32F10x_StdPeriph_Driver/inc \
               -I lib/USART/inc -I lib/COM/inc -I lib/SD/inc -I lib/UTIL/inc \
               -I lib/STM32_USB-FS-Device_Driver/inc -I lib/Virtual_COM_Port/inc \
               -I lib/FreeRTOS/Source/include \
               -I lib/FreeRTOS/Source/portable/GCC/ARM_CM3 \
               -I lib/Startup \
               -I lib/Platform 
               
LIBRARY_DIRS = -L ./lib

COMPILE_OPTS = -fsigned-char $(WARNINGS) $(FIRMWARE_OPTS) $(BOARD_OPTS) $(FREERTOS_OPTS) $(TARGET_OPTS) $(MESSAGES) $(INCLUDE_DIRS)
#WARNINGS =  -W $(W_CONVERSION) -Winline
#WARNINGS = -Wall -W $(W_CONVERSION) -Wshadow -Wcast-qual -Wwrite-strings -Winline
WARNINGS = -Wall -W $(W_CONVERSION)
#W_CONVERSION = -Wconversion

WARNINGS_CXX = -Weffc++ $(W_OLD_STYLE_CAST)
#W_OLD_STYLE_CAST = -Wold-style-cast
#F_NO_EXCEPTIONS = -fno-exceptions	# disabling exceptions saves code space

FIRMWARE_OPTS = -DUSE_STDPERIPH_DRIVER

MESSAGES = -fmessage-length=0

FREERTOS_OPTS = -D GCC_ARMCM3

ifdef DEBUG
 TARGET_OPTS = -O0 -g3
 DEBUG_MACRO = -DDEBUG
else
 F_INLINE = -finline
 F_INLINE_ONCE = -finline-functions-called-once
 #F_UNROLL_LOOPS = -funroll-loops
 TARGET_OPTS = -Os $(F_INLINE) $(F_INLINE_ONCE) $(F_UNROLL_LOOPS)
endif

CPPFLAGS = $(DEBUG_MACRO)

CC = arm-none-eabi-gcc
CFLAGS = -std=gnu99 $(COMPILE_OPTS)

CXX = arm-none-eabi-g++
CXXFLAGS = -std=gnu++98 $(COMPILE_OPTS) $(WARNINGS_CXX) $(F_NO_EXCEPTIONS)

AS = $(CC) -x assembler-with-cpp -c $(TARGET_ARCH)
ASFLAGS = $(COMPILE_OPTS)

LD = $(CC)
#LD = $(CXX)
#LDFLAGS = -Wl,--gc-sections,-Map=$(MAIN_MAP),-cref -T ld/512_64.ld $(INCLUDE_DIRS) $(LIBRARY_DIRS)
#LDFLAGS = -Wl,--gc-sections,-Map=$(MAIN_MAP),-cref -T ld/stm32f10x_flash.ld $(INCLUDE_DIRS) $(LIBRARY_DIRS)
LDFLAGS = -Wl,--gc-sections,-Map=$(MAIN_MAP),-cref -T ld/$(LD_SCRIPT) $(INCLUDE_DIRS) $(LIBRARY_DIRS)

AR = arm-none-eabi-ar
ARFLAGS = cr

OBJCOPY = arm-none-eabi-objcopy
OBJCOPY_BIN_FLAGS = -O binary
OBJCOPY_HEX_FLAGS = -O ihex

MAIN_OUT = bin\main.elf
MAIN_MAP = $(MAIN_OUT:%.elf=%.map)
MAIN_BIN = $(MAIN_OUT:%.elf=%.bin)
MAIN_HEX = $(MAIN_OUT:%.elf=%.hex)
MAIN_DFU = $(MAIN_OUT:%.elf=%.dfu)

MAIN_OBJS = $(sort \
            $(patsubst %.cpp,%.o,$(wildcard *.cpp)) \
            $(patsubst %.cc,%.o,$(wildcard *.cc)) \
            $(patsubst %.c,%.o,$(wildcard *.c)) \
            $(patsubst %.s,%.o,$(wildcard *.s)) \
            $(patsubst %.cpp,%.o,$(wildcard src/*.cpp)) \
            $(patsubst %.cc,%.o,$(wildcard src/*.cc)) \
            $(patsubst %.c,%.o,$(wildcard src/*.c)) \
            $(patsubst %.s,%.o,$(wildcard src/*.s)))

LINK_FWLIB = lib/lib.a

# all

.PHONY: all
all: $(MAIN_BIN) $(MAIN_HEX)

# main

$(MAIN_OUT): $(MAIN_OBJS) $(START_UP) $(LINK_FWLIB)  
	$(LD) $(LDFLAGS) $(TARGET_ARCH) $^ -o $@ 

$(MAIN_OBJS): $(wildcard *.h) $(wildcard inc/*.h)

$(MAIN_BIN): $(MAIN_OUT)
	$(OBJCOPY) $(OBJCOPY_BIN_FLAGS) $< $@

$(MAIN_HEX): $(MAIN_OUT)
	$(OBJCOPY) $(OBJCOPY_HEX_FLAGS) $< $@


# flash

.PHONY: flash
flash: flash-elf
#flash: flash-bin

.PHONY: flash-elf
flash-elf: all
	@copy $(MAIN_OUT) jtag\flash.elf

ifdef DFU	
# download using DFU and DFU Tools from KSK
	@cd bin && DfuConvert.exe -n Internal_Flash -v 0483 -p DF11 -b 0 -ReadFileName main.hex -CreateFileName main.dfu
	@cd bin && DfuUpgrade.exe -DownFileName main.dfu -TargetIdSel 0 -v 0483 -p DF11 -b 0
	@cd bin && DfuVerify.exe -DownFileName main.dfu -TargetIdSel 0 -v 0483 -p DF11 -b 0
endif

ifdef FT2232_JTAG
# download using FT2232 JTAG and OpenOCD
	@cd jtag && openocd -f flash-elf.cfg
endif
	
ifdef STLINK-JTAG
# download using ST-LINK and ST-LINK GDB server	
	@cd jtag && arm-none-eabi-gdb flash.elf -x gdb.ini
endif

ifdef UART
# download using UART and Flash Loader Demonstrator	
	@cd bin && STMFlashLoader.exe -c --pn 4 --br 115200 --pr NONE -i STM32_High-density_512K -e --all -p --drp --dwp -d --a 8000000 --fn "main.bin"
	@del jtag\flash.elf
endif

.PHONY: flash-bin
flash-bin: all
	@cp $(MAIN_BIN) jtag\flash.bin
	@cd jtag && openocd-ftd2xx -f flash-bin.cfg
	@del jtag\flash.bin
	
# asm

.PHONY: asm
asm:
	$(CC) $(COMPILE_OPTS) -S \
             $(wildcard *.c) \
             $(wildcard src/*.c) 

# clean

.PHONY: clean
clean:
#	-cs-rm *.o src/*.o $(MAIN_OUT) $(MAIN_MAP) $(MAIN_BIN) *.s jtag/flash.elf jtag/flash.bin
	-del /f /q *.o src\*.o $(MAIN_OUT) $(MAIN_MAP) $(MAIN_BIN) $(MAIN_HEX) $(MAIN_DFU) *.s jtag\flash.elf jtag\flash.bin
#	@cd fwlib && $(MAKE) clean
