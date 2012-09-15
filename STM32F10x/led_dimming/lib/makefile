# makefile for library

# DEBUG = 1	# or include DEBUG=1 in the command line to compile under debug mode

# making this makefile works even if cygwin is installed
SHELL=cmd.exe

# include board specific setting
include ../makefile.in

# General settings
TARGET_ARCH = -mcpu=cortex-m3 -mthumb
COMPILE_OPTS = -fsigned-char $(WARNINGS) $(FIRMWARE_OPTS) $(BOARD_OPTS) $(FREERTOS_OPTS) $(TARGET_OPTS) $(MESSAGES) $(F_NO_STRICT_ALIASING) $(INCLUDE_DIRS)
WARNINGS = -Wall
#WARNINGS = -Wall -W -Wshadow -Wwrite-strings -Winline
FIRMWARE_OPTS = -DUSE_STDPERIPH_DRIVER

INCLUDE_DIRS = -I . -I ../ -I ../inc \
               -I STM32F10x_StdPeriph_Driver/inc \
               -I USART/inc -I UTIL/inc -I SD/inc \
               -I CMSIS/CM3/CoreSupport -I CMSIS/CM3/DeviceSupport/ST/STM32F10x \
               -I STM32_USB-FS-Device_Driver/inc \
               -I FreeRTOS/Source/include \
               -I FreeRTOS/Source/portable/GCC/ARM_CM3 \
               -I COM/inc \
               -I Virtual_COM_Port/inc \
               -I Platform \
               -I Startup

# needed to safely build the library (until proven otherwise)
F_NO_STRICT_ALIASING = -fno-strict-aliasing

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

AS = $(CC) -x assembler-with-cpp -c $(TARGET_ARCH)
ASFLAGS = $(COMPILE_OPTS)

AR = arm-none-eabi-ar
ARFLAGS = cr

LIB_OUT = lib.a

LIB_OBJS = $(sort \
 $(patsubst %.c,%.o,$(wildcard USART/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard USART/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard COM/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard COM/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard UTIL/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard UTIL/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard SD/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard SD/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard STM32F10x_StdPeriph_Driver/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard STM32F10x_StdPeriph_Driver/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard CMSIS/CM3/CoreSupport/*.c)) \
 $(patsubst %.s,%.o,$(wildcard CMSIS/CM3/CoreSupport/*.s)) \
 $(patsubst %.c,%.o,$(wildcard CMSIS/CM3/DeviceSupport/ST/STM32F10x/*.c)) \
 $(patsubst %.s,%.o,$(wildcard CMSIS/CM3/DeviceSupport/ST/STM32F10x/*.s)) \
 $(patsubst %.c,%.o,$(wildcard FreeRTOS/Source/*.c)) \
 $(patsubst %.s,%.o,$(wildcard FreeRTOS/Source/*.s)) \
 $(patsubst %.c,%.o,$(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.c)) \
 $(patsubst %.s,%.o,$(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.s)) \
 $(patsubst %.c,%.o,$(wildcard FreeRTOS/Source/portable/MemMang/*.c)) \
 $(patsubst %.s,%.o,$(wildcard FreeRTOS/Source/portable/MemMang/*.s)) \
 $(patsubst %.c,%.o,$(wildcard STM32_USB-FS-Device_Driver/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard STM32_USB-FS-Device_Driver/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard Virtual_COM_Port/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard Virtual_COM_Port/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard Startup/*.c)) \
 $(patsubst %.s,%.o,$(wildcard Startup/*.s)) \
 $(patsubst %.c,%.o,$(wildcard Platform/*.c)) \
 $(patsubst %.s,%.o,$(wildcard Platform/*.s)))
 
# all

.PHONY: all
all: $(LIB_OUT)

# lib
.PHONY: lib
$(LIB_OUT): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $(LIB_OBJS)

$(LIB_OBJS): $(wildcard USART/src/*.h) \
             $(wildcard COM/src/*.h) \
             $(wildcard UTIL/src/*.h) \
             $(wildcard SD/src/*.h) \
             $(wildcard STM32F10x_StdPeriph_Driver/src/*.h) \
             $(wildcard CMSIS/CM3/CoreSupport/*.h) \
             $(wildcard CMSIS/CM3/DeviceSupport/ST/STM32F10x/*.h) \
             $(wildcard STM32_USB-FS-Device_Driver/src/*.h) \
             $(wildcard FreeRTOS/Source/*.h) \
             $(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.h) \
             $(wildcard FreeRTOS/Source/portable/MemMang/*.h) \
             $(wildcard Virtual_COM_Port/src/*.h) \
             $(wildcard Startup/*.h) \
             $(wildcard Platform/*.h) 

# asm

.PHONY: asm
asm:
	$(CC) $(COMPILE_OPTS) -S \
             $(wildcard STM32F10x_StdPeriph_Driver/src/*.c) \
             $(wildcard USART/src/*.c) \
             $(wildcard COM/src/*.c) \
             $(wildcard UTIL/src/*.c) \
             $(wildcard SD/src/*.c) \
             $(wildcard CMSIS/CM3/CoreSupport/*.c) \
             $(wildcard CMSIS/CM3/DeviceSupport/ST/STM32F10x/*.c) \
             $(wildcard STM32_USB-FS-Device_Driver/src/*.c) \
             $(wildcard FreeRTOS/Source/*.c) \
             $(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.c) \
             $(wildcard FreeRTOS/Source/portable/MemMang/*.c) \
             $(wildcard Virtual_COM_Port/src/*.c) \
             $(wildcard Startup/*.c) \
             $(wildcard Platform/*.c)
                         
# clean

.PHONY: clean
clean:
	@del /f /q STM32F10x_StdPeriph_Driver\src\*.o COM\src\*.o USART\src\*.o UTIL\src\*.o SD\src\*.o CMSIS\CM3\CoreSupport\*.o CMSIS\CM3\DeviceSupport\ST\STM32F10x\*.o STM32_USB-FS-Device_Driver\src\*.o Virtual_COM_Port\src\*.o FreeRTOS\Source\*.o FreeRTOS\Source\portable\GCC\ARM_CM3\*.o FreeRTOS\Source\portable\MemMang\*.o Startup\*.o Platform\*.o *.s $(LIB_OUT)
