# setup environment


#define DEBUG
#import MESSAGES

TARGET_ARCH = -mcpu=cortex-m3 -mthumb

COMPILE_OPTS = $(WARNINGS) $(FREERTOS_OPTS) $(TARGET_OPTS) $(MESSAGES) $(F_NO_STRICT_ALIASING) $(INCLUDE_DIRS)
WARNINGS = -Wall
#WARNINGS = -Wall -W -Wshadow -Wwrite-strings -Winline
INCLUDE_DIRS = -I . -I ../ -I ../inc \
               -I STM32F10x_StdPeriph_Driver/inc \
               -I USART/inc -I UTIL/inc -I SD/inc \
               -I CMSIS/Core/CM3 \
               -I STM32_USB-FS-Device_Driver/inc \
               -I FreeRTOS/Source/include \
               -I FreeRTOS/Source/portable/GCC/ARM_CM3 \
               -I COM/inc \
               -I Virtual_COM_Port/inc

# needed to safely build the library (until proven otherwise)
F_NO_STRICT_ALIASING = -fno-strict-aliasing

FREERTOS_OPTS = -D GCC_ARMCM3

ifdef DEBUG
 TARGET_OPTS = -O0 -g3
 DEBUG_MACRO = -DDEBUG
else
 TARGET_OPTS = -Os $(F_INLINE) $(F_INLINE_ONCE) $(F_UNROLL_LOOPS)
 F_INLINE = -finline
 F_INLINE_ONCE = -finline-functions-called-once
 #F_UNROLL_LOOPS = -funroll-loops
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
 $(patsubst %.c,%.o,$(wildcard CMSIS/Core/CM3/*.c)) \
 $(patsubst %.s,%.o,$(wildcard CMSIS/Core/CM3/*.s)) \
 $(patsubst %.c,%.o,$(wildcard FreeRTOS/Source/*.c)) \
 $(patsubst %.s,%.o,$(wildcard FreeRTOS/Source/*.s)) \
 $(patsubst %.c,%.o,$(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.c)) \
 $(patsubst %.s,%.o,$(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.s)) \
 $(patsubst %.c,%.o,$(wildcard FreeRTOS/Source/portable/MemMang/*.c)) \
 $(patsubst %.s,%.o,$(wildcard FreeRTOS/Source/portable/MemMang/*.s)) \
 $(patsubst %.c,%.o,$(wildcard STM32_USB-FS-Device_Driver/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard STM32_USB-FS-Device_Driver/src/*.s)) \
 $(patsubst %.c,%.o,$(wildcard Virtual_COM_Port/src/*.c)) \
 $(patsubst %.s,%.o,$(wildcard Virtual_COM_Port/src/*.s)))
 
# all

.PHONY: all
all: $(LIB_OUT)

# lib
.PHONY: lib
$(LIB_OUT): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $(LIB_OBJS)

$(LIB_OBJS): $(wildcard USART/src/*.c) \
             $(wildcard COM/src/*.c) \
             $(wildcard UTIL/src/*.c) \
             $(wildcard SD/src/*.c) \
             $(wildcard STM32F10x_StdPeriph_Driver/src/*.c) \
             $(wildcard CMSIS/Core/CM3/*.c) \
             $(wildcard STM32_USB-FS-Device_Driver/src/*.c) \
             $(wildcard FreeRTOS/Source/*.c) \
             $(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.c) \
             $(wildcard FreeRTOS/Source/portable/MemMang/*.c) \
             $(wildcard Virtual_COM_Port/src/*.c)

# asm

.PHONY: asm
asm:
	$(CC) $(COMPILE_OPTS) -S \
             $(wildcard STM32F10x_StdPeriph_Driver/src/*.c) \
             $(wildcard USART/src/*.c) \
             $(wildcard COM/src/*.c) \
             $(wildcard UTIL/src/*.c) \
             $(wildcard SD/src/*.c) \
             $(wildcard CMSIS/Core/CM3/*.c) \
             $(wildcard STM32_USB-FS-Device_Driver/src/*.c) \
             $(wildcard FreeRTOS/Source/*.c) \
             $(wildcard FreeRTOS/Source/portable/GCC/ARM_CM3/*.c) \
             $(wildcard FreeRTOS/Source/portable/MemMang/*.c) \
             $(wildcard Virtual_COM_Port/src/*.c)
                         
# clean

.PHONY: clean
clean:
	@del /f /q STM32F10x_StdPeriph_Driver\src\*.o CMSIS\Core\CM3\*.o STM32_USB-FS-Device_Driver\src\*.o FreeRTOS\Source\*.o FreeRTOS\Source\portable\GCC\ARM_CM3\*.o FreeRTOS\Source\portable\MemMang\*.o *.s $(LIB_OUT)
