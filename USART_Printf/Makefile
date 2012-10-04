EXECUTABLE=USART_Printf.elf
BIN_IMAGE=USART_Printf.bin

CC=arm-none-eabi-gcc
OBJCOPY=arm-none-eabi-objcopy

CFLAGS=-g -O2 -mlittle-endian -mthumb
CFLAGS+=-mcpu=cortex-m4	
CFLAGS+=-ffreestanding 

#Don't use newlib
CFLAGS+=-nostdlib
#Link to newlib
#CFLAGS+=-L/cygdrive/c/sat/arm-none-eabi/lib/thumb2


# to run from FLASH
CFLAGS+=-Wl,-T,stm32_flash.ld

#CFLAGS+=-v

#Search current directory too
CFLAGS+=-I./

# stm32f4_discovery lib
CFLAGS+=-I../STM32F4xxLibrary/STM32F4xx_StdPeriph_Driver/inc
CFLAGS+=-I../STM32F4xxLibrary/STM32F4xx_StdPeriph_Driver/inc/device_support
CFLAGS+=-I../STM32F4xxLibrary/STM32F4xx_StdPeriph_Driver/inc/core_support

#Utilities
CFLAGS+=-I../STM32F4xxLibrary/Utilities/STM32F4-Discovery
CFLAGS+=-I../STM32F4xxLibrary/Utilities/jnosky

all: $(BIN_IMAGE)

$(BIN_IMAGE): $(EXECUTABLE)
	$(OBJCOPY) -O binary $^ $@

$(EXECUTABLE): main.c system_stm32f4xx.c startup_stm32f4xx.s stm32f4xx_it.c \
	../STM32F4xxLibrary/Utilities/STM32F4-Discovery/stm32f4_discovery.c \
	../STM32F4xxLibrary/Utilities/jnosky/printf.c \
	../STM32F4xxLibrary/Utilities/jnosky/discoveryf4utils.c

#Add for newlib
#	../../Utilities/syscalls.c



	$(CC) $(CFLAGS) $^ -o $@  -L../STM32F4xx -lSTM32F4xx_StdPeriph_Driver \

clean:
	rm -rf $(EXECUTABLE)
	rm -rf $(BIN_IMAGE)

.PHONY: all clean
