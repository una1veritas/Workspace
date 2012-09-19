SRC = ./src/main.c ./src/system_stm32f4xx.c ./src/startup_stm32f4xx.s ./src/stm32f4xx_it.c \
	./board/iar_stm32f217ze_sk.c
#	./Libraries/CMSIS/CM3/DeviceSupport/ST/STM32F2xx/startup/arm/startup_stm32xx.s
EXECUTABLE=STM32_E.elf
BIN_IMAGE=STM32_E.bin

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
CFLAGS+=-I./inc
CFLAGS+=-I./Libraries/CMSIS/CM4/ST/STM32F4xx/Include
CFLAGS+=-I./Libraries/CMSIS/CM4/Include

# stm32f4_discovery lib
#CFLAGS+=-I../libstm/STM32F4xx_StdPeriph_Driver/inc
#CFLAGS+=-I./libstm/STM32F4xx_StdPeriph_Driver/inc
#CFLAGS+=-I../libstm/STM32F4xx_StdPeriph_Driver/inc/device_support
#CFLAGS+=-I../libstm/STM32F4xx_StdPeriph_Driver/inc/core_support

#Utilities
#CFLAGS+=-I../libstm/Utilities/STM32F4-Discovery

all: $(BIN_IMAGE)

$(BIN_IMAGE): $(EXECUTABLE)
	$(OBJCOPY) -O binary $^ $@

$(EXECUTABLE): $(SRC)  #\
#	../libstm/Utilities/STM32F4-Discovery/stm32f4_discovery.c \
#	../libstm/Utilities/printf.c

#Add for newlib
#	../../Utilities/syscalls.c



	$(CC) $(CFLAGS) $^ -o $@  -L../libstm/STM32F4xx_StdPeriph_Driver/build -lSTM32F4xx_StdPeriph_Driver \

clean:
	rm -rf $(EXECUTABLE)
	rm -rf $(BIN_IMAGE)

.PHONY: all clean
