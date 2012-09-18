LIB = libSTM32F4xx_StdPeriph_Driver.a

CC = arm-none-eabi-gcc
AR = arm-none-eabi-ar
RANLIB = arm-none-eabi-ranlib

LIBPATH = ./Libraries/STM32F4xx_StdPeriph_Driver
INCLUDES = -I$(LIBPATH)/inc -I$(LIBPATH)/inc/device_support -I$(LIBPATH)/inc/core_support

CFLAGS = -Wall -O2 -mlittle-endian -mthumb
CFLAGS += -mcpu=cortex-m4 -ffreestanding -nostdlib
CFLAGS += $(INCLUDES)


SRCS = \
$(LIBPATH)/src/misc.c \
$(LIBPATH)/src/stm32f4xx_adc.c \
$(LIBPATH)/src/stm32f4xx_can.c \
$(LIBPATH)/src/stm32f4xx_crc.c \
$(LIBPATH)/src/stm32f4xx_cryp_aes.c \
$(LIBPATH)/src/stm32f4xx_cryp_des.c \
$(LIBPATH)/src/stm32f4xx_cryp_tdes.c \
$(LIBPATH)/src/stm32f4xx_cryp_des.c \
$(LIBPATH)/src/stm32f4xx_dac.c \
$(LIBPATH)/src/stm32f4xx_dbgmcu.c \
$(LIBPATH)/src/stm32f4xx_dcmi.c \
$(LIBPATH)/src/stm32f4xx_dma.c \
$(LIBPATH)/src/stm32f4xx_exti.c \
$(LIBPATH)/src/stm32f4xx_flash.c \
$(LIBPATH)/src/stm32f4xx_fsmc.c \
$(LIBPATH)/src/stm32f4xx_gpio.c \
$(LIBPATH)/src/stm32f4xx_hash_md5.c \
$(LIBPATH)/src/stm32f4xx_hash_sha1.c \
$(LIBPATH)/src/stm32f4xx_hash.c \
$(LIBPATH)/src/stm32f4xx_i2c.c \
$(LIBPATH)/src/stm32f4xx_iwdg.c \
$(LIBPATH)/src/stm32f4xx_pwr.c \
$(LIBPATH)/src/stm32f4xx_rcc.c \
$(LIBPATH)/src/stm32f4xx_rng.c \
$(LIBPATH)/src/stm32f4xx_rtc.c \
$(LIBPATH)/src/stm32f4xx_sdio.c \
$(LIBPATH)/src/stm32f4xx_spi.c \
$(LIBPATH)/src/stm32f4xx_syscfg.c \
$(LIBPATH)/src/stm32f4xx_tim.c \
$(LIBPATH)/src/stm32f4xx_usart.c \
$(LIBPATH)/src/stm32f4xx_wwdg.c \
#../inc/core_support/core_cm4.c

OBJS = $(SRCS:.c=.o)

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) -r $(LIB) $(OBJS)
	$(RANLIB) $(LIB)

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	-rm -f $(OBJS)
#	-rm -f $(LIB)

.PHONY: all clean
