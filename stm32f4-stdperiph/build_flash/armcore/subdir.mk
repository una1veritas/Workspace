################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/delay.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/gpio.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/i2c.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/spi.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/systick.c \
C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/usart.c 

OBJS += \
./armcore/delay.o \
./armcore/gpio.o \
./armcore/i2c.o \
./armcore/spi.o \
./armcore/systick.o \
./armcore/usart.o 

C_DEPS += \
./armcore/delay.d \
./armcore/gpio.d \
./armcore/i2c.d \
./armcore/spi.d \
./armcore/systick.d \
./armcore/usart.d 


# Each subdirectory must supply rules for building sources it contributes
armcore/delay.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/delay.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\STM32F4xx" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/gpio.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/gpio.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\STM32F4xx" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/i2c.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/i2c.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\STM32F4xx" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/spi.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/spi.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\STM32F4xx" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/systick.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/systick.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\STM32F4xx" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

armcore/usart.o: C:/Users/Sin/Documents/Eclipse/Workspace/STM32library/armcore/usart.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\Include" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\CMSIS\STM32F4xx" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32library\STM32F4xx_StdPeriph_Driver\inc" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -I"C:\Users\Sin\Documents\Eclipse\Workspace\STM32library\armcore" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


