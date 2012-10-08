################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../armcore/delay.c \
../armcore/gpio_digital.c \
../armcore/newlib_stubs.c 

OBJS += \
./armcore/delay.o \
./armcore/gpio_digital.o \
./armcore/newlib_stubs.o 

C_DEPS += \
./armcore/delay.d \
./armcore/gpio_digital.d \
./armcore/newlib_stubs.d 


# Each subdirectory must supply rules for building sources it contributes
armcore/%.o: ../armcore/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_test/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLib/Utilities/jnosky" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


