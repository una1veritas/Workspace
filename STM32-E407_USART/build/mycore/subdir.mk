################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../mycore/delay.c \
../mycore/gpio_digital.c 

OBJS += \
./mycore/delay.o \
./mycore/gpio_digital.o 

C_DEPS += \
./mycore/delay.d \
./mycore/gpio_digital.d 


# Each subdirectory must supply rules for building sources it contributes
mycore/%.o: ../mycore/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/mycore" -I"/Users/sin/Documents/Eclipse/Workspace/STM32-E407_USART/USART" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLibrary/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLibrary/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4xxLibrary/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -mfpu=fpv4-sp-d16 -g -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


