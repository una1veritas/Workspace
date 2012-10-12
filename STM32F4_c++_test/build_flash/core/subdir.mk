################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../core/USARTSerial.cpp \
../core/systick.cpp 

C_SRCS += \
../core/gpio_digital.c 

OBJS += \
./core/USARTSerial.o \
./core/gpio_digital.o \
./core/systick.o 

C_DEPS += \
./core/gpio_digital.d 

CPP_DEPS += \
./core/USARTSerial.d \
./core/systick.d 


# Each subdirectory must supply rules for building sources it contributes
core/%.o: ../core/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test/core" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -ffreestanding -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

core/%.o: ../core/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test/core" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -ffreestanding -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


