################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../core/Print.cpp \
../core/Stream.cpp \
../core/USARTSerial.cpp \
../core/gpio_digital.cpp \
../core/systick.cpp 

OBJS += \
./core/Print.o \
./core/Stream.o \
./core/USARTSerial.o \
./core/gpio_digital.o \
./core/systick.o 

CPP_DEPS += \
./core/Print.d \
./core/Stream.d \
./core/USARTSerial.d \
./core/gpio_digital.d \
./core/systick.d 


# Each subdirectory must supply rules for building sources it contributes
core/%.o: ../core/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_c++_test/core" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -Wall -c -fmessage-length=0 -ffreestanding -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


