################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../armlib/Print.cpp \
../armlib/ST7032i.cpp \
../armlib/Wire.cpp 

OBJS += \
./armlib/Print.o \
./armlib/ST7032i.o \
./armlib/Wire.o 

CPP_DEPS += \
./armlib/Print.d \
./armlib/ST7032i.d \
./armlib/Wire.d 


# Each subdirectory must supply rules for building sources it contributes
armlib/%.o: ../armlib/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/CMSIS/STM32F4xx" -I"/Users/sin/Documents/Eclipse/Workspace/STM32Library/armcore" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph" -I"/Users/sin/Documents/Eclipse/Workspace/stm32f4-stdperiph/armlib" -O2 -mcpu=cortex-m4 -mthumb -mlittle-endian  -mfloat-abi=soft -g3 -Wall -c -fmessage-length=0 -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


