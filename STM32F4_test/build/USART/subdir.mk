################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../USART/main.cpp 

<<<<<<< HEAD
C_SRCS += \
../USART/Olimex_stm32f217ze_sk.c 

OBJS += \
./USART/Olimex_stm32f217ze_sk.o \
./USART/main.o 

C_DEPS += \
./USART/Olimex_stm32f217ze_sk.d 

=======
OBJS += \
./USART/main.o 

>>>>>>> 40ecdb381c32971155a08b181ede242212999ebb
CPP_DEPS += \
./USART/main.d 


# Each subdirectory must supply rules for building sources it contributes
<<<<<<< HEAD
USART/%.o: ../USART/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_test/mycore" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -g3 -gdwarf-2 -Wall -c -fmessage-length=0  -ffreestanding -std=c99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

=======
>>>>>>> 40ecdb381c32971155a08b181ede242212999ebb
USART/%.o: ../USART/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-none-eabi-g++ -DUSE_STDPERIPH_DRIVER=1 -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Device/ST/STM32F4xx/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/CMSIS/Include" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_test/GPIO" -I"/Users/sin/Documents/Eclipse/Workspace/STM32_library/stm32f4/STM32F4xx_StdPeriph_Driver/inc" -I"/Users/sin/Documents/Eclipse/Workspace/STM32F4_test/mycore" -Os -mthumb -mcpu=cortex-m4 -mlittle-endian -nostdlib -g3 -gdwarf-2 -Wall -c -fmessage-length=0 -fno-rtti -fno-exceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


