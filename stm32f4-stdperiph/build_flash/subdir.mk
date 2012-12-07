################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../ST7032i.c \
../main.c \
../newlib_stubs.c \
../stm32f4xx_it.c \
../system_stm32f4xx.c 

S_UPPER_SRCS += \
../startup_stm32f4xx.S 

OBJS += \
./ST7032i.o \
./main.o \
./newlib_stubs.o \
./startup_stm32f4xx.o \
./stm32f4xx_it.o \
./system_stm32f4xx.o 

C_DEPS += \
./ST7032i.d \
./main.d \
./newlib_stubs.d \
./stm32f4xx_it.d \
./system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	arm-none-eabi-gcc -DUSE_STDPERIPH_DRIVER=1 -DHSE_VALUE=25000000 -I"C:\Users\sin\Documents\Eclipse\Workspace\stm32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -Os -mcpu=cortex-m4 -mthumb -mlittle-endian -mfloat-abi=soft -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.S
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Assembler'
	arm-none-eabi-as -I"C:\Users\sin\Documents\Eclipse\Workspace\stm32library\armcore" -I"C:\Users\Sin\Documents\Eclipse\Workspace\stm32f4-stdperiph" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


