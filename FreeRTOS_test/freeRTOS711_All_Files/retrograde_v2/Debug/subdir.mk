################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../main.c 

S_SRCS += \
../xitoa.s 

OBJS += \
./main.o \
./xitoa.o 

C_DEPS += \
./main.d 

S_DEPS += \
./xitoa.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/home/phillip/AVR/EclipseWorkspace/freeRTOS710/include" -DGCC_MEGA_AVR -Wall -O2 -fpack-struct -fshort-enums -frename-registers -ffast-math -mcall-prologues -mrelax -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=22118400UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.s
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -mmcu=atmega328p -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


