################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/MegaSDTest/main.c 

S_SRCS += \
../freeRTOS711_All_Files/MegaSDTest/xitoa.s 

OBJS += \
./freeRTOS711_All_Files/MegaSDTest/main.o \
./freeRTOS711_All_Files/MegaSDTest/xitoa.o 

C_DEPS += \
./freeRTOS711_All_Files/MegaSDTest/main.d 

S_DEPS += \
./freeRTOS711_All_Files/MegaSDTest/xitoa.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/MegaSDTest/%.o: ../freeRTOS711_All_Files/MegaSDTest/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

freeRTOS711_All_Files/MegaSDTest/%.o: ../freeRTOS711_All_Files/MegaSDTest/%.s
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -g2 -gstabs -mmcu=atmega16 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


