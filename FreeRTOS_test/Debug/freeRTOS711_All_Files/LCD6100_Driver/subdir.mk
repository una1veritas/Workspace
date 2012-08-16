################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/LCD6100_Driver/LCD_driver.c \
../freeRTOS711_All_Files/LCD6100_Driver/SPI9master.c 

OBJS += \
./freeRTOS711_All_Files/LCD6100_Driver/LCD_driver.o \
./freeRTOS711_All_Files/LCD6100_Driver/SPI9master.o 

C_DEPS += \
./freeRTOS711_All_Files/LCD6100_Driver/LCD_driver.d \
./freeRTOS711_All_Files/LCD6100_Driver/SPI9master.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/LCD6100_Driver/%.o: ../freeRTOS711_All_Files/LCD6100_Driver/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


