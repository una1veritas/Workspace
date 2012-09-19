################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
/Users/sin/Documents/Arduino/libraries/BMP085/ATmega328/main.o 

C_SRCS += \
/Users/sin/Documents/Arduino/libraries/BMP085/ATmega328/main.c 

OBJS += \
./home_libraries/BMP085/ATmega328/main.o 

C_DEPS += \
./home_libraries/BMP085/ATmega328/main.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/BMP085/ATmega328/main.o: /Users/sin/Documents/Arduino/libraries/BMP085/ATmega328/main.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


