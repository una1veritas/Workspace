################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
/Users/sin/Documents/Arduino/hardware/Quaranta/bootloaders/atmega644p/ATmegaBOOT_644P.o 

C_SRCS += \
/Users/sin/Documents/Arduino/hardware/Quaranta/bootloaders/atmega644p/ATmegaBOOT.c 

OBJS += \
./Arduino_sketch_home/hardware/Quaranta/bootloaders/atmega644p/ATmegaBOOT.o 

C_DEPS += \
./Arduino_sketch_home/hardware/Quaranta/bootloaders/atmega644p/ATmegaBOOT.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino_sketch_home/hardware/Quaranta/bootloaders/atmega644p/ATmegaBOOT.o: /Users/sin/Documents/Arduino/hardware/Quaranta/bootloaders/atmega644p/ATmegaBOOT.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

