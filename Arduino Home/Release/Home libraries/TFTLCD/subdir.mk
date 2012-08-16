################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/TFTLCD/TFTLCD.cpp 

C_SRCS += \
/Users/sin/Documents/Arduino/libraries/TFTLCD/glcdfont.c 

OBJS += \
./Home\ libraries/TFTLCD/TFTLCD.o \
./Home\ libraries/TFTLCD/glcdfont.o 

C_DEPS += \
./Home\ libraries/TFTLCD/glcdfont.d 

CPP_DEPS += \
./Home\ libraries/TFTLCD/TFTLCD.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/TFTLCD/TFTLCD.o: /Users/sin/Documents/Arduino/libraries/TFTLCD/TFTLCD.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/TFTLCD/TFTLCD.d" -MT"Home\ libraries/TFTLCD/TFTLCD.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/TFTLCD/glcdfont.o: /Users/sin/Documents/Arduino/libraries/TFTLCD/glcdfont.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/TFTLCD/glcdfont.d" -MT"Home\ libraries/TFTLCD/glcdfont.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


