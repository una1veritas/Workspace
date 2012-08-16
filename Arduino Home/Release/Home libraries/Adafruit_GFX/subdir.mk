################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Adafruit_GFX/Adafruit_GFX.cpp 

C_SRCS += \
/Users/sin/Documents/Arduino/libraries/Adafruit_GFX/glcdfont.c 

OBJS += \
./Home\ libraries/Adafruit_GFX/Adafruit_GFX.o \
./Home\ libraries/Adafruit_GFX/glcdfont.o 

C_DEPS += \
./Home\ libraries/Adafruit_GFX/glcdfont.d 

CPP_DEPS += \
./Home\ libraries/Adafruit_GFX/Adafruit_GFX.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/Adafruit_GFX/Adafruit_GFX.o: /Users/sin/Documents/Arduino/libraries/Adafruit_GFX/Adafruit_GFX.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Adafruit_GFX/Adafruit_GFX.d" -MT"Home\ libraries/Adafruit_GFX/Adafruit_GFX.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/Adafruit_GFX/glcdfont.o: /Users/sin/Documents/Arduino/libraries/Adafruit_GFX/glcdfont.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"Home libraries/Adafruit_GFX/glcdfont.d" -MT"Home\ libraries/Adafruit_GFX/glcdfont.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


