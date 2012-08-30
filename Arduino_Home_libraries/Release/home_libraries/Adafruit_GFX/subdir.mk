################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/Adafruit_GFX/Adafruit_GFX.cpp 

C_SRCS += \
/Users/sin/Documents/Arduino/libraries/Adafruit_GFX/glcdfont.c 

OBJS += \
./home_libraries/Adafruit_GFX/Adafruit_GFX.o \
./home_libraries/Adafruit_GFX/glcdfont.o 

C_DEPS += \
./home_libraries/Adafruit_GFX/glcdfont.d 

CPP_DEPS += \
./home_libraries/Adafruit_GFX/Adafruit_GFX.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/Adafruit_GFX/Adafruit_GFX.o: /Users/sin/Documents/Arduino/libraries/Adafruit_GFX/Adafruit_GFX.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/Adafruit_GFX/glcdfont.o: /Users/sin/Documents/Arduino/libraries/Adafruit_GFX/glcdfont.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DNON_ARDUINO_IDE -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


