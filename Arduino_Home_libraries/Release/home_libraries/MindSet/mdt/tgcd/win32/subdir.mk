################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
/Users/sin/Documents/Arduino/libraries/MindSet/mdt/tgcd/win32/thinkgear_testapp.c 

OBJS += \
./home_libraries/MindSet/mdt/tgcd/win32/thinkgear_testapp.o 

C_DEPS += \
./home_libraries/MindSet/mdt/tgcd/win32/thinkgear_testapp.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/MindSet/mdt/tgcd/win32/thinkgear_testapp.o: /Users/sin/Documents/Arduino/libraries/MindSet/mdt/tgcd/win32/thinkgear_testapp.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

