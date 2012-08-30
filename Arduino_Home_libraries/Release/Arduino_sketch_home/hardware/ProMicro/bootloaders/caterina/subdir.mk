################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.o \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.o 

C_SRCS += \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.c \
/Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.c 

OBJS += \
./Arduino_sketch_home/hardware/ProMicro/bootloaders/caterina/Caterina.o \
./Arduino_sketch_home/hardware/ProMicro/bootloaders/caterina/Descriptors.o 

C_DEPS += \
./Arduino_sketch_home/hardware/ProMicro/bootloaders/caterina/Caterina.d \
./Arduino_sketch_home/hardware/ProMicro/bootloaders/caterina/Descriptors.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino_sketch_home/hardware/ProMicro/bootloaders/caterina/Caterina.o: /Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Caterina.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DNON_ARDUINO_IDE -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/hardware/ProMicro/bootloaders/caterina/Descriptors.o: /Users/sin/Documents/Arduino/hardware/ProMicro/bootloaders/caterina/Descriptors.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries" -DARDUINO=100 -DNON_ARDUINO_IDE -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


