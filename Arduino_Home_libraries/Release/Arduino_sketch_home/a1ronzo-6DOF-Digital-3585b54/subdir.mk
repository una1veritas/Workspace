################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.cpp \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.cpp \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/main.cpp 

C_SRCS += \
/Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/twi.c 

OBJS += \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/ADXL345.o \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/ITG3200.o \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/main.o \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/twi.o 

C_DEPS += \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/twi.d 

CPP_DEPS += \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/ADXL345.d \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/ITG3200.d \
./Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/main.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/ADXL345.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ADXL345.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/ITG3200.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/ITG3200.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/main.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/main.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/a1ronzo-6DOF-Digital-3585b54/twi.o: /Users/sin/Documents/Arduino/a1ronzo-6DOF-Digital-3585b54/twi.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


