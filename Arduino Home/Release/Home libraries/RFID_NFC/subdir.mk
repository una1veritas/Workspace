################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/RFID_NFC/PN532_I2C.cpp \
/Users/sin/Documents/Arduino/libraries/RFID_NFC/RCS620S.cpp 

OBJS += \
./Home\ libraries/RFID_NFC/PN532_I2C.o \
./Home\ libraries/RFID_NFC/RCS620S.o 

CPP_DEPS += \
./Home\ libraries/RFID_NFC/PN532_I2C.d \
./Home\ libraries/RFID_NFC/RCS620S.d 


# Each subdirectory must supply rules for building sources it contributes
Home\ libraries/RFID_NFC/PN532_I2C.o: /Users/sin/Documents/Arduino/libraries/RFID_NFC/PN532_I2C.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/RFID_NFC/PN532_I2C.d" -MT"Home\ libraries/RFID_NFC/PN532_I2C.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Home\ libraries/RFID_NFC/RCS620S.o: /Users/sin/Documents/Arduino/libraries/RFID_NFC/RCS620S.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/Wire/utility" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"Home libraries/RFID_NFC/RCS620S.d" -MT"Home\ libraries/RFID_NFC/RCS620S.d" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


