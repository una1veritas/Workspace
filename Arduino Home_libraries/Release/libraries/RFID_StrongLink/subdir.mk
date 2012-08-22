################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/RFID_StrongLink/SL025.cpp \
/Users/sin/Documents/Arduino/libraries/RFID_StrongLink/StrongLinkI2C.cpp 

OBJS += \
./libraries/RFID_StrongLink/SL025.o \
./libraries/RFID_StrongLink/StrongLinkI2C.o 

CPP_DEPS += \
./libraries/RFID_StrongLink/SL025.d \
./libraries/RFID_StrongLink/StrongLinkI2C.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/RFID_StrongLink/SL025.o: /Users/sin/Documents/Arduino/libraries/RFID_StrongLink/SL025.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

libraries/RFID_StrongLink/StrongLinkI2C.o: /Users/sin/Documents/Arduino/libraries/RFID_StrongLink/StrongLinkI2C.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -DARDUINO=100 -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


