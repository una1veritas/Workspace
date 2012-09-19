################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/libraries/SD_SPI/utility/Sd2Card.cpp \
/Users/sin/Documents/Arduino/libraries/SD_SPI/utility/SdFile.cpp \
/Users/sin/Documents/Arduino/libraries/SD_SPI/utility/SdVolume.cpp 

OBJS += \
./home_libraries/SD_SPI/utility/Sd2Card.o \
./home_libraries/SD_SPI/utility/SdFile.o \
./home_libraries/SD_SPI/utility/SdVolume.o 

CPP_DEPS += \
./home_libraries/SD_SPI/utility/Sd2Card.d \
./home_libraries/SD_SPI/utility/SdFile.d \
./home_libraries/SD_SPI/utility/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
home_libraries/SD_SPI/utility/Sd2Card.o: /Users/sin/Documents/Arduino/libraries/SD_SPI/utility/Sd2Card.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/SD_SPI/utility/SdFile.o: /Users/sin/Documents/Arduino/libraries/SD_SPI/utility/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

home_libraries/SD_SPI/utility/SdVolume.o: /Users/sin/Documents/Arduino/libraries/SD_SPI/utility/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


