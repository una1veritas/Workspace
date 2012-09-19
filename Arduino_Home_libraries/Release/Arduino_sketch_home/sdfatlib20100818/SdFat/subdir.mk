################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/Sd2Card.cpp \
/Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdFile.cpp \
/Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdVolume.cpp 

OBJS += \
./Arduino_sketch_home/sdfatlib20100818/SdFat/Sd2Card.o \
./Arduino_sketch_home/sdfatlib20100818/SdFat/SdFile.o \
./Arduino_sketch_home/sdfatlib20100818/SdFat/SdVolume.o 

CPP_DEPS += \
./Arduino_sketch_home/sdfatlib20100818/SdFat/Sd2Card.d \
./Arduino_sketch_home/sdfatlib20100818/SdFat/SdFile.d \
./Arduino_sketch_home/sdfatlib20100818/SdFat/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
Arduino_sketch_home/sdfatlib20100818/SdFat/Sd2Card.o: /Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/Sd2Card.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/sdfatlib20100818/SdFat/SdFile.o: /Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Arduino_sketch_home/sdfatlib20100818/SdFat/SdVolume.o: /Users/sin/Documents/Arduino/sdfatlib20100818/SdFat/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SPI" -I/usr/local/cross/avr/include/avr -DARDUINO=100 -DARDUINO_MAIN -Wall -Os -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega328p -DF_CPU=16000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


