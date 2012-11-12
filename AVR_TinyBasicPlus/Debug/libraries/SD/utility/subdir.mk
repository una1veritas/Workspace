################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/Sd2Card.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdFile.cpp \
/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdVolume.cpp 

OBJS += \
./libraries/SD/utility/Sd2Card.o \
./libraries/SD/utility/SdFile.o \
./libraries/SD/utility/SdVolume.o 

CPP_DEPS += \
./libraries/SD/utility/Sd2Card.d \
./libraries/SD/utility/SdFile.d \
./libraries/SD/utility/SdVolume.d 


# Each subdirectory must supply rules for building sources it contributes
libraries/SD/utility/Sd2Card.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/Sd2Card.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SD" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility" -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

libraries/SD/utility/SdFile.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdFile.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SD" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility" -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

libraries/SD/utility/SdVolume.o: /Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility/SdVolume.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: AVR C++ Compiler'
	avr-g++ -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino" -I"/Applications/Arduino.app/Contents/Resources/Java/hardware/arduino/variants/standard" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SD" -I"/Applications/Arduino.app/Contents/Resources/Java/libraries/SD/utility" -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -funsigned-char -funsigned-bitfields -fno-exceptions -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


