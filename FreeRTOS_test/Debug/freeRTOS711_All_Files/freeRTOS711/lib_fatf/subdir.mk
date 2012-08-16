################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/freeRTOS711/lib_fatf/cc932.c \
../freeRTOS711_All_Files/freeRTOS711/lib_fatf/ccsbcs.c \
../freeRTOS711_All_Files/freeRTOS711/lib_fatf/diskio.c \
../freeRTOS711_All_Files/freeRTOS711/lib_fatf/ff.c \
../freeRTOS711_All_Files/freeRTOS711/lib_fatf/xmodem.c 

OBJS += \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/cc932.o \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/ccsbcs.o \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/diskio.o \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/ff.o \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/xmodem.o 

C_DEPS += \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/cc932.d \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/ccsbcs.d \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/diskio.d \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/ff.d \
./freeRTOS711_All_Files/freeRTOS711/lib_fatf/xmodem.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/freeRTOS711/lib_fatf/%.o: ../freeRTOS711_All_Files/freeRTOS711/lib_fatf/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


