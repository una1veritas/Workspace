################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../freeRTOS711_All_Files/MegaW5100Test/main.c \
../freeRTOS711_All_Files/MegaW5100Test/webserver.c 

S_SRCS += \
../freeRTOS711_All_Files/MegaW5100Test/xitoa.s 

OBJS += \
./freeRTOS711_All_Files/MegaW5100Test/main.o \
./freeRTOS711_All_Files/MegaW5100Test/webserver.o \
./freeRTOS711_All_Files/MegaW5100Test/xitoa.o 

C_DEPS += \
./freeRTOS711_All_Files/MegaW5100Test/main.d \
./freeRTOS711_All_Files/MegaW5100Test/webserver.d 

S_DEPS += \
./freeRTOS711_All_Files/MegaW5100Test/xitoa.d 


# Each subdirectory must supply rules for building sources it contributes
freeRTOS711_All_Files/MegaW5100Test/%.o: ../freeRTOS711_All_Files/MegaW5100Test/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Compiler'
	avr-gcc -Wall -g2 -gstabs -O0 -fpack-struct -fshort-enums -std=gnu99 -funsigned-char -funsigned-bitfields -mmcu=atmega16 -DF_CPU=1000000UL -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

freeRTOS711_All_Files/MegaW5100Test/%.o: ../freeRTOS711_All_Files/MegaW5100Test/%.s
	@echo 'Building file: $<'
	@echo 'Invoking: AVR Assembler'
	avr-gcc -x assembler-with-cpp -g2 -gstabs -mmcu=atmega16 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


