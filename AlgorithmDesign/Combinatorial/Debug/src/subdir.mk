################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Combinatorial\ Programming.cpp 

C_SRCS += \
../src/main.c 

OBJS += \
./src/Combinatorial\ Programming.o \
./src/main.o 

C_DEPS += \
./src/main.d 

CPP_DEPS += \
./src/Combinatorial\ Programming.d 


# Each subdirectory must supply rules for building sources it contributes
src/Combinatorial\ Programming.o: ../src/Combinatorial\ Programming.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/Combinatorial Programming.d" -MT"src/Combinatorial\ Programming.d" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


