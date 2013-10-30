################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../knapsack/knapsack.cpp 

C_SRCS += \
../knapsack/ksubset.c \
../knapsack/main.c \
../knapsack/subsetenum.c 

CC_SRCS += \
../knapsack/knapsack-dp.cc 

OBJS += \
./knapsack/knapsack-dp.o \
./knapsack/knapsack.o \
./knapsack/ksubset.o \
./knapsack/main.o \
./knapsack/subsetenum.o 

C_DEPS += \
./knapsack/ksubset.d \
./knapsack/main.d \
./knapsack/subsetenum.d 

CC_DEPS += \
./knapsack/knapsack-dp.d 

CPP_DEPS += \
./knapsack/knapsack.d 


# Each subdirectory must supply rules for building sources it contributes
knapsack/%.o: ../knapsack/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

knapsack/%.o: ../knapsack/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

knapsack/%.o: ../knapsack/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


