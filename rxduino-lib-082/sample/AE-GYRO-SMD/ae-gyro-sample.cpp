// RX62N��GCC�T���v���v���O���� 
// ���d�U���W���C���T���v�� 
// (C)Copyright 2011 ����d�q��H 
  
#include <rxduino.h> 
#include <stdlib.h> 
  
void setup() 
{ 
    Serial.begin(38400); 
  
    pinMode(PIN_SW,INPUT); 
    pinMode(PIN_LED3,OUTPUT); 
} 
  
int count = 0; 
  
void loop() 
{ 
    int val1,val2; 
    int analogPin1 = 1; //�A�i���O���͂̂P�ԃs�� 
    int analogPin2 = 2; //�A�i���O���͂̂Q�ԃs�� 
  
    Serial.println("RXDuino ���d�U���W���C���e�X�g"); 
    Serial.println("G1_data, G2_data"); 
    while(1){ 
  
        val1 = analogRead(analogPin1); 
        Serial.print((int)(1000 * val1 * 3.3 / 1024.)); 
        Serial.print("[mV] , "); 
        val2 = analogRead(analogPin2); 
        Serial.print((int)(1000 * val2 * 3.3 / 1024.)); 
        Serial.println("[mV]"); 
        delay(20); 
    } 
} 