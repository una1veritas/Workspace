// RX62N��GCC�T���v���v���O���� 
// 3�������x�Z���T KXM52�T���v�� 
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
    int val1,val2,val3; 
  
    int analogPin1 = 1; //�A�i���O���͂�1�ԃs�� 
    int analogPin2 = 2; //�A�i���O���͂�2�ԃs�� 
    int analogPin3 = 3; //�A�i���O���͂�3�ԃs�� 
  
    Serial.println("RXDuino KXM52-1050 3�������x�Z���T�[�e�X�g"); 
    Serial.println("G1_data, G2_data"); 
    while(1){ 
  
        val1 = analogRead(analogPin1); 
        Serial.print((int)(val1 * 3.3 / 1024. * 1000)); 
        Serial.print("[mV] , "); 
        val2 = analogRead(analogPin2); 
        Serial.print((int)(val2 * 3.3 / 1024. * 1000)); 
        Serial.print("[mV] , "); 
        val3 = analogRead(analogPin3); 
        Serial.print((int)(val3 * 3.3 / 1024. * 1000)); 
        Serial.println("[mV]");    //���s�t��print 
        delay(20); 
    } 
} 
