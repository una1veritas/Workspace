// RX62NのGCCサンプルプログラム 
// 3軸加速度センサ KXM52サンプル 
// (C)Copyright 2011 特殊電子回路 

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
  
    int analogPin1 = 1; //アナログ入力の1番ピン 
    int analogPin2 = 2; //アナログ入力の2番ピン 
    int analogPin3 = 3; //アナログ入力の3番ピン 
  
    Serial.println("RXDuino KXM52-1050 3軸加速度センサーテスト"); 
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
        Serial.println("[mV]");    //改行付きprint 
        delay(20); 
    } 
} 
