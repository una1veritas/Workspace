// RX62NのGCCサンプルプログラム 
// 圧電振動ジャイロサンプル 
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
    int val1,val2; 
    int analogPin1 = 1; //アナログ入力の１番ピン 
    int analogPin2 = 2; //アナログ入力の２番ピン 
  
    Serial.println("RXDuino 圧電振動ジャイロテスト"); 
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