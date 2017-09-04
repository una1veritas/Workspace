//
// LM73_test1
// sample program for reading LM73
// 
//   Figure.1 LM73 elekijack No.8 breakout
//
//              --------------                           
//              |    LM73    |                           
//              |O          O|<-SDA                        
//              |    =--=    |                           
//         GND->|O   =  =   O|                           
//              |    =--=    |                           
//         VDD->|O    C1    O|<-SCL                          
//              |    -||-    |                           
//              |            |                           
//              --------------                           
//     
//  NOTE:
//  Arduino analog input 5 - I2C SCL
//  Arduino analog input 4 - I2C SDA
//

#include <Wire.h>

#define LM73_ADDR 0b1001100 //0x4c          // 0b1001100

int ret;

void setup()                    // run once, when the sketch 
{
  pinMode(13,OUTPUT);
  digitalWrite(13,LOW);
  
  Wire.begin();
  Serial.begin(9600);
  delay(100); // 
  
  Wire.beginTransmission(LM73_ADDR);
  Wire.send((byte)0x04);  // select Control/Status Register
  Wire.send(0x60);        // select 14 bits mode
  ret=Wire.endTransmission();
  Serial.println(ret);

  Wire.beginTransmission(LM73_ADDR);
  Wire.send(0x00);        // select register
  ret=Wire.endTransmission();
  Serial.println(ret);
  delay(100); // bounce
  
}

void loop()                     // run over and over again
{

  long data=0 ;

  Wire.beginTransmission(LM73_ADDR);
  ret=Wire.requestFrom(LM73_ADDR, 2);
  
  if (Wire.available()) { // the 1first byte
    data = Wire.receive();
  } else {
    Serial.println("ERR!");
  }
  if (Wire.available()) { // the 2nd byte
    data = (data << 8 )| Wire.receive() ;
  }
  ret=Wire.endTransmission();
  data = data*100/128;
//  Serial.print(ret);
//  Serial.print("  ");
  Serial.print(data/(float)100);
  Serial.println("C");
  delay(2000);
}

