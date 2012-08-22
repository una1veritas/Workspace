// RXduino�̃T���v���v���O����
// Arduino�̃X�P�b�`���ɊȒP�Ƀv���O���������܂�
// (C) 2012 Tokushu Denshi Kairo Inc.

#include <rxduino.h> // ������C���N���[�h���邾����OK!!
#include <serial.h>
#include <spi.h>
#include <math.h>
#include <tkdn_tone.h>

volatile int count = 0;
bool accel_sensor_exist = false;

void setup()
{
	Serial.begin(38400);
	Serial.println("Start RXduino !");

	//SPI�������x�Z���T�p�ɏ�����
	SPI.begin();
	// ��SPI.begin���s���ƁAClockDivider�ABitLength�ABitOrder=LSBFIRST������������Ă��܂��̂ŁA
	// setClockDivider�Ȃǂ�begin�̌�ɍs������
	SPI.setClockDivider(SPI_CLOCK_DIV64);
	SPI.setBitLength(8);
	SPI.setBitOrder(MSBFIRST) ;
	SPI.port = SPI_PORT_NONE;

	pinMode(PIN_LED0,OUTPUT);
	pinMode(PIN_LED1,OUTPUT);
	pinMode(PIN_LED2,OUTPUT);
	pinMode(PIN_LED3,OUTPUT);

	//�֌W�Ȃ�CS��H���x���ɂ��Ă������Ƃ͕K�v
	pinMode(PIN_SPI_CS0,OUTPUT);
	digitalWrite(PIN_SPI_CS0,1);
	pinMode(PIN_SPI_CS1,OUTPUT);
	digitalWrite(PIN_SPI_CS1,1);
	pinMode(PIN_SPI_CS2,OUTPUT);
	digitalWrite(PIN_SPI_CS2,1);
	pinMode(PIN_SPI_CS3,OUTPUT);
	digitalWrite(PIN_SPI_CS3,1);
	pinMode(SPI_PORT_RAXINO_SDMMC,OUTPUT);
	digitalWrite(SPI_PORT_RAXINO_SDMMC,1);

	// �����x�Z���T��������
	SPI.port = SPI_PORT_RAXINO_ACCEL;
	SPI.setBitLength(16);
	SPI.transfer(0x2047); //

	unsigned long recv;
	recv = SPI.transfer(0x8f00) & 0xff; //
	Serial.print("ACCEL WHO_AM_I=");
	Serial.print(recv,CSerial::HEX);
	Serial.print(" ");
	if((recv != 0) and (recv != 0xff))
	{
		Serial.print("Accel sensor found.");
		accel_sensor_exist = true;
	}

	recv = SPI.transfer(0xa000) & 0xff; //
	Serial.print("CTRL1_REG1=");
	Serial.print(recv,CSerial::HEX);
	Serial.println(" ");

	// �R�}���h(8bit)�ƃf�[�^(24bit)��ʁX�ɑ���M���Ă݂�e�X�g
	SPI.port = SPI_PORT_NONE; // CS�������Ő��䂷��
	digitalWrite(PIN_SPI_CS2,0); // CS2��������
	SPI.setBitLength(8);
	SPI.transfer(0x9f); // �R�}���h���M
	SPI.setBitLength(24);
	recv = SPI.transfer(0); // �f�[�^(24bit)��M
	Serial.print("SPI ROM  JEDEC ID=");
	Serial.print(recv,CSerial::HEX);
	Serial.print(",");
	digitalWrite(PIN_SPI_CS2,1); // CS2��������

	digitalWrite(PIN_SPI_CS2,0); // CS2��������
	SPI.setBitLength(32);
	SPI.transfer(0x90000000); // �R�}���h���M 90 00 00 00
	recv = SPI.transfer(0); // �f�[�^��M
	Serial.print("ID=");
	Serial.print(recv,CSerial::HEX);
	Serial.print(",");
	digitalWrite(PIN_SPI_CS2,1); // CS2��������

	// �R�}���h(8bit)�ƃf�[�^(8bit)���ꏏ�ɑ���M���Ă݂�
	// CS�������œ������Ă݂�e�X�g
	SPI.port = SPI_PORT_RAXINO_ROM; // �|�[�g�ԍ���ݒ肵��CS�������I�ɓ�����
	SPI.setBitLength(16);
	recv = SPI.transfer(0x0500); // �R�}���h���M 05 00
	Serial.print("STATUS_REG=");
	Serial.print(recv & 0xff,CSerial::HEX);
	Serial.println("");

	delay(3000);
}

int anyreceived;

void loop()
{
	digitalWrite(PIN_LED0, count & 1);
	digitalWrite(PIN_LED1, count & 2);
	digitalWrite(PIN_LED2, count & 4);
	digitalWrite(PIN_LED3, count & 8);
	count++;

	unsigned long recv;

	// �����x�Z���T
	SPI.port = SPI_PORT_RAXINO_ACCEL;
	SPI.setBitLength(16);

	if(Serial.available())
	{
		anyreceived = 1;
		Serial.read();
	}

	if(accel_sensor_exist) {
		const float fullscale = 2.3; // 2.3g g=�d�͉����x

		do {
			recv = SPI.transfer(0xa700) & 0xff; //
			if(recv & 0x80) { // �I�[�o�[�������Ă���
				SPI.transfer(0xe900); // �ǂݎ̂�
				SPI.transfer(0xeb00);
				SPI.transfer(0xed00);
				continue;
			}
		} while ((recv & 0x08) == 0); //  �V�����f�[�^���Ȃ��Ȃ烋�[�v

		int x,y,z;
		x = (signed char)(SPI.transfer(0xe900) & 0xff);
		y = (signed char)(SPI.transfer(0xeb00) & 0xff);
		z = (signed char)(SPI.transfer(0xed00) & 0xff);

/*
		if(anyreceived)
		{
			Serial.print("X=");
			Serial.print((int)(x/128.*fullscale*1000.),DEC);
			Serial.print("mg ");
			Serial.print("Y=");
			Serial.print((int)(y/128.*fullscale*1000.),DEC);
			Serial.print("mg ");
			Serial.print("Z=");
			Serial.print((int)(z/128.*fullscale*1000.),DEC);
			Serial.println("mg ");
		}
*/
		
		double dx,dy,dz;
		dx = (int)x * (int)x;
		dy = (int)y * (int)y;
		dz = (int)z * (int)z;
		if(digitalRead(PIN_SW) == 1)
		{
			tone_start(PIN_P51,dx+dy+dz + 256,0);
		}
		else
		{
			tone_start(PIN_P51,dx+dy+dz + 256,50);
			delay(50);
		}
		
		if(anyreceived)
		{
			Serial.print("X=");
			Serial.print((int)(dx),CSerial::DEC);
			Serial.print(" Y=");
			Serial.print((int)(dy),CSerial::DEC);
			Serial.print(" Z=");
			Serial.print((int)(dz),CSerial::DEC);
			Serial.print(" Freq=");
			Serial.print((int)sqrt(dx+dy+dz) + 256,CSerial::DEC);
			Serial.println("Hz ");
		}

	}
}
