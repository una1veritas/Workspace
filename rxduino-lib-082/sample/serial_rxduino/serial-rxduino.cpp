// RX62N��GCC�ŁA�V���A���|�[�g���g���T���v��
// RXduino�o�[�W����
// ����d�q��H��

#include <rxduino.h>

void setup()
{
	Serial.begin(38400);

	Serial.println("Serial sample program (RXduino version)");
	Serial.println("CR��LF�̃R�[�h�ϊ����s���Ă��܂�");

	pinMode(PIN_LED3,OUTPUT);
}

int count = 0;

void loop()
{
	while(1)
	{
		if(Serial.available()) // ������M��������������
		{
			char tmp[10];
			char c = Serial.read(); // 1������M
			tmp[0] = c;
			tmp[1] = '\0';
			Serial.print(tmp); // �G�R�[�o�b�N
			if(c == 0x0d)
			{
				Serial.print("[\\r]");
			}
			else if(c == 0x0a)
			{
				Serial.print("[\\n]");
			}
			else
			{
				Serial.print("[");
				Serial.print(c); // �����R�[�h��10�i�ŕ\��
				Serial.print("]");
			}
			digitalWrite(PIN_LED3 , count++ & 1);
		}
	}

}
