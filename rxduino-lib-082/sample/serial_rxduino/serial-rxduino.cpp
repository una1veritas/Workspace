// RX62NとGCCで、シリアルポートを使うサンプル
// RXduinoバージョン
// 特殊電子回路㈱

#include <rxduino.h>

void setup()
{
	Serial.begin(38400);

	Serial.println("Serial sample program (RXduino version)");
	Serial.println("CRとLFのコード変換も行っています");

	pinMode(PIN_LED3,OUTPUT);
}

int count = 0;

void loop()
{
	while(1)
	{
		if(Serial.available()) // 何か受信した文字がある
		{
			char tmp[10];
			char c = Serial.read(); // 1文字受信
			tmp[0] = c;
			tmp[1] = '\0';
			Serial.print(tmp); // エコーバック
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
				Serial.print(c); // 文字コードを10進で表示
				Serial.print("]");
			}
			digitalWrite(PIN_LED3 , count++ & 1);
		}
	}

}
