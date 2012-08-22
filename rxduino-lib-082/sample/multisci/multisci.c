// RX62NとGCCで、シリアルポートを使うサンプル
// 特殊電子回路㈱

#include <tkdn_hal.h>
#include <tkdn_sci.h>
#include <stdio.h>

int main()
{
	sci_t sci1;
	sci_t sci2;
	int led1,led2;
	
	sci_init_ex(&sci1,SCI_USB0,38400); // sci1をUSB0に割り当て
	sci_convert_crlf_ex(&sci1,CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換 

	sci_init_ex(&sci2,SCI_SCI0P2x,38400); // sci2をSCI0に割り当て
	sci_convert_crlf_ex(&sci2,CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換 

	led1 = 0;
	led2 = 0;

	gpio_set_pinmode(PIN_LED0,1);
	gpio_set_pinmode(PIN_LED1,1);

	while(1)
	{
		if(sci_rxcount_ex(&sci1)) // sci1から何か受信した文字がある
		{
			char c = sci_getc_ex(&sci1); // 1文字受信
			sci_putc_ex(&sci1,c); // エコーバック
			sci_putc_ex(&sci2,c); // エコーバック
			gpio_write_port(PIN_LED1 , led1++ & 1);
		}

		if(sci_rxcount_ex(&sci2)) // sci2から何か受信した文字がある
		{
			char c = sci_getc_ex(&sci2); // 1文字受信
			sci_putc_ex(&sci1,c); // エコーバック
			sci_putc_ex(&sci2,c); // エコーバック
			gpio_write_port(PIN_LED2 , led2++ & 1);
		}
	}

}
