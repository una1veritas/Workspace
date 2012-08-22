// RX62NのGCCサンプルプログラム
// GPIOサンプル
// (C)Copyright 2011 特殊電子回路

// 特電HAL
#include <tkdn_hal.h>

int main()
{
	int count = 0;

	sci_init(SCI_AUTO,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換

	sci_puts("\nGPIO sample program (HAL version)\n");

	gpio_set_pinmode(PIN_LED0,1); // LED0を出力
	gpio_set_pinmode(PIN_LED1,1); // LED1を出力
	gpio_set_pinmode(PIN_LED2,1); // LED2を出力
	gpio_set_pinmode(PIN_LED3,1); // LED3(BUZZ)を出力
	gpio_set_pinmode(PIN_SW,0);   // SWを入力

	while(1)
	{
		char tmp[128];
		sprintf(tmp,"%d ",count & 15);
		sci_puts(tmp);

		gpio_write_port(PIN_LED0,count & 1);
		gpio_write_port(PIN_LED1,count & 2);
		gpio_write_port(PIN_LED2,count & 4);
		gpio_write_port(PIN_LED3,count & 8);
		if(gpio_read_port(PIN_SW) == 1) timer_wait_ms(100);
		timer_wait_ms(1);

		count++;
	}
}

