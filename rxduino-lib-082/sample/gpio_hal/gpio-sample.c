// RX62N��GCC�T���v���v���O����
// GPIO�T���v��
// (C)Copyright 2011 ����d�q��H

// ���dHAL
#include <tkdn_hal.h>

int main()
{
	int count = 0;

	sci_init(SCI_AUTO,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \n��\r\n�ɕϊ�

	sci_puts("\nGPIO sample program (HAL version)\n");

	gpio_set_pinmode(PIN_LED0,1); // LED0���o��
	gpio_set_pinmode(PIN_LED1,1); // LED1���o��
	gpio_set_pinmode(PIN_LED2,1); // LED2���o��
	gpio_set_pinmode(PIN_LED3,1); // LED3(BUZZ)���o��
	gpio_set_pinmode(PIN_SW,0);   // SW�����

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

