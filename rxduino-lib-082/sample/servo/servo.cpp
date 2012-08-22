// RXduinoのサンプルプログラム

// Arduinoのスケッチ風に簡単にプログラムが作れます

#include <stdio.h>
#include <stdlib.h>
#include <tkdn_servo.h>
#include <tkdn_pwm.h>

servo_t *servo;


int x = 0;

unsigned char ledbright(unsigned char x)
{
	return (int)x * (int)x * (int)x / 65536;
}

int main()
{
	gpio_set_pinmode(PIN_LED0,1);
	gpio_set_pinmode(PIN_LED1,1);
	gpio_set_pinmode(PIN_LED2,1);
	gpio_set_pinmode(PIN_LED3,1);
	gpio_set_pinmode(PIN_SW,0);
	gpio_write_port(PIN_LED2,1);

//	sci_init(SCI_AUTO,38400);
	sci_init(SCI_SCI0P2x,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF);

	servo_str servo;
	servo_attach(&servo,3,0,0); //3番ピン使用
	servo_write_us(&servo,1000);
//	pwm_init();

	sci_puts("Hello rxduino\n");
	sci_puts("Compiled at ");
	sci_puts(__DATE__);
	sci_puts(" ");
	sci_puts(__TIME__);
	sci_puts("\n");

	while(1)
	{
		sci_puts(".");
	
		pwm_output(PIN_LED0,ledbright(~(x + 0)));
		pwm_output(PIN_LED1,ledbright(~(x + 64)));
		pwm_output(PIN_LED2,ledbright(~(x + 128)));
		pwm_output(PIN_LED3,ledbright(~(x + 192)));
		printf("servo = %d\n",x);
		servo_write_us(&servo,x);
		x++;
		x = x & 255;
	
		if(sci_rxcount())
		{
			char c = sci_getc();
			pwm_output(PIN_ARD0,x);
			sci_putc(c);
			servo_write(&servo,(c - '0') * 20);
	/*
			Serial.write(" => ");
			Serial.print((c - '0') * 20,DEC);
			Serial.println("degree");
	*/		
		}
		else
		{
			timer_wait_ms(10);
		}
	}
}
