// RX62N��GCC�ŁA�V���A���|�[�g���g���T���v��
// ����d�q��H��

#include <tkdn_hal.h>
#include <tkdn_sci.h>
#include <stdio.h>

int main()
{
	sci_t sci1;
	sci_t sci2;
	int led1,led2;
	
	sci_init_ex(&sci1,SCI_USB0,38400); // sci1��USB0�Ɋ��蓖��
	sci_convert_crlf_ex(&sci1,CRLF_CRLF,CRLF_CRLF); // \n��\r\n�ɕϊ� 

	sci_init_ex(&sci2,SCI_SCI0P2x,38400); // sci2��SCI0�Ɋ��蓖��
	sci_convert_crlf_ex(&sci2,CRLF_CRLF,CRLF_CRLF); // \n��\r\n�ɕϊ� 

	led1 = 0;
	led2 = 0;

	gpio_set_pinmode(PIN_LED0,1);
	gpio_set_pinmode(PIN_LED1,1);

	while(1)
	{
		if(sci_rxcount_ex(&sci1)) // sci1���牽����M��������������
		{
			char c = sci_getc_ex(&sci1); // 1������M
			sci_putc_ex(&sci1,c); // �G�R�[�o�b�N
			sci_putc_ex(&sci2,c); // �G�R�[�o�b�N
			gpio_write_port(PIN_LED1 , led1++ & 1);
		}

		if(sci_rxcount_ex(&sci2)) // sci2���牽����M��������������
		{
			char c = sci_getc_ex(&sci2); // 1������M
			sci_putc_ex(&sci1,c); // �G�R�[�o�b�N
			sci_putc_ex(&sci2,c); // �G�R�[�o�b�N
			gpio_write_port(PIN_LED2 , led2++ & 1);
		}
	}

}
