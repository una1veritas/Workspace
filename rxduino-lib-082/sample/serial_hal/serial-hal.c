// RX62N��GCC�ŁA�V���A���|�[�g���g���T���v��
// ����d�q��H��

#include <tkdn_hal.h>
#include <stdio.h>

int main()
{
	int count = 0;
    sci_init(SCI_AUTO,38400); 
    sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \n��\r\n�ɕϊ� 

	sci_puts("\Serial sample program (HAL version)\n");
	sci_puts("CR��LF�̃R�[�h�ϊ����s���Ă��܂�\n");

	gpio_set_pinmode(PIN_LED3,1); // LED3(BUZZ)���o��

	while(1)
	{
		if(sci_rxcount()) // ������M��������������
		{
			char tmp[128];
			char c = sci_getc(); // 1������M
			sci_putc(c); // �G�R�[�o�b�N
			if(c == 0x0d)
			{
				sprintf(tmp,"[\\r]");
			}
			else if(c == 0x0a)
			{
				sprintf(tmp,"[\\n]");
			}
			else
			{
				sprintf(tmp,"[%d]",c);
			}
			sci_puts(tmp); // �����R�[�h��10�i�ŕ\��
			gpio_write_port(PIN_LED3 , count++ & 1);
		}
	}

}
