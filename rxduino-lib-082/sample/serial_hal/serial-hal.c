// RX62NとGCCで、シリアルポートを使うサンプル
// 特殊電子回路㈱

#include <tkdn_hal.h>
#include <stdio.h>

int main()
{
	int count = 0;
    sci_init(SCI_AUTO,38400); 
    sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換 

	sci_puts("\Serial sample program (HAL version)\n");
	sci_puts("CRとLFのコード変換も行っています\n");

	gpio_set_pinmode(PIN_LED3,1); // LED3(BUZZ)を出力

	while(1)
	{
		if(sci_rxcount()) // 何か受信した文字がある
		{
			char tmp[128];
			char c = sci_getc(); // 1文字受信
			sci_putc(c); // エコーバック
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
			sci_puts(tmp); // 文字コードを10進で表示
			gpio_write_port(PIN_LED3 , count++ & 1);
		}
	}

}
