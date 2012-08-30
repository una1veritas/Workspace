// RX62N��GCC�T���v���v���O����
// �C�[�T�l�b�g
// (C)Copyright 2011 ����d�q��H

// �g����
// ���̃v���O�����̃t�@�C������main.c�ɕύX���āAmake���Ă��������B

#ifdef __GNUC__
  #ifdef CPU_IS_RX62N
    #include "iodefine_gcc62n.h"
  #endif
  #ifdef CPU_IS_RX63N
    #include "iodefine_gcc63n.h"
  #endif
#endif
#ifdef __RENESAS__
  #include "iodefine.h"
#endif

// ���dHAL
#include "tkdn_hal.h"

#include <stdio.h>
#include <ctype.h>

int main()
{
	int i;
	char tmp[128];

	sci_init(SCI_USB0,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \n��\r\n�ɕϊ�

	sci_puts("\nEthernet sample program\n");
	sprintf(tmp,"Compiled at %s %s\n",__DATE__,__TIME__);
	sci_puts(tmp);

//	PORT1.DDR.BYTE |= 0x80;

	unsigned char macaddr[6] = {6,5,4,3,2,1};
	ether_open(macaddr);

	gpio_set_pinmode(PIN_SW,0);

	while(1)
	{
		if(gpio_read_port(PIN_SW) == 0)
		{
			ether_autonegotiate();
			timer_wait_ms(100);
		}

		if(ether_link_changed())
		{
			if(ether_is_linkup())     
			{
				sci_puts("Link:UP   ");
				if(ether_is_100M())       sci_puts("100M ");
				else                      sci_puts("10M  ");
				if(ether_is_fullduplex()) sci_puts("full\n");
				else                      sci_puts("half\n");
			}
			else
			{
				sci_puts("Link:DOWN\n");
			}
		}

		unsigned char rdata[2048];
		int rbytes = ether_read(rdata);

		if(rbytes > 0)
		{
//			PORT1.DR.BYTE ^= (1 << 7);

			sprintf(tmp,"Ethernet packet received %d bytes\n",rbytes);
			sci_puts(tmp);

			for(i=0;i<(rbytes + 15)/16;i++)
			{
				sprintf(tmp,"%04x  ",i*16);
				sci_puts(tmp);
	
				int j;
				for(j=0;j<16;j++)
				{
					if(j == 8) sci_puts(" ");
					if(i*16+j >= rbytes)
					{
						sci_puts("   ");
					}
					else
					{
						sprintf(tmp,"%02x ",rdata[i*16+j]);
						sci_puts(tmp);
					}
				}
	
				sci_puts(" ");
	
				for(j=0;j<16;j++)
				{
					if(j == 8) sci_puts(" ");
					if(i*16+1 >= rbytes)
					{
						sci_puts(" ");
					}
					else
					{
						if(isprint(rdata[i*16+j]))
						{
							sprintf(tmp,"%c",rdata[i*16+j]);
							sci_puts(tmp);
						}
						else
						{
							sci_puts(".");
						}
					}
				}
				sci_puts("\n");
			}
		}
	}
}

