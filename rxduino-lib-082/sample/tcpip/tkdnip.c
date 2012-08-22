// RX62NのGCCサンプルプログラム
// イーサネット
// (C)Copyright 2011 特殊電子回路

// 使い方
// このプログラムのファイル名をmain.cに変更して、makeしてください。

#include "tkdn_hal.h"
#include "tkdnip.h"
#include "arp.h"
#include "ip.h"
#include "icmp.h"

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

int g_dump; // パケットダンプの有効/無効

// 送受信バッファをどこに置くか
//#if (TARGET_BOARD == BOARD_RAXINO) //  ・・・ SDRAMはないので、内蔵RAM上
unsigned char *sendpacket = (unsigned char *)0x00010000;
unsigned char *recvpacket = (unsigned char *)0x00011000;
//#elif (TARGET_BOARD == BOARD_ULT62N) || (TARGET_BOARD == BOARD_RXMEGA)
//                                   //  ・・・ SDRAM上
//unsigned char *sendpacket = (unsigned char *)0x08000000;
//unsigned char *recvpacket = (unsigned char *)0x08001000;
//#endif

// デフォルトの設定値
unsigned char my_macaddr[6] = {0x02,0x00,0x00,0x00,0x00,0x01};
unsigned char my_ipaddr[4]  = {192,168,2,123};
unsigned char my_ipmask[4]  = {255,255,255,0};
unsigned char gw_ipaddr[4]  = {192,168,2,1};

void packet_dump(unsigned char *buf,int len) {
	int i;
	char tmp[80];
	for(i=0;i<(len + 15)/16;i++) {
		sprintf(tmp,"%04x	",i*16);
		sci_puts(tmp);
		int j;
		for(j=0;j<16;j++) {
			if(j == 8) sci_puts(" ");
			if(i*16+j >= len) {
				sci_puts("	 ");
			}
			else {
				sprintf(tmp,"%02x ",buf[i*16+j]);
				sci_puts(tmp);
			}
		}
		sci_puts(" ");
		for(j=0;j<16;j++) {
			if(j == 8) sci_puts(" ");
			if(i*16+1 >= len) {
				sci_puts(" ");
			}
			else {
				if(isprint(buf[i*16+j])) {
					sprintf(tmp,"%c",buf[i*16+j]);
					sci_puts(tmp);
				}
				else {
					sci_puts(".");
				}
			}
		}
		sci_puts("\n");
	}
}

void check_link()
{
	if(ether_is_linkup()) {
		gpio_write_port(PIN_LED0,0);
		sci_puts("Link:UP	 ");
		if(ether_is_100M()) sci_puts("100M ");
		else                sci_puts("10M	");

		if(ether_is_fullduplex()) sci_puts("full\n");
		else                sci_puts("half\n");
		
	}
	else {
		gpio_write_port(PIN_LED0,1);
		sci_puts("Link:DOWN\n");
	}
}

void packet_receive_process()
{
	char tmp[128];
	int rbytes = ether_read(recvpacket);
		
	if(rbytes <= 0) return;
	ethernet_frame_str *ep = (ethernet_frame_str *)recvpacket;

	if(g_dump)
	{
		sprintf(tmp,"[Ethernet frame type=%04x received. %d bytes]\n",__builtin_rx_revw(ep->type),rbytes);
		sci_puts(tmp);
		packet_dump(recvpacket,rbytes);
	}

	 // 何か受信した
	if(ep->type == CONST_REVW(0x0806))
	{
		arp_received(&ep->payload[0],rbytes - sizeof(ethernet_frame_str));
		return;
	}

	if(ep->type == CONST_REVW(0x0800))
	{
		ip_headder_str *ip = (ip_headder_str *)&ep->payload[0];
		unsigned long myip = my_ipaddr[0] | (my_ipaddr[1] << 8) | (my_ipaddr[2] << 16) | (my_ipaddr[3] << 24);

		if(ip->dest_addr != myip)
		{
//			sprintf(tmp,"[Received IP addr %08lx is not my addr %08lx]\n",ip->dest_addr,myip);
//			sci_puts(tmp);
			return;
		}
		
		if(ip->protocol == PROTO_UDP)
		{
			sprintf(tmp,"[UDP received %d bytes]\n",rbytes);
			sci_puts(tmp);
			return;
		}

		if(ip->protocol == PROTO_ICMP)
		{
			icmp_received(recvpacket,rbytes);
			return;
		}

		if(ip->protocol == PROTO_TCP)
		{
			sprintf(tmp,"[TCP received %d bytes]\n",rbytes);
			sci_puts(tmp);
			packet_dump(recvpacket,rbytes);
			return;
		}

		{
			sprintf(tmp,"[Unknown IP packet received %d bytes]\n",rbytes);
			sci_puts(tmp);
			packet_dump(recvpacket,rbytes);
			return;
		}
		return;
	}
	sprintf(tmp,"[Unknown ethernet frame %04x received %d bytes]\n",__builtin_rx_revw(ep->type),rbytes);
	sci_puts(tmp);
	packet_dump(recvpacket,rbytes);
}

void show_prompt()
{
	sci_puts("RX62N> ");
}

void userapp_ping(unsigned char ipaddr[4])
{
	int i;
	for(i=0;;i++) // 送信回数は4回
	{
		char c;
		send_ping(ipaddr);
		while(g_ping_process)
		{
			if(timer_get_us() - g_ping_timer > 5000000) // タイムアウト5秒
			{
				sci_puts("Request timed out.\n");
				break;
			}
			while((c = sci_getc()) != '\0')
			{
				if(c == 0x03) return;
			}
		}
		timer_wait_ms(1000); // 送信間隔は1000ms
	}
}

unsigned long myrand()
{
	return rand() ^ (rand() << 12) ^ (rand() << 20);
}

void sdram_test()
{
	unsigned int i;
	char tmp[80];
	unsigned long *p;

	sci_puts("SDRAM write..");
	srand(0);
	p = (unsigned long *)0x08000000;
	for(i=0;i<16777216/4;i++)
	{
		*p++ = myrand();
	}
	sci_puts("done\n");

	sci_puts("wait 10 sec\n");
	timer_wait_ms(10000);

	sci_puts("SDRAM read..");
	srand(0);
	p = (unsigned long *)0x08000000;
	for(i=0;i<16777216/4;i++)
	{
		unsigned long val = myrand();
		if(*p != val)
		{
			sprintf(tmp,"SDRAM error count=%d read=%08lx write=%08lx\n",i,*p,val);
			sci_puts(tmp);
		}
		p++;
	}
	sci_puts("done\n");
	sci_puts("SDRAM test done.\n");
}

void ipconfig()
{
	char tmp[128];
	sprintf(tmp,"IP Address. . . . . . . . . . . . : %d.%d.%d.%d\n",my_ipaddr[0],my_ipaddr[1],my_ipaddr[2],my_ipaddr[3]);
	sci_puts(tmp);
	sprintf(tmp,"Subnet Mask . . . . . . . . . . . : %d.%d.%d.%d\n",my_ipmask[0],my_ipmask[1],my_ipmask[2],my_ipmask[3]);
	sci_puts(tmp);
	sprintf(tmp,"Default Gateway . . . . . . . . . : %d.%d.%d.%d\n",gw_ipaddr[0],gw_ipaddr[1],gw_ipaddr[2],gw_ipaddr[3]);
	sci_puts(tmp);
}

void dump(char *title,unsigned long startaddr,int len)
{
	char tmp[10];
	int i;
	sci_puts(title);
	sci_puts("\n");
	while(len)
	{
		if((startaddr & 31) == 0)
		{
			sprintf(tmp,"%08lx  ",startaddr);
			sci_puts(tmp);
		}
		sprintf(tmp,"%08lx ",*(unsigned long *)startaddr);
		sci_puts(tmp);
		if((startaddr & 31) == 28)
		{
			sci_puts("\n");
		}
		startaddr+=4;
		len -= 4;
	}
}

int main() {
	char tmp[128];

	gpio_set_pinmode(PIN_LED0,1);
	gpio_set_pinmode(PIN_LED1,1);
	gpio_set_pinmode(PIN_LED2,1);
	gpio_set_pinmode(PIN_LED3,1);
//	gpio_write_port(PIN_LED2,1);
//	gpio_write_port(PIN_LED3,1);
	
	sci_init(SCI_AUTO,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換
	
	sci_puts("\nRX62N Ethernet sample program.\n");
	sci_puts("Tokushu Denshi Kairo Inc.\n");
	sprintf(tmp,"Compiled at %s %s\n",__DATE__,__TIME__);
	sci_puts(tmp);

//	gpio_write_port(PIN_LED3,0);

	ether_open(my_macaddr);

	// イーサフレームを受信したときに実行される割り込み関数を登録
	ether_regist_user_rx_procedure(packet_receive_process);
	if(!ether_is_linkup()) check_link();

	while(1) {
		if(ether_link_changed()) check_link();

		if(sci_rxcount())
		{
			char cmd[32];
			sci_gets(cmd,32);
			char *tmp = strtok(cmd," ");
			if(tmp)
			{
				if(!strcmp(tmp,"help"))
				{
					sci_puts("COMMAND        Description\n");
					sci_puts("------------------------------------------\n");
					sci_puts("help           Show this message.\n");
					sci_puts("arp -a         Show arp table.\n");
					sci_puts("ipconfig       Show IP configuration\n");
					sci_puts("ping target    Ping the specified target.\n");
					sci_puts("link           Retry link-up and negotiation.\n");
					sci_puts("sdtest         Test SDRAM.\n");
					sci_puts("dump [on|off]  On/Off received packet dump.\n");
					show_prompt();
					continue;
				}

				if(!strcmp(tmp,"link"))
				{
					ether_autonegotiate();
					show_prompt();
					continue;
				}

				if(!strcmp(tmp,"arp"))
				{
					tmp = strtok(NULL," ");
					if(tmp && !strcmp(tmp,"-a"))
					{
						arp_table_show();
					}
					show_prompt();
					continue;
				}

				if(!strcmp(tmp,"sdtest"))
				{
					ether_close();
					sdram_test();
					ether_open(my_macaddr);
					show_prompt();
					continue;
				}

				if(!strcmp(tmp,"ipconfig"))
				{
					ipconfig();
					show_prompt();
					continue;
				}

				if(!strcmp(tmp,"dump"))
				{
					tmp = strtok(NULL," ");
					if(tmp)
					{
						if(!strcmp(tmp,"on")) g_dump = 1;
						if(!strcmp(tmp,"off")) g_dump = 0;
					}
					show_prompt();
					continue;
				}

				if(!strcmp(tmp,"ping"))
				{
					unsigned char ipaddr[4];
					
					do
					{
						tmp = strtok(NULL,".");
						if(!tmp) break;
						ipaddr[0] = atoi(tmp);
					
						tmp = strtok(NULL,".");
						if(!tmp) break;
						ipaddr[1] = atoi(tmp);
					
						tmp = strtok(NULL,".");
						if(!tmp) break;
						ipaddr[2] = atoi(tmp);
					
						tmp = strtok(NULL,"");
						if(!tmp) break;
						ipaddr[3] = atoi(tmp);
	
						userapp_ping(ipaddr);
					} 	while(0);
					show_prompt();
					continue;
				}
				sci_puts("Unknwon command ");
				sci_puts(tmp);
				sci_puts("\n");
			}
			show_prompt();
		}
		
	}
}

