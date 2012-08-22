// RX62N��GCC�T���v���v���O����
// �C�[�T�l�b�g
// (C)Copyright 2011 ����d�q��H

// �g����
// ���̃v���O�����̃t�@�C������main.c�ɕύX���āAmake���Ă��������B

#include "tkdnbase.h"
#include "tkdn_hal.h"
#include "tkdnip.h"
#include "arp.h"
#include "ip.h"
#include "icmp.h"

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include "libmisc.h"

int g_success = 0;

int g_dump; // �p�P�b�g�_���v�̗L��/����

// ����M�o�b�t�@���ǂ��ɒu����
#if (TARGET_BOARD == BOARD_RAXINO) //  �E�E�E SDRAM�͂Ȃ��̂ŁA����RAM��
unsigned char *sendpacket = (unsigned char *)0x00010000;
unsigned char *recvpacket = (unsigned char *)0x00011000;
#elif (TARGET_BOARD == BOARD_ULT62N) || (TARGET_BOARD == BOARD_RXMEGA)
                                   //  �E�E�E SDRAM��
unsigned char *sendpacket = (unsigned char *)0x08000000;
unsigned char *recvpacket = (unsigned char *)0x08001000;
#endif

// �f�t�H���g�̐ݒ�l
unsigned char my_macaddr[6] = {0x02,0x00,0x00,0x00,0x00,0x01};
unsigned char my_ipaddr[4]  = {192,168,2,123};
unsigned char my_ipmask[4]  = {255,255,255,0};
unsigned char gw_ipaddr[4]  = {192,168,2,1};

void packet_dump(unsigned char *buf,int len) {
	int i;
	for(i=0;i<(len + 15)/16;i++) {
		printf("%04x	",i*16);
		int j;
		for(j=0;j<16;j++) {
			if(j == 8) sci_puts(" ");
			if(i*16+j >= len) {
				sci_puts("	 ");
			}
			else {
				printf("%02x ",buf[i*16+j]);
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
					printf("%c",buf[i*16+j]);
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
	int rbytes = ether_read(recvpacket);
		
	if(rbytes <= 0) return;
	ethernet_frame_str *ep = (ethernet_frame_str *)recvpacket;

	if(g_dump)
	{
		printf("[Ethernet frame type=%04x received. %d bytes]\n",__builtin_rx_revw(ep->type),rbytes);
		packet_dump(recvpacket,rbytes);
	}

	 // ������M����
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
			printf("[UDP received %d bytes]\n",rbytes);
			return;
		}

		if(ip->protocol == PROTO_ICMP)
		{
			icmp_received(recvpacket,rbytes);
			return;
		}

		if(ip->protocol == PROTO_TCP)
		{
			printf("[TCP received %d bytes]\n",rbytes);
			packet_dump(recvpacket,rbytes);
			return;
		}

		{
			printf("[Unknown IP packet received %d bytes]\n",rbytes);
			packet_dump(recvpacket,rbytes);
			return;
		}
		return;
	}
	printf("[Unknown ethernet frame %04x received %d bytes]\n",__builtin_rx_revw(ep->type),rbytes);
	packet_dump(recvpacket,rbytes);
}

void show_prompt()
{
	sci_puts("RX62N> ");
}

void userapp_ping(unsigned char ipaddr[4])
{
	int i;
	for(i=0;i<10;i++) // ���M�񐔂�4��
	{
		char c;
		send_ping(ipaddr);
		while(g_ping_process)
		{
			if(timer_get_us() - g_ping_timer > 5000000) // �^�C���A�E�g5�b
			{
				sci_puts("Request timed out.\n");
				buzz_ng();
				break;
			}
			while((c = sci_getc()) != '\0')
			{
				if(c == 0x03) return;
			}
		}
		if(!g_ping_process)
		{
			buzz_ok();
			g_success++;
		}
		timer_wait_ms(1000); // ���M�Ԋu��1000ms
	}
}

void ipconfig()
{
	printf("IP Address. . . . . . . . . . . . : %d.%d.%d.%d\n",my_ipaddr[0],my_ipaddr[1],my_ipaddr[2],my_ipaddr[3]);
	printf("Subnet Mask . . . . . . . . . . . : %d.%d.%d.%d\n",my_ipmask[0],my_ipmask[1],my_ipmask[2],my_ipmask[3]);
	printf("Default Gateway . . . . . . . . . : %d.%d.%d.%d\n",gw_ipaddr[0],gw_ipaddr[1],gw_ipaddr[2],gw_ipaddr[3]);
}

BOOL test_ip() {
	gpio_write_port(PIN_LED3,0);

	ether_open(my_macaddr);

	// �C�[�T�t���[������M�����Ƃ��Ɏ��s����銄�荞�݊֐���o�^
	ether_regist_user_rx_procedure(packet_receive_process);

	while(!ether_is_linkup())
	{
		sci_puts("Wait for link up...\n");
		timer_wait_ms(100);
	}
	check_link();

	unsigned char ipaddr[4] = {192,168,2,1};
	g_success = 0;
	userapp_ping(ipaddr);
	
	if(g_success)
	{
		sci_puts("[�C�[�T�l�b�g �e�X�g����]\n");
		return TRUE;
	}
	else
	{
		sci_puts("[�C�[�T�l�b�g �e�X�g����]\n");
		return FALSE;
	}

#if 0
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
#endif
	return TRUE;
}

