// RX62N��TCP/IP�T���v���v���O����
// (C)Copyright 2011 ����d�q��H

#include "tkdnip.h"
#include "ip.h"
#include "icmp.h"

#include <stdio.h>
#include <string.h>

unsigned short g_ping_seq;
unsigned short g_ping_process;
unsigned long  g_ping_timer;

void icmp_received(unsigned char *etherframe,int etherlen)
{
	char tmp[80];

	ethernet_frame_str *ef = (ethernet_frame_str *)etherframe;
	ip_headder_str *ip = (ip_headder_str *)&ef->payload[0];

	// �C�[�T�t���[����MAC�A�h���X�����ւ���
	memcpy(tmp,ef->dst_macaddr,6);
	memcpy(ef->dst_macaddr,ef->src_macaddr,6);
	memcpy(ef->src_macaddr,tmp,6);

	// IP�w�b�_��IP�A�h���X�����ւ���
	unsigned long tmpaddr = ip->dest_addr;
	ip->dest_addr = ip->src_addr;
	ip->src_addr = tmpaddr;
	ip->checksum = 0;
	
	// IP�w�b�_�̏���
	unsigned char *icmpdata = (unsigned char *)ip + (ip->version_length & 0x0f) * 4;

/*
	sprintf(tmp,"\n[ICMP received %d bytes]\n",etherlen);
	sci_puts(tmp);
	packet_dump(etherframe,etherlen);
	sprintf(tmp,"ICMP Type %02x , process %d , ping seq %d \n",icmpdata[0], (icmpdata[5] << 8) | icmpdata[4] , (icmpdata[7] << 8) | icmpdata[6]);
	sci_puts(tmp);
*/

	if(icmpdata[0] == 0x00) // ��M�������̂�Ping�̉�����������
	{
		sprintf(tmp,"Reply from %ld.%ld.%ld.%ld:",ip->dest_addr & 0xff,(ip->dest_addr >> 8) & 0xff,(ip->dest_addr >> 16) & 0xff,(ip->dest_addr >> 24) & 0xff);
		sci_puts(tmp);
		sprintf(tmp,"bytes=%d ",__builtin_rx_revw(ip->dglength));
		sci_puts(tmp);
		sprintf(tmp,"time=%ld[us] ttl=%d\n",timer_get_us() - g_ping_timer,ip->ttl);
		sci_puts(tmp);
		g_ping_process = 0; // Ping�͐i�s���Ă��Ȃ�
		return ;
	}

	if(icmpdata[0] == 0x08) // ��M�������̂�Ping�̃��N�G�X�g��������
	{
		icmpdata[0] = 0x00; // Ping����������
		icmpdata[1] = 0x00;
		icmpdata[2] = 0x00;
		icmpdata[3] = 0x00;
		unsigned short csum = calc_checksum((unsigned short *)icmpdata , etherlen - sizeof(ethernet_frame_str) - (ip->version_length & 0x0f) * 4);
		icmpdata[2] = csum & 0xff;
		icmpdata[3] = csum >> 8;
		ip_send_raw(etherframe,etherlen - sizeof(ethernet_frame_str));
	}

//	sprintf(tmp,"[IP send %d bytes]\n",etherlen);
//	sci_puts(tmp);
}

void send_ping(unsigned char ipaddr[4])
{
	g_ping_seq++;
	g_ping_process = 1;
	g_ping_timer = timer_get_us();

	// Ping�p�P�b�g�̒��g�����
	memset(sendpacket,0,2048); // �������̃N���A
	ethernet_frame_str *ef = (ethernet_frame_str *)sendpacket;
	ip_headder_str *ip = (ip_headder_str *)&ef->payload[0];

	unsigned char *icmpmsg = (unsigned char *)ip + sizeof(ip_headder_str);
	icmpmsg[0] = 0x08; // ���N�G�X�g
	icmpmsg[1] = 0x00;
	icmpmsg[4] = g_ping_process; // �v���Z�X�ԍ�
	icmpmsg[5] = 0x00;
	icmpmsg[6] = g_ping_seq; // �V�[�P���X�ԍ�
	icmpmsg[7] = 0x00;
	int i;
	for(i=0;i<32;i++)
	{
		icmpmsg[8+i] = 0x61 + i % 0x17;
	}
	unsigned short csum = calc_checksum((unsigned short *)icmpmsg,8+32);
	icmpmsg[2] = csum & 0xff;
	icmpmsg[3] = csum >> 8;
	ip_send(ipaddr,PROTO_ICMP,sendpacket,sizeof(ip_headder_str) + 8+32);
}
