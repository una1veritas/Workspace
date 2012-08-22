// RX62N��GCC�T���v���v���O����
// �C�[�T�l�b�g
// (C)Copyright 2011 ����d�q��H

#include "tkdnip.h"
#include "arp.h"
#include "ip.h"

#include <stdio.h>
#include <string.h>

int g_ipseq;

unsigned short calc_checksum(unsigned short *buf, int size)
{
	unsigned long sum = 0;

	while (size > 1) {
		sum += *buf++;
		size -= 2;
	}
	if (size)
		sum += *(unsigned char *)buf;

	sum  = (sum & 0xffff) + (sum >> 16);	// add overflow counts 
	sum  = (sum & 0xffff) + (sum >> 16);	// once again 
	
	return ~sum;
}

void ip_send_raw(unsigned char *etherbuf,int ip_data_len)
{
	ethernet_frame_str *ef = (ethernet_frame_str *)etherbuf;
	ip_headder_str *ip = (ip_headder_str *)&ef->payload[0];

	unsigned long csum = calc_checksum((unsigned short *)ip,(ip->version_length & 0x0f)*4);

	ip->checksum = csum;

	ether_write(etherbuf,ip_data_len + sizeof(ethernet_frame_str));
//	packet_dump(etherbuf,ip_data_len + sizeof(ethernet_frame_str));
}

void ip_send(unsigned char ipaddr[4],unsigned char proto,unsigned char *etherbuf,int ip_data_len)
{
	ethernet_frame_str *ef = (ethernet_frame_str *)etherbuf;

	// IP�w�b�_�̐ݒ�
	ip_headder_str *ip = (ip_headder_str *)&ef->payload[0];
	ip->version_length = 0x45;
	ip->tos            = 0x00;
	ip->dglength       = __builtin_rx_revw(ip_data_len);
	ip->id             = __builtin_rx_revw(g_ipseq++);
	ip->fragment       = 0;
	ip->ttl            = 0x80;
	ip->protocol       = proto;
	ip->src_addr  = my_ipaddr[0] | (my_ipaddr[1] << 8) | (my_ipaddr[2] << 16) | (my_ipaddr[3] << 24);
	ip->dest_addr     = ipaddr[0] | (ipaddr[1] << 8) | (ipaddr[2] << 16) | (ipaddr[3] << 24);
	unsigned long csum = calc_checksum((unsigned short *)ip,(ip->version_length & 0x0f)*4);
	ip->checksum = csum;

	// �C�[�T�t���[���̐ݒ�
	if((IP_AS_ULONG(ipaddr)    & IP_AS_ULONG(my_ipmask)) != 
	   (IP_AS_ULONG(my_ipaddr) & IP_AS_ULONG(my_ipmask)))
	{   //���̃l�b�g���[�N���ɂ���̂Ńf�t�H���gGW�ɓ�����
		if(!arp_get_mac_by_ip(ef->dst_macaddr,gw_ipaddr))
		{
			sci_puts("Target MAC address cannot be resolved.");
			return; // ���s���ă��^�[��
		}
	}
	else
	{   //�����̃l�b�g���[�N���ɂ���̂�MAC�A�h���X�𒲂ׂ�
		if(!arp_get_mac_by_ip(ef->dst_macaddr,ipaddr))
		{
			sci_puts("Target MAC address cannot be resolved.");
			return; // ���s���ă��^�[��
		}
	}

	memcpy(ef->src_macaddr,my_macaddr,6); // ������MAC�A�h���X���Z�b�g����
	ef->type = __builtin_rx_revw(0x0800); // �C�[�T�t���[��

	// ���M
	ether_write(etherbuf,ip_data_len + sizeof(ethernet_frame_str));
//	packet_dump(etherbuf,ip_data_len + sizeof(ethernet_frame_str));
}

