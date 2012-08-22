// RX62NのGCCサンプルプログラム
// イーサネット
// (C)Copyright 2011 特殊電子回路

#include "tkdnip.h"
#include "arp.h"

#include <stdio.h>
#include <string.h>

// ARPパケットの型の定義
typedef struct arp_packet_str {
	unsigned short hwtype;        // 0x00 0x01
	unsigned short prtype;        // 0x08 0x00
	unsigned char  hwlen;         // 0x06
	unsigned char  prlen;         // 0x04
	unsigned short operation;     // 0x00 0x01
	unsigned char  src_macaddr[6];
	unsigned char  src_ipaddr[4];
	unsigned char  dst_macaddr[6];
	unsigned char  dst_ipaddr[4];
} arp_packet_str;

// ARPテーブルの構造
typedef struct arp_table_str {
	unsigned char macaddr[6];
	unsigned long ipaddr; // エンディアンに注意 192.168.1.1 = 0x0101a8c0
	unsigned long time_ms;
} arp_table_str;

// グローバル変数
static arp_table_str arp_table[MAX_ARP_TABLE];
static int arp_table_ptr = 0;

static int arp_table_search(unsigned char *ipaddr)
{
	int i;
	unsigned long _ipaddr = *(unsigned long *)&ipaddr[0];
	for(i=0;i<MAX_ARP_TABLE;i++)
	{
		if(arp_table[i].ipaddr == _ipaddr) return i;
	}
	return -1;
}

static void arp_table_regist(unsigned char *macaddr,unsigned char *ipaddr)
{
	if(arp_table_search(ipaddr) < 0) // 未登録
	{
		unsigned long _ipaddr = *(unsigned long *)ipaddr;
		arp_table_ptr = arp_table_ptr % MAX_ARP_TABLE;
		arp_table[arp_table_ptr].ipaddr = _ipaddr;
		memcpy(arp_table[arp_table_ptr].macaddr,macaddr,6);
		arp_table[arp_table_ptr].time_ms = timer_get_ms();
		arp_table_ptr++;
	}
}

static void arp_send(unsigned char *dst_macaddr,unsigned char *dst_ipaddr,unsigned short operation)
{
	unsigned char buf[74];
	ethernet_frame_str *ef = (ethernet_frame_str *)buf;
	memcpy(ef->dst_macaddr,dst_macaddr,6);
	memcpy(ef->src_macaddr,my_macaddr,6);
	ef->type = 0x0608; // ARP 0x08 0x06 

	arp_packet_str *ap = (arp_packet_str *)&ef->payload[0];
	ap->hwtype    = 0x0100;
	ap->prtype    = 0x0008;
	ap->hwlen     = 0x06;
	ap->prlen     = 0x04;
	ap->operation = (operation & 0xff) << 8;

	memcpy(ap->dst_macaddr,dst_macaddr,6);
	memcpy(ap->dst_ipaddr,dst_ipaddr,4);
	memcpy(ap->src_macaddr,my_macaddr,6);
	memcpy(ap->src_ipaddr,my_ipaddr,4);

//	char tmp[80];
	ether_write(buf,74);
//	sprintf(tmp,"[ARP send %d bytes]\n",74);
//	sci_puts(tmp);
//	packet_dump(buf,74);

//	sprintf(tmp,"\n[ARP reply]\n");
//	sci_puts(tmp);
//	sprintf(tmp,"Destination madaddr  %02x-%02x-%02x-%02x-%02x-%02x\n",ap->dst_macaddr[0],ap->dst_macaddr[1],ap->dst_macaddr[2],ap->dst_macaddr[3],ap->dst_macaddr[4],ap->dst_macaddr[5]);
//	sci_puts(tmp);

}

void arp_table_show()
{
	int i;
	char tmp[80];
	for(i=0;i<MAX_ARP_TABLE;i++)
	{
		unsigned char *macaddr = arp_table[i].macaddr;
		unsigned char *ipaddr = (unsigned char *)&arp_table[i].ipaddr;

		sprintf(tmp,"ARP entry (%2d)  ",i);
		sci_puts(tmp);
		sprintf(tmp,"%02x-%02x-%02x-%02x-%02x-%02x ",macaddr[0],macaddr[1],macaddr[2],macaddr[3],macaddr[4],macaddr[5]);
		sci_puts(tmp);
		sprintf(tmp,"%d.%d.%d.%d ",ipaddr[0],ipaddr[1],ipaddr[2],ipaddr[3]);
		sci_puts(tmp);
		sprintf(tmp,",lapsedtime=%d\n",(int)(timer_get_ms() - arp_table[i].time_ms) / 1000);
		sci_puts(tmp);
	}
}

int arp_get_mac_by_ip(unsigned char dest_macaddr[6],unsigned char ipaddr[4])
{
	int arp_retry_count = 0;
	int arpindex;
	unsigned char MAC_FFFFFFFF[] = {0xff,0xff,0xff,0xff,0xff,0xff,};

	unsigned long starttime = 0;

	while(1)
	{
		arpindex = arp_table_search(ipaddr);
		if(arpindex >= 0) break; // 発見した
		if((arp_retry_count == 0) || (timer_get_ms() - starttime >= 200)) // 500ms経過
		{
			if(++arp_retry_count == 10) return 0; // 2秒でタイムアウト
			arp_send(MAC_FFFFFFFF,ipaddr,0x0001);
			starttime = timer_get_ms();
		}
	}
	unsigned char *dstmac = arp_table[arpindex].macaddr;
	memcpy(dest_macaddr,&dstmac[0],6);    // 相手先MACアドレスをセット
	return 1;
}

void arp_received(unsigned char *rdata,int rbytes)
{
	arp_packet_str *ap = (arp_packet_str *)rdata;
/*
	char tmp[80];
	sprintf(tmp,"\n[ARP packet received %d bytes]\n",rbytes);
	sci_puts(tmp);

	sprintf(tmp,"Hardware Type   %02x%02x\n",ap->hwtype & 0xff,ap->hwtype >> 8);
	sci_puts(tmp);

	sprintf(tmp,"Protocol Type   %02x%02x\n",ap->prtype & 0xff,ap->prtype >> 8);
	sci_puts(tmp);

	sprintf(tmp,"Hardware Length %02x\n",ap->hwlen);
	sci_puts(tmp);

	sprintf(tmp,"Protocol Length %02x\n",ap->prlen);
	sci_puts(tmp);

	sprintf(tmp,"Operation       %02x%02x",ap->operation & 0xff,ap->operation >> 8);
	sci_puts(tmp);
	if(ap->operation == 0x0100) sci_puts(" (Request)\n");
	if(ap->operation == 0x0200) sci_puts(" (Reply)\n");

//	sprintf(tmp,"Source madaddr  %02x-%02x-%02x-%02x-%02x-%02x\n",ap->src_macaddr[0],ap->src_macaddr[1],ap->src_macaddr[2],ap->src_macaddr[3],ap->src_macaddr[4],ap->src_macaddr[5]);
//	sci_puts(tmp);

	sprintf(tmp,"Source ipaddr   %d.%d.%d.%d\n",ap->src_ipaddr[0],ap->src_ipaddr[1],ap->src_ipaddr[2],ap->src_ipaddr[3]);
	sci_puts(tmp);

//	sprintf(tmp,"Dest.  macaddr  %02x-%02x-%02x-%02x-%02x-%02x\n",ap->dst_macaddr[0],ap->dst_macaddr[1],ap->dst_macaddr[2],ap->dst_macaddr[3],ap->dst_macaddr[4],ap->dst_macaddr[5]);
//	sci_puts(tmp);

	sprintf(tmp,"Dest.  ipaddr   %d.%d.%d.%d\n",ap->dst_ipaddr[0],ap->dst_ipaddr[1],ap->dst_ipaddr[2],ap->dst_ipaddr[3]);
	sci_puts(tmp);
*/
	
	if(memcmp(ap->dst_ipaddr,my_ipaddr,4)) return; // MACアドレス相違
	
	if(ap->operation == 0x0100) // ARP request
	{
		arp_send(ap->src_macaddr,ap->src_ipaddr,0x0002);
	}

	if(ap->operation == 0x0200) // ARP reply
	{
		arp_table_regist(ap->src_macaddr,ap->src_ipaddr);
	}
}

