// RX62NのTCP/IPサンプルプログラム
// (C)Copyright 2011 特殊電子回路

#ifndef H_TKDNIP
#define H_TKDNIP

#include "iodefine_gcc.h"

// 特電HAL
#include "tkdn_hal.h"

// 各種マクロ関数
#define CONST_REVW(d) ((((d) & 0xff) << 8) | (((d) & 0xff00) >> 8))
#define IP_AS_ULONG(I) ((I[0]) | (I[1] << 8) | (I[2] << 16) | (I[3] << 24))

// マクロ定数
#define BOOL  int
#define TRUE  1
#define FALSE 0

#define PROTO_ICMP 0x01
#define PROTO_TCP  0x06
#define PROTO_UDP  0x11

// グローバル変数
extern unsigned char my_macaddr[6];
extern unsigned char my_ipaddr[4];
extern unsigned char my_ipmask[4];
extern unsigned char gw_ipaddr[4];

// 送受信バッファ
extern unsigned char *sendpacket;
extern unsigned char *recvpacket;

// パケットの型
typedef struct ethernet_frame_str {
	unsigned char  dst_macaddr[6];
	unsigned char  src_macaddr[6];
	unsigned short type;   // 0x08 0x06
	unsigned char  payload[0];
} ethernet_frame_str;

typedef struct ip_headder_str {
	unsigned char version_length;
	unsigned char tos;
	unsigned short dglength;
	unsigned short id;
	unsigned short fragment;
	unsigned char ttl;
	unsigned char protocol;
	unsigned short checksum;
	unsigned long src_addr;
	unsigned long dest_addr;
	unsigned char data[0];
} ip_headder_str;

#endif
