// RX62NのTCP/IPサンプルプログラム
// (C)Copyright 2011 特殊電子回路

#ifndef H_ARP
#define H_ARP

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ARP_TABLE 4

// ARPパケットを受信した場合に、この関数を呼ぶこと
void arp_received(unsigned char *rdata,int rbytes);

// ARPテーブルを表示する
void arp_table_show();

// IPアドレスに対応するMACアドレスを、dest_macにセットする
// 必要に応じてARPパケットを発行する
int  arp_get_mac_by_ip(unsigned char dest_macaddr[6],unsigned char ipaddr[4]);

#ifdef __cplusplus
}
#endif

#endif
