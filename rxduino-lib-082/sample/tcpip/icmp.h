// RX62NのTCP/IPサンプルプログラム
// (C)Copyright 2011 特殊電子回路

#ifndef H_ICMP
#define H_ICMP

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned short g_ping_seq;     // 最後に送信したID番号
extern unsigned short g_ping_process; // 
extern unsigned long  g_ping_timer;   // 最後に送信してから何ms経つか

void icmp_received(unsigned char *etherframe,int etherlen);
void send_ping(unsigned char ipaddr[4]);

#ifdef __cplusplus
}
#endif

#endif
