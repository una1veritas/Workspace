// RX62N��TCP/IP�T���v���v���O����
// (C)Copyright 2011 ����d�q��H

#ifndef H_ICMP
#define H_ICMP

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned short g_ping_seq;     // �Ō�ɑ��M����ID�ԍ�
extern unsigned short g_ping_process; // 
extern unsigned long  g_ping_timer;   // �Ō�ɑ��M���Ă��牽ms�o��

void icmp_received(unsigned char *etherframe,int etherlen);
void send_ping(unsigned char ipaddr[4]);

#ifdef __cplusplus
}
#endif

#endif
