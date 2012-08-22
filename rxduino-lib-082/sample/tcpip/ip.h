// RX62NのTCP/IPサンプルプログラム
// (C)Copyright 2011 特殊電子回路

#ifndef H_IP
#define H_IP

#ifdef __cplusplus
extern "C" {
#endif

unsigned short calc_checksum(unsigned short *buf, int size);
void ip_send_raw(unsigned char *etherbuf,int ip_data_len);
void ip_send(unsigned char ipaddr[4],unsigned char proto,unsigned char *etherbuf,int ip_data_len);


#ifdef __cplusplus
}
#endif

#endif
