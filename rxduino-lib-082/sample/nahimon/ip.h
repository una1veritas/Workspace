// RX62N��TCP/IP�T���v���v���O����
// (C)Copyright 2011 ����d�q��H

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
