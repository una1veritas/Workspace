// RX62N��TCP/IP�T���v���v���O����
// (C)Copyright 2011 ����d�q��H

#ifndef H_ARP
#define H_ARP

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ARP_TABLE 4

// ARP�p�P�b�g����M�����ꍇ�ɁA���̊֐����ĂԂ���
void arp_received(unsigned char *rdata,int rbytes);

// ARP�e�[�u����\������
void arp_table_show();

// IP�A�h���X�ɑΉ�����MAC�A�h���X���Adest_mac�ɃZ�b�g����
// �K�v�ɉ�����ARP�p�P�b�g�𔭍s����
int  arp_get_mac_by_ip(unsigned char dest_macaddr[6],unsigned char ipaddr[4]);

#ifdef __cplusplus
}
#endif

#endif
