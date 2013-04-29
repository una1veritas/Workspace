/*
 * main.h
 *
 *  Created on: 2013/04/28
 *      Author: sin
 */

#ifndef MAIN_H_
#define MAIN_H_


void init_Ethernet(void);
void PN532_init(void);
byte readMifare(ISO14443 & card);
byte readFCF(ISO14443 & ccard);
void send_header(EthernetClient & client);

#endif /* MAIN_H_ */
