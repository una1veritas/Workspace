#include <stdio.h>
#include <string.h>

#include "des.h"
/*
 const char syskey[]  = "5C7851CB6AB54E5E";
 const char	crd_keya[12] = "4B615A754969";
 const char	crd_keyb[12] = "63457455794B";
 const char	ini_keya[12] = "A0A1A2A3A4A5";
 const char	ini_keyb[12] = "B0B1B2B3B4B5";
 const char	ini_keyc[12] = "FFFFFFFFFFFF";
 */

//const byte Ks[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};

int main (int argc, const char * argv[]) {
    // insert code here...
	char buf[32];
    printf("Hello, World!\n");
	
	unsigned char key[8] = { 0x5C, 0x78, 0x51, 0xCB, 0x6A, 0xB5, 0x4E, 0x5E};
//	unsigned char key[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
//	unsigned char key[8] = {0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01};
	unsigned char plaintext[8];
	unsigned char ciphertext[8];
	unsigned char recoverd[8];
	gl_des_ctx context;

	int i, j, k;

	// Fill 'key' and 'plaintext' with some data
	strncpy((char *) plaintext, "Hello, World!\n", 8);
	for (i = 0; i < 8; i++) {
		printf("%02x ", plaintext[i]);
	}
	printf("\n");
	printf("\n");

	// Set up the DES encryption context
	gl_des_setkey(&context, (char *) key);
	for (i = 0; i < 8; i++) {
		printf("%02x", key[i]);
	}
	printf("\n");
	printf("\n");
	for( i = 0, k = 0; i < 32; i++) {
		printf("%x ", context.encrypt_subkeys[i]);
		printf("\n");
	}
	printf("\n");
	
	// Encrypt the plaintext
	gl_des_ecb_encrypt(&context, (char*) plaintext, (char*) ciphertext);
	
	for (i = 0; i < 8; i++) {
		printf("%02x ", ciphertext[i]);
	}
	printf("\n");
	for (i = 0; i < 8; i++) {
		printf("%c ", ciphertext[i]);
	}
	printf("\n");
	
	// To recover the orginal plaintext from ciphertext use:
	gl_des_ecb_decrypt(&context, (char *) ciphertext, (char*) recoverd);
	
	for (i = 0; i < 8; i++) {
		printf("%02X ", recoverd[i]);
	}
	printf("\n");
	strncpy(buf, recoverd,8);
	buf[8] = '\0';
	printf("%s\n", buf);
    return 0;
}
