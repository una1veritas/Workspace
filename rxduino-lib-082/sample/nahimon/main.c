// RX62NのGCCサンプルプログラム
// RTCサンプル
// (C)Copyright 2011 特殊電子回路

// 使い方
// このプログラムのファイル名をmain.cに変更して、makeしてください。

// 特電HAL
#include <tkdn_hal.h>
#include <tkdn_spi.h>
#include <tkusbhost.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "command.h"
#include "libmisc.h"
#include "tkdnbase.h"
#include "fatfs-test.h"

char hist_cmd[16][PREVCMD_BUFSIZE+1];
volatile int  hist_idx_w = 0;
volatile int  hist_idx_r = 0;

void print_prompt(void);
void usbhost_show(void);

BOOL test_sd();
BOOL test_ip();
BOOL test_sdram();

BOOL process_scirxtx();

int main()
{
	int i;

	gpio_set_pinmode(PIN_LED0,1);
	gpio_set_pinmode(PIN_LED1,1);
	gpio_set_pinmode(PIN_LED2,1);
	gpio_set_pinmode(PIN_LED3,1);
	gpio_set_pinmode(PIN_BUZZ,1);
	gpio_set_pinmode(PIN_SW,0);

	// ブザーとLEDのテスト
	gpio_set_pinmode(PIN_SPI_CS0,1);
	gpio_write_port(PIN_SPI_CS0,1);
	gpio_set_pinmode(PIN_SPI_CS1,1);
	gpio_write_port(PIN_SPI_CS1,1);
	gpio_set_pinmode(PIN_SPI_CS2,1);
	gpio_write_port(PIN_SPI_CS2,1);
	gpio_set_pinmode(PIN_SPI_CS3,1);
	gpio_write_port(PIN_SPI_CS3,1);

	for(i=0;i<4;i++)
	{
		gpio_write_port(PIN_LED0,(i % 4 == 0) ? 1 : 0);
		gpio_write_port(PIN_LED1,(i % 4 == 1) ? 1 : 0);
		gpio_write_port(PIN_LED2,(i % 4 == 2) ? 1 : 0);
		gpio_write_port(PIN_LED3,(i % 4 == 3) ? 1 : 0);
		timer_wait_ms(50);
	}

	for(i=0;i<100;i++)
	{
		gpio_write_port(PIN_LED3,i & 1);
		timer_wait_us(500);
	}
	for(i=0;i<50;i++)
	{
		gpio_write_port(PIN_LED3,i & 1);
		timer_wait_us(1000);
	}

//	sci_init(SCI_SCI0P2x,38400);
	sci_init(SCI_AUTO,38400);
//	sci_init(SCI_SCI1JTAG,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換
	setbuf(stdout, NULL);

	sci_puts("\nSystem boot\n");
	printf("\n\n\n\n\n");
	printf("====================================================\n");
	printf("RX62N monitor version %d.%d.%d ",0,9,6);
	sci_puts("for ");
#ifdef BOARD_NAME
	sci_puts(BOARD_NAME);
	sci_puts("\n");
#else
	#if (TARGET_BOARD == BOARD_RAXINO)
		sci_puts("RaXino\n");
	#elif (TARGET_BOARD == BOARD_ULT62N0_SDRAM) || (TARGET_BOARD == BOARD_ULT62N0_MMC) || (TARGET_BOARD == BOARD_ULT62N)
		sci_puts("ULT62N\n");
	#elif (TARGET_BOARD == BOARD_RXMEGA)
		sci_puts("RXMEGA\n");
	#else
		sci_puts("???\n");
	#endif
#endif	
	printf("(C)Copyright H23-H24 Tokushu Denshi Kairo Inc.\n");
	printf("\n");
	printf("Compiled at %s %s \n",__DATE__,__TIME__);
	printf("====================================================\n");

	sci_puts("Initialize fatfs..");
	fatfs_init();
	sci_puts("done.\n");

	sci_puts("Initialize SPI ROM..");
	spi_init();
	sci_puts("done.\n");

#if ((TARGET_BOARD == BOARD_ULT62N0_SDRAM) || (TARGET_BOARD == BOARD_ULT62N0_MMC) || (TARGET_BOARD == BOARD_ULT62N))
	if(tkusbh_init() == TKUSBH_OK)
	{
		sci_puts("USB Host Initialization success.\n");
	}
	else
	{
		sci_puts("USB Host Initialization ERROR.\n");
	}
#endif

#if (TARGET_BOARD == BOARD_RXMEGA)
	if(sci_getport() != SCI_USB0)
	{
		if(tkusbh_init() == TKUSBH_OK)
		{
			sci_puts("USB Host Initialization success.\n");
		}
		else
		{
			sci_puts("USB Host Initialization ERROR.\n");
		}
	}
	else
	{
		sci_puts("USB0 is used already as USB function.\n");
	}
#endif

#if (TARGET_BOARD == BOARD_RAXINO)
	sci_puts("USB Host can not be used.\n");
#endif

	print_prompt();
	while(1) // メインループ
	{
		if(USBHostInfo.FlagAttach)
		{
			usbhost_show();
			print_prompt();
		}
		if(process_scirxtx()) //SCI送受信処理の起動
		{
			print_prompt();
		}
	}

}

void usbhost_show()
{
	char buf[128];
	if(tkusbh_connect(5000) != TKUSBH_OK)
	{
		printf("*Connection TIMEOUT. Retry...\n");
		return;
	}

	printf("\n\n**USB host decected a target connected !\n");
	ShowDeviceDesc(&USBHostInfo.DeviceDesc,USBHostInfo.LangId);
	ShowConfigDesc(&USBHostInfo.ConfigDesc);
	tkusbh_get_descriptor(CONFIG_DESCRIPTOR_TYPE, 0, buf, 128);

	int len = USBHostInfo.ConfigDesc.wTotalLength;
	int i=0;
	while(i < len)
	{
		switch(buf[i+1])
		{
			case CONFIG_DESCRIPTOR_TYPE:
				break;
			case INTERFACE_DESCRIPTOR_TYPE:
				ShowInterfaceDesc((InterfaceDesc_t *)&buf[i]);
				break;
			case ENDPOINT_DESCRIPTOR_TYPE:
				ShowEndpointDesc((EndpointDesc_t *)&buf[i]);
				break;
			default:
				printf("Unknown descriptor.\n");
				dump((unsigned char *)&buf[i],buf[i]);
		}
		i += buf[i+0];
		continue;
	}
}

BOOL process_scirxtx()
{
	char line[128];

	int p = 0;
	if(sci_rxcount() == 0) return FALSE;

	while(1) {
		char c = sci_getc();
		if(c) {
			if(c == '\n') {
				line[p] = '\0';
				break;
			}
			if(c == 0x03) {
				line[p] = '\0';
				break;
			}
			if(p && (c == 0x08)) {
				sci_putc(0x08);
				sci_putc(0x20);
				sci_putc(0x08);
				p--;
			}
			else {
				sci_putc(c);
				line[p++] = c;
				if(p == 127) p--;
			}
		}
	}
	
	// 受信データ取り出し
	if(!p || (line[p] == 0x03)) { // データがないかCTRL+C
		return TRUE;
	}
	trim(line);

	// 文字列がヌルでなければhist_cmdバッファにコピー
/*
	if(line[0] != '\0')
	{
		strncpy(hist_cmd[hist_idx_w],line,PREVCMD_BUFSIZE);
		hist_idx_w = ++hist_idx_w & 15;
		hist_idx_r = (hist_idx_w - 1) & 15;;
	}
*/

	sci_puts("\n");

	char *cmd = strtok(line," ");
	if(!strcmp(cmd,"help")) {
		cmd_help();
		return TRUE;
	}

	if(!strcmp(cmd,"dump")) {
		cmd_dump();
		return TRUE;
	}

	if(!strcmp(cmd,"memwr")) {
		cmd_memwr();
		return TRUE;
	}

	if(!strcmp(cmd,"memfill")) {
		cmd_memfill();
		return TRUE;
	}

	if(!strcmp(cmd,"spi")) {
		cmd_spi();
		return TRUE;
	}

	if(!strcmp(cmd,"load")) {
		cmd_load();
		return TRUE;
	}

	if(!strcmp(cmd,"dir")) {
		cmd_dir();
		return TRUE;
	}

	if(!strcmp(cmd,"cd")) {
		cmd_cd();
		return TRUE;
	}

	if(!strcmp(cmd,"type")) {
		cmd_type();
		return TRUE;
	}

	if(!strcmp(cmd,"jmp")) {
		cmd_jmp();
		return TRUE;
	}

	if(!strcmp(cmd,"run")) {
		cmd_run();
		return TRUE;
	}

	if(!strcmp(cmd,"reboot")) {
		cmd_reboot();
		return TRUE;
	}

	if(!strcmp(cmd,"time")) {
		cmd_time();
		return TRUE;
	}

	printf("ERR Unknown command '%s'\n",line);
	return TRUE;
}

void print_prompt(void)
{
	SCI_PORT port = sci_getport();
	sci_puts("\n");

	if(port == SCI_USB0)     sci_puts("USB0 ");
	if(port == SCI_SCI0P2x)  sci_puts("SCI0P ");
	if(port == SCI_SCI1JTAG) sci_puts("SCI1J ");
	if(port == SCI_SCI2A)    sci_puts("SCI2A ");
	if(port == SCI_SCI2B)    sci_puts("SCI2B ");
	if(port == SCI_SCI6A)    sci_puts("SCI6A ");
	if(port == SCI_SCI6B)    sci_puts("SCI6B ");

	RX62N_RTC_TIME rtctime;
	if(rtc_get_time(&rtctime)) {
		printf("[%4x/%02x/%02x %02x:%02x:%02x] ",
			rtctime.year,rtctime.mon,rtctime.day,
			rtctime.hour,rtctime.min,rtctime.second
		);
	}

	char cwd[256];
	fatfs_getcwd(cwd,255);
	sci_puts(cwd);

	sci_puts(" $ ");
}
