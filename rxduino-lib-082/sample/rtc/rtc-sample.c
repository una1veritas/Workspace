// RX62NのGCCサンプルプログラム
// RTCサンプル
// (C)Copyright 2011 特殊電子回路

// 使い方
// このプログラムのファイル名をmain.cに変更して、makeしてください。

// 特電HAL
#include <tkdn_hal.h>
#include <stdio.h>

int main()
{
	RX62N_RTC_TIME rtctime = {0x2011,0x08,0x02,0x02,0x23,0x45,0x12}; // BCDで指定する
	int count = 0;

	gpio_set_pinmode(PIN_LED3,1); // LED3(BUZZ)を出力

	sci_init(SCI_AUTO,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換

	sci_puts("\nRTC sample program\n");

	if(rtc_set_time(&rtctime) == 0)
	{
		sci_puts("RTC起動失敗\n");
	}

	rtctime.year = 0;
	rtctime.mon = 0;
	rtctime.day = 0;
	rtctime.hour = 0;
	rtctime.min = 0;
	rtctime.second = 0;

	while(1)
	{
		if(rtc_get_time(&rtctime))
		{
			char tmp[128];
			sprintf(tmp,"%02x/%02x/%02x %02x:%02x:%02x\n",
				rtctime.year,rtctime.mon,rtctime.day,
				rtctime.hour,rtctime.min,rtctime.second
			);
			sci_puts(tmp);
		}
		timer_wait_ms(500);
		gpio_write_port(PIN_LED3,count++ & 1); // LED3(BUZZ)を出力
	}
}

