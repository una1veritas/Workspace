// RX62NのGCCサンプルプログラム
// ADコンバータ
// (C)Copyright 2011 特殊電子回路

// 使い方
// このプログラムのファイル名をmain.cに変更して、makeしてください。

// 特電HAL
#include <tkdn_hal.h>

#include <stdio.h>

#ifdef __GNUC__
  #ifdef CPU_IS_RX62N
    #include "iodefine_gcc62n.h"
  #endif
  #ifdef CPU_IS_RX63N
    #include "iodefine_gcc63n.h"
  #endif
#endif
#ifdef __RENESAS__
  #include "iodefine.h"
#endif

int main()
{
	int i;

	sci_init(SCI_AUTO,38400);
	sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換
	adc_init();

	sci_puts("ADC test start!!\n");
	gpio_set_pinmode(PIN_LED0,1);
	gpio_set_pinmode(PIN_LED1,1);
#ifdef CPU_IS_RX63N
	S12AD.ADSSTR23.WORD = 0xff14;
#endif

	while(1)
	{
		char tmp[128];
		for(i=0;i<8;i++)
		{
			gpio_write_port(PIN_LED0,1);
			int x = adc_sample(i);
			gpio_write_port(PIN_LED0,0);
			sprintf(tmp,"%04d ",x);
			sci_puts(tmp);
		}

#ifdef CPU_IS_RX63N
		sprintf(tmp,"内部基準電圧=%f[V] ",adc_sample(101) * 3.3 / 4096);
		sci_puts(tmp);

		gpio_write_port(PIN_LED1,1);
		double Vs = adc_sample(100) * 3.3 / 4096; // 測定電圧
		gpio_write_port(PIN_LED1,0);
		double V1 = 1.26; // 25℃での電圧
		double Slope = 0.0041; // V /℃
		double T = (Vs - V1) / Slope + 25;
		sprintf(tmp,"Vs=%f[V] ",Vs);
		sci_puts(tmp);
		sprintf(tmp,"temp=%f[℃] ",T);
		sci_puts(tmp);
#endif

		sci_puts("\n");

/*
		char tmp[128];
		sprintf(tmp,"%x %x %x  %x %x %x  %x %x\n",PORT4.PMR.BYTE,MPC.P40PFS.BYTE,AD.ADCR.BYTE,
		                   AD.ADCR.BYTE,AD.ADCR2.BYTE,AD.ADCSR.BYTE,
		                   AD.ADSSTR,AD.ADDIAGR.BYTE);
*/

//		char tmp[128];
//		sprintf(tmp,"ch0=%4d ch1=%4d ch2=%4d ch3=%4d ch4=%4d ch5=%4d ch6=%4d ch7=%4d \n",S12AD.ADDR0,S12AD.ADDR1,S12AD.ADDR2,S12AD.ADDR3,S12AD.ADDR4,S12AD.ADDR5,S12AD.ADDR6,S12AD.ADDR7);
//		sci_puts(tmp);

//		timer_wait_ms(0);
	}
}
