/*
 * RTC DS1307/3234
 *
 */

#include <ctype.h>

#include "armcore.h"
#include "delay.h"
#include "i2c.h"


		i2c_requestFrom(0b1101000, 0, (uint8_t *) &tmp32, 3);
		if (rtctime != (tmp32 & 0xffffff)) {
			rtctime = tmp32 & 0xffffff;
			sprintf(printbuf, "%02x:%02x:%02x\r\n", UINT8(rtctime>>16),
					UINT8(rtctime>>8), UINT8(rtctime) );
			usart_print(&Serial3, printbuf);
			ST7032i_Set_DDRAM(((0 * 0x40) % 0x6c) + 0);
			ST7032i_Print_String((int8_t *) printbuf);
			if ((rtctime & 0xff) == 0) {
				i2c_requestFrom(0b1101000, 3, (uint8_t *) &tmp32, 4);
				sprintf(printbuf, "20%02x %02x/%02x (%x)", UINT8(tmp32>>24),
						UINT8(tmp32>>16), UINT8(tmp32>>8), UINT8(tmp32) );
				usart_print(&Serial3, printbuf);
				ST7032i_Set_DDRAM(((1 * 0x40) % 0x6c) + 0);
				ST7032i_Print_String((int8_t *) printbuf);
			}
		}
		delay_ms(50);
	}
}

