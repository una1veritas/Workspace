#include "tkdnbase.h"

sysinfo_str sysinfo[] =
{
	{"LED         " ,SII_LED,SIS_DOWN},
	{"�u�U�[      " ,SII_BUZZ,SIS_DOWN},
	{"SPI ROM     " ,SII_SPIROM,SIS_DOWN},
	{"�������J�[�h" ,SII_MEMCARD ,SIS_DOWN},
	{"USB�z�X�g   " ,SII_USBHOST ,SIS_DOWN},
	{"USB ̧ݸ��� " ,SII_USBFUNC ,SIS_DOWN},
	{"LAN         " ,SII_ETHER ,SIS_DOWN},
	{"            " ,SII_NULL,SIS_DOWN},
};
			  
const char sysinfo_mes[][8] =
{
	"�Ȃ�  ",
	"init  ",
	"OK    ",
	"�̏�H",
	"????  "
};

void          sysinfo_set(sysinfo_id id,sysinfo_state state)
{
	int i = 0;
	while(sysinfo[i].id != SII_NULL && (i < sizeof(sysinfo)/sizeof(sysinfo_str)))
	{
		if(sysinfo[i].id == id)
		{
			sysinfo[i].state = state;
			return;
		}
		i++;
	}
}

sysinfo_state sysinfo_get(sysinfo_id id)
{
	int i = 0;
	while(sysinfo[i].id != SII_NULL)
	{
		if(sysinfo[i].id == id)
		{
			return sysinfo[i].state;
		}
		i++;
	}
	return SIS_OTHER;
}

char         *sysinfo_strmes(sysinfo_state state)
{
	return (char *)sysinfo_mes[state];
}