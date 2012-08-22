#include "libmisc.h"
#include "string.h"
#include <ctype.h>
#include <stdio.h>
#include <stdarg.h>
#include <tkdn_gpio.h>

static char buf[16];
static const char hexstr[] = "0123456789ABCDEF";

int atoi(char *s)
{
	int a = 0;
	if(!s) return 0;
	while(1)
	{
		if((*s < '0') || (*s > '9')) return a; /* ”ÍˆÍŠO */
		a  = a * 10;
		a += (*s++ - '0');
	}
}

int htoi(char c)
{
	if((c >= '0') && (c <= '9')) return c - '0';
	if((c >= 'A') && (c <= 'F')) return c - 'A' + 10;
	if((c >= 'a') && (c <= 'f')) return c - 'a' + 10;
	return 0;
}

char *int2hex(int val,int digit)
{
	int i;
	static char buf[10];
	if(digit > 8) digit = 8;
	for(i=0;i<digit;i++)
	{
		buf[i] = hexstr[(val >> ((digit - i - 1) * 4)) & 0x0f];
	}
	buf[i] = '\0';
	return buf;
}

char *int2asc(int val)
{
	int i;
	char tmp[11];
	int p = 0;
	if(val < 0)
	{
		buf[p++] = '-';
		val = -val;
	}

	for(i=0; val != 0 && i < 10;i++)
	{
		tmp[i] = (val % 10) + '0';
		val = val / 10;
	}
	i--;
	for(; i >= 0;i--)
	{
		buf[p++] = tmp[i];
	}
	buf[p] = '\0';
	return buf;
}


void trim(char *buffer)
{
	unsigned int w_ptr,r_ptr;
	unsigned int buflen;
	if(!buffer) return;
	if(!buffer[0]) return;

    for(w_ptr = strlen(buffer)-1 ; w_ptr != 0 ;w_ptr--)
    {
		if(iscntrl((unsigned char)buffer[w_ptr]) || isspace((unsigned char)buffer[w_ptr]))
		{
			buffer[w_ptr] = '\0';
		}
		else
		{
			break;
		}
    }

    for(r_ptr=0;r_ptr<strlen(buffer);r_ptr++)
    {
		//æ“ª‚Ì•\Ž¦‰Â”\•¶Žš‚ð’T‚·
		if(isprint((unsigned char)buffer[r_ptr]) && !isspace((unsigned char)buffer[r_ptr]))
			break;
	}

	w_ptr = 0;
	buflen = strlen(buffer);
	while(r_ptr < buflen)
	{
		buffer[w_ptr] = buffer[r_ptr];
		if(buffer[r_ptr] == '\0')
		{
			break;
		}
		w_ptr++;
		r_ptr++;
    }
	buffer[w_ptr] = '\0';
}

char *get_token(char *src,int *p)
{
	int p2,s;
	if(!src) return "";
	if(!p) return "";
	
	p2 = *p;

	// ANK•¶Žš‚ð’T‚·
	while(1)
	{
		char c = src[p2];
		if(!c) return "";
		if((c >= '0') && (c <= '9')) break;
		if((c >= 'A') && (c <= 'Z')) break;
		if((c >= 'a') && (c <= 'z')) break;
		if((c == '_')) break;
		p2++;
		*p = p2;
	}

	s = p2;

	// ”ñANK•¶Žš‚ð’T‚·
	while(1)
	{
		char c = src[p2];
		if(!c) break;
		if((c >= '0') && (c <= '9')) {p2++;continue;}
		if((c >= 'A') && (c <= 'Z')) {p2++;continue;}
		if((c >= 'a') && (c <= 'z')) {p2++;continue;}
		if((c == '_')) {p2++;continue;}

		break;
	}
	if(src[p2]) src[p2++] = '\0';
	*p = p2;
	return &src[s];
}

unsigned long hex_to_ulong(char *src)
{
	unsigned long val = 0;
	if(!src) return 0;
	while(*src)
	{
		val <<= 4;
		if((*src >= '0') && (*src <= '9')) val |= (*src - '0');
		if((*src >= 'A') && (*src <= 'F')) val |= (*src - 'A' + 10);
		if((*src >= 'a') && (*src <= 'f')) val |= (*src - 'a' + 10);
		src++;
	}
	return val;
}

unsigned char htouc(char c)
{
	if((c >= '0') && (c <= '9')) return c - '0';
	if((c >= 'A') && (c <= 'F')) return c - 'A' + 10;
	if((c >= 'a') && (c <= 'f')) return c - 'a' + 10;
	return 0;
}

unsigned long myrand()
{
	return rand() ^ (rand() << 12) ^ (rand() << 20);
}

void buzz_ok() {
	int i;
	for(i=0;i<100;i++)
	{
		gpio_write_port(PIN_BUZZ,1);
		timer_wait_ms(1);
		gpio_write_port(PIN_BUZZ,0);
		timer_wait_ms(1);
	}
}

void buzz_ng() {
	int i;
	for(i=0;i<100;i++)
	{
		gpio_write_port(PIN_BUZZ,1);
		timer_wait_ms(4);
		gpio_write_port(PIN_BUZZ,0);
		timer_wait_ms(4);
	}
}
