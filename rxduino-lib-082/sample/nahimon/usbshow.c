#include <stdio.h>
#include <string.h>

#include "tkusbhost.h"

void dump(unsigned char *buf,int len)
{
	int i;
	if(len < 0) return;
	for(i=0;i<len;i++)
	{
		if((i & 15) == 0)
		{
			printf("%04X ",i);
		}
		printf("%02X ",buf[i]);
		if(((i & 15) == 15) || (i == len-1))
		{
			printf("\n");
		}
	}
}

void ShowDeviceDesc(DeviceDesc_t *desc,unsigned short LangId)
{
	char buf[64];
	printf("\nDevice Desriptor:\n");
	printf("  bLength            = %d\n",desc->bLength);
	printf("  bDescriptorType    = %d\n",desc->bDescriptorType);
	printf("  bcdUSB             = 0x%x\n",desc->bcdUSB);
	printf("  bDeviceClass       = 0x%x\n",desc->bDeviceClass);
	printf("  bDeviceSubClass    = 0x%d\n",desc->bDeviceSubClass);
	printf("  bDeviceProtocol    = 0x%d\n",desc->bDeviceProtocol);
	printf("  bMaxPacketSize     = %d\n",desc->bMaxPacketSize0);
	printf("  idVendor           = 0x%04x\n",desc->idVendor);
	printf("  idProduct          = 0x%04x\n",desc->idProduct);
	printf("  bcdDevice          = 0x%04x\n",desc->bcdDevice);

	if(!desc->iManufacture || (tkusbh_get_string(desc->iManufacture, LangId, buf, 64) <= 0))
	{
		*buf = '\0';
	}
	printf("  iManufacture       = %x (%s)\n",desc->iManufacture,buf);

	if(!desc->iProduct || (tkusbh_get_string(desc->iProduct, LangId, buf, 64) <= 0))
	{
		*buf = '\0';
	}
	printf("  iProduct           = %x (%s)\n",desc->iProduct,buf);

	if(!desc->iSerialNumber || (tkusbh_get_string(desc->iSerialNumber, LangId, buf, 64) <= 0))
	{
		*buf = '\0';
	}
	printf("  iSerialNumber      = %x (%s)\n",desc->iSerialNumber,buf);

	printf("  bNumConfigurations = %x\n",desc->bNumConfigurations);
}

void ShowConfigDesc(ConfigDesc_t *desc)
{
	printf("\nConfiguration Desriptor:\n");
	printf("  bLength             = %d\n",desc->bLength);
	printf("  bDescriptorType     = %d\n",desc->bDescriptorType);
	printf("  wTotalLength        = %d\n",desc->wTotalLength);
	printf("  bNumInterfaces      = %d\n",desc->bNumInterfaces);
	printf("  bConfigurationValue = %d\n",desc->bConfigurationValue);
	printf("  iConfiguraion       = %d\n",desc->iConfiguraion);
	printf("  bmAttributes        = 0x%x ",desc->bmAttributes);
	if(desc->bmAttributes & 0x40) printf("(Self powered)\n");
	else                          printf("(Bus powered)\n");
	printf("  bMaxPower           = %d (%dmA)\n",desc->bMaxPower,desc->bMaxPower*2);
}

void ShowInterfaceDesc(InterfaceDesc_t *desc)
{
	printf("  + Interface Desriptor:\n");
	printf("      bLength             = %d\n",desc->bLength);
	printf("      bDescriptorType     = %d\n",desc->bDescriptorType);
	printf("      bInterfaceNumber    = %d\n",desc->bInterfaceNumber);
	printf("      bAlternateSetting   = %d\n",desc->bAlternateSetting);
	printf("      bNumEndpoints       = %d\n",desc->bNumEndpoints);
	printf("      bInterfaceClass     = 0x%x\n",desc->bInterfaceClass);
	printf("      bInterfaceSubClass  = 0x%x\n",desc->bInterfaceSubClass);
	printf("      bInterfaceProtocol  = 0x%x\n",desc->bInterfaceProtocol);
	printf("      iInterface          = %d\n",desc->iInterface);
}

void ShowEndpointDesc(EndpointDesc_t *desc)
{
	printf("      + Endpoint Desriptor: (EP%d ",desc->bEndpointAddress & 0x0f);
	if(desc->bEndpointAddress & 0x80) printf("IN)\n");
	else                              printf("OUT)\n");
	printf("          bLength             = %d\n",desc->bLength);
	printf("          bDescriptorType     = %d\n",desc->bDescriptorType);
	printf("          bEndpointAddress    = %d\n",desc->bEndpointAddress);
	printf("          bmAttributes        = 0x%x ",desc->bmAttributes);
	if(desc->bmAttributes == 0) printf("CONTROL\n");
	if(desc->bmAttributes == 1) printf("ISO\n");
	if(desc->bmAttributes == 2) printf("BULK\n");
	if(desc->bmAttributes == 3) printf("INTERRUPT\n");
	
	printf("          wMaxPacketSize      = %d\n",desc->wMaxPacketSize);
	printf("          bInterval           = %d\n",desc->bInterval);
	printf("          bRefresh            = %d\n",desc->bRefresh);
	printf("          bSynchAddress       = %d\n",desc->bSynchAddress);
}
