#include <stdio.h>
#include <tkdn_hal.h>
#include <tkdn_sci.h>
#include <tkdn_timer.h>
#include <stdlib.h>

#include "tkdnbase.h"
#include "libmisc.h"
#include "string.h"
#include "command.h"
#include "fatfs-test.h"

#if (TARGET_BOARD == BOARD_RAXINO) || (TARGET_BOARD == BOARD_NP1055)
  #define SPI_ROM_PORT PIN_SPI_CS2
#endif

#if (TARGET_BOARD == BOARD_ULT62N) || (TARGET_BOARD == BOARD_ULT62N0_SDRAM) || (TARGET_BOARD == BOARD_ULT62N0_MMC)
  #define SPI_ROM_PORT PIN_SPI_CS0
#endif

#if (TARGET_BOARD == BOARD_RXMEGA)
  #define SPI_ROM_PORT PIN_SPI_CS3
#endif

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

void print_prompt(void);

char sci_getc_wait() {
	char c;
	while(1) {
		c = sci_getc();
		if(c >= 0x20) break;
	}
	return c;
}

int load_srec_line(unsigned long *entryaddr) {
	int i;
	int alen;
	char type;
	unsigned char len;
	unsigned char csum;
	unsigned char data;
	unsigned long addr;
	char c;
	char str[33];
	int end = 0;

	// S���R�[�h�̗�
	// S00E00006E6168696D6F6E206D6F7497
	// S315FFFF8000FD7302982D0000FD730A982E0000FB52A8

	c = sci_getc_wait(); // 'S' �ł���ׂ�
	if(c != 'S') return -1; // 1�����ڂ�S�łȂ�

	c = sci_getc_wait(); // 2�����ڂ̓^�C�v
	if(!c) return -1;
	type = c;

	c = sci_getc_wait(); // 3�����ڂ�4�����ڂ͒���
	if(!c) return -1;
	len  = htouc(c) << 4;
	c = sci_getc_wait();
	if(!c) return -1;
	len |= htouc(c);

	switch(type)
	{
		case '0': // S0 �X�^�[�g���R�[�h
			alen = 2;
			break;
		case '1': // S1 
			alen = 2;
			break;
		case '2': // S2 
			alen = 3;
			break;
		case '3': // S3 
			alen = 4;
			break;
		case '7': // �A�h���X
			alen = 4;
			end = 1;
			break;
		case '8': // �A�h���X
			alen = 3;
			end = 1;
			break;
		case '9': // �A�h���X
			alen = 2;
			end = 1;
			break;

		default:
			return -1; // �\�����ʃ^�C�v
	}

	// �A�h���X���擾
	addr = 0;
	csum = len;
	for(i=0;i<alen;i++)
	{
		c = sci_getc_wait();
		if(!c) return -1;
		data = htouc(c);

		c = sci_getc_wait();
		if(!c) return -1;
		data = (data << 4) | htouc(c);

		addr = (addr << 8) | data;
		csum += data;
		len--;
	}

	if(end) {
		*entryaddr = addr;
	}

	// �f�[�^���擾
	for(i=0;(i<len-1);i++)
	{
		c = sci_getc_wait();
		if(!c) return -1;
		data = htouc(c);

		c = sci_getc_wait();
		if(!c) return -1;
		data = (data << 4) | htouc(c);

		csum += data;
		if(   ((addr >= 0x00010000) && (addr < 0x08000000))
		   || ((addr >= 0x09000000) && (addr < 0xfff80000))) {
			printf("\n�A�h���X�͈̔͂��I�[�o�[�t���[���Ă��܂� %lx\n",addr);
			return -1;
		}

		if(type == '0')
		{
			str[addr & 0x3f] = data;
		}
		else
		{
			*(unsigned char *)addr = data;
		}
		addr++;
	}
	str[addr & 0x3f] = '\0';

	if(type == '0')
	{
		printf("\nDownload start \"%s\".\n",str);
	}

	// �`�F�b�N�T���̎�M
	c = sci_getc_wait();
	if(!c) return -1;
	data = htouc(c);

	c = sci_getc_wait();
	if(!c) return -1;
	data = (data << 4) | htouc(c);
	csum += data;

	if(csum == 0xff)
	{
		sci_puts("o");
		if(end) return 1;
		return 0;
	}
	sci_puts("x");
	return -1;
}

void cmd_help() {
	printf("General commands\n");
	printf("     dump      Show memory dump\n");
	printf("     memwr     Modify memory\n");
	printf("     memfill   Fill memory.\n");
	printf("     spi       SPI memory diag.\n");
	printf("     time      Show and change current time.\n");
	printf("     help      Show this message\n");
	printf("     jmp       Jump to specified address.\n");
	printf("SD/MMC card commands\n");
	printf("     dir       Show directory.\n");
	printf("     cd        Change directory.\n");
	printf("     type      Type file contents.\n");
	printf("Program load and run commands:\n");
	printf("     load      Load SREC file.\n");
	printf("     run       Run loaded program without reboot.\n");
	printf("     reboot    Reboot and run loaded program.\n");
	printf("\n");
}

void cmd_dump() {
	unsigned long addr,len;
	unsigned char count,uc;
	char tmp[48];

	char *arg = strtok(NULL," ");
	if(!arg) {
		printf("Usage: dump addr len\n");
		return;
	}
	addr = hex_to_ulong(arg);

	len = 0;
	arg = strtok(NULL," ");
	if(arg) len  = hex_to_ulong(arg);
	
	if(len == 0) len = 0x100;

	count = 0;
	while(len) {
		if(sci_rxcount()) break;

        if(count == 0) {
			printf("%08x ",(unsigned int)addr);
        }

        uc = *(unsigned char *)addr;
        if((uc >= ' ') && (uc <= 0x7e)) tmp[count] = uc;
        else                            tmp[count] = '.';
        printf("%02x ",uc);

        addr++; // �A�h���X���₷
        len--; // ���������炷

        count++;

        if(!len) // �c�肪�Ȃ��Ȃ��Ă��܂�����
        {
            while(count != 16) {
                printf("   ");
                tmp[count++] = ' ';
            }
        }
        if(count == 16) {
            tmp[count] = '\0';
            printf("%s\n",tmp);
        }
        count &= 15;
	}
}

void cmd_memwr() {
	unsigned long addr,val;
	char *tmp;
	unsigned char size = 0;
	const char usage[] = "Usage: memwr { b|w|l } addr val\n";

	tmp = strtok(NULL," ");
	if(!tmp) {
		printf(usage);
		return;
	}
	if(!strcmp(tmp,"b")) size = 1;
	if(!strcmp(tmp,"w")) size = 2;
	if(!strcmp(tmp,"l")) size = 4;

	if(size == 0) {
		printf(usage);
		return;
	}

	tmp = strtok(NULL," ");
	if(!tmp)
	{
		printf(usage);
		return;
	}
	addr = hex_to_ulong(tmp);

	tmp = strtok(NULL," ");
	if(!strlen(tmp))
	{
		printf(usage);
		return;
	}
	val = hex_to_ulong(tmp);

	if(size == 1) *(unsigned char *)addr = val;
	if(size == 2) *(unsigned short*)addr = val;
	if(size == 4) *(unsigned long *)addr = val;

	printf("�A�h���X%08x�ɒl%02x�������܂���\n",(unsigned int)addr,(unsigned int)val);
}

void cmd_memfill() {
	unsigned long addr,len,val;
	char *tmp;
	const char usage[] = "Usage: memfill addr len val\n";
	
	tmp = strtok(NULL," ");
	if(!tmp) {
		printf(usage);
		return;
	}
	addr = hex_to_ulong(tmp);

	tmp = strtok(NULL," ");
	if(!tmp) {
		printf(usage);
		return;
	}
	len = hex_to_ulong(tmp);

	tmp = strtok(NULL," ");
	if(!tmp) {
		printf(usage);
		return;
	}
	val = hex_to_ulong(tmp) & 0xff;

	memset((unsigned long *)addr,val,len);
	printf("�A�h���X%08x����%08x�܂ł�l%02x�Ŗ��߂܂���\n",(unsigned int)addr,(unsigned int)(addr+len-1),(unsigned int)val);
}

void cmd_jmp() {
	unsigned long addr;
	char *tmp;

	typedef void (*FUNC)(void);
	FUNC func;
	
	tmp = strtok(NULL," ");
	if(!tmp) {
		printf("Usage: jmp addr\n");
		return;
	}
	addr = hex_to_ulong(tmp);

	printf("�A�h���X%08x�փW�����v���܂�\n",(unsigned int)addr);

	func = (FUNC)addr;
	func();

	printf("���[�U�v���O�����I��\n");
}

void cmd_reboot() {
	if(sci_getport() == SCI_USB0) {
		sci_puts("5�b��Ƀ��Z�b�g���܂��B����܂ł̊Ԃ�USB���zCOM�|�[�g����Ă��������B ");
		buzz_ok();
		timer_wait_ms(1000);
		sci_puts("4 ");
		buzz_ok();
		timer_wait_ms(1000);
		sci_puts("3 ");
		buzz_ok();
		timer_wait_ms(1000);
		sci_puts("2 ");
		buzz_ok();
		timer_wait_ms(1000);
		sci_puts("1 ");
		buzz_ok();
		timer_wait_ms(1000);
		sci_puts("0 ");
	}
	buzz_ok();
#ifdef CPU_IS_RX62N
	WDT.WRITE.WINB = 0x5a5f; // RSTCSR�������݁B�I�[�o�[�t���[�Ń��Z�b�g��������
	WDT.WRITE.WINA = 0xa5fc; // TCSR�������݁BWDT�J�n�B5ms��ɃI�[�o�[�t���[
	sci_puts("system reboot!\n");
#endif
#ifdef CPU_IS_RX63N
	sci_puts("RX63N cannot reboot yet...\n");
#endif
}

void cmd_dir() {
	char *tmp = strtok(NULL," ");
	if(!tmp) fatfs_dir("");
	else fatfs_dir(tmp);
}

void cmd_cd() {
	char *tmp = strtok(NULL," ");
	if(!tmp) fatfs_cd("");
	else fatfs_cd(tmp);
}

void cmd_type() {
	char *tmp = strtok(NULL," ");
	if(!tmp) fatfs_type("");
	else fatfs_type(tmp);
}

void cmd_load() {
	int stat;
	int line = 0;
	unsigned long startaddr;
	sci_puts("SREC�t�@�C���𑗐M���Ă�������\n");
	while(1) {
		stat = load_srec_line(&startaddr);
		line++;
		if(stat == -1) {
			printf("\nSREC�̓ǂݍ��݂ŃG���[���������܂��� %d�s��\n",line);
			break;
		}
		if(stat == 1) {
			sci_puts("\n�_�E�����[�h����\n");
			*(unsigned long *)(0x00000004) = 0x4e444b54; // �L�[���[�h "TKDN"
			*(unsigned long *)(0x00000008) = startaddr;
			break;
		}
	}
}

void cmd_spirom_id() {
	unsigned long recv1;
	unsigned short recv2;

//	spi_init();
	spi_set_port(SPI_PORT_NONE); // CS�������Ő��䂷��
	spi_set_bit_order(SPI_MSBFIRST);

	gpio_set_pinmode(SPI_ROM_PORT,1);
	gpio_write_port(SPI_ROM_PORT,0); // CS��������
	spi_set_bit_length(8);
	spi_transfer(0x9f); // �R�}���h���M
	spi_set_bit_length(24);
	recv1 = spi_transfer(0) & 0xffffff;
	printf("SPI ROM JEDEC ID=%08lx, ",recv1);
//	gpio_write_port(SPI_ROM_PORT,1); // CS���グ��

	gpio_write_port(SPI_ROM_PORT,0); // CS��������
	spi_set_bit_length(32);
	recv2 = spi_transfer(0x90000000) & 0xffff; // �R�}���h���M
	printf("ID=%04x\n",recv2);
	gpio_write_port(SPI_ROM_PORT,1); // CS���グ��

	gpio_set_pinmode(SPI_ROM_PORT,0);
//	spi_terminate();
}

void spirom_read(unsigned long addr,int len,unsigned char *rxdata)
{
	gpio_write_port(SPI_ROM_PORT,0); // CS��������
	spi_set_bit_length(32);
	spi_transfer(0x03000000 | addr); // �R�}���h���M

	spi_set_bit_length(8);
	while(len--) *rxdata++ = spi_transfer(0); // �f�[�^��M

	gpio_write_port(SPI_ROM_PORT,1); // CS���グ��
}

void cmd_spirom_dump() {
	unsigned long addr,len;
	unsigned char count,uc;
	char tmp[48];
	unsigned char rdbuf[16];

	char *arg = strtok(NULL," ");
	if(!arg) {
		printf("Usage: spi dump addr len\n");
		return;
	}
	addr = hex_to_ulong(arg);

	len = 0;
	arg = strtok(NULL," ");
	if(arg) len  = hex_to_ulong(arg);
	
	int p=0;
	if(len == 0) len = 0x100;

	count = 0;
	while(len) {
		if(sci_rxcount()) break;

		spirom_read(addr,16,rdbuf);

        if(count == 0) {
			printf("%08x ",(unsigned int)addr);
        }

        uc = rdbuf[p++];
		p = p & 15;
        if((uc >= ' ') && (uc <= 0x7e)) tmp[count] = uc;
        else                            tmp[count] = '.';
        printf("%02x ",uc);

        addr++; // �A�h���X���₷
        len--; // ���������炷

        count++;

        if(!len) // �c�肪�Ȃ��Ȃ��Ă��܂�����
        {
            while(count != 16) {
                printf("   ");
                tmp[count++] = ' ';
            }
        }
        if(count == 16) {
            tmp[count] = '\0';
            printf("%s\n",tmp);
        }
        count &= 15;
	}
}

void cmd_spi() {
	const char usage[] = "Usage: spi id | dump addr";

	spi_set_port(SPI_PORT_NONE); // CS�������Ő��䂷��
	spi_set_bit_order(SPI_MSBFIRST);
	char *tmp = strtok(NULL," ");
	if(!tmp) {
		sci_puts(usage);
	}
	if(!strcmp(tmp,"id")) {
		cmd_spirom_id();
	}
	if(!strcmp(tmp,"dump")) {
		cmd_spirom_dump();
	}
	spi_set_bit_length(8);
}

void cmd_time() {
	// RTC�̃e�X�g
	RX62N_RTC_TIME rtctime = {0x2012,0x01,0x18,0x05,0x05,0x34,0x00}; // BCD�Ŏw�肷��

	char *tmp = strtok(NULL,"/");
	if(!tmp) {
		if(rtc_get_time(&rtctime)) {
			printf("%04x/%02x/%02x %02x:%02x:%02x\n",
				rtctime.year,rtctime.mon,rtctime.day,
				rtctime.hour,rtctime.min,rtctime.second
			);
		}
		else {
			sci_puts("RTC���ݒ肳��Ă��Ȃ��̂œK���Ȏ�����ݒ肵�܂��B");
			if(rtc_set_time(&rtctime) == 0) {
				sci_puts("RTC�N�����s\n");
			}
		}
		return;
	}

	rtctime.year = hex_to_ulong(tmp);
	tmp = strtok(NULL,"/");
	if(tmp) rtctime.mon = hex_to_ulong(tmp);
	tmp = strtok(NULL," ");
	if(tmp) rtctime.day = hex_to_ulong(tmp);

	tmp = strtok(NULL,":");
	if(tmp) rtctime.hour = hex_to_ulong(tmp);
	tmp = strtok(NULL,":");
	if(tmp) rtctime.min = hex_to_ulong(tmp);
	tmp = strtok(NULL," ");
	if(tmp) rtctime.second = hex_to_ulong(tmp);

	if(rtc_set_time(&rtctime) == 0) {
		sci_puts("RTC�̐ݒ�Ŏ��s���܂���\n");
		return;
	}

/*
	RX62N_RTC_TIME rtctime = {0x2011,0x08,0x02,0x02,0x23,0x45,0x12}; // BCD�Ŏw�肷��
	if(rtc_set_time(&rtctime) == 0)
	{
		sci_puts("[RTC�N�����s]\n");
		return false;
	}

	rtctime.year = 0;
	rtctime.mon = 0;
	rtctime.day = 0;
	rtctime.hour = 0;
	rtctime.min = 0;
	rtctime.second = 0;

	for(i=0;i<10;i++)
	{
		timer_wait_ms(500);
	}
	sci_puts("[RTC�N������]\n");
*/
}

void cmd_run() {
	if(*(unsigned long *)(0x00000004) == 0x4e444b54)  {
		typedef void (* FUNCP)(void);
		FUNCP func;
		func = *(FUNCP *)0x00000008;
		*(unsigned long *)(0x00000004) = 0;
		func();
	}
}

#if 0

void cmf_help(char *arg)
{
	int len = 0;
	int i = 0;
	int j;

	printf("Usage of command \"%s\"\n",helphelp);

	i = 0;
	while(lastcmd[i].func || lastcmd[i].sub)
	{
		if(strlen(lastcmd[i].name) >= len) len = strlen(lastcmd[i].name);
		i++;
	}

	i = 0;
	while(lastcmd[i].func || lastcmd[i].sub)
	{
		sci_puts(" ");
		sci_puts(helphelp);
		sci_puts(" ");
		sci_puts(lastcmd[i].name);
		for(j=0;j<len - strlen(lastcmd[i].name) + 2;j++) sci_puts(" ");
		sci_puts(lastcmd[i].help);
		sci_puts("\n");
		i++;
	}
	return;
}

void cmd_wait(char *arg)
{
	printf("WAIT start ... ");
	timer_wait_ms(atoi(arg));
	printf("done\n");
}

static void empty_loop(void)
{
	int i;
	volatile int j;
	for(i=0;i<1000;i++)
	{
		gpio_write_port(PIN_BUZZ,i & 1);
		for(j=0;j<1000;j++) {}
	}
}

static void empty_loop_next(void)
{
}

void cmd_memtest(char *arg)
{
	unsigned long *sdptr;
	unsigned long i;
	unsigned long loop;
	int errcnt;
	int sec,us,us2;
	const int TEST_SIZE = 0x01000000;

	printf("SDRAM�������e�X�g���J�n���܂�\n");
	
	us = timer_get_ms();
	memset((void *)0x08000000,0x55,0x01000000);
	us2 = timer_get_ms();

	printf("16MBytes��memset���鎞�Ԃ� %d�b�ł���\n",(us2-us)/1000);

	us = timer_get_ms();
	memcpy((void *)0x08000000,(void *)0x09000000,0x01000000);
	us2 = timer_get_ms();
	printf("16MBytes��memcpy���鎞�Ԃ� %d.%06d�b�ł���\n",(us2-us)/1000);

	sdptr = (unsigned long *)0x08000000;
//	timer_start();
	for(i=0;i<16777216/4;i++)
	{
		volatile int t = *sdptr++;
	}
//	timer_check(&sec,&us);
//	timer_stop();
//	printf("16MBytes��for���œǂݏo�����Ԃ� %d.%06d�b�ł���\n",sec,us);

	sdptr = (unsigned long *)0x08000000;
//	timer_start();
	for(i=0;i<16777216/4;i++)
	{
		*sdptr++ = i;
	}
//	timer_check(&sec,&us);
//	timer_stop();
//	printf("16MBytes��for���ŏ������ގ��Ԃ� %d.%06d�b�ł���\n",sec,us);
		
	{
		void (*func)(void);
		unsigned char *src,*dst;

//		timer_start();
		empty_loop();
//		timer_check(&sec,&us);
//		timer_stop();
//		printf("10^6�񃋁[�v�����RAM��Ŏ��s���鎞�Ԃ� %d.%06d�b�ł���\n",sec,us);

		src = (unsigned char *)empty_loop;
		dst = (unsigned char *)0x08000000;
		memcpy(dst,src,(unsigned long)empty_loop_next - (unsigned long)empty_loop);

		func = (void (*)(void))0x08000000;
//		timer_start();
		func();
//		timer_check(&sec,&us2);
//		timer_stop();
		printf("10^6�񃋁[�v��SDRAM��Ŏ��s���鎞�Ԃ� %d.%06d�b�ł���\n",(int)sec,(int)us2);
		
		us2 *= 100;
		printf("��%d.%d�{�̎��Ԃ��������Ă��܂�\n",(int)(us2/us/100),(int)((us2/us)%100));
	}

	printf("�ǂݏ����`�F�b�N���s���܂�\n");
	
	loop = 1;
	while(!sci_rxcount())
	{
		srand(loop);
		sdptr = (unsigned long *)0x08000000;
		for(i=0;i<TEST_SIZE / sizeof(long);i++)
		{
			*sdptr++ = myrand();
		}

		srand(loop);
		errcnt = 0;
		sdptr = (unsigned long *)0x08000000;
		for(i=0;i<TEST_SIZE / sizeof(long);i++)
		{
			if(*sdptr != myrand())
			{
				printf("Error:addr=%08x data=%02x\n",(unsigned int)sdptr,(unsigned char)*sdptr);
				errcnt++;
				if(sci_rxcount()) break;
				if(errcnt > 50) break;
			}
			sdptr++;
		}

		gpio_write_port(PIN_BUZZ,1);
		if(errcnt == 0)
		{
			printf("Memory test success loop=%d\n",(int)loop);
			for(i=0;i<500;i++)
			{
				int j;
				for(j=0;j<20000;j++) {}
				gpio_write_port(PIN_LED2,j & 1);
			}
		}
		else
		{
			printf("Memory test failed loop=%d\n",(int)loop);
			for(i=0;i<100;i++)
			{
				int j;
				for(j=0;j<100000;j++) {}
				gpio_write_port(PIN_LED2,j & 1);
			}
			break;
		}
		gpio_write_port(PIN_BUZZ,0);
		loop++;
	}
}

void cmf_test(char *arg)
{
}

void cmd_reboot(char *arg)
{
/*
	if(!strcmp(arg,"rom"))        reboot(REBOOT_NORMAL);
	else if(!strcmp(arg,"swap"))  reboot(REBOOT_SWAP);
	else if(!strcmp(arg,"ram"))
	{
		unsigned char *src;
		unsigned short crc;
		src = (unsigned char *)runtype_get_srambase();
		xprintf("calculating CRC ...");
		crc = crc_calc_hard(src,SDCONT_PROG_SIZE-2);
		xprintf("%04X\n",crc);
		if(crc != *(unsigned short *)&src[SDCONT_PROG_SIZE-2])
		{
			xprintf("CRC Error. Can not reboot from RAM.\n");
			return;
		}
		reboot(REBOOT_SRAM);
	}
	else if(!strcmp(arg,"dlbuf"))
	{
		unsigned short crc;
		xprintf("calculating CRC ...");
		crc = crc_calc_hard((unsigned char *)dlbuf,SDCONT_PROG_SIZE-2);
		xprintf("%04X\n",crc);
		if(crc != *(unsigned short *)&dlbuf[SDCONT_PROG_SIZE-2])
		{
			xprintf("CRC Error. Can not reboot from dlbuf.\n");
			return;
		}
		firm_update_ram(SDCONT_PROG_SIZE);
	}
	else 
	{
		xprintf(" missing arg. where reboot from ? { rom | swap | ram | dlbuf} \n");
	}
*/
}

functype_cmd search_cmd(const command_str *cmdlist,char *inputcmd,int *argpos)
{
	int ilen = 0;
	int i = 0;
	int a;

	// ���͂��ꂽ�R�}���h�̕����񒷂𐔂���
	while(inputcmd[ilen] && (inputcmd[ilen] != ' ')) ilen++;

	// �����̐擪�ӏ���T��
	a = ilen;
	while(inputcmd[a] && (inputcmd[a] == ' ')) a++;
	if(argpos) *argpos += a;

	// �R�}���h��T��
	while(cmdlist[i].func || cmdlist[i].sub)
	{
		printf("[%s]-[%s]\n",cmdlist[i].name,inputcmd);
		if(!strncmp(cmdlist[i].name,inputcmd,ilen)) // ���O����v�H
		{
			if(strlen(cmdlist[i].name) == ilen) // ��������v�H
			{
				lastcmd = (command_str *)cmdlist;
				strncat(helphelp,inputcmd,a);

				if(cmdlist[i].sub) // �T�u�R�}���h����H
				{
//					strncat(helphelp,inputcmd,a);
					return search_cmd(cmdlist[i].sub,&inputcmd[a],argpos);
				}
				return cmdlist[i].func;
			}
		}
		i++;
	}
	return NULL;
}


const command_str cmd_main[] = 
{
//   ���O          �֐�           �T�u�R�}���h   ����
	{""         , cmd_null      , NULL         , "" },
	{"help"     , cmf_mhelp     , NULL         , "���̃w���v��\\��"},
// 	{"system"   , cmf_system    , NULL         , "�V�X�e������\\��"},
	{"load"     , cmf_load      , NULL         , "SREC�t�@�C���̃��[�h"},
	{"dump"     , cmf_memdump   , NULL         , "�������_���v"},
 	{"memfill"  , cmf_memfill   , NULL         , "�������t�B��"},
	{"memtest"  , cmd_memtest   , NULL         , "�������e�X�g"},
//	{"spi"      , NULL          , cmf_spi      , "SPI�������Ɋւ���l�X�ȃe�X�g"},
//	{"sd"       , cmf_sd        , NULL         , "MMC/SD�J�[�h�Ɋւ���l�X�ȃe�X�g"},
 	{"jmp"      , cmf_jmp       , NULL         , "�C�ӂ̔Ԓn�փW�����v"},
	{"wait"     , cmd_wait      , NULL         , "�^�C�}�̃e�X�g" },
	{"time"     , cmf_time      , NULL         , "���ݎ����̐ݒ�ƕ\\��" },
 	{"reboot"   , cmf_reboot    , NULL         , "�ċN��"},
// 	{"dir"      , cmf_dir       , NULL         , "�t�@�C���̈ꗗ��\��"},
// 	{"upload"   , cmf_upload    , NULL         , "�t�@�C�����A�b�v���[�h"},
//	{"type"     , cmf_type      , NULL         , "�t�@�C���̓��e��\��"},
//	{"del"      , cmf_del       , NULL         , "�t�@�C�����폜����"},
//	{"save"     , cmf_save      , NULL         , "RAM�f�B�X�N��ROM�ɕۑ�"},
//	{"format"   , cmf_format    , NULL         , "RAM�f�B�X�N������������"},
//	{"init"     , cmf_init      , NULL         , "RAM�f�B�X�N���t���b�V�����������烍�[�h����"},
//	{"bpb"      , cmf_bpb       , NULL         , "MBR�̏���\��"},
//	{"sector"   , cmf_sector    , NULL         , "�g�p���Ă���Z�N�^��\��"},
 	{"test"     , cmf_test      , NULL         , "�l�X�ȃe�X�g���s��"},
 	{""         , NULL          , NULL         , "" },
};

#endif
