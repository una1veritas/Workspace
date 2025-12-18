/*==========================================================*/
/*|     CP/M-68K(tm) BIOS for the MES68K8_RAM 				|*/
/*|															|*/
/*|     Copyright 1983, Digital Research.					|*/
/*|															|*/
/*|	Modified 2024.5.22										|*/
/*|															|*/
/*==========================================================*/

#define int32_t		long
#define int16_t		short int
#define int8_t		char
#define uint32_t	unsigned long
#define uint16_t	unsigned short
#define uint8_t		unsigned char

#define REG	register

/************************************************/
/* BIOS  Table Definitions						*/
/************************************************/


/* PIC I/F
//  ---- request command to PIC
// CREQ_COM = 1 ; CONIN  : return char in UNI_CHR
//          = 2 ; CONOUT : UNI_CHR = output char
//          = 3 ; CONST  : return status in UNI_CHR
//                       : ( 0: no key, 1 : key exist )
//          = 4 ; STROUT : string address = (PTRSAV, PTRSAV_SEG)
//          = 5 ; DISK READ
//          = 6 ; DISK WRITE
//          = 0 ; request is done( return this flag from PIC )
//                return status is in CBI_CHR;
*/

#define CONIN_REQ	0x01
#define CONOUT_REQ	0x02
#define CONST_REQ	0x03
#define STROUT_REQ	0x04
#define REQ_DREAD	0x05
#define REQ_DWRITE	0x06

/* shared memory address */
#define SMA 0x100;
#define INVOKE_PIC	0x80000

struct preq_hdr {
	uint8_t  UREQ_COM;		/* unimon CONIN/CONOUT request command */
	uint8_t  UNI_CHR;		/* charcter (CONIN/CONOUT) or number of strings */
	uint8_t  *STR_addr;		/* unimon string offset */
	uint8_t  CREQ_COM;		/* unimon CONIN/CONOUT request command */
	uint8_t  CBI_CHR;		/* charcter (CONIN/CONOUT) or number of strings */
	uint8_t	 disk_drive;
	uint8_t	 dummy;			/* alignment word boundary */
	uint16_t disk_track;
	uint16_t disk_sector;
	uint8_t  *data_dma;
};

struct preq_hdr *preq;
uint8_t *pic;

/* (DPB) Disk Parameter Block Structure */

struct dpb
{
	uint16_t spt;		/* sectors per track */
	uint8_t  bsh;		/* Block SHift factor */
	uint8_t  blm;		/* BLock Mask */
	uint8_t  exm;		/* EXtent Mask */
	uint8_t  dpbjunk;	/* adjust word boundary */
	uint16_t dsm;		/* disk size-1 */
	uint16_t drm;		/* directory max */
	uint8_t  al0;		/* alloc 0 */
	uint8_t  al1;		/* alloc 1 */
	uint16_t cks;		/* check size */
	uint16_t off;		/* track offset */
};


/* Disk Parameter Header Structure */

struct dph
{
	uint8_t  *xltp;			/* sector translate table address */
	uint16_t dphscr[3];		/* reserve 3 words */
	uint8_t  *dirbufp;		/* directory buffer address */
	struct   dpb *dpbp;		/* DPB address */
	uint8_t  *csvp;			/* check sum vector address */
	uint8_t  *alvp;			/* allocation vector address */
};



/********************************************************/
/*	Directory Buffer for use by the BDOS				*/
/********************************************************/

uint8_t dirbuf[128];

/****************************************/
/*	CSV's								*/
/****************************************/
/*
uint8_t csv0[256];
uint8_t csv1[256];
uint8_t csv2[256];
uint8_t csv3[256];
*/
/****************************************/
/*	ALV's								*/
/****************************************/

uint8_t alv0[256];	/* (dsm0 / 8) + 1	*/
uint8_t alv1[256];	/* (dsm1 / 8) + 1	*/
uint8_t alv2[256];	/* (dsm2 / 8) + 1	*/
uint8_t alv3[256];	/* (dsm3 / 8) + 1	*/

/************************************************/
/*	Disk Parameter Blocks						*/
/************************************************/
/* al0 and al1 are reserved in CP/M-68K. */
/* Refer to the CP/M-68K System Guide Section 5.2.3 Disk Parameter Block */

/* original code */
/*struct dpb dpb0 = { 26,   3,   7,   0,   0,   242,   63,  0,   0,  16,   2}; */

/* 4MB DISK DPB definition */
/*************    spt, bsh, blm, exm,  jnk,   dsm,  drm,  al0, al1, cks, off */
struct dpb dpb2 = { 128,  4,  15,   0,   0,  2039, 1023,    0,  0,   0,   0};

/* For the reference : MEZ88_47Q CP/M-86 4MB DISK DPB */
/*struct dpb dpb2 = { 128,  4,  15,   0,   0,  2039, 1023,  255, 255,   0,   0};*/

/********************************************************/
/* Sector Translate Table for Floppy Disks				*/ 
/********************************************************/
/*
uint8_t	xlt[26] = {  1,  7, 13, 19, 25,  5, 11, 17, 23,  3,  9, 15, 21,
		     2,  8, 14, 20, 26,  6, 12, 18, 24,  4, 10, 16, 22 };
*/
/********************************************************/
/* Disk Parameter Headers								*/
/********************************************************/

/* Disk Parameter Headers */
struct dph dphtab[] = {
	{0L, 0, 0, 0, &dirbuf[0], &dpb2, 0L, &alv0[0]}, /*dsk a*/
	{0L, 0, 0, 0, &dirbuf[0], &dpb2, 0L, &alv1[0]}, /*dsk b*/
	{0L, 0, 0, 0, &dirbuf[0], &dpb2, 0L, &alv2[0]}, /*dsk c*/
	{0L, 0, 0, 0, &dirbuf[0], &dpb2, 0L, &alv3[0]}, /*dsk d*/
};

/********************************************/
/*	Memory Region Table						*/
/********************************************/

struct mrt {
	uint16_t count;
	uint32_t tpalow;
	uint32_t tpalen;
} memtab;				/* Initialized in M68KBIOA.S	*/


/****************************************/
/*	IOBYTE								*/
/****************************************/

uint16_t iobyte;	/* The I/O Byte is defined, but not used */

/****************************************************************/
/*	Define the number of disks supported and other disk stuff	*/
/****************************************************************/

#define NUMDSKS 4		/* number of disks defined */

#define MAXDSK  (NUMDSKS-1)	/* maximum disk number 	   */


uint8_t wakeup_pic()
{
	REG uint8_t	dummy;
	dummy = *pic;					/* wake uo PIC */
	while( preq->CREQ_COM ) {};		/* Wait until PIC processing is complete */
	return( preq->CBI_CHR );		/* return status */
}

/* change big endian to little endian */
uint16_t order_b2l( data )
uint16_t data;
{
	uint16_t p1, p2;

	p1 = (data & 0x00ff) << 8;
	p2 = (data & 0xff00) >> 8;

	return( p1 | p2 );

}

void set_dma( dma_adr )
uint8_t *dma_adr;
{
	uint32_t p1, p2, p3, p4;

	p1 = ((uint32_t)dma_adr & 0x000000ff) << 24;
	p2 = ((uint32_t)dma_adr & 0x0000ff00) << 8;
	p3 = ((uint32_t)dma_adr & 0x00ff0000) >> 8;
	p4 = ((uint32_t)dma_adr & 0xff000000) >> 24;

	/* change big endian to little endian */
	preq->data_dma = (uint8_t *)(p1 | p2 | p3 | p4);
}

/*****************************************************/ 
/*	Generic console status input status				*/
/****************************************************/

uint8_t con_st()
{
	preq->CREQ_COM = CONST_REQ;	/* set CONST request */
	return(wakeup_pic());
}
 
 
/********************************************/ 
/*	Generic console input					*/
/********************************************/
 
uint8_t con_in()
{
	preq->CREQ_COM = CONIN_REQ;	/* set CONIN request */
	return(wakeup_pic());
}
 
 
/********************************************/ 
/*	Generic consolet output					*/
/********************************************/
 
con_out(ch)
REG uint8_t ch;
{
	preq->CREQ_COM = CONOUT_REQ;	/* set CONIN request */
	preq->CBI_CHR = ch;				/* set CONOUT character */
	wakeup_pic();
}

/************************************************/
/*	Flush all disk buffers						*/
/************************************************/
/*
uint16_t flush()
{
	return( 0 );
}
*/
/********************************************************/
/*	Bios READ Function -- read one sector				*/
/********************************************************/

uint16_t read()
{
	preq->CREQ_COM = REQ_DREAD;	/* set READ request */
	return((uint16_t)wakeup_pic());
}
/********************************************************/
/*	BIOS WRITE Function -- write one sector 			*/
/********************************************************/

uint16_t write(mode)
uint8_t mode;
{
	preq->CREQ_COM = REQ_DWRITE;	/* set READ request */
	return((uint16_t)wakeup_pic());
}

/****************************************************/
/*	BIOS Sector Translate Function					*/
/****************************************************/

uint16_t sectran(s, xp)
REG uint16_t  s;
REG uint8_t *xp;
{
	if (xp) return (uint16_t)xp[s];
	else	return (s+1);
}


/****************************************************/
/*	BIOS Set Exception Vector Function				*/
/****************************************************/

uint32_t setxvect(vnum, vval)
uint16_t vnum;
uint32_t vval;
{
	uint32_t  oldval;
	uint32_t *vloc;

	vloc = (uint32_t *)(vnum << 2);
	oldval = *vloc;
	*vloc = vval;

	return(oldval);	
/*
	REG LONG  oldval;
	REG BYTE *vloc;

	vloc = ( (long)vnum ) << 2;
	oldval = vloc->lword;
	vloc->lword = vval;

	return(oldval);	
*/
}


/************************************************/
/*	BIOS Select Disk Function					*/
/************************************************/

struct dph *slctdsk(dsk)
REG uint8_t dsk;
{
	if (dsk > MAXDSK ) return(0L);	/* error return */
	preq->disk_drive = dsk;			/* set drive no. to shared memory area */
	return(&dphtab[dsk]);
}

/****************************************************************/
/*																*/
/*	Bios initialization.  Must be done before any regular BIOS	*/
/*	calls are performed.										*/
/*																*/
/****************************************************************/

biosinit()
{
	REG uint8_t c, *p;
	
	preq = (struct preq_hdr *)SMA;	/* Set shared memory address */
	pic = (uint8_t *)INVOKE_PIC;	/* Set the PIC wake up address */

	/* initialize PIC request table */
	p = (uint8_t *)preq;
	for(c = 0; c < sizeof(struct preq_hdr); c++) *p++ = 0;
}


/********************************************************************/
/*																	*/
/*      BIOS MAIN ENTRY -- Branch out to the various functions.		*/
/*																	*/
/********************************************************************/
 
uint32_t cbios(d0, d1, d2)
REG uint16_t	d0;
REG uint32_t	d1, d2;
{
	switch(d0) {
		case 0:	/* INIT		*/
			biosinit();
			break;

		case 1:	/* WBOOT	*/
/*
			flush();
*/
			wboot();
		     /* break; */

		case 2:	/* CONST	*/
			return(con_st());

		case 3:	/* CONIN	*/
			return(con_in());

		case 4:	/* CONOUT	*/
			con_out((uint8_t)d1);
			break;

		case 5:	/* LIST		*/
		case 6: /* PUNCH	*/
		case 7:	/* READER	*/
			return(0);

		case 8:	/* HOME		*/
			preq->disk_track = 0;
			break;

		case 9:	/* SELDSK	*/
			return((uint32_t)slctdsk((uint8_t)d1));

		case 10:	/* SETTRK	*/
			preq->disk_track = order_b2l( (uint16_t)d1 );
			break;

		case 11:	/* SETSEC	*/
/*
			preq->disk_sector = order_b2l((uint16_t)d1-1);
*/
			preq->disk_sector = order_b2l((uint16_t)d1);		/* d1-1 in PIC firmware */
			break;

		case 12:	/* SETDMA	*/
			set_dma((uint8_t *)d1);
			break;

		case 13: 	/* READ		*/
			return(read());

		case 14:	/* WRITE	*/
			/* don't use mode(D1=1, 2) */
			/* MEZ68K_RAM use only 128bytes per sector */
			return(write());

		case 15: /* LIST STATUS */
			return ( 0x000 );

		case 16:	/* SECTRAN	*/
			return(sectran((int)d1, d2));

		case 18:	/* Get Memory Region Table	*/
			return((uint32_t)(&memtab));

		case 19:	/* GETIOB	*/
			return(iobyte);

		case 20:	/* SETIOB	*/
			iobyte = (int)d1;
			 break;

		case 21:	/* FLUSH	*/
/*
			if (flush()) return(0L);
			 else return(0xffffL);
*/
			break;
			
		case 22:	/* Set Exception Vecter Address */
			return(setxvect((uint16_t)d1,d2));

	} /* end switch */
	return(0L);
}
