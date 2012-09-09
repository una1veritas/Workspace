// ＣＱ出版のダウンロードサイトにある、ＬＰＣ２３８８用サンプルソフトを使って、( gcc_sample.c )
// モニタを移植したもの。ＬＰＣ２３８８の内部機能レジスタの設定などは、
// サンプルソフトをそのまま使っている。また、スタートアップもそのままである。
// ＡＲＭ開発環境も、ＣＱ出版のダウンロードサイトにある　arm-tools-new-20080625.tar.zip
// を、Ｗｉｎｄｏｗｓで解凍して、arm-tools-new-20080625.tar.bz2 とし、
// これを、Ｃｙｇｗｉｎの、/usr/local で解凍して使用。解凍方法は、
// tar jxvf arm-tools-new-20080625.tar.bz2 [リターン]
#define COM_BaseAdr	0xE000C000	/* UART0 ベースアドレス */
#define COM_RBR			((volatile unsigned char *)(COM_BaseAdr+0x00))
#define COM_THR			((volatile unsigned char *)(COM_BaseAdr+0x00))
#define COM_DLL			((volatile unsigned char *)(COM_BaseAdr+0x00))
#define COM_DLM			((volatile unsigned char *)(COM_BaseAdr+0x04))
#define COM_IER			((volatile unsigned char *)(COM_BaseAdr+0x04))
#define COM_IIR			((volatile unsigned char *)(COM_BaseAdr+0x08))
#define COM_FCR			((volatile unsigned char *)(COM_BaseAdr+0x08))
#define COM_LCR			((volatile unsigned char *)(COM_BaseAdr+0x0C))
#define COM_LSR			((volatile unsigned char *)(COM_BaseAdr+0x14))
#define COM_SCR			((volatile unsigned char *)(COM_BaseAdr+0x1C))
#define COM_FDR			((volatile unsigned char *)(COM_BaseAdr+0x28))
// ＵＳＢ　のＵＡＲＴから、受信文字があるかをチェック。この部分は追加。
char CheckRecvData()	// ライン・ステータス取得。１文字受信なら１、無ければ０をリターン
{
	if(*COM_LSR & 0x1)
		return 1;
	else
		return 0;
}
// 以下は、インターフェイス２００９年８月のダウンロードサイトの、cq_gnu_resources_20090504.tar.gz
// を解凍した中の、gcc_sample のソース、gcc_sample.c からコピーしたもの
char RecvData(){
  while((*COM_LSR & 0x1) == 0){};	/* ライン・ステータス取得 */
  return (unsigned char)*COM_RBR;	/* 受信データを返す */
}

void SendData(char data){
  while((*COM_LSR & 0x20)==0) {}	/* 送信バッファが空くまで待つ */
  *COM_THR = (unsigned char)data;	/* 送信データ書き込み */
}

/* 割り込みコントローラ(VIC) */
#define VIC_IRQStatus	((volatile unsigned int *)(0xFFFFF000))
#define VIC_IntSelect   ((volatile unsigned int *)(0xFFFFF00C))
#define VIC_IntEnable   ((volatile unsigned int *)(0xFFFFF010))

#define UART_INT_BIT 0x00000040

int IRQ_func(){
  if((*VIC_IRQStatus & (1<<6))){ // UART0
    if(((*COM_IIR >> 1) & 0x7) == 0x2){ // Receive Data Available
      SendData(RecvData());
    }
  }
}

/* システムコントローラ */
#define SYS_SCS			((volatile unsigned int *)(0xE01FC1A0))

/* ポートコントローラ */
#define PINSEL0			((volatile unsigned int *)(0xE002C000))
#define PINSEL1			((volatile unsigned int *)(0xE002C004))
#define GPIO_PINMODE1	((volatile unsigned int *)(0xE002C044))

/* FGPIOコントローラ */
#define FGPIO_FIO1DIR	((volatile unsigned int *)(0x3FFFC020))
#define FGPIO_FIO1MASK	((volatile unsigned int *)(0x3FFFC030))
#define FGPIO_FIO1PIN	((volatile unsigned int *)(0x3FFFC034))
#define FGPIO_FIO1SET	((volatile unsigned int *)(0x3FFFC038))
#define FGPIO_FIO1CLR	((volatile unsigned int *)(0x3FFFC03C))

// クロック＆パワーコントローラ 
#define CLK_PCLKSEL0	((volatile unsigned int *)(0xE01FC1A8))
#define POW_PCONP		((volatile unsigned int *)(0xE01FC0C4))


// システムコントローラ 
#define SYS_SCS			((volatile unsigned int *)(0xE01FC1A0))

// クロック＆パワーコントローラ制御レジスタ 
#define CLK_PLLCON		((volatile unsigned int *)(0xE01FC080))
#define CLK_PLLCFG		((volatile unsigned int *)(0xE01FC084))
#define CLK_PLLSTAT		((volatile unsigned int *)(0xE01FC088))
#define CLK_PLLFEED		((volatile unsigned int *)(0xE01FC08C))

#define CLK_CCLKCFG		((volatile unsigned int *)(0xE01FC104))
#define CLK_USBCLKCFG	((volatile unsigned int *)(0xE01FC108))
#define CLK_CLKSRCSEL	((volatile unsigned int *)(0xE01FC10C))

#define CLK_PCLKSEL0	((volatile unsigned int *)(0xE01FC1A8))
#define CLK_PCLKSEL1	((volatile unsigned int *)(0xE01FC1AC))
// タイマコンローラ 
#define TIMER_T0IR		((volatile unsigned int *)(0xE0004000))
#define TIMER_T0TCR		((volatile unsigned int *)(0xE0004004))
#define TIMER_T0TC		((volatile unsigned int *)(0xE0004008))
#define TIMER_T0PR		((volatile unsigned int *)(0xE000400C))
#define TIMER_T0PC		((volatile unsigned int *)(0xE0004010))
#define TIMER_T0MCR		((volatile unsigned int *)(0xE0004014))
#define TIMER_T0MR0		((volatile unsigned int *)(0xE0004018))
//	MAM Control
#define MAM_MAMCR		((volatile unsigned long *)(0xE01FC000))
#define MAM_MAMTIM		((volatile unsigned long *)(0xE01FC004))

/* CPU固有初期化処理 */
void CPU_Initialize(void)
{
  /*** PLLクロック発振初期化 ***/

  /* すでにPLLが動作中だった場合は停止させる */
  if ( *CLK_PLLSTAT & (1<<25) ) {
    /* PLLCON - PLL Enable & disconnected */
    *CLK_PLLCON   =0x00000001;
    /* PLL Feed operation */
    *CLK_PLLFEED  =0x000000AA;
    *CLK_PLLFEED  =0x00000055;
  }
  /* PLLCON - PLL Disable & disconnected(PLL完全停止) */
  *CLK_PLLCON   =0x00000000;
  /* PLL Feed operation */
  *CLK_PLLFEED  =0x000000AA;
  *CLK_PLLFEED  =0x00000055;

  *SYS_SCS = 0x21;	/* 12MHz OSC Enable & FGPIO Select */
  while((*SYS_SCS&0x40) == 0){}	/* OSCSTAT Wait */

  /* CLKSRCSEL - MASTER oscillator select */
  *CLK_CLKSRCSEL=0x00000001;

  /* PLLCFG - MCLK=12MHz use, FCC0 = 288MHz M=144,N=12 */
  *CLK_PLLCFG   =0x000B008F;
  /* PLL Feed operation. */
  *CLK_PLLFEED  =0x000000AA;
  *CLK_PLLFEED  =0x00000055;

  /* PLLCON - PLL Enable & disconnected */
  *CLK_PLLCON   =0x00000001;
  /* PLL Feed operation */
  *CLK_PLLFEED  =0x000000AA;
  *CLK_PLLFEED  =0x00000055;

  /* CPU Clock Divider 1/4 */
  *CLK_CCLKCFG  =0x00000003;
  /* USB Clock Divider 1/6 */
  *CLK_USBCLKCFG=0x00000005;

  while ( ((*CLK_PLLSTAT & (1<<26)) == 0) ); /* Check lock bit status */

  /* PLLCON - PLL Enable & Connected */
  *CLK_PLLCON   =0x00000003;
  /* PLL Feed operation. */
  *CLK_PLLFEED  =0x000000AA;
  *CLK_PLLFEED  =0x00000055;
  while ( ((*CLK_PLLSTAT & (1<<25)) == 0) ); /* Check connect bit status */
}
void main2(int);// 追加
int main(){
  int i,j;
	char c;
  CPU_Initialize();

  *SYS_SCS = *SYS_SCS | 1;	/* FGPIO Select */
  // LEDはP1[18]に接続されているので、これをデジタルI/Oに設定
  *PINSEL1 =0;			/* GPIO Select */
  *GPIO_PINMODE1=0;			/* PullUp Enable */

  /* LED点灯制御設定 */
  *FGPIO_FIO1DIR =0x00040000;	/* P1[18] OutPut */
  *FGPIO_FIO1MASK=0x00000000;	/* P1[18] Non Mask */
  *FGPIO_FIO1PIN =0x00040000;	/* P1[18] '1' -> LED OFF */

  // UARTはTXD0とRXD0を使う
  /* ポートをUARTモードに設定(UART0) */
  *PINSEL0=(*PINSEL0 & 0xFFFFFF0F) | 0x50;	/* TXD0&RXD0選択 */
  /* UART0供給クロック 72MHz/4=18MHz */
  *CLK_PCLKSEL0=(*CLK_PCLKSEL0 & 0xFFFFFF3F);
  /* UART0パワーイネーブル */
  *POW_PCONP = *POW_PCONP | 8;	/* UART0 */

  /* 割り込みコントローラ設定(UARTx) */
  *VIC_IntSelect = *VIC_IntSelect & (~UART_INT_BIT);
  *VIC_IntEnable = *VIC_IntEnable |   UART_INT_BIT;
  
  /* UART設定 */
  *COM_LCR =0x80;   /* 分周設定レジスタ選択 */
  *COM_DLL =9;     /* 18MHz供給時の115200 bps設定 */
  *COM_DLM =0;
  *COM_FDR =(/*MULVAL=*/ 12 <<4)|/*DIVADDVAL=*/1; /* FDRレジスタ設定 */
  *COM_LCR =0x03;		/* パリティなし/ストップビット1/データ8ビット */

//  *COM_IER =1;		/* 受信割り込み使用 */

  SendData('O');
  SendData('K');
  SendData('\n');
  SendData('\r');
// ３行追加。これが無いと、処理がかなり遅い。
	*MAM_MAMCR = 0;//	Stop MAM
	*MAM_MAMTIM = 4;// over 60MHz
	*MAM_MAMCR = 2;//	Start MAM
// 簡易モニタに移行する。
	main2(0);	//	簡易モニタに移動。戻らない。
  // LEDの点滅。若干変更。
  for(;;){
    *FGPIO_FIO1PIN =0x00040000;	/* P1[18] '1' -> LED OFF */
    for(j=0;j<1000000;j++)
	{
		if(CheckRecvData)
		{
			c = RecvData();
			SendData(c);
			if(c == '\r')
				SendData('\n');
		}
	}
    *FGPIO_FIO1PIN =0x00000000;	/* P1[18] '0' -> LED ON */
    for(j=0;j<300000;j++)
	{
		if(CheckRecvData)
		{
			c = RecvData();
			SendData(c);
			if(c == '\r')
				SendData('\n');
		}
	}
  }

  while(1){}

  return 1;
}
// 	以下は簡易モニタ。
//	メモリのダンプ、変更、任意番地の実行、
//	ホストからのヘキサをメモリにロード、などの機能がある。
char getcon(void);	// 簡易モニタのコンソール入力
void putcon(char);	// 簡易モニタのコンソール出力
char statcon(void);	// 簡易モニタのコンソール入力チェック
union param32{
	long l32;
	int i32;
	unsigned long ul32;
	short s16[2];
	unsigned short us16[2];
	char c8[4];
	unsigned char uc8[4];
	char *cpoint;
	unsigned char *ucpoint;
	short *spoint;
	unsigned short *uspoint;
	long *lpoint;
	unsigned long *ulpoint;
	union param32 *upoint;
	long (*ppoint)();	/* アドレスとして関数呼び出し */
};
#define COLUMN 80
char htol(long *,char **);
void crlf(void);
int chex(char );
void ckasci(short );
void cprintf();
int csprintf();
void spacen(int );
//char buf[80];
//====================================================
char getcon(void);
void putcon(char );
void putscon(char *);
int combuf2f;
//char combuf2[COLUMN];		/* 前回のコマンド記憶 */
char combuf[COLUMN];		/* コマンド入力バッファー */
union param32 param[8];
void dumpm(union param32 *,int);
void dumpmt(void);
void movem(union param32 *,long );
int argck(char **,union param32 *,int );
char htol(long *,char **);
void paramer(void);
int verify(union param32 *);
void srload(union param32 *,int);
int setmemo(union param32 *,int,int );
void findp(union param32 *);
long cpysstr(char *,char *,long );
long gosub(union param32 addr);
int dhrymain(void);
int getscon(char *,int,int);
void main2 (int mode);
int sievemain(void);
#define swapw(x) (x<<8)|((x>>8) & 0xFF)	//　スワップワード。ビッグエンディアンから、リトルエンディアンに変更
#define	getcon RecvData	// 簡易モニタのコンソールを、上記サブルーチンに変更
#define	putcon SendData	// 簡易モニタのコンソールを、上記サブルーチンに変更
#define	statcon CheckRecvData	// 簡易モニタのコンソールを、上記サブルーチンに変更
//	以下、最後までが、モニタ本体。一応３２ビットマシンなら大抵使える。
//	変更すべきは、メモリダンプで、ビッグエンディアンと、リトルエンディアンが関係する部分。
void main2 (int mode) {
	int ret;
	char *comp,c;
	int i,j,l;
	int ts,te;
	*TIMER_T0PR = 17;// LPC2388 のタイマー０のプリスケーラを、1/18 にする。
	*TIMER_T0TCR = 1;// タイマーカウンタを、１μ秒で１カウントする。
	putscon("\r\nLPC2388 monitor 2011.12.5-1\r\n");
	ts=te=0;
	while(1)
	{
LOOP:
		putscon("LPC2388-Bug>");// 簡易モニタのプロンプトを送信。
		if((ret=getscon(combuf,COLUMN-1,0))== -1)// コマンド入力。
		{
				crlf();
				goto LOOP;
		}
		comp=combuf;

		c= *comp;
		switch(c)
		{
//		case	'h':
//			putscon("help");
//			break;
		case	'd':
			j=0;
			comp++;
			if((*comp=='x'))
			{
				j |= 0x10;
				comp++;
			}
//			if(j==0)
			{
				if(*comp ==0)
				{
					l = param[1].l32 - param[0].l32 + 2;
					l &= 0xfffffffe;
					param[0].l32 += l;
					param[1].l32 += l;
				}
				else
				{
					if((i=argck(&comp,param,1))== -1)
					{
						paramer();
						break;
					}
					if(i==1)
					{
						param[1].l32=param[0].l32+0x7e;
					}
				}
			}
				dumpm(param,j);
			break;
		case	'm':// メモリのコピー。ムーブメモリ。
			j=0;
			comp++;
			if((*comp=='W')||(*comp=='w'))
			{
				j=1;
				comp++;
			}
			else if((*comp=='L')||(*comp=='l'))
			{
				j=2;
				comp++;
			}
			if((i=argck(&comp,param,3)) != 3)
			{
				paramer();
				break;
			}
			ts = *TIMER_T0TC;// 実行前のカウンタを読む
			movem(param,(long)j);
			te = *TIMER_T0TC;// 実行後のカウンタを読む
			break;

//		case	'V':
		case	'v':
			comp++;
			if((i=argck(&comp,param,3)) != 3)
			{
				paramer();
				break;
			}
			ts = *TIMER_T0TC;
			verify(param);
			te = *TIMER_T0TC;
			break;
/*
		case	'F':// フィルデータ　この機能は、s コマンドで行える。
			comp++;
//		addr,length,start,add,byte

			if((i=argck(&comp,param,5)) != 5)
			{
				paramer();
				break;
			}
			ts = *TIMER_T0TC;
			fillp(param);
			te = *TIMER_T0TC;
			break;
*/
// ファインドパターン。１バイトから４バイトまでのデータをメモリから探す。
		case	'f':
			comp++;
			if((i=argck(&comp,param,4)) != 4)
			{
				paramer();
				break;
			}
			ts = *TIMER_T0TC;
			findp(param);
			te = *TIMER_T0TC;
			break;
// ゴーコマンド。指定したアドレスをサブルーチンとして実行。
// 戻れた場合は、リターン値を表示する。
		case	'g':
			comp++;
			if((i=argck(&comp,param,1))== -1)
			{
				paramer();
				break;
			}
			ts = *TIMER_T0TC;
			ret = param[0].ppoint();
			te = *TIMER_T0TC;
			cprintf("\r\nret= %08lX",ret);
			break;
//		case	'L':
// ロードコマンド。ホストから送られるヘキサデータをメモリに書く。
		case	'l':
			j=0;
			comp++;
			if((i=argck(&comp,param,1))== -1)
			{
				paramer();
				break;
			}
			crlf();
			srload(param,1);
			break;
// セットメモリ。メモリの内容をインタラクティブに表示、変更する。
// 引数が２個以上は、フィルメモリなど、かなり高機能なメモリ変更ができる。
		case	's':
			j=0;
			comp++;
			if((*comp=='N')||(*comp == 'n'))
			{
				j=4;
				comp++;
			}
			if((*comp=='W')||(*comp=='w'))
			{
				j |= 1;
				comp++;
			}
			else if((*comp=='L')||(*comp=='l'))
			{
				j |= 2;
				comp++;
			}
			param[2].l32 = 1;
			param[3].l32 = 0;
			param[4].l32 = 0;
			param[5].l32 = 0;
			param[6].l32 = 0;
//			param[7].l32 = 0;
			if((i=argck(&comp,param,1))== -1)
			{
				paramer();
				break;
			}
			ts = *TIMER_T0TC;
			setmemo(param,j,i);
			te = *TIMER_T0TC;
			break;
//		case	'D':// dhry stone テスト
//			ret = dhrymain();
//			cprintf("\r\nTimer HEX = %lX,",ret); cprintf(" time = %ld",ret);
//			break;

		case	'S':// エラトステネスのふるいで、素数を求めたときの実行時間
			ret = sievemain();
			cprintf("sieve=%ld ",ret);
			break;

		case	'T':
			l=te-ts;
			cprintf("\r\nTimer HEX = %lX, %ld",l,l);
			cprintf("μs");
			crlf();
			break;
		case	'i':	// interrupt 実装なし。
			break;
		default:// モニタ未定義命令の場合。
			cprintf("default");
			break;
		}
		crlf();


	}
   return ;// 戻るコマンドは実装してないので、ここには来ない。
}
// エラトステネスのふるいで、素数を求めたときの実行時間を表示。素数は１８９９個になる。
// オリジナルは、CP/M-80 時代の、BDSC がデモで使ったサンプルソフト。sieve.c
// サンプルソフトは、Intel 8080で実行され、10 iterations　で、１５秒かかった。(8080 2MHz ???)
// ARM 72MHz では、ＦＲＯＭで、1000 iterations　約６秒なので、２５０倍早いことになる。
#define SIZE 8191
#define SIZEPL 8192
//char buf[SIZEPL];

int sievemain()
{
	long prime,k,i,count,iter;
	long l,ts,te;
//	long time();
	int n;
	char *buf;
//	l=0;
	cprintf("\r\nBuffer = 0x4000A000 --> 0x4000BFFF");
	cprintf("\r\ntype return to do 1000 iterations:");
	getcon();// 何かキーを押すとスタート。
	buf = (char *)0x4000A000;// 作業メモリを、0x4000A000--0x4000BFFF に設定。
	ts = *TIMER_T0TC;
	for(iter=0;iter<1000;iter++)
	{
		count=0;	//	
		for(i=0;i<=SIZE;i++)
		{
			buf[i]=1;
		}
		for(i=0;i<=SIZE;i++)
		{
			if(buf[i])
			{
				k=i+i+3;
				prime=k;
				k += i;
				for(;k<SIZEPL;k += prime)
				{
					buf[k]=0;
				}
				count++;
			}
		}
	}
	te = *TIMER_T0TC;
	cprintf("\r\n%ld primes Timer0=%ld count\r\n",count,te-ts);
	return count;
}

void putscon(char *s)
{
	while(*s)
	{
		putcon(*s++);
	}
}
// モニタ命令の１行を入力する。
// バックスペース機能があるが、その他、１行限定のエディット機能は、コメントアウトしている。
// セットメモリ用の、アドレス更新、戻りのための、'n' 'r' コマンドがある。　引数　f==1
int getscon(char *cp,int cn,int f)
{
	int n,n2;
	char c;
	int s;
//	int i;
//	int insf;

	n=0;
	n2=0;
//	insf=1;
/*
	if(f & 2)	// すでにバッファーには入力文字がいくつか有る 
	{
		for(i=0;i<cn;i++)
		{
			c = cp[i];
			if(c == 0)
			{
				n=i;
				n2=i;
				break;
			}
			putcon(c);
		}
		if((f & 4) ==0)
		{
			while(i--)
				putcon(8);
			n=0;
		}
		f=0;
	}
	else
*/
		*cp=0;
	while(n <= cn)
	{
//		c=getcon_combuf2();
		c=getcon();
		if(c =='\3')
		{
			return(-1);
		}
		if(c==0xd)	// CR
		{
			cp[n2]='\0';
			return(n2);
		}
/*		if(c == 'O'-0x40)	// ctrl O
		{
			insf ^= 1;
			if(insf & 1)
			{
				for(i=0;i<=n;i++)
					putcon(0x8);
				putcon('>');
				for(i=0;i<n;i++)
					putcon(cp[i]);
			}
			else
			{
				for(i=0;i<=n;i++)
					putcon(0x8);
				putcon('=');
				for(i=0;i<n;i++)
					putcon(cp[i]);
			}
		}
		if(c == 'G'-0x40)
		{
			if(n2 > n)
			{
				for(i=n;i<n2;i++)
				{
					cp[i]= cp[i+1];
				}
				n2--;
				for(i=n;i<n2;i++)
				{
					putcon(cp[i]);
				}
				putcon(' ');
				for(i=n;i<n2+1;i++)
				{
					putcon(0x8);
				}
			}
		}
		if((c == 'S'-0x40) && (n > 0))
		{
			n--;
			putcon(0x8);	// back space code 
		}*/
/*		if((c == 'D'-0x40) && (n2 > n))
		{
			putcon(cp[n++]);	// edit point char 
		}*/
		if(((c=='\10') || (c==0x7f))
		&& (n))
		{
			if(n == n2)
			{
				cp[n--]='\0';
				n2--;
				putcon(0x8);	// back space code 
				putcon(0x20);	// space code 
				putcon(0x8);	// back space code 
			}
/*			else if((n2 > n) && (n))
			{
				n--;
				putcon(0x8);	/// back space code 
				for(i=n;i<n2-1;i++)
				{
					c2 = cp[i+1];
					cp[i] = c2;
					putcon(c2);
				}
				cp[--n2]=0;
				putcon(' ');
				for(i=n;i<n2+1;i++)
					putcon(0x8);
			}*/
		}
/*		if(c=='A'-0x40)
		{
			while(n--)
			{
				putcon(0x8);
			}
			n=0;
		}
		if(c=='F'-0x40)
		{
			for(;n<n2;n++)
			{
				putcon(cp[n]);
			}
			n=n2;
		}
		if(c=='X'-0x40)	//  ワンキーモニタのオプション 
		{
			return -2;
		}
		if(c=='E'-0x40)	//  ワンキーモニタのオプション 
		{
			return -3;
		}
		if(c=='R'-0x40)	//  ワンキーモニタのオプション 
		{
			return -4;
		}*/
//		if(c=='"')f=0;
//		if(c=='I'-0x40)c=' ';
		if((f==1) && (c=='r'))
		{
//			return('r' | 0x8000);
			return('r');
		}
		if((f==1) && (c=='n'))
		{
//			return('n' | 0x8000);
			return('n');
		}
		
		if(((c & 0x7f) >= 0x20) && ((c & 0x7f) < 0x7f))
		{
			if(n == n2)
			{
				cp[n++]=c;
				s=c;
				putcon(s);
				if(n > n2)n2=n;
			}
/*			else if(n2 > n)
			{
				if(insf)
				{
					for(i=n2-1;i>=n;i--)
					{
						cp[i+1]= cp[i];
					}
					cp[n] = c;
					for(i=n;i<n2+1;i++)
					{
						putcon(cp[i]);
					}
					for(i=n;i<n2;i++)
					{
						putcon(0x8);
					}
					n++;
					n2++;
				}
				else
				{
					putcon(c);
					cp[n++] = c;
				}
			}*/
		}
	}
	cp[n]='\0';
	return(n2);
}
// コマンドの引数の個数チェックと、ヘキサ文字をバイナリに変換。
int argck(char **cp,union param32 *param,int n)
{
//	char *sp;
	char c;
	short count;
/*	long l;*/

//	sp= *cp;
	count=0;
	while(1)
	{
		c=htol(&param[count++].l32,cp);
		if((c==' ') || (c=='\0'))
		{
			if(count>=n)
			{
				return(count);
			}
			else
			{
				return(-1);
			}
		}
		else
		{
			if(c != ',')
			return(-1);
		}
	}
}
void paramer(void)
{
	putscon("\r\nPARAMETER ERROR !\r\n");
}
/*
void hex8b(long l)
{
	putcon(hex(l>>4));
	putcon(hex(l));
}
void hex16b(long l)
{
	putcon(hex(l>>12));
	putcon(hex(l>>8));
	putcon(hex(l>>4));
	putcon(hex(l));
}
void hex32b(long l)
{
	hex16b(l>>16);
	hex16b(l);
}
*/
// argck で、ヘキサ文字からバイナリに変換。
char htol(long *lp,char **cp)
{
	char c;
	char *p;
	long l;
	short s;

	p= *cp;
	l=0l;
	while(1)
	{
		c= *p++;
//			hex32b(c);
		s = chex(c);
//			hex32b(s);
		if(s== -1)
		{
			*lp = l;
			*cp = p;
			return(c);
		}
		l <<= 4;
		l |= s;
	}
}
// メモリ間のコピー。バイト、ワード、ロングワードの指定ができる。
void movem(union param32 *param,long ll)
{
	register long l;
	register char *s,*d;
	register short *sp;
	long *lp,*ldp;
	short *dp;

	s=param[0].cpoint;
	sp=param[0].spoint;
	d=param[1].cpoint;
	dp=param[1].spoint;
	lp=param[0].lpoint;
	ldp=param[1].lpoint;
	l=param[2].l32;

	if(ll==0)
	{
		while(l--)
		{
			*d++ = *s++;
		}
	}
	else if(ll==1)
	{
		while(l--)
		{
			*dp++ = *sp++;
		}
	}
	else if(ll==2)
	{
		while(l--)
		{
			*ldp++ = *lp++;
		}
	}
}
// 指定したバイト列１～４バイトのデータが、メモリにあるかを調べる。
void findp(union param32 *param)
{
	register long bytec;
	register long l;
	register char *s,*d;
/*	register char c1,c2;*/
	register short count;
	short shot;
	long ld;
	short lfn;

	s=param[0].cpoint;
	d=param[1].cpoint;
	bytec=param[2].l32;
	l=param[3].l32;

	if(bytec==0)return;
	if(bytec==1)l &= 0xffl;
	if(bytec==2)l &= 0xffffl;
	if(bytec==3)l &= 0xffffffl;
	lfn=0;
	crlf();
	while(s != d)
	{
		ld=0;
		for(count=0;count<bytec;count++)
		{
			ld <<= 8;
			shot= s[count];	shot &= 0xff;
			ld |= shot;
		}
		if(ld==l)
		{
			cprintf("%08lX ",(long)s);
//			spacen(1);
			lfn++;
			if(lfn >=8)
			{
				lfn=0;
				crlf();
			}
			if(statcon())
			{
				if(getcon()==3)return;
				if(getcon()==3)return;
			}
		}
		s++;
	}
}
void dumpmt(void)
{
	putscon("\r\nADDR      0 1  2 3  4 5  6 7  8 9  A B  C D  E F  ascii\r\n");
}
void dumpm(union param32 *param,int mode)
{
	short *sp;
	short s,i;
	union param32 u;
	short sbuf[8];
	long l,n;

	param[0].l32 &= 0xfffffff0l;
	param[1].l32 &= 0xffffffffl;
	sp=param[0].spoint;
	if(mode == 0x10)
	{
		n = param[1].l32 - param[0].l32;
		n /= 4;
		while(n--)
		{
			l = *param[0].lpoint++;
			cprintf("%08lX\r\n",l);
			if(param[0].l32 >= param[1].l32)
				return;
		}
	}
	dumpmt();
	while(1)
	{

	cprintf("%08lX ",(long)sp);//	spacen(1);
//	hex32b((long)sp);	spacen(1);
	for(i=0;i<8;i++)
	{
		s= *sp++;	sbuf[i]=s;
		cprintf("%04X ",swapw(s));//	spacen(1);
//		hex16b(s);	spacen(1);
		if(statcon())
		{
			if(getcon()==3)return;
			if(getcon()==3)return;
		}
	}
	putcon(0x2a);	/*   '*'   */
	for(i=0;i<8;i++)
	{
		s=sbuf[i];
		ckasci(s);
		ckasci(s >> 8);
	}
	putcon(0x2a);
	if(sp >= param[1].spoint)
	{
		return;
	}
	u.spoint=sp;
	if((u.l32 & 0xf)==0)
	{
		crlf();
		if((u.l32 & 0xff)==0)
		dumpmt();
	}

	}
}
// セットメモリ。
//	argc == 1 で、インタラクティブにメモリを表示、変更。
//	argc >= 2 で、フィル機能、データを変更しながら書く機能などがある。
int setmemo(union param32 *param,int wf,int argc)
{
	char *cp;
	short *sp;
	long *lp;
//	short s,j;
/*	union param32 u;*/
	char buf[16];
	long l,i,updatea,updated;//,repeat;
	char *c;
	int wof;

	wof=0;
	if(wf & 4)
	{
		wof=1;
		wf &= 3;
	}
	if(wf==1)
	{
		param[0].l32 &= 0xfffffffel;
		sp=param[0].spoint;
	}
	else if(wf==2)
	{
/*		param[0].l32 &= 0xfffffffcl;*/
		lp=param[0].lpoint;
	}
	else
	{
		cp=param[0].cpoint;
	}
	if(argc >= 2)
	{
//	param[2] count	param[3] update  param[4] address segment  param[5] adjust address update  param[6] data update
//	default		param[2].l32 = 1;
//			param[3].l32 = 0;
//			param[4].l32 = 0x0;
//			param[5].l32 = 0;
//			param[6].l32 = 0;
		updatea = param[4].l32;
		updated = param[1].l32;
		while(param[2].l32--)
		{
			switch(wf)
			{
			case	0:	*cp++ = (char)updated;	break;
			case	1:	*sp++ = (short)updated;	break;
			case	2:	*lp++ = (long)updated;	break;
			}
			updated = updated + param[3].l32;
			updatea--;
			if(updatea == 0)
			{
				updatea = param[4].l32;
				cp = &cp[param[5].l32];
				sp = &sp[param[5].l32];
				lp = &lp[param[5].l32];
				param[1].l32 += param[6].l32;
				updated = param[1].l32;
			}
		}
	}
	if(argc == 1)
	while(1)
	{
		crlf();
		if(wf==1)
		{
			cprintf("%08lX",(long)sp);
//			hex32b((long)sp);
		}
		else if(wf==2)
		{
			cprintf("%08lX",(long)lp);
//			hex32b((long)lp);
		}
		else
		{
			cprintf("%08lX",(long)cp);
//			hex32b((long)cp);
		}
		putcon('=');
		if(wof==0)
		{
			if(wf==1)
			{
				cprintf("%04X",*sp);
//				hex16b(*sp);
			}
			else if(wf==2)
			{
				cprintf("%08lX",*lp);
			}
			else
			{
				cprintf("%02X",(short)*cp & 0xff);
			}
		}
		putcon(' ');
		i=getscon(buf,15,1);
		if(i == -1)return(0);
		if(i>15)
		{
			if((i=='R')||(i=='r'))
			{
				if(wf==1)
				{
					sp--;
				}
				else if(wf==2)
				{
					lp--;
				}
				else
				{
					cp--;
				}
			}
			else if(((i=='X')||(i=='x'))&&(wof==1))
			{
				if(wf==1)
				{
					cprintf("%04X",*sp);
				}
				else if(wf==2)
				{
					cprintf("%08lX",*lp);
				}
				else
				{
					cprintf("%02X",(short)*cp & 0xff);
				}
				putcon(' ');
			}
			else	/* 'N'	*/
			{
				if(wf==1)
				{
					sp++;
				}
				if(wf==2)
				{
					lp++;
				}
				else
				{
					cp++;
				}
			}
		}
		else
		{
			c=buf;
			if(*c == '.')break;
			htol(&l,&c);
			i=l;
			if(wf==1)
			{
				*sp++ = i;
			}
			else if(wf==2)
			{
				*lp++ = i;
			}
			else
			{
				*cp++ = i;
			}
		}
	}
	return 0;
}
int verify(union param32 *param)
{
	register long l;
	register char *s,*d;
	register char c1,c2;
	short shot;

	s=param[0].cpoint;
	d=param[1].cpoint;
	l=param[2].l32;

	while(l--)
	{
		c1= *s;
		c2= *d;
		if(c1 != c2)
		{
			crlf();
			cprintf("%08lX=",(long)s);
			shot=c1;
			cprintf("%02X ",shot & 0xff);
			cprintf("%08lX=",(long)d);
			shot=c2;
			cprintf("%02X",shot & 0xff);
			if(statcon())
			{
				if(getcon()==3)return(0);
				if(getcon()==3)return(0);
			}
		}
		s++;	d++;
	}
	return(0);
}
// strcmp もどき。
int cmpstr(char *d,char *s)
{
	register long l;
	register char c1,c2;
	l=250l;
	while(l--)
	{
		c1= *s++;
		c2= *d++;
		if(c1 > c2)
		{
			return(1);
		}
		if(c1 < c2)
		{
			return(-1);
		}
		if((c1 == 0) && (c2 == 0))
		{
			return(0);
		}
		if((c1 == 0) || (c2 == 0))
		break;
	}
	return(1);
}
void srload(union param32 *param,int m)
{
/*	extern short ring0v;
	extern short ring0r;
*/
	char *cp,*cpb;
//	char *rp;
	register long l;
	register int i,j,k,sm;
	int ii,flag;
	long len;

//	char (*func)();
	len=0;	flag=0;
//	if(m==1)
//		func = (char (*)())getcon;
//	len=0;	flag=0;

SRLDL:
	if((i=getcon()) ==3)	goto SRLDE;

	if(i != 'S')goto SRLDL;
	if((i=getcon())==3)	goto SRLDE;

	if(i == '9')		goto SRLDE;
	if(i == '8')		goto SRLDE;
	if(i == '7')		goto SRLDE;
	if(i == '0')		goto SRLDL;
	if(i == '5')		goto SRLDL;
	if((i >= '1') && (i <= '3'))
	{
		j=(i & 3)+1;
		/*	dly(10);*/
		if((i=getcon()) ==3)	goto SRLDE;
		k=chex(i);
		/*	dly(10);*/
		if((i=getcon()) ==3)	goto SRLDE;
		i=chex(i);
		k= k*16 + i;
		if(k==0)
		{
			i='9';	goto SRLDE;
		}
		sm = 0xFF;
		sm -= k;	k--;
		l=0;
		while(j--)
		{
		/*	dly(10);*/
			if((i=getcon()) ==3)	goto SRLDE;
			ii=chex(i);
		/*	dly(10);*/
			if((i=getcon()) ==3)	goto SRLDE;
			ii = ii*16 +chex(i);
			sm -= ii;
			l = (l << 8) + ii;
			k--;
		}
		cp = param[0].cpoint;
		cp = &cp[l];
		if(flag==0)
		{
			flag=1;
			cpb=cp;
		/*	putscon("start loading address = ");*/
		/*	hex32b((long)cp);*/
		/*	crlf();	*/
		}
		while(k--)
		{
		/*	dly(10);*/
			if((i=getcon()) ==3)	goto SRLDE;
			ii=chex(i);
		/*	dly(10);*/
			if((i=getcon()) ==3)	goto SRLDE;
			ii = ii*16 + chex(i);
			sm -= ii;
			*cp++ = ii;
			len++;
		}
		/*	dly(10);*/
		if((i=getcon()) ==3)	goto SRLDE;
		ii=chex(i);
		/*	dly(10);*/
		if((i=getcon()) ==3)	goto SRLDE;
		ii = ii*16+ chex(i);
		if((sm & 0xFF) != (ii & 0xff))
		{
			putscon("check sum error !!  ");
			cprintf("%04X %04X",(short)sm,(short)ii);//cprintf(" %04X",(short)ii);
//			hex16b((short)ii);
		/*	printf("sm=%x ii=%x",sm,ii);*/
			i = -2;	goto SRLDE;
		}
		goto SRLDL;
	}
	else
	{
		putscon("S1~S3 abarble\r\n");
		return;
	}
SRLDE:
/*	s0icls();*/
	if(i == -1)
	{
//		putscon("BREAK BY CONSOLE !!\r\n");
		return;
	}
	if(i == -2)
	{
		putscon("SUM CHECK ERROR\r\n");
		return;
	}
	if((i == '9')||(i == '8')||(i == '7'))
	{
		for(j=28-(i & 0xf)*2;j>0;j--)
			getcon();
		putscon("start loading address = ");
		cprintf("%08lX",(long)cpb);
//		crlf();
		putscon("\r\nEND.     bytes loaded = ");
		cprintf("%08lX\r\n",len);
//		crlf();
		return;
	}
	putscon("????? S load stop\r\n");
	return;
}
// strcpy もどき
long cpysstr(char *d,char *s,long n)
{
	register char c;
	register long l;
	l=0;
	while(n--)
	{
		c= *s++;
		if((c == ' ') || (c == '\0'))break;
		*d++ = c;
		l++;
	}
	*d++ = '\0'; 
	return(l);
}
void spacen(int n)
{
	while(n--)
	{
		putcon(0x20);
	}
}
void crlf()
{
	putcon('\r');
	putcon('\n');
}
int chex(char c)
{
//	hex32b(0);
	if((c >= 'a') && (c <= 'z'))c -= ' ';
	if((c >= '0') && (c <= '9'))return(c & 0xf);
	if((c >= 'A') && (c <= 'F'))return(c -= 0x37);
	return(-1);
}
int hex(int i)
{
	i &= 0xF;
	if(i>9)
	i+=7;
	i+=0x30;
	return(i);
}
void ckasci(short s)
{
	s &= 0xff;
	if(((s >= 0x20) && ( s <= 0x7e)))
//	|| ((s >= 0xa0) && ( s <= 0xfe)))
	{
		putcon(s);
	}
	else
	{
		putcon(0x2e);
	}
}
// strlen もどき
long cstrlen(char *cp,char f)
{
	long i;

	for(i=0;i<COLUMN;i++)
	{
		if(*cp++ ==f)
			return i;
	}
	return i;
}
// printf もどき
// printf 関数の整数版 引数は２個まで。printf lite. Custom printf
// %X 
void cprintf(char *fmt,unsigned long a,unsigned long b /* ,unsigned long c,unsigned long d*/ )
{
	char bf[COLUMN];

	int len;
	len = csprintf(bf,fmt,a,b /* ,c,d */);
	bf[len]=0;
	putscon(bf);
}
// 引数を３２ビットまで有効にするには、%lx などと、'l' or 'L' を付加する。無いと、下位１６ビットが有効。
int csprintf(char *dp,char *fmt,unsigned long a,unsigned long b /* ,unsigned long c,unsigned long d*/ )
{
	char c;
	int len,i,j,keta,zs,siz;
	int wlen,hiki;
	char cb[12];
	int s,lower;
	unsigned long l;
	char *cp;

//	hex32b((long)cb);
//	hex32b((long)dp);
	len = cstrlen(fmt,(char)0);
	wlen=0;
	hiki=0;
	for(i=0;i<len;i++)
	{
		c = fmt[i];
		if(c == '%')
		{
			zs=1;
			keta=1;
			siz=2;
			if(hiki==0)
				l=a;
			else
				l=b;
/*
			switch(hiki)
			{
			case	0:	l=a;	break;
			case	1:	l=b;	break;
			case	2:	l=c;	break;
			case	3:	l=d;	break;
			default:	l=a;	break;
			}
*/
			hiki++;
			c = fmt[++i];
			if((c >= '0') && (c <= '9'))
			{
				if(c=='0')
				{
					zs=0;	i++;
					c = fmt[i];
				}
				if((c >= '1') && (c <= '9'))
				{
					keta=c & 0xf;
					i++;
					c = fmt[i];
					if((c >= '0') && (c <= '9'))
					{
						keta *= 10;
						keta += c & 0xf;
						i++;
						c = fmt[i];
					}
				}
				else
					zs=1;
			}
			if((c == 'l') || (c == 'L'))
			{
				siz=4;
				i++;
				c = fmt[i];
			}
			if((c == 'x') || (c == 'X'))
			{
				if(c == 'x')
					lower = 1;
				else
					lower = 0;
				c='x';
			}
			else if((c == 'd') || (c == 'D'))
			{
				c='d';
			}
			if(c == 'c')
			{
				*dp++ = l;
				wlen++;
			}
			else if(c == 's')
			{
				cp = (char *)l;
				while(*cp)
				{
					*dp++ = *cp++;
					wlen++;
					if(wlen >= (COLUMN -1))break;
				}
			}
			else
			{
				if(siz==2)
					l &= 0xFFFF;
				if(keta>10)keta=10;
				cb[10]=0;
				if(c == 'x')
				{
					cb[0]=' ';
					cb[1]=' ';
					for(j=9;j>=2;j--)
					{
						s = hex((int)l);
						if((lower) && (s > '9'))
							s |= ' ';// 英小文字
						cb[j]=s;
						l >>= 4;
					}
				}
				else
				{
					for(j=9;j>=0;j--)
					{
						s = (unsigned long)l%10;
						cb[j]=s | '0';
						l = (unsigned long)l/10;
					}
				}
				for(j=0;j<9;j++)
				{
					c = cb[j];
					if(c == '0')
					{
						if(zs)
							cb[j]=' ';
					}
					else if(c != ' ')
						break;
				}
				if(keta < (10-j))
					keta=10-j;
				if((wlen + keta) >= (COLUMN -1))
					return wlen;
				for(j=10-keta;j<10;j++)
					*dp++ = cb[j];
				wlen += keta;
			}
		}
		else
		{
			*dp++ = c;
			wlen++;
		}
		if(wlen >= (COLUMN -1))return wlen;
	}
	return wlen;
}
