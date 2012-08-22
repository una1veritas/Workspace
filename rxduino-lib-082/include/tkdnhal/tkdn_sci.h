/*******************************************************************************
* RXduinoライブラリ & 特電HAL
* 
* このソフトウェアは特殊電子回路株式会社によって開発されたものであり、当社製品の
* サポートとして提供されます。このライブラリは当社製品および当社がライセンスした
* 製品に対して使用することができます。
* このソフトウェアはあるがままの状態で提供され、内容および動作についての保障はあ
* りません。弊社はファイルの内容および実行結果についていかなる責任も負いません。
* お客様は、お客様の製品開発のために当ソフトウェアのソースコードを自由に参照し、
* 引用していただくことができます。
* このファイルを単体で第三者へ開示・再配布・貸与・譲渡することはできません。
* コンパイル・リンク後のオブジェクトファイル(ELF ファイルまたはMOT,SRECファイル)
* であって、デバッグ情報が削除されている場合は第三者に再配布することができます。
* (C) Copyright 2011-2012 TokushuDenshiKairo Inc. 特殊電子回路株式会社
* http://rx.tokudenkairo.co.jp/
*******************************************************************************/

#ifndef __H_TKDN_SCI
#define __H_TKDN_SCI

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SCI_TXINTERRUPT_NOTUSE  0
#define SCI_TXINTERRUPT_USE     1

//SCIの動作ポート定数
typedef enum {
	SCI_NONE,     // SCIを使わない
	SCI_AUTO,     // SCIを自動選択
	SCI_USB0,     // USB0 の仮想COMポートを使う 
	SCI_USB1,     // USB1 の仮想COMポートを使う （未実装）
	SCI_SCI0P2x,  // SCI0 (ポートP20,P21と兼用) を使う
	SCI_SCI1JTAG, // SCI1 (ポートPF0,PF2,JTAGと兼用) を使う 
	SCI_SCI2A,    // SCI2A (ポートP13,P12)を使う  
	SCI_SCI2B,    // SCI2B (ポートP50,P52)を使う  
	SCI_SCI6A,    // SCI6A (ポートP00,P01)を使う  
	SCI_SCI6B,    // SCI6B (ポートP32,P33)を使う  
} SCI_PORT;

//CRLFMODE
typedef enum {
	CRLF_NONE,
	CRLF_CR,
	CRLF_CRLF,
} CRLFMODE;


//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所

#define SCI_BUFSIZE  256          // 受信・送信があるので実際にはこの2倍のメモリを使う
                                  // 2^nの値を指定すること

// SCI TX割り込みを使うならここを書き換える
#define SCI_TXINTERRUPT_MODE    SCI_TXINTERRUPT_USE

//  カスタマイズはここまで
//■■■■■■■■■■■■■■■■■■■■■■■■■

typedef struct sci_str
{
	// 送受信バッファ
	char rxbuf[SCI_BUFSIZE];
	char txbuf[SCI_BUFSIZE];
	
	// 送受信バッファのポインタ
	int     tx_rptr;
	int     tx_wptr;
	int     rx_rptr;
	int     rx_wptr;
	
	SCI_PORT port;
	CRLFMODE crlf_tx;
	CRLFMODE crlf_rx;
} sci_str;

typedef struct sci_str sci_t; // SCIハンドラ(拡張版SCIルーチンで使用する)

//各種関数

// SCIポートを初期化する
//  portに使いたいポートを指定する
//  SCI_NONEを指定すると自動的に判別する
//  実際に開かれたポートを戻り値として返す

TKDN_HAL
SCI_PORT sci_init(SCI_PORT _port,int bps);

// SCIの使用ポートを取得
TKDN_HAL
SCI_PORT sci_getport(void);

// １文字送信
//   ※失敗(送信バッファFULL)したら0を返す
TKDN_HAL
int sci_putc(char c);

// 文字列送信
//   ※失敗(送信バッファFULL)したら0を返す
TKDN_HAL
void sci_puts(const char *str);

// １文字受信
//  ※受信した文字がなければ\0を返す
TKDN_HAL
char sci_getc(void);

// 文字列受信
// 最大max文字受信する
// 最後にヌルターミネータをつけるが、最大文字数に達した場合はヌルターミネータはつけない
TKDN_HAL
char *sci_gets(char *s,int max);

// バイナリデータ送信
TKDN_HAL
void sci_writedata(const unsigned char *data,int len);

// バイナリデータ受信
//  ※指定されたデータ量を受信するまで制御を返さない
TKDN_HAL
void sci_readdata(unsigned char *data,int len);

// 受信バッファに溜まっているデータ数を調べる
TKDN_HAL
int  sci_rxcount(void);

// CRやLFの変換をSCIライブラリ内で行う
TKDN_HAL
void sci_convert_crlf(CRLFMODE tx,CRLFMODE rx);

// データを覗き見る
TKDN_HAL
char sci_peek(void);

// 送信データをフラッシュする
TKDN_HAL
void sci_flush(void);


// 拡張版SCIルーチン

// SCIの初期化
TKDN_HAL
void sci_init_ex(sci_t *sci,SCI_PORT _port,int bps);

// SCIの使用ポートを取得
TKDN_HAL
SCI_PORT sci_getport_ex(sci_t *sci);

// １文字送信
//   ※失敗(送信バッファFULL)したら0を返す
TKDN_HAL
int sci_putc_ex(sci_t *sci,char c);

// 文字列送信
//   ※失敗(送信バッファFULL)したら0を返す
TKDN_HAL
void sci_puts_ex(sci_t *sci,const char *str);

// １文字受信
//  ※受信した文字がなければ\0を返す
TKDN_HAL
char sci_getc_ex(sci_t *sci);

// 文字列受信
// 最大max文字受信する
// 最後にヌルターミネータをつけるが、最大文字数に達した場合はヌルターミネータはつけない
TKDN_HAL
char *sci_gets_ex(sci_t *sci,char *s,int max);

// バイナリデータ送信
TKDN_HAL
void sci_writedata_ex(sci_t *sci,const unsigned char *data,int len);

// バイナリデータ受信
//  ※指定されたデータ量を受信するまで制御を返さない
TKDN_HAL
void sci_readdata_ex(sci_t *sci,unsigned char *data,int len);

// 受信バッファに溜まっているデータ数を調べる
TKDN_HAL
int  sci_rxcount_ex(sci_t *sci);

// CRやLFの変換をSCIライブラリ内で行う
TKDN_HAL
void sci_convert_crlf_ex(sci_t *sci,CRLFMODE tx,CRLFMODE rx);

// データを覗き見る
TKDN_HAL
char sci_peek_ex(sci_t *sci);

// 送信データをフラッシュする
TKDN_HAL
void sci_flush_ex(sci_t *sci);

// 
TKDN_HAL
void sci_regist_default(sci_t *sci);

#ifdef __cplusplus
}
#endif

#endif // __H_TKDN_SCI

