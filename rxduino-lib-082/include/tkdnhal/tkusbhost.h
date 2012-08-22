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

#ifndef TKUSBHOST_H
#define TKUSBHOST_H

#define TIMEOUT         1000  //1000ms
#define TIMEOUT_INFINITE -1

#define DEVICE_DESCRIPTOR_TYPE    0x01
#define CONFIG_DESCRIPTOR_TYPE    0x02
#define STRING_DESCRIPTOR_TYPE    0x03
#define INTERFACE_DESCRIPTOR_TYPE 0x04
#define ENDPOINT_DESCRIPTOR_TYPE  0x05

typedef struct DeviceDesc_t
{
	unsigned char  bLength;
	unsigned char  bDescriptorType;
	unsigned short bcdUSB;
	unsigned char  bDeviceClass;
	unsigned char  bDeviceSubClass;
	unsigned char  bDeviceProtocol;
	unsigned char  bMaxPacketSize0;
	unsigned short idVendor;
	unsigned short idProduct;
	unsigned short bcdDevice;
	unsigned char  iManufacture;
	unsigned char  iProduct;
	unsigned char  iSerialNumber;
	unsigned char  bNumConfigurations;
} DeviceDesc_t;

typedef struct ConfigDesc_t
{
	unsigned char  bLength;
	unsigned char  bDescriptorType;
	unsigned short wTotalLength;
	unsigned char  bNumInterfaces;
	unsigned char  bConfigurationValue;
	unsigned char  iConfiguraion;
	unsigned char  bmAttributes;
	unsigned char  bMaxPower;
} ConfigDesc_t;

typedef struct EndpointDesc_t
{
    unsigned char  bLength;
    unsigned char  bDescriptorType;
    unsigned char  bEndpointAddress;
    unsigned char  bmAttributes;
    unsigned short wMaxPacketSize;
    unsigned char  bInterval;
    unsigned char  bRefresh;
    unsigned char  bSynchAddress;
} EndpointDesc_t;

typedef struct InterfaceDesc_t
{
    unsigned char  bLength;
    unsigned char  bDescriptorType;
    unsigned char  bInterfaceNumber;
    unsigned char  bAlternateSetting;
    unsigned char  bNumEndpoints;
    unsigned char  bInterfaceClass;
    unsigned char  bInterfaceSubClass;
    unsigned char  bInterfaceProtocol;
    unsigned char  iInterface;
} InterfaceDesc_t;

typedef struct USBHostInfo_t
{
	DeviceDesc_t    DeviceDesc;
	ConfigDesc_t    ConfigDesc;
	EndpointDesc_t  EndpointDesc;
	InterfaceDesc_t InterfaceDesc;
	unsigned short  LangId;
	unsigned char   FullSpeed; // 1:Fullspeed, 0:Lowspeed
	unsigned char   FlagAttach;
} USBHostInfo_t;

typedef enum TKUSBH_RESULT
{
	TKUSBH_OK      = 0, // エラーなし
	TKUSBH_NOSUPPORT,   // USBホストはサポートされていない
	TKUSBH_DISCONNECT,  // ターゲットが接続されていない
	TKUSBH_NOINIT,      // 初期化されていない
	TKUSBH_TIMEOUT,     // タイムアウト(NAK)
	TKUSBH_STALL,       // STALLが返された
	TKUSBH_ERROR,       // その他のエラー
} TKUSBH_RESULT;

extern USBHostInfo_t USBHostInfo;

/****************************************************************************************
                   USB Hostの関数
****************************************************************************************/

// 特電USBホストモジュールを初期化する。
// 成功するとTKUSBH_OKを返す。
// USBホストをサポートしていないボードではTKUSB_NOSUPPORTを返す
TKUSBH_RESULT tkusbh_init();

// ターゲットが接続されているかどうかを調べる
// 接続されていればTKUSBH_OKを返す。
// 接続されていなければTKUSBH_DISCONNECTを返す。
// USBホストをサポートしていないボードではTKUSB_NOSUPPORTを返す
TKUSBH_RESULT tkusbh_is_connected();

// ターゲットに接続する
// 引数のタイムアウトはms単位で指定する。-1を指定すると無限に待つ
//  (※タイムアウトは未実装)
// 成功したらTKUSBH_OKを返す。
// タイムアウトしたらTKUSBH_TIMEOUTを返す。
// 何らかの接続エラーが発生して失敗したらTKUSBH_ERRORを返す。
// USBホストをサポートしていないボードではTKUSB_NOSUPPORTを返す
TKUSBH_RESULT tkusbh_connect(int timeout_ms);

// ターゲットを切断する。切断されるまで待つ。
// タイムアウトはms単位で指定する。-1を指定すると無限に待つ
// 成功したらTKUSBH_DISCONNECTを返す。
// タイムアウトしたらTKUSBH_OKを返す。
// USBホストをサポートしていないボードではTKUSB_NOSUPPORTを返す
TKUSBH_RESULT tkusbh_disconnect(int timeout_ms);

// ディスクリプタを取得する
// 成功したら取得したディスクリプタの長さを返す
// 失敗(STALL)したら-1を返す
int  tkusbh_get_descriptor(unsigned char type,unsigned char index, void *buf, int size);

// ストリングディスクリプタを取得する
// 成功したら取得したディスクリプタの長さを返す
// 失敗(STALL)したら-1を返す
int  tkusbh_get_string(unsigned short index, unsigned short langid, char *buf,int buflen);

// コントロールトランザクションを発行する
// 成功したら、送受信したデータの長さを返す
// 失敗(STALL)したら-1を返す
int  tkusbh_control_msg(unsigned short req,unsigned short val,
                        unsigned short index,unsigned short len,
                        unsigned char *buf, int timeout);

// SET CONFIGURATIONを実行する
// 成功したら、TKUSBH_OKを返す
// 失敗(STALL)したらTKUSBH_ERRを返す
// USBホストをサポートしていないボードではTKUSB_NOSUPPORTを返す
TKUSBH_RESULT tkusbh_set_configuration(int configuration);

// バルク転送を実行
// 成功したら転送した長さを返す。
// 失敗したら-1を返す
// ※現在のバージョンの制約により、使用できるエンドポイントは送受信とも１つずつです
int  tkusbh_bulk_write(int ep, unsigned char *bytes, int size,int timeout);
int  tkusbh_bulk_read(int ep, unsigned char *bytes, int size,int timeout);

// 以下の関数はまだ作っていない
// int  tkusbh_clear_halt(unsigned int ep);
// int  tkusbh_resetep(unsigned int ep);
// int  tkusbh_reset();

/****************************************************************************************
                   USBホストユーティリティ
****************************************************************************************/

// ディスクリプタの表示
void dump(unsigned char *buf,int len);
void ShowDeviceDesc(DeviceDesc_t *desc,unsigned short LangId);
void ShowConfigDesc(ConfigDesc_t *desc);
void ShowInterfaceDesc(InterfaceDesc_t *desc);
void ShowEndpointDesc(EndpointDesc_t *desc);

#endif
