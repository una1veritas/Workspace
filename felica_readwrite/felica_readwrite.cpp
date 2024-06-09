#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <openssl/des.h>
#include <time.h>
#include <unistd.h>
#include "RCS620S.h"

#define KEY_NUM (3)

int connectRcs620s();
bool compareBuf(uint8_t val1[16], uint8_t val2[16]);
bool authFelica();
bool checkMac();
bool compareMac(uint8_t CARD_MAC_A[8], uint8_t RC1[8], uint8_t RC2[8], uint8_t BLOCK_HIGH[8], uint8_t BLOCK_LOW[8]);
bool readIdWithMacA(uint8_t macBlock[16]);
bool writeWithoutEncryption(uint16_t serviceCode, uint8_t blockNumber, uint8_t felicaBlock[16]);
bool readWithoutEncryption(uint8_t blockNumber, uint8_t felicaBlock[16]);
void arryXor(const uint8_t input1[8], const uint8_t input2[8], uint8_t output[8]);
void tripleDES(uint8_t input[8], uint8_t XorInput[8], uint8_t input_key1[8], uint8_t input_key2[8], uint8_t input_key3[8], uint8_t output[8]);
void swapByteOrder(uint8_t inout[8]);
bool getCK(uint8_t CK1[8], uint8_t CK2[8]);
bool generatePersonalizedCK(uint8_t CK[16]);
bool issuanceFelica();

//定数
const uint16_t ADDRESS_RC = 0x80;
const uint16_t ADDRESS_ID = 0x82;
const uint16_t ADDRESS_CKV = 0x86;
const uint16_t ADDRESS_CK = 0x87;
const uint16_t ADDRESS_MAC_A = 0x91;
const uint16_t READ_WRITE_MODE = 0x0009;
const uint16_t READ_ONLY_MODE = 0x000b;

//個別化マスター鍵
//最上位ビット1にしたい場合
uint8_t  MK1[8] = {0x00, 0x21, 0xFF, 0xA4, 0x87, 0x2F, 0x1C, 0x4B };
uint8_t  MK2[8] = {0x15, 0x9C, 0xF4, 0x14, 0xC4, 0x4E, 0x98, 0xB4 };
uint8_t  MK3[8] = {0x7C, 0x79, 0x99, 0xA0, 0x31, 0x96, 0x6B, 0xE6 };

//最上位ビット0にしたい場合
/*
uint8_t  MK1[8] = {0x00, 0x01, 0xFF, 0xA4, 0x87, 0x2F, 0x1C, 0x4B };
uint8_t  MK2[8] = {0x7C, 0x79, 0x99, 0xA0, 0x31, 0x96, 0x6B, 0xE6 };
uint8_t  MK3[8] = {0x15, 0x9C, 0xF4, 0x14, 0xC4, 0x4E, 0x98, 0xB4 };
*/

//FeliCa リーダー・ライター
RCS620S rcs620s;

/*
値の比較
*/
bool compareBuf(uint8_t val1[16], uint8_t val2[16])
{  
  printf("compareBuf start\n");
  bool rtn;
  rtn = true;
  for(int i=0; i<16; i++)
  {
    printf("%02X %02X\n", val1[i], val2[i]);
    if(val1[i] != val2[i]) rtn = false;
  }
  printf("compareBuf finish\n");
  
  return rtn;
}

/*
カードリーダに接続
*/
int connectRcs620s()
{
  //カードに接続
  int ret = rcs620s.initDevice();
  if (!ret) { 
    printf("カード接続失敗\n");
    return false; 
  }

  ret = rcs620s.polling();
  if (!ret) { 
    printf("polling失敗\n");
    return false; 
  }
  return true;
}

/*
カードの正当性チェック(単独呼び出し用)
*/
bool authFelica()
{
  int ret = connectRcs620s(); //カード接続
  ret = checkMac();
  rcs620s.rfOff();  //カード接続断

  if(ret==0)
  {
    return true;
  }else{
    return false;  
  }
  
}

/*
カードの正当性チェック
戻り値：
カードが正当：0
カードが不正：-1
カード接続失敗：-2
*/
bool checkMac()
{
  int ret;
  uint8_t RC[16];

  /* RC用に1～100の擬似乱数を16個生成 */
  srand((unsigned) time(NULL));
  for (int i=1; i<=16; i++) {
    RC[i] = (uint8_t)rand()%100+1;
  }

  uint8_t RC1[8];
  uint8_t RC2[8];
  for (int i = 0; i < 8; i++)
  {
    RC1[i] = RC[i];
    RC2[i] = RC[8+i];
  }

  //RCを書き込む
  if (!writeWithoutEncryption(READ_WRITE_MODE, ADDRESS_RC, RC)) { 
    printf("writeWithoutEncryption失敗\n");
    return -2; 
  }

  //IDとMAC_Aを読み出す
  uint8_t macBlock[44];
  if (!readIdWithMacA(macBlock)) { 
    printf("readIdWithMacA失敗\n");
    return -2; 
  }

  uint8_t BLOCK_HIGH[8], BLOCK_LOW[8];
  for (int i = 12; i < 28; i++)
  {
    BLOCK_HIGH[i - 12] = macBlock[8 + i];
    BLOCK_LOW[i - 12] = macBlock[0 + i];
  }

  uint8_t CARD_MAC_A[8];
  for (int i = 28; i < 36; i++)
  {
    CARD_MAC_A[i - 28] = macBlock[i];
  }

  //MACの比較
  if(compareMac(CARD_MAC_A, RC1, RC2, BLOCK_HIGH, BLOCK_LOW))
  {
    return 0;
  }else{
    return -1;
  }
}

/*
MACの比較
*/
bool compareMac(uint8_t CARD_MAC_A[8], uint8_t RC1[8], uint8_t RC2[8], uint8_t BLOCK_HIGH[8], uint8_t BLOCK_LOW[8])
{
  uint8_t ZERO[8] = { 0 };
  uint8_t IV1[8];
  uint8_t IV2[8];
  uint8_t SK1[8];
  uint8_t SK2[8];

  uint8_t CK1[8];
  uint8_t CK2[8];
  getCK(CK1, CK2);

  //SK1を生成
  tripleDES(RC1, ZERO, CK1, CK2, CK1, IV1);
  for (int i = 0; i < 8; i++)
  {
    SK1[i] = IV1[i];
  }
  swapByteOrder(SK1);

  //SK2を生成
  tripleDES(RC2, IV1, CK1, CK2, CK1, IV2);
  for (int i = 0; i < 8; i++)
  {
    SK2[i] = IV2[i];
  }
  swapByteOrder(SK2);

  //MAC_Aを生成
  //uint16_t ADDRESS_ID = 0x82;
  //uint16_t ADDRESS_MAC_A = 0x91;

  uint8_t OUT_1[8];
  uint8_t OUT_2[8];
  uint8_t CALC_MAC_A[8];

  uint8_t BlockInfo[8] = { (uint8_t)(ADDRESS_ID & 0xFF),(uint8_t)((ADDRESS_ID >> 8) & 0xFF),(uint8_t)(ADDRESS_MAC_A & 0xFF),(uint8_t)((ADDRESS_MAC_A >> 8) & 0xFF),0xFF,0xFF,0xFF,0xFF };

  swapByteOrder(BlockInfo);
  //swapByteOrder(BLOCK_LOW);
  //swapByteOrder(BLOCK_HIGH);

  tripleDES(RC1, BlockInfo, SK1, SK2, SK1, OUT_1);
  tripleDES(BLOCK_LOW, OUT_1, SK1, SK2, SK1, OUT_2);
  tripleDES(BLOCK_HIGH, OUT_2, SK1, SK2, SK1, CALC_MAC_A);

  swapByteOrder(CALC_MAC_A);

  fprintf(stdout, "生成したMAC_A [");
  for (int i = 0; i < 8; i++)
  {
    printf("%02X ", CALC_MAC_A[i]);
  }
  printf("]\n");


  //内部認証（カードのMAC_Aと生成したMAC_Aを比較）
  fprintf(stdout, "カードのMAC_A [");
  for (int i = 0; i < 8; i++)
  {
    printf("%02X ", CARD_MAC_A[i]);
  }
  printf("]\n");

  for(int i=0; i<8; i++)
  {
    if(CALC_MAC_A[i] != CARD_MAC_A[i])
    {
      printf("MAC不一致\n");
      return false;
    }
  }
  printf("MAC一致\n");

  return true;
}


/*
IDをMAC付で読み出す
*/
bool readIdWithMacA(uint8_t felicaBlock[16]) {
  int ret;
  uint8_t buf[RCS620S_MAX_CARD_RESPONSE_LEN];
  uint8_t responseLen = 0;
  uint16_t serviceCode = 0x000b;

  buf[0] = 0x06;
  memcpy(buf + 1, rcs620s.idm, 8);
  buf[9] = 0x01;      // サービス数
  buf[10] = (uint8_t)((serviceCode >> 0) & 0xff);
  buf[11] = (uint8_t)((serviceCode >> 8) & 0xff);
  buf[12] = 0x02;     // ブロック数
  buf[13] = 0x80;
  buf[14] = 0x82;
  buf[15] = 0x80;
  buf[16] = 0x91;

  ret = rcs620s.cardCommand(buf, 17, buf, &responseLen);
  if (!ret) {
    return false;
  }

  printf("ID, MAC_A Block : ");
  for(int i=0; i<44; i++)
  {
    felicaBlock[i] = buf[i];
    printf("%02X ", buf[i]);
  }
  printf("\n");

  return true;
}

/*
Read Without Encryption
*/
bool readWithoutEncryption(uint8_t blockNumber, uint8_t felicaBlock[16]) {
  int ret;
  uint8_t buf[RCS620S_MAX_CARD_RESPONSE_LEN];
  uint8_t responseLen = 0;
  uint16_t serviceCode = 0x0009;

  buf[0] = 0x06;
  memcpy(buf + 1, rcs620s.idm, 8);
  buf[9] = 0x01;      // サービス数
  buf[10] = (uint8_t)((serviceCode >> 0) & 0xff);
  buf[11] = (uint8_t)((serviceCode >> 8) & 0xff);
  buf[12] = 0x01;     // ブロック数
  buf[13] = 0x80;
  buf[14] = blockNumber;

  ret = rcs620s.cardCommand(buf, 15, buf, &responseLen);

  if (!ret || (responseLen != 28) || (buf[0] != 0x07) ||
      (memcmp(buf + 1, rcs620s.idm, 8) != 0)) {
    printf("read faild.\n");  
    return false;
  }

  for(int i=12; i<28; i++)
  {
    felicaBlock[i - 12] = buf[i];
  }

  return true;
}

/*
Write Without Encryption
*/
bool writeWithoutEncryption(uint16_t serviceCode, uint8_t blockNumber, uint8_t felicaBlock[16]) {
  int ret;
  uint8_t buf[RCS620S_MAX_CARD_RESPONSE_LEN];
  uint8_t responseLen = 0;
  
  buf[0] = 0x08;
  memcpy(buf + 1, rcs620s.idm, 8);
  buf[9] = 0x01;      // サービス数
  buf[10] = (uint8_t)((serviceCode >> 0) & 0xff);
  buf[11] = (uint8_t)((serviceCode >> 8) & 0xff);
  buf[12] = 0x01;     // ブロック数
  buf[13] = 0x80;
  buf[14] = blockNumber;
  for(int i=0; i<16; i++)
  {
    buf[i + 15] = felicaBlock[i];
  }

  ret = rcs620s.cardCommand(buf, 31, buf, &responseLen);
  if (!ret || (responseLen != 11) || (buf[0] != 0x09) || (buf[9] != 0x00) || (buf[10] != 0x00)) {
    return false;
  }

//  for(int i=12; i<28; i++)
//  {
//    felicaBlock[i - 12] = buf[i];
//  }

  return true;
}

/*
8バイトのXOR
*/
void arryXor(const uint8_t input1[8], const uint8_t input2[8], uint8_t output[8])
{
  for (int i = 0; i < 8; i++)
  {
    output[i] = input1[i] ^ input2[i];
  }
}

/*
トリプルDESを実行
*/
void tripleDES(uint8_t input[8], uint8_t XorInput[8], uint8_t input_key1[8], uint8_t input_key2[8], uint8_t input_key3[8], uint8_t output[8])
{
  // キー情報
  DES_cblock key[KEY_NUM];

  DES_cblock rawData;
  DES_cblock cryptData;
  DES_cblock decryptData;

  int ret;
  uint8_t tmpInput[8];
  uint8_t xorResult[8];


  // キー情報作成
  memset(key, 0, KEY_NUM * sizeof(DES_cblock));

  for (int i = 0; i < 8; i++)
  {
    tmpInput[i] = input[i];
    key[0][i] = input_key1[i];
    key[1][i] = input_key2[i];
    key[2][i] = input_key1[i];
  }

  swapByteOrder(tmpInput);
  arryXor(tmpInput, XorInput, xorResult);
  memset(&rawData, 0, sizeof(rawData));
  strncpy((char *)&rawData, (char *)xorResult, sizeof(rawData));

  swapByteOrder(key[0]);
  swapByteOrder(key[1]);
  swapByteOrder(key[2]);

  for (int i = 0; i < KEY_NUM; i++)
  {
    DES_set_odd_parity(&key[i]);
  }

  // キースケジュール作成
  DES_key_schedule schedule[KEY_NUM];
  for (int i = 0; i < KEY_NUM; i++)
  {
    ret = DES_set_key_checked(&key[i], &schedule[i]);
    if (ret != 0)
    {
      fprintf(stderr, "DES_key_set_checked 失敗 ");
      if (ret == -1)
      {
        fprintf(stderr, "パリティ不正\n");
      }
      else if (ret == -2)
      {
        fprintf(stderr, "脆弱なキー\n");
      }
      else
      {
        fprintf(stderr, "不明なエラー\n");
      }
      exit(-1);
    }
  }

  // 暗号化
  DES_ecb3_encrypt(&rawData, &cryptData,
                   &schedule[0], &schedule[1], &schedule[2],
                   DES_ENCRYPT);

  // 復号化
  //DES_ecb3_encrypt(&cryptData, &decryptData,
  //                 &schedule[0], &schedule[1], &schedule[2],
  //                 DES_DECRYPT);


  for (int i = 0; i < 8; i++)
  {
    output[i] = cryptData[i];
  }

}

/*
バイトオーダーを入れ替える(ポインタの先を直に入れ替えるので注意！)
*/
void swapByteOrder(uint8_t inout[8])
{
  uint8_t swap[8];
  for (int i = 0; i < 8; i++)
  {
    swap[7 - i] = inout[i];
  }
  for (int i = 0; i < 8; i++)
  {
    inout[i] = swap[i];
  }
}

/*
カードの個別化カード鍵を取得
*/
bool getCK(uint8_t CK1[8], uint8_t CK2[8])
{
  uint8_t CK[16];
  generatePersonalizedCK(CK);
  for (int i = 0; i < 8; i++)
  {
    CK1[i] = CK[i];
    CK2[i] = CK[8+i];
  }
}

/*
個別化カード鍵の作成
*/
bool generatePersonalizedCK(uint8_t CK[16])
{

  //IDブロックの値を読み出す
  uint8_t ID[16];
  if (!readWithoutEncryption(ADDRESS_ID, ID)) { 
    printf("readWithoutEncryption失敗\n");
    return false; 
  }

  printf("IDブロック　　 [");
  for (int i = 0; i < 16; i++)
  {
    printf("%02X ", ID[i]);
  }
  printf("]\n");

  uint8_t ZERO[8] = { 0 };
  uint8_t ZERO_1B[8] = { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x1b };
  uint8_t K1[8];

  //0と個別化マスター鍵で3DES            
  tripleDES(ZERO, ZERO, MK1, MK2, MK3, K1);
  
  bool msb = false;
  for(int i=7; i>=0; i--)
  {
    bool bak = msb;
    msb = (K1[i] & 0x80 != 0)? true : false;
    K1[i] = K1[i] << 1;
    if(bak)
    {
      //下のバイトからのcarry
      K1[i] = K1[i] | 0x0l;
    }
  }

  //Lの最上位ビットが1の場合、最下位バイトと0x1bをXORする
  if(msb)
  {
    K1[7] = K1[7] ^ 0x1b;
  }

  uint8_t ID1[8];
  uint8_t ID2[8];

  //Mを先頭から8byteずつに分け、M1, M2*とする(M2* xor K1 → M2)
  for(int i=0; i<8; i++)
  {
    ID1[i] = ID[i];
    ID2[i] = ID[8+i] ^ K1[i];
  }
  
  //M1を平文、Kを鍵として3DES→結果C1
  uint8_t C1[8];    
  tripleDES(ID1, ZERO, MK1, MK2, MK3, C1);

  //C1とM2をXORした結果を平文、Kを鍵として3DES→結果T
  uint8_t T1[8];
  uint8_t xorResult[8];
  arryXor(C1, ID2, xorResult);
  tripleDES(xorResult, ZERO, MK1, MK2, MK3, T1);

  //M1の最上位ビットを反転→M1'  
  ID1[0] = ID1[0] ^ 0x80;

  //M1'を平文、Kを鍵として3DES→結果C1'
  uint8_t C1_1[8];
  tripleDES(ID1, ZERO, MK1, MK2, MK3, C1_1);

  // (C1' xor M2)を平文、Kを鍵として3DES→結果T'
  arryXor(C1_1, ID2, xorResult);
  uint8_t T1_1[8];
  tripleDES(xorResult, ZERO, MK1, MK2, MK3, T1_1);

  //Tを上位8byte、T'を下位8byte→結果C→個別化カード鍵
  for(int i=0; i<8; i++)
  {
    CK[i] = T1[i];
    CK[8+i] = T1_1[i];
  }

  printf("個別化カード鍵 [");
  for (int i = 0; i < 16; i++)
  {
    printf("%02X ", CK[i]);
  }
  printf("]\n");

  return true;

}

/*
カードの0.5次発行
*/
bool issuanceFelica()
{

  int ret;
  if(!connectRcs620s()) return false;  //カード接続

  //カード鍵の書き込み
  //カード鍵の設定（書き込み） ブロック135番(0x87番=CKブロック)の内容を16進数[bb]で書き込む
  uint8_t CK[16]; 
  if(!generatePersonalizedCK(CK))
  {
    printf("カード鍵の取得 失敗\n");
    rcs620s.rfOff();  //カード接続断
    return false;
  }
  ret = writeWithoutEncryption(READ_WRITE_MODE, ADDRESS_CK, CK);

  //書き込んだカード鍵の内容を確認
  if(!checkMac)
  {
    printf("書き込んだカード鍵の内容を確認 失敗\n");
    rcs620s.rfOff();  //カード接続断
    return false;
  }
   
  //鍵バージョンの書き込み
  //ブロック134番(0x86番=CKVブロック)の内容を16進数[bb]で書き込む
  uint8_t  CKV[16] = {0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  ret = writeWithoutEncryption(READ_WRITE_MODE, ADDRESS_CKV, CKV);
  if(!ret)
  {
    printf("鍵バージョンの書き込み 失敗\n");
    rcs620s.rfOff();  //カード接続断
    return false;
  }

  uint8_t Val[16];  
  //書き込んだCKVと読み出したCKVを比較
  readWithoutEncryption(ADDRESS_CKV, Val);
  if(!ret || !compareBuf(CKV, Val))
  {
    printf("書き込んだCKVと読み出したCKVを比較 失敗\n");
    rcs620s.rfOff();  //カード接続断
    return false;
  }
  
  rcs620s.rfOff();  //カード接続断
  
  printf("0.5次発行　正常終了\n");
  return true;
}


/*
main
オプション
c : カード検証
i : カード発行
g : 個別化カード鍵の生成→表示
*/
int main(int argc, char *argv[])
{
  int opt;
  opterr = 0;

  while ((opt = getopt(argc, argv, "aig :")) != -1) {
    switch (opt) {
      case 'a':
        //auth(認証 - 内部)
        printf("-aがオプションとして渡されました\n");
        authFelica();
        break;
      case 'i':
        //issuance(発行) ※MCの書き込みは行っていないので、実際は「0.5次発行（仮称）」
        printf("-iがオプションとして渡されました\n");
        issuanceFelica();
        break;
      case 'g':
        //generate(発行)
        printf("-gがオプションとして渡されました\n");
        if(connectRcs620s())   //カード接続
        {
          uint8_t CK[16]; 
          generatePersonalizedCK(CK);
          rcs620s.rfOff();  //カード接続断
        }
        break;
      default:
        printf("Usage: %s [-a] [-i] [-g]\n");
    }
  }
  return 0;
}

