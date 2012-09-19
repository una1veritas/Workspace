/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2008
 *
 *    File name   : sd_spi_mode.c
 *    Description : SD/MMC driver
 *
 *    History :
 *    1. Date        : April 10, 2008
 *       Author      : Stanimir Bonev
 *       Description : Create
 *    $Revision: #1 $
 **************************************************************************/

#define SD_SPI_MODE_GLOBAL
#include "sd_spi_mode.h"

// SD Maximum Block Rad Access Time
#define RD_TIME_OUT               100LL   // ms
// SD Maximum Block Write Access Time
#define WR_TIME_OUT               250LL   // ms

// Card R1 bitmap definitions
#define _SD_OK              0x00
#define _SD_ILDE_STATE      0x01
#define _SD_ERASE_RST       0x02
#define _SD_ILLEGAL_CMD     0x04
#define _SD_CRC_ERROR       0x08
#define _SD_ERASE_ERROR     0x10
#define _SD_ADD_ERROR       0x20
#define _SD_PARAM_ERROR     0x40

#define _SD_DATA_TOLKEN     0xFE
#define _SD_DATA_ERR_TOLKEN 0x1F
#define _SD_STOP_TRAN       0xFD

#define _CSD_GET_TRAN_SPEED_EXP()      (_SdSdCsd[ 3]&0x07)
#define _CSD_GET_TRAN_SPEED_MANT()    ((_SdSdCsd[ 3]&0xF8)>>3 )
#define _CSD_GET_NSAC()                (_SdSdCsd[ 2]          )
#define _CSD_GET_TAAC_EXP()            (_SdSdCsd[ 1]&0x7)
#define _CSD_GET_TAAC_MANT()          ((_SdSdCsd[ 1]&0xF8)>>3 )
#define _CSD_GET_R2W_FACTOR()         ((_SdSdCsd[12]&0x1C)>>2 )
#define _CSD_GET_READ_BL_LEN()         (_SdSdCsd[ 5]&0x0F)
#define _CSD_GET_C_SIZE()            (((_SdSdCsd[ 6]&0x03)<<10) + (_SdSdCsd[7]<<2) + ((_SdSdCsd[8]&0xc0)>>6))
#define _CSD_GET_C_SIZE_MULT()       (((_SdSdCsd[ 9]&0x03)<<1 ) +((_SdSdCsd[10]&0x80)>>7))
#define _CSD_GET_PERM_WRITE_PROTECT() ((_SdSdCsd[14]&0x20)>>5 )
#define _CSD_GET_TMP_WRITE_PROTECT()  ((_SdSdCsd[14]&0x10)>>4 )

#define _CSD_2_0_GET_C_SIZE()        (((_SdSdCsd[7]&0x0F)<<16) + (_SdSdCsd[8]<<8) + _SdSdCsd[9])

#define _OCR            0x003E0000
#define _HC             0x40000000
#define _CMD8_DATA      0x000001AA

typedef enum __SdSpiCmdInd_t
{
  _CMD0 = 0,  // Resets the MultiMediaCard

  _CMD1,      // Activates the card’s initialization process

  _CMD8,      // Send Interface Condition Command
              // for HC cards only

  _CMD9,      // Asks the selected card to send its card-specific data (CSD)

  _CMD10,     // Asks the selected card to send its card identification (CID)

  _CMD12,     // Stop transmission on multiple block read

  _CMD13,     // Asks the selected card to send its status register

  _CMD16,     // Selects a block length (in bytes) for all following block commands (read and write)

  _CMD17,     // Reads a block of the size selected by the SET_BLOCKLEN command

  _CMD18,     // Continuously transfers data blocks from card to host
              // until interrupted by a Stop command or the requested number of data blocks transmitted

  _CMD24,     // Writes a block of the size selected by the SET_BLOCKLEN command

  _CMD25,     // Continuously writes blocks of data until a ‘Stop Tran’
              // Token or the requested number of blocks received

  _CMD27,     // Programming of the programmable bits of the CSD

  _CMD28,     // If the card has write protection features, this
              // command sets the write protection bit of the
              // addressed group. The properties of write protection
              // are coded in the card specific data (WP_GRP_SIZE).

  _CMD29,     // If the card has write protection features, this
              // command clears the write protection bit of the addressed group

  _CMD30,     // If the card has write protection features, this
              // command asks the card to send the status of the write protection bits

  _CMD32,     // Sets the address of the first sector of the erase group

  _CMD33,     // Sets the address of the last sector in a continuous
              // range within the selected erase group, or the address
              // of a single sector to be selected for erase

  _CMD34,     // Removes one previously selected sector from the erase selection

  _CMD35,     // Sets the address of the first erase group within a range to be selected for erase

  _CMD36,     // Sets the address of the last erase group within a
              // continuous range to be selected for erase

  _CMD37,     // Removes one previously selected erase group from the erase selection.

  _CMD38,     // Erases all previously selected sectors

  _CMD42,     // Used to set/reset the password or lock/unlock the
              // card. The size of the Data Block is defined by the SET_BLOCK_LEN command

  _CMD55,     // Notifies the card that the next command is an
              // application specific command rather than a standard command.

  _CMD56,     // Used either to transfer a Data Block to the card or
              // to get a Data Block from the card for general
              // purpose/application specific commands. The size
              // of the Data Block is defined with SET_BLOCK_LEN command

  _CMD58,     // Reads the OCR register of a card

  _CMD59,     // Turns the CRC option on or off. A ‘1’ in the CRC
              // option bit will turn the option on, a ‘0’ will turn it off

  _ACMD41,    // Activates the card’s initialization process (Only for SD)

  _CMD_END    // End of commands index
} _SdSpiCmdInd_t;

typedef enum __SdAgmType_t
{
  _SdNoArg = 0, _SdBlockLen, _SdDataAdd, _SdDummyWord
} _SdAgmType_t;

typedef enum __SdRespType_t
{
  _SdR1 = 0, _SdR1b, _SdR2, _SdR3, _SdR7
} _SdRespType_t;

typedef struct __SdCommads_t
{
  Int8U         TxData;
  _SdAgmType_t  Arg;
  _SdRespType_t Resp;
} _SdCommads_t;

typedef enum __SdState_t
{
  _SdOk = 0, _SdNoPresent, _SdNoResponse, _SdCardError,
  _SdMiscompare, _SdUnsupported
} _SdState_t;


const Int32U _SdTransfExp[] =
{
     10000UL,
    100000UL,
   1000000UL,
  10000000UL,
         0UL,
         0UL,
         0UL,
         0UL,
};

const Int32U _SdAccessTime [] =
{
        1UL,
       10UL,
      100UL,
     1000UL,
    10000UL,
   100000UL,
  1000000UL,
 10000000UL,
};

const Int32U _SdCsdMant[] =
{
  0UL,10UL,12UL,13UL,15UL,
  20UL,25UL,
  30UL,35UL,
  40UL,45UL,
  50UL,55UL,
  60UL,
  70UL,
  80UL,
};

const _SdCommads_t _SdCmd[_CMD_END] =
{
  // CMD0
  {0x40,_SdNoArg    ,_SdR1 },
  // CMD1
  {0x41,_SdNoArg    ,_SdR1 },
  // CMD8
  {0x48,_SdDataAdd  ,_SdR7 },
  // CMD9
  {0x49,_SdNoArg    ,_SdR1 },
  // CMD10
  {0x4A,_SdNoArg    ,_SdR1 },
  // CMD12
  {0x4C,_SdNoArg    ,_SdR1 },
  // CMD13
  {0x4D,_SdNoArg    ,_SdR2 },
  // CMD16
  {0x50,_SdBlockLen ,_SdR1 },
  // CMD17
  {0x51,_SdDataAdd  ,_SdR1 },
  // CMD18
  {0x52,_SdDataAdd  ,_SdR1 },
  // CMD24
  {0x58,_SdDataAdd  ,_SdR1 },
  // CMD25
  {0x59,_SdDataAdd  ,_SdR1 },
  // CMD27
  {0x5B,_SdNoArg    ,_SdR1 },
  // CMD28
  {0x5C,_SdDataAdd  ,_SdR1b},
  // CMD29
  {0x5D,_SdDataAdd  ,_SdR1b},
  // CMD30
  {0x5E,_SdDataAdd  ,_SdR1 },
  // CMD32
  {0x60,_SdDataAdd  ,_SdR1 },
  // CMD33
  {0x61,_SdDataAdd  ,_SdR1 },
  // CMD34
  {0x62,_SdDataAdd  ,_SdR1 },
  // CMD35
  {0x63,_SdDataAdd  ,_SdR1 },
  // CMD36
  {0x64,_SdDataAdd  ,_SdR1 },
  // CMD37
  {0x65,_SdDataAdd  ,_SdR1 },
  // CMD38
  {0x66,_SdDummyWord,_SdR1b},
  // CMD42
  {0x6A,_SdDummyWord,_SdR1b},
  // CMD55
  {0x77,_SdNoArg    ,_SdR1 },
  // CMD56
  {0x78,_SdNoArg    ,_SdR1 },
  // CMD58
  {0x7A,_SdNoArg    ,_SdR3 },
  // CMD59
  {0x7B,_SdDummyWord,_SdR1 },
  // ACMD41
  {0x69,_SdDataAdd  ,_SdR1 }
};

#define _CS_L() SdChipSelect(1)
#define _CS_H() SdChipSelect(0); SdTranserByte(0xFF)

static Int32U _SdSendCmd(_SdSpiCmdInd_t ComdInd, pInt32U pData);
static void _SdCsdImplemet (void);
static _SdState_t _SdInitMedia (void);

static inline _SdState_t _SdReadCardInfo(pInt8U pData);
static inline _SdState_t _SdRead(pInt8U pData, Int32U Add);
static inline _SdState_t _SdWrite(pInt8U pData, Int32U Add);
static inline _SdState_t _SdVerify(pInt8U pData, Int32U Add);

static DiskCtrlBlk_t _SdDskCtrlBlk;
static Int32U        _Tnac;
static Int32U        _Twr;
static Int8U         _SdSdCsd[16];
static Boolean       _bSdPermWriteProtect;
static Boolean       _bHC;

/*************************************************************************
 * Function Name: _SdSendCmd
 * Parameters: _SdSpiCmdInd_t ComdInd, Int32U Arg
 *
 * Return: Int32U
 *
 * Description: SD/MMC commands implement
 *
 *************************************************************************/
static
Int32U _SdSendCmd(_SdSpiCmdInd_t ComdInd, pInt32U pData)
{
Int32U ch = 0xFF;
Int32U i;
  // Chip Select
  _CS_L();
  // Send command code
  SdTranserByte(_SdCmd[ComdInd].TxData);
  // Send command's arguments
  if(_SdCmd[ComdInd].Arg == _SdNoArg)
  {
    SdTranserByte(0x00);
    SdTranserByte(0x00);
    SdTranserByte(0x00);
    SdTranserByte(0x00);
  }
  else
  {
    assert(pData);
    SdTranserByte(*pData >> 24);
    SdTranserByte(*pData >> 16);
    SdTranserByte(*pData >> 8 );
    SdTranserByte(*pData      );
  }
  // Send CRC
  if(_CMD0 == ComdInd)
  {
    SdTranserByte(0x95);
  }
  else
  {
    SdTranserByte(0xFF);
  }
  // wait for command response
  for(i = 9; i && (ch == 0xFF); --i) ch = SdTranserByte(0xFF);

  if (0 == i)
  {
    _CS_H();
    return((Int32U)-1);
  }

  // Implement command response
  switch (_SdCmd[ComdInd].Resp)
  {
  case _SdR1b:
    for (ch = 0,i = _Twr; i && (ch != 0xFF); --i)
    {
      ch = SdTranserByte(0xFF);
    }
  case _SdR1:
    break;
  case _SdR2:
    ch  = ((Int32U)ch << 8)     & 0x0000FF00;
    ch |= SdTranserByte(0xFF)  & 0x000000FF;
    break;
  default:
    assert(pData);
    *pData  = ((Int32U)SdTranserByte(0xFF) << 24) & 0xFF000000;
    *pData |= ((Int32U)SdTranserByte(0xFF) << 16) & 0x00FF0000;
    *pData |= ((Int32U)SdTranserByte(0xFF) << 8 ) & 0x0000FF00;
    *pData |=          SdTranserByte(0xFF)        & 0x000000FF;
  }
  return(ch);
}

/*************************************************************************
 * Function Name: _SdInitMedia
 * Parameters: none
 *
 * Return: _SdState_t
 *
 * Description: SD/MMC detect and initialize
 *
 *************************************************************************/
static _SdState_t _SdInitMedia (void)
{
Int32U i,res, data;
  _Tnac = 1;
  if(!SdPresent())
  {
    SdPowerOff();
    return(_SdNoPresent);
  }
  // Clock Freq. Identification Mode < 400kHz
  SdSetClockFreq(IdentificationModeClock);
  /*Enable MMC/SD power*/
  SdPowerOn();
  // After power up at least 74 clock cycles are required prior to
  // starting bus communication
  _CS_H();
  for(i = 10; i; --i) SdTranserByte(0xFF);

  // CMD0 (Go to IDLE) and put MMC/SD card in SPI mode
  res = _SdSendCmd(_CMD0,NULL);
  _CS_H();
  if(_SD_ILDE_STATE != res)
  {
    return(_SdNoResponse);
  }

  // Determinate Card type - SD Class 1, SD Class2 or MMC
  _SdDskCtrlBlk.DiskType = DiskSD_Spec2_0;
  _bHC = FALSE;

  for(i=100;i;--i)
  {
    data = _CMD8_DATA;
    res = _SdSendCmd(_CMD8,&data);
    _CS_H();
    if(res & _SD_ILLEGAL_CMD)
    {
      // The card doesn't support specification 2.0
       _SdDskCtrlBlk.DiskType = DiskSD_Spec1_x;
    }
    else if (  (_CMD8_DATA & 0xFF) == (data & 0xFF)
             && 0 == ((1<<8) & data))
    {
      return(_SdUnsupported);
    }

    res = _SdSendCmd(_CMD58,&data);
    _CS_H();
    if(~_SD_ILDE_STATE & res)
    {
      // get CID must be always valid
      return(_SdNoResponse);
    }

    if (0 == (_OCR & data))
    {
      return(_SdUnsupported);
    }

    res = _SdSendCmd(_CMD55,NULL);
    _CS_H();
    data = (DiskSD_Spec2_0 == _SdDskCtrlBlk.DiskType)?_HC:0;
    res = _SdSendCmd(_ACMD41,&data);
    _CS_H();
    if(_SD_ILLEGAL_CMD & res)
    {
      // MMC card may be
      _SdDskCtrlBlk.DiskType = DiskMMC;
      break;
    }
    if(_SD_OK == res)
    {
      // SD card is find
      // Get Card's capacity type
      res = _SdSendCmd(_CMD58,&data);
      _CS_H();
      if(~_SD_ILDE_STATE & res)
      {
        // get CID must be always valid
        return(_SdNoResponse);
      }

      _bHC = 0 != (_HC & data);
      break;
    }
    SdDly_1ms(50);
  }
  if(i == 0)
  {
    return(_SdNoResponse);
  }

  if(DiskMMC == _SdDskCtrlBlk.DiskType)
  {
    // CMD1 for MMC Init sequence
    // will be complete within 500ms
    for (i = 100; i;--i)
    {
      res = _SdSendCmd(_CMD1,0);
      _CS_H();
      if(_SD_OK == res)
      {
        // Init complete
        break;
      }
      SdDly_1ms(50);
    }
    if(i == 0)
    {
      return(_SdNoResponse);
    }
  }

  // Read CSD
  res = _SdReadCardInfo(_SdSdCsd);
  if(_SD_OK != res)
  {
    // CSD must be always valid
    return(_SdNoResponse);
  }

  _SdCsdImplemet();

  res = _SdSendCmd(_CMD16,&_SdDskCtrlBlk.BlockSize);
  if(_SD_OK != res)
  {
    // CSD must be always valid
    return(_SdNoResponse);
  }
  _CS_H();
  return(_SdOk);
}

/*************************************************************************
 * Function Name: _SdReadCardInfo
 * Parameters: pInt8U pData,
 *
 * Return: _SdState_t
 *
 * Description: Read CSD registers depend of command
 *
 *************************************************************************/
static inline
_SdState_t _SdReadCardInfo(pInt8U pData)
{
Int32U i;
Int32U res;
  res = _SdSendCmd(_CMD9,NULL);
  if (res == _SD_OK)
  {
    for(i = 8; i ; --i)
    {
      res = SdTranserByte(0xFF);
      if(_SD_DATA_ERR_TOLKEN == (res | _SD_DATA_ERR_TOLKEN))
      {
        return(_SdCardError);
      }
      else if (_SD_DATA_TOLKEN == res)
      {
        // Read CSD or CID data
        for(i = 0; 16 > i; ++i)
        {
          *pData++ = SdTranserByte(0xFF);
        }
        // CRC receive
        SdTranserByte(0xFF);
        SdTranserByte(0xFF);
        _CS_H();
        return(_SdOk);
      }
    }
  }
  _CS_H();
  return(_SdNoResponse);
}

/*************************************************************************
 * Function Name: _SdCsdImplemet
 * Parameters:  none
 *
 * Return: none
 *
 * Description: Implement data from CSD
 *
 *************************************************************************/
static
void _SdCsdImplemet (void)
{
Int32U Freq;
Int64U Tmp;
  // Calculate SPI max clock
  Freq = _SdTransfExp[_CSD_GET_TRAN_SPEED_EXP()] * _SdCsdMant[_CSD_GET_TRAN_SPEED_MANT()];
  Freq = SdSetClockFreq(Freq);
  if(DiskMMC == _SdDskCtrlBlk.DiskType)
  {
    // Calculate Time outs for MMC cards
    Tmp = _SdAccessTime[_CSD_GET_TAAC_EXP()] * _SdCsdMant[_CSD_GET_TAAC_MANT()];
    Tmp /= 10000; // us
    // Freq [Hz], Tmp[1 us], *10
    Tmp = (Freq*Tmp)/100000LL;
    // NSAC*100*10
    Tmp += 1000*_CSD_GET_NSAC();
    // Max time out
    _Tnac = Tmp;
    _Twr  = Tmp * (1<<_CSD_GET_R2W_FACTOR());
  }
  else
  {
    // Calculate Time outs for SD cards
    // Freq [Hz], RD_TIME_OUT[ms]
    Tmp = Freq/1000;
    _Tnac = Tmp * RD_TIME_OUT;
    _Twr  = Tmp * WR_TIME_OUT;
  }
  // Calculate Block size and Block Number
  _SdDskCtrlBlk.BlockSize = 1<<_CSD_GET_READ_BL_LEN();
  if(_bHC)
  {
    _SdDskCtrlBlk.BlockNumb = (_CSD_2_0_GET_C_SIZE()+1) * 1024;
  }
  else
  {
    _SdDskCtrlBlk.BlockNumb = (_CSD_GET_C_SIZE()+1)*(4<<_CSD_GET_C_SIZE_MULT());
  }
  if(_SdDskCtrlBlk.BlockSize != SD_DEF_BLK_SIZE)
  {
    // because Windows support only 512 bytes block
    _SdDskCtrlBlk.BlockNumb *= (_SdDskCtrlBlk.BlockSize>>9);
    _SdDskCtrlBlk.BlockSize  = SD_DEF_BLK_SIZE;
  }
  // Set Write Protect
  _bSdPermWriteProtect = _CSD_GET_PERM_WRITE_PROTECT() |\
                          _CSD_GET_TMP_WRITE_PROTECT();
  _SdDskCtrlBlk.WriteProtect = SdWriteProtect() |\
                              _bSdPermWriteProtect;
}

/*************************************************************************
 * Function Name: _SdRead
 * Parameters: pInt8U pData, Int32U Add, Int32U Length
 *
 * Return: _SdState_t
 *
 * Description: Read from a SD/MMC
 *
 *************************************************************************/
inline static
_SdState_t _SdRead(pInt8U pData, Int32U Add)
{
Int32U res,i;
  res = _SdSendCmd(_CMD17,&Add);
  if(res == _SD_OK)
  {
    for(i = _Tnac; i; --i)
    {
      res = SdTranserByte(0xFF);
      if(_SD_DATA_ERR_TOLKEN == (res | _SD_DATA_ERR_TOLKEN))
      {
        _CS_H();
        return(_SdCardError);
      }
      else if (res == _SD_DATA_TOLKEN)
      {
        // Receive block
        SdReceiveBlock(pData, SD_DEF_BLK_SIZE);
        // CRC receive
        SdTranserByte(0xFF);
        SdTranserByte(0xFF);
        _CS_H();
        return(_SdOk);
      }
    }
  }
  _CS_H();
  return(_SdNoResponse);
}

/*************************************************************************
 * Function Name: _SdWrite
 * Parameters: pInt8U pData, Int32U Add, Int32U Length
 *
 * Return: _SdState_t
 *
 * Description: Write to a SD/MMC
 *
 *************************************************************************/
inline static
_SdState_t _SdWrite(pInt8U pData, Int32U Add)
{
Int32U res,i;
  res = _SdSendCmd(_CMD24,&Add);
  if(_SD_OK == res)
  {
    SdTranserByte(0xFF);
    SdTranserByte(_SD_DATA_TOLKEN);
    // Send block
    SdSendBlock(pData,SD_DEF_BLK_SIZE);
    // CRC Send
    SdTranserByte(0xFF);
    SdTranserByte(0xFF);
    res = SdTranserByte(0xFF) & 0x1F;
    if(res != 0x05)
    {
      _CS_H();
      return(_SdCardError);
    }

    for(i = _Twr; i ;i--)
    {
      if(SdTranserByte(0xFF) == 0xFF)
      {
        break;
      }
    }

    _CS_H();
    if (0 == i)
    {
      return(_SdNoResponse);
    }
    return(_SdOk);
  }
  _CS_H();
  return(_SdNoResponse);
}

/*************************************************************************
 * Function Name: _SdVerify
 * Parameters: pInt8U pData, Int32U Add, Int32U Length
 *
 * Return: _SdState_t
 *
 * Description: Verify on a SD/MMC
 *
 *************************************************************************/
inline static
_SdState_t _SdVerify(pInt8U pData, Int32U Add)
{
Int32U res,i;
  res = _SdSendCmd(_CMD17,&Add);
  if(res == _SD_OK)
  {
    for(i = _Tnac;i;--i)
    {
      res = SdTranserByte(0xFF);
      if(_SD_DATA_ERR_TOLKEN == (res | _SD_DATA_ERR_TOLKEN))
      {
        return(_SdCardError);
      }
      else if (_SD_DATA_TOLKEN == res)
      {
        res = 0;
        for(i = 0; i<SD_DEF_BLK_SIZE;++i,++pData)
        {
          *pData ^= SdTranserByte(0xFF);
          if (*pData != 0)
          {
            res = 1;
          }
        }
        // CRC receive
        SdTranserByte(0xFF);
        SdTranserByte(0xFF);
        _CS_H();
        if (res)
        {
          return(_SdMiscompare);
        }
        return(_SdOk);
      }
    }
  }
  return(_SdNoResponse);
}

/*************************************************************************
 * Function Name: SdStatusUpdate
 * Parameters: none
 *
 * Return: none
 *
 * Description: Update status of SD/MMC card
 *
 *************************************************************************/
void SdStatusUpdate (void)
{
  // Update WP state
  _SdDskCtrlBlk.WriteProtect = SdWriteProtect() |\
                               _bSdPermWriteProtect;
  if(!SdPresent())
  {
    _SdDskCtrlBlk.DiskStatus = DiskNotPresent;
    SdPowerOff();
    return;
  }
  if((DiskCommandPass != _SdDskCtrlBlk.DiskStatus))
  {
    switch (_SdInitMedia())
    {
    case _SdOk:
      _SdDskCtrlBlk.DiskStatus = DiskCommandPass;
      _SdDskCtrlBlk.MediaChanged = TRUE;
      break;
    case _SdCardError:
      _SdDskCtrlBlk.DiskStatus = DiskNotReady;
      break;
    default:
      _SdDskCtrlBlk.DiskStatus = DiskNotPresent;
      break;
    }
  }
  else if (_SdOk != _SdReadCardInfo(_SdSdCsd))
  {
    _SdDskCtrlBlk.DiskStatus = DiskNotReady;
  }
}

/*************************************************************************
 * Function Name: SdDiskInit
 * Parameters:  none
 *
 * Return: none
 *
 * Description: Init MMC/SD disk
 *
 *************************************************************************/
void SdDiskInit (void)
{
  _SdDskCtrlBlk.BlockNumb =\
  _SdDskCtrlBlk.BlockSize = 0;
  // Init SPI
  SdInit();
  // Media Init
  switch (_SdInitMedia())
  {
  case _SdOk:
    _SdDskCtrlBlk.DiskStatus = DiskCommandPass;
    _SdDskCtrlBlk.MediaChanged = TRUE;
    break;
  case _SdCardError:
    _SdDskCtrlBlk.DiskStatus = DiskNotReady;
    break;
  default:
    _SdDskCtrlBlk.DiskStatus = DiskNotPresent;
    break;
  }
}

/*************************************************************************
 * Function Name: SdGetDiskCtrlBkl
 * Parameters:  none
 *
 * Return: pDiskCtrlBlk_t
 *
 * Description: Return pointer to status structure of the disk
 *
 *************************************************************************/
pDiskCtrlBlk_t SdGetDiskCtrlBkl (void)
{
  return(&_SdDskCtrlBlk);
}

/*************************************************************************
 * Function Name: SdDiskIO
 * Parameters: pInt8U pData,Int32U BlockStart,
 *             Int32U BlockNum, DiskIoRequest_t IoRequest
 *
 * Return: DiskStatusCode_t
 *
 * Description: MMC/SD disk I/O
 *
 *************************************************************************/
DiskStatusCode_t SdDiskIO (pInt8U pData,Int32U BlockStart,
                           Int32U BlockNum, DiskIoRequest_t IoRequest)
{
  if((NULL == pData) || (BlockStart+BlockNum > _SdDskCtrlBlk.BlockNumb))
  {
    return(DiskParametersError);
  }
  if (_SdDskCtrlBlk.DiskStatus)
  {
    return(_SdDskCtrlBlk.DiskStatus);
  }
  switch (IoRequest)
  {
  case DiskWrite:
    if(_SdDskCtrlBlk.WriteProtect)
    {
      return(DiskParametersError);
    }
    while(BlockNum--)
    {
      switch (_SdWrite(pData,_bHC?BlockStart:BlockStart*_SdDskCtrlBlk.BlockSize))
      {
      case _SdOk:
        BlockStart = _bHC?++BlockStart:BlockStart+_SdDskCtrlBlk.BlockSize;
        break;
      case _SdCardError:
        _SdDskCtrlBlk.DiskStatus = DiskNotReady;
        return(_SdDskCtrlBlk.DiskStatus);
      default:
        _SdDskCtrlBlk.DiskStatus = DiskNotPresent;
        return(_SdDskCtrlBlk.DiskStatus);
      }
    }
    break;
  case DiskRead:
    while(BlockNum--)
    {
      switch (_SdRead(pData,
                     _bHC?BlockStart:BlockStart*_SdDskCtrlBlk.BlockSize))
      {
      case _SdOk:
        BlockStart = _bHC?++BlockStart:BlockStart+_SdDskCtrlBlk.BlockSize;
        break;
      case _SdCardError:
        _SdDskCtrlBlk.DiskStatus = DiskNotReady;
        return(_SdDskCtrlBlk.DiskStatus);
      default:
        _SdDskCtrlBlk.DiskStatus = DiskNotPresent;
        return(_SdDskCtrlBlk.DiskStatus);
      }
    }
    break;
  case DiskVerify:
    while(BlockNum--)
    {
      switch (_SdVerify(pData,
                        _bHC?BlockStart:BlockStart*_SdDskCtrlBlk.BlockSize))
      {
      case _SdOk:
        BlockStart = _bHC?++BlockStart:BlockStart+_SdDskCtrlBlk.BlockSize;
        break;
      case _SdCardError:
        _SdDskCtrlBlk.DiskStatus = DiskNotReady;
        return(_SdDskCtrlBlk.DiskStatus);
      default:
        _SdDskCtrlBlk.DiskStatus = DiskNotPresent;
        return(_SdDskCtrlBlk.DiskStatus);
      }
    }
    break;
  default:
    return(DiskParametersError);
  }
  return(_SdDskCtrlBlk.DiskStatus);
}

