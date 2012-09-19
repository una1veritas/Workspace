/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2008
 *
 *    File name   : temp_sensor_drv.h
 *    Description : Temperature sensor STCN75 driver
 *
 *    History :
 *    1. Date        : July 28, 2008
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#define TEMP_SENSOR_DRV_GLOBAL
#include "temp_sensor_drv.h"

/* DEFINE LOCAL TYPES HERE */
typedef Int8U BYTE;

/* DEFINE LOCAL CONSTANTS HERE */
#define SOFT_I2C_PORT_CLOCK RCC_AHB1Periph_GPIOG
#define SOFT_I2C_PORT GPIOG
#define SOFT_I2C_SCL_PIN GPIO_Pin_10
#define SOFT_I2C_SDA_PIN GPIO_Pin_12

#define ACK 0
#define NACK 1

/* DECLARE LOCAL FUNCTIONS HERE */
static void local_I2C_Initialize(void);
static void local_I2C_Delay(void);
static void local_I2C_Start(void);
static void local_I2C_Stop(void);
static BYTE local_I2C_ReadByte(char ack);
static char local_I2C_WriteByte(BYTE data); // returns ack state
static Boolean local_I2C_DataTransfer(Int8U SlaveAddr, pInt8U pData, Int32U Size);

typedef enum _temp_sensor_regs_t
{
  TEMP_REG = 0, CONF_REG, THYS_REG, TOS_REG
} temp_sensor_regs_t;

/*************************************************************************
 * Function Name: TempSensor_Init
 * Parameters: none
 *
 * Return: Boolean
 *
 * Description: Init Temperature sensor
 *
 *************************************************************************/
Boolean TempSensor_Init (void)
{
GPIO_InitTypeDef GPIO_InitStructure;
Int8U Data[3];


  // Init temperature sensor interrupt signal
  RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
  GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN;
  GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOE, &GPIO_InitStructure);

  local_I2C_Initialize();

  // Init Temperature sensor to default
  Data[0] = CONF_REG;
  Data[1] = 0;
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 2))
  {
    return(FALSE);
  }

  // THYS 75C
  Data[0] = THYS_REG;
  Data[1] = 0x4B;
  Data[2] = 0x00;
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 3))
  {
    return(FALSE);
  }

  // TOS 80C
  Data[0] = TOS_REG;
  Data[1] = 0x50;
  Data[2] = 0x00;
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 3))
  {
    return(FALSE);
  }

  return(TRUE);
}

/*************************************************************************
 * Function Name: TempSensorShutdown
 * Parameters: Boolean Shutdown
 *
 * Return: Boolean
 *
 * Description: Enable/Disable sensor shutdown
 *
 *************************************************************************/
Boolean TempSensorShutdown (Boolean Shutdown)
{
  Int8U Data[2];
  Data[0] = CONF_REG;

  // Read config reg
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 1))
  {
    return(FALSE);
  }
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR | 0x1, &Data[1], 1))
  {
    return(FALSE);
  }
  if (Shutdown)
  {
    Data[1] |= 1U << 0;
  }
  else
  {
    Data[1] &= ~(1U << 0);
  }
  // Write config reg
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data,2))
  {
    return(FALSE);
  }

  return(TRUE);
}

/*************************************************************************
 * Function Name: TempSensor_Conf
 * Parameters: Flo32 TOS, Flo32 THYS, Boolean Mode, Int32U FaultTol
 *
 * Return: Boolean
 *
 * Description: Temperature sensor config
 *
 *************************************************************************/
Boolean TempSensor_Conf (Flo32 TOS, Flo32 THYS, Boolean Mode,
                         Int32U FaultTol)
{
Int8U Data[3];
Int16S Temp;

  // Configure the fault tolerance
  Data[0] = CONF_REG;
  // Read config reg
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 1))
  {
    return(FALSE);
  }
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR | 0x1, &Data[1], 1))
  {
    return(FALSE);
  }
  Data[1] &= ~0x18;
  Data[1] |= (FaultTol & 3) << 3;

  // Write config reg
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data,2))
  {
    return(FALSE);
  }
  // Set Tos
  Temp = (Int16S)(TOS * 256.0);
  Data[0] = TOS_REG;
  Data[1] = Temp>>8;
  Data[2] = Temp;
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 3))
  {
    return(FALSE);
  }
  // Set Thys
  Temp = (Int16S)(THYS * 256.0);
  Data[0] = THYS_REG;
  Data[1] = Temp>>8;
  Data[2] = Temp;
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 3))
  {
    return(FALSE);
  }

  return(TRUE);
}

/*************************************************************************
 * Function Name: TempSensorGetTemp
 * Parameters: pFlo32 pTemp, pBoolean pAlarm
 *
 * Return: Boolean
 *
 * Description: Temperature sensor get temperature and alarm state
 *
 *************************************************************************/
Boolean TempSensorGetTemp (pFlo32 pTemp, pBoolean pAlarm)
{
Int8U Data[2];
Int16S Temp;

  if(Bit_SET == GPIO_ReadInputDataBit(GPIOE,GPIO_Pin_2))
  {
    *pAlarm = FALSE;
  }
  else
  {
    *pAlarm = TRUE;
  }

  // Configure the fault tolerance
  Data[0] = TEMP_REG;
  // Read config reg
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR, Data, 1))
  {
    return(FALSE);
  }
  if(FALSE == local_I2C_DataTransfer(TEMP_SENSOR_SENSOR_ADDR | 0x1, Data, 2))
  {
    return(FALSE);
  }
  Temp = (Data[0] << 8) | Data[1];

  *pTemp = (Flo32)Temp;
  *pTemp /= 256.0;

  return(TRUE);
}

/* local functions */
static void local_I2C_Initialize(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	
	/* Enable the software I2C Clock */
	RCC_AHB1PeriphClockCmd(SOFT_I2C_PORT_CLOCK, ENABLE);
	
	/* Configure the SDA and SCL pins */
	// RG12 - SDA
	// RG10 - SCL
	GPIO_InitStructure.GPIO_Pin = SOFT_I2C_SCL_PIN | SOFT_I2C_SDA_PIN;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_Init(SOFT_I2C_PORT, &GPIO_InitStructure);
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);

	local_I2C_Delay();
}

static void local_I2C_Delay(void)
{
	Int32U d = 10000;
	while(d--) {
	}	
}	

static void local_I2C_Start(void)
{
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	local_I2C_Delay();
	
}

static void local_I2C_Stop(void)
{
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	local_I2C_Delay();
}

static BYTE local_I2C_ReadByte(char ack)
{
	BYTE data = 0;
	char i;
	
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET); // make input
	for(i = 0; i < 8; i++) {
		local_I2C_Delay();
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
		local_I2C_Delay();
		data |= GPIO_ReadInputDataBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN) & 0x01;
		if(i != 7)
			data <<= 1;
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	}
	
	// issue the ack
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, ack ? Bit_SET : Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	local_I2C_Delay();
	
	return data;
}

// returns ack state, 0 means acknowledged
static char local_I2C_WriteByte(BYTE data)
{
	char i;

	// send the 8 bits
	for(i = 0; i < 8; i++) {
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, (data & 0x80) ? Bit_SET : Bit_RESET);
		data <<= 1;
		local_I2C_Delay();
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
		local_I2C_Delay();
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	}
	
	// read the ack
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
	local_I2C_Delay();
	i = GPIO_ReadInputDataBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN);
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	local_I2C_Delay();
	
	return i;
}

static Boolean local_I2C_DataTransfer(Int8U SlaveAddr, pInt8U pData, Int32U Size)
{
	int result = 0, i;
	
	do {
		local_I2C_Start();
		result |= local_I2C_WriteByte(SlaveAddr);
		if(result) break;

		// read mode requested
		if(SlaveAddr & 0x01) {
			for(i = 0; i < Size; i++) {
				*pData++ = local_I2C_ReadByte( (i == Size -1) ? NACK : ACK );
			}
				
		} else {
			for(i = 0; i < Size; i++) {
				result |= local_I2C_WriteByte(*pData++);
				if(result)
					break;
			}			
		}
		
		local_I2C_Stop();			
		
	} while(0);
	
	// if there was an error reset pins to idle state
	if(result)
        local_I2C_Stop();
	
	return (result ? FALSE : TRUE);	
}


