////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////    main.c
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <avr/io.h>
#include <avr/eeprom.h>
#include <util/delay.h>

/* Scheduler include files. */
#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <semphr.h>

/* Pololu derived include files. */
#include <digitalAnalog.h>

/* serial interface include file. */
#include <lib_serial.h>

/* i2c Interface include file. */
#include <i2cMultiMaster.h>

/* RTC interface (using I2C) include file */
#include <rtc.h>

/* hd44780 LCD control interface file. */
#include <hd44780.h>

/* Servo PWM Timer include file */
#include <servoPWM.h>

/* Clock include file. */
#include "retrograde.h"
#include "xitoa.h"


/*--------------PrivateFunctions-------------------*/

static void get_line (uint8_t *buff, uint8_t len);

static int8_t ReadADCSensors(void);   // Read ADC Sensor for thermal LM335z

/*--------------Functions---------------------------*/

/* Main program loop */
int16_t main(void) __attribute__((OS_main));

int16_t main(void)
{

    // turn on the serial port for setting or querying the time .
	xSerialPort = xSerialPortInitMinimal( 115200, 80, 80); //  serial port: WantedBaud, TxQueueLength, RxQueueLength (8n1)

    // Memory shortages mean that we have to minimise the number of
    // threads, hence there are no longer multiple threads using a resource.
    // Still, semaphores are useful to stop a thread proceeding, where it should be stopped because it is using a resource.
    vSemaphoreCreateBinary( xADCSemaphore ); 		// binary semaphore for ADC - Don't sample temperature when hands are moving (voltage droop).

	// initialise I2C master interface, need to do this once only.
	// If there are two I2C processes, then do it during the system initiation.
	I2C_Master_Initialise((ARDUINO<<I2C_ADR_BITS) | (pdTRUE<<I2C_GEN_BIT));

	avrSerialPrint_P(PSTR("\r\n\nHello World!\r\n")); // Ok, so we're alive...

    xTaskCreate(
		TaskWriteLCD
		,  (const signed portCHAR *)"WriteLCD"
		,  192
		,  NULL
		,  2
		,  NULL ); // */

   xTaskCreate(
		TaskWriteRTCRetrograde
		,  (const signed portCHAR *)"WriteRTCRetrograde"
		,  96
		,  NULL
		,  1
		,  &xTaskWriteRTCRetrograde ); // */

   xTaskCreate(
		TaskMonitor
		,  (const signed portCHAR *)"SDMonitor" // Serial Monitor
		,  208
		,  NULL
		,  3
		,  NULL ); // */

//	avrSerialPrintf_P(PSTR("\r\nFree Heap Size: %u\r\n"),xPortGetFreeHeapSize() ); // needs heap_1 or heap_2 for this function to succeed.

    vTaskStartScheduler();

	avrSerialPrint_P(PSTR("\r\n\n\nGoodbye... no space for idle task!\r\n")); // Doh, so we're dead...

#if defined (portHD44780_LCD)
	lcd_Locate (0, 1);
	lcd_Print_P(PSTR("DEAD BEEF!"));
#endif


}

/*-----------------------------------------------------------*/


static void TaskMonitor(void *pvParameters) // Monitor for Serial Interface
{
    (void) pvParameters;;

	uint8_t *ptr;
	uint32_t p1;

	// create the buffer on the heap (so they can be moved later).
	if(LineBuffer == NULL) // if there is no Line buffer allocated (pointer is NULL), then allocate buffer.
		if( !(LineBuffer = (uint8_t *) pvPortMalloc( sizeof(uint8_t) * LINE_SIZE )))
			xSerialPrint_P(PSTR("pvPortMalloc for *LineBuffer fail..!\r\n"));

#if defined (portRTC_DEFINED)

	if(SetTimeDate == NULL) // if there is no SetTimeDate allocated (pointer is NULL), then allocate buffer.
		if( !(SetTimeDate = (pRTCArraySto) pvPortMalloc( sizeof(xRTCArraySto) )))
			xSerialPrint_P(PSTR("pvPortMalloc for *Buff fail..!\r\n"));
#endif


    while(1)
    {

    	xSerialPutChar(xSerialPort, '>', 100 / portTICK_RATE_MS);

		ptr = LineBuffer;
		get_line(ptr, (uint8_t)(sizeof(uint8_t)* LINE_SIZE)); //sizeof (Line);

		switch (*ptr++) {

#ifdef portRTC_DEFINED
		case 't' :	/* t [<year yy> <month mm> <date dd> <day: Sun=1> <hour hh> <minute mm> <second ss>] */

			if (xatoi(&ptr, &p1)) {
				SetTimeDate->Year = (uint8_t)p1;
				xatoi(&ptr, &p1); SetTimeDate->Month = (uint8_t)p1;
				xatoi(&ptr, &p1); SetTimeDate->Date = (uint8_t)p1;
				xatoi(&ptr, &p1); SetTimeDate->Day = (uint8_t)p1;
				xatoi(&ptr, &p1); SetTimeDate->Hour = (uint8_t)p1;
				xatoi(&ptr, &p1); SetTimeDate->Minute = (uint8_t)p1;
				if (!xatoi(&ptr, &p1))
					break;
				SetTimeDate->Second = (uint8_t)p1;

				xSerialPrintf_P(PSTR("Set: %u/%u/%u %2u:%02u:%02u\r\n"), SetTimeDate->Year, SetTimeDate->Month, SetTimeDate->Date, SetTimeDate->Hour, SetTimeDate->Minute, SetTimeDate->Second);
				if (setDateTimeDS1307( SetTimeDate ) == pdTRUE)
					xSerialPrint_P( PSTR("Setting successful\r\n") );

			} else {

				if (getDateTimeDS1307( &xCurrentTempTime.DateTime) == pdTRUE)
					xSerialPrintf_P(PSTR("Current: %u/%u/%u %2u:%02u:%02u\r\n"), xCurrentTempTime.DateTime.Year + 2000, xCurrentTempTime.DateTime.Month, xCurrentTempTime.DateTime.Date, xCurrentTempTime.DateTime.Hour, xCurrentTempTime.DateTime.Minute, xCurrentTempTime.DateTime.Second);
			}

			break;
#endif

		case 'r' : // reset
			switch (*ptr++) {
			case 't' : // temperature

				xMaximumTempTime = xCurrentTempTime;
				xMinimumTempTime = xCurrentTempTime;
				// Now we commit the time and temperature to the EEPROM, forever...
				eeprom_update_block(&xMaximumTempTime, &xMaximumEverTempTime, sizeof(xMaximumTempTime));
				eeprom_update_block(&xMinimumTempTime, &xMinimumEverTempTime, sizeof(xMinimumTempTime));

				break;
			default :
				break;
			}
			break;

		default :
			break;

		}
// 		xSerialPrintf_P(PSTR("\r\nSD Monitor HighWater @ %u\r\n\n"), uxTaskGetStackHighWaterMark(NULL));
    }

}


/*-----------------------------------------------------------*/


static void TaskWriteLCD(void *pvParameters) // Write to LCD
{
    (void) pvParameters;

    portTickType xLastWakeTime;
	/* The xLastWakeTime variable needs to be initialised with the current tick
	count.  Note that this is the only time we access this variable.  From this
	point on xLastWakeTime is managed automatically by the vTaskDelayUntil()
	API function. */
	xLastWakeTime = xTaskGetTickCount();

    eeprom_read_block(&xMaximumTempTime, &xMaximumEverTempTime, sizeof(xRTCTempArray));
    eeprom_read_block(&xMinimumTempTime, &xMinimumEverTempTime, sizeof(xRTCTempArray));

    setAnalogMode(MODE_10_BIT);    // 10-bit analog-to-digital conversions

    lcd_Init();	// initialise LCD, move cursor to start of top line

    while(1)
    {
    	if(getDateTimeDS1307(&xCurrentTempTime.DateTime))
    	{
			if ( (xCurrentTempTime.Temperature = ReadADCSensors()) != 0x7f) // if 0x7f then no reading returned.
			{							// trigger a temperature reading

				if( (xCurrentTempTime.Temperature < 65) && (xCurrentTempTime.Temperature > xMaximumTempTime.Temperature)) // check for maximum temp
				// we don't expect the temperature sensor to work above 65C
				{
					xMaximumTempTime = xCurrentTempTime;

					// Now we commit the time and temperature to the EEPROM, forever...
					eeprom_update_block(&xMaximumTempTime, &xMaximumEverTempTime, sizeof(xMaximumTempTime));
				}

				if( (xCurrentTempTime.Temperature > (-30)) && (xCurrentTempTime.Temperature < xMinimumTempTime.Temperature)) // and check for minimum temp
				// we don't expect the temperature sensor to work below -30C
				{
					xMinimumTempTime = xCurrentTempTime;

					// Now we commit the time and temperature to the EEPROM, forever...
					eeprom_update_block(&xMinimumTempTime, &xMinimumEverTempTime, sizeof(xMinimumTempTime));
				}
			}

			lcd_Locate(0, 0);  // go to the first character of the first LCD line
			switch( xCurrentTempTime.DateTime.Day )
			{
				case Sunday:
					lcd_Print_P(PSTR("Sunday   "));
					break;
				case Monday:
					lcd_Print_P(PSTR("Monday   "));
					break;
				case Tuesday:
					lcd_Print_P(PSTR("Tuesday  "));
					break;
				case Wednesday:
					lcd_Print_P(PSTR("Wednesday"));
					break;
				case Thursday:
					lcd_Print_P(PSTR("Thursday "));
					break;
				case Friday:
					lcd_Print_P(PSTR("Friday   "));
					break;
				case Saturday:
					lcd_Print_P(PSTR("Saturday "));
					break;
				default:
					lcd_Print_P(PSTR("Any Day  "));
					break;
			}

			// display Day Date/Month/Year
			lcd_Locate(0, 10);              // go to the eleventh character of the first LCD line
			lcd_Printf_P( PSTR("%u/%u/%u"), xCurrentTempTime.DateTime.Date, xCurrentTempTime.DateTime.Month, (xCurrentTempTime.DateTime.Year + 2000) );

			// display the current temperature
			lcd_Locate(1, 1);		// LCD cursor to third character of the second LCD line
			if ( xCurrentTempTime.Temperature != 0x7f) // don't print the temperature if you didn't get it
				lcd_Printf_P( PSTR("%3dC"), xCurrentTempTime.Temperature ); // print temperature

			// display the current time
			lcd_Locate(1, 8);             // go to the ninth character of the second LCD line
			lcd_Printf_P(PSTR("%2u:%02u:%02u"),xCurrentTempTime.DateTime.Hour, xCurrentTempTime.DateTime.Minute, xCurrentTempTime.DateTime.Second);

			// display the maximum temperature, time and date
			lcd_Locate(2, 0);          // go to the first character of the third LCD line
			lcd_Printf_P(PSTR("Max%3dC"),xMaximumTempTime.Temperature);			// print the maximum temperature value

			lcd_Locate(2, 8);          // go to the ninth character of the third LCD line
			lcd_Printf_P(PSTR("%2u:%02u %2u/%u"),xMaximumTempTime.DateTime.Hour, xMaximumTempTime.DateTime.Minute, xMaximumTempTime.DateTime.Date, xMaximumTempTime.DateTime.Month );

			// display the m temperature, time and date
			lcd_Locate(3, 0);          // go to the first character of the forth LCD line
			lcd_Printf_P(PSTR("Min%3dC"),xMinimumTempTime.Temperature);			// print the minimum temperature value

			lcd_Locate(3, 8);          // go to the ninth character of the fourth LCD line
			lcd_Printf_P(PSTR("%2u:%02u %2u/%u"),xMinimumTempTime.DateTime.Hour, xMinimumTempTime.DateTime.Minute, xMinimumTempTime.DateTime.Date, xMinimumTempTime.DateTime.Month );

			if(xCurrentTempTime.DateTime.Second == 0)
			// resume the xTaskWriteRTCRetrograde() task, now that we need to write the analogue hands.
				vTaskResume( xTaskWriteRTCRetrograde );
    	}
//		xSerialPrintf_P(PSTR("LCD HighWater @ %u\r\n"), uxTaskGetStackHighWaterMark(NULL));
        vTaskDelayUntil( &xLastWakeTime, ( 100 / portTICK_RATE_MS ) );
	}
}


static void TaskWriteRTCRetrograde(void *pvParameters) // Write RTC to Retrograde Hands
{
    (void) pvParameters;;

    uint16_t servoHours_uS = 1500;
    uint16_t servoMinutes_uS = 1500;
    uint8_t firstPass = pdTRUE;

	if( xSemaphoreTake( xADCSemaphore, portMAX_DELAY ) == pdTRUE )
	{
		// We were able to obtain the semaphore and can now access the shared resource.
		// We don't want anyone using the ADC during servo moves, so take the semaphore.
		// There is too much noise on Vcc to get a clean sample.

		start_PWM_hardware();  // start the PWM TimerX hardware depending on the Timer #define in FreeRTOSConfig.h
		// Servos driving the hands, drags the Vcc down, drastically affecting the ADC0 (temperature) reading.

		// delay 2000mS to ensure hands are stopped before releasing.
		vTaskDelay( 2000 / portTICK_RATE_MS );

		xSemaphoreGive( xADCSemaphore ); // now the ADC can be used again.
	}

    while(1)
    {
		if( firstPass == pdTRUE) // Set hour hand servo on power-on once,
		{
			// convert to a range of 700uS to 2300uS over 24 hours.
			servoHours_uS = (uint16_t)(2300 - ((float)xCurrentTempTime.DateTime.Minute + (float)xCurrentTempTime.DateTime.Hour*60 )/1439*(2300-700));
			firstPass = pdFALSE;

		} else {

			switch( xCurrentTempTime.DateTime.Minute )  // otherwise update the hour hand once every quarter hour.
			{
				case 0:
				case 15:
				case 30:
				case 45:
					// convert to a range of 700uS to 2300uS over 24 hours.
					servoHours_uS = (uint16_t)(2300 - ((float)xCurrentTempTime.DateTime.Minute + (float)xCurrentTempTime.DateTime.Hour*60 )/1439*(2300-700));
					break;
				default:
					break;
			}

		}

		// convert to a range of 700uS to 2300uS over 60 minutes.
		servoMinutes_uS = (uint16_t)(2300 - (float)xCurrentTempTime.DateTime.Minute/59*(2300-700));

		// See if we can obtain the ADC semaphore.  If the semaphore is not available
		// wait for as long as we can to see if it becomes free.
		if( xSemaphoreTake( xADCSemaphore, portMAX_DELAY ) == pdTRUE )
		{
			// We were able to obtain the semaphore and can now access the shared resource.
			// We don't want anyone using the ADC during servo moves, so take the semaphore.
			// There is too much noise on Vcc to get a clean sample.

			set_PWM_hardware( servoHours_uS, servoMinutes_uS );

			// Servos driving, drags the Vcc down, drastically affecting the ADC0 reading.
			// delay 2000mS to ensure hands are stopped before releasing.
			vTaskDelay( 2000 / portTICK_RATE_MS );
			xSemaphoreGive( xADCSemaphore ); // now the ADC can be used again.
		}

    	vTaskSuspend( NULL );	// suspend ourselves, until we're needed again.

//		xSerialPrintf_P(PSTR("RTC Servo HighWater @ %u\r\n"), uxTaskGetStackHighWaterMark(NULL));
    }
}


/*-----------------------------------------------------------*/
/* Additional helper functions */
/*-----------------------------------------------------------*/

int8_t ReadADCSensors(void) // Read ADC Sensor for Thermal LM335z
{
	// Variables for the analogue conversion on ADC Sensors
    const uint8_t samples = 20;        	// determines the number of samples taken for averaging
    uint16_t sum;               		// holds the summated samples
    uint8_t i = samples;

	if( xADCSemaphore != NULL )
	{
		// See if we can obtain the semaphore.  If the semaphore is not available
		// wait 5 ticks to see if it becomes free.

		if( xSemaphoreTake( xADCSemaphore, ( portTickType ) 5 ) == pdTRUE )
		{
			// We were able to obtain the semaphore and can now access the shared resource.
			// We want to have the ADC for us alone, as it takes some time to sample,
			// so we don't want it getting stolen during the middle of a conversion.

			do
			{
				startAnalogConversion(0, EXTERNAL_REF);   // start next conversion

				while( analogIsConverting() )
				{
					_delay_ms(1);     // wait until conversion ready
				}
				sum += analogConversionResult();	// sum the results

			} while (--i);

			xSemaphoreGive( xADCSemaphore );

			/*
			For the LM335Z we want to calculate the resistance R1 required to ensure that we have 500 uA minimum at the maximum
			temperature we intend to measure.
			Assume maximum 60C this is 273K + 60 = 333K or will be measured at 3330mV

			If Vcc is 4.9V (USB) then the R = V/I calculation gives (4.9 - 3.3)/ 0.0005 = 3200 Ohm

			This leads to using a 3200 Ohm resistor, or there about being 3300 Ohm.

			Testing gives us 0.58mA with the resistor actual at 3250 Ohm.

			Analogue testing gives with this set up: 2.952V at 20C actual... or 22C indicated

			Lets see what the Arduino ADC gives us...
			*/

			// The 497 is the Power Supply Voltage in mV / 10. 1023 is the number of ADC values.
			// 271.15 is the adjustment from Kelvin (273.15) and the offset relating to the Temp Sensor error correction.
		} else {
			return 0x7f; // no sample taken so return 0x7f.
		}
	}
	return (int8_t) (( (portFLOAT) sum * 497.0 / (portFLOAT)(1023 * samples))- 273.15);  // and return the current temp
}

/*-----------------------------------------------------------*/
/* Monitor                                                   */
/*-----------------------------------------------------------*/

static
void get_line (uint8_t *buff, uint8_t len)
{
	uint8_t c;
	uint8_t i = 0;

	for (;;) {
		xSerialGetChar(xSerialPort, &c, portMAX_DELAY);

		if (c == '\r') break;
		if ((c == '\b') && i) {
			--i;
			xSerialPutChar(xSerialPort, c, 100 / portTICK_RATE_MS);
			continue;
		}
		if (c >= ' ' && i < len - 1) {	/* Visible chars */
			buff[i++] = c;
			xSerialPutChar(xSerialPort, c, 100 / portTICK_RATE_MS);
		}
	}
	buff[i] = 0;
	xSerialPrint((uint8_t *)"\r\n");
}



