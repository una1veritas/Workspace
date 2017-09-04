/**** A V R  A P P L I C A T I O N  NOTE 1 3 4 ************************** 
 *
 * Title:           Real Time Clock
 * Version:         1.01
 * Last Updated:    12.10.98
 * Target:          ATmega103 (All AVR Devices with secondary external oscillator)
 *
 * Support E-mail:  avr@atmel.com
 *
 * Description      
 * This application note shows how to implement a Real Time Clock utilizing a secondary 
 * external oscilator. Included a test program that performs this function, which keeps
 * track of time, date, month, and year with auto leap-year configuration. 8 LEDs are used
 * to display the RTC. The 1st LED flashes every second, the next six represents the
 * minute, and the 8th LED represents the hour.
 *
 ******************************************************************************************/ 

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>

char not_leap(void);

typedef struct{ 
unsigned char second;   //enter the current time, date, month, and year
unsigned char minute;
unsigned char hour;                                     
unsigned char date;       
unsigned char month;
unsigned int year;      
            }time;

 time t;      
                                                        
int main(void)   // C_task means "main" is never called from another function
{                                
//    init_rtc();
    int temp0,temp1;   
     
    for(temp0=0;temp0<0x0040;temp0++)   // Wait for external clock crystal to stabilize
    {
        for(temp1=0;temp1<0xFFFF;temp1++);
    }
    DDRB=0xFF;

//    TIMSK &=~((1<<TOIE0)|(1<<OCIE0));     //Disable TC0 interrupt
    TIMSK2 &=~((1<<TOIE2)|(1<<OCIE2A)|(1<<OCIE2B));     //Disable TC0 interrupt
    ASSR |= (1<<AS2);           //set Timer/Counter0 to be asynchronous from the CPU clock 
                                //with a second external clock(32,768kHz)driving it.  
    TCNT2 = 0x00;
    TCCR2B = 0x05;                 //prescale the timer to be clock source / 128 to make it
                                //exactly 1 second for every overflow to occur
    while(ASSR&0x07);           //Wait until TC0 is updated
    TIMSK2 |= (1<<TOIE0);        //set 8-bit Timer/Counter0 Overflow Interrupt Enable                             
    sei();                     //set the Global Interrupt Enable Bit  
                              
    while(1)                     
    {
//        MCUCR = 0x38;           //entering sleeping mode: power save mode
//		sleep_enable();
//        sleep_mode();              //will wake up from time overflow interrupt  
        asm volatile("nop"); //_NOP();
        TCCR2B=0x05;           // Write dummy value to Control register
		PORTB = 2;
        while(ASSR&0x07);     //Wait until TC0 is updated
		PORTB = 0;
    }            
}

ISR (TIMER0_OVF_vect) { // void counter(void) //overflow interrupt vector
//{ 
    
    if (++t.second==60)        //keep track of time, date, month, and year
    {
        t.second=0;
        if (++t.minute==60) 
        {
            t.minute=0;
            if (++t.hour==24)
            {
                t.hour=0;
                if (++t.date==32)
                {
                    t.month++;
                    t.date=1;
                }
                else if (t.date==31) 
                {                    
                    if ((t.month==4) || (t.month==6) || (t.month==9) || (t.month==11)) 
                    {
                        t.month++;
                        t.date=1;
                    }
                }
                else if (t.date==30)
                {
                    if(t.month==2)
                    {
                       t.month++;
                       t.date=1;
                    }
                }              
                else if (t.date==29) 
                {
                    if((t.month==2) && (not_leap()))
                    {
                        t.month++;
                        t.date=1;
                    }                
                }                          
                if (t.month==13)
                {
                    t.month=1;
                    t.year++;
                }
            }
        }
    }  
    PORTB=~(((t.second&0x01)|t.minute<<1)|t.hour<<7); 


}  
 
char not_leap(void)      //check for leap year
{
    if (!(t.year%100))
        return (char)(t.year%400);
    else
        return (char)(t.year%4);
}         
  
          
    
        
