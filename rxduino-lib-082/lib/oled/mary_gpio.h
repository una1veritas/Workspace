#ifndef MARY_GPIO_H_
#define MARY_GPIO_H_

#define MARY_FAIL		0
#define MARY_SUCCESS	1

#define MARY1	1
#define MARY2	2
#define Arduino	3

//======================
// MARY1 Peripheral INIT
//======================
//OLED_CS:	PC0
#define MARY1_CS_H()	{PORTC.DR.BYTE |= (1U<<0);}			//OLED_CS(PC0) = H
#define MARY1_CS_L() 	{PORTC.DR.BYTE &= ~(1U << 0);}		//OLED_CS = L
#define MARY1_CS_OUT()	{PORTC.DDR.BYTE |= (1U << 0);}		//OLED_CS = OUTPUT
#define MARY1_CS_IN()	{PORTC.DDR.BYTE &= ~(1U << 0);}		//OLED_CS = INPUT(Maybe not in use)

//OLED_RES:	P90
#define MARY1_OLED_RES_H()		{PORT9.DR.BYTE |= (1U << 0);}		//RES(P90) = H
#define MARY1_OLED_RES_L()		{PORT9.DR.BYTE &= ~(1U << 0);}		//RES = L
#define MARY1_OLED_RES_OUT()	{PORT9.DDR.BYTE |= (1U << 0);}		//RES(P90) = OUTPUT
#define MARY1_OLED_RES_IN()		{PORT9.DDR.BYTE &= ~(1U << 0);}		//RES(P90) = INPUT(Maybe not in use)

//OLED_VCC_ON: AN5(P45)
#define MARY1_OLED_VCC_ON_H()	{PORT4.DR.BYTE	|= (1U << 5);}	//	OLED_VCC_ON(P45) = H
#define MARY1_OLED_VCC_ON_L()	{PORT4.DR.BYTE	&= ~(1U << 5);}	//	OLED_VCC_ON = L
#define MARY1_OLED_VCC_ON_OUT()	{PORT4.DDR.BYTE	|= (1U << 5);}	//	OLED_VCC_ON(P45) = OUTPUT
#define MARY1_OLED_VCC_ON_IN()	{PORT4.DDR.BYTE	&= ~(1U << 5);}	//	OLED_VCC_ON(P45) = INPUT(Maybe not in use)


//======================
// MARY2 Peripheral INIT
//======================
//OLED_CS:	PC1
#define MARY2_CS_H()	{PORTC.DR.BYTE |= (1U << 1);}		//OLED_CS(PC1) = H
#define MARY2_CS_L() 	{PORTC.DR.BYTE &= ~(1U << 1);}		//OLED_CS = L
#define MARY2_CS_OUT()	{PORTC.DDR.BYTE |= (1U << 1);}		//OLED_CS = OUTPUT
#define MARY2_CS_IN()	{PORTC.DDR.BYTE &= ~(1U << 1);}		//OLED_CS = INPUT(Maybe not in use)

//OLED_RES: P92
#define MARY2_OLED_RES_H()		{PORT9.DR.BYTE |= (1U << 2);}		//RES(P92) = H
#define MARY2_OLED_RES_L()		{PORT9.DR.BYTE &= ~(1U << 2);}		//RES = L
#define MARY2_OLED_RES_OUT()	{PORT9.DDR.BYTE |= (1U << 2);}		//RES(P92) = OUTPUT
#define MARY2_OLED_RES_IN()		{PORT9.DDR.BYTE &= ~(1U << 2);}		//RES(P92) = INPUT(Maybe not in use)

//OLED_VCC_ON:	AN6(P46)
#define MARY2_OLED_VCC_ON_H()	{PORT4.DR.BYTE	|= (1U << 6);}	//	OLED_VCC_ON(P46) = H
#define MARY2_OLED_VCC_ON_L()	{PORT4.DR.BYTE	&= ~(1U << 6);}	//	OLED_VCC_ON = L
#define MARY2_OLED_VCC_ON_OUT()	{PORT4.DDR.BYTE	|= (1U << 6);}	//	OLED_VCC_ON(P46) = OUTPUT
#define MARY2_OLED_VCC_ON_IN()	{PORT4.DDR.BYTE	&= ~(1U << 6);}	//	OLED_VCC_ON(P46) = INPUT(Maybe not in use)

#endif
