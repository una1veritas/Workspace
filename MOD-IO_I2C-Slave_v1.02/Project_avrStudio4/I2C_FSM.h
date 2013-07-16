/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : I2C_FSM.h
 *    Description : implements the I2C FSM
 *
 *    History :
 *    1. Date        : 07 November 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef I2C_FSM_H
#define I2C_FSM_H

/******************************************************************************
* Description: I2C_FSM_Initialize(..) - initializes I2C interface and FSM
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char I2C_FSM_Initialize(void);

/******************************************************************************
* Description: I2C_FSM_Refresh(..) - call periodically to run FSM
* Input: 	none
* Output: 	none
* Return:	none
*******************************************************************************/
void I2C_FSM_Refresh(void);

#endif // I2C_FSM_H
