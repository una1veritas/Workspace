/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : E700_Camera.h
 *    Description : API for the E700 Camera
 *
 *    History :
 *    1. Date        : 18 Oct 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef _E700_CAMERA_H
#define _E700_CAMERA_H

// camera can shoot images with resolution upto 640 x 480
#define CAMERA_HOR_RES 132
#define CAMERA_VER_RES 132

#define CAMERA_MAX_HOR_RES 640
#define CAMERA_MAX_VER_RES 480

/******************************************************************************
* Description: E700_Camera_Initialize(..) - initializes DCMI interface, I2C software interface and configures camera
* Input: 	none
* Output: 	none
* Return:	0 if sucessfully initialized, -1 if error occured
*******************************************************************************/
int E700_Camera_Initialize(void);

/******************************************************************************
* Description: E700_Camera_Deinitialize(..) - deinitializes camera stuff
* Input: 	none
* Output: 	none
* Return:	none
*******************************************************************************/
void E700_Camera_Deinitialize(void);

/******************************************************************************
* Description: E700_Camera_CaptureImage(..) - enable capture of 1 shot and wait for capturing to complete
* Input: 	none
* Output: 	none
* Return:	pointer to the capture image raw data
*******************************************************************************/
pInt8U E700_Camera_CaptureImage(void);

/******************************************************************************
* Description: E700_Camera_ProcessImage(..) - converts 4:2:2 YCbCr values to 24bit RGB
* Input: 	pInData - input buffer containing the raw values
*			width - width in px of the image
*           height - width in px of the image
* Output: 	pOutData - output buffer to contain the converted values
* Return:	0 if sucessfully initialized, -1 if error occured
*******************************************************************************/
void E700_Camera_ProcessImage(pInt8U pInData, pInt8U pOutData, Int16U width, Int16U height);

#endif // _E700_CAMERA_H



