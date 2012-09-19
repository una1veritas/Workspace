/**
  ******************************************************************************
  * @file    lib_std/UTIL/src/comm.c
  * @author  Martin Thomas, Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   communication capsule function for term_io.c/.h
  ******************************************************************************
  * @copy
  *
  * This library is made by Martin Thomas. Yasuo Kawachi made small
  * modification to it.
  *
  * Copyright 2008-2009 Martin Thomas and Yasuo Kawachi All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *  1. Redistributions of source code must retain the above copyright notice,
  *  this list of conditions and the following disclaimer.
  *  2. Redistributions in binary form must reproduce the above copyright notice,
  *  this list of conditions and the following disclaimer in the documentation
  *  and/or other materials provided with the distribution.
  *  3. Neither the name of the copyright holders nor the names of contributors
  *  may be used to endorse or promote products derived from this software
  *  without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY MARTIN THOMAS OR CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL YASUO KAWACHI OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

#include "stm32f10x.h"
#include "platform_config.h"
#include "comm.h"

#ifdef USE_VCP
#include "usb_config.h"
#elif defined USE_USART1
#include "usart_config.h"
#elif defined USE_USART2
#include "usart_config.h"
#elif defined USE_USART3
#include "usart_config.h"
#endif

int comm_test (void)
{
	return ( RX_BUFFER_IS_EMPTY ) ? 0 : 1;
}

unsigned char comm_get (void)
{
	while(RX_BUFFER_IS_EMPTY) { ; }
	return (unsigned char)RECEIVE_DATA;
}

void comm_put (unsigned char d)
{
	cputchar(d);

}

void comm_init (void)
{
	// already done in main.c
}


