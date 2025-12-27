/*
 * Copyright (c) 2023 @hanyazou
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __MODEM_XFER_DEBUG_H__
#define __MODEM_XFER_DEBUG_H__

#define  err(args...) do { modem_xfer_printf(MODEM_XFER_LOG_ERROR,   args); } while(0)
#define warn(args...) do { modem_xfer_printf(MODEM_XFER_LOG_WARNING, args); } while(0)
#define info(args...) do { modem_xfer_printf(MODEM_XFER_LOG_INFO,    args); } while(0)
#ifdef DEBUG
#define  dbg(args...) do { modem_xfer_printf(MODEM_XFER_LOG_DEBUG,   args); } while(0)
#else
#define  dbg(args...) do { } while(0)
#endif

#endif  // __MODEM_XFER_H__
