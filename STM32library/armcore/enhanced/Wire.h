/*
 * Wire.h
 *
 *  Created on: 2012/11/03
 *      Author: sin
 */

#ifndef WIRE_H_
#define WIRE_H_

#include "stm32f4xx_i2c.h"
#include "i2c.h"

#define BUFFER_LENGTH 64

class TwoWire {
	I2C_TypeDef * I2Cx;
	uint8_t dstaddr;
	uint16_t txindex;
	uint8_t txbuffer[BUFFER_LENGTH];

public:
	TwoWire() {
		I2Cx = I2C1;
		dstaddr = 0;
		txindex = 0;
	}

    void begin(uint32_t clk = 100000L) {
    	i2c_begin(I2Cx, clk);
    }
    void beginTransmission(uint8_t addr) {
    	dstaddr = addr;
    	txindex = 0;
    }
    size_t write(uint8_t d) {
    	txbuffer[txindex++] = d;
    	return 1;
    }

    size_t write(const uint8_t * ptr, size_t n) {
    	size_t sz = n;
    	while(n--) {
    		txbuffer[txindex++] = *ptr++;
    	}
    	return sz;
    }

    uint8_t endTransmission(void) {
    	i2c_send(I2Cx, dstaddr, txbuffer, txindex);
    	dstaddr = 0;
    	return 1;
    }
	/*
  private:
    static uint8_t rxBuffer[];
    static uint8_t rxBufferIndex;
    static uint8_t rxBufferLength;

    static uint8_t txAddress;
    static uint8_t txBuffer[];
    static uint8_t txBufferIndex;
    static uint8_t txBufferLength;

    static uint8_t transmitting;
    static void (*user_onRequest)(void);
    static void (*user_onReceive)(int);
    static void onRequestService(void);
    static void onReceiveService(uint8_t*, int);
  public:
    TwoWire();
    void begin();
    void begin(uint8_t);
    void begin(int);
    void beginTransmission(uint8_t);
    void beginTransmission(int);
    uint8_t endTransmission(void);
    uint8_t endTransmission(uint8_t);
    uint8_t requestFrom(uint8_t, uint8_t);
    uint8_t requestFrom(uint8_t, uint8_t, uint8_t);
    uint8_t requestFrom(int, int);
    uint8_t requestFrom(int, int, int);
    virtual size_t write(uint8_t);
    virtual size_t write(const uint8_t *, size_t);
    virtual int available(void);
    virtual int read(void);
    virtual int peek(void);
	virtual void flush(void);
    void onReceive( void (*)(int) );
    void onRequest( void (*)(void) );

    inline size_t write(unsigned long n) { return write((uint8_t)n); }
    inline size_t write(long n) { return write((uint8_t)n); }
    inline size_t write(unsigned int n) { return write((uint8_t)n); }
    inline size_t write(int n) { return write((uint8_t)n); }
    using Print::write;
	 */
};

extern TwoWire Wire;

#endif /* WIRE_H_ */
