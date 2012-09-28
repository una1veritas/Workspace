// - ----------------------------------------------------------------------------
// -          ATMEL Microcontroller Software Support  -  ROUSSET  -
// - ----------------------------------------------------------------------------
// -  DISCLAIMER:  THIS SOFTWARE IS PROVIDED BY ATMEL "AS IS" AND ANY EXPRESS OR
// -  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// -  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT ARE
// -  DISCLAIMED. IN NO EVENT SHALL ATMEL BE LIABLE FOR ANY DIRECT, INDIRECT,
// -  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// -  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// -  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// -  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// -  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// - ----------------------------------------------------------------------------
// - File Name           : AT91SAM9G45.h
// - Object              : AT91SAM9G45 definitions
// - Generated           : AT91 SW Application Group  06/25/2009 (11:40:50)
// - 
// - CVS Reference       : /AT91SAM9G45.pl/1.1/Wed May 20 12:05:51 2009//
// - CVS Reference       : /SYS_SAM9264.pl/1.2/Tue Oct  2 12:19:30 2007//
// - CVS Reference       : /HMATRIX2_SAM9G45.pl/1.1/Wed May 20 09:31:20 2009//
// - CVS Reference       : /PMC_SAM9264.pl/1.9/Thu Nov 29 09:55:11 2007//
// - CVS Reference       : /HDDRSDRC2_6304B.pl/1.1/Thu Oct 11 10:44:49 2007//
// - CVS Reference       : /EBI_SAM9260.pl/1.1/Fri Sep 30 12:12:14 2005//
// - CVS Reference       : /HSMC3_SAM9264.pl/1.1/Wed Oct 10 09:39:31 2007//
// - CVS Reference       : /HECC_6143A.pl/1.1/Wed Feb  9 17:16:57 2005//
// - CVS Reference       : /SFR_SAM9264.pl/1.4/Fri Feb 29 14:02:55 2008//
// - CVS Reference       : /AIC_6075A.pl/1.1/Mon Jul 12 17:04:01 2004//
// - CVS Reference       : /PDC_6074C.pl/1.2/Thu Feb  3 09:02:11 2005//
// - CVS Reference       : /DBGU_6059D.pl/1.1/Mon Jan 31 13:54:41 2005//
// - CVS Reference       : /PIO_SAM9264.pl/1.1/Wed Oct 10 09:38:26 2007//
// - CVS Reference       : /RSTC_6098A.pl/1.4/Fri Oct 17 13:27:55 2008//
// - CVS Reference       : /SHDWC_6122A.pl/1.3/Wed Oct  6 14:16:58 2004//
// - CVS Reference       : /RTTC_6081A.pl/1.2/Thu Nov  4 13:57:22 2004//
// - CVS Reference       : /PITC_6079A.pl/1.2/Thu Nov  4 13:56:22 2004//
// - CVS Reference       : /WDTC_6080A.pl/1.3/Thu Nov  4 13:58:52 2004//
// - CVS Reference       : /TC_6082A.pl/1.8/Fri Oct 17 13:27:58 2008//
// - CVS Reference       : /MCI_6101F.pl/1.3/Fri Jan 23 09:15:32 2009//
// - CVS Reference       : /TWI_6061B.pl/1.3/Fri Oct 17 13:27:59 2008//
// - CVS Reference       : /US_6089J.pl/1.3/Fri Oct 17 13:27:59 2008//
// - CVS Reference       : /SSC_6078B.pl/1.3/Fri Oct 17 13:27:57 2008//
// - CVS Reference       : /SPI_6088D.pl/1.3/Fri May 20 14:23:02 2005//
// - CVS Reference       : /AC97C_XXXX.pl/1.3/Tue Feb 22 17:08:27 2005//
// - CVS Reference       : /PWM_6044D.pl/1.2/Tue May 10 12:39:09 2005//
// - CVS Reference       : /LCDC_6063A.pl/1.6/Tue Jan 20 16:29:59 2009//
// - CVS Reference       : /HDMA_SAM9264.pl/1.2/Thu Sep 13 11:48:30 2007//
// - CVS Reference       : /UDPHS_SAM9_7ept6dma4iso.pl/1.4/Tue Jun 24 13:05:14 2008//
// - CVS Reference       : /TSC_SAM9264.pl/1.2/Thu Jun 25 08:43:26 2009//
// - CVS Reference       : /RTC_1245D.pl/1.3/Fri Sep 17 14:01:31 2004//
// - CVS Reference       : /EMACB_SAM9264.pl/1.1/Tue Sep 25 12:07:23 2007//
// - CVS Reference       : /uhphs_ohci.pl/1.1/Fri Jun 22 14:20:34 2007//
// - CVS Reference       : /uhphs_ehci.pl/1.3/Tue Jul 17 07:50:29 2007//
// - CVS Reference       : /ISI_SAM9264.pl/1.2/Wed Sep  3 08:30:55 2008//
// - CVS Reference       : /TRNG_xxxxx.pl/1.1/Wed Jul 18 12:02:58 2007//
// - ----------------------------------------------------------------------------

#ifndef AT91SAM9G45_H
#define AT91SAM9G45_H

#ifdef __IAR_SYSTEMS_ICC__

typedef volatile unsigned int AT91_REG;// Hardware register definition

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR SPECIAL FUNCTION REGISTER
// *****************************************************************************
typedef struct _AT91S_SFR {
	AT91_REG	 SFR_EMA; 	// memory Extra Margin Adjustment control
	AT91_REG	 SFR_DDRCFG; 	// DDR2 SSTL18 control
	AT91_REG	 SFR_DDRDELAY; 	// DDR2 controller clock delay
	AT91_REG	 SFR_EBIDELAY; 	// EBI DDR controller clock delay
	AT91_REG	 SFR_UTMICFG; 	// UTMI Software Reset, and OHCI suspend interrupt control
	AT91_REG	 SFR_INT; 	// OHCI suspend Interrupt status
} AT91S_SFR, *AT91PS_SFR;

// -------- SFR_EMA : (SFR Offset: 0x0) memory Extra Margin Adjustment control register -------- 
#define AT91C_SFR_RAM_EMA     ((unsigned int) 0x7 <<  0) // (SFR) SRAM EMA
#define 	AT91C_SFR_RAM_EMA_0                    ((unsigned int) 0x0) // (SFR) Normal Mode
#define 	AT91C_SFR_RAM_EMA_1                    ((unsigned int) 0x1) // (SFR) DEBUG MODE 1
#define 	AT91C_SFR_RAM_EMA_2                    ((unsigned int) 0x2) // (SFR) DEBUG MODE 2
#define 	AT91C_SFR_RAM_EMA_3                    ((unsigned int) 0x3) // (SFR) DEBUG MODE 3
#define 	AT91C_SFR_RAM_EMA_4                    ((unsigned int) 0x4) // (SFR) DEBUG MODE 4
#define 	AT91C_SFR_RAM_EMA_5                    ((unsigned int) 0x5) // (SFR) DEBUG MODE 5
#define 	AT91C_SFR_RAM_EMA_6                    ((unsigned int) 0x6) // (SFR) DEBUG MODE 6
#define 	AT91C_SFR_RAM_EMA_7                    ((unsigned int) 0x7) // (SFR) DEBUG MODE 7
#define AT91C_SFR_DPRAM_EMA   ((unsigned int) 0x7 <<  4) // (SFR) SRAM EMA
#define 	AT91C_SFR_DPRAM_EMA_0                    ((unsigned int) 0x0 <<  4) // (SFR) Normal Mode
#define 	AT91C_SFR_DPRAM_EMA_1                    ((unsigned int) 0x1 <<  4) // (SFR) DEBUG MODE 1
#define 	AT91C_SFR_DPRAM_EMA_2                    ((unsigned int) 0x2 <<  4) // (SFR) DEBUG MODE 2
#define 	AT91C_SFR_DPRAM_EMA_3                    ((unsigned int) 0x3 <<  4) // (SFR) DEBUG MODE 3
#define 	AT91C_SFR_DPRAM_EMA_4                    ((unsigned int) 0x4 <<  4) // (SFR) DEBUG MODE 4
#define 	AT91C_SFR_DPRAM_EMA_5                    ((unsigned int) 0x5 <<  4) // (SFR) DEBUG MODE 5
#define 	AT91C_SFR_DPRAM_EMA_6                    ((unsigned int) 0x6 <<  4) // (SFR) DEBUG MODE 6
#define 	AT91C_SFR_DPRAM_EMA_7                    ((unsigned int) 0x7 <<  4) // (SFR) DEBUG MODE 7
#define AT91C_SFR_RF_EMA      ((unsigned int) 0x7 <<  8) // (SFR) SRAM EMA
#define 	AT91C_SFR_RF_EMA_0                    ((unsigned int) 0x0 <<  8) // (SFR) Normal Mode
#define 	AT91C_SFR_RF_EMA_1                    ((unsigned int) 0x1 <<  8) // (SFR) DEBUG MODE 1
#define 	AT91C_SFR_RF_EMA_2                    ((unsigned int) 0x2 <<  8) // (SFR) DEBUG MODE 2
#define 	AT91C_SFR_RF_EMA_3                    ((unsigned int) 0x3 <<  8) // (SFR) DEBUG MODE 3
#define 	AT91C_SFR_RF_EMA_4                    ((unsigned int) 0x4 <<  8) // (SFR) DEBUG MODE 4
#define 	AT91C_SFR_RF_EMA_5                    ((unsigned int) 0x5 <<  8) // (SFR) DEBUG MODE 5
#define 	AT91C_SFR_RF_EMA_6                    ((unsigned int) 0x6 <<  8) // (SFR) DEBUG MODE 6
#define 	AT91C_SFR_RF_EMA_7                    ((unsigned int) 0x7 <<  8) // (SFR) DEBUG MODE 7
#define AT91C_SFR_DPRF_EMA    ((unsigned int) 0x7 << 12) // (SFR) SRAM EMA
#define 	AT91C_SFR_DPRF_EMA_0                    ((unsigned int) 0x0 << 12) // (SFR) Normal Mode
#define 	AT91C_SFR_DPRF_EMA_1                    ((unsigned int) 0x1 << 12) // (SFR) DEBUG MODE 1
#define 	AT91C_SFR_DPRF_EMA_2                    ((unsigned int) 0x2 << 12) // (SFR) DEBUG MODE 2
#define 	AT91C_SFR_DPRF_EMA_3                    ((unsigned int) 0x3 << 12) // (SFR) DEBUG MODE 3
#define 	AT91C_SFR_DPRF_EMA_4                    ((unsigned int) 0x4 << 12) // (SFR) DEBUG MODE 4
#define 	AT91C_SFR_DPRF_EMA_5                    ((unsigned int) 0x5 << 12) // (SFR) DEBUG MODE 5
#define 	AT91C_SFR_DPRF_EMA_6                    ((unsigned int) 0x6 << 12) // (SFR) DEBUG MODE 6
#define 	AT91C_SFR_DPRF_EMA_7                    ((unsigned int) 0x7 << 12) // (SFR) DEBUG MODE 7
#define AT91C_SFR_ROM_EMA     ((unsigned int) 0x7 << 16) // (SFR) SRAM EMA
#define 	AT91C_SFR_ROM_EMA_0                    ((unsigned int) 0x0 << 16) // (SFR) Normal Mode
#define 	AT91C_SFR_ROM_EMA_1                    ((unsigned int) 0x1 << 16) // (SFR) DEBUG MODE 1
#define 	AT91C_SFR_ROM_EMA_2                    ((unsigned int) 0x2 << 16) // (SFR) DEBUG MODE 2
#define 	AT91C_SFR_ROM_EMA_3                    ((unsigned int) 0x3 << 16) // (SFR) DEBUG MODE 3
#define 	AT91C_SFR_ROM_EMA_4                    ((unsigned int) 0x4 << 16) // (SFR) DEBUG MODE 4
#define 	AT91C_SFR_ROM_EMA_5                    ((unsigned int) 0x5 << 16) // (SFR) DEBUG MODE 5
#define 	AT91C_SFR_ROM_EMA_6                    ((unsigned int) 0x6 << 16) // (SFR) DEBUG MODE 6
#define 	AT91C_SFR_ROM_EMA_7                    ((unsigned int) 0x7 << 16) // (SFR) DEBUG MODE 7
// -------- SFR_DDRCFG : (SFR Offset: 0x4) DDR2 SSTL18 control register -------- 
#define AT91C_SFR_DDRCFG_SSTL ((unsigned int) 0x1 <<  0) // (SFR) Control DDR2 pads SSTL mode control
#define 	AT91C_SFR_DDRCFG_SSTL_NORMAL               ((unsigned int) 0x0) // (SFR) Force pads in SSTL18 mode when DDR2 is connected
#define 	AT91C_SFR_DDRCFG_SSTL_COMPATIBLE           ((unsigned int) 0x1) // (SFR) LVCMOS level (compatible SSTL18)
#define AT91C_SFR_DDRCFG_CLKDELAY ((unsigned int) 0x1 <<  8) // (SFR) Control DDR2 pads clocks delay on clk, dqs0, dqs1
#define 	AT91C_SFR_DDRCFG_CLKDELAY_HARD                 ((unsigned int) 0x0 <<  8) // (SFR) Fixed by hardware
#define 	AT91C_SFR_DDRCFG_CLKDELAY_SOFT                 ((unsigned int) 0x1 <<  8) // (SFR) Software must write correct delay value
// -------- SFR_DDRDELAY : (SFR Offset: 0x8) DDR2 controller clock delay -------- 
#define AT91C_SFR_DDRDELAY_CLK ((unsigned int) 0xFF <<  0) // (SFR) Control CLK clock delay
#define 	AT91C_SFR_DDRDELAY_CLK_0                    ((unsigned int) 0x0) // (SFR) minimum delay
#define 	AT91C_SFR_DDRDELAY_CLK_1                    ((unsigned int) 0x1) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_2                    ((unsigned int) 0x2) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_3                    ((unsigned int) 0x3) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_4                    ((unsigned int) 0x4) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_5                    ((unsigned int) 0x5) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_6                    ((unsigned int) 0x6) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_7                    ((unsigned int) 0x7) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_8                    ((unsigned int) 0x8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_9                    ((unsigned int) 0x9) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_10                   ((unsigned int) 0xA) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_11                   ((unsigned int) 0xB) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_12                   ((unsigned int) 0xC) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_13                   ((unsigned int) 0xD) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_14                   ((unsigned int) 0xE) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_15                   ((unsigned int) 0xF) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_CLK_16                   ((unsigned int) 0x10) // (SFR) 
#define AT91C_SFR_DDRDELAY_DQS0 ((unsigned int) 0xFF <<  8) // (SFR) Control DQS0 clock delay
#define 	AT91C_SFR_DDRDELAY_DQS0_0                    ((unsigned int) 0x0 <<  8) // (SFR) minimum delay
#define 	AT91C_SFR_DDRDELAY_DQS0_1                    ((unsigned int) 0x1 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_2                    ((unsigned int) 0x2 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_3                    ((unsigned int) 0x3 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_4                    ((unsigned int) 0x4 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_5                    ((unsigned int) 0x5 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_6                    ((unsigned int) 0x6 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_7                    ((unsigned int) 0x7 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_8                    ((unsigned int) 0x8 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_9                    ((unsigned int) 0x9 <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_10                   ((unsigned int) 0xA <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_11                   ((unsigned int) 0xB <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_12                   ((unsigned int) 0xC <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_13                   ((unsigned int) 0xD <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_14                   ((unsigned int) 0xE <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_15                   ((unsigned int) 0xF <<  8) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS0_16                   ((unsigned int) 0x10 <<  8) // (SFR) 
#define AT91C_SFR_DDRDELAY_DQS1 ((unsigned int) 0xFF << 16) // (SFR) Control DQS1 clock delay
#define 	AT91C_SFR_DDRDELAY_DQS1_0                    ((unsigned int) 0x0 << 16) // (SFR) minimum delay
#define 	AT91C_SFR_DDRDELAY_DQS1_1                    ((unsigned int) 0x1 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_2                    ((unsigned int) 0x2 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_3                    ((unsigned int) 0x3 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_4                    ((unsigned int) 0x4 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_5                    ((unsigned int) 0x5 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_6                    ((unsigned int) 0x6 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_7                    ((unsigned int) 0x7 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_8                    ((unsigned int) 0x8 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_9                    ((unsigned int) 0x9 << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_10                   ((unsigned int) 0xA << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_11                   ((unsigned int) 0xB << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_12                   ((unsigned int) 0xC << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_13                   ((unsigned int) 0xD << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_14                   ((unsigned int) 0xE << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_15                   ((unsigned int) 0xF << 16) // (SFR) 
#define 	AT91C_SFR_DDRDELAY_DQS1_16                   ((unsigned int) 0x10 << 16) // (SFR) 
// -------- SFR_EBIDELAY : (SFR Offset: 0xc) EBI DDR controller clock delay -------- 
#define AT91C_SFR_EBIDELAY_CLK ((unsigned int) 0xFF <<  0) // (SFR) Control CLK clock delay
#define 	AT91C_SFR_EBIDELAY_CLK_0                    ((unsigned int) 0x0) // (SFR) minimum delay
#define 	AT91C_SFR_EBIDELAY_CLK_1                    ((unsigned int) 0x1) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_2                    ((unsigned int) 0x2) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_3                    ((unsigned int) 0x3) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_4                    ((unsigned int) 0x4) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_5                    ((unsigned int) 0x5) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_6                    ((unsigned int) 0x6) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_7                    ((unsigned int) 0x7) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_8                    ((unsigned int) 0x8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_9                    ((unsigned int) 0x9) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_10                   ((unsigned int) 0xA) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_11                   ((unsigned int) 0xB) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_12                   ((unsigned int) 0xC) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_13                   ((unsigned int) 0xD) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_14                   ((unsigned int) 0xE) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_15                   ((unsigned int) 0xF) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_CLK_16                   ((unsigned int) 0x10) // (SFR) 
#define AT91C_SFR_EBIDELAY_DQS0 ((unsigned int) 0xFF <<  8) // (SFR) Control DQS0 clock delay
#define 	AT91C_SFR_EBIDELAY_DQS0_0                    ((unsigned int) 0x0 <<  8) // (SFR) minimum delay
#define 	AT91C_SFR_EBIDELAY_DQS0_1                    ((unsigned int) 0x1 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_2                    ((unsigned int) 0x2 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_3                    ((unsigned int) 0x3 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_4                    ((unsigned int) 0x4 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_5                    ((unsigned int) 0x5 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_6                    ((unsigned int) 0x6 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_7                    ((unsigned int) 0x7 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_8                    ((unsigned int) 0x8 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_9                    ((unsigned int) 0x9 <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_10                   ((unsigned int) 0xA <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_11                   ((unsigned int) 0xB <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_12                   ((unsigned int) 0xC <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_13                   ((unsigned int) 0xD <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_14                   ((unsigned int) 0xE <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_15                   ((unsigned int) 0xF <<  8) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS0_16                   ((unsigned int) 0x10 <<  8) // (SFR) 
#define AT91C_SFR_EBIDELAY_DQS1 ((unsigned int) 0xFF << 16) // (SFR) Control DQS1 clock delay
#define 	AT91C_SFR_EBIDELAY_DQS1_0                    ((unsigned int) 0x0 << 16) // (SFR) minimum delay
#define 	AT91C_SFR_EBIDELAY_DQS1_1                    ((unsigned int) 0x1 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_2                    ((unsigned int) 0x2 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_3                    ((unsigned int) 0x3 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_4                    ((unsigned int) 0x4 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_5                    ((unsigned int) 0x5 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_6                    ((unsigned int) 0x6 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_7                    ((unsigned int) 0x7 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_8                    ((unsigned int) 0x8 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_9                    ((unsigned int) 0x9 << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_10                   ((unsigned int) 0xA << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_11                   ((unsigned int) 0xB << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_12                   ((unsigned int) 0xC << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_13                   ((unsigned int) 0xD << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_14                   ((unsigned int) 0xE << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_15                   ((unsigned int) 0xF << 16) // (SFR) 
#define 	AT91C_SFR_EBIDELAY_DQS1_16                   ((unsigned int) 0x10 << 16) // (SFR) 
// -------- SFR_UTMICFG : (SFR Offset: 0x10) UTMI Software Reset -------- 
#define AT91C_SFR_UTMICFG_PORT0 ((unsigned int) 0x1 <<  0) // (SFR) UTMI Software Reset port 0
#define AT91C_SFR_UTMICFG_PORT1 ((unsigned int) 0x1 <<  1) // (SFR) UTMI Software Reset port 1
#define AT91C_SFR_UTMICFG_OHCI_SUSP_INT_ENABLE ((unsigned int) 0x1 <<  2) // (SFR) OHCI Suspend Interrupt enable
// -------- SFR_OHCI_SUSP_INT : (SFR Offset: 0x14) OHCI suspend Interrupt status -------- 
#define AT91C_SFR_OHCI_SUSP_INT_STATUS ((unsigned int) 0x3 <<  0) // (SFR) OHCI suspend Interrupt status
#define 	AT91C_SFR_OHCI_SUSP_INT_STATUS_PORT0                ((unsigned int) 0x1) // (SFR) OHCI suspend Interrupt status for port 0
#define 	AT91C_SFR_OHCI_SUSP_INT_STATUS_PORT1                ((unsigned int) 0x2) // (SFR) OHCI suspend Interrupt status for port 1

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR System Peripherals
// *****************************************************************************
typedef struct _AT91S_SYS {
	AT91_REG	 Reserved0[3904]; 	// 
	AT91_REG	 SYS_RSTC_RCR; 	// Reset Control Register
	AT91_REG	 SYS_RSTC_RSR; 	// Reset Status Register
	AT91_REG	 SYS_RSTC_RMR; 	// Reset Mode Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 SYS_SHDWC_SHCR; 	// Shut Down Control Register
	AT91_REG	 SYS_SHDWC_SHMR; 	// Shut Down Mode Register
	AT91_REG	 SYS_SHDWC_SHSR; 	// Shut Down Status Register
	AT91_REG	 Reserved2[1]; 	// 
	AT91_REG	 SYS_RTTC0_RTMR; 	// Real-time Mode Register
	AT91_REG	 SYS_RTTC0_RTAR; 	// Real-time Alarm Register
	AT91_REG	 SYS_RTTC0_RTVR; 	// Real-time Value Register
	AT91_REG	 SYS_RTTC0_RTSR; 	// Real-time Status Register
	AT91_REG	 SYS_PITC_PIMR; 	// Period Interval Mode Register
	AT91_REG	 SYS_PITC_PISR; 	// Period Interval Status Register
	AT91_REG	 SYS_PITC_PIVR; 	// Period Interval Value Register
	AT91_REG	 SYS_PITC_PIIR; 	// Period Interval Image Register
	AT91_REG	 SYS_WDTC_WDCR; 	// Watchdog Control Register
	AT91_REG	 SYS_WDTC_WDMR; 	// Watchdog Mode Register
	AT91_REG	 SYS_WDTC_WDSR; 	// Watchdog Status Register
	AT91_REG	 Reserved3[1]; 	// 
	AT91_REG	 SYS_SLCKSEL; 	// Slow Clock Selection Register
	AT91_REG	 Reserved4[3]; 	// 
	AT91_REG	 SYS_GPBR[4]; 	// General Purpose Register
	AT91_REG	 Reserved5[16]; 	// 
	AT91_REG	 RTC_CR; 	// Control Register
	AT91_REG	 RTC_MR; 	// Mode Register
	AT91_REG	 RTC_TIMR; 	// Time Register
	AT91_REG	 RTC_CALR; 	// Calendar Register
	AT91_REG	 RTC_TIMALR; 	// Time Alarm Register
	AT91_REG	 RTC_CALALR; 	// Calendar Alarm Register
	AT91_REG	 RTC_SR; 	// Status Register
	AT91_REG	 RTC_SCCR; 	// Status Clear Command Register
	AT91_REG	 RTC_IER; 	// Interrupt Enable Register
	AT91_REG	 RTC_IDR; 	// Interrupt Disable Register
	AT91_REG	 RTC_IMR; 	// Interrupt Mask Register
	AT91_REG	 RTC_VER; 	// Valid Entry Register
	AT91_REG	 Reserved6[7]; 	// 
	AT91_REG	 SYS_RSTC_VER; 	// Version Register
} AT91S_SYS, *AT91PS_SYS;

// -------- SLCKSEL : (SYS Offset: 0x3d50) Slow Clock Selection Register -------- 
#define AT91C_SLCKSEL_RCEN    ((unsigned int) 0x1 <<  0) // (SYS) Enable Internal RC Oscillator
#define AT91C_SLCKSEL_OSC32EN ((unsigned int) 0x1 <<  1) // (SYS) Enable External Oscillator
#define AT91C_SLCKSEL_OSC32BYP ((unsigned int) 0x1 <<  2) // (SYS) Bypass External Oscillator
#define AT91C_SLCKSEL_OSCSEL  ((unsigned int) 0x1 <<  3) // (SYS) OSC Selection
// -------- GPBR : (SYS Offset: 0x3d60) GPBR General Purpose Register -------- 
#define AT91C_GPBR_GPRV       ((unsigned int) 0x0 <<  0) // (SYS) General Purpose Register Value

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR External Bus Interface
// *****************************************************************************
typedef struct _AT91S_EBI {
	AT91_REG	 EBI_DUMMY; 	// Dummy register - Do not use
} AT91S_EBI, *AT91PS_EBI;


// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR DDR2/SDRAM Controller
// *****************************************************************************
typedef struct _AT91S_HDDRSDRC2 {
	AT91_REG	 HDDRSDRC2_MR; 	// Mode Register
	AT91_REG	 HDDRSDRC2_RTR; 	// Refresh Timer Register
	AT91_REG	 HDDRSDRC2_CR; 	// Configuration Register
	AT91_REG	 HDDRSDRC2_T0PR; 	// Timing0 Register
	AT91_REG	 HDDRSDRC2_T1PR; 	// Timing1 Register
	AT91_REG	 HDDRSDRC2_T2PR; 	// Timing2 Register
	AT91_REG	 HDDRSDRC2_T3PR; 	// Timing3 Register
	AT91_REG	 HDDRSDRC2_LPR; 	// Low-power Register
	AT91_REG	 HDDRSDRC2_MDR; 	// Memory Device Register
	AT91_REG	 HDDRSDRC2_DLL; 	// DLL Information Register
	AT91_REG	 HDDRSDRC2_VER; 	// DLL Version Register
	AT91_REG	 HDDRSDRC2_HS; 	// High Speed Register
	AT91_REG	 HDDRSDRC2_DELAY1; 	// Pad delay1 Register
	AT91_REG	 HDDRSDRC2_DELAY2; 	// Pad delay2 Register
	AT91_REG	 HDDRSDRC2_DELAY3; 	// Pad delay3 Register
	AT91_REG	 HDDRSDRC2_DELAY4; 	// Pad delay4 Register
	AT91_REG	 HDDRSDRC2_DELAY5; 	// Pad delay5 Register
	AT91_REG	 HDDRSDRC2_DELAY6; 	// Pad delay6 Register
	AT91_REG	 HDDRSDRC2_DELAY7; 	// Pad delay7 Register
	AT91_REG	 HDDRSDRC2_DELAY8; 	// Pad delay8 Register
	AT91_REG	 Reserved0[37]; 	// 
	AT91_REG	 HDDRSDRC2_WPCR; 	// High Speed Register
	AT91_REG	 HDDRSDRC2_WPSR; 	// High Speed Register
	AT91_REG	 Reserved1[4]; 	// 
	AT91_REG	 HDDRSDRC2_VERSION; 	// Version Register
} AT91S_HDDRSDRC2, *AT91PS_HDDRSDRC2;

// -------- HDDRSDRC2_MR : (HDDRSDRC2 Offset: 0x0) Mode Register -------- 
#define AT91C_DDRC2_MODE      ((unsigned int) 0x7 <<  0) // (HDDRSDRC2) DDR/SDRAM Command Mode
#define 	AT91C_DDRC2_MODE_NORMAL_CMD           ((unsigned int) 0x0) // (HDDRSDRC2) Normal Mode
#define 	AT91C_DDRC2_MODE_NOP_CMD              ((unsigned int) 0x1) // (HDDRSDRC2) Issue a NOP Command at every access
#define 	AT91C_DDRC2_MODE_PRCGALL_CMD          ((unsigned int) 0x2) // (HDDRSDRC2) Issue a All Banks Precharge Command at every access
#define 	AT91C_DDRC2_MODE_LMR_CMD              ((unsigned int) 0x3) // (HDDRSDRC2) Issue a Load Mode Register at every access
#define 	AT91C_DDRC2_MODE_RFSH_CMD             ((unsigned int) 0x4) // (HDDRSDRC2) Issue a Refresh
#define 	AT91C_DDRC2_MODE_EXT_LMR_CMD          ((unsigned int) 0x5) // (HDDRSDRC2) Issue an Extended Load Mode Register
#define 	AT91C_DDRC2_MODE_DEEP_CMD             ((unsigned int) 0x6) // (HDDRSDRC2) Enter Deep Power Mode
#define 	AT91C_DDRC2_MODE_Reserved             ((unsigned int) 0x7) // (HDDRSDRC2) Reserved value
// -------- HDDRSDRC2_RTR : (HDDRSDRC2 Offset: 0x4) Refresh Timer Register -------- 
#define AT91C_DDRC2_COUNT     ((unsigned int) 0xFFF <<  0) // (HDDRSDRC2) Refresh Timer Count
// -------- HDDRSDRC2_CR : (HDDRSDRC2 Offset: 0x8) Configuration Register -------- 
#define AT91C_DDRC2_NC        ((unsigned int) 0x3 <<  0) // (HDDRSDRC2) Number of Column Bits
#define 	AT91C_DDRC2_NC_DDR9_SDR8            ((unsigned int) 0x0) // (HDDRSDRC2) DDR 9 Bits | SDR 8 Bits
#define 	AT91C_DDRC2_NC_DDR10_SDR9           ((unsigned int) 0x1) // (HDDRSDRC2) DDR 10 Bits | SDR 9 Bits
#define 	AT91C_DDRC2_NC_DDR11_SDR10          ((unsigned int) 0x2) // (HDDRSDRC2) DDR 11 Bits | SDR 10 Bits
#define 	AT91C_DDRC2_NC_DDR12_SDR11          ((unsigned int) 0x3) // (HDDRSDRC2) DDR 12 Bits | SDR 11 Bits
#define AT91C_DDRC2_NR        ((unsigned int) 0x3 <<  2) // (HDDRSDRC2) Number of Row Bits
#define 	AT91C_DDRC2_NR_11                   ((unsigned int) 0x0 <<  2) // (HDDRSDRC2) 11 Bits
#define 	AT91C_DDRC2_NR_12                   ((unsigned int) 0x1 <<  2) // (HDDRSDRC2) 12 Bits
#define 	AT91C_DDRC2_NR_13                   ((unsigned int) 0x2 <<  2) // (HDDRSDRC2) 13 Bits
#define 	AT91C_DDRC2_NR_14                   ((unsigned int) 0x3 <<  2) // (HDDRSDRC2) 14 Bits
#define AT91C_DDRC2_CAS       ((unsigned int) 0x7 <<  4) // (HDDRSDRC2) CAS Latency
#define 	AT91C_DDRC2_CAS_2                    ((unsigned int) 0x2 <<  4) // (HDDRSDRC2) 2 cycles (DDR1/SDR)
#define 	AT91C_DDRC2_CAS_3                    ((unsigned int) 0x3 <<  4) // (HDDRSDRC2) 3 cycles (DDR2/DDR1/SDR)
#define 	AT91C_DDRC2_CAS_4                    ((unsigned int) 0x4 <<  4) // (HDDRSDRC2) 4 cycles (DDR2)
#define 	AT91C_DDRC2_CAS_2_5                  ((unsigned int) 0x6 <<  4) // (HDDRSDRC2) 2.5 cycles (DDR1)
#define AT91C_DDRC2_DLL       ((unsigned int) 0x1 <<  7) // (HDDRSDRC2) DLL Reset
#define 	AT91C_DDRC2_DLL_RESET_DISABLED       ((unsigned int) 0x0 <<  7) // (HDDRSDRC2) DLL normal mode
#define 	AT91C_DDRC2_DLL_RESET_ENABLED        ((unsigned int) 0x1 <<  7) // (HDDRSDRC2) Reset DLL
#define AT91C_DDRC2_DIC_DS    ((unsigned int) 0x1 <<  8) // (HDDRSDRC2) Output driver impedance control
#define AT91C_DDRC2_OCD       ((unsigned int) 0x7 << 12) // (HDDRSDRC2) Off chip driver
#define 	AT91C_DDRC2_OCD_EXIT                 ((unsigned int) 0x0 << 12) // (HDDRSDRC2) Exit OCD calibration mode.
#define 	AT91C_DDRC2_OCD_DEFAULT              ((unsigned int) 0x7 << 12) // (HDDRSDRC2) Program OCD calibration default value.
#define AT91C_DDRC2_DQMS      ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Data Mask share
#define 	AT91C_DDRC2_DQMS_NOT_SHARED           ((unsigned int) 0x0 << 16) // (HDDRSDRC2) Used DQM bits are not shared
#define 	AT91C_DDRC2_DQMS_SHARED               ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Used DQM bits are shared
#define AT91C_DDRC2_ENRDM     ((unsigned int) 0x1 << 17) // (HDDRSDRC2) DQS/DATA phase error correction
#define 	AT91C_DDRC2_ENRDM_OFF                  ((unsigned int) 0x0 << 17) // (HDDRSDRC2) phase error correction disabled
#define 	AT91C_DDRC2_ENRDM_ON                   ((unsigned int) 0x1 << 17) // (HDDRSDRC2) phase error correction enabled
// -------- HDDRSDRC2_T0PR : (HDDRSDRC2 Offset: 0xc) Timing0 Register -------- 
#define AT91C_DDRC2_TRAS      ((unsigned int) 0xF <<  0) // (HDDRSDRC2) Active to precharge delay
#define 	AT91C_DDRC2_TRAS_0                    ((unsigned int) 0x0) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRAS_1                    ((unsigned int) 0x1) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRAS_2                    ((unsigned int) 0x2) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRAS_3                    ((unsigned int) 0x3) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRAS_4                    ((unsigned int) 0x4) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRAS_5                    ((unsigned int) 0x5) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRAS_6                    ((unsigned int) 0x6) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRAS_7                    ((unsigned int) 0x7) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRAS_8                    ((unsigned int) 0x8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRAS_9                    ((unsigned int) 0x9) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRAS_10                   ((unsigned int) 0xA) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRAS_11                   ((unsigned int) 0xB) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRAS_12                   ((unsigned int) 0xC) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRAS_13                   ((unsigned int) 0xD) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRAS_14                   ((unsigned int) 0xE) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRAS_15                   ((unsigned int) 0xF) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TRCD      ((unsigned int) 0xF <<  4) // (HDDRSDRC2) Row to column delay
#define 	AT91C_DDRC2_TRCD_0                    ((unsigned int) 0x0 <<  4) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRCD_1                    ((unsigned int) 0x1 <<  4) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRCD_2                    ((unsigned int) 0x2 <<  4) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRCD_3                    ((unsigned int) 0x3 <<  4) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRCD_4                    ((unsigned int) 0x4 <<  4) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRCD_5                    ((unsigned int) 0x5 <<  4) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRCD_6                    ((unsigned int) 0x6 <<  4) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRCD_7                    ((unsigned int) 0x7 <<  4) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRCD_8                    ((unsigned int) 0x8 <<  4) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRCD_9                    ((unsigned int) 0x9 <<  4) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRCD_10                   ((unsigned int) 0xA <<  4) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRCD_11                   ((unsigned int) 0xB <<  4) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRCD_12                   ((unsigned int) 0xC <<  4) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRCD_13                   ((unsigned int) 0xD <<  4) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRCD_14                   ((unsigned int) 0xE <<  4) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRCD_15                   ((unsigned int) 0xF <<  4) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TWR       ((unsigned int) 0xF <<  8) // (HDDRSDRC2) Write recovery delay
#define 	AT91C_DDRC2_TWR_0                    ((unsigned int) 0x0 <<  8) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TWR_1                    ((unsigned int) 0x1 <<  8) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TWR_2                    ((unsigned int) 0x2 <<  8) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TWR_3                    ((unsigned int) 0x3 <<  8) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TWR_4                    ((unsigned int) 0x4 <<  8) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TWR_5                    ((unsigned int) 0x5 <<  8) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TWR_6                    ((unsigned int) 0x6 <<  8) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TWR_7                    ((unsigned int) 0x7 <<  8) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TWR_8                    ((unsigned int) 0x8 <<  8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TWR_9                    ((unsigned int) 0x9 <<  8) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TWR_10                   ((unsigned int) 0xA <<  8) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TWR_11                   ((unsigned int) 0xB <<  8) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TWR_12                   ((unsigned int) 0xC <<  8) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TWR_13                   ((unsigned int) 0xD <<  8) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TWR_14                   ((unsigned int) 0xE <<  8) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TWR_15                   ((unsigned int) 0xF <<  8) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TRC       ((unsigned int) 0xF << 12) // (HDDRSDRC2) Row cycle delay
#define 	AT91C_DDRC2_TRC_0                    ((unsigned int) 0x0 << 12) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRC_1                    ((unsigned int) 0x1 << 12) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRC_2                    ((unsigned int) 0x2 << 12) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRC_3                    ((unsigned int) 0x3 << 12) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRC_4                    ((unsigned int) 0x4 << 12) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRC_5                    ((unsigned int) 0x5 << 12) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRC_6                    ((unsigned int) 0x6 << 12) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRC_7                    ((unsigned int) 0x7 << 12) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRC_8                    ((unsigned int) 0x8 << 12) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRC_9                    ((unsigned int) 0x9 << 12) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRC_10                   ((unsigned int) 0xA << 12) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRC_11                   ((unsigned int) 0xB << 12) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRC_12                   ((unsigned int) 0xC << 12) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRC_13                   ((unsigned int) 0xD << 12) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRC_14                   ((unsigned int) 0xE << 12) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRC_15                   ((unsigned int) 0xF << 12) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TRP       ((unsigned int) 0xF << 16) // (HDDRSDRC2) Row precharge delay
#define 	AT91C_DDRC2_TRP_0                    ((unsigned int) 0x0 << 16) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRP_1                    ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRP_2                    ((unsigned int) 0x2 << 16) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRP_3                    ((unsigned int) 0x3 << 16) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRP_4                    ((unsigned int) 0x4 << 16) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRP_5                    ((unsigned int) 0x5 << 16) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRP_6                    ((unsigned int) 0x6 << 16) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRP_7                    ((unsigned int) 0x7 << 16) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRP_8                    ((unsigned int) 0x8 << 16) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRP_9                    ((unsigned int) 0x9 << 16) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRP_10                   ((unsigned int) 0xA << 16) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRP_11                   ((unsigned int) 0xB << 16) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRP_12                   ((unsigned int) 0xC << 16) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRP_13                   ((unsigned int) 0xD << 16) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRP_14                   ((unsigned int) 0xE << 16) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRP_15                   ((unsigned int) 0xF << 16) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TRRD      ((unsigned int) 0xF << 20) // (HDDRSDRC2) Active bankA to Active bankB
#define 	AT91C_DDRC2_TRRD_0                    ((unsigned int) 0x0 << 20) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRRD_1                    ((unsigned int) 0x1 << 20) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRRD_2                    ((unsigned int) 0x2 << 20) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRRD_3                    ((unsigned int) 0x3 << 20) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRRD_4                    ((unsigned int) 0x4 << 20) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRRD_5                    ((unsigned int) 0x5 << 20) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRRD_6                    ((unsigned int) 0x6 << 20) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRRD_7                    ((unsigned int) 0x7 << 20) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRRD_8                    ((unsigned int) 0x8 << 20) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRRD_9                    ((unsigned int) 0x9 << 20) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRRD_10                   ((unsigned int) 0xA << 20) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRRD_11                   ((unsigned int) 0xB << 20) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRRD_12                   ((unsigned int) 0xC << 20) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRRD_13                   ((unsigned int) 0xD << 20) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRRD_14                   ((unsigned int) 0xE << 20) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRRD_15                   ((unsigned int) 0xF << 20) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TWTR      ((unsigned int) 0x1 << 24) // (HDDRSDRC2) Internal write to read delay
#define 	AT91C_DDRC2_TWTR_0                    ((unsigned int) 0x0 << 24) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TWTR_1                    ((unsigned int) 0x1 << 24) // (HDDRSDRC2) Value :  1
#define AT91C_DDRC2_TMRD      ((unsigned int) 0xF << 28) // (HDDRSDRC2) Load mode register command to active or refresh command
#define 	AT91C_DDRC2_TMRD_0                    ((unsigned int) 0x0 << 28) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TMRD_1                    ((unsigned int) 0x1 << 28) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TMRD_2                    ((unsigned int) 0x2 << 28) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TMRD_3                    ((unsigned int) 0x3 << 28) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TMRD_4                    ((unsigned int) 0x4 << 28) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TMRD_5                    ((unsigned int) 0x5 << 28) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TMRD_6                    ((unsigned int) 0x6 << 28) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TMRD_7                    ((unsigned int) 0x7 << 28) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TMRD_8                    ((unsigned int) 0x8 << 28) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TMRD_9                    ((unsigned int) 0x9 << 28) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TMRD_10                   ((unsigned int) 0xA << 28) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TMRD_11                   ((unsigned int) 0xB << 28) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TMRD_12                   ((unsigned int) 0xC << 28) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TMRD_13                   ((unsigned int) 0xD << 28) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TMRD_14                   ((unsigned int) 0xE << 28) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TMRD_15                   ((unsigned int) 0xF << 28) // (HDDRSDRC2) Value : 15
// -------- HDDRSDRC2_T1PR : (HDDRSDRC2 Offset: 0x10) Timing1 Register -------- 
#define AT91C_DDRC2_TRFC      ((unsigned int) 0x1F <<  0) // (HDDRSDRC2) row cycle delay
#define 	AT91C_DDRC2_TRFC_0                    ((unsigned int) 0x0) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRFC_1                    ((unsigned int) 0x1) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRFC_2                    ((unsigned int) 0x2) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRFC_3                    ((unsigned int) 0x3) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRFC_4                    ((unsigned int) 0x4) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRFC_5                    ((unsigned int) 0x5) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRFC_6                    ((unsigned int) 0x6) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRFC_7                    ((unsigned int) 0x7) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRFC_8                    ((unsigned int) 0x8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRFC_9                    ((unsigned int) 0x9) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRFC_10                   ((unsigned int) 0xA) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRFC_11                   ((unsigned int) 0xB) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRFC_12                   ((unsigned int) 0xC) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRFC_13                   ((unsigned int) 0xD) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRFC_14                   ((unsigned int) 0xE) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRFC_15                   ((unsigned int) 0xF) // (HDDRSDRC2) Value : 15
#define 	AT91C_DDRC2_TRFC_16                   ((unsigned int) 0x10) // (HDDRSDRC2) Value : 16
#define 	AT91C_DDRC2_TRFC_17                   ((unsigned int) 0x11) // (HDDRSDRC2) Value : 17
#define 	AT91C_DDRC2_TRFC_18                   ((unsigned int) 0x12) // (HDDRSDRC2) Value : 18
#define 	AT91C_DDRC2_TRFC_19                   ((unsigned int) 0x13) // (HDDRSDRC2) Value : 19
#define 	AT91C_DDRC2_TRFC_20                   ((unsigned int) 0x14) // (HDDRSDRC2) Value : 20
#define 	AT91C_DDRC2_TRFC_21                   ((unsigned int) 0x15) // (HDDRSDRC2) Value : 21
#define 	AT91C_DDRC2_TRFC_22                   ((unsigned int) 0x16) // (HDDRSDRC2) Value : 22
#define 	AT91C_DDRC2_TRFC_23                   ((unsigned int) 0x17) // (HDDRSDRC2) Value : 23
#define 	AT91C_DDRC2_TRFC_24                   ((unsigned int) 0x18) // (HDDRSDRC2) Value : 24
#define 	AT91C_DDRC2_TRFC_25                   ((unsigned int) 0x19) // (HDDRSDRC2) Value : 25
#define 	AT91C_DDRC2_TRFC_26                   ((unsigned int) 0x1A) // (HDDRSDRC2) Value : 26
#define 	AT91C_DDRC2_TRFC_27                   ((unsigned int) 0x1B) // (HDDRSDRC2) Value : 27
#define 	AT91C_DDRC2_TRFC_28                   ((unsigned int) 0x1C) // (HDDRSDRC2) Value : 28
#define 	AT91C_DDRC2_TRFC_29                   ((unsigned int) 0x1D) // (HDDRSDRC2) Value : 29
#define 	AT91C_DDRC2_TRFC_30                   ((unsigned int) 0x1E) // (HDDRSDRC2) Value : 30
#define 	AT91C_DDRC2_TRFC_31                   ((unsigned int) 0x1F) // (HDDRSDRC2) Value : 31
#define AT91C_DDRC2_TXSNR     ((unsigned int) 0xFF <<  8) // (HDDRSDRC2) Exit self refresh delay to Read command
#define 	AT91C_DDRC2_TXSNR_0                    ((unsigned int) 0x0 <<  8) // (HDDRSDRC2) Value :   0
#define 	AT91C_DDRC2_TXSNR_8                    ((unsigned int) 0x8 <<  8) // (HDDRSDRC2) Value :   8
#define 	AT91C_DDRC2_TXSNR_16                   ((unsigned int) 0x10 <<  8) // (HDDRSDRC2) Value :  16
#define 	AT91C_DDRC2_TXSNR_32                   ((unsigned int) 0x20 <<  8) // (HDDRSDRC2) Value :  32
#define 	AT91C_DDRC2_TXSNR_48                   ((unsigned int) 0x30 <<  8) // (HDDRSDRC2) Value :  48
#define 	AT91C_DDRC2_TXSNR_64                   ((unsigned int) 0x40 <<  8) // (HDDRSDRC2) Value :  64
#define 	AT91C_DDRC2_TXSNR_80                   ((unsigned int) 0x50 <<  8) // (HDDRSDRC2) Value :  80
#define 	AT91C_DDRC2_TXSNR_96                   ((unsigned int) 0x60 <<  8) // (HDDRSDRC2) Value :  96
#define 	AT91C_DDRC2_TXSNR_112                  ((unsigned int) 0x70 <<  8) // (HDDRSDRC2) Value : 112
#define 	AT91C_DDRC2_TXSNR_128                  ((unsigned int) 0x80 <<  8) // (HDDRSDRC2) Value : 128
#define 	AT91C_DDRC2_TXSNR_144                  ((unsigned int) 0x90 <<  8) // (HDDRSDRC2) Value : 144
#define 	AT91C_DDRC2_TXSNR_160                  ((unsigned int) 0xA0 <<  8) // (HDDRSDRC2) Value : 160
#define 	AT91C_DDRC2_TXSNR_176                  ((unsigned int) 0xB0 <<  8) // (HDDRSDRC2) Value : 176
#define 	AT91C_DDRC2_TXSNR_192                  ((unsigned int) 0xC0 <<  8) // (HDDRSDRC2) Value : 192
#define 	AT91C_DDRC2_TXSNR_208                  ((unsigned int) 0xD0 <<  8) // (HDDRSDRC2) Value : 208
#define 	AT91C_DDRC2_TXSNR_224                  ((unsigned int) 0xE0 <<  8) // (HDDRSDRC2) Value : 224
#define 	AT91C_DDRC2_TXSNR_240                  ((unsigned int) 0xF0 <<  8) // (HDDRSDRC2) Value : 240
#define 	AT91C_DDRC2_TXSNR_255                  ((unsigned int) 0xFF <<  8) // (HDDRSDRC2) Value : 255
#define AT91C_DDRC2_TXSRD     ((unsigned int) 0xFF << 16) // (HDDRSDRC2) Exit self refresh delay to Read command
#define 	AT91C_DDRC2_TXSRD_0                    ((unsigned int) 0x0 << 16) // (HDDRSDRC2) Value :   0
#define 	AT91C_DDRC2_TXSRD_8                    ((unsigned int) 0x8 << 16) // (HDDRSDRC2) Value :   8
#define 	AT91C_DDRC2_TXSRD_16                   ((unsigned int) 0x10 << 16) // (HDDRSDRC2) Value :  16
#define 	AT91C_DDRC2_TXSRD_32                   ((unsigned int) 0x20 << 16) // (HDDRSDRC2) Value :  32
#define 	AT91C_DDRC2_TXSRD_48                   ((unsigned int) 0x30 << 16) // (HDDRSDRC2) Value :  48
#define 	AT91C_DDRC2_TXSRD_64                   ((unsigned int) 0x40 << 16) // (HDDRSDRC2) Value :  64
#define 	AT91C_DDRC2_TXSRD_80                   ((unsigned int) 0x50 << 16) // (HDDRSDRC2) Value :  80
#define 	AT91C_DDRC2_TXSRD_96                   ((unsigned int) 0x60 << 16) // (HDDRSDRC2) Value :  96
#define 	AT91C_DDRC2_TXSRD_112                  ((unsigned int) 0x70 << 16) // (HDDRSDRC2) Value : 112
#define 	AT91C_DDRC2_TXSRD_128                  ((unsigned int) 0x80 << 16) // (HDDRSDRC2) Value : 128
#define 	AT91C_DDRC2_TXSRD_144                  ((unsigned int) 0x90 << 16) // (HDDRSDRC2) Value : 144
#define 	AT91C_DDRC2_TXSRD_160                  ((unsigned int) 0xA0 << 16) // (HDDRSDRC2) Value : 160
#define 	AT91C_DDRC2_TXSRD_176                  ((unsigned int) 0xB0 << 16) // (HDDRSDRC2) Value : 176
#define 	AT91C_DDRC2_TXSRD_192                  ((unsigned int) 0xC0 << 16) // (HDDRSDRC2) Value : 192
#define 	AT91C_DDRC2_TXSRD_208                  ((unsigned int) 0xD0 << 16) // (HDDRSDRC2) Value : 208
#define 	AT91C_DDRC2_TXSRD_224                  ((unsigned int) 0xE0 << 16) // (HDDRSDRC2) Value : 224
#define 	AT91C_DDRC2_TXSRD_240                  ((unsigned int) 0xF0 << 16) // (HDDRSDRC2) Value : 240
#define 	AT91C_DDRC2_TXSRD_255                  ((unsigned int) 0xFF << 16) // (HDDRSDRC2) Value : 255
#define AT91C_DDRC2_TXP       ((unsigned int) 0xF << 24) // (HDDRSDRC2) Exit Power-down delay to first command
#define 	AT91C_DDRC2_TXP_0                    ((unsigned int) 0x0 << 24) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TXP_1                    ((unsigned int) 0x1 << 24) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TXP_2                    ((unsigned int) 0x2 << 24) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TXP_3                    ((unsigned int) 0x3 << 24) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TXP_4                    ((unsigned int) 0x4 << 24) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TXP_5                    ((unsigned int) 0x5 << 24) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TXP_6                    ((unsigned int) 0x6 << 24) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TXP_7                    ((unsigned int) 0x7 << 24) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TXP_8                    ((unsigned int) 0x8 << 24) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TXP_9                    ((unsigned int) 0x9 << 24) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TXP_10                   ((unsigned int) 0xA << 24) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TXP_11                   ((unsigned int) 0xB << 24) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TXP_12                   ((unsigned int) 0xC << 24) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TXP_13                   ((unsigned int) 0xD << 24) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TXP_14                   ((unsigned int) 0xE << 24) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TXP_15                   ((unsigned int) 0xF << 24) // (HDDRSDRC2) Value : 15
// -------- HDDRSDRC2_T2PR : (HDDRSDRC2 Offset: 0x14) Timing2 Register -------- 
#define AT91C_DDRC2_TXARD     ((unsigned int) 0xF <<  0) // (HDDRSDRC2) Exit active power down delay to read command in 'Fast Exit' mode.
#define 	AT91C_DDRC2_TXARD_0                    ((unsigned int) 0x0) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TXARD_1                    ((unsigned int) 0x1) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TXARD_2                    ((unsigned int) 0x2) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TXARD_3                    ((unsigned int) 0x3) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TXARD_4                    ((unsigned int) 0x4) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TXARD_5                    ((unsigned int) 0x5) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TXARD_6                    ((unsigned int) 0x6) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TXARD_7                    ((unsigned int) 0x7) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TXARD_8                    ((unsigned int) 0x8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TXARD_9                    ((unsigned int) 0x9) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TXARD_10                   ((unsigned int) 0xA) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TXARD_11                   ((unsigned int) 0xB) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TXARD_12                   ((unsigned int) 0xC) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TXARD_13                   ((unsigned int) 0xD) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TXARD_14                   ((unsigned int) 0xE) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TXARD_15                   ((unsigned int) 0xF) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TXARDS    ((unsigned int) 0xF <<  8) // (HDDRSDRC2) Exit active power down delay to read command in 'Slow Exit' mode.
#define 	AT91C_DDRC2_TXARDS_0                    ((unsigned int) 0x0 <<  8) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TXARDS_1                    ((unsigned int) 0x1 <<  8) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TXARDS_2                    ((unsigned int) 0x2 <<  8) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TXARDS_3                    ((unsigned int) 0x3 <<  8) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TXARDS_4                    ((unsigned int) 0x4 <<  8) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TXARDS_5                    ((unsigned int) 0x5 <<  8) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TXARDS_6                    ((unsigned int) 0x6 <<  8) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TXARDS_7                    ((unsigned int) 0x7 <<  8) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TXARDS_8                    ((unsigned int) 0x8 <<  8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TXARDS_9                    ((unsigned int) 0x9 <<  8) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TXARDS_10                   ((unsigned int) 0xA <<  8) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TXARDS_11                   ((unsigned int) 0xB <<  8) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TXARDS_12                   ((unsigned int) 0xC <<  8) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TXARDS_13                   ((unsigned int) 0xD <<  8) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TXARDS_14                   ((unsigned int) 0xE <<  8) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TXARDS_15                   ((unsigned int) 0xF <<  8) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TRPA      ((unsigned int) 0xF << 16) // (HDDRSDRC2) Row precharge all delay
#define 	AT91C_DDRC2_TRPA_0                    ((unsigned int) 0x0 << 16) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRPA_1                    ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRPA_2                    ((unsigned int) 0x2 << 16) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRPA_3                    ((unsigned int) 0x3 << 16) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRPA_4                    ((unsigned int) 0x4 << 16) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRPA_5                    ((unsigned int) 0x5 << 16) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRPA_6                    ((unsigned int) 0x6 << 16) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRPA_7                    ((unsigned int) 0x7 << 16) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRPA_8                    ((unsigned int) 0x8 << 16) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRPA_9                    ((unsigned int) 0x9 << 16) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRPA_10                   ((unsigned int) 0xA << 16) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRPA_11                   ((unsigned int) 0xB << 16) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRPA_12                   ((unsigned int) 0xC << 16) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRPA_13                   ((unsigned int) 0xD << 16) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRPA_14                   ((unsigned int) 0xE << 16) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRPA_15                   ((unsigned int) 0xF << 16) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TRTP      ((unsigned int) 0xF << 24) // (HDDRSDRC2) Read to Precharge delay
#define 	AT91C_DDRC2_TRTP_0                    ((unsigned int) 0x0 << 24) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TRTP_1                    ((unsigned int) 0x1 << 24) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TRTP_2                    ((unsigned int) 0x2 << 24) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TRTP_3                    ((unsigned int) 0x3 << 24) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TRTP_4                    ((unsigned int) 0x4 << 24) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TRTP_5                    ((unsigned int) 0x5 << 24) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TRTP_6                    ((unsigned int) 0x6 << 24) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TRTP_7                    ((unsigned int) 0x7 << 24) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TRTP_8                    ((unsigned int) 0x8 << 24) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TRTP_9                    ((unsigned int) 0x9 << 24) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TRTP_10                   ((unsigned int) 0xA << 24) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TRTP_11                   ((unsigned int) 0xB << 24) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TRTP_12                   ((unsigned int) 0xC << 24) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TRTP_13                   ((unsigned int) 0xD << 24) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TRTP_14                   ((unsigned int) 0xE << 24) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TRTP_15                   ((unsigned int) 0xF << 24) // (HDDRSDRC2) Value : 15
// -------- HDDRSDRC2_T3PR : (HDDRSDRC2 Offset: 0x18) Timing3 Register -------- 
#define AT91C_DDRC2_TANPD     ((unsigned int) 0xF <<  0) // (HDDRSDRC2) ODT to power down entry
#define 	AT91C_DDRC2_TANPD_0                    ((unsigned int) 0x0) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TANPD_1                    ((unsigned int) 0x1) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TANPD_2                    ((unsigned int) 0x2) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TANPD_3                    ((unsigned int) 0x3) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TANPD_4                    ((unsigned int) 0x4) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TANPD_5                    ((unsigned int) 0x5) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TANPD_6                    ((unsigned int) 0x6) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TANPD_7                    ((unsigned int) 0x7) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TANPD_8                    ((unsigned int) 0x8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TANPD_9                    ((unsigned int) 0x9) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TANPD_10                   ((unsigned int) 0xA) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TANPD_11                   ((unsigned int) 0xB) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TANPD_12                   ((unsigned int) 0xC) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TANPD_13                   ((unsigned int) 0xD) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TANPD_14                   ((unsigned int) 0xE) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TANPD_15                   ((unsigned int) 0xF) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TAXPD     ((unsigned int) 0xF <<  4) // (HDDRSDRC2) ODT power down exit
#define 	AT91C_DDRC2_TAXPD_0                    ((unsigned int) 0x0 <<  4) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TAXPD_1                    ((unsigned int) 0x1 <<  4) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TAXPD_2                    ((unsigned int) 0x2 <<  4) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TAXPD_3                    ((unsigned int) 0x3 <<  4) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TAXPD_4                    ((unsigned int) 0x4 <<  4) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TAXPD_5                    ((unsigned int) 0x5 <<  4) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TAXPD_6                    ((unsigned int) 0x6 <<  4) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TAXPD_7                    ((unsigned int) 0x7 <<  4) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TAXPD_8                    ((unsigned int) 0x8 <<  4) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TAXPD_9                    ((unsigned int) 0x9 <<  4) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TAXPD_10                   ((unsigned int) 0xA <<  4) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TAXPD_11                   ((unsigned int) 0xB <<  4) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TAXPD_12                   ((unsigned int) 0xC <<  4) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TAXPD_13                   ((unsigned int) 0xD <<  4) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TAXPD_14                   ((unsigned int) 0xE <<  4) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TAXPD_15                   ((unsigned int) 0xF <<  4) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TAOFPD    ((unsigned int) 0xF <<  8) // (HDDRSDRC2) ODT turn off in power down mode
#define 	AT91C_DDRC2_TAOFPD_0                    ((unsigned int) 0x0 <<  8) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TAOFPD_1                    ((unsigned int) 0x1 <<  8) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TAOFPD_2                    ((unsigned int) 0x2 <<  8) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TAOFPD_3                    ((unsigned int) 0x3 <<  8) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TAOFPD_4                    ((unsigned int) 0x4 <<  8) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TAOFPD_5                    ((unsigned int) 0x5 <<  8) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TAOFPD_6                    ((unsigned int) 0x6 <<  8) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TAOFPD_7                    ((unsigned int) 0x7 <<  8) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TAOFPD_8                    ((unsigned int) 0x8 <<  8) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TAOFPD_9                    ((unsigned int) 0x9 <<  8) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TAOFPD_10                   ((unsigned int) 0xA <<  8) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TAOFPD_11                   ((unsigned int) 0xB <<  8) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TAOFPD_12                   ((unsigned int) 0xC <<  8) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TAOFPD_13                   ((unsigned int) 0xD <<  8) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TAOFPD_14                   ((unsigned int) 0xE <<  8) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TAOFPD_15                   ((unsigned int) 0xF <<  8) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TAOFD     ((unsigned int) 0xF << 12) // (HDDRSDRC2) ODT turn off delay
#define 	AT91C_DDRC2_TAOFD_0                    ((unsigned int) 0x0 << 12) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TAOFD_1                    ((unsigned int) 0x1 << 12) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TAOFD_2                    ((unsigned int) 0x2 << 12) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TAOFD_3                    ((unsigned int) 0x3 << 12) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TAOFD_4                    ((unsigned int) 0x4 << 12) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TAOFD_5                    ((unsigned int) 0x5 << 12) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TAOFD_6                    ((unsigned int) 0x6 << 12) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TAOFD_7                    ((unsigned int) 0x7 << 12) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TAOFD_8                    ((unsigned int) 0x8 << 12) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TAOFD_9                    ((unsigned int) 0x9 << 12) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TAOFD_10                   ((unsigned int) 0xA << 12) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TAOFD_11                   ((unsigned int) 0xB << 12) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TAOFD_12                   ((unsigned int) 0xC << 12) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TAOFD_13                   ((unsigned int) 0xD << 12) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TAOFD_14                   ((unsigned int) 0xE << 12) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TAOFD_15                   ((unsigned int) 0xF << 12) // (HDDRSDRC2) Value : 15
#define AT91C_DDRC2_TAONPD    ((unsigned int) 0xF << 16) // (HDDRSDRC2) ODT turn on in power down mode
#define 	AT91C_DDRC2_TAONPD_0                    ((unsigned int) 0x0 << 16) // (HDDRSDRC2) Value :  0
#define 	AT91C_DDRC2_TAONPD_1                    ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Value :  1
#define 	AT91C_DDRC2_TAONPD_2                    ((unsigned int) 0x2 << 16) // (HDDRSDRC2) Value :  2
#define 	AT91C_DDRC2_TAONPD_3                    ((unsigned int) 0x3 << 16) // (HDDRSDRC2) Value :  3
#define 	AT91C_DDRC2_TAONPD_4                    ((unsigned int) 0x4 << 16) // (HDDRSDRC2) Value :  4
#define 	AT91C_DDRC2_TAONPD_5                    ((unsigned int) 0x5 << 16) // (HDDRSDRC2) Value :  5
#define 	AT91C_DDRC2_TAONPD_6                    ((unsigned int) 0x6 << 16) // (HDDRSDRC2) Value :  6
#define 	AT91C_DDRC2_TAONPD_7                    ((unsigned int) 0x7 << 16) // (HDDRSDRC2) Value :  7
#define 	AT91C_DDRC2_TAONPD_8                    ((unsigned int) 0x8 << 16) // (HDDRSDRC2) Value :  8
#define 	AT91C_DDRC2_TAONPD_9                    ((unsigned int) 0x9 << 16) // (HDDRSDRC2) Value :  9
#define 	AT91C_DDRC2_TAONPD_10                   ((unsigned int) 0xA << 16) // (HDDRSDRC2) Value : 10
#define 	AT91C_DDRC2_TAONPD_11                   ((unsigned int) 0xB << 16) // (HDDRSDRC2) Value : 11
#define 	AT91C_DDRC2_TAONPD_12                   ((unsigned int) 0xC << 16) // (HDDRSDRC2) Value : 12
#define 	AT91C_DDRC2_TAONPD_13                   ((unsigned int) 0xD << 16) // (HDDRSDRC2) Value : 13
#define 	AT91C_DDRC2_TAONPD_14                   ((unsigned int) 0xE << 16) // (HDDRSDRC2) Value : 14
#define 	AT91C_DDRC2_TAONPD_15                   ((unsigned int) 0xF << 16) // (HDDRSDRC2) Value : 15
// -------- HDDRSDRC2_LPR : (HDDRSDRC2 Offset: 0x1c)  -------- 
#define AT91C_DDRC2_LPCB      ((unsigned int) 0x3 <<  0) // (HDDRSDRC2) Low-power Command Bit
#define 	AT91C_DDRC2_LPCB_DISABLED             ((unsigned int) 0x0) // (HDDRSDRC2) Low-power Feature is inhibited: no power-down, self refresh and Deep power mode are issued to the SDRAM device.
#define 	AT91C_DDRC2_LPCB_SELFREFRESH          ((unsigned int) 0x1) // (HDDRSDRC2) The DDRSDRAMC Controller issues a Self Refresh Command to the SDRAM device, the clock(s) is/are de-activated and the CKE signal is set low. The SDRAM device leaves the self refresh mode when accessed and enters it after the access.
#define 	AT91C_DDRC2_LPCB_POWERDOWN            ((unsigned int) 0x2) // (HDDRSDRC2) The HDDRSDRC2 Controller issues a Power-down Command to the SDRAM device after each access, the CKE signal is set low. The SDRAM device leaves the power-down mode when accessed and enters it after the access.
#define 	AT91C_DDRC2_LPCB_DEEP_PWD             ((unsigned int) 0x3) // (HDDRSDRC2) The HDDRSDRC2 Controller issues a Deep Power-down Command to the Mobile SDRAM device. This mode is unique to Mobile SDRAM devices
#define AT91C_DDRC2_CLK_FR    ((unsigned int) 0x1 <<  2) // (HDDRSDRC2) Clock frozen Command Bit
#define 	AT91C_DDRC2_CLK_FR_DISABLED             ((unsigned int) 0x0 <<  2) // (HDDRSDRC2) Low-power Feature is inhibited: no power-down, self refresh and Deep power mode are issued to the SDRAM device.
#define 	AT91C_DDRC2_CLK_FR_SELFREFRESH          ((unsigned int) 0x1 <<  2) // (HDDRSDRC2) The DDRSDRAMC Controller issues a Self Refresh Command to the SDRAM device, the clock(s) is/are de-activated and the CKE signal is set low. The SDRAM device leaves the self refresh mode when accessed and enters it after the access.
#define 	AT91C_DDRC2_CLK_FR_POWERDOWN            ((unsigned int) 0x2 <<  2) // (HDDRSDRC2) The HDDRSDRC2 Controller issues a Power-down Command to the SDRAM device after each access, the CKE signal is set low. The SDRAM device leaves the power-down mode when accessed and enters it after the access.
#define 	AT91C_DDRC2_CLK_FR_DEEP_PWD             ((unsigned int) 0x3 <<  2) // (HDDRSDRC2) The HDDRSDRC2 Controller issues a Deep Power-down Command to the Mobile SDRAM device. This mode is unique to Mobile SDRAM devices
#define AT91C_DDRC2_PASR      ((unsigned int) 0x7 <<  4) // (HDDRSDRC2) Partial Array Self Refresh
#define AT91C_                ((unsigned int) 0x0 <<  8) // (HDDRSDRC2) 
#define AT91C_DDRC2_DS        ((unsigned int) 0x3 << 10) // (HDDRSDRC2) Drive strength
#define AT91C_DDRC2_TIMEOUT   ((unsigned int) 0x3 << 12) // (HDDRSDRC2) low-power mode delay
#define 	AT91C_DDRC2_TIMEOUT_0                    ((unsigned int) 0x0 << 12) // (HDDRSDRC2) The SDRAM controller activates the SDRAM low-power mode immediately after the end of the last transfer.
#define 	AT91C_DDRC2_TIMEOUT_64                   ((unsigned int) 0x1 << 12) // (HDDRSDRC2) The SDRAM controller activates the SDRAM low-power mode 64 clock cycles after the end of the last transfer.
#define 	AT91C_DDRC2_TIMEOUT_128                  ((unsigned int) 0x2 << 12) // (HDDRSDRC2) The SDRAM controller activates the SDRAM low-power mode 128 clock cycles after the end of the last transfer.
#define 	AT91C_DDRC2_TIMEOUT_Reserved             ((unsigned int) 0x3 << 12) // (HDDRSDRC2) Reserved
#define AT91C_DDRC2_ADPE      ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Active Power Down Exit time
#define 	AT91C_DDRC2_ADPE_FAST                 ((unsigned int) 0x0 << 16) // (HDDRSDRC2) Fast Exit
#define 	AT91C_DDRC2_ADPE_SLOW                 ((unsigned int) 0x1 << 16) // (HDDRSDRC2) Slow Exit
// -------- HDDRSDRC2_MDR : (HDDRSDRC2 Offset: 0x20) Memory Device Register -------- 
#define AT91C_DDRC2_MD        ((unsigned int) 0x7 <<  0) // (HDDRSDRC2) memory device
#define 	AT91C_DDRC2_MD_SDR_SDRAM            ((unsigned int) 0x0) // (HDDRSDRC2) SDR SDRAM
#define 	AT91C_DDRC2_MD_LP_SDR_SDRAM         ((unsigned int) 0x1) // (HDDRSDRC2) Low Power SDR SDRAM
#define 	AT91C_DDRC2_MD_DDR_SDRAM            ((unsigned int) 0x2) // (HDDRSDRC2) DDR SDRAM
#define 	AT91C_DDRC2_MD_LP_DDR_SDRAM         ((unsigned int) 0x3) // (HDDRSDRC2) Low Power DDR SDRAM
#define 	AT91C_DDRC2_MD_DDR2_SDRAM           ((unsigned int) 0x6) // (HDDRSDRC2) DDR2 SDRAM
#define AT91C_DDRC2_DBW       ((unsigned int) 0x1 <<  4) // (HDDRSDRC2) Data Bus Width
#define 	AT91C_DDRC2_DBW_32_BITS              ((unsigned int) 0x0 <<  4) // (HDDRSDRC2) 32 Bits datas bus
#define 	AT91C_DDRC2_DBW_16_BITS              ((unsigned int) 0x1 <<  4) // (HDDRSDRC2) 16 Bits datas bus
// -------- HDDRSDRC2_DLL : (HDDRSDRC2 Offset: 0x24) DLL Information Register -------- 
#define AT91C_DDRC2_MDINC     ((unsigned int) 0x1 <<  0) // (HDDRSDRC2) DLL Master Delay Increment
#define AT91C_DDRC2_MDDEC     ((unsigned int) 0x1 <<  1) // (HDDRSDRC2) DLL Master Delay Decrement
#define AT91C_DDRC2_MDOVF     ((unsigned int) 0x1 <<  2) // (HDDRSDRC2) DLL Master Delay Overflow Flag
#define AT91C_DDRC2_SDCOVF    ((unsigned int) 0x1 <<  3) // (HDDRSDRC2) DLL Slave Delay Correction Overflow Flag
#define AT91C_DDRC2_SDCUDF    ((unsigned int) 0x1 <<  4) // (HDDRSDRC2) DLL Slave Delay Correction Underflow Flag
#define AT91C_DDRC2_SDERF     ((unsigned int) 0x1 <<  5) // (HDDRSDRC2) DLL Slave Delay Correction Error Flag
#define AT91C_DDRC2_MDVAL     ((unsigned int) 0xFF <<  8) // (HDDRSDRC2) DLL Master Delay Value
#define AT91C_DDRC2_SDVAL     ((unsigned int) 0xFF << 16) // (HDDRSDRC2) DLL Slave Delay Value
#define AT91C_DDRC2_SDCVAL    ((unsigned int) 0xFF << 24) // (HDDRSDRC2) DLL Slave Delay Correction Value
// -------- HDDRSDRC2_HS : (HDDRSDRC2 Offset: 0x2c) High Speed Register -------- 
#define AT91C_DDRC2_DA        ((unsigned int) 0x1 <<  0) // (HDDRSDRC2) Decode cycle
#define AT91C_DDRC2_NO_OPT    ((unsigned int) 0x1 <<  1) // (HDDRSDRC2) Disable optimization
#define AT91C_DDRC2_NO_ANT    ((unsigned int) 0x1 <<  2) // (HDDRSDRC2) Disable Anticipated read
#define AT91C_DDRC2_RES_DQS   ((unsigned int) 0x1 <<  3) // (HDDRSDRC2) Reset DQS Active
// -------- HDDRSDRC2_DELAY1 : (HDDRSDRC2 Offset: 0x30) Pad delay1 Register -------- 
#define AT91C_DDRC2_DELAY     ((unsigned int) 0xF <<  0) // (HDDRSDRC2) Pad delay value
// -------- HDDRSDRC2_DELAY2 : (HDDRSDRC2 Offset: 0x34) Pad delay2 Register -------- 
// -------- HDDRSDRC2_DELAY3 : (HDDRSDRC2 Offset: 0x38) Pad delay3 Register -------- 
// -------- HDDRSDRC2_DELAY4 : (HDDRSDRC2 Offset: 0x3c) Pad delay4 Register -------- 
// -------- HDDRSDRC2_DELAY5 : (HDDRSDRC2 Offset: 0x40) Pad delay5 Register -------- 
// -------- HDDRSDRC2_DELAY6 : (HDDRSDRC2 Offset: 0x44) Pad delay6 Register -------- 
// -------- HDDRSDRC2_DELAY7 : (HDDRSDRC2 Offset: 0x48) Pad delay7 Register -------- 
// -------- HDDRSDRC2_DELAY8 : (HDDRSDRC2 Offset: 0x4c) Pad delay8 Register -------- 
// -------- HDDRSDRC2_WPCR : (HDDRSDRC2 Offset: 0xe4) Write Protect Control Register -------- 
#define AT91C_DDRC2_WPEN      ((unsigned int) 0x1 <<  0) // (HDDRSDRC2) write protect enable
#define AT91C_DDRC2_WPKEY     ((unsigned int) 0xFFFFFF <<  8) // (HDDRSDRC2) write protect key
// -------- HDDRSDRC2_WPSR : (HDDRSDRC2 Offset: 0xe8) Write Protect Status Register -------- 
#define AT91C_DDRC2_WPVS      ((unsigned int) 0x1 <<  0) // (HDDRSDRC2) write protect violation status
#define AT91C_DDRC2_WPSRC     ((unsigned int) 0xFFFF <<  8) // (HDDRSDRC2) write protect source address

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Static Memory Controller Interface
// *****************************************************************************
typedef struct _AT91S_SMC {
	AT91_REG	 SMC_SETUP0; 	//  Setup Register for CS 0
	AT91_REG	 SMC_PULSE0; 	//  Pulse Register for CS 0
	AT91_REG	 SMC_CYCLE0; 	//  Cycle Register for CS 0
	AT91_REG	 SMC_CTRL0; 	//  Control Register for CS 0
	AT91_REG	 SMC_SETUP1; 	//  Setup Register for CS 1
	AT91_REG	 SMC_PULSE1; 	//  Pulse Register for CS 1
	AT91_REG	 SMC_CYCLE1; 	//  Cycle Register for CS 1
	AT91_REG	 SMC_CTRL1; 	//  Control Register for CS 1
	AT91_REG	 SMC_SETUP2; 	//  Setup Register for CS 2
	AT91_REG	 SMC_PULSE2; 	//  Pulse Register for CS 2
	AT91_REG	 SMC_CYCLE2; 	//  Cycle Register for CS 2
	AT91_REG	 SMC_CTRL2; 	//  Control Register for CS 2
	AT91_REG	 SMC_SETUP3; 	//  Setup Register for CS 3
	AT91_REG	 SMC_PULSE3; 	//  Pulse Register for CS 3
	AT91_REG	 SMC_CYCLE3; 	//  Cycle Register for CS 3
	AT91_REG	 SMC_CTRL3; 	//  Control Register for CS 3
	AT91_REG	 SMC_SETUP4; 	//  Setup Register for CS 4
	AT91_REG	 SMC_PULSE4; 	//  Pulse Register for CS 4
	AT91_REG	 SMC_CYCLE4; 	//  Cycle Register for CS 4
	AT91_REG	 SMC_CTRL4; 	//  Control Register for CS 4
	AT91_REG	 SMC_SETUP5; 	//  Setup Register for CS 5
	AT91_REG	 SMC_PULSE5; 	//  Pulse Register for CS 5
	AT91_REG	 SMC_CYCLE5; 	//  Cycle Register for CS 5
	AT91_REG	 SMC_CTRL5; 	//  Control Register for CS 5
	AT91_REG	 SMC_SETUP6; 	//  Setup Register for CS 6
	AT91_REG	 SMC_PULSE6; 	//  Pulse Register for CS 6
	AT91_REG	 SMC_CYCLE6; 	//  Cycle Register for CS 6
	AT91_REG	 SMC_CTRL6; 	//  Control Register for CS 6
	AT91_REG	 SMC_SETUP7; 	//  Setup Register for CS 7
	AT91_REG	 SMC_PULSE7; 	//  Pulse Register for CS 7
	AT91_REG	 SMC_CYCLE7; 	//  Cycle Register for CS 7
	AT91_REG	 SMC_CTRL7; 	//  Control Register for CS 7
	AT91_REG	 Reserved0[16]; 	// 
	AT91_REG	 SMC_DELAY1; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY2; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY3; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY4; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY5; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY6; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY7; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY8; 	// SMC Delay Control Register
} AT91S_SMC, *AT91PS_SMC;

// -------- SMC_SETUP : (SMC Offset: 0x0) Setup Register for CS x -------- 
#define AT91C_SMC_NWESETUP    ((unsigned int) 0x3F <<  0) // (SMC) NWE Setup Length
#define AT91C_SMC_NCSSETUPWR  ((unsigned int) 0x3F <<  8) // (SMC) NCS Setup Length in WRite Access
#define AT91C_SMC_NRDSETUP    ((unsigned int) 0x3F << 16) // (SMC) NRD Setup Length
#define AT91C_SMC_NCSSETUPRD  ((unsigned int) 0x3F << 24) // (SMC) NCS Setup Length in ReaD Access
// -------- SMC_PULSE : (SMC Offset: 0x4) Pulse Register for CS x -------- 
#define AT91C_SMC_NWEPULSE    ((unsigned int) 0x7F <<  0) // (SMC) NWE Pulse Length
#define AT91C_SMC_NCSPULSEWR  ((unsigned int) 0x7F <<  8) // (SMC) NCS Pulse Length in WRite Access
#define AT91C_SMC_NRDPULSE    ((unsigned int) 0x7F << 16) // (SMC) NRD Pulse Length
#define AT91C_SMC_NCSPULSERD  ((unsigned int) 0x7F << 24) // (SMC) NCS Pulse Length in ReaD Access
// -------- SMC_CYC : (SMC Offset: 0x8) Cycle Register for CS x -------- 
#define AT91C_SMC_NWECYCLE    ((unsigned int) 0x1FF <<  0) // (SMC) Total Write Cycle Length
#define AT91C_SMC_NRDCYCLE    ((unsigned int) 0x1FF << 16) // (SMC) Total Read Cycle Length
// -------- SMC_CTRL : (SMC Offset: 0xc) Control Register for CS x -------- 
#define AT91C_SMC_READMODE    ((unsigned int) 0x1 <<  0) // (SMC) Read Mode
#define AT91C_SMC_WRITEMODE   ((unsigned int) 0x1 <<  1) // (SMC) Write Mode
#define AT91C_SMC_NWAITM      ((unsigned int) 0x3 <<  5) // (SMC) NWAIT Mode
#define 	AT91C_SMC_NWAITM_NWAIT_DISABLE        ((unsigned int) 0x0 <<  5) // (SMC) External NWAIT disabled.
#define 	AT91C_SMC_NWAITM_NWAIT_ENABLE_FROZEN  ((unsigned int) 0x2 <<  5) // (SMC) External NWAIT enabled in frozen mode.
#define 	AT91C_SMC_NWAITM_NWAIT_ENABLE_READY   ((unsigned int) 0x3 <<  5) // (SMC) External NWAIT enabled in ready mode.
#define AT91C_SMC_BAT         ((unsigned int) 0x1 <<  8) // (SMC) Byte Access Type
#define 	AT91C_SMC_BAT_BYTE_SELECT          ((unsigned int) 0x0 <<  8) // (SMC) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.
#define 	AT91C_SMC_BAT_BYTE_WRITE           ((unsigned int) 0x1 <<  8) // (SMC) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.
#define AT91C_SMC_DBW         ((unsigned int) 0x3 << 12) // (SMC) Data Bus Width
#define 	AT91C_SMC_DBW_WIDTH_EIGTH_BITS     ((unsigned int) 0x0 << 12) // (SMC) 8 bits.
#define 	AT91C_SMC_DBW_WIDTH_SIXTEEN_BITS   ((unsigned int) 0x1 << 12) // (SMC) 16 bits.
#define 	AT91C_SMC_DBW_WIDTH_THIRTY_TWO_BITS ((unsigned int) 0x2 << 12) // (SMC) 32 bits.
#define AT91C_SMC_TDF         ((unsigned int) 0xF << 16) // (SMC) Data Float Time.
#define AT91C_SMC_TDFEN       ((unsigned int) 0x1 << 20) // (SMC) TDF Enabled.
#define AT91C_SMC_PMEN        ((unsigned int) 0x1 << 24) // (SMC) Page Mode Enabled.
#define AT91C_SMC_PS          ((unsigned int) 0x3 << 28) // (SMC) Page Size
#define 	AT91C_SMC_PS_SIZE_FOUR_BYTES      ((unsigned int) 0x0 << 28) // (SMC) 4 bytes.
#define 	AT91C_SMC_PS_SIZE_EIGHT_BYTES     ((unsigned int) 0x1 << 28) // (SMC) 8 bytes.
#define 	AT91C_SMC_PS_SIZE_SIXTEEN_BYTES   ((unsigned int) 0x2 << 28) // (SMC) 16 bytes.
#define 	AT91C_SMC_PS_SIZE_THIRTY_TWO_BYTES ((unsigned int) 0x3 << 28) // (SMC) 32 bytes.
// -------- SMC_SETUP : (SMC Offset: 0x10) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x14) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x18) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x1c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x20) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x24) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x28) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x2c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x30) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x34) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x38) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x3c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x40) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x44) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x48) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x4c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x50) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x54) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x58) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x5c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x60) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x64) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x68) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x6c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x70) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x74) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x78) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x7c) Control Register for CS x -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR AHB Matrix Interface
// *****************************************************************************
typedef struct _AT91S_MATRIX {
	AT91_REG	 MATRIX_MCFG0; 	//  Master Configuration Register 0 : ARM-I
	AT91_REG	 MATRIX_MCFG1; 	//  Master Configuration Register 1 ; ARM-D
	AT91_REG	 MATRIX_MCFG2; 	//  Master Configuration Register 2 : pdc
	AT91_REG	 MATRIX_MCFG3; 	//  Master Configuration Register 3 : USB Host OHCI
	AT91_REG	 MATRIX_MCFG4; 	//  Master Configuration Register 4 : DMA0
	AT91_REG	 MATRIX_MCFG5; 	//  Master Configuration Register 5 : DMA1
	AT91_REG	 MATRIX_MCFG6; 	//  Master Configuration Register 6 : hisi
	AT91_REG	 MATRIX_MCFG7; 	//  Master Configuration Register 7 : lcdc
	AT91_REG	 MATRIX_MCFG8; 	//  Master Configuration Register 8 : eMAC
	AT91_REG	 MATRIX_MCFG9; 	//  Master Configuration Register 9 : USB Device
	AT91_REG	 MATRIX_MCFG10; 	//  Master Configuration Register 10 : USB Host EHCI
	AT91_REG	 Reserved0[5]; 	// 
	AT91_REG	 MATRIX_SCFG0; 	//  Slave Configuration Register 0 : SRAM S0
	AT91_REG	 MATRIX_SCFG1; 	//  Slave Configuration Register 1 : SRAM S1
	AT91_REG	 MATRIX_SCFG2; 	//  Slave Configuration Register 2 : SRAM S2
	AT91_REG	 MATRIX_SCFG3; 	//  Slave Configuration Register 3 : SRAM S3
	AT91_REG	 MATRIX_SCFG4; 	//  Slave Configuration Register 4 ; ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
	AT91_REG	 MATRIX_SCFG5; 	//  Slave Configuration Register 5 : DDR2 S0
	AT91_REG	 MATRIX_SCFG6; 	//  Slave Configuration Register 6 : DDR2 S1
	AT91_REG	 MATRIX_SCFG7; 	//  Slave Configuration Register 7 : DDR2 S2
	AT91_REG	 Reserved1[8]; 	// 
	AT91_REG	 MATRIX_PRAS0; 	//  PRAS0 : SRAM S0
	AT91_REG	 MATRIX_PRBS0; 	//  PRBS0 : SRAM S0
	AT91_REG	 MATRIX_PRAS1; 	//  PRAS1 : SRAM S1
	AT91_REG	 MATRIX_PRBS1; 	//  PRBS1 : SRAM S1
	AT91_REG	 MATRIX_PRAS2; 	//  PRAS2 : SRAM S2
	AT91_REG	 MATRIX_PRBS2; 	//  PRBS2 : SRAM S2
	AT91_REG	 MATRIX_PRAS3; 	//  PRAS3 : SRAM S3
	AT91_REG	 MATRIX_PRBS3; 	//  PRBS3 : SRAM S3
	AT91_REG	 MATRIX_PRAS4; 	//  PRAS4 : ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
	AT91_REG	 MATRIX_PRBS4; 	//  PRBS4 : ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
	AT91_REG	 MATRIX_PRAS5; 	//  PRAS5 : DDR2 S0
	AT91_REG	 MATRIX_PRBS5; 	//  PRBS5 : DDR2 S0
	AT91_REG	 MATRIX_PRAS6; 	//  PRAS6 : DDR2 S1
	AT91_REG	 MATRIX_PRBS6; 	//  PRBS6 : DDR2 S1
	AT91_REG	 MATRIX_PRAS7; 	//  PRAS7 : DDR2 S2
	AT91_REG	 MATRIX_PRBS7; 	//  PRBS7 : DDR2 S2
	AT91_REG	 Reserved2[16]; 	// 
	AT91_REG	 MATRIX_MRCR; 	//  Master Remap Control Register 
	AT91_REG	 Reserved3[3]; 	// 
	AT91_REG	 MATRIX_TCMR; 	//  Bus Matrix TCM Configuration Register 
	AT91_REG	 Reserved4[1]; 	// 
	AT91_REG	 MATRIX_DDRMPR; 	//  DDR Multi-Port Register 
	AT91_REG	 Reserved5[3]; 	// 
	AT91_REG	 MATRIX_EBICSA; 	//  EBI Chip Select Assignment Register 
	AT91_REG	 Reserved6[46]; 	// 
	AT91_REG	 MATRIX_WRPROTEN; 	//  Write Protection Control Register 
	AT91_REG	 MATRIX_WRPROTST; 	//  Write Protection Status Register 
} AT91S_MATRIX, *AT91PS_MATRIX;

// -------- MATRIX_MCFG0 : (MATRIX Offset: 0x0) Master Configuration Register ARM-I -------- 
#define AT91C_MATRIX_ULBT     ((unsigned int) 0x7 <<  0) // (MATRIX) Undefined Length Burst Type
// -------- MATRIX_MCFG1 : (MATRIX Offset: 0x4) Master Configuration Register ARM-D -------- 
// -------- MATRIX_MCFG2 : (MATRIX Offset: 0x8) Master Configuration Register PDC -------- 
// -------- MATRIX_MCFG3 : (MATRIX Offset: 0xc) Master Configuration Register USB Host OHCI -------- 
// -------- MATRIX_MCFG4 : (MATRIX Offset: 0x10) Master Configuration Register DMA0 -------- 
// -------- MATRIX_MCFG5 : (MATRIX Offset: 0x14) Master Configuration Register DMA1 -------- 
// -------- MATRIX_MCFG6 : (MATRIX Offset: 0x18) Master Configuration Register HISI -------- 
// -------- MATRIX_MCFG7 : (MATRIX Offset: 0x1c) Master Configuration Register LCD -------- 
// -------- MATRIX_MCFG8 : (MATRIX Offset: 0x20) Master Configuration Register EMAC -------- 
// -------- MATRIX_MCFG9 : (MATRIX Offset: 0x24) Master Configuration Register USB Device -------- 
// -------- MATRIX_MCFG10 : (MATRIX Offset: 0x28) Master Configuration Register USB Host EHCI -------- 
// -------- MATRIX_SCFG0 : (MATRIX Offset: 0x40) Slave Configuration Register 0 -------- 
#define AT91C_MATRIX_SLOT_CYCLE ((unsigned int) 0xFF <<  0) // (MATRIX) Maximum Number of Allowed Cycles for a Burst
#define AT91C_MATRIX_DEFMSTR_TYPE ((unsigned int) 0x3 << 16) // (MATRIX) Default Master Type
#define 	AT91C_MATRIX_DEFMSTR_TYPE_NO_DEFMSTR           ((unsigned int) 0x0 << 16) // (MATRIX) No Default Master. At the end of current slave access, if no other master request is pending, the slave is deconnected from all masters. This results in having a one cycle latency for the first transfer of a burst.
#define 	AT91C_MATRIX_DEFMSTR_TYPE_LAST_DEFMSTR         ((unsigned int) 0x1 << 16) // (MATRIX) Last Default Master. At the end of current slave access, if no other master request is pending, the slave stay connected with the last master having accessed it. This results in not having the one cycle latency when the last master re-trying access on the slave.
#define 	AT91C_MATRIX_DEFMSTR_TYPE_FIXED_DEFMSTR        ((unsigned int) 0x2 << 16) // (MATRIX) Fixed Default Master. At the end of current slave access, if no other master request is pending, the slave connects with fixed which number is in FIXED_DEFMSTR field. This results in not having the one cycle latency when the fixed master re-trying access on the slave.
#define AT91C_MATRIX_FIXED_DEFMSTR0 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_ARM926I              ((unsigned int) 0x0 << 18) // (MATRIX) ARM926EJ-S Instruction Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_ARM926D              ((unsigned int) 0x1 << 18) // (MATRIX) ARM926EJ-S Data Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_PDC                  ((unsigned int) 0x2 << 18) // (MATRIX) PDC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_UHPHS_OHCI           ((unsigned int) 0x3 << 18) // (MATRIX) USB Host OHCI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_DMA0                 ((unsigned int) 0x4 << 18) // (MATRIX) DMA0 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_DMA1                 ((unsigned int) 0x5 << 18) // (MATRIX) DMA1 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_ISI                  ((unsigned int) 0x6 << 18) // (MATRIX) ISI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_EMAC                 ((unsigned int) 0x8 << 18) // (MATRIX) EMAC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_UDPHS                ((unsigned int) 0x9 << 18) // (MATRIX) USB Device Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_UHPHS_EHCI           ((unsigned int) 0xA << 18) // (MATRIX) USB Host EHCI Master is Default Master
#define AT91C_MATRIX_ARBT     ((unsigned int) 0x3 << 24) // (MATRIX) Arbitration Type
// -------- MATRIX_SCFG1 : (MATRIX Offset: 0x44) Slave Configuration Register 1 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR1 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_ARM926I              ((unsigned int) 0x0 << 18) // (MATRIX) ARM926EJ-S Instruction Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_ARM926D              ((unsigned int) 0x1 << 18) // (MATRIX) ARM926EJ-S Data Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_PDC                  ((unsigned int) 0x2 << 18) // (MATRIX) PDC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_ISI                  ((unsigned int) 0x6 << 18) // (MATRIX) ISI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_UDPHS                ((unsigned int) 0x9 << 18) // (MATRIX) USB Device Master is Default Master
// -------- MATRIX_SCFG2 : (MATRIX Offset: 0x48) Slave Configuration Register 2 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR2 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR2_ARM926I              ((unsigned int) 0x0 << 18) // (MATRIX) ARM926EJ-S Instruction Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR2_VDEC                 ((unsigned int) 0xB << 18) // (MATRIX) Video Decoder Master is Default Master
// -------- MATRIX_SCFG3 : (MATRIX Offset: 0x4c) Slave Configuration Register 3 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR3 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_ARM926D              ((unsigned int) 0x1 << 18) // (MATRIX) ARM926EJ-S Data Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_LCD                  ((unsigned int) 0x7 << 18) // (MATRIX) LCD Master is Default Master
// -------- MATRIX_SCFG4 : (MATRIX Offset: 0x50) Slave Configuration Register 4 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR4 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_ARM926I              ((unsigned int) 0x0 << 18) // (MATRIX) ARM926EJ-S Instruction Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_PDC                  ((unsigned int) 0x2 << 18) // (MATRIX) PDC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_UHPHS_OHCI           ((unsigned int) 0x3 << 18) // (MATRIX) USB Host OHCI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_DMA0                 ((unsigned int) 0x4 << 18) // (MATRIX) DMA0 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_DMA1                 ((unsigned int) 0x5 << 18) // (MATRIX) DMA1 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_ISI                  ((unsigned int) 0x6 << 18) // (MATRIX) ISI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_EMAC                 ((unsigned int) 0x8 << 18) // (MATRIX) EMAC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_UDPHS                ((unsigned int) 0x9 << 18) // (MATRIX) USB Device Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_UHPHS_EHCI           ((unsigned int) 0xA << 18) // (MATRIX) USB Host EHCI Master is Default Master
// -------- MATRIX_SCFG5 : (MATRIX Offset: 0x54) Slave Configuration Register 5 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR5 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_ARM926D              ((unsigned int) 0x1 << 18) // (MATRIX) ARM926EJ-S Data Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_PDC                  ((unsigned int) 0x2 << 18) // (MATRIX) PDC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_UHPHS_OHCI           ((unsigned int) 0x3 << 18) // (MATRIX) USB Host OHCI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_DMA0                 ((unsigned int) 0x4 << 18) // (MATRIX) DMA0 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_DMA1                 ((unsigned int) 0x5 << 18) // (MATRIX) DMA1 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_ISI                  ((unsigned int) 0x6 << 18) // (MATRIX) ISI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_EMAC                 ((unsigned int) 0x8 << 18) // (MATRIX) EMAC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_UDPHS                ((unsigned int) 0x9 << 18) // (MATRIX) USB Device Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_UHPHS_EHCI           ((unsigned int) 0xA << 18) // (MATRIX) USB Host EHCI Master is Default Master
// -------- MATRIX_SCFG6 : (MATRIX Offset: 0x58) Slave Configuration Register 6 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR6 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_ARM926I              ((unsigned int) 0x0 << 18) // (MATRIX) ARM926EJ-S Instruction Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_ARM926D              ((unsigned int) 0x1 << 18) // (MATRIX) ARM926EJ-S Data Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_PDC                  ((unsigned int) 0x2 << 18) // (MATRIX) PDC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_UHPHS_OHCI           ((unsigned int) 0x3 << 18) // (MATRIX) USB Host OHCI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_DMA0                 ((unsigned int) 0x4 << 18) // (MATRIX) DMA0 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_DMA1                 ((unsigned int) 0x5 << 18) // (MATRIX) DMA1 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_ISI                  ((unsigned int) 0x6 << 18) // (MATRIX) ISI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_LCD                  ((unsigned int) 0x7 << 18) // (MATRIX) LCD Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_EMAC                 ((unsigned int) 0x8 << 18) // (MATRIX) EMAC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_UDPHS                ((unsigned int) 0x9 << 18) // (MATRIX) USB Device Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_UHPHS_EHCI           ((unsigned int) 0xA << 18) // (MATRIX) USB Host EHCI Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_VDEC                 ((unsigned int) 0xB << 18) // (MATRIX) Video Decoder Master is Default Master
// -------- MATRIX_SCFG7 : (MATRIX Offset: 0x5c) Slave Configuration Register 7 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR7 ((unsigned int) 0xF << 18) // (MATRIX) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_ARM926I              ((unsigned int) 0x0 << 18) // (MATRIX) ARM926EJ-S Instruction Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_ARM926D              ((unsigned int) 0x1 << 18) // (MATRIX) ARM926EJ-S Data Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_PDC                  ((unsigned int) 0x2 << 18) // (MATRIX) PDC Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_DMA0                 ((unsigned int) 0x4 << 18) // (MATRIX) DMA0 Master is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_DMA1                 ((unsigned int) 0x5 << 18) // (MATRIX) DMA1 Master is Default Master
// -------- MATRIX_PRAS0 : (MATRIX Offset: 0x80) PRAS0 Register -------- 
#define AT91C_MATRIX_M0PR     ((unsigned int) 0x3 <<  0) // (MATRIX) ARM926EJ-S Instruction priority
#define AT91C_MATRIX_M1PR     ((unsigned int) 0x3 <<  4) // (MATRIX) ARM926EJ-S Data priority
#define AT91C_MATRIX_M2PR     ((unsigned int) 0x3 <<  8) // (MATRIX) PDC priority
#define AT91C_MATRIX_M3PR     ((unsigned int) 0x3 << 12) // (MATRIX) USB Host OHCI priority
#define AT91C_MATRIX_M4PR     ((unsigned int) 0x3 << 16) // (MATRIX) DMA0 priority
#define AT91C_MATRIX_M5PR     ((unsigned int) 0x3 << 20) // (MATRIX) DMA1 priority
#define AT91C_MATRIX_M6PR     ((unsigned int) 0x3 << 24) // (MATRIX) ISI priority
#define AT91C_MATRIX_M7PR     ((unsigned int) 0x3 << 28) // (MATRIX) LCD priority
// -------- MATRIX_PRBS0 : (MATRIX Offset: 0x84) PRBS0 Register -------- 
#define AT91C_MATRIX_M8PR     ((unsigned int) 0x3 <<  0) // (MATRIX) EMAC priority
#define AT91C_MATRIX_M9PR     ((unsigned int) 0x3 <<  4) // (MATRIX) USB Device priority
#define AT91C_MATRIX_M10PR    ((unsigned int) 0x3 <<  8) // (MATRIX) USB Host EHCI priority
#define AT91C_MATRIX_M11PR    ((unsigned int) 0x3 << 12) // (MATRIX) Video Decoder priority
// -------- MATRIX_PRAS1 : (MATRIX Offset: 0x88) PRAS1 Register -------- 
// -------- MATRIX_PRBS1 : (MATRIX Offset: 0x8c) PRBS1 Register -------- 
// -------- MATRIX_PRAS2 : (MATRIX Offset: 0x90) PRAS2 Register -------- 
// -------- MATRIX_PRBS2 : (MATRIX Offset: 0x94) PRBS2 Register -------- 
// -------- MATRIX_PRAS3 : (MATRIX Offset: 0x98) PRAS3 Register -------- 
// -------- MATRIX_PRBS3 : (MATRIX Offset: 0x9c) PRBS3 Register -------- 
// -------- MATRIX_PRAS4 : (MATRIX Offset: 0xa0) PRAS4 Register -------- 
// -------- MATRIX_PRBS4 : (MATRIX Offset: 0xa4) PRBS4 Register -------- 
// -------- MATRIX_PRAS5 : (MATRIX Offset: 0xa8) PRAS5 Register -------- 
// -------- MATRIX_PRBS5 : (MATRIX Offset: 0xac) PRBS5 Register -------- 
// -------- MATRIX_PRAS6 : (MATRIX Offset: 0xb0) PRAS6 Register -------- 
// -------- MATRIX_PRBS6 : (MATRIX Offset: 0xb4) PRBS6 Register -------- 
// -------- MATRIX_PRAS7 : (MATRIX Offset: 0xb8) PRAS7 Register -------- 
// -------- MATRIX_PRBS7 : (MATRIX Offset: 0xbc) PRBS7 Register -------- 
// -------- MATRIX_MRCR : (MATRIX Offset: 0x100) MRCR Register -------- 
#define AT91C_MATRIX_RCA926I  ((unsigned int) 0x1 <<  0) // (MATRIX) Remap Command Bit for ARM926EJ-S Instruction
#define AT91C_MATRIX_RCA926D  ((unsigned int) 0x1 <<  1) // (MATRIX) Remap Command Bit for ARM926EJ-S Data
#define AT91C_MATRIX_RCB2     ((unsigned int) 0x1 <<  2) // (MATRIX) Remap Command Bit for PDC
#define AT91C_MATRIX_RCB3     ((unsigned int) 0x1 <<  3) // (MATRIX) Remap Command Bit for USB Host OHCI
#define AT91C_MATRIX_RCB4     ((unsigned int) 0x1 <<  4) // (MATRIX) Remap Command Bit for DMA0
#define AT91C_MATRIX_RCB5     ((unsigned int) 0x1 <<  5) // (MATRIX) Remap Command Bit for DMA1
#define AT91C_MATRIX_RCB6     ((unsigned int) 0x1 <<  6) // (MATRIX) Remap Command Bit for ISI
#define AT91C_MATRIX_RCB7     ((unsigned int) 0x1 <<  7) // (MATRIX) Remap Command Bit for LCD
#define AT91C_MATRIX_RCB8     ((unsigned int) 0x1 <<  8) // (MATRIX) Remap Command Bit for EMAC
#define AT91C_MATRIX_RCB9     ((unsigned int) 0x1 <<  9) // (MATRIX) Remap Command Bit for USB Device
#define AT91C_MATRIX_RCB10    ((unsigned int) 0x1 << 10) // (MATRIX) Remap Command Bit for USB Host EHCI
#define AT91C_MATRIX_RCB11    ((unsigned int) 0x1 << 11) // (MATRIX) Remap Command Bit for Video Decoder
// -------- HMATRIX_CCFG_TCMR : (MATRIX Offset: 0x110) TCMR Register -------- 
#define AT91C_ITCM_SIZE       ((unsigned int) 0xF <<  0) // (MATRIX) Size of ITCM enabled memory block
#define 	AT91C_ITCM_SIZE_0KB                  ((unsigned int) 0x0) // (MATRIX) 0 KB (No ITCM Memory)
#define 	AT91C_ITCM_SIZE_32KB                 ((unsigned int) 0x6) // (MATRIX) 32 KB
#define AT91C_DTCM_SIZE       ((unsigned int) 0xF <<  4) // (MATRIX) Size of DTCM enabled memory block
#define 	AT91C_DTCM_SIZE_0KB                  ((unsigned int) 0x0 <<  4) // (MATRIX) 0 KB (No DTCM Memory)
#define 	AT91C_DTCM_SIZE_32KB                 ((unsigned int) 0x6 <<  4) // (MATRIX) 32 KB
#define 	AT91C_DTCM_SIZE_64KB                 ((unsigned int) 0x7 <<  4) // (MATRIX) 64 KB
#define AT91C_TCM_NWS         ((unsigned int) 0x1 << 11) // (MATRIX) TCM Wait State
#define 	AT91C_TCM_NWS_NO_WS                ((unsigned int) 0x0 << 11) // (MATRIX) NO WAIT STATE : 0 WS
#define 	AT91C_TCM_NWS_ONE_WS               ((unsigned int) 0x1 << 11) // (MATRIX) 1 WS activated (only for RATIO 3:1 or 4:1
// -------- HMATRIX_CCFG_DDRMPR : (MATRIX Offset: 0x118) DDR Multi-Port Register -------- 
#define AT91C_DDRMP_DIS       ((unsigned int) 0x1 <<  0) // (MATRIX) DDR Multi-Port Disable
// -------- HMATRIX_CCFG_EBICSA : (MATRIX Offset: 0x128) CCFG_EBICSA Register -------- 
#define AT91C_EBI_CS1A        ((unsigned int) 0x1 <<  1) // (MATRIX) EBI Chip Select 1 Assignment
#define 	AT91C_EBI_CS1A_SMC                  ((unsigned int) 0x0 <<  1) // (MATRIX) Chip Select 1 is assigned to the Static Memory Controller.
#define 	AT91C_EBI_CS1A_SDRAMC               ((unsigned int) 0x1 <<  1) // (MATRIX) Chip Select 1 is assigned to the SDRAM Controller.
#define AT91C_EBI_CS3A        ((unsigned int) 0x1 <<  3) // (MATRIX) EBI Chip Select 3 Assignment
#define 	AT91C_EBI_CS3A_SMC                  ((unsigned int) 0x0 <<  3) // (MATRIX) Chip Select 3 is only assigned to the Static Memory Controller and NCS3 behaves as defined by the SMC.
#define 	AT91C_EBI_CS3A_SM                   ((unsigned int) 0x1 <<  3) // (MATRIX) Chip Select 3 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
#define AT91C_EBI_CS4A        ((unsigned int) 0x1 <<  4) // (MATRIX) EBI Chip Select 4 Assignment
#define 	AT91C_EBI_CS4A_SMC                  ((unsigned int) 0x0 <<  4) // (MATRIX) Chip Select 4 is only assigned to the Static Memory Controller and NCS4 behaves as defined by the SMC.
#define 	AT91C_EBI_CS4A_CF                   ((unsigned int) 0x1 <<  4) // (MATRIX) Chip Select 4 is assigned to the Static Memory Controller and the CompactFlash Logic (first slot) is activated.
#define AT91C_EBI_CS5A        ((unsigned int) 0x1 <<  5) // (MATRIX) EBI Chip Select 5 Assignment
#define 	AT91C_EBI_CS5A_SMC                  ((unsigned int) 0x0 <<  5) // (MATRIX) Chip Select 5 is only assigned to the Static Memory Controller and NCS5 behaves as defined by the SMC
#define 	AT91C_EBI_CS5A_CF                   ((unsigned int) 0x1 <<  5) // (MATRIX) Chip Select 5 is assigned to the Static Memory Controller and the CompactFlash Logic (second slot) is activated.
#define AT91C_EBI_DBPUC       ((unsigned int) 0x1 <<  8) // (MATRIX) EBI Data Bus Pull-Up Configuration
#define AT91C_EBI_DRIVE       ((unsigned int) 0x3 << 16) // (MATRIX) EBI I/O Drive Configuration
#define 	AT91C_EBI_DRIVE_18RD                 ((unsigned int) 0x0 << 16) // (MATRIX) optimized for 1.8V powered memories with Reduced Drive
#define 	AT91C_EBI_DRIVE_33RD                 ((unsigned int) 0x1 << 16) // (MATRIX) optimized for 3.3V powered memories with Reduced Drive
#define 	AT91C_EBI_DRIVE_18ND                 ((unsigned int) 0x2 << 16) // (MATRIX) optimized for 1.8V powered memories with Normal Drive
#define 	AT91C_EBI_DRIVE_33ND                 ((unsigned int) 0x3 << 16) // (MATRIX) optimized for 3.3V powered memories with Normal Drive
#define AT91C_DDR_DRIVE       ((unsigned int) 0x1 << 18) // (MATRIX) DDR2 dedicated port I/O slew rate selection
// -------- MATRIX_WRPROTEN : (MATRIX Offset: 0x1e4) Write Protection Control Register -------- 
#define AT91C_MATRIX_WRPROT   ((unsigned int) 0x1 <<  0) // (MATRIX) Enable/Disable Write Protection of HMATRIX2 configuration registers (requires key)
#define 	AT91C_MATRIX_WRPROT_DISABLE              ((unsigned int) 0x0) // (MATRIX) Disable Write Protection of HMATRIX2 configuration registers
#define 	AT91C_MATRIX_WRPROT_ENABLE               ((unsigned int) 0x1) // (MATRIX) Enable  Write Protection of HMATRIX2 configuration registers
// -------- MATRIX_WRPROTST : (MATRIX Offset: 0x1e8) Write Protection Status Register -------- 
#define AT91C_MATRIX_WRPROT_VIOLATION ((unsigned int) 0x1 <<  0) // (MATRIX) Violation of Write Protection of HMATRIX2 configuration registers
#define 	AT91C_MATRIX_WRPROT_VIOLATION_UNDETECTED           ((unsigned int) 0x0) // (MATRIX) No violation of Write Protection of HMATRIX2 configuration registers
#define 	AT91C_MATRIX_WRPROT_VIOLATION_DETECTED             ((unsigned int) 0x1) // (MATRIX) Violation of  Write Protection of HMATRIX2 configuration registers
#define AT91C_MATRIX_WRPROT_VIOLATION_OFFSET ((unsigned int) 0x1FF <<  8) // (MATRIX) Offset where violation of Write Protection of HMATRIX2 configuration registers is detected

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Advanced Interrupt Controller
// *****************************************************************************
typedef struct _AT91S_AIC {
	AT91_REG	 AIC_SMR[32]; 	// Source Mode Register
	AT91_REG	 AIC_SVR[32]; 	// Source Vector Register
	AT91_REG	 AIC_IVR; 	// IRQ Vector Register
	AT91_REG	 AIC_FVR; 	// FIQ Vector Register
	AT91_REG	 AIC_ISR; 	// Interrupt Status Register
	AT91_REG	 AIC_IPR; 	// Interrupt Pending Register
	AT91_REG	 AIC_IMR; 	// Interrupt Mask Register
	AT91_REG	 AIC_CISR; 	// Core Interrupt Status Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 AIC_IECR; 	// Interrupt Enable Command Register
	AT91_REG	 AIC_IDCR; 	// Interrupt Disable Command Register
	AT91_REG	 AIC_ICCR; 	// Interrupt Clear Command Register
	AT91_REG	 AIC_ISCR; 	// Interrupt Set Command Register
	AT91_REG	 AIC_EOICR; 	// End of Interrupt Command Register
	AT91_REG	 AIC_SPU; 	// Spurious Vector Register
	AT91_REG	 AIC_DCR; 	// Debug Control Register (Protect)
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 AIC_FFER; 	// Fast Forcing Enable Register
	AT91_REG	 AIC_FFDR; 	// Fast Forcing Disable Register
	AT91_REG	 AIC_FFSR; 	// Fast Forcing Status Register
} AT91S_AIC, *AT91PS_AIC;

// -------- AIC_SMR : (AIC Offset: 0x0) Control Register -------- 
#define AT91C_AIC_PRIOR       ((unsigned int) 0x7 <<  0) // (AIC) Priority Level
#define 	AT91C_AIC_PRIOR_LOWEST               ((unsigned int) 0x0) // (AIC) Lowest priority level
#define 	AT91C_AIC_PRIOR_HIGHEST              ((unsigned int) 0x7) // (AIC) Highest priority level
#define AT91C_AIC_SRCTYPE     ((unsigned int) 0x3 <<  5) // (AIC) Interrupt Source Type
#define 	AT91C_AIC_SRCTYPE_INT_LEVEL_SENSITIVE  ((unsigned int) 0x0 <<  5) // (AIC) Internal Sources Code Label Level Sensitive
#define 	AT91C_AIC_SRCTYPE_INT_EDGE_TRIGGERED   ((unsigned int) 0x1 <<  5) // (AIC) Internal Sources Code Label Edge triggered
#define 	AT91C_AIC_SRCTYPE_EXT_HIGH_LEVEL       ((unsigned int) 0x2 <<  5) // (AIC) External Sources Code Label High-level Sensitive
#define 	AT91C_AIC_SRCTYPE_EXT_POSITIVE_EDGE    ((unsigned int) 0x3 <<  5) // (AIC) External Sources Code Label Positive Edge triggered
// -------- AIC_CISR : (AIC Offset: 0x114) AIC Core Interrupt Status Register -------- 
#define AT91C_AIC_NFIQ        ((unsigned int) 0x1 <<  0) // (AIC) NFIQ Status
#define AT91C_AIC_NIRQ        ((unsigned int) 0x1 <<  1) // (AIC) NIRQ Status
// -------- AIC_DCR : (AIC Offset: 0x138) AIC Debug Control Register (Protect) -------- 
#define AT91C_AIC_DCR_PROT    ((unsigned int) 0x1 <<  0) // (AIC) Protection Mode
#define AT91C_AIC_DCR_GMSK    ((unsigned int) 0x1 <<  1) // (AIC) General Mask

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Peripheral DMA Controller
// *****************************************************************************
typedef struct _AT91S_PDC {
	AT91_REG	 PDC_RPR; 	// Receive Pointer Register
	AT91_REG	 PDC_RCR; 	// Receive Counter Register
	AT91_REG	 PDC_TPR; 	// Transmit Pointer Register
	AT91_REG	 PDC_TCR; 	// Transmit Counter Register
	AT91_REG	 PDC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PDC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PDC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PDC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PDC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PDC_PTSR; 	// PDC Transfer Status Register
} AT91S_PDC, *AT91PS_PDC;

// -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- 
#define AT91C_PDC_RXTEN       ((unsigned int) 0x1 <<  0) // (PDC) Receiver Transfer Enable
#define AT91C_PDC_RXTDIS      ((unsigned int) 0x1 <<  1) // (PDC) Receiver Transfer Disable
#define AT91C_PDC_TXTEN       ((unsigned int) 0x1 <<  8) // (PDC) Transmitter Transfer Enable
#define AT91C_PDC_TXTDIS      ((unsigned int) 0x1 <<  9) // (PDC) Transmitter Transfer Disable
// -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Debug Unit
// *****************************************************************************
typedef struct _AT91S_DBGU {
	AT91_REG	 DBGU_CR; 	// Control Register
	AT91_REG	 DBGU_MR; 	// Mode Register
	AT91_REG	 DBGU_IER; 	// Interrupt Enable Register
	AT91_REG	 DBGU_IDR; 	// Interrupt Disable Register
	AT91_REG	 DBGU_IMR; 	// Interrupt Mask Register
	AT91_REG	 DBGU_CSR; 	// Channel Status Register
	AT91_REG	 DBGU_RHR; 	// Receiver Holding Register
	AT91_REG	 DBGU_THR; 	// Transmitter Holding Register
	AT91_REG	 DBGU_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 Reserved0[7]; 	// 
	AT91_REG	 DBGU_CIDR; 	// Chip ID Register
	AT91_REG	 DBGU_EXID; 	// Chip ID Extension Register
	AT91_REG	 DBGU_FNTR; 	// Force NTRST Register
	AT91_REG	 Reserved1[45]; 	// 
	AT91_REG	 DBGU_RPR; 	// Receive Pointer Register
	AT91_REG	 DBGU_RCR; 	// Receive Counter Register
	AT91_REG	 DBGU_TPR; 	// Transmit Pointer Register
	AT91_REG	 DBGU_TCR; 	// Transmit Counter Register
	AT91_REG	 DBGU_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 DBGU_RNCR; 	// Receive Next Counter Register
	AT91_REG	 DBGU_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 DBGU_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 DBGU_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 DBGU_PTSR; 	// PDC Transfer Status Register
} AT91S_DBGU, *AT91PS_DBGU;

// -------- DBGU_CR : (DBGU Offset: 0x0) Debug Unit Control Register -------- 
#define AT91C_US_RSTRX        ((unsigned int) 0x1 <<  2) // (DBGU) Reset Receiver
#define AT91C_US_RSTTX        ((unsigned int) 0x1 <<  3) // (DBGU) Reset Transmitter
#define AT91C_US_RXEN         ((unsigned int) 0x1 <<  4) // (DBGU) Receiver Enable
#define AT91C_US_RXDIS        ((unsigned int) 0x1 <<  5) // (DBGU) Receiver Disable
#define AT91C_US_TXEN         ((unsigned int) 0x1 <<  6) // (DBGU) Transmitter Enable
#define AT91C_US_TXDIS        ((unsigned int) 0x1 <<  7) // (DBGU) Transmitter Disable
#define AT91C_US_RSTSTA       ((unsigned int) 0x1 <<  8) // (DBGU) Reset Status Bits
// -------- DBGU_MR : (DBGU Offset: 0x4) Debug Unit Mode Register -------- 
#define AT91C_US_PAR          ((unsigned int) 0x7 <<  9) // (DBGU) Parity type
#define 	AT91C_US_PAR_EVEN                 ((unsigned int) 0x0 <<  9) // (DBGU) Even Parity
#define 	AT91C_US_PAR_ODD                  ((unsigned int) 0x1 <<  9) // (DBGU) Odd Parity
#define 	AT91C_US_PAR_SPACE                ((unsigned int) 0x2 <<  9) // (DBGU) Parity forced to 0 (Space)
#define 	AT91C_US_PAR_MARK                 ((unsigned int) 0x3 <<  9) // (DBGU) Parity forced to 1 (Mark)
#define 	AT91C_US_PAR_NONE                 ((unsigned int) 0x4 <<  9) // (DBGU) No Parity
#define 	AT91C_US_PAR_MULTI_DROP           ((unsigned int) 0x6 <<  9) // (DBGU) Multi-drop mode
#define AT91C_US_CHMODE       ((unsigned int) 0x3 << 14) // (DBGU) Channel Mode
#define 	AT91C_US_CHMODE_NORMAL               ((unsigned int) 0x0 << 14) // (DBGU) Normal Mode: The USART channel operates as an RX/TX USART.
#define 	AT91C_US_CHMODE_AUTO                 ((unsigned int) 0x1 << 14) // (DBGU) Automatic Echo: Receiver Data Input is connected to the TXD pin.
#define 	AT91C_US_CHMODE_LOCAL                ((unsigned int) 0x2 << 14) // (DBGU) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
#define 	AT91C_US_CHMODE_REMOTE               ((unsigned int) 0x3 << 14) // (DBGU) Remote Loopback: RXD pin is internally connected to TXD pin.
// -------- DBGU_IER : (DBGU Offset: 0x8) Debug Unit Interrupt Enable Register -------- 
#define AT91C_US_RXRDY        ((unsigned int) 0x1 <<  0) // (DBGU) RXRDY Interrupt
#define AT91C_US_TXRDY        ((unsigned int) 0x1 <<  1) // (DBGU) TXRDY Interrupt
#define AT91C_US_ENDRX        ((unsigned int) 0x1 <<  3) // (DBGU) End of Receive Transfer Interrupt
#define AT91C_US_ENDTX        ((unsigned int) 0x1 <<  4) // (DBGU) End of Transmit Interrupt
#define AT91C_US_OVRE         ((unsigned int) 0x1 <<  5) // (DBGU) Overrun Interrupt
#define AT91C_US_FRAME        ((unsigned int) 0x1 <<  6) // (DBGU) Framing Error Interrupt
#define AT91C_US_PARE         ((unsigned int) 0x1 <<  7) // (DBGU) Parity Error Interrupt
#define AT91C_US_TXEMPTY      ((unsigned int) 0x1 <<  9) // (DBGU) TXEMPTY Interrupt
#define AT91C_US_TXBUFE       ((unsigned int) 0x1 << 11) // (DBGU) TXBUFE Interrupt
#define AT91C_US_RXBUFF       ((unsigned int) 0x1 << 12) // (DBGU) RXBUFF Interrupt
#define AT91C_US_COMM_TX      ((unsigned int) 0x1 << 30) // (DBGU) COMM_TX Interrupt
#define AT91C_US_COMM_RX      ((unsigned int) 0x1 << 31) // (DBGU) COMM_RX Interrupt
// -------- DBGU_IDR : (DBGU Offset: 0xc) Debug Unit Interrupt Disable Register -------- 
// -------- DBGU_IMR : (DBGU Offset: 0x10) Debug Unit Interrupt Mask Register -------- 
// -------- DBGU_CSR : (DBGU Offset: 0x14) Debug Unit Channel Status Register -------- 
// -------- DBGU_FNTR : (DBGU Offset: 0x48) Debug Unit FORCE_NTRST Register -------- 
#define AT91C_US_FORCE_NTRST  ((unsigned int) 0x1 <<  0) // (DBGU) Force NTRST in JTAG

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler
// *****************************************************************************
typedef struct _AT91S_PIO {
	AT91_REG	 PIO_PER; 	// PIO Enable Register
	AT91_REG	 PIO_PDR; 	// PIO Disable Register
	AT91_REG	 PIO_PSR; 	// PIO Status Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 PIO_OER; 	// Output Enable Register
	AT91_REG	 PIO_ODR; 	// Output Disable Registerr
	AT91_REG	 PIO_OSR; 	// Output Status Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 PIO_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIO_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIO_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved2[1]; 	// 
	AT91_REG	 PIO_SODR; 	// Set Output Data Register
	AT91_REG	 PIO_CODR; 	// Clear Output Data Register
	AT91_REG	 PIO_ODSR; 	// Output Data Status Register
	AT91_REG	 PIO_PDSR; 	// Pin Data Status Register
	AT91_REG	 PIO_IER; 	// Interrupt Enable Register
	AT91_REG	 PIO_IDR; 	// Interrupt Disable Register
	AT91_REG	 PIO_IMR; 	// Interrupt Mask Register
	AT91_REG	 PIO_ISR; 	// Interrupt Status Register
	AT91_REG	 PIO_MDER; 	// Multi-driver Enable Register
	AT91_REG	 PIO_MDDR; 	// Multi-driver Disable Register
	AT91_REG	 PIO_MDSR; 	// Multi-driver Status Register
	AT91_REG	 Reserved3[1]; 	// 
	AT91_REG	 PIO_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIO_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIO_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved4[1]; 	// 
	AT91_REG	 PIO_ASR; 	// Select A Register
	AT91_REG	 PIO_BSR; 	// Select B Register
	AT91_REG	 PIO_ABSR; 	// AB Select Status Register
	AT91_REG	 Reserved5[9]; 	// 
	AT91_REG	 PIO_OWER; 	// Output Write Enable Register
	AT91_REG	 PIO_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIO_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved6[1]; 	// 
	AT91_REG	 PIO_SLEWRATE1; 	// PIO Slewrate Control Register
	AT91_REG	 Reserved7[3]; 	// 
	AT91_REG	 PIO_DELAY1; 	// PIO Delay Control Register
	AT91_REG	 PIO_DELAY2; 	// PIO Delay Control Register
	AT91_REG	 PIO_DELAY3; 	// PIO Delay Control Register
	AT91_REG	 PIO_DELAY4; 	// PIO Delay Control Register
	AT91_REG	 Reserved8[11]; 	// 
	AT91_REG	 PIO_VERSION; 	// PIO Version Register
} AT91S_PIO, *AT91PS_PIO;


// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Power Management Controller V610
// *****************************************************************************
typedef struct _AT91S_PMC {
	AT91_REG	 PMC_SCER; 	// System Clock Enable Register
	AT91_REG	 PMC_SCDR; 	// System Clock Disable Register
	AT91_REG	 PMC_SCSR; 	// System Clock Status Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 PMC_PCER; 	// Peripheral Clock Enable Register
	AT91_REG	 PMC_PCDR; 	// Peripheral Clock Disable Register
	AT91_REG	 PMC_PCSR; 	// Peripheral Clock Status Register
	AT91_REG	 PMC_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 PMC_MOR; 	// Main Oscillator Register
	AT91_REG	 PMC_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 PMC_PLLAR; 	// PLL A Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 PMC_MCKR; 	// Master Clock Register
	AT91_REG	 Reserved2[1]; 	// 
	AT91_REG	 PMC_USB; 	// USB clock register
	AT91_REG	 Reserved3[1]; 	// 
	AT91_REG	 PMC_PCKR[2]; 	// Programmable Clock 0 Register
	AT91_REG	 Reserved4[6]; 	// 
	AT91_REG	 PMC_IER; 	// Interrupt Enable Register
	AT91_REG	 PMC_IDR; 	// Interrupt Disable Register
	AT91_REG	 PMC_SR; 	// Status Register
	AT91_REG	 PMC_IMR; 	// Interrupt Mask Register
	AT91_REG	 Reserved5[4]; 	// 
	AT91_REG	 PMC_PLLICPR; 	// PLL Charge Pump Current Register
	AT91_REG	 Reserved6[26]; 	// 
	AT91_REG	 PMC_ADDRSIZE; 	// 
	AT91_REG	 PMC_NAME1; 	// 
	AT91_REG	 PMC_NAME2; 	// 
	AT91_REG	 PMC_FEATURES; 	// 
	AT91_REG	 PMC_VERSION; 	// 
} AT91S_PMC, *AT91PS_PMC;

// -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- 
#define AT91C_PMC_DDR         ((unsigned int) 0x1 <<  2) // (PMC) DDR controller Clock2x
#define AT91C_PMC_UHP         ((unsigned int) 0x1 <<  6) // (PMC) USB Host Port Clock
#define AT91C_PMC_UDP         ((unsigned int) 0x1 <<  7) // (PMC) USB Device Port Clock
#define AT91C_PMC_PCK0        ((unsigned int) 0x1 <<  8) // (PMC) Programmable Clock Output
#define AT91C_PMC_PCK1        ((unsigned int) 0x1 <<  9) // (PMC) Programmable Clock Output
// -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- 
#define AT91C_PMC_PCK         ((unsigned int) 0x1 <<  0) // (PMC) Processor Clock
// -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- 
// -------- CKGR_UCKR : (PMC Offset: 0x1c) UTMI Clock Configuration Register -------- 
#define AT91C_CKGR_UPLLEN     ((unsigned int) 0x1 << 16) // (PMC) UTMI PLL Enable
#define 	AT91C_CKGR_UPLLEN_DISABLED             ((unsigned int) 0x0 << 16) // (PMC) The UTMI PLL is disabled
#define 	AT91C_CKGR_UPLLEN_ENABLED              ((unsigned int) 0x1 << 16) // (PMC) The UTMI PLL is enabled
#define AT91C_CKGR_PLLCOUNT   ((unsigned int) 0xF << 20) // (PMC) UTMI Oscillator Start-up Time
#define AT91C_CKGR_BIASEN     ((unsigned int) 0x1 << 24) // (PMC) UTMI BIAS Enable
#define 	AT91C_CKGR_BIASEN_DISABLED             ((unsigned int) 0x0 << 24) // (PMC) The UTMI BIAS is disabled
#define 	AT91C_CKGR_BIASEN_ENABLED              ((unsigned int) 0x1 << 24) // (PMC) The UTMI BIAS is enabled
#define AT91C_CKGR_BIASCOUNT  ((unsigned int) 0xF << 28) // (PMC) UTMI BIAS Start-up Time
// -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- 
#define AT91C_CKGR_MOSCEN     ((unsigned int) 0x1 <<  0) // (PMC) Main Oscillator Enable
#define AT91C_CKGR_OSCBYPASS  ((unsigned int) 0x1 <<  1) // (PMC) Main Oscillator Bypass
#define AT91C_CKGR_OSCOUNT    ((unsigned int) 0xFF <<  8) // (PMC) Main Oscillator Start-up Time
// -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- 
#define AT91C_CKGR_MAINF      ((unsigned int) 0xFFFF <<  0) // (PMC) Main Clock Frequency
#define AT91C_CKGR_MAINRDY    ((unsigned int) 0x1 << 16) // (PMC) Main Clock Ready
// -------- CKGR_PLLAR : (PMC Offset: 0x28) PLL A Register -------- 
#define AT91C_CKGR_DIVA       ((unsigned int) 0xFF <<  0) // (PMC) Divider A Selected
#define 	AT91C_CKGR_DIVA_0                    ((unsigned int) 0x0) // (PMC) Divider A output is 0
#define 	AT91C_CKGR_DIVA_BYPASS               ((unsigned int) 0x1) // (PMC) Divider A is bypassed
#define AT91C_CKGR_PLLACOUNT  ((unsigned int) 0x3F <<  8) // (PMC) PLL A Counter
#define AT91C_CKGR_OUTA       ((unsigned int) 0x3 << 14) // (PMC) PLL A Output Frequency Range
#define 	AT91C_CKGR_OUTA_0                    ((unsigned int) 0x0 << 14) // (PMC) Please refer to the PLLA datasheet
#define 	AT91C_CKGR_OUTA_1                    ((unsigned int) 0x1 << 14) // (PMC) Please refer to the PLLA datasheet
#define 	AT91C_CKGR_OUTA_2                    ((unsigned int) 0x2 << 14) // (PMC) Please refer to the PLLA datasheet
#define 	AT91C_CKGR_OUTA_3                    ((unsigned int) 0x3 << 14) // (PMC) Please refer to the PLLA datasheet
#define AT91C_CKGR_MULA       ((unsigned int) 0xFF << 16) // (PMC) PLL A Multiplier
#define AT91C_CKGR_SRCA       ((unsigned int) 0x1 << 29) // (PMC) 
// -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- 
#define AT91C_PMC_CSS         ((unsigned int) 0x3 <<  0) // (PMC) Programmable Clock Selection
#define 	AT91C_PMC_CSS_SLOW_CLK             ((unsigned int) 0x0) // (PMC) Slow Clock is selected
#define 	AT91C_PMC_CSS_MAIN_CLK             ((unsigned int) 0x1) // (PMC) Main Clock is selected
#define 	AT91C_PMC_CSS_PLLA_CLK             ((unsigned int) 0x2) // (PMC) Clock from PLL A is selected
#define 	AT91C_PMC_CSS_UPLL_CLK             ((unsigned int) 0x3) // (PMC) Clock from UTMI PLL is selected
#define AT91C_PMC_PRES        ((unsigned int) 0x7 <<  2) // (PMC) Programmable Clock Prescaler
#define 	AT91C_PMC_PRES_CLK                  ((unsigned int) 0x0 <<  2) // (PMC) Selected clock
#define 	AT91C_PMC_PRES_CLK_2                ((unsigned int) 0x1 <<  2) // (PMC) Selected clock divided by 2
#define 	AT91C_PMC_PRES_CLK_4                ((unsigned int) 0x2 <<  2) // (PMC) Selected clock divided by 4
#define 	AT91C_PMC_PRES_CLK_8                ((unsigned int) 0x3 <<  2) // (PMC) Selected clock divided by 8
#define 	AT91C_PMC_PRES_CLK_16               ((unsigned int) 0x4 <<  2) // (PMC) Selected clock divided by 16
#define 	AT91C_PMC_PRES_CLK_32               ((unsigned int) 0x5 <<  2) // (PMC) Selected clock divided by 32
#define 	AT91C_PMC_PRES_CLK_64               ((unsigned int) 0x6 <<  2) // (PMC) Selected clock divided by 64
#define AT91C_PMC_MDIV        ((unsigned int) 0x3 <<  8) // (PMC) Master Clock Division
#define 	AT91C_PMC_MDIV_1                    ((unsigned int) 0x0 <<  8) // (PMC) Processor clock = Master Clock ; DDR Clock = Master Clock
#define 	AT91C_PMC_MDIV_2                    ((unsigned int) 0x1 <<  8) // (PMC) Processor clock = 2 * Master Clock ; DDR Clock = 2 * Master Clock
#define 	AT91C_PMC_MDIV_4                    ((unsigned int) 0x2 <<  8) // (PMC) Processor clock = 4 * Master Clock ; DDR Clock = 2 * Master Clock
#define 	AT91C_PMC_MDIV_3                    ((unsigned int) 0x3 <<  8) // (PMC) Processor clock = 3 * Master Clock ; DDR Clock = 2 * Master Clock
#define AT91C_PMC_PLLADIV2    ((unsigned int) 0x1 << 12) // (PMC) PLLA divisor by 2
#define 	AT91C_PMC_PLLADIV2_1                    ((unsigned int) 0x0 << 12) // (PMC) PLLA clock frequency is divided by 1
#define 	AT91C_PMC_PLLADIV2_2                    ((unsigned int) 0x1 << 12) // (PMC) PLLA clock frequency is divided by 2
// -------- PMC_USB : (PMC Offset: 0x38) USB Clock Register -------- 
#define AT91C_PMC_USBS        ((unsigned int) 0x1 <<  0) // (PMC) USBS
#define 	AT91C_PMC_USBS_USB_PLLA             ((unsigned int) 0x0) // (PMC) USB Clock Input is PLLA
#define 	AT91C_PMC_USBS_USB_UPLL             ((unsigned int) 0x1) // (PMC) USB Clock Input is UPLL
#define AT91C_PMC_USBDIV      ((unsigned int) 0xF <<  8) // (PMC) USBDIV
#define 	AT91C_PMC_USBDIV_1                    ((unsigned int) 0x0 <<  8) // (PMC) USB Clock divided by 1
#define 	AT91C_PMC_USBDIV_2                    ((unsigned int) 0x1 <<  8) // (PMC) USB Clock divided by 2
#define 	AT91C_PMC_USBDIV_3                    ((unsigned int) 0x2 <<  8) // (PMC) USB Clock divided by 3
#define 	AT91C_PMC_USBDIV_4                    ((unsigned int) 0x3 <<  8) // (PMC) USB Clock divided by 4
#define 	AT91C_PMC_USBDIV_5                    ((unsigned int) 0x4 <<  8) // (PMC) USB Clock divided by 5
#define 	AT91C_PMC_USBDIV_6                    ((unsigned int) 0x5 <<  8) // (PMC) USB Clock divided by 6
#define 	AT91C_PMC_USBDIV_7                    ((unsigned int) 0x6 <<  8) // (PMC) USB Clock divided by 7
#define 	AT91C_PMC_USBDIV_8                    ((unsigned int) 0x7 <<  8) // (PMC) USB Clock divided by 8
#define 	AT91C_PMC_USBDIV_9                    ((unsigned int) 0x8 <<  8) // (PMC) USB Clock divided by 9
#define 	AT91C_PMC_USBDIV_10                   ((unsigned int) 0x9 <<  8) // (PMC) USB Clock divided by 10
#define 	AT91C_PMC_USBDIV_11                   ((unsigned int) 0xA <<  8) // (PMC) USB Clock divided by 11
#define 	AT91C_PMC_USBDIV_12                   ((unsigned int) 0xB <<  8) // (PMC) USB Clock divided by 12
#define 	AT91C_PMC_USBDIV_13                   ((unsigned int) 0xC <<  8) // (PMC) USB Clock divided by 13
#define 	AT91C_PMC_USBDIV_14                   ((unsigned int) 0xD <<  8) // (PMC) USB Clock divided by 14
#define 	AT91C_PMC_USBDIV_15                   ((unsigned int) 0xE <<  8) // (PMC) USB Clock divided by 15
#define 	AT91C_PMC_USBDIV_16                   ((unsigned int) 0xF <<  8) // (PMC) USB Clock divided by 16
// -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock 0 Register -------- 
#define AT91C_PMC_SLCKMCK     ((unsigned int) 0x1 <<  8) // (PMC) Programmable Clock Prescaler
#define 	AT91C_PMC_SLCKMCK_SLCK                 ((unsigned int) 0x0 <<  8) // (PMC) Slow Clock selected
#define 	AT91C_PMC_SLCKMCK_MCK                  ((unsigned int) 0x1 <<  8) // (PMC) Master Clock selected
// -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- 
#define AT91C_PMC_MOSCS       ((unsigned int) 0x1 <<  0) // (PMC) MOSC Status/Enable/Disable/Mask
#define AT91C_PMC_LOCKA       ((unsigned int) 0x1 <<  1) // (PMC) PLL A Status/Enable/Disable/Mask
#define AT91C_PMC_MCKRDY      ((unsigned int) 0x1 <<  3) // (PMC) Master Clock Status/Enable/Disable/Mask
#define AT91C_PMC_LOCKU       ((unsigned int) 0x1 <<  6) // (PMC) PLL UTMI Status/Enable/Disable/Mask
#define AT91C_PMC_PCK0RDY     ((unsigned int) 0x1 <<  8) // (PMC) PCK0_RDY Status/Enable/Disable/Mask
#define AT91C_PMC_PCK1RDY     ((unsigned int) 0x1 <<  9) // (PMC) PCK1_RDY Status/Enable/Disable/Mask
// -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- 
// -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- 
// -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- 
// -------- PMC_PLLICPR : (PMC Offset: 0x80) PLL Charge Pump Current Register -------- 
#define AT91C_PMC_ICPPLLA     ((unsigned int) 0xF <<  0) // (PMC) PLLA charge pump current setting
#define 	AT91C_PMC_ICPPLLA_0                    ((unsigned int) 0x0) // (PMC) 595-800 MHz
#define 	AT91C_PMC_ICPPLLA_1                    ((unsigned int) 0x1) // (PMC) 395-600 MHz
#define AT91C_PMC_REALLOCK    ((unsigned int) 0x1 <<  7) // (PMC) PLLs use real lock signals when 1
#define AT91C_PMC_IPLLA       ((unsigned int) 0xF <<  8) // (PMC) PLLA special setting
#define 	AT91C_PMC_IPLLA_0                    ((unsigned int) 0x0 <<  8) // (PMC) Internal LFT
#define 	AT91C_PMC_IPLLA_1                    ((unsigned int) 0x1 <<  8) // (PMC) External LFT
// -------- PMC_FEATURES : (PMC Offset: 0xf8)   -------- 
#define AT91C_PMC_CFGAHBCLK   ((unsigned int) 0x1 <<  0) // (PMC) 
#define 	AT91C_PMC_CFGAHBCLK_0                    ((unsigned int) 0x0) // (PMC) 
#define 	AT91C_PMC_CFGAHBCLK_1                    ((unsigned int) 0x1) // (PMC) 
#define AT91C_PMC_HCLKEN      ((unsigned int) 0x1 <<  1) // (PMC) 
#define 	AT91C_PMC_HCLKEN_0                    ((unsigned int) 0x0 <<  1) // (PMC) 
#define 	AT91C_PMC_HCLKEN_1                    ((unsigned int) 0x1 <<  1) // (PMC) 
#define AT91C_PMC_PERMCLK     ((unsigned int) 0x1 <<  2) // (PMC) 
#define 	AT91C_PMC_PERMCLK_0                    ((unsigned int) 0x0 <<  2) // (PMC) 
#define 	AT91C_PMC_PERMCLK_1                    ((unsigned int) 0x1 <<  2) // (PMC) 
#define AT91C_PMC_CORE2       ((unsigned int) 0x1 <<  3) // (PMC) 
#define 	AT91C_PMC_CORE2_0                    ((unsigned int) 0x0 <<  3) // (PMC) 
#define 	AT91C_PMC_CORE2_1                    ((unsigned int) 0x1 <<  3) // (PMC) 
#define AT91C_PMC_USBDEVCK    ((unsigned int) 0x1 <<  4) // (PMC) 
#define 	AT91C_PMC_USBDEVCK_0                    ((unsigned int) 0x0 <<  4) // (PMC) 
#define 	AT91C_PMC_USBDEVCK_1                    ((unsigned int) 0x1 <<  4) // (PMC) 
#define AT91C_PMC_USBHOSTCK   ((unsigned int) 0x1 <<  5) // (PMC) 
#define 	AT91C_PMC_USBHOSTCK_0                    ((unsigned int) 0x0 <<  5) // (PMC) 
#define 	AT91C_PMC_USBHOSTCK_1                    ((unsigned int) 0x1 <<  5) // (PMC) 
#define AT91C_PMC_USBOTGCK    ((unsigned int) 0x1 <<  6) // (PMC) 
#define 	AT91C_PMC_USBOTGCK_0                    ((unsigned int) 0x0 <<  6) // (PMC) 
#define 	AT91C_PMC_USBOTGCK_1                    ((unsigned int) 0x1 <<  6) // (PMC) 
#define AT91C_PMC_UHSYNRST    ((unsigned int) 0x1 <<  7) // (PMC) 
#define 	AT91C_PMC_UHSYNRST_0                    ((unsigned int) 0x0 <<  7) // (PMC) 
#define 	AT91C_PMC_UHSYNRST_1                    ((unsigned int) 0x1 <<  7) // (PMC) 
#define AT91C_PMC_UOSYNRST    ((unsigned int) 0x1 <<  8) // (PMC) 
#define 	AT91C_PMC_UOSYNRST_0                    ((unsigned int) 0x0 <<  8) // (PMC) 
#define 	AT91C_PMC_UOSYNRST_1                    ((unsigned int) 0x1 <<  8) // (PMC) 
#define AT91C_PMC_PLLENPOL    ((unsigned int) 0x1 <<  9) // (PMC) 
#define 	AT91C_PMC_PLLENPOL_0                    ((unsigned int) 0x0 <<  9) // (PMC) 
#define 	AT91C_PMC_PLLENPOL_1                    ((unsigned int) 0x1 <<  9) // (PMC) 
#define AT91C_PMC_BIASREG     ((unsigned int) 0x1 << 10) // (PMC) 
#define 	AT91C_PMC_BIASREG_0                    ((unsigned int) 0x0 << 10) // (PMC) 
#define 	AT91C_PMC_BIASREG_1                    ((unsigned int) 0x1 << 10) // (PMC) 
#define AT91C_PMC_OUTPLL      ((unsigned int) 0x1 << 11) // (PMC) 
#define 	AT91C_PMC_OUTPLL_0                    ((unsigned int) 0x0 << 11) // (PMC) 
#define 	AT91C_PMC_OUTPLL_1                    ((unsigned int) 0x1 << 11) // (PMC) 
#define AT91C_PMC_OUTCURR     ((unsigned int) 0x1 << 12) // (PMC) 
#define 	AT91C_PMC_OUTCURR_0                    ((unsigned int) 0x0 << 12) // (PMC) 
#define 	AT91C_PMC_OUTCURR_1                    ((unsigned int) 0x1 << 12) // (PMC) 
#define AT91C_PMC_FWUP        ((unsigned int) 0x1 << 13) // (PMC) 
#define 	AT91C_PMC_FWUP_0                    ((unsigned int) 0x0 << 13) // (PMC) 
#define 	AT91C_PMC_FWUP_1                    ((unsigned int) 0x1 << 13) // (PMC) 
#define AT91C_PMC_SELMAINCLK  ((unsigned int) 0x1 << 14) // (PMC) 
#define 	AT91C_PMC_SELMAINCLK_0                    ((unsigned int) 0x0 << 14) // (PMC) 
#define 	AT91C_PMC_SELMAINCLK_1                    ((unsigned int) 0x1 << 14) // (PMC) 
#define AT91C_PMC_RSTCLKM     ((unsigned int) 0x1 << 15) // (PMC) 
#define 	AT91C_PMC_RSTCLKM_0                    ((unsigned int) 0x0 << 15) // (PMC) 
#define 	AT91C_PMC_RSTCLKM_1                    ((unsigned int) 0x1 << 15) // (PMC) 
#define AT91C_PMC_NB_PERIPH_CLOCK ((unsigned int) 0xFF << 16) // (PMC) 
// -------- PMC_VERSION : (PMC Offset: 0xfc)   -------- 
#define AT91C_PMC_Version     ((unsigned int) 0xFFFF <<  0) // (PMC) 
#define 	AT91C_PMC_Version_0                    ((unsigned int) 0x0) // (PMC) 
#define 	AT91C_PMC_Version_1                    ((unsigned int) 0x1) // (PMC) 
#define AT91C_PMC_MFN         ((unsigned int) 0x7 << 16) // (PMC) 
#define 	AT91C_PMC_MFN_0                    ((unsigned int) 0x0 << 16) // (PMC) 
#define 	AT91C_PMC_MFN_1                    ((unsigned int) 0x1 << 16) // (PMC) 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Clock Generator Controler
// *****************************************************************************
typedef struct _AT91S_CKGR {
	AT91_REG	 CKGR_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 CKGR_MOR; 	// Main Oscillator Register
	AT91_REG	 CKGR_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 CKGR_PLLAR; 	// PLL A Register
} AT91S_CKGR, *AT91PS_CKGR;

// -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- 
// -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- 
// -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- 
// -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Reset Controller Interface
// *****************************************************************************
typedef struct _AT91S_RSTC {
	AT91_REG	 RSTC_RCR; 	// Reset Control Register
	AT91_REG	 RSTC_RSR; 	// Reset Status Register
	AT91_REG	 RSTC_RMR; 	// Reset Mode Register
	AT91_REG	 Reserved0[60]; 	// 
	AT91_REG	 RSTC_VER; 	// Version Register
} AT91S_RSTC, *AT91PS_RSTC;

// -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- 
#define AT91C_RSTC_PROCRST    ((unsigned int) 0x1 <<  0) // (RSTC) Processor Reset
#define AT91C_RSTC_ICERST     ((unsigned int) 0x1 <<  1) // (RSTC) ICE Interface Reset
#define AT91C_RSTC_PERRST     ((unsigned int) 0x1 <<  2) // (RSTC) Peripheral Reset
#define AT91C_RSTC_EXTRST     ((unsigned int) 0x1 <<  3) // (RSTC) External Reset
#define AT91C_RSTC_KEY        ((unsigned int) 0xFF << 24) // (RSTC) Password
// -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- 
#define AT91C_RSTC_URSTS      ((unsigned int) 0x1 <<  0) // (RSTC) User Reset Status
#define AT91C_RSTC_RSTTYP     ((unsigned int) 0x7 <<  8) // (RSTC) Reset Type
#define 	AT91C_RSTC_RSTTYP_GENERAL              ((unsigned int) 0x0 <<  8) // (RSTC) General reset. Both VDDCORE and VDDBU rising.
#define 	AT91C_RSTC_RSTTYP_WAKEUP               ((unsigned int) 0x1 <<  8) // (RSTC) WakeUp Reset. VDDCORE rising.
#define 	AT91C_RSTC_RSTTYP_WATCHDOG             ((unsigned int) 0x2 <<  8) // (RSTC) Watchdog Reset. Watchdog overflow occured.
#define 	AT91C_RSTC_RSTTYP_SOFTWARE             ((unsigned int) 0x3 <<  8) // (RSTC) Software Reset. Processor reset required by the software.
#define 	AT91C_RSTC_RSTTYP_USER                 ((unsigned int) 0x4 <<  8) // (RSTC) User Reset. NRST pin detected low.
#define AT91C_RSTC_NRSTL      ((unsigned int) 0x1 << 16) // (RSTC) NRST pin level
#define AT91C_RSTC_SRCMP      ((unsigned int) 0x1 << 17) // (RSTC) Software Reset Command in Progress.
// -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- 
#define AT91C_RSTC_URSTEN     ((unsigned int) 0x1 <<  0) // (RSTC) User Reset Enable
#define AT91C_RSTC_URSTIEN    ((unsigned int) 0x1 <<  4) // (RSTC) User Reset Interrupt Enable
#define AT91C_RSTC_ERSTL      ((unsigned int) 0xF <<  8) // (RSTC) User Reset Enable

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Shut Down Controller Interface
// *****************************************************************************
typedef struct _AT91S_SHDWC {
	AT91_REG	 SHDWC_SHCR; 	// Shut Down Control Register
	AT91_REG	 SHDWC_SHMR; 	// Shut Down Mode Register
	AT91_REG	 SHDWC_SHSR; 	// Shut Down Status Register
} AT91S_SHDWC, *AT91PS_SHDWC;

// -------- SHDWC_SHCR : (SHDWC Offset: 0x0) Shut Down Control Register -------- 
#define AT91C_SHDWC_SHDW      ((unsigned int) 0x1 <<  0) // (SHDWC) Processor Reset
#define AT91C_SHDWC_KEY       ((unsigned int) 0xFF << 24) // (SHDWC) Shut down KEY Password
// -------- SHDWC_SHMR : (SHDWC Offset: 0x4) Shut Down Mode Register -------- 
#define AT91C_SHDWC_WKMODE0   ((unsigned int) 0x3 <<  0) // (SHDWC) Wake Up 0 Mode Selection
#define 	AT91C_SHDWC_WKMODE0_NONE                 ((unsigned int) 0x0) // (SHDWC) None. No detection is performed on the wake up input.
#define 	AT91C_SHDWC_WKMODE0_HIGH                 ((unsigned int) 0x1) // (SHDWC) High Level.
#define 	AT91C_SHDWC_WKMODE0_LOW                  ((unsigned int) 0x2) // (SHDWC) Low Level.
#define 	AT91C_SHDWC_WKMODE0_ANYLEVEL             ((unsigned int) 0x3) // (SHDWC) Any level change.
#define AT91C_SHDWC_CPTWK0    ((unsigned int) 0xF <<  4) // (SHDWC) Counter On Wake Up 0
#define AT91C_SHDWC_WKMODE1   ((unsigned int) 0x3 <<  8) // (SHDWC) Wake Up 1 Mode Selection
#define 	AT91C_SHDWC_WKMODE1_NONE                 ((unsigned int) 0x0 <<  8) // (SHDWC) None. No detection is performed on the wake up input.
#define 	AT91C_SHDWC_WKMODE1_HIGH                 ((unsigned int) 0x1 <<  8) // (SHDWC) High Level.
#define 	AT91C_SHDWC_WKMODE1_LOW                  ((unsigned int) 0x2 <<  8) // (SHDWC) Low Level.
#define 	AT91C_SHDWC_WKMODE1_ANYLEVEL             ((unsigned int) 0x3 <<  8) // (SHDWC) Any level change.
#define AT91C_SHDWC_CPTWK1    ((unsigned int) 0xF << 12) // (SHDWC) Counter On Wake Up 1
#define AT91C_SHDWC_RTTWKEN   ((unsigned int) 0x1 << 16) // (SHDWC) Real Time Timer Wake Up Enable
#define AT91C_SHDWC_RTCWKEN   ((unsigned int) 0x1 << 17) // (SHDWC) Real Time Clock Wake Up Enable
// -------- SHDWC_SHSR : (SHDWC Offset: 0x8) Shut Down Status Register -------- 
#define AT91C_SHDWC_WAKEUP0   ((unsigned int) 0x1 <<  0) // (SHDWC) Wake Up 0 Status
#define AT91C_SHDWC_WAKEUP1   ((unsigned int) 0x1 <<  1) // (SHDWC) Wake Up 1 Status
#define AT91C_SHDWC_FWKUP     ((unsigned int) 0x1 <<  2) // (SHDWC) Force Wake Up Status
#define AT91C_SHDWC_RTTWK     ((unsigned int) 0x1 << 16) // (SHDWC) Real Time Timer wake Up
#define AT91C_SHDWC_RTCWK     ((unsigned int) 0x1 << 17) // (SHDWC) Real Time Clock wake Up

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Real Time Timer Controller Interface
// *****************************************************************************
typedef struct _AT91S_RTTC {
	AT91_REG	 RTTC_RTMR; 	// Real-time Mode Register
	AT91_REG	 RTTC_RTAR; 	// Real-time Alarm Register
	AT91_REG	 RTTC_RTVR; 	// Real-time Value Register
	AT91_REG	 RTTC_RTSR; 	// Real-time Status Register
} AT91S_RTTC, *AT91PS_RTTC;

// -------- RTTC_RTMR : (RTTC Offset: 0x0) Real-time Mode Register -------- 
#define AT91C_RTTC_RTPRES     ((unsigned int) 0xFFFF <<  0) // (RTTC) Real-time Timer Prescaler Value
#define AT91C_RTTC_ALMIEN     ((unsigned int) 0x1 << 16) // (RTTC) Alarm Interrupt Enable
#define AT91C_RTTC_RTTINCIEN  ((unsigned int) 0x1 << 17) // (RTTC) Real Time Timer Increment Interrupt Enable
#define AT91C_RTTC_RTTRST     ((unsigned int) 0x1 << 18) // (RTTC) Real Time Timer Restart
// -------- RTTC_RTAR : (RTTC Offset: 0x4) Real-time Alarm Register -------- 
#define AT91C_RTTC_ALMV       ((unsigned int) 0x0 <<  0) // (RTTC) Alarm Value
// -------- RTTC_RTVR : (RTTC Offset: 0x8) Current Real-time Value Register -------- 
#define AT91C_RTTC_CRTV       ((unsigned int) 0x0 <<  0) // (RTTC) Current Real-time Value
// -------- RTTC_RTSR : (RTTC Offset: 0xc) Real-time Status Register -------- 
#define AT91C_RTTC_ALMS       ((unsigned int) 0x1 <<  0) // (RTTC) Real-time Alarm Status
#define AT91C_RTTC_RTTINC     ((unsigned int) 0x1 <<  1) // (RTTC) Real-time Timer Increment

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Periodic Interval Timer Controller Interface
// *****************************************************************************
typedef struct _AT91S_PITC {
	AT91_REG	 PITC_PIMR; 	// Period Interval Mode Register
	AT91_REG	 PITC_PISR; 	// Period Interval Status Register
	AT91_REG	 PITC_PIVR; 	// Period Interval Value Register
	AT91_REG	 PITC_PIIR; 	// Period Interval Image Register
} AT91S_PITC, *AT91PS_PITC;

// -------- PITC_PIMR : (PITC Offset: 0x0) Periodic Interval Mode Register -------- 
#define AT91C_PITC_PIV        ((unsigned int) 0xFFFFF <<  0) // (PITC) Periodic Interval Value
#define AT91C_PITC_PITEN      ((unsigned int) 0x1 << 24) // (PITC) Periodic Interval Timer Enabled
#define AT91C_PITC_PITIEN     ((unsigned int) 0x1 << 25) // (PITC) Periodic Interval Timer Interrupt Enable
// -------- PITC_PISR : (PITC Offset: 0x4) Periodic Interval Status Register -------- 
#define AT91C_PITC_PITS       ((unsigned int) 0x1 <<  0) // (PITC) Periodic Interval Timer Status
// -------- PITC_PIVR : (PITC Offset: 0x8) Periodic Interval Value Register -------- 
#define AT91C_PITC_CPIV       ((unsigned int) 0xFFFFF <<  0) // (PITC) Current Periodic Interval Value
#define AT91C_PITC_PICNT      ((unsigned int) 0xFFF << 20) // (PITC) Periodic Interval Counter
// -------- PITC_PIIR : (PITC Offset: 0xc) Periodic Interval Image Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Watchdog Timer Controller Interface
// *****************************************************************************
typedef struct _AT91S_WDTC {
	AT91_REG	 WDTC_WDCR; 	// Watchdog Control Register
	AT91_REG	 WDTC_WDMR; 	// Watchdog Mode Register
	AT91_REG	 WDTC_WDSR; 	// Watchdog Status Register
} AT91S_WDTC, *AT91PS_WDTC;

// -------- WDTC_WDCR : (WDTC Offset: 0x0) Periodic Interval Image Register -------- 
#define AT91C_WDTC_WDRSTT     ((unsigned int) 0x1 <<  0) // (WDTC) Watchdog Restart
#define AT91C_WDTC_KEY        ((unsigned int) 0xFF << 24) // (WDTC) Watchdog KEY Password
// -------- WDTC_WDMR : (WDTC Offset: 0x4) Watchdog Mode Register -------- 
#define AT91C_WDTC_WDV        ((unsigned int) 0xFFF <<  0) // (WDTC) Watchdog Timer Restart
#define AT91C_WDTC_WDFIEN     ((unsigned int) 0x1 << 12) // (WDTC) Watchdog Fault Interrupt Enable
#define AT91C_WDTC_WDRSTEN    ((unsigned int) 0x1 << 13) // (WDTC) Watchdog Reset Enable
#define AT91C_WDTC_WDRPROC    ((unsigned int) 0x1 << 14) // (WDTC) Watchdog Timer Restart
#define AT91C_WDTC_WDDIS      ((unsigned int) 0x1 << 15) // (WDTC) Watchdog Disable
#define AT91C_WDTC_WDD        ((unsigned int) 0xFFF << 16) // (WDTC) Watchdog Delta Value
#define AT91C_WDTC_WDDBGHLT   ((unsigned int) 0x1 << 28) // (WDTC) Watchdog Debug Halt
#define AT91C_WDTC_WDIDLEHLT  ((unsigned int) 0x1 << 29) // (WDTC) Watchdog Idle Halt
// -------- WDTC_WDSR : (WDTC Offset: 0x8) Watchdog Status Register -------- 
#define AT91C_WDTC_WDUNF      ((unsigned int) 0x1 <<  0) // (WDTC) Watchdog Underflow
#define AT91C_WDTC_WDERR      ((unsigned int) 0x1 <<  1) // (WDTC) Watchdog Error

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm and Parallel Load Interface
// *****************************************************************************
typedef struct _AT91S_RTC {
	AT91_REG	 RTC_CR; 	// Control Register
	AT91_REG	 RTC_MR; 	// Mode Register
	AT91_REG	 RTC_TIMR; 	// Time Register
	AT91_REG	 RTC_CALR; 	// Calendar Register
	AT91_REG	 RTC_TIMALR; 	// Time Alarm Register
	AT91_REG	 RTC_CALALR; 	// Calendar Alarm Register
	AT91_REG	 RTC_SR; 	// Status Register
	AT91_REG	 RTC_SCCR; 	// Status Clear Command Register
	AT91_REG	 RTC_IER; 	// Interrupt Enable Register
	AT91_REG	 RTC_IDR; 	// Interrupt Disable Register
	AT91_REG	 RTC_IMR; 	// Interrupt Mask Register
	AT91_REG	 RTC_VER; 	// Valid Entry Register
} AT91S_RTC, *AT91PS_RTC;

// -------- RTC_CR : (RTC Offset: 0x0) RTC Control Register -------- 
#define AT91C_RTC_UPDTIM      ((unsigned int) 0x1 <<  0) // (RTC) Update Request Time Register
#define AT91C_RTC_UPDCAL      ((unsigned int) 0x1 <<  1) // (RTC) Update Request Calendar Register
#define AT91C_RTC_TIMEVSEL    ((unsigned int) 0x3 <<  8) // (RTC) Time Event Selection
#define 	AT91C_RTC_TIMEVSEL_MINUTE               ((unsigned int) 0x0 <<  8) // (RTC) Minute change.
#define 	AT91C_RTC_TIMEVSEL_HOUR                 ((unsigned int) 0x1 <<  8) // (RTC) Hour change.
#define 	AT91C_RTC_TIMEVSEL_DAY24                ((unsigned int) 0x2 <<  8) // (RTC) Every day at midnight.
#define 	AT91C_RTC_TIMEVSEL_DAY12                ((unsigned int) 0x3 <<  8) // (RTC) Every day at noon.
#define AT91C_RTC_CALEVSEL    ((unsigned int) 0x3 << 16) // (RTC) Calendar Event Selection
#define 	AT91C_RTC_CALEVSEL_WEEK                 ((unsigned int) 0x0 << 16) // (RTC) Week change (every Monday at time 00:00:00).
#define 	AT91C_RTC_CALEVSEL_MONTH                ((unsigned int) 0x1 << 16) // (RTC) Month change (every 01 of each month at time 00:00:00).
#define 	AT91C_RTC_CALEVSEL_YEAR                 ((unsigned int) 0x2 << 16) // (RTC) Year change (every January 1 at time 00:00:00).
// -------- RTC_MR : (RTC Offset: 0x4) RTC Mode Register -------- 
#define AT91C_RTC_HRMOD       ((unsigned int) 0x1 <<  0) // (RTC) 12-24 hour Mode
// -------- RTC_TIMR : (RTC Offset: 0x8) RTC Time Register -------- 
#define AT91C_RTC_SEC         ((unsigned int) 0x7F <<  0) // (RTC) Current Second
#define AT91C_RTC_MIN         ((unsigned int) 0x7F <<  8) // (RTC) Current Minute
#define AT91C_RTC_HOUR        ((unsigned int) 0x3F << 16) // (RTC) Current Hour
#define AT91C_RTC_AMPM        ((unsigned int) 0x1 << 22) // (RTC) Ante Meridiem, Post Meridiem Indicator
// -------- RTC_CALR : (RTC Offset: 0xc) RTC Calendar Register -------- 
#define AT91C_RTC_CENT        ((unsigned int) 0x3F <<  0) // (RTC) Current Century
#define AT91C_RTC_YEAR        ((unsigned int) 0xFF <<  8) // (RTC) Current Year
#define AT91C_RTC_MONTH       ((unsigned int) 0x1F << 16) // (RTC) Current Month
#define AT91C_RTC_DAY         ((unsigned int) 0x7 << 21) // (RTC) Current Day
#define AT91C_RTC_DATE        ((unsigned int) 0x3F << 24) // (RTC) Current Date
// -------- RTC_TIMALR : (RTC Offset: 0x10) RTC Time Alarm Register -------- 
#define AT91C_RTC_SECEN       ((unsigned int) 0x1 <<  7) // (RTC) Second Alarm Enable
#define AT91C_RTC_MINEN       ((unsigned int) 0x1 << 15) // (RTC) Minute Alarm
#define AT91C_RTC_HOUREN      ((unsigned int) 0x1 << 23) // (RTC) Current Hour
// -------- RTC_CALALR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- 
#define AT91C_RTC_MONTHEN     ((unsigned int) 0x1 << 23) // (RTC) Month Alarm Enable
#define AT91C_RTC_DATEEN      ((unsigned int) 0x1 << 31) // (RTC) Date Alarm Enable
// -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- 
#define AT91C_RTC_ACKUPD      ((unsigned int) 0x1 <<  0) // (RTC) Acknowledge for Update
#define AT91C_RTC_ALARM       ((unsigned int) 0x1 <<  1) // (RTC) Alarm Flag
#define AT91C_RTC_SECEV       ((unsigned int) 0x1 <<  2) // (RTC) Second Event
#define AT91C_RTC_TIMEV       ((unsigned int) 0x1 <<  3) // (RTC) Time Event
#define AT91C_RTC_CALEV       ((unsigned int) 0x1 <<  4) // (RTC) Calendar event
// -------- RTC_SCCR : (RTC Offset: 0x1c) RTC Status Clear Command Register -------- 
// -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- 
// -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- 
// -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- 
// -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- 
#define AT91C_RTC_NVTIM       ((unsigned int) 0x1 <<  0) // (RTC) Non valid Time
#define AT91C_RTC_NVCAL       ((unsigned int) 0x1 <<  1) // (RTC) Non valid Calendar
#define AT91C_RTC_NVTIMALR    ((unsigned int) 0x1 <<  2) // (RTC) Non valid time Alarm
#define AT91C_RTC_NVCALALR    ((unsigned int) 0x1 <<  3) // (RTC) Nonvalid Calendar Alarm

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Timer Counter Channel Interface
// *****************************************************************************
typedef struct _AT91S_TC {
	AT91_REG	 TC_CCR; 	// Channel Control Register
	AT91_REG	 TC_CMR; 	// Channel Mode Register (Capture Mode / Waveform Mode)
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 TC_CV; 	// Counter Value
	AT91_REG	 TC_RA; 	// Register A
	AT91_REG	 TC_RB; 	// Register B
	AT91_REG	 TC_RC; 	// Register C
	AT91_REG	 TC_SR; 	// Status Register
	AT91_REG	 TC_IER; 	// Interrupt Enable Register
	AT91_REG	 TC_IDR; 	// Interrupt Disable Register
	AT91_REG	 TC_IMR; 	// Interrupt Mask Register
} AT91S_TC, *AT91PS_TC;

// -------- TC_CCR : (TC Offset: 0x0) TC Channel Control Register -------- 
#define AT91C_TC_CLKEN        ((unsigned int) 0x1 <<  0) // (TC) Counter Clock Enable Command
#define AT91C_TC_CLKDIS       ((unsigned int) 0x1 <<  1) // (TC) Counter Clock Disable Command
#define AT91C_TC_SWTRG        ((unsigned int) 0x1 <<  2) // (TC) Software Trigger Command
// -------- TC_CMR : (TC Offset: 0x4) TC Channel Mode Register: Capture Mode / Waveform Mode -------- 
#define AT91C_TC_CLKS         ((unsigned int) 0x7 <<  0) // (TC) Clock Selection
#define 	AT91C_TC_CLKS_TIMER_DIV1_CLOCK     ((unsigned int) 0x0) // (TC) Clock selected: TIMER_DIV1_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV2_CLOCK     ((unsigned int) 0x1) // (TC) Clock selected: TIMER_DIV2_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV3_CLOCK     ((unsigned int) 0x2) // (TC) Clock selected: TIMER_DIV3_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV4_CLOCK     ((unsigned int) 0x3) // (TC) Clock selected: TIMER_DIV4_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV5_CLOCK     ((unsigned int) 0x4) // (TC) Clock selected: TIMER_DIV5_CLOCK
#define 	AT91C_TC_CLKS_XC0                  ((unsigned int) 0x5) // (TC) Clock selected: XC0
#define 	AT91C_TC_CLKS_XC1                  ((unsigned int) 0x6) // (TC) Clock selected: XC1
#define 	AT91C_TC_CLKS_XC2                  ((unsigned int) 0x7) // (TC) Clock selected: XC2
#define AT91C_TC_CLKI         ((unsigned int) 0x1 <<  3) // (TC) Clock Invert
#define AT91C_TC_BURST        ((unsigned int) 0x3 <<  4) // (TC) Burst Signal Selection
#define 	AT91C_TC_BURST_NONE                 ((unsigned int) 0x0 <<  4) // (TC) The clock is not gated by an external signal
#define 	AT91C_TC_BURST_XC0                  ((unsigned int) 0x1 <<  4) // (TC) XC0 is ANDed with the selected clock
#define 	AT91C_TC_BURST_XC1                  ((unsigned int) 0x2 <<  4) // (TC) XC1 is ANDed with the selected clock
#define 	AT91C_TC_BURST_XC2                  ((unsigned int) 0x3 <<  4) // (TC) XC2 is ANDed with the selected clock
#define AT91C_TC_CPCSTOP      ((unsigned int) 0x1 <<  6) // (TC) Counter Clock Stopped with RC Compare
#define AT91C_TC_LDBSTOP      ((unsigned int) 0x1 <<  6) // (TC) Counter Clock Stopped with RB Loading
#define AT91C_TC_CPCDIS       ((unsigned int) 0x1 <<  7) // (TC) Counter Clock Disable with RC Compare
#define AT91C_TC_LDBDIS       ((unsigned int) 0x1 <<  7) // (TC) Counter Clock Disabled with RB Loading
#define AT91C_TC_ETRGEDG      ((unsigned int) 0x3 <<  8) // (TC) External Trigger Edge Selection
#define 	AT91C_TC_ETRGEDG_NONE                 ((unsigned int) 0x0 <<  8) // (TC) Edge: None
#define 	AT91C_TC_ETRGEDG_RISING               ((unsigned int) 0x1 <<  8) // (TC) Edge: rising edge
#define 	AT91C_TC_ETRGEDG_FALLING              ((unsigned int) 0x2 <<  8) // (TC) Edge: falling edge
#define 	AT91C_TC_ETRGEDG_BOTH                 ((unsigned int) 0x3 <<  8) // (TC) Edge: each edge
#define AT91C_TC_EEVTEDG      ((unsigned int) 0x3 <<  8) // (TC) External Event Edge Selection
#define 	AT91C_TC_EEVTEDG_NONE                 ((unsigned int) 0x0 <<  8) // (TC) Edge: None
#define 	AT91C_TC_EEVTEDG_RISING               ((unsigned int) 0x1 <<  8) // (TC) Edge: rising edge
#define 	AT91C_TC_EEVTEDG_FALLING              ((unsigned int) 0x2 <<  8) // (TC) Edge: falling edge
#define 	AT91C_TC_EEVTEDG_BOTH                 ((unsigned int) 0x3 <<  8) // (TC) Edge: each edge
#define AT91C_TC_EEVT         ((unsigned int) 0x3 << 10) // (TC) External Event  Selection
#define 	AT91C_TC_EEVT_TIOB                 ((unsigned int) 0x0 << 10) // (TC) Signal selected as external event: TIOB TIOB direction: input
#define 	AT91C_TC_EEVT_XC0                  ((unsigned int) 0x1 << 10) // (TC) Signal selected as external event: XC0 TIOB direction: output
#define 	AT91C_TC_EEVT_XC1                  ((unsigned int) 0x2 << 10) // (TC) Signal selected as external event: XC1 TIOB direction: output
#define 	AT91C_TC_EEVT_XC2                  ((unsigned int) 0x3 << 10) // (TC) Signal selected as external event: XC2 TIOB direction: output
#define AT91C_TC_ABETRG       ((unsigned int) 0x1 << 10) // (TC) TIOA or TIOB External Trigger Selection
#define AT91C_TC_ENETRG       ((unsigned int) 0x1 << 12) // (TC) External Event Trigger enable
#define AT91C_TC_WAVESEL      ((unsigned int) 0x3 << 13) // (TC) Waveform  Selection
#define 	AT91C_TC_WAVESEL_UP                   ((unsigned int) 0x0 << 13) // (TC) UP mode without atomatic trigger on RC Compare
#define 	AT91C_TC_WAVESEL_UPDOWN               ((unsigned int) 0x1 << 13) // (TC) UPDOWN mode without automatic trigger on RC Compare
#define 	AT91C_TC_WAVESEL_UP_AUTO              ((unsigned int) 0x2 << 13) // (TC) UP mode with automatic trigger on RC Compare
#define 	AT91C_TC_WAVESEL_UPDOWN_AUTO          ((unsigned int) 0x3 << 13) // (TC) UPDOWN mode with automatic trigger on RC Compare
#define AT91C_TC_CPCTRG       ((unsigned int) 0x1 << 14) // (TC) RC Compare Trigger Enable
#define AT91C_TC_WAVE         ((unsigned int) 0x1 << 15) // (TC) 
#define AT91C_TC_ACPA         ((unsigned int) 0x3 << 16) // (TC) RA Compare Effect on TIOA
#define 	AT91C_TC_ACPA_NONE                 ((unsigned int) 0x0 << 16) // (TC) Effect: none
#define 	AT91C_TC_ACPA_SET                  ((unsigned int) 0x1 << 16) // (TC) Effect: set
#define 	AT91C_TC_ACPA_CLEAR                ((unsigned int) 0x2 << 16) // (TC) Effect: clear
#define 	AT91C_TC_ACPA_TOGGLE               ((unsigned int) 0x3 << 16) // (TC) Effect: toggle
#define AT91C_TC_LDRA         ((unsigned int) 0x3 << 16) // (TC) RA Loading Selection
#define 	AT91C_TC_LDRA_NONE                 ((unsigned int) 0x0 << 16) // (TC) Edge: None
#define 	AT91C_TC_LDRA_RISING               ((unsigned int) 0x1 << 16) // (TC) Edge: rising edge of TIOA
#define 	AT91C_TC_LDRA_FALLING              ((unsigned int) 0x2 << 16) // (TC) Edge: falling edge of TIOA
#define 	AT91C_TC_LDRA_BOTH                 ((unsigned int) 0x3 << 16) // (TC) Edge: each edge of TIOA
#define AT91C_TC_ACPC         ((unsigned int) 0x3 << 18) // (TC) RC Compare Effect on TIOA
#define 	AT91C_TC_ACPC_NONE                 ((unsigned int) 0x0 << 18) // (TC) Effect: none
#define 	AT91C_TC_ACPC_SET                  ((unsigned int) 0x1 << 18) // (TC) Effect: set
#define 	AT91C_TC_ACPC_CLEAR                ((unsigned int) 0x2 << 18) // (TC) Effect: clear
#define 	AT91C_TC_ACPC_TOGGLE               ((unsigned int) 0x3 << 18) // (TC) Effect: toggle
#define AT91C_TC_LDRB         ((unsigned int) 0x3 << 18) // (TC) RB Loading Selection
#define 	AT91C_TC_LDRB_NONE                 ((unsigned int) 0x0 << 18) // (TC) Edge: None
#define 	AT91C_TC_LDRB_RISING               ((unsigned int) 0x1 << 18) // (TC) Edge: rising edge of TIOA
#define 	AT91C_TC_LDRB_FALLING              ((unsigned int) 0x2 << 18) // (TC) Edge: falling edge of TIOA
#define 	AT91C_TC_LDRB_BOTH                 ((unsigned int) 0x3 << 18) // (TC) Edge: each edge of TIOA
#define AT91C_TC_AEEVT        ((unsigned int) 0x3 << 20) // (TC) External Event Effect on TIOA
#define 	AT91C_TC_AEEVT_NONE                 ((unsigned int) 0x0 << 20) // (TC) Effect: none
#define 	AT91C_TC_AEEVT_SET                  ((unsigned int) 0x1 << 20) // (TC) Effect: set
#define 	AT91C_TC_AEEVT_CLEAR                ((unsigned int) 0x2 << 20) // (TC) Effect: clear
#define 	AT91C_TC_AEEVT_TOGGLE               ((unsigned int) 0x3 << 20) // (TC) Effect: toggle
#define AT91C_TC_ASWTRG       ((unsigned int) 0x3 << 22) // (TC) Software Trigger Effect on TIOA
#define 	AT91C_TC_ASWTRG_NONE                 ((unsigned int) 0x0 << 22) // (TC) Effect: none
#define 	AT91C_TC_ASWTRG_SET                  ((unsigned int) 0x1 << 22) // (TC) Effect: set
#define 	AT91C_TC_ASWTRG_CLEAR                ((unsigned int) 0x2 << 22) // (TC) Effect: clear
#define 	AT91C_TC_ASWTRG_TOGGLE               ((unsigned int) 0x3 << 22) // (TC) Effect: toggle
#define AT91C_TC_BCPB         ((unsigned int) 0x3 << 24) // (TC) RB Compare Effect on TIOB
#define 	AT91C_TC_BCPB_NONE                 ((unsigned int) 0x0 << 24) // (TC) Effect: none
#define 	AT91C_TC_BCPB_SET                  ((unsigned int) 0x1 << 24) // (TC) Effect: set
#define 	AT91C_TC_BCPB_CLEAR                ((unsigned int) 0x2 << 24) // (TC) Effect: clear
#define 	AT91C_TC_BCPB_TOGGLE               ((unsigned int) 0x3 << 24) // (TC) Effect: toggle
#define AT91C_TC_BCPC         ((unsigned int) 0x3 << 26) // (TC) RC Compare Effect on TIOB
#define 	AT91C_TC_BCPC_NONE                 ((unsigned int) 0x0 << 26) // (TC) Effect: none
#define 	AT91C_TC_BCPC_SET                  ((unsigned int) 0x1 << 26) // (TC) Effect: set
#define 	AT91C_TC_BCPC_CLEAR                ((unsigned int) 0x2 << 26) // (TC) Effect: clear
#define 	AT91C_TC_BCPC_TOGGLE               ((unsigned int) 0x3 << 26) // (TC) Effect: toggle
#define AT91C_TC_BEEVT        ((unsigned int) 0x3 << 28) // (TC) External Event Effect on TIOB
#define 	AT91C_TC_BEEVT_NONE                 ((unsigned int) 0x0 << 28) // (TC) Effect: none
#define 	AT91C_TC_BEEVT_SET                  ((unsigned int) 0x1 << 28) // (TC) Effect: set
#define 	AT91C_TC_BEEVT_CLEAR                ((unsigned int) 0x2 << 28) // (TC) Effect: clear
#define 	AT91C_TC_BEEVT_TOGGLE               ((unsigned int) 0x3 << 28) // (TC) Effect: toggle
#define AT91C_TC_BSWTRG       ((unsigned int) 0x3 << 30) // (TC) Software Trigger Effect on TIOB
#define 	AT91C_TC_BSWTRG_NONE                 ((unsigned int) 0x0 << 30) // (TC) Effect: none
#define 	AT91C_TC_BSWTRG_SET                  ((unsigned int) 0x1 << 30) // (TC) Effect: set
#define 	AT91C_TC_BSWTRG_CLEAR                ((unsigned int) 0x2 << 30) // (TC) Effect: clear
#define 	AT91C_TC_BSWTRG_TOGGLE               ((unsigned int) 0x3 << 30) // (TC) Effect: toggle
// -------- TC_SR : (TC Offset: 0x20) TC Channel Status Register -------- 
#define AT91C_TC_COVFS        ((unsigned int) 0x1 <<  0) // (TC) Counter Overflow
#define AT91C_TC_LOVRS        ((unsigned int) 0x1 <<  1) // (TC) Load Overrun
#define AT91C_TC_CPAS         ((unsigned int) 0x1 <<  2) // (TC) RA Compare
#define AT91C_TC_CPBS         ((unsigned int) 0x1 <<  3) // (TC) RB Compare
#define AT91C_TC_CPCS         ((unsigned int) 0x1 <<  4) // (TC) RC Compare
#define AT91C_TC_LDRAS        ((unsigned int) 0x1 <<  5) // (TC) RA Loading
#define AT91C_TC_LDRBS        ((unsigned int) 0x1 <<  6) // (TC) RB Loading
#define AT91C_TC_ETRGS        ((unsigned int) 0x1 <<  7) // (TC) External Trigger
#define AT91C_TC_CLKSTA       ((unsigned int) 0x1 << 16) // (TC) Clock Enabling
#define AT91C_TC_MTIOA        ((unsigned int) 0x1 << 17) // (TC) TIOA Mirror
#define AT91C_TC_MTIOB        ((unsigned int) 0x1 << 18) // (TC) TIOA Mirror
// -------- TC_IER : (TC Offset: 0x24) TC Channel Interrupt Enable Register -------- 
// -------- TC_IDR : (TC Offset: 0x28) TC Channel Interrupt Disable Register -------- 
// -------- TC_IMR : (TC Offset: 0x2c) TC Channel Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Timer Counter Interface
// *****************************************************************************
typedef struct _AT91S_TCB {
	AT91S_TC	 TCB_TC0; 	// TC Channel 0
	AT91_REG	 Reserved0[4]; 	// 
	AT91S_TC	 TCB_TC1; 	// TC Channel 1
	AT91_REG	 Reserved1[4]; 	// 
	AT91S_TC	 TCB_TC2; 	// TC Channel 2
	AT91_REG	 Reserved2[4]; 	// 
	AT91_REG	 TCB_BCR; 	// TC Block Control Register
	AT91_REG	 TCB_BMR; 	// TC Block Mode Register
	AT91_REG	 Reserved3[9]; 	// 
	AT91_REG	 TCB_ADDRSIZE; 	// TC ADDRSIZE REGISTER 
	AT91_REG	 TCB_IPNAME1; 	// TC IPNAME1 REGISTER 
	AT91_REG	 TCB_IPNAME2; 	// TC IPNAME2 REGISTER 
	AT91_REG	 TCB_FEATURES; 	// TC FEATURES REGISTER 
	AT91_REG	 TCB_VER; 	//  Version Register
} AT91S_TCB, *AT91PS_TCB;

// -------- TCB_BCR : (TCB Offset: 0xc0) TC Block Control Register -------- 
#define AT91C_TCB_SYNC        ((unsigned int) 0x1 <<  0) // (TCB) Synchro Command
// -------- TCB_BMR : (TCB Offset: 0xc4) TC Block Mode Register -------- 
#define AT91C_TCB_TC0XC0S     ((unsigned int) 0x3 <<  0) // (TCB) External Clock Signal 0 Selection
#define 	AT91C_TCB_TC0XC0S_TCLK0                ((unsigned int) 0x0) // (TCB) TCLK0 connected to XC0
#define 	AT91C_TCB_TC0XC0S_NONE                 ((unsigned int) 0x1) // (TCB) None signal connected to XC0
#define 	AT91C_TCB_TC0XC0S_TIOA1                ((unsigned int) 0x2) // (TCB) TIOA1 connected to XC0
#define 	AT91C_TCB_TC0XC0S_TIOA2                ((unsigned int) 0x3) // (TCB) TIOA2 connected to XC0
#define AT91C_TCB_TC1XC1S     ((unsigned int) 0x3 <<  2) // (TCB) External Clock Signal 1 Selection
#define 	AT91C_TCB_TC1XC1S_TCLK1                ((unsigned int) 0x0 <<  2) // (TCB) TCLK1 connected to XC1
#define 	AT91C_TCB_TC1XC1S_NONE                 ((unsigned int) 0x1 <<  2) // (TCB) None signal connected to XC1
#define 	AT91C_TCB_TC1XC1S_TIOA0                ((unsigned int) 0x2 <<  2) // (TCB) TIOA0 connected to XC1
#define 	AT91C_TCB_TC1XC1S_TIOA2                ((unsigned int) 0x3 <<  2) // (TCB) TIOA2 connected to XC1
#define AT91C_TCB_TC2XC2S     ((unsigned int) 0x3 <<  4) // (TCB) External Clock Signal 2 Selection
#define 	AT91C_TCB_TC2XC2S_TCLK2                ((unsigned int) 0x0 <<  4) // (TCB) TCLK2 connected to XC2
#define 	AT91C_TCB_TC2XC2S_NONE                 ((unsigned int) 0x1 <<  4) // (TCB) None signal connected to XC2
#define 	AT91C_TCB_TC2XC2S_TIOA0                ((unsigned int) 0x2 <<  4) // (TCB) TIOA0 connected to XC2
#define 	AT91C_TCB_TC2XC2S_TIOA1                ((unsigned int) 0x3 <<  4) // (TCB) TIOA2 connected to XC2

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Multimedia Card Interface
// *****************************************************************************
typedef struct _AT91S_MCI {
	AT91_REG	 MCI_CR; 	// MCI Control Register
	AT91_REG	 MCI_MR; 	// MCI Mode Register
	AT91_REG	 MCI_DTOR; 	// MCI Data Timeout Register
	AT91_REG	 MCI_SDCR; 	// MCI SD/SDIO Card Register
	AT91_REG	 MCI_ARGR; 	// MCI Argument Register
	AT91_REG	 MCI_CMDR; 	// MCI Command Register
	AT91_REG	 MCI_BLKR; 	// MCI Block Register
	AT91_REG	 MCI_CSTOR; 	// MCI Completion Signal Timeout Register
	AT91_REG	 MCI_RSPR[4]; 	// MCI Response Register
	AT91_REG	 MCI_RDR; 	// MCI Receive Data Register
	AT91_REG	 MCI_TDR; 	// MCI Transmit Data Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 MCI_SR; 	// MCI Status Register
	AT91_REG	 MCI_IER; 	// MCI Interrupt Enable Register
	AT91_REG	 MCI_IDR; 	// MCI Interrupt Disable Register
	AT91_REG	 MCI_IMR; 	// MCI Interrupt Mask Register
	AT91_REG	 MCI_DMA; 	// MCI DMA Configuration Register
	AT91_REG	 MCI_CFG; 	// MCI Configuration Register
	AT91_REG	 Reserved1[35]; 	// 
	AT91_REG	 MCI_WPCR; 	// MCI Write Protection Control Register
	AT91_REG	 MCI_WPSR; 	// MCI Write Protection Status Register
	AT91_REG	 MCI_ADDRSIZE; 	// MCI ADDRSIZE REGISTER 
	AT91_REG	 MCI_IPNAME1; 	// MCI IPNAME1 REGISTER 
	AT91_REG	 MCI_IPNAME2; 	// MCI IPNAME2 REGISTER 
	AT91_REG	 MCI_FEATURES; 	// MCI FEATURES REGISTER 
	AT91_REG	 MCI_VER; 	// MCI VERSION REGISTER 
	AT91_REG	 MCI_RPR; 	// Receive Pointer Register
	AT91_REG	 MCI_RCR; 	// Receive Counter Register
	AT91_REG	 MCI_TPR; 	// Transmit Pointer Register
	AT91_REG	 MCI_TCR; 	// Transmit Counter Register
	AT91_REG	 MCI_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 MCI_RNCR; 	// Receive Next Counter Register
	AT91_REG	 MCI_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 MCI_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 MCI_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 MCI_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved2[54]; 	// 
	AT91_REG	 MCI_FIFO; 	// MCI FIFO Aperture Register
} AT91S_MCI, *AT91PS_MCI;

// -------- MCI_CR : (MCI Offset: 0x0) MCI Control Register -------- 
#define AT91C_MCI_MCIEN       ((unsigned int) 0x1 <<  0) // (MCI) Multimedia Interface Enable
#define 	AT91C_MCI_MCIEN_0                    ((unsigned int) 0x0) // (MCI) No effect
#define 	AT91C_MCI_MCIEN_1                    ((unsigned int) 0x1) // (MCI) Enable the MultiMedia Interface if MCIDIS is 0
#define AT91C_MCI_MCIDIS      ((unsigned int) 0x1 <<  1) // (MCI) Multimedia Interface Disable
#define 	AT91C_MCI_MCIDIS_0                    ((unsigned int) 0x0 <<  1) // (MCI) No effect
#define 	AT91C_MCI_MCIDIS_1                    ((unsigned int) 0x1 <<  1) // (MCI) Disable the MultiMedia Interface
#define AT91C_MCI_PWSEN       ((unsigned int) 0x1 <<  2) // (MCI) Power Save Mode Enable
#define 	AT91C_MCI_PWSEN_0                    ((unsigned int) 0x0 <<  2) // (MCI) No effect
#define 	AT91C_MCI_PWSEN_1                    ((unsigned int) 0x1 <<  2) // (MCI) Enable the Power-saving mode if PWSDIS is 0.
#define AT91C_MCI_PWSDIS      ((unsigned int) 0x1 <<  3) // (MCI) Power Save Mode Disable
#define 	AT91C_MCI_PWSDIS_0                    ((unsigned int) 0x0 <<  3) // (MCI) No effect
#define 	AT91C_MCI_PWSDIS_1                    ((unsigned int) 0x1 <<  3) // (MCI) Disable the Power-saving mode.
#define AT91C_MCI_IOWAITEN    ((unsigned int) 0x1 <<  4) // (MCI) SDIO Read Wait Enable
#define 	AT91C_MCI_IOWAITEN_0                    ((unsigned int) 0x0 <<  4) // (MCI) No effect
#define 	AT91C_MCI_IOWAITEN_1                    ((unsigned int) 0x1 <<  4) // (MCI) Enables the SDIO Read Wait Operation.
#define AT91C_MCI_IOWAITDIS   ((unsigned int) 0x1 <<  5) // (MCI) SDIO Read Wait Disable
#define 	AT91C_MCI_IOWAITDIS_0                    ((unsigned int) 0x0 <<  5) // (MCI) No effect
#define 	AT91C_MCI_IOWAITDIS_1                    ((unsigned int) 0x1 <<  5) // (MCI) Disables the SDIO Read Wait Operation.
#define AT91C_MCI_SWRST       ((unsigned int) 0x1 <<  7) // (MCI) MCI Software reset
#define 	AT91C_MCI_SWRST_0                    ((unsigned int) 0x0 <<  7) // (MCI) No effect
#define 	AT91C_MCI_SWRST_1                    ((unsigned int) 0x1 <<  7) // (MCI) Resets the MCI
// -------- MCI_MR : (MCI Offset: 0x4) MCI Mode Register -------- 
#define AT91C_MCI_CLKDIV      ((unsigned int) 0xFF <<  0) // (MCI) Clock Divider
#define AT91C_MCI_PWSDIV      ((unsigned int) 0x7 <<  8) // (MCI) Power Saving Divider
#define AT91C_MCI_RDPROOF     ((unsigned int) 0x1 << 11) // (MCI) Read Proof Enable
#define 	AT91C_MCI_RDPROOF_DISABLE              ((unsigned int) 0x0 << 11) // (MCI) Disables Read Proof
#define 	AT91C_MCI_RDPROOF_ENABLE               ((unsigned int) 0x1 << 11) // (MCI) Enables Read Proof
#define AT91C_MCI_WRPROOF     ((unsigned int) 0x1 << 12) // (MCI) Write Proof Enable
#define 	AT91C_MCI_WRPROOF_DISABLE              ((unsigned int) 0x0 << 12) // (MCI) Disables Write Proof
#define 	AT91C_MCI_WRPROOF_ENABLE               ((unsigned int) 0x1 << 12) // (MCI) Enables Write Proof
#define AT91C_MCI_PDCFBYTE    ((unsigned int) 0x1 << 13) // (MCI) PDC Force Byte Transfer
#define 	AT91C_MCI_PDCFBYTE_DISABLE              ((unsigned int) 0x0 << 13) // (MCI) Disables PDC Force Byte Transfer
#define 	AT91C_MCI_PDCFBYTE_ENABLE               ((unsigned int) 0x1 << 13) // (MCI) Enables PDC Force Byte Transfer
#define AT91C_MCI_PDCPADV     ((unsigned int) 0x1 << 14) // (MCI) PDC Padding Value
#define AT91C_MCI_PDCMODE     ((unsigned int) 0x1 << 15) // (MCI) PDC Oriented Mode
#define 	AT91C_MCI_PDCMODE_DISABLE              ((unsigned int) 0x0 << 15) // (MCI) Disables PDC Transfer
#define 	AT91C_MCI_PDCMODE_ENABLE               ((unsigned int) 0x1 << 15) // (MCI) Enables PDC Transfer
#define AT91C_MCI_BLKLEN      ((unsigned int) 0xFFFF << 16) // (MCI) Data Block Length
// -------- MCI_DTOR : (MCI Offset: 0x8) MCI Data Timeout Register -------- 
#define AT91C_MCI_DTOCYC      ((unsigned int) 0xF <<  0) // (MCI) Data Timeout Cycle Number
#define AT91C_MCI_DTOMUL      ((unsigned int) 0x7 <<  4) // (MCI) Data Timeout Multiplier
#define 	AT91C_MCI_DTOMUL_1                    ((unsigned int) 0x0 <<  4) // (MCI) DTOCYC x 1
#define 	AT91C_MCI_DTOMUL_16                   ((unsigned int) 0x1 <<  4) // (MCI) DTOCYC x 16
#define 	AT91C_MCI_DTOMUL_128                  ((unsigned int) 0x2 <<  4) // (MCI) DTOCYC x 128
#define 	AT91C_MCI_DTOMUL_256                  ((unsigned int) 0x3 <<  4) // (MCI) DTOCYC x 256
#define 	AT91C_MCI_DTOMUL_1024                 ((unsigned int) 0x4 <<  4) // (MCI) DTOCYC x 1024
#define 	AT91C_MCI_DTOMUL_4096                 ((unsigned int) 0x5 <<  4) // (MCI) DTOCYC x 4096
#define 	AT91C_MCI_DTOMUL_65536                ((unsigned int) 0x6 <<  4) // (MCI) DTOCYC x 65536
#define 	AT91C_MCI_DTOMUL_1048576              ((unsigned int) 0x7 <<  4) // (MCI) DTOCYC x 1048576
// -------- MCI_SDCR : (MCI Offset: 0xc) MCI SD Card Register -------- 
#define AT91C_MCI_SCDSEL      ((unsigned int) 0x3 <<  0) // (MCI) SD Card/SDIO Selector
#define 	AT91C_MCI_SCDSEL_SLOTA                ((unsigned int) 0x0) // (MCI) Slot A selected
#define 	AT91C_MCI_SCDSEL_SLOTB                ((unsigned int) 0x1) // (MCI) Slot B selected
#define 	AT91C_MCI_SCDSEL_SLOTC                ((unsigned int) 0x2) // (MCI) Slot C selected
#define 	AT91C_MCI_SCDSEL_SLOTD                ((unsigned int) 0x3) // (MCI) Slot D selected
#define AT91C_MCI_SCDBUS      ((unsigned int) 0x3 <<  6) // (MCI) SDCard/SDIO Bus Width
#define 	AT91C_MCI_SCDBUS_1BIT                 ((unsigned int) 0x0 <<  6) // (MCI) 1-bit data bus
#define 	AT91C_MCI_SCDBUS_4BITS                ((unsigned int) 0x2 <<  6) // (MCI) 4-bits data bus
#define 	AT91C_MCI_SCDBUS_8BITS                ((unsigned int) 0x3 <<  6) // (MCI) 8-bits data bus
// -------- MCI_CMDR : (MCI Offset: 0x14) MCI Command Register -------- 
#define AT91C_MCI_CMDNB       ((unsigned int) 0x3F <<  0) // (MCI) Command Number
#define AT91C_MCI_RSPTYP      ((unsigned int) 0x3 <<  6) // (MCI) Response Type
#define 	AT91C_MCI_RSPTYP_NO                   ((unsigned int) 0x0 <<  6) // (MCI) No response
#define 	AT91C_MCI_RSPTYP_48                   ((unsigned int) 0x1 <<  6) // (MCI) 48-bit response
#define 	AT91C_MCI_RSPTYP_136                  ((unsigned int) 0x2 <<  6) // (MCI) 136-bit response
#define 	AT91C_MCI_RSPTYP_R1B                  ((unsigned int) 0x3 <<  6) // (MCI) R1b response
#define AT91C_MCI_SPCMD       ((unsigned int) 0x7 <<  8) // (MCI) Special CMD
#define 	AT91C_MCI_SPCMD_NONE                 ((unsigned int) 0x0 <<  8) // (MCI) Not a special CMD
#define 	AT91C_MCI_SPCMD_INIT                 ((unsigned int) 0x1 <<  8) // (MCI) Initialization CMD
#define 	AT91C_MCI_SPCMD_SYNC                 ((unsigned int) 0x2 <<  8) // (MCI) Synchronized CMD
#define 	AT91C_MCI_SPCMD_CE_ATA               ((unsigned int) 0x3 <<  8) // (MCI) CE-ATA Completion Signal disable CMD
#define 	AT91C_MCI_SPCMD_IT_CMD               ((unsigned int) 0x4 <<  8) // (MCI) Interrupt command
#define 	AT91C_MCI_SPCMD_IT_REP               ((unsigned int) 0x5 <<  8) // (MCI) Interrupt response
#define AT91C_MCI_OPDCMD      ((unsigned int) 0x1 << 11) // (MCI) Open Drain Command
#define 	AT91C_MCI_OPDCMD_PUSHPULL             ((unsigned int) 0x0 << 11) // (MCI) Push/pull command
#define 	AT91C_MCI_OPDCMD_OPENDRAIN            ((unsigned int) 0x1 << 11) // (MCI) Open drain command
#define AT91C_MCI_MAXLAT      ((unsigned int) 0x1 << 12) // (MCI) Maximum Latency for Command to respond
#define 	AT91C_MCI_MAXLAT_5                    ((unsigned int) 0x0 << 12) // (MCI) 5 cycles maximum latency
#define 	AT91C_MCI_MAXLAT_64                   ((unsigned int) 0x1 << 12) // (MCI) 64 cycles maximum latency
#define AT91C_MCI_TRCMD       ((unsigned int) 0x3 << 16) // (MCI) Transfer CMD
#define 	AT91C_MCI_TRCMD_NO                   ((unsigned int) 0x0 << 16) // (MCI) No transfer
#define 	AT91C_MCI_TRCMD_START                ((unsigned int) 0x1 << 16) // (MCI) Start transfer
#define 	AT91C_MCI_TRCMD_STOP                 ((unsigned int) 0x2 << 16) // (MCI) Stop transfer
#define AT91C_MCI_TRDIR       ((unsigned int) 0x1 << 18) // (MCI) Transfer Direction
#define 	AT91C_MCI_TRDIR_WRITE                ((unsigned int) 0x0 << 18) // (MCI) Write
#define 	AT91C_MCI_TRDIR_READ                 ((unsigned int) 0x1 << 18) // (MCI) Read
#define AT91C_MCI_TRTYP       ((unsigned int) 0x7 << 19) // (MCI) Transfer Type
#define 	AT91C_MCI_TRTYP_BLOCK                ((unsigned int) 0x0 << 19) // (MCI) MMC/SDCard Single Block Transfer type
#define 	AT91C_MCI_TRTYP_MULTIPLE             ((unsigned int) 0x1 << 19) // (MCI) MMC/SDCard Multiple Block transfer type
#define 	AT91C_MCI_TRTYP_STREAM               ((unsigned int) 0x2 << 19) // (MCI) MMC Stream transfer type
#define 	AT91C_MCI_TRTYP_SDIO_BYTE            ((unsigned int) 0x4 << 19) // (MCI) SDIO Byte transfer type
#define 	AT91C_MCI_TRTYP_SDIO_BLOCK           ((unsigned int) 0x5 << 19) // (MCI) SDIO Block transfer type
#define AT91C_MCI_IOSPCMD     ((unsigned int) 0x3 << 24) // (MCI) SDIO Special Command
#define 	AT91C_MCI_IOSPCMD_NONE                 ((unsigned int) 0x0 << 24) // (MCI) NOT a special command
#define 	AT91C_MCI_IOSPCMD_SUSPEND              ((unsigned int) 0x1 << 24) // (MCI) SDIO Suspend Command
#define 	AT91C_MCI_IOSPCMD_RESUME               ((unsigned int) 0x2 << 24) // (MCI) SDIO Resume Command
#define AT91C_MCI_ATACS       ((unsigned int) 0x1 << 26) // (MCI) ATA with command completion signal
#define 	AT91C_MCI_ATACS_NORMAL               ((unsigned int) 0x0 << 26) // (MCI) normal operation mode
#define 	AT91C_MCI_ATACS_COMPLETION           ((unsigned int) 0x1 << 26) // (MCI) completion signal is expected within MCI_CSTOR
// -------- MCI_BLKR : (MCI Offset: 0x18) MCI Block Register -------- 
#define AT91C_MCI_BCNT        ((unsigned int) 0xFFFF <<  0) // (MCI) MMC/SDIO Block Count / SDIO Byte Count
// -------- MCI_CSTOR : (MCI Offset: 0x1c) MCI Completion Signal Timeout Register -------- 
#define AT91C_MCI_CSTOCYC     ((unsigned int) 0xF <<  0) // (MCI) Completion Signal Timeout Cycle Number
#define AT91C_MCI_CSTOMUL     ((unsigned int) 0x7 <<  4) // (MCI) Completion Signal Timeout Multiplier
#define 	AT91C_MCI_CSTOMUL_1                    ((unsigned int) 0x0 <<  4) // (MCI) CSTOCYC x 1
#define 	AT91C_MCI_CSTOMUL_16                   ((unsigned int) 0x1 <<  4) // (MCI) CSTOCYC x  16
#define 	AT91C_MCI_CSTOMUL_128                  ((unsigned int) 0x2 <<  4) // (MCI) CSTOCYC x  128
#define 	AT91C_MCI_CSTOMUL_256                  ((unsigned int) 0x3 <<  4) // (MCI) CSTOCYC x  256
#define 	AT91C_MCI_CSTOMUL_1024                 ((unsigned int) 0x4 <<  4) // (MCI) CSTOCYC x  1024
#define 	AT91C_MCI_CSTOMUL_4096                 ((unsigned int) 0x5 <<  4) // (MCI) CSTOCYC x  4096
#define 	AT91C_MCI_CSTOMUL_65536                ((unsigned int) 0x6 <<  4) // (MCI) CSTOCYC x  65536
#define 	AT91C_MCI_CSTOMUL_1048576              ((unsigned int) 0x7 <<  4) // (MCI) CSTOCYC x  1048576
// -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- 
#define AT91C_MCI_CMDRDY      ((unsigned int) 0x1 <<  0) // (MCI) Command Ready flag
#define AT91C_MCI_RXRDY       ((unsigned int) 0x1 <<  1) // (MCI) RX Ready flag
#define AT91C_MCI_TXRDY       ((unsigned int) 0x1 <<  2) // (MCI) TX Ready flag
#define AT91C_MCI_BLKE        ((unsigned int) 0x1 <<  3) // (MCI) Data Block Transfer Ended flag
#define AT91C_MCI_DTIP        ((unsigned int) 0x1 <<  4) // (MCI) Data Transfer in Progress flag
#define AT91C_MCI_NOTBUSY     ((unsigned int) 0x1 <<  5) // (MCI) Data Line Not Busy flag
#define AT91C_MCI_ENDRX       ((unsigned int) 0x1 <<  6) // (MCI) End of RX Buffer flag
#define AT91C_MCI_ENDTX       ((unsigned int) 0x1 <<  7) // (MCI) End of TX Buffer flag
#define AT91C_MCI_SDIOIRQA    ((unsigned int) 0x1 <<  8) // (MCI) SDIO Interrupt for Slot A
#define AT91C_MCI_SDIOIRQB    ((unsigned int) 0x1 <<  9) // (MCI) SDIO Interrupt for Slot B
#define AT91C_MCI_SDIOIRQC    ((unsigned int) 0x1 << 10) // (MCI) SDIO Interrupt for Slot C
#define AT91C_MCI_SDIOIRQD    ((unsigned int) 0x1 << 11) // (MCI) SDIO Interrupt for Slot D
#define AT91C_MCI_SDIOWAIT    ((unsigned int) 0x1 << 12) // (MCI) SDIO Read Wait operation flag
#define AT91C_MCI_CSRCV       ((unsigned int) 0x1 << 13) // (MCI) CE-ATA Completion Signal flag
#define AT91C_MCI_RXBUFF      ((unsigned int) 0x1 << 14) // (MCI) RX Buffer Full flag
#define AT91C_MCI_TXBUFE      ((unsigned int) 0x1 << 15) // (MCI) TX Buffer Empty flag
#define AT91C_MCI_RINDE       ((unsigned int) 0x1 << 16) // (MCI) Response Index Error flag
#define AT91C_MCI_RDIRE       ((unsigned int) 0x1 << 17) // (MCI) Response Direction Error flag
#define AT91C_MCI_RCRCE       ((unsigned int) 0x1 << 18) // (MCI) Response CRC Error flag
#define AT91C_MCI_RENDE       ((unsigned int) 0x1 << 19) // (MCI) Response End Bit Error flag
#define AT91C_MCI_RTOE        ((unsigned int) 0x1 << 20) // (MCI) Response Time-out Error flag
#define AT91C_MCI_DCRCE       ((unsigned int) 0x1 << 21) // (MCI) data CRC Error flag
#define AT91C_MCI_DTOE        ((unsigned int) 0x1 << 22) // (MCI) Data timeout Error flag
#define AT91C_MCI_CSTOE       ((unsigned int) 0x1 << 23) // (MCI) Completion Signal timeout Error flag
#define AT91C_MCI_BLKOVRE     ((unsigned int) 0x1 << 24) // (MCI) DMA Block Overrun Error flag
#define AT91C_MCI_DMADONE     ((unsigned int) 0x1 << 25) // (MCI) DMA Transfer Done flag
#define AT91C_MCI_FIFOEMPTY   ((unsigned int) 0x1 << 26) // (MCI) FIFO Empty flag
#define AT91C_MCI_XFRDONE     ((unsigned int) 0x1 << 27) // (MCI) Transfer Done flag
#define AT91C_MCI_OVRE        ((unsigned int) 0x1 << 30) // (MCI) Overrun flag
#define AT91C_MCI_UNRE        ((unsigned int) 0x1 << 31) // (MCI) Underrun flag
// -------- MCI_IER : (MCI Offset: 0x44) MCI Interrupt Enable Register -------- 
// -------- MCI_IDR : (MCI Offset: 0x48) MCI Interrupt Disable Register -------- 
// -------- MCI_IMR : (MCI Offset: 0x4c) MCI Interrupt Mask Register -------- 
// -------- MCI_DMA : (MCI Offset: 0x50) MCI DMA Configuration Register -------- 
#define AT91C_MCI_OFFSET      ((unsigned int) 0x3 <<  0) // (MCI) DMA Write Buffer Offset
#define AT91C_MCI_CHKSIZE     ((unsigned int) 0x7 <<  4) // (MCI) DMA Channel Read/Write Chunk Size
#define 	AT91C_MCI_CHKSIZE_1                    ((unsigned int) 0x0 <<  4) // (MCI) Number of data transferred is 1
#define 	AT91C_MCI_CHKSIZE_4                    ((unsigned int) 0x1 <<  4) // (MCI) Number of data transferred is 4
#define 	AT91C_MCI_CHKSIZE_8                    ((unsigned int) 0x2 <<  4) // (MCI) Number of data transferred is 8
#define 	AT91C_MCI_CHKSIZE_16                   ((unsigned int) 0x3 <<  4) // (MCI) Number of data transferred is 16
#define 	AT91C_MCI_CHKSIZE_32                   ((unsigned int) 0x4 <<  4) // (MCI) Number of data transferred is 32
#define AT91C_MCI_DMAEN       ((unsigned int) 0x1 <<  8) // (MCI) DMA Hardware Handshaking Enable
#define 	AT91C_MCI_DMAEN_DISABLE              ((unsigned int) 0x0 <<  8) // (MCI) DMA interface is disabled
#define 	AT91C_MCI_DMAEN_ENABLE               ((unsigned int) 0x1 <<  8) // (MCI) DMA interface is enabled
// -------- MCI_CFG : (MCI Offset: 0x54) MCI Configuration Register -------- 
#define AT91C_MCI_FIFOMODE    ((unsigned int) 0x1 <<  0) // (MCI) MCI Internal FIFO Control Mode
#define 	AT91C_MCI_FIFOMODE_AMOUNTDATA           ((unsigned int) 0x0) // (MCI) A write transfer starts when a sufficient amount of datas is written into the FIFO
#define 	AT91C_MCI_FIFOMODE_ONEDATA              ((unsigned int) 0x1) // (MCI) A write transfer starts as soon as one data is written into the FIFO
#define AT91C_MCI_FERRCTRL    ((unsigned int) 0x1 <<  4) // (MCI) Flow Error Flag Reset Control Mode
#define 	AT91C_MCI_FERRCTRL_RWCMD                ((unsigned int) 0x0 <<  4) // (MCI) When an underflow/overflow condition flag is set, a new Write/Read command is needed to reset the flag
#define 	AT91C_MCI_FERRCTRL_READSR               ((unsigned int) 0x1 <<  4) // (MCI) When an underflow/overflow condition flag is set, a read status resets the flag
#define AT91C_MCI_HSMODE      ((unsigned int) 0x1 <<  8) // (MCI) High Speed Mode
#define 	AT91C_MCI_HSMODE_DISABLE              ((unsigned int) 0x0 <<  8) // (MCI) Default Bus Timing Mode
#define 	AT91C_MCI_HSMODE_ENABLE               ((unsigned int) 0x1 <<  8) // (MCI) High Speed Mode
#define AT91C_MCI_LSYNC       ((unsigned int) 0x1 << 12) // (MCI) Synchronize on last block
#define 	AT91C_MCI_LSYNC_CURRENT              ((unsigned int) 0x0 << 12) // (MCI) Pending command sent at end of current data block
#define 	AT91C_MCI_LSYNC_INFINITE             ((unsigned int) 0x1 << 12) // (MCI) Pending command sent at end of block transfer when transfer length is not infinite
// -------- MCI_WPCR : (MCI Offset: 0xe4) Write Protection Control Register -------- 
#define AT91C_MCI_WP_EN       ((unsigned int) 0x1 <<  0) // (MCI) Write Protection Enable
#define 	AT91C_MCI_WP_EN_DISABLE              ((unsigned int) 0x0) // (MCI) Write Operation is disabled (if WP_KEY corresponds)
#define 	AT91C_MCI_WP_EN_ENABLE               ((unsigned int) 0x1) // (MCI) Write Operation is enabled (if WP_KEY corresponds)
#define AT91C_MCI_WP_KEY      ((unsigned int) 0xFFFFFF <<  8) // (MCI) Write Protection Key
// -------- MCI_WPSR : (MCI Offset: 0xe8) Write Protection Status Register -------- 
#define AT91C_MCI_WP_VS       ((unsigned int) 0xF <<  0) // (MCI) Write Protection Violation Status
#define 	AT91C_MCI_WP_VS_NO_VIOLATION         ((unsigned int) 0x0) // (MCI) No Write Protection Violation detected since last read
#define 	AT91C_MCI_WP_VS_ON_WRITE             ((unsigned int) 0x1) // (MCI) Write Protection Violation detected since last read
#define 	AT91C_MCI_WP_VS_ON_RESET             ((unsigned int) 0x2) // (MCI) Software Reset Violation detected since last read
#define 	AT91C_MCI_WP_VS_ON_BOTH              ((unsigned int) 0x3) // (MCI) Write Protection and Software Reset Violation detected since last read
#define AT91C_MCI_WP_VSRC     ((unsigned int) 0xF <<  8) // (MCI) Write Protection Violation Source
#define 	AT91C_MCI_WP_VSRC_NO_VIOLATION         ((unsigned int) 0x0 <<  8) // (MCI) No Write Protection Violation detected since last read
#define 	AT91C_MCI_WP_VSRC_MCI_MR               ((unsigned int) 0x1 <<  8) // (MCI) Write Protection Violation detected on MCI_MR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_DTOR             ((unsigned int) 0x2 <<  8) // (MCI) Write Protection Violation detected on MCI_DTOR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_SDCR             ((unsigned int) 0x3 <<  8) // (MCI) Write Protection Violation detected on MCI_SDCR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_CSTOR            ((unsigned int) 0x4 <<  8) // (MCI) Write Protection Violation detected on MCI_CSTOR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_DMA              ((unsigned int) 0x5 <<  8) // (MCI) Write Protection Violation detected on MCI_DMA since last read
#define 	AT91C_MCI_WP_VSRC_MCI_CFG              ((unsigned int) 0x6 <<  8) // (MCI) Write Protection Violation detected on MCI_CFG since last read
#define 	AT91C_MCI_WP_VSRC_MCI_DEL              ((unsigned int) 0x7 <<  8) // (MCI) Write Protection Violation detected on MCI_DEL since last read
// -------- MCI_VER : (MCI Offset: 0xfc)  VERSION  Register -------- 
#define AT91C_MCI_VER         ((unsigned int) 0xF <<  0) // (MCI)  VERSION  Register

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Two-wire Interface
// *****************************************************************************
typedef struct _AT91S_TWI {
	AT91_REG	 TWI_CR; 	// Control Register
	AT91_REG	 TWI_MMR; 	// Master Mode Register
	AT91_REG	 TWI_SMR; 	// Slave Mode Register
	AT91_REG	 TWI_IADR; 	// Internal Address Register
	AT91_REG	 TWI_CWGR; 	// Clock Waveform Generator Register
	AT91_REG	 Reserved0[3]; 	// 
	AT91_REG	 TWI_SR; 	// Status Register
	AT91_REG	 TWI_IER; 	// Interrupt Enable Register
	AT91_REG	 TWI_IDR; 	// Interrupt Disable Register
	AT91_REG	 TWI_IMR; 	// Interrupt Mask Register
	AT91_REG	 TWI_RHR; 	// Receive Holding Register
	AT91_REG	 TWI_THR; 	// Transmit Holding Register
	AT91_REG	 Reserved1[45]; 	// 
	AT91_REG	 TWI_ADDRSIZE; 	// TWI ADDRSIZE REGISTER 
	AT91_REG	 TWI_IPNAME1; 	// TWI IPNAME1 REGISTER 
	AT91_REG	 TWI_IPNAME2; 	// TWI IPNAME2 REGISTER 
	AT91_REG	 TWI_FEATURES; 	// TWI FEATURES REGISTER 
	AT91_REG	 TWI_VER; 	// Version Register
	AT91_REG	 TWI_RPR; 	// Receive Pointer Register
	AT91_REG	 TWI_RCR; 	// Receive Counter Register
	AT91_REG	 TWI_TPR; 	// Transmit Pointer Register
	AT91_REG	 TWI_TCR; 	// Transmit Counter Register
	AT91_REG	 TWI_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 TWI_RNCR; 	// Receive Next Counter Register
	AT91_REG	 TWI_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 TWI_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 TWI_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 TWI_PTSR; 	// PDC Transfer Status Register
} AT91S_TWI, *AT91PS_TWI;

// -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- 
#define AT91C_TWI_START       ((unsigned int) 0x1 <<  0) // (TWI) Send a START Condition
#define AT91C_TWI_STOP        ((unsigned int) 0x1 <<  1) // (TWI) Send a STOP Condition
#define AT91C_TWI_MSEN        ((unsigned int) 0x1 <<  2) // (TWI) TWI Master Transfer Enabled
#define AT91C_TWI_MSDIS       ((unsigned int) 0x1 <<  3) // (TWI) TWI Master Transfer Disabled
#define AT91C_TWI_SVEN        ((unsigned int) 0x1 <<  4) // (TWI) TWI Slave mode Enabled
#define AT91C_TWI_SVDIS       ((unsigned int) 0x1 <<  5) // (TWI) TWI Slave mode Disabled
#define AT91C_TWI_SWRST       ((unsigned int) 0x1 <<  7) // (TWI) Software Reset
// -------- TWI_MMR : (TWI Offset: 0x4) TWI Master Mode Register -------- 
#define AT91C_TWI_IADRSZ      ((unsigned int) 0x3 <<  8) // (TWI) Internal Device Address Size
#define 	AT91C_TWI_IADRSZ_NO                   ((unsigned int) 0x0 <<  8) // (TWI) No internal device address
#define 	AT91C_TWI_IADRSZ_1_BYTE               ((unsigned int) 0x1 <<  8) // (TWI) One-byte internal device address
#define 	AT91C_TWI_IADRSZ_2_BYTE               ((unsigned int) 0x2 <<  8) // (TWI) Two-byte internal device address
#define 	AT91C_TWI_IADRSZ_3_BYTE               ((unsigned int) 0x3 <<  8) // (TWI) Three-byte internal device address
#define AT91C_TWI_MREAD       ((unsigned int) 0x1 << 12) // (TWI) Master Read Direction
#define AT91C_TWI_DADR        ((unsigned int) 0x7F << 16) // (TWI) Device Address
// -------- TWI_SMR : (TWI Offset: 0x8) TWI Slave Mode Register -------- 
#define AT91C_TWI_SADR        ((unsigned int) 0x7F << 16) // (TWI) Slave Address
// -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- 
#define AT91C_TWI_CLDIV       ((unsigned int) 0xFF <<  0) // (TWI) Clock Low Divider
#define AT91C_TWI_CHDIV       ((unsigned int) 0xFF <<  8) // (TWI) Clock High Divider
#define AT91C_TWI_CKDIV       ((unsigned int) 0x7 << 16) // (TWI) Clock Divider
// -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- 
#define AT91C_TWI_TXCOMP_SLAVE ((unsigned int) 0x1 <<  0) // (TWI) Transmission Completed
#define AT91C_TWI_TXCOMP_MASTER ((unsigned int) 0x1 <<  0) // (TWI) Transmission Completed
#define AT91C_TWI_RXRDY       ((unsigned int) 0x1 <<  1) // (TWI) Receive holding register ReaDY
#define AT91C_TWI_TXRDY_MASTER ((unsigned int) 0x1 <<  2) // (TWI) Transmit holding register ReaDY
#define AT91C_TWI_TXRDY_SLAVE ((unsigned int) 0x1 <<  2) // (TWI) Transmit holding register ReaDY
#define AT91C_TWI_SVREAD      ((unsigned int) 0x1 <<  3) // (TWI) Slave READ (used only in Slave mode)
#define AT91C_TWI_SVACC       ((unsigned int) 0x1 <<  4) // (TWI) Slave ACCess (used only in Slave mode)
#define AT91C_TWI_GACC        ((unsigned int) 0x1 <<  5) // (TWI) General Call ACcess (used only in Slave mode)
#define AT91C_TWI_OVRE        ((unsigned int) 0x1 <<  6) // (TWI) Overrun Error (used only in Master and Multi-master mode)
#define AT91C_TWI_NACK_SLAVE  ((unsigned int) 0x1 <<  8) // (TWI) Not Acknowledged
#define AT91C_TWI_NACK_MASTER ((unsigned int) 0x1 <<  8) // (TWI) Not Acknowledged
#define AT91C_TWI_ARBLST_MULTI_MASTER ((unsigned int) 0x1 <<  9) // (TWI) Arbitration Lost (used only in Multimaster mode)
#define AT91C_TWI_SCLWS       ((unsigned int) 0x1 << 10) // (TWI) Clock Wait State (used only in Slave mode)
#define AT91C_TWI_EOSACC      ((unsigned int) 0x1 << 11) // (TWI) End Of Slave ACCess (used only in Slave mode)
#define AT91C_TWI_ENDRX       ((unsigned int) 0x1 << 12) // (TWI) End of Receiver Transfer
#define AT91C_TWI_ENDTX       ((unsigned int) 0x1 << 13) // (TWI) End of Receiver Transfer
#define AT91C_TWI_RXBUFF      ((unsigned int) 0x1 << 14) // (TWI) RXBUFF Interrupt
#define AT91C_TWI_TXBUFE      ((unsigned int) 0x1 << 15) // (TWI) TXBUFE Interrupt
// -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- 
// -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- 
// -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Usart
// *****************************************************************************
typedef struct _AT91S_USART {
	AT91_REG	 US_CR; 	// Control Register
	AT91_REG	 US_MR; 	// Mode Register
	AT91_REG	 US_IER; 	// Interrupt Enable Register
	AT91_REG	 US_IDR; 	// Interrupt Disable Register
	AT91_REG	 US_IMR; 	// Interrupt Mask Register
	AT91_REG	 US_CSR; 	// Channel Status Register
	AT91_REG	 US_RHR; 	// Receiver Holding Register
	AT91_REG	 US_THR; 	// Transmitter Holding Register
	AT91_REG	 US_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 US_RTOR; 	// Receiver Time-out Register
	AT91_REG	 US_TTGR; 	// Transmitter Time-guard Register
	AT91_REG	 Reserved0[5]; 	// 
	AT91_REG	 US_FIDI; 	// FI_DI_Ratio Register
	AT91_REG	 US_NER; 	// Nb Errors Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 US_IF; 	// IRDA_FILTER Register
	AT91_REG	 US_MAN; 	// Manchester Encoder Decoder Register
	AT91_REG	 Reserved2[38]; 	// 
	AT91_REG	 US_ADDRSIZE; 	// US ADDRSIZE REGISTER 
	AT91_REG	 US_IPNAME1; 	// US IPNAME1 REGISTER 
	AT91_REG	 US_IPNAME2; 	// US IPNAME2 REGISTER 
	AT91_REG	 US_FEATURES; 	// US FEATURES REGISTER 
	AT91_REG	 US_VER; 	// VERSION Register
	AT91_REG	 US_RPR; 	// Receive Pointer Register
	AT91_REG	 US_RCR; 	// Receive Counter Register
	AT91_REG	 US_TPR; 	// Transmit Pointer Register
	AT91_REG	 US_TCR; 	// Transmit Counter Register
	AT91_REG	 US_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 US_RNCR; 	// Receive Next Counter Register
	AT91_REG	 US_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 US_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 US_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 US_PTSR; 	// PDC Transfer Status Register
} AT91S_USART, *AT91PS_USART;

// -------- US_CR : (USART Offset: 0x0)  Control Register -------- 
#define AT91C_US_STTBRK       ((unsigned int) 0x1 <<  9) // (USART) Start Break
#define AT91C_US_STPBRK       ((unsigned int) 0x1 << 10) // (USART) Stop Break
#define AT91C_US_STTTO        ((unsigned int) 0x1 << 11) // (USART) Start Time-out
#define AT91C_US_SENDA        ((unsigned int) 0x1 << 12) // (USART) Send Address
#define AT91C_US_RSTIT        ((unsigned int) 0x1 << 13) // (USART) Reset Iterations
#define AT91C_US_RSTNACK      ((unsigned int) 0x1 << 14) // (USART) Reset Non Acknowledge
#define AT91C_US_RETTO        ((unsigned int) 0x1 << 15) // (USART) Rearm Time-out
#define AT91C_US_DTREN        ((unsigned int) 0x1 << 16) // (USART) Data Terminal ready Enable
#define AT91C_US_DTRDIS       ((unsigned int) 0x1 << 17) // (USART) Data Terminal ready Disable
#define AT91C_US_RTSEN        ((unsigned int) 0x1 << 18) // (USART) Request to Send enable
#define AT91C_US_RTSDIS       ((unsigned int) 0x1 << 19) // (USART) Request to Send Disable
// -------- US_MR : (USART Offset: 0x4)  Mode Register -------- 
#define AT91C_US_USMODE       ((unsigned int) 0xF <<  0) // (USART) Usart mode
#define 	AT91C_US_USMODE_NORMAL               ((unsigned int) 0x0) // (USART) Normal
#define 	AT91C_US_USMODE_RS485                ((unsigned int) 0x1) // (USART) RS485
#define 	AT91C_US_USMODE_HWHSH                ((unsigned int) 0x2) // (USART) Hardware Handshaking
#define 	AT91C_US_USMODE_MODEM                ((unsigned int) 0x3) // (USART) Modem
#define 	AT91C_US_USMODE_ISO7816_0            ((unsigned int) 0x4) // (USART) ISO7816 protocol: T = 0
#define 	AT91C_US_USMODE_ISO7816_1            ((unsigned int) 0x6) // (USART) ISO7816 protocol: T = 1
#define 	AT91C_US_USMODE_IRDA                 ((unsigned int) 0x8) // (USART) IrDA
#define 	AT91C_US_USMODE_SWHSH                ((unsigned int) 0xC) // (USART) Software Handshaking
#define AT91C_US_CLKS         ((unsigned int) 0x3 <<  4) // (USART) Clock Selection (Baud Rate generator Input Clock
#define 	AT91C_US_CLKS_CLOCK                ((unsigned int) 0x0 <<  4) // (USART) Clock
#define 	AT91C_US_CLKS_FDIV1                ((unsigned int) 0x1 <<  4) // (USART) fdiv1
#define 	AT91C_US_CLKS_SLOW                 ((unsigned int) 0x2 <<  4) // (USART) slow_clock (ARM)
#define 	AT91C_US_CLKS_EXT                  ((unsigned int) 0x3 <<  4) // (USART) External (SCK)
#define AT91C_US_CHRL         ((unsigned int) 0x3 <<  6) // (USART) Clock Selection (Baud Rate generator Input Clock
#define 	AT91C_US_CHRL_5_BITS               ((unsigned int) 0x0 <<  6) // (USART) Character Length: 5 bits
#define 	AT91C_US_CHRL_6_BITS               ((unsigned int) 0x1 <<  6) // (USART) Character Length: 6 bits
#define 	AT91C_US_CHRL_7_BITS               ((unsigned int) 0x2 <<  6) // (USART) Character Length: 7 bits
#define 	AT91C_US_CHRL_8_BITS               ((unsigned int) 0x3 <<  6) // (USART) Character Length: 8 bits
#define AT91C_US_SYNC         ((unsigned int) 0x1 <<  8) // (USART) Synchronous Mode Select
#define AT91C_US_NBSTOP       ((unsigned int) 0x3 << 12) // (USART) Number of Stop bits
#define 	AT91C_US_NBSTOP_1_BIT                ((unsigned int) 0x0 << 12) // (USART) 1 stop bit
#define 	AT91C_US_NBSTOP_15_BIT               ((unsigned int) 0x1 << 12) // (USART) Asynchronous (SYNC=0) 2 stop bits Synchronous (SYNC=1) 2 stop bits
#define 	AT91C_US_NBSTOP_2_BIT                ((unsigned int) 0x2 << 12) // (USART) 2 stop bits
#define AT91C_US_MSBF         ((unsigned int) 0x1 << 16) // (USART) Bit Order
#define AT91C_US_MODE9        ((unsigned int) 0x1 << 17) // (USART) 9-bit Character length
#define AT91C_US_CKLO         ((unsigned int) 0x1 << 18) // (USART) Clock Output Select
#define AT91C_US_OVER         ((unsigned int) 0x1 << 19) // (USART) Over Sampling Mode
#define AT91C_US_INACK        ((unsigned int) 0x1 << 20) // (USART) Inhibit Non Acknowledge
#define AT91C_US_DSNACK       ((unsigned int) 0x1 << 21) // (USART) Disable Successive NACK
#define AT91C_US_VAR_SYNC     ((unsigned int) 0x1 << 22) // (USART) Variable synchronization of command/data sync Start Frame Delimiter
#define AT91C_US_MAX_ITER     ((unsigned int) 0x1 << 24) // (USART) Number of Repetitions
#define AT91C_US_FILTER       ((unsigned int) 0x1 << 28) // (USART) Receive Line Filter
#define AT91C_US_MANMODE      ((unsigned int) 0x1 << 29) // (USART) Manchester Encoder/Decoder Enable
#define AT91C_US_MODSYNC      ((unsigned int) 0x1 << 30) // (USART) Manchester Synchronization mode
#define AT91C_US_ONEBIT       ((unsigned int) 0x1 << 31) // (USART) Start Frame Delimiter selector
// -------- US_IER : (USART Offset: 0x8)  Interrupt Enable Register -------- 
#define AT91C_US_RXBRK        ((unsigned int) 0x1 <<  2) // (USART) Break Received/End of Break
#define AT91C_US_TIMEOUT      ((unsigned int) 0x1 <<  8) // (USART) Receiver Time-out
#define AT91C_US_ITERATION    ((unsigned int) 0x1 << 10) // (USART) Max number of Repetitions Reached
#define AT91C_US_NACK         ((unsigned int) 0x1 << 13) // (USART) Non Acknowledge
#define AT91C_US_RIIC         ((unsigned int) 0x1 << 16) // (USART) Ring INdicator Input Change Flag
#define AT91C_US_DSRIC        ((unsigned int) 0x1 << 17) // (USART) Data Set Ready Input Change Flag
#define AT91C_US_DCDIC        ((unsigned int) 0x1 << 18) // (USART) Data Carrier Flag
#define AT91C_US_CTSIC        ((unsigned int) 0x1 << 19) // (USART) Clear To Send Input Change Flag
#define AT91C_US_MANE         ((unsigned int) 0x1 << 20) // (USART) Manchester Error Interrupt
// -------- US_IDR : (USART Offset: 0xc)  Interrupt Disable Register -------- 
// -------- US_IMR : (USART Offset: 0x10)  Interrupt Mask Register -------- 
// -------- US_CSR : (USART Offset: 0x14)  Channel Status Register -------- 
#define AT91C_US_RI           ((unsigned int) 0x1 << 20) // (USART) Image of RI Input
#define AT91C_US_DSR          ((unsigned int) 0x1 << 21) // (USART) Image of DSR Input
#define AT91C_US_DCD          ((unsigned int) 0x1 << 22) // (USART) Image of DCD Input
#define AT91C_US_CTS          ((unsigned int) 0x1 << 23) // (USART) Image of CTS Input
#define AT91C_US_MANERR       ((unsigned int) 0x1 << 24) // (USART) Manchester Error
// -------- US_MAN : (USART Offset: 0x50) Manchester Encoder Decoder Register -------- 
#define AT91C_US_TX_PL        ((unsigned int) 0xF <<  0) // (USART) Transmitter Preamble Length
#define AT91C_US_TX_PP        ((unsigned int) 0x3 <<  8) // (USART) Transmitter Preamble Pattern
#define 	AT91C_US_TX_PP_ALL_ONE              ((unsigned int) 0x0 <<  8) // (USART) ALL_ONE
#define 	AT91C_US_TX_PP_ALL_ZERO             ((unsigned int) 0x1 <<  8) // (USART) ALL_ZERO
#define 	AT91C_US_TX_PP_ZERO_ONE             ((unsigned int) 0x2 <<  8) // (USART) ZERO_ONE
#define 	AT91C_US_TX_PP_ONE_ZERO             ((unsigned int) 0x3 <<  8) // (USART) ONE_ZERO
#define AT91C_US_TX_MPOL      ((unsigned int) 0x1 << 12) // (USART) Transmitter Manchester Polarity
#define AT91C_US_RX_PL        ((unsigned int) 0xF << 16) // (USART) Receiver Preamble Length
#define AT91C_US_RX_PP        ((unsigned int) 0x3 << 24) // (USART) Receiver Preamble Pattern detected
#define 	AT91C_US_RX_PP_ALL_ONE              ((unsigned int) 0x0 << 24) // (USART) ALL_ONE
#define 	AT91C_US_RX_PP_ALL_ZERO             ((unsigned int) 0x1 << 24) // (USART) ALL_ZERO
#define 	AT91C_US_RX_PP_ZERO_ONE             ((unsigned int) 0x2 << 24) // (USART) ZERO_ONE
#define 	AT91C_US_RX_PP_ONE_ZERO             ((unsigned int) 0x3 << 24) // (USART) ONE_ZERO
#define AT91C_US_RX_MPOL      ((unsigned int) 0x1 << 28) // (USART) Receiver Manchester Polarity
#define AT91C_US_DRIFT        ((unsigned int) 0x1 << 30) // (USART) Drift compensation

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Synchronous Serial Controller Interface
// *****************************************************************************
typedef struct _AT91S_SSC {
	AT91_REG	 SSC_CR; 	// Control Register
	AT91_REG	 SSC_CMR; 	// Clock Mode Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 SSC_RCMR; 	// Receive Clock ModeRegister
	AT91_REG	 SSC_RFMR; 	// Receive Frame Mode Register
	AT91_REG	 SSC_TCMR; 	// Transmit Clock Mode Register
	AT91_REG	 SSC_TFMR; 	// Transmit Frame Mode Register
	AT91_REG	 SSC_RHR; 	// Receive Holding Register
	AT91_REG	 SSC_THR; 	// Transmit Holding Register
	AT91_REG	 Reserved1[2]; 	// 
	AT91_REG	 SSC_RSHR; 	// Receive Sync Holding Register
	AT91_REG	 SSC_TSHR; 	// Transmit Sync Holding Register
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 SSC_SR; 	// Status Register
	AT91_REG	 SSC_IER; 	// Interrupt Enable Register
	AT91_REG	 SSC_IDR; 	// Interrupt Disable Register
	AT91_REG	 SSC_IMR; 	// Interrupt Mask Register
	AT91_REG	 Reserved3[39]; 	// 
	AT91_REG	 SSC_ADDRSIZE; 	// SSC ADDRSIZE REGISTER 
	AT91_REG	 SSC_IPNAME1; 	// SSC IPNAME1 REGISTER 
	AT91_REG	 SSC_IPNAME2; 	// SSC IPNAME2 REGISTER 
	AT91_REG	 SSC_FEATURES; 	// SSC FEATURES REGISTER 
	AT91_REG	 SSC_VER; 	// Version Register
	AT91_REG	 SSC_RPR; 	// Receive Pointer Register
	AT91_REG	 SSC_RCR; 	// Receive Counter Register
	AT91_REG	 SSC_TPR; 	// Transmit Pointer Register
	AT91_REG	 SSC_TCR; 	// Transmit Counter Register
	AT91_REG	 SSC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 SSC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 SSC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 SSC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 SSC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 SSC_PTSR; 	// PDC Transfer Status Register
} AT91S_SSC, *AT91PS_SSC;

// -------- SSC_CR : (SSC Offset: 0x0) SSC Control Register -------- 
#define AT91C_SSC_RXEN        ((unsigned int) 0x1 <<  0) // (SSC) Receive Enable
#define AT91C_SSC_RXDIS       ((unsigned int) 0x1 <<  1) // (SSC) Receive Disable
#define AT91C_SSC_TXEN        ((unsigned int) 0x1 <<  8) // (SSC) Transmit Enable
#define AT91C_SSC_TXDIS       ((unsigned int) 0x1 <<  9) // (SSC) Transmit Disable
#define AT91C_SSC_SWRST       ((unsigned int) 0x1 << 15) // (SSC) Software Reset
// -------- SSC_RCMR : (SSC Offset: 0x10) SSC Receive Clock Mode Register -------- 
#define AT91C_SSC_CKS         ((unsigned int) 0x3 <<  0) // (SSC) Receive/Transmit Clock Selection
#define 	AT91C_SSC_CKS_DIV                  ((unsigned int) 0x0) // (SSC) Divided Clock
#define 	AT91C_SSC_CKS_TK                   ((unsigned int) 0x1) // (SSC) TK Clock signal
#define 	AT91C_SSC_CKS_RK                   ((unsigned int) 0x2) // (SSC) RK pin
#define AT91C_SSC_CKO         ((unsigned int) 0x7 <<  2) // (SSC) Receive/Transmit Clock Output Mode Selection
#define 	AT91C_SSC_CKO_NONE                 ((unsigned int) 0x0 <<  2) // (SSC) Receive/Transmit Clock Output Mode: None RK pin: Input-only
#define 	AT91C_SSC_CKO_CONTINOUS            ((unsigned int) 0x1 <<  2) // (SSC) Continuous Receive/Transmit Clock RK pin: Output
#define 	AT91C_SSC_CKO_DATA_TX              ((unsigned int) 0x2 <<  2) // (SSC) Receive/Transmit Clock only during data transfers RK pin: Output
#define AT91C_SSC_CKI         ((unsigned int) 0x1 <<  5) // (SSC) Receive/Transmit Clock Inversion
#define AT91C_SSC_CKG         ((unsigned int) 0x3 <<  6) // (SSC) Receive/Transmit Clock Gating Selection
#define 	AT91C_SSC_CKG_NONE                 ((unsigned int) 0x0 <<  6) // (SSC) Receive/Transmit Clock Gating: None, continuous clock
#define 	AT91C_SSC_CKG_LOW                  ((unsigned int) 0x1 <<  6) // (SSC) Receive/Transmit Clock enabled only if RF Low
#define 	AT91C_SSC_CKG_HIGH                 ((unsigned int) 0x2 <<  6) // (SSC) Receive/Transmit Clock enabled only if RF High
#define AT91C_SSC_START       ((unsigned int) 0xF <<  8) // (SSC) Receive/Transmit Start Selection
#define 	AT91C_SSC_START_CONTINOUS            ((unsigned int) 0x0 <<  8) // (SSC) Continuous, as soon as the receiver is enabled, and immediately after the end of transfer of the previous data.
#define 	AT91C_SSC_START_TX                   ((unsigned int) 0x1 <<  8) // (SSC) Transmit/Receive start
#define 	AT91C_SSC_START_LOW_RF               ((unsigned int) 0x2 <<  8) // (SSC) Detection of a low level on RF input
#define 	AT91C_SSC_START_HIGH_RF              ((unsigned int) 0x3 <<  8) // (SSC) Detection of a high level on RF input
#define 	AT91C_SSC_START_FALL_RF              ((unsigned int) 0x4 <<  8) // (SSC) Detection of a falling edge on RF input
#define 	AT91C_SSC_START_RISE_RF              ((unsigned int) 0x5 <<  8) // (SSC) Detection of a rising edge on RF input
#define 	AT91C_SSC_START_LEVEL_RF             ((unsigned int) 0x6 <<  8) // (SSC) Detection of any level change on RF input
#define 	AT91C_SSC_START_EDGE_RF              ((unsigned int) 0x7 <<  8) // (SSC) Detection of any edge on RF input
#define 	AT91C_SSC_START_0                    ((unsigned int) 0x8 <<  8) // (SSC) Compare 0
#define AT91C_SSC_STOP        ((unsigned int) 0x1 << 12) // (SSC) Receive Stop Selection
#define AT91C_SSC_STTDLY      ((unsigned int) 0xFF << 16) // (SSC) Receive/Transmit Start Delay
#define AT91C_SSC_PERIOD      ((unsigned int) 0xFF << 24) // (SSC) Receive/Transmit Period Divider Selection
// -------- SSC_RFMR : (SSC Offset: 0x14) SSC Receive Frame Mode Register -------- 
#define AT91C_SSC_DATLEN      ((unsigned int) 0x1F <<  0) // (SSC) Data Length
#define AT91C_SSC_LOOP        ((unsigned int) 0x1 <<  5) // (SSC) Loop Mode
#define AT91C_SSC_MSBF        ((unsigned int) 0x1 <<  7) // (SSC) Most Significant Bit First
#define AT91C_SSC_DATNB       ((unsigned int) 0xF <<  8) // (SSC) Data Number per Frame
#define AT91C_SSC_FSLEN       ((unsigned int) 0xF << 16) // (SSC) Receive/Transmit Frame Sync length
#define AT91C_SSC_FSOS        ((unsigned int) 0x7 << 20) // (SSC) Receive/Transmit Frame Sync Output Selection
#define 	AT91C_SSC_FSOS_NONE                 ((unsigned int) 0x0 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: None RK pin Input-only
#define 	AT91C_SSC_FSOS_NEGATIVE             ((unsigned int) 0x1 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Negative Pulse
#define 	AT91C_SSC_FSOS_POSITIVE             ((unsigned int) 0x2 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Positive Pulse
#define 	AT91C_SSC_FSOS_LOW                  ((unsigned int) 0x3 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Driver Low during data transfer
#define 	AT91C_SSC_FSOS_HIGH                 ((unsigned int) 0x4 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Driver High during data transfer
#define 	AT91C_SSC_FSOS_TOGGLE               ((unsigned int) 0x5 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Toggling at each start of data transfer
#define AT91C_SSC_FSEDGE      ((unsigned int) 0x1 << 24) // (SSC) Frame Sync Edge Detection
// -------- SSC_TCMR : (SSC Offset: 0x18) SSC Transmit Clock Mode Register -------- 
// -------- SSC_TFMR : (SSC Offset: 0x1c) SSC Transmit Frame Mode Register -------- 
#define AT91C_SSC_DATDEF      ((unsigned int) 0x1 <<  5) // (SSC) Data Default Value
#define AT91C_SSC_FSDEN       ((unsigned int) 0x1 << 23) // (SSC) Frame Sync Data Enable
// -------- SSC_SR : (SSC Offset: 0x40) SSC Status Register -------- 
#define AT91C_SSC_TXRDY       ((unsigned int) 0x1 <<  0) // (SSC) Transmit Ready
#define AT91C_SSC_TXEMPTY     ((unsigned int) 0x1 <<  1) // (SSC) Transmit Empty
#define AT91C_SSC_ENDTX       ((unsigned int) 0x1 <<  2) // (SSC) End Of Transmission
#define AT91C_SSC_TXBUFE      ((unsigned int) 0x1 <<  3) // (SSC) Transmit Buffer Empty
#define AT91C_SSC_RXRDY       ((unsigned int) 0x1 <<  4) // (SSC) Receive Ready
#define AT91C_SSC_OVRUN       ((unsigned int) 0x1 <<  5) // (SSC) Receive Overrun
#define AT91C_SSC_ENDRX       ((unsigned int) 0x1 <<  6) // (SSC) End of Reception
#define AT91C_SSC_RXBUFF      ((unsigned int) 0x1 <<  7) // (SSC) Receive Buffer Full
#define AT91C_SSC_CP0         ((unsigned int) 0x1 <<  8) // (SSC) Compare 0
#define AT91C_SSC_CP1         ((unsigned int) 0x1 <<  9) // (SSC) Compare 1
#define AT91C_SSC_TXSYN       ((unsigned int) 0x1 << 10) // (SSC) Transmit Sync
#define AT91C_SSC_RXSYN       ((unsigned int) 0x1 << 11) // (SSC) Receive Sync
#define AT91C_SSC_TXENA       ((unsigned int) 0x1 << 16) // (SSC) Transmit Enable
#define AT91C_SSC_RXENA       ((unsigned int) 0x1 << 17) // (SSC) Receive Enable
// -------- SSC_IER : (SSC Offset: 0x44) SSC Interrupt Enable Register -------- 
// -------- SSC_IDR : (SSC Offset: 0x48) SSC Interrupt Disable Register -------- 
// -------- SSC_IMR : (SSC Offset: 0x4c) SSC Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR PWMC Channel Interface
// *****************************************************************************
typedef struct _AT91S_PWMC_CH {
	AT91_REG	 PWMC_CMR; 	// Channel Mode Register
	AT91_REG	 PWMC_CDTYR; 	// Channel Duty Cycle Register
	AT91_REG	 PWMC_CPRDR; 	// Channel Period Register
	AT91_REG	 PWMC_CCNTR; 	// Channel Counter Register
	AT91_REG	 PWMC_CUPDR; 	// Channel Update Register
	AT91_REG	 PWMC_Reserved[3]; 	// Reserved
} AT91S_PWMC_CH, *AT91PS_PWMC_CH;

// -------- PWMC_CMR : (PWMC_CH Offset: 0x0) PWMC Channel Mode Register -------- 
#define AT91C_PWMC_CPRE       ((unsigned int) 0xF <<  0) // (PWMC_CH) Channel Pre-scaler : PWMC_CLKx
#define 	AT91C_PWMC_CPRE_MCK                  ((unsigned int) 0x0) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCKA                 ((unsigned int) 0xB) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCKB                 ((unsigned int) 0xC) // (PWMC_CH) 
#define AT91C_PWMC_CALG       ((unsigned int) 0x1 <<  8) // (PWMC_CH) Channel Alignment
#define AT91C_PWMC_CPOL       ((unsigned int) 0x1 <<  9) // (PWMC_CH) Channel Polarity
#define AT91C_PWMC_CPD        ((unsigned int) 0x1 << 10) // (PWMC_CH) Channel Update Period
// -------- PWMC_CDTYR : (PWMC_CH Offset: 0x4) PWMC Channel Duty Cycle Register -------- 
#define AT91C_PWMC_CDTY       ((unsigned int) 0x0 <<  0) // (PWMC_CH) Channel Duty Cycle
// -------- PWMC_CPRDR : (PWMC_CH Offset: 0x8) PWMC Channel Period Register -------- 
#define AT91C_PWMC_CPRD       ((unsigned int) 0x0 <<  0) // (PWMC_CH) Channel Period
// -------- PWMC_CCNTR : (PWMC_CH Offset: 0xc) PWMC Channel Counter Register -------- 
#define AT91C_PWMC_CCNT       ((unsigned int) 0x0 <<  0) // (PWMC_CH) Channel Counter
// -------- PWMC_CUPDR : (PWMC_CH Offset: 0x10) PWMC Channel Update Register -------- 
#define AT91C_PWMC_CUPD       ((unsigned int) 0x0 <<  0) // (PWMC_CH) Channel Update

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Pulse Width Modulation Controller Interface
// *****************************************************************************
typedef struct _AT91S_PWMC {
	AT91_REG	 PWMC_MR; 	// PWMC Mode Register
	AT91_REG	 PWMC_ENA; 	// PWMC Enable Register
	AT91_REG	 PWMC_DIS; 	// PWMC Disable Register
	AT91_REG	 PWMC_SR; 	// PWMC Status Register
	AT91_REG	 PWMC_IER; 	// PWMC Interrupt Enable Register
	AT91_REG	 PWMC_IDR; 	// PWMC Interrupt Disable Register
	AT91_REG	 PWMC_IMR; 	// PWMC Interrupt Mask Register
	AT91_REG	 PWMC_ISR; 	// PWMC Interrupt Status Register
	AT91_REG	 Reserved0[55]; 	// 
	AT91_REG	 PWMC_VR; 	// PWMC Version Register
	AT91_REG	 Reserved1[64]; 	// 
	AT91S_PWMC_CH	 PWMC_CH[32]; 	// PWMC Channel
} AT91S_PWMC, *AT91PS_PWMC;

// -------- PWMC_MR : (PWMC Offset: 0x0) PWMC Mode Register -------- 
#define AT91C_PWMC_DIVA       ((unsigned int) 0xFF <<  0) // (PWMC) CLKA divide factor.
#define AT91C_PWMC_PREA       ((unsigned int) 0xF <<  8) // (PWMC) Divider Input Clock Prescaler A
#define 	AT91C_PWMC_PREA_MCK                  ((unsigned int) 0x0 <<  8) // (PWMC) 
#define AT91C_PWMC_DIVB       ((unsigned int) 0xFF << 16) // (PWMC) CLKB divide factor.
#define AT91C_PWMC_PREB       ((unsigned int) 0xF << 24) // (PWMC) Divider Input Clock Prescaler B
#define 	AT91C_PWMC_PREB_MCK                  ((unsigned int) 0x0 << 24) // (PWMC) 
// -------- PWMC_ENA : (PWMC Offset: 0x4) PWMC Enable Register -------- 
#define AT91C_PWMC_CHID0      ((unsigned int) 0x1 <<  0) // (PWMC) Channel ID 0
#define AT91C_PWMC_CHID1      ((unsigned int) 0x1 <<  1) // (PWMC) Channel ID 1
#define AT91C_PWMC_CHID2      ((unsigned int) 0x1 <<  2) // (PWMC) Channel ID 2
#define AT91C_PWMC_CHID3      ((unsigned int) 0x1 <<  3) // (PWMC) Channel ID 3
#define AT91C_PWMC_CHID4      ((unsigned int) 0x1 <<  4) // (PWMC) Channel ID 4
#define AT91C_PWMC_CHID5      ((unsigned int) 0x1 <<  5) // (PWMC) Channel ID 5
#define AT91C_PWMC_CHID6      ((unsigned int) 0x1 <<  6) // (PWMC) Channel ID 6
#define AT91C_PWMC_CHID7      ((unsigned int) 0x1 <<  7) // (PWMC) Channel ID 7
// -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- 
// -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- 
// -------- PWMC_IER : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- 
// -------- PWMC_IDR : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- 
// -------- PWMC_IMR : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- 
// -------- PWMC_ISR : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Serial Parallel Interface
// *****************************************************************************
typedef struct _AT91S_SPI {
	AT91_REG	 SPI_CR; 	// Control Register
	AT91_REG	 SPI_MR; 	// Mode Register
	AT91_REG	 SPI_RDR; 	// Receive Data Register
	AT91_REG	 SPI_TDR; 	// Transmit Data Register
	AT91_REG	 SPI_SR; 	// Status Register
	AT91_REG	 SPI_IER; 	// Interrupt Enable Register
	AT91_REG	 SPI_IDR; 	// Interrupt Disable Register
	AT91_REG	 SPI_IMR; 	// Interrupt Mask Register
	AT91_REG	 Reserved0[4]; 	// 
	AT91_REG	 SPI_CSR[4]; 	// Chip Select Register
	AT91_REG	 Reserved1[48]; 	// 
	AT91_REG	 SPI_RPR; 	// Receive Pointer Register
	AT91_REG	 SPI_RCR; 	// Receive Counter Register
	AT91_REG	 SPI_TPR; 	// Transmit Pointer Register
	AT91_REG	 SPI_TCR; 	// Transmit Counter Register
	AT91_REG	 SPI_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 SPI_RNCR; 	// Receive Next Counter Register
	AT91_REG	 SPI_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 SPI_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 SPI_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 SPI_PTSR; 	// PDC Transfer Status Register
} AT91S_SPI, *AT91PS_SPI;

// -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- 
#define AT91C_SPI_SPIEN       ((unsigned int) 0x1 <<  0) // (SPI) SPI Enable
#define AT91C_SPI_SPIDIS      ((unsigned int) 0x1 <<  1) // (SPI) SPI Disable
#define AT91C_SPI_SWRST       ((unsigned int) 0x1 <<  7) // (SPI) SPI Software reset
#define AT91C_SPI_LASTXFER    ((unsigned int) 0x1 << 24) // (SPI) SPI Last Transfer
// -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- 
#define AT91C_SPI_MSTR        ((unsigned int) 0x1 <<  0) // (SPI) Master/Slave Mode
#define AT91C_SPI_PS          ((unsigned int) 0x1 <<  1) // (SPI) Peripheral Select
#define 	AT91C_SPI_PS_FIXED                ((unsigned int) 0x0 <<  1) // (SPI) Fixed Peripheral Select
#define 	AT91C_SPI_PS_VARIABLE             ((unsigned int) 0x1 <<  1) // (SPI) Variable Peripheral Select
#define AT91C_SPI_PCSDEC      ((unsigned int) 0x1 <<  2) // (SPI) Chip Select Decode
#define AT91C_SPI_FDIV        ((unsigned int) 0x1 <<  3) // (SPI) Clock Selection
#define AT91C_SPI_MODFDIS     ((unsigned int) 0x1 <<  4) // (SPI) Mode Fault Detection
#define AT91C_SPI_LLB         ((unsigned int) 0x1 <<  7) // (SPI) Clock Selection
#define AT91C_SPI_PCS         ((unsigned int) 0xF << 16) // (SPI) Peripheral Chip Select
#define AT91C_SPI_DLYBCS      ((unsigned int) 0xFF << 24) // (SPI) Delay Between Chip Selects
// -------- SPI_RDR : (SPI Offset: 0x8) Receive Data Register -------- 
#define AT91C_SPI_RD          ((unsigned int) 0xFFFF <<  0) // (SPI) Receive Data
#define AT91C_SPI_RPCS        ((unsigned int) 0xF << 16) // (SPI) Peripheral Chip Select Status
// -------- SPI_TDR : (SPI Offset: 0xc) Transmit Data Register -------- 
#define AT91C_SPI_TD          ((unsigned int) 0xFFFF <<  0) // (SPI) Transmit Data
#define AT91C_SPI_TPCS        ((unsigned int) 0xF << 16) // (SPI) Peripheral Chip Select Status
// -------- SPI_SR : (SPI Offset: 0x10) Status Register -------- 
#define AT91C_SPI_RDRF        ((unsigned int) 0x1 <<  0) // (SPI) Receive Data Register Full
#define AT91C_SPI_TDRE        ((unsigned int) 0x1 <<  1) // (SPI) Transmit Data Register Empty
#define AT91C_SPI_MODF        ((unsigned int) 0x1 <<  2) // (SPI) Mode Fault Error
#define AT91C_SPI_OVRES       ((unsigned int) 0x1 <<  3) // (SPI) Overrun Error Status
#define AT91C_SPI_ENDRX       ((unsigned int) 0x1 <<  4) // (SPI) End of Receiver Transfer
#define AT91C_SPI_ENDTX       ((unsigned int) 0x1 <<  5) // (SPI) End of Receiver Transfer
#define AT91C_SPI_RXBUFF      ((unsigned int) 0x1 <<  6) // (SPI) RXBUFF Interrupt
#define AT91C_SPI_TXBUFE      ((unsigned int) 0x1 <<  7) // (SPI) TXBUFE Interrupt
#define AT91C_SPI_NSSR        ((unsigned int) 0x1 <<  8) // (SPI) NSSR Interrupt
#define AT91C_SPI_TXEMPTY     ((unsigned int) 0x1 <<  9) // (SPI) TXEMPTY Interrupt
#define AT91C_SPI_SPIENS      ((unsigned int) 0x1 << 16) // (SPI) Enable Status
// -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- 
// -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- 
// -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- 
// -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- 
#define AT91C_SPI_CPOL        ((unsigned int) 0x1 <<  0) // (SPI) Clock Polarity
#define AT91C_SPI_NCPHA       ((unsigned int) 0x1 <<  1) // (SPI) Clock Phase
#define AT91C_SPI_CSAAT       ((unsigned int) 0x1 <<  3) // (SPI) Chip Select Active After Transfer
#define AT91C_SPI_BITS        ((unsigned int) 0xF <<  4) // (SPI) Bits Per Transfer
#define 	AT91C_SPI_BITS_8                    ((unsigned int) 0x0 <<  4) // (SPI) 8 Bits Per transfer
#define 	AT91C_SPI_BITS_9                    ((unsigned int) 0x1 <<  4) // (SPI) 9 Bits Per transfer
#define 	AT91C_SPI_BITS_10                   ((unsigned int) 0x2 <<  4) // (SPI) 10 Bits Per transfer
#define 	AT91C_SPI_BITS_11                   ((unsigned int) 0x3 <<  4) // (SPI) 11 Bits Per transfer
#define 	AT91C_SPI_BITS_12                   ((unsigned int) 0x4 <<  4) // (SPI) 12 Bits Per transfer
#define 	AT91C_SPI_BITS_13                   ((unsigned int) 0x5 <<  4) // (SPI) 13 Bits Per transfer
#define 	AT91C_SPI_BITS_14                   ((unsigned int) 0x6 <<  4) // (SPI) 14 Bits Per transfer
#define 	AT91C_SPI_BITS_15                   ((unsigned int) 0x7 <<  4) // (SPI) 15 Bits Per transfer
#define 	AT91C_SPI_BITS_16                   ((unsigned int) 0x8 <<  4) // (SPI) 16 Bits Per transfer
#define AT91C_SPI_SCBR        ((unsigned int) 0xFF <<  8) // (SPI) Serial Clock Baud Rate
#define AT91C_SPI_DLYBS       ((unsigned int) 0xFF << 16) // (SPI) Delay Before SPCK
#define AT91C_SPI_DLYBCT      ((unsigned int) 0xFF << 24) // (SPI) Delay Between Consecutive Transfers

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR TSADC
// *****************************************************************************
typedef struct _AT91S_TSADC {
	AT91_REG	 TSADC_CR; 	// Control Register
	AT91_REG	 TSADC_MR; 	// Mode Register
	AT91_REG	 TSADC_TRGR; 	// Trigger Register
	AT91_REG	 TSADC_TSR; 	// Touch Screen Register
	AT91_REG	 TSADC_CHER; 	// Channel Enable Register
	AT91_REG	 TSADC_CHDR; 	// Channel Disable Register
	AT91_REG	 TSADC_CHSR; 	// Channel Status Register
	AT91_REG	 TSADC_SR; 	// Status Register
	AT91_REG	 TSADC_LCDR; 	// Last Converted Register
	AT91_REG	 TSADC_IER; 	// Interrupt Enable Register
	AT91_REG	 TSADC_IDR; 	// Interrupt Disable Register
	AT91_REG	 TSADC_IMR; 	// Interrupt Mask Register
	AT91_REG	 TSADC_CDR0; 	// Channel Data Register 0
	AT91_REG	 TSADC_CDR1; 	// Channel Data Register 1
	AT91_REG	 TSADC_CDR2; 	// Channel Data Register 2
	AT91_REG	 TSADC_CDR3; 	// Channel Data Register 3
	AT91_REG	 TSADC_CDR4; 	// Channel Data Register 4
	AT91_REG	 TSADC_CDR5; 	// Channel Data Register 5
	AT91_REG	 TSADC_CDR6; 	// Channel Data Register 6
	AT91_REG	 TSADC_CDR7; 	// Channel Data Register 7
	AT91_REG	 Reserved0[44]; 	// 
	AT91_REG	 TSADC_RPR; 	// Receive Pointer Register
	AT91_REG	 TSADC_RCR; 	// Receive Counter Register
	AT91_REG	 TSADC_TPR; 	// Transmit Pointer Register
	AT91_REG	 TSADC_TCR; 	// Transmit Counter Register
	AT91_REG	 TSADC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 TSADC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 TSADC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 TSADC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 TSADC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 TSADC_PTSR; 	// PDC Transfer Status Register
} AT91S_TSADC, *AT91PS_TSADC;

// -------- TSADC_CR : (TSADC Offset: 0x0) Control Register -------- 
#define AT91C_TSADC_SWRST     ((unsigned int) 0x1 <<  0) // (TSADC) Software Reset
#define AT91C_TSADC_START     ((unsigned int) 0x1 <<  1) // (TSADC) Start Conversion
// -------- TSADC_MR : (TSADC Offset: 0x4) Mode Register -------- 
#define AT91C_TSADC_TSAMOD    ((unsigned int) 0x3 <<  0) // (TSADC) Touch Screen ADC Mode
#define 	AT91C_TSADC_TSAMOD_ADC_ONLY_MODE        ((unsigned int) 0x0) // (TSADC) ADC Mode
#define 	AT91C_TSADC_TSAMOD_TS_ONLY_MODE         ((unsigned int) 0x1) // (TSADC) Touch Screen Only Mode
#define AT91C_TSADC_LOWRES    ((unsigned int) 0x1 <<  4) // (TSADC) ADC Resolution
#define AT91C_TSADC_SLEEP     ((unsigned int) 0x1 <<  5) // (TSADC) Sleep Mode
#define AT91C_TSADC_PENDET    ((unsigned int) 0x1 <<  6) // (TSADC) Pen Detect Selection
#define AT91C_TSADC_PRESCAL   ((unsigned int) 0xFF <<  8) // (TSADC) Prescaler Rate Selection
#define AT91C_TSADC_STARTUP   ((unsigned int) 0x7F << 16) // (TSADC) Startup Time
#define AT91C_TSADC_SHTIM     ((unsigned int) 0xF << 24) // (TSADC) Sample and Hold Time for ADC Channels
#define AT91C_TSADC_PENDBC    ((unsigned int) 0xF << 28) // (TSADC) Pen Detect Debouncing Period
// -------- TSADC_TRGR : (TSADC Offset: 0x8) Trigger Register -------- 
#define AT91C_TSADC_TRGMOD    ((unsigned int) 0x7 <<  0) // (TSADC) Trigger Mode
#define 	AT91C_TSADC_TRGMOD_NO_TRIGGER           ((unsigned int) 0x0) // (TSADC) No Trigger
#define 	AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_RE  ((unsigned int) 0x1) // (TSADC) External Trigger Rising Edge
#define 	AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_FE  ((unsigned int) 0x2) // (TSADC) External Trigger Falling Edge
#define 	AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_AE  ((unsigned int) 0x3) // (TSADC) External Trigger Any Edge
#define 	AT91C_TSADC_TRGMOD_PENDET_TRIGGER       ((unsigned int) 0x4) // (TSADC) Pen Detect Trigger (only if PENDET is set and in Touch Screen mode only)
#define 	AT91C_TSADC_TRGMOD_PERIODIC_TRIGGER     ((unsigned int) 0x5) // (TSADC) Periodic Trigger (wrt TRGPER)
#define 	AT91C_TSADC_TRGMOD_CONT_TRIGGER         ((unsigned int) 0x6) // (TSADC) Continuous Trigger
#define AT91C_TSADC_TRGPER    ((unsigned int) 0xFFFF << 16) // (TSADC) Trigger Period
// -------- TSADC_TSR : (TSADC Offset: 0xc) Touch Screen Register -------- 
#define AT91C_TSADC_TSSHTIM   ((unsigned int) 0xF << 24) // (TSADC) Sample and Hold Time for Touch Screen Channels
// -------- TSADC_CHER : (TSADC Offset: 0x10) Channel Enable Register -------- 
#define AT91C_TSADC_CHENA0    ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 Enable
#define AT91C_TSADC_CHENA1    ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 Enable
#define AT91C_TSADC_CHENA2    ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 Enable
#define AT91C_TSADC_CHENA3    ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 Enable
#define AT91C_TSADC_CHENA4    ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 Enable
#define AT91C_TSADC_CHENA5    ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 Enable
#define AT91C_TSADC_CHENA6    ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 Enable
#define AT91C_TSADC_CHENA7    ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 Enable
// -------- TSADC_CHDR : (TSADC Offset: 0x14) Channel Disable Register -------- 
#define AT91C_TSADC_CHDIS0    ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 Disable
#define AT91C_TSADC_CHDIS1    ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 Disable
#define AT91C_TSADC_CHDIS2    ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 Disable
#define AT91C_TSADC_CHDIS3    ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 Disable
#define AT91C_TSADC_CHDIS4    ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 Disable
#define AT91C_TSADC_CHDIS5    ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 Disable
#define AT91C_TSADC_CHDIS6    ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 Disable
#define AT91C_TSADC_CHDIS7    ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 Disable
// -------- TSADC_CHSR : (TSADC Offset: 0x18) Channel Status Register -------- 
#define AT91C_TSADC_CHS0      ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 Status
#define AT91C_TSADC_CHS1      ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 Status
#define AT91C_TSADC_CHS2      ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 Status
#define AT91C_TSADC_CHS3      ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 Status
#define AT91C_TSADC_CHS4      ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 Status
#define AT91C_TSADC_CHS5      ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 Status
#define AT91C_TSADC_CHS6      ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 Status
#define AT91C_TSADC_CHS7      ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 Status
// -------- TSADC_SR : (TSADC Offset: 0x1c) Status Register -------- 
#define AT91C_TSADC_EOC0      ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 End Of Conversion
#define AT91C_TSADC_EOC1      ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 End Of Conversion
#define AT91C_TSADC_EOC2      ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 End Of Conversion
#define AT91C_TSADC_EOC3      ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 End Of Conversion
#define AT91C_TSADC_EOC4      ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 End Of Conversion
#define AT91C_TSADC_EOC5      ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 End Of Conversion
#define AT91C_TSADC_EOC6      ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 End Of Conversion
#define AT91C_TSADC_EOC7      ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 End Of Conversion
#define AT91C_TSADC_OVRE0     ((unsigned int) 0x1 <<  8) // (TSADC) Channel 0 Overrun Error
#define AT91C_TSADC_OVRE1     ((unsigned int) 0x1 <<  9) // (TSADC) Channel 1 Overrun Error
#define AT91C_TSADC_OVRE2     ((unsigned int) 0x1 << 10) // (TSADC) Channel 2 Overrun Error
#define AT91C_TSADC_OVRE3     ((unsigned int) 0x1 << 11) // (TSADC) Channel 3 Overrun Error
#define AT91C_TSADC_OVRE4     ((unsigned int) 0x1 << 12) // (TSADC) Channel 4 Overrun Error
#define AT91C_TSADC_OVRE5     ((unsigned int) 0x1 << 13) // (TSADC) Channel 5 Overrun Error
#define AT91C_TSADC_OVRE6     ((unsigned int) 0x1 << 14) // (TSADC) Channel 6 Overrun Error
#define AT91C_TSADC_OVRE7     ((unsigned int) 0x1 << 15) // (TSADC) Channel 7 Overrun Error
#define AT91C_TSADC_DRDY      ((unsigned int) 0x1 << 16) // (TSADC) Data Ready
#define AT91C_TSADC_GOVRE     ((unsigned int) 0x1 << 17) // (TSADC) General Overrun Error
#define AT91C_TSADC_ENDRX     ((unsigned int) 0x1 << 18) // (TSADC) End of RX Buffer
#define AT91C_TSADC_RXBUFF    ((unsigned int) 0x1 << 19) // (TSADC) RX Buffer Full
#define AT91C_TSADC_PENCNT    ((unsigned int) 0x1 << 20) // (TSADC) Pen Contact
#define AT91C_TSADC_NOCNT     ((unsigned int) 0x1 << 21) // (TSADC) No Contact
// -------- TSADC_LCDR : (TSADC Offset: 0x20) Last Converted Data Register -------- 
#define AT91C_TSADC_LDATA     ((unsigned int) 0x3FF <<  0) // (TSADC) Last Converted Data
// -------- TSADC_IER : (TSADC Offset: 0x24) Interrupt Enable Register -------- 
#define AT91C_TSADC_IENAEOC0  ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC1  ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC2  ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC3  ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC4  ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC5  ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC6  ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAEOC7  ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 End Of Conversion Interrupt Enable
#define AT91C_TSADC_IENAOVRE0 ((unsigned int) 0x1 <<  8) // (TSADC) Channel 0 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE1 ((unsigned int) 0x1 <<  9) // (TSADC) Channel 1 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE2 ((unsigned int) 0x1 << 10) // (TSADC) Channel 2 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE3 ((unsigned int) 0x1 << 11) // (TSADC) Channel 3 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE4 ((unsigned int) 0x1 << 12) // (TSADC) Channel 4 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE5 ((unsigned int) 0x1 << 13) // (TSADC) Channel 5 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE6 ((unsigned int) 0x1 << 14) // (TSADC) Channel 6 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAOVRE7 ((unsigned int) 0x1 << 15) // (TSADC) Channel 7 Overrun Error Interrupt Enable
#define AT91C_TSADC_IENADRDY  ((unsigned int) 0x1 << 16) // (TSADC) Data Ready Interrupt Enable
#define AT91C_TSADC_IENAGOVRE ((unsigned int) 0x1 << 17) // (TSADC) General Overrun Error Interrupt Enable
#define AT91C_TSADC_IENAENDRX ((unsigned int) 0x1 << 18) // (TSADC) End of RX Buffer Interrupt Enable
#define AT91C_TSADC_IENARXBUFF ((unsigned int) 0x1 << 19) // (TSADC) RX Buffer Full Interrupt Enable
#define AT91C_TSADC_IENAPENCNT ((unsigned int) 0x1 << 20) // (TSADC) Pen Contact Interrupt Enable
#define AT91C_TSADC_IENANOCNT ((unsigned int) 0x1 << 21) // (TSADC) No Contact Interrupt Enable
// -------- TSADC_IDR : (TSADC Offset: 0x28) Interrupt Disable Register -------- 
#define AT91C_TSADC_IDISEOC0  ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC1  ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC2  ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC3  ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC4  ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC5  ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC6  ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISEOC7  ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 End Of Conversion Interrupt Disable
#define AT91C_TSADC_IDISOVRE0 ((unsigned int) 0x1 <<  8) // (TSADC) Channel 0 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE1 ((unsigned int) 0x1 <<  9) // (TSADC) Channel 1 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE2 ((unsigned int) 0x1 << 10) // (TSADC) Channel 2 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE3 ((unsigned int) 0x1 << 11) // (TSADC) Channel 3 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE4 ((unsigned int) 0x1 << 12) // (TSADC) Channel 4 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE5 ((unsigned int) 0x1 << 13) // (TSADC) Channel 5 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE6 ((unsigned int) 0x1 << 14) // (TSADC) Channel 6 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISOVRE7 ((unsigned int) 0x1 << 15) // (TSADC) Channel 7 Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISDRDY  ((unsigned int) 0x1 << 16) // (TSADC) Data Ready Interrupt Disable
#define AT91C_TSADC_IDISGOVRE ((unsigned int) 0x1 << 17) // (TSADC) General Overrun Error Interrupt Disable
#define AT91C_TSADC_IDISENDRX ((unsigned int) 0x1 << 18) // (TSADC) End of RX Buffer Interrupt Disable
#define AT91C_TSADC_IDISRXBUFF ((unsigned int) 0x1 << 19) // (TSADC) RX Buffer Full Interrupt Disable
#define AT91C_TSADC_IDISPENCNT ((unsigned int) 0x1 << 20) // (TSADC) Pen Contact Interrupt Disable
#define AT91C_TSADC_IDISNOCNT ((unsigned int) 0x1 << 21) // (TSADC) No Contact Interrupt Disable
// -------- TSADC_IMR : (TSADC Offset: 0x2c) Interrupt Mask Register -------- 
#define AT91C_TSADC_IMSKEOC0  ((unsigned int) 0x1 <<  0) // (TSADC) Channel 0 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC1  ((unsigned int) 0x1 <<  1) // (TSADC) Channel 1 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC2  ((unsigned int) 0x1 <<  2) // (TSADC) Channel 2 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC3  ((unsigned int) 0x1 <<  3) // (TSADC) Channel 3 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC4  ((unsigned int) 0x1 <<  4) // (TSADC) Channel 4 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC5  ((unsigned int) 0x1 <<  5) // (TSADC) Channel 5 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC6  ((unsigned int) 0x1 <<  6) // (TSADC) Channel 6 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKEOC7  ((unsigned int) 0x1 <<  7) // (TSADC) Channel 7 End Of Conversion Interrupt Mask
#define AT91C_TSADC_IMSKOVRE0 ((unsigned int) 0x1 <<  8) // (TSADC) Channel 0 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE1 ((unsigned int) 0x1 <<  9) // (TSADC) Channel 1 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE2 ((unsigned int) 0x1 << 10) // (TSADC) Channel 2 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE3 ((unsigned int) 0x1 << 11) // (TSADC) Channel 3 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE4 ((unsigned int) 0x1 << 12) // (TSADC) Channel 4 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE5 ((unsigned int) 0x1 << 13) // (TSADC) Channel 5 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE6 ((unsigned int) 0x1 << 14) // (TSADC) Channel 6 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKOVRE7 ((unsigned int) 0x1 << 15) // (TSADC) Channel 7 Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKDRDY  ((unsigned int) 0x1 << 16) // (TSADC) Data Ready Interrupt Mask
#define AT91C_TSADC_IMSKGOVRE ((unsigned int) 0x1 << 17) // (TSADC) General Overrun Error Interrupt Mask
#define AT91C_TSADC_IMSKENDRX ((unsigned int) 0x1 << 18) // (TSADC) End of RX Buffer Interrupt Mask
#define AT91C_TSADC_IMSKRXBUFF ((unsigned int) 0x1 << 19) // (TSADC) RX Buffer Full Interrupt Mask
#define AT91C_TSADC_IMSKPENCNT ((unsigned int) 0x1 << 20) // (TSADC) Pen Contact Interrupt Mask
#define AT91C_TSADC_IMSKNOCNT ((unsigned int) 0x1 << 21) // (TSADC) No Contact Interrupt Mask
// -------- TSADC_CDR0 : (TSADC Offset: 0x30) Channel 0 Data Register -------- 
#define AT91C_TSADC_DATA0     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 0 Data
// -------- TSADC_CDR1 : (TSADC Offset: 0x34) Channel 1 Data Register -------- 
#define AT91C_TSADC_DATA1     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 1 Data
// -------- TSADC_CDR2 : (TSADC Offset: 0x38) Channel 2 Data Register -------- 
#define AT91C_TSADC_DATA2     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 2 Data
// -------- TSADC_CDR3 : (TSADC Offset: 0x3c) Channel 3 Data Register -------- 
#define AT91C_TSADC_DATA3     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 3 Data
// -------- TSADC_CDR4 : (TSADC Offset: 0x40) Channel 4 Data Register -------- 
#define AT91C_TSADC_DATA4     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 4 Data
// -------- TSADC_CDR5 : (TSADC Offset: 0x44) Channel 5 Data Register -------- 
#define AT91C_TSADC_DATA5     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 5 Data
// -------- TSADC_CDR6 : (TSADC Offset: 0x48) Channel 6 Data Register -------- 
#define AT91C_TSADC_DATA6     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 6 Data
// -------- TSADC_CDR7 : (TSADC Offset: 0x4c) Channel 7 Data Register -------- 
#define AT91C_TSADC_DATA7     ((unsigned int) 0x3FF <<  0) // (TSADC) Channel 7 Data

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS Enpoint FIFO data register
// *****************************************************************************
typedef struct _AT91S_UDPHS_EPTFIFO {
	AT91_REG	 UDPHS_READEPT0[16384]; 	// FIFO Endpoint Data Register 0
	AT91_REG	 UDPHS_READEPT1[16384]; 	// FIFO Endpoint Data Register 1
	AT91_REG	 UDPHS_READEPT2[16384]; 	// FIFO Endpoint Data Register 2
	AT91_REG	 UDPHS_READEPT3[16384]; 	// FIFO Endpoint Data Register 3
	AT91_REG	 UDPHS_READEPT4[16384]; 	// FIFO Endpoint Data Register 4
	AT91_REG	 UDPHS_READEPT5[16384]; 	// FIFO Endpoint Data Register 5
	AT91_REG	 UDPHS_READEPT6[16384]; 	// FIFO Endpoint Data Register 6
} AT91S_UDPHS_EPTFIFO, *AT91PS_UDPHS_EPTFIFO;


// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS Endpoint struct
// *****************************************************************************
typedef struct _AT91S_UDPHS_EPT {
	AT91_REG	 UDPHS_EPTCFG; 	// UDPHS Endpoint Config Register
	AT91_REG	 UDPHS_EPTCTLENB; 	// UDPHS Endpoint Control Enable Register
	AT91_REG	 UDPHS_EPTCTLDIS; 	// UDPHS Endpoint Control Disable Register
	AT91_REG	 UDPHS_EPTCTL; 	// UDPHS Endpoint Control Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 UDPHS_EPTSETSTA; 	// UDPHS Endpoint Set Status Register
	AT91_REG	 UDPHS_EPTCLRSTA; 	// UDPHS Endpoint Clear Status Register
	AT91_REG	 UDPHS_EPTSTA; 	// UDPHS Endpoint Status Register
} AT91S_UDPHS_EPT, *AT91PS_UDPHS_EPT;

// -------- UDPHS_EPTCFG : (UDPHS_EPT Offset: 0x0) UDPHS Endpoint Config Register -------- 
#define AT91C_UDPHS_EPT_SIZE  ((unsigned int) 0x7 <<  0) // (UDPHS_EPT) Endpoint Size
#define 	AT91C_UDPHS_EPT_SIZE_8                    ((unsigned int) 0x0) // (UDPHS_EPT)    8 bytes
#define 	AT91C_UDPHS_EPT_SIZE_16                   ((unsigned int) 0x1) // (UDPHS_EPT)   16 bytes
#define 	AT91C_UDPHS_EPT_SIZE_32                   ((unsigned int) 0x2) // (UDPHS_EPT)   32 bytes
#define 	AT91C_UDPHS_EPT_SIZE_64                   ((unsigned int) 0x3) // (UDPHS_EPT)   64 bytes
#define 	AT91C_UDPHS_EPT_SIZE_128                  ((unsigned int) 0x4) // (UDPHS_EPT)  128 bytes
#define 	AT91C_UDPHS_EPT_SIZE_256                  ((unsigned int) 0x5) // (UDPHS_EPT)  256 bytes (if possible)
#define 	AT91C_UDPHS_EPT_SIZE_512                  ((unsigned int) 0x6) // (UDPHS_EPT)  512 bytes (if possible)
#define 	AT91C_UDPHS_EPT_SIZE_1024                 ((unsigned int) 0x7) // (UDPHS_EPT) 1024 bytes (if possible)
#define AT91C_UDPHS_EPT_DIR   ((unsigned int) 0x1 <<  3) // (UDPHS_EPT) Endpoint Direction 0:OUT, 1:IN
#define 	AT91C_UDPHS_EPT_DIR_OUT                  ((unsigned int) 0x0 <<  3) // (UDPHS_EPT) Direction OUT
#define 	AT91C_UDPHS_EPT_DIR_IN                   ((unsigned int) 0x1 <<  3) // (UDPHS_EPT) Direction IN
#define AT91C_UDPHS_EPT_TYPE  ((unsigned int) 0x3 <<  4) // (UDPHS_EPT) Endpoint Type
#define 	AT91C_UDPHS_EPT_TYPE_CTL_EPT              ((unsigned int) 0x0 <<  4) // (UDPHS_EPT) Control endpoint
#define 	AT91C_UDPHS_EPT_TYPE_ISO_EPT              ((unsigned int) 0x1 <<  4) // (UDPHS_EPT) Isochronous endpoint
#define 	AT91C_UDPHS_EPT_TYPE_BUL_EPT              ((unsigned int) 0x2 <<  4) // (UDPHS_EPT) Bulk endpoint
#define 	AT91C_UDPHS_EPT_TYPE_INT_EPT              ((unsigned int) 0x3 <<  4) // (UDPHS_EPT) Interrupt endpoint
#define AT91C_UDPHS_BK_NUMBER ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) Number of Banks
#define 	AT91C_UDPHS_BK_NUMBER_0                    ((unsigned int) 0x0 <<  6) // (UDPHS_EPT) Zero Bank, the EndPoint is not mapped in memory
#define 	AT91C_UDPHS_BK_NUMBER_1                    ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) One Bank (Bank0)
#define 	AT91C_UDPHS_BK_NUMBER_2                    ((unsigned int) 0x2 <<  6) // (UDPHS_EPT) Double bank (Ping-Pong : Bank0 / Bank1)
#define 	AT91C_UDPHS_BK_NUMBER_3                    ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) Triple Bank (Bank0 / Bank1 / Bank2) (if possible)
#define AT91C_UDPHS_NB_TRANS  ((unsigned int) 0x3 <<  8) // (UDPHS_EPT) Number Of Transaction per Micro-Frame (High-Bandwidth iso only)
#define AT91C_UDPHS_EPT_MAPD  ((unsigned int) 0x1 << 31) // (UDPHS_EPT) Endpoint Mapped (read only
// -------- UDPHS_EPTCTLENB : (UDPHS_EPT Offset: 0x4) UDPHS Endpoint Control Enable Register -------- 
#define AT91C_UDPHS_EPT_ENABL ((unsigned int) 0x1 <<  0) // (UDPHS_EPT) Endpoint Enable
#define AT91C_UDPHS_AUTO_VALID ((unsigned int) 0x1 <<  1) // (UDPHS_EPT) Packet Auto-Valid Enable/Disable
#define AT91C_UDPHS_INTDIS_DMA ((unsigned int) 0x1 <<  3) // (UDPHS_EPT) Endpoint Interrupts DMA Request Enable/Disable
#define AT91C_UDPHS_NYET_DIS  ((unsigned int) 0x1 <<  4) // (UDPHS_EPT) NYET Enable/Disable
#define AT91C_UDPHS_DATAX_RX  ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) DATAx Interrupt Enable/Disable
#define AT91C_UDPHS_MDATA_RX  ((unsigned int) 0x1 <<  7) // (UDPHS_EPT) MDATA Interrupt Enabled/Disable
#define AT91C_UDPHS_ERR_OVFLW ((unsigned int) 0x1 <<  8) // (UDPHS_EPT) OverFlow Error Interrupt Enable/Disable/Status
#define AT91C_UDPHS_RX_BK_RDY ((unsigned int) 0x1 <<  9) // (UDPHS_EPT) Received OUT Data
#define AT91C_UDPHS_TX_COMPLT ((unsigned int) 0x1 << 10) // (UDPHS_EPT) Transmitted IN Data Complete Interrupt Enable/Disable or Transmitted IN Data Complete (clear)
#define AT91C_UDPHS_ERR_TRANS ((unsigned int) 0x1 << 11) // (UDPHS_EPT) Transaction Error Interrupt Enable/Disable
#define AT91C_UDPHS_TX_PK_RDY ((unsigned int) 0x1 << 11) // (UDPHS_EPT) TX Packet Ready Interrupt Enable/Disable
#define AT91C_UDPHS_RX_SETUP  ((unsigned int) 0x1 << 12) // (UDPHS_EPT) Received SETUP Interrupt Enable/Disable
#define AT91C_UDPHS_ERR_FL_ISO ((unsigned int) 0x1 << 12) // (UDPHS_EPT) Error Flow Clear/Interrupt Enable/Disable
#define AT91C_UDPHS_STALL_SNT ((unsigned int) 0x1 << 13) // (UDPHS_EPT) Stall Sent Clear
#define AT91C_UDPHS_ERR_CRISO ((unsigned int) 0x1 << 13) // (UDPHS_EPT) CRC error / Error NB Trans / Interrupt Enable/Disable
#define AT91C_UDPHS_NAK_IN    ((unsigned int) 0x1 << 14) // (UDPHS_EPT) NAKIN ERROR FLUSH / Clear / Interrupt Enable/Disable
#define AT91C_UDPHS_NAK_OUT   ((unsigned int) 0x1 << 15) // (UDPHS_EPT) NAKOUT / Clear / Interrupt Enable/Disable
#define AT91C_UDPHS_BUSY_BANK ((unsigned int) 0x1 << 18) // (UDPHS_EPT) Busy Bank Interrupt Enable/Disable
#define AT91C_UDPHS_SHRT_PCKT ((unsigned int) 0x1 << 31) // (UDPHS_EPT) Short Packet / Interrupt Enable/Disable
// -------- UDPHS_EPTCTLDIS : (UDPHS_EPT Offset: 0x8) UDPHS Endpoint Control Disable Register -------- 
#define AT91C_UDPHS_EPT_DISABL ((unsigned int) 0x1 <<  0) // (UDPHS_EPT) Endpoint Disable
// -------- UDPHS_EPTCTL : (UDPHS_EPT Offset: 0xc) UDPHS Endpoint Control Register -------- 
// -------- UDPHS_EPTSETSTA : (UDPHS_EPT Offset: 0x14) UDPHS Endpoint Set Status Register -------- 
#define AT91C_UDPHS_FRCESTALL ((unsigned int) 0x1 <<  5) // (UDPHS_EPT) Stall Handshake Request Set/Clear/Status
#define AT91C_UDPHS_KILL_BANK ((unsigned int) 0x1 <<  9) // (UDPHS_EPT) KILL Bank
// -------- UDPHS_EPTCLRSTA : (UDPHS_EPT Offset: 0x18) UDPHS Endpoint Clear Status Register -------- 
#define AT91C_UDPHS_TOGGLESQ  ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) Data Toggle Clear
// -------- UDPHS_EPTSTA : (UDPHS_EPT Offset: 0x1c) UDPHS Endpoint Status Register -------- 
#define AT91C_UDPHS_TOGGLESQ_STA ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) Toggle Sequencing
#define 	AT91C_UDPHS_TOGGLESQ_STA_00                   ((unsigned int) 0x0 <<  6) // (UDPHS_EPT) Data0
#define 	AT91C_UDPHS_TOGGLESQ_STA_01                   ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) Data1
#define 	AT91C_UDPHS_TOGGLESQ_STA_10                   ((unsigned int) 0x2 <<  6) // (UDPHS_EPT) Data2 (only for High-Bandwidth Isochronous EndPoint)
#define 	AT91C_UDPHS_TOGGLESQ_STA_11                   ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) MData (only for High-Bandwidth Isochronous EndPoint)
#define AT91C_UDPHS_CONTROL_DIR ((unsigned int) 0x3 << 16) // (UDPHS_EPT) 
#define 	AT91C_UDPHS_CONTROL_DIR_00                   ((unsigned int) 0x0 << 16) // (UDPHS_EPT) Bank 0
#define 	AT91C_UDPHS_CONTROL_DIR_01                   ((unsigned int) 0x1 << 16) // (UDPHS_EPT) Bank 1
#define 	AT91C_UDPHS_CONTROL_DIR_10                   ((unsigned int) 0x2 << 16) // (UDPHS_EPT) Bank 2
#define 	AT91C_UDPHS_CONTROL_DIR_11                   ((unsigned int) 0x3 << 16) // (UDPHS_EPT) Invalid
#define AT91C_UDPHS_CURRENT_BANK ((unsigned int) 0x3 << 16) // (UDPHS_EPT) 
#define 	AT91C_UDPHS_CURRENT_BANK_00                   ((unsigned int) 0x0 << 16) // (UDPHS_EPT) Bank 0
#define 	AT91C_UDPHS_CURRENT_BANK_01                   ((unsigned int) 0x1 << 16) // (UDPHS_EPT) Bank 1
#define 	AT91C_UDPHS_CURRENT_BANK_10                   ((unsigned int) 0x2 << 16) // (UDPHS_EPT) Bank 2
#define 	AT91C_UDPHS_CURRENT_BANK_11                   ((unsigned int) 0x3 << 16) // (UDPHS_EPT) Invalid
#define AT91C_UDPHS_BUSY_BANK_STA ((unsigned int) 0x3 << 18) // (UDPHS_EPT) Busy Bank Number
#define 	AT91C_UDPHS_BUSY_BANK_STA_00                   ((unsigned int) 0x0 << 18) // (UDPHS_EPT) All banks are free
#define 	AT91C_UDPHS_BUSY_BANK_STA_01                   ((unsigned int) 0x1 << 18) // (UDPHS_EPT) 1 busy bank
#define 	AT91C_UDPHS_BUSY_BANK_STA_10                   ((unsigned int) 0x2 << 18) // (UDPHS_EPT) 2 busy banks
#define 	AT91C_UDPHS_BUSY_BANK_STA_11                   ((unsigned int) 0x3 << 18) // (UDPHS_EPT) 3 busy banks (if possible)
#define AT91C_UDPHS_BYTE_COUNT ((unsigned int) 0x7FF << 20) // (UDPHS_EPT) UDPHS Byte Count

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS DMA struct
// *****************************************************************************
typedef struct _AT91S_UDPHS_DMA {
	AT91_REG	 UDPHS_DMANXTDSC; 	// UDPHS DMA Channel Next Descriptor Address
	AT91_REG	 UDPHS_DMAADDRESS; 	// UDPHS DMA Channel Address Register
	AT91_REG	 UDPHS_DMACONTROL; 	// UDPHS DMA Channel Control Register
	AT91_REG	 UDPHS_DMASTATUS; 	// UDPHS DMA Channel Status Register
} AT91S_UDPHS_DMA, *AT91PS_UDPHS_DMA;

// -------- UDPHS_DMANXTDSC : (UDPHS_DMA Offset: 0x0) UDPHS DMA Next Descriptor Address Register -------- 
#define AT91C_UDPHS_NXT_DSC_ADD ((unsigned int) 0xFFFFFFF <<  4) // (UDPHS_DMA) next Channel Descriptor
// -------- UDPHS_DMAADDRESS : (UDPHS_DMA Offset: 0x4) UDPHS DMA Channel Address Register -------- 
#define AT91C_UDPHS_BUFF_ADD  ((unsigned int) 0x0 <<  0) // (UDPHS_DMA) starting address of a DMA Channel transfer
// -------- UDPHS_DMACONTROL : (UDPHS_DMA Offset: 0x8) UDPHS DMA Channel Control Register -------- 
#define AT91C_UDPHS_CHANN_ENB ((unsigned int) 0x1 <<  0) // (UDPHS_DMA) Channel Enabled
#define AT91C_UDPHS_LDNXT_DSC ((unsigned int) 0x1 <<  1) // (UDPHS_DMA) Load Next Channel Transfer Descriptor Enable
#define AT91C_UDPHS_END_TR_EN ((unsigned int) 0x1 <<  2) // (UDPHS_DMA) Buffer Close Input Enable
#define AT91C_UDPHS_END_B_EN  ((unsigned int) 0x1 <<  3) // (UDPHS_DMA) End of DMA Buffer Packet Validation
#define AT91C_UDPHS_END_TR_IT ((unsigned int) 0x1 <<  4) // (UDPHS_DMA) End Of Transfer Interrupt Enable
#define AT91C_UDPHS_END_BUFFIT ((unsigned int) 0x1 <<  5) // (UDPHS_DMA) End Of Channel Buffer Interrupt Enable
#define AT91C_UDPHS_DESC_LD_IT ((unsigned int) 0x1 <<  6) // (UDPHS_DMA) Descriptor Loaded Interrupt Enable
#define AT91C_UDPHS_BURST_LCK ((unsigned int) 0x1 <<  7) // (UDPHS_DMA) Burst Lock Enable
#define AT91C_UDPHS_BUFF_LENGTH ((unsigned int) 0xFFFF << 16) // (UDPHS_DMA) Buffer Byte Length (write only)
// -------- UDPHS_DMASTATUS : (UDPHS_DMA Offset: 0xc) UDPHS DMA Channelx Status Register -------- 
#define AT91C_UDPHS_CHANN_ACT ((unsigned int) 0x1 <<  1) // (UDPHS_DMA) 
#define AT91C_UDPHS_END_TR_ST ((unsigned int) 0x1 <<  4) // (UDPHS_DMA) 
#define AT91C_UDPHS_END_BF_ST ((unsigned int) 0x1 <<  5) // (UDPHS_DMA) 
#define AT91C_UDPHS_DESC_LDST ((unsigned int) 0x1 <<  6) // (UDPHS_DMA) 
#define AT91C_UDPHS_BUFF_COUNT ((unsigned int) 0xFFFF << 16) // (UDPHS_DMA) 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS High Speed Device Interface
// *****************************************************************************
typedef struct _AT91S_UDPHS {
	AT91_REG	 UDPHS_CTRL; 	// UDPHS Control Register
	AT91_REG	 UDPHS_FNUM; 	// UDPHS Frame Number Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 UDPHS_IEN; 	// UDPHS Interrupt Enable Register
	AT91_REG	 UDPHS_INTSTA; 	// UDPHS Interrupt Status Register
	AT91_REG	 UDPHS_CLRINT; 	// UDPHS Clear Interrupt Register
	AT91_REG	 UDPHS_EPTRST; 	// UDPHS Endpoints Reset Register
	AT91_REG	 Reserved1[44]; 	// 
	AT91_REG	 UDPHS_TSTSOFCNT; 	// UDPHS Test SOF Counter Register
	AT91_REG	 UDPHS_TSTCNTA; 	// UDPHS Test A Counter Register
	AT91_REG	 UDPHS_TSTCNTB; 	// UDPHS Test B Counter Register
	AT91_REG	 UDPHS_TSTMODREG; 	// UDPHS Test Mode Register
	AT91_REG	 UDPHS_TST; 	// UDPHS Test Register
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 UDPHS_RIPPADDRSIZE; 	// UDPHS PADDRSIZE Register
	AT91_REG	 UDPHS_RIPNAME1; 	// UDPHS Name1 Register
	AT91_REG	 UDPHS_RIPNAME2; 	// UDPHS Name2 Register
	AT91_REG	 UDPHS_IPFEATURES; 	// UDPHS Features Register
	AT91_REG	 UDPHS_IPVERSION; 	// UDPHS Version Register
	AT91S_UDPHS_EPT	 UDPHS_EPT[7]; 	// UDPHS Endpoint struct
	AT91_REG	 Reserved3[72]; 	// 
	AT91S_UDPHS_DMA	 UDPHS_DMA[6]; 	// UDPHS DMA channel struct (not use [0])
} AT91S_UDPHS, *AT91PS_UDPHS;

// -------- UDPHS_CTRL : (UDPHS Offset: 0x0) UDPHS Control Register -------- 
#define AT91C_UDPHS_DEV_ADDR  ((unsigned int) 0x7F <<  0) // (UDPHS) UDPHS Address
#define AT91C_UDPHS_FADDR_EN  ((unsigned int) 0x1 <<  7) // (UDPHS) Function Address Enable
#define AT91C_UDPHS_EN_UDPHS  ((unsigned int) 0x1 <<  8) // (UDPHS) UDPHS Enable
#define AT91C_UDPHS_DETACH    ((unsigned int) 0x1 <<  9) // (UDPHS) Detach Command
#define AT91C_UDPHS_REWAKEUP  ((unsigned int) 0x1 << 10) // (UDPHS) Send Remote Wake Up
#define AT91C_UDPHS_PULLD_DIS ((unsigned int) 0x1 << 11) // (UDPHS) PullDown Disable
// -------- UDPHS_FNUM : (UDPHS Offset: 0x4) UDPHS Frame Number Register -------- 
#define AT91C_UDPHS_MICRO_FRAME_NUM ((unsigned int) 0x7 <<  0) // (UDPHS) Micro Frame Number
#define AT91C_UDPHS_FRAME_NUMBER ((unsigned int) 0x7FF <<  3) // (UDPHS) Frame Number as defined in the Packet Field Formats
#define AT91C_UDPHS_FNUM_ERR  ((unsigned int) 0x1 << 31) // (UDPHS) Frame Number CRC Error
// -------- UDPHS_IEN : (UDPHS Offset: 0x10) UDPHS Interrupt Enable Register -------- 
#define AT91C_UDPHS_DET_SUSPD ((unsigned int) 0x1 <<  1) // (UDPHS) Suspend Interrupt Enable/Clear/Status
#define AT91C_UDPHS_MICRO_SOF ((unsigned int) 0x1 <<  2) // (UDPHS) Micro-SOF Interrupt Enable/Clear/Status
#define AT91C_UDPHS_IEN_SOF   ((unsigned int) 0x1 <<  3) // (UDPHS) SOF Interrupt Enable/Clear/Status
#define AT91C_UDPHS_ENDRESET  ((unsigned int) 0x1 <<  4) // (UDPHS) End Of Reset Interrupt Enable/Clear/Status
#define AT91C_UDPHS_WAKE_UP   ((unsigned int) 0x1 <<  5) // (UDPHS) Wake Up CPU Interrupt Enable/Clear/Status
#define AT91C_UDPHS_ENDOFRSM  ((unsigned int) 0x1 <<  6) // (UDPHS) End Of Resume Interrupt Enable/Clear/Status
#define AT91C_UDPHS_UPSTR_RES ((unsigned int) 0x1 <<  7) // (UDPHS) Upstream Resume Interrupt Enable/Clear/Status
#define AT91C_UDPHS_EPT_INT_0 ((unsigned int) 0x1 <<  8) // (UDPHS) Endpoint 0 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_1 ((unsigned int) 0x1 <<  9) // (UDPHS) Endpoint 1 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_2 ((unsigned int) 0x1 << 10) // (UDPHS) Endpoint 2 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_3 ((unsigned int) 0x1 << 11) // (UDPHS) Endpoint 3 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_4 ((unsigned int) 0x1 << 12) // (UDPHS) Endpoint 4 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_5 ((unsigned int) 0x1 << 13) // (UDPHS) Endpoint 5 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_6 ((unsigned int) 0x1 << 14) // (UDPHS) Endpoint 6 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_1 ((unsigned int) 0x1 << 25) // (UDPHS) DMA Channel 1 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_2 ((unsigned int) 0x1 << 26) // (UDPHS) DMA Channel 2 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_3 ((unsigned int) 0x1 << 27) // (UDPHS) DMA Channel 3 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_4 ((unsigned int) 0x1 << 28) // (UDPHS) DMA Channel 4 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_5 ((unsigned int) 0x1 << 29) // (UDPHS) DMA Channel 5 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_6 ((unsigned int) 0x1 << 30) // (UDPHS) DMA Channel 6 Interrupt Enable/Status
// -------- UDPHS_INTSTA : (UDPHS Offset: 0x14) UDPHS Interrupt Status Register -------- 
#define AT91C_UDPHS_SPEED     ((unsigned int) 0x1 <<  0) // (UDPHS) Speed Status
// -------- UDPHS_CLRINT : (UDPHS Offset: 0x18) UDPHS Clear Interrupt Register -------- 
// -------- UDPHS_EPTRST : (UDPHS Offset: 0x1c) UDPHS Endpoints Reset Register -------- 
#define AT91C_UDPHS_RST_EPT_0 ((unsigned int) 0x1 <<  0) // (UDPHS) Endpoint Reset 0
#define AT91C_UDPHS_RST_EPT_1 ((unsigned int) 0x1 <<  1) // (UDPHS) Endpoint Reset 1
#define AT91C_UDPHS_RST_EPT_2 ((unsigned int) 0x1 <<  2) // (UDPHS) Endpoint Reset 2
#define AT91C_UDPHS_RST_EPT_3 ((unsigned int) 0x1 <<  3) // (UDPHS) Endpoint Reset 3
#define AT91C_UDPHS_RST_EPT_4 ((unsigned int) 0x1 <<  4) // (UDPHS) Endpoint Reset 4
#define AT91C_UDPHS_RST_EPT_5 ((unsigned int) 0x1 <<  5) // (UDPHS) Endpoint Reset 5
#define AT91C_UDPHS_RST_EPT_6 ((unsigned int) 0x1 <<  6) // (UDPHS) Endpoint Reset 6
// -------- UDPHS_TSTSOFCNT : (UDPHS Offset: 0xd0) UDPHS Test SOF Counter Register -------- 
#define AT91C_UDPHS_SOFCNTMAX ((unsigned int) 0x3 <<  0) // (UDPHS) SOF Counter Max Value
#define AT91C_UDPHS_SOFCTLOAD ((unsigned int) 0x1 <<  7) // (UDPHS) SOF Counter Load
// -------- UDPHS_TSTCNTA : (UDPHS Offset: 0xd4) UDPHS Test A Counter Register -------- 
#define AT91C_UDPHS_CNTAMAX   ((unsigned int) 0x7FFF <<  0) // (UDPHS) A Counter Max Value
#define AT91C_UDPHS_CNTALOAD  ((unsigned int) 0x1 << 15) // (UDPHS) A Counter Load
// -------- UDPHS_TSTCNTB : (UDPHS Offset: 0xd8) UDPHS Test B Counter Register -------- 
#define AT91C_UDPHS_CNTBMAX   ((unsigned int) 0x7FFF <<  0) // (UDPHS) B Counter Max Value
#define AT91C_UDPHS_CNTBLOAD  ((unsigned int) 0x1 << 15) // (UDPHS) B Counter Load
// -------- UDPHS_TSTMODREG : (UDPHS Offset: 0xdc) UDPHS Test Mode Register -------- 
#define AT91C_UDPHS_TSTMODE   ((unsigned int) 0x1F <<  1) // (UDPHS) UDPHS Core TestModeReg
// -------- UDPHS_TST : (UDPHS Offset: 0xe0) UDPHS Test Register -------- 
#define AT91C_UDPHS_SPEED_CFG ((unsigned int) 0x3 <<  0) // (UDPHS) Speed Configuration
#define 	AT91C_UDPHS_SPEED_CFG_NM                   ((unsigned int) 0x0) // (UDPHS) Normal Mode
#define 	AT91C_UDPHS_SPEED_CFG_RS                   ((unsigned int) 0x1) // (UDPHS) Reserved
#define 	AT91C_UDPHS_SPEED_CFG_HS                   ((unsigned int) 0x2) // (UDPHS) Force High Speed
#define 	AT91C_UDPHS_SPEED_CFG_FS                   ((unsigned int) 0x3) // (UDPHS) Force Full-Speed
#define AT91C_UDPHS_TST_J     ((unsigned int) 0x1 <<  2) // (UDPHS) TestJMode
#define AT91C_UDPHS_TST_K     ((unsigned int) 0x1 <<  3) // (UDPHS) TestKMode
#define AT91C_UDPHS_TST_PKT   ((unsigned int) 0x1 <<  4) // (UDPHS) TestPacketMode
#define AT91C_UDPHS_OPMODE2   ((unsigned int) 0x1 <<  5) // (UDPHS) OpMode2
// -------- UDPHS_RIPPADDRSIZE : (UDPHS Offset: 0xec) UDPHS PADDRSIZE Register -------- 
#define AT91C_UDPHS_IPPADDRSIZE ((unsigned int) 0x0 <<  0) // (UDPHS) 2^UDPHSDEV_PADDR_SIZE
// -------- UDPHS_RIPNAME1 : (UDPHS Offset: 0xf0) UDPHS Name Register -------- 
#define AT91C_UDPHS_IPNAME1   ((unsigned int) 0x0 <<  0) // (UDPHS) ASCII string HUSB
// -------- UDPHS_RIPNAME2 : (UDPHS Offset: 0xf4) UDPHS Name Register -------- 
#define AT91C_UDPHS_IPNAME2   ((unsigned int) 0x0 <<  0) // (UDPHS) ASCII string 2DEV
// -------- UDPHS_IPFEATURES : (UDPHS Offset: 0xf8) UDPHS Features Register -------- 
#define AT91C_UDPHS_EPT_NBR_MAX ((unsigned int) 0xF <<  0) // (UDPHS) Max Number of Endpoints
#define AT91C_UDPHS_DMA_CHANNEL_NBR ((unsigned int) 0x7 <<  4) // (UDPHS) Number of DMA Channels
#define AT91C_UDPHS_DMA_B_SIZ ((unsigned int) 0x1 <<  7) // (UDPHS) DMA Buffer Size
#define AT91C_UDPHS_DMA_FIFO_WORD_DEPTH ((unsigned int) 0xF <<  8) // (UDPHS) DMA FIFO Depth in words
#define AT91C_UDPHS_FIFO_MAX_SIZE ((unsigned int) 0x7 << 12) // (UDPHS) DPRAM size
#define AT91C_UDPHS_BW_DPRAM  ((unsigned int) 0x1 << 15) // (UDPHS) DPRAM byte write capability
#define AT91C_UDPHS_DATAB16_8 ((unsigned int) 0x1 << 16) // (UDPHS) UTMI DataBus16_8
#define AT91C_UDPHS_ISO_EPT_1 ((unsigned int) 0x1 << 17) // (UDPHS) Endpoint 1 High Bandwidth Isochronous Capability
#define AT91C_UDPHS_ISO_EPT_2 ((unsigned int) 0x1 << 18) // (UDPHS) Endpoint 2 High Bandwidth Isochronous Capability
#define AT91C_UDPHS_ISO_EPT_5 ((unsigned int) 0x1 << 21) // (UDPHS) Endpoint 5 High Bandwidth Isochronous Capability
#define AT91C_UDPHS_ISO_EPT_6 ((unsigned int) 0x1 << 22) // (UDPHS) Endpoint 6 High Bandwidth Isochronous Capability
// -------- UDPHS_IPVERSION : (UDPHS Offset: 0xfc) UDPHS Version Register -------- 
#define AT91C_UDPHS_VERSION_NUM ((unsigned int) 0xFFFF <<  0) // (UDPHS) Give the IP version
#define AT91C_UDPHS_METAL_FIX_NUM ((unsigned int) 0x7 << 16) // (UDPHS) Give the number of metal fixes

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR AC97 Controller Interface
// *****************************************************************************
typedef struct _AT91S_AC97C {
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 AC97C_MR; 	// Mode Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 AC97C_ICA; 	// Input Channel AssignementRegister
	AT91_REG	 AC97C_OCA; 	// Output Channel Assignement Register
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 AC97C_CARHR; 	// Channel A Receive Holding Register
	AT91_REG	 AC97C_CATHR; 	// Channel A Transmit Holding Register
	AT91_REG	 AC97C_CASR; 	// Channel A Status Register
	AT91_REG	 AC97C_CAMR; 	// Channel A Mode Register
	AT91_REG	 AC97C_CBRHR; 	// Channel B Receive Holding Register (optional)
	AT91_REG	 AC97C_CBTHR; 	// Channel B Transmit Holding Register (optional)
	AT91_REG	 AC97C_CBSR; 	// Channel B Status Register
	AT91_REG	 AC97C_CBMR; 	// Channel B Mode Register
	AT91_REG	 AC97C_CORHR; 	// COdec Transmit Holding Register
	AT91_REG	 AC97C_COTHR; 	// COdec Transmit Holding Register
	AT91_REG	 AC97C_COSR; 	// CODEC Status Register
	AT91_REG	 AC97C_COMR; 	// CODEC Mask Status Register
	AT91_REG	 AC97C_SR; 	// Status Register
	AT91_REG	 AC97C_IER; 	// Interrupt Enable Register
	AT91_REG	 AC97C_IDR; 	// Interrupt Disable Register
	AT91_REG	 AC97C_IMR; 	// Interrupt Mask Register
	AT91_REG	 Reserved3[39]; 	// 
	AT91_REG	 AC97C_VERSION; 	// Version Register
	AT91_REG	 AC97C_RPR; 	// Receive Pointer Register
	AT91_REG	 AC97C_RCR; 	// Receive Counter Register
	AT91_REG	 AC97C_TPR; 	// Transmit Pointer Register
	AT91_REG	 AC97C_TCR; 	// Transmit Counter Register
	AT91_REG	 AC97C_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 AC97C_RNCR; 	// Receive Next Counter Register
	AT91_REG	 AC97C_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 AC97C_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 AC97C_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 AC97C_PTSR; 	// PDC Transfer Status Register
} AT91S_AC97C, *AT91PS_AC97C;

// -------- AC97C_MR : (AC97C Offset: 0x8) AC97C Mode Register -------- 
#define AT91C_AC97C_ENA       ((unsigned int) 0x1 <<  0) // (AC97C) AC97 Controller Global Enable
#define AT91C_AC97C_WRST      ((unsigned int) 0x1 <<  1) // (AC97C) Warm Reset
#define AT91C_AC97C_VRA       ((unsigned int) 0x1 <<  2) // (AC97C) Variable RAte (for Data Slots)
// -------- AC97C_ICA : (AC97C Offset: 0x10) AC97C Input Channel Assignement Register -------- 
#define AT91C_AC97C_CHID3     ((unsigned int) 0x7 <<  0) // (AC97C) Channel Id for the input slot 3
#define 	AT91C_AC97C_CHID3_NONE                 ((unsigned int) 0x0) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID3_CA                   ((unsigned int) 0x1) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID3_CB                   ((unsigned int) 0x2) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID3_CC                   ((unsigned int) 0x3) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID4     ((unsigned int) 0x7 <<  3) // (AC97C) Channel Id for the input slot 4
#define 	AT91C_AC97C_CHID4_NONE                 ((unsigned int) 0x0 <<  3) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID4_CA                   ((unsigned int) 0x1 <<  3) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID4_CB                   ((unsigned int) 0x2 <<  3) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID4_CC                   ((unsigned int) 0x3 <<  3) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID5     ((unsigned int) 0x7 <<  6) // (AC97C) Channel Id for the input slot 5
#define 	AT91C_AC97C_CHID5_NONE                 ((unsigned int) 0x0 <<  6) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID5_CA                   ((unsigned int) 0x1 <<  6) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID5_CB                   ((unsigned int) 0x2 <<  6) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID5_CC                   ((unsigned int) 0x3 <<  6) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID6     ((unsigned int) 0x7 <<  9) // (AC97C) Channel Id for the input slot 6
#define 	AT91C_AC97C_CHID6_NONE                 ((unsigned int) 0x0 <<  9) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID6_CA                   ((unsigned int) 0x1 <<  9) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID6_CB                   ((unsigned int) 0x2 <<  9) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID6_CC                   ((unsigned int) 0x3 <<  9) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID7     ((unsigned int) 0x7 << 12) // (AC97C) Channel Id for the input slot 7
#define 	AT91C_AC97C_CHID7_NONE                 ((unsigned int) 0x0 << 12) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID7_CA                   ((unsigned int) 0x1 << 12) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID7_CB                   ((unsigned int) 0x2 << 12) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID7_CC                   ((unsigned int) 0x3 << 12) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID8     ((unsigned int) 0x7 << 15) // (AC97C) Channel Id for the input slot 8
#define 	AT91C_AC97C_CHID8_NONE                 ((unsigned int) 0x0 << 15) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID8_CA                   ((unsigned int) 0x1 << 15) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID8_CB                   ((unsigned int) 0x2 << 15) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID8_CC                   ((unsigned int) 0x3 << 15) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID9     ((unsigned int) 0x7 << 18) // (AC97C) Channel Id for the input slot 9
#define 	AT91C_AC97C_CHID9_NONE                 ((unsigned int) 0x0 << 18) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID9_CA                   ((unsigned int) 0x1 << 18) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID9_CB                   ((unsigned int) 0x2 << 18) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID9_CC                   ((unsigned int) 0x3 << 18) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID10    ((unsigned int) 0x7 << 21) // (AC97C) Channel Id for the input slot 10
#define 	AT91C_AC97C_CHID10_NONE                 ((unsigned int) 0x0 << 21) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID10_CA                   ((unsigned int) 0x1 << 21) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID10_CB                   ((unsigned int) 0x2 << 21) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID10_CC                   ((unsigned int) 0x3 << 21) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID11    ((unsigned int) 0x7 << 24) // (AC97C) Channel Id for the input slot 11
#define 	AT91C_AC97C_CHID11_NONE                 ((unsigned int) 0x0 << 24) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID11_CA                   ((unsigned int) 0x1 << 24) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID11_CB                   ((unsigned int) 0x2 << 24) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID11_CC                   ((unsigned int) 0x3 << 24) // (AC97C) Channel C data will be transmitted during this slot
#define AT91C_AC97C_CHID12    ((unsigned int) 0x7 << 27) // (AC97C) Channel Id for the input slot 12
#define 	AT91C_AC97C_CHID12_NONE                 ((unsigned int) 0x0 << 27) // (AC97C) No data will be transmitted during this slot
#define 	AT91C_AC97C_CHID12_CA                   ((unsigned int) 0x1 << 27) // (AC97C) Channel A data will be transmitted during this slot
#define 	AT91C_AC97C_CHID12_CB                   ((unsigned int) 0x2 << 27) // (AC97C) Channel B data will be transmitted during this slot
#define 	AT91C_AC97C_CHID12_CC                   ((unsigned int) 0x3 << 27) // (AC97C) Channel C data will be transmitted during this slot
// -------- AC97C_OCA : (AC97C Offset: 0x14) AC97C Output Channel Assignement Register -------- 
// -------- AC97C_CARHR : (AC97C Offset: 0x20) AC97C Channel A Receive Holding Register -------- 
#define AT91C_AC97C_RDATA     ((unsigned int) 0xFFFFF <<  0) // (AC97C) Receive data
// -------- AC97C_CATHR : (AC97C Offset: 0x24) AC97C Channel A Transmit Holding Register -------- 
#define AT91C_AC97C_TDATA     ((unsigned int) 0xFFFFF <<  0) // (AC97C) Transmit data
// -------- AC97C_CASR : (AC97C Offset: 0x28) AC97C Channel A Status Register -------- 
#define AT91C_AC97C_TXRDY     ((unsigned int) 0x1 <<  0) // (AC97C) 
#define AT91C_AC97C_TXEMPTY   ((unsigned int) 0x1 <<  1) // (AC97C) 
#define AT91C_AC97C_UNRUN     ((unsigned int) 0x1 <<  2) // (AC97C) 
#define AT91C_AC97C_RXRDY     ((unsigned int) 0x1 <<  4) // (AC97C) 
#define AT91C_AC97C_OVRUN     ((unsigned int) 0x1 <<  5) // (AC97C) 
#define AT91C_AC97C_ENDTX     ((unsigned int) 0x1 << 10) // (AC97C) 
#define AT91C_AC97C_TXBUFE    ((unsigned int) 0x1 << 11) // (AC97C) 
#define AT91C_AC97C_ENDRX     ((unsigned int) 0x1 << 14) // (AC97C) 
#define AT91C_AC97C_RXBUFF    ((unsigned int) 0x1 << 15) // (AC97C) 
// -------- AC97C_CAMR : (AC97C Offset: 0x2c) AC97C Channel A Mode Register -------- 
#define AT91C_AC97C_SIZE      ((unsigned int) 0x3 << 16) // (AC97C) 
#define 	AT91C_AC97C_SIZE_20_BITS              ((unsigned int) 0x0 << 16) // (AC97C) Data size is 20 bits
#define 	AT91C_AC97C_SIZE_18_BITS              ((unsigned int) 0x1 << 16) // (AC97C) Data size is 18 bits
#define 	AT91C_AC97C_SIZE_16_BITS              ((unsigned int) 0x2 << 16) // (AC97C) Data size is 16 bits
#define 	AT91C_AC97C_SIZE_10_BITS              ((unsigned int) 0x3 << 16) // (AC97C) Data size is 10 bits
#define AT91C_AC97C_CEM       ((unsigned int) 0x1 << 18) // (AC97C) 
#define AT91C_AC97C_CEN       ((unsigned int) 0x1 << 21) // (AC97C) 
#define AT91C_AC97C_PDCEN     ((unsigned int) 0x1 << 22) // (AC97C) 
// -------- AC97C_CBRHR : (AC97C Offset: 0x30) AC97C Channel B Receive Holding Register -------- 
// -------- AC97C_CBTHR : (AC97C Offset: 0x34) AC97C Channel B Transmit Holding Register -------- 
// -------- AC97C_CBSR : (AC97C Offset: 0x38) AC97C Channel B Status Register -------- 
// -------- AC97C_CBMR : (AC97C Offset: 0x3c) AC97C Channel B Mode Register -------- 
// -------- AC97C_CORHR : (AC97C Offset: 0x40) AC97C Codec Channel Receive Holding Register -------- 
#define AT91C_AC97C_SDATA     ((unsigned int) 0xFFFF <<  0) // (AC97C) Status Data
// -------- AC97C_COTHR : (AC97C Offset: 0x44) AC97C Codec Channel Transmit Holding Register -------- 
#define AT91C_AC97C_CDATA     ((unsigned int) 0xFFFF <<  0) // (AC97C) Command Data
#define AT91C_AC97C_CADDR     ((unsigned int) 0x7F << 16) // (AC97C) COdec control register index
#define AT91C_AC97C_READ      ((unsigned int) 0x1 << 23) // (AC97C) Read/Write command
// -------- AC97C_COSR : (AC97C Offset: 0x48) AC97C CODEC Status Register -------- 
// -------- AC97C_COMR : (AC97C Offset: 0x4c) AC97C CODEC Mode Register -------- 
// -------- AC97C_SR : (AC97C Offset: 0x50) AC97C Status Register -------- 
#define AT91C_AC97C_SOF       ((unsigned int) 0x1 <<  0) // (AC97C) 
#define AT91C_AC97C_WKUP      ((unsigned int) 0x1 <<  1) // (AC97C) 
#define AT91C_AC97C_COEVT     ((unsigned int) 0x1 <<  2) // (AC97C) 
#define AT91C_AC97C_CAEVT     ((unsigned int) 0x1 <<  3) // (AC97C) 
#define AT91C_AC97C_CBEVT     ((unsigned int) 0x1 <<  4) // (AC97C) 
// -------- AC97C_IER : (AC97C Offset: 0x54) AC97C Interrupt Enable Register -------- 
// -------- AC97C_IDR : (AC97C Offset: 0x58) AC97C Interrupt Disable Register -------- 
// -------- AC97C_IMR : (AC97C Offset: 0x5c) AC97C Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR LCD Controller
// *****************************************************************************
typedef struct _AT91S_LCDC {
	AT91_REG	 LCDC_BA1; 	// DMA Base Address Register 1
	AT91_REG	 LCDC_BA2; 	// DMA Base Address Register 2
	AT91_REG	 LCDC_FRMP1; 	// DMA Frame Pointer Register 1
	AT91_REG	 LCDC_FRMP2; 	// DMA Frame Pointer Register 2
	AT91_REG	 LCDC_FRMA1; 	// DMA Frame Address Register 1
	AT91_REG	 LCDC_FRMA2; 	// DMA Frame Address Register 2
	AT91_REG	 LCDC_FRMCFG; 	// DMA Frame Configuration Register
	AT91_REG	 LCDC_DMACON; 	// DMA Control Register
	AT91_REG	 LCDC_DMA2DCFG; 	// DMA 2D addressing configuration
	AT91_REG	 Reserved0[503]; 	// 
	AT91_REG	 LCDC_LCDCON1; 	// LCD Control 1 Register
	AT91_REG	 LCDC_LCDCON2; 	// LCD Control 2 Register
	AT91_REG	 LCDC_TIM1; 	// LCD Timing Config 1 Register
	AT91_REG	 LCDC_TIM2; 	// LCD Timing Config 2 Register
	AT91_REG	 LCDC_LCDFRCFG; 	// LCD Frame Config Register
	AT91_REG	 LCDC_FIFO; 	// LCD FIFO Register
	AT91_REG	 LCDC_MVAL; 	// LCD Mode Toggle Rate Value Register
	AT91_REG	 LCDC_DP1_2; 	// Dithering Pattern DP1_2 Register
	AT91_REG	 LCDC_DP4_7; 	// Dithering Pattern DP4_7 Register
	AT91_REG	 LCDC_DP3_5; 	// Dithering Pattern DP3_5 Register
	AT91_REG	 LCDC_DP2_3; 	// Dithering Pattern DP2_3 Register
	AT91_REG	 LCDC_DP5_7; 	// Dithering Pattern DP5_7 Register
	AT91_REG	 LCDC_DP3_4; 	// Dithering Pattern DP3_4 Register
	AT91_REG	 LCDC_DP4_5; 	// Dithering Pattern DP4_5 Register
	AT91_REG	 LCDC_DP6_7; 	// Dithering Pattern DP6_7 Register
	AT91_REG	 LCDC_PWRCON; 	// Power Control Register
	AT91_REG	 LCDC_CTRSTCON; 	// Contrast Control Register
	AT91_REG	 LCDC_CTRSTVAL; 	// Contrast Value Register
	AT91_REG	 LCDC_IER; 	// Interrupt Enable Register
	AT91_REG	 LCDC_IDR; 	// Interrupt Disable Register
	AT91_REG	 LCDC_IMR; 	// Interrupt Mask Register
	AT91_REG	 LCDC_ISR; 	// Interrupt Enable Register
	AT91_REG	 LCDC_ICR; 	// Interrupt Clear Register
	AT91_REG	 LCDC_GPR; 	// General Purpose Register
	AT91_REG	 LCDC_ITR; 	// Interrupts Test Register
	AT91_REG	 LCDC_IRR; 	// Interrupts Raw Status Register
	AT91_REG	 Reserved1[230]; 	// 
	AT91_REG	 LCDC_LUT_ENTRY[256]; 	// LUT Entries Register
} AT91S_LCDC, *AT91PS_LCDC;

// -------- LCDC_FRMP1 : (LCDC Offset: 0x8) DMA Frame Pointer 1 Register -------- 
#define AT91C_LCDC_FRMPT1     ((unsigned int) 0x3FFFFF <<  0) // (LCDC) Frame Pointer Address 1
// -------- LCDC_FRMP2 : (LCDC Offset: 0xc) DMA Frame Pointer 2 Register -------- 
#define AT91C_LCDC_FRMPT2     ((unsigned int) 0x1FFFFF <<  0) // (LCDC) Frame Pointer Address 2
// -------- LCDC_FRMCFG : (LCDC Offset: 0x18) DMA Frame Config Register -------- 
#define AT91C_LCDC_FRSIZE     ((unsigned int) 0x7FFFFF <<  0) // (LCDC) FRAME SIZE
#define AT91C_LCDC_BLENGTH    ((unsigned int) 0x7F << 24) // (LCDC) BURST LENGTH
// -------- LCDC_DMACON : (LCDC Offset: 0x1c) DMA Control Register -------- 
#define AT91C_LCDC_DMAEN      ((unsigned int) 0x1 <<  0) // (LCDC) DAM Enable
#define AT91C_LCDC_DMARST     ((unsigned int) 0x1 <<  1) // (LCDC) DMA Reset (WO)
#define AT91C_LCDC_DMABUSY    ((unsigned int) 0x1 <<  2) // (LCDC) DMA Reset (WO)
#define AT91C_LCDC_DMAUPDT    ((unsigned int) 0x1 <<  3) // (LCDC) DMA Configuration Update
#define AT91C_LCDC_DMA2DEN    ((unsigned int) 0x1 <<  4) // (LCDC) 2D Addressing Enable
// -------- LCDC_DMA2DCFG : (LCDC Offset: 0x20) DMA 2D addressing configuration Register -------- 
#define AT91C_LCDC_ADDRINC    ((unsigned int) 0xFFFF <<  0) // (LCDC) Number of 32b words that the DMA must jump when going to the next line
#define AT91C_LCDC_PIXELOFF   ((unsigned int) 0x1F << 24) // (LCDC) Offset (in bits) of the first pixel of the screen in the memory word which contain it
// -------- LCDC_LCDCON1 : (LCDC Offset: 0x800) LCD Control 1 Register -------- 
#define AT91C_LCDC_BYPASS     ((unsigned int) 0x1 <<  0) // (LCDC) Bypass lcd_pccklk divider
#define AT91C_LCDC_CLKVAL     ((unsigned int) 0x1FF << 12) // (LCDC) 9-bit Divider for pixel clock frequency
#define AT91C_LCDC_LINCNT     ((unsigned int) 0x7FF << 21) // (LCDC) Line Counter (RO)
// -------- LCDC_LCDCON2 : (LCDC Offset: 0x804) LCD Control 2 Register -------- 
#define AT91C_LCDC_DISTYPE    ((unsigned int) 0x3 <<  0) // (LCDC) Display Type
#define 	AT91C_LCDC_DISTYPE_STNMONO              ((unsigned int) 0x0) // (LCDC) STN Mono
#define 	AT91C_LCDC_DISTYPE_STNCOLOR             ((unsigned int) 0x1) // (LCDC) STN Color
#define 	AT91C_LCDC_DISTYPE_TFT                  ((unsigned int) 0x2) // (LCDC) TFT
#define AT91C_LCDC_SCANMOD    ((unsigned int) 0x1 <<  2) // (LCDC) Scan Mode
#define 	AT91C_LCDC_SCANMOD_SINGLESCAN           ((unsigned int) 0x0 <<  2) // (LCDC) Single Scan
#define 	AT91C_LCDC_SCANMOD_DUALSCAN             ((unsigned int) 0x1 <<  2) // (LCDC) Dual Scan
#define AT91C_LCDC_IFWIDTH    ((unsigned int) 0x3 <<  3) // (LCDC) Interface Width
#define 	AT91C_LCDC_IFWIDTH_FOURBITSWIDTH        ((unsigned int) 0x0 <<  3) // (LCDC) 4 Bits
#define 	AT91C_LCDC_IFWIDTH_EIGTHBITSWIDTH       ((unsigned int) 0x1 <<  3) // (LCDC) 8 Bits
#define 	AT91C_LCDC_IFWIDTH_SIXTEENBITSWIDTH     ((unsigned int) 0x2 <<  3) // (LCDC) 16 Bits
#define AT91C_LCDC_PIXELSIZE  ((unsigned int) 0x7 <<  5) // (LCDC) Bits per pixel
#define 	AT91C_LCDC_PIXELSIZE_ONEBITSPERPIXEL      ((unsigned int) 0x0 <<  5) // (LCDC) 1 Bits
#define 	AT91C_LCDC_PIXELSIZE_TWOBITSPERPIXEL      ((unsigned int) 0x1 <<  5) // (LCDC) 2 Bits
#define 	AT91C_LCDC_PIXELSIZE_FOURBITSPERPIXEL     ((unsigned int) 0x2 <<  5) // (LCDC) 4 Bits
#define 	AT91C_LCDC_PIXELSIZE_EIGTHBITSPERPIXEL    ((unsigned int) 0x3 <<  5) // (LCDC) 8 Bits
#define 	AT91C_LCDC_PIXELSIZE_SIXTEENBITSPERPIXEL  ((unsigned int) 0x4 <<  5) // (LCDC) 16 Bits
#define 	AT91C_LCDC_PIXELSIZE_TWENTYFOURBITSPERPIXEL ((unsigned int) 0x5 <<  5) // (LCDC) 24 Bits
#define AT91C_LCDC_INVVD      ((unsigned int) 0x1 <<  8) // (LCDC) lcd datas polarity
#define 	AT91C_LCDC_INVVD_NORMALPOL            ((unsigned int) 0x0 <<  8) // (LCDC) Normal Polarity
#define 	AT91C_LCDC_INVVD_INVERTEDPOL          ((unsigned int) 0x1 <<  8) // (LCDC) Inverted Polarity
#define AT91C_LCDC_INVFRAME   ((unsigned int) 0x1 <<  9) // (LCDC) lcd vsync polarity
#define 	AT91C_LCDC_INVFRAME_NORMALPOL            ((unsigned int) 0x0 <<  9) // (LCDC) Normal Polarity
#define 	AT91C_LCDC_INVFRAME_INVERTEDPOL          ((unsigned int) 0x1 <<  9) // (LCDC) Inverted Polarity
#define AT91C_LCDC_INVLINE    ((unsigned int) 0x1 << 10) // (LCDC) lcd hsync polarity
#define 	AT91C_LCDC_INVLINE_NORMALPOL            ((unsigned int) 0x0 << 10) // (LCDC) Normal Polarity
#define 	AT91C_LCDC_INVLINE_INVERTEDPOL          ((unsigned int) 0x1 << 10) // (LCDC) Inverted Polarity
#define AT91C_LCDC_INVCLK     ((unsigned int) 0x1 << 11) // (LCDC) lcd pclk polarity
#define 	AT91C_LCDC_INVCLK_NORMALPOL            ((unsigned int) 0x0 << 11) // (LCDC) Normal Polarity
#define 	AT91C_LCDC_INVCLK_INVERTEDPOL          ((unsigned int) 0x1 << 11) // (LCDC) Inverted Polarity
#define AT91C_LCDC_INVDVAL    ((unsigned int) 0x1 << 12) // (LCDC) lcd dval polarity
#define 	AT91C_LCDC_INVDVAL_NORMALPOL            ((unsigned int) 0x0 << 12) // (LCDC) Normal Polarity
#define 	AT91C_LCDC_INVDVAL_INVERTEDPOL          ((unsigned int) 0x1 << 12) // (LCDC) Inverted Polarity
#define AT91C_LCDC_CLKMOD     ((unsigned int) 0x1 << 15) // (LCDC) lcd pclk Mode
#define 	AT91C_LCDC_CLKMOD_ACTIVEONLYDISP       ((unsigned int) 0x0 << 15) // (LCDC) Active during display period
#define 	AT91C_LCDC_CLKMOD_ALWAYSACTIVE         ((unsigned int) 0x1 << 15) // (LCDC) Always Active
#define AT91C_LCDC_MEMOR      ((unsigned int) 0x3 << 30) // (LCDC) Memory Ordering Format
#define 	AT91C_LCDC_MEMOR_BIGIND               ((unsigned int) 0x0 << 30) // (LCDC) Big Endian
#define 	AT91C_LCDC_MEMOR_LITTLEIND            ((unsigned int) 0x2 << 30) // (LCDC) Little Endian
// -------- LCDC_TIM1 : (LCDC Offset: 0x808) LCDC Timing Config 1 Register -------- 
#define AT91C_LCDC_VFP        ((unsigned int) 0xFF <<  0) // (LCDC) Vertical Front Porch
#define AT91C_LCDC_VBP        ((unsigned int) 0xFF <<  8) // (LCDC) Vertical Back Porch
#define AT91C_LCDC_VPW        ((unsigned int) 0x3F << 16) // (LCDC) Vertical Synchronization Pulse Width
#define AT91C_LCDC_VHDLY      ((unsigned int) 0xF << 24) // (LCDC) Vertical to Horizontal Delay
// -------- LCDC_TIM2 : (LCDC Offset: 0x80c) LCDC Timing Config 2 Register -------- 
#define AT91C_LCDC_HBP        ((unsigned int) 0xFF <<  0) // (LCDC) Horizontal Back Porch
#define AT91C_LCDC_HPW        ((unsigned int) 0x3F <<  8) // (LCDC) Horizontal Synchronization Pulse Width
#define AT91C_LCDC_HFP        ((unsigned int) 0x7FF << 21) // (LCDC) Horizontal Front Porch
// -------- LCDC_LCDFRCFG : (LCDC Offset: 0x810) LCD Frame Config Register -------- 
#define AT91C_LCDC_LINEVAL    ((unsigned int) 0x7FF <<  0) // (LCDC) Vertical Size of LCD Module
#define AT91C_LCDC_HOZVAL     ((unsigned int) 0x7FF << 21) // (LCDC) Horizontal Size of LCD Module
// -------- LCDC_FIFO : (LCDC Offset: 0x814) LCD FIFO Register -------- 
#define AT91C_LCDC_FIFOTH     ((unsigned int) 0xFFFF <<  0) // (LCDC) FIFO Threshold
// -------- LCDC_MVAL : (LCDC Offset: 0x818) LCD Mode Toggle Rate Value Register -------- 
#define AT91C_LCDC_MVALUE     ((unsigned int) 0xFF <<  0) // (LCDC) Toggle Rate Value
#define AT91C_LCDC_MMODE      ((unsigned int) 0x1 << 31) // (LCDC) Toggle Rate Sel
#define 	AT91C_LCDC_MMODE_EACHFRAME            ((unsigned int) 0x0 << 31) // (LCDC) Each Frame
#define 	AT91C_LCDC_MMODE_MVALDEFINED          ((unsigned int) 0x1 << 31) // (LCDC) Defined by MVAL
// -------- LCDC_DP1_2 : (LCDC Offset: 0x81c) Dithering Pattern 1/2 -------- 
#define AT91C_LCDC_DP1_2_FIELD ((unsigned int) 0xFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP4_7 : (LCDC Offset: 0x820) Dithering Pattern 4/7 -------- 
#define AT91C_LCDC_DP4_7_FIELD ((unsigned int) 0xFFFFFFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP3_5 : (LCDC Offset: 0x824) Dithering Pattern 3/5 -------- 
#define AT91C_LCDC_DP3_5_FIELD ((unsigned int) 0xFFFFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP2_3 : (LCDC Offset: 0x828) Dithering Pattern 2/3 -------- 
#define AT91C_LCDC_DP2_3_FIELD ((unsigned int) 0xFFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP5_7 : (LCDC Offset: 0x82c) Dithering Pattern 5/7 -------- 
#define AT91C_LCDC_DP5_7_FIELD ((unsigned int) 0xFFFFFFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP3_4 : (LCDC Offset: 0x830) Dithering Pattern 3/4 -------- 
#define AT91C_LCDC_DP3_4_FIELD ((unsigned int) 0xFFFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP4_5 : (LCDC Offset: 0x834) Dithering Pattern 4/5 -------- 
#define AT91C_LCDC_DP4_5_FIELD ((unsigned int) 0xFFFFF <<  0) // (LCDC) Ratio
// -------- LCDC_DP6_7 : (LCDC Offset: 0x838) Dithering Pattern 6/7 -------- 
#define AT91C_LCDC_DP6_7_FIELD ((unsigned int) 0xFFFFFFF <<  0) // (LCDC) Ratio
// -------- LCDC_PWRCON : (LCDC Offset: 0x83c) LCDC Power Control Register -------- 
#define AT91C_LCDC_PWR        ((unsigned int) 0x1 <<  0) // (LCDC) LCD Module Power Control
#define AT91C_LCDC_GUARDT     ((unsigned int) 0x7F <<  1) // (LCDC) Delay in Frame Period
#define AT91C_LCDC_BUSY       ((unsigned int) 0x1 << 31) // (LCDC) Read Only : 1 indicates that LCDC is busy
#define 	AT91C_LCDC_BUSY_LCDNOTBUSY           ((unsigned int) 0x0 << 31) // (LCDC) LCD is Not Busy
#define 	AT91C_LCDC_BUSY_LCDBUSY              ((unsigned int) 0x1 << 31) // (LCDC) LCD is Busy
// -------- LCDC_CTRSTCON : (LCDC Offset: 0x840) LCDC Contrast Control Register -------- 
#define AT91C_LCDC_PS         ((unsigned int) 0x3 <<  0) // (LCDC) LCD Contrast Counter Prescaler
#define 	AT91C_LCDC_PS_NOTDIVIDED           ((unsigned int) 0x0) // (LCDC) Counter Freq is System Freq.
#define 	AT91C_LCDC_PS_DIVIDEDBYTWO         ((unsigned int) 0x1) // (LCDC) Counter Freq is System Freq divided by 2.
#define 	AT91C_LCDC_PS_DIVIDEDBYFOUR        ((unsigned int) 0x2) // (LCDC) Counter Freq is System Freq divided by 4.
#define 	AT91C_LCDC_PS_DIVIDEDBYEIGHT       ((unsigned int) 0x3) // (LCDC) Counter Freq is System Freq divided by 8.
#define AT91C_LCDC_POL        ((unsigned int) 0x1 <<  2) // (LCDC) Polarity of output Pulse
#define 	AT91C_LCDC_POL_NEGATIVEPULSE        ((unsigned int) 0x0 <<  2) // (LCDC) Negative Pulse
#define 	AT91C_LCDC_POL_POSITIVEPULSE        ((unsigned int) 0x1 <<  2) // (LCDC) Positive Pulse
#define AT91C_LCDC_ENA        ((unsigned int) 0x1 <<  3) // (LCDC) PWM generator Control
#define 	AT91C_LCDC_ENA_PWMGEMDISABLED       ((unsigned int) 0x0 <<  3) // (LCDC) PWM Generator Disabled
#define 	AT91C_LCDC_ENA_PWMGEMENABLED        ((unsigned int) 0x1 <<  3) // (LCDC) PWM Generator Disabled
// -------- LCDC_CTRSTVAL : (LCDC Offset: 0x844) Contrast Value Register -------- 
#define AT91C_LCDC_CVAL       ((unsigned int) 0xFF <<  0) // (LCDC) PWM Compare Value
// -------- LCDC_IER : (LCDC Offset: 0x848) LCDC Interrupt Enable Register -------- 
#define AT91C_LCDC_LNI        ((unsigned int) 0x1 <<  0) // (LCDC) Line Interrupt
#define AT91C_LCDC_LSTLNI     ((unsigned int) 0x1 <<  1) // (LCDC) Last Line Interrupt
#define AT91C_LCDC_EOFI       ((unsigned int) 0x1 <<  2) // (LCDC) End Of Frame Interrupt
#define AT91C_LCDC_UFLWI      ((unsigned int) 0x1 <<  4) // (LCDC) FIFO Underflow Interrupt
#define AT91C_LCDC_OWRI       ((unsigned int) 0x1 <<  5) // (LCDC) Over Write Interrupt
#define AT91C_LCDC_MERI       ((unsigned int) 0x1 <<  6) // (LCDC) Memory Error  Interrupt
// -------- LCDC_IDR : (LCDC Offset: 0x84c) LCDC Interrupt Disable Register -------- 
// -------- LCDC_IMR : (LCDC Offset: 0x850) LCDC Interrupt Mask Register -------- 
// -------- LCDC_ISR : (LCDC Offset: 0x854) LCDC Interrupt Status Register -------- 
// -------- LCDC_ICR : (LCDC Offset: 0x858) LCDC Interrupt Clear Register -------- 
// -------- LCDC_GPR : (LCDC Offset: 0x85c) LCDC General Purpose Register -------- 
#define AT91C_LCDC_GPRBUS     ((unsigned int) 0xFF <<  0) // (LCDC) 8 bits available
// -------- LCDC_ITR : (LCDC Offset: 0x860) Interrupts Test Register -------- 
// -------- LCDC_IRR : (LCDC Offset: 0x864) Interrupts Raw Status Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR HDMA Channel structure
// *****************************************************************************
typedef struct _AT91S_HDMA_CH {
	AT91_REG	 HDMA_SADDR; 	// HDMA Channel Source Address Register
	AT91_REG	 HDMA_DADDR; 	// HDMA Channel Destination Address Register
	AT91_REG	 HDMA_DSCR; 	// HDMA Channel Descriptor Address Register
	AT91_REG	 HDMA_CTRLA; 	// HDMA Channel Control A Register
	AT91_REG	 HDMA_CTRLB; 	// HDMA Channel Control B Register
	AT91_REG	 HDMA_CFG; 	// HDMA Channel Configuration Register
	AT91_REG	 HDMA_SPIP; 	// HDMA Channel Source Picture in Picture Configuration Register
	AT91_REG	 HDMA_DPIP; 	// HDMA Channel Destination Picture in Picture Configuration Register
	AT91_REG	 HDMA_BDSCR; 	// HDMA Reserved
	AT91_REG	 HDMA_CADDR; 	// HDMA Reserved
} AT91S_HDMA_CH, *AT91PS_HDMA_CH;

// -------- HDMA_SADDR : (HDMA_CH Offset: 0x0)  -------- 
#define AT91C_SADDR           ((unsigned int) 0x0 <<  0) // (HDMA_CH) 
// -------- HDMA_DADDR : (HDMA_CH Offset: 0x4)  -------- 
#define AT91C_DADDR           ((unsigned int) 0x0 <<  0) // (HDMA_CH) 
// -------- HDMA_DSCR : (HDMA_CH Offset: 0x8)  -------- 
#define AT91C_HDMA_DSCR_IF    ((unsigned int) 0x3 <<  0) // (HDMA_CH) Select AHB-Lite Interface for current channel
#define 	AT91C_HDMA_DSCR_IF_0                    ((unsigned int) 0x0) // (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 0.
#define 	AT91C_HDMA_DSCR_IF_1                    ((unsigned int) 0x1) // (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 1.
#define 	AT91C_HDMA_DSCR_IF_2                    ((unsigned int) 0x2) // (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 2.
#define 	AT91C_HDMA_DSCR_IF_3                    ((unsigned int) 0x3) // (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 3.
#define AT91C_HDMA_DSCR       ((unsigned int) 0x3FFFFFFF <<  2) // (HDMA_CH) Buffer Transfer descriptor address. This address is word aligned.
// -------- HDMA_CTRLA : (HDMA_CH Offset: 0xc)  -------- 
#define AT91C_HDMA_BTSIZE     ((unsigned int) 0xFFFF <<  0) // (HDMA_CH) Buffer Transfer Size.
#define AT91C_HDMA_SCSIZE     ((unsigned int) 0x7 << 16) // (HDMA_CH) Source Chunk Transfer Size.
#define 	AT91C_HDMA_SCSIZE_1                    ((unsigned int) 0x0 << 16) // (HDMA_CH) 1.
#define 	AT91C_HDMA_SCSIZE_4                    ((unsigned int) 0x1 << 16) // (HDMA_CH) 4.
#define 	AT91C_HDMA_SCSIZE_8                    ((unsigned int) 0x2 << 16) // (HDMA_CH) 8.
#define 	AT91C_HDMA_SCSIZE_16                   ((unsigned int) 0x3 << 16) // (HDMA_CH) 16.
#define 	AT91C_HDMA_SCSIZE_32                   ((unsigned int) 0x4 << 16) // (HDMA_CH) 32.
#define 	AT91C_HDMA_SCSIZE_64                   ((unsigned int) 0x5 << 16) // (HDMA_CH) 64.
#define 	AT91C_HDMA_SCSIZE_128                  ((unsigned int) 0x6 << 16) // (HDMA_CH) 128.
#define 	AT91C_HDMA_SCSIZE_256                  ((unsigned int) 0x7 << 16) // (HDMA_CH) 256.
#define AT91C_HDMA_DCSIZE     ((unsigned int) 0x7 << 20) // (HDMA_CH) Destination Chunk Transfer Size
#define 	AT91C_HDMA_DCSIZE_1                    ((unsigned int) 0x0 << 20) // (HDMA_CH) 1.
#define 	AT91C_HDMA_DCSIZE_4                    ((unsigned int) 0x1 << 20) // (HDMA_CH) 4.
#define 	AT91C_HDMA_DCSIZE_8                    ((unsigned int) 0x2 << 20) // (HDMA_CH) 8.
#define 	AT91C_HDMA_DCSIZE_16                   ((unsigned int) 0x3 << 20) // (HDMA_CH) 16.
#define 	AT91C_HDMA_DCSIZE_32                   ((unsigned int) 0x4 << 20) // (HDMA_CH) 32.
#define 	AT91C_HDMA_DCSIZE_64                   ((unsigned int) 0x5 << 20) // (HDMA_CH) 64.
#define 	AT91C_HDMA_DCSIZE_128                  ((unsigned int) 0x6 << 20) // (HDMA_CH) 128.
#define 	AT91C_HDMA_DCSIZE_256                  ((unsigned int) 0x7 << 20) // (HDMA_CH) 256.
#define AT91C_HDMA_SRC_WIDTH  ((unsigned int) 0x3 << 24) // (HDMA_CH) Source Single Transfer Size
#define 	AT91C_HDMA_SRC_WIDTH_BYTE                 ((unsigned int) 0x0 << 24) // (HDMA_CH) BYTE.
#define 	AT91C_HDMA_SRC_WIDTH_HALFWORD             ((unsigned int) 0x1 << 24) // (HDMA_CH) HALF-WORD.
#define 	AT91C_HDMA_SRC_WIDTH_WORD                 ((unsigned int) 0x2 << 24) // (HDMA_CH) WORD.
#define AT91C_HDMA_DST_WIDTH  ((unsigned int) 0x3 << 28) // (HDMA_CH) Destination Single Transfer Size
#define 	AT91C_HDMA_DST_WIDTH_BYTE                 ((unsigned int) 0x0 << 28) // (HDMA_CH) BYTE.
#define 	AT91C_HDMA_DST_WIDTH_HALFWORD             ((unsigned int) 0x1 << 28) // (HDMA_CH) HALF-WORD.
#define 	AT91C_HDMA_DST_WIDTH_WORD                 ((unsigned int) 0x2 << 28) // (HDMA_CH) WORD.
#define AT91C_HDMA_DONE       ((unsigned int) 0x1 << 31) // (HDMA_CH) 
// -------- HDMA_CTRLB : (HDMA_CH Offset: 0x10)  -------- 
#define AT91C_HDMA_SIF        ((unsigned int) 0x3 <<  0) // (HDMA_CH) Source Interface Selection Field.
#define 	AT91C_HDMA_SIF_0                    ((unsigned int) 0x0) // (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 0.
#define 	AT91C_HDMA_SIF_1                    ((unsigned int) 0x1) // (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 1.
#define 	AT91C_HDMA_SIF_2                    ((unsigned int) 0x2) // (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 2.
#define 	AT91C_HDMA_SIF_3                    ((unsigned int) 0x3) // (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 3.
#define AT91C_HDMA_DIF        ((unsigned int) 0x3 <<  4) // (HDMA_CH) Destination Interface Selection Field.
#define 	AT91C_HDMA_DIF_0                    ((unsigned int) 0x0 <<  4) // (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 0.
#define 	AT91C_HDMA_DIF_1                    ((unsigned int) 0x1 <<  4) // (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 1.
#define 	AT91C_HDMA_DIF_2                    ((unsigned int) 0x2 <<  4) // (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 2.
#define 	AT91C_HDMA_DIF_3                    ((unsigned int) 0x3 <<  4) // (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 3.
#define AT91C_HDMA_SRC_PIP    ((unsigned int) 0x1 <<  8) // (HDMA_CH) Source Picture-in-Picture Mode
#define 	AT91C_HDMA_SRC_PIP_DISABLE              ((unsigned int) 0x0 <<  8) // (HDMA_CH) Source Picture-in-Picture mode is disabled.
#define 	AT91C_HDMA_SRC_PIP_ENABLE               ((unsigned int) 0x1 <<  8) // (HDMA_CH) Source Picture-in-Picture mode is enabled.
#define AT91C_HDMA_DST_PIP    ((unsigned int) 0x1 << 12) // (HDMA_CH) Destination Picture-in-Picture Mode
#define 	AT91C_HDMA_DST_PIP_DISABLE              ((unsigned int) 0x0 << 12) // (HDMA_CH) Destination Picture-in-Picture mode is disabled.
#define 	AT91C_HDMA_DST_PIP_ENABLE               ((unsigned int) 0x1 << 12) // (HDMA_CH) Destination Picture-in-Picture mode is enabled.
#define AT91C_HDMA_SRC_DSCR   ((unsigned int) 0x1 << 16) // (HDMA_CH) Source Buffer Descriptor Fetch operation
#define 	AT91C_HDMA_SRC_DSCR_FETCH_FROM_MEM       ((unsigned int) 0x0 << 16) // (HDMA_CH) Source address is updated when the descriptor is fetched from the memory.
#define 	AT91C_HDMA_SRC_DSCR_FETCH_DISABLE        ((unsigned int) 0x1 << 16) // (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the Source.
#define AT91C_HDMA_DST_DSCR   ((unsigned int) 0x1 << 20) // (HDMA_CH) Destination Buffer Descriptor operation
#define 	AT91C_HDMA_DST_DSCR_FETCH_FROM_MEM       ((unsigned int) 0x0 << 20) // (HDMA_CH) Destination address is updated when the descriptor is fetched from the memory.
#define 	AT91C_HDMA_DST_DSCR_FETCH_DISABLE        ((unsigned int) 0x1 << 20) // (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the destination.
#define AT91C_HDMA_FC         ((unsigned int) 0x7 << 21) // (HDMA_CH) This field defines which devices controls the size of the buffer transfer, also referred as to the Flow Controller.
#define 	AT91C_HDMA_FC_MEM2MEM              ((unsigned int) 0x0 << 21) // (HDMA_CH) Memory-to-Memory (DMA Controller).
#define 	AT91C_HDMA_FC_MEM2PER              ((unsigned int) 0x1 << 21) // (HDMA_CH) Memory-to-Peripheral (DMA Controller).
#define 	AT91C_HDMA_FC_PER2MEM              ((unsigned int) 0x2 << 21) // (HDMA_CH) Peripheral-to-Memory (DMA Controller).
#define 	AT91C_HDMA_FC_PER2PER              ((unsigned int) 0x3 << 21) // (HDMA_CH) Peripheral-to-Peripheral (DMA Controller).
#define 	AT91C_HDMA_FC_PER2MEM_PER          ((unsigned int) 0x4 << 21) // (HDMA_CH) Peripheral-to-Memory (Peripheral).
#define 	AT91C_HDMA_FC_MEM2PER_PER          ((unsigned int) 0x5 << 21) // (HDMA_CH) Memory-to-Peripheral (Peripheral).
#define 	AT91C_HDMA_FC_PER2PER_PER          ((unsigned int) 0x6 << 21) // (HDMA_CH) Peripheral-to-Peripheral (Source Peripheral).
#define AT91C_HDMA_SRC_ADDRESS_MODE ((unsigned int) 0x3 << 24) // (HDMA_CH) Type of addressing mode
#define 	AT91C_HDMA_SRC_ADDRESS_MODE_INCR                 ((unsigned int) 0x0 << 24) // (HDMA_CH) Incrementing Mode.
#define 	AT91C_HDMA_SRC_ADDRESS_MODE_DECR                 ((unsigned int) 0x1 << 24) // (HDMA_CH) Decrementing Mode.
#define 	AT91C_HDMA_SRC_ADDRESS_MODE_FIXED                ((unsigned int) 0x2 << 24) // (HDMA_CH) Fixed Mode.
#define AT91C_HDMA_DST_ADDRESS_MODE ((unsigned int) 0x3 << 28) // (HDMA_CH) Type of addressing mode
#define 	AT91C_HDMA_DST_ADDRESS_MODE_INCR                 ((unsigned int) 0x0 << 28) // (HDMA_CH) Incrementing Mode.
#define 	AT91C_HDMA_DST_ADDRESS_MODE_DECR                 ((unsigned int) 0x1 << 28) // (HDMA_CH) Decrementing Mode.
#define 	AT91C_HDMA_DST_ADDRESS_MODE_FIXED                ((unsigned int) 0x2 << 28) // (HDMA_CH) Fixed Mode.
#define AT91C_HDMA_AUTO       ((unsigned int) 0x1 << 31) // (HDMA_CH) Automatic multiple buffer transfer enable
#define 	AT91C_HDMA_AUTO_DISABLE              ((unsigned int) 0x0 << 31) // (HDMA_CH) Automatic multiple buffer transfer is disabled.
#define 	AT91C_HDMA_AUTO_ENABLE               ((unsigned int) 0x1 << 31) // (HDMA_CH) Automatic multiple buffer transfer is enabled. This enables replay mode or contiguous mode when several buffers are transferred.
// -------- HDMA_CFG : (HDMA_CH Offset: 0x14)  -------- 
#define AT91C_HDMA_SRC_PER    ((unsigned int) 0xF <<  0) // (HDMA_CH) Channel Source Request is associated with peripheral identifier coded SRC_PER handshaking interface.
#define 	AT91C_HDMA_SRC_PER_0                    ((unsigned int) 0x0) // (HDMA_CH) HW Handshaking Interface number 0.
#define 	AT91C_HDMA_SRC_PER_1                    ((unsigned int) 0x1) // (HDMA_CH) HW Handshaking Interface number 1.
#define 	AT91C_HDMA_SRC_PER_2                    ((unsigned int) 0x2) // (HDMA_CH) HW Handshaking Interface number 2.
#define 	AT91C_HDMA_SRC_PER_3                    ((unsigned int) 0x3) // (HDMA_CH) HW Handshaking Interface number 3.
#define 	AT91C_HDMA_SRC_PER_4                    ((unsigned int) 0x4) // (HDMA_CH) HW Handshaking Interface number 4.
#define 	AT91C_HDMA_SRC_PER_5                    ((unsigned int) 0x5) // (HDMA_CH) HW Handshaking Interface number 5.
#define 	AT91C_HDMA_SRC_PER_6                    ((unsigned int) 0x6) // (HDMA_CH) HW Handshaking Interface number 6.
#define 	AT91C_HDMA_SRC_PER_7                    ((unsigned int) 0x7) // (HDMA_CH) HW Handshaking Interface number 7.
#define 	AT91C_HDMA_SRC_PER_8                    ((unsigned int) 0x8) // (HDMA_CH) HW Handshaking Interface number 8.
#define 	AT91C_HDMA_SRC_PER_9                    ((unsigned int) 0x9) // (HDMA_CH) HW Handshaking Interface number 9.
#define 	AT91C_HDMA_SRC_PER_10                   ((unsigned int) 0xA) // (HDMA_CH) HW Handshaking Interface number 10.
#define 	AT91C_HDMA_SRC_PER_11                   ((unsigned int) 0xB) // (HDMA_CH) HW Handshaking Interface number 11.
#define 	AT91C_HDMA_SRC_PER_12                   ((unsigned int) 0xC) // (HDMA_CH) HW Handshaking Interface number 12.
#define 	AT91C_HDMA_SRC_PER_13                   ((unsigned int) 0xD) // (HDMA_CH) HW Handshaking Interface number 13.
#define 	AT91C_HDMA_SRC_PER_14                   ((unsigned int) 0xE) // (HDMA_CH) HW Handshaking Interface number 14.
#define 	AT91C_HDMA_SRC_PER_15                   ((unsigned int) 0xF) // (HDMA_CH) HW Handshaking Interface number 15.
#define AT91C_HDMA_DST_PER    ((unsigned int) 0xF <<  4) // (HDMA_CH) Channel Destination Request is associated with peripheral identifier coded DST_PER handshaking interface.
#define 	AT91C_HDMA_DST_PER_0                    ((unsigned int) 0x0 <<  4) // (HDMA_CH) HW Handshaking Interface number 0.
#define 	AT91C_HDMA_DST_PER_1                    ((unsigned int) 0x1 <<  4) // (HDMA_CH) HW Handshaking Interface number 1.
#define 	AT91C_HDMA_DST_PER_2                    ((unsigned int) 0x2 <<  4) // (HDMA_CH) HW Handshaking Interface number 2.
#define 	AT91C_HDMA_DST_PER_3                    ((unsigned int) 0x3 <<  4) // (HDMA_CH) HW Handshaking Interface number 3.
#define 	AT91C_HDMA_DST_PER_4                    ((unsigned int) 0x4 <<  4) // (HDMA_CH) HW Handshaking Interface number 4.
#define 	AT91C_HDMA_DST_PER_5                    ((unsigned int) 0x5 <<  4) // (HDMA_CH) HW Handshaking Interface number 5.
#define 	AT91C_HDMA_DST_PER_6                    ((unsigned int) 0x6 <<  4) // (HDMA_CH) HW Handshaking Interface number 6.
#define 	AT91C_HDMA_DST_PER_7                    ((unsigned int) 0x7 <<  4) // (HDMA_CH) HW Handshaking Interface number 7.
#define 	AT91C_HDMA_DST_PER_8                    ((unsigned int) 0x8 <<  4) // (HDMA_CH) HW Handshaking Interface number 8.
#define 	AT91C_HDMA_DST_PER_9                    ((unsigned int) 0x9 <<  4) // (HDMA_CH) HW Handshaking Interface number 9.
#define 	AT91C_HDMA_DST_PER_10                   ((unsigned int) 0xA <<  4) // (HDMA_CH) HW Handshaking Interface number 10.
#define 	AT91C_HDMA_DST_PER_11                   ((unsigned int) 0xB <<  4) // (HDMA_CH) HW Handshaking Interface number 11.
#define 	AT91C_HDMA_DST_PER_12                   ((unsigned int) 0xC <<  4) // (HDMA_CH) HW Handshaking Interface number 12.
#define 	AT91C_HDMA_DST_PER_13                   ((unsigned int) 0xD <<  4) // (HDMA_CH) HW Handshaking Interface number 13.
#define 	AT91C_HDMA_DST_PER_14                   ((unsigned int) 0xE <<  4) // (HDMA_CH) HW Handshaking Interface number 14.
#define 	AT91C_HDMA_DST_PER_15                   ((unsigned int) 0xF <<  4) // (HDMA_CH) HW Handshaking Interface number 15.
#define AT91C_HDMA_SRC_REP    ((unsigned int) 0x1 <<  8) // (HDMA_CH) Source Replay Mode
#define 	AT91C_HDMA_SRC_REP_CONTIGUOUS_ADDR      ((unsigned int) 0x0 <<  8) // (HDMA_CH) When automatic mode is activated, source address is contiguous between two buffers.
#define 	AT91C_HDMA_SRC_REP_RELOAD_ADDR          ((unsigned int) 0x1 <<  8) // (HDMA_CH) When automatic mode is activated, the source address and the control register are reloaded from previous transfer..
#define AT91C_HDMA_SRC_H2SEL  ((unsigned int) 0x1 <<  9) // (HDMA_CH) Source Handshaking Mode
#define 	AT91C_HDMA_SRC_H2SEL_SW                   ((unsigned int) 0x0 <<  9) // (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
#define 	AT91C_HDMA_SRC_H2SEL_HW                   ((unsigned int) 0x1 <<  9) // (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
#define AT91C_HDMA_DST_REP    ((unsigned int) 0x1 << 12) // (HDMA_CH) Destination Replay Mode
#define 	AT91C_HDMA_DST_REP_CONTIGUOUS_ADDR      ((unsigned int) 0x0 << 12) // (HDMA_CH) When automatic mode is activated, destination address is contiguous between two buffers.
#define 	AT91C_HDMA_DST_REP_RELOAD_ADDR          ((unsigned int) 0x1 << 12) // (HDMA_CH) When automatic mode is activated, the destination address and the control register are reloaded from previous transfer..
#define AT91C_HDMA_DST_H2SEL  ((unsigned int) 0x1 << 13) // (HDMA_CH) Destination Handshaking Mode
#define 	AT91C_HDMA_DST_H2SEL_SW                   ((unsigned int) 0x0 << 13) // (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
#define 	AT91C_HDMA_DST_H2SEL_HW                   ((unsigned int) 0x1 << 13) // (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
#define AT91C_HDMA_SOD        ((unsigned int) 0x1 << 16) // (HDMA_CH) STOP ON DONE
#define 	AT91C_HDMA_SOD_DISABLE              ((unsigned int) 0x0 << 16) // (HDMA_CH) STOP ON DONE disabled, the descriptor fetch operation ignores DONE Field of CTRLA register.
#define 	AT91C_HDMA_SOD_ENABLE               ((unsigned int) 0x1 << 16) // (HDMA_CH) STOP ON DONE activated, the DMAC module is automatically disabled if DONE FIELD is set to 1.
#define AT91C_HDMA_LOCK_IF    ((unsigned int) 0x1 << 20) // (HDMA_CH) Interface Lock
#define 	AT91C_HDMA_LOCK_IF_DISABLE              ((unsigned int) 0x0 << 20) // (HDMA_CH) Interface Lock capability is disabled.
#define 	AT91C_HDMA_LOCK_IF_ENABLE               ((unsigned int) 0x1 << 20) // (HDMA_CH) Interface Lock capability is enabled.
#define AT91C_HDMA_LOCK_B     ((unsigned int) 0x1 << 21) // (HDMA_CH) AHB Bus Lock
#define 	AT91C_HDMA_LOCK_B_DISABLE              ((unsigned int) 0x0 << 21) // (HDMA_CH) AHB Bus Locking capability is disabled.
#define 	AT91C_HDMA_LOCK_B_ENABLE               ((unsigned int) 0x1 << 21) // (HDMA_CH) AHB Bus Locking capability is enabled.
#define AT91C_HDMA_LOCK_IF_L  ((unsigned int) 0x1 << 22) // (HDMA_CH) Master Interface Arbiter Lock
#define 	AT91C_HDMA_LOCK_IF_L_CHUNK                ((unsigned int) 0x0 << 22) // (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a chunk transfer.
#define 	AT91C_HDMA_LOCK_IF_L_BUFFER               ((unsigned int) 0x1 << 22) // (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a buffer transfer.
#define AT91C_HDMA_AHB_PROT   ((unsigned int) 0x7 << 24) // (HDMA_CH) AHB Prot
#define AT91C_HDMA_FIFOCFG    ((unsigned int) 0x3 << 28) // (HDMA_CH) FIFO Request Configuration
#define 	AT91C_HDMA_FIFOCFG_LARGESTBURST         ((unsigned int) 0x0 << 28) // (HDMA_CH) The largest defined length AHB burst is performed on the destination AHB interface.
#define 	AT91C_HDMA_FIFOCFG_HALFFIFO             ((unsigned int) 0x1 << 28) // (HDMA_CH) When half fifo size is available/filled a source/destination request is serviced.
#define 	AT91C_HDMA_FIFOCFG_ENOUGHSPACE          ((unsigned int) 0x2 << 28) // (HDMA_CH) When there is enough space/data available to perfom a single AHB access then the request is serviced.
// -------- HDMA_SPIP : (HDMA_CH Offset: 0x18)  -------- 
#define AT91C_SPIP_HOLE       ((unsigned int) 0xFFFF <<  0) // (HDMA_CH) This field indicates the value to add to the address when the programmable boundary has been reached.
#define AT91C_SPIP_BOUNDARY   ((unsigned int) 0x3FF << 16) // (HDMA_CH) This field indicates the number of source transfers to perform before the automatic address increment operation.
// -------- HDMA_DPIP : (HDMA_CH Offset: 0x1c)  -------- 
#define AT91C_DPIP_HOLE       ((unsigned int) 0xFFFF <<  0) // (HDMA_CH) This field indicates the value to add to the address when the programmable boundary has been reached.
#define AT91C_DPIP_BOUNDARY   ((unsigned int) 0x3FF << 16) // (HDMA_CH) This field indicates the number of source transfers to perform before the automatic address increment operation.
// -------- HDMA_BDSCR : (HDMA_CH Offset: 0x20)  -------- 
// -------- HDMA_CADDR : (HDMA_CH Offset: 0x24)  -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR HDMA controller
// *****************************************************************************
typedef struct _AT91S_HDMA {
	AT91_REG	 HDMA_GCFG; 	// HDMA Global Configuration Register
	AT91_REG	 HDMA_EN; 	// HDMA Controller Enable Register
	AT91_REG	 HDMA_SREQ; 	// HDMA Software Single Request Register
	AT91_REG	 HDMA_CREQ; 	// HDMA Software Chunk Transfer Request Register
	AT91_REG	 HDMA_LAST; 	// HDMA Software Last Transfer Flag Register
	AT91_REG	 HDMA_SYNC; 	// HDMA Request Synchronization Register
	AT91_REG	 HDMA_EBCIER; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
	AT91_REG	 HDMA_EBCIDR; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register
	AT91_REG	 HDMA_EBCIMR; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
	AT91_REG	 HDMA_EBCISR; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
	AT91_REG	 HDMA_CHER; 	// HDMA Channel Handler Enable Register
	AT91_REG	 HDMA_CHDR; 	// HDMA Channel Handler Disable Register
	AT91_REG	 HDMA_CHSR; 	// HDMA Channel Handler Status Register
	AT91_REG	 HDMA_RSVD0; 	// HDMA Reserved
	AT91_REG	 HDMA_RSVD1; 	// HDMA Reserved
	AT91S_HDMA_CH	 HDMA_CH[8]; 	// HDMA Channel structure
} AT91S_HDMA, *AT91PS_HDMA;

// -------- HDMA_GCFG : (HDMA Offset: 0x0)  -------- 
#define AT91C_HDMA_IF0_BIGEND ((unsigned int) 0x1 <<  0) // (HDMA) AHB-Lite Interface 0 endian mode.
#define 	AT91C_HDMA_IF0_BIGEND_IS_LITTLE_ENDIAN     ((unsigned int) 0x0) // (HDMA) AHB-Lite Interface 0 is little endian.
#define 	AT91C_HDMA_IF0_BIGEND_IS_BIG_ENDIAN        ((unsigned int) 0x1) // (HDMA) AHB-Lite Interface 0 is big endian.
#define AT91C_HDMA_IF1_BIGEND ((unsigned int) 0x1 <<  1) // (HDMA) AHB-Lite Interface 1 endian mode.
#define 	AT91C_HDMA_IF1_BIGEND_IS_LITTLE_ENDIAN     ((unsigned int) 0x0 <<  1) // (HDMA) AHB-Lite Interface 1 is little endian.
#define 	AT91C_HDMA_IF1_BIGEND_IS_BIG_ENDIAN        ((unsigned int) 0x1 <<  1) // (HDMA) AHB-Lite Interface 1 is big endian.
#define AT91C_HDMA_IF2_BIGEND ((unsigned int) 0x1 <<  2) // (HDMA) AHB-Lite Interface 2 endian mode.
#define 	AT91C_HDMA_IF2_BIGEND_IS_LITTLE_ENDIAN     ((unsigned int) 0x0 <<  2) // (HDMA) AHB-Lite Interface 2 is little endian.
#define 	AT91C_HDMA_IF2_BIGEND_IS_BIG_ENDIAN        ((unsigned int) 0x1 <<  2) // (HDMA) AHB-Lite Interface 2 is big endian.
#define AT91C_HDMA_IF3_BIGEND ((unsigned int) 0x1 <<  3) // (HDMA) AHB-Lite Interface 3 endian mode.
#define 	AT91C_HDMA_IF3_BIGEND_IS_LITTLE_ENDIAN     ((unsigned int) 0x0 <<  3) // (HDMA) AHB-Lite Interface 3 is little endian.
#define 	AT91C_HDMA_IF3_BIGEND_IS_BIG_ENDIAN        ((unsigned int) 0x1 <<  3) // (HDMA) AHB-Lite Interface 3 is big endian.
#define AT91C_HDMA_ARB_CFG    ((unsigned int) 0x1 <<  4) // (HDMA) Arbiter mode.
#define 	AT91C_HDMA_ARB_CFG_FIXED                ((unsigned int) 0x0 <<  4) // (HDMA) Fixed priority arbiter.
#define 	AT91C_HDMA_ARB_CFG_ROUND_ROBIN          ((unsigned int) 0x1 <<  4) // (HDMA) Modified round robin arbiter.
// -------- HDMA_EN : (HDMA Offset: 0x4)  -------- 
#define AT91C_HDMA_ENABLE     ((unsigned int) 0x1 <<  0) // (HDMA) 
#define 	AT91C_HDMA_ENABLE_DISABLE              ((unsigned int) 0x0) // (HDMA) Disables HDMA.
#define 	AT91C_HDMA_ENABLE_ENABLE               ((unsigned int) 0x1) // (HDMA) Enables HDMA.
// -------- HDMA_SREQ : (HDMA Offset: 0x8)  -------- 
#define AT91C_HDMA_SSREQ0     ((unsigned int) 0x1 <<  0) // (HDMA) Request a source single transfer on channel 0
#define 	AT91C_HDMA_SSREQ0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ0_1                    ((unsigned int) 0x1) // (HDMA) Request a source single transfer on channel 0.
#define AT91C_HDMA_DSREQ0     ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination single transfer on channel 0
#define 	AT91C_HDMA_DSREQ0_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ0_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination single transfer on channel 0.
#define AT91C_HDMA_SSREQ1     ((unsigned int) 0x1 <<  2) // (HDMA) Request a source single transfer on channel 1
#define 	AT91C_HDMA_SSREQ1_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ1_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Request a source single transfer on channel 1.
#define AT91C_HDMA_DSREQ1     ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination single transfer on channel 1
#define 	AT91C_HDMA_DSREQ1_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ1_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination single transfer on channel 1.
#define AT91C_HDMA_SSREQ2     ((unsigned int) 0x1 <<  4) // (HDMA) Request a source single transfer on channel 2
#define 	AT91C_HDMA_SSREQ2_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ2_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Request a source single transfer on channel 2.
#define AT91C_HDMA_DSREQ2     ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination single transfer on channel 2
#define 	AT91C_HDMA_DSREQ2_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ2_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination single transfer on channel 2.
#define AT91C_HDMA_SSREQ3     ((unsigned int) 0x1 <<  6) // (HDMA) Request a source single transfer on channel 3
#define 	AT91C_HDMA_SSREQ3_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ3_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Request a source single transfer on channel 3.
#define AT91C_HDMA_DSREQ3     ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination single transfer on channel 3
#define 	AT91C_HDMA_DSREQ3_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ3_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination single transfer on channel 3.
#define AT91C_HDMA_SSREQ4     ((unsigned int) 0x1 <<  8) // (HDMA) Request a source single transfer on channel 4
#define 	AT91C_HDMA_SSREQ4_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ4_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Request a source single transfer on channel 4.
#define AT91C_HDMA_DSREQ4     ((unsigned int) 0x1 <<  9) // (HDMA) Request a destination single transfer on channel 4
#define 	AT91C_HDMA_DSREQ4_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ4_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Request a destination single transfer on channel 4.
#define AT91C_HDMA_SSREQ5     ((unsigned int) 0x1 << 10) // (HDMA) Request a source single transfer on channel 5
#define 	AT91C_HDMA_SSREQ5_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ5_1                    ((unsigned int) 0x1 << 10) // (HDMA) Request a source single transfer on channel 5.
#define AT91C_HDMA_DSREQ6     ((unsigned int) 0x1 << 11) // (HDMA) Request a destination single transfer on channel 5
#define 	AT91C_HDMA_DSREQ6_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ6_1                    ((unsigned int) 0x1 << 11) // (HDMA) Request a destination single transfer on channel 5.
#define AT91C_HDMA_SSREQ6     ((unsigned int) 0x1 << 12) // (HDMA) Request a source single transfer on channel 6
#define 	AT91C_HDMA_SSREQ6_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ6_1                    ((unsigned int) 0x1 << 12) // (HDMA) Request a source single transfer on channel 6.
#define AT91C_HDMA_SSREQ7     ((unsigned int) 0x1 << 14) // (HDMA) Request a source single transfer on channel 7
#define 	AT91C_HDMA_SSREQ7_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ7_1                    ((unsigned int) 0x1 << 14) // (HDMA) Request a source single transfer on channel 7.
#define AT91C_HDMA_DSREQ7     ((unsigned int) 0x1 << 15) // (HDMA) Request a destination single transfer on channel 7
#define 	AT91C_HDMA_DSREQ7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Request a destination single transfer on channel 7.
// -------- HDMA_CREQ : (HDMA Offset: 0xc)  -------- 
#define AT91C_HDMA_SCREQ0     ((unsigned int) 0x1 <<  0) // (HDMA) Request a source chunk transfer on channel 0
#define 	AT91C_HDMA_SCREQ0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ0_1                    ((unsigned int) 0x1) // (HDMA) Request a source chunk transfer on channel 0.
#define AT91C_HDMA_DCREQ0     ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination chunk transfer on channel 0
#define 	AT91C_HDMA_DCREQ0_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ0_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination chunk transfer on channel 0.
#define AT91C_HDMA_SCREQ1     ((unsigned int) 0x1 <<  2) // (HDMA) Request a source chunk transfer on channel 1
#define 	AT91C_HDMA_SCREQ1_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ1_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Request a source chunk transfer on channel 1.
#define AT91C_HDMA_DCREQ1     ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination chunk transfer on channel 1
#define 	AT91C_HDMA_DCREQ1_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ1_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination chunk transfer on channel 1.
#define AT91C_HDMA_SCREQ2     ((unsigned int) 0x1 <<  4) // (HDMA) Request a source chunk transfer on channel 2
#define 	AT91C_HDMA_SCREQ2_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ2_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Request a source chunk transfer on channel 2.
#define AT91C_HDMA_DCREQ2     ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination chunk transfer on channel 2
#define 	AT91C_HDMA_DCREQ2_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ2_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination chunk transfer on channel 2.
#define AT91C_HDMA_SCREQ3     ((unsigned int) 0x1 <<  6) // (HDMA) Request a source chunk transfer on channel 3
#define 	AT91C_HDMA_SCREQ3_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ3_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Request a source chunk transfer on channel 3.
#define AT91C_HDMA_DCREQ3     ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination chunk transfer on channel 3
#define 	AT91C_HDMA_DCREQ3_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ3_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination chunk transfer on channel 3.
#define AT91C_HDMA_SCREQ4     ((unsigned int) 0x1 <<  8) // (HDMA) Request a source chunk transfer on channel 4
#define 	AT91C_HDMA_SCREQ4_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ4_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Request a source chunk transfer on channel 4.
#define AT91C_HDMA_DCREQ4     ((unsigned int) 0x1 <<  9) // (HDMA) Request a destination chunk transfer on channel 4
#define 	AT91C_HDMA_DCREQ4_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ4_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Request a destination chunk transfer on channel 4.
#define AT91C_HDMA_SCREQ5     ((unsigned int) 0x1 << 10) // (HDMA) Request a source chunk transfer on channel 5
#define 	AT91C_HDMA_SCREQ5_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ5_1                    ((unsigned int) 0x1 << 10) // (HDMA) Request a source chunk transfer on channel 5.
#define AT91C_HDMA_DCREQ6     ((unsigned int) 0x1 << 11) // (HDMA) Request a destination chunk transfer on channel 5
#define 	AT91C_HDMA_DCREQ6_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ6_1                    ((unsigned int) 0x1 << 11) // (HDMA) Request a destination chunk transfer on channel 5.
#define AT91C_HDMA_SCREQ6     ((unsigned int) 0x1 << 12) // (HDMA) Request a source chunk transfer on channel 6
#define 	AT91C_HDMA_SCREQ6_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ6_1                    ((unsigned int) 0x1 << 12) // (HDMA) Request a source chunk transfer on channel 6.
#define AT91C_HDMA_SCREQ7     ((unsigned int) 0x1 << 14) // (HDMA) Request a source chunk transfer on channel 7
#define 	AT91C_HDMA_SCREQ7_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ7_1                    ((unsigned int) 0x1 << 14) // (HDMA) Request a source chunk transfer on channel 7.
#define AT91C_HDMA_DCREQ7     ((unsigned int) 0x1 << 15) // (HDMA) Request a destination chunk transfer on channel 7
#define 	AT91C_HDMA_DCREQ7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Request a destination chunk transfer on channel 7.
// -------- HDMA_LAST : (HDMA Offset: 0x10)  -------- 
#define AT91C_HDMA_SLAST0     ((unsigned int) 0x1 <<  0) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 0
#define 	AT91C_HDMA_SLAST0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST0_1                    ((unsigned int) 0x1) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 0.
#define AT91C_HDMA_DLAST0     ((unsigned int) 0x1 <<  1) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 0
#define 	AT91C_HDMA_DLAST0_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST0_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 0.
#define AT91C_HDMA_SLAST1     ((unsigned int) 0x1 <<  2) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 1
#define 	AT91C_HDMA_SLAST1_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST1_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 1.
#define AT91C_HDMA_DLAST1     ((unsigned int) 0x1 <<  3) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 1
#define 	AT91C_HDMA_DLAST1_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST1_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 1.
#define AT91C_HDMA_SLAST2     ((unsigned int) 0x1 <<  4) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 2
#define 	AT91C_HDMA_SLAST2_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST2_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 2.
#define AT91C_HDMA_DLAST2     ((unsigned int) 0x1 <<  5) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 2
#define 	AT91C_HDMA_DLAST2_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST2_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 2.
#define AT91C_HDMA_SLAST3     ((unsigned int) 0x1 <<  6) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 3
#define 	AT91C_HDMA_SLAST3_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST3_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 3.
#define AT91C_HDMA_DLAST3     ((unsigned int) 0x1 <<  7) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 3
#define 	AT91C_HDMA_DLAST3_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST3_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 3.
#define AT91C_HDMA_SLAST4     ((unsigned int) 0x1 <<  8) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 4
#define 	AT91C_HDMA_SLAST4_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST4_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 4.
#define AT91C_HDMA_DLAST4     ((unsigned int) 0x1 <<  9) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 4
#define 	AT91C_HDMA_DLAST4_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST4_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 4.
#define AT91C_HDMA_SLAST5     ((unsigned int) 0x1 << 10) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 5
#define 	AT91C_HDMA_SLAST5_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST5_1                    ((unsigned int) 0x1 << 10) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 5.
#define AT91C_HDMA_DLAST6     ((unsigned int) 0x1 << 11) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 5
#define 	AT91C_HDMA_DLAST6_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST6_1                    ((unsigned int) 0x1 << 11) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 5.
#define AT91C_HDMA_SLAST6     ((unsigned int) 0x1 << 12) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 6
#define 	AT91C_HDMA_SLAST6_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST6_1                    ((unsigned int) 0x1 << 12) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 6.
#define AT91C_HDMA_SLAST7     ((unsigned int) 0x1 << 14) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 7
#define 	AT91C_HDMA_SLAST7_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST7_1                    ((unsigned int) 0x1 << 14) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 7.
#define AT91C_HDMA_DLAST7     ((unsigned int) 0x1 << 15) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 7
#define 	AT91C_HDMA_DLAST7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 7.
// -------- HDMA_SYNC : (HDMA Offset: 0x14)  -------- 
#define AT91C_SYNC_REQ        ((unsigned int) 0xFFFF <<  0) // (HDMA) 
// -------- HDMA_EBCIER : (HDMA Offset: 0x18) Buffer Transfer Completed/Chained Buffer Transfer Completed/Access Error Interrupt Enable Register -------- 
#define AT91C_HDMA_BTC0       ((unsigned int) 0x1 <<  0) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC1       ((unsigned int) 0x1 <<  1) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC2       ((unsigned int) 0x1 <<  2) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC3       ((unsigned int) 0x1 <<  3) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC4       ((unsigned int) 0x1 <<  4) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC5       ((unsigned int) 0x1 <<  5) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC6       ((unsigned int) 0x1 <<  6) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC7       ((unsigned int) 0x1 <<  7) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC0      ((unsigned int) 0x1 <<  8) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC1      ((unsigned int) 0x1 <<  9) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC2      ((unsigned int) 0x1 << 10) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC3      ((unsigned int) 0x1 << 11) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC4      ((unsigned int) 0x1 << 12) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC5      ((unsigned int) 0x1 << 13) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC6      ((unsigned int) 0x1 << 14) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC7      ((unsigned int) 0x1 << 15) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR0       ((unsigned int) 0x1 << 16) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR1       ((unsigned int) 0x1 << 17) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR2       ((unsigned int) 0x1 << 18) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR3       ((unsigned int) 0x1 << 19) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR4       ((unsigned int) 0x1 << 20) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR5       ((unsigned int) 0x1 << 21) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR6       ((unsigned int) 0x1 << 22) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR7       ((unsigned int) 0x1 << 23) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
// -------- HDMA_EBCIDR : (HDMA Offset: 0x1c)  -------- 
// -------- HDMA_EBCIMR : (HDMA Offset: 0x20)  -------- 
// -------- HDMA_EBCISR : (HDMA Offset: 0x24)  -------- 
// -------- HDMA_CHER : (HDMA Offset: 0x28)  -------- 
#define AT91C_HDMA_ENA0       ((unsigned int) 0x1 <<  0) // (HDMA) When set, channel 0 enabled.
#define 	AT91C_HDMA_ENA0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA0_1                    ((unsigned int) 0x1) // (HDMA) Channel 0 enabled.
#define AT91C_HDMA_ENA1       ((unsigned int) 0x1 <<  1) // (HDMA) When set, channel 1 enabled.
#define 	AT91C_HDMA_ENA1_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA1_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Channel 1 enabled.
#define AT91C_HDMA_ENA2       ((unsigned int) 0x1 <<  2) // (HDMA) When set, channel 2 enabled.
#define 	AT91C_HDMA_ENA2_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA2_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Channel 2 enabled.
#define AT91C_HDMA_ENA3       ((unsigned int) 0x1 <<  3) // (HDMA) When set, channel 3 enabled.
#define 	AT91C_HDMA_ENA3_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA3_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Channel 3 enabled.
#define AT91C_HDMA_ENA4       ((unsigned int) 0x1 <<  4) // (HDMA) When set, channel 4 enabled.
#define 	AT91C_HDMA_ENA4_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA4_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Channel 4 enabled.
#define AT91C_HDMA_ENA5       ((unsigned int) 0x1 <<  5) // (HDMA) When set, channel 5 enabled.
#define 	AT91C_HDMA_ENA5_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA5_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Channel 5 enabled.
#define AT91C_HDMA_ENA6       ((unsigned int) 0x1 <<  6) // (HDMA) When set, channel 6 enabled.
#define 	AT91C_HDMA_ENA6_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA6_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Channel 6 enabled.
#define AT91C_HDMA_ENA7       ((unsigned int) 0x1 <<  7) // (HDMA) When set, channel 7 enabled.
#define 	AT91C_HDMA_ENA7_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA7_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Channel 7 enabled.
#define AT91C_HDMA_SUSP0      ((unsigned int) 0x1 <<  8) // (HDMA) When set, channel 0 freezed and its current context.
#define 	AT91C_HDMA_SUSP0_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP0_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Channel 0 freezed.
#define AT91C_HDMA_SUSP1      ((unsigned int) 0x1 <<  9) // (HDMA) When set, channel 1 freezed and its current context.
#define 	AT91C_HDMA_SUSP1_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP1_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Channel 1 freezed.
#define AT91C_HDMA_SUSP2      ((unsigned int) 0x1 << 10) // (HDMA) When set, channel 2 freezed and its current context.
#define 	AT91C_HDMA_SUSP2_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP2_1                    ((unsigned int) 0x1 << 10) // (HDMA) Channel 2 freezed.
#define AT91C_HDMA_SUSP3      ((unsigned int) 0x1 << 11) // (HDMA) When set, channel 3 freezed and its current context.
#define 	AT91C_HDMA_SUSP3_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP3_1                    ((unsigned int) 0x1 << 11) // (HDMA) Channel 3 freezed.
#define AT91C_HDMA_SUSP4      ((unsigned int) 0x1 << 12) // (HDMA) When set, channel 4 freezed and its current context.
#define 	AT91C_HDMA_SUSP4_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP4_1                    ((unsigned int) 0x1 << 12) // (HDMA) Channel 4 freezed.
#define AT91C_HDMA_SUSP5      ((unsigned int) 0x1 << 13) // (HDMA) When set, channel 5 freezed and its current context.
#define 	AT91C_HDMA_SUSP5_0                    ((unsigned int) 0x0 << 13) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP5_1                    ((unsigned int) 0x1 << 13) // (HDMA) Channel 5 freezed.
#define AT91C_HDMA_SUSP6      ((unsigned int) 0x1 << 14) // (HDMA) When set, channel 6 freezed and its current context.
#define 	AT91C_HDMA_SUSP6_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP6_1                    ((unsigned int) 0x1 << 14) // (HDMA) Channel 6 freezed.
#define AT91C_HDMA_SUSP7      ((unsigned int) 0x1 << 15) // (HDMA) When set, channel 7 freezed and its current context.
#define 	AT91C_HDMA_SUSP7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Channel 7 freezed.
#define AT91C_HDMA_KEEP0      ((unsigned int) 0x1 << 24) // (HDMA) When set, it resumes the channel 0 from an automatic stall state.
#define 	AT91C_HDMA_KEEP0_0                    ((unsigned int) 0x0 << 24) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP0_1                    ((unsigned int) 0x1 << 24) // (HDMA) Resumes the channel 0.
#define AT91C_HDMA_KEEP1      ((unsigned int) 0x1 << 25) // (HDMA) When set, it resumes the channel 1 from an automatic stall state.
#define 	AT91C_HDMA_KEEP1_0                    ((unsigned int) 0x0 << 25) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP1_1                    ((unsigned int) 0x1 << 25) // (HDMA) Resumes the channel 1.
#define AT91C_HDMA_KEEP2      ((unsigned int) 0x1 << 26) // (HDMA) When set, it resumes the channel 2 from an automatic stall state.
#define 	AT91C_HDMA_KEEP2_0                    ((unsigned int) 0x0 << 26) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP2_1                    ((unsigned int) 0x1 << 26) // (HDMA) Resumes the channel 2.
#define AT91C_HDMA_KEEP3      ((unsigned int) 0x1 << 27) // (HDMA) When set, it resumes the channel 3 from an automatic stall state.
#define 	AT91C_HDMA_KEEP3_0                    ((unsigned int) 0x0 << 27) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP3_1                    ((unsigned int) 0x1 << 27) // (HDMA) Resumes the channel 3.
#define AT91C_HDMA_KEEP4      ((unsigned int) 0x1 << 28) // (HDMA) When set, it resumes the channel 4 from an automatic stall state.
#define 	AT91C_HDMA_KEEP4_0                    ((unsigned int) 0x0 << 28) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP4_1                    ((unsigned int) 0x1 << 28) // (HDMA) Resumes the channel 4.
#define AT91C_HDMA_KEEP5      ((unsigned int) 0x1 << 29) // (HDMA) When set, it resumes the channel 5 from an automatic stall state.
#define 	AT91C_HDMA_KEEP5_0                    ((unsigned int) 0x0 << 29) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP5_1                    ((unsigned int) 0x1 << 29) // (HDMA) Resumes the channel 5.
#define AT91C_HDMA_KEEP6      ((unsigned int) 0x1 << 30) // (HDMA) When set, it resumes the channel 6 from an automatic stall state.
#define 	AT91C_HDMA_KEEP6_0                    ((unsigned int) 0x0 << 30) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP6_1                    ((unsigned int) 0x1 << 30) // (HDMA) Resumes the channel 6.
#define AT91C_HDMA_KEEP7      ((unsigned int) 0x1 << 31) // (HDMA) When set, it resumes the channel 7 from an automatic stall state.
#define 	AT91C_HDMA_KEEP7_0                    ((unsigned int) 0x0 << 31) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP7_1                    ((unsigned int) 0x1 << 31) // (HDMA) Resumes the channel 7.
// -------- HDMA_CHDR : (HDMA Offset: 0x2c)  -------- 
#define AT91C_HDMA_DIS0       ((unsigned int) 0x1 <<  0) // (HDMA) Write one to this field to disable the channel 0.
#define 	AT91C_HDMA_DIS0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS0_1                    ((unsigned int) 0x1) // (HDMA) Disables the channel 0.
#define AT91C_HDMA_DIS1       ((unsigned int) 0x1 <<  1) // (HDMA) Write one to this field to disable the channel 1.
#define 	AT91C_HDMA_DIS1_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS1_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Disables the channel 1.
#define AT91C_HDMA_DIS2       ((unsigned int) 0x1 <<  2) // (HDMA) Write one to this field to disable the channel 2.
#define 	AT91C_HDMA_DIS2_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS2_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Disables the channel 2.
#define AT91C_HDMA_DIS3       ((unsigned int) 0x1 <<  3) // (HDMA) Write one to this field to disable the channel 3.
#define 	AT91C_HDMA_DIS3_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS3_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Disables the channel 3.
#define AT91C_HDMA_DIS4       ((unsigned int) 0x1 <<  4) // (HDMA) Write one to this field to disable the channel 4.
#define 	AT91C_HDMA_DIS4_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS4_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Disables the channel 4.
#define AT91C_HDMA_DIS5       ((unsigned int) 0x1 <<  5) // (HDMA) Write one to this field to disable the channel 5.
#define 	AT91C_HDMA_DIS5_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS5_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Disables the channel 5.
#define AT91C_HDMA_DIS6       ((unsigned int) 0x1 <<  6) // (HDMA) Write one to this field to disable the channel 6.
#define 	AT91C_HDMA_DIS6_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS6_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Disables the channel 6.
#define AT91C_HDMA_DIS7       ((unsigned int) 0x1 <<  7) // (HDMA) Write one to this field to disable the channel 7.
#define 	AT91C_HDMA_DIS7_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS7_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Disables the channel 7.
#define AT91C_HDMA_RES0       ((unsigned int) 0x1 <<  8) // (HDMA) Write one to this field to resume the channel 0 transfer restoring its context.
#define 	AT91C_HDMA_RES0_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_RES0_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Resumes the channel 0.
#define AT91C_HDMA_RES1       ((unsigned int) 0x1 <<  9) // (HDMA) Write one to this field to resume the channel 1 transfer restoring its context.
#define 	AT91C_HDMA_RES1_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_RES1_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Resumes the channel 1.
#define AT91C_HDMA_RES2       ((unsigned int) 0x1 << 10) // (HDMA) Write one to this field to resume the channel 2 transfer restoring its context.
#define 	AT91C_HDMA_RES2_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_RES2_1                    ((unsigned int) 0x1 << 10) // (HDMA) Resumes the channel 2.
#define AT91C_HDMA_RES3       ((unsigned int) 0x1 << 11) // (HDMA) Write one to this field to resume the channel 3 transfer restoring its context.
#define 	AT91C_HDMA_RES3_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_RES3_1                    ((unsigned int) 0x1 << 11) // (HDMA) Resumes the channel 3.
#define AT91C_HDMA_RES4       ((unsigned int) 0x1 << 12) // (HDMA) Write one to this field to resume the channel 4 transfer restoring its context.
#define 	AT91C_HDMA_RES4_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_RES4_1                    ((unsigned int) 0x1 << 12) // (HDMA) Resumes the channel 4.
#define AT91C_HDMA_RES5       ((unsigned int) 0x1 << 13) // (HDMA) Write one to this field to resume the channel 5 transfer restoring its context.
#define 	AT91C_HDMA_RES5_0                    ((unsigned int) 0x0 << 13) // (HDMA) No effect.
#define 	AT91C_HDMA_RES5_1                    ((unsigned int) 0x1 << 13) // (HDMA) Resumes the channel 5.
#define AT91C_HDMA_RES6       ((unsigned int) 0x1 << 14) // (HDMA) Write one to this field to resume the channel 6 transfer restoring its context.
#define 	AT91C_HDMA_RES6_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_RES6_1                    ((unsigned int) 0x1 << 14) // (HDMA) Resumes the channel 6.
#define AT91C_HDMA_RES7       ((unsigned int) 0x1 << 15) // (HDMA) Write one to this field to resume the channel 7 transfer restoring its context.
#define 	AT91C_HDMA_RES7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_RES7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Resumes the channel 7.
// -------- HDMA_CHSR : (HDMA Offset: 0x30)  -------- 
#define AT91C_HDMA_EMPT0      ((unsigned int) 0x1 << 16) // (HDMA) When set, channel 0 is empty.
#define 	AT91C_HDMA_EMPT0_0                    ((unsigned int) 0x0 << 16) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT0_1                    ((unsigned int) 0x1 << 16) // (HDMA) Channel 0 empty.
#define AT91C_HDMA_EMPT1      ((unsigned int) 0x1 << 17) // (HDMA) When set, channel 1 is empty.
#define 	AT91C_HDMA_EMPT1_0                    ((unsigned int) 0x0 << 17) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT1_1                    ((unsigned int) 0x1 << 17) // (HDMA) Channel 1 empty.
#define AT91C_HDMA_EMPT2      ((unsigned int) 0x1 << 18) // (HDMA) When set, channel 2 is empty.
#define 	AT91C_HDMA_EMPT2_0                    ((unsigned int) 0x0 << 18) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT2_1                    ((unsigned int) 0x1 << 18) // (HDMA) Channel 2 empty.
#define AT91C_HDMA_EMPT3      ((unsigned int) 0x1 << 19) // (HDMA) When set, channel 3 is empty.
#define 	AT91C_HDMA_EMPT3_0                    ((unsigned int) 0x0 << 19) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT3_1                    ((unsigned int) 0x1 << 19) // (HDMA) Channel 3 empty.
#define AT91C_HDMA_EMPT4      ((unsigned int) 0x1 << 20) // (HDMA) When set, channel 4 is empty.
#define 	AT91C_HDMA_EMPT4_0                    ((unsigned int) 0x0 << 20) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT4_1                    ((unsigned int) 0x1 << 20) // (HDMA) Channel 4 empty.
#define AT91C_HDMA_EMPT5      ((unsigned int) 0x1 << 21) // (HDMA) When set, channel 5 is empty.
#define 	AT91C_HDMA_EMPT5_0                    ((unsigned int) 0x0 << 21) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT5_1                    ((unsigned int) 0x1 << 21) // (HDMA) Channel 5 empty.
#define AT91C_HDMA_EMPT6      ((unsigned int) 0x1 << 22) // (HDMA) When set, channel 6 is empty.
#define 	AT91C_HDMA_EMPT6_0                    ((unsigned int) 0x0 << 22) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT6_1                    ((unsigned int) 0x1 << 22) // (HDMA) Channel 6 empty.
#define AT91C_HDMA_EMPT7      ((unsigned int) 0x1 << 23) // (HDMA) When set, channel 7 is empty.
#define 	AT91C_HDMA_EMPT7_0                    ((unsigned int) 0x0 << 23) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT7_1                    ((unsigned int) 0x1 << 23) // (HDMA) Channel 7 empty.
#define AT91C_HDMA_STAL0      ((unsigned int) 0x1 << 24) // (HDMA) When set, channel 0 is stalled.
#define 	AT91C_HDMA_STAL0_0                    ((unsigned int) 0x0 << 24) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL0_1                    ((unsigned int) 0x1 << 24) // (HDMA) Channel 0 stalled.
#define AT91C_HDMA_STAL1      ((unsigned int) 0x1 << 25) // (HDMA) When set, channel 1 is stalled.
#define 	AT91C_HDMA_STAL1_0                    ((unsigned int) 0x0 << 25) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL1_1                    ((unsigned int) 0x1 << 25) // (HDMA) Channel 1 stalled.
#define AT91C_HDMA_STAL2      ((unsigned int) 0x1 << 26) // (HDMA) When set, channel 2 is stalled.
#define 	AT91C_HDMA_STAL2_0                    ((unsigned int) 0x0 << 26) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL2_1                    ((unsigned int) 0x1 << 26) // (HDMA) Channel 2 stalled.
#define AT91C_HDMA_STAL3      ((unsigned int) 0x1 << 27) // (HDMA) When set, channel 3 is stalled.
#define 	AT91C_HDMA_STAL3_0                    ((unsigned int) 0x0 << 27) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL3_1                    ((unsigned int) 0x1 << 27) // (HDMA) Channel 3 stalled.
#define AT91C_HDMA_STAL4      ((unsigned int) 0x1 << 28) // (HDMA) When set, channel 4 is stalled.
#define 	AT91C_HDMA_STAL4_0                    ((unsigned int) 0x0 << 28) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL4_1                    ((unsigned int) 0x1 << 28) // (HDMA) Channel 4 stalled.
#define AT91C_HDMA_STAL5      ((unsigned int) 0x1 << 29) // (HDMA) When set, channel 5 is stalled.
#define 	AT91C_HDMA_STAL5_0                    ((unsigned int) 0x0 << 29) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL5_1                    ((unsigned int) 0x1 << 29) // (HDMA) Channel 5 stalled.
#define AT91C_HDMA_STAL6      ((unsigned int) 0x1 << 30) // (HDMA) When set, channel 6 is stalled.
#define 	AT91C_HDMA_STAL6_0                    ((unsigned int) 0x0 << 30) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL6_1                    ((unsigned int) 0x1 << 30) // (HDMA) Channel 6 stalled.
#define AT91C_HDMA_STAL7      ((unsigned int) 0x1 << 31) // (HDMA) When set, channel 7 is stalled.
#define 	AT91C_HDMA_STAL7_0                    ((unsigned int) 0x0 << 31) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL7_1                    ((unsigned int) 0x1 << 31) // (HDMA) Channel 7 stalled.
// -------- HDMA_RSVD : (HDMA Offset: 0x34)  -------- 
// -------- HDMA_RSVD : (HDMA Offset: 0x38)  -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Error Correction Code controller
// *****************************************************************************
typedef struct _AT91S_ECC {
	AT91_REG	 ECC_CR; 	//  ECC reset register
	AT91_REG	 ECC_MR; 	//  ECC Page size register
	AT91_REG	 ECC_SR; 	//  ECC Status register
	AT91_REG	 ECC_PR; 	//  ECC Parity register
	AT91_REG	 ECC_NPR; 	//  ECC Parity N register
	AT91_REG	 Reserved0[58]; 	// 
	AT91_REG	 ECC_VR; 	//  ECC Version register
} AT91S_ECC, *AT91PS_ECC;

// -------- ECC_CR : (ECC Offset: 0x0) ECC reset register -------- 
#define AT91C_ECC_RST         ((unsigned int) 0x1 <<  0) // (ECC) ECC reset parity
// -------- ECC_MR : (ECC Offset: 0x4) ECC page size register -------- 
#define AT91C_ECC_PAGE_SIZE   ((unsigned int) 0x3 <<  0) // (ECC) Nand Flash page size
// -------- ECC_SR : (ECC Offset: 0x8) ECC status register -------- 
#define AT91C_ECC_RECERR      ((unsigned int) 0x1 <<  0) // (ECC) ECC error
#define AT91C_ECC_ECCERR      ((unsigned int) 0x1 <<  1) // (ECC) ECC single error
#define AT91C_ECC_MULERR      ((unsigned int) 0x1 <<  2) // (ECC) ECC_MULERR
// -------- ECC_PR : (ECC Offset: 0xc) ECC parity register -------- 
#define AT91C_ECC_BITADDR     ((unsigned int) 0xF <<  0) // (ECC) Bit address error
#define AT91C_ECC_WORDADDR    ((unsigned int) 0xFFF <<  4) // (ECC) address of the failing bit
// -------- ECC_NPR : (ECC Offset: 0x10) ECC N parity register -------- 
#define AT91C_ECC_NPARITY     ((unsigned int) 0xFFFF <<  0) // (ECC) ECC parity N 
// -------- ECC_VR : (ECC Offset: 0xfc) ECC version register -------- 
#define AT91C_ECC_VR          ((unsigned int) 0xF <<  0) // (ECC) ECC version register

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Ethernet MAC 10/100
// *****************************************************************************
typedef struct _AT91S_EMAC {
	AT91_REG	 EMAC_NCR; 	// Network Control Register
	AT91_REG	 EMAC_NCFGR; 	// Network Configuration Register
	AT91_REG	 EMAC_NSR; 	// Network Status Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 EMAC_TSR; 	// Transmit Status Register
	AT91_REG	 EMAC_RBQP; 	// Receive Buffer Queue Pointer
	AT91_REG	 EMAC_TBQP; 	// Transmit Buffer Queue Pointer
	AT91_REG	 EMAC_RSR; 	// Receive Status Register
	AT91_REG	 EMAC_ISR; 	// Interrupt Status Register
	AT91_REG	 EMAC_IER; 	// Interrupt Enable Register
	AT91_REG	 EMAC_IDR; 	// Interrupt Disable Register
	AT91_REG	 EMAC_IMR; 	// Interrupt Mask Register
	AT91_REG	 EMAC_MAN; 	// PHY Maintenance Register
	AT91_REG	 EMAC_PTR; 	// Pause Time Register
	AT91_REG	 EMAC_PFR; 	// Pause Frames received Register
	AT91_REG	 EMAC_FTO; 	// Frames Transmitted OK Register
	AT91_REG	 EMAC_SCF; 	// Single Collision Frame Register
	AT91_REG	 EMAC_MCF; 	// Multiple Collision Frame Register
	AT91_REG	 EMAC_FRO; 	// Frames Received OK Register
	AT91_REG	 EMAC_FCSE; 	// Frame Check Sequence Error Register
	AT91_REG	 EMAC_ALE; 	// Alignment Error Register
	AT91_REG	 EMAC_DTF; 	// Deferred Transmission Frame Register
	AT91_REG	 EMAC_LCOL; 	// Late Collision Register
	AT91_REG	 EMAC_ECOL; 	// Excessive Collision Register
	AT91_REG	 EMAC_TUND; 	// Transmit Underrun Error Register
	AT91_REG	 EMAC_CSE; 	// Carrier Sense Error Register
	AT91_REG	 EMAC_RRE; 	// Receive Ressource Error Register
	AT91_REG	 EMAC_ROV; 	// Receive Overrun Errors Register
	AT91_REG	 EMAC_RSE; 	// Receive Symbol Errors Register
	AT91_REG	 EMAC_ELE; 	// Excessive Length Errors Register
	AT91_REG	 EMAC_RJA; 	// Receive Jabbers Register
	AT91_REG	 EMAC_USF; 	// Undersize Frames Register
	AT91_REG	 EMAC_STE; 	// SQE Test Error Register
	AT91_REG	 EMAC_RLE; 	// Receive Length Field Mismatch Register
	AT91_REG	 EMAC_TPF; 	// Transmitted Pause Frames Register
	AT91_REG	 EMAC_HRB; 	// Hash Address Bottom[31:0]
	AT91_REG	 EMAC_HRT; 	// Hash Address Top[63:32]
	AT91_REG	 EMAC_SA1L; 	// Specific Address 1 Bottom, First 4 bytes
	AT91_REG	 EMAC_SA1H; 	// Specific Address 1 Top, Last 2 bytes
	AT91_REG	 EMAC_SA2L; 	// Specific Address 2 Bottom, First 4 bytes
	AT91_REG	 EMAC_SA2H; 	// Specific Address 2 Top, Last 2 bytes
	AT91_REG	 EMAC_SA3L; 	// Specific Address 3 Bottom, First 4 bytes
	AT91_REG	 EMAC_SA3H; 	// Specific Address 3 Top, Last 2 bytes
	AT91_REG	 EMAC_SA4L; 	// Specific Address 4 Bottom, First 4 bytes
	AT91_REG	 EMAC_SA4H; 	// Specific Address 4 Top, Last 2 bytes
	AT91_REG	 EMAC_TID; 	// Type ID Checking Register
	AT91_REG	 EMAC_TPQ; 	// Transmit Pause Quantum Register
	AT91_REG	 EMAC_USRIO; 	// USER Input/Output Register
	AT91_REG	 EMAC_WOL; 	// Wake On LAN Register
	AT91_REG	 Reserved1[13]; 	// 
	AT91_REG	 EMAC_REV; 	// Revision Register
} AT91S_EMAC, *AT91PS_EMAC;

// -------- EMAC_NCR : (EMAC Offset: 0x0)  -------- 
#define AT91C_EMAC_LB         ((unsigned int) 0x1 <<  0) // (EMAC) Loopback. Optional. When set, loopback signal is at high level.
#define AT91C_EMAC_LLB        ((unsigned int) 0x1 <<  1) // (EMAC) Loopback local. 
#define AT91C_EMAC_RE         ((unsigned int) 0x1 <<  2) // (EMAC) Receive enable. 
#define AT91C_EMAC_TE         ((unsigned int) 0x1 <<  3) // (EMAC) Transmit enable. 
#define AT91C_EMAC_MPE        ((unsigned int) 0x1 <<  4) // (EMAC) Management port enable. 
#define AT91C_EMAC_CLRSTAT    ((unsigned int) 0x1 <<  5) // (EMAC) Clear statistics registers. 
#define AT91C_EMAC_INCSTAT    ((unsigned int) 0x1 <<  6) // (EMAC) Increment statistics registers. 
#define AT91C_EMAC_WESTAT     ((unsigned int) 0x1 <<  7) // (EMAC) Write enable for statistics registers. 
#define AT91C_EMAC_BP         ((unsigned int) 0x1 <<  8) // (EMAC) Back pressure. 
#define AT91C_EMAC_TSTART     ((unsigned int) 0x1 <<  9) // (EMAC) Start Transmission. 
#define AT91C_EMAC_THALT      ((unsigned int) 0x1 << 10) // (EMAC) Transmission Halt. 
#define AT91C_EMAC_TPFR       ((unsigned int) 0x1 << 11) // (EMAC) Transmit pause frame 
#define AT91C_EMAC_TZQ        ((unsigned int) 0x1 << 12) // (EMAC) Transmit zero quantum pause frame
// -------- EMAC_NCFGR : (EMAC Offset: 0x4) Network Configuration Register -------- 
#define AT91C_EMAC_SPD        ((unsigned int) 0x1 <<  0) // (EMAC) Speed. 
#define AT91C_EMAC_FD         ((unsigned int) 0x1 <<  1) // (EMAC) Full duplex. 
#define AT91C_EMAC_JFRAME     ((unsigned int) 0x1 <<  3) // (EMAC) Jumbo Frames. 
#define AT91C_EMAC_CAF        ((unsigned int) 0x1 <<  4) // (EMAC) Copy all frames. 
#define AT91C_EMAC_NBC        ((unsigned int) 0x1 <<  5) // (EMAC) No broadcast. 
#define AT91C_EMAC_MTI        ((unsigned int) 0x1 <<  6) // (EMAC) Multicast hash event enable
#define AT91C_EMAC_UNI        ((unsigned int) 0x1 <<  7) // (EMAC) Unicast hash enable. 
#define AT91C_EMAC_BIG        ((unsigned int) 0x1 <<  8) // (EMAC) Receive 1522 bytes. 
#define AT91C_EMAC_EAE        ((unsigned int) 0x1 <<  9) // (EMAC) External address match enable. 
#define AT91C_EMAC_CLK        ((unsigned int) 0x3 << 10) // (EMAC) 
#define 	AT91C_EMAC_CLK_HCLK_8               ((unsigned int) 0x0 << 10) // (EMAC) HCLK divided by 8
#define 	AT91C_EMAC_CLK_HCLK_16              ((unsigned int) 0x1 << 10) // (EMAC) HCLK divided by 16
#define 	AT91C_EMAC_CLK_HCLK_32              ((unsigned int) 0x2 << 10) // (EMAC) HCLK divided by 32
#define 	AT91C_EMAC_CLK_HCLK_64              ((unsigned int) 0x3 << 10) // (EMAC) HCLK divided by 64
#define AT91C_EMAC_RTY        ((unsigned int) 0x1 << 12) // (EMAC) 
#define AT91C_EMAC_PAE        ((unsigned int) 0x1 << 13) // (EMAC) 
#define AT91C_EMAC_RBOF       ((unsigned int) 0x3 << 14) // (EMAC) 
#define 	AT91C_EMAC_RBOF_OFFSET_0             ((unsigned int) 0x0 << 14) // (EMAC) no offset from start of receive buffer
#define 	AT91C_EMAC_RBOF_OFFSET_1             ((unsigned int) 0x1 << 14) // (EMAC) one byte offset from start of receive buffer
#define 	AT91C_EMAC_RBOF_OFFSET_2             ((unsigned int) 0x2 << 14) // (EMAC) two bytes offset from start of receive buffer
#define 	AT91C_EMAC_RBOF_OFFSET_3             ((unsigned int) 0x3 << 14) // (EMAC) three bytes offset from start of receive buffer
#define AT91C_EMAC_RLCE       ((unsigned int) 0x1 << 16) // (EMAC) Receive Length field Checking Enable
#define AT91C_EMAC_DRFCS      ((unsigned int) 0x1 << 17) // (EMAC) Discard Receive FCS
#define AT91C_EMAC_EFRHD      ((unsigned int) 0x1 << 18) // (EMAC) 
#define AT91C_EMAC_IRXFCS     ((unsigned int) 0x1 << 19) // (EMAC) Ignore RX FCS
// -------- EMAC_NSR : (EMAC Offset: 0x8) Network Status Register -------- 
#define AT91C_EMAC_LINKR      ((unsigned int) 0x1 <<  0) // (EMAC) 
#define AT91C_EMAC_MDIO       ((unsigned int) 0x1 <<  1) // (EMAC) 
#define AT91C_EMAC_IDLE       ((unsigned int) 0x1 <<  2) // (EMAC) 
// -------- EMAC_TSR : (EMAC Offset: 0x14) Transmit Status Register -------- 
#define AT91C_EMAC_UBR        ((unsigned int) 0x1 <<  0) // (EMAC) 
#define AT91C_EMAC_COL        ((unsigned int) 0x1 <<  1) // (EMAC) 
#define AT91C_EMAC_RLES       ((unsigned int) 0x1 <<  2) // (EMAC) 
#define AT91C_EMAC_TGO        ((unsigned int) 0x1 <<  3) // (EMAC) Transmit Go
#define AT91C_EMAC_BEX        ((unsigned int) 0x1 <<  4) // (EMAC) Buffers exhausted mid frame
#define AT91C_EMAC_COMP       ((unsigned int) 0x1 <<  5) // (EMAC) 
#define AT91C_EMAC_UND        ((unsigned int) 0x1 <<  6) // (EMAC) 
// -------- EMAC_RSR : (EMAC Offset: 0x20) Receive Status Register -------- 
#define AT91C_EMAC_BNA        ((unsigned int) 0x1 <<  0) // (EMAC) 
#define AT91C_EMAC_REC        ((unsigned int) 0x1 <<  1) // (EMAC) 
#define AT91C_EMAC_OVR        ((unsigned int) 0x1 <<  2) // (EMAC) 
// -------- EMAC_ISR : (EMAC Offset: 0x24) Interrupt Status Register -------- 
#define AT91C_EMAC_MFD        ((unsigned int) 0x1 <<  0) // (EMAC) 
#define AT91C_EMAC_RCOMP      ((unsigned int) 0x1 <<  1) // (EMAC) 
#define AT91C_EMAC_RXUBR      ((unsigned int) 0x1 <<  2) // (EMAC) 
#define AT91C_EMAC_TXUBR      ((unsigned int) 0x1 <<  3) // (EMAC) 
#define AT91C_EMAC_TUNDR      ((unsigned int) 0x1 <<  4) // (EMAC) 
#define AT91C_EMAC_RLEX       ((unsigned int) 0x1 <<  5) // (EMAC) 
#define AT91C_EMAC_TXERR      ((unsigned int) 0x1 <<  6) // (EMAC) 
#define AT91C_EMAC_TCOMP      ((unsigned int) 0x1 <<  7) // (EMAC) 
#define AT91C_EMAC_LINK       ((unsigned int) 0x1 <<  9) // (EMAC) 
#define AT91C_EMAC_ROVR       ((unsigned int) 0x1 << 10) // (EMAC) 
#define AT91C_EMAC_HRESP      ((unsigned int) 0x1 << 11) // (EMAC) 
#define AT91C_EMAC_PFRE       ((unsigned int) 0x1 << 12) // (EMAC) 
#define AT91C_EMAC_PTZ        ((unsigned int) 0x1 << 13) // (EMAC) 
#define AT91C_EMAC_WOLEV      ((unsigned int) 0x1 << 14) // (EMAC) 
// -------- EMAC_IER : (EMAC Offset: 0x28) Interrupt Enable Register -------- 
// -------- EMAC_IDR : (EMAC Offset: 0x2c) Interrupt Disable Register -------- 
// -------- EMAC_IMR : (EMAC Offset: 0x30) Interrupt Mask Register -------- 
// -------- EMAC_MAN : (EMAC Offset: 0x34) PHY Maintenance Register -------- 
#define AT91C_EMAC_DATA       ((unsigned int) 0xFFFF <<  0) // (EMAC) 
#define AT91C_EMAC_CODE       ((unsigned int) 0x3 << 16) // (EMAC) 
#define AT91C_EMAC_REGA       ((unsigned int) 0x1F << 18) // (EMAC) 
#define AT91C_EMAC_PHYA       ((unsigned int) 0x1F << 23) // (EMAC) 
#define AT91C_EMAC_RW         ((unsigned int) 0x3 << 28) // (EMAC) 
#define AT91C_EMAC_SOF        ((unsigned int) 0x3 << 30) // (EMAC) 
// -------- EMAC_USRIO : (EMAC Offset: 0xc0) USER Input Output Register -------- 
#define AT91C_EMAC_RMII       ((unsigned int) 0x1 <<  0) // (EMAC) Reduce MII
#define AT91C_EMAC_CLKEN      ((unsigned int) 0x1 <<  1) // (EMAC) Clock Enable
// -------- EMAC_WOL : (EMAC Offset: 0xc4) Wake On LAN Register -------- 
#define AT91C_EMAC_IP         ((unsigned int) 0xFFFF <<  0) // (EMAC) ARP request IP address
#define AT91C_EMAC_MAG        ((unsigned int) 0x1 << 16) // (EMAC) Magic packet event enable
#define AT91C_EMAC_ARP        ((unsigned int) 0x1 << 17) // (EMAC) ARP request event enable
#define AT91C_EMAC_SA1        ((unsigned int) 0x1 << 18) // (EMAC) Specific address register 1 event enable
// -------- EMAC_REV : (EMAC Offset: 0xfc) Revision Register -------- 
#define AT91C_EMAC_REVREF     ((unsigned int) 0xFFFF <<  0) // (EMAC) 
#define AT91C_EMAC_PARTREF    ((unsigned int) 0xFFFF << 16) // (EMAC) 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Image Sensor Interface
// *****************************************************************************
typedef struct _AT91S_ISI {
	AT91_REG	 ISI_CFG1; 	// Configuration Register 1
	AT91_REG	 ISI_CFG2; 	// Configuration Register 2
	AT91_REG	 ISI_PSIZE; 	// Preview Size Register
	AT91_REG	 ISI_PDECF; 	// Preview Decimation Factor Register
	AT91_REG	 ISI_Y2RSET0; 	// Color Space Conversion YCrCb to RGB Register
	AT91_REG	 ISI_Y2RSET1; 	// Color Space Conversion YCrCb to RGB Register
	AT91_REG	 ISI_R2YSET0; 	// Color Space Conversion RGB to YCrCb Register
	AT91_REG	 ISI_R2YSET1; 	// Color Space Conversion RGB to YCrCb Register
	AT91_REG	 ISI_R2YSET2; 	// Color Space Conversion RGB to YCrCb Register
	AT91_REG	 ISI_CTRL; 	// Control Register
	AT91_REG	 ISI_SR; 	// Status Register
	AT91_REG	 ISI_IER; 	// Interrupt Enable Register
	AT91_REG	 ISI_IDR; 	// Interrupt Disable Register
	AT91_REG	 ISI_IMR; 	// Interrupt Mask Register
	AT91_REG	 ISI_DMACHER; 	// DMA Channel Enable Register
	AT91_REG	 ISI_DMACHDR; 	// DMA Channel Disable Register
	AT91_REG	 ISI_DMACHSR; 	// DMA Channel Status Register
	AT91_REG	 ISI_DMAPADDR; 	// DMA Preview Base Address Register
	AT91_REG	 ISI_DMAPCTRL; 	// DMA Preview Control Register
	AT91_REG	 ISI_DMAPDSCR; 	// DMA Preview Descriptor Address Register
	AT91_REG	 ISI_DMACADDR; 	// DMA Codec Base Address Register
	AT91_REG	 ISI_DMACCTRL; 	// DMA Codec Control Register
	AT91_REG	 ISI_DMACDSCR; 	// DMA Codec Descriptor Address Register
	AT91_REG	 Reserved0[34]; 	// 
	AT91_REG	 ISI_WPCR; 	// Write Protection Control Register
	AT91_REG	 ISI_WPSR; 	// Write Protection Status Register
	AT91_REG	 Reserved1[4]; 	// 
	AT91_REG	 ISI_VER; 	// Version Register
} AT91S_ISI, *AT91PS_ISI;

// -------- ISI_CFG1 : (ISI Offset: 0x0) ISI Configuration Register 1 -------- 
#define AT91C_ISI_HSYNC_POL   ((unsigned int) 0x1 <<  2) // (ISI) Horizontal synchronization polarity
#define 	AT91C_ISI_HSYNC_POL_ACTIVE_HIGH          ((unsigned int) 0x0 <<  2) // (ISI) HSYNC active high.
#define 	AT91C_ISI_HSYNC_POL_ACTIVE_LOW           ((unsigned int) 0x1 <<  2) // (ISI) HSYNC active low.
#define AT91C_ISI_VSYNC_POL   ((unsigned int) 0x1 <<  3) // (ISI) Vertical synchronization polarity
#define 	AT91C_ISI_VSYNC_POL_ACTIVE_HIGH          ((unsigned int) 0x0 <<  3) // (ISI) VSYNC active high.
#define 	AT91C_ISI_VSYNC_POL_ACTIVE_LOW           ((unsigned int) 0x1 <<  3) // (ISI) VSYNC active low.
#define AT91C_ISI_PIXCLK_POL  ((unsigned int) 0x1 <<  4) // (ISI) Pixel Clock Polarity
#define 	AT91C_ISI_PIXCLK_POL_RISING_EDGE          ((unsigned int) 0x0 <<  4) // (ISI) Data is sampled on rising edge of pixel clock.
#define 	AT91C_ISI_PIXCLK_POL_FALLING_EDGE         ((unsigned int) 0x1 <<  4) // (ISI) Data is sampled on falling edge of pixel clock.
#define AT91C_ISI_EMB_SYNC    ((unsigned int) 0x1 <<  6) // (ISI) Embedded synchronisation
#define 	AT91C_ISI_EMB_SYNC_HSYNC_VSYNC          ((unsigned int) 0x0 <<  6) // (ISI) Synchronization by HSYNC, VSYNC.
#define 	AT91C_ISI_EMB_SYNC_SAV_EAV              ((unsigned int) 0x1 <<  6) // (ISI) Synchronisation by Embedded Synchronization Sequence SAV/EAV.
#define AT91C_ISI_CRC_SYNC    ((unsigned int) 0x1 <<  7) // (ISI) CRC correction
#define 	AT91C_ISI_CRC_SYNC_CORRECTION_OFF       ((unsigned int) 0x0 <<  7) // (ISI) No CRC correction performed on embedded synchronization.
#define 	AT91C_ISI_CRC_SYNC_CORRECTION_ON        ((unsigned int) 0x1 <<  7) // (ISI) CRC correction is performed.
#define AT91C_ISI_FRATE       ((unsigned int) 0x7 <<  8) // (ISI) Frame rate capture
#define AT91C_ISI_FULL        ((unsigned int) 0x1 << 12) // (ISI) Full mode is allowed
#define 	AT91C_ISI_FULL_MODE_DISABLE         ((unsigned int) 0x0 << 12) // (ISI) Full mode disabled.
#define 	AT91C_ISI_FULL_MODE_ENABLE          ((unsigned int) 0x1 << 12) // (ISI) both codec and preview datapath are working simultaneously.
#define AT91C_ISI_THMASK      ((unsigned int) 0x3 << 13) // (ISI) DMA Burst Mask
#define 	AT91C_ISI_THMASK_4_BURST              ((unsigned int) 0x0 << 13) // (ISI) Only 4 beats AHB bursts are allowed
#define 	AT91C_ISI_THMASK_4_8_BURST            ((unsigned int) 0x1 << 13) // (ISI) Only 4 and 8 beats AHB bursts are allowed
#define 	AT91C_ISI_THMASK_4_8_16_BURST         ((unsigned int) 0x2 << 13) // (ISI) 4, 8 and 16 beats AHB bursts are allowed
#define AT91C_ISI_SLD         ((unsigned int) 0xFF << 16) // (ISI) Start of Line Delay
#define AT91C_ISI_SFD         ((unsigned int) 0xFF << 24) // (ISI) Start of frame Delay
// -------- ISI_CFG2 : (ISI Offset: 0x4) ISI Control Register 2 -------- 
#define AT91C_ISI_IM_VSIZE    ((unsigned int) 0x7FF <<  0) // (ISI) Vertical size of the Image sensor [0..2047]
#define AT91C_ISI_GS_MODE     ((unsigned int) 0x1 << 11) // (ISI) Grayscale Memory Mode
#define 	AT91C_ISI_GS_MODE_2_PIXELS             ((unsigned int) 0x0 << 11) // (ISI) 2 pixels per word.
#define 	AT91C_ISI_GS_MODE_1_PIXEL              ((unsigned int) 0x1 << 11) // (ISI) 1 pixel per word.
#define AT91C_ISI_RGB_MODE    ((unsigned int) 0x1 << 12) // (ISI) RGB mode
#define 	AT91C_ISI_RGB_MODE_RGB_888              ((unsigned int) 0x0 << 12) // (ISI) RGB 8:8:8 24 bits
#define 	AT91C_ISI_RGB_MODE_RGB_565              ((unsigned int) 0x1 << 12) // (ISI) RGB 5:6:5 16 bits
#define AT91C_ISI_GRAYSCALE   ((unsigned int) 0x1 << 13) // (ISI) Grayscale Mode
#define 	AT91C_ISI_GRAYSCALE_DISABLE              ((unsigned int) 0x0 << 13) // (ISI) Grayscale mode is disabled
#define 	AT91C_ISI_GRAYSCALE_ENABLE               ((unsigned int) 0x1 << 13) // (ISI) Input image is assumed to be grayscale coded
#define AT91C_ISI_RGB_SWAP    ((unsigned int) 0x1 << 14) // (ISI) RGB Swap
#define 	AT91C_ISI_RGB_SWAP_DISABLE              ((unsigned int) 0x0 << 14) // (ISI) D7 -> R7
#define 	AT91C_ISI_RGB_SWAP_ENABLE               ((unsigned int) 0x1 << 14) // (ISI) D0 -> R7
#define AT91C_ISI_COL_SPACE   ((unsigned int) 0x1 << 15) // (ISI) Color space for the image data
#define 	AT91C_ISI_COL_SPACE_YCBCR                ((unsigned int) 0x0 << 15) // (ISI) YCbCr
#define 	AT91C_ISI_COL_SPACE_RGB                  ((unsigned int) 0x1 << 15) // (ISI) RGB
#define AT91C_ISI_IM_HSIZE    ((unsigned int) 0x7FF << 16) // (ISI) Horizontal size of the Image sensor [0..2047]
#define AT91C_ISI_YCC_SWAP    ((unsigned int) 0x3 << 28) // (ISI) Ycc swap
#define 	AT91C_ISI_YCC_SWAP_YCC_DEFAULT          ((unsigned int) 0x0 << 28) // (ISI) Cb(i) Y(i) Cr(i) Y(i+1)
#define 	AT91C_ISI_YCC_SWAP_YCC_MODE1            ((unsigned int) 0x1 << 28) // (ISI) Cr(i) Y(i) Cb(i) Y(i+1)
#define 	AT91C_ISI_YCC_SWAP_YCC_MODE2            ((unsigned int) 0x2 << 28) // (ISI) Y(i) Cb(i) Y(i+1) Cr(i)
#define 	AT91C_ISI_YCC_SWAP_YCC_MODE3            ((unsigned int) 0x3 << 28) // (ISI) Y(i) Cr(i) Y(i+1) Cb(i)
#define AT91C_ISI_RGB_CFG     ((unsigned int) 0x3 << 30) // (ISI) RGB configuration
#define 	AT91C_ISI_RGB_CFG_RGB_DEFAULT          ((unsigned int) 0x0 << 30) // (ISI) R/G(MSB)  G(LSB)/B  R/G(MSB)  G(LSB)/B
#define 	AT91C_ISI_RGB_CFG_RGB_MODE1            ((unsigned int) 0x1 << 30) // (ISI) B/G(MSB)  G(LSB)/R  B/G(MSB)  G(LSB)/R
#define 	AT91C_ISI_RGB_CFG_RGB_MODE2            ((unsigned int) 0x2 << 30) // (ISI) G(LSB)/R  B/G(MSB)  G(LSB)/R  B/G(MSB)
#define 	AT91C_ISI_RGB_CFG_RGB_MODE3            ((unsigned int) 0x3 << 30) // (ISI) G(LSB)/B  R/G(MSB)  G(LSB)/B  R/G(MSB)
// -------- ISI_PSIZE : (ISI Offset: 0x8) ISI Preview Register -------- 
#define AT91C_ISI_PREV_VSIZE  ((unsigned int) 0x3FF <<  0) // (ISI) Vertical size for the preview path
#define AT91C_ISI_PREV_HSIZE  ((unsigned int) 0x3FF << 16) // (ISI) Horizontal size for the preview path
// -------- ISI_PDECF : (ISI Offset: 0xc) ISI Preview Decimation Factor Register -------- 
#define AT91C_ISI_DEC_FACTOR  ((unsigned int) 0xFF <<  0) // (ISI) Decimation factor
// -------- ISI_Y2RSET0 : (ISI Offset: 0x10) Color Space Conversion YCrCb to RGB Register -------- 
#define AT91C_ISI_Y2R_C0      ((unsigned int) 0xFF <<  0) // (ISI) Color Space Conversion Matrix Coefficient C0
#define AT91C_ISI_Y2R_C1      ((unsigned int) 0xFF <<  8) // (ISI) Color Space Conversion Matrix Coefficient C1
#define AT91C_ISI_Y2R_C2      ((unsigned int) 0xFF << 16) // (ISI) Color Space Conversion Matrix Coefficient C2
#define AT91C_ISI_Y2R_C3      ((unsigned int) 0xFF << 24) // (ISI) Color Space Conversion Matrix Coefficient C3
// -------- ISI_Y2RSET1 : (ISI Offset: 0x14) ISI Color Space Conversion YCrCb to RGB set 1 Register -------- 
#define AT91C_ISI_Y2R_C4      ((unsigned int) 0x1FF <<  0) // (ISI) Color Space Conversion Matrix Coefficient C4
#define AT91C_ISI_Y2R_YOFF    ((unsigned int) 0x1 << 12) // (ISI) Color Space Conversion Luninance default offset
#define 	AT91C_ISI_Y2R_YOFF_0                    ((unsigned int) 0x0 << 12) // (ISI) Offset is 0
#define 	AT91C_ISI_Y2R_YOFF_128                  ((unsigned int) 0x1 << 12) // (ISI) Offset is 128
#define AT91C_ISI_Y2R_CROFF   ((unsigned int) 0x1 << 13) // (ISI) Color Space Conversion Red Chrominance default offset
#define 	AT91C_ISI_Y2R_CROFF_0                    ((unsigned int) 0x0 << 13) // (ISI) Offset is 0
#define 	AT91C_ISI_Y2R_CROFF_16                   ((unsigned int) 0x1 << 13) // (ISI) Offset is 16
#define AT91C_ISI_Y2R_CBOFF   ((unsigned int) 0x1 << 14) // (ISI) Color Space Conversion Blue Chrominance default offset
#define 	AT91C_ISI_Y2R_CBOFF_0                    ((unsigned int) 0x0 << 14) // (ISI) Offset is 0
#define 	AT91C_ISI_Y2R_CBOFF_16                   ((unsigned int) 0x1 << 14) // (ISI) Offset is 16
// -------- ISI_R2YSET0 : (ISI Offset: 0x18) Color Space Conversion RGB to YCrCb set 0 register -------- 
#define AT91C_ISI_R2Y_C0      ((unsigned int) 0x7F <<  0) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C0
#define AT91C_ISI_R2Y_C1      ((unsigned int) 0x7F <<  8) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C1
#define AT91C_ISI_R2Y_C2      ((unsigned int) 0x7F << 16) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C2
#define AT91C_ISI_R2Y_ROFF    ((unsigned int) 0x1 << 24) // (ISI) Color Space Conversion Red component offset
#define 	AT91C_ISI_R2Y_ROFF_0                    ((unsigned int) 0x0 << 24) // (ISI) Offset is 0
#define 	AT91C_ISI_R2Y_ROFF_16                   ((unsigned int) 0x1 << 24) // (ISI) Offset is 16
// -------- ISI_R2YSET1 : (ISI Offset: 0x1c) Color Space Conversion RGB to YCrCb set 1 register -------- 
#define AT91C_ISI_R2Y_C3      ((unsigned int) 0x7F <<  0) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C3
#define AT91C_ISI_R2Y_C4      ((unsigned int) 0x7F <<  8) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C4
#define AT91C_ISI_R2Y_C5      ((unsigned int) 0x7F << 16) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C5
#define AT91C_ISI_R2Y_GOFF    ((unsigned int) 0x1 << 24) // (ISI) Color Space Conversion Green component offset
#define 	AT91C_ISI_R2Y_GOFF_0                    ((unsigned int) 0x0 << 24) // (ISI) Offset is 0
#define 	AT91C_ISI_R2Y_GOFF_128                  ((unsigned int) 0x1 << 24) // (ISI) Offset is 128
// -------- ISI_R2YSET2 : (ISI Offset: 0x20) Color Space Conversion RGB to YCrCb set 2 register -------- 
#define AT91C_ISI_R2Y_C6      ((unsigned int) 0x7F <<  0) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C6
#define AT91C_ISI_R2Y_C7      ((unsigned int) 0x7F <<  8) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C7
#define AT91C_ISI_R2Y_C8      ((unsigned int) 0x7F << 16) // (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C8
#define AT91C_ISI_R2Y_BOFF    ((unsigned int) 0x1 << 24) // (ISI) Color Space Conversion Blue component offset
#define 	AT91C_ISI_R2Y_BOFF_0                    ((unsigned int) 0x0 << 24) // (ISI) Offset is 0
#define 	AT91C_ISI_R2Y_BOFF_128                  ((unsigned int) 0x1 << 24) // (ISI) Offset is 128
// -------- ISI_CTRL : (ISI Offset: 0x24) ISI Control Register -------- 
#define AT91C_ISI_EN          ((unsigned int) 0x1 <<  0) // (ISI) Image Sensor Interface Enable Request
#define 	AT91C_ISI_EN_0                    ((unsigned int) 0x0) // (ISI) No effect
#define 	AT91C_ISI_EN_1                    ((unsigned int) 0x1) // (ISI) Enable the module and the capture
#define AT91C_ISI_DIS         ((unsigned int) 0x1 <<  1) // (ISI) Image Sensor Interface Disable Request
#define 	AT91C_ISI_DIS_0                    ((unsigned int) 0x0 <<  1) // (ISI) No effect
#define 	AT91C_ISI_DIS_1                    ((unsigned int) 0x1 <<  1) // (ISI) Disable the module and the capture
#define AT91C_ISI_SRST        ((unsigned int) 0x1 <<  2) // (ISI) Software Reset Request
#define 	AT91C_ISI_SRST_0                    ((unsigned int) 0x0 <<  2) // (ISI) No effect
#define 	AT91C_ISI_SRST_1                    ((unsigned int) 0x1 <<  2) // (ISI) Reset the module
#define AT91C_ISI_CDC         ((unsigned int) 0x1 <<  8) // (ISI) Codec Request
#define 	AT91C_ISI_CDC_0                    ((unsigned int) 0x0 <<  8) // (ISI) No effect
#define 	AT91C_ISI_CDC_1                    ((unsigned int) 0x1 <<  8) // (ISI) Enable the Codec
// -------- ISI_SR : (ISI Offset: 0x28) ISI Status Register -------- 
#define AT91C_ISI_VSYNC       ((unsigned int) 0x1 << 10) // (ISI) Vertical Synchronization
#define 	AT91C_ISI_VSYNC_0                    ((unsigned int) 0x0 << 10) // (ISI) No effect
#define 	AT91C_ISI_VSYNC_1                    ((unsigned int) 0x1 << 10) // (ISI) Indicates that a Vertical Synchronization has been detected since last read
#define AT91C_ISI_PXFR_DONE   ((unsigned int) 0x1 << 16) // (ISI) Preview DMA transfer terminated
#define 	AT91C_ISI_PXFR_DONE_0                    ((unsigned int) 0x0 << 16) // (ISI) No effect
#define 	AT91C_ISI_PXFR_DONE_1                    ((unsigned int) 0x1 << 16) // (ISI) Indicates that DATA transfer on preview channel has completed since last read
#define AT91C_ISI_CXFR_DONE   ((unsigned int) 0x1 << 17) // (ISI) Codec DMA transfer terminated
#define 	AT91C_ISI_CXFR_DONE_0                    ((unsigned int) 0x0 << 17) // (ISI) No effect
#define 	AT91C_ISI_CXFR_DONE_1                    ((unsigned int) 0x1 << 17) // (ISI) Indicates that DATA transfer on preview channel has completed since last read
#define AT91C_ISI_SIP         ((unsigned int) 0x1 << 19) // (ISI) Synchronization In Progress
#define 	AT91C_ISI_SIP_0                    ((unsigned int) 0x0 << 19) // (ISI) No effect
#define 	AT91C_ISI_SIP_1                    ((unsigned int) 0x1 << 19) // (ISI) Indicates that Synchronization is in progress
#define AT91C_ISI_P_OVR       ((unsigned int) 0x1 << 24) // (ISI) Fifo Preview Overflow 
#define 	AT91C_ISI_P_OVR_0                    ((unsigned int) 0x0 << 24) // (ISI) No error
#define 	AT91C_ISI_P_OVR_1                    ((unsigned int) 0x1 << 24) // (ISI) An overrun condition has occurred in input FIFO on the preview path
#define AT91C_ISI_C_OVR       ((unsigned int) 0x1 << 25) // (ISI) Fifo Codec Overflow 
#define 	AT91C_ISI_C_OVR_0                    ((unsigned int) 0x0 << 25) // (ISI) No error
#define 	AT91C_ISI_C_OVR_1                    ((unsigned int) 0x1 << 25) // (ISI) An overrun condition has occurred in input FIFO on the codec path
#define AT91C_ISI_CRC_ERR     ((unsigned int) 0x1 << 26) // (ISI) CRC synchronisation error
#define 	AT91C_ISI_CRC_ERR_0                    ((unsigned int) 0x0 << 26) // (ISI) No error
#define 	AT91C_ISI_CRC_ERR_1                    ((unsigned int) 0x1 << 26) // (ISI) CRC_SYNC is enabled in the control register and an error has been detected and not corrected. The frame is discarded and the ISI waits for a new one.
#define AT91C_ISI_FR_OVR      ((unsigned int) 0x1 << 27) // (ISI) Frame rate overun
#define 	AT91C_ISI_FR_OVR_0                    ((unsigned int) 0x0 << 27) // (ISI) No error
#define 	AT91C_ISI_FR_OVR_1                    ((unsigned int) 0x1 << 27) // (ISI) Frame overrun, the current frame is being skipped because a vsync signal has been detected while flushing FIFOs.
// -------- ISI_IER : (ISI Offset: 0x2c) ISI Interrupt Enable Register -------- 
// -------- ISI_IDR : (ISI Offset: 0x30) ISI Interrupt Disable Register -------- 
// -------- ISI_IMR : (ISI Offset: 0x34) ISI Interrupt Mask Register -------- 
// -------- ISI_DMACHER : (ISI Offset: 0x38) DMA Channel Enable Register -------- 
#define AT91C_ISI_P_CH_EN     ((unsigned int) 0x1 <<  0) // (ISI) Preview Channel Enable
#define 	AT91C_ISI_P_CH_EN_0                    ((unsigned int) 0x0) // (ISI) No effect
#define 	AT91C_ISI_P_CH_EN_1                    ((unsigned int) 0x1) // (ISI) Enable the Preview Channel
#define AT91C_ISI_C_CH_EN     ((unsigned int) 0x1 <<  1) // (ISI) Codec Channel Enable
#define 	AT91C_ISI_C_CH_EN_0                    ((unsigned int) 0x0 <<  1) // (ISI) No effect
#define 	AT91C_ISI_C_CH_EN_1                    ((unsigned int) 0x1 <<  1) // (ISI) Enable the Codec Channel
// -------- ISI_DMACHDR : (ISI Offset: 0x3c) DMA Channel Enable Register -------- 
#define AT91C_ISI_P_CH_DIS    ((unsigned int) 0x1 <<  0) // (ISI) Preview Channel Disable
#define 	AT91C_ISI_P_CH_DIS_0                    ((unsigned int) 0x0) // (ISI) No effect
#define 	AT91C_ISI_P_CH_DIS_1                    ((unsigned int) 0x1) // (ISI) Disable the Preview Channel
#define AT91C_ISI_C_CH_DIS    ((unsigned int) 0x1 <<  1) // (ISI) Codec Channel Disable
#define 	AT91C_ISI_C_CH_DIS_0                    ((unsigned int) 0x0 <<  1) // (ISI) No effect
#define 	AT91C_ISI_C_CH_DIS_1                    ((unsigned int) 0x1 <<  1) // (ISI) Disable the Codec Channel
// -------- ISI_DMACHSR : (ISI Offset: 0x40) DMA Channel Status Register -------- 
#define AT91C_ISI_P_CH_S      ((unsigned int) 0x1 <<  0) // (ISI) Preview Channel Disable
#define 	AT91C_ISI_P_CH_S_0                    ((unsigned int) 0x0) // (ISI) Preview Channel is disabled
#define 	AT91C_ISI_P_CH_S_1                    ((unsigned int) 0x1) // (ISI) Preview Channel is enabled
#define AT91C_ISI_C_CH_S      ((unsigned int) 0x1 <<  1) // (ISI) Codec Channel Disable
#define 	AT91C_ISI_C_CH_S_0                    ((unsigned int) 0x0 <<  1) // (ISI) Codec Channel is disabled
#define 	AT91C_ISI_C_CH_S_1                    ((unsigned int) 0x1 <<  1) // (ISI) Codec Channel is enabled
// -------- ISI_DMAPCTRL : (ISI Offset: 0x48) DMA Preview Control Register -------- 
#define AT91C_ISI_P_FETCH     ((unsigned int) 0x1 <<  0) // (ISI) Preview Descriptor Fetch Control Field
#define 	AT91C_ISI_P_FETCH_DISABLE              ((unsigned int) 0x0) // (ISI) Preview Channel Fetch Operation is disabled
#define 	AT91C_ISI_P_FETCH_ENABLE               ((unsigned int) 0x1) // (ISI) Preview Channel Fetch Operation is enabled
#define AT91C_ISI_P_DONE      ((unsigned int) 0x1 <<  1) // (ISI) Preview Transfer Done Flag
#define 	AT91C_ISI_P_DONE_0                    ((unsigned int) 0x0 <<  1) // (ISI) Preview Transfer has not been performed
#define 	AT91C_ISI_P_DONE_1                    ((unsigned int) 0x1 <<  1) // (ISI) Preview Transfer has completed
// -------- ISI_DMACCTRL : (ISI Offset: 0x54) DMA Codec Control Register -------- 
#define AT91C_ISI_C_FETCH     ((unsigned int) 0x1 <<  0) // (ISI) Codec Descriptor Fetch Control Field
#define 	AT91C_ISI_C_FETCH_DISABLE              ((unsigned int) 0x0) // (ISI) Codec Channel Fetch Operation is disabled
#define 	AT91C_ISI_C_FETCH_ENABLE               ((unsigned int) 0x1) // (ISI) Codec Channel Fetch Operation is enabled
#define AT91C_ISI_C_DONE      ((unsigned int) 0x1 <<  1) // (ISI) Codec Transfer Done Flag
#define 	AT91C_ISI_C_DONE_0                    ((unsigned int) 0x0 <<  1) // (ISI) Codec Transfer has not been performed
#define 	AT91C_ISI_C_DONE_1                    ((unsigned int) 0x1 <<  1) // (ISI) Codec Transfer has completed
// -------- ISI_WPCR : (ISI Offset: 0xe4) Write Protection Control Register -------- 
#define AT91C_ISI_WP_EN       ((unsigned int) 0x1 <<  0) // (ISI) Write Protection Enable
#define 	AT91C_ISI_WP_EN_DISABLE              ((unsigned int) 0x0) // (ISI) Write Operation is disabled (if WP_KEY corresponds)
#define 	AT91C_ISI_WP_EN_ENABLE               ((unsigned int) 0x1) // (ISI) Write Operation is enabled (if WP_KEY corresponds)
#define AT91C_ISI_WP_KEY      ((unsigned int) 0xFFFFFF <<  8) // (ISI) Write Protection Key
// -------- ISI_WPSR : (ISI Offset: 0xe8) Write Protection Status Register -------- 
#define AT91C_ISI_WP_VS       ((unsigned int) 0xF <<  0) // (ISI) Write Protection Violation Status
#define 	AT91C_ISI_WP_VS_NO_VIOLATION         ((unsigned int) 0x0) // (ISI) No Write Protection Violation detected since last read
#define 	AT91C_ISI_WP_VS_ON_WRITE             ((unsigned int) 0x1) // (ISI) Write Protection Violation detected since last read
#define 	AT91C_ISI_WP_VS_ON_RESET             ((unsigned int) 0x2) // (ISI) Software Reset Violation detected since last read
#define 	AT91C_ISI_WP_VS_ON_BOTH              ((unsigned int) 0x3) // (ISI) Write Protection and Software Reset Violation detected since last read
#define AT91C_ISI_WP_VSRC     ((unsigned int) 0xF <<  8) // (ISI) Write Protection Violation Source
#define 	AT91C_ISI_WP_VSRC_NO_VIOLATION         ((unsigned int) 0x0 <<  8) // (ISI) No Write Protection Violation detected since last read
#define 	AT91C_ISI_WP_VSRC_ISI_CFG1             ((unsigned int) 0x1 <<  8) // (ISI) Write Protection Violation detected on ISI_CFG1 since last read
#define 	AT91C_ISI_WP_VSRC_ISI_CFG2             ((unsigned int) 0x2 <<  8) // (ISI) Write Protection Violation detected on ISI_CFG2 since last read
#define 	AT91C_ISI_WP_VSRC_ISI_PSIZE            ((unsigned int) 0x3 <<  8) // (ISI) Write Protection Violation detected on ISI_PSIZE since last read
#define 	AT91C_ISI_WP_VSRC_ISI_PDECF            ((unsigned int) 0x4 <<  8) // (ISI) Write Protection Violation detected on ISI_PDECF since last read
#define 	AT91C_ISI_WP_VSRC_ISI_Y2RSET0          ((unsigned int) 0x5 <<  8) // (ISI) Write Protection Violation detected on ISI_Y2RSET0 since last read
#define 	AT91C_ISI_WP_VSRC_ISI_Y2RSET1          ((unsigned int) 0x6 <<  8) // (ISI) Write Protection Violation detected on ISI_Y2RSET1 since last read
#define 	AT91C_ISI_WP_VSRC_ISI_R2YSET0          ((unsigned int) 0x7 <<  8) // (ISI) Write Protection Violation detected on ISI_R2YSET0 since last read
#define 	AT91C_ISI_WP_VSRC_ISI_R2YSET1          ((unsigned int) 0x8 <<  8) // (ISI) Write Protection Violation detected on ISI_R2YSET1 since last read
#define 	AT91C_ISI_WP_VSRC_ISI_R2YSET2          ((unsigned int) 0x9 <<  8) // (ISI) Write Protection Violation detected on ISI_R2YSET2 since last read

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR USB Host Interface
// *****************************************************************************
typedef struct _AT91S_UHPHS_OHCI {
	AT91_REG	 UHPHS_OHCI_HcRevision; 	// Revision
	AT91_REG	 UHPHS_OHCI_HcControl; 	// Operating modes for the Host Controller
	AT91_REG	 UHPHS_OHCI_HcCommandStatus; 	// Command & status Register
	AT91_REG	 UHPHS_OHCI_HcInterruptStatus; 	// Interrupt Status Register
	AT91_REG	 UHPHS_OHCI_HcInterruptEnable; 	// Interrupt Enable Register
	AT91_REG	 UHPHS_OHCI_HcInterruptDisable; 	// Interrupt Disable Register
	AT91_REG	 UHPHS_OHCI_HcHCCA; 	// Pointer to the Host Controller Communication Area
	AT91_REG	 UHPHS_OHCI_HcPeriodCurrentED; 	// Current Isochronous or Interrupt Endpoint Descriptor
	AT91_REG	 UHPHS_OHCI_HcControlHeadED; 	// First Endpoint Descriptor of the Control list
	AT91_REG	 UHPHS_OHCI_HcControlCurrentED; 	// Endpoint Control and Status Register
	AT91_REG	 UHPHS_OHCI_HcBulkHeadED; 	// First endpoint register of the Bulk list
	AT91_REG	 UHPHS_OHCI_HcBulkCurrentED; 	// Current endpoint of the Bulk list
	AT91_REG	 UHPHS_OHCI_HcBulkDoneHead; 	// Last completed transfer descriptor
	AT91_REG	 UHPHS_OHCI_HcFmInterval; 	// Bit time between 2 consecutive SOFs
	AT91_REG	 UHPHS_OHCI_HcFmRemaining; 	// Bit time remaining in the current Frame
	AT91_REG	 UHPHS_OHCI_HcFmNumber; 	// Frame number
	AT91_REG	 UHPHS_OHCI_HcPeriodicStart; 	// Periodic Start
	AT91_REG	 UHPHS_OHCI_HcLSThreshold; 	// LS Threshold
	AT91_REG	 UHPHS_OHCI_HcRhDescriptorA; 	// Root Hub characteristics A
	AT91_REG	 UHPHS_OHCI_HcRhDescriptorB; 	// Root Hub characteristics B
	AT91_REG	 UHPHS_OHCI_HcRhStatus; 	// Root Hub Status register
	AT91_REG	 UHPHS_OHCI_HcRhPortStatus[2]; 	// Root Hub Port Status Register
} AT91S_UHPHS_OHCI, *AT91PS_UHPHS_OHCI;


// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR USB Host Interface
// *****************************************************************************
typedef struct _AT91S_UHPHS_EHCI {
	AT91_REG	 UHPHS_EHCI_VERSION; 	// 
	AT91_REG	 UHPHS_EHCI_HCSPARAMS; 	// 
	AT91_REG	 UHPHS_EHCI_HCCPARAMS; 	// 
	AT91_REG	 UHPHS_EHCI_HCSPPORTROUTE; 	// 
	AT91_REG	 UHPHS_EHCI_USBCMD; 	// 
	AT91_REG	 UHPHS_EHCI_USBSTS; 	// 
	AT91_REG	 UHPHS_EHCI_USBINTR; 	// 
	AT91_REG	 UHPHS_EHCI_FRINDEX; 	// 
	AT91_REG	 UHPHS_EHCI_CTRLDSSEGMENT; 	// 
	AT91_REG	 UHPHS_EHCI_PERIODICLISTBASE; 	// 
	AT91_REG	 UHPHS_EHCI_ASYNCLISTADDR; 	// 
	AT91_REG	 Reserved0[9]; 	// 
	AT91_REG	 UHPHS_EHCI_CONFIGFLAG; 	// 
	AT91_REG	 UHPHS_EHCI_PORTSC[2]; 	// 
	AT91_REG	 Reserved1[13]; 	// 
	AT91_REG	 UHPHS_EHCI_INSNREG00; 	// 
	AT91_REG	 UHPHS_EHCI_INSNREG01; 	// 
	AT91_REG	 UHPHS_EHCI_INSNREG02; 	// 
	AT91_REG	 UHPHS_EHCI_INSNREG03; 	// 
	AT91_REG	 UHPHS_EHCI_INSNREG04; 	// 
	AT91_REG	 UHPHS_EHCI_INSNREG05; 	// 
} AT91S_UHPHS_EHCI, *AT91PS_UHPHS_EHCI;

// -------- VERSION : (UHPHS_EHCI Offset: 0x0)  -------- 
#define AT91C_UHPHS_CAPLENGTH ((unsigned int) 0xFF <<  0) // (UHPHS_EHCI) CapLength : Offset for control registers
#define AT91C_UHPHS_HCIVERSION ((unsigned int) 0xFF << 16) // (UHPHS_EHCI) Hci Version

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR True Random Generator
// *****************************************************************************
typedef struct _AT91S_TRNG {
	AT91_REG	 TRNG_CR; 	// Control Register
	AT91_REG	 Reserved0[3]; 	// 
	AT91_REG	 TRNG_IER; 	// Interrupt Enable Register
	AT91_REG	 TRNG_IDR; 	// Interrupt Disable Register
	AT91_REG	 TRNG_IMR; 	// Interrupt Mask Register
	AT91_REG	 TRNG_ISR; 	// Interrupt Status Register
	AT91_REG	 Reserved1[12]; 	// 
	AT91_REG	 TRNG_ODATA; 	// Output Data Register
	AT91_REG	 Reserved2[42]; 	// 
	AT91_REG	 TRNG_VERSION; 	// TRNG Version Register
} AT91S_TRNG, *AT91PS_TRNG;

// -------- TRNG_CR : (TRNG Offset: 0x0) Control Register -------- 
#define AT91C_TRNG_ENABLE     ((unsigned int) 0x1 <<  0) // (TRNG) Enable TRNG
// -------- TRNG_IER : (TRNG Offset: 0x10) Interrupt Enable Register -------- 
#define AT91C_TRNG_DATRDY     ((unsigned int) 0x1 <<  0) // (TRNG) DATRDY
// -------- TRNG_IDR : (TRNG Offset: 0x14) Interrupt Disable Register -------- 
// -------- TRNG_IMR : (TRNG Offset: 0x18) Interrupt Mask Register -------- 
// -------- TRNG_ISR : (TRNG Offset: 0x1c) Interrupt Status Register -------- 

// *****************************************************************************
//               REGISTER ADDRESS DEFINITION FOR AT91SAM9G45
// *****************************************************************************
// ========== Register definition for SFR peripheral ========== 
#define AT91C_SFR_INT   ((AT91_REG *) 	0xFFF74014) // (SFR) OHCI suspend Interrupt status
#define AT91C_SFR_DDRCFG ((AT91_REG *) 	0xFFF74004) // (SFR) DDR2 SSTL18 control
#define AT91C_SFR_UTMICFG ((AT91_REG *) 	0xFFF74010) // (SFR) UTMI Software Reset, and OHCI suspend interrupt control
#define AT91C_SFR_EBIDELAY ((AT91_REG *) 	0xFFF7400C) // (SFR) EBI DDR controller clock delay
#define AT91C_SFR_DDRDELAY ((AT91_REG *) 	0xFFF74008) // (SFR) DDR2 controller clock delay
#define AT91C_SFR_EMA   ((AT91_REG *) 	0xFFF74000) // (SFR) memory Extra Margin Adjustment control
// ========== Register definition for SYS peripheral ========== 
#define AT91C_SYS_SLCKSEL ((AT91_REG *) 	0xFFFFFD50) // (SYS) Slow Clock Selection Register
#define AT91C_SYS_GPBR  ((AT91_REG *) 	0xFFFFFD60) // (SYS) General Purpose Register
// ========== Register definition for EBI peripheral ========== 
#define AT91C_EBI_DUMMY ((AT91_REG *) 	0xFFFFE200) // (EBI) Dummy register - Do not use
// ========== Register definition for DDR2CP1 peripheral ========== 
#define AT91C_DDR2CP1_DELAY2 ((AT91_REG *) 	0xFFFFE434) // (DDR2CP1) Pad delay2 Register
#define AT91C_DDR2CP1_T3PR ((AT91_REG *) 	0xFFFFE418) // (DDR2CP1) Timing3 Register
#define AT91C_DDR2CP1_RTR ((AT91_REG *) 	0xFFFFE404) // (DDR2CP1) Refresh Timer Register
#define AT91C_DDR2CP1_T0PR ((AT91_REG *) 	0xFFFFE40C) // (DDR2CP1) Timing0 Register
#define AT91C_DDR2CP1_WPSR ((AT91_REG *) 	0xFFFFE4E8) // (DDR2CP1) High Speed Register
#define AT91C_DDR2CP1_DELAY8 ((AT91_REG *) 	0xFFFFE44C) // (DDR2CP1) Pad delay8 Register
#define AT91C_DDR2CP1_LPR ((AT91_REG *) 	0xFFFFE41C) // (DDR2CP1) Low-power Register
#define AT91C_DDR2CP1_VER ((AT91_REG *) 	0xFFFFE428) // (DDR2CP1) DLL Version Register
#define AT91C_DDR2CP1_DELAY7 ((AT91_REG *) 	0xFFFFE448) // (DDR2CP1) Pad delay7 Register
#define AT91C_DDR2CP1_CR ((AT91_REG *) 	0xFFFFE408) // (DDR2CP1) Configuration Register
#define AT91C_DDR2CP1_WPCR ((AT91_REG *) 	0xFFFFE4E4) // (DDR2CP1) High Speed Register
#define AT91C_DDR2CP1_MR ((AT91_REG *) 	0xFFFFE400) // (DDR2CP1) Mode Register
#define AT91C_DDR2CP1_DELAY5 ((AT91_REG *) 	0xFFFFE440) // (DDR2CP1) Pad delay5 Register
#define AT91C_DDR2CP1_T2PR ((AT91_REG *) 	0xFFFFE414) // (DDR2CP1) Timing2 Register
#define AT91C_DDR2CP1_HS ((AT91_REG *) 	0xFFFFE42C) // (DDR2CP1) High Speed Register
#define AT91C_DDR2CP1_MDR ((AT91_REG *) 	0xFFFFE420) // (DDR2CP1) Memory Device Register
#define AT91C_DDR2CP1_DELAY4 ((AT91_REG *) 	0xFFFFE43C) // (DDR2CP1) Pad delay4 Register
#define AT91C_DDR2CP1_DELAY1 ((AT91_REG *) 	0xFFFFE430) // (DDR2CP1) Pad delay1 Register
#define AT91C_DDR2CP1_DELAY6 ((AT91_REG *) 	0xFFFFE444) // (DDR2CP1) Pad delay6 Register
#define AT91C_DDR2CP1_DLL ((AT91_REG *) 	0xFFFFE424) // (DDR2CP1) DLL Information Register
#define AT91C_DDR2CP1_DELAY3 ((AT91_REG *) 	0xFFFFE438) // (DDR2CP1) Pad delay3 Register
#define AT91C_DDR2CP1_VERSION ((AT91_REG *) 	0xFFFFE4FC) // (DDR2CP1) Version Register
#define AT91C_DDR2CP1_T1PR ((AT91_REG *) 	0xFFFFE410) // (DDR2CP1) Timing1 Register
// ========== Register definition for DDR2C peripheral ========== 
#define AT91C_DDR2C_DELAY8 ((AT91_REG *) 	0xFFFFE64C) // (DDR2C) Pad delay8 Register
#define AT91C_DDR2C_VER ((AT91_REG *) 	0xFFFFE628) // (DDR2C) DLL Version Register
#define AT91C_DDR2C_RTR ((AT91_REG *) 	0xFFFFE604) // (DDR2C) Refresh Timer Register
#define AT91C_DDR2C_T0PR ((AT91_REG *) 	0xFFFFE60C) // (DDR2C) Timing0 Register
#define AT91C_DDR2C_DELAY5 ((AT91_REG *) 	0xFFFFE640) // (DDR2C) Pad delay5 Register
#define AT91C_DDR2C_LPR ((AT91_REG *) 	0xFFFFE61C) // (DDR2C) Low-power Register
#define AT91C_DDR2C_HS  ((AT91_REG *) 	0xFFFFE62C) // (DDR2C) High Speed Register
#define AT91C_DDR2C_DELAY2 ((AT91_REG *) 	0xFFFFE634) // (DDR2C) Pad delay2 Register
#define AT91C_DDR2C_T2PR ((AT91_REG *) 	0xFFFFE614) // (DDR2C) Timing2 Register
#define AT91C_DDR2C_DELAY1 ((AT91_REG *) 	0xFFFFE630) // (DDR2C) Pad delay1 Register
#define AT91C_DDR2C_T1PR ((AT91_REG *) 	0xFFFFE610) // (DDR2C) Timing1 Register
#define AT91C_DDR2C_MDR ((AT91_REG *) 	0xFFFFE620) // (DDR2C) Memory Device Register
#define AT91C_DDR2C_DELAY6 ((AT91_REG *) 	0xFFFFE644) // (DDR2C) Pad delay6 Register
#define AT91C_DDR2C_VERSION ((AT91_REG *) 	0xFFFFE6FC) // (DDR2C) Version Register
#define AT91C_DDR2C_MR  ((AT91_REG *) 	0xFFFFE600) // (DDR2C) Mode Register
#define AT91C_DDR2C_DLL ((AT91_REG *) 	0xFFFFE624) // (DDR2C) DLL Information Register
#define AT91C_DDR2C_DELAY4 ((AT91_REG *) 	0xFFFFE63C) // (DDR2C) Pad delay4 Register
#define AT91C_DDR2C_WPCR ((AT91_REG *) 	0xFFFFE6E4) // (DDR2C) High Speed Register
#define AT91C_DDR2C_CR  ((AT91_REG *) 	0xFFFFE608) // (DDR2C) Configuration Register
#define AT91C_DDR2C_DELAY3 ((AT91_REG *) 	0xFFFFE638) // (DDR2C) Pad delay3 Register
#define AT91C_DDR2C_WPSR ((AT91_REG *) 	0xFFFFE6E8) // (DDR2C) High Speed Register
#define AT91C_DDR2C_DELAY7 ((AT91_REG *) 	0xFFFFE648) // (DDR2C) Pad delay7 Register
#define AT91C_DDR2C_T3PR ((AT91_REG *) 	0xFFFFE618) // (DDR2C) Timing3 Register
// ========== Register definition for SMC peripheral ========== 
#define AT91C_SMC_PULSE7 ((AT91_REG *) 	0xFFFFE874) // (SMC)  Pulse Register for CS 7
#define AT91C_SMC_DELAY1 ((AT91_REG *) 	0xFFFFE8C0) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CYCLE2 ((AT91_REG *) 	0xFFFFE828) // (SMC)  Cycle Register for CS 2
#define AT91C_SMC_DELAY5 ((AT91_REG *) 	0xFFFFE8D0) // (SMC) SMC Delay Control Register
#define AT91C_SMC_DELAY6 ((AT91_REG *) 	0xFFFFE8D4) // (SMC) SMC Delay Control Register
#define AT91C_SMC_PULSE2 ((AT91_REG *) 	0xFFFFE824) // (SMC)  Pulse Register for CS 2
#define AT91C_SMC_SETUP6 ((AT91_REG *) 	0xFFFFE860) // (SMC)  Setup Register for CS 6
#define AT91C_SMC_SETUP5 ((AT91_REG *) 	0xFFFFE850) // (SMC)  Setup Register for CS 5
#define AT91C_SMC_CYCLE6 ((AT91_REG *) 	0xFFFFE868) // (SMC)  Cycle Register for CS 6
#define AT91C_SMC_PULSE6 ((AT91_REG *) 	0xFFFFE864) // (SMC)  Pulse Register for CS 6
#define AT91C_SMC_CTRL5 ((AT91_REG *) 	0xFFFFE85C) // (SMC)  Control Register for CS 5
#define AT91C_SMC_CTRL3 ((AT91_REG *) 	0xFFFFE83C) // (SMC)  Control Register for CS 3
#define AT91C_SMC_DELAY7 ((AT91_REG *) 	0xFFFFE8D8) // (SMC) SMC Delay Control Register
#define AT91C_SMC_DELAY3 ((AT91_REG *) 	0xFFFFE8C8) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CYCLE0 ((AT91_REG *) 	0xFFFFE808) // (SMC)  Cycle Register for CS 0
#define AT91C_SMC_SETUP1 ((AT91_REG *) 	0xFFFFE810) // (SMC)  Setup Register for CS 1
#define AT91C_SMC_PULSE5 ((AT91_REG *) 	0xFFFFE854) // (SMC)  Pulse Register for CS 5
#define AT91C_SMC_SETUP7 ((AT91_REG *) 	0xFFFFE870) // (SMC)  Setup Register for CS 7
#define AT91C_SMC_CTRL4 ((AT91_REG *) 	0xFFFFE84C) // (SMC)  Control Register for CS 4
#define AT91C_SMC_DELAY2 ((AT91_REG *) 	0xFFFFE8C4) // (SMC) SMC Delay Control Register
#define AT91C_SMC_PULSE3 ((AT91_REG *) 	0xFFFFE834) // (SMC)  Pulse Register for CS 3
#define AT91C_SMC_CYCLE4 ((AT91_REG *) 	0xFFFFE848) // (SMC)  Cycle Register for CS 4
#define AT91C_SMC_CTRL1 ((AT91_REG *) 	0xFFFFE81C) // (SMC)  Control Register for CS 1
#define AT91C_SMC_SETUP3 ((AT91_REG *) 	0xFFFFE830) // (SMC)  Setup Register for CS 3
#define AT91C_SMC_CTRL0 ((AT91_REG *) 	0xFFFFE80C) // (SMC)  Control Register for CS 0
#define AT91C_SMC_CYCLE7 ((AT91_REG *) 	0xFFFFE878) // (SMC)  Cycle Register for CS 7
#define AT91C_SMC_DELAY4 ((AT91_REG *) 	0xFFFFE8CC) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CYCLE1 ((AT91_REG *) 	0xFFFFE818) // (SMC)  Cycle Register for CS 1
#define AT91C_SMC_SETUP2 ((AT91_REG *) 	0xFFFFE820) // (SMC)  Setup Register for CS 2
#define AT91C_SMC_PULSE1 ((AT91_REG *) 	0xFFFFE814) // (SMC)  Pulse Register for CS 1
#define AT91C_SMC_DELAY8 ((AT91_REG *) 	0xFFFFE8DC) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CTRL2 ((AT91_REG *) 	0xFFFFE82C) // (SMC)  Control Register for CS 2
#define AT91C_SMC_PULSE4 ((AT91_REG *) 	0xFFFFE844) // (SMC)  Pulse Register for CS 4
#define AT91C_SMC_SETUP4 ((AT91_REG *) 	0xFFFFE840) // (SMC)  Setup Register for CS 4
#define AT91C_SMC_CYCLE3 ((AT91_REG *) 	0xFFFFE838) // (SMC)  Cycle Register for CS 3
#define AT91C_SMC_SETUP0 ((AT91_REG *) 	0xFFFFE800) // (SMC)  Setup Register for CS 0
#define AT91C_SMC_CYCLE5 ((AT91_REG *) 	0xFFFFE858) // (SMC)  Cycle Register for CS 5
#define AT91C_SMC_PULSE0 ((AT91_REG *) 	0xFFFFE804) // (SMC)  Pulse Register for CS 0
#define AT91C_SMC_CTRL6 ((AT91_REG *) 	0xFFFFE86C) // (SMC)  Control Register for CS 6
#define AT91C_SMC_CTRL7 ((AT91_REG *) 	0xFFFFE87C) // (SMC)  Control Register for CS 7
// ========== Register definition for MATRIX peripheral ========== 
#define AT91C_MATRIX_SCFG1 ((AT91_REG *) 	0xFFFFEA44) // (MATRIX)  Slave Configuration Register 1 : SRAM S1
#define AT91C_MATRIX_MRCR ((AT91_REG *) 	0xFFFFEB00) // (MATRIX)  Master Remap Control Register 
#define AT91C_MATRIX_PRAS2 ((AT91_REG *) 	0xFFFFEA90) // (MATRIX)  PRAS2 : SRAM S2
#define AT91C_MATRIX_PRAS1 ((AT91_REG *) 	0xFFFFEA88) // (MATRIX)  PRAS1 : SRAM S1
#define AT91C_MATRIX_PRAS0 ((AT91_REG *) 	0xFFFFEA80) // (MATRIX)  PRAS0 : SRAM S0
#define AT91C_MATRIX_MCFG8 ((AT91_REG *) 	0xFFFFEA20) // (MATRIX)  Master Configuration Register 8 : eMAC
#define AT91C_MATRIX_MCFG2 ((AT91_REG *) 	0xFFFFEA08) // (MATRIX)  Master Configuration Register 2 : pdc
#define AT91C_MATRIX_EBICSA ((AT91_REG *) 	0xFFFFEB28) // (MATRIX)  EBI Chip Select Assignment Register 
#define AT91C_MATRIX_PRAS4 ((AT91_REG *) 	0xFFFFEAA0) // (MATRIX)  PRAS4 : ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
#define AT91C_MATRIX_MCFG3 ((AT91_REG *) 	0xFFFFEA0C) // (MATRIX)  Master Configuration Register 3 : USB Host OHCI
#define AT91C_MATRIX_SCFG0 ((AT91_REG *) 	0xFFFFEA40) // (MATRIX)  Slave Configuration Register 0 : SRAM S0
#define AT91C_MATRIX_MCFG7 ((AT91_REG *) 	0xFFFFEA1C) // (MATRIX)  Master Configuration Register 7 : lcdc
#define AT91C_MATRIX_PRAS6 ((AT91_REG *) 	0xFFFFEAB0) // (MATRIX)  PRAS6 : DDR2 S1
#define AT91C_MATRIX_SCFG7 ((AT91_REG *) 	0xFFFFEA5C) // (MATRIX)  Slave Configuration Register 7 : DDR2 S2
#define AT91C_MATRIX_PRAS7 ((AT91_REG *) 	0xFFFFEAB8) // (MATRIX)  PRAS7 : DDR2 S2
#define AT91C_MATRIX_SCFG2 ((AT91_REG *) 	0xFFFFEA48) // (MATRIX)  Slave Configuration Register 2 : SRAM S2
#define AT91C_MATRIX_WRPROTST ((AT91_REG *) 	0xFFFFEBE8) // (MATRIX)  Write Protection Status Register 
#define AT91C_MATRIX_PRBS0 ((AT91_REG *) 	0xFFFFEA84) // (MATRIX)  PRBS0 : SRAM S0
#define AT91C_MATRIX_PRBS2 ((AT91_REG *) 	0xFFFFEA94) // (MATRIX)  PRBS2 : SRAM S2
#define AT91C_MATRIX_MCFG4 ((AT91_REG *) 	0xFFFFEA10) // (MATRIX)  Master Configuration Register 4 : DMA0
#define AT91C_MATRIX_SCFG5 ((AT91_REG *) 	0xFFFFEA54) // (MATRIX)  Slave Configuration Register 5 : DDR2 S0
#define AT91C_MATRIX_PRBS6 ((AT91_REG *) 	0xFFFFEAB4) // (MATRIX)  PRBS6 : DDR2 S1
#define AT91C_MATRIX_MCFG1 ((AT91_REG *) 	0xFFFFEA04) // (MATRIX)  Master Configuration Register 1 ; ARM-D
#define AT91C_MATRIX_SCFG6 ((AT91_REG *) 	0xFFFFEA58) // (MATRIX)  Slave Configuration Register 6 : DDR2 S1
#define AT91C_MATRIX_SCFG4 ((AT91_REG *) 	0xFFFFEA50) // (MATRIX)  Slave Configuration Register 4 ; ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
#define AT91C_MATRIX_PRBS1 ((AT91_REG *) 	0xFFFFEA8C) // (MATRIX)  PRBS1 : SRAM S1
#define AT91C_MATRIX_PRBS3 ((AT91_REG *) 	0xFFFFEA9C) // (MATRIX)  PRBS3 : SRAM S3
#define AT91C_MATRIX_WRPROTEN ((AT91_REG *) 	0xFFFFEBE4) // (MATRIX)  Write Protection Control Register 
#define AT91C_MATRIX_TCMR ((AT91_REG *) 	0xFFFFEB10) // (MATRIX)  Bus Matrix TCM Configuration Register 
#define AT91C_MATRIX_MCFG0 ((AT91_REG *) 	0xFFFFEA00) // (MATRIX)  Master Configuration Register 0 : ARM-I
#define AT91C_MATRIX_PRAS5 ((AT91_REG *) 	0xFFFFEAA8) // (MATRIX)  PRAS5 : DDR2 S0
#define AT91C_MATRIX_DDRMPR ((AT91_REG *) 	0xFFFFEB18) // (MATRIX)  DDR Multi-Port Register 
#define AT91C_MATRIX_PRBS4 ((AT91_REG *) 	0xFFFFEAA4) // (MATRIX)  PRBS4 : ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
#define AT91C_MATRIX_MCFG6 ((AT91_REG *) 	0xFFFFEA18) // (MATRIX)  Master Configuration Register 6 : hisi
#define AT91C_MATRIX_PRBS5 ((AT91_REG *) 	0xFFFFEAAC) // (MATRIX)  PRBS5 : DDR2 S0
#define AT91C_MATRIX_MCFG9 ((AT91_REG *) 	0xFFFFEA24) // (MATRIX)  Master Configuration Register 9 : USB Device
#define AT91C_MATRIX_MCFG5 ((AT91_REG *) 	0xFFFFEA14) // (MATRIX)  Master Configuration Register 5 : DMA1
#define AT91C_MATRIX_PRBS7 ((AT91_REG *) 	0xFFFFEABC) // (MATRIX)  PRBS7 : DDR2 S2
#define AT91C_MATRIX_MCFG10 ((AT91_REG *) 	0xFFFFEA28) // (MATRIX)  Master Configuration Register 10 : USB Host EHCI
#define AT91C_MATRIX_PRAS3 ((AT91_REG *) 	0xFFFFEA98) // (MATRIX)  PRAS3 : SRAM S3
#define AT91C_MATRIX_SCFG3 ((AT91_REG *) 	0xFFFFEA4C) // (MATRIX)  Slave Configuration Register 3 : SRAM S3
// ========== Register definition for AIC peripheral ========== 
#define AT91C_AIC_IVR   ((AT91_REG *) 	0xFFFFF100) // (AIC) IRQ Vector Register
#define AT91C_AIC_SMR   ((AT91_REG *) 	0xFFFFF000) // (AIC) Source Mode Register
#define AT91C_AIC_FVR   ((AT91_REG *) 	0xFFFFF104) // (AIC) FIQ Vector Register
#define AT91C_AIC_DCR   ((AT91_REG *) 	0xFFFFF138) // (AIC) Debug Control Register (Protect)
#define AT91C_AIC_EOICR ((AT91_REG *) 	0xFFFFF130) // (AIC) End of Interrupt Command Register
#define AT91C_AIC_SVR   ((AT91_REG *) 	0xFFFFF080) // (AIC) Source Vector Register
#define AT91C_AIC_FFSR  ((AT91_REG *) 	0xFFFFF148) // (AIC) Fast Forcing Status Register
#define AT91C_AIC_ICCR  ((AT91_REG *) 	0xFFFFF128) // (AIC) Interrupt Clear Command Register
#define AT91C_AIC_ISR   ((AT91_REG *) 	0xFFFFF108) // (AIC) Interrupt Status Register
#define AT91C_AIC_IMR   ((AT91_REG *) 	0xFFFFF110) // (AIC) Interrupt Mask Register
#define AT91C_AIC_IPR   ((AT91_REG *) 	0xFFFFF10C) // (AIC) Interrupt Pending Register
#define AT91C_AIC_FFER  ((AT91_REG *) 	0xFFFFF140) // (AIC) Fast Forcing Enable Register
#define AT91C_AIC_IECR  ((AT91_REG *) 	0xFFFFF120) // (AIC) Interrupt Enable Command Register
#define AT91C_AIC_ISCR  ((AT91_REG *) 	0xFFFFF12C) // (AIC) Interrupt Set Command Register
#define AT91C_AIC_FFDR  ((AT91_REG *) 	0xFFFFF144) // (AIC) Fast Forcing Disable Register
#define AT91C_AIC_CISR  ((AT91_REG *) 	0xFFFFF114) // (AIC) Core Interrupt Status Register
#define AT91C_AIC_IDCR  ((AT91_REG *) 	0xFFFFF124) // (AIC) Interrupt Disable Command Register
#define AT91C_AIC_SPU   ((AT91_REG *) 	0xFFFFF134) // (AIC) Spurious Vector Register
// ========== Register definition for PDC_DBGU peripheral ========== 
#define AT91C_DBGU_PTCR ((AT91_REG *) 	0xFFFFEF20) // (PDC_DBGU) PDC Transfer Control Register
#define AT91C_DBGU_RCR  ((AT91_REG *) 	0xFFFFEF04) // (PDC_DBGU) Receive Counter Register
#define AT91C_DBGU_TCR  ((AT91_REG *) 	0xFFFFEF0C) // (PDC_DBGU) Transmit Counter Register
#define AT91C_DBGU_RNCR ((AT91_REG *) 	0xFFFFEF14) // (PDC_DBGU) Receive Next Counter Register
#define AT91C_DBGU_TNPR ((AT91_REG *) 	0xFFFFEF18) // (PDC_DBGU) Transmit Next Pointer Register
#define AT91C_DBGU_RNPR ((AT91_REG *) 	0xFFFFEF10) // (PDC_DBGU) Receive Next Pointer Register
#define AT91C_DBGU_PTSR ((AT91_REG *) 	0xFFFFEF24) // (PDC_DBGU) PDC Transfer Status Register
#define AT91C_DBGU_RPR  ((AT91_REG *) 	0xFFFFEF00) // (PDC_DBGU) Receive Pointer Register
#define AT91C_DBGU_TPR  ((AT91_REG *) 	0xFFFFEF08) // (PDC_DBGU) Transmit Pointer Register
#define AT91C_DBGU_TNCR ((AT91_REG *) 	0xFFFFEF1C) // (PDC_DBGU) Transmit Next Counter Register
// ========== Register definition for DBGU peripheral ========== 
#define AT91C_DBGU_BRGR ((AT91_REG *) 	0xFFFFEE20) // (DBGU) Baud Rate Generator Register
#define AT91C_DBGU_CR   ((AT91_REG *) 	0xFFFFEE00) // (DBGU) Control Register
#define AT91C_DBGU_THR  ((AT91_REG *) 	0xFFFFEE1C) // (DBGU) Transmitter Holding Register
#define AT91C_DBGU_IDR  ((AT91_REG *) 	0xFFFFEE0C) // (DBGU) Interrupt Disable Register
#define AT91C_DBGU_EXID ((AT91_REG *) 	0xFFFFEE44) // (DBGU) Chip ID Extension Register
#define AT91C_DBGU_IMR  ((AT91_REG *) 	0xFFFFEE10) // (DBGU) Interrupt Mask Register
#define AT91C_DBGU_FNTR ((AT91_REG *) 	0xFFFFEE48) // (DBGU) Force NTRST Register
#define AT91C_DBGU_IER  ((AT91_REG *) 	0xFFFFEE08) // (DBGU) Interrupt Enable Register
#define AT91C_DBGU_CSR  ((AT91_REG *) 	0xFFFFEE14) // (DBGU) Channel Status Register
#define AT91C_DBGU_MR   ((AT91_REG *) 	0xFFFFEE04) // (DBGU) Mode Register
#define AT91C_DBGU_RHR  ((AT91_REG *) 	0xFFFFEE18) // (DBGU) Receiver Holding Register
#define AT91C_DBGU_CIDR ((AT91_REG *) 	0xFFFFEE40) // (DBGU) Chip ID Register
// ========== Register definition for PIOA peripheral ========== 
#define AT91C_PIOA_OWDR ((AT91_REG *) 	0xFFFFF2A4) // (PIOA) Output Write Disable Register
#define AT91C_PIOA_DELAY3 ((AT91_REG *) 	0xFFFFF2C8) // (PIOA) PIO Delay Control Register
#define AT91C_PIOA_ISR  ((AT91_REG *) 	0xFFFFF24C) // (PIOA) Interrupt Status Register
#define AT91C_PIOA_PDR  ((AT91_REG *) 	0xFFFFF204) // (PIOA) PIO Disable Register
#define AT91C_PIOA_OSR  ((AT91_REG *) 	0xFFFFF218) // (PIOA) Output Status Register
#define AT91C_PIOA_ABSR ((AT91_REG *) 	0xFFFFF278) // (PIOA) AB Select Status Register
#define AT91C_PIOA_DELAY2 ((AT91_REG *) 	0xFFFFF2C4) // (PIOA) PIO Delay Control Register
#define AT91C_PIOA_PDSR ((AT91_REG *) 	0xFFFFF23C) // (PIOA) Pin Data Status Register
#define AT91C_PIOA_BSR  ((AT91_REG *) 	0xFFFFF274) // (PIOA) Select B Register
#define AT91C_PIOA_DELAY1 ((AT91_REG *) 	0xFFFFF2C0) // (PIOA) PIO Delay Control Register
#define AT91C_PIOA_PPUER ((AT91_REG *) 	0xFFFFF264) // (PIOA) Pull-up Enable Register
#define AT91C_PIOA_OER  ((AT91_REG *) 	0xFFFFF210) // (PIOA) Output Enable Register
#define AT91C_PIOA_PER  ((AT91_REG *) 	0xFFFFF200) // (PIOA) PIO Enable Register
#define AT91C_PIOA_VERSION ((AT91_REG *) 	0xFFFFF2FC) // (PIOA) PIO Version Register
#define AT91C_PIOA_PPUDR ((AT91_REG *) 	0xFFFFF260) // (PIOA) Pull-up Disable Register
#define AT91C_PIOA_ODSR ((AT91_REG *) 	0xFFFFF238) // (PIOA) Output Data Status Register
#define AT91C_PIOA_SLEWRATE1 ((AT91_REG *) 	0xFFFFF2B0) // (PIOA) PIO Slewrate Control Register
#define AT91C_PIOA_MDDR ((AT91_REG *) 	0xFFFFF254) // (PIOA) Multi-driver Disable Register
#define AT91C_PIOA_IFSR ((AT91_REG *) 	0xFFFFF228) // (PIOA) Input Filter Status Register
#define AT91C_PIOA_CODR ((AT91_REG *) 	0xFFFFF234) // (PIOA) Clear Output Data Register
#define AT91C_PIOA_ASR  ((AT91_REG *) 	0xFFFFF270) // (PIOA) Select A Register
#define AT91C_PIOA_OWSR ((AT91_REG *) 	0xFFFFF2A8) // (PIOA) Output Write Status Register
#define AT91C_PIOA_IMR  ((AT91_REG *) 	0xFFFFF248) // (PIOA) Interrupt Mask Register
#define AT91C_PIOA_PPUSR ((AT91_REG *) 	0xFFFFF268) // (PIOA) Pull-up Status Register
#define AT91C_PIOA_MDER ((AT91_REG *) 	0xFFFFF250) // (PIOA) Multi-driver Enable Register
#define AT91C_PIOA_IFDR ((AT91_REG *) 	0xFFFFF224) // (PIOA) Input Filter Disable Register
#define AT91C_PIOA_SODR ((AT91_REG *) 	0xFFFFF230) // (PIOA) Set Output Data Register
#define AT91C_PIOA_OWER ((AT91_REG *) 	0xFFFFF2A0) // (PIOA) Output Write Enable Register
#define AT91C_PIOA_IDR  ((AT91_REG *) 	0xFFFFF244) // (PIOA) Interrupt Disable Register
#define AT91C_PIOA_IFER ((AT91_REG *) 	0xFFFFF220) // (PIOA) Input Filter Enable Register
#define AT91C_PIOA_IER  ((AT91_REG *) 	0xFFFFF240) // (PIOA) Interrupt Enable Register
#define AT91C_PIOA_ODR  ((AT91_REG *) 	0xFFFFF214) // (PIOA) Output Disable Registerr
#define AT91C_PIOA_MDSR ((AT91_REG *) 	0xFFFFF258) // (PIOA) Multi-driver Status Register
#define AT91C_PIOA_DELAY4 ((AT91_REG *) 	0xFFFFF2CC) // (PIOA) PIO Delay Control Register
#define AT91C_PIOA_PSR  ((AT91_REG *) 	0xFFFFF208) // (PIOA) PIO Status Register
// ========== Register definition for PIOB peripheral ========== 
#define AT91C_PIOB_ODR  ((AT91_REG *) 	0xFFFFF414) // (PIOB) Output Disable Registerr
#define AT91C_PIOB_DELAY4 ((AT91_REG *) 	0xFFFFF4CC) // (PIOB) PIO Delay Control Register
#define AT91C_PIOB_SODR ((AT91_REG *) 	0xFFFFF430) // (PIOB) Set Output Data Register
#define AT91C_PIOB_ISR  ((AT91_REG *) 	0xFFFFF44C) // (PIOB) Interrupt Status Register
#define AT91C_PIOB_ABSR ((AT91_REG *) 	0xFFFFF478) // (PIOB) AB Select Status Register
#define AT91C_PIOB_IMR  ((AT91_REG *) 	0xFFFFF448) // (PIOB) Interrupt Mask Register
#define AT91C_PIOB_MDSR ((AT91_REG *) 	0xFFFFF458) // (PIOB) Multi-driver Status Register
#define AT91C_PIOB_PPUSR ((AT91_REG *) 	0xFFFFF468) // (PIOB) Pull-up Status Register
#define AT91C_PIOB_PDSR ((AT91_REG *) 	0xFFFFF43C) // (PIOB) Pin Data Status Register
#define AT91C_PIOB_DELAY3 ((AT91_REG *) 	0xFFFFF4C8) // (PIOB) PIO Delay Control Register
#define AT91C_PIOB_MDDR ((AT91_REG *) 	0xFFFFF454) // (PIOB) Multi-driver Disable Register
#define AT91C_PIOB_CODR ((AT91_REG *) 	0xFFFFF434) // (PIOB) Clear Output Data Register
#define AT91C_PIOB_MDER ((AT91_REG *) 	0xFFFFF450) // (PIOB) Multi-driver Enable Register
#define AT91C_PIOB_PDR  ((AT91_REG *) 	0xFFFFF404) // (PIOB) PIO Disable Register
#define AT91C_PIOB_IFSR ((AT91_REG *) 	0xFFFFF428) // (PIOB) Input Filter Status Register
#define AT91C_PIOB_PSR  ((AT91_REG *) 	0xFFFFF408) // (PIOB) PIO Status Register
#define AT91C_PIOB_SLEWRATE1 ((AT91_REG *) 	0xFFFFF4B0) // (PIOB) PIO Slewrate Control Register
#define AT91C_PIOB_IER  ((AT91_REG *) 	0xFFFFF440) // (PIOB) Interrupt Enable Register
#define AT91C_PIOB_PPUDR ((AT91_REG *) 	0xFFFFF460) // (PIOB) Pull-up Disable Register
#define AT91C_PIOB_PER  ((AT91_REG *) 	0xFFFFF400) // (PIOB) PIO Enable Register
#define AT91C_PIOB_IFDR ((AT91_REG *) 	0xFFFFF424) // (PIOB) Input Filter Disable Register
#define AT91C_PIOB_IDR  ((AT91_REG *) 	0xFFFFF444) // (PIOB) Interrupt Disable Register
#define AT91C_PIOB_OWDR ((AT91_REG *) 	0xFFFFF4A4) // (PIOB) Output Write Disable Register
#define AT91C_PIOB_ODSR ((AT91_REG *) 	0xFFFFF438) // (PIOB) Output Data Status Register
#define AT91C_PIOB_DELAY2 ((AT91_REG *) 	0xFFFFF4C4) // (PIOB) PIO Delay Control Register
#define AT91C_PIOB_OWSR ((AT91_REG *) 	0xFFFFF4A8) // (PIOB) Output Write Status Register
#define AT91C_PIOB_BSR  ((AT91_REG *) 	0xFFFFF474) // (PIOB) Select B Register
#define AT91C_PIOB_IFER ((AT91_REG *) 	0xFFFFF420) // (PIOB) Input Filter Enable Register
#define AT91C_PIOB_OWER ((AT91_REG *) 	0xFFFFF4A0) // (PIOB) Output Write Enable Register
#define AT91C_PIOB_PPUER ((AT91_REG *) 	0xFFFFF464) // (PIOB) Pull-up Enable Register
#define AT91C_PIOB_OSR  ((AT91_REG *) 	0xFFFFF418) // (PIOB) Output Status Register
#define AT91C_PIOB_ASR  ((AT91_REG *) 	0xFFFFF470) // (PIOB) Select A Register
#define AT91C_PIOB_OER  ((AT91_REG *) 	0xFFFFF410) // (PIOB) Output Enable Register
#define AT91C_PIOB_VERSION ((AT91_REG *) 	0xFFFFF4FC) // (PIOB) PIO Version Register
#define AT91C_PIOB_DELAY1 ((AT91_REG *) 	0xFFFFF4C0) // (PIOB) PIO Delay Control Register
// ========== Register definition for PIOC peripheral ========== 
#define AT91C_PIOC_OWDR ((AT91_REG *) 	0xFFFFF6A4) // (PIOC) Output Write Disable Register
#define AT91C_PIOC_IMR  ((AT91_REG *) 	0xFFFFF648) // (PIOC) Interrupt Mask Register
#define AT91C_PIOC_ASR  ((AT91_REG *) 	0xFFFFF670) // (PIOC) Select A Register
#define AT91C_PIOC_PPUDR ((AT91_REG *) 	0xFFFFF660) // (PIOC) Pull-up Disable Register
#define AT91C_PIOC_CODR ((AT91_REG *) 	0xFFFFF634) // (PIOC) Clear Output Data Register
#define AT91C_PIOC_OWER ((AT91_REG *) 	0xFFFFF6A0) // (PIOC) Output Write Enable Register
#define AT91C_PIOC_ABSR ((AT91_REG *) 	0xFFFFF678) // (PIOC) AB Select Status Register
#define AT91C_PIOC_IFDR ((AT91_REG *) 	0xFFFFF624) // (PIOC) Input Filter Disable Register
#define AT91C_PIOC_VERSION ((AT91_REG *) 	0xFFFFF6FC) // (PIOC) PIO Version Register
#define AT91C_PIOC_ODR  ((AT91_REG *) 	0xFFFFF614) // (PIOC) Output Disable Registerr
#define AT91C_PIOC_PPUER ((AT91_REG *) 	0xFFFFF664) // (PIOC) Pull-up Enable Register
#define AT91C_PIOC_SODR ((AT91_REG *) 	0xFFFFF630) // (PIOC) Set Output Data Register
#define AT91C_PIOC_ISR  ((AT91_REG *) 	0xFFFFF64C) // (PIOC) Interrupt Status Register
#define AT91C_PIOC_OSR  ((AT91_REG *) 	0xFFFFF618) // (PIOC) Output Status Register
#define AT91C_PIOC_MDSR ((AT91_REG *) 	0xFFFFF658) // (PIOC) Multi-driver Status Register
#define AT91C_PIOC_IFER ((AT91_REG *) 	0xFFFFF620) // (PIOC) Input Filter Enable Register
#define AT91C_PIOC_DELAY2 ((AT91_REG *) 	0xFFFFF6C4) // (PIOC) PIO Delay Control Register
#define AT91C_PIOC_MDER ((AT91_REG *) 	0xFFFFF650) // (PIOC) Multi-driver Enable Register
#define AT91C_PIOC_PPUSR ((AT91_REG *) 	0xFFFFF668) // (PIOC) Pull-up Status Register
#define AT91C_PIOC_PSR  ((AT91_REG *) 	0xFFFFF608) // (PIOC) PIO Status Register
#define AT91C_PIOC_DELAY4 ((AT91_REG *) 	0xFFFFF6CC) // (PIOC) PIO Delay Control Register
#define AT91C_PIOC_DELAY3 ((AT91_REG *) 	0xFFFFF6C8) // (PIOC) PIO Delay Control Register
#define AT91C_PIOC_IER  ((AT91_REG *) 	0xFFFFF640) // (PIOC) Interrupt Enable Register
#define AT91C_PIOC_SLEWRATE1 ((AT91_REG *) 	0xFFFFF6B0) // (PIOC) PIO Slewrate Control Register
#define AT91C_PIOC_IDR  ((AT91_REG *) 	0xFFFFF644) // (PIOC) Interrupt Disable Register
#define AT91C_PIOC_PDSR ((AT91_REG *) 	0xFFFFF63C) // (PIOC) Pin Data Status Register
#define AT91C_PIOC_DELAY1 ((AT91_REG *) 	0xFFFFF6C0) // (PIOC) PIO Delay Control Register
#define AT91C_PIOC_PDR  ((AT91_REG *) 	0xFFFFF604) // (PIOC) PIO Disable Register
#define AT91C_PIOC_OWSR ((AT91_REG *) 	0xFFFFF6A8) // (PIOC) Output Write Status Register
#define AT91C_PIOC_IFSR ((AT91_REG *) 	0xFFFFF628) // (PIOC) Input Filter Status Register
#define AT91C_PIOC_ODSR ((AT91_REG *) 	0xFFFFF638) // (PIOC) Output Data Status Register
#define AT91C_PIOC_OER  ((AT91_REG *) 	0xFFFFF610) // (PIOC) Output Enable Register
#define AT91C_PIOC_MDDR ((AT91_REG *) 	0xFFFFF654) // (PIOC) Multi-driver Disable Register
#define AT91C_PIOC_BSR  ((AT91_REG *) 	0xFFFFF674) // (PIOC) Select B Register
#define AT91C_PIOC_PER  ((AT91_REG *) 	0xFFFFF600) // (PIOC) PIO Enable Register
// ========== Register definition for PIOD peripheral ========== 
#define AT91C_PIOD_DELAY1 ((AT91_REG *) 	0xFFFFF8C0) // (PIOD) PIO Delay Control Register
#define AT91C_PIOD_OWDR ((AT91_REG *) 	0xFFFFF8A4) // (PIOD) Output Write Disable Register
#define AT91C_PIOD_SODR ((AT91_REG *) 	0xFFFFF830) // (PIOD) Set Output Data Register
#define AT91C_PIOD_PPUER ((AT91_REG *) 	0xFFFFF864) // (PIOD) Pull-up Enable Register
#define AT91C_PIOD_CODR ((AT91_REG *) 	0xFFFFF834) // (PIOD) Clear Output Data Register
#define AT91C_PIOD_DELAY4 ((AT91_REG *) 	0xFFFFF8CC) // (PIOD) PIO Delay Control Register
#define AT91C_PIOD_PSR  ((AT91_REG *) 	0xFFFFF808) // (PIOD) PIO Status Register
#define AT91C_PIOD_PDR  ((AT91_REG *) 	0xFFFFF804) // (PIOD) PIO Disable Register
#define AT91C_PIOD_ODR  ((AT91_REG *) 	0xFFFFF814) // (PIOD) Output Disable Registerr
#define AT91C_PIOD_PPUSR ((AT91_REG *) 	0xFFFFF868) // (PIOD) Pull-up Status Register
#define AT91C_PIOD_IFSR ((AT91_REG *) 	0xFFFFF828) // (PIOD) Input Filter Status Register
#define AT91C_PIOD_IMR  ((AT91_REG *) 	0xFFFFF848) // (PIOD) Interrupt Mask Register
#define AT91C_PIOD_ASR  ((AT91_REG *) 	0xFFFFF870) // (PIOD) Select A Register
#define AT91C_PIOD_DELAY2 ((AT91_REG *) 	0xFFFFF8C4) // (PIOD) PIO Delay Control Register
#define AT91C_PIOD_OWSR ((AT91_REG *) 	0xFFFFF8A8) // (PIOD) Output Write Status Register
#define AT91C_PIOD_PER  ((AT91_REG *) 	0xFFFFF800) // (PIOD) PIO Enable Register
#define AT91C_PIOD_MDER ((AT91_REG *) 	0xFFFFF850) // (PIOD) Multi-driver Enable Register
#define AT91C_PIOD_PDSR ((AT91_REG *) 	0xFFFFF83C) // (PIOD) Pin Data Status Register
#define AT91C_PIOD_MDSR ((AT91_REG *) 	0xFFFFF858) // (PIOD) Multi-driver Status Register
#define AT91C_PIOD_OWER ((AT91_REG *) 	0xFFFFF8A0) // (PIOD) Output Write Enable Register
#define AT91C_PIOD_BSR  ((AT91_REG *) 	0xFFFFF874) // (PIOD) Select B Register
#define AT91C_PIOD_IFDR ((AT91_REG *) 	0xFFFFF824) // (PIOD) Input Filter Disable Register
#define AT91C_PIOD_DELAY3 ((AT91_REG *) 	0xFFFFF8C8) // (PIOD) PIO Delay Control Register
#define AT91C_PIOD_ABSR ((AT91_REG *) 	0xFFFFF878) // (PIOD) AB Select Status Register
#define AT91C_PIOD_OER  ((AT91_REG *) 	0xFFFFF810) // (PIOD) Output Enable Register
#define AT91C_PIOD_MDDR ((AT91_REG *) 	0xFFFFF854) // (PIOD) Multi-driver Disable Register
#define AT91C_PIOD_IDR  ((AT91_REG *) 	0xFFFFF844) // (PIOD) Interrupt Disable Register
#define AT91C_PIOD_IER  ((AT91_REG *) 	0xFFFFF840) // (PIOD) Interrupt Enable Register
#define AT91C_PIOD_PPUDR ((AT91_REG *) 	0xFFFFF860) // (PIOD) Pull-up Disable Register
#define AT91C_PIOD_VERSION ((AT91_REG *) 	0xFFFFF8FC) // (PIOD) PIO Version Register
#define AT91C_PIOD_ISR  ((AT91_REG *) 	0xFFFFF84C) // (PIOD) Interrupt Status Register
#define AT91C_PIOD_ODSR ((AT91_REG *) 	0xFFFFF838) // (PIOD) Output Data Status Register
#define AT91C_PIOD_OSR  ((AT91_REG *) 	0xFFFFF818) // (PIOD) Output Status Register
#define AT91C_PIOD_IFER ((AT91_REG *) 	0xFFFFF820) // (PIOD) Input Filter Enable Register
#define AT91C_PIOD_SLEWRATE1 ((AT91_REG *) 	0xFFFFF8B0) // (PIOD) PIO Slewrate Control Register
// ========== Register definition for PIOE peripheral ========== 
#define AT91C_PIOE_ODSR ((AT91_REG *) 	0xFFFFFA38) // (PIOE) Output Data Status Register
#define AT91C_PIOE_ABSR ((AT91_REG *) 	0xFFFFFA78) // (PIOE) AB Select Status Register
#define AT91C_PIOE_PSR  ((AT91_REG *) 	0xFFFFFA08) // (PIOE) PIO Status Register
#define AT91C_PIOE_PPUDR ((AT91_REG *) 	0xFFFFFA60) // (PIOE) Pull-up Disable Register
#define AT91C_PIOE_OER  ((AT91_REG *) 	0xFFFFFA10) // (PIOE) Output Enable Register
#define AT91C_PIOE_IFSR ((AT91_REG *) 	0xFFFFFA28) // (PIOE) Input Filter Status Register
#define AT91C_PIOE_IFER ((AT91_REG *) 	0xFFFFFA20) // (PIOE) Input Filter Enable Register
#define AT91C_PIOE_DELAY3 ((AT91_REG *) 	0xFFFFFAC8) // (PIOE) PIO Delay Control Register
#define AT91C_PIOE_ODR  ((AT91_REG *) 	0xFFFFFA14) // (PIOE) Output Disable Registerr
#define AT91C_PIOE_IDR  ((AT91_REG *) 	0xFFFFFA44) // (PIOE) Interrupt Disable Register
#define AT91C_PIOE_DELAY1 ((AT91_REG *) 	0xFFFFFAC0) // (PIOE) PIO Delay Control Register
#define AT91C_PIOE_OSR  ((AT91_REG *) 	0xFFFFFA18) // (PIOE) Output Status Register
#define AT91C_PIOE_CODR ((AT91_REG *) 	0xFFFFFA34) // (PIOE) Clear Output Data Register
#define AT91C_PIOE_VERSION ((AT91_REG *) 	0xFFFFFAFC) // (PIOE) PIO Version Register
#define AT91C_PIOE_MDSR ((AT91_REG *) 	0xFFFFFA58) // (PIOE) Multi-driver Status Register
#define AT91C_PIOE_PDR  ((AT91_REG *) 	0xFFFFFA04) // (PIOE) PIO Disable Register
#define AT91C_PIOE_IER  ((AT91_REG *) 	0xFFFFFA40) // (PIOE) Interrupt Enable Register
#define AT91C_PIOE_OWSR ((AT91_REG *) 	0xFFFFFAA8) // (PIOE) Output Write Status Register
#define AT91C_PIOE_BSR  ((AT91_REG *) 	0xFFFFFA74) // (PIOE) Select B Register
#define AT91C_PIOE_SLEWRATE1 ((AT91_REG *) 	0xFFFFFAB0) // (PIOE) PIO Slewrate Control Register
#define AT91C_PIOE_DELAY4 ((AT91_REG *) 	0xFFFFFACC) // (PIOE) PIO Delay Control Register
#define AT91C_PIOE_PER  ((AT91_REG *) 	0xFFFFFA00) // (PIOE) PIO Enable Register
#define AT91C_PIOE_OWDR ((AT91_REG *) 	0xFFFFFAA4) // (PIOE) Output Write Disable Register
#define AT91C_PIOE_IFDR ((AT91_REG *) 	0xFFFFFA24) // (PIOE) Input Filter Disable Register
#define AT91C_PIOE_PPUSR ((AT91_REG *) 	0xFFFFFA68) // (PIOE) Pull-up Status Register
#define AT91C_PIOE_PDSR ((AT91_REG *) 	0xFFFFFA3C) // (PIOE) Pin Data Status Register
#define AT91C_PIOE_PPUER ((AT91_REG *) 	0xFFFFFA64) // (PIOE) Pull-up Enable Register
#define AT91C_PIOE_MDDR ((AT91_REG *) 	0xFFFFFA54) // (PIOE) Multi-driver Disable Register
#define AT91C_PIOE_ISR  ((AT91_REG *) 	0xFFFFFA4C) // (PIOE) Interrupt Status Register
#define AT91C_PIOE_DELAY2 ((AT91_REG *) 	0xFFFFFAC4) // (PIOE) PIO Delay Control Register
#define AT91C_PIOE_SODR ((AT91_REG *) 	0xFFFFFA30) // (PIOE) Set Output Data Register
#define AT91C_PIOE_ASR  ((AT91_REG *) 	0xFFFFFA70) // (PIOE) Select A Register
#define AT91C_PIOE_IMR  ((AT91_REG *) 	0xFFFFFA48) // (PIOE) Interrupt Mask Register
#define AT91C_PIOE_OWER ((AT91_REG *) 	0xFFFFFAA0) // (PIOE) Output Write Enable Register
#define AT91C_PIOE_MDER ((AT91_REG *) 	0xFFFFFA50) // (PIOE) Multi-driver Enable Register
// ========== Register definition for PMC peripheral ========== 
#define AT91C_PMC_ADDRSIZE ((AT91_REG *) 	0xFFFFFCEC) // (PMC) 
#define AT91C_PMC_PCER  ((AT91_REG *) 	0xFFFFFC10) // (PMC) Peripheral Clock Enable Register
#define AT91C_PMC_PCKR  ((AT91_REG *) 	0xFFFFFC40) // (PMC) Programmable Clock 0 Register
#define AT91C_PMC_MCKR  ((AT91_REG *) 	0xFFFFFC30) // (PMC) Master Clock Register
#define AT91C_PMC_PLLAR ((AT91_REG *) 	0xFFFFFC28) // (PMC) PLL A Register
#define AT91C_PMC_NAME2 ((AT91_REG *) 	0xFFFFFCF4) // (PMC) 
#define AT91C_PMC_PCDR  ((AT91_REG *) 	0xFFFFFC14) // (PMC) Peripheral Clock Disable Register
#define AT91C_PMC_SCSR  ((AT91_REG *) 	0xFFFFFC08) // (PMC) System Clock Status Register
#define AT91C_PMC_MCFR  ((AT91_REG *) 	0xFFFFFC24) // (PMC) Main Clock  Frequency Register
#define AT91C_PMC_FEATURES ((AT91_REG *) 	0xFFFFFCF8) // (PMC) 
#define AT91C_PMC_PLLICPR ((AT91_REG *) 	0xFFFFFC80) // (PMC) PLL Charge Pump Current Register
#define AT91C_PMC_IMR   ((AT91_REG *) 	0xFFFFFC6C) // (PMC) Interrupt Mask Register
#define AT91C_PMC_IER   ((AT91_REG *) 	0xFFFFFC60) // (PMC) Interrupt Enable Register
#define AT91C_PMC_UCKR  ((AT91_REG *) 	0xFFFFFC1C) // (PMC) UTMI Clock Configuration Register
#define AT91C_PMC_USB   ((AT91_REG *) 	0xFFFFFC38) // (PMC) USB clock register
#define AT91C_PMC_MOR   ((AT91_REG *) 	0xFFFFFC20) // (PMC) Main Oscillator Register
#define AT91C_PMC_IDR   ((AT91_REG *) 	0xFFFFFC64) // (PMC) Interrupt Disable Register
#define AT91C_PMC_NAME1 ((AT91_REG *) 	0xFFFFFCF0) // (PMC) 
#define AT91C_PMC_SCDR  ((AT91_REG *) 	0xFFFFFC04) // (PMC) System Clock Disable Register
#define AT91C_PMC_PCSR  ((AT91_REG *) 	0xFFFFFC18) // (PMC) Peripheral Clock Status Register
#define AT91C_PMC_SCER  ((AT91_REG *) 	0xFFFFFC00) // (PMC) System Clock Enable Register
#define AT91C_PMC_VERSION ((AT91_REG *) 	0xFFFFFCFC) // (PMC) 
#define AT91C_PMC_SR    ((AT91_REG *) 	0xFFFFFC68) // (PMC) Status Register
// ========== Register definition for CKGR peripheral ========== 
#define AT91C_CKGR_MOR  ((AT91_REG *) 	0xFFFFFC20) // (CKGR) Main Oscillator Register
#define AT91C_CKGR_MCFR ((AT91_REG *) 	0xFFFFFC24) // (CKGR) Main Clock  Frequency Register
#define AT91C_CKGR_PLLAR ((AT91_REG *) 	0xFFFFFC28) // (CKGR) PLL A Register
#define AT91C_CKGR_UCKR ((AT91_REG *) 	0xFFFFFC1C) // (CKGR) UTMI Clock Configuration Register
// ========== Register definition for RSTC peripheral ========== 
#define AT91C_RSTC_RCR  ((AT91_REG *) 	0xFFFFFD00) // (RSTC) Reset Control Register
#define AT91C_RSTC_VER  ((AT91_REG *) 	0xFFFFFDFC) // (RSTC) Version Register
#define AT91C_RSTC_RMR  ((AT91_REG *) 	0xFFFFFD08) // (RSTC) Reset Mode Register
#define AT91C_RSTC_RSR  ((AT91_REG *) 	0xFFFFFD04) // (RSTC) Reset Status Register
// ========== Register definition for SHDWC peripheral ========== 
#define AT91C_SHDWC_SHSR ((AT91_REG *) 	0xFFFFFD18) // (SHDWC) Shut Down Status Register
#define AT91C_SHDWC_SHMR ((AT91_REG *) 	0xFFFFFD14) // (SHDWC) Shut Down Mode Register
#define AT91C_SHDWC_SHCR ((AT91_REG *) 	0xFFFFFD10) // (SHDWC) Shut Down Control Register
// ========== Register definition for RTTC peripheral ========== 
#define AT91C_RTTC_RTSR ((AT91_REG *) 	0xFFFFFD2C) // (RTTC) Real-time Status Register
#define AT91C_RTTC_RTMR ((AT91_REG *) 	0xFFFFFD20) // (RTTC) Real-time Mode Register
#define AT91C_RTTC_RTVR ((AT91_REG *) 	0xFFFFFD28) // (RTTC) Real-time Value Register
#define AT91C_RTTC_RTAR ((AT91_REG *) 	0xFFFFFD24) // (RTTC) Real-time Alarm Register
// ========== Register definition for PITC peripheral ========== 
#define AT91C_PITC_PIVR ((AT91_REG *) 	0xFFFFFD38) // (PITC) Period Interval Value Register
#define AT91C_PITC_PISR ((AT91_REG *) 	0xFFFFFD34) // (PITC) Period Interval Status Register
#define AT91C_PITC_PIIR ((AT91_REG *) 	0xFFFFFD3C) // (PITC) Period Interval Image Register
#define AT91C_PITC_PIMR ((AT91_REG *) 	0xFFFFFD30) // (PITC) Period Interval Mode Register
// ========== Register definition for WDTC peripheral ========== 
#define AT91C_WDTC_WDCR ((AT91_REG *) 	0xFFFFFD40) // (WDTC) Watchdog Control Register
#define AT91C_WDTC_WDSR ((AT91_REG *) 	0xFFFFFD48) // (WDTC) Watchdog Status Register
#define AT91C_WDTC_WDMR ((AT91_REG *) 	0xFFFFFD44) // (WDTC) Watchdog Mode Register
// ========== Register definition for RTC peripheral ========== 
#define AT91C_RTC_IDR   ((AT91_REG *) 	0xFFFFFDD4) // (RTC) Interrupt Disable Register
#define AT91C_RTC_SR    ((AT91_REG *) 	0xFFFFFDC8) // (RTC) Status Register
#define AT91C_RTC_MR    ((AT91_REG *) 	0xFFFFFDB4) // (RTC) Mode Register
#define AT91C_RTC_IER   ((AT91_REG *) 	0xFFFFFDD0) // (RTC) Interrupt Enable Register
#define AT91C_RTC_TIMALR ((AT91_REG *) 	0xFFFFFDC0) // (RTC) Time Alarm Register
#define AT91C_RTC_CALALR ((AT91_REG *) 	0xFFFFFDC4) // (RTC) Calendar Alarm Register
#define AT91C_RTC_CR    ((AT91_REG *) 	0xFFFFFDB0) // (RTC) Control Register
#define AT91C_RTC_TIMR  ((AT91_REG *) 	0xFFFFFDB8) // (RTC) Time Register
#define AT91C_RTC_CALR  ((AT91_REG *) 	0xFFFFFDBC) // (RTC) Calendar Register
#define AT91C_RTC_VER   ((AT91_REG *) 	0xFFFFFDDC) // (RTC) Valid Entry Register
#define AT91C_RTC_SCCR  ((AT91_REG *) 	0xFFFFFDCC) // (RTC) Status Clear Command Register
#define AT91C_RTC_IMR   ((AT91_REG *) 	0xFFFFFDD8) // (RTC) Interrupt Mask Register
// ========== Register definition for TC0 peripheral ========== 
#define AT91C_TC0_IER   ((AT91_REG *) 	0xFFF7C024) // (TC0) Interrupt Enable Register
#define AT91C_TC0_IMR   ((AT91_REG *) 	0xFFF7C02C) // (TC0) Interrupt Mask Register
#define AT91C_TC0_CCR   ((AT91_REG *) 	0xFFF7C000) // (TC0) Channel Control Register
#define AT91C_TC0_RB    ((AT91_REG *) 	0xFFF7C018) // (TC0) Register B
#define AT91C_TC0_CV    ((AT91_REG *) 	0xFFF7C010) // (TC0) Counter Value
#define AT91C_TC0_SR    ((AT91_REG *) 	0xFFF7C020) // (TC0) Status Register
#define AT91C_TC0_CMR   ((AT91_REG *) 	0xFFF7C004) // (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC0_RA    ((AT91_REG *) 	0xFFF7C014) // (TC0) Register A
#define AT91C_TC0_RC    ((AT91_REG *) 	0xFFF7C01C) // (TC0) Register C
#define AT91C_TC0_IDR   ((AT91_REG *) 	0xFFF7C028) // (TC0) Interrupt Disable Register
// ========== Register definition for TC1 peripheral ========== 
#define AT91C_TC1_IER   ((AT91_REG *) 	0xFFF7C064) // (TC1) Interrupt Enable Register
#define AT91C_TC1_SR    ((AT91_REG *) 	0xFFF7C060) // (TC1) Status Register
#define AT91C_TC1_RC    ((AT91_REG *) 	0xFFF7C05C) // (TC1) Register C
#define AT91C_TC1_CV    ((AT91_REG *) 	0xFFF7C050) // (TC1) Counter Value
#define AT91C_TC1_RA    ((AT91_REG *) 	0xFFF7C054) // (TC1) Register A
#define AT91C_TC1_CMR   ((AT91_REG *) 	0xFFF7C044) // (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC1_IDR   ((AT91_REG *) 	0xFFF7C068) // (TC1) Interrupt Disable Register
#define AT91C_TC1_RB    ((AT91_REG *) 	0xFFF7C058) // (TC1) Register B
#define AT91C_TC1_IMR   ((AT91_REG *) 	0xFFF7C06C) // (TC1) Interrupt Mask Register
#define AT91C_TC1_CCR   ((AT91_REG *) 	0xFFF7C040) // (TC1) Channel Control Register
// ========== Register definition for TC2 peripheral ========== 
#define AT91C_TC2_SR    ((AT91_REG *) 	0xFFF7C0A0) // (TC2) Status Register
#define AT91C_TC2_IMR   ((AT91_REG *) 	0xFFF7C0AC) // (TC2) Interrupt Mask Register
#define AT91C_TC2_IER   ((AT91_REG *) 	0xFFF7C0A4) // (TC2) Interrupt Enable Register
#define AT91C_TC2_CV    ((AT91_REG *) 	0xFFF7C090) // (TC2) Counter Value
#define AT91C_TC2_RB    ((AT91_REG *) 	0xFFF7C098) // (TC2) Register B
#define AT91C_TC2_CCR   ((AT91_REG *) 	0xFFF7C080) // (TC2) Channel Control Register
#define AT91C_TC2_CMR   ((AT91_REG *) 	0xFFF7C084) // (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC2_RA    ((AT91_REG *) 	0xFFF7C094) // (TC2) Register A
#define AT91C_TC2_IDR   ((AT91_REG *) 	0xFFF7C0A8) // (TC2) Interrupt Disable Register
#define AT91C_TC2_RC    ((AT91_REG *) 	0xFFF7C09C) // (TC2) Register C
// ========== Register definition for TC3 peripheral ========== 
#define AT91C_TC3_SR    ((AT91_REG *) 	0xFFFD4020) // (TC3) Status Register
#define AT91C_TC3_RC    ((AT91_REG *) 	0xFFFD401C) // (TC3) Register C
#define AT91C_TC3_IER   ((AT91_REG *) 	0xFFFD4024) // (TC3) Interrupt Enable Register
#define AT91C_TC3_CV    ((AT91_REG *) 	0xFFFD4010) // (TC3) Counter Value
#define AT91C_TC3_IDR   ((AT91_REG *) 	0xFFFD4028) // (TC3) Interrupt Disable Register
#define AT91C_TC3_IMR   ((AT91_REG *) 	0xFFFD402C) // (TC3) Interrupt Mask Register
#define AT91C_TC3_CMR   ((AT91_REG *) 	0xFFFD4004) // (TC3) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC3_RB    ((AT91_REG *) 	0xFFFD4018) // (TC3) Register B
#define AT91C_TC3_CCR   ((AT91_REG *) 	0xFFFD4000) // (TC3) Channel Control Register
#define AT91C_TC3_RA    ((AT91_REG *) 	0xFFFD4014) // (TC3) Register A
// ========== Register definition for TC4 peripheral ========== 
#define AT91C_TC4_CV    ((AT91_REG *) 	0xFFFD4050) // (TC4) Counter Value
#define AT91C_TC4_RA    ((AT91_REG *) 	0xFFFD4054) // (TC4) Register A
#define AT91C_TC4_IDR   ((AT91_REG *) 	0xFFFD4068) // (TC4) Interrupt Disable Register
#define AT91C_TC4_SR    ((AT91_REG *) 	0xFFFD4060) // (TC4) Status Register
#define AT91C_TC4_CMR   ((AT91_REG *) 	0xFFFD4044) // (TC4) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC4_CCR   ((AT91_REG *) 	0xFFFD4040) // (TC4) Channel Control Register
#define AT91C_TC4_RB    ((AT91_REG *) 	0xFFFD4058) // (TC4) Register B
#define AT91C_TC4_RC    ((AT91_REG *) 	0xFFFD405C) // (TC4) Register C
#define AT91C_TC4_IER   ((AT91_REG *) 	0xFFFD4064) // (TC4) Interrupt Enable Register
#define AT91C_TC4_IMR   ((AT91_REG *) 	0xFFFD406C) // (TC4) Interrupt Mask Register
// ========== Register definition for TC5 peripheral ========== 
#define AT91C_TC5_IER   ((AT91_REG *) 	0xFFFD40A4) // (TC5) Interrupt Enable Register
#define AT91C_TC5_IDR   ((AT91_REG *) 	0xFFFD40A8) // (TC5) Interrupt Disable Register
#define AT91C_TC5_RA    ((AT91_REG *) 	0xFFFD4094) // (TC5) Register A
#define AT91C_TC5_RB    ((AT91_REG *) 	0xFFFD4098) // (TC5) Register B
#define AT91C_TC5_CCR   ((AT91_REG *) 	0xFFFD4080) // (TC5) Channel Control Register
#define AT91C_TC5_SR    ((AT91_REG *) 	0xFFFD40A0) // (TC5) Status Register
#define AT91C_TC5_CV    ((AT91_REG *) 	0xFFFD4090) // (TC5) Counter Value
#define AT91C_TC5_RC    ((AT91_REG *) 	0xFFFD409C) // (TC5) Register C
#define AT91C_TC5_IMR   ((AT91_REG *) 	0xFFFD40AC) // (TC5) Interrupt Mask Register
#define AT91C_TC5_CMR   ((AT91_REG *) 	0xFFFD4084) // (TC5) Channel Mode Register (Capture Mode / Waveform Mode)
// ========== Register definition for TCB0 peripheral ========== 
#define AT91C_TCB0_IPNAME1 ((AT91_REG *) 	0xFFF7C0F0) // (TCB0) TC IPNAME1 REGISTER 
#define AT91C_TCB0_IPNAME2 ((AT91_REG *) 	0xFFF7C0F4) // (TCB0) TC IPNAME2 REGISTER 
#define AT91C_TCB0_FEATURES ((AT91_REG *) 	0xFFF7C0F8) // (TCB0) TC FEATURES REGISTER 
#define AT91C_TCB0_BCR  ((AT91_REG *) 	0xFFF7C0C0) // (TCB0) TC Block Control Register
#define AT91C_TCB0_VER  ((AT91_REG *) 	0xFFF7C0FC) // (TCB0)  Version Register
#define AT91C_TCB0_BMR  ((AT91_REG *) 	0xFFF7C0C4) // (TCB0) TC Block Mode Register
#define AT91C_TCB0_ADDRSIZE ((AT91_REG *) 	0xFFF7C0EC) // (TCB0) TC ADDRSIZE REGISTER 
// ========== Register definition for TCB1 peripheral ========== 
#define AT91C_TCB1_VER  ((AT91_REG *) 	0xFFFD40FC) // (TCB1)  Version Register
#define AT91C_TCB1_IPNAME2 ((AT91_REG *) 	0xFFFD40F4) // (TCB1) TC IPNAME2 REGISTER 
#define AT91C_TCB1_ADDRSIZE ((AT91_REG *) 	0xFFFD40EC) // (TCB1) TC ADDRSIZE REGISTER 
#define AT91C_TCB1_BMR  ((AT91_REG *) 	0xFFFD40C4) // (TCB1) TC Block Mode Register
#define AT91C_TCB1_FEATURES ((AT91_REG *) 	0xFFFD40F8) // (TCB1) TC FEATURES REGISTER 
#define AT91C_TCB1_IPNAME1 ((AT91_REG *) 	0xFFFD40F0) // (TCB1) TC IPNAME1 REGISTER 
#define AT91C_TCB1_BCR  ((AT91_REG *) 	0xFFFD40C0) // (TCB1) TC Block Control Register
// ========== Register definition for MCI0 peripheral ========== 
#define AT91C_MCI0_IMR  ((AT91_REG *) 	0xFFF8004C) // (MCI0) MCI Interrupt Mask Register
#define AT91C_MCI0_MR   ((AT91_REG *) 	0xFFF80004) // (MCI0) MCI Mode Register
#define AT91C_MCI0_ADDRSIZE ((AT91_REG *) 	0xFFF800EC) // (MCI0) MCI ADDRSIZE REGISTER 
#define AT91C_MCI0_CR   ((AT91_REG *) 	0xFFF80000) // (MCI0) MCI Control Register
#define AT91C_MCI0_IPNAME2 ((AT91_REG *) 	0xFFF800F4) // (MCI0) MCI IPNAME2 REGISTER 
#define AT91C_MCI0_IER  ((AT91_REG *) 	0xFFF80044) // (MCI0) MCI Interrupt Enable Register
#define AT91C_MCI0_FIFO ((AT91_REG *) 	0xFFF80200) // (MCI0) MCI FIFO Aperture Register
#define AT91C_MCI0_DTOR ((AT91_REG *) 	0xFFF80008) // (MCI0) MCI Data Timeout Register
#define AT91C_MCI0_SDCR ((AT91_REG *) 	0xFFF8000C) // (MCI0) MCI SD/SDIO Card Register
#define AT91C_MCI0_BLKR ((AT91_REG *) 	0xFFF80018) // (MCI0) MCI Block Register
#define AT91C_MCI0_VER  ((AT91_REG *) 	0xFFF800FC) // (MCI0) MCI VERSION REGISTER 
#define AT91C_MCI0_WPSR ((AT91_REG *) 	0xFFF800E8) // (MCI0) MCI Write Protection Status Register
#define AT91C_MCI0_CMDR ((AT91_REG *) 	0xFFF80014) // (MCI0) MCI Command Register
#define AT91C_MCI0_CSTOR ((AT91_REG *) 	0xFFF8001C) // (MCI0) MCI Completion Signal Timeout Register
#define AT91C_MCI0_IPNAME1 ((AT91_REG *) 	0xFFF800F0) // (MCI0) MCI IPNAME1 REGISTER 
#define AT91C_MCI0_DMA  ((AT91_REG *) 	0xFFF80050) // (MCI0) MCI DMA Configuration Register
#define AT91C_MCI0_FEATURES ((AT91_REG *) 	0xFFF800F8) // (MCI0) MCI FEATURES REGISTER 
#define AT91C_MCI0_RDR  ((AT91_REG *) 	0xFFF80030) // (MCI0) MCI Receive Data Register
#define AT91C_MCI0_SR   ((AT91_REG *) 	0xFFF80040) // (MCI0) MCI Status Register
#define AT91C_MCI0_TDR  ((AT91_REG *) 	0xFFF80034) // (MCI0) MCI Transmit Data Register
#define AT91C_MCI0_CFG  ((AT91_REG *) 	0xFFF80054) // (MCI0) MCI Configuration Register
#define AT91C_MCI0_ARGR ((AT91_REG *) 	0xFFF80010) // (MCI0) MCI Argument Register
#define AT91C_MCI0_RSPR ((AT91_REG *) 	0xFFF80020) // (MCI0) MCI Response Register
#define AT91C_MCI0_WPCR ((AT91_REG *) 	0xFFF800E4) // (MCI0) MCI Write Protection Control Register
#define AT91C_MCI0_IDR  ((AT91_REG *) 	0xFFF80048) // (MCI0) MCI Interrupt Disable Register
// ========== Register definition for MCI1 peripheral ========== 
#define AT91C_MCI1_IPNAME1 ((AT91_REG *) 	0xFFFD00F0) // (MCI1) MCI IPNAME1 REGISTER 
#define AT91C_MCI1_IDR  ((AT91_REG *) 	0xFFFD0048) // (MCI1) MCI Interrupt Disable Register
#define AT91C_MCI1_IPNAME2 ((AT91_REG *) 	0xFFFD00F4) // (MCI1) MCI IPNAME2 REGISTER 
#define AT91C_MCI1_MR   ((AT91_REG *) 	0xFFFD0004) // (MCI1) MCI Mode Register
#define AT91C_MCI1_SR   ((AT91_REG *) 	0xFFFD0040) // (MCI1) MCI Status Register
#define AT91C_MCI1_DTOR ((AT91_REG *) 	0xFFFD0008) // (MCI1) MCI Data Timeout Register
#define AT91C_MCI1_WPCR ((AT91_REG *) 	0xFFFD00E4) // (MCI1) MCI Write Protection Control Register
#define AT91C_MCI1_ADDRSIZE ((AT91_REG *) 	0xFFFD00EC) // (MCI1) MCI ADDRSIZE REGISTER 
#define AT91C_MCI1_ARGR ((AT91_REG *) 	0xFFFD0010) // (MCI1) MCI Argument Register
#define AT91C_MCI1_FIFO ((AT91_REG *) 	0xFFFD0200) // (MCI1) MCI FIFO Aperture Register
#define AT91C_MCI1_IER  ((AT91_REG *) 	0xFFFD0044) // (MCI1) MCI Interrupt Enable Register
#define AT91C_MCI1_SDCR ((AT91_REG *) 	0xFFFD000C) // (MCI1) MCI SD/SDIO Card Register
#define AT91C_MCI1_FEATURES ((AT91_REG *) 	0xFFFD00F8) // (MCI1) MCI FEATURES REGISTER 
#define AT91C_MCI1_CR   ((AT91_REG *) 	0xFFFD0000) // (MCI1) MCI Control Register
#define AT91C_MCI1_CSTOR ((AT91_REG *) 	0xFFFD001C) // (MCI1) MCI Completion Signal Timeout Register
#define AT91C_MCI1_RSPR ((AT91_REG *) 	0xFFFD0020) // (MCI1) MCI Response Register
#define AT91C_MCI1_IMR  ((AT91_REG *) 	0xFFFD004C) // (MCI1) MCI Interrupt Mask Register
#define AT91C_MCI1_DMA  ((AT91_REG *) 	0xFFFD0050) // (MCI1) MCI DMA Configuration Register
#define AT91C_MCI1_BLKR ((AT91_REG *) 	0xFFFD0018) // (MCI1) MCI Block Register
#define AT91C_MCI1_RDR  ((AT91_REG *) 	0xFFFD0030) // (MCI1) MCI Receive Data Register
#define AT91C_MCI1_CFG  ((AT91_REG *) 	0xFFFD0054) // (MCI1) MCI Configuration Register
#define AT91C_MCI1_WPSR ((AT91_REG *) 	0xFFFD00E8) // (MCI1) MCI Write Protection Status Register
#define AT91C_MCI1_CMDR ((AT91_REG *) 	0xFFFD0014) // (MCI1) MCI Command Register
#define AT91C_MCI1_TDR  ((AT91_REG *) 	0xFFFD0034) // (MCI1) MCI Transmit Data Register
#define AT91C_MCI1_VER  ((AT91_REG *) 	0xFFFD00FC) // (MCI1) MCI VERSION REGISTER 
// ========== Register definition for TWI0 peripheral ========== 
#define AT91C_TWI0_RHR  ((AT91_REG *) 	0xFFF84030) // (TWI0) Receive Holding Register
#define AT91C_TWI0_ADDRSIZE ((AT91_REG *) 	0xFFF840EC) // (TWI0) TWI ADDRSIZE REGISTER 
#define AT91C_TWI0_SMR  ((AT91_REG *) 	0xFFF84008) // (TWI0) Slave Mode Register
#define AT91C_TWI0_IDR  ((AT91_REG *) 	0xFFF84028) // (TWI0) Interrupt Disable Register
#define AT91C_TWI0_CWGR ((AT91_REG *) 	0xFFF84010) // (TWI0) Clock Waveform Generator Register
#define AT91C_TWI0_IER  ((AT91_REG *) 	0xFFF84024) // (TWI0) Interrupt Enable Register
#define AT91C_TWI0_THR  ((AT91_REG *) 	0xFFF84034) // (TWI0) Transmit Holding Register
#define AT91C_TWI0_IPNAME2 ((AT91_REG *) 	0xFFF840F4) // (TWI0) TWI IPNAME2 REGISTER 
#define AT91C_TWI0_CR   ((AT91_REG *) 	0xFFF84000) // (TWI0) Control Register
#define AT91C_TWI0_MMR  ((AT91_REG *) 	0xFFF84004) // (TWI0) Master Mode Register
#define AT91C_TWI0_SR   ((AT91_REG *) 	0xFFF84020) // (TWI0) Status Register
#define AT91C_TWI0_IMR  ((AT91_REG *) 	0xFFF8402C) // (TWI0) Interrupt Mask Register
#define AT91C_TWI0_FEATURES ((AT91_REG *) 	0xFFF840F8) // (TWI0) TWI FEATURES REGISTER 
#define AT91C_TWI0_IADR ((AT91_REG *) 	0xFFF8400C) // (TWI0) Internal Address Register
#define AT91C_TWI0_IPNAME1 ((AT91_REG *) 	0xFFF840F0) // (TWI0) TWI IPNAME1 REGISTER 
#define AT91C_TWI0_VER  ((AT91_REG *) 	0xFFF840FC) // (TWI0) Version Register
// ========== Register definition for TWI1 peripheral ========== 
#define AT91C_TWI1_VER  ((AT91_REG *) 	0xFFF880FC) // (TWI1) Version Register
#define AT91C_TWI1_IMR  ((AT91_REG *) 	0xFFF8802C) // (TWI1) Interrupt Mask Register
#define AT91C_TWI1_THR  ((AT91_REG *) 	0xFFF88034) // (TWI1) Transmit Holding Register
#define AT91C_TWI1_IER  ((AT91_REG *) 	0xFFF88024) // (TWI1) Interrupt Enable Register
#define AT91C_TWI1_MMR  ((AT91_REG *) 	0xFFF88004) // (TWI1) Master Mode Register
#define AT91C_TWI1_IPNAME1 ((AT91_REG *) 	0xFFF880F0) // (TWI1) TWI IPNAME1 REGISTER 
#define AT91C_TWI1_CR   ((AT91_REG *) 	0xFFF88000) // (TWI1) Control Register
#define AT91C_TWI1_SR   ((AT91_REG *) 	0xFFF88020) // (TWI1) Status Register
#define AT91C_TWI1_CWGR ((AT91_REG *) 	0xFFF88010) // (TWI1) Clock Waveform Generator Register
#define AT91C_TWI1_IPNAME2 ((AT91_REG *) 	0xFFF880F4) // (TWI1) TWI IPNAME2 REGISTER 
#define AT91C_TWI1_IDR  ((AT91_REG *) 	0xFFF88028) // (TWI1) Interrupt Disable Register
#define AT91C_TWI1_RHR  ((AT91_REG *) 	0xFFF88030) // (TWI1) Receive Holding Register
#define AT91C_TWI1_IADR ((AT91_REG *) 	0xFFF8800C) // (TWI1) Internal Address Register
#define AT91C_TWI1_SMR  ((AT91_REG *) 	0xFFF88008) // (TWI1) Slave Mode Register
#define AT91C_TWI1_FEATURES ((AT91_REG *) 	0xFFF880F8) // (TWI1) TWI FEATURES REGISTER 
#define AT91C_TWI1_ADDRSIZE ((AT91_REG *) 	0xFFF880EC) // (TWI1) TWI ADDRSIZE REGISTER 
// ========== Register definition for PDC_US0 peripheral ========== 
#define AT91C_US0_TNPR  ((AT91_REG *) 	0xFFF8C118) // (PDC_US0) Transmit Next Pointer Register
#define AT91C_US0_PTSR  ((AT91_REG *) 	0xFFF8C124) // (PDC_US0) PDC Transfer Status Register
#define AT91C_US0_PTCR  ((AT91_REG *) 	0xFFF8C120) // (PDC_US0) PDC Transfer Control Register
#define AT91C_US0_RNCR  ((AT91_REG *) 	0xFFF8C114) // (PDC_US0) Receive Next Counter Register
#define AT91C_US0_RCR   ((AT91_REG *) 	0xFFF8C104) // (PDC_US0) Receive Counter Register
#define AT91C_US0_TNCR  ((AT91_REG *) 	0xFFF8C11C) // (PDC_US0) Transmit Next Counter Register
#define AT91C_US0_TCR   ((AT91_REG *) 	0xFFF8C10C) // (PDC_US0) Transmit Counter Register
#define AT91C_US0_RNPR  ((AT91_REG *) 	0xFFF8C110) // (PDC_US0) Receive Next Pointer Register
#define AT91C_US0_RPR   ((AT91_REG *) 	0xFFF8C100) // (PDC_US0) Receive Pointer Register
#define AT91C_US0_TPR   ((AT91_REG *) 	0xFFF8C108) // (PDC_US0) Transmit Pointer Register
// ========== Register definition for US0 peripheral ========== 
#define AT91C_US0_MAN   ((AT91_REG *) 	0xFFF8C050) // (US0) Manchester Encoder Decoder Register
#define AT91C_US0_MR    ((AT91_REG *) 	0xFFF8C004) // (US0) Mode Register
#define AT91C_US0_RHR   ((AT91_REG *) 	0xFFF8C018) // (US0) Receiver Holding Register
#define AT91C_US0_CSR   ((AT91_REG *) 	0xFFF8C014) // (US0) Channel Status Register
#define AT91C_US0_CR    ((AT91_REG *) 	0xFFF8C000) // (US0) Control Register
#define AT91C_US0_VER   ((AT91_REG *) 	0xFFF8C0FC) // (US0) VERSION Register
#define AT91C_US0_IER   ((AT91_REG *) 	0xFFF8C008) // (US0) Interrupt Enable Register
#define AT91C_US0_BRGR  ((AT91_REG *) 	0xFFF8C020) // (US0) Baud Rate Generator Register
#define AT91C_US0_FEATURES ((AT91_REG *) 	0xFFF8C0F8) // (US0) US FEATURES REGISTER 
#define AT91C_US0_RTOR  ((AT91_REG *) 	0xFFF8C024) // (US0) Receiver Time-out Register
#define AT91C_US0_THR   ((AT91_REG *) 	0xFFF8C01C) // (US0) Transmitter Holding Register
#define AT91C_US0_NER   ((AT91_REG *) 	0xFFF8C044) // (US0) Nb Errors Register
#define AT91C_US0_IPNAME1 ((AT91_REG *) 	0xFFF8C0F0) // (US0) US IPNAME1 REGISTER 
#define AT91C_US0_IMR   ((AT91_REG *) 	0xFFF8C010) // (US0) Interrupt Mask Register
#define AT91C_US0_ADDRSIZE ((AT91_REG *) 	0xFFF8C0EC) // (US0) US ADDRSIZE REGISTER 
#define AT91C_US0_IDR   ((AT91_REG *) 	0xFFF8C00C) // (US0) Interrupt Disable Register
#define AT91C_US0_FIDI  ((AT91_REG *) 	0xFFF8C040) // (US0) FI_DI_Ratio Register
#define AT91C_US0_IPNAME2 ((AT91_REG *) 	0xFFF8C0F4) // (US0) US IPNAME2 REGISTER 
#define AT91C_US0_TTGR  ((AT91_REG *) 	0xFFF8C028) // (US0) Transmitter Time-guard Register
#define AT91C_US0_IF    ((AT91_REG *) 	0xFFF8C04C) // (US0) IRDA_FILTER Register
// ========== Register definition for PDC_US1 peripheral ========== 
#define AT91C_US1_PTCR  ((AT91_REG *) 	0xFFF90120) // (PDC_US1) PDC Transfer Control Register
#define AT91C_US1_TNCR  ((AT91_REG *) 	0xFFF9011C) // (PDC_US1) Transmit Next Counter Register
#define AT91C_US1_RCR   ((AT91_REG *) 	0xFFF90104) // (PDC_US1) Receive Counter Register
#define AT91C_US1_RPR   ((AT91_REG *) 	0xFFF90100) // (PDC_US1) Receive Pointer Register
#define AT91C_US1_TPR   ((AT91_REG *) 	0xFFF90108) // (PDC_US1) Transmit Pointer Register
#define AT91C_US1_TCR   ((AT91_REG *) 	0xFFF9010C) // (PDC_US1) Transmit Counter Register
#define AT91C_US1_RNPR  ((AT91_REG *) 	0xFFF90110) // (PDC_US1) Receive Next Pointer Register
#define AT91C_US1_TNPR  ((AT91_REG *) 	0xFFF90118) // (PDC_US1) Transmit Next Pointer Register
#define AT91C_US1_RNCR  ((AT91_REG *) 	0xFFF90114) // (PDC_US1) Receive Next Counter Register
#define AT91C_US1_PTSR  ((AT91_REG *) 	0xFFF90124) // (PDC_US1) PDC Transfer Status Register
// ========== Register definition for US1 peripheral ========== 
#define AT91C_US1_FEATURES ((AT91_REG *) 	0xFFF900F8) // (US1) US FEATURES REGISTER 
#define AT91C_US1_ADDRSIZE ((AT91_REG *) 	0xFFF900EC) // (US1) US ADDRSIZE REGISTER 
#define AT91C_US1_NER   ((AT91_REG *) 	0xFFF90044) // (US1) Nb Errors Register
#define AT91C_US1_RHR   ((AT91_REG *) 	0xFFF90018) // (US1) Receiver Holding Register
#define AT91C_US1_IF    ((AT91_REG *) 	0xFFF9004C) // (US1) IRDA_FILTER Register
#define AT91C_US1_CR    ((AT91_REG *) 	0xFFF90000) // (US1) Control Register
#define AT91C_US1_TTGR  ((AT91_REG *) 	0xFFF90028) // (US1) Transmitter Time-guard Register
#define AT91C_US1_MR    ((AT91_REG *) 	0xFFF90004) // (US1) Mode Register
#define AT91C_US1_FIDI  ((AT91_REG *) 	0xFFF90040) // (US1) FI_DI_Ratio Register
#define AT91C_US1_RTOR  ((AT91_REG *) 	0xFFF90024) // (US1) Receiver Time-out Register
#define AT91C_US1_IER   ((AT91_REG *) 	0xFFF90008) // (US1) Interrupt Enable Register
#define AT91C_US1_MAN   ((AT91_REG *) 	0xFFF90050) // (US1) Manchester Encoder Decoder Register
#define AT91C_US1_IPNAME2 ((AT91_REG *) 	0xFFF900F4) // (US1) US IPNAME2 REGISTER 
#define AT91C_US1_IMR   ((AT91_REG *) 	0xFFF90010) // (US1) Interrupt Mask Register
#define AT91C_US1_IDR   ((AT91_REG *) 	0xFFF9000C) // (US1) Interrupt Disable Register
#define AT91C_US1_CSR   ((AT91_REG *) 	0xFFF90014) // (US1) Channel Status Register
#define AT91C_US1_BRGR  ((AT91_REG *) 	0xFFF90020) // (US1) Baud Rate Generator Register
#define AT91C_US1_THR   ((AT91_REG *) 	0xFFF9001C) // (US1) Transmitter Holding Register
#define AT91C_US1_VER   ((AT91_REG *) 	0xFFF900FC) // (US1) VERSION Register
#define AT91C_US1_IPNAME1 ((AT91_REG *) 	0xFFF900F0) // (US1) US IPNAME1 REGISTER 
// ========== Register definition for PDC_US2 peripheral ========== 
#define AT91C_US2_RNCR  ((AT91_REG *) 	0xFFF94114) // (PDC_US2) Receive Next Counter Register
#define AT91C_US2_PTCR  ((AT91_REG *) 	0xFFF94120) // (PDC_US2) PDC Transfer Control Register
#define AT91C_US2_TNPR  ((AT91_REG *) 	0xFFF94118) // (PDC_US2) Transmit Next Pointer Register
#define AT91C_US2_TNCR  ((AT91_REG *) 	0xFFF9411C) // (PDC_US2) Transmit Next Counter Register
#define AT91C_US2_TPR   ((AT91_REG *) 	0xFFF94108) // (PDC_US2) Transmit Pointer Register
#define AT91C_US2_RCR   ((AT91_REG *) 	0xFFF94104) // (PDC_US2) Receive Counter Register
#define AT91C_US2_PTSR  ((AT91_REG *) 	0xFFF94124) // (PDC_US2) PDC Transfer Status Register
#define AT91C_US2_TCR   ((AT91_REG *) 	0xFFF9410C) // (PDC_US2) Transmit Counter Register
#define AT91C_US2_RPR   ((AT91_REG *) 	0xFFF94100) // (PDC_US2) Receive Pointer Register
#define AT91C_US2_RNPR  ((AT91_REG *) 	0xFFF94110) // (PDC_US2) Receive Next Pointer Register
// ========== Register definition for US2 peripheral ========== 
#define AT91C_US2_TTGR  ((AT91_REG *) 	0xFFF94028) // (US2) Transmitter Time-guard Register
#define AT91C_US2_IER   ((AT91_REG *) 	0xFFF94008) // (US2) Interrupt Enable Register
#define AT91C_US2_ADDRSIZE ((AT91_REG *) 	0xFFF940EC) // (US2) US ADDRSIZE REGISTER 
#define AT91C_US2_NER   ((AT91_REG *) 	0xFFF94044) // (US2) Nb Errors Register
#define AT91C_US2_CR    ((AT91_REG *) 	0xFFF94000) // (US2) Control Register
#define AT91C_US2_IDR   ((AT91_REG *) 	0xFFF9400C) // (US2) Interrupt Disable Register
#define AT91C_US2_VER   ((AT91_REG *) 	0xFFF940FC) // (US2) VERSION Register
#define AT91C_US2_THR   ((AT91_REG *) 	0xFFF9401C) // (US2) Transmitter Holding Register
#define AT91C_US2_BRGR  ((AT91_REG *) 	0xFFF94020) // (US2) Baud Rate Generator Register
#define AT91C_US2_CSR   ((AT91_REG *) 	0xFFF94014) // (US2) Channel Status Register
#define AT91C_US2_IPNAME2 ((AT91_REG *) 	0xFFF940F4) // (US2) US IPNAME2 REGISTER 
#define AT91C_US2_IMR   ((AT91_REG *) 	0xFFF94010) // (US2) Interrupt Mask Register
#define AT91C_US2_RHR   ((AT91_REG *) 	0xFFF94018) // (US2) Receiver Holding Register
#define AT91C_US2_MR    ((AT91_REG *) 	0xFFF94004) // (US2) Mode Register
#define AT91C_US2_FIDI  ((AT91_REG *) 	0xFFF94040) // (US2) FI_DI_Ratio Register
#define AT91C_US2_MAN   ((AT91_REG *) 	0xFFF94050) // (US2) Manchester Encoder Decoder Register
#define AT91C_US2_IF    ((AT91_REG *) 	0xFFF9404C) // (US2) IRDA_FILTER Register
#define AT91C_US2_FEATURES ((AT91_REG *) 	0xFFF940F8) // (US2) US FEATURES REGISTER 
#define AT91C_US2_IPNAME1 ((AT91_REG *) 	0xFFF940F0) // (US2) US IPNAME1 REGISTER 
#define AT91C_US2_RTOR  ((AT91_REG *) 	0xFFF94024) // (US2) Receiver Time-out Register
// ========== Register definition for PDC_US3 peripheral ========== 
#define AT91C_US3_PTSR  ((AT91_REG *) 	0xFFF98124) // (PDC_US3) PDC Transfer Status Register
#define AT91C_US3_TCR   ((AT91_REG *) 	0xFFF9810C) // (PDC_US3) Transmit Counter Register
#define AT91C_US3_RNPR  ((AT91_REG *) 	0xFFF98110) // (PDC_US3) Receive Next Pointer Register
#define AT91C_US3_RNCR  ((AT91_REG *) 	0xFFF98114) // (PDC_US3) Receive Next Counter Register
#define AT91C_US3_TNPR  ((AT91_REG *) 	0xFFF98118) // (PDC_US3) Transmit Next Pointer Register
#define AT91C_US3_RPR   ((AT91_REG *) 	0xFFF98100) // (PDC_US3) Receive Pointer Register
#define AT91C_US3_TPR   ((AT91_REG *) 	0xFFF98108) // (PDC_US3) Transmit Pointer Register
#define AT91C_US3_RCR   ((AT91_REG *) 	0xFFF98104) // (PDC_US3) Receive Counter Register
#define AT91C_US3_TNCR  ((AT91_REG *) 	0xFFF9811C) // (PDC_US3) Transmit Next Counter Register
#define AT91C_US3_PTCR  ((AT91_REG *) 	0xFFF98120) // (PDC_US3) PDC Transfer Control Register
// ========== Register definition for US3 peripheral ========== 
#define AT91C_US3_VER   ((AT91_REG *) 	0xFFF980FC) // (US3) VERSION Register
#define AT91C_US3_BRGR  ((AT91_REG *) 	0xFFF98020) // (US3) Baud Rate Generator Register
#define AT91C_US3_TTGR  ((AT91_REG *) 	0xFFF98028) // (US3) Transmitter Time-guard Register
#define AT91C_US3_RTOR  ((AT91_REG *) 	0xFFF98024) // (US3) Receiver Time-out Register
#define AT91C_US3_MAN   ((AT91_REG *) 	0xFFF98050) // (US3) Manchester Encoder Decoder Register
#define AT91C_US3_NER   ((AT91_REG *) 	0xFFF98044) // (US3) Nb Errors Register
#define AT91C_US3_CR    ((AT91_REG *) 	0xFFF98000) // (US3) Control Register
#define AT91C_US3_IDR   ((AT91_REG *) 	0xFFF9800C) // (US3) Interrupt Disable Register
#define AT91C_US3_ADDRSIZE ((AT91_REG *) 	0xFFF980EC) // (US3) US ADDRSIZE REGISTER 
#define AT91C_US3_CSR   ((AT91_REG *) 	0xFFF98014) // (US3) Channel Status Register
#define AT91C_US3_IPNAME2 ((AT91_REG *) 	0xFFF980F4) // (US3) US IPNAME2 REGISTER 
#define AT91C_US3_IER   ((AT91_REG *) 	0xFFF98008) // (US3) Interrupt Enable Register
#define AT91C_US3_FEATURES ((AT91_REG *) 	0xFFF980F8) // (US3) US FEATURES REGISTER 
#define AT91C_US3_MR    ((AT91_REG *) 	0xFFF98004) // (US3) Mode Register
#define AT91C_US3_IF    ((AT91_REG *) 	0xFFF9804C) // (US3) IRDA_FILTER Register
#define AT91C_US3_THR   ((AT91_REG *) 	0xFFF9801C) // (US3) Transmitter Holding Register
#define AT91C_US3_IMR   ((AT91_REG *) 	0xFFF98010) // (US3) Interrupt Mask Register
#define AT91C_US3_FIDI  ((AT91_REG *) 	0xFFF98040) // (US3) FI_DI_Ratio Register
#define AT91C_US3_RHR   ((AT91_REG *) 	0xFFF98018) // (US3) Receiver Holding Register
#define AT91C_US3_IPNAME1 ((AT91_REG *) 	0xFFF980F0) // (US3) US IPNAME1 REGISTER 
// ========== Register definition for PDC_SSC0 peripheral ========== 
#define AT91C_SSC0_TNPR ((AT91_REG *) 	0xFFF9C118) // (PDC_SSC0) Transmit Next Pointer Register
#define AT91C_SSC0_PTSR ((AT91_REG *) 	0xFFF9C124) // (PDC_SSC0) PDC Transfer Status Register
#define AT91C_SSC0_TNCR ((AT91_REG *) 	0xFFF9C11C) // (PDC_SSC0) Transmit Next Counter Register
#define AT91C_SSC0_RNCR ((AT91_REG *) 	0xFFF9C114) // (PDC_SSC0) Receive Next Counter Register
#define AT91C_SSC0_TPR  ((AT91_REG *) 	0xFFF9C108) // (PDC_SSC0) Transmit Pointer Register
#define AT91C_SSC0_RCR  ((AT91_REG *) 	0xFFF9C104) // (PDC_SSC0) Receive Counter Register
#define AT91C_SSC0_PTCR ((AT91_REG *) 	0xFFF9C120) // (PDC_SSC0) PDC Transfer Control Register
#define AT91C_SSC0_RNPR ((AT91_REG *) 	0xFFF9C110) // (PDC_SSC0) Receive Next Pointer Register
#define AT91C_SSC0_TCR  ((AT91_REG *) 	0xFFF9C10C) // (PDC_SSC0) Transmit Counter Register
#define AT91C_SSC0_RPR  ((AT91_REG *) 	0xFFF9C100) // (PDC_SSC0) Receive Pointer Register
// ========== Register definition for SSC0 peripheral ========== 
#define AT91C_SSC0_IPNAME2 ((AT91_REG *) 	0xFFF9C0F4) // (SSC0) SSC IPNAME2 REGISTER 
#define AT91C_SSC0_SR   ((AT91_REG *) 	0xFFF9C040) // (SSC0) Status Register
#define AT91C_SSC0_TSHR ((AT91_REG *) 	0xFFF9C034) // (SSC0) Transmit Sync Holding Register
#define AT91C_SSC0_TCMR ((AT91_REG *) 	0xFFF9C018) // (SSC0) Transmit Clock Mode Register
#define AT91C_SSC0_ADDRSIZE ((AT91_REG *) 	0xFFF9C0EC) // (SSC0) SSC ADDRSIZE REGISTER 
#define AT91C_SSC0_IMR  ((AT91_REG *) 	0xFFF9C04C) // (SSC0) Interrupt Mask Register
#define AT91C_SSC0_IDR  ((AT91_REG *) 	0xFFF9C048) // (SSC0) Interrupt Disable Register
#define AT91C_SSC0_CR   ((AT91_REG *) 	0xFFF9C000) // (SSC0) Control Register
#define AT91C_SSC0_VER  ((AT91_REG *) 	0xFFF9C0FC) // (SSC0) Version Register
#define AT91C_SSC0_RHR  ((AT91_REG *) 	0xFFF9C020) // (SSC0) Receive Holding Register
#define AT91C_SSC0_THR  ((AT91_REG *) 	0xFFF9C024) // (SSC0) Transmit Holding Register
#define AT91C_SSC0_CMR  ((AT91_REG *) 	0xFFF9C004) // (SSC0) Clock Mode Register
#define AT91C_SSC0_FEATURES ((AT91_REG *) 	0xFFF9C0F8) // (SSC0) SSC FEATURES REGISTER 
#define AT91C_SSC0_RCMR ((AT91_REG *) 	0xFFF9C010) // (SSC0) Receive Clock ModeRegister
#define AT91C_SSC0_IPNAME1 ((AT91_REG *) 	0xFFF9C0F0) // (SSC0) SSC IPNAME1 REGISTER 
#define AT91C_SSC0_RSHR ((AT91_REG *) 	0xFFF9C030) // (SSC0) Receive Sync Holding Register
#define AT91C_SSC0_IER  ((AT91_REG *) 	0xFFF9C044) // (SSC0) Interrupt Enable Register
#define AT91C_SSC0_TFMR ((AT91_REG *) 	0xFFF9C01C) // (SSC0) Transmit Frame Mode Register
#define AT91C_SSC0_RFMR ((AT91_REG *) 	0xFFF9C014) // (SSC0) Receive Frame Mode Register
// ========== Register definition for PDC_SSC1 peripheral ========== 
#define AT91C_SSC1_RNPR ((AT91_REG *) 	0xFFFA0110) // (PDC_SSC1) Receive Next Pointer Register
#define AT91C_SSC1_TCR  ((AT91_REG *) 	0xFFFA010C) // (PDC_SSC1) Transmit Counter Register
#define AT91C_SSC1_TNCR ((AT91_REG *) 	0xFFFA011C) // (PDC_SSC1) Transmit Next Counter Register
#define AT91C_SSC1_RCR  ((AT91_REG *) 	0xFFFA0104) // (PDC_SSC1) Receive Counter Register
#define AT91C_SSC1_RNCR ((AT91_REG *) 	0xFFFA0114) // (PDC_SSC1) Receive Next Counter Register
#define AT91C_SSC1_PTCR ((AT91_REG *) 	0xFFFA0120) // (PDC_SSC1) PDC Transfer Control Register
#define AT91C_SSC1_TPR  ((AT91_REG *) 	0xFFFA0108) // (PDC_SSC1) Transmit Pointer Register
#define AT91C_SSC1_RPR  ((AT91_REG *) 	0xFFFA0100) // (PDC_SSC1) Receive Pointer Register
#define AT91C_SSC1_PTSR ((AT91_REG *) 	0xFFFA0124) // (PDC_SSC1) PDC Transfer Status Register
#define AT91C_SSC1_TNPR ((AT91_REG *) 	0xFFFA0118) // (PDC_SSC1) Transmit Next Pointer Register
// ========== Register definition for SSC1 peripheral ========== 
#define AT91C_SSC1_SR   ((AT91_REG *) 	0xFFFA0040) // (SSC1) Status Register
#define AT91C_SSC1_TFMR ((AT91_REG *) 	0xFFFA001C) // (SSC1) Transmit Frame Mode Register
#define AT91C_SSC1_IPNAME1 ((AT91_REG *) 	0xFFFA00F0) // (SSC1) SSC IPNAME1 REGISTER 
#define AT91C_SSC1_CMR  ((AT91_REG *) 	0xFFFA0004) // (SSC1) Clock Mode Register
#define AT91C_SSC1_THR  ((AT91_REG *) 	0xFFFA0024) // (SSC1) Transmit Holding Register
#define AT91C_SSC1_FEATURES ((AT91_REG *) 	0xFFFA00F8) // (SSC1) SSC FEATURES REGISTER 
#define AT91C_SSC1_TSHR ((AT91_REG *) 	0xFFFA0034) // (SSC1) Transmit Sync Holding Register
#define AT91C_SSC1_RCMR ((AT91_REG *) 	0xFFFA0010) // (SSC1) Receive Clock ModeRegister
#define AT91C_SSC1_RHR  ((AT91_REG *) 	0xFFFA0020) // (SSC1) Receive Holding Register
#define AT91C_SSC1_VER  ((AT91_REG *) 	0xFFFA00FC) // (SSC1) Version Register
#define AT91C_SSC1_TCMR ((AT91_REG *) 	0xFFFA0018) // (SSC1) Transmit Clock Mode Register
#define AT91C_SSC1_CR   ((AT91_REG *) 	0xFFFA0000) // (SSC1) Control Register
#define AT91C_SSC1_RSHR ((AT91_REG *) 	0xFFFA0030) // (SSC1) Receive Sync Holding Register
#define AT91C_SSC1_IER  ((AT91_REG *) 	0xFFFA0044) // (SSC1) Interrupt Enable Register
#define AT91C_SSC1_ADDRSIZE ((AT91_REG *) 	0xFFFA00EC) // (SSC1) SSC ADDRSIZE REGISTER 
#define AT91C_SSC1_RFMR ((AT91_REG *) 	0xFFFA0014) // (SSC1) Receive Frame Mode Register
#define AT91C_SSC1_IMR  ((AT91_REG *) 	0xFFFA004C) // (SSC1) Interrupt Mask Register
#define AT91C_SSC1_IDR  ((AT91_REG *) 	0xFFFA0048) // (SSC1) Interrupt Disable Register
#define AT91C_SSC1_IPNAME2 ((AT91_REG *) 	0xFFFA00F4) // (SSC1) SSC IPNAME2 REGISTER 
// ========== Register definition for PWMC_CH0 peripheral ========== 
#define AT91C_PWMC_CH0_CCNTR ((AT91_REG *) 	0xFFFB820C) // (PWMC_CH0) Channel Counter Register
#define AT91C_PWMC_CH0_CPRDR ((AT91_REG *) 	0xFFFB8208) // (PWMC_CH0) Channel Period Register
#define AT91C_PWMC_CH0_CUPDR ((AT91_REG *) 	0xFFFB8210) // (PWMC_CH0) Channel Update Register
#define AT91C_PWMC_CH0_CDTYR ((AT91_REG *) 	0xFFFB8204) // (PWMC_CH0) Channel Duty Cycle Register
#define AT91C_PWMC_CH0_CMR ((AT91_REG *) 	0xFFFB8200) // (PWMC_CH0) Channel Mode Register
#define AT91C_PWMC_CH0_Reserved ((AT91_REG *) 	0xFFFB8214) // (PWMC_CH0) Reserved
// ========== Register definition for PWMC_CH1 peripheral ========== 
#define AT91C_PWMC_CH1_CCNTR ((AT91_REG *) 	0xFFFB822C) // (PWMC_CH1) Channel Counter Register
#define AT91C_PWMC_CH1_CDTYR ((AT91_REG *) 	0xFFFB8224) // (PWMC_CH1) Channel Duty Cycle Register
#define AT91C_PWMC_CH1_CMR ((AT91_REG *) 	0xFFFB8220) // (PWMC_CH1) Channel Mode Register
#define AT91C_PWMC_CH1_CPRDR ((AT91_REG *) 	0xFFFB8228) // (PWMC_CH1) Channel Period Register
#define AT91C_PWMC_CH1_Reserved ((AT91_REG *) 	0xFFFB8234) // (PWMC_CH1) Reserved
#define AT91C_PWMC_CH1_CUPDR ((AT91_REG *) 	0xFFFB8230) // (PWMC_CH1) Channel Update Register
// ========== Register definition for PWMC_CH2 peripheral ========== 
#define AT91C_PWMC_CH2_CUPDR ((AT91_REG *) 	0xFFFB8250) // (PWMC_CH2) Channel Update Register
#define AT91C_PWMC_CH2_CMR ((AT91_REG *) 	0xFFFB8240) // (PWMC_CH2) Channel Mode Register
#define AT91C_PWMC_CH2_Reserved ((AT91_REG *) 	0xFFFB8254) // (PWMC_CH2) Reserved
#define AT91C_PWMC_CH2_CPRDR ((AT91_REG *) 	0xFFFB8248) // (PWMC_CH2) Channel Period Register
#define AT91C_PWMC_CH2_CDTYR ((AT91_REG *) 	0xFFFB8244) // (PWMC_CH2) Channel Duty Cycle Register
#define AT91C_PWMC_CH2_CCNTR ((AT91_REG *) 	0xFFFB824C) // (PWMC_CH2) Channel Counter Register
// ========== Register definition for PWMC_CH3 peripheral ========== 
#define AT91C_PWMC_CH3_CPRDR ((AT91_REG *) 	0xFFFB8268) // (PWMC_CH3) Channel Period Register
#define AT91C_PWMC_CH3_Reserved ((AT91_REG *) 	0xFFFB8274) // (PWMC_CH3) Reserved
#define AT91C_PWMC_CH3_CUPDR ((AT91_REG *) 	0xFFFB8270) // (PWMC_CH3) Channel Update Register
#define AT91C_PWMC_CH3_CDTYR ((AT91_REG *) 	0xFFFB8264) // (PWMC_CH3) Channel Duty Cycle Register
#define AT91C_PWMC_CH3_CCNTR ((AT91_REG *) 	0xFFFB826C) // (PWMC_CH3) Channel Counter Register
#define AT91C_PWMC_CH3_CMR ((AT91_REG *) 	0xFFFB8260) // (PWMC_CH3) Channel Mode Register
// ========== Register definition for PWMC peripheral ========== 
#define AT91C_PWMC_IDR  ((AT91_REG *) 	0xFFFB8014) // (PWMC) PWMC Interrupt Disable Register
#define AT91C_PWMC_MR   ((AT91_REG *) 	0xFFFB8000) // (PWMC) PWMC Mode Register
#define AT91C_PWMC_VR   ((AT91_REG *) 	0xFFFB80FC) // (PWMC) PWMC Version Register
#define AT91C_PWMC_IMR  ((AT91_REG *) 	0xFFFB8018) // (PWMC) PWMC Interrupt Mask Register
#define AT91C_PWMC_SR   ((AT91_REG *) 	0xFFFB800C) // (PWMC) PWMC Status Register
#define AT91C_PWMC_ISR  ((AT91_REG *) 	0xFFFB801C) // (PWMC) PWMC Interrupt Status Register
#define AT91C_PWMC_ENA  ((AT91_REG *) 	0xFFFB8004) // (PWMC) PWMC Enable Register
#define AT91C_PWMC_IER  ((AT91_REG *) 	0xFFFB8010) // (PWMC) PWMC Interrupt Enable Register
#define AT91C_PWMC_DIS  ((AT91_REG *) 	0xFFFB8008) // (PWMC) PWMC Disable Register
// ========== Register definition for PDC_SPI0 peripheral ========== 
#define AT91C_SPI0_TPR  ((AT91_REG *) 	0xFFFA4108) // (PDC_SPI0) Transmit Pointer Register
#define AT91C_SPI0_PTCR ((AT91_REG *) 	0xFFFA4120) // (PDC_SPI0) PDC Transfer Control Register
#define AT91C_SPI0_RNPR ((AT91_REG *) 	0xFFFA4110) // (PDC_SPI0) Receive Next Pointer Register
#define AT91C_SPI0_TNCR ((AT91_REG *) 	0xFFFA411C) // (PDC_SPI0) Transmit Next Counter Register
#define AT91C_SPI0_TCR  ((AT91_REG *) 	0xFFFA410C) // (PDC_SPI0) Transmit Counter Register
#define AT91C_SPI0_RCR  ((AT91_REG *) 	0xFFFA4104) // (PDC_SPI0) Receive Counter Register
#define AT91C_SPI0_RNCR ((AT91_REG *) 	0xFFFA4114) // (PDC_SPI0) Receive Next Counter Register
#define AT91C_SPI0_TNPR ((AT91_REG *) 	0xFFFA4118) // (PDC_SPI0) Transmit Next Pointer Register
#define AT91C_SPI0_RPR  ((AT91_REG *) 	0xFFFA4100) // (PDC_SPI0) Receive Pointer Register
#define AT91C_SPI0_PTSR ((AT91_REG *) 	0xFFFA4124) // (PDC_SPI0) PDC Transfer Status Register
// ========== Register definition for PDC_SPI1 peripheral ========== 
#define AT91C_SPI1_RNCR ((AT91_REG *) 	0xFFFA8114) // (PDC_SPI1) Receive Next Counter Register
#define AT91C_SPI1_TCR  ((AT91_REG *) 	0xFFFA810C) // (PDC_SPI1) Transmit Counter Register
#define AT91C_SPI1_RCR  ((AT91_REG *) 	0xFFFA8104) // (PDC_SPI1) Receive Counter Register
#define AT91C_SPI1_TNPR ((AT91_REG *) 	0xFFFA8118) // (PDC_SPI1) Transmit Next Pointer Register
#define AT91C_SPI1_RNPR ((AT91_REG *) 	0xFFFA8110) // (PDC_SPI1) Receive Next Pointer Register
#define AT91C_SPI1_RPR  ((AT91_REG *) 	0xFFFA8100) // (PDC_SPI1) Receive Pointer Register
#define AT91C_SPI1_TNCR ((AT91_REG *) 	0xFFFA811C) // (PDC_SPI1) Transmit Next Counter Register
#define AT91C_SPI1_TPR  ((AT91_REG *) 	0xFFFA8108) // (PDC_SPI1) Transmit Pointer Register
#define AT91C_SPI1_PTSR ((AT91_REG *) 	0xFFFA8124) // (PDC_SPI1) PDC Transfer Status Register
#define AT91C_SPI1_PTCR ((AT91_REG *) 	0xFFFA8120) // (PDC_SPI1) PDC Transfer Control Register
// ========== Register definition for SPI0 peripheral ========== 
#define AT91C_SPI0_MR   ((AT91_REG *) 	0xFFFA4004) // (SPI0) Mode Register
#define AT91C_SPI0_RDR  ((AT91_REG *) 	0xFFFA4008) // (SPI0) Receive Data Register
#define AT91C_SPI0_CR   ((AT91_REG *) 	0xFFFA4000) // (SPI0) Control Register
#define AT91C_SPI0_IER  ((AT91_REG *) 	0xFFFA4014) // (SPI0) Interrupt Enable Register
#define AT91C_SPI0_TDR  ((AT91_REG *) 	0xFFFA400C) // (SPI0) Transmit Data Register
#define AT91C_SPI0_IDR  ((AT91_REG *) 	0xFFFA4018) // (SPI0) Interrupt Disable Register
#define AT91C_SPI0_CSR  ((AT91_REG *) 	0xFFFA4030) // (SPI0) Chip Select Register
#define AT91C_SPI0_SR   ((AT91_REG *) 	0xFFFA4010) // (SPI0) Status Register
#define AT91C_SPI0_IMR  ((AT91_REG *) 	0xFFFA401C) // (SPI0) Interrupt Mask Register
// ========== Register definition for SPI1 peripheral ========== 
#define AT91C_SPI1_CSR  ((AT91_REG *) 	0xFFFA8030) // (SPI1) Chip Select Register
#define AT91C_SPI1_IER  ((AT91_REG *) 	0xFFFA8014) // (SPI1) Interrupt Enable Register
#define AT91C_SPI1_RDR  ((AT91_REG *) 	0xFFFA8008) // (SPI1) Receive Data Register
#define AT91C_SPI1_IDR  ((AT91_REG *) 	0xFFFA8018) // (SPI1) Interrupt Disable Register
#define AT91C_SPI1_MR   ((AT91_REG *) 	0xFFFA8004) // (SPI1) Mode Register
#define AT91C_SPI1_CR   ((AT91_REG *) 	0xFFFA8000) // (SPI1) Control Register
#define AT91C_SPI1_SR   ((AT91_REG *) 	0xFFFA8010) // (SPI1) Status Register
#define AT91C_SPI1_TDR  ((AT91_REG *) 	0xFFFA800C) // (SPI1) Transmit Data Register
#define AT91C_SPI1_IMR  ((AT91_REG *) 	0xFFFA801C) // (SPI1) Interrupt Mask Register
// ========== Register definition for PDC_TSADC peripheral ========== 
#define AT91C_TSADC_TCR ((AT91_REG *) 	0xFFFB010C) // (PDC_TSADC) Transmit Counter Register
#define AT91C_TSADC_PTCR ((AT91_REG *) 	0xFFFB0120) // (PDC_TSADC) PDC Transfer Control Register
#define AT91C_TSADC_RNCR ((AT91_REG *) 	0xFFFB0114) // (PDC_TSADC) Receive Next Counter Register
#define AT91C_TSADC_PTSR ((AT91_REG *) 	0xFFFB0124) // (PDC_TSADC) PDC Transfer Status Register
#define AT91C_TSADC_TNCR ((AT91_REG *) 	0xFFFB011C) // (PDC_TSADC) Transmit Next Counter Register
#define AT91C_TSADC_RNPR ((AT91_REG *) 	0xFFFB0110) // (PDC_TSADC) Receive Next Pointer Register
#define AT91C_TSADC_RCR ((AT91_REG *) 	0xFFFB0104) // (PDC_TSADC) Receive Counter Register
#define AT91C_TSADC_TPR ((AT91_REG *) 	0xFFFB0108) // (PDC_TSADC) Transmit Pointer Register
#define AT91C_TSADC_TNPR ((AT91_REG *) 	0xFFFB0118) // (PDC_TSADC) Transmit Next Pointer Register
#define AT91C_TSADC_RPR ((AT91_REG *) 	0xFFFB0100) // (PDC_TSADC) Receive Pointer Register
// ========== Register definition for TSADC peripheral ========== 
#define AT91C_TSADC_CHSR ((AT91_REG *) 	0xFFFB0018) // (TSADC) Channel Status Register
#define AT91C_TSADC_CDR5 ((AT91_REG *) 	0xFFFB0044) // (TSADC) Channel Data Register 5
#define AT91C_TSADC_CR  ((AT91_REG *) 	0xFFFB0000) // (TSADC) Control Register
#define AT91C_TSADC_IMR ((AT91_REG *) 	0xFFFB002C) // (TSADC) Interrupt Mask Register
#define AT91C_TSADC_CHDR ((AT91_REG *) 	0xFFFB0014) // (TSADC) Channel Disable Register
#define AT91C_TSADC_LCDR ((AT91_REG *) 	0xFFFB0020) // (TSADC) Last Converted Register
#define AT91C_TSADC_IER ((AT91_REG *) 	0xFFFB0024) // (TSADC) Interrupt Enable Register
#define AT91C_TSADC_TSR ((AT91_REG *) 	0xFFFB000C) // (TSADC) Touch Screen Register
#define AT91C_TSADC_CDR2 ((AT91_REG *) 	0xFFFB0038) // (TSADC) Channel Data Register 2
#define AT91C_TSADC_CDR4 ((AT91_REG *) 	0xFFFB0040) // (TSADC) Channel Data Register 4
#define AT91C_TSADC_CHER ((AT91_REG *) 	0xFFFB0010) // (TSADC) Channel Enable Register
#define AT91C_TSADC_TRGR ((AT91_REG *) 	0xFFFB0008) // (TSADC) Trigger Register
#define AT91C_TSADC_CDR3 ((AT91_REG *) 	0xFFFB003C) // (TSADC) Channel Data Register 3
#define AT91C_TSADC_SR  ((AT91_REG *) 	0xFFFB001C) // (TSADC) Status Register
#define AT91C_TSADC_CDR0 ((AT91_REG *) 	0xFFFB0030) // (TSADC) Channel Data Register 0
#define AT91C_TSADC_CDR6 ((AT91_REG *) 	0xFFFB0048) // (TSADC) Channel Data Register 6
#define AT91C_TSADC_IDR ((AT91_REG *) 	0xFFFB0028) // (TSADC) Interrupt Disable Register
#define AT91C_TSADC_MR  ((AT91_REG *) 	0xFFFB0004) // (TSADC) Mode Register
#define AT91C_TSADC_CDR7 ((AT91_REG *) 	0xFFFB004C) // (TSADC) Channel Data Register 7
#define AT91C_TSADC_CDR1 ((AT91_REG *) 	0xFFFB0034) // (TSADC) Channel Data Register 1
// ========== Register definition for UDPHS_EPTFIFO peripheral ========== 
#define AT91C_UDPHS_EPTFIFO_READEPT3 ((AT91_REG *) 	0x00630000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 3
#define AT91C_UDPHS_EPTFIFO_READEPT5 ((AT91_REG *) 	0x00650000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 5
#define AT91C_UDPHS_EPTFIFO_READEPT1 ((AT91_REG *) 	0x00610000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 1
#define AT91C_UDPHS_EPTFIFO_READEPT0 ((AT91_REG *) 	0x00600000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 0
#define AT91C_UDPHS_EPTFIFO_READEPT6 ((AT91_REG *) 	0x00660000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 6
#define AT91C_UDPHS_EPTFIFO_READEPT2 ((AT91_REG *) 	0x00620000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 2
#define AT91C_UDPHS_EPTFIFO_READEPT4 ((AT91_REG *) 	0x00640000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 4
// ========== Register definition for UDPHS_EPT_0 peripheral ========== 
#define AT91C_UDPHS_EPT_0_EPTSTA ((AT91_REG *) 	0xFFF7811C) // (UDPHS_EPT_0) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_0_EPTCTL ((AT91_REG *) 	0xFFF7810C) // (UDPHS_EPT_0) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_0_EPTCTLDIS ((AT91_REG *) 	0xFFF78108) // (UDPHS_EPT_0) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_0_EPTCFG ((AT91_REG *) 	0xFFF78100) // (UDPHS_EPT_0) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_0_EPTCLRSTA ((AT91_REG *) 	0xFFF78118) // (UDPHS_EPT_0) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_0_EPTSETSTA ((AT91_REG *) 	0xFFF78114) // (UDPHS_EPT_0) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_0_EPTCTLENB ((AT91_REG *) 	0xFFF78104) // (UDPHS_EPT_0) UDPHS Endpoint Control Enable Register
// ========== Register definition for UDPHS_EPT_1 peripheral ========== 
#define AT91C_UDPHS_EPT_1_EPTCTLENB ((AT91_REG *) 	0xFFF78124) // (UDPHS_EPT_1) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_1_EPTCFG ((AT91_REG *) 	0xFFF78120) // (UDPHS_EPT_1) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_1_EPTCTL ((AT91_REG *) 	0xFFF7812C) // (UDPHS_EPT_1) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_1_EPTSTA ((AT91_REG *) 	0xFFF7813C) // (UDPHS_EPT_1) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_1_EPTCLRSTA ((AT91_REG *) 	0xFFF78138) // (UDPHS_EPT_1) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_1_EPTSETSTA ((AT91_REG *) 	0xFFF78134) // (UDPHS_EPT_1) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_1_EPTCTLDIS ((AT91_REG *) 	0xFFF78128) // (UDPHS_EPT_1) UDPHS Endpoint Control Disable Register
// ========== Register definition for UDPHS_EPT_2 peripheral ========== 
#define AT91C_UDPHS_EPT_2_EPTCLRSTA ((AT91_REG *) 	0xFFF78158) // (UDPHS_EPT_2) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_2_EPTCTLDIS ((AT91_REG *) 	0xFFF78148) // (UDPHS_EPT_2) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_2_EPTSTA ((AT91_REG *) 	0xFFF7815C) // (UDPHS_EPT_2) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_2_EPTSETSTA ((AT91_REG *) 	0xFFF78154) // (UDPHS_EPT_2) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_2_EPTCTL ((AT91_REG *) 	0xFFF7814C) // (UDPHS_EPT_2) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_2_EPTCFG ((AT91_REG *) 	0xFFF78140) // (UDPHS_EPT_2) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_2_EPTCTLENB ((AT91_REG *) 	0xFFF78144) // (UDPHS_EPT_2) UDPHS Endpoint Control Enable Register
// ========== Register definition for UDPHS_EPT_3 peripheral ========== 
#define AT91C_UDPHS_EPT_3_EPTCTL ((AT91_REG *) 	0xFFF7816C) // (UDPHS_EPT_3) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_3_EPTCLRSTA ((AT91_REG *) 	0xFFF78178) // (UDPHS_EPT_3) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_3_EPTCTLDIS ((AT91_REG *) 	0xFFF78168) // (UDPHS_EPT_3) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_3_EPTSTA ((AT91_REG *) 	0xFFF7817C) // (UDPHS_EPT_3) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_3_EPTSETSTA ((AT91_REG *) 	0xFFF78174) // (UDPHS_EPT_3) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_3_EPTCTLENB ((AT91_REG *) 	0xFFF78164) // (UDPHS_EPT_3) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_3_EPTCFG ((AT91_REG *) 	0xFFF78160) // (UDPHS_EPT_3) UDPHS Endpoint Config Register
// ========== Register definition for UDPHS_EPT_4 peripheral ========== 
#define AT91C_UDPHS_EPT_4_EPTCLRSTA ((AT91_REG *) 	0xFFF78198) // (UDPHS_EPT_4) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_4_EPTCTL ((AT91_REG *) 	0xFFF7818C) // (UDPHS_EPT_4) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_4_EPTCTLENB ((AT91_REG *) 	0xFFF78184) // (UDPHS_EPT_4) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_4_EPTSTA ((AT91_REG *) 	0xFFF7819C) // (UDPHS_EPT_4) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_4_EPTSETSTA ((AT91_REG *) 	0xFFF78194) // (UDPHS_EPT_4) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_4_EPTCFG ((AT91_REG *) 	0xFFF78180) // (UDPHS_EPT_4) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_4_EPTCTLDIS ((AT91_REG *) 	0xFFF78188) // (UDPHS_EPT_4) UDPHS Endpoint Control Disable Register
// ========== Register definition for UDPHS_EPT_5 peripheral ========== 
#define AT91C_UDPHS_EPT_5_EPTSTA ((AT91_REG *) 	0xFFF781BC) // (UDPHS_EPT_5) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_5_EPTCLRSTA ((AT91_REG *) 	0xFFF781B8) // (UDPHS_EPT_5) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_5_EPTCTLENB ((AT91_REG *) 	0xFFF781A4) // (UDPHS_EPT_5) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_5_EPTSETSTA ((AT91_REG *) 	0xFFF781B4) // (UDPHS_EPT_5) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_5_EPTCTLDIS ((AT91_REG *) 	0xFFF781A8) // (UDPHS_EPT_5) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_5_EPTCFG ((AT91_REG *) 	0xFFF781A0) // (UDPHS_EPT_5) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_5_EPTCTL ((AT91_REG *) 	0xFFF781AC) // (UDPHS_EPT_5) UDPHS Endpoint Control Register
// ========== Register definition for UDPHS_EPT_6 peripheral ========== 
#define AT91C_UDPHS_EPT_6_EPTCLRSTA ((AT91_REG *) 	0xFFF781D8) // (UDPHS_EPT_6) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_6_EPTCTLENB ((AT91_REG *) 	0xFFF781C4) // (UDPHS_EPT_6) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_6_EPTCTL ((AT91_REG *) 	0xFFF781CC) // (UDPHS_EPT_6) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_6_EPTSETSTA ((AT91_REG *) 	0xFFF781D4) // (UDPHS_EPT_6) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_6_EPTCTLDIS ((AT91_REG *) 	0xFFF781C8) // (UDPHS_EPT_6) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_6_EPTSTA ((AT91_REG *) 	0xFFF781DC) // (UDPHS_EPT_6) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_6_EPTCFG ((AT91_REG *) 	0xFFF781C0) // (UDPHS_EPT_6) UDPHS Endpoint Config Register
// ========== Register definition for UDPHS_DMA_1 peripheral ========== 
#define AT91C_UDPHS_DMA_1_DMASTATUS ((AT91_REG *) 	0xFFF7831C) // (UDPHS_DMA_1) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_1_DMANXTDSC ((AT91_REG *) 	0xFFF78310) // (UDPHS_DMA_1) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_1_DMACONTROL ((AT91_REG *) 	0xFFF78318) // (UDPHS_DMA_1) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_1_DMAADDRESS ((AT91_REG *) 	0xFFF78314) // (UDPHS_DMA_1) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS_DMA_2 peripheral ========== 
#define AT91C_UDPHS_DMA_2_DMACONTROL ((AT91_REG *) 	0xFFF78328) // (UDPHS_DMA_2) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_2_DMASTATUS ((AT91_REG *) 	0xFFF7832C) // (UDPHS_DMA_2) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_2_DMAADDRESS ((AT91_REG *) 	0xFFF78324) // (UDPHS_DMA_2) UDPHS DMA Channel Address Register
#define AT91C_UDPHS_DMA_2_DMANXTDSC ((AT91_REG *) 	0xFFF78320) // (UDPHS_DMA_2) UDPHS DMA Channel Next Descriptor Address
// ========== Register definition for UDPHS_DMA_3 peripheral ========== 
#define AT91C_UDPHS_DMA_3_DMAADDRESS ((AT91_REG *) 	0xFFF78334) // (UDPHS_DMA_3) UDPHS DMA Channel Address Register
#define AT91C_UDPHS_DMA_3_DMANXTDSC ((AT91_REG *) 	0xFFF78330) // (UDPHS_DMA_3) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_3_DMACONTROL ((AT91_REG *) 	0xFFF78338) // (UDPHS_DMA_3) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_3_DMASTATUS ((AT91_REG *) 	0xFFF7833C) // (UDPHS_DMA_3) UDPHS DMA Channel Status Register
// ========== Register definition for UDPHS_DMA_4 peripheral ========== 
#define AT91C_UDPHS_DMA_4_DMANXTDSC ((AT91_REG *) 	0xFFF78340) // (UDPHS_DMA_4) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_4_DMAADDRESS ((AT91_REG *) 	0xFFF78344) // (UDPHS_DMA_4) UDPHS DMA Channel Address Register
#define AT91C_UDPHS_DMA_4_DMACONTROL ((AT91_REG *) 	0xFFF78348) // (UDPHS_DMA_4) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_4_DMASTATUS ((AT91_REG *) 	0xFFF7834C) // (UDPHS_DMA_4) UDPHS DMA Channel Status Register
// ========== Register definition for UDPHS_DMA_5 peripheral ========== 
#define AT91C_UDPHS_DMA_5_DMASTATUS ((AT91_REG *) 	0xFFF7835C) // (UDPHS_DMA_5) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_5_DMACONTROL ((AT91_REG *) 	0xFFF78358) // (UDPHS_DMA_5) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_5_DMANXTDSC ((AT91_REG *) 	0xFFF78350) // (UDPHS_DMA_5) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_5_DMAADDRESS ((AT91_REG *) 	0xFFF78354) // (UDPHS_DMA_5) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS_DMA_6 peripheral ========== 
#define AT91C_UDPHS_DMA_6_DMANXTDSC ((AT91_REG *) 	0xFFF78360) // (UDPHS_DMA_6) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_6_DMACONTROL ((AT91_REG *) 	0xFFF78368) // (UDPHS_DMA_6) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_6_DMASTATUS ((AT91_REG *) 	0xFFF7836C) // (UDPHS_DMA_6) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_6_DMAADDRESS ((AT91_REG *) 	0xFFF78364) // (UDPHS_DMA_6) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS peripheral ========== 
#define AT91C_UDPHS_IEN ((AT91_REG *) 	0xFFF78010) // (UDPHS) UDPHS Interrupt Enable Register
#define AT91C_UDPHS_TSTSOFCNT ((AT91_REG *) 	0xFFF780D0) // (UDPHS) UDPHS Test SOF Counter Register
#define AT91C_UDPHS_IPFEATURES ((AT91_REG *) 	0xFFF780F8) // (UDPHS) UDPHS Features Register
#define AT91C_UDPHS_TST ((AT91_REG *) 	0xFFF780E0) // (UDPHS) UDPHS Test Register
#define AT91C_UDPHS_FNUM ((AT91_REG *) 	0xFFF78004) // (UDPHS) UDPHS Frame Number Register
#define AT91C_UDPHS_TSTCNTB ((AT91_REG *) 	0xFFF780D8) // (UDPHS) UDPHS Test B Counter Register
#define AT91C_UDPHS_RIPPADDRSIZE ((AT91_REG *) 	0xFFF780EC) // (UDPHS) UDPHS PADDRSIZE Register
#define AT91C_UDPHS_INTSTA ((AT91_REG *) 	0xFFF78014) // (UDPHS) UDPHS Interrupt Status Register
#define AT91C_UDPHS_EPTRST ((AT91_REG *) 	0xFFF7801C) // (UDPHS) UDPHS Endpoints Reset Register
#define AT91C_UDPHS_TSTCNTA ((AT91_REG *) 	0xFFF780D4) // (UDPHS) UDPHS Test A Counter Register
#define AT91C_UDPHS_RIPNAME2 ((AT91_REG *) 	0xFFF780F4) // (UDPHS) UDPHS Name2 Register
#define AT91C_UDPHS_RIPNAME1 ((AT91_REG *) 	0xFFF780F0) // (UDPHS) UDPHS Name1 Register
#define AT91C_UDPHS_TSTMODREG ((AT91_REG *) 	0xFFF780DC) // (UDPHS) UDPHS Test Mode Register
#define AT91C_UDPHS_CLRINT ((AT91_REG *) 	0xFFF78018) // (UDPHS) UDPHS Clear Interrupt Register
#define AT91C_UDPHS_IPVERSION ((AT91_REG *) 	0xFFF780FC) // (UDPHS) UDPHS Version Register
#define AT91C_UDPHS_CTRL ((AT91_REG *) 	0xFFF78000) // (UDPHS) UDPHS Control Register
// ========== Register definition for PDC_AC97C peripheral ========== 
#define AT91C_AC97C_PTSR ((AT91_REG *) 	0xFFFAC124) // (PDC_AC97C) PDC Transfer Status Register
#define AT91C_AC97C_RPR ((AT91_REG *) 	0xFFFAC100) // (PDC_AC97C) Receive Pointer Register
#define AT91C_AC97C_RNCR ((AT91_REG *) 	0xFFFAC114) // (PDC_AC97C) Receive Next Counter Register
#define AT91C_AC97C_RCR ((AT91_REG *) 	0xFFFAC104) // (PDC_AC97C) Receive Counter Register
#define AT91C_AC97C_PTCR ((AT91_REG *) 	0xFFFAC120) // (PDC_AC97C) PDC Transfer Control Register
#define AT91C_AC97C_TPR ((AT91_REG *) 	0xFFFAC108) // (PDC_AC97C) Transmit Pointer Register
#define AT91C_AC97C_RNPR ((AT91_REG *) 	0xFFFAC110) // (PDC_AC97C) Receive Next Pointer Register
#define AT91C_AC97C_TNPR ((AT91_REG *) 	0xFFFAC118) // (PDC_AC97C) Transmit Next Pointer Register
#define AT91C_AC97C_TCR ((AT91_REG *) 	0xFFFAC10C) // (PDC_AC97C) Transmit Counter Register
#define AT91C_AC97C_TNCR ((AT91_REG *) 	0xFFFAC11C) // (PDC_AC97C) Transmit Next Counter Register
// ========== Register definition for AC97C peripheral ========== 
#define AT91C_AC97C_IER ((AT91_REG *) 	0xFFFAC054) // (AC97C) Interrupt Enable Register
#define AT91C_AC97C_COTHR ((AT91_REG *) 	0xFFFAC044) // (AC97C) COdec Transmit Holding Register
#define AT91C_AC97C_IDR ((AT91_REG *) 	0xFFFAC058) // (AC97C) Interrupt Disable Register
#define AT91C_AC97C_ICA ((AT91_REG *) 	0xFFFAC010) // (AC97C) Input Channel AssignementRegister
#define AT91C_AC97C_CATHR ((AT91_REG *) 	0xFFFAC024) // (AC97C) Channel A Transmit Holding Register
#define AT91C_AC97C_CBSR ((AT91_REG *) 	0xFFFAC038) // (AC97C) Channel B Status Register
#define AT91C_AC97C_CAMR ((AT91_REG *) 	0xFFFAC02C) // (AC97C) Channel A Mode Register
#define AT91C_AC97C_SR  ((AT91_REG *) 	0xFFFAC050) // (AC97C) Status Register
#define AT91C_AC97C_CBTHR ((AT91_REG *) 	0xFFFAC034) // (AC97C) Channel B Transmit Holding Register (optional)
#define AT91C_AC97C_CASR ((AT91_REG *) 	0xFFFAC028) // (AC97C) Channel A Status Register
#define AT91C_AC97C_COSR ((AT91_REG *) 	0xFFFAC048) // (AC97C) CODEC Status Register
#define AT91C_AC97C_MR  ((AT91_REG *) 	0xFFFAC008) // (AC97C) Mode Register
#define AT91C_AC97C_OCA ((AT91_REG *) 	0xFFFAC014) // (AC97C) Output Channel Assignement Register
#define AT91C_AC97C_CORHR ((AT91_REG *) 	0xFFFAC040) // (AC97C) COdec Transmit Holding Register
#define AT91C_AC97C_CBRHR ((AT91_REG *) 	0xFFFAC030) // (AC97C) Channel B Receive Holding Register (optional)
#define AT91C_AC97C_IMR ((AT91_REG *) 	0xFFFAC05C) // (AC97C) Interrupt Mask Register
#define AT91C_AC97C_COMR ((AT91_REG *) 	0xFFFAC04C) // (AC97C) CODEC Mask Status Register
#define AT91C_AC97C_CARHR ((AT91_REG *) 	0xFFFAC020) // (AC97C) Channel A Receive Holding Register
#define AT91C_AC97C_VERSION ((AT91_REG *) 	0xFFFAC0FC) // (AC97C) Version Register
#define AT91C_AC97C_CBMR ((AT91_REG *) 	0xFFFAC03C) // (AC97C) Channel B Mode Register
// ========== Register definition for LCDC peripheral ========== 
#define AT91C_LCDC_MVAL ((AT91_REG *) 	0x00500818) // (LCDC) LCD Mode Toggle Rate Value Register
#define AT91C_LCDC_PWRCON ((AT91_REG *) 	0x0050083C) // (LCDC) Power Control Register
#define AT91C_LCDC_ISR  ((AT91_REG *) 	0x00500854) // (LCDC) Interrupt Enable Register
#define AT91C_LCDC_FRMP1 ((AT91_REG *) 	0x00500008) // (LCDC) DMA Frame Pointer Register 1
#define AT91C_LCDC_CTRSTVAL ((AT91_REG *) 	0x00500844) // (LCDC) Contrast Value Register
#define AT91C_LCDC_ICR  ((AT91_REG *) 	0x00500858) // (LCDC) Interrupt Clear Register
#define AT91C_LCDC_TIM1 ((AT91_REG *) 	0x00500808) // (LCDC) LCD Timing Config 1 Register
#define AT91C_LCDC_DMACON ((AT91_REG *) 	0x0050001C) // (LCDC) DMA Control Register
#define AT91C_LCDC_ITR  ((AT91_REG *) 	0x00500860) // (LCDC) Interrupts Test Register
#define AT91C_LCDC_IDR  ((AT91_REG *) 	0x0050084C) // (LCDC) Interrupt Disable Register
#define AT91C_LCDC_DP4_7 ((AT91_REG *) 	0x00500820) // (LCDC) Dithering Pattern DP4_7 Register
#define AT91C_LCDC_DP5_7 ((AT91_REG *) 	0x0050082C) // (LCDC) Dithering Pattern DP5_7 Register
#define AT91C_LCDC_IRR  ((AT91_REG *) 	0x00500864) // (LCDC) Interrupts Raw Status Register
#define AT91C_LCDC_DP3_4 ((AT91_REG *) 	0x00500830) // (LCDC) Dithering Pattern DP3_4 Register
#define AT91C_LCDC_IMR  ((AT91_REG *) 	0x00500850) // (LCDC) Interrupt Mask Register
#define AT91C_LCDC_LCDFRCFG ((AT91_REG *) 	0x00500810) // (LCDC) LCD Frame Config Register
#define AT91C_LCDC_CTRSTCON ((AT91_REG *) 	0x00500840) // (LCDC) Contrast Control Register
#define AT91C_LCDC_DP1_2 ((AT91_REG *) 	0x0050081C) // (LCDC) Dithering Pattern DP1_2 Register
#define AT91C_LCDC_FRMP2 ((AT91_REG *) 	0x0050000C) // (LCDC) DMA Frame Pointer Register 2
#define AT91C_LCDC_LCDCON1 ((AT91_REG *) 	0x00500800) // (LCDC) LCD Control 1 Register
#define AT91C_LCDC_DP4_5 ((AT91_REG *) 	0x00500834) // (LCDC) Dithering Pattern DP4_5 Register
#define AT91C_LCDC_FRMA2 ((AT91_REG *) 	0x00500014) // (LCDC) DMA Frame Address Register 2
#define AT91C_LCDC_BA1  ((AT91_REG *) 	0x00500000) // (LCDC) DMA Base Address Register 1
#define AT91C_LCDC_DMA2DCFG ((AT91_REG *) 	0x00500020) // (LCDC) DMA 2D addressing configuration
#define AT91C_LCDC_LUT_ENTRY ((AT91_REG *) 	0x00500C00) // (LCDC) LUT Entries Register
#define AT91C_LCDC_DP6_7 ((AT91_REG *) 	0x00500838) // (LCDC) Dithering Pattern DP6_7 Register
#define AT91C_LCDC_FRMCFG ((AT91_REG *) 	0x00500018) // (LCDC) DMA Frame Configuration Register
#define AT91C_LCDC_TIM2 ((AT91_REG *) 	0x0050080C) // (LCDC) LCD Timing Config 2 Register
#define AT91C_LCDC_DP3_5 ((AT91_REG *) 	0x00500824) // (LCDC) Dithering Pattern DP3_5 Register
#define AT91C_LCDC_FRMA1 ((AT91_REG *) 	0x00500010) // (LCDC) DMA Frame Address Register 1
#define AT91C_LCDC_IER  ((AT91_REG *) 	0x00500848) // (LCDC) Interrupt Enable Register
#define AT91C_LCDC_DP2_3 ((AT91_REG *) 	0x00500828) // (LCDC) Dithering Pattern DP2_3 Register
#define AT91C_LCDC_FIFO ((AT91_REG *) 	0x00500814) // (LCDC) LCD FIFO Register
#define AT91C_LCDC_BA2  ((AT91_REG *) 	0x00500004) // (LCDC) DMA Base Address Register 2
#define AT91C_LCDC_LCDCON2 ((AT91_REG *) 	0x00500804) // (LCDC) LCD Control 2 Register
#define AT91C_LCDC_GPR  ((AT91_REG *) 	0x0050085C) // (LCDC) General Purpose Register
// ========== Register definition for LCDC_16B_TFT peripheral ========== 
#define AT91C_TFT_MVAL  ((AT91_REG *) 	0x00500818) // (LCDC_16B_TFT) LCD Mode Toggle Rate Value Register
#define AT91C_TFT_PWRCON ((AT91_REG *) 	0x0050083C) // (LCDC_16B_TFT) Power Control Register
#define AT91C_TFT_ISR   ((AT91_REG *) 	0x00500854) // (LCDC_16B_TFT) Interrupt Enable Register
#define AT91C_TFT_FRMP1 ((AT91_REG *) 	0x00500008) // (LCDC_16B_TFT) DMA Frame Pointer Register 1
#define AT91C_TFT_CTRSTVAL ((AT91_REG *) 	0x00500844) // (LCDC_16B_TFT) Contrast Value Register
#define AT91C_TFT_ICR   ((AT91_REG *) 	0x00500858) // (LCDC_16B_TFT) Interrupt Clear Register
#define AT91C_TFT_TIM1  ((AT91_REG *) 	0x00500808) // (LCDC_16B_TFT) LCD Timing Config 1 Register
#define AT91C_TFT_DMACON ((AT91_REG *) 	0x0050001C) // (LCDC_16B_TFT) DMA Control Register
#define AT91C_TFT_ITR   ((AT91_REG *) 	0x00500860) // (LCDC_16B_TFT) Interrupts Test Register
#define AT91C_TFT_IDR   ((AT91_REG *) 	0x0050084C) // (LCDC_16B_TFT) Interrupt Disable Register
#define AT91C_TFT_DP4_7 ((AT91_REG *) 	0x00500820) // (LCDC_16B_TFT) Dithering Pattern DP4_7 Register
#define AT91C_TFT_DP5_7 ((AT91_REG *) 	0x0050082C) // (LCDC_16B_TFT) Dithering Pattern DP5_7 Register
#define AT91C_TFT_IRR   ((AT91_REG *) 	0x00500864) // (LCDC_16B_TFT) Interrupts Raw Status Register
#define AT91C_TFT_DP3_4 ((AT91_REG *) 	0x00500830) // (LCDC_16B_TFT) Dithering Pattern DP3_4 Register
#define AT91C_TFT_IMR   ((AT91_REG *) 	0x00500850) // (LCDC_16B_TFT) Interrupt Mask Register
#define AT91C_TFT_LCDFRCFG ((AT91_REG *) 	0x00500810) // (LCDC_16B_TFT) LCD Frame Config Register
#define AT91C_TFT_CTRSTCON ((AT91_REG *) 	0x00500840) // (LCDC_16B_TFT) Contrast Control Register
#define AT91C_TFT_DP1_2 ((AT91_REG *) 	0x0050081C) // (LCDC_16B_TFT) Dithering Pattern DP1_2 Register
#define AT91C_TFT_FRMP2 ((AT91_REG *) 	0x0050000C) // (LCDC_16B_TFT) DMA Frame Pointer Register 2
#define AT91C_TFT_LCDCON1 ((AT91_REG *) 	0x00500800) // (LCDC_16B_TFT) LCD Control 1 Register
#define AT91C_TFT_DP4_5 ((AT91_REG *) 	0x00500834) // (LCDC_16B_TFT) Dithering Pattern DP4_5 Register
#define AT91C_TFT_FRMA2 ((AT91_REG *) 	0x00500014) // (LCDC_16B_TFT) DMA Frame Address Register 2
#define AT91C_TFT_BA1   ((AT91_REG *) 	0x00500000) // (LCDC_16B_TFT) DMA Base Address Register 1
#define AT91C_TFT_DMA2DCFG ((AT91_REG *) 	0x00500020) // (LCDC_16B_TFT) DMA 2D addressing configuration
#define AT91C_TFT_LUT_ENTRY ((AT91_REG *) 	0x00500C00) // (LCDC_16B_TFT) LUT Entries Register
#define AT91C_TFT_DP6_7 ((AT91_REG *) 	0x00500838) // (LCDC_16B_TFT) Dithering Pattern DP6_7 Register
#define AT91C_TFT_FRMCFG ((AT91_REG *) 	0x00500018) // (LCDC_16B_TFT) DMA Frame Configuration Register
#define AT91C_TFT_TIM2  ((AT91_REG *) 	0x0050080C) // (LCDC_16B_TFT) LCD Timing Config 2 Register
#define AT91C_TFT_DP3_5 ((AT91_REG *) 	0x00500824) // (LCDC_16B_TFT) Dithering Pattern DP3_5 Register
#define AT91C_TFT_FRMA1 ((AT91_REG *) 	0x00500010) // (LCDC_16B_TFT) DMA Frame Address Register 1
#define AT91C_TFT_IER   ((AT91_REG *) 	0x00500848) // (LCDC_16B_TFT) Interrupt Enable Register
#define AT91C_TFT_DP2_3 ((AT91_REG *) 	0x00500828) // (LCDC_16B_TFT) Dithering Pattern DP2_3 Register
#define AT91C_TFT_FIFO  ((AT91_REG *) 	0x00500814) // (LCDC_16B_TFT) LCD FIFO Register
#define AT91C_TFT_BA2   ((AT91_REG *) 	0x00500004) // (LCDC_16B_TFT) DMA Base Address Register 2
#define AT91C_TFT_LCDCON2 ((AT91_REG *) 	0x00500804) // (LCDC_16B_TFT) LCD Control 2 Register
#define AT91C_TFT_GPR   ((AT91_REG *) 	0x0050085C) // (LCDC_16B_TFT) General Purpose Register
// ========== Register definition for HDMA_CH_0 peripheral ========== 
#define AT91C_HDMA_CH_0_BDSCR ((AT91_REG *) 	0xFFFFEC5C) // (HDMA_CH_0) HDMA Reserved
#define AT91C_HDMA_CH_0_DADDR ((AT91_REG *) 	0xFFFFEC40) // (HDMA_CH_0) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_0_DPIP ((AT91_REG *) 	0xFFFFEC58) // (HDMA_CH_0) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_0_DSCR ((AT91_REG *) 	0xFFFFEC44) // (HDMA_CH_0) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_0_CFG ((AT91_REG *) 	0xFFFFEC50) // (HDMA_CH_0) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_0_SPIP ((AT91_REG *) 	0xFFFFEC54) // (HDMA_CH_0) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_0_CADDR ((AT91_REG *) 	0xFFFFEC60) // (HDMA_CH_0) HDMA Reserved
#define AT91C_HDMA_CH_0_CTRLA ((AT91_REG *) 	0xFFFFEC48) // (HDMA_CH_0) HDMA Channel Control A Register
#define AT91C_HDMA_CH_0_CTRLB ((AT91_REG *) 	0xFFFFEC4C) // (HDMA_CH_0) HDMA Channel Control B Register
#define AT91C_HDMA_CH_0_SADDR ((AT91_REG *) 	0xFFFFEC3C) // (HDMA_CH_0) HDMA Channel Source Address Register
// ========== Register definition for HDMA_CH_1 peripheral ========== 
#define AT91C_HDMA_CH_1_CADDR ((AT91_REG *) 	0xFFFFEC88) // (HDMA_CH_1) HDMA Reserved
#define AT91C_HDMA_CH_1_DPIP ((AT91_REG *) 	0xFFFFEC80) // (HDMA_CH_1) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_1_CTRLB ((AT91_REG *) 	0xFFFFEC74) // (HDMA_CH_1) HDMA Channel Control B Register
#define AT91C_HDMA_CH_1_SADDR ((AT91_REG *) 	0xFFFFEC64) // (HDMA_CH_1) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_1_BDSCR ((AT91_REG *) 	0xFFFFEC84) // (HDMA_CH_1) HDMA Reserved
#define AT91C_HDMA_CH_1_CFG ((AT91_REG *) 	0xFFFFEC78) // (HDMA_CH_1) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_1_DSCR ((AT91_REG *) 	0xFFFFEC6C) // (HDMA_CH_1) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_1_DADDR ((AT91_REG *) 	0xFFFFEC68) // (HDMA_CH_1) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_1_CTRLA ((AT91_REG *) 	0xFFFFEC70) // (HDMA_CH_1) HDMA Channel Control A Register
#define AT91C_HDMA_CH_1_SPIP ((AT91_REG *) 	0xFFFFEC7C) // (HDMA_CH_1) HDMA Channel Source Picture in Picture Configuration Register
// ========== Register definition for HDMA_CH_2 peripheral ========== 
#define AT91C_HDMA_CH_2_SADDR ((AT91_REG *) 	0xFFFFEC8C) // (HDMA_CH_2) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_2_BDSCR ((AT91_REG *) 	0xFFFFECAC) // (HDMA_CH_2) HDMA Reserved
#define AT91C_HDMA_CH_2_DPIP ((AT91_REG *) 	0xFFFFECA8) // (HDMA_CH_2) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_2_SPIP ((AT91_REG *) 	0xFFFFECA4) // (HDMA_CH_2) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_2_DADDR ((AT91_REG *) 	0xFFFFEC90) // (HDMA_CH_2) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_2_DSCR ((AT91_REG *) 	0xFFFFEC94) // (HDMA_CH_2) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_2_CTRLA ((AT91_REG *) 	0xFFFFEC98) // (HDMA_CH_2) HDMA Channel Control A Register
#define AT91C_HDMA_CH_2_CADDR ((AT91_REG *) 	0xFFFFECB0) // (HDMA_CH_2) HDMA Reserved
#define AT91C_HDMA_CH_2_CFG ((AT91_REG *) 	0xFFFFECA0) // (HDMA_CH_2) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_2_CTRLB ((AT91_REG *) 	0xFFFFEC9C) // (HDMA_CH_2) HDMA Channel Control B Register
// ========== Register definition for HDMA_CH_3 peripheral ========== 
#define AT91C_HDMA_CH_3_SPIP ((AT91_REG *) 	0xFFFFECCC) // (HDMA_CH_3) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_3_CTRLA ((AT91_REG *) 	0xFFFFECC0) // (HDMA_CH_3) HDMA Channel Control A Register
#define AT91C_HDMA_CH_3_DPIP ((AT91_REG *) 	0xFFFFECD0) // (HDMA_CH_3) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_3_CTRLB ((AT91_REG *) 	0xFFFFECC4) // (HDMA_CH_3) HDMA Channel Control B Register
#define AT91C_HDMA_CH_3_BDSCR ((AT91_REG *) 	0xFFFFECD4) // (HDMA_CH_3) HDMA Reserved
#define AT91C_HDMA_CH_3_DSCR ((AT91_REG *) 	0xFFFFECBC) // (HDMA_CH_3) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_3_CADDR ((AT91_REG *) 	0xFFFFECD8) // (HDMA_CH_3) HDMA Reserved
#define AT91C_HDMA_CH_3_CFG ((AT91_REG *) 	0xFFFFECC8) // (HDMA_CH_3) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_3_DADDR ((AT91_REG *) 	0xFFFFECB8) // (HDMA_CH_3) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_3_SADDR ((AT91_REG *) 	0xFFFFECB4) // (HDMA_CH_3) HDMA Channel Source Address Register
// ========== Register definition for HDMA_CH_4 peripheral ========== 
#define AT91C_HDMA_CH_4_CFG ((AT91_REG *) 	0xFFFFECF0) // (HDMA_CH_4) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_4_BDSCR ((AT91_REG *) 	0xFFFFECFC) // (HDMA_CH_4) HDMA Reserved
#define AT91C_HDMA_CH_4_CADDR ((AT91_REG *) 	0xFFFFED00) // (HDMA_CH_4) HDMA Reserved
#define AT91C_HDMA_CH_4_DADDR ((AT91_REG *) 	0xFFFFECE0) // (HDMA_CH_4) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_4_CTRLA ((AT91_REG *) 	0xFFFFECE8) // (HDMA_CH_4) HDMA Channel Control A Register
#define AT91C_HDMA_CH_4_SADDR ((AT91_REG *) 	0xFFFFECDC) // (HDMA_CH_4) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_4_CTRLB ((AT91_REG *) 	0xFFFFECEC) // (HDMA_CH_4) HDMA Channel Control B Register
#define AT91C_HDMA_CH_4_DSCR ((AT91_REG *) 	0xFFFFECE4) // (HDMA_CH_4) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_4_SPIP ((AT91_REG *) 	0xFFFFECF4) // (HDMA_CH_4) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_4_DPIP ((AT91_REG *) 	0xFFFFECF8) // (HDMA_CH_4) HDMA Channel Destination Picture in Picture Configuration Register
// ========== Register definition for HDMA_CH_5 peripheral ========== 
#define AT91C_HDMA_CH_5_CTRLB ((AT91_REG *) 	0xFFFFED14) // (HDMA_CH_5) HDMA Channel Control B Register
#define AT91C_HDMA_CH_5_DADDR ((AT91_REG *) 	0xFFFFED08) // (HDMA_CH_5) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_5_SPIP ((AT91_REG *) 	0xFFFFED1C) // (HDMA_CH_5) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_5_DSCR ((AT91_REG *) 	0xFFFFED0C) // (HDMA_CH_5) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_5_DPIP ((AT91_REG *) 	0xFFFFED20) // (HDMA_CH_5) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_5_CFG ((AT91_REG *) 	0xFFFFED18) // (HDMA_CH_5) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_5_CADDR ((AT91_REG *) 	0xFFFFED28) // (HDMA_CH_5) HDMA Reserved
#define AT91C_HDMA_CH_5_SADDR ((AT91_REG *) 	0xFFFFED04) // (HDMA_CH_5) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_5_BDSCR ((AT91_REG *) 	0xFFFFED24) // (HDMA_CH_5) HDMA Reserved
#define AT91C_HDMA_CH_5_CTRLA ((AT91_REG *) 	0xFFFFED10) // (HDMA_CH_5) HDMA Channel Control A Register
// ========== Register definition for HDMA_CH_6 peripheral ========== 
#define AT91C_HDMA_CH_6_SADDR ((AT91_REG *) 	0xFFFFED2C) // (HDMA_CH_6) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_6_BDSCR ((AT91_REG *) 	0xFFFFED4C) // (HDMA_CH_6) HDMA Reserved
#define AT91C_HDMA_CH_6_CADDR ((AT91_REG *) 	0xFFFFED50) // (HDMA_CH_6) HDMA Reserved
#define AT91C_HDMA_CH_6_SPIP ((AT91_REG *) 	0xFFFFED44) // (HDMA_CH_6) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_6_CTRLB ((AT91_REG *) 	0xFFFFED3C) // (HDMA_CH_6) HDMA Channel Control B Register
#define AT91C_HDMA_CH_6_DSCR ((AT91_REG *) 	0xFFFFED34) // (HDMA_CH_6) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_6_CFG ((AT91_REG *) 	0xFFFFED40) // (HDMA_CH_6) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_6_DPIP ((AT91_REG *) 	0xFFFFED48) // (HDMA_CH_6) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_6_DADDR ((AT91_REG *) 	0xFFFFED30) // (HDMA_CH_6) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_6_CTRLA ((AT91_REG *) 	0xFFFFED38) // (HDMA_CH_6) HDMA Channel Control A Register
// ========== Register definition for HDMA_CH_7 peripheral ========== 
#define AT91C_HDMA_CH_7_CADDR ((AT91_REG *) 	0xFFFFED78) // (HDMA_CH_7) HDMA Reserved
#define AT91C_HDMA_CH_7_CFG ((AT91_REG *) 	0xFFFFED68) // (HDMA_CH_7) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_7_DADDR ((AT91_REG *) 	0xFFFFED58) // (HDMA_CH_7) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_7_CTRLB ((AT91_REG *) 	0xFFFFED64) // (HDMA_CH_7) HDMA Channel Control B Register
#define AT91C_HDMA_CH_7_DSCR ((AT91_REG *) 	0xFFFFED5C) // (HDMA_CH_7) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_7_DPIP ((AT91_REG *) 	0xFFFFED70) // (HDMA_CH_7) HDMA Channel Destination Picture in Picture Configuration Register
#define AT91C_HDMA_CH_7_CTRLA ((AT91_REG *) 	0xFFFFED60) // (HDMA_CH_7) HDMA Channel Control A Register
#define AT91C_HDMA_CH_7_BDSCR ((AT91_REG *) 	0xFFFFED74) // (HDMA_CH_7) HDMA Reserved
#define AT91C_HDMA_CH_7_SPIP ((AT91_REG *) 	0xFFFFED6C) // (HDMA_CH_7) HDMA Channel Source Picture in Picture Configuration Register
#define AT91C_HDMA_CH_7_SADDR ((AT91_REG *) 	0xFFFFED54) // (HDMA_CH_7) HDMA Channel Source Address Register
// ========== Register definition for HDMA peripheral ========== 
#define AT91C_HDMA_EBCIDR ((AT91_REG *) 	0xFFFFEC1C) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register
#define AT91C_HDMA_LAST ((AT91_REG *) 	0xFFFFEC10) // (HDMA) HDMA Software Last Transfer Flag Register
#define AT91C_HDMA_SREQ ((AT91_REG *) 	0xFFFFEC08) // (HDMA) HDMA Software Single Request Register
#define AT91C_HDMA_RSVD0 ((AT91_REG *) 	0xFFFFEC34) // (HDMA) HDMA Reserved
#define AT91C_HDMA_EBCIER ((AT91_REG *) 	0xFFFFEC18) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
#define AT91C_HDMA_GCFG ((AT91_REG *) 	0xFFFFEC00) // (HDMA) HDMA Global Configuration Register
#define AT91C_HDMA_CHER ((AT91_REG *) 	0xFFFFEC28) // (HDMA) HDMA Channel Handler Enable Register
#define AT91C_HDMA_RSVD1 ((AT91_REG *) 	0xFFFFEC38) // (HDMA) HDMA Reserved
#define AT91C_HDMA_CHDR ((AT91_REG *) 	0xFFFFEC2C) // (HDMA) HDMA Channel Handler Disable Register
#define AT91C_HDMA_EBCIMR ((AT91_REG *) 	0xFFFFEC20) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
#define AT91C_HDMA_CREQ ((AT91_REG *) 	0xFFFFEC0C) // (HDMA) HDMA Software Chunk Transfer Request Register
#define AT91C_HDMA_SYNC ((AT91_REG *) 	0xFFFFEC14) // (HDMA) HDMA Request Synchronization Register
#define AT91C_HDMA_EN   ((AT91_REG *) 	0xFFFFEC04) // (HDMA) HDMA Controller Enable Register
#define AT91C_HDMA_EBCISR ((AT91_REG *) 	0xFFFFEC24) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
#define AT91C_HDMA_CHSR ((AT91_REG *) 	0xFFFFEC30) // (HDMA) HDMA Channel Handler Status Register
// ========== Register definition for HECC peripheral ========== 
#define AT91C_HECC_VR   ((AT91_REG *) 	0xFFFFE2FC) // (HECC)  ECC Version register
#define AT91C_HECC_SR   ((AT91_REG *) 	0xFFFFE208) // (HECC)  ECC Status register
#define AT91C_HECC_CR   ((AT91_REG *) 	0xFFFFE200) // (HECC)  ECC reset register
#define AT91C_HECC_NPR  ((AT91_REG *) 	0xFFFFE210) // (HECC)  ECC Parity N register
#define AT91C_HECC_PR   ((AT91_REG *) 	0xFFFFE20C) // (HECC)  ECC Parity register
#define AT91C_HECC_MR   ((AT91_REG *) 	0xFFFFE204) // (HECC)  ECC Page size register
// ========== Register definition for EMACB peripheral ========== 
#define AT91C_EMACB_ALE ((AT91_REG *) 	0xFFFBC054) // (EMACB) Alignment Error Register
#define AT91C_EMACB_RRE ((AT91_REG *) 	0xFFFBC06C) // (EMACB) Receive Ressource Error Register
#define AT91C_EMACB_SA4H ((AT91_REG *) 	0xFFFBC0B4) // (EMACB) Specific Address 4 Top, Last 2 bytes
#define AT91C_EMACB_TPQ ((AT91_REG *) 	0xFFFBC0BC) // (EMACB) Transmit Pause Quantum Register
#define AT91C_EMACB_RJA ((AT91_REG *) 	0xFFFBC07C) // (EMACB) Receive Jabbers Register
#define AT91C_EMACB_SA2H ((AT91_REG *) 	0xFFFBC0A4) // (EMACB) Specific Address 2 Top, Last 2 bytes
#define AT91C_EMACB_TPF ((AT91_REG *) 	0xFFFBC08C) // (EMACB) Transmitted Pause Frames Register
#define AT91C_EMACB_ROV ((AT91_REG *) 	0xFFFBC070) // (EMACB) Receive Overrun Errors Register
#define AT91C_EMACB_SA4L ((AT91_REG *) 	0xFFFBC0B0) // (EMACB) Specific Address 4 Bottom, First 4 bytes
#define AT91C_EMACB_MAN ((AT91_REG *) 	0xFFFBC034) // (EMACB) PHY Maintenance Register
#define AT91C_EMACB_TID ((AT91_REG *) 	0xFFFBC0B8) // (EMACB) Type ID Checking Register
#define AT91C_EMACB_TBQP ((AT91_REG *) 	0xFFFBC01C) // (EMACB) Transmit Buffer Queue Pointer
#define AT91C_EMACB_SA3L ((AT91_REG *) 	0xFFFBC0A8) // (EMACB) Specific Address 3 Bottom, First 4 bytes
#define AT91C_EMACB_DTF ((AT91_REG *) 	0xFFFBC058) // (EMACB) Deferred Transmission Frame Register
#define AT91C_EMACB_PTR ((AT91_REG *) 	0xFFFBC038) // (EMACB) Pause Time Register
#define AT91C_EMACB_CSE ((AT91_REG *) 	0xFFFBC068) // (EMACB) Carrier Sense Error Register
#define AT91C_EMACB_ECOL ((AT91_REG *) 	0xFFFBC060) // (EMACB) Excessive Collision Register
#define AT91C_EMACB_STE ((AT91_REG *) 	0xFFFBC084) // (EMACB) SQE Test Error Register
#define AT91C_EMACB_MCF ((AT91_REG *) 	0xFFFBC048) // (EMACB) Multiple Collision Frame Register
#define AT91C_EMACB_IER ((AT91_REG *) 	0xFFFBC028) // (EMACB) Interrupt Enable Register
#define AT91C_EMACB_ELE ((AT91_REG *) 	0xFFFBC078) // (EMACB) Excessive Length Errors Register
#define AT91C_EMACB_USRIO ((AT91_REG *) 	0xFFFBC0C0) // (EMACB) USER Input/Output Register
#define AT91C_EMACB_PFR ((AT91_REG *) 	0xFFFBC03C) // (EMACB) Pause Frames received Register
#define AT91C_EMACB_FCSE ((AT91_REG *) 	0xFFFBC050) // (EMACB) Frame Check Sequence Error Register
#define AT91C_EMACB_SA1L ((AT91_REG *) 	0xFFFBC098) // (EMACB) Specific Address 1 Bottom, First 4 bytes
#define AT91C_EMACB_NCR ((AT91_REG *) 	0xFFFBC000) // (EMACB) Network Control Register
#define AT91C_EMACB_HRT ((AT91_REG *) 	0xFFFBC094) // (EMACB) Hash Address Top[63:32]
#define AT91C_EMACB_NCFGR ((AT91_REG *) 	0xFFFBC004) // (EMACB) Network Configuration Register
#define AT91C_EMACB_SCF ((AT91_REG *) 	0xFFFBC044) // (EMACB) Single Collision Frame Register
#define AT91C_EMACB_LCOL ((AT91_REG *) 	0xFFFBC05C) // (EMACB) Late Collision Register
#define AT91C_EMACB_SA3H ((AT91_REG *) 	0xFFFBC0AC) // (EMACB) Specific Address 3 Top, Last 2 bytes
#define AT91C_EMACB_HRB ((AT91_REG *) 	0xFFFBC090) // (EMACB) Hash Address Bottom[31:0]
#define AT91C_EMACB_ISR ((AT91_REG *) 	0xFFFBC024) // (EMACB) Interrupt Status Register
#define AT91C_EMACB_IMR ((AT91_REG *) 	0xFFFBC030) // (EMACB) Interrupt Mask Register
#define AT91C_EMACB_WOL ((AT91_REG *) 	0xFFFBC0C4) // (EMACB) Wake On LAN Register
#define AT91C_EMACB_USF ((AT91_REG *) 	0xFFFBC080) // (EMACB) Undersize Frames Register
#define AT91C_EMACB_TSR ((AT91_REG *) 	0xFFFBC014) // (EMACB) Transmit Status Register
#define AT91C_EMACB_FRO ((AT91_REG *) 	0xFFFBC04C) // (EMACB) Frames Received OK Register
#define AT91C_EMACB_IDR ((AT91_REG *) 	0xFFFBC02C) // (EMACB) Interrupt Disable Register
#define AT91C_EMACB_SA1H ((AT91_REG *) 	0xFFFBC09C) // (EMACB) Specific Address 1 Top, Last 2 bytes
#define AT91C_EMACB_RLE ((AT91_REG *) 	0xFFFBC088) // (EMACB) Receive Length Field Mismatch Register
#define AT91C_EMACB_TUND ((AT91_REG *) 	0xFFFBC064) // (EMACB) Transmit Underrun Error Register
#define AT91C_EMACB_RSR ((AT91_REG *) 	0xFFFBC020) // (EMACB) Receive Status Register
#define AT91C_EMACB_SA2L ((AT91_REG *) 	0xFFFBC0A0) // (EMACB) Specific Address 2 Bottom, First 4 bytes
#define AT91C_EMACB_FTO ((AT91_REG *) 	0xFFFBC040) // (EMACB) Frames Transmitted OK Register
#define AT91C_EMACB_RSE ((AT91_REG *) 	0xFFFBC074) // (EMACB) Receive Symbol Errors Register
#define AT91C_EMACB_NSR ((AT91_REG *) 	0xFFFBC008) // (EMACB) Network Status Register
#define AT91C_EMACB_RBQP ((AT91_REG *) 	0xFFFBC018) // (EMACB) Receive Buffer Queue Pointer
#define AT91C_EMACB_REV ((AT91_REG *) 	0xFFFBC0FC) // (EMACB) Revision Register
// ========== Register definition for ISI peripheral ========== 
#define AT91C_ISI_DMACHDR ((AT91_REG *) 	0xFFFB403C) // (ISI) DMA Channel Disable Register
#define AT91C_ISI_IMR   ((AT91_REG *) 	0xFFFB4034) // (ISI) Interrupt Mask Register
#define AT91C_ISI_R2YSET1 ((AT91_REG *) 	0xFFFB401C) // (ISI) Color Space Conversion RGB to YCrCb Register
#define AT91C_ISI_PDECF ((AT91_REG *) 	0xFFFB400C) // (ISI) Preview Decimation Factor Register
#define AT91C_ISI_CFG2  ((AT91_REG *) 	0xFFFB4004) // (ISI) Configuration Register 2
#define AT91C_ISI_DMACCTRL ((AT91_REG *) 	0xFFFB4054) // (ISI) DMA Codec Control Register
#define AT91C_ISI_CTRL  ((AT91_REG *) 	0xFFFB4024) // (ISI) Control Register
#define AT91C_ISI_Y2RSET0 ((AT91_REG *) 	0xFFFB4010) // (ISI) Color Space Conversion YCrCb to RGB Register
#define AT91C_ISI_WPSR  ((AT91_REG *) 	0xFFFB40E8) // (ISI) Write Protection Status Register
#define AT91C_ISI_DMACHER ((AT91_REG *) 	0xFFFB4038) // (ISI) DMA Channel Enable Register
#define AT91C_ISI_DMACHSR ((AT91_REG *) 	0xFFFB4040) // (ISI) DMA Channel Status Register
#define AT91C_ISI_IDR   ((AT91_REG *) 	0xFFFB4030) // (ISI) Interrupt Disable Register
#define AT91C_ISI_VER   ((AT91_REG *) 	0xFFFB40FC) // (ISI) Version Register
#define AT91C_ISI_Y2RSET1 ((AT91_REG *) 	0xFFFB4014) // (ISI) Color Space Conversion YCrCb to RGB Register
#define AT91C_ISI_R2YSET2 ((AT91_REG *) 	0xFFFB4020) // (ISI) Color Space Conversion RGB to YCrCb Register
#define AT91C_ISI_SR    ((AT91_REG *) 	0xFFFB4028) // (ISI) Status Register
#define AT91C_ISI_DMACDSCR ((AT91_REG *) 	0xFFFB4058) // (ISI) DMA Codec Descriptor Address Register
#define AT91C_ISI_IER   ((AT91_REG *) 	0xFFFB402C) // (ISI) Interrupt Enable Register
#define AT91C_ISI_WPCR  ((AT91_REG *) 	0xFFFB40E4) // (ISI) Write Protection Control Register
#define AT91C_ISI_DMACADDR ((AT91_REG *) 	0xFFFB4050) // (ISI) DMA Codec Base Address Register
#define AT91C_ISI_CFG1  ((AT91_REG *) 	0xFFFB4000) // (ISI) Configuration Register 1
#define AT91C_ISI_R2YSET0 ((AT91_REG *) 	0xFFFB4018) // (ISI) Color Space Conversion RGB to YCrCb Register
#define AT91C_ISI_PSIZE ((AT91_REG *) 	0xFFFB4008) // (ISI) Preview Size Register
#define AT91C_ISI_DMAPDSCR ((AT91_REG *) 	0xFFFB404C) // (ISI) DMA Preview Descriptor Address Register
#define AT91C_ISI_DMAPADDR ((AT91_REG *) 	0xFFFB4044) // (ISI) DMA Preview Base Address Register
#define AT91C_ISI_DMAPCTRL ((AT91_REG *) 	0xFFFB4048) // (ISI) DMA Preview Control Register
// ========== Register definition for UHPHS_OHCI peripheral ========== 
#define AT91C_OHCI_HcRhPortStatus ((AT91_REG *) 	0x00700054) // (UHPHS_OHCI) Root Hub Port Status Register
#define AT91C_OHCI_HcFmRemaining ((AT91_REG *) 	0x00700038) // (UHPHS_OHCI) Bit time remaining in the current Frame
#define AT91C_OHCI_HcInterruptEnable ((AT91_REG *) 	0x00700010) // (UHPHS_OHCI) Interrupt Enable Register
#define AT91C_OHCI_HcControl ((AT91_REG *) 	0x00700004) // (UHPHS_OHCI) Operating modes for the Host Controller
#define AT91C_OHCI_HcPeriodicStart ((AT91_REG *) 	0x00700040) // (UHPHS_OHCI) Periodic Start
#define AT91C_OHCI_HcInterruptStatus ((AT91_REG *) 	0x0070000C) // (UHPHS_OHCI) Interrupt Status Register
#define AT91C_OHCI_HcRhDescriptorB ((AT91_REG *) 	0x0070004C) // (UHPHS_OHCI) Root Hub characteristics B
#define AT91C_OHCI_HcInterruptDisable ((AT91_REG *) 	0x00700014) // (UHPHS_OHCI) Interrupt Disable Register
#define AT91C_OHCI_HcPeriodCurrentED ((AT91_REG *) 	0x0070001C) // (UHPHS_OHCI) Current Isochronous or Interrupt Endpoint Descriptor
#define AT91C_OHCI_HcRhDescriptorA ((AT91_REG *) 	0x00700048) // (UHPHS_OHCI) Root Hub characteristics A
#define AT91C_OHCI_HcRhStatus ((AT91_REG *) 	0x00700050) // (UHPHS_OHCI) Root Hub Status register
#define AT91C_OHCI_HcBulkCurrentED ((AT91_REG *) 	0x0070002C) // (UHPHS_OHCI) Current endpoint of the Bulk list
#define AT91C_OHCI_HcControlHeadED ((AT91_REG *) 	0x00700020) // (UHPHS_OHCI) First Endpoint Descriptor of the Control list
#define AT91C_OHCI_HcLSThreshold ((AT91_REG *) 	0x00700044) // (UHPHS_OHCI) LS Threshold
#define AT91C_OHCI_HcRevision ((AT91_REG *) 	0x00700000) // (UHPHS_OHCI) Revision
#define AT91C_OHCI_HcBulkDoneHead ((AT91_REG *) 	0x00700030) // (UHPHS_OHCI) Last completed transfer descriptor
#define AT91C_OHCI_HcFmNumber ((AT91_REG *) 	0x0070003C) // (UHPHS_OHCI) Frame number
#define AT91C_OHCI_HcFmInterval ((AT91_REG *) 	0x00700034) // (UHPHS_OHCI) Bit time between 2 consecutive SOFs
#define AT91C_OHCI_HcBulkHeadED ((AT91_REG *) 	0x00700028) // (UHPHS_OHCI) First endpoint register of the Bulk list
#define AT91C_OHCI_HcHCCA ((AT91_REG *) 	0x00700018) // (UHPHS_OHCI) Pointer to the Host Controller Communication Area
#define AT91C_OHCI_HcCommandStatus ((AT91_REG *) 	0x00700008) // (UHPHS_OHCI) Command & status Register
#define AT91C_OHCI_HcControlCurrentED ((AT91_REG *) 	0x00700024) // (UHPHS_OHCI) Endpoint Control and Status Register
// ========== Register definition for UHPHS_EHCI peripheral ========== 
#define AT91C_EHCI_USBSTS ((AT91_REG *) 	0x00800014) // (UHPHS_EHCI) 
#define AT91C_EHCI_INSNREG03 ((AT91_REG *) 	0x0080009C) // (UHPHS_EHCI) 
#define AT91C_EHCI_INSNREG00 ((AT91_REG *) 	0x00800090) // (UHPHS_EHCI) 
#define AT91C_EHCI_HCSPPORTROUTE ((AT91_REG *) 	0x0080000C) // (UHPHS_EHCI) 
#define AT91C_EHCI_FRINDEX ((AT91_REG *) 	0x0080001C) // (UHPHS_EHCI) 
#define AT91C_EHCI_CONFIGFLAG ((AT91_REG *) 	0x00800050) // (UHPHS_EHCI) 
#define AT91C_EHCI_HCCPARAMS ((AT91_REG *) 	0x00800008) // (UHPHS_EHCI) 
#define AT91C_EHCI_USBINTR ((AT91_REG *) 	0x00800018) // (UHPHS_EHCI) 
#define AT91C_EHCI_PORTSC ((AT91_REG *) 	0x00800054) // (UHPHS_EHCI) 
#define AT91C_EHCI_CTRLDSSEGMENT ((AT91_REG *) 	0x00800020) // (UHPHS_EHCI) 
#define AT91C_EHCI_VERSION ((AT91_REG *) 	0x00800000) // (UHPHS_EHCI) 
#define AT91C_EHCI_USBCMD ((AT91_REG *) 	0x00800010) // (UHPHS_EHCI) 
#define AT91C_EHCI_INSNREG04 ((AT91_REG *) 	0x008000A0) // (UHPHS_EHCI) 
#define AT91C_EHCI_PERIODICLISTBASE ((AT91_REG *) 	0x00800024) // (UHPHS_EHCI) 
#define AT91C_EHCI_INSNREG01 ((AT91_REG *) 	0x00800094) // (UHPHS_EHCI) 
#define AT91C_EHCI_HCSPARAMS ((AT91_REG *) 	0x00800004) // (UHPHS_EHCI) 
#define AT91C_EHCI_INSNREG05 ((AT91_REG *) 	0x008000A4) // (UHPHS_EHCI) 
#define AT91C_EHCI_ASYNCLISTADDR ((AT91_REG *) 	0x00800028) // (UHPHS_EHCI) 
#define AT91C_EHCI_INSNREG02 ((AT91_REG *) 	0x00800098) // (UHPHS_EHCI) 
// ========== Register definition for TRNG peripheral ========== 
#define AT91C_TRNG_IDR  ((AT91_REG *) 	0xFFFCC014) // (TRNG) Interrupt Disable Register
#define AT91C_TRNG_IER  ((AT91_REG *) 	0xFFFCC010) // (TRNG) Interrupt Enable Register
#define AT91C_TRNG_VERSION ((AT91_REG *) 	0xFFFCC0FC) // (TRNG) TRNG Version Register
#define AT91C_TRNG_ISR  ((AT91_REG *) 	0xFFFCC01C) // (TRNG) Interrupt Status Register
#define AT91C_TRNG_CR   ((AT91_REG *) 	0xFFFCC000) // (TRNG) Control Register
#define AT91C_TRNG_ODATA ((AT91_REG *) 	0xFFFCC050) // (TRNG) Output Data Register
#define AT91C_TRNG_IMR  ((AT91_REG *) 	0xFFFCC018) // (TRNG) Interrupt Mask Register

// *****************************************************************************
//               PIO DEFINITIONS FOR AT91SAM9G45
// *****************************************************************************
#define AT91C_PIO_PA0        ((unsigned int) 1 <<  0) // Pin Controlled by PA0
#define AT91C_PA0_MCI0_CK  ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PA0_TCLK3    ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PIO_PA1        ((unsigned int) 1 <<  1) // Pin Controlled by PA1
#define AT91C_PA1_MCI0_CDA ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PA1_TIOA3    ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PIO_PA10       ((unsigned int) 1 << 10) // Pin Controlled by PA10
#define AT91C_PA10_ETX0     ((unsigned int) AT91C_PIO_PA10) //  Ethernet MAC Transmit Data 0
#define AT91C_PIO_PA11       ((unsigned int) 1 << 11) // Pin Controlled by PA11
#define AT91C_PA11_ETX1     ((unsigned int) AT91C_PIO_PA11) //  Ethernet MAC Transmit Data 1
#define AT91C_PIO_PA12       ((unsigned int) 1 << 12) // Pin Controlled by PA12
#define AT91C_PA12_ERX0     ((unsigned int) AT91C_PIO_PA12) //  Ethernet MAC Receive Data 0
#define AT91C_PIO_PA13       ((unsigned int) 1 << 13) // Pin Controlled by PA13
#define AT91C_PA13_ERX1     ((unsigned int) AT91C_PIO_PA13) //  Ethernet MAC Receive Data 1
#define AT91C_PIO_PA14       ((unsigned int) 1 << 14) // Pin Controlled by PA14
#define AT91C_PA14_ETXEN    ((unsigned int) AT91C_PIO_PA14) //  Ethernet MAC Transmit Enable
#define AT91C_PIO_PA15       ((unsigned int) 1 << 15) // Pin Controlled by PA15
#define AT91C_PA15_ERXDV    ((unsigned int) AT91C_PIO_PA15) //  Ethernet MAC Receive Data Valid
#define AT91C_PIO_PA16       ((unsigned int) 1 << 16) // Pin Controlled by PA16
#define AT91C_PA16_ERXER    ((unsigned int) AT91C_PIO_PA16) //  Ethernet MAC Receive Error
#define AT91C_PIO_PA17       ((unsigned int) 1 << 17) // Pin Controlled by PA17
#define AT91C_PA17_ETXCK_EREFCK ((unsigned int) AT91C_PIO_PA17) //  Ethernet MAC Transmit Clock/Reference Clock
#define AT91C_PIO_PA18       ((unsigned int) 1 << 18) // Pin Controlled by PA18
#define AT91C_PA18_EMDC     ((unsigned int) AT91C_PIO_PA18) //  Ethernet MAC Management Data Clock
#define AT91C_PIO_PA19       ((unsigned int) 1 << 19) // Pin Controlled by PA19
#define AT91C_PA19_EMDIO    ((unsigned int) AT91C_PIO_PA19) //  Ethernet MAC Management Data Input/Output
#define AT91C_PIO_PA2        ((unsigned int) 1 <<  2) // Pin Controlled by PA2
#define AT91C_PA2_MCI0_DA0 ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PA2_TIOB3    ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PIO_PA20       ((unsigned int) 1 << 20) // Pin Controlled by PA20
#define AT91C_PA20_TWD0     ((unsigned int) AT91C_PIO_PA20) //  TWI Two-wire Serial Data
#define AT91C_PIO_PA21       ((unsigned int) 1 << 21) // Pin Controlled by PA21
#define AT91C_PA21_TWCK0    ((unsigned int) AT91C_PIO_PA21) //  TWI Two-wire Serial Clock
#define AT91C_PIO_PA22       ((unsigned int) 1 << 22) // Pin Controlled by PA22
#define AT91C_PA22_MCI1_CDA ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PA22_SCK3     ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PIO_PA23       ((unsigned int) 1 << 23) // Pin Controlled by PA23
#define AT91C_PA23_MCI1_DA0 ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PA23_RTS3     ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PIO_PA24       ((unsigned int) 1 << 24) // Pin Controlled by PA24
#define AT91C_PA24_MCI1_DA1 ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PA24_CTS3     ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PIO_PA25       ((unsigned int) 1 << 25) // Pin Controlled by PA25
#define AT91C_PA25_MCI1_DA2 ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PA25_PWM3     ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PIO_PA26       ((unsigned int) 1 << 26) // Pin Controlled by PA26
#define AT91C_PA26_MCI1_DA3 ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PA26_TIOB2    ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PIO_PA27       ((unsigned int) 1 << 27) // Pin Controlled by PA27
#define AT91C_PA27_MCI1_DA4 ((unsigned int) AT91C_PIO_PA27) //  
#define AT91C_PA27_ETXER    ((unsigned int) AT91C_PIO_PA27) //  Ethernet MAC Transmikt Coding Error
#define AT91C_PIO_PA28       ((unsigned int) 1 << 28) // Pin Controlled by PA28
#define AT91C_PA28_MCI1_DA5 ((unsigned int) AT91C_PIO_PA28) //  
#define AT91C_PA28_ERXCK    ((unsigned int) AT91C_PIO_PA28) //  Ethernet MAC Receive Clock
#define AT91C_PIO_PA29       ((unsigned int) 1 << 29) // Pin Controlled by PA29
#define AT91C_PA29_MCI1_DA6 ((unsigned int) AT91C_PIO_PA29) //  
#define AT91C_PA29_ECRS     ((unsigned int) AT91C_PIO_PA29) //  Ethernet MAC Carrier Sense/Carrier Sense and Data Valid
#define AT91C_PIO_PA3        ((unsigned int) 1 <<  3) // Pin Controlled by PA3
#define AT91C_PA3_MCI0_DA1 ((unsigned int) AT91C_PIO_PA3) //  
#define AT91C_PA3_TCLK4    ((unsigned int) AT91C_PIO_PA3) //  
#define AT91C_PIO_PA30       ((unsigned int) 1 << 30) // Pin Controlled by PA30
#define AT91C_PA30_MCI1_DA7 ((unsigned int) AT91C_PIO_PA30) //  
#define AT91C_PA30_ECOL     ((unsigned int) AT91C_PIO_PA30) //  Ethernet MAC Collision Detected
#define AT91C_PIO_PA31       ((unsigned int) 1 << 31) // Pin Controlled by PA31
#define AT91C_PA31_MCI1_CK  ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PA31_PCK0     ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PIO_PA4        ((unsigned int) 1 <<  4) // Pin Controlled by PA4
#define AT91C_PA4_MCI0_DA2 ((unsigned int) AT91C_PIO_PA4) //  
#define AT91C_PA4_TIOA4    ((unsigned int) AT91C_PIO_PA4) //  
#define AT91C_PIO_PA5        ((unsigned int) 1 <<  5) // Pin Controlled by PA5
#define AT91C_PA5_MCI0_DA3 ((unsigned int) AT91C_PIO_PA5) //  
#define AT91C_PA5_TIOB4    ((unsigned int) AT91C_PIO_PA5) //  
#define AT91C_PIO_PA6        ((unsigned int) 1 <<  6) // Pin Controlled by PA6
#define AT91C_PA6_MCI0_DA4 ((unsigned int) AT91C_PIO_PA6) //  
#define AT91C_PA6_ETX2     ((unsigned int) AT91C_PIO_PA6) //  Ethernet MAC Transmit Data 2
#define AT91C_PIO_PA7        ((unsigned int) 1 <<  7) // Pin Controlled by PA7
#define AT91C_PA7_MCI0_DA5 ((unsigned int) AT91C_PIO_PA7) //  
#define AT91C_PA7_ETX3     ((unsigned int) AT91C_PIO_PA7) //  Ethernet MAC Transmit Data 3
#define AT91C_PIO_PA8        ((unsigned int) 1 <<  8) // Pin Controlled by PA8
#define AT91C_PA8_MCI0_DA6 ((unsigned int) AT91C_PIO_PA8) //  
#define AT91C_PA8_ERX2     ((unsigned int) AT91C_PIO_PA8) //  Ethernet MAC Receive Data 2
#define AT91C_PIO_PA9        ((unsigned int) 1 <<  9) // Pin Controlled by PA9
#define AT91C_PA9_MCI0_DA7 ((unsigned int) AT91C_PIO_PA9) //  
#define AT91C_PA9_ERX3     ((unsigned int) AT91C_PIO_PA9) //  Ethernet MAC Receive Data 3
#define AT91C_PIO_PB0        ((unsigned int) 1 <<  0) // Pin Controlled by PB0
#define AT91C_PB0_SPI0_MISO ((unsigned int) AT91C_PIO_PB0) //  SPI 0 Master In Slave
#define AT91C_PIO_PB1        ((unsigned int) 1 <<  1) // Pin Controlled by PB1
#define AT91C_PB1_SPI0_MOSI ((unsigned int) AT91C_PIO_PB1) //  SPI 0 Master Out Slave
#define AT91C_PIO_PB10       ((unsigned int) 1 << 10) // Pin Controlled by PB10
#define AT91C_PB10_TWD1     ((unsigned int) AT91C_PIO_PB10) //  
#define AT91C_PB10_ISI_D10  ((unsigned int) AT91C_PIO_PB10) //  
#define AT91C_PIO_PB11       ((unsigned int) 1 << 11) // Pin Controlled by PB11
#define AT91C_PB11_TWCK1    ((unsigned int) AT91C_PIO_PB11) //  
#define AT91C_PB11_ISI_D11  ((unsigned int) AT91C_PIO_PB11) //  
#define AT91C_PIO_PB12       ((unsigned int) 1 << 12) // Pin Controlled by PB12
#define AT91C_PB12_DRXD     ((unsigned int) AT91C_PIO_PB12) //  
#define AT91C_PIO_PB13       ((unsigned int) 1 << 13) // Pin Controlled by PB13
#define AT91C_PB13_DTXD     ((unsigned int) AT91C_PIO_PB13) //  
#define AT91C_PIO_PB14       ((unsigned int) 1 << 14) // Pin Controlled by PB14
#define AT91C_PB14_SPI1_MISO ((unsigned int) AT91C_PIO_PB14) //  
#define AT91C_PIO_PB15       ((unsigned int) 1 << 15) // Pin Controlled by PB15
#define AT91C_PB15_SPI1_MOSI ((unsigned int) AT91C_PIO_PB15) //  
#define AT91C_PB15_CTS0     ((unsigned int) AT91C_PIO_PB15) //  
#define AT91C_PIO_PB16       ((unsigned int) 1 << 16) // Pin Controlled by PB16
#define AT91C_PB16_SPI1_SPCK ((unsigned int) AT91C_PIO_PB16) //  
#define AT91C_PB16_SCK0     ((unsigned int) AT91C_PIO_PB16) //  
#define AT91C_PIO_PB17       ((unsigned int) 1 << 17) // Pin Controlled by PB17
#define AT91C_PB17_SPI1_NPCS0 ((unsigned int) AT91C_PIO_PB17) //  
#define AT91C_PB17_RTS0     ((unsigned int) AT91C_PIO_PB17) //  
#define AT91C_PIO_PB18       ((unsigned int) 1 << 18) // Pin Controlled by PB18
#define AT91C_PB18_RXD0     ((unsigned int) AT91C_PIO_PB18) //  
#define AT91C_PB18_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PB18) //  
#define AT91C_PIO_PB19       ((unsigned int) 1 << 19) // Pin Controlled by PB19
#define AT91C_PB19_TXD0     ((unsigned int) AT91C_PIO_PB19) //  
#define AT91C_PB19_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PB19) //  
#define AT91C_PIO_PB2        ((unsigned int) 1 <<  2) // Pin Controlled by PB2
#define AT91C_PB2_SPI0_SPCK ((unsigned int) AT91C_PIO_PB2) //  SPI 0 Serial Clock
#define AT91C_PIO_PB20       ((unsigned int) 1 << 20) // Pin Controlled by PB20
#define AT91C_PB20_ISI_D0   ((unsigned int) AT91C_PIO_PB20) //  
#define AT91C_PIO_PB21       ((unsigned int) 1 << 21) // Pin Controlled by PB21
#define AT91C_PB21_ISI_D1   ((unsigned int) AT91C_PIO_PB21) //  
#define AT91C_PIO_PB22       ((unsigned int) 1 << 22) // Pin Controlled by PB22
#define AT91C_PB22_ISI_D2   ((unsigned int) AT91C_PIO_PB22) //  
#define AT91C_PIO_PB23       ((unsigned int) 1 << 23) // Pin Controlled by PB23
#define AT91C_PB23_ISI_D3   ((unsigned int) AT91C_PIO_PB23) //  
#define AT91C_PIO_PB24       ((unsigned int) 1 << 24) // Pin Controlled by PB24
#define AT91C_PB24_ISI_D4   ((unsigned int) AT91C_PIO_PB24) //  
#define AT91C_PIO_PB25       ((unsigned int) 1 << 25) // Pin Controlled by PB25
#define AT91C_PB25_ISI_D5   ((unsigned int) AT91C_PIO_PB25) //  
#define AT91C_PIO_PB26       ((unsigned int) 1 << 26) // Pin Controlled by PB26
#define AT91C_PB26_ISI_D6   ((unsigned int) AT91C_PIO_PB26) //  
#define AT91C_PIO_PB27       ((unsigned int) 1 << 27) // Pin Controlled by PB27
#define AT91C_PB27_ISI_D7   ((unsigned int) AT91C_PIO_PB27) //  
#define AT91C_PIO_PB28       ((unsigned int) 1 << 28) // Pin Controlled by PB28
#define AT91C_PB28_ISI_PCK  ((unsigned int) AT91C_PIO_PB28) //  
#define AT91C_PIO_PB29       ((unsigned int) 1 << 29) // Pin Controlled by PB29
#define AT91C_PB29_ISI_VSYNC ((unsigned int) AT91C_PIO_PB29) //  
#define AT91C_PIO_PB3        ((unsigned int) 1 <<  3) // Pin Controlled by PB3
#define AT91C_PB3_SPI0_NPCS0 ((unsigned int) AT91C_PIO_PB3) //  SPI 0 Peripheral Chip Select 0
#define AT91C_PIO_PB30       ((unsigned int) 1 << 30) // Pin Controlled by PB30
#define AT91C_PB30_ISI_HSYNC ((unsigned int) AT91C_PIO_PB30) //  
#define AT91C_PIO_PB31       ((unsigned int) 1 << 31) // Pin Controlled by PB31
#define AT91C_PB31_         ((unsigned int) AT91C_PIO_PB31) //  
#define AT91C_PB31_PCK1     ((unsigned int) AT91C_PIO_PB31) //  
#define AT91C_PIO_PB4        ((unsigned int) 1 <<  4) // Pin Controlled by PB4
#define AT91C_PB4_TXD1     ((unsigned int) AT91C_PIO_PB4) //  USART 1 Transmit Data
#define AT91C_PIO_PB5        ((unsigned int) 1 <<  5) // Pin Controlled by PB5
#define AT91C_PB5_RXD1     ((unsigned int) AT91C_PIO_PB5) //  USART 1 Receive Data
#define AT91C_PIO_PB6        ((unsigned int) 1 <<  6) // Pin Controlled by PB6
#define AT91C_PB6_TXD2     ((unsigned int) AT91C_PIO_PB6) //  USART 2 Transmit Data
#define AT91C_PIO_PB7        ((unsigned int) 1 <<  7) // Pin Controlled by PB7
#define AT91C_PB7_RXD2     ((unsigned int) AT91C_PIO_PB7) //  USART 2 Receive Data
#define AT91C_PIO_PB8        ((unsigned int) 1 <<  8) // Pin Controlled by PB8
#define AT91C_PB8_TXD3     ((unsigned int) AT91C_PIO_PB8) //  USART 3 Transmit Data
#define AT91C_PB8_ISI_D8   ((unsigned int) AT91C_PIO_PB8) //  
#define AT91C_PIO_PB9        ((unsigned int) 1 <<  9) // Pin Controlled by PB9
#define AT91C_PB9_RXD3     ((unsigned int) AT91C_PIO_PB9) //  USART 3 Receive Data
#define AT91C_PB9_ISI_D9   ((unsigned int) AT91C_PIO_PB9) //  
#define AT91C_PIO_PC0        ((unsigned int) 1 <<  0) // Pin Controlled by PC0
#define AT91C_PC0_DQM2     ((unsigned int) AT91C_PIO_PC0) //  DQM2
#define AT91C_PIO_PC1        ((unsigned int) 1 <<  1) // Pin Controlled by PC1
#define AT91C_PC1_DQM3     ((unsigned int) AT91C_PIO_PC1) //  DQM3
#define AT91C_PIO_PC10       ((unsigned int) 1 << 10) // Pin Controlled by PC10
#define AT91C_PC10_NCS4_CFCS0 ((unsigned int) AT91C_PIO_PC10) //  
#define AT91C_PC10_TCLK2    ((unsigned int) AT91C_PIO_PC10) //  
#define AT91C_PIO_PC11       ((unsigned int) 1 << 11) // Pin Controlled by PC11
#define AT91C_PC11_NCS5_CFCS1 ((unsigned int) AT91C_PIO_PC11) //  
#define AT91C_PC11_CTS2     ((unsigned int) AT91C_PIO_PC11) //  
#define AT91C_PIO_PC12       ((unsigned int) 1 << 12) // Pin Controlled by PC12
#define AT91C_PC12_A25_CFRNW ((unsigned int) AT91C_PIO_PC12) //  
#define AT91C_PIO_PC13       ((unsigned int) 1 << 13) // Pin Controlled by PC13
#define AT91C_PC13_NCS2     ((unsigned int) AT91C_PIO_PC13) //  
#define AT91C_PIO_PC14       ((unsigned int) 1 << 14) // Pin Controlled by PC14
#define AT91C_PC14_NCS3_NANDCS ((unsigned int) AT91C_PIO_PC14) //  
#define AT91C_PIO_PC15       ((unsigned int) 1 << 15) // Pin Controlled by PC15
#define AT91C_PC15_NWAIT    ((unsigned int) AT91C_PIO_PC15) //  
#define AT91C_PIO_PC16       ((unsigned int) 1 << 16) // Pin Controlled by PC16
#define AT91C_PC16_D16      ((unsigned int) AT91C_PIO_PC16) //  
#define AT91C_PIO_PC17       ((unsigned int) 1 << 17) // Pin Controlled by PC17
#define AT91C_PC17_D17      ((unsigned int) AT91C_PIO_PC17) //  
#define AT91C_PIO_PC18       ((unsigned int) 1 << 18) // Pin Controlled by PC18
#define AT91C_PC18_D18      ((unsigned int) AT91C_PIO_PC18) //  
#define AT91C_PIO_PC19       ((unsigned int) 1 << 19) // Pin Controlled by PC19
#define AT91C_PC19_D19      ((unsigned int) AT91C_PIO_PC19) //  
#define AT91C_PIO_PC2        ((unsigned int) 1 <<  2) // Pin Controlled by PC2
#define AT91C_PC2_A19      ((unsigned int) AT91C_PIO_PC2) //  
#define AT91C_PIO_PC20       ((unsigned int) 1 << 20) // Pin Controlled by PC20
#define AT91C_PC20_D20      ((unsigned int) AT91C_PIO_PC20) //  
#define AT91C_PIO_PC21       ((unsigned int) 1 << 21) // Pin Controlled by PC21
#define AT91C_PC21_D21      ((unsigned int) AT91C_PIO_PC21) //  
#define AT91C_PIO_PC22       ((unsigned int) 1 << 22) // Pin Controlled by PC22
#define AT91C_PC22_D22      ((unsigned int) AT91C_PIO_PC22) //  
#define AT91C_PIO_PC23       ((unsigned int) 1 << 23) // Pin Controlled by PC23
#define AT91C_PC23_D23      ((unsigned int) AT91C_PIO_PC23) //  
#define AT91C_PIO_PC24       ((unsigned int) 1 << 24) // Pin Controlled by PC24
#define AT91C_PC24_D24      ((unsigned int) AT91C_PIO_PC24) //  
#define AT91C_PIO_PC25       ((unsigned int) 1 << 25) // Pin Controlled by PC25
#define AT91C_PC25_D25      ((unsigned int) AT91C_PIO_PC25) //  
#define AT91C_PIO_PC26       ((unsigned int) 1 << 26) // Pin Controlled by PC26
#define AT91C_PC26_D26      ((unsigned int) AT91C_PIO_PC26) //  
#define AT91C_PIO_PC27       ((unsigned int) 1 << 27) // Pin Controlled by PC27
#define AT91C_PC27_D27      ((unsigned int) AT91C_PIO_PC27) //  
#define AT91C_PIO_PC28       ((unsigned int) 1 << 28) // Pin Controlled by PC28
#define AT91C_PC28_D28      ((unsigned int) AT91C_PIO_PC28) //  
#define AT91C_PIO_PC29       ((unsigned int) 1 << 29) // Pin Controlled by PC29
#define AT91C_PC29_D29      ((unsigned int) AT91C_PIO_PC29) //  
#define AT91C_PIO_PC3        ((unsigned int) 1 <<  3) // Pin Controlled by PC3
#define AT91C_PC3_A20      ((unsigned int) AT91C_PIO_PC3) //  
#define AT91C_PIO_PC30       ((unsigned int) 1 << 30) // Pin Controlled by PC30
#define AT91C_PC30_D30      ((unsigned int) AT91C_PIO_PC30) //  
#define AT91C_PIO_PC31       ((unsigned int) 1 << 31) // Pin Controlled by PC31
#define AT91C_PC31_D31      ((unsigned int) AT91C_PIO_PC31) //  
#define AT91C_PIO_PC4        ((unsigned int) 1 <<  4) // Pin Controlled by PC4
#define AT91C_PC4_A21_NANDALE ((unsigned int) AT91C_PIO_PC4) //  
#define AT91C_PIO_PC5        ((unsigned int) 1 <<  5) // Pin Controlled by PC5
#define AT91C_PC5_A22_NANDCLE ((unsigned int) AT91C_PIO_PC5) //  
#define AT91C_PIO_PC6        ((unsigned int) 1 <<  6) // Pin Controlled by PC6
#define AT91C_PC6_A23      ((unsigned int) AT91C_PIO_PC6) //  
#define AT91C_PIO_PC7        ((unsigned int) 1 <<  7) // Pin Controlled by PC7
#define AT91C_PC7_A24      ((unsigned int) AT91C_PIO_PC7) //  
#define AT91C_PIO_PC8        ((unsigned int) 1 <<  8) // Pin Controlled by PC8
#define AT91C_PC8_CFCE1    ((unsigned int) AT91C_PIO_PC8) //  
#define AT91C_PIO_PC9        ((unsigned int) 1 <<  9) // Pin Controlled by PC9
#define AT91C_PC9_CFCE2    ((unsigned int) AT91C_PIO_PC9) //  
#define AT91C_PC9_RTS2     ((unsigned int) AT91C_PIO_PC9) //  
#define AT91C_PIO_PD0        ((unsigned int) 1 <<  0) // Pin Controlled by PD0
#define AT91C_PD0_TK0      ((unsigned int) AT91C_PIO_PD0) //  
#define AT91C_PD0_PWM3     ((unsigned int) AT91C_PIO_PD0) //  
#define AT91C_PIO_PD1        ((unsigned int) 1 <<  1) // Pin Controlled by PD1
#define AT91C_PD1_TF0      ((unsigned int) AT91C_PIO_PD1) //  
#define AT91C_PIO_PD10       ((unsigned int) 1 << 10) // Pin Controlled by PD10
#define AT91C_PD10_TD1      ((unsigned int) AT91C_PIO_PD10) //  
#define AT91C_PIO_PD11       ((unsigned int) 1 << 11) // Pin Controlled by PD11
#define AT91C_PD11_RD1      ((unsigned int) AT91C_PIO_PD11) //  
#define AT91C_PIO_PD12       ((unsigned int) 1 << 12) // Pin Controlled by PD12
#define AT91C_PD12_TK1      ((unsigned int) AT91C_PIO_PD12) //  
#define AT91C_PD12_PCK0     ((unsigned int) AT91C_PIO_PD12) //  
#define AT91C_PIO_PD13       ((unsigned int) 1 << 13) // Pin Controlled by PD13
#define AT91C_PD13_RK1      ((unsigned int) AT91C_PIO_PD13) //  
#define AT91C_PIO_PD14       ((unsigned int) 1 << 14) // Pin Controlled by PD14
#define AT91C_PD14_TF1      ((unsigned int) AT91C_PIO_PD14) //  
#define AT91C_PIO_PD15       ((unsigned int) 1 << 15) // Pin Controlled by PD15
#define AT91C_PD15_RF1      ((unsigned int) AT91C_PIO_PD15) //  
#define AT91C_PIO_PD16       ((unsigned int) 1 << 16) // Pin Controlled by PD16
#define AT91C_PD16_RTS1     ((unsigned int) AT91C_PIO_PD16) //  
#define AT91C_PIO_PD17       ((unsigned int) 1 << 17) // Pin Controlled by PD17
#define AT91C_PD17_CTS1     ((unsigned int) AT91C_PIO_PD17) //  
#define AT91C_PIO_PD18       ((unsigned int) 1 << 18) // Pin Controlled by PD18
#define AT91C_PD18_SPI1_NPCS2 ((unsigned int) AT91C_PIO_PD18) //  
#define AT91C_PD18_IRQ      ((unsigned int) AT91C_PIO_PD18) //  
#define AT91C_PIO_PD19       ((unsigned int) 1 << 19) // Pin Controlled by PD19
#define AT91C_PD19_SPI1_NPCS3 ((unsigned int) AT91C_PIO_PD19) //  
#define AT91C_PD19_FIQ      ((unsigned int) AT91C_PIO_PD19) //  
#define AT91C_PIO_PD2        ((unsigned int) 1 <<  2) // Pin Controlled by PD2
#define AT91C_PD2_TD0      ((unsigned int) AT91C_PIO_PD2) //  
#define AT91C_PIO_PD20       ((unsigned int) 1 << 20) // Pin Controlled by PD20
#define AT91C_PD20_TIOA0    ((unsigned int) AT91C_PIO_PD20) //  
#define AT91C_PIO_PD21       ((unsigned int) 1 << 21) // Pin Controlled by PD21
#define AT91C_PD21_TIOA1    ((unsigned int) AT91C_PIO_PD21) //  
#define AT91C_PIO_PD22       ((unsigned int) 1 << 22) // Pin Controlled by PD22
#define AT91C_PD22_TIOA2    ((unsigned int) AT91C_PIO_PD22) //  
#define AT91C_PIO_PD23       ((unsigned int) 1 << 23) // Pin Controlled by PD23
#define AT91C_PD23_TCLK0    ((unsigned int) AT91C_PIO_PD23) //  
#define AT91C_PIO_PD24       ((unsigned int) 1 << 24) // Pin Controlled by PD24
#define AT91C_PD24_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PD24) //  
#define AT91C_PD24_PWM0     ((unsigned int) AT91C_PIO_PD24) //  
#define AT91C_PIO_PD25       ((unsigned int) 1 << 25) // Pin Controlled by PD25
#define AT91C_PD25_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PD25) //  
#define AT91C_PD25_PWM1     ((unsigned int) AT91C_PIO_PD25) //  
#define AT91C_PIO_PD26       ((unsigned int) 1 << 26) // Pin Controlled by PD26
#define AT91C_PD26_PCK0     ((unsigned int) AT91C_PIO_PD26) //  
#define AT91C_PD26_PWM2     ((unsigned int) AT91C_PIO_PD26) //  
#define AT91C_PIO_PD27       ((unsigned int) 1 << 27) // Pin Controlled by PD27
#define AT91C_PD27_PCK1     ((unsigned int) AT91C_PIO_PD27) //  
#define AT91C_PD27_SPI0_NPCS3 ((unsigned int) AT91C_PIO_PD27) //  
#define AT91C_PIO_PD28       ((unsigned int) 1 << 28) // Pin Controlled by PD28
#define AT91C_PD28_TSADTRG  ((unsigned int) AT91C_PIO_PD28) //  
#define AT91C_PD28_SPI1_NPCS1 ((unsigned int) AT91C_PIO_PD28) //  
#define AT91C_PIO_PD29       ((unsigned int) 1 << 29) // Pin Controlled by PD29
#define AT91C_PD29_TCLK1    ((unsigned int) AT91C_PIO_PD29) //  
#define AT91C_PD29_SCK1     ((unsigned int) AT91C_PIO_PD29) //  
#define AT91C_PIO_PD3        ((unsigned int) 1 <<  3) // Pin Controlled by PD3
#define AT91C_PD3_RD0      ((unsigned int) AT91C_PIO_PD3) //  
#define AT91C_PIO_PD30       ((unsigned int) 1 << 30) // Pin Controlled by PD30
#define AT91C_PD30_TIOB0    ((unsigned int) AT91C_PIO_PD30) //  
#define AT91C_PD30_SCK2     ((unsigned int) AT91C_PIO_PD30) //  
#define AT91C_PIO_PD31       ((unsigned int) 1 << 31) // Pin Controlled by PD31
#define AT91C_PD31_TIOB1    ((unsigned int) AT91C_PIO_PD31) //  
#define AT91C_PD31_PWM1     ((unsigned int) AT91C_PIO_PD31) //  
#define AT91C_PIO_PD4        ((unsigned int) 1 <<  4) // Pin Controlled by PD4
#define AT91C_PD4_RK0      ((unsigned int) AT91C_PIO_PD4) //  
#define AT91C_PIO_PD5        ((unsigned int) 1 <<  5) // Pin Controlled by PD5
#define AT91C_PD5_RF0      ((unsigned int) AT91C_PIO_PD5) //  
#define AT91C_PIO_PD6        ((unsigned int) 1 <<  6) // Pin Controlled by PD6
#define AT91C_PD6_AC97RX   ((unsigned int) AT91C_PIO_PD6) //  
#define AT91C_PIO_PD7        ((unsigned int) 1 <<  7) // Pin Controlled by PD7
#define AT91C_PD7_AC97TX   ((unsigned int) AT91C_PIO_PD7) //  
#define AT91C_PD7_TIOA5    ((unsigned int) AT91C_PIO_PD7) //  
#define AT91C_PIO_PD8        ((unsigned int) 1 <<  8) // Pin Controlled by PD8
#define AT91C_PD8_AC97FS   ((unsigned int) AT91C_PIO_PD8) //  
#define AT91C_PD8_TIOB5    ((unsigned int) AT91C_PIO_PD8) //  
#define AT91C_PIO_PD9        ((unsigned int) 1 <<  9) // Pin Controlled by PD9
#define AT91C_PD9_AC97CK   ((unsigned int) AT91C_PIO_PD9) //  
#define AT91C_PD9_TCLK5    ((unsigned int) AT91C_PIO_PD9) //  
#define AT91C_PIO_PE0        ((unsigned int) 1 <<  0) // Pin Controlled by PE0
#define AT91C_PE0_LCDPWR   ((unsigned int) AT91C_PIO_PE0) //  
#define AT91C_PE0_PCK0     ((unsigned int) AT91C_PIO_PE0) //  
#define AT91C_PIO_PE1        ((unsigned int) 1 <<  1) // Pin Controlled by PE1
#define AT91C_PE1_LCDMOD   ((unsigned int) AT91C_PIO_PE1) //  
#define AT91C_PIO_PE10       ((unsigned int) 1 << 10) // Pin Controlled by PE10
#define AT91C_PE10_LCDD3    ((unsigned int) AT91C_PIO_PE10) //  
#define AT91C_PE10_LCDD5    ((unsigned int) AT91C_PIO_PE10) //  
#define AT91C_PIO_PE11       ((unsigned int) 1 << 11) // Pin Controlled by PE11
#define AT91C_PE11_LCDD4    ((unsigned int) AT91C_PIO_PE11) //  
#define AT91C_PE11_LCDD6    ((unsigned int) AT91C_PIO_PE11) //  
#define AT91C_PIO_PE12       ((unsigned int) 1 << 12) // Pin Controlled by PE12
#define AT91C_PE12_LCDD5    ((unsigned int) AT91C_PIO_PE12) //  
#define AT91C_PE12_LCDD7    ((unsigned int) AT91C_PIO_PE12) //  
#define AT91C_PIO_PE13       ((unsigned int) 1 << 13) // Pin Controlled by PE13
#define AT91C_PE13_LCDD6    ((unsigned int) AT91C_PIO_PE13) //  
#define AT91C_PE13_LCDD10   ((unsigned int) AT91C_PIO_PE13) //  
#define AT91C_PIO_PE14       ((unsigned int) 1 << 14) // Pin Controlled by PE14
#define AT91C_PE14_LCDD7    ((unsigned int) AT91C_PIO_PE14) //  
#define AT91C_PE14_LCDD11   ((unsigned int) AT91C_PIO_PE14) //  
#define AT91C_PIO_PE15       ((unsigned int) 1 << 15) // Pin Controlled by PE15
#define AT91C_PE15_LCDD8    ((unsigned int) AT91C_PIO_PE15) //  
#define AT91C_PE15_LCDD12   ((unsigned int) AT91C_PIO_PE15) //  
#define AT91C_PIO_PE16       ((unsigned int) 1 << 16) // Pin Controlled by PE16
#define AT91C_PE16_LCDD9    ((unsigned int) AT91C_PIO_PE16) //  
#define AT91C_PE16_LCDD13   ((unsigned int) AT91C_PIO_PE16) //  
#define AT91C_PIO_PE17       ((unsigned int) 1 << 17) // Pin Controlled by PE17
#define AT91C_PE17_LCDD10   ((unsigned int) AT91C_PIO_PE17) //  
#define AT91C_PE17_LCDD14   ((unsigned int) AT91C_PIO_PE17) //  
#define AT91C_PIO_PE18       ((unsigned int) 1 << 18) // Pin Controlled by PE18
#define AT91C_PE18_LCDD11   ((unsigned int) AT91C_PIO_PE18) //  
#define AT91C_PE18_LCDD15   ((unsigned int) AT91C_PIO_PE18) //  
#define AT91C_PIO_PE19       ((unsigned int) 1 << 19) // Pin Controlled by PE19
#define AT91C_PE19_LCDD12   ((unsigned int) AT91C_PIO_PE19) //  
#define AT91C_PE19_LCDD18   ((unsigned int) AT91C_PIO_PE19) //  
#define AT91C_PIO_PE2        ((unsigned int) 1 <<  2) // Pin Controlled by PE2
#define AT91C_PE2_LCDCC    ((unsigned int) AT91C_PIO_PE2) //  
#define AT91C_PIO_PE20       ((unsigned int) 1 << 20) // Pin Controlled by PE20
#define AT91C_PE20_LCDD13   ((unsigned int) AT91C_PIO_PE20) //  
#define AT91C_PE20_LCDD19   ((unsigned int) AT91C_PIO_PE20) //  
#define AT91C_PIO_PE21       ((unsigned int) 1 << 21) // Pin Controlled by PE21
#define AT91C_PE21_LCDD14   ((unsigned int) AT91C_PIO_PE21) //  
#define AT91C_PE21_LCDD20   ((unsigned int) AT91C_PIO_PE21) //  
#define AT91C_PIO_PE22       ((unsigned int) 1 << 22) // Pin Controlled by PE22
#define AT91C_PE22_LCDD15   ((unsigned int) AT91C_PIO_PE22) //  
#define AT91C_PE22_LCDD21   ((unsigned int) AT91C_PIO_PE22) //  
#define AT91C_PIO_PE23       ((unsigned int) 1 << 23) // Pin Controlled by PE23
#define AT91C_PE23_LCDD16   ((unsigned int) AT91C_PIO_PE23) //  
#define AT91C_PE23_LCDD22   ((unsigned int) AT91C_PIO_PE23) //  
#define AT91C_PIO_PE24       ((unsigned int) 1 << 24) // Pin Controlled by PE24
#define AT91C_PE24_LCDD17   ((unsigned int) AT91C_PIO_PE24) //  
#define AT91C_PE24_LCDD23   ((unsigned int) AT91C_PIO_PE24) //  
#define AT91C_PIO_PE25       ((unsigned int) 1 << 25) // Pin Controlled by PE25
#define AT91C_PE25_LCDD18   ((unsigned int) AT91C_PIO_PE25) //  
#define AT91C_PIO_PE26       ((unsigned int) 1 << 26) // Pin Controlled by PE26
#define AT91C_PE26_LCDD19   ((unsigned int) AT91C_PIO_PE26) //  
#define AT91C_PIO_PE27       ((unsigned int) 1 << 27) // Pin Controlled by PE27
#define AT91C_PE27_LCDD20   ((unsigned int) AT91C_PIO_PE27) //  
#define AT91C_PIO_PE28       ((unsigned int) 1 << 28) // Pin Controlled by PE28
#define AT91C_PE28_LCDD21   ((unsigned int) AT91C_PIO_PE28) //  
#define AT91C_PIO_PE29       ((unsigned int) 1 << 29) // Pin Controlled by PE29
#define AT91C_PE29_LCDD22   ((unsigned int) AT91C_PIO_PE29) //  
#define AT91C_PIO_PE3        ((unsigned int) 1 <<  3) // Pin Controlled by PE3
#define AT91C_PE3_LCDVSYNC ((unsigned int) AT91C_PIO_PE3) //  
#define AT91C_PIO_PE30       ((unsigned int) 1 << 30) // Pin Controlled by PE30
#define AT91C_PE30_LCDD23   ((unsigned int) AT91C_PIO_PE30) //  
#define AT91C_PIO_PE31       ((unsigned int) 1 << 31) // Pin Controlled by PE31
#define AT91C_PE31_PWM2     ((unsigned int) AT91C_PIO_PE31) //  
#define AT91C_PE31_PCK1     ((unsigned int) AT91C_PIO_PE31) //  
#define AT91C_PIO_PE4        ((unsigned int) 1 <<  4) // Pin Controlled by PE4
#define AT91C_PE4_LCDHSYNC ((unsigned int) AT91C_PIO_PE4) //  
#define AT91C_PIO_PE5        ((unsigned int) 1 <<  5) // Pin Controlled by PE5
#define AT91C_PE5_LCDDOTCK ((unsigned int) AT91C_PIO_PE5) //  
#define AT91C_PIO_PE6        ((unsigned int) 1 <<  6) // Pin Controlled by PE6
#define AT91C_PE6_LCDDEN   ((unsigned int) AT91C_PIO_PE6) //  
#define AT91C_PIO_PE7        ((unsigned int) 1 <<  7) // Pin Controlled by PE7
#define AT91C_PE7_LCDD0    ((unsigned int) AT91C_PIO_PE7) //  
#define AT91C_PE7_LCDD2    ((unsigned int) AT91C_PIO_PE7) //  
#define AT91C_PIO_PE8        ((unsigned int) 1 <<  8) // Pin Controlled by PE8
#define AT91C_PE8_LCDD1    ((unsigned int) AT91C_PIO_PE8) //  
#define AT91C_PE8_LCDD3    ((unsigned int) AT91C_PIO_PE8) //  
#define AT91C_PIO_PE9        ((unsigned int) 1 <<  9) // Pin Controlled by PE9
#define AT91C_PE9_LCDD2    ((unsigned int) AT91C_PIO_PE9) //  
#define AT91C_PE9_LCDD4    ((unsigned int) AT91C_PIO_PE9) //  

// *****************************************************************************
//               PERIPHERAL ID DEFINITIONS FOR AT91SAM9G45
// *****************************************************************************
#define AT91C_ID_FIQ    ((unsigned int)  0) // Advanced Interrupt Controller (FIQ)
#define AT91C_ID_SYS    ((unsigned int)  1) // System Controller
#define AT91C_ID_PIOA   ((unsigned int)  2) // Parallel IO Controller A
#define AT91C_ID_PIOB   ((unsigned int)  3) // Parallel IO Controller B
#define AT91C_ID_PIOC   ((unsigned int)  4) // Parallel IO Controller C
#define AT91C_ID_PIOD_E ((unsigned int)  5) // Parallel IO Controller D and E
#define AT91C_ID_TRNG   ((unsigned int)  6) // True Random Number Generator
#define AT91C_ID_US0    ((unsigned int)  7) // USART 0
#define AT91C_ID_US1    ((unsigned int)  8) // USART 1
#define AT91C_ID_US2    ((unsigned int)  9) // USART 2
#define AT91C_ID_US3    ((unsigned int) 10) // USART 2
#define AT91C_ID_MCI0   ((unsigned int) 11) // Multimedia Card Interface 0
#define AT91C_ID_TWI0   ((unsigned int) 12) // TWI 0
#define AT91C_ID_TWI1   ((unsigned int) 13) // TWI 1
#define AT91C_ID_SPI0   ((unsigned int) 14) // Serial Peripheral Interface
#define AT91C_ID_SPI1   ((unsigned int) 15) // Serial Peripheral Interface
#define AT91C_ID_SSC0   ((unsigned int) 16) // Serial Synchronous Controller 0
#define AT91C_ID_SSC1   ((unsigned int) 17) // Serial Synchronous Controller 1
#define AT91C_ID_TC     ((unsigned int) 18) // Timer Counter 0, 1, 2, 3, 4, 5
#define AT91C_ID_PWMC   ((unsigned int) 19) // Pulse Width Modulation Controller
#define AT91C_ID_TSADC  ((unsigned int) 20) // Touch Screen Controller
#define AT91C_ID_HDMA   ((unsigned int) 21) // HDMA
#define AT91C_ID_UHPHS  ((unsigned int) 22) // USB Host High Speed
#define AT91C_ID_LCDC   ((unsigned int) 23) // LCD Controller
#define AT91C_ID_AC97C  ((unsigned int) 24) // AC97 Controller
#define AT91C_ID_EMAC   ((unsigned int) 25) // Ethernet MAC
#define AT91C_ID_ISI    ((unsigned int) 26) // Image Sensor Interface
#define AT91C_ID_UDPHS  ((unsigned int) 27) // USB Device HS
#define AT91C_ID_MCI1   ((unsigned int) 29) // Multimedia Card Interface 1
#define AT91C_ID_VDEC   ((unsigned int) 30) // Video Decoder
#define AT91C_ID_IRQ0   ((unsigned int) 31) // Advanced Interrupt Controller (IRQ0)
#define AT91C_ALL_INT   ((unsigned int) 0xEFFFFFFF) // ALL VALID INTERRUPTS

// *****************************************************************************
//               BASE ADDRESS DEFINITIONS FOR AT91SAM9G45
// *****************************************************************************
#define AT91C_BASE_SFR       ((AT91PS_SFR) 	0xFFF74000) // (SFR) Base Address
#define AT91C_BASE_SYS       ((AT91PS_SYS) 	0xFFFFC000) // (SYS) Base Address
#define AT91C_BASE_EBI       ((AT91PS_EBI) 	0xFFFFE200) // (EBI) Base Address
#define AT91C_BASE_DDR2CP1   ((AT91PS_HDDRSDRC2) 	0xFFFFE400) // (DDR2CP1) Base Address
#define AT91C_BASE_DDR2C     ((AT91PS_HDDRSDRC2) 	0xFFFFE600) // (DDR2C) Base Address
#define AT91C_BASE_SMC       ((AT91PS_SMC) 	0xFFFFE800) // (SMC) Base Address
#define AT91C_BASE_MATRIX    ((AT91PS_MATRIX) 	0xFFFFEA00) // (MATRIX) Base Address
#define AT91C_BASE_AIC       ((AT91PS_AIC) 	0xFFFFF000) // (AIC) Base Address
#define AT91C_BASE_PDC_DBGU  ((AT91PS_PDC) 	0xFFFFEF00) // (PDC_DBGU) Base Address
#define AT91C_BASE_DBGU      ((AT91PS_DBGU) 	0xFFFFEE00) // (DBGU) Base Address
#define AT91C_BASE_PIOA      ((AT91PS_PIO) 	0xFFFFF200) // (PIOA) Base Address
#define AT91C_BASE_PIOB      ((AT91PS_PIO) 	0xFFFFF400) // (PIOB) Base Address
#define AT91C_BASE_PIOC      ((AT91PS_PIO) 	0xFFFFF600) // (PIOC) Base Address
#define AT91C_BASE_PIOD      ((AT91PS_PIO) 	0xFFFFF800) // (PIOD) Base Address
#define AT91C_BASE_PIOE      ((AT91PS_PIO) 	0xFFFFFA00) // (PIOE) Base Address
#define AT91C_BASE_PMC       ((AT91PS_PMC) 	0xFFFFFC00) // (PMC) Base Address
#define AT91C_BASE_CKGR      ((AT91PS_CKGR) 	0xFFFFFC1C) // (CKGR) Base Address
#define AT91C_BASE_RSTC      ((AT91PS_RSTC) 	0xFFFFFD00) // (RSTC) Base Address
#define AT91C_BASE_SHDWC     ((AT91PS_SHDWC) 	0xFFFFFD10) // (SHDWC) Base Address
#define AT91C_BASE_RTTC      ((AT91PS_RTTC) 	0xFFFFFD20) // (RTTC) Base Address
#define AT91C_BASE_PITC      ((AT91PS_PITC) 	0xFFFFFD30) // (PITC) Base Address
#define AT91C_BASE_WDTC      ((AT91PS_WDTC) 	0xFFFFFD40) // (WDTC) Base Address
#define AT91C_BASE_RTC       ((AT91PS_RTC) 	0xFFFFFDB0) // (RTC) Base Address
#define AT91C_BASE_TC0       ((AT91PS_TC) 	0xFFF7C000) // (TC0) Base Address
#define AT91C_BASE_TC1       ((AT91PS_TC) 	0xFFF7C040) // (TC1) Base Address
#define AT91C_BASE_TC2       ((AT91PS_TC) 	0xFFF7C080) // (TC2) Base Address
#define AT91C_BASE_TC3       ((AT91PS_TC) 	0xFFFD4000) // (TC3) Base Address
#define AT91C_BASE_TC4       ((AT91PS_TC) 	0xFFFD4040) // (TC4) Base Address
#define AT91C_BASE_TC5       ((AT91PS_TC) 	0xFFFD4080) // (TC5) Base Address
#define AT91C_BASE_TCB0      ((AT91PS_TCB) 	0xFFF7C000) // (TCB0) Base Address
#define AT91C_BASE_TCB1      ((AT91PS_TCB) 	0xFFFD4000) // (TCB1) Base Address
#define AT91C_BASE_MCI0      ((AT91PS_MCI) 	0xFFF80000) // (MCI0) Base Address
#define AT91C_BASE_MCI1      ((AT91PS_MCI) 	0xFFFD0000) // (MCI1) Base Address
#define AT91C_BASE_TWI0      ((AT91PS_TWI) 	0xFFF84000) // (TWI0) Base Address
#define AT91C_BASE_TWI1      ((AT91PS_TWI) 	0xFFF88000) // (TWI1) Base Address
#define AT91C_BASE_PDC_US0   ((AT91PS_PDC) 	0xFFF8C100) // (PDC_US0) Base Address
#define AT91C_BASE_US0       ((AT91PS_USART) 	0xFFF8C000) // (US0) Base Address
#define AT91C_BASE_PDC_US1   ((AT91PS_PDC) 	0xFFF90100) // (PDC_US1) Base Address
#define AT91C_BASE_US1       ((AT91PS_USART) 	0xFFF90000) // (US1) Base Address
#define AT91C_BASE_PDC_US2   ((AT91PS_PDC) 	0xFFF94100) // (PDC_US2) Base Address
#define AT91C_BASE_US2       ((AT91PS_USART) 	0xFFF94000) // (US2) Base Address
#define AT91C_BASE_PDC_US3   ((AT91PS_PDC) 	0xFFF98100) // (PDC_US3) Base Address
#define AT91C_BASE_US3       ((AT91PS_USART) 	0xFFF98000) // (US3) Base Address
#define AT91C_BASE_PDC_SSC0  ((AT91PS_PDC) 	0xFFF9C100) // (PDC_SSC0) Base Address
#define AT91C_BASE_SSC0      ((AT91PS_SSC) 	0xFFF9C000) // (SSC0) Base Address
#define AT91C_BASE_PDC_SSC1  ((AT91PS_PDC) 	0xFFFA0100) // (PDC_SSC1) Base Address
#define AT91C_BASE_SSC1      ((AT91PS_SSC) 	0xFFFA0000) // (SSC1) Base Address
#define AT91C_BASE_PWMC_CH0  ((AT91PS_PWMC_CH) 	0xFFFB8200) // (PWMC_CH0) Base Address
#define AT91C_BASE_PWMC_CH1  ((AT91PS_PWMC_CH) 	0xFFFB8220) // (PWMC_CH1) Base Address
#define AT91C_BASE_PWMC_CH2  ((AT91PS_PWMC_CH) 	0xFFFB8240) // (PWMC_CH2) Base Address
#define AT91C_BASE_PWMC_CH3  ((AT91PS_PWMC_CH) 	0xFFFB8260) // (PWMC_CH3) Base Address
#define AT91C_BASE_PWMC      ((AT91PS_PWMC) 	0xFFFB8000) // (PWMC) Base Address
#define AT91C_BASE_PDC_SPI0  ((AT91PS_PDC) 	0xFFFA4100) // (PDC_SPI0) Base Address
#define AT91C_BASE_PDC_SPI1  ((AT91PS_PDC) 	0xFFFA8100) // (PDC_SPI1) Base Address
#define AT91C_BASE_SPI0      ((AT91PS_SPI) 	0xFFFA4000) // (SPI0) Base Address
#define AT91C_BASE_SPI1      ((AT91PS_SPI) 	0xFFFA8000) // (SPI1) Base Address
#define AT91C_BASE_PDC_TSADC ((AT91PS_PDC) 	0xFFFB0100) // (PDC_TSADC) Base Address
#define AT91C_BASE_TSADC     ((AT91PS_TSADC) 	0xFFFB0000) // (TSADC) Base Address
#define AT91C_BASE_UDPHS_EPTFIFO ((AT91PS_UDPHS_EPTFIFO) 	0x00600000) // (UDPHS_EPTFIFO) Base Address
#define AT91C_BASE_UDPHS_EPT_0 ((AT91PS_UDPHS_EPT) 	0xFFF78100) // (UDPHS_EPT_0) Base Address
#define AT91C_BASE_UDPHS_EPT_1 ((AT91PS_UDPHS_EPT) 	0xFFF78120) // (UDPHS_EPT_1) Base Address
#define AT91C_BASE_UDPHS_EPT_2 ((AT91PS_UDPHS_EPT) 	0xFFF78140) // (UDPHS_EPT_2) Base Address
#define AT91C_BASE_UDPHS_EPT_3 ((AT91PS_UDPHS_EPT) 	0xFFF78160) // (UDPHS_EPT_3) Base Address
#define AT91C_BASE_UDPHS_EPT_4 ((AT91PS_UDPHS_EPT) 	0xFFF78180) // (UDPHS_EPT_4) Base Address
#define AT91C_BASE_UDPHS_EPT_5 ((AT91PS_UDPHS_EPT) 	0xFFF781A0) // (UDPHS_EPT_5) Base Address
#define AT91C_BASE_UDPHS_EPT_6 ((AT91PS_UDPHS_EPT) 	0xFFF781C0) // (UDPHS_EPT_6) Base Address
#define AT91C_BASE_UDPHS_DMA_1 ((AT91PS_UDPHS_DMA) 	0xFFF78310) // (UDPHS_DMA_1) Base Address
#define AT91C_BASE_UDPHS_DMA_2 ((AT91PS_UDPHS_DMA) 	0xFFF78320) // (UDPHS_DMA_2) Base Address
#define AT91C_BASE_UDPHS_DMA_3 ((AT91PS_UDPHS_DMA) 	0xFFF78330) // (UDPHS_DMA_3) Base Address
#define AT91C_BASE_UDPHS_DMA_4 ((AT91PS_UDPHS_DMA) 	0xFFF78340) // (UDPHS_DMA_4) Base Address
#define AT91C_BASE_UDPHS_DMA_5 ((AT91PS_UDPHS_DMA) 	0xFFF78350) // (UDPHS_DMA_5) Base Address
#define AT91C_BASE_UDPHS_DMA_6 ((AT91PS_UDPHS_DMA) 	0xFFF78360) // (UDPHS_DMA_6) Base Address
#define AT91C_BASE_UDPHS     ((AT91PS_UDPHS) 	0xFFF78000) // (UDPHS) Base Address
#define AT91C_BASE_PDC_AC97C ((AT91PS_PDC) 	0xFFFAC100) // (PDC_AC97C) Base Address
#define AT91C_BASE_AC97C     ((AT91PS_AC97C) 	0xFFFAC000) // (AC97C) Base Address
#define AT91C_BASE_LCDC      ((AT91PS_LCDC) 	0x00500000) // (LCDC) Base Address
#define AT91C_BASE_LCDC_16B_TFT ((AT91PS_LCDC) 	0x00500000) // (LCDC_16B_TFT) Base Address
#define AT91C_BASE_HDMA_CH_0 ((AT91PS_HDMA_CH) 	0xFFFFEC3C) // (HDMA_CH_0) Base Address
#define AT91C_BASE_HDMA_CH_1 ((AT91PS_HDMA_CH) 	0xFFFFEC64) // (HDMA_CH_1) Base Address
#define AT91C_BASE_HDMA_CH_2 ((AT91PS_HDMA_CH) 	0xFFFFEC8C) // (HDMA_CH_2) Base Address
#define AT91C_BASE_HDMA_CH_3 ((AT91PS_HDMA_CH) 	0xFFFFECB4) // (HDMA_CH_3) Base Address
#define AT91C_BASE_HDMA_CH_4 ((AT91PS_HDMA_CH) 	0xFFFFECDC) // (HDMA_CH_4) Base Address
#define AT91C_BASE_HDMA_CH_5 ((AT91PS_HDMA_CH) 	0xFFFFED04) // (HDMA_CH_5) Base Address
#define AT91C_BASE_HDMA_CH_6 ((AT91PS_HDMA_CH) 	0xFFFFED2C) // (HDMA_CH_6) Base Address
#define AT91C_BASE_HDMA_CH_7 ((AT91PS_HDMA_CH) 	0xFFFFED54) // (HDMA_CH_7) Base Address
#define AT91C_BASE_HDMA      ((AT91PS_HDMA) 	0xFFFFEC00) // (HDMA) Base Address
#define AT91C_BASE_HECC      ((AT91PS_ECC) 	0xFFFFE200) // (HECC) Base Address
#define AT91C_BASE_EMACB     ((AT91PS_EMAC) 	0xFFFBC000) // (EMACB) Base Address
#define AT91C_BASE_ISI       ((AT91PS_ISI) 	0xFFFB4000) // (ISI) Base Address
#define AT91C_BASE_UHPHS_OHCI ((AT91PS_UHPHS_OHCI) 	0x00700000) // (UHPHS_OHCI) Base Address
#define AT91C_BASE_UHPHS_EHCI ((AT91PS_UHPHS_EHCI) 	0x00800000) // (UHPHS_EHCI) Base Address
#define AT91C_BASE_TRNG      ((AT91PS_TRNG) 	0xFFFCC000) // (TRNG) Base Address

// *****************************************************************************
//               MEMORY MAPPING DEFINITIONS FOR AT91SAM9G45
// *****************************************************************************
// ITCM
#define AT91C_ITCM 	 ((char *) 	0x00100000) // Maximum ITCM Area base address
// DTCM
#define AT91C_DTCM 	 ((char *) 	0x00200000) // Maximum DTCM Area base address
// IRAM
#define AT91C_IRAM 	 ((char *) 	0x00300000) // Maximum Internal SRAM base address
#define AT91C_IRAM_SIZE	 ((unsigned int) 0x00010000) // Maximum Internal SRAM size in byte (64 Kbytes)
// IRAM_MIN
#define AT91C_IRAM_MIN	 ((char *) 	0x00300000) // Minimum Internal RAM base address
#define AT91C_IRAM_MIN_SIZE	 ((unsigned int) 0x00004000) // Minimum Internal RAM size in byte (16 Kbytes)
// IROM
#define AT91C_IROM 	 ((char *) 	0x00400000) // Internal ROM base address
#define AT91C_IROM_SIZE	 ((unsigned int) 0x00010000) // Internal ROM size in byte (64 Kbytes)
// EBI_CS0
#define AT91C_EBI_CS0	 ((char *) 	0x10000000) // EBI Chip Select 0 base address
#define AT91C_EBI_CS0_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 0 size in byte (262144 Kbytes)
// EBI_CS1
#define AT91C_EBI_CS1	 ((char *) 	0x20000000) // EBI Chip Select 1 base address
#define AT91C_EBI_CS1_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 1 size in byte (262144 Kbytes)
// EBI_SDRAM
#define AT91C_EBI_SDRAM	 ((char *) 	0x20000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_SIZE	 ((unsigned int) 0x10000000) // SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
// EBI_SDRAM_16BIT
#define AT91C_EBI_SDRAM_16BIT	 ((char *) 	0x20000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_16BIT_SIZE	 ((unsigned int) 0x02000000) // SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
// EBI_SDRAM_32BIT
#define AT91C_EBI_SDRAM_32BIT	 ((char *) 	0x20000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_32BIT_SIZE	 ((unsigned int) 0x04000000) // SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
// EBI_CS2
#define AT91C_EBI_CS2	 ((char *) 	0x30000000) // EBI Chip Select 2 base address
#define AT91C_EBI_CS2_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 2 size in byte (262144 Kbytes)
// EBI_CS3
#define AT91C_EBI_CS3	 ((char *) 	0x40000000) // EBI Chip Select 3 base address
#define AT91C_EBI_CS3_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 3 size in byte (262144 Kbytes)
// EBI_SM
#define AT91C_EBI_SM	 ((char *) 	0x40000000) // NANDFLASH on EBI Chip Select 3 base address
#define AT91C_EBI_SM_SIZE	 ((unsigned int) 0x10000000) // NANDFLASH on EBI Chip Select 3 size in byte (262144 Kbytes)
// EBI_CS4
#define AT91C_EBI_CS4	 ((char *) 	0x50000000) // EBI Chip Select 4 base address
#define AT91C_EBI_CS4_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 4 size in byte (262144 Kbytes)
// EBI_CF0
#define AT91C_EBI_CF0	 ((char *) 	0x50000000) // CompactFlash 0 on EBI Chip Select 4 base address
#define AT91C_EBI_CF0_SIZE	 ((unsigned int) 0x10000000) // CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
// EBI_CS5
#define AT91C_EBI_CS5	 ((char *) 	0x60000000) // EBI Chip Select 5 base address
#define AT91C_EBI_CS5_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 5 size in byte (262144 Kbytes)
// EBI_CF1
#define AT91C_EBI_CF1	 ((char *) 	0x60000000) // CompactFlash 1 on EBIChip Select 5 base address
#define AT91C_EBI_CF1_SIZE	 ((unsigned int) 0x10000000) // CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
// DDR2
#define AT91C_DDR2 	 ((char *) 	0x70000000) // DDR2/LPDDR space base address
#define AT91C_DDR2_SIZE	 ((unsigned int) 0x10000000) // DDR2/LPDDR space size in byte (262144 Kbytes)
#endif /* __IAR_SYSTEMS_ICC__ */

#ifdef __IAR_SYSTEMS_ASM__

// - Hardware register definition

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR SPECIAL FUNCTION REGISTER
// - *****************************************************************************
// - -------- SFR_EMA : (SFR Offset: 0x0) memory Extra Margin Adjustment control register -------- 
AT91C_SFR_RAM_EMA         EQU (0x7 <<  0) ;- (SFR) SRAM EMA
AT91C_SFR_RAM_EMA_0       EQU (0x0) ;- (SFR) Normal Mode
AT91C_SFR_RAM_EMA_1       EQU (0x1) ;- (SFR) DEBUG MODE 1
AT91C_SFR_RAM_EMA_2       EQU (0x2) ;- (SFR) DEBUG MODE 2
AT91C_SFR_RAM_EMA_3       EQU (0x3) ;- (SFR) DEBUG MODE 3
AT91C_SFR_RAM_EMA_4       EQU (0x4) ;- (SFR) DEBUG MODE 4
AT91C_SFR_RAM_EMA_5       EQU (0x5) ;- (SFR) DEBUG MODE 5
AT91C_SFR_RAM_EMA_6       EQU (0x6) ;- (SFR) DEBUG MODE 6
AT91C_SFR_RAM_EMA_7       EQU (0x7) ;- (SFR) DEBUG MODE 7
AT91C_SFR_DPRAM_EMA       EQU (0x7 <<  4) ;- (SFR) SRAM EMA
AT91C_SFR_DPRAM_EMA_0     EQU (0x0 <<  4) ;- (SFR) Normal Mode
AT91C_SFR_DPRAM_EMA_1     EQU (0x1 <<  4) ;- (SFR) DEBUG MODE 1
AT91C_SFR_DPRAM_EMA_2     EQU (0x2 <<  4) ;- (SFR) DEBUG MODE 2
AT91C_SFR_DPRAM_EMA_3     EQU (0x3 <<  4) ;- (SFR) DEBUG MODE 3
AT91C_SFR_DPRAM_EMA_4     EQU (0x4 <<  4) ;- (SFR) DEBUG MODE 4
AT91C_SFR_DPRAM_EMA_5     EQU (0x5 <<  4) ;- (SFR) DEBUG MODE 5
AT91C_SFR_DPRAM_EMA_6     EQU (0x6 <<  4) ;- (SFR) DEBUG MODE 6
AT91C_SFR_DPRAM_EMA_7     EQU (0x7 <<  4) ;- (SFR) DEBUG MODE 7
AT91C_SFR_RF_EMA          EQU (0x7 <<  8) ;- (SFR) SRAM EMA
AT91C_SFR_RF_EMA_0        EQU (0x0 <<  8) ;- (SFR) Normal Mode
AT91C_SFR_RF_EMA_1        EQU (0x1 <<  8) ;- (SFR) DEBUG MODE 1
AT91C_SFR_RF_EMA_2        EQU (0x2 <<  8) ;- (SFR) DEBUG MODE 2
AT91C_SFR_RF_EMA_3        EQU (0x3 <<  8) ;- (SFR) DEBUG MODE 3
AT91C_SFR_RF_EMA_4        EQU (0x4 <<  8) ;- (SFR) DEBUG MODE 4
AT91C_SFR_RF_EMA_5        EQU (0x5 <<  8) ;- (SFR) DEBUG MODE 5
AT91C_SFR_RF_EMA_6        EQU (0x6 <<  8) ;- (SFR) DEBUG MODE 6
AT91C_SFR_RF_EMA_7        EQU (0x7 <<  8) ;- (SFR) DEBUG MODE 7
AT91C_SFR_DPRF_EMA        EQU (0x7 << 12) ;- (SFR) SRAM EMA
AT91C_SFR_DPRF_EMA_0      EQU (0x0 << 12) ;- (SFR) Normal Mode
AT91C_SFR_DPRF_EMA_1      EQU (0x1 << 12) ;- (SFR) DEBUG MODE 1
AT91C_SFR_DPRF_EMA_2      EQU (0x2 << 12) ;- (SFR) DEBUG MODE 2
AT91C_SFR_DPRF_EMA_3      EQU (0x3 << 12) ;- (SFR) DEBUG MODE 3
AT91C_SFR_DPRF_EMA_4      EQU (0x4 << 12) ;- (SFR) DEBUG MODE 4
AT91C_SFR_DPRF_EMA_5      EQU (0x5 << 12) ;- (SFR) DEBUG MODE 5
AT91C_SFR_DPRF_EMA_6      EQU (0x6 << 12) ;- (SFR) DEBUG MODE 6
AT91C_SFR_DPRF_EMA_7      EQU (0x7 << 12) ;- (SFR) DEBUG MODE 7
AT91C_SFR_ROM_EMA         EQU (0x7 << 16) ;- (SFR) SRAM EMA
AT91C_SFR_ROM_EMA_0       EQU (0x0 << 16) ;- (SFR) Normal Mode
AT91C_SFR_ROM_EMA_1       EQU (0x1 << 16) ;- (SFR) DEBUG MODE 1
AT91C_SFR_ROM_EMA_2       EQU (0x2 << 16) ;- (SFR) DEBUG MODE 2
AT91C_SFR_ROM_EMA_3       EQU (0x3 << 16) ;- (SFR) DEBUG MODE 3
AT91C_SFR_ROM_EMA_4       EQU (0x4 << 16) ;- (SFR) DEBUG MODE 4
AT91C_SFR_ROM_EMA_5       EQU (0x5 << 16) ;- (SFR) DEBUG MODE 5
AT91C_SFR_ROM_EMA_6       EQU (0x6 << 16) ;- (SFR) DEBUG MODE 6
AT91C_SFR_ROM_EMA_7       EQU (0x7 << 16) ;- (SFR) DEBUG MODE 7
// - -------- SFR_DDRCFG : (SFR Offset: 0x4) DDR2 SSTL18 control register -------- 
AT91C_SFR_DDRCFG_SSTL     EQU (0x1 <<  0) ;- (SFR) Control DDR2 pads SSTL mode control
AT91C_SFR_DDRCFG_SSTL_NORMAL EQU (0x0) ;- (SFR) Force pads in SSTL18 mode when DDR2 is connected
AT91C_SFR_DDRCFG_SSTL_COMPATIBLE EQU (0x1) ;- (SFR) LVCMOS level (compatible SSTL18)
AT91C_SFR_DDRCFG_CLKDELAY EQU (0x1 <<  8) ;- (SFR) Control DDR2 pads clocks delay on clk, dqs0, dqs1
AT91C_SFR_DDRCFG_CLKDELAY_HARD EQU (0x0 <<  8) ;- (SFR) Fixed by hardware
AT91C_SFR_DDRCFG_CLKDELAY_SOFT EQU (0x1 <<  8) ;- (SFR) Software must write correct delay value
// - -------- SFR_DDRDELAY : (SFR Offset: 0x8) DDR2 controller clock delay -------- 
AT91C_SFR_DDRDELAY_CLK    EQU (0xFF <<  0) ;- (SFR) Control CLK clock delay
AT91C_SFR_DDRDELAY_CLK_0  EQU (0x0) ;- (SFR) minimum delay
AT91C_SFR_DDRDELAY_CLK_1  EQU (0x1) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_2  EQU (0x2) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_3  EQU (0x3) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_4  EQU (0x4) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_5  EQU (0x5) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_6  EQU (0x6) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_7  EQU (0x7) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_8  EQU (0x8) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_9  EQU (0x9) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_10 EQU (0xA) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_11 EQU (0xB) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_12 EQU (0xC) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_13 EQU (0xD) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_14 EQU (0xE) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_15 EQU (0xF) ;- (SFR) 
AT91C_SFR_DDRDELAY_CLK_16 EQU (0x10) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0   EQU (0xFF <<  8) ;- (SFR) Control DQS0 clock delay
AT91C_SFR_DDRDELAY_DQS0_0 EQU (0x0 <<  8) ;- (SFR) minimum delay
AT91C_SFR_DDRDELAY_DQS0_1 EQU (0x1 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_2 EQU (0x2 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_3 EQU (0x3 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_4 EQU (0x4 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_5 EQU (0x5 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_6 EQU (0x6 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_7 EQU (0x7 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_8 EQU (0x8 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_9 EQU (0x9 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_10 EQU (0xA <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_11 EQU (0xB <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_12 EQU (0xC <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_13 EQU (0xD <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_14 EQU (0xE <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_15 EQU (0xF <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS0_16 EQU (0x10 <<  8) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1   EQU (0xFF << 16) ;- (SFR) Control DQS1 clock delay
AT91C_SFR_DDRDELAY_DQS1_0 EQU (0x0 << 16) ;- (SFR) minimum delay
AT91C_SFR_DDRDELAY_DQS1_1 EQU (0x1 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_2 EQU (0x2 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_3 EQU (0x3 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_4 EQU (0x4 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_5 EQU (0x5 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_6 EQU (0x6 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_7 EQU (0x7 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_8 EQU (0x8 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_9 EQU (0x9 << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_10 EQU (0xA << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_11 EQU (0xB << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_12 EQU (0xC << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_13 EQU (0xD << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_14 EQU (0xE << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_15 EQU (0xF << 16) ;- (SFR) 
AT91C_SFR_DDRDELAY_DQS1_16 EQU (0x10 << 16) ;- (SFR) 
// - -------- SFR_EBIDELAY : (SFR Offset: 0xc) EBI DDR controller clock delay -------- 
AT91C_SFR_EBIDELAY_CLK    EQU (0xFF <<  0) ;- (SFR) Control CLK clock delay
AT91C_SFR_EBIDELAY_CLK_0  EQU (0x0) ;- (SFR) minimum delay
AT91C_SFR_EBIDELAY_CLK_1  EQU (0x1) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_2  EQU (0x2) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_3  EQU (0x3) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_4  EQU (0x4) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_5  EQU (0x5) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_6  EQU (0x6) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_7  EQU (0x7) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_8  EQU (0x8) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_9  EQU (0x9) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_10 EQU (0xA) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_11 EQU (0xB) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_12 EQU (0xC) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_13 EQU (0xD) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_14 EQU (0xE) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_15 EQU (0xF) ;- (SFR) 
AT91C_SFR_EBIDELAY_CLK_16 EQU (0x10) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0   EQU (0xFF <<  8) ;- (SFR) Control DQS0 clock delay
AT91C_SFR_EBIDELAY_DQS0_0 EQU (0x0 <<  8) ;- (SFR) minimum delay
AT91C_SFR_EBIDELAY_DQS0_1 EQU (0x1 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_2 EQU (0x2 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_3 EQU (0x3 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_4 EQU (0x4 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_5 EQU (0x5 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_6 EQU (0x6 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_7 EQU (0x7 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_8 EQU (0x8 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_9 EQU (0x9 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_10 EQU (0xA <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_11 EQU (0xB <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_12 EQU (0xC <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_13 EQU (0xD <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_14 EQU (0xE <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_15 EQU (0xF <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS0_16 EQU (0x10 <<  8) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1   EQU (0xFF << 16) ;- (SFR) Control DQS1 clock delay
AT91C_SFR_EBIDELAY_DQS1_0 EQU (0x0 << 16) ;- (SFR) minimum delay
AT91C_SFR_EBIDELAY_DQS1_1 EQU (0x1 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_2 EQU (0x2 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_3 EQU (0x3 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_4 EQU (0x4 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_5 EQU (0x5 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_6 EQU (0x6 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_7 EQU (0x7 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_8 EQU (0x8 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_9 EQU (0x9 << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_10 EQU (0xA << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_11 EQU (0xB << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_12 EQU (0xC << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_13 EQU (0xD << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_14 EQU (0xE << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_15 EQU (0xF << 16) ;- (SFR) 
AT91C_SFR_EBIDELAY_DQS1_16 EQU (0x10 << 16) ;- (SFR) 
// - -------- SFR_UTMICFG : (SFR Offset: 0x10) UTMI Software Reset -------- 
AT91C_SFR_UTMICFG_PORT0   EQU (0x1 <<  0) ;- (SFR) UTMI Software Reset port 0
AT91C_SFR_UTMICFG_PORT1   EQU (0x1 <<  1) ;- (SFR) UTMI Software Reset port 1
AT91C_SFR_UTMICFG_OHCI_SUSP_INT_ENABLE EQU (0x1 <<  2) ;- (SFR) OHCI Suspend Interrupt enable
// - -------- SFR_OHCI_SUSP_INT : (SFR Offset: 0x14) OHCI suspend Interrupt status -------- 
AT91C_SFR_OHCI_SUSP_INT_STATUS EQU (0x3 <<  0) ;- (SFR) OHCI suspend Interrupt status
AT91C_SFR_OHCI_SUSP_INT_STATUS_PORT0 EQU (0x1) ;- (SFR) OHCI suspend Interrupt status for port 0
AT91C_SFR_OHCI_SUSP_INT_STATUS_PORT1 EQU (0x2) ;- (SFR) OHCI suspend Interrupt status for port 1

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR System Peripherals
// - *****************************************************************************
// - -------- SLCKSEL : (SYS Offset: 0x3d50) Slow Clock Selection Register -------- 
AT91C_SLCKSEL_RCEN        EQU (0x1 <<  0) ;- (SYS) Enable Internal RC Oscillator
AT91C_SLCKSEL_OSC32EN     EQU (0x1 <<  1) ;- (SYS) Enable External Oscillator
AT91C_SLCKSEL_OSC32BYP    EQU (0x1 <<  2) ;- (SYS) Bypass External Oscillator
AT91C_SLCKSEL_OSCSEL      EQU (0x1 <<  3) ;- (SYS) OSC Selection
// - -------- GPBR : (SYS Offset: 0x3d60) GPBR General Purpose Register -------- 
AT91C_GPBR_GPRV           EQU (0x0 <<  0) ;- (SYS) General Purpose Register Value

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR External Bus Interface
// - *****************************************************************************

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR DDR2/SDRAM Controller
// - *****************************************************************************
// - -------- HDDRSDRC2_MR : (HDDRSDRC2 Offset: 0x0) Mode Register -------- 
AT91C_DDRC2_MODE          EQU (0x7 <<  0) ;- (HDDRSDRC2) DDR/SDRAM Command Mode
AT91C_DDRC2_MODE_NORMAL_CMD EQU (0x0) ;- (HDDRSDRC2) Normal Mode
AT91C_DDRC2_MODE_NOP_CMD  EQU (0x1) ;- (HDDRSDRC2) Issue a NOP Command at every access
AT91C_DDRC2_MODE_PRCGALL_CMD EQU (0x2) ;- (HDDRSDRC2) Issue a All Banks Precharge Command at every access
AT91C_DDRC2_MODE_LMR_CMD  EQU (0x3) ;- (HDDRSDRC2) Issue a Load Mode Register at every access
AT91C_DDRC2_MODE_RFSH_CMD EQU (0x4) ;- (HDDRSDRC2) Issue a Refresh
AT91C_DDRC2_MODE_EXT_LMR_CMD EQU (0x5) ;- (HDDRSDRC2) Issue an Extended Load Mode Register
AT91C_DDRC2_MODE_DEEP_CMD EQU (0x6) ;- (HDDRSDRC2) Enter Deep Power Mode
AT91C_DDRC2_MODE_Reserved EQU (0x7) ;- (HDDRSDRC2) Reserved value
// - -------- HDDRSDRC2_RTR : (HDDRSDRC2 Offset: 0x4) Refresh Timer Register -------- 
AT91C_DDRC2_COUNT         EQU (0xFFF <<  0) ;- (HDDRSDRC2) Refresh Timer Count
// - -------- HDDRSDRC2_CR : (HDDRSDRC2 Offset: 0x8) Configuration Register -------- 
AT91C_DDRC2_NC            EQU (0x3 <<  0) ;- (HDDRSDRC2) Number of Column Bits
AT91C_DDRC2_NC_DDR9_SDR8  EQU (0x0) ;- (HDDRSDRC2) DDR 9 Bits | SDR 8 Bits
AT91C_DDRC2_NC_DDR10_SDR9 EQU (0x1) ;- (HDDRSDRC2) DDR 10 Bits | SDR 9 Bits
AT91C_DDRC2_NC_DDR11_SDR10 EQU (0x2) ;- (HDDRSDRC2) DDR 11 Bits | SDR 10 Bits
AT91C_DDRC2_NC_DDR12_SDR11 EQU (0x3) ;- (HDDRSDRC2) DDR 12 Bits | SDR 11 Bits
AT91C_DDRC2_NR            EQU (0x3 <<  2) ;- (HDDRSDRC2) Number of Row Bits
AT91C_DDRC2_NR_11         EQU (0x0 <<  2) ;- (HDDRSDRC2) 11 Bits
AT91C_DDRC2_NR_12         EQU (0x1 <<  2) ;- (HDDRSDRC2) 12 Bits
AT91C_DDRC2_NR_13         EQU (0x2 <<  2) ;- (HDDRSDRC2) 13 Bits
AT91C_DDRC2_NR_14         EQU (0x3 <<  2) ;- (HDDRSDRC2) 14 Bits
AT91C_DDRC2_CAS           EQU (0x7 <<  4) ;- (HDDRSDRC2) CAS Latency
AT91C_DDRC2_CAS_2         EQU (0x2 <<  4) ;- (HDDRSDRC2) 2 cycles (DDR1/SDR)
AT91C_DDRC2_CAS_3         EQU (0x3 <<  4) ;- (HDDRSDRC2) 3 cycles (DDR2/DDR1/SDR)
AT91C_DDRC2_CAS_4         EQU (0x4 <<  4) ;- (HDDRSDRC2) 4 cycles (DDR2)
AT91C_DDRC2_CAS_2_5       EQU (0x6 <<  4) ;- (HDDRSDRC2) 2.5 cycles (DDR1)
AT91C_DDRC2_DLL           EQU (0x1 <<  7) ;- (HDDRSDRC2) DLL Reset
AT91C_DDRC2_DLL_RESET_DISABLED EQU (0x0 <<  7) ;- (HDDRSDRC2) DLL normal mode
AT91C_DDRC2_DLL_RESET_ENABLED EQU (0x1 <<  7) ;- (HDDRSDRC2) Reset DLL
AT91C_DDRC2_DIC_DS        EQU (0x1 <<  8) ;- (HDDRSDRC2) Output driver impedance control
AT91C_DDRC2_OCD           EQU (0x7 << 12) ;- (HDDRSDRC2) Off chip driver
AT91C_DDRC2_OCD_EXIT      EQU (0x0 << 12) ;- (HDDRSDRC2) Exit OCD calibration mode.
AT91C_DDRC2_OCD_DEFAULT   EQU (0x7 << 12) ;- (HDDRSDRC2) Program OCD calibration default value.
AT91C_DDRC2_DQMS          EQU (0x1 << 16) ;- (HDDRSDRC2) Data Mask share
AT91C_DDRC2_DQMS_NOT_SHARED EQU (0x0 << 16) ;- (HDDRSDRC2) Used DQM bits are not shared
AT91C_DDRC2_DQMS_SHARED   EQU (0x1 << 16) ;- (HDDRSDRC2) Used DQM bits are shared
AT91C_DDRC2_ENRDM         EQU (0x1 << 17) ;- (HDDRSDRC2) DQS/DATA phase error correction
AT91C_DDRC2_ENRDM_OFF     EQU (0x0 << 17) ;- (HDDRSDRC2) phase error correction disabled
AT91C_DDRC2_ENRDM_ON      EQU (0x1 << 17) ;- (HDDRSDRC2) phase error correction enabled
// - -------- HDDRSDRC2_T0PR : (HDDRSDRC2 Offset: 0xc) Timing0 Register -------- 
AT91C_DDRC2_TRAS          EQU (0xF <<  0) ;- (HDDRSDRC2) Active to precharge delay
AT91C_DDRC2_TRAS_0        EQU (0x0) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRAS_1        EQU (0x1) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRAS_2        EQU (0x2) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRAS_3        EQU (0x3) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRAS_4        EQU (0x4) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRAS_5        EQU (0x5) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRAS_6        EQU (0x6) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRAS_7        EQU (0x7) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRAS_8        EQU (0x8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRAS_9        EQU (0x9) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRAS_10       EQU (0xA) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRAS_11       EQU (0xB) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRAS_12       EQU (0xC) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRAS_13       EQU (0xD) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRAS_14       EQU (0xE) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRAS_15       EQU (0xF) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRCD          EQU (0xF <<  4) ;- (HDDRSDRC2) Row to column delay
AT91C_DDRC2_TRCD_0        EQU (0x0 <<  4) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRCD_1        EQU (0x1 <<  4) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRCD_2        EQU (0x2 <<  4) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRCD_3        EQU (0x3 <<  4) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRCD_4        EQU (0x4 <<  4) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRCD_5        EQU (0x5 <<  4) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRCD_6        EQU (0x6 <<  4) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRCD_7        EQU (0x7 <<  4) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRCD_8        EQU (0x8 <<  4) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRCD_9        EQU (0x9 <<  4) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRCD_10       EQU (0xA <<  4) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRCD_11       EQU (0xB <<  4) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRCD_12       EQU (0xC <<  4) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRCD_13       EQU (0xD <<  4) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRCD_14       EQU (0xE <<  4) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRCD_15       EQU (0xF <<  4) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TWR           EQU (0xF <<  8) ;- (HDDRSDRC2) Write recovery delay
AT91C_DDRC2_TWR_0         EQU (0x0 <<  8) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TWR_1         EQU (0x1 <<  8) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TWR_2         EQU (0x2 <<  8) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TWR_3         EQU (0x3 <<  8) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TWR_4         EQU (0x4 <<  8) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TWR_5         EQU (0x5 <<  8) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TWR_6         EQU (0x6 <<  8) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TWR_7         EQU (0x7 <<  8) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TWR_8         EQU (0x8 <<  8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TWR_9         EQU (0x9 <<  8) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TWR_10        EQU (0xA <<  8) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TWR_11        EQU (0xB <<  8) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TWR_12        EQU (0xC <<  8) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TWR_13        EQU (0xD <<  8) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TWR_14        EQU (0xE <<  8) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TWR_15        EQU (0xF <<  8) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRC           EQU (0xF << 12) ;- (HDDRSDRC2) Row cycle delay
AT91C_DDRC2_TRC_0         EQU (0x0 << 12) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRC_1         EQU (0x1 << 12) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRC_2         EQU (0x2 << 12) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRC_3         EQU (0x3 << 12) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRC_4         EQU (0x4 << 12) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRC_5         EQU (0x5 << 12) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRC_6         EQU (0x6 << 12) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRC_7         EQU (0x7 << 12) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRC_8         EQU (0x8 << 12) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRC_9         EQU (0x9 << 12) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRC_10        EQU (0xA << 12) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRC_11        EQU (0xB << 12) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRC_12        EQU (0xC << 12) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRC_13        EQU (0xD << 12) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRC_14        EQU (0xE << 12) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRC_15        EQU (0xF << 12) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRP           EQU (0xF << 16) ;- (HDDRSDRC2) Row precharge delay
AT91C_DDRC2_TRP_0         EQU (0x0 << 16) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRP_1         EQU (0x1 << 16) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRP_2         EQU (0x2 << 16) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRP_3         EQU (0x3 << 16) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRP_4         EQU (0x4 << 16) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRP_5         EQU (0x5 << 16) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRP_6         EQU (0x6 << 16) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRP_7         EQU (0x7 << 16) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRP_8         EQU (0x8 << 16) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRP_9         EQU (0x9 << 16) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRP_10        EQU (0xA << 16) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRP_11        EQU (0xB << 16) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRP_12        EQU (0xC << 16) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRP_13        EQU (0xD << 16) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRP_14        EQU (0xE << 16) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRP_15        EQU (0xF << 16) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRRD          EQU (0xF << 20) ;- (HDDRSDRC2) Active bankA to Active bankB
AT91C_DDRC2_TRRD_0        EQU (0x0 << 20) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRRD_1        EQU (0x1 << 20) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRRD_2        EQU (0x2 << 20) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRRD_3        EQU (0x3 << 20) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRRD_4        EQU (0x4 << 20) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRRD_5        EQU (0x5 << 20) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRRD_6        EQU (0x6 << 20) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRRD_7        EQU (0x7 << 20) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRRD_8        EQU (0x8 << 20) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRRD_9        EQU (0x9 << 20) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRRD_10       EQU (0xA << 20) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRRD_11       EQU (0xB << 20) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRRD_12       EQU (0xC << 20) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRRD_13       EQU (0xD << 20) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRRD_14       EQU (0xE << 20) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRRD_15       EQU (0xF << 20) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TWTR          EQU (0x1 << 24) ;- (HDDRSDRC2) Internal write to read delay
AT91C_DDRC2_TWTR_0        EQU (0x0 << 24) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TWTR_1        EQU (0x1 << 24) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TMRD          EQU (0xF << 28) ;- (HDDRSDRC2) Load mode register command to active or refresh command
AT91C_DDRC2_TMRD_0        EQU (0x0 << 28) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TMRD_1        EQU (0x1 << 28) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TMRD_2        EQU (0x2 << 28) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TMRD_3        EQU (0x3 << 28) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TMRD_4        EQU (0x4 << 28) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TMRD_5        EQU (0x5 << 28) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TMRD_6        EQU (0x6 << 28) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TMRD_7        EQU (0x7 << 28) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TMRD_8        EQU (0x8 << 28) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TMRD_9        EQU (0x9 << 28) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TMRD_10       EQU (0xA << 28) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TMRD_11       EQU (0xB << 28) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TMRD_12       EQU (0xC << 28) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TMRD_13       EQU (0xD << 28) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TMRD_14       EQU (0xE << 28) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TMRD_15       EQU (0xF << 28) ;- (HDDRSDRC2) Value : 15
// - -------- HDDRSDRC2_T1PR : (HDDRSDRC2 Offset: 0x10) Timing1 Register -------- 
AT91C_DDRC2_TRFC          EQU (0x1F <<  0) ;- (HDDRSDRC2) row cycle delay
AT91C_DDRC2_TRFC_0        EQU (0x0) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRFC_1        EQU (0x1) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRFC_2        EQU (0x2) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRFC_3        EQU (0x3) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRFC_4        EQU (0x4) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRFC_5        EQU (0x5) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRFC_6        EQU (0x6) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRFC_7        EQU (0x7) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRFC_8        EQU (0x8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRFC_9        EQU (0x9) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRFC_10       EQU (0xA) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRFC_11       EQU (0xB) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRFC_12       EQU (0xC) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRFC_13       EQU (0xD) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRFC_14       EQU (0xE) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRFC_15       EQU (0xF) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRFC_16       EQU (0x10) ;- (HDDRSDRC2) Value : 16
AT91C_DDRC2_TRFC_17       EQU (0x11) ;- (HDDRSDRC2) Value : 17
AT91C_DDRC2_TRFC_18       EQU (0x12) ;- (HDDRSDRC2) Value : 18
AT91C_DDRC2_TRFC_19       EQU (0x13) ;- (HDDRSDRC2) Value : 19
AT91C_DDRC2_TRFC_20       EQU (0x14) ;- (HDDRSDRC2) Value : 20
AT91C_DDRC2_TRFC_21       EQU (0x15) ;- (HDDRSDRC2) Value : 21
AT91C_DDRC2_TRFC_22       EQU (0x16) ;- (HDDRSDRC2) Value : 22
AT91C_DDRC2_TRFC_23       EQU (0x17) ;- (HDDRSDRC2) Value : 23
AT91C_DDRC2_TRFC_24       EQU (0x18) ;- (HDDRSDRC2) Value : 24
AT91C_DDRC2_TRFC_25       EQU (0x19) ;- (HDDRSDRC2) Value : 25
AT91C_DDRC2_TRFC_26       EQU (0x1A) ;- (HDDRSDRC2) Value : 26
AT91C_DDRC2_TRFC_27       EQU (0x1B) ;- (HDDRSDRC2) Value : 27
AT91C_DDRC2_TRFC_28       EQU (0x1C) ;- (HDDRSDRC2) Value : 28
AT91C_DDRC2_TRFC_29       EQU (0x1D) ;- (HDDRSDRC2) Value : 29
AT91C_DDRC2_TRFC_30       EQU (0x1E) ;- (HDDRSDRC2) Value : 30
AT91C_DDRC2_TRFC_31       EQU (0x1F) ;- (HDDRSDRC2) Value : 31
AT91C_DDRC2_TXSNR         EQU (0xFF <<  8) ;- (HDDRSDRC2) Exit self refresh delay to Read command
AT91C_DDRC2_TXSNR_0       EQU (0x0 <<  8) ;- (HDDRSDRC2) Value :   0
AT91C_DDRC2_TXSNR_8       EQU (0x8 <<  8) ;- (HDDRSDRC2) Value :   8
AT91C_DDRC2_TXSNR_16      EQU (0x10 <<  8) ;- (HDDRSDRC2) Value :  16
AT91C_DDRC2_TXSNR_32      EQU (0x20 <<  8) ;- (HDDRSDRC2) Value :  32
AT91C_DDRC2_TXSNR_48      EQU (0x30 <<  8) ;- (HDDRSDRC2) Value :  48
AT91C_DDRC2_TXSNR_64      EQU (0x40 <<  8) ;- (HDDRSDRC2) Value :  64
AT91C_DDRC2_TXSNR_80      EQU (0x50 <<  8) ;- (HDDRSDRC2) Value :  80
AT91C_DDRC2_TXSNR_96      EQU (0x60 <<  8) ;- (HDDRSDRC2) Value :  96
AT91C_DDRC2_TXSNR_112     EQU (0x70 <<  8) ;- (HDDRSDRC2) Value : 112
AT91C_DDRC2_TXSNR_128     EQU (0x80 <<  8) ;- (HDDRSDRC2) Value : 128
AT91C_DDRC2_TXSNR_144     EQU (0x90 <<  8) ;- (HDDRSDRC2) Value : 144
AT91C_DDRC2_TXSNR_160     EQU (0xA0 <<  8) ;- (HDDRSDRC2) Value : 160
AT91C_DDRC2_TXSNR_176     EQU (0xB0 <<  8) ;- (HDDRSDRC2) Value : 176
AT91C_DDRC2_TXSNR_192     EQU (0xC0 <<  8) ;- (HDDRSDRC2) Value : 192
AT91C_DDRC2_TXSNR_208     EQU (0xD0 <<  8) ;- (HDDRSDRC2) Value : 208
AT91C_DDRC2_TXSNR_224     EQU (0xE0 <<  8) ;- (HDDRSDRC2) Value : 224
AT91C_DDRC2_TXSNR_240     EQU (0xF0 <<  8) ;- (HDDRSDRC2) Value : 240
AT91C_DDRC2_TXSNR_255     EQU (0xFF <<  8) ;- (HDDRSDRC2) Value : 255
AT91C_DDRC2_TXSRD         EQU (0xFF << 16) ;- (HDDRSDRC2) Exit self refresh delay to Read command
AT91C_DDRC2_TXSRD_0       EQU (0x0 << 16) ;- (HDDRSDRC2) Value :   0
AT91C_DDRC2_TXSRD_8       EQU (0x8 << 16) ;- (HDDRSDRC2) Value :   8
AT91C_DDRC2_TXSRD_16      EQU (0x10 << 16) ;- (HDDRSDRC2) Value :  16
AT91C_DDRC2_TXSRD_32      EQU (0x20 << 16) ;- (HDDRSDRC2) Value :  32
AT91C_DDRC2_TXSRD_48      EQU (0x30 << 16) ;- (HDDRSDRC2) Value :  48
AT91C_DDRC2_TXSRD_64      EQU (0x40 << 16) ;- (HDDRSDRC2) Value :  64
AT91C_DDRC2_TXSRD_80      EQU (0x50 << 16) ;- (HDDRSDRC2) Value :  80
AT91C_DDRC2_TXSRD_96      EQU (0x60 << 16) ;- (HDDRSDRC2) Value :  96
AT91C_DDRC2_TXSRD_112     EQU (0x70 << 16) ;- (HDDRSDRC2) Value : 112
AT91C_DDRC2_TXSRD_128     EQU (0x80 << 16) ;- (HDDRSDRC2) Value : 128
AT91C_DDRC2_TXSRD_144     EQU (0x90 << 16) ;- (HDDRSDRC2) Value : 144
AT91C_DDRC2_TXSRD_160     EQU (0xA0 << 16) ;- (HDDRSDRC2) Value : 160
AT91C_DDRC2_TXSRD_176     EQU (0xB0 << 16) ;- (HDDRSDRC2) Value : 176
AT91C_DDRC2_TXSRD_192     EQU (0xC0 << 16) ;- (HDDRSDRC2) Value : 192
AT91C_DDRC2_TXSRD_208     EQU (0xD0 << 16) ;- (HDDRSDRC2) Value : 208
AT91C_DDRC2_TXSRD_224     EQU (0xE0 << 16) ;- (HDDRSDRC2) Value : 224
AT91C_DDRC2_TXSRD_240     EQU (0xF0 << 16) ;- (HDDRSDRC2) Value : 240
AT91C_DDRC2_TXSRD_255     EQU (0xFF << 16) ;- (HDDRSDRC2) Value : 255
AT91C_DDRC2_TXP           EQU (0xF << 24) ;- (HDDRSDRC2) Exit Power-down delay to first command
AT91C_DDRC2_TXP_0         EQU (0x0 << 24) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TXP_1         EQU (0x1 << 24) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TXP_2         EQU (0x2 << 24) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TXP_3         EQU (0x3 << 24) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TXP_4         EQU (0x4 << 24) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TXP_5         EQU (0x5 << 24) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TXP_6         EQU (0x6 << 24) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TXP_7         EQU (0x7 << 24) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TXP_8         EQU (0x8 << 24) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TXP_9         EQU (0x9 << 24) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TXP_10        EQU (0xA << 24) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TXP_11        EQU (0xB << 24) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TXP_12        EQU (0xC << 24) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TXP_13        EQU (0xD << 24) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TXP_14        EQU (0xE << 24) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TXP_15        EQU (0xF << 24) ;- (HDDRSDRC2) Value : 15
// - -------- HDDRSDRC2_T2PR : (HDDRSDRC2 Offset: 0x14) Timing2 Register -------- 
AT91C_DDRC2_TXARD         EQU (0xF <<  0) ;- (HDDRSDRC2) Exit active power down delay to read command in 'Fast Exit' mode.
AT91C_DDRC2_TXARD_0       EQU (0x0) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TXARD_1       EQU (0x1) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TXARD_2       EQU (0x2) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TXARD_3       EQU (0x3) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TXARD_4       EQU (0x4) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TXARD_5       EQU (0x5) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TXARD_6       EQU (0x6) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TXARD_7       EQU (0x7) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TXARD_8       EQU (0x8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TXARD_9       EQU (0x9) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TXARD_10      EQU (0xA) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TXARD_11      EQU (0xB) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TXARD_12      EQU (0xC) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TXARD_13      EQU (0xD) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TXARD_14      EQU (0xE) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TXARD_15      EQU (0xF) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TXARDS        EQU (0xF <<  8) ;- (HDDRSDRC2) Exit active power down delay to read command in 'Slow Exit' mode.
AT91C_DDRC2_TXARDS_0      EQU (0x0 <<  8) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TXARDS_1      EQU (0x1 <<  8) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TXARDS_2      EQU (0x2 <<  8) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TXARDS_3      EQU (0x3 <<  8) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TXARDS_4      EQU (0x4 <<  8) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TXARDS_5      EQU (0x5 <<  8) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TXARDS_6      EQU (0x6 <<  8) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TXARDS_7      EQU (0x7 <<  8) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TXARDS_8      EQU (0x8 <<  8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TXARDS_9      EQU (0x9 <<  8) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TXARDS_10     EQU (0xA <<  8) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TXARDS_11     EQU (0xB <<  8) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TXARDS_12     EQU (0xC <<  8) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TXARDS_13     EQU (0xD <<  8) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TXARDS_14     EQU (0xE <<  8) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TXARDS_15     EQU (0xF <<  8) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRPA          EQU (0xF << 16) ;- (HDDRSDRC2) Row precharge all delay
AT91C_DDRC2_TRPA_0        EQU (0x0 << 16) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRPA_1        EQU (0x1 << 16) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRPA_2        EQU (0x2 << 16) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRPA_3        EQU (0x3 << 16) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRPA_4        EQU (0x4 << 16) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRPA_5        EQU (0x5 << 16) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRPA_6        EQU (0x6 << 16) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRPA_7        EQU (0x7 << 16) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRPA_8        EQU (0x8 << 16) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRPA_9        EQU (0x9 << 16) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRPA_10       EQU (0xA << 16) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRPA_11       EQU (0xB << 16) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRPA_12       EQU (0xC << 16) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRPA_13       EQU (0xD << 16) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRPA_14       EQU (0xE << 16) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRPA_15       EQU (0xF << 16) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TRTP          EQU (0xF << 24) ;- (HDDRSDRC2) Read to Precharge delay
AT91C_DDRC2_TRTP_0        EQU (0x0 << 24) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TRTP_1        EQU (0x1 << 24) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TRTP_2        EQU (0x2 << 24) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TRTP_3        EQU (0x3 << 24) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TRTP_4        EQU (0x4 << 24) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TRTP_5        EQU (0x5 << 24) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TRTP_6        EQU (0x6 << 24) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TRTP_7        EQU (0x7 << 24) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TRTP_8        EQU (0x8 << 24) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TRTP_9        EQU (0x9 << 24) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TRTP_10       EQU (0xA << 24) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TRTP_11       EQU (0xB << 24) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TRTP_12       EQU (0xC << 24) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TRTP_13       EQU (0xD << 24) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TRTP_14       EQU (0xE << 24) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TRTP_15       EQU (0xF << 24) ;- (HDDRSDRC2) Value : 15
// - -------- HDDRSDRC2_T3PR : (HDDRSDRC2 Offset: 0x18) Timing3 Register -------- 
AT91C_DDRC2_TANPD         EQU (0xF <<  0) ;- (HDDRSDRC2) ODT to power down entry
AT91C_DDRC2_TANPD_0       EQU (0x0) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TANPD_1       EQU (0x1) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TANPD_2       EQU (0x2) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TANPD_3       EQU (0x3) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TANPD_4       EQU (0x4) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TANPD_5       EQU (0x5) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TANPD_6       EQU (0x6) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TANPD_7       EQU (0x7) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TANPD_8       EQU (0x8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TANPD_9       EQU (0x9) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TANPD_10      EQU (0xA) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TANPD_11      EQU (0xB) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TANPD_12      EQU (0xC) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TANPD_13      EQU (0xD) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TANPD_14      EQU (0xE) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TANPD_15      EQU (0xF) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TAXPD         EQU (0xF <<  4) ;- (HDDRSDRC2) ODT power down exit
AT91C_DDRC2_TAXPD_0       EQU (0x0 <<  4) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TAXPD_1       EQU (0x1 <<  4) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TAXPD_2       EQU (0x2 <<  4) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TAXPD_3       EQU (0x3 <<  4) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TAXPD_4       EQU (0x4 <<  4) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TAXPD_5       EQU (0x5 <<  4) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TAXPD_6       EQU (0x6 <<  4) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TAXPD_7       EQU (0x7 <<  4) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TAXPD_8       EQU (0x8 <<  4) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TAXPD_9       EQU (0x9 <<  4) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TAXPD_10      EQU (0xA <<  4) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TAXPD_11      EQU (0xB <<  4) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TAXPD_12      EQU (0xC <<  4) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TAXPD_13      EQU (0xD <<  4) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TAXPD_14      EQU (0xE <<  4) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TAXPD_15      EQU (0xF <<  4) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TAOFPD        EQU (0xF <<  8) ;- (HDDRSDRC2) ODT turn off in power down mode
AT91C_DDRC2_TAOFPD_0      EQU (0x0 <<  8) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TAOFPD_1      EQU (0x1 <<  8) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TAOFPD_2      EQU (0x2 <<  8) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TAOFPD_3      EQU (0x3 <<  8) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TAOFPD_4      EQU (0x4 <<  8) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TAOFPD_5      EQU (0x5 <<  8) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TAOFPD_6      EQU (0x6 <<  8) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TAOFPD_7      EQU (0x7 <<  8) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TAOFPD_8      EQU (0x8 <<  8) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TAOFPD_9      EQU (0x9 <<  8) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TAOFPD_10     EQU (0xA <<  8) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TAOFPD_11     EQU (0xB <<  8) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TAOFPD_12     EQU (0xC <<  8) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TAOFPD_13     EQU (0xD <<  8) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TAOFPD_14     EQU (0xE <<  8) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TAOFPD_15     EQU (0xF <<  8) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TAOFD         EQU (0xF << 12) ;- (HDDRSDRC2) ODT turn off delay
AT91C_DDRC2_TAOFD_0       EQU (0x0 << 12) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TAOFD_1       EQU (0x1 << 12) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TAOFD_2       EQU (0x2 << 12) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TAOFD_3       EQU (0x3 << 12) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TAOFD_4       EQU (0x4 << 12) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TAOFD_5       EQU (0x5 << 12) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TAOFD_6       EQU (0x6 << 12) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TAOFD_7       EQU (0x7 << 12) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TAOFD_8       EQU (0x8 << 12) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TAOFD_9       EQU (0x9 << 12) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TAOFD_10      EQU (0xA << 12) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TAOFD_11      EQU (0xB << 12) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TAOFD_12      EQU (0xC << 12) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TAOFD_13      EQU (0xD << 12) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TAOFD_14      EQU (0xE << 12) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TAOFD_15      EQU (0xF << 12) ;- (HDDRSDRC2) Value : 15
AT91C_DDRC2_TAONPD        EQU (0xF << 16) ;- (HDDRSDRC2) ODT turn on in power down mode
AT91C_DDRC2_TAONPD_0      EQU (0x0 << 16) ;- (HDDRSDRC2) Value :  0
AT91C_DDRC2_TAONPD_1      EQU (0x1 << 16) ;- (HDDRSDRC2) Value :  1
AT91C_DDRC2_TAONPD_2      EQU (0x2 << 16) ;- (HDDRSDRC2) Value :  2
AT91C_DDRC2_TAONPD_3      EQU (0x3 << 16) ;- (HDDRSDRC2) Value :  3
AT91C_DDRC2_TAONPD_4      EQU (0x4 << 16) ;- (HDDRSDRC2) Value :  4
AT91C_DDRC2_TAONPD_5      EQU (0x5 << 16) ;- (HDDRSDRC2) Value :  5
AT91C_DDRC2_TAONPD_6      EQU (0x6 << 16) ;- (HDDRSDRC2) Value :  6
AT91C_DDRC2_TAONPD_7      EQU (0x7 << 16) ;- (HDDRSDRC2) Value :  7
AT91C_DDRC2_TAONPD_8      EQU (0x8 << 16) ;- (HDDRSDRC2) Value :  8
AT91C_DDRC2_TAONPD_9      EQU (0x9 << 16) ;- (HDDRSDRC2) Value :  9
AT91C_DDRC2_TAONPD_10     EQU (0xA << 16) ;- (HDDRSDRC2) Value : 10
AT91C_DDRC2_TAONPD_11     EQU (0xB << 16) ;- (HDDRSDRC2) Value : 11
AT91C_DDRC2_TAONPD_12     EQU (0xC << 16) ;- (HDDRSDRC2) Value : 12
AT91C_DDRC2_TAONPD_13     EQU (0xD << 16) ;- (HDDRSDRC2) Value : 13
AT91C_DDRC2_TAONPD_14     EQU (0xE << 16) ;- (HDDRSDRC2) Value : 14
AT91C_DDRC2_TAONPD_15     EQU (0xF << 16) ;- (HDDRSDRC2) Value : 15
// - -------- HDDRSDRC2_LPR : (HDDRSDRC2 Offset: 0x1c)  -------- 
AT91C_DDRC2_LPCB          EQU (0x3 <<  0) ;- (HDDRSDRC2) Low-power Command Bit
AT91C_DDRC2_LPCB_DISABLED EQU (0x0) ;- (HDDRSDRC2) Low-power Feature is inhibited: no power-down, self refresh and Deep power mode are issued to the SDRAM device.
AT91C_DDRC2_LPCB_SELFREFRESH EQU (0x1) ;- (HDDRSDRC2) The DDRSDRAMC Controller issues a Self Refresh Command to the SDRAM device, the clock(s) is/are de-activated and the CKE signal is set low. The SDRAM device leaves the self refresh mode when accessed and enters it after the access.
AT91C_DDRC2_LPCB_POWERDOWN EQU (0x2) ;- (HDDRSDRC2) The HDDRSDRC2 Controller issues a Power-down Command to the SDRAM device after each access, the CKE signal is set low. The SDRAM device leaves the power-down mode when accessed and enters it after the access.
AT91C_DDRC2_LPCB_DEEP_PWD EQU (0x3) ;- (HDDRSDRC2) The HDDRSDRC2 Controller issues a Deep Power-down Command to the Mobile SDRAM device. This mode is unique to Mobile SDRAM devices
AT91C_DDRC2_CLK_FR        EQU (0x1 <<  2) ;- (HDDRSDRC2) Clock frozen Command Bit
AT91C_DDRC2_CLK_FR_DISABLED EQU (0x0 <<  2) ;- (HDDRSDRC2) Low-power Feature is inhibited: no power-down, self refresh and Deep power mode are issued to the SDRAM device.
AT91C_DDRC2_CLK_FR_SELFREFRESH EQU (0x1 <<  2) ;- (HDDRSDRC2) The DDRSDRAMC Controller issues a Self Refresh Command to the SDRAM device, the clock(s) is/are de-activated and the CKE signal is set low. The SDRAM device leaves the self refresh mode when accessed and enters it after the access.
AT91C_DDRC2_CLK_FR_POWERDOWN EQU (0x2 <<  2) ;- (HDDRSDRC2) The HDDRSDRC2 Controller issues a Power-down Command to the SDRAM device after each access, the CKE signal is set low. The SDRAM device leaves the power-down mode when accessed and enters it after the access.
AT91C_DDRC2_CLK_FR_DEEP_PWD EQU (0x3 <<  2) ;- (HDDRSDRC2) The HDDRSDRC2 Controller issues a Deep Power-down Command to the Mobile SDRAM device. This mode is unique to Mobile SDRAM devices
AT91C_DDRC2_PASR          EQU (0x7 <<  4) ;- (HDDRSDRC2) Partial Array Self Refresh
AT91C_                    EQU (0x0 <<  8) ;- (HDDRSDRC2) 
AT91C_DDRC2_DS            EQU (0x3 << 10) ;- (HDDRSDRC2) Drive strength
AT91C_DDRC2_TIMEOUT       EQU (0x3 << 12) ;- (HDDRSDRC2) low-power mode delay
AT91C_DDRC2_TIMEOUT_0     EQU (0x0 << 12) ;- (HDDRSDRC2) The SDRAM controller activates the SDRAM low-power mode immediately after the end of the last transfer.
AT91C_DDRC2_TIMEOUT_64    EQU (0x1 << 12) ;- (HDDRSDRC2) The SDRAM controller activates the SDRAM low-power mode 64 clock cycles after the end of the last transfer.
AT91C_DDRC2_TIMEOUT_128   EQU (0x2 << 12) ;- (HDDRSDRC2) The SDRAM controller activates the SDRAM low-power mode 128 clock cycles after the end of the last transfer.
AT91C_DDRC2_TIMEOUT_Reserved EQU (0x3 << 12) ;- (HDDRSDRC2) Reserved
AT91C_DDRC2_ADPE          EQU (0x1 << 16) ;- (HDDRSDRC2) Active Power Down Exit time
AT91C_DDRC2_ADPE_FAST     EQU (0x0 << 16) ;- (HDDRSDRC2) Fast Exit
AT91C_DDRC2_ADPE_SLOW     EQU (0x1 << 16) ;- (HDDRSDRC2) Slow Exit
// - -------- HDDRSDRC2_MDR : (HDDRSDRC2 Offset: 0x20) Memory Device Register -------- 
AT91C_DDRC2_MD            EQU (0x7 <<  0) ;- (HDDRSDRC2) memory device
AT91C_DDRC2_MD_SDR_SDRAM  EQU (0x0) ;- (HDDRSDRC2) SDR SDRAM
AT91C_DDRC2_MD_LP_SDR_SDRAM EQU (0x1) ;- (HDDRSDRC2) Low Power SDR SDRAM
AT91C_DDRC2_MD_DDR_SDRAM  EQU (0x2) ;- (HDDRSDRC2) DDR SDRAM
AT91C_DDRC2_MD_LP_DDR_SDRAM EQU (0x3) ;- (HDDRSDRC2) Low Power DDR SDRAM
AT91C_DDRC2_MD_DDR2_SDRAM EQU (0x6) ;- (HDDRSDRC2) DDR2 SDRAM
AT91C_DDRC2_DBW           EQU (0x1 <<  4) ;- (HDDRSDRC2) Data Bus Width
AT91C_DDRC2_DBW_32_BITS   EQU (0x0 <<  4) ;- (HDDRSDRC2) 32 Bits datas bus
AT91C_DDRC2_DBW_16_BITS   EQU (0x1 <<  4) ;- (HDDRSDRC2) 16 Bits datas bus
// - -------- HDDRSDRC2_DLL : (HDDRSDRC2 Offset: 0x24) DLL Information Register -------- 
AT91C_DDRC2_MDINC         EQU (0x1 <<  0) ;- (HDDRSDRC2) DLL Master Delay Increment
AT91C_DDRC2_MDDEC         EQU (0x1 <<  1) ;- (HDDRSDRC2) DLL Master Delay Decrement
AT91C_DDRC2_MDOVF         EQU (0x1 <<  2) ;- (HDDRSDRC2) DLL Master Delay Overflow Flag
AT91C_DDRC2_SDCOVF        EQU (0x1 <<  3) ;- (HDDRSDRC2) DLL Slave Delay Correction Overflow Flag
AT91C_DDRC2_SDCUDF        EQU (0x1 <<  4) ;- (HDDRSDRC2) DLL Slave Delay Correction Underflow Flag
AT91C_DDRC2_SDERF         EQU (0x1 <<  5) ;- (HDDRSDRC2) DLL Slave Delay Correction Error Flag
AT91C_DDRC2_MDVAL         EQU (0xFF <<  8) ;- (HDDRSDRC2) DLL Master Delay Value
AT91C_DDRC2_SDVAL         EQU (0xFF << 16) ;- (HDDRSDRC2) DLL Slave Delay Value
AT91C_DDRC2_SDCVAL        EQU (0xFF << 24) ;- (HDDRSDRC2) DLL Slave Delay Correction Value
// - -------- HDDRSDRC2_HS : (HDDRSDRC2 Offset: 0x2c) High Speed Register -------- 
AT91C_DDRC2_DA            EQU (0x1 <<  0) ;- (HDDRSDRC2) Decode cycle
AT91C_DDRC2_NO_OPT        EQU (0x1 <<  1) ;- (HDDRSDRC2) Disable optimization
AT91C_DDRC2_NO_ANT        EQU (0x1 <<  2) ;- (HDDRSDRC2) Disable Anticipated read
AT91C_DDRC2_RES_DQS       EQU (0x1 <<  3) ;- (HDDRSDRC2) Reset DQS Active
// - -------- HDDRSDRC2_DELAY1 : (HDDRSDRC2 Offset: 0x30) Pad delay1 Register -------- 
AT91C_DDRC2_DELAY         EQU (0xF <<  0) ;- (HDDRSDRC2) Pad delay value
// - -------- HDDRSDRC2_DELAY2 : (HDDRSDRC2 Offset: 0x34) Pad delay2 Register -------- 
// - -------- HDDRSDRC2_DELAY3 : (HDDRSDRC2 Offset: 0x38) Pad delay3 Register -------- 
// - -------- HDDRSDRC2_DELAY4 : (HDDRSDRC2 Offset: 0x3c) Pad delay4 Register -------- 
// - -------- HDDRSDRC2_DELAY5 : (HDDRSDRC2 Offset: 0x40) Pad delay5 Register -------- 
// - -------- HDDRSDRC2_DELAY6 : (HDDRSDRC2 Offset: 0x44) Pad delay6 Register -------- 
// - -------- HDDRSDRC2_DELAY7 : (HDDRSDRC2 Offset: 0x48) Pad delay7 Register -------- 
// - -------- HDDRSDRC2_DELAY8 : (HDDRSDRC2 Offset: 0x4c) Pad delay8 Register -------- 
// - -------- HDDRSDRC2_WPCR : (HDDRSDRC2 Offset: 0xe4) Write Protect Control Register -------- 
AT91C_DDRC2_WPEN          EQU (0x1 <<  0) ;- (HDDRSDRC2) write protect enable
AT91C_DDRC2_WPKEY         EQU (0xFFFFFF <<  8) ;- (HDDRSDRC2) write protect key
// - -------- HDDRSDRC2_WPSR : (HDDRSDRC2 Offset: 0xe8) Write Protect Status Register -------- 
AT91C_DDRC2_WPVS          EQU (0x1 <<  0) ;- (HDDRSDRC2) write protect violation status
AT91C_DDRC2_WPSRC         EQU (0xFFFF <<  8) ;- (HDDRSDRC2) write protect source address

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Static Memory Controller Interface
// - *****************************************************************************
// - -------- SMC_SETUP : (SMC Offset: 0x0) Setup Register for CS x -------- 
AT91C_SMC_NWESETUP        EQU (0x3F <<  0) ;- (SMC) NWE Setup Length
AT91C_SMC_NCSSETUPWR      EQU (0x3F <<  8) ;- (SMC) NCS Setup Length in WRite Access
AT91C_SMC_NRDSETUP        EQU (0x3F << 16) ;- (SMC) NRD Setup Length
AT91C_SMC_NCSSETUPRD      EQU (0x3F << 24) ;- (SMC) NCS Setup Length in ReaD Access
// - -------- SMC_PULSE : (SMC Offset: 0x4) Pulse Register for CS x -------- 
AT91C_SMC_NWEPULSE        EQU (0x7F <<  0) ;- (SMC) NWE Pulse Length
AT91C_SMC_NCSPULSEWR      EQU (0x7F <<  8) ;- (SMC) NCS Pulse Length in WRite Access
AT91C_SMC_NRDPULSE        EQU (0x7F << 16) ;- (SMC) NRD Pulse Length
AT91C_SMC_NCSPULSERD      EQU (0x7F << 24) ;- (SMC) NCS Pulse Length in ReaD Access
// - -------- SMC_CYC : (SMC Offset: 0x8) Cycle Register for CS x -------- 
AT91C_SMC_NWECYCLE        EQU (0x1FF <<  0) ;- (SMC) Total Write Cycle Length
AT91C_SMC_NRDCYCLE        EQU (0x1FF << 16) ;- (SMC) Total Read Cycle Length
// - -------- SMC_CTRL : (SMC Offset: 0xc) Control Register for CS x -------- 
AT91C_SMC_READMODE        EQU (0x1 <<  0) ;- (SMC) Read Mode
AT91C_SMC_WRITEMODE       EQU (0x1 <<  1) ;- (SMC) Write Mode
AT91C_SMC_NWAITM          EQU (0x3 <<  5) ;- (SMC) NWAIT Mode
AT91C_SMC_NWAITM_NWAIT_DISABLE EQU (0x0 <<  5) ;- (SMC) External NWAIT disabled.
AT91C_SMC_NWAITM_NWAIT_ENABLE_FROZEN EQU (0x2 <<  5) ;- (SMC) External NWAIT enabled in frozen mode.
AT91C_SMC_NWAITM_NWAIT_ENABLE_READY EQU (0x3 <<  5) ;- (SMC) External NWAIT enabled in ready mode.
AT91C_SMC_BAT             EQU (0x1 <<  8) ;- (SMC) Byte Access Type
AT91C_SMC_BAT_BYTE_SELECT EQU (0x0 <<  8) ;- (SMC) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.
AT91C_SMC_BAT_BYTE_WRITE  EQU (0x1 <<  8) ;- (SMC) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.
AT91C_SMC_DBW             EQU (0x3 << 12) ;- (SMC) Data Bus Width
AT91C_SMC_DBW_WIDTH_EIGTH_BITS EQU (0x0 << 12) ;- (SMC) 8 bits.
AT91C_SMC_DBW_WIDTH_SIXTEEN_BITS EQU (0x1 << 12) ;- (SMC) 16 bits.
AT91C_SMC_DBW_WIDTH_THIRTY_TWO_BITS EQU (0x2 << 12) ;- (SMC) 32 bits.
AT91C_SMC_TDF             EQU (0xF << 16) ;- (SMC) Data Float Time.
AT91C_SMC_TDFEN           EQU (0x1 << 20) ;- (SMC) TDF Enabled.
AT91C_SMC_PMEN            EQU (0x1 << 24) ;- (SMC) Page Mode Enabled.
AT91C_SMC_PS              EQU (0x3 << 28) ;- (SMC) Page Size
AT91C_SMC_PS_SIZE_FOUR_BYTES EQU (0x0 << 28) ;- (SMC) 4 bytes.
AT91C_SMC_PS_SIZE_EIGHT_BYTES EQU (0x1 << 28) ;- (SMC) 8 bytes.
AT91C_SMC_PS_SIZE_SIXTEEN_BYTES EQU (0x2 << 28) ;- (SMC) 16 bytes.
AT91C_SMC_PS_SIZE_THIRTY_TWO_BYTES EQU (0x3 << 28) ;- (SMC) 32 bytes.
// - -------- SMC_SETUP : (SMC Offset: 0x10) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x14) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x18) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x1c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x20) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x24) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x28) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x2c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x30) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x34) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x38) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x3c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x40) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x44) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x48) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x4c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x50) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x54) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x58) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x5c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x60) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x64) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x68) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x6c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x70) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x74) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x78) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x7c) Control Register for CS x -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR AHB Matrix Interface
// - *****************************************************************************
// - -------- MATRIX_MCFG0 : (MATRIX Offset: 0x0) Master Configuration Register ARM-I -------- 
AT91C_MATRIX_ULBT         EQU (0x7 <<  0) ;- (MATRIX) Undefined Length Burst Type
// - -------- MATRIX_MCFG1 : (MATRIX Offset: 0x4) Master Configuration Register ARM-D -------- 
// - -------- MATRIX_MCFG2 : (MATRIX Offset: 0x8) Master Configuration Register PDC -------- 
// - -------- MATRIX_MCFG3 : (MATRIX Offset: 0xc) Master Configuration Register USB Host OHCI -------- 
// - -------- MATRIX_MCFG4 : (MATRIX Offset: 0x10) Master Configuration Register DMA0 -------- 
// - -------- MATRIX_MCFG5 : (MATRIX Offset: 0x14) Master Configuration Register DMA1 -------- 
// - -------- MATRIX_MCFG6 : (MATRIX Offset: 0x18) Master Configuration Register HISI -------- 
// - -------- MATRIX_MCFG7 : (MATRIX Offset: 0x1c) Master Configuration Register LCD -------- 
// - -------- MATRIX_MCFG8 : (MATRIX Offset: 0x20) Master Configuration Register EMAC -------- 
// - -------- MATRIX_MCFG9 : (MATRIX Offset: 0x24) Master Configuration Register USB Device -------- 
// - -------- MATRIX_MCFG10 : (MATRIX Offset: 0x28) Master Configuration Register USB Host EHCI -------- 
// - -------- MATRIX_SCFG0 : (MATRIX Offset: 0x40) Slave Configuration Register 0 -------- 
AT91C_MATRIX_SLOT_CYCLE   EQU (0xFF <<  0) ;- (MATRIX) Maximum Number of Allowed Cycles for a Burst
AT91C_MATRIX_DEFMSTR_TYPE EQU (0x3 << 16) ;- (MATRIX) Default Master Type
AT91C_MATRIX_DEFMSTR_TYPE_NO_DEFMSTR EQU (0x0 << 16) ;- (MATRIX) No Default Master. At the end of current slave access, if no other master request is pending, the slave is deconnected from all masters. This results in having a one cycle latency for the first transfer of a burst.
AT91C_MATRIX_DEFMSTR_TYPE_LAST_DEFMSTR EQU (0x1 << 16) ;- (MATRIX) Last Default Master. At the end of current slave access, if no other master request is pending, the slave stay connected with the last master having accessed it. This results in not having the one cycle latency when the last master re-trying access on the slave.
AT91C_MATRIX_DEFMSTR_TYPE_FIXED_DEFMSTR EQU (0x2 << 16) ;- (MATRIX) Fixed Default Master. At the end of current slave access, if no other master request is pending, the slave connects with fixed which number is in FIXED_DEFMSTR field. This results in not having the one cycle latency when the fixed master re-trying access on the slave.
AT91C_MATRIX_FIXED_DEFMSTR0 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_UHPHS_OHCI EQU (0x3 << 18) ;- (MATRIX) USB Host OHCI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_DMA0 EQU (0x4 << 18) ;- (MATRIX) DMA0 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_DMA1 EQU (0x5 << 18) ;- (MATRIX) DMA1 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_ISI EQU (0x6 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_EMAC EQU (0x8 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_UDPHS EQU (0x9 << 18) ;- (MATRIX) USB Device Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_UHPHS_EHCI EQU (0xA << 18) ;- (MATRIX) USB Host EHCI Master is Default Master
AT91C_MATRIX_ARBT         EQU (0x3 << 24) ;- (MATRIX) Arbitration Type
// - -------- MATRIX_SCFG1 : (MATRIX Offset: 0x44) Slave Configuration Register 1 -------- 
AT91C_MATRIX_FIXED_DEFMSTR1 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_ISI EQU (0x6 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_UDPHS EQU (0x9 << 18) ;- (MATRIX) USB Device Master is Default Master
// - -------- MATRIX_SCFG2 : (MATRIX Offset: 0x48) Slave Configuration Register 2 -------- 
AT91C_MATRIX_FIXED_DEFMSTR2 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR2_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR2_VDEC EQU (0xB << 18) ;- (MATRIX) Video Decoder Master is Default Master
// - -------- MATRIX_SCFG3 : (MATRIX Offset: 0x4c) Slave Configuration Register 3 -------- 
AT91C_MATRIX_FIXED_DEFMSTR3 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_LCD EQU (0x7 << 18) ;- (MATRIX) LCD Master is Default Master
// - -------- MATRIX_SCFG4 : (MATRIX Offset: 0x50) Slave Configuration Register 4 -------- 
AT91C_MATRIX_FIXED_DEFMSTR4 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_UHPHS_OHCI EQU (0x3 << 18) ;- (MATRIX) USB Host OHCI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_DMA0 EQU (0x4 << 18) ;- (MATRIX) DMA0 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_DMA1 EQU (0x5 << 18) ;- (MATRIX) DMA1 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_ISI EQU (0x6 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_EMAC EQU (0x8 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_UDPHS EQU (0x9 << 18) ;- (MATRIX) USB Device Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_UHPHS_EHCI EQU (0xA << 18) ;- (MATRIX) USB Host EHCI Master is Default Master
// - -------- MATRIX_SCFG5 : (MATRIX Offset: 0x54) Slave Configuration Register 5 -------- 
AT91C_MATRIX_FIXED_DEFMSTR5 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_UHPHS_OHCI EQU (0x3 << 18) ;- (MATRIX) USB Host OHCI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_DMA0 EQU (0x4 << 18) ;- (MATRIX) DMA0 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_DMA1 EQU (0x5 << 18) ;- (MATRIX) DMA1 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_ISI EQU (0x6 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_EMAC EQU (0x8 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_UDPHS EQU (0x9 << 18) ;- (MATRIX) USB Device Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_UHPHS_EHCI EQU (0xA << 18) ;- (MATRIX) USB Host EHCI Master is Default Master
// - -------- MATRIX_SCFG6 : (MATRIX Offset: 0x58) Slave Configuration Register 6 -------- 
AT91C_MATRIX_FIXED_DEFMSTR6 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_UHPHS_OHCI EQU (0x3 << 18) ;- (MATRIX) USB Host OHCI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_DMA0 EQU (0x4 << 18) ;- (MATRIX) DMA0 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_DMA1 EQU (0x5 << 18) ;- (MATRIX) DMA1 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_ISI EQU (0x6 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_LCD EQU (0x7 << 18) ;- (MATRIX) LCD Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_EMAC EQU (0x8 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_UDPHS EQU (0x9 << 18) ;- (MATRIX) USB Device Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_UHPHS_EHCI EQU (0xA << 18) ;- (MATRIX) USB Host EHCI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_VDEC EQU (0xB << 18) ;- (MATRIX) Video Decoder Master is Default Master
// - -------- MATRIX_SCFG7 : (MATRIX Offset: 0x5c) Slave Configuration Register 7 -------- 
AT91C_MATRIX_FIXED_DEFMSTR7 EQU (0xF << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_DMA0 EQU (0x4 << 18) ;- (MATRIX) DMA0 Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_DMA1 EQU (0x5 << 18) ;- (MATRIX) DMA1 Master is Default Master
// - -------- MATRIX_PRAS0 : (MATRIX Offset: 0x80) PRAS0 Register -------- 
AT91C_MATRIX_M0PR         EQU (0x3 <<  0) ;- (MATRIX) ARM926EJ-S Instruction priority
AT91C_MATRIX_M1PR         EQU (0x3 <<  4) ;- (MATRIX) ARM926EJ-S Data priority
AT91C_MATRIX_M2PR         EQU (0x3 <<  8) ;- (MATRIX) PDC priority
AT91C_MATRIX_M3PR         EQU (0x3 << 12) ;- (MATRIX) USB Host OHCI priority
AT91C_MATRIX_M4PR         EQU (0x3 << 16) ;- (MATRIX) DMA0 priority
AT91C_MATRIX_M5PR         EQU (0x3 << 20) ;- (MATRIX) DMA1 priority
AT91C_MATRIX_M6PR         EQU (0x3 << 24) ;- (MATRIX) ISI priority
AT91C_MATRIX_M7PR         EQU (0x3 << 28) ;- (MATRIX) LCD priority
// - -------- MATRIX_PRBS0 : (MATRIX Offset: 0x84) PRBS0 Register -------- 
AT91C_MATRIX_M8PR         EQU (0x3 <<  0) ;- (MATRIX) EMAC priority
AT91C_MATRIX_M9PR         EQU (0x3 <<  4) ;- (MATRIX) USB Device priority
AT91C_MATRIX_M10PR        EQU (0x3 <<  8) ;- (MATRIX) USB Host EHCI priority
AT91C_MATRIX_M11PR        EQU (0x3 << 12) ;- (MATRIX) Video Decoder priority
// - -------- MATRIX_PRAS1 : (MATRIX Offset: 0x88) PRAS1 Register -------- 
// - -------- MATRIX_PRBS1 : (MATRIX Offset: 0x8c) PRBS1 Register -------- 
// - -------- MATRIX_PRAS2 : (MATRIX Offset: 0x90) PRAS2 Register -------- 
// - -------- MATRIX_PRBS2 : (MATRIX Offset: 0x94) PRBS2 Register -------- 
// - -------- MATRIX_PRAS3 : (MATRIX Offset: 0x98) PRAS3 Register -------- 
// - -------- MATRIX_PRBS3 : (MATRIX Offset: 0x9c) PRBS3 Register -------- 
// - -------- MATRIX_PRAS4 : (MATRIX Offset: 0xa0) PRAS4 Register -------- 
// - -------- MATRIX_PRBS4 : (MATRIX Offset: 0xa4) PRBS4 Register -------- 
// - -------- MATRIX_PRAS5 : (MATRIX Offset: 0xa8) PRAS5 Register -------- 
// - -------- MATRIX_PRBS5 : (MATRIX Offset: 0xac) PRBS5 Register -------- 
// - -------- MATRIX_PRAS6 : (MATRIX Offset: 0xb0) PRAS6 Register -------- 
// - -------- MATRIX_PRBS6 : (MATRIX Offset: 0xb4) PRBS6 Register -------- 
// - -------- MATRIX_PRAS7 : (MATRIX Offset: 0xb8) PRAS7 Register -------- 
// - -------- MATRIX_PRBS7 : (MATRIX Offset: 0xbc) PRBS7 Register -------- 
// - -------- MATRIX_MRCR : (MATRIX Offset: 0x100) MRCR Register -------- 
AT91C_MATRIX_RCA926I      EQU (0x1 <<  0) ;- (MATRIX) Remap Command Bit for ARM926EJ-S Instruction
AT91C_MATRIX_RCA926D      EQU (0x1 <<  1) ;- (MATRIX) Remap Command Bit for ARM926EJ-S Data
AT91C_MATRIX_RCB2         EQU (0x1 <<  2) ;- (MATRIX) Remap Command Bit for PDC
AT91C_MATRIX_RCB3         EQU (0x1 <<  3) ;- (MATRIX) Remap Command Bit for USB Host OHCI
AT91C_MATRIX_RCB4         EQU (0x1 <<  4) ;- (MATRIX) Remap Command Bit for DMA0
AT91C_MATRIX_RCB5         EQU (0x1 <<  5) ;- (MATRIX) Remap Command Bit for DMA1
AT91C_MATRIX_RCB6         EQU (0x1 <<  6) ;- (MATRIX) Remap Command Bit for ISI
AT91C_MATRIX_RCB7         EQU (0x1 <<  7) ;- (MATRIX) Remap Command Bit for LCD
AT91C_MATRIX_RCB8         EQU (0x1 <<  8) ;- (MATRIX) Remap Command Bit for EMAC
AT91C_MATRIX_RCB9         EQU (0x1 <<  9) ;- (MATRIX) Remap Command Bit for USB Device
AT91C_MATRIX_RCB10        EQU (0x1 << 10) ;- (MATRIX) Remap Command Bit for USB Host EHCI
AT91C_MATRIX_RCB11        EQU (0x1 << 11) ;- (MATRIX) Remap Command Bit for Video Decoder
// - -------- HMATRIX_CCFG_TCMR : (MATRIX Offset: 0x110) TCMR Register -------- 
AT91C_ITCM_SIZE           EQU (0xF <<  0) ;- (MATRIX) Size of ITCM enabled memory block
AT91C_ITCM_SIZE_0KB       EQU (0x0) ;- (MATRIX) 0 KB (No ITCM Memory)
AT91C_ITCM_SIZE_32KB      EQU (0x6) ;- (MATRIX) 32 KB
AT91C_DTCM_SIZE           EQU (0xF <<  4) ;- (MATRIX) Size of DTCM enabled memory block
AT91C_DTCM_SIZE_0KB       EQU (0x0 <<  4) ;- (MATRIX) 0 KB (No DTCM Memory)
AT91C_DTCM_SIZE_32KB      EQU (0x6 <<  4) ;- (MATRIX) 32 KB
AT91C_DTCM_SIZE_64KB      EQU (0x7 <<  4) ;- (MATRIX) 64 KB
AT91C_TCM_NWS             EQU (0x1 << 11) ;- (MATRIX) TCM Wait State
AT91C_TCM_NWS_NO_WS       EQU (0x0 << 11) ;- (MATRIX) NO WAIT STATE : 0 WS
AT91C_TCM_NWS_ONE_WS      EQU (0x1 << 11) ;- (MATRIX) 1 WS activated (only for RATIO 3:1 or 4:1
// - -------- HMATRIX_CCFG_DDRMPR : (MATRIX Offset: 0x118) DDR Multi-Port Register -------- 
AT91C_DDRMP_DIS           EQU (0x1 <<  0) ;- (MATRIX) DDR Multi-Port Disable
// - -------- HMATRIX_CCFG_EBICSA : (MATRIX Offset: 0x128) CCFG_EBICSA Register -------- 
AT91C_EBI_CS1A            EQU (0x1 <<  1) ;- (MATRIX) EBI Chip Select 1 Assignment
AT91C_EBI_CS1A_SMC        EQU (0x0 <<  1) ;- (MATRIX) Chip Select 1 is assigned to the Static Memory Controller.
AT91C_EBI_CS1A_SDRAMC     EQU (0x1 <<  1) ;- (MATRIX) Chip Select 1 is assigned to the SDRAM Controller.
AT91C_EBI_CS3A            EQU (0x1 <<  3) ;- (MATRIX) EBI Chip Select 3 Assignment
AT91C_EBI_CS3A_SMC        EQU (0x0 <<  3) ;- (MATRIX) Chip Select 3 is only assigned to the Static Memory Controller and NCS3 behaves as defined by the SMC.
AT91C_EBI_CS3A_SM         EQU (0x1 <<  3) ;- (MATRIX) Chip Select 3 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
AT91C_EBI_CS4A            EQU (0x1 <<  4) ;- (MATRIX) EBI Chip Select 4 Assignment
AT91C_EBI_CS4A_SMC        EQU (0x0 <<  4) ;- (MATRIX) Chip Select 4 is only assigned to the Static Memory Controller and NCS4 behaves as defined by the SMC.
AT91C_EBI_CS4A_CF         EQU (0x1 <<  4) ;- (MATRIX) Chip Select 4 is assigned to the Static Memory Controller and the CompactFlash Logic (first slot) is activated.
AT91C_EBI_CS5A            EQU (0x1 <<  5) ;- (MATRIX) EBI Chip Select 5 Assignment
AT91C_EBI_CS5A_SMC        EQU (0x0 <<  5) ;- (MATRIX) Chip Select 5 is only assigned to the Static Memory Controller and NCS5 behaves as defined by the SMC
AT91C_EBI_CS5A_CF         EQU (0x1 <<  5) ;- (MATRIX) Chip Select 5 is assigned to the Static Memory Controller and the CompactFlash Logic (second slot) is activated.
AT91C_EBI_DBPUC           EQU (0x1 <<  8) ;- (MATRIX) EBI Data Bus Pull-Up Configuration
AT91C_EBI_DRIVE           EQU (0x3 << 16) ;- (MATRIX) EBI I/O Drive Configuration
AT91C_EBI_DRIVE_18RD      EQU (0x0 << 16) ;- (MATRIX) optimized for 1.8V powered memories with Reduced Drive
AT91C_EBI_DRIVE_33RD      EQU (0x1 << 16) ;- (MATRIX) optimized for 3.3V powered memories with Reduced Drive
AT91C_EBI_DRIVE_18ND      EQU (0x2 << 16) ;- (MATRIX) optimized for 1.8V powered memories with Normal Drive
AT91C_EBI_DRIVE_33ND      EQU (0x3 << 16) ;- (MATRIX) optimized for 3.3V powered memories with Normal Drive
AT91C_DDR_DRIVE           EQU (0x1 << 18) ;- (MATRIX) DDR2 dedicated port I/O slew rate selection
// - -------- MATRIX_WRPROTEN : (MATRIX Offset: 0x1e4) Write Protection Control Register -------- 
AT91C_MATRIX_WRPROT       EQU (0x1 <<  0) ;- (MATRIX) Enable/Disable Write Protection of HMATRIX2 configuration registers (requires key)
AT91C_MATRIX_WRPROT_DISABLE EQU (0x0) ;- (MATRIX) Disable Write Protection of HMATRIX2 configuration registers
AT91C_MATRIX_WRPROT_ENABLE EQU (0x1) ;- (MATRIX) Enable  Write Protection of HMATRIX2 configuration registers
// - -------- MATRIX_WRPROTST : (MATRIX Offset: 0x1e8) Write Protection Status Register -------- 
AT91C_MATRIX_WRPROT_VIOLATION EQU (0x1 <<  0) ;- (MATRIX) Violation of Write Protection of HMATRIX2 configuration registers
AT91C_MATRIX_WRPROT_VIOLATION_UNDETECTED EQU (0x0) ;- (MATRIX) No violation of Write Protection of HMATRIX2 configuration registers
AT91C_MATRIX_WRPROT_VIOLATION_DETECTED EQU (0x1) ;- (MATRIX) Violation of  Write Protection of HMATRIX2 configuration registers
AT91C_MATRIX_WRPROT_VIOLATION_OFFSET EQU (0x1FF <<  8) ;- (MATRIX) Offset where violation of Write Protection of HMATRIX2 configuration registers is detected

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Advanced Interrupt Controller
// - *****************************************************************************
// - -------- AIC_SMR : (AIC Offset: 0x0) Control Register -------- 
AT91C_AIC_PRIOR           EQU (0x7 <<  0) ;- (AIC) Priority Level
AT91C_AIC_PRIOR_LOWEST    EQU (0x0) ;- (AIC) Lowest priority level
AT91C_AIC_PRIOR_HIGHEST   EQU (0x7) ;- (AIC) Highest priority level
AT91C_AIC_SRCTYPE         EQU (0x3 <<  5) ;- (AIC) Interrupt Source Type
AT91C_AIC_SRCTYPE_INT_LEVEL_SENSITIVE EQU (0x0 <<  5) ;- (AIC) Internal Sources Code Label Level Sensitive
AT91C_AIC_SRCTYPE_INT_EDGE_TRIGGERED EQU (0x1 <<  5) ;- (AIC) Internal Sources Code Label Edge triggered
AT91C_AIC_SRCTYPE_EXT_HIGH_LEVEL EQU (0x2 <<  5) ;- (AIC) External Sources Code Label High-level Sensitive
AT91C_AIC_SRCTYPE_EXT_POSITIVE_EDGE EQU (0x3 <<  5) ;- (AIC) External Sources Code Label Positive Edge triggered
// - -------- AIC_CISR : (AIC Offset: 0x114) AIC Core Interrupt Status Register -------- 
AT91C_AIC_NFIQ            EQU (0x1 <<  0) ;- (AIC) NFIQ Status
AT91C_AIC_NIRQ            EQU (0x1 <<  1) ;- (AIC) NIRQ Status
// - -------- AIC_DCR : (AIC Offset: 0x138) AIC Debug Control Register (Protect) -------- 
AT91C_AIC_DCR_PROT        EQU (0x1 <<  0) ;- (AIC) Protection Mode
AT91C_AIC_DCR_GMSK        EQU (0x1 <<  1) ;- (AIC) General Mask

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Peripheral DMA Controller
// - *****************************************************************************
// - -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- 
AT91C_PDC_RXTEN           EQU (0x1 <<  0) ;- (PDC) Receiver Transfer Enable
AT91C_PDC_RXTDIS          EQU (0x1 <<  1) ;- (PDC) Receiver Transfer Disable
AT91C_PDC_TXTEN           EQU (0x1 <<  8) ;- (PDC) Transmitter Transfer Enable
AT91C_PDC_TXTDIS          EQU (0x1 <<  9) ;- (PDC) Transmitter Transfer Disable
// - -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Debug Unit
// - *****************************************************************************
// - -------- DBGU_CR : (DBGU Offset: 0x0) Debug Unit Control Register -------- 
AT91C_US_RSTRX            EQU (0x1 <<  2) ;- (DBGU) Reset Receiver
AT91C_US_RSTTX            EQU (0x1 <<  3) ;- (DBGU) Reset Transmitter
AT91C_US_RXEN             EQU (0x1 <<  4) ;- (DBGU) Receiver Enable
AT91C_US_RXDIS            EQU (0x1 <<  5) ;- (DBGU) Receiver Disable
AT91C_US_TXEN             EQU (0x1 <<  6) ;- (DBGU) Transmitter Enable
AT91C_US_TXDIS            EQU (0x1 <<  7) ;- (DBGU) Transmitter Disable
AT91C_US_RSTSTA           EQU (0x1 <<  8) ;- (DBGU) Reset Status Bits
// - -------- DBGU_MR : (DBGU Offset: 0x4) Debug Unit Mode Register -------- 
AT91C_US_PAR              EQU (0x7 <<  9) ;- (DBGU) Parity type
AT91C_US_PAR_EVEN         EQU (0x0 <<  9) ;- (DBGU) Even Parity
AT91C_US_PAR_ODD          EQU (0x1 <<  9) ;- (DBGU) Odd Parity
AT91C_US_PAR_SPACE        EQU (0x2 <<  9) ;- (DBGU) Parity forced to 0 (Space)
AT91C_US_PAR_MARK         EQU (0x3 <<  9) ;- (DBGU) Parity forced to 1 (Mark)
AT91C_US_PAR_NONE         EQU (0x4 <<  9) ;- (DBGU) No Parity
AT91C_US_PAR_MULTI_DROP   EQU (0x6 <<  9) ;- (DBGU) Multi-drop mode
AT91C_US_CHMODE           EQU (0x3 << 14) ;- (DBGU) Channel Mode
AT91C_US_CHMODE_NORMAL    EQU (0x0 << 14) ;- (DBGU) Normal Mode: The USART channel operates as an RX/TX USART.
AT91C_US_CHMODE_AUTO      EQU (0x1 << 14) ;- (DBGU) Automatic Echo: Receiver Data Input is connected to the TXD pin.
AT91C_US_CHMODE_LOCAL     EQU (0x2 << 14) ;- (DBGU) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
AT91C_US_CHMODE_REMOTE    EQU (0x3 << 14) ;- (DBGU) Remote Loopback: RXD pin is internally connected to TXD pin.
// - -------- DBGU_IER : (DBGU Offset: 0x8) Debug Unit Interrupt Enable Register -------- 
AT91C_US_RXRDY            EQU (0x1 <<  0) ;- (DBGU) RXRDY Interrupt
AT91C_US_TXRDY            EQU (0x1 <<  1) ;- (DBGU) TXRDY Interrupt
AT91C_US_ENDRX            EQU (0x1 <<  3) ;- (DBGU) End of Receive Transfer Interrupt
AT91C_US_ENDTX            EQU (0x1 <<  4) ;- (DBGU) End of Transmit Interrupt
AT91C_US_OVRE             EQU (0x1 <<  5) ;- (DBGU) Overrun Interrupt
AT91C_US_FRAME            EQU (0x1 <<  6) ;- (DBGU) Framing Error Interrupt
AT91C_US_PARE             EQU (0x1 <<  7) ;- (DBGU) Parity Error Interrupt
AT91C_US_TXEMPTY          EQU (0x1 <<  9) ;- (DBGU) TXEMPTY Interrupt
AT91C_US_TXBUFE           EQU (0x1 << 11) ;- (DBGU) TXBUFE Interrupt
AT91C_US_RXBUFF           EQU (0x1 << 12) ;- (DBGU) RXBUFF Interrupt
AT91C_US_COMM_TX          EQU (0x1 << 30) ;- (DBGU) COMM_TX Interrupt
AT91C_US_COMM_RX          EQU (0x1 << 31) ;- (DBGU) COMM_RX Interrupt
// - -------- DBGU_IDR : (DBGU Offset: 0xc) Debug Unit Interrupt Disable Register -------- 
// - -------- DBGU_IMR : (DBGU Offset: 0x10) Debug Unit Interrupt Mask Register -------- 
// - -------- DBGU_CSR : (DBGU Offset: 0x14) Debug Unit Channel Status Register -------- 
// - -------- DBGU_FNTR : (DBGU Offset: 0x48) Debug Unit FORCE_NTRST Register -------- 
AT91C_US_FORCE_NTRST      EQU (0x1 <<  0) ;- (DBGU) Force NTRST in JTAG

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler
// - *****************************************************************************

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Power Management Controller V610
// - *****************************************************************************
// - -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- 
AT91C_PMC_DDR             EQU (0x1 <<  2) ;- (PMC) DDR controller Clock2x
AT91C_PMC_UHP             EQU (0x1 <<  6) ;- (PMC) USB Host Port Clock
AT91C_PMC_UDP             EQU (0x1 <<  7) ;- (PMC) USB Device Port Clock
AT91C_PMC_PCK0            EQU (0x1 <<  8) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK1            EQU (0x1 <<  9) ;- (PMC) Programmable Clock Output
// - -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- 
AT91C_PMC_PCK             EQU (0x1 <<  0) ;- (PMC) Processor Clock
// - -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- 
// - -------- CKGR_UCKR : (PMC Offset: 0x1c) UTMI Clock Configuration Register -------- 
AT91C_CKGR_UPLLEN         EQU (0x1 << 16) ;- (PMC) UTMI PLL Enable
AT91C_CKGR_UPLLEN_DISABLED EQU (0x0 << 16) ;- (PMC) The UTMI PLL is disabled
AT91C_CKGR_UPLLEN_ENABLED EQU (0x1 << 16) ;- (PMC) The UTMI PLL is enabled
AT91C_CKGR_PLLCOUNT       EQU (0xF << 20) ;- (PMC) UTMI Oscillator Start-up Time
AT91C_CKGR_BIASEN         EQU (0x1 << 24) ;- (PMC) UTMI BIAS Enable
AT91C_CKGR_BIASEN_DISABLED EQU (0x0 << 24) ;- (PMC) The UTMI BIAS is disabled
AT91C_CKGR_BIASEN_ENABLED EQU (0x1 << 24) ;- (PMC) The UTMI BIAS is enabled
AT91C_CKGR_BIASCOUNT      EQU (0xF << 28) ;- (PMC) UTMI BIAS Start-up Time
// - -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- 
AT91C_CKGR_MOSCEN         EQU (0x1 <<  0) ;- (PMC) Main Oscillator Enable
AT91C_CKGR_OSCBYPASS      EQU (0x1 <<  1) ;- (PMC) Main Oscillator Bypass
AT91C_CKGR_OSCOUNT        EQU (0xFF <<  8) ;- (PMC) Main Oscillator Start-up Time
// - -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- 
AT91C_CKGR_MAINF          EQU (0xFFFF <<  0) ;- (PMC) Main Clock Frequency
AT91C_CKGR_MAINRDY        EQU (0x1 << 16) ;- (PMC) Main Clock Ready
// - -------- CKGR_PLLAR : (PMC Offset: 0x28) PLL A Register -------- 
AT91C_CKGR_DIVA           EQU (0xFF <<  0) ;- (PMC) Divider A Selected
AT91C_CKGR_DIVA_0         EQU (0x0) ;- (PMC) Divider A output is 0
AT91C_CKGR_DIVA_BYPASS    EQU (0x1) ;- (PMC) Divider A is bypassed
AT91C_CKGR_PLLACOUNT      EQU (0x3F <<  8) ;- (PMC) PLL A Counter
AT91C_CKGR_OUTA           EQU (0x3 << 14) ;- (PMC) PLL A Output Frequency Range
AT91C_CKGR_OUTA_0         EQU (0x0 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_1         EQU (0x1 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_2         EQU (0x2 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_3         EQU (0x3 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_MULA           EQU (0xFF << 16) ;- (PMC) PLL A Multiplier
AT91C_CKGR_SRCA           EQU (0x1 << 29) ;- (PMC) 
// - -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- 
AT91C_PMC_CSS             EQU (0x3 <<  0) ;- (PMC) Programmable Clock Selection
AT91C_PMC_CSS_SLOW_CLK    EQU (0x0) ;- (PMC) Slow Clock is selected
AT91C_PMC_CSS_MAIN_CLK    EQU (0x1) ;- (PMC) Main Clock is selected
AT91C_PMC_CSS_PLLA_CLK    EQU (0x2) ;- (PMC) Clock from PLL A is selected
AT91C_PMC_CSS_UPLL_CLK    EQU (0x3) ;- (PMC) Clock from UTMI PLL is selected
AT91C_PMC_PRES            EQU (0x7 <<  2) ;- (PMC) Programmable Clock Prescaler
AT91C_PMC_PRES_CLK        EQU (0x0 <<  2) ;- (PMC) Selected clock
AT91C_PMC_PRES_CLK_2      EQU (0x1 <<  2) ;- (PMC) Selected clock divided by 2
AT91C_PMC_PRES_CLK_4      EQU (0x2 <<  2) ;- (PMC) Selected clock divided by 4
AT91C_PMC_PRES_CLK_8      EQU (0x3 <<  2) ;- (PMC) Selected clock divided by 8
AT91C_PMC_PRES_CLK_16     EQU (0x4 <<  2) ;- (PMC) Selected clock divided by 16
AT91C_PMC_PRES_CLK_32     EQU (0x5 <<  2) ;- (PMC) Selected clock divided by 32
AT91C_PMC_PRES_CLK_64     EQU (0x6 <<  2) ;- (PMC) Selected clock divided by 64
AT91C_PMC_MDIV            EQU (0x3 <<  8) ;- (PMC) Master Clock Division
AT91C_PMC_MDIV_1          EQU (0x0 <<  8) ;- (PMC) Processor clock = Master Clock ; DDR Clock = Master Clock
AT91C_PMC_MDIV_2          EQU (0x1 <<  8) ;- (PMC) Processor clock = 2 * Master Clock ; DDR Clock = 2 * Master Clock
AT91C_PMC_MDIV_4          EQU (0x2 <<  8) ;- (PMC) Processor clock = 4 * Master Clock ; DDR Clock = 2 * Master Clock
AT91C_PMC_MDIV_3          EQU (0x3 <<  8) ;- (PMC) Processor clock = 3 * Master Clock ; DDR Clock = 2 * Master Clock
AT91C_PMC_PLLADIV2        EQU (0x1 << 12) ;- (PMC) PLLA divisor by 2
AT91C_PMC_PLLADIV2_1      EQU (0x0 << 12) ;- (PMC) PLLA clock frequency is divided by 1
AT91C_PMC_PLLADIV2_2      EQU (0x1 << 12) ;- (PMC) PLLA clock frequency is divided by 2
// - -------- PMC_USB : (PMC Offset: 0x38) USB Clock Register -------- 
AT91C_PMC_USBS            EQU (0x1 <<  0) ;- (PMC) USBS
AT91C_PMC_USBS_USB_PLLA   EQU (0x0) ;- (PMC) USB Clock Input is PLLA
AT91C_PMC_USBS_USB_UPLL   EQU (0x1) ;- (PMC) USB Clock Input is UPLL
AT91C_PMC_USBDIV          EQU (0xF <<  8) ;- (PMC) USBDIV
AT91C_PMC_USBDIV_1        EQU (0x0 <<  8) ;- (PMC) USB Clock divided by 1
AT91C_PMC_USBDIV_2        EQU (0x1 <<  8) ;- (PMC) USB Clock divided by 2
AT91C_PMC_USBDIV_3        EQU (0x2 <<  8) ;- (PMC) USB Clock divided by 3
AT91C_PMC_USBDIV_4        EQU (0x3 <<  8) ;- (PMC) USB Clock divided by 4
AT91C_PMC_USBDIV_5        EQU (0x4 <<  8) ;- (PMC) USB Clock divided by 5
AT91C_PMC_USBDIV_6        EQU (0x5 <<  8) ;- (PMC) USB Clock divided by 6
AT91C_PMC_USBDIV_7        EQU (0x6 <<  8) ;- (PMC) USB Clock divided by 7
AT91C_PMC_USBDIV_8        EQU (0x7 <<  8) ;- (PMC) USB Clock divided by 8
AT91C_PMC_USBDIV_9        EQU (0x8 <<  8) ;- (PMC) USB Clock divided by 9
AT91C_PMC_USBDIV_10       EQU (0x9 <<  8) ;- (PMC) USB Clock divided by 10
AT91C_PMC_USBDIV_11       EQU (0xA <<  8) ;- (PMC) USB Clock divided by 11
AT91C_PMC_USBDIV_12       EQU (0xB <<  8) ;- (PMC) USB Clock divided by 12
AT91C_PMC_USBDIV_13       EQU (0xC <<  8) ;- (PMC) USB Clock divided by 13
AT91C_PMC_USBDIV_14       EQU (0xD <<  8) ;- (PMC) USB Clock divided by 14
AT91C_PMC_USBDIV_15       EQU (0xE <<  8) ;- (PMC) USB Clock divided by 15
AT91C_PMC_USBDIV_16       EQU (0xF <<  8) ;- (PMC) USB Clock divided by 16
// - -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock 0 Register -------- 
AT91C_PMC_SLCKMCK         EQU (0x1 <<  8) ;- (PMC) Programmable Clock Prescaler
AT91C_PMC_SLCKMCK_SLCK    EQU (0x0 <<  8) ;- (PMC) Slow Clock selected
AT91C_PMC_SLCKMCK_MCK     EQU (0x1 <<  8) ;- (PMC) Master Clock selected
// - -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- 
AT91C_PMC_MOSCS           EQU (0x1 <<  0) ;- (PMC) MOSC Status/Enable/Disable/Mask
AT91C_PMC_LOCKA           EQU (0x1 <<  1) ;- (PMC) PLL A Status/Enable/Disable/Mask
AT91C_PMC_MCKRDY          EQU (0x1 <<  3) ;- (PMC) Master Clock Status/Enable/Disable/Mask
AT91C_PMC_LOCKU           EQU (0x1 <<  6) ;- (PMC) PLL UTMI Status/Enable/Disable/Mask
AT91C_PMC_PCK0RDY         EQU (0x1 <<  8) ;- (PMC) PCK0_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK1RDY         EQU (0x1 <<  9) ;- (PMC) PCK1_RDY Status/Enable/Disable/Mask
// - -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- 
// - -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- 
// - -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- 
// - -------- PMC_PLLICPR : (PMC Offset: 0x80) PLL Charge Pump Current Register -------- 
AT91C_PMC_ICPPLLA         EQU (0xF <<  0) ;- (PMC) PLLA charge pump current setting
AT91C_PMC_ICPPLLA_0       EQU (0x0) ;- (PMC) 595-800 MHz
AT91C_PMC_ICPPLLA_1       EQU (0x1) ;- (PMC) 395-600 MHz
AT91C_PMC_REALLOCK        EQU (0x1 <<  7) ;- (PMC) PLLs use real lock signals when 1
AT91C_PMC_IPLLA           EQU (0xF <<  8) ;- (PMC) PLLA special setting
AT91C_PMC_IPLLA_0         EQU (0x0 <<  8) ;- (PMC) Internal LFT
AT91C_PMC_IPLLA_1         EQU (0x1 <<  8) ;- (PMC) External LFT
// - -------- PMC_FEATURES : (PMC Offset: 0xf8)   -------- 
AT91C_PMC_CFGAHBCLK       EQU (0x1 <<  0) ;- (PMC) 
AT91C_PMC_CFGAHBCLK_0     EQU (0x0) ;- (PMC) 
AT91C_PMC_CFGAHBCLK_1     EQU (0x1) ;- (PMC) 
AT91C_PMC_HCLKEN          EQU (0x1 <<  1) ;- (PMC) 
AT91C_PMC_HCLKEN_0        EQU (0x0 <<  1) ;- (PMC) 
AT91C_PMC_HCLKEN_1        EQU (0x1 <<  1) ;- (PMC) 
AT91C_PMC_PERMCLK         EQU (0x1 <<  2) ;- (PMC) 
AT91C_PMC_PERMCLK_0       EQU (0x0 <<  2) ;- (PMC) 
AT91C_PMC_PERMCLK_1       EQU (0x1 <<  2) ;- (PMC) 
AT91C_PMC_CORE2           EQU (0x1 <<  3) ;- (PMC) 
AT91C_PMC_CORE2_0         EQU (0x0 <<  3) ;- (PMC) 
AT91C_PMC_CORE2_1         EQU (0x1 <<  3) ;- (PMC) 
AT91C_PMC_USBDEVCK        EQU (0x1 <<  4) ;- (PMC) 
AT91C_PMC_USBDEVCK_0      EQU (0x0 <<  4) ;- (PMC) 
AT91C_PMC_USBDEVCK_1      EQU (0x1 <<  4) ;- (PMC) 
AT91C_PMC_USBHOSTCK       EQU (0x1 <<  5) ;- (PMC) 
AT91C_PMC_USBHOSTCK_0     EQU (0x0 <<  5) ;- (PMC) 
AT91C_PMC_USBHOSTCK_1     EQU (0x1 <<  5) ;- (PMC) 
AT91C_PMC_USBOTGCK        EQU (0x1 <<  6) ;- (PMC) 
AT91C_PMC_USBOTGCK_0      EQU (0x0 <<  6) ;- (PMC) 
AT91C_PMC_USBOTGCK_1      EQU (0x1 <<  6) ;- (PMC) 
AT91C_PMC_UHSYNRST        EQU (0x1 <<  7) ;- (PMC) 
AT91C_PMC_UHSYNRST_0      EQU (0x0 <<  7) ;- (PMC) 
AT91C_PMC_UHSYNRST_1      EQU (0x1 <<  7) ;- (PMC) 
AT91C_PMC_UOSYNRST        EQU (0x1 <<  8) ;- (PMC) 
AT91C_PMC_UOSYNRST_0      EQU (0x0 <<  8) ;- (PMC) 
AT91C_PMC_UOSYNRST_1      EQU (0x1 <<  8) ;- (PMC) 
AT91C_PMC_PLLENPOL        EQU (0x1 <<  9) ;- (PMC) 
AT91C_PMC_PLLENPOL_0      EQU (0x0 <<  9) ;- (PMC) 
AT91C_PMC_PLLENPOL_1      EQU (0x1 <<  9) ;- (PMC) 
AT91C_PMC_BIASREG         EQU (0x1 << 10) ;- (PMC) 
AT91C_PMC_BIASREG_0       EQU (0x0 << 10) ;- (PMC) 
AT91C_PMC_BIASREG_1       EQU (0x1 << 10) ;- (PMC) 
AT91C_PMC_OUTPLL          EQU (0x1 << 11) ;- (PMC) 
AT91C_PMC_OUTPLL_0        EQU (0x0 << 11) ;- (PMC) 
AT91C_PMC_OUTPLL_1        EQU (0x1 << 11) ;- (PMC) 
AT91C_PMC_OUTCURR         EQU (0x1 << 12) ;- (PMC) 
AT91C_PMC_OUTCURR_0       EQU (0x0 << 12) ;- (PMC) 
AT91C_PMC_OUTCURR_1       EQU (0x1 << 12) ;- (PMC) 
AT91C_PMC_FWUP            EQU (0x1 << 13) ;- (PMC) 
AT91C_PMC_FWUP_0          EQU (0x0 << 13) ;- (PMC) 
AT91C_PMC_FWUP_1          EQU (0x1 << 13) ;- (PMC) 
AT91C_PMC_SELMAINCLK      EQU (0x1 << 14) ;- (PMC) 
AT91C_PMC_SELMAINCLK_0    EQU (0x0 << 14) ;- (PMC) 
AT91C_PMC_SELMAINCLK_1    EQU (0x1 << 14) ;- (PMC) 
AT91C_PMC_RSTCLKM         EQU (0x1 << 15) ;- (PMC) 
AT91C_PMC_RSTCLKM_0       EQU (0x0 << 15) ;- (PMC) 
AT91C_PMC_RSTCLKM_1       EQU (0x1 << 15) ;- (PMC) 
AT91C_PMC_NB_PERIPH_CLOCK EQU (0xFF << 16) ;- (PMC) 
// - -------- PMC_VERSION : (PMC Offset: 0xfc)   -------- 
AT91C_PMC_Version         EQU (0xFFFF <<  0) ;- (PMC) 
AT91C_PMC_Version_0       EQU (0x0) ;- (PMC) 
AT91C_PMC_Version_1       EQU (0x1) ;- (PMC) 
AT91C_PMC_MFN             EQU (0x7 << 16) ;- (PMC) 
AT91C_PMC_MFN_0           EQU (0x0 << 16) ;- (PMC) 
AT91C_PMC_MFN_1           EQU (0x1 << 16) ;- (PMC) 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Clock Generator Controler
// - *****************************************************************************
// - -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- 
// - -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- 
// - -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- 
// - -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Reset Controller Interface
// - *****************************************************************************
// - -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- 
AT91C_RSTC_PROCRST        EQU (0x1 <<  0) ;- (RSTC) Processor Reset
AT91C_RSTC_ICERST         EQU (0x1 <<  1) ;- (RSTC) ICE Interface Reset
AT91C_RSTC_PERRST         EQU (0x1 <<  2) ;- (RSTC) Peripheral Reset
AT91C_RSTC_EXTRST         EQU (0x1 <<  3) ;- (RSTC) External Reset
AT91C_RSTC_KEY            EQU (0xFF << 24) ;- (RSTC) Password
// - -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- 
AT91C_RSTC_URSTS          EQU (0x1 <<  0) ;- (RSTC) User Reset Status
AT91C_RSTC_RSTTYP         EQU (0x7 <<  8) ;- (RSTC) Reset Type
AT91C_RSTC_RSTTYP_GENERAL EQU (0x0 <<  8) ;- (RSTC) General reset. Both VDDCORE and VDDBU rising.
AT91C_RSTC_RSTTYP_WAKEUP  EQU (0x1 <<  8) ;- (RSTC) WakeUp Reset. VDDCORE rising.
AT91C_RSTC_RSTTYP_WATCHDOG EQU (0x2 <<  8) ;- (RSTC) Watchdog Reset. Watchdog overflow occured.
AT91C_RSTC_RSTTYP_SOFTWARE EQU (0x3 <<  8) ;- (RSTC) Software Reset. Processor reset required by the software.
AT91C_RSTC_RSTTYP_USER    EQU (0x4 <<  8) ;- (RSTC) User Reset. NRST pin detected low.
AT91C_RSTC_NRSTL          EQU (0x1 << 16) ;- (RSTC) NRST pin level
AT91C_RSTC_SRCMP          EQU (0x1 << 17) ;- (RSTC) Software Reset Command in Progress.
// - -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- 
AT91C_RSTC_URSTEN         EQU (0x1 <<  0) ;- (RSTC) User Reset Enable
AT91C_RSTC_URSTIEN        EQU (0x1 <<  4) ;- (RSTC) User Reset Interrupt Enable
AT91C_RSTC_ERSTL          EQU (0xF <<  8) ;- (RSTC) User Reset Enable

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Shut Down Controller Interface
// - *****************************************************************************
// - -------- SHDWC_SHCR : (SHDWC Offset: 0x0) Shut Down Control Register -------- 
AT91C_SHDWC_SHDW          EQU (0x1 <<  0) ;- (SHDWC) Processor Reset
AT91C_SHDWC_KEY           EQU (0xFF << 24) ;- (SHDWC) Shut down KEY Password
// - -------- SHDWC_SHMR : (SHDWC Offset: 0x4) Shut Down Mode Register -------- 
AT91C_SHDWC_WKMODE0       EQU (0x3 <<  0) ;- (SHDWC) Wake Up 0 Mode Selection
AT91C_SHDWC_WKMODE0_NONE  EQU (0x0) ;- (SHDWC) None. No detection is performed on the wake up input.
AT91C_SHDWC_WKMODE0_HIGH  EQU (0x1) ;- (SHDWC) High Level.
AT91C_SHDWC_WKMODE0_LOW   EQU (0x2) ;- (SHDWC) Low Level.
AT91C_SHDWC_WKMODE0_ANYLEVEL EQU (0x3) ;- (SHDWC) Any level change.
AT91C_SHDWC_CPTWK0        EQU (0xF <<  4) ;- (SHDWC) Counter On Wake Up 0
AT91C_SHDWC_WKMODE1       EQU (0x3 <<  8) ;- (SHDWC) Wake Up 1 Mode Selection
AT91C_SHDWC_WKMODE1_NONE  EQU (0x0 <<  8) ;- (SHDWC) None. No detection is performed on the wake up input.
AT91C_SHDWC_WKMODE1_HIGH  EQU (0x1 <<  8) ;- (SHDWC) High Level.
AT91C_SHDWC_WKMODE1_LOW   EQU (0x2 <<  8) ;- (SHDWC) Low Level.
AT91C_SHDWC_WKMODE1_ANYLEVEL EQU (0x3 <<  8) ;- (SHDWC) Any level change.
AT91C_SHDWC_CPTWK1        EQU (0xF << 12) ;- (SHDWC) Counter On Wake Up 1
AT91C_SHDWC_RTTWKEN       EQU (0x1 << 16) ;- (SHDWC) Real Time Timer Wake Up Enable
AT91C_SHDWC_RTCWKEN       EQU (0x1 << 17) ;- (SHDWC) Real Time Clock Wake Up Enable
// - -------- SHDWC_SHSR : (SHDWC Offset: 0x8) Shut Down Status Register -------- 
AT91C_SHDWC_WAKEUP0       EQU (0x1 <<  0) ;- (SHDWC) Wake Up 0 Status
AT91C_SHDWC_WAKEUP1       EQU (0x1 <<  1) ;- (SHDWC) Wake Up 1 Status
AT91C_SHDWC_FWKUP         EQU (0x1 <<  2) ;- (SHDWC) Force Wake Up Status
AT91C_SHDWC_RTTWK         EQU (0x1 << 16) ;- (SHDWC) Real Time Timer wake Up
AT91C_SHDWC_RTCWK         EQU (0x1 << 17) ;- (SHDWC) Real Time Clock wake Up

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Real Time Timer Controller Interface
// - *****************************************************************************
// - -------- RTTC_RTMR : (RTTC Offset: 0x0) Real-time Mode Register -------- 
AT91C_RTTC_RTPRES         EQU (0xFFFF <<  0) ;- (RTTC) Real-time Timer Prescaler Value
AT91C_RTTC_ALMIEN         EQU (0x1 << 16) ;- (RTTC) Alarm Interrupt Enable
AT91C_RTTC_RTTINCIEN      EQU (0x1 << 17) ;- (RTTC) Real Time Timer Increment Interrupt Enable
AT91C_RTTC_RTTRST         EQU (0x1 << 18) ;- (RTTC) Real Time Timer Restart
// - -------- RTTC_RTAR : (RTTC Offset: 0x4) Real-time Alarm Register -------- 
AT91C_RTTC_ALMV           EQU (0x0 <<  0) ;- (RTTC) Alarm Value
// - -------- RTTC_RTVR : (RTTC Offset: 0x8) Current Real-time Value Register -------- 
AT91C_RTTC_CRTV           EQU (0x0 <<  0) ;- (RTTC) Current Real-time Value
// - -------- RTTC_RTSR : (RTTC Offset: 0xc) Real-time Status Register -------- 
AT91C_RTTC_ALMS           EQU (0x1 <<  0) ;- (RTTC) Real-time Alarm Status
AT91C_RTTC_RTTINC         EQU (0x1 <<  1) ;- (RTTC) Real-time Timer Increment

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Periodic Interval Timer Controller Interface
// - *****************************************************************************
// - -------- PITC_PIMR : (PITC Offset: 0x0) Periodic Interval Mode Register -------- 
AT91C_PITC_PIV            EQU (0xFFFFF <<  0) ;- (PITC) Periodic Interval Value
AT91C_PITC_PITEN          EQU (0x1 << 24) ;- (PITC) Periodic Interval Timer Enabled
AT91C_PITC_PITIEN         EQU (0x1 << 25) ;- (PITC) Periodic Interval Timer Interrupt Enable
// - -------- PITC_PISR : (PITC Offset: 0x4) Periodic Interval Status Register -------- 
AT91C_PITC_PITS           EQU (0x1 <<  0) ;- (PITC) Periodic Interval Timer Status
// - -------- PITC_PIVR : (PITC Offset: 0x8) Periodic Interval Value Register -------- 
AT91C_PITC_CPIV           EQU (0xFFFFF <<  0) ;- (PITC) Current Periodic Interval Value
AT91C_PITC_PICNT          EQU (0xFFF << 20) ;- (PITC) Periodic Interval Counter
// - -------- PITC_PIIR : (PITC Offset: 0xc) Periodic Interval Image Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Watchdog Timer Controller Interface
// - *****************************************************************************
// - -------- WDTC_WDCR : (WDTC Offset: 0x0) Periodic Interval Image Register -------- 
AT91C_WDTC_WDRSTT         EQU (0x1 <<  0) ;- (WDTC) Watchdog Restart
AT91C_WDTC_KEY            EQU (0xFF << 24) ;- (WDTC) Watchdog KEY Password
// - -------- WDTC_WDMR : (WDTC Offset: 0x4) Watchdog Mode Register -------- 
AT91C_WDTC_WDV            EQU (0xFFF <<  0) ;- (WDTC) Watchdog Timer Restart
AT91C_WDTC_WDFIEN         EQU (0x1 << 12) ;- (WDTC) Watchdog Fault Interrupt Enable
AT91C_WDTC_WDRSTEN        EQU (0x1 << 13) ;- (WDTC) Watchdog Reset Enable
AT91C_WDTC_WDRPROC        EQU (0x1 << 14) ;- (WDTC) Watchdog Timer Restart
AT91C_WDTC_WDDIS          EQU (0x1 << 15) ;- (WDTC) Watchdog Disable
AT91C_WDTC_WDD            EQU (0xFFF << 16) ;- (WDTC) Watchdog Delta Value
AT91C_WDTC_WDDBGHLT       EQU (0x1 << 28) ;- (WDTC) Watchdog Debug Halt
AT91C_WDTC_WDIDLEHLT      EQU (0x1 << 29) ;- (WDTC) Watchdog Idle Halt
// - -------- WDTC_WDSR : (WDTC Offset: 0x8) Watchdog Status Register -------- 
AT91C_WDTC_WDUNF          EQU (0x1 <<  0) ;- (WDTC) Watchdog Underflow
AT91C_WDTC_WDERR          EQU (0x1 <<  1) ;- (WDTC) Watchdog Error

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm and Parallel Load Interface
// - *****************************************************************************
// - -------- RTC_CR : (RTC Offset: 0x0) RTC Control Register -------- 
AT91C_RTC_UPDTIM          EQU (0x1 <<  0) ;- (RTC) Update Request Time Register
AT91C_RTC_UPDCAL          EQU (0x1 <<  1) ;- (RTC) Update Request Calendar Register
AT91C_RTC_TIMEVSEL        EQU (0x3 <<  8) ;- (RTC) Time Event Selection
AT91C_RTC_TIMEVSEL_MINUTE EQU (0x0 <<  8) ;- (RTC) Minute change.
AT91C_RTC_TIMEVSEL_HOUR   EQU (0x1 <<  8) ;- (RTC) Hour change.
AT91C_RTC_TIMEVSEL_DAY24  EQU (0x2 <<  8) ;- (RTC) Every day at midnight.
AT91C_RTC_TIMEVSEL_DAY12  EQU (0x3 <<  8) ;- (RTC) Every day at noon.
AT91C_RTC_CALEVSEL        EQU (0x3 << 16) ;- (RTC) Calendar Event Selection
AT91C_RTC_CALEVSEL_WEEK   EQU (0x0 << 16) ;- (RTC) Week change (every Monday at time 00:00:00).
AT91C_RTC_CALEVSEL_MONTH  EQU (0x1 << 16) ;- (RTC) Month change (every 01 of each month at time 00:00:00).
AT91C_RTC_CALEVSEL_YEAR   EQU (0x2 << 16) ;- (RTC) Year change (every January 1 at time 00:00:00).
// - -------- RTC_MR : (RTC Offset: 0x4) RTC Mode Register -------- 
AT91C_RTC_HRMOD           EQU (0x1 <<  0) ;- (RTC) 12-24 hour Mode
// - -------- RTC_TIMR : (RTC Offset: 0x8) RTC Time Register -------- 
AT91C_RTC_SEC             EQU (0x7F <<  0) ;- (RTC) Current Second
AT91C_RTC_MIN             EQU (0x7F <<  8) ;- (RTC) Current Minute
AT91C_RTC_HOUR            EQU (0x3F << 16) ;- (RTC) Current Hour
AT91C_RTC_AMPM            EQU (0x1 << 22) ;- (RTC) Ante Meridiem, Post Meridiem Indicator
// - -------- RTC_CALR : (RTC Offset: 0xc) RTC Calendar Register -------- 
AT91C_RTC_CENT            EQU (0x3F <<  0) ;- (RTC) Current Century
AT91C_RTC_YEAR            EQU (0xFF <<  8) ;- (RTC) Current Year
AT91C_RTC_MONTH           EQU (0x1F << 16) ;- (RTC) Current Month
AT91C_RTC_DAY             EQU (0x7 << 21) ;- (RTC) Current Day
AT91C_RTC_DATE            EQU (0x3F << 24) ;- (RTC) Current Date
// - -------- RTC_TIMALR : (RTC Offset: 0x10) RTC Time Alarm Register -------- 
AT91C_RTC_SECEN           EQU (0x1 <<  7) ;- (RTC) Second Alarm Enable
AT91C_RTC_MINEN           EQU (0x1 << 15) ;- (RTC) Minute Alarm
AT91C_RTC_HOUREN          EQU (0x1 << 23) ;- (RTC) Current Hour
// - -------- RTC_CALALR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- 
AT91C_RTC_MONTHEN         EQU (0x1 << 23) ;- (RTC) Month Alarm Enable
AT91C_RTC_DATEEN          EQU (0x1 << 31) ;- (RTC) Date Alarm Enable
// - -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- 
AT91C_RTC_ACKUPD          EQU (0x1 <<  0) ;- (RTC) Acknowledge for Update
AT91C_RTC_ALARM           EQU (0x1 <<  1) ;- (RTC) Alarm Flag
AT91C_RTC_SECEV           EQU (0x1 <<  2) ;- (RTC) Second Event
AT91C_RTC_TIMEV           EQU (0x1 <<  3) ;- (RTC) Time Event
AT91C_RTC_CALEV           EQU (0x1 <<  4) ;- (RTC) Calendar event
// - -------- RTC_SCCR : (RTC Offset: 0x1c) RTC Status Clear Command Register -------- 
// - -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- 
// - -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- 
// - -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- 
// - -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- 
AT91C_RTC_NVTIM           EQU (0x1 <<  0) ;- (RTC) Non valid Time
AT91C_RTC_NVCAL           EQU (0x1 <<  1) ;- (RTC) Non valid Calendar
AT91C_RTC_NVTIMALR        EQU (0x1 <<  2) ;- (RTC) Non valid time Alarm
AT91C_RTC_NVCALALR        EQU (0x1 <<  3) ;- (RTC) Nonvalid Calendar Alarm

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Timer Counter Channel Interface
// - *****************************************************************************
// - -------- TC_CCR : (TC Offset: 0x0) TC Channel Control Register -------- 
AT91C_TC_CLKEN            EQU (0x1 <<  0) ;- (TC) Counter Clock Enable Command
AT91C_TC_CLKDIS           EQU (0x1 <<  1) ;- (TC) Counter Clock Disable Command
AT91C_TC_SWTRG            EQU (0x1 <<  2) ;- (TC) Software Trigger Command
// - -------- TC_CMR : (TC Offset: 0x4) TC Channel Mode Register: Capture Mode / Waveform Mode -------- 
AT91C_TC_CLKS             EQU (0x7 <<  0) ;- (TC) Clock Selection
AT91C_TC_CLKS_TIMER_DIV1_CLOCK EQU (0x0) ;- (TC) Clock selected: TIMER_DIV1_CLOCK
AT91C_TC_CLKS_TIMER_DIV2_CLOCK EQU (0x1) ;- (TC) Clock selected: TIMER_DIV2_CLOCK
AT91C_TC_CLKS_TIMER_DIV3_CLOCK EQU (0x2) ;- (TC) Clock selected: TIMER_DIV3_CLOCK
AT91C_TC_CLKS_TIMER_DIV4_CLOCK EQU (0x3) ;- (TC) Clock selected: TIMER_DIV4_CLOCK
AT91C_TC_CLKS_TIMER_DIV5_CLOCK EQU (0x4) ;- (TC) Clock selected: TIMER_DIV5_CLOCK
AT91C_TC_CLKS_XC0         EQU (0x5) ;- (TC) Clock selected: XC0
AT91C_TC_CLKS_XC1         EQU (0x6) ;- (TC) Clock selected: XC1
AT91C_TC_CLKS_XC2         EQU (0x7) ;- (TC) Clock selected: XC2
AT91C_TC_CLKI             EQU (0x1 <<  3) ;- (TC) Clock Invert
AT91C_TC_BURST            EQU (0x3 <<  4) ;- (TC) Burst Signal Selection
AT91C_TC_BURST_NONE       EQU (0x0 <<  4) ;- (TC) The clock is not gated by an external signal
AT91C_TC_BURST_XC0        EQU (0x1 <<  4) ;- (TC) XC0 is ANDed with the selected clock
AT91C_TC_BURST_XC1        EQU (0x2 <<  4) ;- (TC) XC1 is ANDed with the selected clock
AT91C_TC_BURST_XC2        EQU (0x3 <<  4) ;- (TC) XC2 is ANDed with the selected clock
AT91C_TC_CPCSTOP          EQU (0x1 <<  6) ;- (TC) Counter Clock Stopped with RC Compare
AT91C_TC_LDBSTOP          EQU (0x1 <<  6) ;- (TC) Counter Clock Stopped with RB Loading
AT91C_TC_CPCDIS           EQU (0x1 <<  7) ;- (TC) Counter Clock Disable with RC Compare
AT91C_TC_LDBDIS           EQU (0x1 <<  7) ;- (TC) Counter Clock Disabled with RB Loading
AT91C_TC_ETRGEDG          EQU (0x3 <<  8) ;- (TC) External Trigger Edge Selection
AT91C_TC_ETRGEDG_NONE     EQU (0x0 <<  8) ;- (TC) Edge: None
AT91C_TC_ETRGEDG_RISING   EQU (0x1 <<  8) ;- (TC) Edge: rising edge
AT91C_TC_ETRGEDG_FALLING  EQU (0x2 <<  8) ;- (TC) Edge: falling edge
AT91C_TC_ETRGEDG_BOTH     EQU (0x3 <<  8) ;- (TC) Edge: each edge
AT91C_TC_EEVTEDG          EQU (0x3 <<  8) ;- (TC) External Event Edge Selection
AT91C_TC_EEVTEDG_NONE     EQU (0x0 <<  8) ;- (TC) Edge: None
AT91C_TC_EEVTEDG_RISING   EQU (0x1 <<  8) ;- (TC) Edge: rising edge
AT91C_TC_EEVTEDG_FALLING  EQU (0x2 <<  8) ;- (TC) Edge: falling edge
AT91C_TC_EEVTEDG_BOTH     EQU (0x3 <<  8) ;- (TC) Edge: each edge
AT91C_TC_EEVT             EQU (0x3 << 10) ;- (TC) External Event  Selection
AT91C_TC_EEVT_TIOB        EQU (0x0 << 10) ;- (TC) Signal selected as external event: TIOB TIOB direction: input
AT91C_TC_EEVT_XC0         EQU (0x1 << 10) ;- (TC) Signal selected as external event: XC0 TIOB direction: output
AT91C_TC_EEVT_XC1         EQU (0x2 << 10) ;- (TC) Signal selected as external event: XC1 TIOB direction: output
AT91C_TC_EEVT_XC2         EQU (0x3 << 10) ;- (TC) Signal selected as external event: XC2 TIOB direction: output
AT91C_TC_ABETRG           EQU (0x1 << 10) ;- (TC) TIOA or TIOB External Trigger Selection
AT91C_TC_ENETRG           EQU (0x1 << 12) ;- (TC) External Event Trigger enable
AT91C_TC_WAVESEL          EQU (0x3 << 13) ;- (TC) Waveform  Selection
AT91C_TC_WAVESEL_UP       EQU (0x0 << 13) ;- (TC) UP mode without atomatic trigger on RC Compare
AT91C_TC_WAVESEL_UPDOWN   EQU (0x1 << 13) ;- (TC) UPDOWN mode without automatic trigger on RC Compare
AT91C_TC_WAVESEL_UP_AUTO  EQU (0x2 << 13) ;- (TC) UP mode with automatic trigger on RC Compare
AT91C_TC_WAVESEL_UPDOWN_AUTO EQU (0x3 << 13) ;- (TC) UPDOWN mode with automatic trigger on RC Compare
AT91C_TC_CPCTRG           EQU (0x1 << 14) ;- (TC) RC Compare Trigger Enable
AT91C_TC_WAVE             EQU (0x1 << 15) ;- (TC) 
AT91C_TC_ACPA             EQU (0x3 << 16) ;- (TC) RA Compare Effect on TIOA
AT91C_TC_ACPA_NONE        EQU (0x0 << 16) ;- (TC) Effect: none
AT91C_TC_ACPA_SET         EQU (0x1 << 16) ;- (TC) Effect: set
AT91C_TC_ACPA_CLEAR       EQU (0x2 << 16) ;- (TC) Effect: clear
AT91C_TC_ACPA_TOGGLE      EQU (0x3 << 16) ;- (TC) Effect: toggle
AT91C_TC_LDRA             EQU (0x3 << 16) ;- (TC) RA Loading Selection
AT91C_TC_LDRA_NONE        EQU (0x0 << 16) ;- (TC) Edge: None
AT91C_TC_LDRA_RISING      EQU (0x1 << 16) ;- (TC) Edge: rising edge of TIOA
AT91C_TC_LDRA_FALLING     EQU (0x2 << 16) ;- (TC) Edge: falling edge of TIOA
AT91C_TC_LDRA_BOTH        EQU (0x3 << 16) ;- (TC) Edge: each edge of TIOA
AT91C_TC_ACPC             EQU (0x3 << 18) ;- (TC) RC Compare Effect on TIOA
AT91C_TC_ACPC_NONE        EQU (0x0 << 18) ;- (TC) Effect: none
AT91C_TC_ACPC_SET         EQU (0x1 << 18) ;- (TC) Effect: set
AT91C_TC_ACPC_CLEAR       EQU (0x2 << 18) ;- (TC) Effect: clear
AT91C_TC_ACPC_TOGGLE      EQU (0x3 << 18) ;- (TC) Effect: toggle
AT91C_TC_LDRB             EQU (0x3 << 18) ;- (TC) RB Loading Selection
AT91C_TC_LDRB_NONE        EQU (0x0 << 18) ;- (TC) Edge: None
AT91C_TC_LDRB_RISING      EQU (0x1 << 18) ;- (TC) Edge: rising edge of TIOA
AT91C_TC_LDRB_FALLING     EQU (0x2 << 18) ;- (TC) Edge: falling edge of TIOA
AT91C_TC_LDRB_BOTH        EQU (0x3 << 18) ;- (TC) Edge: each edge of TIOA
AT91C_TC_AEEVT            EQU (0x3 << 20) ;- (TC) External Event Effect on TIOA
AT91C_TC_AEEVT_NONE       EQU (0x0 << 20) ;- (TC) Effect: none
AT91C_TC_AEEVT_SET        EQU (0x1 << 20) ;- (TC) Effect: set
AT91C_TC_AEEVT_CLEAR      EQU (0x2 << 20) ;- (TC) Effect: clear
AT91C_TC_AEEVT_TOGGLE     EQU (0x3 << 20) ;- (TC) Effect: toggle
AT91C_TC_ASWTRG           EQU (0x3 << 22) ;- (TC) Software Trigger Effect on TIOA
AT91C_TC_ASWTRG_NONE      EQU (0x0 << 22) ;- (TC) Effect: none
AT91C_TC_ASWTRG_SET       EQU (0x1 << 22) ;- (TC) Effect: set
AT91C_TC_ASWTRG_CLEAR     EQU (0x2 << 22) ;- (TC) Effect: clear
AT91C_TC_ASWTRG_TOGGLE    EQU (0x3 << 22) ;- (TC) Effect: toggle
AT91C_TC_BCPB             EQU (0x3 << 24) ;- (TC) RB Compare Effect on TIOB
AT91C_TC_BCPB_NONE        EQU (0x0 << 24) ;- (TC) Effect: none
AT91C_TC_BCPB_SET         EQU (0x1 << 24) ;- (TC) Effect: set
AT91C_TC_BCPB_CLEAR       EQU (0x2 << 24) ;- (TC) Effect: clear
AT91C_TC_BCPB_TOGGLE      EQU (0x3 << 24) ;- (TC) Effect: toggle
AT91C_TC_BCPC             EQU (0x3 << 26) ;- (TC) RC Compare Effect on TIOB
AT91C_TC_BCPC_NONE        EQU (0x0 << 26) ;- (TC) Effect: none
AT91C_TC_BCPC_SET         EQU (0x1 << 26) ;- (TC) Effect: set
AT91C_TC_BCPC_CLEAR       EQU (0x2 << 26) ;- (TC) Effect: clear
AT91C_TC_BCPC_TOGGLE      EQU (0x3 << 26) ;- (TC) Effect: toggle
AT91C_TC_BEEVT            EQU (0x3 << 28) ;- (TC) External Event Effect on TIOB
AT91C_TC_BEEVT_NONE       EQU (0x0 << 28) ;- (TC) Effect: none
AT91C_TC_BEEVT_SET        EQU (0x1 << 28) ;- (TC) Effect: set
AT91C_TC_BEEVT_CLEAR      EQU (0x2 << 28) ;- (TC) Effect: clear
AT91C_TC_BEEVT_TOGGLE     EQU (0x3 << 28) ;- (TC) Effect: toggle
AT91C_TC_BSWTRG           EQU (0x3 << 30) ;- (TC) Software Trigger Effect on TIOB
AT91C_TC_BSWTRG_NONE      EQU (0x0 << 30) ;- (TC) Effect: none
AT91C_TC_BSWTRG_SET       EQU (0x1 << 30) ;- (TC) Effect: set
AT91C_TC_BSWTRG_CLEAR     EQU (0x2 << 30) ;- (TC) Effect: clear
AT91C_TC_BSWTRG_TOGGLE    EQU (0x3 << 30) ;- (TC) Effect: toggle
// - -------- TC_SR : (TC Offset: 0x20) TC Channel Status Register -------- 
AT91C_TC_COVFS            EQU (0x1 <<  0) ;- (TC) Counter Overflow
AT91C_TC_LOVRS            EQU (0x1 <<  1) ;- (TC) Load Overrun
AT91C_TC_CPAS             EQU (0x1 <<  2) ;- (TC) RA Compare
AT91C_TC_CPBS             EQU (0x1 <<  3) ;- (TC) RB Compare
AT91C_TC_CPCS             EQU (0x1 <<  4) ;- (TC) RC Compare
AT91C_TC_LDRAS            EQU (0x1 <<  5) ;- (TC) RA Loading
AT91C_TC_LDRBS            EQU (0x1 <<  6) ;- (TC) RB Loading
AT91C_TC_ETRGS            EQU (0x1 <<  7) ;- (TC) External Trigger
AT91C_TC_CLKSTA           EQU (0x1 << 16) ;- (TC) Clock Enabling
AT91C_TC_MTIOA            EQU (0x1 << 17) ;- (TC) TIOA Mirror
AT91C_TC_MTIOB            EQU (0x1 << 18) ;- (TC) TIOA Mirror
// - -------- TC_IER : (TC Offset: 0x24) TC Channel Interrupt Enable Register -------- 
// - -------- TC_IDR : (TC Offset: 0x28) TC Channel Interrupt Disable Register -------- 
// - -------- TC_IMR : (TC Offset: 0x2c) TC Channel Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Timer Counter Interface
// - *****************************************************************************
// - -------- TCB_BCR : (TCB Offset: 0xc0) TC Block Control Register -------- 
AT91C_TCB_SYNC            EQU (0x1 <<  0) ;- (TCB) Synchro Command
// - -------- TCB_BMR : (TCB Offset: 0xc4) TC Block Mode Register -------- 
AT91C_TCB_TC0XC0S         EQU (0x3 <<  0) ;- (TCB) External Clock Signal 0 Selection
AT91C_TCB_TC0XC0S_TCLK0   EQU (0x0) ;- (TCB) TCLK0 connected to XC0
AT91C_TCB_TC0XC0S_NONE    EQU (0x1) ;- (TCB) None signal connected to XC0
AT91C_TCB_TC0XC0S_TIOA1   EQU (0x2) ;- (TCB) TIOA1 connected to XC0
AT91C_TCB_TC0XC0S_TIOA2   EQU (0x3) ;- (TCB) TIOA2 connected to XC0
AT91C_TCB_TC1XC1S         EQU (0x3 <<  2) ;- (TCB) External Clock Signal 1 Selection
AT91C_TCB_TC1XC1S_TCLK1   EQU (0x0 <<  2) ;- (TCB) TCLK1 connected to XC1
AT91C_TCB_TC1XC1S_NONE    EQU (0x1 <<  2) ;- (TCB) None signal connected to XC1
AT91C_TCB_TC1XC1S_TIOA0   EQU (0x2 <<  2) ;- (TCB) TIOA0 connected to XC1
AT91C_TCB_TC1XC1S_TIOA2   EQU (0x3 <<  2) ;- (TCB) TIOA2 connected to XC1
AT91C_TCB_TC2XC2S         EQU (0x3 <<  4) ;- (TCB) External Clock Signal 2 Selection
AT91C_TCB_TC2XC2S_TCLK2   EQU (0x0 <<  4) ;- (TCB) TCLK2 connected to XC2
AT91C_TCB_TC2XC2S_NONE    EQU (0x1 <<  4) ;- (TCB) None signal connected to XC2
AT91C_TCB_TC2XC2S_TIOA0   EQU (0x2 <<  4) ;- (TCB) TIOA0 connected to XC2
AT91C_TCB_TC2XC2S_TIOA1   EQU (0x3 <<  4) ;- (TCB) TIOA2 connected to XC2

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Multimedia Card Interface
// - *****************************************************************************
// - -------- MCI_CR : (MCI Offset: 0x0) MCI Control Register -------- 
AT91C_MCI_MCIEN           EQU (0x1 <<  0) ;- (MCI) Multimedia Interface Enable
AT91C_MCI_MCIEN_0         EQU (0x0) ;- (MCI) No effect
AT91C_MCI_MCIEN_1         EQU (0x1) ;- (MCI) Enable the MultiMedia Interface if MCIDIS is 0
AT91C_MCI_MCIDIS          EQU (0x1 <<  1) ;- (MCI) Multimedia Interface Disable
AT91C_MCI_MCIDIS_0        EQU (0x0 <<  1) ;- (MCI) No effect
AT91C_MCI_MCIDIS_1        EQU (0x1 <<  1) ;- (MCI) Disable the MultiMedia Interface
AT91C_MCI_PWSEN           EQU (0x1 <<  2) ;- (MCI) Power Save Mode Enable
AT91C_MCI_PWSEN_0         EQU (0x0 <<  2) ;- (MCI) No effect
AT91C_MCI_PWSEN_1         EQU (0x1 <<  2) ;- (MCI) Enable the Power-saving mode if PWSDIS is 0.
AT91C_MCI_PWSDIS          EQU (0x1 <<  3) ;- (MCI) Power Save Mode Disable
AT91C_MCI_PWSDIS_0        EQU (0x0 <<  3) ;- (MCI) No effect
AT91C_MCI_PWSDIS_1        EQU (0x1 <<  3) ;- (MCI) Disable the Power-saving mode.
AT91C_MCI_IOWAITEN        EQU (0x1 <<  4) ;- (MCI) SDIO Read Wait Enable
AT91C_MCI_IOWAITEN_0      EQU (0x0 <<  4) ;- (MCI) No effect
AT91C_MCI_IOWAITEN_1      EQU (0x1 <<  4) ;- (MCI) Enables the SDIO Read Wait Operation.
AT91C_MCI_IOWAITDIS       EQU (0x1 <<  5) ;- (MCI) SDIO Read Wait Disable
AT91C_MCI_IOWAITDIS_0     EQU (0x0 <<  5) ;- (MCI) No effect
AT91C_MCI_IOWAITDIS_1     EQU (0x1 <<  5) ;- (MCI) Disables the SDIO Read Wait Operation.
AT91C_MCI_SWRST           EQU (0x1 <<  7) ;- (MCI) MCI Software reset
AT91C_MCI_SWRST_0         EQU (0x0 <<  7) ;- (MCI) No effect
AT91C_MCI_SWRST_1         EQU (0x1 <<  7) ;- (MCI) Resets the MCI
// - -------- MCI_MR : (MCI Offset: 0x4) MCI Mode Register -------- 
AT91C_MCI_CLKDIV          EQU (0xFF <<  0) ;- (MCI) Clock Divider
AT91C_MCI_PWSDIV          EQU (0x7 <<  8) ;- (MCI) Power Saving Divider
AT91C_MCI_RDPROOF         EQU (0x1 << 11) ;- (MCI) Read Proof Enable
AT91C_MCI_RDPROOF_DISABLE EQU (0x0 << 11) ;- (MCI) Disables Read Proof
AT91C_MCI_RDPROOF_ENABLE  EQU (0x1 << 11) ;- (MCI) Enables Read Proof
AT91C_MCI_WRPROOF         EQU (0x1 << 12) ;- (MCI) Write Proof Enable
AT91C_MCI_WRPROOF_DISABLE EQU (0x0 << 12) ;- (MCI) Disables Write Proof
AT91C_MCI_WRPROOF_ENABLE  EQU (0x1 << 12) ;- (MCI) Enables Write Proof
AT91C_MCI_PDCFBYTE        EQU (0x1 << 13) ;- (MCI) PDC Force Byte Transfer
AT91C_MCI_PDCFBYTE_DISABLE EQU (0x0 << 13) ;- (MCI) Disables PDC Force Byte Transfer
AT91C_MCI_PDCFBYTE_ENABLE EQU (0x1 << 13) ;- (MCI) Enables PDC Force Byte Transfer
AT91C_MCI_PDCPADV         EQU (0x1 << 14) ;- (MCI) PDC Padding Value
AT91C_MCI_PDCMODE         EQU (0x1 << 15) ;- (MCI) PDC Oriented Mode
AT91C_MCI_PDCMODE_DISABLE EQU (0x0 << 15) ;- (MCI) Disables PDC Transfer
AT91C_MCI_PDCMODE_ENABLE  EQU (0x1 << 15) ;- (MCI) Enables PDC Transfer
AT91C_MCI_BLKLEN          EQU (0xFFFF << 16) ;- (MCI) Data Block Length
// - -------- MCI_DTOR : (MCI Offset: 0x8) MCI Data Timeout Register -------- 
AT91C_MCI_DTOCYC          EQU (0xF <<  0) ;- (MCI) Data Timeout Cycle Number
AT91C_MCI_DTOMUL          EQU (0x7 <<  4) ;- (MCI) Data Timeout Multiplier
AT91C_MCI_DTOMUL_1        EQU (0x0 <<  4) ;- (MCI) DTOCYC x 1
AT91C_MCI_DTOMUL_16       EQU (0x1 <<  4) ;- (MCI) DTOCYC x 16
AT91C_MCI_DTOMUL_128      EQU (0x2 <<  4) ;- (MCI) DTOCYC x 128
AT91C_MCI_DTOMUL_256      EQU (0x3 <<  4) ;- (MCI) DTOCYC x 256
AT91C_MCI_DTOMUL_1024     EQU (0x4 <<  4) ;- (MCI) DTOCYC x 1024
AT91C_MCI_DTOMUL_4096     EQU (0x5 <<  4) ;- (MCI) DTOCYC x 4096
AT91C_MCI_DTOMUL_65536    EQU (0x6 <<  4) ;- (MCI) DTOCYC x 65536
AT91C_MCI_DTOMUL_1048576  EQU (0x7 <<  4) ;- (MCI) DTOCYC x 1048576
// - -------- MCI_SDCR : (MCI Offset: 0xc) MCI SD Card Register -------- 
AT91C_MCI_SCDSEL          EQU (0x3 <<  0) ;- (MCI) SD Card/SDIO Selector
AT91C_MCI_SCDSEL_SLOTA    EQU (0x0) ;- (MCI) Slot A selected
AT91C_MCI_SCDSEL_SLOTB    EQU (0x1) ;- (MCI) Slot B selected
AT91C_MCI_SCDSEL_SLOTC    EQU (0x2) ;- (MCI) Slot C selected
AT91C_MCI_SCDSEL_SLOTD    EQU (0x3) ;- (MCI) Slot D selected
AT91C_MCI_SCDBUS          EQU (0x3 <<  6) ;- (MCI) SDCard/SDIO Bus Width
AT91C_MCI_SCDBUS_1BIT     EQU (0x0 <<  6) ;- (MCI) 1-bit data bus
AT91C_MCI_SCDBUS_4BITS    EQU (0x2 <<  6) ;- (MCI) 4-bits data bus
AT91C_MCI_SCDBUS_8BITS    EQU (0x3 <<  6) ;- (MCI) 8-bits data bus
// - -------- MCI_CMDR : (MCI Offset: 0x14) MCI Command Register -------- 
AT91C_MCI_CMDNB           EQU (0x3F <<  0) ;- (MCI) Command Number
AT91C_MCI_RSPTYP          EQU (0x3 <<  6) ;- (MCI) Response Type
AT91C_MCI_RSPTYP_NO       EQU (0x0 <<  6) ;- (MCI) No response
AT91C_MCI_RSPTYP_48       EQU (0x1 <<  6) ;- (MCI) 48-bit response
AT91C_MCI_RSPTYP_136      EQU (0x2 <<  6) ;- (MCI) 136-bit response
AT91C_MCI_RSPTYP_R1B      EQU (0x3 <<  6) ;- (MCI) R1b response
AT91C_MCI_SPCMD           EQU (0x7 <<  8) ;- (MCI) Special CMD
AT91C_MCI_SPCMD_NONE      EQU (0x0 <<  8) ;- (MCI) Not a special CMD
AT91C_MCI_SPCMD_INIT      EQU (0x1 <<  8) ;- (MCI) Initialization CMD
AT91C_MCI_SPCMD_SYNC      EQU (0x2 <<  8) ;- (MCI) Synchronized CMD
AT91C_MCI_SPCMD_CE_ATA    EQU (0x3 <<  8) ;- (MCI) CE-ATA Completion Signal disable CMD
AT91C_MCI_SPCMD_IT_CMD    EQU (0x4 <<  8) ;- (MCI) Interrupt command
AT91C_MCI_SPCMD_IT_REP    EQU (0x5 <<  8) ;- (MCI) Interrupt response
AT91C_MCI_OPDCMD          EQU (0x1 << 11) ;- (MCI) Open Drain Command
AT91C_MCI_OPDCMD_PUSHPULL EQU (0x0 << 11) ;- (MCI) Push/pull command
AT91C_MCI_OPDCMD_OPENDRAIN EQU (0x1 << 11) ;- (MCI) Open drain command
AT91C_MCI_MAXLAT          EQU (0x1 << 12) ;- (MCI) Maximum Latency for Command to respond
AT91C_MCI_MAXLAT_5        EQU (0x0 << 12) ;- (MCI) 5 cycles maximum latency
AT91C_MCI_MAXLAT_64       EQU (0x1 << 12) ;- (MCI) 64 cycles maximum latency
AT91C_MCI_TRCMD           EQU (0x3 << 16) ;- (MCI) Transfer CMD
AT91C_MCI_TRCMD_NO        EQU (0x0 << 16) ;- (MCI) No transfer
AT91C_MCI_TRCMD_START     EQU (0x1 << 16) ;- (MCI) Start transfer
AT91C_MCI_TRCMD_STOP      EQU (0x2 << 16) ;- (MCI) Stop transfer
AT91C_MCI_TRDIR           EQU (0x1 << 18) ;- (MCI) Transfer Direction
AT91C_MCI_TRDIR_WRITE     EQU (0x0 << 18) ;- (MCI) Write
AT91C_MCI_TRDIR_READ      EQU (0x1 << 18) ;- (MCI) Read
AT91C_MCI_TRTYP           EQU (0x7 << 19) ;- (MCI) Transfer Type
AT91C_MCI_TRTYP_BLOCK     EQU (0x0 << 19) ;- (MCI) MMC/SDCard Single Block Transfer type
AT91C_MCI_TRTYP_MULTIPLE  EQU (0x1 << 19) ;- (MCI) MMC/SDCard Multiple Block transfer type
AT91C_MCI_TRTYP_STREAM    EQU (0x2 << 19) ;- (MCI) MMC Stream transfer type
AT91C_MCI_TRTYP_SDIO_BYTE EQU (0x4 << 19) ;- (MCI) SDIO Byte transfer type
AT91C_MCI_TRTYP_SDIO_BLOCK EQU (0x5 << 19) ;- (MCI) SDIO Block transfer type
AT91C_MCI_IOSPCMD         EQU (0x3 << 24) ;- (MCI) SDIO Special Command
AT91C_MCI_IOSPCMD_NONE    EQU (0x0 << 24) ;- (MCI) NOT a special command
AT91C_MCI_IOSPCMD_SUSPEND EQU (0x1 << 24) ;- (MCI) SDIO Suspend Command
AT91C_MCI_IOSPCMD_RESUME  EQU (0x2 << 24) ;- (MCI) SDIO Resume Command
AT91C_MCI_ATACS           EQU (0x1 << 26) ;- (MCI) ATA with command completion signal
AT91C_MCI_ATACS_NORMAL    EQU (0x0 << 26) ;- (MCI) normal operation mode
AT91C_MCI_ATACS_COMPLETION EQU (0x1 << 26) ;- (MCI) completion signal is expected within MCI_CSTOR
// - -------- MCI_BLKR : (MCI Offset: 0x18) MCI Block Register -------- 
AT91C_MCI_BCNT            EQU (0xFFFF <<  0) ;- (MCI) MMC/SDIO Block Count / SDIO Byte Count
// - -------- MCI_CSTOR : (MCI Offset: 0x1c) MCI Completion Signal Timeout Register -------- 
AT91C_MCI_CSTOCYC         EQU (0xF <<  0) ;- (MCI) Completion Signal Timeout Cycle Number
AT91C_MCI_CSTOMUL         EQU (0x7 <<  4) ;- (MCI) Completion Signal Timeout Multiplier
AT91C_MCI_CSTOMUL_1       EQU (0x0 <<  4) ;- (MCI) CSTOCYC x 1
AT91C_MCI_CSTOMUL_16      EQU (0x1 <<  4) ;- (MCI) CSTOCYC x  16
AT91C_MCI_CSTOMUL_128     EQU (0x2 <<  4) ;- (MCI) CSTOCYC x  128
AT91C_MCI_CSTOMUL_256     EQU (0x3 <<  4) ;- (MCI) CSTOCYC x  256
AT91C_MCI_CSTOMUL_1024    EQU (0x4 <<  4) ;- (MCI) CSTOCYC x  1024
AT91C_MCI_CSTOMUL_4096    EQU (0x5 <<  4) ;- (MCI) CSTOCYC x  4096
AT91C_MCI_CSTOMUL_65536   EQU (0x6 <<  4) ;- (MCI) CSTOCYC x  65536
AT91C_MCI_CSTOMUL_1048576 EQU (0x7 <<  4) ;- (MCI) CSTOCYC x  1048576
// - -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- 
AT91C_MCI_CMDRDY          EQU (0x1 <<  0) ;- (MCI) Command Ready flag
AT91C_MCI_RXRDY           EQU (0x1 <<  1) ;- (MCI) RX Ready flag
AT91C_MCI_TXRDY           EQU (0x1 <<  2) ;- (MCI) TX Ready flag
AT91C_MCI_BLKE            EQU (0x1 <<  3) ;- (MCI) Data Block Transfer Ended flag
AT91C_MCI_DTIP            EQU (0x1 <<  4) ;- (MCI) Data Transfer in Progress flag
AT91C_MCI_NOTBUSY         EQU (0x1 <<  5) ;- (MCI) Data Line Not Busy flag
AT91C_MCI_ENDRX           EQU (0x1 <<  6) ;- (MCI) End of RX Buffer flag
AT91C_MCI_ENDTX           EQU (0x1 <<  7) ;- (MCI) End of TX Buffer flag
AT91C_MCI_SDIOIRQA        EQU (0x1 <<  8) ;- (MCI) SDIO Interrupt for Slot A
AT91C_MCI_SDIOIRQB        EQU (0x1 <<  9) ;- (MCI) SDIO Interrupt for Slot B
AT91C_MCI_SDIOIRQC        EQU (0x1 << 10) ;- (MCI) SDIO Interrupt for Slot C
AT91C_MCI_SDIOIRQD        EQU (0x1 << 11) ;- (MCI) SDIO Interrupt for Slot D
AT91C_MCI_SDIOWAIT        EQU (0x1 << 12) ;- (MCI) SDIO Read Wait operation flag
AT91C_MCI_CSRCV           EQU (0x1 << 13) ;- (MCI) CE-ATA Completion Signal flag
AT91C_MCI_RXBUFF          EQU (0x1 << 14) ;- (MCI) RX Buffer Full flag
AT91C_MCI_TXBUFE          EQU (0x1 << 15) ;- (MCI) TX Buffer Empty flag
AT91C_MCI_RINDE           EQU (0x1 << 16) ;- (MCI) Response Index Error flag
AT91C_MCI_RDIRE           EQU (0x1 << 17) ;- (MCI) Response Direction Error flag
AT91C_MCI_RCRCE           EQU (0x1 << 18) ;- (MCI) Response CRC Error flag
AT91C_MCI_RENDE           EQU (0x1 << 19) ;- (MCI) Response End Bit Error flag
AT91C_MCI_RTOE            EQU (0x1 << 20) ;- (MCI) Response Time-out Error flag
AT91C_MCI_DCRCE           EQU (0x1 << 21) ;- (MCI) data CRC Error flag
AT91C_MCI_DTOE            EQU (0x1 << 22) ;- (MCI) Data timeout Error flag
AT91C_MCI_CSTOE           EQU (0x1 << 23) ;- (MCI) Completion Signal timeout Error flag
AT91C_MCI_BLKOVRE         EQU (0x1 << 24) ;- (MCI) DMA Block Overrun Error flag
AT91C_MCI_DMADONE         EQU (0x1 << 25) ;- (MCI) DMA Transfer Done flag
AT91C_MCI_FIFOEMPTY       EQU (0x1 << 26) ;- (MCI) FIFO Empty flag
AT91C_MCI_XFRDONE         EQU (0x1 << 27) ;- (MCI) Transfer Done flag
AT91C_MCI_OVRE            EQU (0x1 << 30) ;- (MCI) Overrun flag
AT91C_MCI_UNRE            EQU (0x1 << 31) ;- (MCI) Underrun flag
// - -------- MCI_IER : (MCI Offset: 0x44) MCI Interrupt Enable Register -------- 
// - -------- MCI_IDR : (MCI Offset: 0x48) MCI Interrupt Disable Register -------- 
// - -------- MCI_IMR : (MCI Offset: 0x4c) MCI Interrupt Mask Register -------- 
// - -------- MCI_DMA : (MCI Offset: 0x50) MCI DMA Configuration Register -------- 
AT91C_MCI_OFFSET          EQU (0x3 <<  0) ;- (MCI) DMA Write Buffer Offset
AT91C_MCI_CHKSIZE         EQU (0x7 <<  4) ;- (MCI) DMA Channel Read/Write Chunk Size
AT91C_MCI_CHKSIZE_1       EQU (0x0 <<  4) ;- (MCI) Number of data transferred is 1
AT91C_MCI_CHKSIZE_4       EQU (0x1 <<  4) ;- (MCI) Number of data transferred is 4
AT91C_MCI_CHKSIZE_8       EQU (0x2 <<  4) ;- (MCI) Number of data transferred is 8
AT91C_MCI_CHKSIZE_16      EQU (0x3 <<  4) ;- (MCI) Number of data transferred is 16
AT91C_MCI_CHKSIZE_32      EQU (0x4 <<  4) ;- (MCI) Number of data transferred is 32
AT91C_MCI_DMAEN           EQU (0x1 <<  8) ;- (MCI) DMA Hardware Handshaking Enable
AT91C_MCI_DMAEN_DISABLE   EQU (0x0 <<  8) ;- (MCI) DMA interface is disabled
AT91C_MCI_DMAEN_ENABLE    EQU (0x1 <<  8) ;- (MCI) DMA interface is enabled
// - -------- MCI_CFG : (MCI Offset: 0x54) MCI Configuration Register -------- 
AT91C_MCI_FIFOMODE        EQU (0x1 <<  0) ;- (MCI) MCI Internal FIFO Control Mode
AT91C_MCI_FIFOMODE_AMOUNTDATA EQU (0x0) ;- (MCI) A write transfer starts when a sufficient amount of datas is written into the FIFO
AT91C_MCI_FIFOMODE_ONEDATA EQU (0x1) ;- (MCI) A write transfer starts as soon as one data is written into the FIFO
AT91C_MCI_FERRCTRL        EQU (0x1 <<  4) ;- (MCI) Flow Error Flag Reset Control Mode
AT91C_MCI_FERRCTRL_RWCMD  EQU (0x0 <<  4) ;- (MCI) When an underflow/overflow condition flag is set, a new Write/Read command is needed to reset the flag
AT91C_MCI_FERRCTRL_READSR EQU (0x1 <<  4) ;- (MCI) When an underflow/overflow condition flag is set, a read status resets the flag
AT91C_MCI_HSMODE          EQU (0x1 <<  8) ;- (MCI) High Speed Mode
AT91C_MCI_HSMODE_DISABLE  EQU (0x0 <<  8) ;- (MCI) Default Bus Timing Mode
AT91C_MCI_HSMODE_ENABLE   EQU (0x1 <<  8) ;- (MCI) High Speed Mode
AT91C_MCI_LSYNC           EQU (0x1 << 12) ;- (MCI) Synchronize on last block
AT91C_MCI_LSYNC_CURRENT   EQU (0x0 << 12) ;- (MCI) Pending command sent at end of current data block
AT91C_MCI_LSYNC_INFINITE  EQU (0x1 << 12) ;- (MCI) Pending command sent at end of block transfer when transfer length is not infinite
// - -------- MCI_WPCR : (MCI Offset: 0xe4) Write Protection Control Register -------- 
AT91C_MCI_WP_EN           EQU (0x1 <<  0) ;- (MCI) Write Protection Enable
AT91C_MCI_WP_EN_DISABLE   EQU (0x0) ;- (MCI) Write Operation is disabled (if WP_KEY corresponds)
AT91C_MCI_WP_EN_ENABLE    EQU (0x1) ;- (MCI) Write Operation is enabled (if WP_KEY corresponds)
AT91C_MCI_WP_KEY          EQU (0xFFFFFF <<  8) ;- (MCI) Write Protection Key
// - -------- MCI_WPSR : (MCI Offset: 0xe8) Write Protection Status Register -------- 
AT91C_MCI_WP_VS           EQU (0xF <<  0) ;- (MCI) Write Protection Violation Status
AT91C_MCI_WP_VS_NO_VIOLATION EQU (0x0) ;- (MCI) No Write Protection Violation detected since last read
AT91C_MCI_WP_VS_ON_WRITE  EQU (0x1) ;- (MCI) Write Protection Violation detected since last read
AT91C_MCI_WP_VS_ON_RESET  EQU (0x2) ;- (MCI) Software Reset Violation detected since last read
AT91C_MCI_WP_VS_ON_BOTH   EQU (0x3) ;- (MCI) Write Protection and Software Reset Violation detected since last read
AT91C_MCI_WP_VSRC         EQU (0xF <<  8) ;- (MCI) Write Protection Violation Source
AT91C_MCI_WP_VSRC_NO_VIOLATION EQU (0x0 <<  8) ;- (MCI) No Write Protection Violation detected since last read
AT91C_MCI_WP_VSRC_MCI_MR  EQU (0x1 <<  8) ;- (MCI) Write Protection Violation detected on MCI_MR since last read
AT91C_MCI_WP_VSRC_MCI_DTOR EQU (0x2 <<  8) ;- (MCI) Write Protection Violation detected on MCI_DTOR since last read
AT91C_MCI_WP_VSRC_MCI_SDCR EQU (0x3 <<  8) ;- (MCI) Write Protection Violation detected on MCI_SDCR since last read
AT91C_MCI_WP_VSRC_MCI_CSTOR EQU (0x4 <<  8) ;- (MCI) Write Protection Violation detected on MCI_CSTOR since last read
AT91C_MCI_WP_VSRC_MCI_DMA EQU (0x5 <<  8) ;- (MCI) Write Protection Violation detected on MCI_DMA since last read
AT91C_MCI_WP_VSRC_MCI_CFG EQU (0x6 <<  8) ;- (MCI) Write Protection Violation detected on MCI_CFG since last read
AT91C_MCI_WP_VSRC_MCI_DEL EQU (0x7 <<  8) ;- (MCI) Write Protection Violation detected on MCI_DEL since last read
// - -------- MCI_VER : (MCI Offset: 0xfc)  VERSION  Register -------- 
AT91C_MCI_VER             EQU (0xF <<  0) ;- (MCI)  VERSION  Register

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Two-wire Interface
// - *****************************************************************************
// - -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- 
AT91C_TWI_START           EQU (0x1 <<  0) ;- (TWI) Send a START Condition
AT91C_TWI_STOP            EQU (0x1 <<  1) ;- (TWI) Send a STOP Condition
AT91C_TWI_MSEN            EQU (0x1 <<  2) ;- (TWI) TWI Master Transfer Enabled
AT91C_TWI_MSDIS           EQU (0x1 <<  3) ;- (TWI) TWI Master Transfer Disabled
AT91C_TWI_SVEN            EQU (0x1 <<  4) ;- (TWI) TWI Slave mode Enabled
AT91C_TWI_SVDIS           EQU (0x1 <<  5) ;- (TWI) TWI Slave mode Disabled
AT91C_TWI_SWRST           EQU (0x1 <<  7) ;- (TWI) Software Reset
// - -------- TWI_MMR : (TWI Offset: 0x4) TWI Master Mode Register -------- 
AT91C_TWI_IADRSZ          EQU (0x3 <<  8) ;- (TWI) Internal Device Address Size
AT91C_TWI_IADRSZ_NO       EQU (0x0 <<  8) ;- (TWI) No internal device address
AT91C_TWI_IADRSZ_1_BYTE   EQU (0x1 <<  8) ;- (TWI) One-byte internal device address
AT91C_TWI_IADRSZ_2_BYTE   EQU (0x2 <<  8) ;- (TWI) Two-byte internal device address
AT91C_TWI_IADRSZ_3_BYTE   EQU (0x3 <<  8) ;- (TWI) Three-byte internal device address
AT91C_TWI_MREAD           EQU (0x1 << 12) ;- (TWI) Master Read Direction
AT91C_TWI_DADR            EQU (0x7F << 16) ;- (TWI) Device Address
// - -------- TWI_SMR : (TWI Offset: 0x8) TWI Slave Mode Register -------- 
AT91C_TWI_SADR            EQU (0x7F << 16) ;- (TWI) Slave Address
// - -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- 
AT91C_TWI_CLDIV           EQU (0xFF <<  0) ;- (TWI) Clock Low Divider
AT91C_TWI_CHDIV           EQU (0xFF <<  8) ;- (TWI) Clock High Divider
AT91C_TWI_CKDIV           EQU (0x7 << 16) ;- (TWI) Clock Divider
// - -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- 
AT91C_TWI_TXCOMP_SLAVE    EQU (0x1 <<  0) ;- (TWI) Transmission Completed
AT91C_TWI_TXCOMP_MASTER   EQU (0x1 <<  0) ;- (TWI) Transmission Completed
AT91C_TWI_RXRDY           EQU (0x1 <<  1) ;- (TWI) Receive holding register ReaDY
AT91C_TWI_TXRDY_MASTER    EQU (0x1 <<  2) ;- (TWI) Transmit holding register ReaDY
AT91C_TWI_TXRDY_SLAVE     EQU (0x1 <<  2) ;- (TWI) Transmit holding register ReaDY
AT91C_TWI_SVREAD          EQU (0x1 <<  3) ;- (TWI) Slave READ (used only in Slave mode)
AT91C_TWI_SVACC           EQU (0x1 <<  4) ;- (TWI) Slave ACCess (used only in Slave mode)
AT91C_TWI_GACC            EQU (0x1 <<  5) ;- (TWI) General Call ACcess (used only in Slave mode)
AT91C_TWI_OVRE            EQU (0x1 <<  6) ;- (TWI) Overrun Error (used only in Master and Multi-master mode)
AT91C_TWI_NACK_SLAVE      EQU (0x1 <<  8) ;- (TWI) Not Acknowledged
AT91C_TWI_NACK_MASTER     EQU (0x1 <<  8) ;- (TWI) Not Acknowledged
AT91C_TWI_ARBLST_MULTI_MASTER EQU (0x1 <<  9) ;- (TWI) Arbitration Lost (used only in Multimaster mode)
AT91C_TWI_SCLWS           EQU (0x1 << 10) ;- (TWI) Clock Wait State (used only in Slave mode)
AT91C_TWI_EOSACC          EQU (0x1 << 11) ;- (TWI) End Of Slave ACCess (used only in Slave mode)
AT91C_TWI_ENDRX           EQU (0x1 << 12) ;- (TWI) End of Receiver Transfer
AT91C_TWI_ENDTX           EQU (0x1 << 13) ;- (TWI) End of Receiver Transfer
AT91C_TWI_RXBUFF          EQU (0x1 << 14) ;- (TWI) RXBUFF Interrupt
AT91C_TWI_TXBUFE          EQU (0x1 << 15) ;- (TWI) TXBUFE Interrupt
// - -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- 
// - -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- 
// - -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Usart
// - *****************************************************************************
// - -------- US_CR : (USART Offset: 0x0)  Control Register -------- 
AT91C_US_STTBRK           EQU (0x1 <<  9) ;- (USART) Start Break
AT91C_US_STPBRK           EQU (0x1 << 10) ;- (USART) Stop Break
AT91C_US_STTTO            EQU (0x1 << 11) ;- (USART) Start Time-out
AT91C_US_SENDA            EQU (0x1 << 12) ;- (USART) Send Address
AT91C_US_RSTIT            EQU (0x1 << 13) ;- (USART) Reset Iterations
AT91C_US_RSTNACK          EQU (0x1 << 14) ;- (USART) Reset Non Acknowledge
AT91C_US_RETTO            EQU (0x1 << 15) ;- (USART) Rearm Time-out
AT91C_US_DTREN            EQU (0x1 << 16) ;- (USART) Data Terminal ready Enable
AT91C_US_DTRDIS           EQU (0x1 << 17) ;- (USART) Data Terminal ready Disable
AT91C_US_RTSEN            EQU (0x1 << 18) ;- (USART) Request to Send enable
AT91C_US_RTSDIS           EQU (0x1 << 19) ;- (USART) Request to Send Disable
// - -------- US_MR : (USART Offset: 0x4)  Mode Register -------- 
AT91C_US_USMODE           EQU (0xF <<  0) ;- (USART) Usart mode
AT91C_US_USMODE_NORMAL    EQU (0x0) ;- (USART) Normal
AT91C_US_USMODE_RS485     EQU (0x1) ;- (USART) RS485
AT91C_US_USMODE_HWHSH     EQU (0x2) ;- (USART) Hardware Handshaking
AT91C_US_USMODE_MODEM     EQU (0x3) ;- (USART) Modem
AT91C_US_USMODE_ISO7816_0 EQU (0x4) ;- (USART) ISO7816 protocol: T = 0
AT91C_US_USMODE_ISO7816_1 EQU (0x6) ;- (USART) ISO7816 protocol: T = 1
AT91C_US_USMODE_IRDA      EQU (0x8) ;- (USART) IrDA
AT91C_US_USMODE_SWHSH     EQU (0xC) ;- (USART) Software Handshaking
AT91C_US_CLKS             EQU (0x3 <<  4) ;- (USART) Clock Selection (Baud Rate generator Input Clock
AT91C_US_CLKS_CLOCK       EQU (0x0 <<  4) ;- (USART) Clock
AT91C_US_CLKS_FDIV1       EQU (0x1 <<  4) ;- (USART) fdiv1
AT91C_US_CLKS_SLOW        EQU (0x2 <<  4) ;- (USART) slow_clock (ARM)
AT91C_US_CLKS_EXT         EQU (0x3 <<  4) ;- (USART) External (SCK)
AT91C_US_CHRL             EQU (0x3 <<  6) ;- (USART) Clock Selection (Baud Rate generator Input Clock
AT91C_US_CHRL_5_BITS      EQU (0x0 <<  6) ;- (USART) Character Length: 5 bits
AT91C_US_CHRL_6_BITS      EQU (0x1 <<  6) ;- (USART) Character Length: 6 bits
AT91C_US_CHRL_7_BITS      EQU (0x2 <<  6) ;- (USART) Character Length: 7 bits
AT91C_US_CHRL_8_BITS      EQU (0x3 <<  6) ;- (USART) Character Length: 8 bits
AT91C_US_SYNC             EQU (0x1 <<  8) ;- (USART) Synchronous Mode Select
AT91C_US_NBSTOP           EQU (0x3 << 12) ;- (USART) Number of Stop bits
AT91C_US_NBSTOP_1_BIT     EQU (0x0 << 12) ;- (USART) 1 stop bit
AT91C_US_NBSTOP_15_BIT    EQU (0x1 << 12) ;- (USART) Asynchronous (SYNC=0) 2 stop bits Synchronous (SYNC=1) 2 stop bits
AT91C_US_NBSTOP_2_BIT     EQU (0x2 << 12) ;- (USART) 2 stop bits
AT91C_US_MSBF             EQU (0x1 << 16) ;- (USART) Bit Order
AT91C_US_MODE9            EQU (0x1 << 17) ;- (USART) 9-bit Character length
AT91C_US_CKLO             EQU (0x1 << 18) ;- (USART) Clock Output Select
AT91C_US_OVER             EQU (0x1 << 19) ;- (USART) Over Sampling Mode
AT91C_US_INACK            EQU (0x1 << 20) ;- (USART) Inhibit Non Acknowledge
AT91C_US_DSNACK           EQU (0x1 << 21) ;- (USART) Disable Successive NACK
AT91C_US_VAR_SYNC         EQU (0x1 << 22) ;- (USART) Variable synchronization of command/data sync Start Frame Delimiter
AT91C_US_MAX_ITER         EQU (0x1 << 24) ;- (USART) Number of Repetitions
AT91C_US_FILTER           EQU (0x1 << 28) ;- (USART) Receive Line Filter
AT91C_US_MANMODE          EQU (0x1 << 29) ;- (USART) Manchester Encoder/Decoder Enable
AT91C_US_MODSYNC          EQU (0x1 << 30) ;- (USART) Manchester Synchronization mode
AT91C_US_ONEBIT           EQU (0x1 << 31) ;- (USART) Start Frame Delimiter selector
// - -------- US_IER : (USART Offset: 0x8)  Interrupt Enable Register -------- 
AT91C_US_RXBRK            EQU (0x1 <<  2) ;- (USART) Break Received/End of Break
AT91C_US_TIMEOUT          EQU (0x1 <<  8) ;- (USART) Receiver Time-out
AT91C_US_ITERATION        EQU (0x1 << 10) ;- (USART) Max number of Repetitions Reached
AT91C_US_NACK             EQU (0x1 << 13) ;- (USART) Non Acknowledge
AT91C_US_RIIC             EQU (0x1 << 16) ;- (USART) Ring INdicator Input Change Flag
AT91C_US_DSRIC            EQU (0x1 << 17) ;- (USART) Data Set Ready Input Change Flag
AT91C_US_DCDIC            EQU (0x1 << 18) ;- (USART) Data Carrier Flag
AT91C_US_CTSIC            EQU (0x1 << 19) ;- (USART) Clear To Send Input Change Flag
AT91C_US_MANE             EQU (0x1 << 20) ;- (USART) Manchester Error Interrupt
// - -------- US_IDR : (USART Offset: 0xc)  Interrupt Disable Register -------- 
// - -------- US_IMR : (USART Offset: 0x10)  Interrupt Mask Register -------- 
// - -------- US_CSR : (USART Offset: 0x14)  Channel Status Register -------- 
AT91C_US_RI               EQU (0x1 << 20) ;- (USART) Image of RI Input
AT91C_US_DSR              EQU (0x1 << 21) ;- (USART) Image of DSR Input
AT91C_US_DCD              EQU (0x1 << 22) ;- (USART) Image of DCD Input
AT91C_US_CTS              EQU (0x1 << 23) ;- (USART) Image of CTS Input
AT91C_US_MANERR           EQU (0x1 << 24) ;- (USART) Manchester Error
// - -------- US_MAN : (USART Offset: 0x50) Manchester Encoder Decoder Register -------- 
AT91C_US_TX_PL            EQU (0xF <<  0) ;- (USART) Transmitter Preamble Length
AT91C_US_TX_PP            EQU (0x3 <<  8) ;- (USART) Transmitter Preamble Pattern
AT91C_US_TX_PP_ALL_ONE    EQU (0x0 <<  8) ;- (USART) ALL_ONE
AT91C_US_TX_PP_ALL_ZERO   EQU (0x1 <<  8) ;- (USART) ALL_ZERO
AT91C_US_TX_PP_ZERO_ONE   EQU (0x2 <<  8) ;- (USART) ZERO_ONE
AT91C_US_TX_PP_ONE_ZERO   EQU (0x3 <<  8) ;- (USART) ONE_ZERO
AT91C_US_TX_MPOL          EQU (0x1 << 12) ;- (USART) Transmitter Manchester Polarity
AT91C_US_RX_PL            EQU (0xF << 16) ;- (USART) Receiver Preamble Length
AT91C_US_RX_PP            EQU (0x3 << 24) ;- (USART) Receiver Preamble Pattern detected
AT91C_US_RX_PP_ALL_ONE    EQU (0x0 << 24) ;- (USART) ALL_ONE
AT91C_US_RX_PP_ALL_ZERO   EQU (0x1 << 24) ;- (USART) ALL_ZERO
AT91C_US_RX_PP_ZERO_ONE   EQU (0x2 << 24) ;- (USART) ZERO_ONE
AT91C_US_RX_PP_ONE_ZERO   EQU (0x3 << 24) ;- (USART) ONE_ZERO
AT91C_US_RX_MPOL          EQU (0x1 << 28) ;- (USART) Receiver Manchester Polarity
AT91C_US_DRIFT            EQU (0x1 << 30) ;- (USART) Drift compensation

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Synchronous Serial Controller Interface
// - *****************************************************************************
// - -------- SSC_CR : (SSC Offset: 0x0) SSC Control Register -------- 
AT91C_SSC_RXEN            EQU (0x1 <<  0) ;- (SSC) Receive Enable
AT91C_SSC_RXDIS           EQU (0x1 <<  1) ;- (SSC) Receive Disable
AT91C_SSC_TXEN            EQU (0x1 <<  8) ;- (SSC) Transmit Enable
AT91C_SSC_TXDIS           EQU (0x1 <<  9) ;- (SSC) Transmit Disable
AT91C_SSC_SWRST           EQU (0x1 << 15) ;- (SSC) Software Reset
// - -------- SSC_RCMR : (SSC Offset: 0x10) SSC Receive Clock Mode Register -------- 
AT91C_SSC_CKS             EQU (0x3 <<  0) ;- (SSC) Receive/Transmit Clock Selection
AT91C_SSC_CKS_DIV         EQU (0x0) ;- (SSC) Divided Clock
AT91C_SSC_CKS_TK          EQU (0x1) ;- (SSC) TK Clock signal
AT91C_SSC_CKS_RK          EQU (0x2) ;- (SSC) RK pin
AT91C_SSC_CKO             EQU (0x7 <<  2) ;- (SSC) Receive/Transmit Clock Output Mode Selection
AT91C_SSC_CKO_NONE        EQU (0x0 <<  2) ;- (SSC) Receive/Transmit Clock Output Mode: None RK pin: Input-only
AT91C_SSC_CKO_CONTINOUS   EQU (0x1 <<  2) ;- (SSC) Continuous Receive/Transmit Clock RK pin: Output
AT91C_SSC_CKO_DATA_TX     EQU (0x2 <<  2) ;- (SSC) Receive/Transmit Clock only during data transfers RK pin: Output
AT91C_SSC_CKI             EQU (0x1 <<  5) ;- (SSC) Receive/Transmit Clock Inversion
AT91C_SSC_CKG             EQU (0x3 <<  6) ;- (SSC) Receive/Transmit Clock Gating Selection
AT91C_SSC_CKG_NONE        EQU (0x0 <<  6) ;- (SSC) Receive/Transmit Clock Gating: None, continuous clock
AT91C_SSC_CKG_LOW         EQU (0x1 <<  6) ;- (SSC) Receive/Transmit Clock enabled only if RF Low
AT91C_SSC_CKG_HIGH        EQU (0x2 <<  6) ;- (SSC) Receive/Transmit Clock enabled only if RF High
AT91C_SSC_START           EQU (0xF <<  8) ;- (SSC) Receive/Transmit Start Selection
AT91C_SSC_START_CONTINOUS EQU (0x0 <<  8) ;- (SSC) Continuous, as soon as the receiver is enabled, and immediately after the end of transfer of the previous data.
AT91C_SSC_START_TX        EQU (0x1 <<  8) ;- (SSC) Transmit/Receive start
AT91C_SSC_START_LOW_RF    EQU (0x2 <<  8) ;- (SSC) Detection of a low level on RF input
AT91C_SSC_START_HIGH_RF   EQU (0x3 <<  8) ;- (SSC) Detection of a high level on RF input
AT91C_SSC_START_FALL_RF   EQU (0x4 <<  8) ;- (SSC) Detection of a falling edge on RF input
AT91C_SSC_START_RISE_RF   EQU (0x5 <<  8) ;- (SSC) Detection of a rising edge on RF input
AT91C_SSC_START_LEVEL_RF  EQU (0x6 <<  8) ;- (SSC) Detection of any level change on RF input
AT91C_SSC_START_EDGE_RF   EQU (0x7 <<  8) ;- (SSC) Detection of any edge on RF input
AT91C_SSC_START_0         EQU (0x8 <<  8) ;- (SSC) Compare 0
AT91C_SSC_STOP            EQU (0x1 << 12) ;- (SSC) Receive Stop Selection
AT91C_SSC_STTDLY          EQU (0xFF << 16) ;- (SSC) Receive/Transmit Start Delay
AT91C_SSC_PERIOD          EQU (0xFF << 24) ;- (SSC) Receive/Transmit Period Divider Selection
// - -------- SSC_RFMR : (SSC Offset: 0x14) SSC Receive Frame Mode Register -------- 
AT91C_SSC_DATLEN          EQU (0x1F <<  0) ;- (SSC) Data Length
AT91C_SSC_LOOP            EQU (0x1 <<  5) ;- (SSC) Loop Mode
AT91C_SSC_MSBF            EQU (0x1 <<  7) ;- (SSC) Most Significant Bit First
AT91C_SSC_DATNB           EQU (0xF <<  8) ;- (SSC) Data Number per Frame
AT91C_SSC_FSLEN           EQU (0xF << 16) ;- (SSC) Receive/Transmit Frame Sync length
AT91C_SSC_FSOS            EQU (0x7 << 20) ;- (SSC) Receive/Transmit Frame Sync Output Selection
AT91C_SSC_FSOS_NONE       EQU (0x0 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: None RK pin Input-only
AT91C_SSC_FSOS_NEGATIVE   EQU (0x1 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Negative Pulse
AT91C_SSC_FSOS_POSITIVE   EQU (0x2 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Positive Pulse
AT91C_SSC_FSOS_LOW        EQU (0x3 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Driver Low during data transfer
AT91C_SSC_FSOS_HIGH       EQU (0x4 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Driver High during data transfer
AT91C_SSC_FSOS_TOGGLE     EQU (0x5 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Toggling at each start of data transfer
AT91C_SSC_FSEDGE          EQU (0x1 << 24) ;- (SSC) Frame Sync Edge Detection
// - -------- SSC_TCMR : (SSC Offset: 0x18) SSC Transmit Clock Mode Register -------- 
// - -------- SSC_TFMR : (SSC Offset: 0x1c) SSC Transmit Frame Mode Register -------- 
AT91C_SSC_DATDEF          EQU (0x1 <<  5) ;- (SSC) Data Default Value
AT91C_SSC_FSDEN           EQU (0x1 << 23) ;- (SSC) Frame Sync Data Enable
// - -------- SSC_SR : (SSC Offset: 0x40) SSC Status Register -------- 
AT91C_SSC_TXRDY           EQU (0x1 <<  0) ;- (SSC) Transmit Ready
AT91C_SSC_TXEMPTY         EQU (0x1 <<  1) ;- (SSC) Transmit Empty
AT91C_SSC_ENDTX           EQU (0x1 <<  2) ;- (SSC) End Of Transmission
AT91C_SSC_TXBUFE          EQU (0x1 <<  3) ;- (SSC) Transmit Buffer Empty
AT91C_SSC_RXRDY           EQU (0x1 <<  4) ;- (SSC) Receive Ready
AT91C_SSC_OVRUN           EQU (0x1 <<  5) ;- (SSC) Receive Overrun
AT91C_SSC_ENDRX           EQU (0x1 <<  6) ;- (SSC) End of Reception
AT91C_SSC_RXBUFF          EQU (0x1 <<  7) ;- (SSC) Receive Buffer Full
AT91C_SSC_CP0             EQU (0x1 <<  8) ;- (SSC) Compare 0
AT91C_SSC_CP1             EQU (0x1 <<  9) ;- (SSC) Compare 1
AT91C_SSC_TXSYN           EQU (0x1 << 10) ;- (SSC) Transmit Sync
AT91C_SSC_RXSYN           EQU (0x1 << 11) ;- (SSC) Receive Sync
AT91C_SSC_TXENA           EQU (0x1 << 16) ;- (SSC) Transmit Enable
AT91C_SSC_RXENA           EQU (0x1 << 17) ;- (SSC) Receive Enable
// - -------- SSC_IER : (SSC Offset: 0x44) SSC Interrupt Enable Register -------- 
// - -------- SSC_IDR : (SSC Offset: 0x48) SSC Interrupt Disable Register -------- 
// - -------- SSC_IMR : (SSC Offset: 0x4c) SSC Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR PWMC Channel Interface
// - *****************************************************************************
// - -------- PWMC_CMR : (PWMC_CH Offset: 0x0) PWMC Channel Mode Register -------- 
AT91C_PWMC_CPRE           EQU (0xF <<  0) ;- (PWMC_CH) Channel Pre-scaler : PWMC_CLKx
AT91C_PWMC_CPRE_MCK       EQU (0x0) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCKA      EQU (0xB) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCKB      EQU (0xC) ;- (PWMC_CH) 
AT91C_PWMC_CALG           EQU (0x1 <<  8) ;- (PWMC_CH) Channel Alignment
AT91C_PWMC_CPOL           EQU (0x1 <<  9) ;- (PWMC_CH) Channel Polarity
AT91C_PWMC_CPD            EQU (0x1 << 10) ;- (PWMC_CH) Channel Update Period
// - -------- PWMC_CDTYR : (PWMC_CH Offset: 0x4) PWMC Channel Duty Cycle Register -------- 
AT91C_PWMC_CDTY           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Duty Cycle
// - -------- PWMC_CPRDR : (PWMC_CH Offset: 0x8) PWMC Channel Period Register -------- 
AT91C_PWMC_CPRD           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Period
// - -------- PWMC_CCNTR : (PWMC_CH Offset: 0xc) PWMC Channel Counter Register -------- 
AT91C_PWMC_CCNT           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Counter
// - -------- PWMC_CUPDR : (PWMC_CH Offset: 0x10) PWMC Channel Update Register -------- 
AT91C_PWMC_CUPD           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Update

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Pulse Width Modulation Controller Interface
// - *****************************************************************************
// - -------- PWMC_MR : (PWMC Offset: 0x0) PWMC Mode Register -------- 
AT91C_PWMC_DIVA           EQU (0xFF <<  0) ;- (PWMC) CLKA divide factor.
AT91C_PWMC_PREA           EQU (0xF <<  8) ;- (PWMC) Divider Input Clock Prescaler A
AT91C_PWMC_PREA_MCK       EQU (0x0 <<  8) ;- (PWMC) 
AT91C_PWMC_DIVB           EQU (0xFF << 16) ;- (PWMC) CLKB divide factor.
AT91C_PWMC_PREB           EQU (0xF << 24) ;- (PWMC) Divider Input Clock Prescaler B
AT91C_PWMC_PREB_MCK       EQU (0x0 << 24) ;- (PWMC) 
// - -------- PWMC_ENA : (PWMC Offset: 0x4) PWMC Enable Register -------- 
AT91C_PWMC_CHID0          EQU (0x1 <<  0) ;- (PWMC) Channel ID 0
AT91C_PWMC_CHID1          EQU (0x1 <<  1) ;- (PWMC) Channel ID 1
AT91C_PWMC_CHID2          EQU (0x1 <<  2) ;- (PWMC) Channel ID 2
AT91C_PWMC_CHID3          EQU (0x1 <<  3) ;- (PWMC) Channel ID 3
AT91C_PWMC_CHID4          EQU (0x1 <<  4) ;- (PWMC) Channel ID 4
AT91C_PWMC_CHID5          EQU (0x1 <<  5) ;- (PWMC) Channel ID 5
AT91C_PWMC_CHID6          EQU (0x1 <<  6) ;- (PWMC) Channel ID 6
AT91C_PWMC_CHID7          EQU (0x1 <<  7) ;- (PWMC) Channel ID 7
// - -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- 
// - -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- 
// - -------- PWMC_IER : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- 
// - -------- PWMC_IDR : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- 
// - -------- PWMC_IMR : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- 
// - -------- PWMC_ISR : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Serial Parallel Interface
// - *****************************************************************************
// - -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- 
AT91C_SPI_SPIEN           EQU (0x1 <<  0) ;- (SPI) SPI Enable
AT91C_SPI_SPIDIS          EQU (0x1 <<  1) ;- (SPI) SPI Disable
AT91C_SPI_SWRST           EQU (0x1 <<  7) ;- (SPI) SPI Software reset
AT91C_SPI_LASTXFER        EQU (0x1 << 24) ;- (SPI) SPI Last Transfer
// - -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- 
AT91C_SPI_MSTR            EQU (0x1 <<  0) ;- (SPI) Master/Slave Mode
AT91C_SPI_PS              EQU (0x1 <<  1) ;- (SPI) Peripheral Select
AT91C_SPI_PS_FIXED        EQU (0x0 <<  1) ;- (SPI) Fixed Peripheral Select
AT91C_SPI_PS_VARIABLE     EQU (0x1 <<  1) ;- (SPI) Variable Peripheral Select
AT91C_SPI_PCSDEC          EQU (0x1 <<  2) ;- (SPI) Chip Select Decode
AT91C_SPI_FDIV            EQU (0x1 <<  3) ;- (SPI) Clock Selection
AT91C_SPI_MODFDIS         EQU (0x1 <<  4) ;- (SPI) Mode Fault Detection
AT91C_SPI_LLB             EQU (0x1 <<  7) ;- (SPI) Clock Selection
AT91C_SPI_PCS             EQU (0xF << 16) ;- (SPI) Peripheral Chip Select
AT91C_SPI_DLYBCS          EQU (0xFF << 24) ;- (SPI) Delay Between Chip Selects
// - -------- SPI_RDR : (SPI Offset: 0x8) Receive Data Register -------- 
AT91C_SPI_RD              EQU (0xFFFF <<  0) ;- (SPI) Receive Data
AT91C_SPI_RPCS            EQU (0xF << 16) ;- (SPI) Peripheral Chip Select Status
// - -------- SPI_TDR : (SPI Offset: 0xc) Transmit Data Register -------- 
AT91C_SPI_TD              EQU (0xFFFF <<  0) ;- (SPI) Transmit Data
AT91C_SPI_TPCS            EQU (0xF << 16) ;- (SPI) Peripheral Chip Select Status
// - -------- SPI_SR : (SPI Offset: 0x10) Status Register -------- 
AT91C_SPI_RDRF            EQU (0x1 <<  0) ;- (SPI) Receive Data Register Full
AT91C_SPI_TDRE            EQU (0x1 <<  1) ;- (SPI) Transmit Data Register Empty
AT91C_SPI_MODF            EQU (0x1 <<  2) ;- (SPI) Mode Fault Error
AT91C_SPI_OVRES           EQU (0x1 <<  3) ;- (SPI) Overrun Error Status
AT91C_SPI_ENDRX           EQU (0x1 <<  4) ;- (SPI) End of Receiver Transfer
AT91C_SPI_ENDTX           EQU (0x1 <<  5) ;- (SPI) End of Receiver Transfer
AT91C_SPI_RXBUFF          EQU (0x1 <<  6) ;- (SPI) RXBUFF Interrupt
AT91C_SPI_TXBUFE          EQU (0x1 <<  7) ;- (SPI) TXBUFE Interrupt
AT91C_SPI_NSSR            EQU (0x1 <<  8) ;- (SPI) NSSR Interrupt
AT91C_SPI_TXEMPTY         EQU (0x1 <<  9) ;- (SPI) TXEMPTY Interrupt
AT91C_SPI_SPIENS          EQU (0x1 << 16) ;- (SPI) Enable Status
// - -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- 
// - -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- 
// - -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- 
// - -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- 
AT91C_SPI_CPOL            EQU (0x1 <<  0) ;- (SPI) Clock Polarity
AT91C_SPI_NCPHA           EQU (0x1 <<  1) ;- (SPI) Clock Phase
AT91C_SPI_CSAAT           EQU (0x1 <<  3) ;- (SPI) Chip Select Active After Transfer
AT91C_SPI_BITS            EQU (0xF <<  4) ;- (SPI) Bits Per Transfer
AT91C_SPI_BITS_8          EQU (0x0 <<  4) ;- (SPI) 8 Bits Per transfer
AT91C_SPI_BITS_9          EQU (0x1 <<  4) ;- (SPI) 9 Bits Per transfer
AT91C_SPI_BITS_10         EQU (0x2 <<  4) ;- (SPI) 10 Bits Per transfer
AT91C_SPI_BITS_11         EQU (0x3 <<  4) ;- (SPI) 11 Bits Per transfer
AT91C_SPI_BITS_12         EQU (0x4 <<  4) ;- (SPI) 12 Bits Per transfer
AT91C_SPI_BITS_13         EQU (0x5 <<  4) ;- (SPI) 13 Bits Per transfer
AT91C_SPI_BITS_14         EQU (0x6 <<  4) ;- (SPI) 14 Bits Per transfer
AT91C_SPI_BITS_15         EQU (0x7 <<  4) ;- (SPI) 15 Bits Per transfer
AT91C_SPI_BITS_16         EQU (0x8 <<  4) ;- (SPI) 16 Bits Per transfer
AT91C_SPI_SCBR            EQU (0xFF <<  8) ;- (SPI) Serial Clock Baud Rate
AT91C_SPI_DLYBS           EQU (0xFF << 16) ;- (SPI) Delay Before SPCK
AT91C_SPI_DLYBCT          EQU (0xFF << 24) ;- (SPI) Delay Between Consecutive Transfers

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR TSADC
// - *****************************************************************************
// - -------- TSADC_CR : (TSADC Offset: 0x0) Control Register -------- 
AT91C_TSADC_SWRST         EQU (0x1 <<  0) ;- (TSADC) Software Reset
AT91C_TSADC_START         EQU (0x1 <<  1) ;- (TSADC) Start Conversion
// - -------- TSADC_MR : (TSADC Offset: 0x4) Mode Register -------- 
AT91C_TSADC_TSAMOD        EQU (0x3 <<  0) ;- (TSADC) Touch Screen ADC Mode
AT91C_TSADC_TSAMOD_ADC_ONLY_MODE EQU (0x0) ;- (TSADC) ADC Mode
AT91C_TSADC_TSAMOD_TS_ONLY_MODE EQU (0x1) ;- (TSADC) Touch Screen Only Mode
AT91C_TSADC_LOWRES        EQU (0x1 <<  4) ;- (TSADC) ADC Resolution
AT91C_TSADC_SLEEP         EQU (0x1 <<  5) ;- (TSADC) Sleep Mode
AT91C_TSADC_PENDET        EQU (0x1 <<  6) ;- (TSADC) Pen Detect Selection
AT91C_TSADC_PRESCAL       EQU (0xFF <<  8) ;- (TSADC) Prescaler Rate Selection
AT91C_TSADC_STARTUP       EQU (0x7F << 16) ;- (TSADC) Startup Time
AT91C_TSADC_SHTIM         EQU (0xF << 24) ;- (TSADC) Sample and Hold Time for ADC Channels
AT91C_TSADC_PENDBC        EQU (0xF << 28) ;- (TSADC) Pen Detect Debouncing Period
// - -------- TSADC_TRGR : (TSADC Offset: 0x8) Trigger Register -------- 
AT91C_TSADC_TRGMOD        EQU (0x7 <<  0) ;- (TSADC) Trigger Mode
AT91C_TSADC_TRGMOD_NO_TRIGGER EQU (0x0) ;- (TSADC) No Trigger
AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_RE EQU (0x1) ;- (TSADC) External Trigger Rising Edge
AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_FE EQU (0x2) ;- (TSADC) External Trigger Falling Edge
AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_AE EQU (0x3) ;- (TSADC) External Trigger Any Edge
AT91C_TSADC_TRGMOD_PENDET_TRIGGER EQU (0x4) ;- (TSADC) Pen Detect Trigger (only if PENDET is set and in Touch Screen mode only)
AT91C_TSADC_TRGMOD_PERIODIC_TRIGGER EQU (0x5) ;- (TSADC) Periodic Trigger (wrt TRGPER)
AT91C_TSADC_TRGMOD_CONT_TRIGGER EQU (0x6) ;- (TSADC) Continuous Trigger
AT91C_TSADC_TRGPER        EQU (0xFFFF << 16) ;- (TSADC) Trigger Period
// - -------- TSADC_TSR : (TSADC Offset: 0xc) Touch Screen Register -------- 
AT91C_TSADC_TSSHTIM       EQU (0xF << 24) ;- (TSADC) Sample and Hold Time for Touch Screen Channels
// - -------- TSADC_CHER : (TSADC Offset: 0x10) Channel Enable Register -------- 
AT91C_TSADC_CHENA0        EQU (0x1 <<  0) ;- (TSADC) Channel 0 Enable
AT91C_TSADC_CHENA1        EQU (0x1 <<  1) ;- (TSADC) Channel 1 Enable
AT91C_TSADC_CHENA2        EQU (0x1 <<  2) ;- (TSADC) Channel 2 Enable
AT91C_TSADC_CHENA3        EQU (0x1 <<  3) ;- (TSADC) Channel 3 Enable
AT91C_TSADC_CHENA4        EQU (0x1 <<  4) ;- (TSADC) Channel 4 Enable
AT91C_TSADC_CHENA5        EQU (0x1 <<  5) ;- (TSADC) Channel 5 Enable
AT91C_TSADC_CHENA6        EQU (0x1 <<  6) ;- (TSADC) Channel 6 Enable
AT91C_TSADC_CHENA7        EQU (0x1 <<  7) ;- (TSADC) Channel 7 Enable
// - -------- TSADC_CHDR : (TSADC Offset: 0x14) Channel Disable Register -------- 
AT91C_TSADC_CHDIS0        EQU (0x1 <<  0) ;- (TSADC) Channel 0 Disable
AT91C_TSADC_CHDIS1        EQU (0x1 <<  1) ;- (TSADC) Channel 1 Disable
AT91C_TSADC_CHDIS2        EQU (0x1 <<  2) ;- (TSADC) Channel 2 Disable
AT91C_TSADC_CHDIS3        EQU (0x1 <<  3) ;- (TSADC) Channel 3 Disable
AT91C_TSADC_CHDIS4        EQU (0x1 <<  4) ;- (TSADC) Channel 4 Disable
AT91C_TSADC_CHDIS5        EQU (0x1 <<  5) ;- (TSADC) Channel 5 Disable
AT91C_TSADC_CHDIS6        EQU (0x1 <<  6) ;- (TSADC) Channel 6 Disable
AT91C_TSADC_CHDIS7        EQU (0x1 <<  7) ;- (TSADC) Channel 7 Disable
// - -------- TSADC_CHSR : (TSADC Offset: 0x18) Channel Status Register -------- 
AT91C_TSADC_CHS0          EQU (0x1 <<  0) ;- (TSADC) Channel 0 Status
AT91C_TSADC_CHS1          EQU (0x1 <<  1) ;- (TSADC) Channel 1 Status
AT91C_TSADC_CHS2          EQU (0x1 <<  2) ;- (TSADC) Channel 2 Status
AT91C_TSADC_CHS3          EQU (0x1 <<  3) ;- (TSADC) Channel 3 Status
AT91C_TSADC_CHS4          EQU (0x1 <<  4) ;- (TSADC) Channel 4 Status
AT91C_TSADC_CHS5          EQU (0x1 <<  5) ;- (TSADC) Channel 5 Status
AT91C_TSADC_CHS6          EQU (0x1 <<  6) ;- (TSADC) Channel 6 Status
AT91C_TSADC_CHS7          EQU (0x1 <<  7) ;- (TSADC) Channel 7 Status
// - -------- TSADC_SR : (TSADC Offset: 0x1c) Status Register -------- 
AT91C_TSADC_EOC0          EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion
AT91C_TSADC_EOC1          EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion
AT91C_TSADC_EOC2          EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion
AT91C_TSADC_EOC3          EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion
AT91C_TSADC_EOC4          EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion
AT91C_TSADC_EOC5          EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion
AT91C_TSADC_EOC6          EQU (0x1 <<  6) ;- (TSADC) Channel 6 End Of Conversion
AT91C_TSADC_EOC7          EQU (0x1 <<  7) ;- (TSADC) Channel 7 End Of Conversion
AT91C_TSADC_OVRE0         EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error
AT91C_TSADC_OVRE1         EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error
AT91C_TSADC_OVRE2         EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error
AT91C_TSADC_OVRE3         EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error
AT91C_TSADC_OVRE4         EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error
AT91C_TSADC_OVRE5         EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error
AT91C_TSADC_OVRE6         EQU (0x1 << 14) ;- (TSADC) Channel 6 Overrun Error
AT91C_TSADC_OVRE7         EQU (0x1 << 15) ;- (TSADC) Channel 7 Overrun Error
AT91C_TSADC_DRDY          EQU (0x1 << 16) ;- (TSADC) Data Ready
AT91C_TSADC_GOVRE         EQU (0x1 << 17) ;- (TSADC) General Overrun Error
AT91C_TSADC_ENDRX         EQU (0x1 << 18) ;- (TSADC) End of RX Buffer
AT91C_TSADC_RXBUFF        EQU (0x1 << 19) ;- (TSADC) RX Buffer Full
AT91C_TSADC_PENCNT        EQU (0x1 << 20) ;- (TSADC) Pen Contact
AT91C_TSADC_NOCNT         EQU (0x1 << 21) ;- (TSADC) No Contact
// - -------- TSADC_LCDR : (TSADC Offset: 0x20) Last Converted Data Register -------- 
AT91C_TSADC_LDATA         EQU (0x3FF <<  0) ;- (TSADC) Last Converted Data
// - -------- TSADC_IER : (TSADC Offset: 0x24) Interrupt Enable Register -------- 
AT91C_TSADC_IENAEOC0      EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC1      EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC2      EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC3      EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC4      EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC5      EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC6      EQU (0x1 <<  6) ;- (TSADC) Channel 6 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC7      EQU (0x1 <<  7) ;- (TSADC) Channel 7 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAOVRE0     EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE1     EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE2     EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE3     EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE4     EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE5     EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE6     EQU (0x1 << 14) ;- (TSADC) Channel 6 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE7     EQU (0x1 << 15) ;- (TSADC) Channel 7 Overrun Error Interrupt Enable
AT91C_TSADC_IENADRDY      EQU (0x1 << 16) ;- (TSADC) Data Ready Interrupt Enable
AT91C_TSADC_IENAGOVRE     EQU (0x1 << 17) ;- (TSADC) General Overrun Error Interrupt Enable
AT91C_TSADC_IENAENDRX     EQU (0x1 << 18) ;- (TSADC) End of RX Buffer Interrupt Enable
AT91C_TSADC_IENARXBUFF    EQU (0x1 << 19) ;- (TSADC) RX Buffer Full Interrupt Enable
AT91C_TSADC_IENAPENCNT    EQU (0x1 << 20) ;- (TSADC) Pen Contact Interrupt Enable
AT91C_TSADC_IENANOCNT     EQU (0x1 << 21) ;- (TSADC) No Contact Interrupt Enable
// - -------- TSADC_IDR : (TSADC Offset: 0x28) Interrupt Disable Register -------- 
AT91C_TSADC_IDISEOC0      EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC1      EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC2      EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC3      EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC4      EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC5      EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC6      EQU (0x1 <<  6) ;- (TSADC) Channel 6 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC7      EQU (0x1 <<  7) ;- (TSADC) Channel 7 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISOVRE0     EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE1     EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE2     EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE3     EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE4     EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE5     EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE6     EQU (0x1 << 14) ;- (TSADC) Channel 6 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE7     EQU (0x1 << 15) ;- (TSADC) Channel 7 Overrun Error Interrupt Disable
AT91C_TSADC_IDISDRDY      EQU (0x1 << 16) ;- (TSADC) Data Ready Interrupt Disable
AT91C_TSADC_IDISGOVRE     EQU (0x1 << 17) ;- (TSADC) General Overrun Error Interrupt Disable
AT91C_TSADC_IDISENDRX     EQU (0x1 << 18) ;- (TSADC) End of RX Buffer Interrupt Disable
AT91C_TSADC_IDISRXBUFF    EQU (0x1 << 19) ;- (TSADC) RX Buffer Full Interrupt Disable
AT91C_TSADC_IDISPENCNT    EQU (0x1 << 20) ;- (TSADC) Pen Contact Interrupt Disable
AT91C_TSADC_IDISNOCNT     EQU (0x1 << 21) ;- (TSADC) No Contact Interrupt Disable
// - -------- TSADC_IMR : (TSADC Offset: 0x2c) Interrupt Mask Register -------- 
AT91C_TSADC_IMSKEOC0      EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC1      EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC2      EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC3      EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC4      EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC5      EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC6      EQU (0x1 <<  6) ;- (TSADC) Channel 6 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC7      EQU (0x1 <<  7) ;- (TSADC) Channel 7 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKOVRE0     EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE1     EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE2     EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE3     EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE4     EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE5     EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE6     EQU (0x1 << 14) ;- (TSADC) Channel 6 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE7     EQU (0x1 << 15) ;- (TSADC) Channel 7 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKDRDY      EQU (0x1 << 16) ;- (TSADC) Data Ready Interrupt Mask
AT91C_TSADC_IMSKGOVRE     EQU (0x1 << 17) ;- (TSADC) General Overrun Error Interrupt Mask
AT91C_TSADC_IMSKENDRX     EQU (0x1 << 18) ;- (TSADC) End of RX Buffer Interrupt Mask
AT91C_TSADC_IMSKRXBUFF    EQU (0x1 << 19) ;- (TSADC) RX Buffer Full Interrupt Mask
AT91C_TSADC_IMSKPENCNT    EQU (0x1 << 20) ;- (TSADC) Pen Contact Interrupt Mask
AT91C_TSADC_IMSKNOCNT     EQU (0x1 << 21) ;- (TSADC) No Contact Interrupt Mask
// - -------- TSADC_CDR0 : (TSADC Offset: 0x30) Channel 0 Data Register -------- 
AT91C_TSADC_DATA0         EQU (0x3FF <<  0) ;- (TSADC) Channel 0 Data
// - -------- TSADC_CDR1 : (TSADC Offset: 0x34) Channel 1 Data Register -------- 
AT91C_TSADC_DATA1         EQU (0x3FF <<  0) ;- (TSADC) Channel 1 Data
// - -------- TSADC_CDR2 : (TSADC Offset: 0x38) Channel 2 Data Register -------- 
AT91C_TSADC_DATA2         EQU (0x3FF <<  0) ;- (TSADC) Channel 2 Data
// - -------- TSADC_CDR3 : (TSADC Offset: 0x3c) Channel 3 Data Register -------- 
AT91C_TSADC_DATA3         EQU (0x3FF <<  0) ;- (TSADC) Channel 3 Data
// - -------- TSADC_CDR4 : (TSADC Offset: 0x40) Channel 4 Data Register -------- 
AT91C_TSADC_DATA4         EQU (0x3FF <<  0) ;- (TSADC) Channel 4 Data
// - -------- TSADC_CDR5 : (TSADC Offset: 0x44) Channel 5 Data Register -------- 
AT91C_TSADC_DATA5         EQU (0x3FF <<  0) ;- (TSADC) Channel 5 Data
// - -------- TSADC_CDR6 : (TSADC Offset: 0x48) Channel 6 Data Register -------- 
AT91C_TSADC_DATA6         EQU (0x3FF <<  0) ;- (TSADC) Channel 6 Data
// - -------- TSADC_CDR7 : (TSADC Offset: 0x4c) Channel 7 Data Register -------- 
AT91C_TSADC_DATA7         EQU (0x3FF <<  0) ;- (TSADC) Channel 7 Data

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS Enpoint FIFO data register
// - *****************************************************************************

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS Endpoint struct
// - *****************************************************************************
// - -------- UDPHS_EPTCFG : (UDPHS_EPT Offset: 0x0) UDPHS Endpoint Config Register -------- 
AT91C_UDPHS_EPT_SIZE      EQU (0x7 <<  0) ;- (UDPHS_EPT) Endpoint Size
AT91C_UDPHS_EPT_SIZE_8    EQU (0x0) ;- (UDPHS_EPT)    8 bytes
AT91C_UDPHS_EPT_SIZE_16   EQU (0x1) ;- (UDPHS_EPT)   16 bytes
AT91C_UDPHS_EPT_SIZE_32   EQU (0x2) ;- (UDPHS_EPT)   32 bytes
AT91C_UDPHS_EPT_SIZE_64   EQU (0x3) ;- (UDPHS_EPT)   64 bytes
AT91C_UDPHS_EPT_SIZE_128  EQU (0x4) ;- (UDPHS_EPT)  128 bytes
AT91C_UDPHS_EPT_SIZE_256  EQU (0x5) ;- (UDPHS_EPT)  256 bytes (if possible)
AT91C_UDPHS_EPT_SIZE_512  EQU (0x6) ;- (UDPHS_EPT)  512 bytes (if possible)
AT91C_UDPHS_EPT_SIZE_1024 EQU (0x7) ;- (UDPHS_EPT) 1024 bytes (if possible)
AT91C_UDPHS_EPT_DIR       EQU (0x1 <<  3) ;- (UDPHS_EPT) Endpoint Direction 0:OUT, 1:IN
AT91C_UDPHS_EPT_DIR_OUT   EQU (0x0 <<  3) ;- (UDPHS_EPT) Direction OUT
AT91C_UDPHS_EPT_DIR_IN    EQU (0x1 <<  3) ;- (UDPHS_EPT) Direction IN
AT91C_UDPHS_EPT_TYPE      EQU (0x3 <<  4) ;- (UDPHS_EPT) Endpoint Type
AT91C_UDPHS_EPT_TYPE_CTL_EPT EQU (0x0 <<  4) ;- (UDPHS_EPT) Control endpoint
AT91C_UDPHS_EPT_TYPE_ISO_EPT EQU (0x1 <<  4) ;- (UDPHS_EPT) Isochronous endpoint
AT91C_UDPHS_EPT_TYPE_BUL_EPT EQU (0x2 <<  4) ;- (UDPHS_EPT) Bulk endpoint
AT91C_UDPHS_EPT_TYPE_INT_EPT EQU (0x3 <<  4) ;- (UDPHS_EPT) Interrupt endpoint
AT91C_UDPHS_BK_NUMBER     EQU (0x3 <<  6) ;- (UDPHS_EPT) Number of Banks
AT91C_UDPHS_BK_NUMBER_0   EQU (0x0 <<  6) ;- (UDPHS_EPT) Zero Bank, the EndPoint is not mapped in memory
AT91C_UDPHS_BK_NUMBER_1   EQU (0x1 <<  6) ;- (UDPHS_EPT) One Bank (Bank0)
AT91C_UDPHS_BK_NUMBER_2   EQU (0x2 <<  6) ;- (UDPHS_EPT) Double bank (Ping-Pong : Bank0 / Bank1)
AT91C_UDPHS_BK_NUMBER_3   EQU (0x3 <<  6) ;- (UDPHS_EPT) Triple Bank (Bank0 / Bank1 / Bank2) (if possible)
AT91C_UDPHS_NB_TRANS      EQU (0x3 <<  8) ;- (UDPHS_EPT) Number Of Transaction per Micro-Frame (High-Bandwidth iso only)
AT91C_UDPHS_EPT_MAPD      EQU (0x1 << 31) ;- (UDPHS_EPT) Endpoint Mapped (read only
// - -------- UDPHS_EPTCTLENB : (UDPHS_EPT Offset: 0x4) UDPHS Endpoint Control Enable Register -------- 
AT91C_UDPHS_EPT_ENABL     EQU (0x1 <<  0) ;- (UDPHS_EPT) Endpoint Enable
AT91C_UDPHS_AUTO_VALID    EQU (0x1 <<  1) ;- (UDPHS_EPT) Packet Auto-Valid Enable/Disable
AT91C_UDPHS_INTDIS_DMA    EQU (0x1 <<  3) ;- (UDPHS_EPT) Endpoint Interrupts DMA Request Enable/Disable
AT91C_UDPHS_NYET_DIS      EQU (0x1 <<  4) ;- (UDPHS_EPT) NYET Enable/Disable
AT91C_UDPHS_DATAX_RX      EQU (0x1 <<  6) ;- (UDPHS_EPT) DATAx Interrupt Enable/Disable
AT91C_UDPHS_MDATA_RX      EQU (0x1 <<  7) ;- (UDPHS_EPT) MDATA Interrupt Enabled/Disable
AT91C_UDPHS_ERR_OVFLW     EQU (0x1 <<  8) ;- (UDPHS_EPT) OverFlow Error Interrupt Enable/Disable/Status
AT91C_UDPHS_RX_BK_RDY     EQU (0x1 <<  9) ;- (UDPHS_EPT) Received OUT Data
AT91C_UDPHS_TX_COMPLT     EQU (0x1 << 10) ;- (UDPHS_EPT) Transmitted IN Data Complete Interrupt Enable/Disable or Transmitted IN Data Complete (clear)
AT91C_UDPHS_ERR_TRANS     EQU (0x1 << 11) ;- (UDPHS_EPT) Transaction Error Interrupt Enable/Disable
AT91C_UDPHS_TX_PK_RDY     EQU (0x1 << 11) ;- (UDPHS_EPT) TX Packet Ready Interrupt Enable/Disable
AT91C_UDPHS_RX_SETUP      EQU (0x1 << 12) ;- (UDPHS_EPT) Received SETUP Interrupt Enable/Disable
AT91C_UDPHS_ERR_FL_ISO    EQU (0x1 << 12) ;- (UDPHS_EPT) Error Flow Clear/Interrupt Enable/Disable
AT91C_UDPHS_STALL_SNT     EQU (0x1 << 13) ;- (UDPHS_EPT) Stall Sent Clear
AT91C_UDPHS_ERR_CRISO     EQU (0x1 << 13) ;- (UDPHS_EPT) CRC error / Error NB Trans / Interrupt Enable/Disable
AT91C_UDPHS_NAK_IN        EQU (0x1 << 14) ;- (UDPHS_EPT) NAKIN ERROR FLUSH / Clear / Interrupt Enable/Disable
AT91C_UDPHS_NAK_OUT       EQU (0x1 << 15) ;- (UDPHS_EPT) NAKOUT / Clear / Interrupt Enable/Disable
AT91C_UDPHS_BUSY_BANK     EQU (0x1 << 18) ;- (UDPHS_EPT) Busy Bank Interrupt Enable/Disable
AT91C_UDPHS_SHRT_PCKT     EQU (0x1 << 31) ;- (UDPHS_EPT) Short Packet / Interrupt Enable/Disable
// - -------- UDPHS_EPTCTLDIS : (UDPHS_EPT Offset: 0x8) UDPHS Endpoint Control Disable Register -------- 
AT91C_UDPHS_EPT_DISABL    EQU (0x1 <<  0) ;- (UDPHS_EPT) Endpoint Disable
// - -------- UDPHS_EPTCTL : (UDPHS_EPT Offset: 0xc) UDPHS Endpoint Control Register -------- 
// - -------- UDPHS_EPTSETSTA : (UDPHS_EPT Offset: 0x14) UDPHS Endpoint Set Status Register -------- 
AT91C_UDPHS_FRCESTALL     EQU (0x1 <<  5) ;- (UDPHS_EPT) Stall Handshake Request Set/Clear/Status
AT91C_UDPHS_KILL_BANK     EQU (0x1 <<  9) ;- (UDPHS_EPT) KILL Bank
// - -------- UDPHS_EPTCLRSTA : (UDPHS_EPT Offset: 0x18) UDPHS Endpoint Clear Status Register -------- 
AT91C_UDPHS_TOGGLESQ      EQU (0x1 <<  6) ;- (UDPHS_EPT) Data Toggle Clear
// - -------- UDPHS_EPTSTA : (UDPHS_EPT Offset: 0x1c) UDPHS Endpoint Status Register -------- 
AT91C_UDPHS_TOGGLESQ_STA  EQU (0x3 <<  6) ;- (UDPHS_EPT) Toggle Sequencing
AT91C_UDPHS_TOGGLESQ_STA_00 EQU (0x0 <<  6) ;- (UDPHS_EPT) Data0
AT91C_UDPHS_TOGGLESQ_STA_01 EQU (0x1 <<  6) ;- (UDPHS_EPT) Data1
AT91C_UDPHS_TOGGLESQ_STA_10 EQU (0x2 <<  6) ;- (UDPHS_EPT) Data2 (only for High-Bandwidth Isochronous EndPoint)
AT91C_UDPHS_TOGGLESQ_STA_11 EQU (0x3 <<  6) ;- (UDPHS_EPT) MData (only for High-Bandwidth Isochronous EndPoint)
AT91C_UDPHS_CONTROL_DIR   EQU (0x3 << 16) ;- (UDPHS_EPT) 
AT91C_UDPHS_CONTROL_DIR_00 EQU (0x0 << 16) ;- (UDPHS_EPT) Bank 0
AT91C_UDPHS_CONTROL_DIR_01 EQU (0x1 << 16) ;- (UDPHS_EPT) Bank 1
AT91C_UDPHS_CONTROL_DIR_10 EQU (0x2 << 16) ;- (UDPHS_EPT) Bank 2
AT91C_UDPHS_CONTROL_DIR_11 EQU (0x3 << 16) ;- (UDPHS_EPT) Invalid
AT91C_UDPHS_CURRENT_BANK  EQU (0x3 << 16) ;- (UDPHS_EPT) 
AT91C_UDPHS_CURRENT_BANK_00 EQU (0x0 << 16) ;- (UDPHS_EPT) Bank 0
AT91C_UDPHS_CURRENT_BANK_01 EQU (0x1 << 16) ;- (UDPHS_EPT) Bank 1
AT91C_UDPHS_CURRENT_BANK_10 EQU (0x2 << 16) ;- (UDPHS_EPT) Bank 2
AT91C_UDPHS_CURRENT_BANK_11 EQU (0x3 << 16) ;- (UDPHS_EPT) Invalid
AT91C_UDPHS_BUSY_BANK_STA EQU (0x3 << 18) ;- (UDPHS_EPT) Busy Bank Number
AT91C_UDPHS_BUSY_BANK_STA_00 EQU (0x0 << 18) ;- (UDPHS_EPT) All banks are free
AT91C_UDPHS_BUSY_BANK_STA_01 EQU (0x1 << 18) ;- (UDPHS_EPT) 1 busy bank
AT91C_UDPHS_BUSY_BANK_STA_10 EQU (0x2 << 18) ;- (UDPHS_EPT) 2 busy banks
AT91C_UDPHS_BUSY_BANK_STA_11 EQU (0x3 << 18) ;- (UDPHS_EPT) 3 busy banks (if possible)
AT91C_UDPHS_BYTE_COUNT    EQU (0x7FF << 20) ;- (UDPHS_EPT) UDPHS Byte Count

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS DMA struct
// - *****************************************************************************
// - -------- UDPHS_DMANXTDSC : (UDPHS_DMA Offset: 0x0) UDPHS DMA Next Descriptor Address Register -------- 
AT91C_UDPHS_NXT_DSC_ADD   EQU (0xFFFFFFF <<  4) ;- (UDPHS_DMA) next Channel Descriptor
// - -------- UDPHS_DMAADDRESS : (UDPHS_DMA Offset: 0x4) UDPHS DMA Channel Address Register -------- 
AT91C_UDPHS_BUFF_ADD      EQU (0x0 <<  0) ;- (UDPHS_DMA) starting address of a DMA Channel transfer
// - -------- UDPHS_DMACONTROL : (UDPHS_DMA Offset: 0x8) UDPHS DMA Channel Control Register -------- 
AT91C_UDPHS_CHANN_ENB     EQU (0x1 <<  0) ;- (UDPHS_DMA) Channel Enabled
AT91C_UDPHS_LDNXT_DSC     EQU (0x1 <<  1) ;- (UDPHS_DMA) Load Next Channel Transfer Descriptor Enable
AT91C_UDPHS_END_TR_EN     EQU (0x1 <<  2) ;- (UDPHS_DMA) Buffer Close Input Enable
AT91C_UDPHS_END_B_EN      EQU (0x1 <<  3) ;- (UDPHS_DMA) End of DMA Buffer Packet Validation
AT91C_UDPHS_END_TR_IT     EQU (0x1 <<  4) ;- (UDPHS_DMA) End Of Transfer Interrupt Enable
AT91C_UDPHS_END_BUFFIT    EQU (0x1 <<  5) ;- (UDPHS_DMA) End Of Channel Buffer Interrupt Enable
AT91C_UDPHS_DESC_LD_IT    EQU (0x1 <<  6) ;- (UDPHS_DMA) Descriptor Loaded Interrupt Enable
AT91C_UDPHS_BURST_LCK     EQU (0x1 <<  7) ;- (UDPHS_DMA) Burst Lock Enable
AT91C_UDPHS_BUFF_LENGTH   EQU (0xFFFF << 16) ;- (UDPHS_DMA) Buffer Byte Length (write only)
// - -------- UDPHS_DMASTATUS : (UDPHS_DMA Offset: 0xc) UDPHS DMA Channelx Status Register -------- 
AT91C_UDPHS_CHANN_ACT     EQU (0x1 <<  1) ;- (UDPHS_DMA) 
AT91C_UDPHS_END_TR_ST     EQU (0x1 <<  4) ;- (UDPHS_DMA) 
AT91C_UDPHS_END_BF_ST     EQU (0x1 <<  5) ;- (UDPHS_DMA) 
AT91C_UDPHS_DESC_LDST     EQU (0x1 <<  6) ;- (UDPHS_DMA) 
AT91C_UDPHS_BUFF_COUNT    EQU (0xFFFF << 16) ;- (UDPHS_DMA) 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS High Speed Device Interface
// - *****************************************************************************
// - -------- UDPHS_CTRL : (UDPHS Offset: 0x0) UDPHS Control Register -------- 
AT91C_UDPHS_DEV_ADDR      EQU (0x7F <<  0) ;- (UDPHS) UDPHS Address
AT91C_UDPHS_FADDR_EN      EQU (0x1 <<  7) ;- (UDPHS) Function Address Enable
AT91C_UDPHS_EN_UDPHS      EQU (0x1 <<  8) ;- (UDPHS) UDPHS Enable
AT91C_UDPHS_DETACH        EQU (0x1 <<  9) ;- (UDPHS) Detach Command
AT91C_UDPHS_REWAKEUP      EQU (0x1 << 10) ;- (UDPHS) Send Remote Wake Up
AT91C_UDPHS_PULLD_DIS     EQU (0x1 << 11) ;- (UDPHS) PullDown Disable
// - -------- UDPHS_FNUM : (UDPHS Offset: 0x4) UDPHS Frame Number Register -------- 
AT91C_UDPHS_MICRO_FRAME_NUM EQU (0x7 <<  0) ;- (UDPHS) Micro Frame Number
AT91C_UDPHS_FRAME_NUMBER  EQU (0x7FF <<  3) ;- (UDPHS) Frame Number as defined in the Packet Field Formats
AT91C_UDPHS_FNUM_ERR      EQU (0x1 << 31) ;- (UDPHS) Frame Number CRC Error
// - -------- UDPHS_IEN : (UDPHS Offset: 0x10) UDPHS Interrupt Enable Register -------- 
AT91C_UDPHS_DET_SUSPD     EQU (0x1 <<  1) ;- (UDPHS) Suspend Interrupt Enable/Clear/Status
AT91C_UDPHS_MICRO_SOF     EQU (0x1 <<  2) ;- (UDPHS) Micro-SOF Interrupt Enable/Clear/Status
AT91C_UDPHS_IEN_SOF       EQU (0x1 <<  3) ;- (UDPHS) SOF Interrupt Enable/Clear/Status
AT91C_UDPHS_ENDRESET      EQU (0x1 <<  4) ;- (UDPHS) End Of Reset Interrupt Enable/Clear/Status
AT91C_UDPHS_WAKE_UP       EQU (0x1 <<  5) ;- (UDPHS) Wake Up CPU Interrupt Enable/Clear/Status
AT91C_UDPHS_ENDOFRSM      EQU (0x1 <<  6) ;- (UDPHS) End Of Resume Interrupt Enable/Clear/Status
AT91C_UDPHS_UPSTR_RES     EQU (0x1 <<  7) ;- (UDPHS) Upstream Resume Interrupt Enable/Clear/Status
AT91C_UDPHS_EPT_INT_0     EQU (0x1 <<  8) ;- (UDPHS) Endpoint 0 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_1     EQU (0x1 <<  9) ;- (UDPHS) Endpoint 1 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_2     EQU (0x1 << 10) ;- (UDPHS) Endpoint 2 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_3     EQU (0x1 << 11) ;- (UDPHS) Endpoint 3 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_4     EQU (0x1 << 12) ;- (UDPHS) Endpoint 4 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_5     EQU (0x1 << 13) ;- (UDPHS) Endpoint 5 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_6     EQU (0x1 << 14) ;- (UDPHS) Endpoint 6 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_1     EQU (0x1 << 25) ;- (UDPHS) DMA Channel 1 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_2     EQU (0x1 << 26) ;- (UDPHS) DMA Channel 2 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_3     EQU (0x1 << 27) ;- (UDPHS) DMA Channel 3 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_4     EQU (0x1 << 28) ;- (UDPHS) DMA Channel 4 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_5     EQU (0x1 << 29) ;- (UDPHS) DMA Channel 5 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_6     EQU (0x1 << 30) ;- (UDPHS) DMA Channel 6 Interrupt Enable/Status
// - -------- UDPHS_INTSTA : (UDPHS Offset: 0x14) UDPHS Interrupt Status Register -------- 
AT91C_UDPHS_SPEED         EQU (0x1 <<  0) ;- (UDPHS) Speed Status
// - -------- UDPHS_CLRINT : (UDPHS Offset: 0x18) UDPHS Clear Interrupt Register -------- 
// - -------- UDPHS_EPTRST : (UDPHS Offset: 0x1c) UDPHS Endpoints Reset Register -------- 
AT91C_UDPHS_RST_EPT_0     EQU (0x1 <<  0) ;- (UDPHS) Endpoint Reset 0
AT91C_UDPHS_RST_EPT_1     EQU (0x1 <<  1) ;- (UDPHS) Endpoint Reset 1
AT91C_UDPHS_RST_EPT_2     EQU (0x1 <<  2) ;- (UDPHS) Endpoint Reset 2
AT91C_UDPHS_RST_EPT_3     EQU (0x1 <<  3) ;- (UDPHS) Endpoint Reset 3
AT91C_UDPHS_RST_EPT_4     EQU (0x1 <<  4) ;- (UDPHS) Endpoint Reset 4
AT91C_UDPHS_RST_EPT_5     EQU (0x1 <<  5) ;- (UDPHS) Endpoint Reset 5
AT91C_UDPHS_RST_EPT_6     EQU (0x1 <<  6) ;- (UDPHS) Endpoint Reset 6
// - -------- UDPHS_TSTSOFCNT : (UDPHS Offset: 0xd0) UDPHS Test SOF Counter Register -------- 
AT91C_UDPHS_SOFCNTMAX     EQU (0x3 <<  0) ;- (UDPHS) SOF Counter Max Value
AT91C_UDPHS_SOFCTLOAD     EQU (0x1 <<  7) ;- (UDPHS) SOF Counter Load
// - -------- UDPHS_TSTCNTA : (UDPHS Offset: 0xd4) UDPHS Test A Counter Register -------- 
AT91C_UDPHS_CNTAMAX       EQU (0x7FFF <<  0) ;- (UDPHS) A Counter Max Value
AT91C_UDPHS_CNTALOAD      EQU (0x1 << 15) ;- (UDPHS) A Counter Load
// - -------- UDPHS_TSTCNTB : (UDPHS Offset: 0xd8) UDPHS Test B Counter Register -------- 
AT91C_UDPHS_CNTBMAX       EQU (0x7FFF <<  0) ;- (UDPHS) B Counter Max Value
AT91C_UDPHS_CNTBLOAD      EQU (0x1 << 15) ;- (UDPHS) B Counter Load
// - -------- UDPHS_TSTMODREG : (UDPHS Offset: 0xdc) UDPHS Test Mode Register -------- 
AT91C_UDPHS_TSTMODE       EQU (0x1F <<  1) ;- (UDPHS) UDPHS Core TestModeReg
// - -------- UDPHS_TST : (UDPHS Offset: 0xe0) UDPHS Test Register -------- 
AT91C_UDPHS_SPEED_CFG     EQU (0x3 <<  0) ;- (UDPHS) Speed Configuration
AT91C_UDPHS_SPEED_CFG_NM  EQU (0x0) ;- (UDPHS) Normal Mode
AT91C_UDPHS_SPEED_CFG_RS  EQU (0x1) ;- (UDPHS) Reserved
AT91C_UDPHS_SPEED_CFG_HS  EQU (0x2) ;- (UDPHS) Force High Speed
AT91C_UDPHS_SPEED_CFG_FS  EQU (0x3) ;- (UDPHS) Force Full-Speed
AT91C_UDPHS_TST_J         EQU (0x1 <<  2) ;- (UDPHS) TestJMode
AT91C_UDPHS_TST_K         EQU (0x1 <<  3) ;- (UDPHS) TestKMode
AT91C_UDPHS_TST_PKT       EQU (0x1 <<  4) ;- (UDPHS) TestPacketMode
AT91C_UDPHS_OPMODE2       EQU (0x1 <<  5) ;- (UDPHS) OpMode2
// - -------- UDPHS_RIPPADDRSIZE : (UDPHS Offset: 0xec) UDPHS PADDRSIZE Register -------- 
AT91C_UDPHS_IPPADDRSIZE   EQU (0x0 <<  0) ;- (UDPHS) 2^UDPHSDEV_PADDR_SIZE
// - -------- UDPHS_RIPNAME1 : (UDPHS Offset: 0xf0) UDPHS Name Register -------- 
AT91C_UDPHS_IPNAME1       EQU (0x0 <<  0) ;- (UDPHS) ASCII string HUSB
// - -------- UDPHS_RIPNAME2 : (UDPHS Offset: 0xf4) UDPHS Name Register -------- 
AT91C_UDPHS_IPNAME2       EQU (0x0 <<  0) ;- (UDPHS) ASCII string 2DEV
// - -------- UDPHS_IPFEATURES : (UDPHS Offset: 0xf8) UDPHS Features Register -------- 
AT91C_UDPHS_EPT_NBR_MAX   EQU (0xF <<  0) ;- (UDPHS) Max Number of Endpoints
AT91C_UDPHS_DMA_CHANNEL_NBR EQU (0x7 <<  4) ;- (UDPHS) Number of DMA Channels
AT91C_UDPHS_DMA_B_SIZ     EQU (0x1 <<  7) ;- (UDPHS) DMA Buffer Size
AT91C_UDPHS_DMA_FIFO_WORD_DEPTH EQU (0xF <<  8) ;- (UDPHS) DMA FIFO Depth in words
AT91C_UDPHS_FIFO_MAX_SIZE EQU (0x7 << 12) ;- (UDPHS) DPRAM size
AT91C_UDPHS_BW_DPRAM      EQU (0x1 << 15) ;- (UDPHS) DPRAM byte write capability
AT91C_UDPHS_DATAB16_8     EQU (0x1 << 16) ;- (UDPHS) UTMI DataBus16_8
AT91C_UDPHS_ISO_EPT_1     EQU (0x1 << 17) ;- (UDPHS) Endpoint 1 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_2     EQU (0x1 << 18) ;- (UDPHS) Endpoint 2 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_5     EQU (0x1 << 21) ;- (UDPHS) Endpoint 5 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_6     EQU (0x1 << 22) ;- (UDPHS) Endpoint 6 High Bandwidth Isochronous Capability
// - -------- UDPHS_IPVERSION : (UDPHS Offset: 0xfc) UDPHS Version Register -------- 
AT91C_UDPHS_VERSION_NUM   EQU (0xFFFF <<  0) ;- (UDPHS) Give the IP version
AT91C_UDPHS_METAL_FIX_NUM EQU (0x7 << 16) ;- (UDPHS) Give the number of metal fixes

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR AC97 Controller Interface
// - *****************************************************************************
// - -------- AC97C_MR : (AC97C Offset: 0x8) AC97C Mode Register -------- 
AT91C_AC97C_ENA           EQU (0x1 <<  0) ;- (AC97C) AC97 Controller Global Enable
AT91C_AC97C_WRST          EQU (0x1 <<  1) ;- (AC97C) Warm Reset
AT91C_AC97C_VRA           EQU (0x1 <<  2) ;- (AC97C) Variable RAte (for Data Slots)
// - -------- AC97C_ICA : (AC97C Offset: 0x10) AC97C Input Channel Assignement Register -------- 
AT91C_AC97C_CHID3         EQU (0x7 <<  0) ;- (AC97C) Channel Id for the input slot 3
AT91C_AC97C_CHID3_NONE    EQU (0x0) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID3_CA      EQU (0x1) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID3_CB      EQU (0x2) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID3_CC      EQU (0x3) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID4         EQU (0x7 <<  3) ;- (AC97C) Channel Id for the input slot 4
AT91C_AC97C_CHID4_NONE    EQU (0x0 <<  3) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID4_CA      EQU (0x1 <<  3) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID4_CB      EQU (0x2 <<  3) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID4_CC      EQU (0x3 <<  3) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID5         EQU (0x7 <<  6) ;- (AC97C) Channel Id for the input slot 5
AT91C_AC97C_CHID5_NONE    EQU (0x0 <<  6) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID5_CA      EQU (0x1 <<  6) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID5_CB      EQU (0x2 <<  6) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID5_CC      EQU (0x3 <<  6) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID6         EQU (0x7 <<  9) ;- (AC97C) Channel Id for the input slot 6
AT91C_AC97C_CHID6_NONE    EQU (0x0 <<  9) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID6_CA      EQU (0x1 <<  9) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID6_CB      EQU (0x2 <<  9) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID6_CC      EQU (0x3 <<  9) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID7         EQU (0x7 << 12) ;- (AC97C) Channel Id for the input slot 7
AT91C_AC97C_CHID7_NONE    EQU (0x0 << 12) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID7_CA      EQU (0x1 << 12) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID7_CB      EQU (0x2 << 12) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID7_CC      EQU (0x3 << 12) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID8         EQU (0x7 << 15) ;- (AC97C) Channel Id for the input slot 8
AT91C_AC97C_CHID8_NONE    EQU (0x0 << 15) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID8_CA      EQU (0x1 << 15) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID8_CB      EQU (0x2 << 15) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID8_CC      EQU (0x3 << 15) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID9         EQU (0x7 << 18) ;- (AC97C) Channel Id for the input slot 9
AT91C_AC97C_CHID9_NONE    EQU (0x0 << 18) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID9_CA      EQU (0x1 << 18) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID9_CB      EQU (0x2 << 18) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID9_CC      EQU (0x3 << 18) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID10        EQU (0x7 << 21) ;- (AC97C) Channel Id for the input slot 10
AT91C_AC97C_CHID10_NONE   EQU (0x0 << 21) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID10_CA     EQU (0x1 << 21) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID10_CB     EQU (0x2 << 21) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID10_CC     EQU (0x3 << 21) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID11        EQU (0x7 << 24) ;- (AC97C) Channel Id for the input slot 11
AT91C_AC97C_CHID11_NONE   EQU (0x0 << 24) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID11_CA     EQU (0x1 << 24) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID11_CB     EQU (0x2 << 24) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID11_CC     EQU (0x3 << 24) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID12        EQU (0x7 << 27) ;- (AC97C) Channel Id for the input slot 12
AT91C_AC97C_CHID12_NONE   EQU (0x0 << 27) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID12_CA     EQU (0x1 << 27) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID12_CB     EQU (0x2 << 27) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID12_CC     EQU (0x3 << 27) ;- (AC97C) Channel C data will be transmitted during this slot
// - -------- AC97C_OCA : (AC97C Offset: 0x14) AC97C Output Channel Assignement Register -------- 
// - -------- AC97C_CARHR : (AC97C Offset: 0x20) AC97C Channel A Receive Holding Register -------- 
AT91C_AC97C_RDATA         EQU (0xFFFFF <<  0) ;- (AC97C) Receive data
// - -------- AC97C_CATHR : (AC97C Offset: 0x24) AC97C Channel A Transmit Holding Register -------- 
AT91C_AC97C_TDATA         EQU (0xFFFFF <<  0) ;- (AC97C) Transmit data
// - -------- AC97C_CASR : (AC97C Offset: 0x28) AC97C Channel A Status Register -------- 
AT91C_AC97C_TXRDY         EQU (0x1 <<  0) ;- (AC97C) 
AT91C_AC97C_TXEMPTY       EQU (0x1 <<  1) ;- (AC97C) 
AT91C_AC97C_UNRUN         EQU (0x1 <<  2) ;- (AC97C) 
AT91C_AC97C_RXRDY         EQU (0x1 <<  4) ;- (AC97C) 
AT91C_AC97C_OVRUN         EQU (0x1 <<  5) ;- (AC97C) 
AT91C_AC97C_ENDTX         EQU (0x1 << 10) ;- (AC97C) 
AT91C_AC97C_TXBUFE        EQU (0x1 << 11) ;- (AC97C) 
AT91C_AC97C_ENDRX         EQU (0x1 << 14) ;- (AC97C) 
AT91C_AC97C_RXBUFF        EQU (0x1 << 15) ;- (AC97C) 
// - -------- AC97C_CAMR : (AC97C Offset: 0x2c) AC97C Channel A Mode Register -------- 
AT91C_AC97C_SIZE          EQU (0x3 << 16) ;- (AC97C) 
AT91C_AC97C_SIZE_20_BITS  EQU (0x0 << 16) ;- (AC97C) Data size is 20 bits
AT91C_AC97C_SIZE_18_BITS  EQU (0x1 << 16) ;- (AC97C) Data size is 18 bits
AT91C_AC97C_SIZE_16_BITS  EQU (0x2 << 16) ;- (AC97C) Data size is 16 bits
AT91C_AC97C_SIZE_10_BITS  EQU (0x3 << 16) ;- (AC97C) Data size is 10 bits
AT91C_AC97C_CEM           EQU (0x1 << 18) ;- (AC97C) 
AT91C_AC97C_CEN           EQU (0x1 << 21) ;- (AC97C) 
AT91C_AC97C_PDCEN         EQU (0x1 << 22) ;- (AC97C) 
// - -------- AC97C_CBRHR : (AC97C Offset: 0x30) AC97C Channel B Receive Holding Register -------- 
// - -------- AC97C_CBTHR : (AC97C Offset: 0x34) AC97C Channel B Transmit Holding Register -------- 
// - -------- AC97C_CBSR : (AC97C Offset: 0x38) AC97C Channel B Status Register -------- 
// - -------- AC97C_CBMR : (AC97C Offset: 0x3c) AC97C Channel B Mode Register -------- 
// - -------- AC97C_CORHR : (AC97C Offset: 0x40) AC97C Codec Channel Receive Holding Register -------- 
AT91C_AC97C_SDATA         EQU (0xFFFF <<  0) ;- (AC97C) Status Data
// - -------- AC97C_COTHR : (AC97C Offset: 0x44) AC97C Codec Channel Transmit Holding Register -------- 
AT91C_AC97C_CDATA         EQU (0xFFFF <<  0) ;- (AC97C) Command Data
AT91C_AC97C_CADDR         EQU (0x7F << 16) ;- (AC97C) COdec control register index
AT91C_AC97C_READ          EQU (0x1 << 23) ;- (AC97C) Read/Write command
// - -------- AC97C_COSR : (AC97C Offset: 0x48) AC97C CODEC Status Register -------- 
// - -------- AC97C_COMR : (AC97C Offset: 0x4c) AC97C CODEC Mode Register -------- 
// - -------- AC97C_SR : (AC97C Offset: 0x50) AC97C Status Register -------- 
AT91C_AC97C_SOF           EQU (0x1 <<  0) ;- (AC97C) 
AT91C_AC97C_WKUP          EQU (0x1 <<  1) ;- (AC97C) 
AT91C_AC97C_COEVT         EQU (0x1 <<  2) ;- (AC97C) 
AT91C_AC97C_CAEVT         EQU (0x1 <<  3) ;- (AC97C) 
AT91C_AC97C_CBEVT         EQU (0x1 <<  4) ;- (AC97C) 
// - -------- AC97C_IER : (AC97C Offset: 0x54) AC97C Interrupt Enable Register -------- 
// - -------- AC97C_IDR : (AC97C Offset: 0x58) AC97C Interrupt Disable Register -------- 
// - -------- AC97C_IMR : (AC97C Offset: 0x5c) AC97C Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR LCD Controller
// - *****************************************************************************
// - -------- LCDC_FRMP1 : (LCDC Offset: 0x8) DMA Frame Pointer 1 Register -------- 
AT91C_LCDC_FRMPT1         EQU (0x3FFFFF <<  0) ;- (LCDC) Frame Pointer Address 1
// - -------- LCDC_FRMP2 : (LCDC Offset: 0xc) DMA Frame Pointer 2 Register -------- 
AT91C_LCDC_FRMPT2         EQU (0x1FFFFF <<  0) ;- (LCDC) Frame Pointer Address 2
// - -------- LCDC_FRMCFG : (LCDC Offset: 0x18) DMA Frame Config Register -------- 
AT91C_LCDC_FRSIZE         EQU (0x7FFFFF <<  0) ;- (LCDC) FRAME SIZE
AT91C_LCDC_BLENGTH        EQU (0x7F << 24) ;- (LCDC) BURST LENGTH
// - -------- LCDC_DMACON : (LCDC Offset: 0x1c) DMA Control Register -------- 
AT91C_LCDC_DMAEN          EQU (0x1 <<  0) ;- (LCDC) DAM Enable
AT91C_LCDC_DMARST         EQU (0x1 <<  1) ;- (LCDC) DMA Reset (WO)
AT91C_LCDC_DMABUSY        EQU (0x1 <<  2) ;- (LCDC) DMA Reset (WO)
AT91C_LCDC_DMAUPDT        EQU (0x1 <<  3) ;- (LCDC) DMA Configuration Update
AT91C_LCDC_DMA2DEN        EQU (0x1 <<  4) ;- (LCDC) 2D Addressing Enable
// - -------- LCDC_DMA2DCFG : (LCDC Offset: 0x20) DMA 2D addressing configuration Register -------- 
AT91C_LCDC_ADDRINC        EQU (0xFFFF <<  0) ;- (LCDC) Number of 32b words that the DMA must jump when going to the next line
AT91C_LCDC_PIXELOFF       EQU (0x1F << 24) ;- (LCDC) Offset (in bits) of the first pixel of the screen in the memory word which contain it
// - -------- LCDC_LCDCON1 : (LCDC Offset: 0x800) LCD Control 1 Register -------- 
AT91C_LCDC_BYPASS         EQU (0x1 <<  0) ;- (LCDC) Bypass lcd_pccklk divider
AT91C_LCDC_CLKVAL         EQU (0x1FF << 12) ;- (LCDC) 9-bit Divider for pixel clock frequency
AT91C_LCDC_LINCNT         EQU (0x7FF << 21) ;- (LCDC) Line Counter (RO)
// - -------- LCDC_LCDCON2 : (LCDC Offset: 0x804) LCD Control 2 Register -------- 
AT91C_LCDC_DISTYPE        EQU (0x3 <<  0) ;- (LCDC) Display Type
AT91C_LCDC_DISTYPE_STNMONO EQU (0x0) ;- (LCDC) STN Mono
AT91C_LCDC_DISTYPE_STNCOLOR EQU (0x1) ;- (LCDC) STN Color
AT91C_LCDC_DISTYPE_TFT    EQU (0x2) ;- (LCDC) TFT
AT91C_LCDC_SCANMOD        EQU (0x1 <<  2) ;- (LCDC) Scan Mode
AT91C_LCDC_SCANMOD_SINGLESCAN EQU (0x0 <<  2) ;- (LCDC) Single Scan
AT91C_LCDC_SCANMOD_DUALSCAN EQU (0x1 <<  2) ;- (LCDC) Dual Scan
AT91C_LCDC_IFWIDTH        EQU (0x3 <<  3) ;- (LCDC) Interface Width
AT91C_LCDC_IFWIDTH_FOURBITSWIDTH EQU (0x0 <<  3) ;- (LCDC) 4 Bits
AT91C_LCDC_IFWIDTH_EIGTHBITSWIDTH EQU (0x1 <<  3) ;- (LCDC) 8 Bits
AT91C_LCDC_IFWIDTH_SIXTEENBITSWIDTH EQU (0x2 <<  3) ;- (LCDC) 16 Bits
AT91C_LCDC_PIXELSIZE      EQU (0x7 <<  5) ;- (LCDC) Bits per pixel
AT91C_LCDC_PIXELSIZE_ONEBITSPERPIXEL EQU (0x0 <<  5) ;- (LCDC) 1 Bits
AT91C_LCDC_PIXELSIZE_TWOBITSPERPIXEL EQU (0x1 <<  5) ;- (LCDC) 2 Bits
AT91C_LCDC_PIXELSIZE_FOURBITSPERPIXEL EQU (0x2 <<  5) ;- (LCDC) 4 Bits
AT91C_LCDC_PIXELSIZE_EIGTHBITSPERPIXEL EQU (0x3 <<  5) ;- (LCDC) 8 Bits
AT91C_LCDC_PIXELSIZE_SIXTEENBITSPERPIXEL EQU (0x4 <<  5) ;- (LCDC) 16 Bits
AT91C_LCDC_PIXELSIZE_TWENTYFOURBITSPERPIXEL EQU (0x5 <<  5) ;- (LCDC) 24 Bits
AT91C_LCDC_INVVD          EQU (0x1 <<  8) ;- (LCDC) lcd datas polarity
AT91C_LCDC_INVVD_NORMALPOL EQU (0x0 <<  8) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVVD_INVERTEDPOL EQU (0x1 <<  8) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVFRAME       EQU (0x1 <<  9) ;- (LCDC) lcd vsync polarity
AT91C_LCDC_INVFRAME_NORMALPOL EQU (0x0 <<  9) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVFRAME_INVERTEDPOL EQU (0x1 <<  9) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVLINE        EQU (0x1 << 10) ;- (LCDC) lcd hsync polarity
AT91C_LCDC_INVLINE_NORMALPOL EQU (0x0 << 10) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVLINE_INVERTEDPOL EQU (0x1 << 10) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVCLK         EQU (0x1 << 11) ;- (LCDC) lcd pclk polarity
AT91C_LCDC_INVCLK_NORMALPOL EQU (0x0 << 11) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVCLK_INVERTEDPOL EQU (0x1 << 11) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVDVAL        EQU (0x1 << 12) ;- (LCDC) lcd dval polarity
AT91C_LCDC_INVDVAL_NORMALPOL EQU (0x0 << 12) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVDVAL_INVERTEDPOL EQU (0x1 << 12) ;- (LCDC) Inverted Polarity
AT91C_LCDC_CLKMOD         EQU (0x1 << 15) ;- (LCDC) lcd pclk Mode
AT91C_LCDC_CLKMOD_ACTIVEONLYDISP EQU (0x0 << 15) ;- (LCDC) Active during display period
AT91C_LCDC_CLKMOD_ALWAYSACTIVE EQU (0x1 << 15) ;- (LCDC) Always Active
AT91C_LCDC_MEMOR          EQU (0x3 << 30) ;- (LCDC) Memory Ordering Format
AT91C_LCDC_MEMOR_BIGIND   EQU (0x0 << 30) ;- (LCDC) Big Endian
AT91C_LCDC_MEMOR_LITTLEIND EQU (0x2 << 30) ;- (LCDC) Little Endian
// - -------- LCDC_TIM1 : (LCDC Offset: 0x808) LCDC Timing Config 1 Register -------- 
AT91C_LCDC_VFP            EQU (0xFF <<  0) ;- (LCDC) Vertical Front Porch
AT91C_LCDC_VBP            EQU (0xFF <<  8) ;- (LCDC) Vertical Back Porch
AT91C_LCDC_VPW            EQU (0x3F << 16) ;- (LCDC) Vertical Synchronization Pulse Width
AT91C_LCDC_VHDLY          EQU (0xF << 24) ;- (LCDC) Vertical to Horizontal Delay
// - -------- LCDC_TIM2 : (LCDC Offset: 0x80c) LCDC Timing Config 2 Register -------- 
AT91C_LCDC_HBP            EQU (0xFF <<  0) ;- (LCDC) Horizontal Back Porch
AT91C_LCDC_HPW            EQU (0x3F <<  8) ;- (LCDC) Horizontal Synchronization Pulse Width
AT91C_LCDC_HFP            EQU (0x7FF << 21) ;- (LCDC) Horizontal Front Porch
// - -------- LCDC_LCDFRCFG : (LCDC Offset: 0x810) LCD Frame Config Register -------- 
AT91C_LCDC_LINEVAL        EQU (0x7FF <<  0) ;- (LCDC) Vertical Size of LCD Module
AT91C_LCDC_HOZVAL         EQU (0x7FF << 21) ;- (LCDC) Horizontal Size of LCD Module
// - -------- LCDC_FIFO : (LCDC Offset: 0x814) LCD FIFO Register -------- 
AT91C_LCDC_FIFOTH         EQU (0xFFFF <<  0) ;- (LCDC) FIFO Threshold
// - -------- LCDC_MVAL : (LCDC Offset: 0x818) LCD Mode Toggle Rate Value Register -------- 
AT91C_LCDC_MVALUE         EQU (0xFF <<  0) ;- (LCDC) Toggle Rate Value
AT91C_LCDC_MMODE          EQU (0x1 << 31) ;- (LCDC) Toggle Rate Sel
AT91C_LCDC_MMODE_EACHFRAME EQU (0x0 << 31) ;- (LCDC) Each Frame
AT91C_LCDC_MMODE_MVALDEFINED EQU (0x1 << 31) ;- (LCDC) Defined by MVAL
// - -------- LCDC_DP1_2 : (LCDC Offset: 0x81c) Dithering Pattern 1/2 -------- 
AT91C_LCDC_DP1_2_FIELD    EQU (0xFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP4_7 : (LCDC Offset: 0x820) Dithering Pattern 4/7 -------- 
AT91C_LCDC_DP4_7_FIELD    EQU (0xFFFFFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP3_5 : (LCDC Offset: 0x824) Dithering Pattern 3/5 -------- 
AT91C_LCDC_DP3_5_FIELD    EQU (0xFFFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP2_3 : (LCDC Offset: 0x828) Dithering Pattern 2/3 -------- 
AT91C_LCDC_DP2_3_FIELD    EQU (0xFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP5_7 : (LCDC Offset: 0x82c) Dithering Pattern 5/7 -------- 
AT91C_LCDC_DP5_7_FIELD    EQU (0xFFFFFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP3_4 : (LCDC Offset: 0x830) Dithering Pattern 3/4 -------- 
AT91C_LCDC_DP3_4_FIELD    EQU (0xFFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP4_5 : (LCDC Offset: 0x834) Dithering Pattern 4/5 -------- 
AT91C_LCDC_DP4_5_FIELD    EQU (0xFFFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_DP6_7 : (LCDC Offset: 0x838) Dithering Pattern 6/7 -------- 
AT91C_LCDC_DP6_7_FIELD    EQU (0xFFFFFFF <<  0) ;- (LCDC) Ratio
// - -------- LCDC_PWRCON : (LCDC Offset: 0x83c) LCDC Power Control Register -------- 
AT91C_LCDC_PWR            EQU (0x1 <<  0) ;- (LCDC) LCD Module Power Control
AT91C_LCDC_GUARDT         EQU (0x7F <<  1) ;- (LCDC) Delay in Frame Period
AT91C_LCDC_BUSY           EQU (0x1 << 31) ;- (LCDC) Read Only : 1 indicates that LCDC is busy
AT91C_LCDC_BUSY_LCDNOTBUSY EQU (0x0 << 31) ;- (LCDC) LCD is Not Busy
AT91C_LCDC_BUSY_LCDBUSY   EQU (0x1 << 31) ;- (LCDC) LCD is Busy
// - -------- LCDC_CTRSTCON : (LCDC Offset: 0x840) LCDC Contrast Control Register -------- 
AT91C_LCDC_PS             EQU (0x3 <<  0) ;- (LCDC) LCD Contrast Counter Prescaler
AT91C_LCDC_PS_NOTDIVIDED  EQU (0x0) ;- (LCDC) Counter Freq is System Freq.
AT91C_LCDC_PS_DIVIDEDBYTWO EQU (0x1) ;- (LCDC) Counter Freq is System Freq divided by 2.
AT91C_LCDC_PS_DIVIDEDBYFOUR EQU (0x2) ;- (LCDC) Counter Freq is System Freq divided by 4.
AT91C_LCDC_PS_DIVIDEDBYEIGHT EQU (0x3) ;- (LCDC) Counter Freq is System Freq divided by 8.
AT91C_LCDC_POL            EQU (0x1 <<  2) ;- (LCDC) Polarity of output Pulse
AT91C_LCDC_POL_NEGATIVEPULSE EQU (0x0 <<  2) ;- (LCDC) Negative Pulse
AT91C_LCDC_POL_POSITIVEPULSE EQU (0x1 <<  2) ;- (LCDC) Positive Pulse
AT91C_LCDC_ENA            EQU (0x1 <<  3) ;- (LCDC) PWM generator Control
AT91C_LCDC_ENA_PWMGEMDISABLED EQU (0x0 <<  3) ;- (LCDC) PWM Generator Disabled
AT91C_LCDC_ENA_PWMGEMENABLED EQU (0x1 <<  3) ;- (LCDC) PWM Generator Disabled
// - -------- LCDC_CTRSTVAL : (LCDC Offset: 0x844) Contrast Value Register -------- 
AT91C_LCDC_CVAL           EQU (0xFF <<  0) ;- (LCDC) PWM Compare Value
// - -------- LCDC_IER : (LCDC Offset: 0x848) LCDC Interrupt Enable Register -------- 
AT91C_LCDC_LNI            EQU (0x1 <<  0) ;- (LCDC) Line Interrupt
AT91C_LCDC_LSTLNI         EQU (0x1 <<  1) ;- (LCDC) Last Line Interrupt
AT91C_LCDC_EOFI           EQU (0x1 <<  2) ;- (LCDC) End Of Frame Interrupt
AT91C_LCDC_UFLWI          EQU (0x1 <<  4) ;- (LCDC) FIFO Underflow Interrupt
AT91C_LCDC_OWRI           EQU (0x1 <<  5) ;- (LCDC) Over Write Interrupt
AT91C_LCDC_MERI           EQU (0x1 <<  6) ;- (LCDC) Memory Error  Interrupt
// - -------- LCDC_IDR : (LCDC Offset: 0x84c) LCDC Interrupt Disable Register -------- 
// - -------- LCDC_IMR : (LCDC Offset: 0x850) LCDC Interrupt Mask Register -------- 
// - -------- LCDC_ISR : (LCDC Offset: 0x854) LCDC Interrupt Status Register -------- 
// - -------- LCDC_ICR : (LCDC Offset: 0x858) LCDC Interrupt Clear Register -------- 
// - -------- LCDC_GPR : (LCDC Offset: 0x85c) LCDC General Purpose Register -------- 
AT91C_LCDC_GPRBUS         EQU (0xFF <<  0) ;- (LCDC) 8 bits available
// - -------- LCDC_ITR : (LCDC Offset: 0x860) Interrupts Test Register -------- 
// - -------- LCDC_IRR : (LCDC Offset: 0x864) Interrupts Raw Status Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR HDMA Channel structure
// - *****************************************************************************
// - -------- HDMA_SADDR : (HDMA_CH Offset: 0x0)  -------- 
AT91C_SADDR               EQU (0x0 <<  0) ;- (HDMA_CH) 
// - -------- HDMA_DADDR : (HDMA_CH Offset: 0x4)  -------- 
AT91C_DADDR               EQU (0x0 <<  0) ;- (HDMA_CH) 
// - -------- HDMA_DSCR : (HDMA_CH Offset: 0x8)  -------- 
AT91C_HDMA_DSCR_IF        EQU (0x3 <<  0) ;- (HDMA_CH) Select AHB-Lite Interface for current channel
AT91C_HDMA_DSCR_IF_0      EQU (0x0) ;- (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 0.
AT91C_HDMA_DSCR_IF_1      EQU (0x1) ;- (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 1.
AT91C_HDMA_DSCR_IF_2      EQU (0x2) ;- (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 2.
AT91C_HDMA_DSCR_IF_3      EQU (0x3) ;- (HDMA_CH) The Buffer Transfer descriptor is fetched via AHB-Lite Interface 3.
AT91C_HDMA_DSCR           EQU (0x3FFFFFFF <<  2) ;- (HDMA_CH) Buffer Transfer descriptor address. This address is word aligned.
// - -------- HDMA_CTRLA : (HDMA_CH Offset: 0xc)  -------- 
AT91C_HDMA_BTSIZE         EQU (0xFFFF <<  0) ;- (HDMA_CH) Buffer Transfer Size.
AT91C_HDMA_SCSIZE         EQU (0x7 << 16) ;- (HDMA_CH) Source Chunk Transfer Size.
AT91C_HDMA_SCSIZE_1       EQU (0x0 << 16) ;- (HDMA_CH) 1.
AT91C_HDMA_SCSIZE_4       EQU (0x1 << 16) ;- (HDMA_CH) 4.
AT91C_HDMA_SCSIZE_8       EQU (0x2 << 16) ;- (HDMA_CH) 8.
AT91C_HDMA_SCSIZE_16      EQU (0x3 << 16) ;- (HDMA_CH) 16.
AT91C_HDMA_SCSIZE_32      EQU (0x4 << 16) ;- (HDMA_CH) 32.
AT91C_HDMA_SCSIZE_64      EQU (0x5 << 16) ;- (HDMA_CH) 64.
AT91C_HDMA_SCSIZE_128     EQU (0x6 << 16) ;- (HDMA_CH) 128.
AT91C_HDMA_SCSIZE_256     EQU (0x7 << 16) ;- (HDMA_CH) 256.
AT91C_HDMA_DCSIZE         EQU (0x7 << 20) ;- (HDMA_CH) Destination Chunk Transfer Size
AT91C_HDMA_DCSIZE_1       EQU (0x0 << 20) ;- (HDMA_CH) 1.
AT91C_HDMA_DCSIZE_4       EQU (0x1 << 20) ;- (HDMA_CH) 4.
AT91C_HDMA_DCSIZE_8       EQU (0x2 << 20) ;- (HDMA_CH) 8.
AT91C_HDMA_DCSIZE_16      EQU (0x3 << 20) ;- (HDMA_CH) 16.
AT91C_HDMA_DCSIZE_32      EQU (0x4 << 20) ;- (HDMA_CH) 32.
AT91C_HDMA_DCSIZE_64      EQU (0x5 << 20) ;- (HDMA_CH) 64.
AT91C_HDMA_DCSIZE_128     EQU (0x6 << 20) ;- (HDMA_CH) 128.
AT91C_HDMA_DCSIZE_256     EQU (0x7 << 20) ;- (HDMA_CH) 256.
AT91C_HDMA_SRC_WIDTH      EQU (0x3 << 24) ;- (HDMA_CH) Source Single Transfer Size
AT91C_HDMA_SRC_WIDTH_BYTE EQU (0x0 << 24) ;- (HDMA_CH) BYTE.
AT91C_HDMA_SRC_WIDTH_HALFWORD EQU (0x1 << 24) ;- (HDMA_CH) HALF-WORD.
AT91C_HDMA_SRC_WIDTH_WORD EQU (0x2 << 24) ;- (HDMA_CH) WORD.
AT91C_HDMA_DST_WIDTH      EQU (0x3 << 28) ;- (HDMA_CH) Destination Single Transfer Size
AT91C_HDMA_DST_WIDTH_BYTE EQU (0x0 << 28) ;- (HDMA_CH) BYTE.
AT91C_HDMA_DST_WIDTH_HALFWORD EQU (0x1 << 28) ;- (HDMA_CH) HALF-WORD.
AT91C_HDMA_DST_WIDTH_WORD EQU (0x2 << 28) ;- (HDMA_CH) WORD.
AT91C_HDMA_DONE           EQU (0x1 << 31) ;- (HDMA_CH) 
// - -------- HDMA_CTRLB : (HDMA_CH Offset: 0x10)  -------- 
AT91C_HDMA_SIF            EQU (0x3 <<  0) ;- (HDMA_CH) Source Interface Selection Field.
AT91C_HDMA_SIF_0          EQU (0x0) ;- (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 0.
AT91C_HDMA_SIF_1          EQU (0x1) ;- (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 1.
AT91C_HDMA_SIF_2          EQU (0x2) ;- (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 2.
AT91C_HDMA_SIF_3          EQU (0x3) ;- (HDMA_CH) The Source Transfer is done via AHB-Lite Interface 3.
AT91C_HDMA_DIF            EQU (0x3 <<  4) ;- (HDMA_CH) Destination Interface Selection Field.
AT91C_HDMA_DIF_0          EQU (0x0 <<  4) ;- (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 0.
AT91C_HDMA_DIF_1          EQU (0x1 <<  4) ;- (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 1.
AT91C_HDMA_DIF_2          EQU (0x2 <<  4) ;- (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 2.
AT91C_HDMA_DIF_3          EQU (0x3 <<  4) ;- (HDMA_CH) The Destination Transfer is done via AHB-Lite Interface 3.
AT91C_HDMA_SRC_PIP        EQU (0x1 <<  8) ;- (HDMA_CH) Source Picture-in-Picture Mode
AT91C_HDMA_SRC_PIP_DISABLE EQU (0x0 <<  8) ;- (HDMA_CH) Source Picture-in-Picture mode is disabled.
AT91C_HDMA_SRC_PIP_ENABLE EQU (0x1 <<  8) ;- (HDMA_CH) Source Picture-in-Picture mode is enabled.
AT91C_HDMA_DST_PIP        EQU (0x1 << 12) ;- (HDMA_CH) Destination Picture-in-Picture Mode
AT91C_HDMA_DST_PIP_DISABLE EQU (0x0 << 12) ;- (HDMA_CH) Destination Picture-in-Picture mode is disabled.
AT91C_HDMA_DST_PIP_ENABLE EQU (0x1 << 12) ;- (HDMA_CH) Destination Picture-in-Picture mode is enabled.
AT91C_HDMA_SRC_DSCR       EQU (0x1 << 16) ;- (HDMA_CH) Source Buffer Descriptor Fetch operation
AT91C_HDMA_SRC_DSCR_FETCH_FROM_MEM EQU (0x0 << 16) ;- (HDMA_CH) Source address is updated when the descriptor is fetched from the memory.
AT91C_HDMA_SRC_DSCR_FETCH_DISABLE EQU (0x1 << 16) ;- (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the Source.
AT91C_HDMA_DST_DSCR       EQU (0x1 << 20) ;- (HDMA_CH) Destination Buffer Descriptor operation
AT91C_HDMA_DST_DSCR_FETCH_FROM_MEM EQU (0x0 << 20) ;- (HDMA_CH) Destination address is updated when the descriptor is fetched from the memory.
AT91C_HDMA_DST_DSCR_FETCH_DISABLE EQU (0x1 << 20) ;- (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the destination.
AT91C_HDMA_FC             EQU (0x7 << 21) ;- (HDMA_CH) This field defines which devices controls the size of the buffer transfer, also referred as to the Flow Controller.
AT91C_HDMA_FC_MEM2MEM     EQU (0x0 << 21) ;- (HDMA_CH) Memory-to-Memory (DMA Controller).
AT91C_HDMA_FC_MEM2PER     EQU (0x1 << 21) ;- (HDMA_CH) Memory-to-Peripheral (DMA Controller).
AT91C_HDMA_FC_PER2MEM     EQU (0x2 << 21) ;- (HDMA_CH) Peripheral-to-Memory (DMA Controller).
AT91C_HDMA_FC_PER2PER     EQU (0x3 << 21) ;- (HDMA_CH) Peripheral-to-Peripheral (DMA Controller).
AT91C_HDMA_FC_PER2MEM_PER EQU (0x4 << 21) ;- (HDMA_CH) Peripheral-to-Memory (Peripheral).
AT91C_HDMA_FC_MEM2PER_PER EQU (0x5 << 21) ;- (HDMA_CH) Memory-to-Peripheral (Peripheral).
AT91C_HDMA_FC_PER2PER_PER EQU (0x6 << 21) ;- (HDMA_CH) Peripheral-to-Peripheral (Source Peripheral).
AT91C_HDMA_SRC_ADDRESS_MODE EQU (0x3 << 24) ;- (HDMA_CH) Type of addressing mode
AT91C_HDMA_SRC_ADDRESS_MODE_INCR EQU (0x0 << 24) ;- (HDMA_CH) Incrementing Mode.
AT91C_HDMA_SRC_ADDRESS_MODE_DECR EQU (0x1 << 24) ;- (HDMA_CH) Decrementing Mode.
AT91C_HDMA_SRC_ADDRESS_MODE_FIXED EQU (0x2 << 24) ;- (HDMA_CH) Fixed Mode.
AT91C_HDMA_DST_ADDRESS_MODE EQU (0x3 << 28) ;- (HDMA_CH) Type of addressing mode
AT91C_HDMA_DST_ADDRESS_MODE_INCR EQU (0x0 << 28) ;- (HDMA_CH) Incrementing Mode.
AT91C_HDMA_DST_ADDRESS_MODE_DECR EQU (0x1 << 28) ;- (HDMA_CH) Decrementing Mode.
AT91C_HDMA_DST_ADDRESS_MODE_FIXED EQU (0x2 << 28) ;- (HDMA_CH) Fixed Mode.
AT91C_HDMA_AUTO           EQU (0x1 << 31) ;- (HDMA_CH) Automatic multiple buffer transfer enable
AT91C_HDMA_AUTO_DISABLE   EQU (0x0 << 31) ;- (HDMA_CH) Automatic multiple buffer transfer is disabled.
AT91C_HDMA_AUTO_ENABLE    EQU (0x1 << 31) ;- (HDMA_CH) Automatic multiple buffer transfer is enabled. This enables replay mode or contiguous mode when several buffers are transferred.
// - -------- HDMA_CFG : (HDMA_CH Offset: 0x14)  -------- 
AT91C_HDMA_SRC_PER        EQU (0xF <<  0) ;- (HDMA_CH) Channel Source Request is associated with peripheral identifier coded SRC_PER handshaking interface.
AT91C_HDMA_SRC_PER_0      EQU (0x0) ;- (HDMA_CH) HW Handshaking Interface number 0.
AT91C_HDMA_SRC_PER_1      EQU (0x1) ;- (HDMA_CH) HW Handshaking Interface number 1.
AT91C_HDMA_SRC_PER_2      EQU (0x2) ;- (HDMA_CH) HW Handshaking Interface number 2.
AT91C_HDMA_SRC_PER_3      EQU (0x3) ;- (HDMA_CH) HW Handshaking Interface number 3.
AT91C_HDMA_SRC_PER_4      EQU (0x4) ;- (HDMA_CH) HW Handshaking Interface number 4.
AT91C_HDMA_SRC_PER_5      EQU (0x5) ;- (HDMA_CH) HW Handshaking Interface number 5.
AT91C_HDMA_SRC_PER_6      EQU (0x6) ;- (HDMA_CH) HW Handshaking Interface number 6.
AT91C_HDMA_SRC_PER_7      EQU (0x7) ;- (HDMA_CH) HW Handshaking Interface number 7.
AT91C_HDMA_SRC_PER_8      EQU (0x8) ;- (HDMA_CH) HW Handshaking Interface number 8.
AT91C_HDMA_SRC_PER_9      EQU (0x9) ;- (HDMA_CH) HW Handshaking Interface number 9.
AT91C_HDMA_SRC_PER_10     EQU (0xA) ;- (HDMA_CH) HW Handshaking Interface number 10.
AT91C_HDMA_SRC_PER_11     EQU (0xB) ;- (HDMA_CH) HW Handshaking Interface number 11.
AT91C_HDMA_SRC_PER_12     EQU (0xC) ;- (HDMA_CH) HW Handshaking Interface number 12.
AT91C_HDMA_SRC_PER_13     EQU (0xD) ;- (HDMA_CH) HW Handshaking Interface number 13.
AT91C_HDMA_SRC_PER_14     EQU (0xE) ;- (HDMA_CH) HW Handshaking Interface number 14.
AT91C_HDMA_SRC_PER_15     EQU (0xF) ;- (HDMA_CH) HW Handshaking Interface number 15.
AT91C_HDMA_DST_PER        EQU (0xF <<  4) ;- (HDMA_CH) Channel Destination Request is associated with peripheral identifier coded DST_PER handshaking interface.
AT91C_HDMA_DST_PER_0      EQU (0x0 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 0.
AT91C_HDMA_DST_PER_1      EQU (0x1 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 1.
AT91C_HDMA_DST_PER_2      EQU (0x2 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 2.
AT91C_HDMA_DST_PER_3      EQU (0x3 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 3.
AT91C_HDMA_DST_PER_4      EQU (0x4 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 4.
AT91C_HDMA_DST_PER_5      EQU (0x5 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 5.
AT91C_HDMA_DST_PER_6      EQU (0x6 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 6.
AT91C_HDMA_DST_PER_7      EQU (0x7 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 7.
AT91C_HDMA_DST_PER_8      EQU (0x8 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 8.
AT91C_HDMA_DST_PER_9      EQU (0x9 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 9.
AT91C_HDMA_DST_PER_10     EQU (0xA <<  4) ;- (HDMA_CH) HW Handshaking Interface number 10.
AT91C_HDMA_DST_PER_11     EQU (0xB <<  4) ;- (HDMA_CH) HW Handshaking Interface number 11.
AT91C_HDMA_DST_PER_12     EQU (0xC <<  4) ;- (HDMA_CH) HW Handshaking Interface number 12.
AT91C_HDMA_DST_PER_13     EQU (0xD <<  4) ;- (HDMA_CH) HW Handshaking Interface number 13.
AT91C_HDMA_DST_PER_14     EQU (0xE <<  4) ;- (HDMA_CH) HW Handshaking Interface number 14.
AT91C_HDMA_DST_PER_15     EQU (0xF <<  4) ;- (HDMA_CH) HW Handshaking Interface number 15.
AT91C_HDMA_SRC_REP        EQU (0x1 <<  8) ;- (HDMA_CH) Source Replay Mode
AT91C_HDMA_SRC_REP_CONTIGUOUS_ADDR EQU (0x0 <<  8) ;- (HDMA_CH) When automatic mode is activated, source address is contiguous between two buffers.
AT91C_HDMA_SRC_REP_RELOAD_ADDR EQU (0x1 <<  8) ;- (HDMA_CH) When automatic mode is activated, the source address and the control register are reloaded from previous transfer..
AT91C_HDMA_SRC_H2SEL      EQU (0x1 <<  9) ;- (HDMA_CH) Source Handshaking Mode
AT91C_HDMA_SRC_H2SEL_SW   EQU (0x0 <<  9) ;- (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
AT91C_HDMA_SRC_H2SEL_HW   EQU (0x1 <<  9) ;- (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
AT91C_HDMA_DST_REP        EQU (0x1 << 12) ;- (HDMA_CH) Destination Replay Mode
AT91C_HDMA_DST_REP_CONTIGUOUS_ADDR EQU (0x0 << 12) ;- (HDMA_CH) When automatic mode is activated, destination address is contiguous between two buffers.
AT91C_HDMA_DST_REP_RELOAD_ADDR EQU (0x1 << 12) ;- (HDMA_CH) When automatic mode is activated, the destination address and the control register are reloaded from previous transfer..
AT91C_HDMA_DST_H2SEL      EQU (0x1 << 13) ;- (HDMA_CH) Destination Handshaking Mode
AT91C_HDMA_DST_H2SEL_SW   EQU (0x0 << 13) ;- (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
AT91C_HDMA_DST_H2SEL_HW   EQU (0x1 << 13) ;- (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
AT91C_HDMA_SOD            EQU (0x1 << 16) ;- (HDMA_CH) STOP ON DONE
AT91C_HDMA_SOD_DISABLE    EQU (0x0 << 16) ;- (HDMA_CH) STOP ON DONE disabled, the descriptor fetch operation ignores DONE Field of CTRLA register.
AT91C_HDMA_SOD_ENABLE     EQU (0x1 << 16) ;- (HDMA_CH) STOP ON DONE activated, the DMAC module is automatically disabled if DONE FIELD is set to 1.
AT91C_HDMA_LOCK_IF        EQU (0x1 << 20) ;- (HDMA_CH) Interface Lock
AT91C_HDMA_LOCK_IF_DISABLE EQU (0x0 << 20) ;- (HDMA_CH) Interface Lock capability is disabled.
AT91C_HDMA_LOCK_IF_ENABLE EQU (0x1 << 20) ;- (HDMA_CH) Interface Lock capability is enabled.
AT91C_HDMA_LOCK_B         EQU (0x1 << 21) ;- (HDMA_CH) AHB Bus Lock
AT91C_HDMA_LOCK_B_DISABLE EQU (0x0 << 21) ;- (HDMA_CH) AHB Bus Locking capability is disabled.
AT91C_HDMA_LOCK_B_ENABLE  EQU (0x1 << 21) ;- (HDMA_CH) AHB Bus Locking capability is enabled.
AT91C_HDMA_LOCK_IF_L      EQU (0x1 << 22) ;- (HDMA_CH) Master Interface Arbiter Lock
AT91C_HDMA_LOCK_IF_L_CHUNK EQU (0x0 << 22) ;- (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a chunk transfer.
AT91C_HDMA_LOCK_IF_L_BUFFER EQU (0x1 << 22) ;- (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a buffer transfer.
AT91C_HDMA_AHB_PROT       EQU (0x7 << 24) ;- (HDMA_CH) AHB Prot
AT91C_HDMA_FIFOCFG        EQU (0x3 << 28) ;- (HDMA_CH) FIFO Request Configuration
AT91C_HDMA_FIFOCFG_LARGESTBURST EQU (0x0 << 28) ;- (HDMA_CH) The largest defined length AHB burst is performed on the destination AHB interface.
AT91C_HDMA_FIFOCFG_HALFFIFO EQU (0x1 << 28) ;- (HDMA_CH) When half fifo size is available/filled a source/destination request is serviced.
AT91C_HDMA_FIFOCFG_ENOUGHSPACE EQU (0x2 << 28) ;- (HDMA_CH) When there is enough space/data available to perfom a single AHB access then the request is serviced.
// - -------- HDMA_SPIP : (HDMA_CH Offset: 0x18)  -------- 
AT91C_SPIP_HOLE           EQU (0xFFFF <<  0) ;- (HDMA_CH) This field indicates the value to add to the address when the programmable boundary has been reached.
AT91C_SPIP_BOUNDARY       EQU (0x3FF << 16) ;- (HDMA_CH) This field indicates the number of source transfers to perform before the automatic address increment operation.
// - -------- HDMA_DPIP : (HDMA_CH Offset: 0x1c)  -------- 
AT91C_DPIP_HOLE           EQU (0xFFFF <<  0) ;- (HDMA_CH) This field indicates the value to add to the address when the programmable boundary has been reached.
AT91C_DPIP_BOUNDARY       EQU (0x3FF << 16) ;- (HDMA_CH) This field indicates the number of source transfers to perform before the automatic address increment operation.
// - -------- HDMA_BDSCR : (HDMA_CH Offset: 0x20)  -------- 
// - -------- HDMA_CADDR : (HDMA_CH Offset: 0x24)  -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR HDMA controller
// - *****************************************************************************
// - -------- HDMA_GCFG : (HDMA Offset: 0x0)  -------- 
AT91C_HDMA_IF0_BIGEND     EQU (0x1 <<  0) ;- (HDMA) AHB-Lite Interface 0 endian mode.
AT91C_HDMA_IF0_BIGEND_IS_LITTLE_ENDIAN EQU (0x0) ;- (HDMA) AHB-Lite Interface 0 is little endian.
AT91C_HDMA_IF0_BIGEND_IS_BIG_ENDIAN EQU (0x1) ;- (HDMA) AHB-Lite Interface 0 is big endian.
AT91C_HDMA_IF1_BIGEND     EQU (0x1 <<  1) ;- (HDMA) AHB-Lite Interface 1 endian mode.
AT91C_HDMA_IF1_BIGEND_IS_LITTLE_ENDIAN EQU (0x0 <<  1) ;- (HDMA) AHB-Lite Interface 1 is little endian.
AT91C_HDMA_IF1_BIGEND_IS_BIG_ENDIAN EQU (0x1 <<  1) ;- (HDMA) AHB-Lite Interface 1 is big endian.
AT91C_HDMA_IF2_BIGEND     EQU (0x1 <<  2) ;- (HDMA) AHB-Lite Interface 2 endian mode.
AT91C_HDMA_IF2_BIGEND_IS_LITTLE_ENDIAN EQU (0x0 <<  2) ;- (HDMA) AHB-Lite Interface 2 is little endian.
AT91C_HDMA_IF2_BIGEND_IS_BIG_ENDIAN EQU (0x1 <<  2) ;- (HDMA) AHB-Lite Interface 2 is big endian.
AT91C_HDMA_IF3_BIGEND     EQU (0x1 <<  3) ;- (HDMA) AHB-Lite Interface 3 endian mode.
AT91C_HDMA_IF3_BIGEND_IS_LITTLE_ENDIAN EQU (0x0 <<  3) ;- (HDMA) AHB-Lite Interface 3 is little endian.
AT91C_HDMA_IF3_BIGEND_IS_BIG_ENDIAN EQU (0x1 <<  3) ;- (HDMA) AHB-Lite Interface 3 is big endian.
AT91C_HDMA_ARB_CFG        EQU (0x1 <<  4) ;- (HDMA) Arbiter mode.
AT91C_HDMA_ARB_CFG_FIXED  EQU (0x0 <<  4) ;- (HDMA) Fixed priority arbiter.
AT91C_HDMA_ARB_CFG_ROUND_ROBIN EQU (0x1 <<  4) ;- (HDMA) Modified round robin arbiter.
// - -------- HDMA_EN : (HDMA Offset: 0x4)  -------- 
AT91C_HDMA_ENABLE         EQU (0x1 <<  0) ;- (HDMA) 
AT91C_HDMA_ENABLE_DISABLE EQU (0x0) ;- (HDMA) Disables HDMA.
AT91C_HDMA_ENABLE_ENABLE  EQU (0x1) ;- (HDMA) Enables HDMA.
// - -------- HDMA_SREQ : (HDMA Offset: 0x8)  -------- 
AT91C_HDMA_SSREQ0         EQU (0x1 <<  0) ;- (HDMA) Request a source single transfer on channel 0
AT91C_HDMA_SSREQ0_0       EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ0_1       EQU (0x1) ;- (HDMA) Request a source single transfer on channel 0.
AT91C_HDMA_DSREQ0         EQU (0x1 <<  1) ;- (HDMA) Request a destination single transfer on channel 0
AT91C_HDMA_DSREQ0_0       EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ0_1       EQU (0x1 <<  1) ;- (HDMA) Request a destination single transfer on channel 0.
AT91C_HDMA_SSREQ1         EQU (0x1 <<  2) ;- (HDMA) Request a source single transfer on channel 1
AT91C_HDMA_SSREQ1_0       EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ1_1       EQU (0x1 <<  2) ;- (HDMA) Request a source single transfer on channel 1.
AT91C_HDMA_DSREQ1         EQU (0x1 <<  3) ;- (HDMA) Request a destination single transfer on channel 1
AT91C_HDMA_DSREQ1_0       EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ1_1       EQU (0x1 <<  3) ;- (HDMA) Request a destination single transfer on channel 1.
AT91C_HDMA_SSREQ2         EQU (0x1 <<  4) ;- (HDMA) Request a source single transfer on channel 2
AT91C_HDMA_SSREQ2_0       EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ2_1       EQU (0x1 <<  4) ;- (HDMA) Request a source single transfer on channel 2.
AT91C_HDMA_DSREQ2         EQU (0x1 <<  5) ;- (HDMA) Request a destination single transfer on channel 2
AT91C_HDMA_DSREQ2_0       EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ2_1       EQU (0x1 <<  5) ;- (HDMA) Request a destination single transfer on channel 2.
AT91C_HDMA_SSREQ3         EQU (0x1 <<  6) ;- (HDMA) Request a source single transfer on channel 3
AT91C_HDMA_SSREQ3_0       EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ3_1       EQU (0x1 <<  6) ;- (HDMA) Request a source single transfer on channel 3.
AT91C_HDMA_DSREQ3         EQU (0x1 <<  7) ;- (HDMA) Request a destination single transfer on channel 3
AT91C_HDMA_DSREQ3_0       EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ3_1       EQU (0x1 <<  7) ;- (HDMA) Request a destination single transfer on channel 3.
AT91C_HDMA_SSREQ4         EQU (0x1 <<  8) ;- (HDMA) Request a source single transfer on channel 4
AT91C_HDMA_SSREQ4_0       EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ4_1       EQU (0x1 <<  8) ;- (HDMA) Request a source single transfer on channel 4.
AT91C_HDMA_DSREQ4         EQU (0x1 <<  9) ;- (HDMA) Request a destination single transfer on channel 4
AT91C_HDMA_DSREQ4_0       EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ4_1       EQU (0x1 <<  9) ;- (HDMA) Request a destination single transfer on channel 4.
AT91C_HDMA_SSREQ5         EQU (0x1 << 10) ;- (HDMA) Request a source single transfer on channel 5
AT91C_HDMA_SSREQ5_0       EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ5_1       EQU (0x1 << 10) ;- (HDMA) Request a source single transfer on channel 5.
AT91C_HDMA_DSREQ6         EQU (0x1 << 11) ;- (HDMA) Request a destination single transfer on channel 5
AT91C_HDMA_DSREQ6_0       EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ6_1       EQU (0x1 << 11) ;- (HDMA) Request a destination single transfer on channel 5.
AT91C_HDMA_SSREQ6         EQU (0x1 << 12) ;- (HDMA) Request a source single transfer on channel 6
AT91C_HDMA_SSREQ6_0       EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ6_1       EQU (0x1 << 12) ;- (HDMA) Request a source single transfer on channel 6.
AT91C_HDMA_SSREQ7         EQU (0x1 << 14) ;- (HDMA) Request a source single transfer on channel 7
AT91C_HDMA_SSREQ7_0       EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ7_1       EQU (0x1 << 14) ;- (HDMA) Request a source single transfer on channel 7.
AT91C_HDMA_DSREQ7         EQU (0x1 << 15) ;- (HDMA) Request a destination single transfer on channel 7
AT91C_HDMA_DSREQ7_0       EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ7_1       EQU (0x1 << 15) ;- (HDMA) Request a destination single transfer on channel 7.
// - -------- HDMA_CREQ : (HDMA Offset: 0xc)  -------- 
AT91C_HDMA_SCREQ0         EQU (0x1 <<  0) ;- (HDMA) Request a source chunk transfer on channel 0
AT91C_HDMA_SCREQ0_0       EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ0_1       EQU (0x1) ;- (HDMA) Request a source chunk transfer on channel 0.
AT91C_HDMA_DCREQ0         EQU (0x1 <<  1) ;- (HDMA) Request a destination chunk transfer on channel 0
AT91C_HDMA_DCREQ0_0       EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ0_1       EQU (0x1 <<  1) ;- (HDMA) Request a destination chunk transfer on channel 0.
AT91C_HDMA_SCREQ1         EQU (0x1 <<  2) ;- (HDMA) Request a source chunk transfer on channel 1
AT91C_HDMA_SCREQ1_0       EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ1_1       EQU (0x1 <<  2) ;- (HDMA) Request a source chunk transfer on channel 1.
AT91C_HDMA_DCREQ1         EQU (0x1 <<  3) ;- (HDMA) Request a destination chunk transfer on channel 1
AT91C_HDMA_DCREQ1_0       EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ1_1       EQU (0x1 <<  3) ;- (HDMA) Request a destination chunk transfer on channel 1.
AT91C_HDMA_SCREQ2         EQU (0x1 <<  4) ;- (HDMA) Request a source chunk transfer on channel 2
AT91C_HDMA_SCREQ2_0       EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ2_1       EQU (0x1 <<  4) ;- (HDMA) Request a source chunk transfer on channel 2.
AT91C_HDMA_DCREQ2         EQU (0x1 <<  5) ;- (HDMA) Request a destination chunk transfer on channel 2
AT91C_HDMA_DCREQ2_0       EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ2_1       EQU (0x1 <<  5) ;- (HDMA) Request a destination chunk transfer on channel 2.
AT91C_HDMA_SCREQ3         EQU (0x1 <<  6) ;- (HDMA) Request a source chunk transfer on channel 3
AT91C_HDMA_SCREQ3_0       EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ3_1       EQU (0x1 <<  6) ;- (HDMA) Request a source chunk transfer on channel 3.
AT91C_HDMA_DCREQ3         EQU (0x1 <<  7) ;- (HDMA) Request a destination chunk transfer on channel 3
AT91C_HDMA_DCREQ3_0       EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ3_1       EQU (0x1 <<  7) ;- (HDMA) Request a destination chunk transfer on channel 3.
AT91C_HDMA_SCREQ4         EQU (0x1 <<  8) ;- (HDMA) Request a source chunk transfer on channel 4
AT91C_HDMA_SCREQ4_0       EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ4_1       EQU (0x1 <<  8) ;- (HDMA) Request a source chunk transfer on channel 4.
AT91C_HDMA_DCREQ4         EQU (0x1 <<  9) ;- (HDMA) Request a destination chunk transfer on channel 4
AT91C_HDMA_DCREQ4_0       EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ4_1       EQU (0x1 <<  9) ;- (HDMA) Request a destination chunk transfer on channel 4.
AT91C_HDMA_SCREQ5         EQU (0x1 << 10) ;- (HDMA) Request a source chunk transfer on channel 5
AT91C_HDMA_SCREQ5_0       EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ5_1       EQU (0x1 << 10) ;- (HDMA) Request a source chunk transfer on channel 5.
AT91C_HDMA_DCREQ6         EQU (0x1 << 11) ;- (HDMA) Request a destination chunk transfer on channel 5
AT91C_HDMA_DCREQ6_0       EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ6_1       EQU (0x1 << 11) ;- (HDMA) Request a destination chunk transfer on channel 5.
AT91C_HDMA_SCREQ6         EQU (0x1 << 12) ;- (HDMA) Request a source chunk transfer on channel 6
AT91C_HDMA_SCREQ6_0       EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ6_1       EQU (0x1 << 12) ;- (HDMA) Request a source chunk transfer on channel 6.
AT91C_HDMA_SCREQ7         EQU (0x1 << 14) ;- (HDMA) Request a source chunk transfer on channel 7
AT91C_HDMA_SCREQ7_0       EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ7_1       EQU (0x1 << 14) ;- (HDMA) Request a source chunk transfer on channel 7.
AT91C_HDMA_DCREQ7         EQU (0x1 << 15) ;- (HDMA) Request a destination chunk transfer on channel 7
AT91C_HDMA_DCREQ7_0       EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ7_1       EQU (0x1 << 15) ;- (HDMA) Request a destination chunk transfer on channel 7.
// - -------- HDMA_LAST : (HDMA Offset: 0x10)  -------- 
AT91C_HDMA_SLAST0         EQU (0x1 <<  0) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 0
AT91C_HDMA_SLAST0_0       EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_SLAST0_1       EQU (0x1) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 0.
AT91C_HDMA_DLAST0         EQU (0x1 <<  1) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 0
AT91C_HDMA_DLAST0_0       EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DLAST0_1       EQU (0x1 <<  1) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 0.
AT91C_HDMA_SLAST1         EQU (0x1 <<  2) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 1
AT91C_HDMA_SLAST1_0       EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_SLAST1_1       EQU (0x1 <<  2) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 1.
AT91C_HDMA_DLAST1         EQU (0x1 <<  3) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 1
AT91C_HDMA_DLAST1_0       EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DLAST1_1       EQU (0x1 <<  3) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 1.
AT91C_HDMA_SLAST2         EQU (0x1 <<  4) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 2
AT91C_HDMA_SLAST2_0       EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_SLAST2_1       EQU (0x1 <<  4) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 2.
AT91C_HDMA_DLAST2         EQU (0x1 <<  5) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 2
AT91C_HDMA_DLAST2_0       EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DLAST2_1       EQU (0x1 <<  5) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 2.
AT91C_HDMA_SLAST3         EQU (0x1 <<  6) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 3
AT91C_HDMA_SLAST3_0       EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_SLAST3_1       EQU (0x1 <<  6) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 3.
AT91C_HDMA_DLAST3         EQU (0x1 <<  7) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 3
AT91C_HDMA_DLAST3_0       EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DLAST3_1       EQU (0x1 <<  7) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 3.
AT91C_HDMA_SLAST4         EQU (0x1 <<  8) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 4
AT91C_HDMA_SLAST4_0       EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_SLAST4_1       EQU (0x1 <<  8) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 4.
AT91C_HDMA_DLAST4         EQU (0x1 <<  9) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 4
AT91C_HDMA_DLAST4_0       EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_DLAST4_1       EQU (0x1 <<  9) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 4.
AT91C_HDMA_SLAST5         EQU (0x1 << 10) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 5
AT91C_HDMA_SLAST5_0       EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_SLAST5_1       EQU (0x1 << 10) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 5.
AT91C_HDMA_DLAST6         EQU (0x1 << 11) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 5
AT91C_HDMA_DLAST6_0       EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_DLAST6_1       EQU (0x1 << 11) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 5.
AT91C_HDMA_SLAST6         EQU (0x1 << 12) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 6
AT91C_HDMA_SLAST6_0       EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_SLAST6_1       EQU (0x1 << 12) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 6.
AT91C_HDMA_SLAST7         EQU (0x1 << 14) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 7
AT91C_HDMA_SLAST7_0       EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_SLAST7_1       EQU (0x1 << 14) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 7.
AT91C_HDMA_DLAST7         EQU (0x1 << 15) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 7
AT91C_HDMA_DLAST7_0       EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_DLAST7_1       EQU (0x1 << 15) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 7.
// - -------- HDMA_SYNC : (HDMA Offset: 0x14)  -------- 
AT91C_SYNC_REQ            EQU (0xFFFF <<  0) ;- (HDMA) 
// - -------- HDMA_EBCIER : (HDMA Offset: 0x18) Buffer Transfer Completed/Chained Buffer Transfer Completed/Access Error Interrupt Enable Register -------- 
AT91C_HDMA_BTC0           EQU (0x1 <<  0) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC1           EQU (0x1 <<  1) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC2           EQU (0x1 <<  2) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC3           EQU (0x1 <<  3) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC4           EQU (0x1 <<  4) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC5           EQU (0x1 <<  5) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC6           EQU (0x1 <<  6) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC7           EQU (0x1 <<  7) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC0          EQU (0x1 <<  8) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC1          EQU (0x1 <<  9) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC2          EQU (0x1 << 10) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC3          EQU (0x1 << 11) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC4          EQU (0x1 << 12) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC5          EQU (0x1 << 13) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC6          EQU (0x1 << 14) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC7          EQU (0x1 << 15) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR0           EQU (0x1 << 16) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR1           EQU (0x1 << 17) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR2           EQU (0x1 << 18) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR3           EQU (0x1 << 19) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR4           EQU (0x1 << 20) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR5           EQU (0x1 << 21) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR6           EQU (0x1 << 22) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR7           EQU (0x1 << 23) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
// - -------- HDMA_EBCIDR : (HDMA Offset: 0x1c)  -------- 
// - -------- HDMA_EBCIMR : (HDMA Offset: 0x20)  -------- 
// - -------- HDMA_EBCISR : (HDMA Offset: 0x24)  -------- 
// - -------- HDMA_CHER : (HDMA Offset: 0x28)  -------- 
AT91C_HDMA_ENA0           EQU (0x1 <<  0) ;- (HDMA) When set, channel 0 enabled.
AT91C_HDMA_ENA0_0         EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_ENA0_1         EQU (0x1) ;- (HDMA) Channel 0 enabled.
AT91C_HDMA_ENA1           EQU (0x1 <<  1) ;- (HDMA) When set, channel 1 enabled.
AT91C_HDMA_ENA1_0         EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_ENA1_1         EQU (0x1 <<  1) ;- (HDMA) Channel 1 enabled.
AT91C_HDMA_ENA2           EQU (0x1 <<  2) ;- (HDMA) When set, channel 2 enabled.
AT91C_HDMA_ENA2_0         EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_ENA2_1         EQU (0x1 <<  2) ;- (HDMA) Channel 2 enabled.
AT91C_HDMA_ENA3           EQU (0x1 <<  3) ;- (HDMA) When set, channel 3 enabled.
AT91C_HDMA_ENA3_0         EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_ENA3_1         EQU (0x1 <<  3) ;- (HDMA) Channel 3 enabled.
AT91C_HDMA_ENA4           EQU (0x1 <<  4) ;- (HDMA) When set, channel 4 enabled.
AT91C_HDMA_ENA4_0         EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_ENA4_1         EQU (0x1 <<  4) ;- (HDMA) Channel 4 enabled.
AT91C_HDMA_ENA5           EQU (0x1 <<  5) ;- (HDMA) When set, channel 5 enabled.
AT91C_HDMA_ENA5_0         EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_ENA5_1         EQU (0x1 <<  5) ;- (HDMA) Channel 5 enabled.
AT91C_HDMA_ENA6           EQU (0x1 <<  6) ;- (HDMA) When set, channel 6 enabled.
AT91C_HDMA_ENA6_0         EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_ENA6_1         EQU (0x1 <<  6) ;- (HDMA) Channel 6 enabled.
AT91C_HDMA_ENA7           EQU (0x1 <<  7) ;- (HDMA) When set, channel 7 enabled.
AT91C_HDMA_ENA7_0         EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_ENA7_1         EQU (0x1 <<  7) ;- (HDMA) Channel 7 enabled.
AT91C_HDMA_SUSP0          EQU (0x1 <<  8) ;- (HDMA) When set, channel 0 freezed and its current context.
AT91C_HDMA_SUSP0_0        EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_SUSP0_1        EQU (0x1 <<  8) ;- (HDMA) Channel 0 freezed.
AT91C_HDMA_SUSP1          EQU (0x1 <<  9) ;- (HDMA) When set, channel 1 freezed and its current context.
AT91C_HDMA_SUSP1_0        EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_SUSP1_1        EQU (0x1 <<  9) ;- (HDMA) Channel 1 freezed.
AT91C_HDMA_SUSP2          EQU (0x1 << 10) ;- (HDMA) When set, channel 2 freezed and its current context.
AT91C_HDMA_SUSP2_0        EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_SUSP2_1        EQU (0x1 << 10) ;- (HDMA) Channel 2 freezed.
AT91C_HDMA_SUSP3          EQU (0x1 << 11) ;- (HDMA) When set, channel 3 freezed and its current context.
AT91C_HDMA_SUSP3_0        EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_SUSP3_1        EQU (0x1 << 11) ;- (HDMA) Channel 3 freezed.
AT91C_HDMA_SUSP4          EQU (0x1 << 12) ;- (HDMA) When set, channel 4 freezed and its current context.
AT91C_HDMA_SUSP4_0        EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_SUSP4_1        EQU (0x1 << 12) ;- (HDMA) Channel 4 freezed.
AT91C_HDMA_SUSP5          EQU (0x1 << 13) ;- (HDMA) When set, channel 5 freezed and its current context.
AT91C_HDMA_SUSP5_0        EQU (0x0 << 13) ;- (HDMA) No effect.
AT91C_HDMA_SUSP5_1        EQU (0x1 << 13) ;- (HDMA) Channel 5 freezed.
AT91C_HDMA_SUSP6          EQU (0x1 << 14) ;- (HDMA) When set, channel 6 freezed and its current context.
AT91C_HDMA_SUSP6_0        EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_SUSP6_1        EQU (0x1 << 14) ;- (HDMA) Channel 6 freezed.
AT91C_HDMA_SUSP7          EQU (0x1 << 15) ;- (HDMA) When set, channel 7 freezed and its current context.
AT91C_HDMA_SUSP7_0        EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_SUSP7_1        EQU (0x1 << 15) ;- (HDMA) Channel 7 freezed.
AT91C_HDMA_KEEP0          EQU (0x1 << 24) ;- (HDMA) When set, it resumes the channel 0 from an automatic stall state.
AT91C_HDMA_KEEP0_0        EQU (0x0 << 24) ;- (HDMA) No effect.
AT91C_HDMA_KEEP0_1        EQU (0x1 << 24) ;- (HDMA) Resumes the channel 0.
AT91C_HDMA_KEEP1          EQU (0x1 << 25) ;- (HDMA) When set, it resumes the channel 1 from an automatic stall state.
AT91C_HDMA_KEEP1_0        EQU (0x0 << 25) ;- (HDMA) No effect.
AT91C_HDMA_KEEP1_1        EQU (0x1 << 25) ;- (HDMA) Resumes the channel 1.
AT91C_HDMA_KEEP2          EQU (0x1 << 26) ;- (HDMA) When set, it resumes the channel 2 from an automatic stall state.
AT91C_HDMA_KEEP2_0        EQU (0x0 << 26) ;- (HDMA) No effect.
AT91C_HDMA_KEEP2_1        EQU (0x1 << 26) ;- (HDMA) Resumes the channel 2.
AT91C_HDMA_KEEP3          EQU (0x1 << 27) ;- (HDMA) When set, it resumes the channel 3 from an automatic stall state.
AT91C_HDMA_KEEP3_0        EQU (0x0 << 27) ;- (HDMA) No effect.
AT91C_HDMA_KEEP3_1        EQU (0x1 << 27) ;- (HDMA) Resumes the channel 3.
AT91C_HDMA_KEEP4          EQU (0x1 << 28) ;- (HDMA) When set, it resumes the channel 4 from an automatic stall state.
AT91C_HDMA_KEEP4_0        EQU (0x0 << 28) ;- (HDMA) No effect.
AT91C_HDMA_KEEP4_1        EQU (0x1 << 28) ;- (HDMA) Resumes the channel 4.
AT91C_HDMA_KEEP5          EQU (0x1 << 29) ;- (HDMA) When set, it resumes the channel 5 from an automatic stall state.
AT91C_HDMA_KEEP5_0        EQU (0x0 << 29) ;- (HDMA) No effect.
AT91C_HDMA_KEEP5_1        EQU (0x1 << 29) ;- (HDMA) Resumes the channel 5.
AT91C_HDMA_KEEP6          EQU (0x1 << 30) ;- (HDMA) When set, it resumes the channel 6 from an automatic stall state.
AT91C_HDMA_KEEP6_0        EQU (0x0 << 30) ;- (HDMA) No effect.
AT91C_HDMA_KEEP6_1        EQU (0x1 << 30) ;- (HDMA) Resumes the channel 6.
AT91C_HDMA_KEEP7          EQU (0x1 << 31) ;- (HDMA) When set, it resumes the channel 7 from an automatic stall state.
AT91C_HDMA_KEEP7_0        EQU (0x0 << 31) ;- (HDMA) No effect.
AT91C_HDMA_KEEP7_1        EQU (0x1 << 31) ;- (HDMA) Resumes the channel 7.
// - -------- HDMA_CHDR : (HDMA Offset: 0x2c)  -------- 
AT91C_HDMA_DIS0           EQU (0x1 <<  0) ;- (HDMA) Write one to this field to disable the channel 0.
AT91C_HDMA_DIS0_0         EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_DIS0_1         EQU (0x1) ;- (HDMA) Disables the channel 0.
AT91C_HDMA_DIS1           EQU (0x1 <<  1) ;- (HDMA) Write one to this field to disable the channel 1.
AT91C_HDMA_DIS1_0         EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DIS1_1         EQU (0x1 <<  1) ;- (HDMA) Disables the channel 1.
AT91C_HDMA_DIS2           EQU (0x1 <<  2) ;- (HDMA) Write one to this field to disable the channel 2.
AT91C_HDMA_DIS2_0         EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_DIS2_1         EQU (0x1 <<  2) ;- (HDMA) Disables the channel 2.
AT91C_HDMA_DIS3           EQU (0x1 <<  3) ;- (HDMA) Write one to this field to disable the channel 3.
AT91C_HDMA_DIS3_0         EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DIS3_1         EQU (0x1 <<  3) ;- (HDMA) Disables the channel 3.
AT91C_HDMA_DIS4           EQU (0x1 <<  4) ;- (HDMA) Write one to this field to disable the channel 4.
AT91C_HDMA_DIS4_0         EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_DIS4_1         EQU (0x1 <<  4) ;- (HDMA) Disables the channel 4.
AT91C_HDMA_DIS5           EQU (0x1 <<  5) ;- (HDMA) Write one to this field to disable the channel 5.
AT91C_HDMA_DIS5_0         EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DIS5_1         EQU (0x1 <<  5) ;- (HDMA) Disables the channel 5.
AT91C_HDMA_DIS6           EQU (0x1 <<  6) ;- (HDMA) Write one to this field to disable the channel 6.
AT91C_HDMA_DIS6_0         EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_DIS6_1         EQU (0x1 <<  6) ;- (HDMA) Disables the channel 6.
AT91C_HDMA_DIS7           EQU (0x1 <<  7) ;- (HDMA) Write one to this field to disable the channel 7.
AT91C_HDMA_DIS7_0         EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DIS7_1         EQU (0x1 <<  7) ;- (HDMA) Disables the channel 7.
AT91C_HDMA_RES0           EQU (0x1 <<  8) ;- (HDMA) Write one to this field to resume the channel 0 transfer restoring its context.
AT91C_HDMA_RES0_0         EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_RES0_1         EQU (0x1 <<  8) ;- (HDMA) Resumes the channel 0.
AT91C_HDMA_RES1           EQU (0x1 <<  9) ;- (HDMA) Write one to this field to resume the channel 1 transfer restoring its context.
AT91C_HDMA_RES1_0         EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_RES1_1         EQU (0x1 <<  9) ;- (HDMA) Resumes the channel 1.
AT91C_HDMA_RES2           EQU (0x1 << 10) ;- (HDMA) Write one to this field to resume the channel 2 transfer restoring its context.
AT91C_HDMA_RES2_0         EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_RES2_1         EQU (0x1 << 10) ;- (HDMA) Resumes the channel 2.
AT91C_HDMA_RES3           EQU (0x1 << 11) ;- (HDMA) Write one to this field to resume the channel 3 transfer restoring its context.
AT91C_HDMA_RES3_0         EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_RES3_1         EQU (0x1 << 11) ;- (HDMA) Resumes the channel 3.
AT91C_HDMA_RES4           EQU (0x1 << 12) ;- (HDMA) Write one to this field to resume the channel 4 transfer restoring its context.
AT91C_HDMA_RES4_0         EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_RES4_1         EQU (0x1 << 12) ;- (HDMA) Resumes the channel 4.
AT91C_HDMA_RES5           EQU (0x1 << 13) ;- (HDMA) Write one to this field to resume the channel 5 transfer restoring its context.
AT91C_HDMA_RES5_0         EQU (0x0 << 13) ;- (HDMA) No effect.
AT91C_HDMA_RES5_1         EQU (0x1 << 13) ;- (HDMA) Resumes the channel 5.
AT91C_HDMA_RES6           EQU (0x1 << 14) ;- (HDMA) Write one to this field to resume the channel 6 transfer restoring its context.
AT91C_HDMA_RES6_0         EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_RES6_1         EQU (0x1 << 14) ;- (HDMA) Resumes the channel 6.
AT91C_HDMA_RES7           EQU (0x1 << 15) ;- (HDMA) Write one to this field to resume the channel 7 transfer restoring its context.
AT91C_HDMA_RES7_0         EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_RES7_1         EQU (0x1 << 15) ;- (HDMA) Resumes the channel 7.
// - -------- HDMA_CHSR : (HDMA Offset: 0x30)  -------- 
AT91C_HDMA_EMPT0          EQU (0x1 << 16) ;- (HDMA) When set, channel 0 is empty.
AT91C_HDMA_EMPT0_0        EQU (0x0 << 16) ;- (HDMA) No effect.
AT91C_HDMA_EMPT0_1        EQU (0x1 << 16) ;- (HDMA) Channel 0 empty.
AT91C_HDMA_EMPT1          EQU (0x1 << 17) ;- (HDMA) When set, channel 1 is empty.
AT91C_HDMA_EMPT1_0        EQU (0x0 << 17) ;- (HDMA) No effect.
AT91C_HDMA_EMPT1_1        EQU (0x1 << 17) ;- (HDMA) Channel 1 empty.
AT91C_HDMA_EMPT2          EQU (0x1 << 18) ;- (HDMA) When set, channel 2 is empty.
AT91C_HDMA_EMPT2_0        EQU (0x0 << 18) ;- (HDMA) No effect.
AT91C_HDMA_EMPT2_1        EQU (0x1 << 18) ;- (HDMA) Channel 2 empty.
AT91C_HDMA_EMPT3          EQU (0x1 << 19) ;- (HDMA) When set, channel 3 is empty.
AT91C_HDMA_EMPT3_0        EQU (0x0 << 19) ;- (HDMA) No effect.
AT91C_HDMA_EMPT3_1        EQU (0x1 << 19) ;- (HDMA) Channel 3 empty.
AT91C_HDMA_EMPT4          EQU (0x1 << 20) ;- (HDMA) When set, channel 4 is empty.
AT91C_HDMA_EMPT4_0        EQU (0x0 << 20) ;- (HDMA) No effect.
AT91C_HDMA_EMPT4_1        EQU (0x1 << 20) ;- (HDMA) Channel 4 empty.
AT91C_HDMA_EMPT5          EQU (0x1 << 21) ;- (HDMA) When set, channel 5 is empty.
AT91C_HDMA_EMPT5_0        EQU (0x0 << 21) ;- (HDMA) No effect.
AT91C_HDMA_EMPT5_1        EQU (0x1 << 21) ;- (HDMA) Channel 5 empty.
AT91C_HDMA_EMPT6          EQU (0x1 << 22) ;- (HDMA) When set, channel 6 is empty.
AT91C_HDMA_EMPT6_0        EQU (0x0 << 22) ;- (HDMA) No effect.
AT91C_HDMA_EMPT6_1        EQU (0x1 << 22) ;- (HDMA) Channel 6 empty.
AT91C_HDMA_EMPT7          EQU (0x1 << 23) ;- (HDMA) When set, channel 7 is empty.
AT91C_HDMA_EMPT7_0        EQU (0x0 << 23) ;- (HDMA) No effect.
AT91C_HDMA_EMPT7_1        EQU (0x1 << 23) ;- (HDMA) Channel 7 empty.
AT91C_HDMA_STAL0          EQU (0x1 << 24) ;- (HDMA) When set, channel 0 is stalled.
AT91C_HDMA_STAL0_0        EQU (0x0 << 24) ;- (HDMA) No effect.
AT91C_HDMA_STAL0_1        EQU (0x1 << 24) ;- (HDMA) Channel 0 stalled.
AT91C_HDMA_STAL1          EQU (0x1 << 25) ;- (HDMA) When set, channel 1 is stalled.
AT91C_HDMA_STAL1_0        EQU (0x0 << 25) ;- (HDMA) No effect.
AT91C_HDMA_STAL1_1        EQU (0x1 << 25) ;- (HDMA) Channel 1 stalled.
AT91C_HDMA_STAL2          EQU (0x1 << 26) ;- (HDMA) When set, channel 2 is stalled.
AT91C_HDMA_STAL2_0        EQU (0x0 << 26) ;- (HDMA) No effect.
AT91C_HDMA_STAL2_1        EQU (0x1 << 26) ;- (HDMA) Channel 2 stalled.
AT91C_HDMA_STAL3          EQU (0x1 << 27) ;- (HDMA) When set, channel 3 is stalled.
AT91C_HDMA_STAL3_0        EQU (0x0 << 27) ;- (HDMA) No effect.
AT91C_HDMA_STAL3_1        EQU (0x1 << 27) ;- (HDMA) Channel 3 stalled.
AT91C_HDMA_STAL4          EQU (0x1 << 28) ;- (HDMA) When set, channel 4 is stalled.
AT91C_HDMA_STAL4_0        EQU (0x0 << 28) ;- (HDMA) No effect.
AT91C_HDMA_STAL4_1        EQU (0x1 << 28) ;- (HDMA) Channel 4 stalled.
AT91C_HDMA_STAL5          EQU (0x1 << 29) ;- (HDMA) When set, channel 5 is stalled.
AT91C_HDMA_STAL5_0        EQU (0x0 << 29) ;- (HDMA) No effect.
AT91C_HDMA_STAL5_1        EQU (0x1 << 29) ;- (HDMA) Channel 5 stalled.
AT91C_HDMA_STAL6          EQU (0x1 << 30) ;- (HDMA) When set, channel 6 is stalled.
AT91C_HDMA_STAL6_0        EQU (0x0 << 30) ;- (HDMA) No effect.
AT91C_HDMA_STAL6_1        EQU (0x1 << 30) ;- (HDMA) Channel 6 stalled.
AT91C_HDMA_STAL7          EQU (0x1 << 31) ;- (HDMA) When set, channel 7 is stalled.
AT91C_HDMA_STAL7_0        EQU (0x0 << 31) ;- (HDMA) No effect.
AT91C_HDMA_STAL7_1        EQU (0x1 << 31) ;- (HDMA) Channel 7 stalled.
// - -------- HDMA_RSVD : (HDMA Offset: 0x34)  -------- 
// - -------- HDMA_RSVD : (HDMA Offset: 0x38)  -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Error Correction Code controller
// - *****************************************************************************
// - -------- ECC_CR : (ECC Offset: 0x0) ECC reset register -------- 
AT91C_ECC_RST             EQU (0x1 <<  0) ;- (ECC) ECC reset parity
// - -------- ECC_MR : (ECC Offset: 0x4) ECC page size register -------- 
AT91C_ECC_PAGE_SIZE       EQU (0x3 <<  0) ;- (ECC) Nand Flash page size
// - -------- ECC_SR : (ECC Offset: 0x8) ECC status register -------- 
AT91C_ECC_RECERR          EQU (0x1 <<  0) ;- (ECC) ECC error
AT91C_ECC_ECCERR          EQU (0x1 <<  1) ;- (ECC) ECC single error
AT91C_ECC_MULERR          EQU (0x1 <<  2) ;- (ECC) ECC_MULERR
// - -------- ECC_PR : (ECC Offset: 0xc) ECC parity register -------- 
AT91C_ECC_BITADDR         EQU (0xF <<  0) ;- (ECC) Bit address error
AT91C_ECC_WORDADDR        EQU (0xFFF <<  4) ;- (ECC) address of the failing bit
// - -------- ECC_NPR : (ECC Offset: 0x10) ECC N parity register -------- 
AT91C_ECC_NPARITY         EQU (0xFFFF <<  0) ;- (ECC) ECC parity N 
// - -------- ECC_VR : (ECC Offset: 0xfc) ECC version register -------- 
AT91C_ECC_VR              EQU (0xF <<  0) ;- (ECC) ECC version register

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Ethernet MAC 10/100
// - *****************************************************************************
// - -------- EMAC_NCR : (EMAC Offset: 0x0)  -------- 
AT91C_EMAC_LB             EQU (0x1 <<  0) ;- (EMAC) Loopback. Optional. When set, loopback signal is at high level.
AT91C_EMAC_LLB            EQU (0x1 <<  1) ;- (EMAC) Loopback local. 
AT91C_EMAC_RE             EQU (0x1 <<  2) ;- (EMAC) Receive enable. 
AT91C_EMAC_TE             EQU (0x1 <<  3) ;- (EMAC) Transmit enable. 
AT91C_EMAC_MPE            EQU (0x1 <<  4) ;- (EMAC) Management port enable. 
AT91C_EMAC_CLRSTAT        EQU (0x1 <<  5) ;- (EMAC) Clear statistics registers. 
AT91C_EMAC_INCSTAT        EQU (0x1 <<  6) ;- (EMAC) Increment statistics registers. 
AT91C_EMAC_WESTAT         EQU (0x1 <<  7) ;- (EMAC) Write enable for statistics registers. 
AT91C_EMAC_BP             EQU (0x1 <<  8) ;- (EMAC) Back pressure. 
AT91C_EMAC_TSTART         EQU (0x1 <<  9) ;- (EMAC) Start Transmission. 
AT91C_EMAC_THALT          EQU (0x1 << 10) ;- (EMAC) Transmission Halt. 
AT91C_EMAC_TPFR           EQU (0x1 << 11) ;- (EMAC) Transmit pause frame 
AT91C_EMAC_TZQ            EQU (0x1 << 12) ;- (EMAC) Transmit zero quantum pause frame
// - -------- EMAC_NCFGR : (EMAC Offset: 0x4) Network Configuration Register -------- 
AT91C_EMAC_SPD            EQU (0x1 <<  0) ;- (EMAC) Speed. 
AT91C_EMAC_FD             EQU (0x1 <<  1) ;- (EMAC) Full duplex. 
AT91C_EMAC_JFRAME         EQU (0x1 <<  3) ;- (EMAC) Jumbo Frames. 
AT91C_EMAC_CAF            EQU (0x1 <<  4) ;- (EMAC) Copy all frames. 
AT91C_EMAC_NBC            EQU (0x1 <<  5) ;- (EMAC) No broadcast. 
AT91C_EMAC_MTI            EQU (0x1 <<  6) ;- (EMAC) Multicast hash event enable
AT91C_EMAC_UNI            EQU (0x1 <<  7) ;- (EMAC) Unicast hash enable. 
AT91C_EMAC_BIG            EQU (0x1 <<  8) ;- (EMAC) Receive 1522 bytes. 
AT91C_EMAC_EAE            EQU (0x1 <<  9) ;- (EMAC) External address match enable. 
AT91C_EMAC_CLK            EQU (0x3 << 10) ;- (EMAC) 
AT91C_EMAC_CLK_HCLK_8     EQU (0x0 << 10) ;- (EMAC) HCLK divided by 8
AT91C_EMAC_CLK_HCLK_16    EQU (0x1 << 10) ;- (EMAC) HCLK divided by 16
AT91C_EMAC_CLK_HCLK_32    EQU (0x2 << 10) ;- (EMAC) HCLK divided by 32
AT91C_EMAC_CLK_HCLK_64    EQU (0x3 << 10) ;- (EMAC) HCLK divided by 64
AT91C_EMAC_RTY            EQU (0x1 << 12) ;- (EMAC) 
AT91C_EMAC_PAE            EQU (0x1 << 13) ;- (EMAC) 
AT91C_EMAC_RBOF           EQU (0x3 << 14) ;- (EMAC) 
AT91C_EMAC_RBOF_OFFSET_0  EQU (0x0 << 14) ;- (EMAC) no offset from start of receive buffer
AT91C_EMAC_RBOF_OFFSET_1  EQU (0x1 << 14) ;- (EMAC) one byte offset from start of receive buffer
AT91C_EMAC_RBOF_OFFSET_2  EQU (0x2 << 14) ;- (EMAC) two bytes offset from start of receive buffer
AT91C_EMAC_RBOF_OFFSET_3  EQU (0x3 << 14) ;- (EMAC) three bytes offset from start of receive buffer
AT91C_EMAC_RLCE           EQU (0x1 << 16) ;- (EMAC) Receive Length field Checking Enable
AT91C_EMAC_DRFCS          EQU (0x1 << 17) ;- (EMAC) Discard Receive FCS
AT91C_EMAC_EFRHD          EQU (0x1 << 18) ;- (EMAC) 
AT91C_EMAC_IRXFCS         EQU (0x1 << 19) ;- (EMAC) Ignore RX FCS
// - -------- EMAC_NSR : (EMAC Offset: 0x8) Network Status Register -------- 
AT91C_EMAC_LINKR          EQU (0x1 <<  0) ;- (EMAC) 
AT91C_EMAC_MDIO           EQU (0x1 <<  1) ;- (EMAC) 
AT91C_EMAC_IDLE           EQU (0x1 <<  2) ;- (EMAC) 
// - -------- EMAC_TSR : (EMAC Offset: 0x14) Transmit Status Register -------- 
AT91C_EMAC_UBR            EQU (0x1 <<  0) ;- (EMAC) 
AT91C_EMAC_COL            EQU (0x1 <<  1) ;- (EMAC) 
AT91C_EMAC_RLES           EQU (0x1 <<  2) ;- (EMAC) 
AT91C_EMAC_TGO            EQU (0x1 <<  3) ;- (EMAC) Transmit Go
AT91C_EMAC_BEX            EQU (0x1 <<  4) ;- (EMAC) Buffers exhausted mid frame
AT91C_EMAC_COMP           EQU (0x1 <<  5) ;- (EMAC) 
AT91C_EMAC_UND            EQU (0x1 <<  6) ;- (EMAC) 
// - -------- EMAC_RSR : (EMAC Offset: 0x20) Receive Status Register -------- 
AT91C_EMAC_BNA            EQU (0x1 <<  0) ;- (EMAC) 
AT91C_EMAC_REC            EQU (0x1 <<  1) ;- (EMAC) 
AT91C_EMAC_OVR            EQU (0x1 <<  2) ;- (EMAC) 
// - -------- EMAC_ISR : (EMAC Offset: 0x24) Interrupt Status Register -------- 
AT91C_EMAC_MFD            EQU (0x1 <<  0) ;- (EMAC) 
AT91C_EMAC_RCOMP          EQU (0x1 <<  1) ;- (EMAC) 
AT91C_EMAC_RXUBR          EQU (0x1 <<  2) ;- (EMAC) 
AT91C_EMAC_TXUBR          EQU (0x1 <<  3) ;- (EMAC) 
AT91C_EMAC_TUNDR          EQU (0x1 <<  4) ;- (EMAC) 
AT91C_EMAC_RLEX           EQU (0x1 <<  5) ;- (EMAC) 
AT91C_EMAC_TXERR          EQU (0x1 <<  6) ;- (EMAC) 
AT91C_EMAC_TCOMP          EQU (0x1 <<  7) ;- (EMAC) 
AT91C_EMAC_LINK           EQU (0x1 <<  9) ;- (EMAC) 
AT91C_EMAC_ROVR           EQU (0x1 << 10) ;- (EMAC) 
AT91C_EMAC_HRESP          EQU (0x1 << 11) ;- (EMAC) 
AT91C_EMAC_PFRE           EQU (0x1 << 12) ;- (EMAC) 
AT91C_EMAC_PTZ            EQU (0x1 << 13) ;- (EMAC) 
AT91C_EMAC_WOLEV          EQU (0x1 << 14) ;- (EMAC) 
// - -------- EMAC_IER : (EMAC Offset: 0x28) Interrupt Enable Register -------- 
// - -------- EMAC_IDR : (EMAC Offset: 0x2c) Interrupt Disable Register -------- 
// - -------- EMAC_IMR : (EMAC Offset: 0x30) Interrupt Mask Register -------- 
// - -------- EMAC_MAN : (EMAC Offset: 0x34) PHY Maintenance Register -------- 
AT91C_EMAC_DATA           EQU (0xFFFF <<  0) ;- (EMAC) 
AT91C_EMAC_CODE           EQU (0x3 << 16) ;- (EMAC) 
AT91C_EMAC_REGA           EQU (0x1F << 18) ;- (EMAC) 
AT91C_EMAC_PHYA           EQU (0x1F << 23) ;- (EMAC) 
AT91C_EMAC_RW             EQU (0x3 << 28) ;- (EMAC) 
AT91C_EMAC_SOF            EQU (0x3 << 30) ;- (EMAC) 
// - -------- EMAC_USRIO : (EMAC Offset: 0xc0) USER Input Output Register -------- 
AT91C_EMAC_RMII           EQU (0x1 <<  0) ;- (EMAC) Reduce MII
AT91C_EMAC_CLKEN          EQU (0x1 <<  1) ;- (EMAC) Clock Enable
// - -------- EMAC_WOL : (EMAC Offset: 0xc4) Wake On LAN Register -------- 
AT91C_EMAC_IP             EQU (0xFFFF <<  0) ;- (EMAC) ARP request IP address
AT91C_EMAC_MAG            EQU (0x1 << 16) ;- (EMAC) Magic packet event enable
AT91C_EMAC_ARP            EQU (0x1 << 17) ;- (EMAC) ARP request event enable
AT91C_EMAC_SA1            EQU (0x1 << 18) ;- (EMAC) Specific address register 1 event enable
// - -------- EMAC_REV : (EMAC Offset: 0xfc) Revision Register -------- 
AT91C_EMAC_REVREF         EQU (0xFFFF <<  0) ;- (EMAC) 
AT91C_EMAC_PARTREF        EQU (0xFFFF << 16) ;- (EMAC) 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Image Sensor Interface
// - *****************************************************************************
// - -------- ISI_CFG1 : (ISI Offset: 0x0) ISI Configuration Register 1 -------- 
AT91C_ISI_HSYNC_POL       EQU (0x1 <<  2) ;- (ISI) Horizontal synchronization polarity
AT91C_ISI_HSYNC_POL_ACTIVE_HIGH EQU (0x0 <<  2) ;- (ISI) HSYNC active high.
AT91C_ISI_HSYNC_POL_ACTIVE_LOW EQU (0x1 <<  2) ;- (ISI) HSYNC active low.
AT91C_ISI_VSYNC_POL       EQU (0x1 <<  3) ;- (ISI) Vertical synchronization polarity
AT91C_ISI_VSYNC_POL_ACTIVE_HIGH EQU (0x0 <<  3) ;- (ISI) VSYNC active high.
AT91C_ISI_VSYNC_POL_ACTIVE_LOW EQU (0x1 <<  3) ;- (ISI) VSYNC active low.
AT91C_ISI_PIXCLK_POL      EQU (0x1 <<  4) ;- (ISI) Pixel Clock Polarity
AT91C_ISI_PIXCLK_POL_RISING_EDGE EQU (0x0 <<  4) ;- (ISI) Data is sampled on rising edge of pixel clock.
AT91C_ISI_PIXCLK_POL_FALLING_EDGE EQU (0x1 <<  4) ;- (ISI) Data is sampled on falling edge of pixel clock.
AT91C_ISI_EMB_SYNC        EQU (0x1 <<  6) ;- (ISI) Embedded synchronisation
AT91C_ISI_EMB_SYNC_HSYNC_VSYNC EQU (0x0 <<  6) ;- (ISI) Synchronization by HSYNC, VSYNC.
AT91C_ISI_EMB_SYNC_SAV_EAV EQU (0x1 <<  6) ;- (ISI) Synchronisation by Embedded Synchronization Sequence SAV/EAV.
AT91C_ISI_CRC_SYNC        EQU (0x1 <<  7) ;- (ISI) CRC correction
AT91C_ISI_CRC_SYNC_CORRECTION_OFF EQU (0x0 <<  7) ;- (ISI) No CRC correction performed on embedded synchronization.
AT91C_ISI_CRC_SYNC_CORRECTION_ON EQU (0x1 <<  7) ;- (ISI) CRC correction is performed.
AT91C_ISI_FRATE           EQU (0x7 <<  8) ;- (ISI) Frame rate capture
AT91C_ISI_FULL            EQU (0x1 << 12) ;- (ISI) Full mode is allowed
AT91C_ISI_FULL_MODE_DISABLE EQU (0x0 << 12) ;- (ISI) Full mode disabled.
AT91C_ISI_FULL_MODE_ENABLE EQU (0x1 << 12) ;- (ISI) both codec and preview datapath are working simultaneously.
AT91C_ISI_THMASK          EQU (0x3 << 13) ;- (ISI) DMA Burst Mask
AT91C_ISI_THMASK_4_BURST  EQU (0x0 << 13) ;- (ISI) Only 4 beats AHB bursts are allowed
AT91C_ISI_THMASK_4_8_BURST EQU (0x1 << 13) ;- (ISI) Only 4 and 8 beats AHB bursts are allowed
AT91C_ISI_THMASK_4_8_16_BURST EQU (0x2 << 13) ;- (ISI) 4, 8 and 16 beats AHB bursts are allowed
AT91C_ISI_SLD             EQU (0xFF << 16) ;- (ISI) Start of Line Delay
AT91C_ISI_SFD             EQU (0xFF << 24) ;- (ISI) Start of frame Delay
// - -------- ISI_CFG2 : (ISI Offset: 0x4) ISI Control Register 2 -------- 
AT91C_ISI_IM_VSIZE        EQU (0x7FF <<  0) ;- (ISI) Vertical size of the Image sensor [0..2047]
AT91C_ISI_GS_MODE         EQU (0x1 << 11) ;- (ISI) Grayscale Memory Mode
AT91C_ISI_GS_MODE_2_PIXELS EQU (0x0 << 11) ;- (ISI) 2 pixels per word.
AT91C_ISI_GS_MODE_1_PIXEL EQU (0x1 << 11) ;- (ISI) 1 pixel per word.
AT91C_ISI_RGB_MODE        EQU (0x1 << 12) ;- (ISI) RGB mode
AT91C_ISI_RGB_MODE_RGB_888 EQU (0x0 << 12) ;- (ISI) RGB 8:8:8 24 bits
AT91C_ISI_RGB_MODE_RGB_565 EQU (0x1 << 12) ;- (ISI) RGB 5:6:5 16 bits
AT91C_ISI_GRAYSCALE       EQU (0x1 << 13) ;- (ISI) Grayscale Mode
AT91C_ISI_GRAYSCALE_DISABLE EQU (0x0 << 13) ;- (ISI) Grayscale mode is disabled
AT91C_ISI_GRAYSCALE_ENABLE EQU (0x1 << 13) ;- (ISI) Input image is assumed to be grayscale coded
AT91C_ISI_RGB_SWAP        EQU (0x1 << 14) ;- (ISI) RGB Swap
AT91C_ISI_RGB_SWAP_DISABLE EQU (0x0 << 14) ;- (ISI) D7 -> R7
AT91C_ISI_RGB_SWAP_ENABLE EQU (0x1 << 14) ;- (ISI) D0 -> R7
AT91C_ISI_COL_SPACE       EQU (0x1 << 15) ;- (ISI) Color space for the image data
AT91C_ISI_COL_SPACE_YCBCR EQU (0x0 << 15) ;- (ISI) YCbCr
AT91C_ISI_COL_SPACE_RGB   EQU (0x1 << 15) ;- (ISI) RGB
AT91C_ISI_IM_HSIZE        EQU (0x7FF << 16) ;- (ISI) Horizontal size of the Image sensor [0..2047]
AT91C_ISI_YCC_SWAP        EQU (0x3 << 28) ;- (ISI) Ycc swap
AT91C_ISI_YCC_SWAP_YCC_DEFAULT EQU (0x0 << 28) ;- (ISI) Cb(i) Y(i) Cr(i) Y(i+1)
AT91C_ISI_YCC_SWAP_YCC_MODE1 EQU (0x1 << 28) ;- (ISI) Cr(i) Y(i) Cb(i) Y(i+1)
AT91C_ISI_YCC_SWAP_YCC_MODE2 EQU (0x2 << 28) ;- (ISI) Y(i) Cb(i) Y(i+1) Cr(i)
AT91C_ISI_YCC_SWAP_YCC_MODE3 EQU (0x3 << 28) ;- (ISI) Y(i) Cr(i) Y(i+1) Cb(i)
AT91C_ISI_RGB_CFG         EQU (0x3 << 30) ;- (ISI) RGB configuration
AT91C_ISI_RGB_CFG_RGB_DEFAULT EQU (0x0 << 30) ;- (ISI) R/G(MSB)  G(LSB)/B  R/G(MSB)  G(LSB)/B
AT91C_ISI_RGB_CFG_RGB_MODE1 EQU (0x1 << 30) ;- (ISI) B/G(MSB)  G(LSB)/R  B/G(MSB)  G(LSB)/R
AT91C_ISI_RGB_CFG_RGB_MODE2 EQU (0x2 << 30) ;- (ISI) G(LSB)/R  B/G(MSB)  G(LSB)/R  B/G(MSB)
AT91C_ISI_RGB_CFG_RGB_MODE3 EQU (0x3 << 30) ;- (ISI) G(LSB)/B  R/G(MSB)  G(LSB)/B  R/G(MSB)
// - -------- ISI_PSIZE : (ISI Offset: 0x8) ISI Preview Register -------- 
AT91C_ISI_PREV_VSIZE      EQU (0x3FF <<  0) ;- (ISI) Vertical size for the preview path
AT91C_ISI_PREV_HSIZE      EQU (0x3FF << 16) ;- (ISI) Horizontal size for the preview path
// - -------- ISI_PDECF : (ISI Offset: 0xc) ISI Preview Decimation Factor Register -------- 
AT91C_ISI_DEC_FACTOR      EQU (0xFF <<  0) ;- (ISI) Decimation factor
// - -------- ISI_Y2RSET0 : (ISI Offset: 0x10) Color Space Conversion YCrCb to RGB Register -------- 
AT91C_ISI_Y2R_C0          EQU (0xFF <<  0) ;- (ISI) Color Space Conversion Matrix Coefficient C0
AT91C_ISI_Y2R_C1          EQU (0xFF <<  8) ;- (ISI) Color Space Conversion Matrix Coefficient C1
AT91C_ISI_Y2R_C2          EQU (0xFF << 16) ;- (ISI) Color Space Conversion Matrix Coefficient C2
AT91C_ISI_Y2R_C3          EQU (0xFF << 24) ;- (ISI) Color Space Conversion Matrix Coefficient C3
// - -------- ISI_Y2RSET1 : (ISI Offset: 0x14) ISI Color Space Conversion YCrCb to RGB set 1 Register -------- 
AT91C_ISI_Y2R_C4          EQU (0x1FF <<  0) ;- (ISI) Color Space Conversion Matrix Coefficient C4
AT91C_ISI_Y2R_YOFF        EQU (0x1 << 12) ;- (ISI) Color Space Conversion Luninance default offset
AT91C_ISI_Y2R_YOFF_0      EQU (0x0 << 12) ;- (ISI) Offset is 0
AT91C_ISI_Y2R_YOFF_128    EQU (0x1 << 12) ;- (ISI) Offset is 128
AT91C_ISI_Y2R_CROFF       EQU (0x1 << 13) ;- (ISI) Color Space Conversion Red Chrominance default offset
AT91C_ISI_Y2R_CROFF_0     EQU (0x0 << 13) ;- (ISI) Offset is 0
AT91C_ISI_Y2R_CROFF_16    EQU (0x1 << 13) ;- (ISI) Offset is 16
AT91C_ISI_Y2R_CBOFF       EQU (0x1 << 14) ;- (ISI) Color Space Conversion Blue Chrominance default offset
AT91C_ISI_Y2R_CBOFF_0     EQU (0x0 << 14) ;- (ISI) Offset is 0
AT91C_ISI_Y2R_CBOFF_16    EQU (0x1 << 14) ;- (ISI) Offset is 16
// - -------- ISI_R2YSET0 : (ISI Offset: 0x18) Color Space Conversion RGB to YCrCb set 0 register -------- 
AT91C_ISI_R2Y_C0          EQU (0x7F <<  0) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C0
AT91C_ISI_R2Y_C1          EQU (0x7F <<  8) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C1
AT91C_ISI_R2Y_C2          EQU (0x7F << 16) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C2
AT91C_ISI_R2Y_ROFF        EQU (0x1 << 24) ;- (ISI) Color Space Conversion Red component offset
AT91C_ISI_R2Y_ROFF_0      EQU (0x0 << 24) ;- (ISI) Offset is 0
AT91C_ISI_R2Y_ROFF_16     EQU (0x1 << 24) ;- (ISI) Offset is 16
// - -------- ISI_R2YSET1 : (ISI Offset: 0x1c) Color Space Conversion RGB to YCrCb set 1 register -------- 
AT91C_ISI_R2Y_C3          EQU (0x7F <<  0) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C3
AT91C_ISI_R2Y_C4          EQU (0x7F <<  8) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C4
AT91C_ISI_R2Y_C5          EQU (0x7F << 16) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C5
AT91C_ISI_R2Y_GOFF        EQU (0x1 << 24) ;- (ISI) Color Space Conversion Green component offset
AT91C_ISI_R2Y_GOFF_0      EQU (0x0 << 24) ;- (ISI) Offset is 0
AT91C_ISI_R2Y_GOFF_128    EQU (0x1 << 24) ;- (ISI) Offset is 128
// - -------- ISI_R2YSET2 : (ISI Offset: 0x20) Color Space Conversion RGB to YCrCb set 2 register -------- 
AT91C_ISI_R2Y_C6          EQU (0x7F <<  0) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C6
AT91C_ISI_R2Y_C7          EQU (0x7F <<  8) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C7
AT91C_ISI_R2Y_C8          EQU (0x7F << 16) ;- (ISI) Color Space Conversion RGB to YCrCb Matrix coefficient C8
AT91C_ISI_R2Y_BOFF        EQU (0x1 << 24) ;- (ISI) Color Space Conversion Blue component offset
AT91C_ISI_R2Y_BOFF_0      EQU (0x0 << 24) ;- (ISI) Offset is 0
AT91C_ISI_R2Y_BOFF_128    EQU (0x1 << 24) ;- (ISI) Offset is 128
// - -------- ISI_CTRL : (ISI Offset: 0x24) ISI Control Register -------- 
AT91C_ISI_EN              EQU (0x1 <<  0) ;- (ISI) Image Sensor Interface Enable Request
AT91C_ISI_EN_0            EQU (0x0) ;- (ISI) No effect
AT91C_ISI_EN_1            EQU (0x1) ;- (ISI) Enable the module and the capture
AT91C_ISI_DIS             EQU (0x1 <<  1) ;- (ISI) Image Sensor Interface Disable Request
AT91C_ISI_DIS_0           EQU (0x0 <<  1) ;- (ISI) No effect
AT91C_ISI_DIS_1           EQU (0x1 <<  1) ;- (ISI) Disable the module and the capture
AT91C_ISI_SRST            EQU (0x1 <<  2) ;- (ISI) Software Reset Request
AT91C_ISI_SRST_0          EQU (0x0 <<  2) ;- (ISI) No effect
AT91C_ISI_SRST_1          EQU (0x1 <<  2) ;- (ISI) Reset the module
AT91C_ISI_CDC             EQU (0x1 <<  8) ;- (ISI) Codec Request
AT91C_ISI_CDC_0           EQU (0x0 <<  8) ;- (ISI) No effect
AT91C_ISI_CDC_1           EQU (0x1 <<  8) ;- (ISI) Enable the Codec
// - -------- ISI_SR : (ISI Offset: 0x28) ISI Status Register -------- 
AT91C_ISI_VSYNC           EQU (0x1 << 10) ;- (ISI) Vertical Synchronization
AT91C_ISI_VSYNC_0         EQU (0x0 << 10) ;- (ISI) No effect
AT91C_ISI_VSYNC_1         EQU (0x1 << 10) ;- (ISI) Indicates that a Vertical Synchronization has been detected since last read
AT91C_ISI_PXFR_DONE       EQU (0x1 << 16) ;- (ISI) Preview DMA transfer terminated
AT91C_ISI_PXFR_DONE_0     EQU (0x0 << 16) ;- (ISI) No effect
AT91C_ISI_PXFR_DONE_1     EQU (0x1 << 16) ;- (ISI) Indicates that DATA transfer on preview channel has completed since last read
AT91C_ISI_CXFR_DONE       EQU (0x1 << 17) ;- (ISI) Codec DMA transfer terminated
AT91C_ISI_CXFR_DONE_0     EQU (0x0 << 17) ;- (ISI) No effect
AT91C_ISI_CXFR_DONE_1     EQU (0x1 << 17) ;- (ISI) Indicates that DATA transfer on preview channel has completed since last read
AT91C_ISI_SIP             EQU (0x1 << 19) ;- (ISI) Synchronization In Progress
AT91C_ISI_SIP_0           EQU (0x0 << 19) ;- (ISI) No effect
AT91C_ISI_SIP_1           EQU (0x1 << 19) ;- (ISI) Indicates that Synchronization is in progress
AT91C_ISI_P_OVR           EQU (0x1 << 24) ;- (ISI) Fifo Preview Overflow 
AT91C_ISI_P_OVR_0         EQU (0x0 << 24) ;- (ISI) No error
AT91C_ISI_P_OVR_1         EQU (0x1 << 24) ;- (ISI) An overrun condition has occurred in input FIFO on the preview path
AT91C_ISI_C_OVR           EQU (0x1 << 25) ;- (ISI) Fifo Codec Overflow 
AT91C_ISI_C_OVR_0         EQU (0x0 << 25) ;- (ISI) No error
AT91C_ISI_C_OVR_1         EQU (0x1 << 25) ;- (ISI) An overrun condition has occurred in input FIFO on the codec path
AT91C_ISI_CRC_ERR         EQU (0x1 << 26) ;- (ISI) CRC synchronisation error
AT91C_ISI_CRC_ERR_0       EQU (0x0 << 26) ;- (ISI) No error
AT91C_ISI_CRC_ERR_1       EQU (0x1 << 26) ;- (ISI) CRC_SYNC is enabled in the control register and an error has been detected and not corrected. The frame is discarded and the ISI waits for a new one.
AT91C_ISI_FR_OVR          EQU (0x1 << 27) ;- (ISI) Frame rate overun
AT91C_ISI_FR_OVR_0        EQU (0x0 << 27) ;- (ISI) No error
AT91C_ISI_FR_OVR_1        EQU (0x1 << 27) ;- (ISI) Frame overrun, the current frame is being skipped because a vsync signal has been detected while flushing FIFOs.
// - -------- ISI_IER : (ISI Offset: 0x2c) ISI Interrupt Enable Register -------- 
// - -------- ISI_IDR : (ISI Offset: 0x30) ISI Interrupt Disable Register -------- 
// - -------- ISI_IMR : (ISI Offset: 0x34) ISI Interrupt Mask Register -------- 
// - -------- ISI_DMACHER : (ISI Offset: 0x38) DMA Channel Enable Register -------- 
AT91C_ISI_P_CH_EN         EQU (0x1 <<  0) ;- (ISI) Preview Channel Enable
AT91C_ISI_P_CH_EN_0       EQU (0x0) ;- (ISI) No effect
AT91C_ISI_P_CH_EN_1       EQU (0x1) ;- (ISI) Enable the Preview Channel
AT91C_ISI_C_CH_EN         EQU (0x1 <<  1) ;- (ISI) Codec Channel Enable
AT91C_ISI_C_CH_EN_0       EQU (0x0 <<  1) ;- (ISI) No effect
AT91C_ISI_C_CH_EN_1       EQU (0x1 <<  1) ;- (ISI) Enable the Codec Channel
// - -------- ISI_DMACHDR : (ISI Offset: 0x3c) DMA Channel Enable Register -------- 
AT91C_ISI_P_CH_DIS        EQU (0x1 <<  0) ;- (ISI) Preview Channel Disable
AT91C_ISI_P_CH_DIS_0      EQU (0x0) ;- (ISI) No effect
AT91C_ISI_P_CH_DIS_1      EQU (0x1) ;- (ISI) Disable the Preview Channel
AT91C_ISI_C_CH_DIS        EQU (0x1 <<  1) ;- (ISI) Codec Channel Disable
AT91C_ISI_C_CH_DIS_0      EQU (0x0 <<  1) ;- (ISI) No effect
AT91C_ISI_C_CH_DIS_1      EQU (0x1 <<  1) ;- (ISI) Disable the Codec Channel
// - -------- ISI_DMACHSR : (ISI Offset: 0x40) DMA Channel Status Register -------- 
AT91C_ISI_P_CH_S          EQU (0x1 <<  0) ;- (ISI) Preview Channel Disable
AT91C_ISI_P_CH_S_0        EQU (0x0) ;- (ISI) Preview Channel is disabled
AT91C_ISI_P_CH_S_1        EQU (0x1) ;- (ISI) Preview Channel is enabled
AT91C_ISI_C_CH_S          EQU (0x1 <<  1) ;- (ISI) Codec Channel Disable
AT91C_ISI_C_CH_S_0        EQU (0x0 <<  1) ;- (ISI) Codec Channel is disabled
AT91C_ISI_C_CH_S_1        EQU (0x1 <<  1) ;- (ISI) Codec Channel is enabled
// - -------- ISI_DMAPCTRL : (ISI Offset: 0x48) DMA Preview Control Register -------- 
AT91C_ISI_P_FETCH         EQU (0x1 <<  0) ;- (ISI) Preview Descriptor Fetch Control Field
AT91C_ISI_P_FETCH_DISABLE EQU (0x0) ;- (ISI) Preview Channel Fetch Operation is disabled
AT91C_ISI_P_FETCH_ENABLE  EQU (0x1) ;- (ISI) Preview Channel Fetch Operation is enabled
AT91C_ISI_P_DONE          EQU (0x1 <<  1) ;- (ISI) Preview Transfer Done Flag
AT91C_ISI_P_DONE_0        EQU (0x0 <<  1) ;- (ISI) Preview Transfer has not been performed
AT91C_ISI_P_DONE_1        EQU (0x1 <<  1) ;- (ISI) Preview Transfer has completed
// - -------- ISI_DMACCTRL : (ISI Offset: 0x54) DMA Codec Control Register -------- 
AT91C_ISI_C_FETCH         EQU (0x1 <<  0) ;- (ISI) Codec Descriptor Fetch Control Field
AT91C_ISI_C_FETCH_DISABLE EQU (0x0) ;- (ISI) Codec Channel Fetch Operation is disabled
AT91C_ISI_C_FETCH_ENABLE  EQU (0x1) ;- (ISI) Codec Channel Fetch Operation is enabled
AT91C_ISI_C_DONE          EQU (0x1 <<  1) ;- (ISI) Codec Transfer Done Flag
AT91C_ISI_C_DONE_0        EQU (0x0 <<  1) ;- (ISI) Codec Transfer has not been performed
AT91C_ISI_C_DONE_1        EQU (0x1 <<  1) ;- (ISI) Codec Transfer has completed
// - -------- ISI_WPCR : (ISI Offset: 0xe4) Write Protection Control Register -------- 
AT91C_ISI_WP_EN           EQU (0x1 <<  0) ;- (ISI) Write Protection Enable
AT91C_ISI_WP_EN_DISABLE   EQU (0x0) ;- (ISI) Write Operation is disabled (if WP_KEY corresponds)
AT91C_ISI_WP_EN_ENABLE    EQU (0x1) ;- (ISI) Write Operation is enabled (if WP_KEY corresponds)
AT91C_ISI_WP_KEY          EQU (0xFFFFFF <<  8) ;- (ISI) Write Protection Key
// - -------- ISI_WPSR : (ISI Offset: 0xe8) Write Protection Status Register -------- 
AT91C_ISI_WP_VS           EQU (0xF <<  0) ;- (ISI) Write Protection Violation Status
AT91C_ISI_WP_VS_NO_VIOLATION EQU (0x0) ;- (ISI) No Write Protection Violation detected since last read
AT91C_ISI_WP_VS_ON_WRITE  EQU (0x1) ;- (ISI) Write Protection Violation detected since last read
AT91C_ISI_WP_VS_ON_RESET  EQU (0x2) ;- (ISI) Software Reset Violation detected since last read
AT91C_ISI_WP_VS_ON_BOTH   EQU (0x3) ;- (ISI) Write Protection and Software Reset Violation detected since last read
AT91C_ISI_WP_VSRC         EQU (0xF <<  8) ;- (ISI) Write Protection Violation Source
AT91C_ISI_WP_VSRC_NO_VIOLATION EQU (0x0 <<  8) ;- (ISI) No Write Protection Violation detected since last read
AT91C_ISI_WP_VSRC_ISI_CFG1 EQU (0x1 <<  8) ;- (ISI) Write Protection Violation detected on ISI_CFG1 since last read
AT91C_ISI_WP_VSRC_ISI_CFG2 EQU (0x2 <<  8) ;- (ISI) Write Protection Violation detected on ISI_CFG2 since last read
AT91C_ISI_WP_VSRC_ISI_PSIZE EQU (0x3 <<  8) ;- (ISI) Write Protection Violation detected on ISI_PSIZE since last read
AT91C_ISI_WP_VSRC_ISI_PDECF EQU (0x4 <<  8) ;- (ISI) Write Protection Violation detected on ISI_PDECF since last read
AT91C_ISI_WP_VSRC_ISI_Y2RSET0 EQU (0x5 <<  8) ;- (ISI) Write Protection Violation detected on ISI_Y2RSET0 since last read
AT91C_ISI_WP_VSRC_ISI_Y2RSET1 EQU (0x6 <<  8) ;- (ISI) Write Protection Violation detected on ISI_Y2RSET1 since last read
AT91C_ISI_WP_VSRC_ISI_R2YSET0 EQU (0x7 <<  8) ;- (ISI) Write Protection Violation detected on ISI_R2YSET0 since last read
AT91C_ISI_WP_VSRC_ISI_R2YSET1 EQU (0x8 <<  8) ;- (ISI) Write Protection Violation detected on ISI_R2YSET1 since last read
AT91C_ISI_WP_VSRC_ISI_R2YSET2 EQU (0x9 <<  8) ;- (ISI) Write Protection Violation detected on ISI_R2YSET2 since last read

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR USB Host Interface
// - *****************************************************************************

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR USB Host Interface
// - *****************************************************************************
// - -------- VERSION : (UHPHS_EHCI Offset: 0x0)  -------- 
AT91C_UHPHS_CAPLENGTH     EQU (0xFF <<  0) ;- (UHPHS_EHCI) CapLength : Offset for control registers
AT91C_UHPHS_HCIVERSION    EQU (0xFF << 16) ;- (UHPHS_EHCI) Hci Version

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR True Random Generator
// - *****************************************************************************
// - -------- TRNG_CR : (TRNG Offset: 0x0) Control Register -------- 
AT91C_TRNG_ENABLE         EQU (0x1 <<  0) ;- (TRNG) Enable TRNG
// - -------- TRNG_IER : (TRNG Offset: 0x10) Interrupt Enable Register -------- 
AT91C_TRNG_DATRDY         EQU (0x1 <<  0) ;- (TRNG) DATRDY
// - -------- TRNG_IDR : (TRNG Offset: 0x14) Interrupt Disable Register -------- 
// - -------- TRNG_IMR : (TRNG Offset: 0x18) Interrupt Mask Register -------- 
// - -------- TRNG_ISR : (TRNG Offset: 0x1c) Interrupt Status Register -------- 

// - *****************************************************************************
// -               REGISTER ADDRESS DEFINITION FOR AT91SAM9G45
// - *****************************************************************************
// - ========== Register definition for SFR peripheral ========== 
AT91C_SFR_INT             EQU (0xFFF74014) ;- (SFR) OHCI suspend Interrupt status
AT91C_SFR_DDRCFG          EQU (0xFFF74004) ;- (SFR) DDR2 SSTL18 control
AT91C_SFR_UTMICFG         EQU (0xFFF74010) ;- (SFR) UTMI Software Reset, and OHCI suspend interrupt control
AT91C_SFR_EBIDELAY        EQU (0xFFF7400C) ;- (SFR) EBI DDR controller clock delay
AT91C_SFR_DDRDELAY        EQU (0xFFF74008) ;- (SFR) DDR2 controller clock delay
AT91C_SFR_EMA             EQU (0xFFF74000) ;- (SFR) memory Extra Margin Adjustment control
// - ========== Register definition for SYS peripheral ========== 
AT91C_SYS_SLCKSEL         EQU (0xFFFFFD50) ;- (SYS) Slow Clock Selection Register
AT91C_SYS_GPBR            EQU (0xFFFFFD60) ;- (SYS) General Purpose Register
// - ========== Register definition for EBI peripheral ========== 
AT91C_EBI_DUMMY           EQU (0xFFFFE200) ;- (EBI) Dummy register - Do not use
// - ========== Register definition for DDR2CP1 peripheral ========== 
AT91C_DDR2CP1_DELAY2      EQU (0xFFFFE434) ;- (DDR2CP1) Pad delay2 Register
AT91C_DDR2CP1_T3PR        EQU (0xFFFFE418) ;- (DDR2CP1) Timing3 Register
AT91C_DDR2CP1_RTR         EQU (0xFFFFE404) ;- (DDR2CP1) Refresh Timer Register
AT91C_DDR2CP1_T0PR        EQU (0xFFFFE40C) ;- (DDR2CP1) Timing0 Register
AT91C_DDR2CP1_WPSR        EQU (0xFFFFE4E8) ;- (DDR2CP1) High Speed Register
AT91C_DDR2CP1_DELAY8      EQU (0xFFFFE44C) ;- (DDR2CP1) Pad delay8 Register
AT91C_DDR2CP1_LPR         EQU (0xFFFFE41C) ;- (DDR2CP1) Low-power Register
AT91C_DDR2CP1_VER         EQU (0xFFFFE428) ;- (DDR2CP1) DLL Version Register
AT91C_DDR2CP1_DELAY7      EQU (0xFFFFE448) ;- (DDR2CP1) Pad delay7 Register
AT91C_DDR2CP1_CR          EQU (0xFFFFE408) ;- (DDR2CP1) Configuration Register
AT91C_DDR2CP1_WPCR        EQU (0xFFFFE4E4) ;- (DDR2CP1) High Speed Register
AT91C_DDR2CP1_MR          EQU (0xFFFFE400) ;- (DDR2CP1) Mode Register
AT91C_DDR2CP1_DELAY5      EQU (0xFFFFE440) ;- (DDR2CP1) Pad delay5 Register
AT91C_DDR2CP1_T2PR        EQU (0xFFFFE414) ;- (DDR2CP1) Timing2 Register
AT91C_DDR2CP1_HS          EQU (0xFFFFE42C) ;- (DDR2CP1) High Speed Register
AT91C_DDR2CP1_MDR         EQU (0xFFFFE420) ;- (DDR2CP1) Memory Device Register
AT91C_DDR2CP1_DELAY4      EQU (0xFFFFE43C) ;- (DDR2CP1) Pad delay4 Register
AT91C_DDR2CP1_DELAY1      EQU (0xFFFFE430) ;- (DDR2CP1) Pad delay1 Register
AT91C_DDR2CP1_DELAY6      EQU (0xFFFFE444) ;- (DDR2CP1) Pad delay6 Register
AT91C_DDR2CP1_DLL         EQU (0xFFFFE424) ;- (DDR2CP1) DLL Information Register
AT91C_DDR2CP1_DELAY3      EQU (0xFFFFE438) ;- (DDR2CP1) Pad delay3 Register
AT91C_DDR2CP1_VERSION     EQU (0xFFFFE4FC) ;- (DDR2CP1) Version Register
AT91C_DDR2CP1_T1PR        EQU (0xFFFFE410) ;- (DDR2CP1) Timing1 Register
// - ========== Register definition for DDR2C peripheral ========== 
AT91C_DDR2C_DELAY8        EQU (0xFFFFE64C) ;- (DDR2C) Pad delay8 Register
AT91C_DDR2C_VER           EQU (0xFFFFE628) ;- (DDR2C) DLL Version Register
AT91C_DDR2C_RTR           EQU (0xFFFFE604) ;- (DDR2C) Refresh Timer Register
AT91C_DDR2C_T0PR          EQU (0xFFFFE60C) ;- (DDR2C) Timing0 Register
AT91C_DDR2C_DELAY5        EQU (0xFFFFE640) ;- (DDR2C) Pad delay5 Register
AT91C_DDR2C_LPR           EQU (0xFFFFE61C) ;- (DDR2C) Low-power Register
AT91C_DDR2C_HS            EQU (0xFFFFE62C) ;- (DDR2C) High Speed Register
AT91C_DDR2C_DELAY2        EQU (0xFFFFE634) ;- (DDR2C) Pad delay2 Register
AT91C_DDR2C_T2PR          EQU (0xFFFFE614) ;- (DDR2C) Timing2 Register
AT91C_DDR2C_DELAY1        EQU (0xFFFFE630) ;- (DDR2C) Pad delay1 Register
AT91C_DDR2C_T1PR          EQU (0xFFFFE610) ;- (DDR2C) Timing1 Register
AT91C_DDR2C_MDR           EQU (0xFFFFE620) ;- (DDR2C) Memory Device Register
AT91C_DDR2C_DELAY6        EQU (0xFFFFE644) ;- (DDR2C) Pad delay6 Register
AT91C_DDR2C_VERSION       EQU (0xFFFFE6FC) ;- (DDR2C) Version Register
AT91C_DDR2C_MR            EQU (0xFFFFE600) ;- (DDR2C) Mode Register
AT91C_DDR2C_DLL           EQU (0xFFFFE624) ;- (DDR2C) DLL Information Register
AT91C_DDR2C_DELAY4        EQU (0xFFFFE63C) ;- (DDR2C) Pad delay4 Register
AT91C_DDR2C_WPCR          EQU (0xFFFFE6E4) ;- (DDR2C) High Speed Register
AT91C_DDR2C_CR            EQU (0xFFFFE608) ;- (DDR2C) Configuration Register
AT91C_DDR2C_DELAY3        EQU (0xFFFFE638) ;- (DDR2C) Pad delay3 Register
AT91C_DDR2C_WPSR          EQU (0xFFFFE6E8) ;- (DDR2C) High Speed Register
AT91C_DDR2C_DELAY7        EQU (0xFFFFE648) ;- (DDR2C) Pad delay7 Register
AT91C_DDR2C_T3PR          EQU (0xFFFFE618) ;- (DDR2C) Timing3 Register
// - ========== Register definition for SMC peripheral ========== 
AT91C_SMC_PULSE7          EQU (0xFFFFE874) ;- (SMC)  Pulse Register for CS 7
AT91C_SMC_DELAY1          EQU (0xFFFFE8C0) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CYCLE2          EQU (0xFFFFE828) ;- (SMC)  Cycle Register for CS 2
AT91C_SMC_DELAY5          EQU (0xFFFFE8D0) ;- (SMC) SMC Delay Control Register
AT91C_SMC_DELAY6          EQU (0xFFFFE8D4) ;- (SMC) SMC Delay Control Register
AT91C_SMC_PULSE2          EQU (0xFFFFE824) ;- (SMC)  Pulse Register for CS 2
AT91C_SMC_SETUP6          EQU (0xFFFFE860) ;- (SMC)  Setup Register for CS 6
AT91C_SMC_SETUP5          EQU (0xFFFFE850) ;- (SMC)  Setup Register for CS 5
AT91C_SMC_CYCLE6          EQU (0xFFFFE868) ;- (SMC)  Cycle Register for CS 6
AT91C_SMC_PULSE6          EQU (0xFFFFE864) ;- (SMC)  Pulse Register for CS 6
AT91C_SMC_CTRL5           EQU (0xFFFFE85C) ;- (SMC)  Control Register for CS 5
AT91C_SMC_CTRL3           EQU (0xFFFFE83C) ;- (SMC)  Control Register for CS 3
AT91C_SMC_DELAY7          EQU (0xFFFFE8D8) ;- (SMC) SMC Delay Control Register
AT91C_SMC_DELAY3          EQU (0xFFFFE8C8) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CYCLE0          EQU (0xFFFFE808) ;- (SMC)  Cycle Register for CS 0
AT91C_SMC_SETUP1          EQU (0xFFFFE810) ;- (SMC)  Setup Register for CS 1
AT91C_SMC_PULSE5          EQU (0xFFFFE854) ;- (SMC)  Pulse Register for CS 5
AT91C_SMC_SETUP7          EQU (0xFFFFE870) ;- (SMC)  Setup Register for CS 7
AT91C_SMC_CTRL4           EQU (0xFFFFE84C) ;- (SMC)  Control Register for CS 4
AT91C_SMC_DELAY2          EQU (0xFFFFE8C4) ;- (SMC) SMC Delay Control Register
AT91C_SMC_PULSE3          EQU (0xFFFFE834) ;- (SMC)  Pulse Register for CS 3
AT91C_SMC_CYCLE4          EQU (0xFFFFE848) ;- (SMC)  Cycle Register for CS 4
AT91C_SMC_CTRL1           EQU (0xFFFFE81C) ;- (SMC)  Control Register for CS 1
AT91C_SMC_SETUP3          EQU (0xFFFFE830) ;- (SMC)  Setup Register for CS 3
AT91C_SMC_CTRL0           EQU (0xFFFFE80C) ;- (SMC)  Control Register for CS 0
AT91C_SMC_CYCLE7          EQU (0xFFFFE878) ;- (SMC)  Cycle Register for CS 7
AT91C_SMC_DELAY4          EQU (0xFFFFE8CC) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CYCLE1          EQU (0xFFFFE818) ;- (SMC)  Cycle Register for CS 1
AT91C_SMC_SETUP2          EQU (0xFFFFE820) ;- (SMC)  Setup Register for CS 2
AT91C_SMC_PULSE1          EQU (0xFFFFE814) ;- (SMC)  Pulse Register for CS 1
AT91C_SMC_DELAY8          EQU (0xFFFFE8DC) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CTRL2           EQU (0xFFFFE82C) ;- (SMC)  Control Register for CS 2
AT91C_SMC_PULSE4          EQU (0xFFFFE844) ;- (SMC)  Pulse Register for CS 4
AT91C_SMC_SETUP4          EQU (0xFFFFE840) ;- (SMC)  Setup Register for CS 4
AT91C_SMC_CYCLE3          EQU (0xFFFFE838) ;- (SMC)  Cycle Register for CS 3
AT91C_SMC_SETUP0          EQU (0xFFFFE800) ;- (SMC)  Setup Register for CS 0
AT91C_SMC_CYCLE5          EQU (0xFFFFE858) ;- (SMC)  Cycle Register for CS 5
AT91C_SMC_PULSE0          EQU (0xFFFFE804) ;- (SMC)  Pulse Register for CS 0
AT91C_SMC_CTRL6           EQU (0xFFFFE86C) ;- (SMC)  Control Register for CS 6
AT91C_SMC_CTRL7           EQU (0xFFFFE87C) ;- (SMC)  Control Register for CS 7
// - ========== Register definition for MATRIX peripheral ========== 
AT91C_MATRIX_SCFG1        EQU (0xFFFFEA44) ;- (MATRIX)  Slave Configuration Register 1 : SRAM S1
AT91C_MATRIX_MRCR         EQU (0xFFFFEB00) ;- (MATRIX)  Master Remap Control Register 
AT91C_MATRIX_PRAS2        EQU (0xFFFFEA90) ;- (MATRIX)  PRAS2 : SRAM S2
AT91C_MATRIX_PRAS1        EQU (0xFFFFEA88) ;- (MATRIX)  PRAS1 : SRAM S1
AT91C_MATRIX_PRAS0        EQU (0xFFFFEA80) ;- (MATRIX)  PRAS0 : SRAM S0
AT91C_MATRIX_MCFG8        EQU (0xFFFFEA20) ;- (MATRIX)  Master Configuration Register 8 : eMAC
AT91C_MATRIX_MCFG2        EQU (0xFFFFEA08) ;- (MATRIX)  Master Configuration Register 2 : pdc
AT91C_MATRIX_EBICSA       EQU (0xFFFFEB28) ;- (MATRIX)  EBI Chip Select Assignment Register 
AT91C_MATRIX_PRAS4        EQU (0xFFFFEAA0) ;- (MATRIX)  PRAS4 : ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
AT91C_MATRIX_MCFG3        EQU (0xFFFFEA0C) ;- (MATRIX)  Master Configuration Register 3 : USB Host OHCI
AT91C_MATRIX_SCFG0        EQU (0xFFFFEA40) ;- (MATRIX)  Slave Configuration Register 0 : SRAM S0
AT91C_MATRIX_MCFG7        EQU (0xFFFFEA1C) ;- (MATRIX)  Master Configuration Register 7 : lcdc
AT91C_MATRIX_PRAS6        EQU (0xFFFFEAB0) ;- (MATRIX)  PRAS6 : DDR2 S1
AT91C_MATRIX_SCFG7        EQU (0xFFFFEA5C) ;- (MATRIX)  Slave Configuration Register 7 : DDR2 S2
AT91C_MATRIX_PRAS7        EQU (0xFFFFEAB8) ;- (MATRIX)  PRAS7 : DDR2 S2
AT91C_MATRIX_SCFG2        EQU (0xFFFFEA48) ;- (MATRIX)  Slave Configuration Register 2 : SRAM S2
AT91C_MATRIX_WRPROTST     EQU (0xFFFFEBE8) ;- (MATRIX)  Write Protection Status Register 
AT91C_MATRIX_PRBS0        EQU (0xFFFFEA84) ;- (MATRIX)  PRBS0 : SRAM S0
AT91C_MATRIX_PRBS2        EQU (0xFFFFEA94) ;- (MATRIX)  PRBS2 : SRAM S2
AT91C_MATRIX_MCFG4        EQU (0xFFFFEA10) ;- (MATRIX)  Master Configuration Register 4 : DMA0
AT91C_MATRIX_SCFG5        EQU (0xFFFFEA54) ;- (MATRIX)  Slave Configuration Register 5 : DDR2 S0
AT91C_MATRIX_PRBS6        EQU (0xFFFFEAB4) ;- (MATRIX)  PRBS6 : DDR2 S1
AT91C_MATRIX_MCFG1        EQU (0xFFFFEA04) ;- (MATRIX)  Master Configuration Register 1 ; ARM-D
AT91C_MATRIX_SCFG6        EQU (0xFFFFEA58) ;- (MATRIX)  Slave Configuration Register 6 : DDR2 S1
AT91C_MATRIX_SCFG4        EQU (0xFFFFEA50) ;- (MATRIX)  Slave Configuration Register 4 ; ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
AT91C_MATRIX_PRBS1        EQU (0xFFFFEA8C) ;- (MATRIX)  PRBS1 : SRAM S1
AT91C_MATRIX_PRBS3        EQU (0xFFFFEA9C) ;- (MATRIX)  PRBS3 : SRAM S3
AT91C_MATRIX_WRPROTEN     EQU (0xFFFFEBE4) ;- (MATRIX)  Write Protection Control Register 
AT91C_MATRIX_TCMR         EQU (0xFFFFEB10) ;- (MATRIX)  Bus Matrix TCM Configuration Register 
AT91C_MATRIX_MCFG0        EQU (0xFFFFEA00) ;- (MATRIX)  Master Configuration Register 0 : ARM-I
AT91C_MATRIX_PRAS5        EQU (0xFFFFEAA8) ;- (MATRIX)  PRAS5 : DDR2 S0
AT91C_MATRIX_DDRMPR       EQU (0xFFFFEB18) ;- (MATRIX)  DDR Multi-Port Register 
AT91C_MATRIX_PRBS4        EQU (0xFFFFEAA4) ;- (MATRIX)  PRBS4 : ROM + USB Dev + USB EHCI + USB OHCI + LCD + Video Decoder
AT91C_MATRIX_MCFG6        EQU (0xFFFFEA18) ;- (MATRIX)  Master Configuration Register 6 : hisi
AT91C_MATRIX_PRBS5        EQU (0xFFFFEAAC) ;- (MATRIX)  PRBS5 : DDR2 S0
AT91C_MATRIX_MCFG9        EQU (0xFFFFEA24) ;- (MATRIX)  Master Configuration Register 9 : USB Device
AT91C_MATRIX_MCFG5        EQU (0xFFFFEA14) ;- (MATRIX)  Master Configuration Register 5 : DMA1
AT91C_MATRIX_PRBS7        EQU (0xFFFFEABC) ;- (MATRIX)  PRBS7 : DDR2 S2
AT91C_MATRIX_MCFG10       EQU (0xFFFFEA28) ;- (MATRIX)  Master Configuration Register 10 : USB Host EHCI
AT91C_MATRIX_PRAS3        EQU (0xFFFFEA98) ;- (MATRIX)  PRAS3 : SRAM S3
AT91C_MATRIX_SCFG3        EQU (0xFFFFEA4C) ;- (MATRIX)  Slave Configuration Register 3 : SRAM S3
// - ========== Register definition for AIC peripheral ========== 
AT91C_AIC_IVR             EQU (0xFFFFF100) ;- (AIC) IRQ Vector Register
AT91C_AIC_SMR             EQU (0xFFFFF000) ;- (AIC) Source Mode Register
AT91C_AIC_FVR             EQU (0xFFFFF104) ;- (AIC) FIQ Vector Register
AT91C_AIC_DCR             EQU (0xFFFFF138) ;- (AIC) Debug Control Register (Protect)
AT91C_AIC_EOICR           EQU (0xFFFFF130) ;- (AIC) End of Interrupt Command Register
AT91C_AIC_SVR             EQU (0xFFFFF080) ;- (AIC) Source Vector Register
AT91C_AIC_FFSR            EQU (0xFFFFF148) ;- (AIC) Fast Forcing Status Register
AT91C_AIC_ICCR            EQU (0xFFFFF128) ;- (AIC) Interrupt Clear Command Register
AT91C_AIC_ISR             EQU (0xFFFFF108) ;- (AIC) Interrupt Status Register
AT91C_AIC_IMR             EQU (0xFFFFF110) ;- (AIC) Interrupt Mask Register
AT91C_AIC_IPR             EQU (0xFFFFF10C) ;- (AIC) Interrupt Pending Register
AT91C_AIC_FFER            EQU (0xFFFFF140) ;- (AIC) Fast Forcing Enable Register
AT91C_AIC_IECR            EQU (0xFFFFF120) ;- (AIC) Interrupt Enable Command Register
AT91C_AIC_ISCR            EQU (0xFFFFF12C) ;- (AIC) Interrupt Set Command Register
AT91C_AIC_FFDR            EQU (0xFFFFF144) ;- (AIC) Fast Forcing Disable Register
AT91C_AIC_CISR            EQU (0xFFFFF114) ;- (AIC) Core Interrupt Status Register
AT91C_AIC_IDCR            EQU (0xFFFFF124) ;- (AIC) Interrupt Disable Command Register
AT91C_AIC_SPU             EQU (0xFFFFF134) ;- (AIC) Spurious Vector Register
// - ========== Register definition for PDC_DBGU peripheral ========== 
AT91C_DBGU_PTCR           EQU (0xFFFFEF20) ;- (PDC_DBGU) PDC Transfer Control Register
AT91C_DBGU_RCR            EQU (0xFFFFEF04) ;- (PDC_DBGU) Receive Counter Register
AT91C_DBGU_TCR            EQU (0xFFFFEF0C) ;- (PDC_DBGU) Transmit Counter Register
AT91C_DBGU_RNCR           EQU (0xFFFFEF14) ;- (PDC_DBGU) Receive Next Counter Register
AT91C_DBGU_TNPR           EQU (0xFFFFEF18) ;- (PDC_DBGU) Transmit Next Pointer Register
AT91C_DBGU_RNPR           EQU (0xFFFFEF10) ;- (PDC_DBGU) Receive Next Pointer Register
AT91C_DBGU_PTSR           EQU (0xFFFFEF24) ;- (PDC_DBGU) PDC Transfer Status Register
AT91C_DBGU_RPR            EQU (0xFFFFEF00) ;- (PDC_DBGU) Receive Pointer Register
AT91C_DBGU_TPR            EQU (0xFFFFEF08) ;- (PDC_DBGU) Transmit Pointer Register
AT91C_DBGU_TNCR           EQU (0xFFFFEF1C) ;- (PDC_DBGU) Transmit Next Counter Register
// - ========== Register definition for DBGU peripheral ========== 
AT91C_DBGU_BRGR           EQU (0xFFFFEE20) ;- (DBGU) Baud Rate Generator Register
AT91C_DBGU_CR             EQU (0xFFFFEE00) ;- (DBGU) Control Register
AT91C_DBGU_THR            EQU (0xFFFFEE1C) ;- (DBGU) Transmitter Holding Register
AT91C_DBGU_IDR            EQU (0xFFFFEE0C) ;- (DBGU) Interrupt Disable Register
AT91C_DBGU_EXID           EQU (0xFFFFEE44) ;- (DBGU) Chip ID Extension Register
AT91C_DBGU_IMR            EQU (0xFFFFEE10) ;- (DBGU) Interrupt Mask Register
AT91C_DBGU_FNTR           EQU (0xFFFFEE48) ;- (DBGU) Force NTRST Register
AT91C_DBGU_IER            EQU (0xFFFFEE08) ;- (DBGU) Interrupt Enable Register
AT91C_DBGU_CSR            EQU (0xFFFFEE14) ;- (DBGU) Channel Status Register
AT91C_DBGU_MR             EQU (0xFFFFEE04) ;- (DBGU) Mode Register
AT91C_DBGU_RHR            EQU (0xFFFFEE18) ;- (DBGU) Receiver Holding Register
AT91C_DBGU_CIDR           EQU (0xFFFFEE40) ;- (DBGU) Chip ID Register
// - ========== Register definition for PIOA peripheral ========== 
AT91C_PIOA_OWDR           EQU (0xFFFFF2A4) ;- (PIOA) Output Write Disable Register
AT91C_PIOA_DELAY3         EQU (0xFFFFF2C8) ;- (PIOA) PIO Delay Control Register
AT91C_PIOA_ISR            EQU (0xFFFFF24C) ;- (PIOA) Interrupt Status Register
AT91C_PIOA_PDR            EQU (0xFFFFF204) ;- (PIOA) PIO Disable Register
AT91C_PIOA_OSR            EQU (0xFFFFF218) ;- (PIOA) Output Status Register
AT91C_PIOA_ABSR           EQU (0xFFFFF278) ;- (PIOA) AB Select Status Register
AT91C_PIOA_DELAY2         EQU (0xFFFFF2C4) ;- (PIOA) PIO Delay Control Register
AT91C_PIOA_PDSR           EQU (0xFFFFF23C) ;- (PIOA) Pin Data Status Register
AT91C_PIOA_BSR            EQU (0xFFFFF274) ;- (PIOA) Select B Register
AT91C_PIOA_DELAY1         EQU (0xFFFFF2C0) ;- (PIOA) PIO Delay Control Register
AT91C_PIOA_PPUER          EQU (0xFFFFF264) ;- (PIOA) Pull-up Enable Register
AT91C_PIOA_OER            EQU (0xFFFFF210) ;- (PIOA) Output Enable Register
AT91C_PIOA_PER            EQU (0xFFFFF200) ;- (PIOA) PIO Enable Register
AT91C_PIOA_VERSION        EQU (0xFFFFF2FC) ;- (PIOA) PIO Version Register
AT91C_PIOA_PPUDR          EQU (0xFFFFF260) ;- (PIOA) Pull-up Disable Register
AT91C_PIOA_ODSR           EQU (0xFFFFF238) ;- (PIOA) Output Data Status Register
AT91C_PIOA_SLEWRATE1      EQU (0xFFFFF2B0) ;- (PIOA) PIO Slewrate Control Register
AT91C_PIOA_MDDR           EQU (0xFFFFF254) ;- (PIOA) Multi-driver Disable Register
AT91C_PIOA_IFSR           EQU (0xFFFFF228) ;- (PIOA) Input Filter Status Register
AT91C_PIOA_CODR           EQU (0xFFFFF234) ;- (PIOA) Clear Output Data Register
AT91C_PIOA_ASR            EQU (0xFFFFF270) ;- (PIOA) Select A Register
AT91C_PIOA_OWSR           EQU (0xFFFFF2A8) ;- (PIOA) Output Write Status Register
AT91C_PIOA_IMR            EQU (0xFFFFF248) ;- (PIOA) Interrupt Mask Register
AT91C_PIOA_PPUSR          EQU (0xFFFFF268) ;- (PIOA) Pull-up Status Register
AT91C_PIOA_MDER           EQU (0xFFFFF250) ;- (PIOA) Multi-driver Enable Register
AT91C_PIOA_IFDR           EQU (0xFFFFF224) ;- (PIOA) Input Filter Disable Register
AT91C_PIOA_SODR           EQU (0xFFFFF230) ;- (PIOA) Set Output Data Register
AT91C_PIOA_OWER           EQU (0xFFFFF2A0) ;- (PIOA) Output Write Enable Register
AT91C_PIOA_IDR            EQU (0xFFFFF244) ;- (PIOA) Interrupt Disable Register
AT91C_PIOA_IFER           EQU (0xFFFFF220) ;- (PIOA) Input Filter Enable Register
AT91C_PIOA_IER            EQU (0xFFFFF240) ;- (PIOA) Interrupt Enable Register
AT91C_PIOA_ODR            EQU (0xFFFFF214) ;- (PIOA) Output Disable Registerr
AT91C_PIOA_MDSR           EQU (0xFFFFF258) ;- (PIOA) Multi-driver Status Register
AT91C_PIOA_DELAY4         EQU (0xFFFFF2CC) ;- (PIOA) PIO Delay Control Register
AT91C_PIOA_PSR            EQU (0xFFFFF208) ;- (PIOA) PIO Status Register
// - ========== Register definition for PIOB peripheral ========== 
AT91C_PIOB_ODR            EQU (0xFFFFF414) ;- (PIOB) Output Disable Registerr
AT91C_PIOB_DELAY4         EQU (0xFFFFF4CC) ;- (PIOB) PIO Delay Control Register
AT91C_PIOB_SODR           EQU (0xFFFFF430) ;- (PIOB) Set Output Data Register
AT91C_PIOB_ISR            EQU (0xFFFFF44C) ;- (PIOB) Interrupt Status Register
AT91C_PIOB_ABSR           EQU (0xFFFFF478) ;- (PIOB) AB Select Status Register
AT91C_PIOB_IMR            EQU (0xFFFFF448) ;- (PIOB) Interrupt Mask Register
AT91C_PIOB_MDSR           EQU (0xFFFFF458) ;- (PIOB) Multi-driver Status Register
AT91C_PIOB_PPUSR          EQU (0xFFFFF468) ;- (PIOB) Pull-up Status Register
AT91C_PIOB_PDSR           EQU (0xFFFFF43C) ;- (PIOB) Pin Data Status Register
AT91C_PIOB_DELAY3         EQU (0xFFFFF4C8) ;- (PIOB) PIO Delay Control Register
AT91C_PIOB_MDDR           EQU (0xFFFFF454) ;- (PIOB) Multi-driver Disable Register
AT91C_PIOB_CODR           EQU (0xFFFFF434) ;- (PIOB) Clear Output Data Register
AT91C_PIOB_MDER           EQU (0xFFFFF450) ;- (PIOB) Multi-driver Enable Register
AT91C_PIOB_PDR            EQU (0xFFFFF404) ;- (PIOB) PIO Disable Register
AT91C_PIOB_IFSR           EQU (0xFFFFF428) ;- (PIOB) Input Filter Status Register
AT91C_PIOB_PSR            EQU (0xFFFFF408) ;- (PIOB) PIO Status Register
AT91C_PIOB_SLEWRATE1      EQU (0xFFFFF4B0) ;- (PIOB) PIO Slewrate Control Register
AT91C_PIOB_IER            EQU (0xFFFFF440) ;- (PIOB) Interrupt Enable Register
AT91C_PIOB_PPUDR          EQU (0xFFFFF460) ;- (PIOB) Pull-up Disable Register
AT91C_PIOB_PER            EQU (0xFFFFF400) ;- (PIOB) PIO Enable Register
AT91C_PIOB_IFDR           EQU (0xFFFFF424) ;- (PIOB) Input Filter Disable Register
AT91C_PIOB_IDR            EQU (0xFFFFF444) ;- (PIOB) Interrupt Disable Register
AT91C_PIOB_OWDR           EQU (0xFFFFF4A4) ;- (PIOB) Output Write Disable Register
AT91C_PIOB_ODSR           EQU (0xFFFFF438) ;- (PIOB) Output Data Status Register
AT91C_PIOB_DELAY2         EQU (0xFFFFF4C4) ;- (PIOB) PIO Delay Control Register
AT91C_PIOB_OWSR           EQU (0xFFFFF4A8) ;- (PIOB) Output Write Status Register
AT91C_PIOB_BSR            EQU (0xFFFFF474) ;- (PIOB) Select B Register
AT91C_PIOB_IFER           EQU (0xFFFFF420) ;- (PIOB) Input Filter Enable Register
AT91C_PIOB_OWER           EQU (0xFFFFF4A0) ;- (PIOB) Output Write Enable Register
AT91C_PIOB_PPUER          EQU (0xFFFFF464) ;- (PIOB) Pull-up Enable Register
AT91C_PIOB_OSR            EQU (0xFFFFF418) ;- (PIOB) Output Status Register
AT91C_PIOB_ASR            EQU (0xFFFFF470) ;- (PIOB) Select A Register
AT91C_PIOB_OER            EQU (0xFFFFF410) ;- (PIOB) Output Enable Register
AT91C_PIOB_VERSION        EQU (0xFFFFF4FC) ;- (PIOB) PIO Version Register
AT91C_PIOB_DELAY1         EQU (0xFFFFF4C0) ;- (PIOB) PIO Delay Control Register
// - ========== Register definition for PIOC peripheral ========== 
AT91C_PIOC_OWDR           EQU (0xFFFFF6A4) ;- (PIOC) Output Write Disable Register
AT91C_PIOC_IMR            EQU (0xFFFFF648) ;- (PIOC) Interrupt Mask Register
AT91C_PIOC_ASR            EQU (0xFFFFF670) ;- (PIOC) Select A Register
AT91C_PIOC_PPUDR          EQU (0xFFFFF660) ;- (PIOC) Pull-up Disable Register
AT91C_PIOC_CODR           EQU (0xFFFFF634) ;- (PIOC) Clear Output Data Register
AT91C_PIOC_OWER           EQU (0xFFFFF6A0) ;- (PIOC) Output Write Enable Register
AT91C_PIOC_ABSR           EQU (0xFFFFF678) ;- (PIOC) AB Select Status Register
AT91C_PIOC_IFDR           EQU (0xFFFFF624) ;- (PIOC) Input Filter Disable Register
AT91C_PIOC_VERSION        EQU (0xFFFFF6FC) ;- (PIOC) PIO Version Register
AT91C_PIOC_ODR            EQU (0xFFFFF614) ;- (PIOC) Output Disable Registerr
AT91C_PIOC_PPUER          EQU (0xFFFFF664) ;- (PIOC) Pull-up Enable Register
AT91C_PIOC_SODR           EQU (0xFFFFF630) ;- (PIOC) Set Output Data Register
AT91C_PIOC_ISR            EQU (0xFFFFF64C) ;- (PIOC) Interrupt Status Register
AT91C_PIOC_OSR            EQU (0xFFFFF618) ;- (PIOC) Output Status Register
AT91C_PIOC_MDSR           EQU (0xFFFFF658) ;- (PIOC) Multi-driver Status Register
AT91C_PIOC_IFER           EQU (0xFFFFF620) ;- (PIOC) Input Filter Enable Register
AT91C_PIOC_DELAY2         EQU (0xFFFFF6C4) ;- (PIOC) PIO Delay Control Register
AT91C_PIOC_MDER           EQU (0xFFFFF650) ;- (PIOC) Multi-driver Enable Register
AT91C_PIOC_PPUSR          EQU (0xFFFFF668) ;- (PIOC) Pull-up Status Register
AT91C_PIOC_PSR            EQU (0xFFFFF608) ;- (PIOC) PIO Status Register
AT91C_PIOC_DELAY4         EQU (0xFFFFF6CC) ;- (PIOC) PIO Delay Control Register
AT91C_PIOC_DELAY3         EQU (0xFFFFF6C8) ;- (PIOC) PIO Delay Control Register
AT91C_PIOC_IER            EQU (0xFFFFF640) ;- (PIOC) Interrupt Enable Register
AT91C_PIOC_SLEWRATE1      EQU (0xFFFFF6B0) ;- (PIOC) PIO Slewrate Control Register
AT91C_PIOC_IDR            EQU (0xFFFFF644) ;- (PIOC) Interrupt Disable Register
AT91C_PIOC_PDSR           EQU (0xFFFFF63C) ;- (PIOC) Pin Data Status Register
AT91C_PIOC_DELAY1         EQU (0xFFFFF6C0) ;- (PIOC) PIO Delay Control Register
AT91C_PIOC_PDR            EQU (0xFFFFF604) ;- (PIOC) PIO Disable Register
AT91C_PIOC_OWSR           EQU (0xFFFFF6A8) ;- (PIOC) Output Write Status Register
AT91C_PIOC_IFSR           EQU (0xFFFFF628) ;- (PIOC) Input Filter Status Register
AT91C_PIOC_ODSR           EQU (0xFFFFF638) ;- (PIOC) Output Data Status Register
AT91C_PIOC_OER            EQU (0xFFFFF610) ;- (PIOC) Output Enable Register
AT91C_PIOC_MDDR           EQU (0xFFFFF654) ;- (PIOC) Multi-driver Disable Register
AT91C_PIOC_BSR            EQU (0xFFFFF674) ;- (PIOC) Select B Register
AT91C_PIOC_PER            EQU (0xFFFFF600) ;- (PIOC) PIO Enable Register
// - ========== Register definition for PIOD peripheral ========== 
AT91C_PIOD_DELAY1         EQU (0xFFFFF8C0) ;- (PIOD) PIO Delay Control Register
AT91C_PIOD_OWDR           EQU (0xFFFFF8A4) ;- (PIOD) Output Write Disable Register
AT91C_PIOD_SODR           EQU (0xFFFFF830) ;- (PIOD) Set Output Data Register
AT91C_PIOD_PPUER          EQU (0xFFFFF864) ;- (PIOD) Pull-up Enable Register
AT91C_PIOD_CODR           EQU (0xFFFFF834) ;- (PIOD) Clear Output Data Register
AT91C_PIOD_DELAY4         EQU (0xFFFFF8CC) ;- (PIOD) PIO Delay Control Register
AT91C_PIOD_PSR            EQU (0xFFFFF808) ;- (PIOD) PIO Status Register
AT91C_PIOD_PDR            EQU (0xFFFFF804) ;- (PIOD) PIO Disable Register
AT91C_PIOD_ODR            EQU (0xFFFFF814) ;- (PIOD) Output Disable Registerr
AT91C_PIOD_PPUSR          EQU (0xFFFFF868) ;- (PIOD) Pull-up Status Register
AT91C_PIOD_IFSR           EQU (0xFFFFF828) ;- (PIOD) Input Filter Status Register
AT91C_PIOD_IMR            EQU (0xFFFFF848) ;- (PIOD) Interrupt Mask Register
AT91C_PIOD_ASR            EQU (0xFFFFF870) ;- (PIOD) Select A Register
AT91C_PIOD_DELAY2         EQU (0xFFFFF8C4) ;- (PIOD) PIO Delay Control Register
AT91C_PIOD_OWSR           EQU (0xFFFFF8A8) ;- (PIOD) Output Write Status Register
AT91C_PIOD_PER            EQU (0xFFFFF800) ;- (PIOD) PIO Enable Register
AT91C_PIOD_MDER           EQU (0xFFFFF850) ;- (PIOD) Multi-driver Enable Register
AT91C_PIOD_PDSR           EQU (0xFFFFF83C) ;- (PIOD) Pin Data Status Register
AT91C_PIOD_MDSR           EQU (0xFFFFF858) ;- (PIOD) Multi-driver Status Register
AT91C_PIOD_OWER           EQU (0xFFFFF8A0) ;- (PIOD) Output Write Enable Register
AT91C_PIOD_BSR            EQU (0xFFFFF874) ;- (PIOD) Select B Register
AT91C_PIOD_IFDR           EQU (0xFFFFF824) ;- (PIOD) Input Filter Disable Register
AT91C_PIOD_DELAY3         EQU (0xFFFFF8C8) ;- (PIOD) PIO Delay Control Register
AT91C_PIOD_ABSR           EQU (0xFFFFF878) ;- (PIOD) AB Select Status Register
AT91C_PIOD_OER            EQU (0xFFFFF810) ;- (PIOD) Output Enable Register
AT91C_PIOD_MDDR           EQU (0xFFFFF854) ;- (PIOD) Multi-driver Disable Register
AT91C_PIOD_IDR            EQU (0xFFFFF844) ;- (PIOD) Interrupt Disable Register
AT91C_PIOD_IER            EQU (0xFFFFF840) ;- (PIOD) Interrupt Enable Register
AT91C_PIOD_PPUDR          EQU (0xFFFFF860) ;- (PIOD) Pull-up Disable Register
AT91C_PIOD_VERSION        EQU (0xFFFFF8FC) ;- (PIOD) PIO Version Register
AT91C_PIOD_ISR            EQU (0xFFFFF84C) ;- (PIOD) Interrupt Status Register
AT91C_PIOD_ODSR           EQU (0xFFFFF838) ;- (PIOD) Output Data Status Register
AT91C_PIOD_OSR            EQU (0xFFFFF818) ;- (PIOD) Output Status Register
AT91C_PIOD_IFER           EQU (0xFFFFF820) ;- (PIOD) Input Filter Enable Register
AT91C_PIOD_SLEWRATE1      EQU (0xFFFFF8B0) ;- (PIOD) PIO Slewrate Control Register
// - ========== Register definition for PIOE peripheral ========== 
AT91C_PIOE_ODSR           EQU (0xFFFFFA38) ;- (PIOE) Output Data Status Register
AT91C_PIOE_ABSR           EQU (0xFFFFFA78) ;- (PIOE) AB Select Status Register
AT91C_PIOE_PSR            EQU (0xFFFFFA08) ;- (PIOE) PIO Status Register
AT91C_PIOE_PPUDR          EQU (0xFFFFFA60) ;- (PIOE) Pull-up Disable Register
AT91C_PIOE_OER            EQU (0xFFFFFA10) ;- (PIOE) Output Enable Register
AT91C_PIOE_IFSR           EQU (0xFFFFFA28) ;- (PIOE) Input Filter Status Register
AT91C_PIOE_IFER           EQU (0xFFFFFA20) ;- (PIOE) Input Filter Enable Register
AT91C_PIOE_DELAY3         EQU (0xFFFFFAC8) ;- (PIOE) PIO Delay Control Register
AT91C_PIOE_ODR            EQU (0xFFFFFA14) ;- (PIOE) Output Disable Registerr
AT91C_PIOE_IDR            EQU (0xFFFFFA44) ;- (PIOE) Interrupt Disable Register
AT91C_PIOE_DELAY1         EQU (0xFFFFFAC0) ;- (PIOE) PIO Delay Control Register
AT91C_PIOE_OSR            EQU (0xFFFFFA18) ;- (PIOE) Output Status Register
AT91C_PIOE_CODR           EQU (0xFFFFFA34) ;- (PIOE) Clear Output Data Register
AT91C_PIOE_VERSION        EQU (0xFFFFFAFC) ;- (PIOE) PIO Version Register
AT91C_PIOE_MDSR           EQU (0xFFFFFA58) ;- (PIOE) Multi-driver Status Register
AT91C_PIOE_PDR            EQU (0xFFFFFA04) ;- (PIOE) PIO Disable Register
AT91C_PIOE_IER            EQU (0xFFFFFA40) ;- (PIOE) Interrupt Enable Register
AT91C_PIOE_OWSR           EQU (0xFFFFFAA8) ;- (PIOE) Output Write Status Register
AT91C_PIOE_BSR            EQU (0xFFFFFA74) ;- (PIOE) Select B Register
AT91C_PIOE_SLEWRATE1      EQU (0xFFFFFAB0) ;- (PIOE) PIO Slewrate Control Register
AT91C_PIOE_DELAY4         EQU (0xFFFFFACC) ;- (PIOE) PIO Delay Control Register
AT91C_PIOE_PER            EQU (0xFFFFFA00) ;- (PIOE) PIO Enable Register
AT91C_PIOE_OWDR           EQU (0xFFFFFAA4) ;- (PIOE) Output Write Disable Register
AT91C_PIOE_IFDR           EQU (0xFFFFFA24) ;- (PIOE) Input Filter Disable Register
AT91C_PIOE_PPUSR          EQU (0xFFFFFA68) ;- (PIOE) Pull-up Status Register
AT91C_PIOE_PDSR           EQU (0xFFFFFA3C) ;- (PIOE) Pin Data Status Register
AT91C_PIOE_PPUER          EQU (0xFFFFFA64) ;- (PIOE) Pull-up Enable Register
AT91C_PIOE_MDDR           EQU (0xFFFFFA54) ;- (PIOE) Multi-driver Disable Register
AT91C_PIOE_ISR            EQU (0xFFFFFA4C) ;- (PIOE) Interrupt Status Register
AT91C_PIOE_DELAY2         EQU (0xFFFFFAC4) ;- (PIOE) PIO Delay Control Register
AT91C_PIOE_SODR           EQU (0xFFFFFA30) ;- (PIOE) Set Output Data Register
AT91C_PIOE_ASR            EQU (0xFFFFFA70) ;- (PIOE) Select A Register
AT91C_PIOE_IMR            EQU (0xFFFFFA48) ;- (PIOE) Interrupt Mask Register
AT91C_PIOE_OWER           EQU (0xFFFFFAA0) ;- (PIOE) Output Write Enable Register
AT91C_PIOE_MDER           EQU (0xFFFFFA50) ;- (PIOE) Multi-driver Enable Register
// - ========== Register definition for PMC peripheral ========== 
AT91C_PMC_ADDRSIZE        EQU (0xFFFFFCEC) ;- (PMC) 
AT91C_PMC_PCER            EQU (0xFFFFFC10) ;- (PMC) Peripheral Clock Enable Register
AT91C_PMC_PCKR            EQU (0xFFFFFC40) ;- (PMC) Programmable Clock 0 Register
AT91C_PMC_MCKR            EQU (0xFFFFFC30) ;- (PMC) Master Clock Register
AT91C_PMC_PLLAR           EQU (0xFFFFFC28) ;- (PMC) PLL A Register
AT91C_PMC_NAME2           EQU (0xFFFFFCF4) ;- (PMC) 
AT91C_PMC_PCDR            EQU (0xFFFFFC14) ;- (PMC) Peripheral Clock Disable Register
AT91C_PMC_SCSR            EQU (0xFFFFFC08) ;- (PMC) System Clock Status Register
AT91C_PMC_MCFR            EQU (0xFFFFFC24) ;- (PMC) Main Clock  Frequency Register
AT91C_PMC_FEATURES        EQU (0xFFFFFCF8) ;- (PMC) 
AT91C_PMC_PLLICPR         EQU (0xFFFFFC80) ;- (PMC) PLL Charge Pump Current Register
AT91C_PMC_IMR             EQU (0xFFFFFC6C) ;- (PMC) Interrupt Mask Register
AT91C_PMC_IER             EQU (0xFFFFFC60) ;- (PMC) Interrupt Enable Register
AT91C_PMC_UCKR            EQU (0xFFFFFC1C) ;- (PMC) UTMI Clock Configuration Register
AT91C_PMC_USB             EQU (0xFFFFFC38) ;- (PMC) USB clock register
AT91C_PMC_MOR             EQU (0xFFFFFC20) ;- (PMC) Main Oscillator Register
AT91C_PMC_IDR             EQU (0xFFFFFC64) ;- (PMC) Interrupt Disable Register
AT91C_PMC_NAME1           EQU (0xFFFFFCF0) ;- (PMC) 
AT91C_PMC_SCDR            EQU (0xFFFFFC04) ;- (PMC) System Clock Disable Register
AT91C_PMC_PCSR            EQU (0xFFFFFC18) ;- (PMC) Peripheral Clock Status Register
AT91C_PMC_SCER            EQU (0xFFFFFC00) ;- (PMC) System Clock Enable Register
AT91C_PMC_VERSION         EQU (0xFFFFFCFC) ;- (PMC) 
AT91C_PMC_SR              EQU (0xFFFFFC68) ;- (PMC) Status Register
// - ========== Register definition for CKGR peripheral ========== 
AT91C_CKGR_MOR            EQU (0xFFFFFC20) ;- (CKGR) Main Oscillator Register
AT91C_CKGR_MCFR           EQU (0xFFFFFC24) ;- (CKGR) Main Clock  Frequency Register
AT91C_CKGR_PLLAR          EQU (0xFFFFFC28) ;- (CKGR) PLL A Register
AT91C_CKGR_UCKR           EQU (0xFFFFFC1C) ;- (CKGR) UTMI Clock Configuration Register
// - ========== Register definition for RSTC peripheral ========== 
AT91C_RSTC_RCR            EQU (0xFFFFFD00) ;- (RSTC) Reset Control Register
AT91C_RSTC_VER            EQU (0xFFFFFDFC) ;- (RSTC) Version Register
AT91C_RSTC_RMR            EQU (0xFFFFFD08) ;- (RSTC) Reset Mode Register
AT91C_RSTC_RSR            EQU (0xFFFFFD04) ;- (RSTC) Reset Status Register
// - ========== Register definition for SHDWC peripheral ========== 
AT91C_SHDWC_SHSR          EQU (0xFFFFFD18) ;- (SHDWC) Shut Down Status Register
AT91C_SHDWC_SHMR          EQU (0xFFFFFD14) ;- (SHDWC) Shut Down Mode Register
AT91C_SHDWC_SHCR          EQU (0xFFFFFD10) ;- (SHDWC) Shut Down Control Register
// - ========== Register definition for RTTC peripheral ========== 
AT91C_RTTC_RTSR           EQU (0xFFFFFD2C) ;- (RTTC) Real-time Status Register
AT91C_RTTC_RTMR           EQU (0xFFFFFD20) ;- (RTTC) Real-time Mode Register
AT91C_RTTC_RTVR           EQU (0xFFFFFD28) ;- (RTTC) Real-time Value Register
AT91C_RTTC_RTAR           EQU (0xFFFFFD24) ;- (RTTC) Real-time Alarm Register
// - ========== Register definition for PITC peripheral ========== 
AT91C_PITC_PIVR           EQU (0xFFFFFD38) ;- (PITC) Period Interval Value Register
AT91C_PITC_PISR           EQU (0xFFFFFD34) ;- (PITC) Period Interval Status Register
AT91C_PITC_PIIR           EQU (0xFFFFFD3C) ;- (PITC) Period Interval Image Register
AT91C_PITC_PIMR           EQU (0xFFFFFD30) ;- (PITC) Period Interval Mode Register
// - ========== Register definition for WDTC peripheral ========== 
AT91C_WDTC_WDCR           EQU (0xFFFFFD40) ;- (WDTC) Watchdog Control Register
AT91C_WDTC_WDSR           EQU (0xFFFFFD48) ;- (WDTC) Watchdog Status Register
AT91C_WDTC_WDMR           EQU (0xFFFFFD44) ;- (WDTC) Watchdog Mode Register
// - ========== Register definition for RTC peripheral ========== 
AT91C_RTC_IDR             EQU (0xFFFFFDD4) ;- (RTC) Interrupt Disable Register
AT91C_RTC_SR              EQU (0xFFFFFDC8) ;- (RTC) Status Register
AT91C_RTC_MR              EQU (0xFFFFFDB4) ;- (RTC) Mode Register
AT91C_RTC_IER             EQU (0xFFFFFDD0) ;- (RTC) Interrupt Enable Register
AT91C_RTC_TIMALR          EQU (0xFFFFFDC0) ;- (RTC) Time Alarm Register
AT91C_RTC_CALALR          EQU (0xFFFFFDC4) ;- (RTC) Calendar Alarm Register
AT91C_RTC_CR              EQU (0xFFFFFDB0) ;- (RTC) Control Register
AT91C_RTC_TIMR            EQU (0xFFFFFDB8) ;- (RTC) Time Register
AT91C_RTC_CALR            EQU (0xFFFFFDBC) ;- (RTC) Calendar Register
AT91C_RTC_VER             EQU (0xFFFFFDDC) ;- (RTC) Valid Entry Register
AT91C_RTC_SCCR            EQU (0xFFFFFDCC) ;- (RTC) Status Clear Command Register
AT91C_RTC_IMR             EQU (0xFFFFFDD8) ;- (RTC) Interrupt Mask Register
// - ========== Register definition for TC0 peripheral ========== 
AT91C_TC0_IER             EQU (0xFFF7C024) ;- (TC0) Interrupt Enable Register
AT91C_TC0_IMR             EQU (0xFFF7C02C) ;- (TC0) Interrupt Mask Register
AT91C_TC0_CCR             EQU (0xFFF7C000) ;- (TC0) Channel Control Register
AT91C_TC0_RB              EQU (0xFFF7C018) ;- (TC0) Register B
AT91C_TC0_CV              EQU (0xFFF7C010) ;- (TC0) Counter Value
AT91C_TC0_SR              EQU (0xFFF7C020) ;- (TC0) Status Register
AT91C_TC0_CMR             EQU (0xFFF7C004) ;- (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC0_RA              EQU (0xFFF7C014) ;- (TC0) Register A
AT91C_TC0_RC              EQU (0xFFF7C01C) ;- (TC0) Register C
AT91C_TC0_IDR             EQU (0xFFF7C028) ;- (TC0) Interrupt Disable Register
// - ========== Register definition for TC1 peripheral ========== 
AT91C_TC1_IER             EQU (0xFFF7C064) ;- (TC1) Interrupt Enable Register
AT91C_TC1_SR              EQU (0xFFF7C060) ;- (TC1) Status Register
AT91C_TC1_RC              EQU (0xFFF7C05C) ;- (TC1) Register C
AT91C_TC1_CV              EQU (0xFFF7C050) ;- (TC1) Counter Value
AT91C_TC1_RA              EQU (0xFFF7C054) ;- (TC1) Register A
AT91C_TC1_CMR             EQU (0xFFF7C044) ;- (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC1_IDR             EQU (0xFFF7C068) ;- (TC1) Interrupt Disable Register
AT91C_TC1_RB              EQU (0xFFF7C058) ;- (TC1) Register B
AT91C_TC1_IMR             EQU (0xFFF7C06C) ;- (TC1) Interrupt Mask Register
AT91C_TC1_CCR             EQU (0xFFF7C040) ;- (TC1) Channel Control Register
// - ========== Register definition for TC2 peripheral ========== 
AT91C_TC2_SR              EQU (0xFFF7C0A0) ;- (TC2) Status Register
AT91C_TC2_IMR             EQU (0xFFF7C0AC) ;- (TC2) Interrupt Mask Register
AT91C_TC2_IER             EQU (0xFFF7C0A4) ;- (TC2) Interrupt Enable Register
AT91C_TC2_CV              EQU (0xFFF7C090) ;- (TC2) Counter Value
AT91C_TC2_RB              EQU (0xFFF7C098) ;- (TC2) Register B
AT91C_TC2_CCR             EQU (0xFFF7C080) ;- (TC2) Channel Control Register
AT91C_TC2_CMR             EQU (0xFFF7C084) ;- (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC2_RA              EQU (0xFFF7C094) ;- (TC2) Register A
AT91C_TC2_IDR             EQU (0xFFF7C0A8) ;- (TC2) Interrupt Disable Register
AT91C_TC2_RC              EQU (0xFFF7C09C) ;- (TC2) Register C
// - ========== Register definition for TC3 peripheral ========== 
AT91C_TC3_SR              EQU (0xFFFD4020) ;- (TC3) Status Register
AT91C_TC3_RC              EQU (0xFFFD401C) ;- (TC3) Register C
AT91C_TC3_IER             EQU (0xFFFD4024) ;- (TC3) Interrupt Enable Register
AT91C_TC3_CV              EQU (0xFFFD4010) ;- (TC3) Counter Value
AT91C_TC3_IDR             EQU (0xFFFD4028) ;- (TC3) Interrupt Disable Register
AT91C_TC3_IMR             EQU (0xFFFD402C) ;- (TC3) Interrupt Mask Register
AT91C_TC3_CMR             EQU (0xFFFD4004) ;- (TC3) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC3_RB              EQU (0xFFFD4018) ;- (TC3) Register B
AT91C_TC3_CCR             EQU (0xFFFD4000) ;- (TC3) Channel Control Register
AT91C_TC3_RA              EQU (0xFFFD4014) ;- (TC3) Register A
// - ========== Register definition for TC4 peripheral ========== 
AT91C_TC4_CV              EQU (0xFFFD4050) ;- (TC4) Counter Value
AT91C_TC4_RA              EQU (0xFFFD4054) ;- (TC4) Register A
AT91C_TC4_IDR             EQU (0xFFFD4068) ;- (TC4) Interrupt Disable Register
AT91C_TC4_SR              EQU (0xFFFD4060) ;- (TC4) Status Register
AT91C_TC4_CMR             EQU (0xFFFD4044) ;- (TC4) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC4_CCR             EQU (0xFFFD4040) ;- (TC4) Channel Control Register
AT91C_TC4_RB              EQU (0xFFFD4058) ;- (TC4) Register B
AT91C_TC4_RC              EQU (0xFFFD405C) ;- (TC4) Register C
AT91C_TC4_IER             EQU (0xFFFD4064) ;- (TC4) Interrupt Enable Register
AT91C_TC4_IMR             EQU (0xFFFD406C) ;- (TC4) Interrupt Mask Register
// - ========== Register definition for TC5 peripheral ========== 
AT91C_TC5_IER             EQU (0xFFFD40A4) ;- (TC5) Interrupt Enable Register
AT91C_TC5_IDR             EQU (0xFFFD40A8) ;- (TC5) Interrupt Disable Register
AT91C_TC5_RA              EQU (0xFFFD4094) ;- (TC5) Register A
AT91C_TC5_RB              EQU (0xFFFD4098) ;- (TC5) Register B
AT91C_TC5_CCR             EQU (0xFFFD4080) ;- (TC5) Channel Control Register
AT91C_TC5_SR              EQU (0xFFFD40A0) ;- (TC5) Status Register
AT91C_TC5_CV              EQU (0xFFFD4090) ;- (TC5) Counter Value
AT91C_TC5_RC              EQU (0xFFFD409C) ;- (TC5) Register C
AT91C_TC5_IMR             EQU (0xFFFD40AC) ;- (TC5) Interrupt Mask Register
AT91C_TC5_CMR             EQU (0xFFFD4084) ;- (TC5) Channel Mode Register (Capture Mode / Waveform Mode)
// - ========== Register definition for TCB0 peripheral ========== 
AT91C_TCB0_IPNAME1        EQU (0xFFF7C0F0) ;- (TCB0) TC IPNAME1 REGISTER 
AT91C_TCB0_IPNAME2        EQU (0xFFF7C0F4) ;- (TCB0) TC IPNAME2 REGISTER 
AT91C_TCB0_FEATURES       EQU (0xFFF7C0F8) ;- (TCB0) TC FEATURES REGISTER 
AT91C_TCB0_BCR            EQU (0xFFF7C0C0) ;- (TCB0) TC Block Control Register
AT91C_TCB0_VER            EQU (0xFFF7C0FC) ;- (TCB0)  Version Register
AT91C_TCB0_BMR            EQU (0xFFF7C0C4) ;- (TCB0) TC Block Mode Register
AT91C_TCB0_ADDRSIZE       EQU (0xFFF7C0EC) ;- (TCB0) TC ADDRSIZE REGISTER 
// - ========== Register definition for TCB1 peripheral ========== 
AT91C_TCB1_VER            EQU (0xFFFD40FC) ;- (TCB1)  Version Register
AT91C_TCB1_IPNAME2        EQU (0xFFFD40F4) ;- (TCB1) TC IPNAME2 REGISTER 
AT91C_TCB1_ADDRSIZE       EQU (0xFFFD40EC) ;- (TCB1) TC ADDRSIZE REGISTER 
AT91C_TCB1_BMR            EQU (0xFFFD40C4) ;- (TCB1) TC Block Mode Register
AT91C_TCB1_FEATURES       EQU (0xFFFD40F8) ;- (TCB1) TC FEATURES REGISTER 
AT91C_TCB1_IPNAME1        EQU (0xFFFD40F0) ;- (TCB1) TC IPNAME1 REGISTER 
AT91C_TCB1_BCR            EQU (0xFFFD40C0) ;- (TCB1) TC Block Control Register
// - ========== Register definition for MCI0 peripheral ========== 
AT91C_MCI0_IMR            EQU (0xFFF8004C) ;- (MCI0) MCI Interrupt Mask Register
AT91C_MCI0_MR             EQU (0xFFF80004) ;- (MCI0) MCI Mode Register
AT91C_MCI0_ADDRSIZE       EQU (0xFFF800EC) ;- (MCI0) MCI ADDRSIZE REGISTER 
AT91C_MCI0_CR             EQU (0xFFF80000) ;- (MCI0) MCI Control Register
AT91C_MCI0_IPNAME2        EQU (0xFFF800F4) ;- (MCI0) MCI IPNAME2 REGISTER 
AT91C_MCI0_IER            EQU (0xFFF80044) ;- (MCI0) MCI Interrupt Enable Register
AT91C_MCI0_FIFO           EQU (0xFFF80200) ;- (MCI0) MCI FIFO Aperture Register
AT91C_MCI0_DTOR           EQU (0xFFF80008) ;- (MCI0) MCI Data Timeout Register
AT91C_MCI0_SDCR           EQU (0xFFF8000C) ;- (MCI0) MCI SD/SDIO Card Register
AT91C_MCI0_BLKR           EQU (0xFFF80018) ;- (MCI0) MCI Block Register
AT91C_MCI0_VER            EQU (0xFFF800FC) ;- (MCI0) MCI VERSION REGISTER 
AT91C_MCI0_WPSR           EQU (0xFFF800E8) ;- (MCI0) MCI Write Protection Status Register
AT91C_MCI0_CMDR           EQU (0xFFF80014) ;- (MCI0) MCI Command Register
AT91C_MCI0_CSTOR          EQU (0xFFF8001C) ;- (MCI0) MCI Completion Signal Timeout Register
AT91C_MCI0_IPNAME1        EQU (0xFFF800F0) ;- (MCI0) MCI IPNAME1 REGISTER 
AT91C_MCI0_DMA            EQU (0xFFF80050) ;- (MCI0) MCI DMA Configuration Register
AT91C_MCI0_FEATURES       EQU (0xFFF800F8) ;- (MCI0) MCI FEATURES REGISTER 
AT91C_MCI0_RDR            EQU (0xFFF80030) ;- (MCI0) MCI Receive Data Register
AT91C_MCI0_SR             EQU (0xFFF80040) ;- (MCI0) MCI Status Register
AT91C_MCI0_TDR            EQU (0xFFF80034) ;- (MCI0) MCI Transmit Data Register
AT91C_MCI0_CFG            EQU (0xFFF80054) ;- (MCI0) MCI Configuration Register
AT91C_MCI0_ARGR           EQU (0xFFF80010) ;- (MCI0) MCI Argument Register
AT91C_MCI0_RSPR           EQU (0xFFF80020) ;- (MCI0) MCI Response Register
AT91C_MCI0_WPCR           EQU (0xFFF800E4) ;- (MCI0) MCI Write Protection Control Register
AT91C_MCI0_IDR            EQU (0xFFF80048) ;- (MCI0) MCI Interrupt Disable Register
// - ========== Register definition for MCI1 peripheral ========== 
AT91C_MCI1_IPNAME1        EQU (0xFFFD00F0) ;- (MCI1) MCI IPNAME1 REGISTER 
AT91C_MCI1_IDR            EQU (0xFFFD0048) ;- (MCI1) MCI Interrupt Disable Register
AT91C_MCI1_IPNAME2        EQU (0xFFFD00F4) ;- (MCI1) MCI IPNAME2 REGISTER 
AT91C_MCI1_MR             EQU (0xFFFD0004) ;- (MCI1) MCI Mode Register
AT91C_MCI1_SR             EQU (0xFFFD0040) ;- (MCI1) MCI Status Register
AT91C_MCI1_DTOR           EQU (0xFFFD0008) ;- (MCI1) MCI Data Timeout Register
AT91C_MCI1_WPCR           EQU (0xFFFD00E4) ;- (MCI1) MCI Write Protection Control Register
AT91C_MCI1_ADDRSIZE       EQU (0xFFFD00EC) ;- (MCI1) MCI ADDRSIZE REGISTER 
AT91C_MCI1_ARGR           EQU (0xFFFD0010) ;- (MCI1) MCI Argument Register
AT91C_MCI1_FIFO           EQU (0xFFFD0200) ;- (MCI1) MCI FIFO Aperture Register
AT91C_MCI1_IER            EQU (0xFFFD0044) ;- (MCI1) MCI Interrupt Enable Register
AT91C_MCI1_SDCR           EQU (0xFFFD000C) ;- (MCI1) MCI SD/SDIO Card Register
AT91C_MCI1_FEATURES       EQU (0xFFFD00F8) ;- (MCI1) MCI FEATURES REGISTER 
AT91C_MCI1_CR             EQU (0xFFFD0000) ;- (MCI1) MCI Control Register
AT91C_MCI1_CSTOR          EQU (0xFFFD001C) ;- (MCI1) MCI Completion Signal Timeout Register
AT91C_MCI1_RSPR           EQU (0xFFFD0020) ;- (MCI1) MCI Response Register
AT91C_MCI1_IMR            EQU (0xFFFD004C) ;- (MCI1) MCI Interrupt Mask Register
AT91C_MCI1_DMA            EQU (0xFFFD0050) ;- (MCI1) MCI DMA Configuration Register
AT91C_MCI1_BLKR           EQU (0xFFFD0018) ;- (MCI1) MCI Block Register
AT91C_MCI1_RDR            EQU (0xFFFD0030) ;- (MCI1) MCI Receive Data Register
AT91C_MCI1_CFG            EQU (0xFFFD0054) ;- (MCI1) MCI Configuration Register
AT91C_MCI1_WPSR           EQU (0xFFFD00E8) ;- (MCI1) MCI Write Protection Status Register
AT91C_MCI1_CMDR           EQU (0xFFFD0014) ;- (MCI1) MCI Command Register
AT91C_MCI1_TDR            EQU (0xFFFD0034) ;- (MCI1) MCI Transmit Data Register
AT91C_MCI1_VER            EQU (0xFFFD00FC) ;- (MCI1) MCI VERSION REGISTER 
// - ========== Register definition for TWI0 peripheral ========== 
AT91C_TWI0_RHR            EQU (0xFFF84030) ;- (TWI0) Receive Holding Register
AT91C_TWI0_ADDRSIZE       EQU (0xFFF840EC) ;- (TWI0) TWI ADDRSIZE REGISTER 
AT91C_TWI0_SMR            EQU (0xFFF84008) ;- (TWI0) Slave Mode Register
AT91C_TWI0_IDR            EQU (0xFFF84028) ;- (TWI0) Interrupt Disable Register
AT91C_TWI0_CWGR           EQU (0xFFF84010) ;- (TWI0) Clock Waveform Generator Register
AT91C_TWI0_IER            EQU (0xFFF84024) ;- (TWI0) Interrupt Enable Register
AT91C_TWI0_THR            EQU (0xFFF84034) ;- (TWI0) Transmit Holding Register
AT91C_TWI0_IPNAME2        EQU (0xFFF840F4) ;- (TWI0) TWI IPNAME2 REGISTER 
AT91C_TWI0_CR             EQU (0xFFF84000) ;- (TWI0) Control Register
AT91C_TWI0_MMR            EQU (0xFFF84004) ;- (TWI0) Master Mode Register
AT91C_TWI0_SR             EQU (0xFFF84020) ;- (TWI0) Status Register
AT91C_TWI0_IMR            EQU (0xFFF8402C) ;- (TWI0) Interrupt Mask Register
AT91C_TWI0_FEATURES       EQU (0xFFF840F8) ;- (TWI0) TWI FEATURES REGISTER 
AT91C_TWI0_IADR           EQU (0xFFF8400C) ;- (TWI0) Internal Address Register
AT91C_TWI0_IPNAME1        EQU (0xFFF840F0) ;- (TWI0) TWI IPNAME1 REGISTER 
AT91C_TWI0_VER            EQU (0xFFF840FC) ;- (TWI0) Version Register
// - ========== Register definition for TWI1 peripheral ========== 
AT91C_TWI1_VER            EQU (0xFFF880FC) ;- (TWI1) Version Register
AT91C_TWI1_IMR            EQU (0xFFF8802C) ;- (TWI1) Interrupt Mask Register
AT91C_TWI1_THR            EQU (0xFFF88034) ;- (TWI1) Transmit Holding Register
AT91C_TWI1_IER            EQU (0xFFF88024) ;- (TWI1) Interrupt Enable Register
AT91C_TWI1_MMR            EQU (0xFFF88004) ;- (TWI1) Master Mode Register
AT91C_TWI1_IPNAME1        EQU (0xFFF880F0) ;- (TWI1) TWI IPNAME1 REGISTER 
AT91C_TWI1_CR             EQU (0xFFF88000) ;- (TWI1) Control Register
AT91C_TWI1_SR             EQU (0xFFF88020) ;- (TWI1) Status Register
AT91C_TWI1_CWGR           EQU (0xFFF88010) ;- (TWI1) Clock Waveform Generator Register
AT91C_TWI1_IPNAME2        EQU (0xFFF880F4) ;- (TWI1) TWI IPNAME2 REGISTER 
AT91C_TWI1_IDR            EQU (0xFFF88028) ;- (TWI1) Interrupt Disable Register
AT91C_TWI1_RHR            EQU (0xFFF88030) ;- (TWI1) Receive Holding Register
AT91C_TWI1_IADR           EQU (0xFFF8800C) ;- (TWI1) Internal Address Register
AT91C_TWI1_SMR            EQU (0xFFF88008) ;- (TWI1) Slave Mode Register
AT91C_TWI1_FEATURES       EQU (0xFFF880F8) ;- (TWI1) TWI FEATURES REGISTER 
AT91C_TWI1_ADDRSIZE       EQU (0xFFF880EC) ;- (TWI1) TWI ADDRSIZE REGISTER 
// - ========== Register definition for PDC_US0 peripheral ========== 
AT91C_US0_TNPR            EQU (0xFFF8C118) ;- (PDC_US0) Transmit Next Pointer Register
AT91C_US0_PTSR            EQU (0xFFF8C124) ;- (PDC_US0) PDC Transfer Status Register
AT91C_US0_PTCR            EQU (0xFFF8C120) ;- (PDC_US0) PDC Transfer Control Register
AT91C_US0_RNCR            EQU (0xFFF8C114) ;- (PDC_US0) Receive Next Counter Register
AT91C_US0_RCR             EQU (0xFFF8C104) ;- (PDC_US0) Receive Counter Register
AT91C_US0_TNCR            EQU (0xFFF8C11C) ;- (PDC_US0) Transmit Next Counter Register
AT91C_US0_TCR             EQU (0xFFF8C10C) ;- (PDC_US0) Transmit Counter Register
AT91C_US0_RNPR            EQU (0xFFF8C110) ;- (PDC_US0) Receive Next Pointer Register
AT91C_US0_RPR             EQU (0xFFF8C100) ;- (PDC_US0) Receive Pointer Register
AT91C_US0_TPR             EQU (0xFFF8C108) ;- (PDC_US0) Transmit Pointer Register
// - ========== Register definition for US0 peripheral ========== 
AT91C_US0_MAN             EQU (0xFFF8C050) ;- (US0) Manchester Encoder Decoder Register
AT91C_US0_MR              EQU (0xFFF8C004) ;- (US0) Mode Register
AT91C_US0_RHR             EQU (0xFFF8C018) ;- (US0) Receiver Holding Register
AT91C_US0_CSR             EQU (0xFFF8C014) ;- (US0) Channel Status Register
AT91C_US0_CR              EQU (0xFFF8C000) ;- (US0) Control Register
AT91C_US0_VER             EQU (0xFFF8C0FC) ;- (US0) VERSION Register
AT91C_US0_IER             EQU (0xFFF8C008) ;- (US0) Interrupt Enable Register
AT91C_US0_BRGR            EQU (0xFFF8C020) ;- (US0) Baud Rate Generator Register
AT91C_US0_FEATURES        EQU (0xFFF8C0F8) ;- (US0) US FEATURES REGISTER 
AT91C_US0_RTOR            EQU (0xFFF8C024) ;- (US0) Receiver Time-out Register
AT91C_US0_THR             EQU (0xFFF8C01C) ;- (US0) Transmitter Holding Register
AT91C_US0_NER             EQU (0xFFF8C044) ;- (US0) Nb Errors Register
AT91C_US0_IPNAME1         EQU (0xFFF8C0F0) ;- (US0) US IPNAME1 REGISTER 
AT91C_US0_IMR             EQU (0xFFF8C010) ;- (US0) Interrupt Mask Register
AT91C_US0_ADDRSIZE        EQU (0xFFF8C0EC) ;- (US0) US ADDRSIZE REGISTER 
AT91C_US0_IDR             EQU (0xFFF8C00C) ;- (US0) Interrupt Disable Register
AT91C_US0_FIDI            EQU (0xFFF8C040) ;- (US0) FI_DI_Ratio Register
AT91C_US0_IPNAME2         EQU (0xFFF8C0F4) ;- (US0) US IPNAME2 REGISTER 
AT91C_US0_TTGR            EQU (0xFFF8C028) ;- (US0) Transmitter Time-guard Register
AT91C_US0_IF              EQU (0xFFF8C04C) ;- (US0) IRDA_FILTER Register
// - ========== Register definition for PDC_US1 peripheral ========== 
AT91C_US1_PTCR            EQU (0xFFF90120) ;- (PDC_US1) PDC Transfer Control Register
AT91C_US1_TNCR            EQU (0xFFF9011C) ;- (PDC_US1) Transmit Next Counter Register
AT91C_US1_RCR             EQU (0xFFF90104) ;- (PDC_US1) Receive Counter Register
AT91C_US1_RPR             EQU (0xFFF90100) ;- (PDC_US1) Receive Pointer Register
AT91C_US1_TPR             EQU (0xFFF90108) ;- (PDC_US1) Transmit Pointer Register
AT91C_US1_TCR             EQU (0xFFF9010C) ;- (PDC_US1) Transmit Counter Register
AT91C_US1_RNPR            EQU (0xFFF90110) ;- (PDC_US1) Receive Next Pointer Register
AT91C_US1_TNPR            EQU (0xFFF90118) ;- (PDC_US1) Transmit Next Pointer Register
AT91C_US1_RNCR            EQU (0xFFF90114) ;- (PDC_US1) Receive Next Counter Register
AT91C_US1_PTSR            EQU (0xFFF90124) ;- (PDC_US1) PDC Transfer Status Register
// - ========== Register definition for US1 peripheral ========== 
AT91C_US1_FEATURES        EQU (0xFFF900F8) ;- (US1) US FEATURES REGISTER 
AT91C_US1_ADDRSIZE        EQU (0xFFF900EC) ;- (US1) US ADDRSIZE REGISTER 
AT91C_US1_NER             EQU (0xFFF90044) ;- (US1) Nb Errors Register
AT91C_US1_RHR             EQU (0xFFF90018) ;- (US1) Receiver Holding Register
AT91C_US1_IF              EQU (0xFFF9004C) ;- (US1) IRDA_FILTER Register
AT91C_US1_CR              EQU (0xFFF90000) ;- (US1) Control Register
AT91C_US1_TTGR            EQU (0xFFF90028) ;- (US1) Transmitter Time-guard Register
AT91C_US1_MR              EQU (0xFFF90004) ;- (US1) Mode Register
AT91C_US1_FIDI            EQU (0xFFF90040) ;- (US1) FI_DI_Ratio Register
AT91C_US1_RTOR            EQU (0xFFF90024) ;- (US1) Receiver Time-out Register
AT91C_US1_IER             EQU (0xFFF90008) ;- (US1) Interrupt Enable Register
AT91C_US1_MAN             EQU (0xFFF90050) ;- (US1) Manchester Encoder Decoder Register
AT91C_US1_IPNAME2         EQU (0xFFF900F4) ;- (US1) US IPNAME2 REGISTER 
AT91C_US1_IMR             EQU (0xFFF90010) ;- (US1) Interrupt Mask Register
AT91C_US1_IDR             EQU (0xFFF9000C) ;- (US1) Interrupt Disable Register
AT91C_US1_CSR             EQU (0xFFF90014) ;- (US1) Channel Status Register
AT91C_US1_BRGR            EQU (0xFFF90020) ;- (US1) Baud Rate Generator Register
AT91C_US1_THR             EQU (0xFFF9001C) ;- (US1) Transmitter Holding Register
AT91C_US1_VER             EQU (0xFFF900FC) ;- (US1) VERSION Register
AT91C_US1_IPNAME1         EQU (0xFFF900F0) ;- (US1) US IPNAME1 REGISTER 
// - ========== Register definition for PDC_US2 peripheral ========== 
AT91C_US2_RNCR            EQU (0xFFF94114) ;- (PDC_US2) Receive Next Counter Register
AT91C_US2_PTCR            EQU (0xFFF94120) ;- (PDC_US2) PDC Transfer Control Register
AT91C_US2_TNPR            EQU (0xFFF94118) ;- (PDC_US2) Transmit Next Pointer Register
AT91C_US2_TNCR            EQU (0xFFF9411C) ;- (PDC_US2) Transmit Next Counter Register
AT91C_US2_TPR             EQU (0xFFF94108) ;- (PDC_US2) Transmit Pointer Register
AT91C_US2_RCR             EQU (0xFFF94104) ;- (PDC_US2) Receive Counter Register
AT91C_US2_PTSR            EQU (0xFFF94124) ;- (PDC_US2) PDC Transfer Status Register
AT91C_US2_TCR             EQU (0xFFF9410C) ;- (PDC_US2) Transmit Counter Register
AT91C_US2_RPR             EQU (0xFFF94100) ;- (PDC_US2) Receive Pointer Register
AT91C_US2_RNPR            EQU (0xFFF94110) ;- (PDC_US2) Receive Next Pointer Register
// - ========== Register definition for US2 peripheral ========== 
AT91C_US2_TTGR            EQU (0xFFF94028) ;- (US2) Transmitter Time-guard Register
AT91C_US2_IER             EQU (0xFFF94008) ;- (US2) Interrupt Enable Register
AT91C_US2_ADDRSIZE        EQU (0xFFF940EC) ;- (US2) US ADDRSIZE REGISTER 
AT91C_US2_NER             EQU (0xFFF94044) ;- (US2) Nb Errors Register
AT91C_US2_CR              EQU (0xFFF94000) ;- (US2) Control Register
AT91C_US2_IDR             EQU (0xFFF9400C) ;- (US2) Interrupt Disable Register
AT91C_US2_VER             EQU (0xFFF940FC) ;- (US2) VERSION Register
AT91C_US2_THR             EQU (0xFFF9401C) ;- (US2) Transmitter Holding Register
AT91C_US2_BRGR            EQU (0xFFF94020) ;- (US2) Baud Rate Generator Register
AT91C_US2_CSR             EQU (0xFFF94014) ;- (US2) Channel Status Register
AT91C_US2_IPNAME2         EQU (0xFFF940F4) ;- (US2) US IPNAME2 REGISTER 
AT91C_US2_IMR             EQU (0xFFF94010) ;- (US2) Interrupt Mask Register
AT91C_US2_RHR             EQU (0xFFF94018) ;- (US2) Receiver Holding Register
AT91C_US2_MR              EQU (0xFFF94004) ;- (US2) Mode Register
AT91C_US2_FIDI            EQU (0xFFF94040) ;- (US2) FI_DI_Ratio Register
AT91C_US2_MAN             EQU (0xFFF94050) ;- (US2) Manchester Encoder Decoder Register
AT91C_US2_IF              EQU (0xFFF9404C) ;- (US2) IRDA_FILTER Register
AT91C_US2_FEATURES        EQU (0xFFF940F8) ;- (US2) US FEATURES REGISTER 
AT91C_US2_IPNAME1         EQU (0xFFF940F0) ;- (US2) US IPNAME1 REGISTER 
AT91C_US2_RTOR            EQU (0xFFF94024) ;- (US2) Receiver Time-out Register
// - ========== Register definition for PDC_US3 peripheral ========== 
AT91C_US3_PTSR            EQU (0xFFF98124) ;- (PDC_US3) PDC Transfer Status Register
AT91C_US3_TCR             EQU (0xFFF9810C) ;- (PDC_US3) Transmit Counter Register
AT91C_US3_RNPR            EQU (0xFFF98110) ;- (PDC_US3) Receive Next Pointer Register
AT91C_US3_RNCR            EQU (0xFFF98114) ;- (PDC_US3) Receive Next Counter Register
AT91C_US3_TNPR            EQU (0xFFF98118) ;- (PDC_US3) Transmit Next Pointer Register
AT91C_US3_RPR             EQU (0xFFF98100) ;- (PDC_US3) Receive Pointer Register
AT91C_US3_TPR             EQU (0xFFF98108) ;- (PDC_US3) Transmit Pointer Register
AT91C_US3_RCR             EQU (0xFFF98104) ;- (PDC_US3) Receive Counter Register
AT91C_US3_TNCR            EQU (0xFFF9811C) ;- (PDC_US3) Transmit Next Counter Register
AT91C_US3_PTCR            EQU (0xFFF98120) ;- (PDC_US3) PDC Transfer Control Register
// - ========== Register definition for US3 peripheral ========== 
AT91C_US3_VER             EQU (0xFFF980FC) ;- (US3) VERSION Register
AT91C_US3_BRGR            EQU (0xFFF98020) ;- (US3) Baud Rate Generator Register
AT91C_US3_TTGR            EQU (0xFFF98028) ;- (US3) Transmitter Time-guard Register
AT91C_US3_RTOR            EQU (0xFFF98024) ;- (US3) Receiver Time-out Register
AT91C_US3_MAN             EQU (0xFFF98050) ;- (US3) Manchester Encoder Decoder Register
AT91C_US3_NER             EQU (0xFFF98044) ;- (US3) Nb Errors Register
AT91C_US3_CR              EQU (0xFFF98000) ;- (US3) Control Register
AT91C_US3_IDR             EQU (0xFFF9800C) ;- (US3) Interrupt Disable Register
AT91C_US3_ADDRSIZE        EQU (0xFFF980EC) ;- (US3) US ADDRSIZE REGISTER 
AT91C_US3_CSR             EQU (0xFFF98014) ;- (US3) Channel Status Register
AT91C_US3_IPNAME2         EQU (0xFFF980F4) ;- (US3) US IPNAME2 REGISTER 
AT91C_US3_IER             EQU (0xFFF98008) ;- (US3) Interrupt Enable Register
AT91C_US3_FEATURES        EQU (0xFFF980F8) ;- (US3) US FEATURES REGISTER 
AT91C_US3_MR              EQU (0xFFF98004) ;- (US3) Mode Register
AT91C_US3_IF              EQU (0xFFF9804C) ;- (US3) IRDA_FILTER Register
AT91C_US3_THR             EQU (0xFFF9801C) ;- (US3) Transmitter Holding Register
AT91C_US3_IMR             EQU (0xFFF98010) ;- (US3) Interrupt Mask Register
AT91C_US3_FIDI            EQU (0xFFF98040) ;- (US3) FI_DI_Ratio Register
AT91C_US3_RHR             EQU (0xFFF98018) ;- (US3) Receiver Holding Register
AT91C_US3_IPNAME1         EQU (0xFFF980F0) ;- (US3) US IPNAME1 REGISTER 
// - ========== Register definition for PDC_SSC0 peripheral ========== 
AT91C_SSC0_TNPR           EQU (0xFFF9C118) ;- (PDC_SSC0) Transmit Next Pointer Register
AT91C_SSC0_PTSR           EQU (0xFFF9C124) ;- (PDC_SSC0) PDC Transfer Status Register
AT91C_SSC0_TNCR           EQU (0xFFF9C11C) ;- (PDC_SSC0) Transmit Next Counter Register
AT91C_SSC0_RNCR           EQU (0xFFF9C114) ;- (PDC_SSC0) Receive Next Counter Register
AT91C_SSC0_TPR            EQU (0xFFF9C108) ;- (PDC_SSC0) Transmit Pointer Register
AT91C_SSC0_RCR            EQU (0xFFF9C104) ;- (PDC_SSC0) Receive Counter Register
AT91C_SSC0_PTCR           EQU (0xFFF9C120) ;- (PDC_SSC0) PDC Transfer Control Register
AT91C_SSC0_RNPR           EQU (0xFFF9C110) ;- (PDC_SSC0) Receive Next Pointer Register
AT91C_SSC0_TCR            EQU (0xFFF9C10C) ;- (PDC_SSC0) Transmit Counter Register
AT91C_SSC0_RPR            EQU (0xFFF9C100) ;- (PDC_SSC0) Receive Pointer Register
// - ========== Register definition for SSC0 peripheral ========== 
AT91C_SSC0_IPNAME2        EQU (0xFFF9C0F4) ;- (SSC0) SSC IPNAME2 REGISTER 
AT91C_SSC0_SR             EQU (0xFFF9C040) ;- (SSC0) Status Register
AT91C_SSC0_TSHR           EQU (0xFFF9C034) ;- (SSC0) Transmit Sync Holding Register
AT91C_SSC0_TCMR           EQU (0xFFF9C018) ;- (SSC0) Transmit Clock Mode Register
AT91C_SSC0_ADDRSIZE       EQU (0xFFF9C0EC) ;- (SSC0) SSC ADDRSIZE REGISTER 
AT91C_SSC0_IMR            EQU (0xFFF9C04C) ;- (SSC0) Interrupt Mask Register
AT91C_SSC0_IDR            EQU (0xFFF9C048) ;- (SSC0) Interrupt Disable Register
AT91C_SSC0_CR             EQU (0xFFF9C000) ;- (SSC0) Control Register
AT91C_SSC0_VER            EQU (0xFFF9C0FC) ;- (SSC0) Version Register
AT91C_SSC0_RHR            EQU (0xFFF9C020) ;- (SSC0) Receive Holding Register
AT91C_SSC0_THR            EQU (0xFFF9C024) ;- (SSC0) Transmit Holding Register
AT91C_SSC0_CMR            EQU (0xFFF9C004) ;- (SSC0) Clock Mode Register
AT91C_SSC0_FEATURES       EQU (0xFFF9C0F8) ;- (SSC0) SSC FEATURES REGISTER 
AT91C_SSC0_RCMR           EQU (0xFFF9C010) ;- (SSC0) Receive Clock ModeRegister
AT91C_SSC0_IPNAME1        EQU (0xFFF9C0F0) ;- (SSC0) SSC IPNAME1 REGISTER 
AT91C_SSC0_RSHR           EQU (0xFFF9C030) ;- (SSC0) Receive Sync Holding Register
AT91C_SSC0_IER            EQU (0xFFF9C044) ;- (SSC0) Interrupt Enable Register
AT91C_SSC0_TFMR           EQU (0xFFF9C01C) ;- (SSC0) Transmit Frame Mode Register
AT91C_SSC0_RFMR           EQU (0xFFF9C014) ;- (SSC0) Receive Frame Mode Register
// - ========== Register definition for PDC_SSC1 peripheral ========== 
AT91C_SSC1_RNPR           EQU (0xFFFA0110) ;- (PDC_SSC1) Receive Next Pointer Register
AT91C_SSC1_TCR            EQU (0xFFFA010C) ;- (PDC_SSC1) Transmit Counter Register
AT91C_SSC1_TNCR           EQU (0xFFFA011C) ;- (PDC_SSC1) Transmit Next Counter Register
AT91C_SSC1_RCR            EQU (0xFFFA0104) ;- (PDC_SSC1) Receive Counter Register
AT91C_SSC1_RNCR           EQU (0xFFFA0114) ;- (PDC_SSC1) Receive Next Counter Register
AT91C_SSC1_PTCR           EQU (0xFFFA0120) ;- (PDC_SSC1) PDC Transfer Control Register
AT91C_SSC1_TPR            EQU (0xFFFA0108) ;- (PDC_SSC1) Transmit Pointer Register
AT91C_SSC1_RPR            EQU (0xFFFA0100) ;- (PDC_SSC1) Receive Pointer Register
AT91C_SSC1_PTSR           EQU (0xFFFA0124) ;- (PDC_SSC1) PDC Transfer Status Register
AT91C_SSC1_TNPR           EQU (0xFFFA0118) ;- (PDC_SSC1) Transmit Next Pointer Register
// - ========== Register definition for SSC1 peripheral ========== 
AT91C_SSC1_SR             EQU (0xFFFA0040) ;- (SSC1) Status Register
AT91C_SSC1_TFMR           EQU (0xFFFA001C) ;- (SSC1) Transmit Frame Mode Register
AT91C_SSC1_IPNAME1        EQU (0xFFFA00F0) ;- (SSC1) SSC IPNAME1 REGISTER 
AT91C_SSC1_CMR            EQU (0xFFFA0004) ;- (SSC1) Clock Mode Register
AT91C_SSC1_THR            EQU (0xFFFA0024) ;- (SSC1) Transmit Holding Register
AT91C_SSC1_FEATURES       EQU (0xFFFA00F8) ;- (SSC1) SSC FEATURES REGISTER 
AT91C_SSC1_TSHR           EQU (0xFFFA0034) ;- (SSC1) Transmit Sync Holding Register
AT91C_SSC1_RCMR           EQU (0xFFFA0010) ;- (SSC1) Receive Clock ModeRegister
AT91C_SSC1_RHR            EQU (0xFFFA0020) ;- (SSC1) Receive Holding Register
AT91C_SSC1_VER            EQU (0xFFFA00FC) ;- (SSC1) Version Register
AT91C_SSC1_TCMR           EQU (0xFFFA0018) ;- (SSC1) Transmit Clock Mode Register
AT91C_SSC1_CR             EQU (0xFFFA0000) ;- (SSC1) Control Register
AT91C_SSC1_RSHR           EQU (0xFFFA0030) ;- (SSC1) Receive Sync Holding Register
AT91C_SSC1_IER            EQU (0xFFFA0044) ;- (SSC1) Interrupt Enable Register
AT91C_SSC1_ADDRSIZE       EQU (0xFFFA00EC) ;- (SSC1) SSC ADDRSIZE REGISTER 
AT91C_SSC1_RFMR           EQU (0xFFFA0014) ;- (SSC1) Receive Frame Mode Register
AT91C_SSC1_IMR            EQU (0xFFFA004C) ;- (SSC1) Interrupt Mask Register
AT91C_SSC1_IDR            EQU (0xFFFA0048) ;- (SSC1) Interrupt Disable Register
AT91C_SSC1_IPNAME2        EQU (0xFFFA00F4) ;- (SSC1) SSC IPNAME2 REGISTER 
// - ========== Register definition for PWMC_CH0 peripheral ========== 
AT91C_PWMC_CH0_CCNTR      EQU (0xFFFB820C) ;- (PWMC_CH0) Channel Counter Register
AT91C_PWMC_CH0_CPRDR      EQU (0xFFFB8208) ;- (PWMC_CH0) Channel Period Register
AT91C_PWMC_CH0_CUPDR      EQU (0xFFFB8210) ;- (PWMC_CH0) Channel Update Register
AT91C_PWMC_CH0_CDTYR      EQU (0xFFFB8204) ;- (PWMC_CH0) Channel Duty Cycle Register
AT91C_PWMC_CH0_CMR        EQU (0xFFFB8200) ;- (PWMC_CH0) Channel Mode Register
AT91C_PWMC_CH0_Reserved   EQU (0xFFFB8214) ;- (PWMC_CH0) Reserved
// - ========== Register definition for PWMC_CH1 peripheral ========== 
AT91C_PWMC_CH1_CCNTR      EQU (0xFFFB822C) ;- (PWMC_CH1) Channel Counter Register
AT91C_PWMC_CH1_CDTYR      EQU (0xFFFB8224) ;- (PWMC_CH1) Channel Duty Cycle Register
AT91C_PWMC_CH1_CMR        EQU (0xFFFB8220) ;- (PWMC_CH1) Channel Mode Register
AT91C_PWMC_CH1_CPRDR      EQU (0xFFFB8228) ;- (PWMC_CH1) Channel Period Register
AT91C_PWMC_CH1_Reserved   EQU (0xFFFB8234) ;- (PWMC_CH1) Reserved
AT91C_PWMC_CH1_CUPDR      EQU (0xFFFB8230) ;- (PWMC_CH1) Channel Update Register
// - ========== Register definition for PWMC_CH2 peripheral ========== 
AT91C_PWMC_CH2_CUPDR      EQU (0xFFFB8250) ;- (PWMC_CH2) Channel Update Register
AT91C_PWMC_CH2_CMR        EQU (0xFFFB8240) ;- (PWMC_CH2) Channel Mode Register
AT91C_PWMC_CH2_Reserved   EQU (0xFFFB8254) ;- (PWMC_CH2) Reserved
AT91C_PWMC_CH2_CPRDR      EQU (0xFFFB8248) ;- (PWMC_CH2) Channel Period Register
AT91C_PWMC_CH2_CDTYR      EQU (0xFFFB8244) ;- (PWMC_CH2) Channel Duty Cycle Register
AT91C_PWMC_CH2_CCNTR      EQU (0xFFFB824C) ;- (PWMC_CH2) Channel Counter Register
// - ========== Register definition for PWMC_CH3 peripheral ========== 
AT91C_PWMC_CH3_CPRDR      EQU (0xFFFB8268) ;- (PWMC_CH3) Channel Period Register
AT91C_PWMC_CH3_Reserved   EQU (0xFFFB8274) ;- (PWMC_CH3) Reserved
AT91C_PWMC_CH3_CUPDR      EQU (0xFFFB8270) ;- (PWMC_CH3) Channel Update Register
AT91C_PWMC_CH3_CDTYR      EQU (0xFFFB8264) ;- (PWMC_CH3) Channel Duty Cycle Register
AT91C_PWMC_CH3_CCNTR      EQU (0xFFFB826C) ;- (PWMC_CH3) Channel Counter Register
AT91C_PWMC_CH3_CMR        EQU (0xFFFB8260) ;- (PWMC_CH3) Channel Mode Register
// - ========== Register definition for PWMC peripheral ========== 
AT91C_PWMC_IDR            EQU (0xFFFB8014) ;- (PWMC) PWMC Interrupt Disable Register
AT91C_PWMC_MR             EQU (0xFFFB8000) ;- (PWMC) PWMC Mode Register
AT91C_PWMC_VR             EQU (0xFFFB80FC) ;- (PWMC) PWMC Version Register
AT91C_PWMC_IMR            EQU (0xFFFB8018) ;- (PWMC) PWMC Interrupt Mask Register
AT91C_PWMC_SR             EQU (0xFFFB800C) ;- (PWMC) PWMC Status Register
AT91C_PWMC_ISR            EQU (0xFFFB801C) ;- (PWMC) PWMC Interrupt Status Register
AT91C_PWMC_ENA            EQU (0xFFFB8004) ;- (PWMC) PWMC Enable Register
AT91C_PWMC_IER            EQU (0xFFFB8010) ;- (PWMC) PWMC Interrupt Enable Register
AT91C_PWMC_DIS            EQU (0xFFFB8008) ;- (PWMC) PWMC Disable Register
// - ========== Register definition for PDC_SPI0 peripheral ========== 
AT91C_SPI0_TPR            EQU (0xFFFA4108) ;- (PDC_SPI0) Transmit Pointer Register
AT91C_SPI0_PTCR           EQU (0xFFFA4120) ;- (PDC_SPI0) PDC Transfer Control Register
AT91C_SPI0_RNPR           EQU (0xFFFA4110) ;- (PDC_SPI0) Receive Next Pointer Register
AT91C_SPI0_TNCR           EQU (0xFFFA411C) ;- (PDC_SPI0) Transmit Next Counter Register
AT91C_SPI0_TCR            EQU (0xFFFA410C) ;- (PDC_SPI0) Transmit Counter Register
AT91C_SPI0_RCR            EQU (0xFFFA4104) ;- (PDC_SPI0) Receive Counter Register
AT91C_SPI0_RNCR           EQU (0xFFFA4114) ;- (PDC_SPI0) Receive Next Counter Register
AT91C_SPI0_TNPR           EQU (0xFFFA4118) ;- (PDC_SPI0) Transmit Next Pointer Register
AT91C_SPI0_RPR            EQU (0xFFFA4100) ;- (PDC_SPI0) Receive Pointer Register
AT91C_SPI0_PTSR           EQU (0xFFFA4124) ;- (PDC_SPI0) PDC Transfer Status Register
// - ========== Register definition for PDC_SPI1 peripheral ========== 
AT91C_SPI1_RNCR           EQU (0xFFFA8114) ;- (PDC_SPI1) Receive Next Counter Register
AT91C_SPI1_TCR            EQU (0xFFFA810C) ;- (PDC_SPI1) Transmit Counter Register
AT91C_SPI1_RCR            EQU (0xFFFA8104) ;- (PDC_SPI1) Receive Counter Register
AT91C_SPI1_TNPR           EQU (0xFFFA8118) ;- (PDC_SPI1) Transmit Next Pointer Register
AT91C_SPI1_RNPR           EQU (0xFFFA8110) ;- (PDC_SPI1) Receive Next Pointer Register
AT91C_SPI1_RPR            EQU (0xFFFA8100) ;- (PDC_SPI1) Receive Pointer Register
AT91C_SPI1_TNCR           EQU (0xFFFA811C) ;- (PDC_SPI1) Transmit Next Counter Register
AT91C_SPI1_TPR            EQU (0xFFFA8108) ;- (PDC_SPI1) Transmit Pointer Register
AT91C_SPI1_PTSR           EQU (0xFFFA8124) ;- (PDC_SPI1) PDC Transfer Status Register
AT91C_SPI1_PTCR           EQU (0xFFFA8120) ;- (PDC_SPI1) PDC Transfer Control Register
// - ========== Register definition for SPI0 peripheral ========== 
AT91C_SPI0_MR             EQU (0xFFFA4004) ;- (SPI0) Mode Register
AT91C_SPI0_RDR            EQU (0xFFFA4008) ;- (SPI0) Receive Data Register
AT91C_SPI0_CR             EQU (0xFFFA4000) ;- (SPI0) Control Register
AT91C_SPI0_IER            EQU (0xFFFA4014) ;- (SPI0) Interrupt Enable Register
AT91C_SPI0_TDR            EQU (0xFFFA400C) ;- (SPI0) Transmit Data Register
AT91C_SPI0_IDR            EQU (0xFFFA4018) ;- (SPI0) Interrupt Disable Register
AT91C_SPI0_CSR            EQU (0xFFFA4030) ;- (SPI0) Chip Select Register
AT91C_SPI0_SR             EQU (0xFFFA4010) ;- (SPI0) Status Register
AT91C_SPI0_IMR            EQU (0xFFFA401C) ;- (SPI0) Interrupt Mask Register
// - ========== Register definition for SPI1 peripheral ========== 
AT91C_SPI1_CSR            EQU (0xFFFA8030) ;- (SPI1) Chip Select Register
AT91C_SPI1_IER            EQU (0xFFFA8014) ;- (SPI1) Interrupt Enable Register
AT91C_SPI1_RDR            EQU (0xFFFA8008) ;- (SPI1) Receive Data Register
AT91C_SPI1_IDR            EQU (0xFFFA8018) ;- (SPI1) Interrupt Disable Register
AT91C_SPI1_MR             EQU (0xFFFA8004) ;- (SPI1) Mode Register
AT91C_SPI1_CR             EQU (0xFFFA8000) ;- (SPI1) Control Register
AT91C_SPI1_SR             EQU (0xFFFA8010) ;- (SPI1) Status Register
AT91C_SPI1_TDR            EQU (0xFFFA800C) ;- (SPI1) Transmit Data Register
AT91C_SPI1_IMR            EQU (0xFFFA801C) ;- (SPI1) Interrupt Mask Register
// - ========== Register definition for PDC_TSADC peripheral ========== 
AT91C_TSADC_TCR           EQU (0xFFFB010C) ;- (PDC_TSADC) Transmit Counter Register
AT91C_TSADC_PTCR          EQU (0xFFFB0120) ;- (PDC_TSADC) PDC Transfer Control Register
AT91C_TSADC_RNCR          EQU (0xFFFB0114) ;- (PDC_TSADC) Receive Next Counter Register
AT91C_TSADC_PTSR          EQU (0xFFFB0124) ;- (PDC_TSADC) PDC Transfer Status Register
AT91C_TSADC_TNCR          EQU (0xFFFB011C) ;- (PDC_TSADC) Transmit Next Counter Register
AT91C_TSADC_RNPR          EQU (0xFFFB0110) ;- (PDC_TSADC) Receive Next Pointer Register
AT91C_TSADC_RCR           EQU (0xFFFB0104) ;- (PDC_TSADC) Receive Counter Register
AT91C_TSADC_TPR           EQU (0xFFFB0108) ;- (PDC_TSADC) Transmit Pointer Register
AT91C_TSADC_TNPR          EQU (0xFFFB0118) ;- (PDC_TSADC) Transmit Next Pointer Register
AT91C_TSADC_RPR           EQU (0xFFFB0100) ;- (PDC_TSADC) Receive Pointer Register
// - ========== Register definition for TSADC peripheral ========== 
AT91C_TSADC_CHSR          EQU (0xFFFB0018) ;- (TSADC) Channel Status Register
AT91C_TSADC_CDR5          EQU (0xFFFB0044) ;- (TSADC) Channel Data Register 5
AT91C_TSADC_CR            EQU (0xFFFB0000) ;- (TSADC) Control Register
AT91C_TSADC_IMR           EQU (0xFFFB002C) ;- (TSADC) Interrupt Mask Register
AT91C_TSADC_CHDR          EQU (0xFFFB0014) ;- (TSADC) Channel Disable Register
AT91C_TSADC_LCDR          EQU (0xFFFB0020) ;- (TSADC) Last Converted Register
AT91C_TSADC_IER           EQU (0xFFFB0024) ;- (TSADC) Interrupt Enable Register
AT91C_TSADC_TSR           EQU (0xFFFB000C) ;- (TSADC) Touch Screen Register
AT91C_TSADC_CDR2          EQU (0xFFFB0038) ;- (TSADC) Channel Data Register 2
AT91C_TSADC_CDR4          EQU (0xFFFB0040) ;- (TSADC) Channel Data Register 4
AT91C_TSADC_CHER          EQU (0xFFFB0010) ;- (TSADC) Channel Enable Register
AT91C_TSADC_TRGR          EQU (0xFFFB0008) ;- (TSADC) Trigger Register
AT91C_TSADC_CDR3          EQU (0xFFFB003C) ;- (TSADC) Channel Data Register 3
AT91C_TSADC_SR            EQU (0xFFFB001C) ;- (TSADC) Status Register
AT91C_TSADC_CDR0          EQU (0xFFFB0030) ;- (TSADC) Channel Data Register 0
AT91C_TSADC_CDR6          EQU (0xFFFB0048) ;- (TSADC) Channel Data Register 6
AT91C_TSADC_IDR           EQU (0xFFFB0028) ;- (TSADC) Interrupt Disable Register
AT91C_TSADC_MR            EQU (0xFFFB0004) ;- (TSADC) Mode Register
AT91C_TSADC_CDR7          EQU (0xFFFB004C) ;- (TSADC) Channel Data Register 7
AT91C_TSADC_CDR1          EQU (0xFFFB0034) ;- (TSADC) Channel Data Register 1
// - ========== Register definition for UDPHS_EPTFIFO peripheral ========== 
AT91C_UDPHS_EPTFIFO_READEPT3 EQU (0x00630000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 3
AT91C_UDPHS_EPTFIFO_READEPT5 EQU (0x00650000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 5
AT91C_UDPHS_EPTFIFO_READEPT1 EQU (0x00610000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 1
AT91C_UDPHS_EPTFIFO_READEPT0 EQU (0x00600000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 0
AT91C_UDPHS_EPTFIFO_READEPT6 EQU (0x00660000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 6
AT91C_UDPHS_EPTFIFO_READEPT2 EQU (0x00620000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 2
AT91C_UDPHS_EPTFIFO_READEPT4 EQU (0x00640000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 4
// - ========== Register definition for UDPHS_EPT_0 peripheral ========== 
AT91C_UDPHS_EPT_0_EPTSTA  EQU (0xFFF7811C) ;- (UDPHS_EPT_0) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_0_EPTCTL  EQU (0xFFF7810C) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_0_EPTCTLDIS EQU (0xFFF78108) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_0_EPTCFG  EQU (0xFFF78100) ;- (UDPHS_EPT_0) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_0_EPTCLRSTA EQU (0xFFF78118) ;- (UDPHS_EPT_0) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_0_EPTSETSTA EQU (0xFFF78114) ;- (UDPHS_EPT_0) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_0_EPTCTLENB EQU (0xFFF78104) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Enable Register
// - ========== Register definition for UDPHS_EPT_1 peripheral ========== 
AT91C_UDPHS_EPT_1_EPTCTLENB EQU (0xFFF78124) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_1_EPTCFG  EQU (0xFFF78120) ;- (UDPHS_EPT_1) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_1_EPTCTL  EQU (0xFFF7812C) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_1_EPTSTA  EQU (0xFFF7813C) ;- (UDPHS_EPT_1) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_1_EPTCLRSTA EQU (0xFFF78138) ;- (UDPHS_EPT_1) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_1_EPTSETSTA EQU (0xFFF78134) ;- (UDPHS_EPT_1) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_1_EPTCTLDIS EQU (0xFFF78128) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Disable Register
// - ========== Register definition for UDPHS_EPT_2 peripheral ========== 
AT91C_UDPHS_EPT_2_EPTCLRSTA EQU (0xFFF78158) ;- (UDPHS_EPT_2) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_2_EPTCTLDIS EQU (0xFFF78148) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_2_EPTSTA  EQU (0xFFF7815C) ;- (UDPHS_EPT_2) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_2_EPTSETSTA EQU (0xFFF78154) ;- (UDPHS_EPT_2) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_2_EPTCTL  EQU (0xFFF7814C) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_2_EPTCFG  EQU (0xFFF78140) ;- (UDPHS_EPT_2) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_2_EPTCTLENB EQU (0xFFF78144) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Enable Register
// - ========== Register definition for UDPHS_EPT_3 peripheral ========== 
AT91C_UDPHS_EPT_3_EPTCTL  EQU (0xFFF7816C) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_3_EPTCLRSTA EQU (0xFFF78178) ;- (UDPHS_EPT_3) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_3_EPTCTLDIS EQU (0xFFF78168) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_3_EPTSTA  EQU (0xFFF7817C) ;- (UDPHS_EPT_3) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_3_EPTSETSTA EQU (0xFFF78174) ;- (UDPHS_EPT_3) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_3_EPTCTLENB EQU (0xFFF78164) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_3_EPTCFG  EQU (0xFFF78160) ;- (UDPHS_EPT_3) UDPHS Endpoint Config Register
// - ========== Register definition for UDPHS_EPT_4 peripheral ========== 
AT91C_UDPHS_EPT_4_EPTCLRSTA EQU (0xFFF78198) ;- (UDPHS_EPT_4) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_4_EPTCTL  EQU (0xFFF7818C) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_4_EPTCTLENB EQU (0xFFF78184) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_4_EPTSTA  EQU (0xFFF7819C) ;- (UDPHS_EPT_4) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_4_EPTSETSTA EQU (0xFFF78194) ;- (UDPHS_EPT_4) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_4_EPTCFG  EQU (0xFFF78180) ;- (UDPHS_EPT_4) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_4_EPTCTLDIS EQU (0xFFF78188) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Disable Register
// - ========== Register definition for UDPHS_EPT_5 peripheral ========== 
AT91C_UDPHS_EPT_5_EPTSTA  EQU (0xFFF781BC) ;- (UDPHS_EPT_5) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_5_EPTCLRSTA EQU (0xFFF781B8) ;- (UDPHS_EPT_5) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_5_EPTCTLENB EQU (0xFFF781A4) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_5_EPTSETSTA EQU (0xFFF781B4) ;- (UDPHS_EPT_5) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_5_EPTCTLDIS EQU (0xFFF781A8) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_5_EPTCFG  EQU (0xFFF781A0) ;- (UDPHS_EPT_5) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_5_EPTCTL  EQU (0xFFF781AC) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Register
// - ========== Register definition for UDPHS_EPT_6 peripheral ========== 
AT91C_UDPHS_EPT_6_EPTCLRSTA EQU (0xFFF781D8) ;- (UDPHS_EPT_6) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_6_EPTCTLENB EQU (0xFFF781C4) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_6_EPTCTL  EQU (0xFFF781CC) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_6_EPTSETSTA EQU (0xFFF781D4) ;- (UDPHS_EPT_6) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_6_EPTCTLDIS EQU (0xFFF781C8) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_6_EPTSTA  EQU (0xFFF781DC) ;- (UDPHS_EPT_6) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_6_EPTCFG  EQU (0xFFF781C0) ;- (UDPHS_EPT_6) UDPHS Endpoint Config Register
// - ========== Register definition for UDPHS_DMA_1 peripheral ========== 
AT91C_UDPHS_DMA_1_DMASTATUS EQU (0xFFF7831C) ;- (UDPHS_DMA_1) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_1_DMANXTDSC EQU (0xFFF78310) ;- (UDPHS_DMA_1) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_1_DMACONTROL EQU (0xFFF78318) ;- (UDPHS_DMA_1) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_1_DMAADDRESS EQU (0xFFF78314) ;- (UDPHS_DMA_1) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS_DMA_2 peripheral ========== 
AT91C_UDPHS_DMA_2_DMACONTROL EQU (0xFFF78328) ;- (UDPHS_DMA_2) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_2_DMASTATUS EQU (0xFFF7832C) ;- (UDPHS_DMA_2) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_2_DMAADDRESS EQU (0xFFF78324) ;- (UDPHS_DMA_2) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_2_DMANXTDSC EQU (0xFFF78320) ;- (UDPHS_DMA_2) UDPHS DMA Channel Next Descriptor Address
// - ========== Register definition for UDPHS_DMA_3 peripheral ========== 
AT91C_UDPHS_DMA_3_DMAADDRESS EQU (0xFFF78334) ;- (UDPHS_DMA_3) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_3_DMANXTDSC EQU (0xFFF78330) ;- (UDPHS_DMA_3) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_3_DMACONTROL EQU (0xFFF78338) ;- (UDPHS_DMA_3) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_3_DMASTATUS EQU (0xFFF7833C) ;- (UDPHS_DMA_3) UDPHS DMA Channel Status Register
// - ========== Register definition for UDPHS_DMA_4 peripheral ========== 
AT91C_UDPHS_DMA_4_DMANXTDSC EQU (0xFFF78340) ;- (UDPHS_DMA_4) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_4_DMAADDRESS EQU (0xFFF78344) ;- (UDPHS_DMA_4) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_4_DMACONTROL EQU (0xFFF78348) ;- (UDPHS_DMA_4) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_4_DMASTATUS EQU (0xFFF7834C) ;- (UDPHS_DMA_4) UDPHS DMA Channel Status Register
// - ========== Register definition for UDPHS_DMA_5 peripheral ========== 
AT91C_UDPHS_DMA_5_DMASTATUS EQU (0xFFF7835C) ;- (UDPHS_DMA_5) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_5_DMACONTROL EQU (0xFFF78358) ;- (UDPHS_DMA_5) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_5_DMANXTDSC EQU (0xFFF78350) ;- (UDPHS_DMA_5) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_5_DMAADDRESS EQU (0xFFF78354) ;- (UDPHS_DMA_5) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS_DMA_6 peripheral ========== 
AT91C_UDPHS_DMA_6_DMANXTDSC EQU (0xFFF78360) ;- (UDPHS_DMA_6) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_6_DMACONTROL EQU (0xFFF78368) ;- (UDPHS_DMA_6) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_6_DMASTATUS EQU (0xFFF7836C) ;- (UDPHS_DMA_6) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_6_DMAADDRESS EQU (0xFFF78364) ;- (UDPHS_DMA_6) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS peripheral ========== 
AT91C_UDPHS_IEN           EQU (0xFFF78010) ;- (UDPHS) UDPHS Interrupt Enable Register
AT91C_UDPHS_TSTSOFCNT     EQU (0xFFF780D0) ;- (UDPHS) UDPHS Test SOF Counter Register
AT91C_UDPHS_IPFEATURES    EQU (0xFFF780F8) ;- (UDPHS) UDPHS Features Register
AT91C_UDPHS_TST           EQU (0xFFF780E0) ;- (UDPHS) UDPHS Test Register
AT91C_UDPHS_FNUM          EQU (0xFFF78004) ;- (UDPHS) UDPHS Frame Number Register
AT91C_UDPHS_TSTCNTB       EQU (0xFFF780D8) ;- (UDPHS) UDPHS Test B Counter Register
AT91C_UDPHS_RIPPADDRSIZE  EQU (0xFFF780EC) ;- (UDPHS) UDPHS PADDRSIZE Register
AT91C_UDPHS_INTSTA        EQU (0xFFF78014) ;- (UDPHS) UDPHS Interrupt Status Register
AT91C_UDPHS_EPTRST        EQU (0xFFF7801C) ;- (UDPHS) UDPHS Endpoints Reset Register
AT91C_UDPHS_TSTCNTA       EQU (0xFFF780D4) ;- (UDPHS) UDPHS Test A Counter Register
AT91C_UDPHS_RIPNAME2      EQU (0xFFF780F4) ;- (UDPHS) UDPHS Name2 Register
AT91C_UDPHS_RIPNAME1      EQU (0xFFF780F0) ;- (UDPHS) UDPHS Name1 Register
AT91C_UDPHS_TSTMODREG     EQU (0xFFF780DC) ;- (UDPHS) UDPHS Test Mode Register
AT91C_UDPHS_CLRINT        EQU (0xFFF78018) ;- (UDPHS) UDPHS Clear Interrupt Register
AT91C_UDPHS_IPVERSION     EQU (0xFFF780FC) ;- (UDPHS) UDPHS Version Register
AT91C_UDPHS_CTRL          EQU (0xFFF78000) ;- (UDPHS) UDPHS Control Register
// - ========== Register definition for PDC_AC97C peripheral ========== 
AT91C_AC97C_PTSR          EQU (0xFFFAC124) ;- (PDC_AC97C) PDC Transfer Status Register
AT91C_AC97C_RPR           EQU (0xFFFAC100) ;- (PDC_AC97C) Receive Pointer Register
AT91C_AC97C_RNCR          EQU (0xFFFAC114) ;- (PDC_AC97C) Receive Next Counter Register
AT91C_AC97C_RCR           EQU (0xFFFAC104) ;- (PDC_AC97C) Receive Counter Register
AT91C_AC97C_PTCR          EQU (0xFFFAC120) ;- (PDC_AC97C) PDC Transfer Control Register
AT91C_AC97C_TPR           EQU (0xFFFAC108) ;- (PDC_AC97C) Transmit Pointer Register
AT91C_AC97C_RNPR          EQU (0xFFFAC110) ;- (PDC_AC97C) Receive Next Pointer Register
AT91C_AC97C_TNPR          EQU (0xFFFAC118) ;- (PDC_AC97C) Transmit Next Pointer Register
AT91C_AC97C_TCR           EQU (0xFFFAC10C) ;- (PDC_AC97C) Transmit Counter Register
AT91C_AC97C_TNCR          EQU (0xFFFAC11C) ;- (PDC_AC97C) Transmit Next Counter Register
// - ========== Register definition for AC97C peripheral ========== 
AT91C_AC97C_IER           EQU (0xFFFAC054) ;- (AC97C) Interrupt Enable Register
AT91C_AC97C_COTHR         EQU (0xFFFAC044) ;- (AC97C) COdec Transmit Holding Register
AT91C_AC97C_IDR           EQU (0xFFFAC058) ;- (AC97C) Interrupt Disable Register
AT91C_AC97C_ICA           EQU (0xFFFAC010) ;- (AC97C) Input Channel AssignementRegister
AT91C_AC97C_CATHR         EQU (0xFFFAC024) ;- (AC97C) Channel A Transmit Holding Register
AT91C_AC97C_CBSR          EQU (0xFFFAC038) ;- (AC97C) Channel B Status Register
AT91C_AC97C_CAMR          EQU (0xFFFAC02C) ;- (AC97C) Channel A Mode Register
AT91C_AC97C_SR            EQU (0xFFFAC050) ;- (AC97C) Status Register
AT91C_AC97C_CBTHR         EQU (0xFFFAC034) ;- (AC97C) Channel B Transmit Holding Register (optional)
AT91C_AC97C_CASR          EQU (0xFFFAC028) ;- (AC97C) Channel A Status Register
AT91C_AC97C_COSR          EQU (0xFFFAC048) ;- (AC97C) CODEC Status Register
AT91C_AC97C_MR            EQU (0xFFFAC008) ;- (AC97C) Mode Register
AT91C_AC97C_OCA           EQU (0xFFFAC014) ;- (AC97C) Output Channel Assignement Register
AT91C_AC97C_CORHR         EQU (0xFFFAC040) ;- (AC97C) COdec Transmit Holding Register
AT91C_AC97C_CBRHR         EQU (0xFFFAC030) ;- (AC97C) Channel B Receive Holding Register (optional)
AT91C_AC97C_IMR           EQU (0xFFFAC05C) ;- (AC97C) Interrupt Mask Register
AT91C_AC97C_COMR          EQU (0xFFFAC04C) ;- (AC97C) CODEC Mask Status Register
AT91C_AC97C_CARHR         EQU (0xFFFAC020) ;- (AC97C) Channel A Receive Holding Register
AT91C_AC97C_VERSION       EQU (0xFFFAC0FC) ;- (AC97C) Version Register
AT91C_AC97C_CBMR          EQU (0xFFFAC03C) ;- (AC97C) Channel B Mode Register
// - ========== Register definition for LCDC peripheral ========== 
AT91C_LCDC_MVAL           EQU (0x00500818) ;- (LCDC) LCD Mode Toggle Rate Value Register
AT91C_LCDC_PWRCON         EQU (0x0050083C) ;- (LCDC) Power Control Register
AT91C_LCDC_ISR            EQU (0x00500854) ;- (LCDC) Interrupt Enable Register
AT91C_LCDC_FRMP1          EQU (0x00500008) ;- (LCDC) DMA Frame Pointer Register 1
AT91C_LCDC_CTRSTVAL       EQU (0x00500844) ;- (LCDC) Contrast Value Register
AT91C_LCDC_ICR            EQU (0x00500858) ;- (LCDC) Interrupt Clear Register
AT91C_LCDC_TIM1           EQU (0x00500808) ;- (LCDC) LCD Timing Config 1 Register
AT91C_LCDC_DMACON         EQU (0x0050001C) ;- (LCDC) DMA Control Register
AT91C_LCDC_ITR            EQU (0x00500860) ;- (LCDC) Interrupts Test Register
AT91C_LCDC_IDR            EQU (0x0050084C) ;- (LCDC) Interrupt Disable Register
AT91C_LCDC_DP4_7          EQU (0x00500820) ;- (LCDC) Dithering Pattern DP4_7 Register
AT91C_LCDC_DP5_7          EQU (0x0050082C) ;- (LCDC) Dithering Pattern DP5_7 Register
AT91C_LCDC_IRR            EQU (0x00500864) ;- (LCDC) Interrupts Raw Status Register
AT91C_LCDC_DP3_4          EQU (0x00500830) ;- (LCDC) Dithering Pattern DP3_4 Register
AT91C_LCDC_IMR            EQU (0x00500850) ;- (LCDC) Interrupt Mask Register
AT91C_LCDC_LCDFRCFG       EQU (0x00500810) ;- (LCDC) LCD Frame Config Register
AT91C_LCDC_CTRSTCON       EQU (0x00500840) ;- (LCDC) Contrast Control Register
AT91C_LCDC_DP1_2          EQU (0x0050081C) ;- (LCDC) Dithering Pattern DP1_2 Register
AT91C_LCDC_FRMP2          EQU (0x0050000C) ;- (LCDC) DMA Frame Pointer Register 2
AT91C_LCDC_LCDCON1        EQU (0x00500800) ;- (LCDC) LCD Control 1 Register
AT91C_LCDC_DP4_5          EQU (0x00500834) ;- (LCDC) Dithering Pattern DP4_5 Register
AT91C_LCDC_FRMA2          EQU (0x00500014) ;- (LCDC) DMA Frame Address Register 2
AT91C_LCDC_BA1            EQU (0x00500000) ;- (LCDC) DMA Base Address Register 1
AT91C_LCDC_DMA2DCFG       EQU (0x00500020) ;- (LCDC) DMA 2D addressing configuration
AT91C_LCDC_LUT_ENTRY      EQU (0x00500C00) ;- (LCDC) LUT Entries Register
AT91C_LCDC_DP6_7          EQU (0x00500838) ;- (LCDC) Dithering Pattern DP6_7 Register
AT91C_LCDC_FRMCFG         EQU (0x00500018) ;- (LCDC) DMA Frame Configuration Register
AT91C_LCDC_TIM2           EQU (0x0050080C) ;- (LCDC) LCD Timing Config 2 Register
AT91C_LCDC_DP3_5          EQU (0x00500824) ;- (LCDC) Dithering Pattern DP3_5 Register
AT91C_LCDC_FRMA1          EQU (0x00500010) ;- (LCDC) DMA Frame Address Register 1
AT91C_LCDC_IER            EQU (0x00500848) ;- (LCDC) Interrupt Enable Register
AT91C_LCDC_DP2_3          EQU (0x00500828) ;- (LCDC) Dithering Pattern DP2_3 Register
AT91C_LCDC_FIFO           EQU (0x00500814) ;- (LCDC) LCD FIFO Register
AT91C_LCDC_BA2            EQU (0x00500004) ;- (LCDC) DMA Base Address Register 2
AT91C_LCDC_LCDCON2        EQU (0x00500804) ;- (LCDC) LCD Control 2 Register
AT91C_LCDC_GPR            EQU (0x0050085C) ;- (LCDC) General Purpose Register
// - ========== Register definition for LCDC_16B_TFT peripheral ========== 
AT91C_TFT_MVAL            EQU (0x00500818) ;- (LCDC_16B_TFT) LCD Mode Toggle Rate Value Register
AT91C_TFT_PWRCON          EQU (0x0050083C) ;- (LCDC_16B_TFT) Power Control Register
AT91C_TFT_ISR             EQU (0x00500854) ;- (LCDC_16B_TFT) Interrupt Enable Register
AT91C_TFT_FRMP1           EQU (0x00500008) ;- (LCDC_16B_TFT) DMA Frame Pointer Register 1
AT91C_TFT_CTRSTVAL        EQU (0x00500844) ;- (LCDC_16B_TFT) Contrast Value Register
AT91C_TFT_ICR             EQU (0x00500858) ;- (LCDC_16B_TFT) Interrupt Clear Register
AT91C_TFT_TIM1            EQU (0x00500808) ;- (LCDC_16B_TFT) LCD Timing Config 1 Register
AT91C_TFT_DMACON          EQU (0x0050001C) ;- (LCDC_16B_TFT) DMA Control Register
AT91C_TFT_ITR             EQU (0x00500860) ;- (LCDC_16B_TFT) Interrupts Test Register
AT91C_TFT_IDR             EQU (0x0050084C) ;- (LCDC_16B_TFT) Interrupt Disable Register
AT91C_TFT_DP4_7           EQU (0x00500820) ;- (LCDC_16B_TFT) Dithering Pattern DP4_7 Register
AT91C_TFT_DP5_7           EQU (0x0050082C) ;- (LCDC_16B_TFT) Dithering Pattern DP5_7 Register
AT91C_TFT_IRR             EQU (0x00500864) ;- (LCDC_16B_TFT) Interrupts Raw Status Register
AT91C_TFT_DP3_4           EQU (0x00500830) ;- (LCDC_16B_TFT) Dithering Pattern DP3_4 Register
AT91C_TFT_IMR             EQU (0x00500850) ;- (LCDC_16B_TFT) Interrupt Mask Register
AT91C_TFT_LCDFRCFG        EQU (0x00500810) ;- (LCDC_16B_TFT) LCD Frame Config Register
AT91C_TFT_CTRSTCON        EQU (0x00500840) ;- (LCDC_16B_TFT) Contrast Control Register
AT91C_TFT_DP1_2           EQU (0x0050081C) ;- (LCDC_16B_TFT) Dithering Pattern DP1_2 Register
AT91C_TFT_FRMP2           EQU (0x0050000C) ;- (LCDC_16B_TFT) DMA Frame Pointer Register 2
AT91C_TFT_LCDCON1         EQU (0x00500800) ;- (LCDC_16B_TFT) LCD Control 1 Register
AT91C_TFT_DP4_5           EQU (0x00500834) ;- (LCDC_16B_TFT) Dithering Pattern DP4_5 Register
AT91C_TFT_FRMA2           EQU (0x00500014) ;- (LCDC_16B_TFT) DMA Frame Address Register 2
AT91C_TFT_BA1             EQU (0x00500000) ;- (LCDC_16B_TFT) DMA Base Address Register 1
AT91C_TFT_DMA2DCFG        EQU (0x00500020) ;- (LCDC_16B_TFT) DMA 2D addressing configuration
AT91C_TFT_LUT_ENTRY       EQU (0x00500C00) ;- (LCDC_16B_TFT) LUT Entries Register
AT91C_TFT_DP6_7           EQU (0x00500838) ;- (LCDC_16B_TFT) Dithering Pattern DP6_7 Register
AT91C_TFT_FRMCFG          EQU (0x00500018) ;- (LCDC_16B_TFT) DMA Frame Configuration Register
AT91C_TFT_TIM2            EQU (0x0050080C) ;- (LCDC_16B_TFT) LCD Timing Config 2 Register
AT91C_TFT_DP3_5           EQU (0x00500824) ;- (LCDC_16B_TFT) Dithering Pattern DP3_5 Register
AT91C_TFT_FRMA1           EQU (0x00500010) ;- (LCDC_16B_TFT) DMA Frame Address Register 1
AT91C_TFT_IER             EQU (0x00500848) ;- (LCDC_16B_TFT) Interrupt Enable Register
AT91C_TFT_DP2_3           EQU (0x00500828) ;- (LCDC_16B_TFT) Dithering Pattern DP2_3 Register
AT91C_TFT_FIFO            EQU (0x00500814) ;- (LCDC_16B_TFT) LCD FIFO Register
AT91C_TFT_BA2             EQU (0x00500004) ;- (LCDC_16B_TFT) DMA Base Address Register 2
AT91C_TFT_LCDCON2         EQU (0x00500804) ;- (LCDC_16B_TFT) LCD Control 2 Register
AT91C_TFT_GPR             EQU (0x0050085C) ;- (LCDC_16B_TFT) General Purpose Register
// - ========== Register definition for HDMA_CH_0 peripheral ========== 
AT91C_HDMA_CH_0_BDSCR     EQU (0xFFFFEC5C) ;- (HDMA_CH_0) HDMA Reserved
AT91C_HDMA_CH_0_DADDR     EQU (0xFFFFEC40) ;- (HDMA_CH_0) HDMA Channel Destination Address Register
AT91C_HDMA_CH_0_DPIP      EQU (0xFFFFEC58) ;- (HDMA_CH_0) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_0_DSCR      EQU (0xFFFFEC44) ;- (HDMA_CH_0) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_0_CFG       EQU (0xFFFFEC50) ;- (HDMA_CH_0) HDMA Channel Configuration Register
AT91C_HDMA_CH_0_SPIP      EQU (0xFFFFEC54) ;- (HDMA_CH_0) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_0_CADDR     EQU (0xFFFFEC60) ;- (HDMA_CH_0) HDMA Reserved
AT91C_HDMA_CH_0_CTRLA     EQU (0xFFFFEC48) ;- (HDMA_CH_0) HDMA Channel Control A Register
AT91C_HDMA_CH_0_CTRLB     EQU (0xFFFFEC4C) ;- (HDMA_CH_0) HDMA Channel Control B Register
AT91C_HDMA_CH_0_SADDR     EQU (0xFFFFEC3C) ;- (HDMA_CH_0) HDMA Channel Source Address Register
// - ========== Register definition for HDMA_CH_1 peripheral ========== 
AT91C_HDMA_CH_1_CADDR     EQU (0xFFFFEC88) ;- (HDMA_CH_1) HDMA Reserved
AT91C_HDMA_CH_1_DPIP      EQU (0xFFFFEC80) ;- (HDMA_CH_1) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_1_CTRLB     EQU (0xFFFFEC74) ;- (HDMA_CH_1) HDMA Channel Control B Register
AT91C_HDMA_CH_1_SADDR     EQU (0xFFFFEC64) ;- (HDMA_CH_1) HDMA Channel Source Address Register
AT91C_HDMA_CH_1_BDSCR     EQU (0xFFFFEC84) ;- (HDMA_CH_1) HDMA Reserved
AT91C_HDMA_CH_1_CFG       EQU (0xFFFFEC78) ;- (HDMA_CH_1) HDMA Channel Configuration Register
AT91C_HDMA_CH_1_DSCR      EQU (0xFFFFEC6C) ;- (HDMA_CH_1) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_1_DADDR     EQU (0xFFFFEC68) ;- (HDMA_CH_1) HDMA Channel Destination Address Register
AT91C_HDMA_CH_1_CTRLA     EQU (0xFFFFEC70) ;- (HDMA_CH_1) HDMA Channel Control A Register
AT91C_HDMA_CH_1_SPIP      EQU (0xFFFFEC7C) ;- (HDMA_CH_1) HDMA Channel Source Picture in Picture Configuration Register
// - ========== Register definition for HDMA_CH_2 peripheral ========== 
AT91C_HDMA_CH_2_SADDR     EQU (0xFFFFEC8C) ;- (HDMA_CH_2) HDMA Channel Source Address Register
AT91C_HDMA_CH_2_BDSCR     EQU (0xFFFFECAC) ;- (HDMA_CH_2) HDMA Reserved
AT91C_HDMA_CH_2_DPIP      EQU (0xFFFFECA8) ;- (HDMA_CH_2) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_2_SPIP      EQU (0xFFFFECA4) ;- (HDMA_CH_2) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_2_DADDR     EQU (0xFFFFEC90) ;- (HDMA_CH_2) HDMA Channel Destination Address Register
AT91C_HDMA_CH_2_DSCR      EQU (0xFFFFEC94) ;- (HDMA_CH_2) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_2_CTRLA     EQU (0xFFFFEC98) ;- (HDMA_CH_2) HDMA Channel Control A Register
AT91C_HDMA_CH_2_CADDR     EQU (0xFFFFECB0) ;- (HDMA_CH_2) HDMA Reserved
AT91C_HDMA_CH_2_CFG       EQU (0xFFFFECA0) ;- (HDMA_CH_2) HDMA Channel Configuration Register
AT91C_HDMA_CH_2_CTRLB     EQU (0xFFFFEC9C) ;- (HDMA_CH_2) HDMA Channel Control B Register
// - ========== Register definition for HDMA_CH_3 peripheral ========== 
AT91C_HDMA_CH_3_SPIP      EQU (0xFFFFECCC) ;- (HDMA_CH_3) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_3_CTRLA     EQU (0xFFFFECC0) ;- (HDMA_CH_3) HDMA Channel Control A Register
AT91C_HDMA_CH_3_DPIP      EQU (0xFFFFECD0) ;- (HDMA_CH_3) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_3_CTRLB     EQU (0xFFFFECC4) ;- (HDMA_CH_3) HDMA Channel Control B Register
AT91C_HDMA_CH_3_BDSCR     EQU (0xFFFFECD4) ;- (HDMA_CH_3) HDMA Reserved
AT91C_HDMA_CH_3_DSCR      EQU (0xFFFFECBC) ;- (HDMA_CH_3) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_3_CADDR     EQU (0xFFFFECD8) ;- (HDMA_CH_3) HDMA Reserved
AT91C_HDMA_CH_3_CFG       EQU (0xFFFFECC8) ;- (HDMA_CH_3) HDMA Channel Configuration Register
AT91C_HDMA_CH_3_DADDR     EQU (0xFFFFECB8) ;- (HDMA_CH_3) HDMA Channel Destination Address Register
AT91C_HDMA_CH_3_SADDR     EQU (0xFFFFECB4) ;- (HDMA_CH_3) HDMA Channel Source Address Register
// - ========== Register definition for HDMA_CH_4 peripheral ========== 
AT91C_HDMA_CH_4_CFG       EQU (0xFFFFECF0) ;- (HDMA_CH_4) HDMA Channel Configuration Register
AT91C_HDMA_CH_4_BDSCR     EQU (0xFFFFECFC) ;- (HDMA_CH_4) HDMA Reserved
AT91C_HDMA_CH_4_CADDR     EQU (0xFFFFED00) ;- (HDMA_CH_4) HDMA Reserved
AT91C_HDMA_CH_4_DADDR     EQU (0xFFFFECE0) ;- (HDMA_CH_4) HDMA Channel Destination Address Register
AT91C_HDMA_CH_4_CTRLA     EQU (0xFFFFECE8) ;- (HDMA_CH_4) HDMA Channel Control A Register
AT91C_HDMA_CH_4_SADDR     EQU (0xFFFFECDC) ;- (HDMA_CH_4) HDMA Channel Source Address Register
AT91C_HDMA_CH_4_CTRLB     EQU (0xFFFFECEC) ;- (HDMA_CH_4) HDMA Channel Control B Register
AT91C_HDMA_CH_4_DSCR      EQU (0xFFFFECE4) ;- (HDMA_CH_4) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_4_SPIP      EQU (0xFFFFECF4) ;- (HDMA_CH_4) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_4_DPIP      EQU (0xFFFFECF8) ;- (HDMA_CH_4) HDMA Channel Destination Picture in Picture Configuration Register
// - ========== Register definition for HDMA_CH_5 peripheral ========== 
AT91C_HDMA_CH_5_CTRLB     EQU (0xFFFFED14) ;- (HDMA_CH_5) HDMA Channel Control B Register
AT91C_HDMA_CH_5_DADDR     EQU (0xFFFFED08) ;- (HDMA_CH_5) HDMA Channel Destination Address Register
AT91C_HDMA_CH_5_SPIP      EQU (0xFFFFED1C) ;- (HDMA_CH_5) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_5_DSCR      EQU (0xFFFFED0C) ;- (HDMA_CH_5) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_5_DPIP      EQU (0xFFFFED20) ;- (HDMA_CH_5) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_5_CFG       EQU (0xFFFFED18) ;- (HDMA_CH_5) HDMA Channel Configuration Register
AT91C_HDMA_CH_5_CADDR     EQU (0xFFFFED28) ;- (HDMA_CH_5) HDMA Reserved
AT91C_HDMA_CH_5_SADDR     EQU (0xFFFFED04) ;- (HDMA_CH_5) HDMA Channel Source Address Register
AT91C_HDMA_CH_5_BDSCR     EQU (0xFFFFED24) ;- (HDMA_CH_5) HDMA Reserved
AT91C_HDMA_CH_5_CTRLA     EQU (0xFFFFED10) ;- (HDMA_CH_5) HDMA Channel Control A Register
// - ========== Register definition for HDMA_CH_6 peripheral ========== 
AT91C_HDMA_CH_6_SADDR     EQU (0xFFFFED2C) ;- (HDMA_CH_6) HDMA Channel Source Address Register
AT91C_HDMA_CH_6_BDSCR     EQU (0xFFFFED4C) ;- (HDMA_CH_6) HDMA Reserved
AT91C_HDMA_CH_6_CADDR     EQU (0xFFFFED50) ;- (HDMA_CH_6) HDMA Reserved
AT91C_HDMA_CH_6_SPIP      EQU (0xFFFFED44) ;- (HDMA_CH_6) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_6_CTRLB     EQU (0xFFFFED3C) ;- (HDMA_CH_6) HDMA Channel Control B Register
AT91C_HDMA_CH_6_DSCR      EQU (0xFFFFED34) ;- (HDMA_CH_6) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_6_CFG       EQU (0xFFFFED40) ;- (HDMA_CH_6) HDMA Channel Configuration Register
AT91C_HDMA_CH_6_DPIP      EQU (0xFFFFED48) ;- (HDMA_CH_6) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_6_DADDR     EQU (0xFFFFED30) ;- (HDMA_CH_6) HDMA Channel Destination Address Register
AT91C_HDMA_CH_6_CTRLA     EQU (0xFFFFED38) ;- (HDMA_CH_6) HDMA Channel Control A Register
// - ========== Register definition for HDMA_CH_7 peripheral ========== 
AT91C_HDMA_CH_7_CADDR     EQU (0xFFFFED78) ;- (HDMA_CH_7) HDMA Reserved
AT91C_HDMA_CH_7_CFG       EQU (0xFFFFED68) ;- (HDMA_CH_7) HDMA Channel Configuration Register
AT91C_HDMA_CH_7_DADDR     EQU (0xFFFFED58) ;- (HDMA_CH_7) HDMA Channel Destination Address Register
AT91C_HDMA_CH_7_CTRLB     EQU (0xFFFFED64) ;- (HDMA_CH_7) HDMA Channel Control B Register
AT91C_HDMA_CH_7_DSCR      EQU (0xFFFFED5C) ;- (HDMA_CH_7) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_7_DPIP      EQU (0xFFFFED70) ;- (HDMA_CH_7) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_7_CTRLA     EQU (0xFFFFED60) ;- (HDMA_CH_7) HDMA Channel Control A Register
AT91C_HDMA_CH_7_BDSCR     EQU (0xFFFFED74) ;- (HDMA_CH_7) HDMA Reserved
AT91C_HDMA_CH_7_SPIP      EQU (0xFFFFED6C) ;- (HDMA_CH_7) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_7_SADDR     EQU (0xFFFFED54) ;- (HDMA_CH_7) HDMA Channel Source Address Register
// - ========== Register definition for HDMA peripheral ========== 
AT91C_HDMA_EBCIDR         EQU (0xFFFFEC1C) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register
AT91C_HDMA_LAST           EQU (0xFFFFEC10) ;- (HDMA) HDMA Software Last Transfer Flag Register
AT91C_HDMA_SREQ           EQU (0xFFFFEC08) ;- (HDMA) HDMA Software Single Request Register
AT91C_HDMA_RSVD0          EQU (0xFFFFEC34) ;- (HDMA) HDMA Reserved
AT91C_HDMA_EBCIER         EQU (0xFFFFEC18) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
AT91C_HDMA_GCFG           EQU (0xFFFFEC00) ;- (HDMA) HDMA Global Configuration Register
AT91C_HDMA_CHER           EQU (0xFFFFEC28) ;- (HDMA) HDMA Channel Handler Enable Register
AT91C_HDMA_RSVD1          EQU (0xFFFFEC38) ;- (HDMA) HDMA Reserved
AT91C_HDMA_CHDR           EQU (0xFFFFEC2C) ;- (HDMA) HDMA Channel Handler Disable Register
AT91C_HDMA_EBCIMR         EQU (0xFFFFEC20) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
AT91C_HDMA_CREQ           EQU (0xFFFFEC0C) ;- (HDMA) HDMA Software Chunk Transfer Request Register
AT91C_HDMA_SYNC           EQU (0xFFFFEC14) ;- (HDMA) HDMA Request Synchronization Register
AT91C_HDMA_EN             EQU (0xFFFFEC04) ;- (HDMA) HDMA Controller Enable Register
AT91C_HDMA_EBCISR         EQU (0xFFFFEC24) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
AT91C_HDMA_CHSR           EQU (0xFFFFEC30) ;- (HDMA) HDMA Channel Handler Status Register
// - ========== Register definition for HECC peripheral ========== 
AT91C_HECC_VR             EQU (0xFFFFE2FC) ;- (HECC)  ECC Version register
AT91C_HECC_SR             EQU (0xFFFFE208) ;- (HECC)  ECC Status register
AT91C_HECC_CR             EQU (0xFFFFE200) ;- (HECC)  ECC reset register
AT91C_HECC_NPR            EQU (0xFFFFE210) ;- (HECC)  ECC Parity N register
AT91C_HECC_PR             EQU (0xFFFFE20C) ;- (HECC)  ECC Parity register
AT91C_HECC_MR             EQU (0xFFFFE204) ;- (HECC)  ECC Page size register
// - ========== Register definition for EMACB peripheral ========== 
AT91C_EMACB_ALE           EQU (0xFFFBC054) ;- (EMACB) Alignment Error Register
AT91C_EMACB_RRE           EQU (0xFFFBC06C) ;- (EMACB) Receive Ressource Error Register
AT91C_EMACB_SA4H          EQU (0xFFFBC0B4) ;- (EMACB) Specific Address 4 Top, Last 2 bytes
AT91C_EMACB_TPQ           EQU (0xFFFBC0BC) ;- (EMACB) Transmit Pause Quantum Register
AT91C_EMACB_RJA           EQU (0xFFFBC07C) ;- (EMACB) Receive Jabbers Register
AT91C_EMACB_SA2H          EQU (0xFFFBC0A4) ;- (EMACB) Specific Address 2 Top, Last 2 bytes
AT91C_EMACB_TPF           EQU (0xFFFBC08C) ;- (EMACB) Transmitted Pause Frames Register
AT91C_EMACB_ROV           EQU (0xFFFBC070) ;- (EMACB) Receive Overrun Errors Register
AT91C_EMACB_SA4L          EQU (0xFFFBC0B0) ;- (EMACB) Specific Address 4 Bottom, First 4 bytes
AT91C_EMACB_MAN           EQU (0xFFFBC034) ;- (EMACB) PHY Maintenance Register
AT91C_EMACB_TID           EQU (0xFFFBC0B8) ;- (EMACB) Type ID Checking Register
AT91C_EMACB_TBQP          EQU (0xFFFBC01C) ;- (EMACB) Transmit Buffer Queue Pointer
AT91C_EMACB_SA3L          EQU (0xFFFBC0A8) ;- (EMACB) Specific Address 3 Bottom, First 4 bytes
AT91C_EMACB_DTF           EQU (0xFFFBC058) ;- (EMACB) Deferred Transmission Frame Register
AT91C_EMACB_PTR           EQU (0xFFFBC038) ;- (EMACB) Pause Time Register
AT91C_EMACB_CSE           EQU (0xFFFBC068) ;- (EMACB) Carrier Sense Error Register
AT91C_EMACB_ECOL          EQU (0xFFFBC060) ;- (EMACB) Excessive Collision Register
AT91C_EMACB_STE           EQU (0xFFFBC084) ;- (EMACB) SQE Test Error Register
AT91C_EMACB_MCF           EQU (0xFFFBC048) ;- (EMACB) Multiple Collision Frame Register
AT91C_EMACB_IER           EQU (0xFFFBC028) ;- (EMACB) Interrupt Enable Register
AT91C_EMACB_ELE           EQU (0xFFFBC078) ;- (EMACB) Excessive Length Errors Register
AT91C_EMACB_USRIO         EQU (0xFFFBC0C0) ;- (EMACB) USER Input/Output Register
AT91C_EMACB_PFR           EQU (0xFFFBC03C) ;- (EMACB) Pause Frames received Register
AT91C_EMACB_FCSE          EQU (0xFFFBC050) ;- (EMACB) Frame Check Sequence Error Register
AT91C_EMACB_SA1L          EQU (0xFFFBC098) ;- (EMACB) Specific Address 1 Bottom, First 4 bytes
AT91C_EMACB_NCR           EQU (0xFFFBC000) ;- (EMACB) Network Control Register
AT91C_EMACB_HRT           EQU (0xFFFBC094) ;- (EMACB) Hash Address Top[63:32]
AT91C_EMACB_NCFGR         EQU (0xFFFBC004) ;- (EMACB) Network Configuration Register
AT91C_EMACB_SCF           EQU (0xFFFBC044) ;- (EMACB) Single Collision Frame Register
AT91C_EMACB_LCOL          EQU (0xFFFBC05C) ;- (EMACB) Late Collision Register
AT91C_EMACB_SA3H          EQU (0xFFFBC0AC) ;- (EMACB) Specific Address 3 Top, Last 2 bytes
AT91C_EMACB_HRB           EQU (0xFFFBC090) ;- (EMACB) Hash Address Bottom[31:0]
AT91C_EMACB_ISR           EQU (0xFFFBC024) ;- (EMACB) Interrupt Status Register
AT91C_EMACB_IMR           EQU (0xFFFBC030) ;- (EMACB) Interrupt Mask Register
AT91C_EMACB_WOL           EQU (0xFFFBC0C4) ;- (EMACB) Wake On LAN Register
AT91C_EMACB_USF           EQU (0xFFFBC080) ;- (EMACB) Undersize Frames Register
AT91C_EMACB_TSR           EQU (0xFFFBC014) ;- (EMACB) Transmit Status Register
AT91C_EMACB_FRO           EQU (0xFFFBC04C) ;- (EMACB) Frames Received OK Register
AT91C_EMACB_IDR           EQU (0xFFFBC02C) ;- (EMACB) Interrupt Disable Register
AT91C_EMACB_SA1H          EQU (0xFFFBC09C) ;- (EMACB) Specific Address 1 Top, Last 2 bytes
AT91C_EMACB_RLE           EQU (0xFFFBC088) ;- (EMACB) Receive Length Field Mismatch Register
AT91C_EMACB_TUND          EQU (0xFFFBC064) ;- (EMACB) Transmit Underrun Error Register
AT91C_EMACB_RSR           EQU (0xFFFBC020) ;- (EMACB) Receive Status Register
AT91C_EMACB_SA2L          EQU (0xFFFBC0A0) ;- (EMACB) Specific Address 2 Bottom, First 4 bytes
AT91C_EMACB_FTO           EQU (0xFFFBC040) ;- (EMACB) Frames Transmitted OK Register
AT91C_EMACB_RSE           EQU (0xFFFBC074) ;- (EMACB) Receive Symbol Errors Register
AT91C_EMACB_NSR           EQU (0xFFFBC008) ;- (EMACB) Network Status Register
AT91C_EMACB_RBQP          EQU (0xFFFBC018) ;- (EMACB) Receive Buffer Queue Pointer
AT91C_EMACB_REV           EQU (0xFFFBC0FC) ;- (EMACB) Revision Register
// - ========== Register definition for ISI peripheral ========== 
AT91C_ISI_DMACHDR         EQU (0xFFFB403C) ;- (ISI) DMA Channel Disable Register
AT91C_ISI_IMR             EQU (0xFFFB4034) ;- (ISI) Interrupt Mask Register
AT91C_ISI_R2YSET1         EQU (0xFFFB401C) ;- (ISI) Color Space Conversion RGB to YCrCb Register
AT91C_ISI_PDECF           EQU (0xFFFB400C) ;- (ISI) Preview Decimation Factor Register
AT91C_ISI_CFG2            EQU (0xFFFB4004) ;- (ISI) Configuration Register 2
AT91C_ISI_DMACCTRL        EQU (0xFFFB4054) ;- (ISI) DMA Codec Control Register
AT91C_ISI_CTRL            EQU (0xFFFB4024) ;- (ISI) Control Register
AT91C_ISI_Y2RSET0         EQU (0xFFFB4010) ;- (ISI) Color Space Conversion YCrCb to RGB Register
AT91C_ISI_WPSR            EQU (0xFFFB40E8) ;- (ISI) Write Protection Status Register
AT91C_ISI_DMACHER         EQU (0xFFFB4038) ;- (ISI) DMA Channel Enable Register
AT91C_ISI_DMACHSR         EQU (0xFFFB4040) ;- (ISI) DMA Channel Status Register
AT91C_ISI_IDR             EQU (0xFFFB4030) ;- (ISI) Interrupt Disable Register
AT91C_ISI_VER             EQU (0xFFFB40FC) ;- (ISI) Version Register
AT91C_ISI_Y2RSET1         EQU (0xFFFB4014) ;- (ISI) Color Space Conversion YCrCb to RGB Register
AT91C_ISI_R2YSET2         EQU (0xFFFB4020) ;- (ISI) Color Space Conversion RGB to YCrCb Register
AT91C_ISI_SR              EQU (0xFFFB4028) ;- (ISI) Status Register
AT91C_ISI_DMACDSCR        EQU (0xFFFB4058) ;- (ISI) DMA Codec Descriptor Address Register
AT91C_ISI_IER             EQU (0xFFFB402C) ;- (ISI) Interrupt Enable Register
AT91C_ISI_WPCR            EQU (0xFFFB40E4) ;- (ISI) Write Protection Control Register
AT91C_ISI_DMACADDR        EQU (0xFFFB4050) ;- (ISI) DMA Codec Base Address Register
AT91C_ISI_CFG1            EQU (0xFFFB4000) ;- (ISI) Configuration Register 1
AT91C_ISI_R2YSET0         EQU (0xFFFB4018) ;- (ISI) Color Space Conversion RGB to YCrCb Register
AT91C_ISI_PSIZE           EQU (0xFFFB4008) ;- (ISI) Preview Size Register
AT91C_ISI_DMAPDSCR        EQU (0xFFFB404C) ;- (ISI) DMA Preview Descriptor Address Register
AT91C_ISI_DMAPADDR        EQU (0xFFFB4044) ;- (ISI) DMA Preview Base Address Register
AT91C_ISI_DMAPCTRL        EQU (0xFFFB4048) ;- (ISI) DMA Preview Control Register
// - ========== Register definition for UHPHS_OHCI peripheral ========== 
AT91C_OHCI_HcRhPortStatus EQU (0x00700054) ;- (UHPHS_OHCI) Root Hub Port Status Register
AT91C_OHCI_HcFmRemaining  EQU (0x00700038) ;- (UHPHS_OHCI) Bit time remaining in the current Frame
AT91C_OHCI_HcInterruptEnable EQU (0x00700010) ;- (UHPHS_OHCI) Interrupt Enable Register
AT91C_OHCI_HcControl      EQU (0x00700004) ;- (UHPHS_OHCI) Operating modes for the Host Controller
AT91C_OHCI_HcPeriodicStart EQU (0x00700040) ;- (UHPHS_OHCI) Periodic Start
AT91C_OHCI_HcInterruptStatus EQU (0x0070000C) ;- (UHPHS_OHCI) Interrupt Status Register
AT91C_OHCI_HcRhDescriptorB EQU (0x0070004C) ;- (UHPHS_OHCI) Root Hub characteristics B
AT91C_OHCI_HcInterruptDisable EQU (0x00700014) ;- (UHPHS_OHCI) Interrupt Disable Register
AT91C_OHCI_HcPeriodCurrentED EQU (0x0070001C) ;- (UHPHS_OHCI) Current Isochronous or Interrupt Endpoint Descriptor
AT91C_OHCI_HcRhDescriptorA EQU (0x00700048) ;- (UHPHS_OHCI) Root Hub characteristics A
AT91C_OHCI_HcRhStatus     EQU (0x00700050) ;- (UHPHS_OHCI) Root Hub Status register
AT91C_OHCI_HcBulkCurrentED EQU (0x0070002C) ;- (UHPHS_OHCI) Current endpoint of the Bulk list
AT91C_OHCI_HcControlHeadED EQU (0x00700020) ;- (UHPHS_OHCI) First Endpoint Descriptor of the Control list
AT91C_OHCI_HcLSThreshold  EQU (0x00700044) ;- (UHPHS_OHCI) LS Threshold
AT91C_OHCI_HcRevision     EQU (0x00700000) ;- (UHPHS_OHCI) Revision
AT91C_OHCI_HcBulkDoneHead EQU (0x00700030) ;- (UHPHS_OHCI) Last completed transfer descriptor
AT91C_OHCI_HcFmNumber     EQU (0x0070003C) ;- (UHPHS_OHCI) Frame number
AT91C_OHCI_HcFmInterval   EQU (0x00700034) ;- (UHPHS_OHCI) Bit time between 2 consecutive SOFs
AT91C_OHCI_HcBulkHeadED   EQU (0x00700028) ;- (UHPHS_OHCI) First endpoint register of the Bulk list
AT91C_OHCI_HcHCCA         EQU (0x00700018) ;- (UHPHS_OHCI) Pointer to the Host Controller Communication Area
AT91C_OHCI_HcCommandStatus EQU (0x00700008) ;- (UHPHS_OHCI) Command & status Register
AT91C_OHCI_HcControlCurrentED EQU (0x00700024) ;- (UHPHS_OHCI) Endpoint Control and Status Register
// - ========== Register definition for UHPHS_EHCI peripheral ========== 
AT91C_EHCI_USBSTS         EQU (0x00800014) ;- (UHPHS_EHCI) 
AT91C_EHCI_INSNREG03      EQU (0x0080009C) ;- (UHPHS_EHCI) 
AT91C_EHCI_INSNREG00      EQU (0x00800090) ;- (UHPHS_EHCI) 
AT91C_EHCI_HCSPPORTROUTE  EQU (0x0080000C) ;- (UHPHS_EHCI) 
AT91C_EHCI_FRINDEX        EQU (0x0080001C) ;- (UHPHS_EHCI) 
AT91C_EHCI_CONFIGFLAG     EQU (0x00800050) ;- (UHPHS_EHCI) 
AT91C_EHCI_HCCPARAMS      EQU (0x00800008) ;- (UHPHS_EHCI) 
AT91C_EHCI_USBINTR        EQU (0x00800018) ;- (UHPHS_EHCI) 
AT91C_EHCI_PORTSC         EQU (0x00800054) ;- (UHPHS_EHCI) 
AT91C_EHCI_CTRLDSSEGMENT  EQU (0x00800020) ;- (UHPHS_EHCI) 
AT91C_EHCI_VERSION        EQU (0x00800000) ;- (UHPHS_EHCI) 
AT91C_EHCI_USBCMD         EQU (0x00800010) ;- (UHPHS_EHCI) 
AT91C_EHCI_INSNREG04      EQU (0x008000A0) ;- (UHPHS_EHCI) 
AT91C_EHCI_PERIODICLISTBASE EQU (0x00800024) ;- (UHPHS_EHCI) 
AT91C_EHCI_INSNREG01      EQU (0x00800094) ;- (UHPHS_EHCI) 
AT91C_EHCI_HCSPARAMS      EQU (0x00800004) ;- (UHPHS_EHCI) 
AT91C_EHCI_INSNREG05      EQU (0x008000A4) ;- (UHPHS_EHCI) 
AT91C_EHCI_ASYNCLISTADDR  EQU (0x00800028) ;- (UHPHS_EHCI) 
AT91C_EHCI_INSNREG02      EQU (0x00800098) ;- (UHPHS_EHCI) 
// - ========== Register definition for TRNG peripheral ========== 
AT91C_TRNG_IDR            EQU (0xFFFCC014) ;- (TRNG) Interrupt Disable Register
AT91C_TRNG_IER            EQU (0xFFFCC010) ;- (TRNG) Interrupt Enable Register
AT91C_TRNG_VERSION        EQU (0xFFFCC0FC) ;- (TRNG) TRNG Version Register
AT91C_TRNG_ISR            EQU (0xFFFCC01C) ;- (TRNG) Interrupt Status Register
AT91C_TRNG_CR             EQU (0xFFFCC000) ;- (TRNG) Control Register
AT91C_TRNG_ODATA          EQU (0xFFFCC050) ;- (TRNG) Output Data Register
AT91C_TRNG_IMR            EQU (0xFFFCC018) ;- (TRNG) Interrupt Mask Register

// - *****************************************************************************
// -               PIO DEFINITIONS FOR AT91SAM9G45
// - *****************************************************************************
AT91C_PIO_PA0             EQU (1 <<  0) ;- Pin Controlled by PA0
AT91C_PA0_MCI0_CK         EQU (AT91C_PIO_PA0) ;-  
AT91C_PA0_TCLK3           EQU (AT91C_PIO_PA0) ;-  
AT91C_PIO_PA1             EQU (1 <<  1) ;- Pin Controlled by PA1
AT91C_PA1_MCI0_CDA        EQU (AT91C_PIO_PA1) ;-  
AT91C_PA1_TIOA3           EQU (AT91C_PIO_PA1) ;-  
AT91C_PIO_PA10            EQU (1 << 10) ;- Pin Controlled by PA10
AT91C_PA10_ETX0           EQU (AT91C_PIO_PA10) ;-  Ethernet MAC Transmit Data 0
AT91C_PIO_PA11            EQU (1 << 11) ;- Pin Controlled by PA11
AT91C_PA11_ETX1           EQU (AT91C_PIO_PA11) ;-  Ethernet MAC Transmit Data 1
AT91C_PIO_PA12            EQU (1 << 12) ;- Pin Controlled by PA12
AT91C_PA12_ERX0           EQU (AT91C_PIO_PA12) ;-  Ethernet MAC Receive Data 0
AT91C_PIO_PA13            EQU (1 << 13) ;- Pin Controlled by PA13
AT91C_PA13_ERX1           EQU (AT91C_PIO_PA13) ;-  Ethernet MAC Receive Data 1
AT91C_PIO_PA14            EQU (1 << 14) ;- Pin Controlled by PA14
AT91C_PA14_ETXEN          EQU (AT91C_PIO_PA14) ;-  Ethernet MAC Transmit Enable
AT91C_PIO_PA15            EQU (1 << 15) ;- Pin Controlled by PA15
AT91C_PA15_ERXDV          EQU (AT91C_PIO_PA15) ;-  Ethernet MAC Receive Data Valid
AT91C_PIO_PA16            EQU (1 << 16) ;- Pin Controlled by PA16
AT91C_PA16_ERXER          EQU (AT91C_PIO_PA16) ;-  Ethernet MAC Receive Error
AT91C_PIO_PA17            EQU (1 << 17) ;- Pin Controlled by PA17
AT91C_PA17_ETXCK_EREFCK   EQU (AT91C_PIO_PA17) ;-  Ethernet MAC Transmit Clock/Reference Clock
AT91C_PIO_PA18            EQU (1 << 18) ;- Pin Controlled by PA18
AT91C_PA18_EMDC           EQU (AT91C_PIO_PA18) ;-  Ethernet MAC Management Data Clock
AT91C_PIO_PA19            EQU (1 << 19) ;- Pin Controlled by PA19
AT91C_PA19_EMDIO          EQU (AT91C_PIO_PA19) ;-  Ethernet MAC Management Data Input/Output
AT91C_PIO_PA2             EQU (1 <<  2) ;- Pin Controlled by PA2
AT91C_PA2_MCI0_DA0        EQU (AT91C_PIO_PA2) ;-  
AT91C_PA2_TIOB3           EQU (AT91C_PIO_PA2) ;-  
AT91C_PIO_PA20            EQU (1 << 20) ;- Pin Controlled by PA20
AT91C_PA20_TWD0           EQU (AT91C_PIO_PA20) ;-  TWI Two-wire Serial Data
AT91C_PIO_PA21            EQU (1 << 21) ;- Pin Controlled by PA21
AT91C_PA21_TWCK0          EQU (AT91C_PIO_PA21) ;-  TWI Two-wire Serial Clock
AT91C_PIO_PA22            EQU (1 << 22) ;- Pin Controlled by PA22
AT91C_PA22_MCI1_CDA       EQU (AT91C_PIO_PA22) ;-  
AT91C_PA22_SCK3           EQU (AT91C_PIO_PA22) ;-  
AT91C_PIO_PA23            EQU (1 << 23) ;- Pin Controlled by PA23
AT91C_PA23_MCI1_DA0       EQU (AT91C_PIO_PA23) ;-  
AT91C_PA23_RTS3           EQU (AT91C_PIO_PA23) ;-  
AT91C_PIO_PA24            EQU (1 << 24) ;- Pin Controlled by PA24
AT91C_PA24_MCI1_DA1       EQU (AT91C_PIO_PA24) ;-  
AT91C_PA24_CTS3           EQU (AT91C_PIO_PA24) ;-  
AT91C_PIO_PA25            EQU (1 << 25) ;- Pin Controlled by PA25
AT91C_PA25_MCI1_DA2       EQU (AT91C_PIO_PA25) ;-  
AT91C_PA25_PWM3           EQU (AT91C_PIO_PA25) ;-  
AT91C_PIO_PA26            EQU (1 << 26) ;- Pin Controlled by PA26
AT91C_PA26_MCI1_DA3       EQU (AT91C_PIO_PA26) ;-  
AT91C_PA26_TIOB2          EQU (AT91C_PIO_PA26) ;-  
AT91C_PIO_PA27            EQU (1 << 27) ;- Pin Controlled by PA27
AT91C_PA27_MCI1_DA4       EQU (AT91C_PIO_PA27) ;-  
AT91C_PA27_ETXER          EQU (AT91C_PIO_PA27) ;-  Ethernet MAC Transmikt Coding Error
AT91C_PIO_PA28            EQU (1 << 28) ;- Pin Controlled by PA28
AT91C_PA28_MCI1_DA5       EQU (AT91C_PIO_PA28) ;-  
AT91C_PA28_ERXCK          EQU (AT91C_PIO_PA28) ;-  Ethernet MAC Receive Clock
AT91C_PIO_PA29            EQU (1 << 29) ;- Pin Controlled by PA29
AT91C_PA29_MCI1_DA6       EQU (AT91C_PIO_PA29) ;-  
AT91C_PA29_ECRS           EQU (AT91C_PIO_PA29) ;-  Ethernet MAC Carrier Sense/Carrier Sense and Data Valid
AT91C_PIO_PA3             EQU (1 <<  3) ;- Pin Controlled by PA3
AT91C_PA3_MCI0_DA1        EQU (AT91C_PIO_PA3) ;-  
AT91C_PA3_TCLK4           EQU (AT91C_PIO_PA3) ;-  
AT91C_PIO_PA30            EQU (1 << 30) ;- Pin Controlled by PA30
AT91C_PA30_MCI1_DA7       EQU (AT91C_PIO_PA30) ;-  
AT91C_PA30_ECOL           EQU (AT91C_PIO_PA30) ;-  Ethernet MAC Collision Detected
AT91C_PIO_PA31            EQU (1 << 31) ;- Pin Controlled by PA31
AT91C_PA31_MCI1_CK        EQU (AT91C_PIO_PA31) ;-  
AT91C_PA31_PCK0           EQU (AT91C_PIO_PA31) ;-  
AT91C_PIO_PA4             EQU (1 <<  4) ;- Pin Controlled by PA4
AT91C_PA4_MCI0_DA2        EQU (AT91C_PIO_PA4) ;-  
AT91C_PA4_TIOA4           EQU (AT91C_PIO_PA4) ;-  
AT91C_PIO_PA5             EQU (1 <<  5) ;- Pin Controlled by PA5
AT91C_PA5_MCI0_DA3        EQU (AT91C_PIO_PA5) ;-  
AT91C_PA5_TIOB4           EQU (AT91C_PIO_PA5) ;-  
AT91C_PIO_PA6             EQU (1 <<  6) ;- Pin Controlled by PA6
AT91C_PA6_MCI0_DA4        EQU (AT91C_PIO_PA6) ;-  
AT91C_PA6_ETX2            EQU (AT91C_PIO_PA6) ;-  Ethernet MAC Transmit Data 2
AT91C_PIO_PA7             EQU (1 <<  7) ;- Pin Controlled by PA7
AT91C_PA7_MCI0_DA5        EQU (AT91C_PIO_PA7) ;-  
AT91C_PA7_ETX3            EQU (AT91C_PIO_PA7) ;-  Ethernet MAC Transmit Data 3
AT91C_PIO_PA8             EQU (1 <<  8) ;- Pin Controlled by PA8
AT91C_PA8_MCI0_DA6        EQU (AT91C_PIO_PA8) ;-  
AT91C_PA8_ERX2            EQU (AT91C_PIO_PA8) ;-  Ethernet MAC Receive Data 2
AT91C_PIO_PA9             EQU (1 <<  9) ;- Pin Controlled by PA9
AT91C_PA9_MCI0_DA7        EQU (AT91C_PIO_PA9) ;-  
AT91C_PA9_ERX3            EQU (AT91C_PIO_PA9) ;-  Ethernet MAC Receive Data 3
AT91C_PIO_PB0             EQU (1 <<  0) ;- Pin Controlled by PB0
AT91C_PB0_SPI0_MISO       EQU (AT91C_PIO_PB0) ;-  SPI 0 Master In Slave
AT91C_PIO_PB1             EQU (1 <<  1) ;- Pin Controlled by PB1
AT91C_PB1_SPI0_MOSI       EQU (AT91C_PIO_PB1) ;-  SPI 0 Master Out Slave
AT91C_PIO_PB10            EQU (1 << 10) ;- Pin Controlled by PB10
AT91C_PB10_TWD1           EQU (AT91C_PIO_PB10) ;-  
AT91C_PB10_ISI_D10        EQU (AT91C_PIO_PB10) ;-  
AT91C_PIO_PB11            EQU (1 << 11) ;- Pin Controlled by PB11
AT91C_PB11_TWCK1          EQU (AT91C_PIO_PB11) ;-  
AT91C_PB11_ISI_D11        EQU (AT91C_PIO_PB11) ;-  
AT91C_PIO_PB12            EQU (1 << 12) ;- Pin Controlled by PB12
AT91C_PB12_DRXD           EQU (AT91C_PIO_PB12) ;-  
AT91C_PIO_PB13            EQU (1 << 13) ;- Pin Controlled by PB13
AT91C_PB13_DTXD           EQU (AT91C_PIO_PB13) ;-  
AT91C_PIO_PB14            EQU (1 << 14) ;- Pin Controlled by PB14
AT91C_PB14_SPI1_MISO      EQU (AT91C_PIO_PB14) ;-  
AT91C_PIO_PB15            EQU (1 << 15) ;- Pin Controlled by PB15
AT91C_PB15_SPI1_MOSI      EQU (AT91C_PIO_PB15) ;-  
AT91C_PB15_CTS0           EQU (AT91C_PIO_PB15) ;-  
AT91C_PIO_PB16            EQU (1 << 16) ;- Pin Controlled by PB16
AT91C_PB16_SPI1_SPCK      EQU (AT91C_PIO_PB16) ;-  
AT91C_PB16_SCK0           EQU (AT91C_PIO_PB16) ;-  
AT91C_PIO_PB17            EQU (1 << 17) ;- Pin Controlled by PB17
AT91C_PB17_SPI1_NPCS0     EQU (AT91C_PIO_PB17) ;-  
AT91C_PB17_RTS0           EQU (AT91C_PIO_PB17) ;-  
AT91C_PIO_PB18            EQU (1 << 18) ;- Pin Controlled by PB18
AT91C_PB18_RXD0           EQU (AT91C_PIO_PB18) ;-  
AT91C_PB18_SPI0_NPCS1     EQU (AT91C_PIO_PB18) ;-  
AT91C_PIO_PB19            EQU (1 << 19) ;- Pin Controlled by PB19
AT91C_PB19_TXD0           EQU (AT91C_PIO_PB19) ;-  
AT91C_PB19_SPI0_NPCS2     EQU (AT91C_PIO_PB19) ;-  
AT91C_PIO_PB2             EQU (1 <<  2) ;- Pin Controlled by PB2
AT91C_PB2_SPI0_SPCK       EQU (AT91C_PIO_PB2) ;-  SPI 0 Serial Clock
AT91C_PIO_PB20            EQU (1 << 20) ;- Pin Controlled by PB20
AT91C_PB20_ISI_D0         EQU (AT91C_PIO_PB20) ;-  
AT91C_PIO_PB21            EQU (1 << 21) ;- Pin Controlled by PB21
AT91C_PB21_ISI_D1         EQU (AT91C_PIO_PB21) ;-  
AT91C_PIO_PB22            EQU (1 << 22) ;- Pin Controlled by PB22
AT91C_PB22_ISI_D2         EQU (AT91C_PIO_PB22) ;-  
AT91C_PIO_PB23            EQU (1 << 23) ;- Pin Controlled by PB23
AT91C_PB23_ISI_D3         EQU (AT91C_PIO_PB23) ;-  
AT91C_PIO_PB24            EQU (1 << 24) ;- Pin Controlled by PB24
AT91C_PB24_ISI_D4         EQU (AT91C_PIO_PB24) ;-  
AT91C_PIO_PB25            EQU (1 << 25) ;- Pin Controlled by PB25
AT91C_PB25_ISI_D5         EQU (AT91C_PIO_PB25) ;-  
AT91C_PIO_PB26            EQU (1 << 26) ;- Pin Controlled by PB26
AT91C_PB26_ISI_D6         EQU (AT91C_PIO_PB26) ;-  
AT91C_PIO_PB27            EQU (1 << 27) ;- Pin Controlled by PB27
AT91C_PB27_ISI_D7         EQU (AT91C_PIO_PB27) ;-  
AT91C_PIO_PB28            EQU (1 << 28) ;- Pin Controlled by PB28
AT91C_PB28_ISI_PCK        EQU (AT91C_PIO_PB28) ;-  
AT91C_PIO_PB29            EQU (1 << 29) ;- Pin Controlled by PB29
AT91C_PB29_ISI_VSYNC      EQU (AT91C_PIO_PB29) ;-  
AT91C_PIO_PB3             EQU (1 <<  3) ;- Pin Controlled by PB3
AT91C_PB3_SPI0_NPCS0      EQU (AT91C_PIO_PB3) ;-  SPI 0 Peripheral Chip Select 0
AT91C_PIO_PB30            EQU (1 << 30) ;- Pin Controlled by PB30
AT91C_PB30_ISI_HSYNC      EQU (AT91C_PIO_PB30) ;-  
AT91C_PIO_PB31            EQU (1 << 31) ;- Pin Controlled by PB31
AT91C_PB31_               EQU (AT91C_PIO_PB31) ;-  
AT91C_PB31_PCK1           EQU (AT91C_PIO_PB31) ;-  
AT91C_PIO_PB4             EQU (1 <<  4) ;- Pin Controlled by PB4
AT91C_PB4_TXD1            EQU (AT91C_PIO_PB4) ;-  USART 1 Transmit Data
AT91C_PIO_PB5             EQU (1 <<  5) ;- Pin Controlled by PB5
AT91C_PB5_RXD1            EQU (AT91C_PIO_PB5) ;-  USART 1 Receive Data
AT91C_PIO_PB6             EQU (1 <<  6) ;- Pin Controlled by PB6
AT91C_PB6_TXD2            EQU (AT91C_PIO_PB6) ;-  USART 2 Transmit Data
AT91C_PIO_PB7             EQU (1 <<  7) ;- Pin Controlled by PB7
AT91C_PB7_RXD2            EQU (AT91C_PIO_PB7) ;-  USART 2 Receive Data
AT91C_PIO_PB8             EQU (1 <<  8) ;- Pin Controlled by PB8
AT91C_PB8_TXD3            EQU (AT91C_PIO_PB8) ;-  USART 3 Transmit Data
AT91C_PB8_ISI_D8          EQU (AT91C_PIO_PB8) ;-  
AT91C_PIO_PB9             EQU (1 <<  9) ;- Pin Controlled by PB9
AT91C_PB9_RXD3            EQU (AT91C_PIO_PB9) ;-  USART 3 Receive Data
AT91C_PB9_ISI_D9          EQU (AT91C_PIO_PB9) ;-  
AT91C_PIO_PC0             EQU (1 <<  0) ;- Pin Controlled by PC0
AT91C_PC0_DQM2            EQU (AT91C_PIO_PC0) ;-  DQM2
AT91C_PIO_PC1             EQU (1 <<  1) ;- Pin Controlled by PC1
AT91C_PC1_DQM3            EQU (AT91C_PIO_PC1) ;-  DQM3
AT91C_PIO_PC10            EQU (1 << 10) ;- Pin Controlled by PC10
AT91C_PC10_NCS4_CFCS0     EQU (AT91C_PIO_PC10) ;-  
AT91C_PC10_TCLK2          EQU (AT91C_PIO_PC10) ;-  
AT91C_PIO_PC11            EQU (1 << 11) ;- Pin Controlled by PC11
AT91C_PC11_NCS5_CFCS1     EQU (AT91C_PIO_PC11) ;-  
AT91C_PC11_CTS2           EQU (AT91C_PIO_PC11) ;-  
AT91C_PIO_PC12            EQU (1 << 12) ;- Pin Controlled by PC12
AT91C_PC12_A25_CFRNW      EQU (AT91C_PIO_PC12) ;-  
AT91C_PIO_PC13            EQU (1 << 13) ;- Pin Controlled by PC13
AT91C_PC13_NCS2           EQU (AT91C_PIO_PC13) ;-  
AT91C_PIO_PC14            EQU (1 << 14) ;- Pin Controlled by PC14
AT91C_PC14_NCS3_NANDCS    EQU (AT91C_PIO_PC14) ;-  
AT91C_PIO_PC15            EQU (1 << 15) ;- Pin Controlled by PC15
AT91C_PC15_NWAIT          EQU (AT91C_PIO_PC15) ;-  
AT91C_PIO_PC16            EQU (1 << 16) ;- Pin Controlled by PC16
AT91C_PC16_D16            EQU (AT91C_PIO_PC16) ;-  
AT91C_PIO_PC17            EQU (1 << 17) ;- Pin Controlled by PC17
AT91C_PC17_D17            EQU (AT91C_PIO_PC17) ;-  
AT91C_PIO_PC18            EQU (1 << 18) ;- Pin Controlled by PC18
AT91C_PC18_D18            EQU (AT91C_PIO_PC18) ;-  
AT91C_PIO_PC19            EQU (1 << 19) ;- Pin Controlled by PC19
AT91C_PC19_D19            EQU (AT91C_PIO_PC19) ;-  
AT91C_PIO_PC2             EQU (1 <<  2) ;- Pin Controlled by PC2
AT91C_PC2_A19             EQU (AT91C_PIO_PC2) ;-  
AT91C_PIO_PC20            EQU (1 << 20) ;- Pin Controlled by PC20
AT91C_PC20_D20            EQU (AT91C_PIO_PC20) ;-  
AT91C_PIO_PC21            EQU (1 << 21) ;- Pin Controlled by PC21
AT91C_PC21_D21            EQU (AT91C_PIO_PC21) ;-  
AT91C_PIO_PC22            EQU (1 << 22) ;- Pin Controlled by PC22
AT91C_PC22_D22            EQU (AT91C_PIO_PC22) ;-  
AT91C_PIO_PC23            EQU (1 << 23) ;- Pin Controlled by PC23
AT91C_PC23_D23            EQU (AT91C_PIO_PC23) ;-  
AT91C_PIO_PC24            EQU (1 << 24) ;- Pin Controlled by PC24
AT91C_PC24_D24            EQU (AT91C_PIO_PC24) ;-  
AT91C_PIO_PC25            EQU (1 << 25) ;- Pin Controlled by PC25
AT91C_PC25_D25            EQU (AT91C_PIO_PC25) ;-  
AT91C_PIO_PC26            EQU (1 << 26) ;- Pin Controlled by PC26
AT91C_PC26_D26            EQU (AT91C_PIO_PC26) ;-  
AT91C_PIO_PC27            EQU (1 << 27) ;- Pin Controlled by PC27
AT91C_PC27_D27            EQU (AT91C_PIO_PC27) ;-  
AT91C_PIO_PC28            EQU (1 << 28) ;- Pin Controlled by PC28
AT91C_PC28_D28            EQU (AT91C_PIO_PC28) ;-  
AT91C_PIO_PC29            EQU (1 << 29) ;- Pin Controlled by PC29
AT91C_PC29_D29            EQU (AT91C_PIO_PC29) ;-  
AT91C_PIO_PC3             EQU (1 <<  3) ;- Pin Controlled by PC3
AT91C_PC3_A20             EQU (AT91C_PIO_PC3) ;-  
AT91C_PIO_PC30            EQU (1 << 30) ;- Pin Controlled by PC30
AT91C_PC30_D30            EQU (AT91C_PIO_PC30) ;-  
AT91C_PIO_PC31            EQU (1 << 31) ;- Pin Controlled by PC31
AT91C_PC31_D31            EQU (AT91C_PIO_PC31) ;-  
AT91C_PIO_PC4             EQU (1 <<  4) ;- Pin Controlled by PC4
AT91C_PC4_A21_NANDALE     EQU (AT91C_PIO_PC4) ;-  
AT91C_PIO_PC5             EQU (1 <<  5) ;- Pin Controlled by PC5
AT91C_PC5_A22_NANDCLE     EQU (AT91C_PIO_PC5) ;-  
AT91C_PIO_PC6             EQU (1 <<  6) ;- Pin Controlled by PC6
AT91C_PC6_A23             EQU (AT91C_PIO_PC6) ;-  
AT91C_PIO_PC7             EQU (1 <<  7) ;- Pin Controlled by PC7
AT91C_PC7_A24             EQU (AT91C_PIO_PC7) ;-  
AT91C_PIO_PC8             EQU (1 <<  8) ;- Pin Controlled by PC8
AT91C_PC8_CFCE1           EQU (AT91C_PIO_PC8) ;-  
AT91C_PIO_PC9             EQU (1 <<  9) ;- Pin Controlled by PC9
AT91C_PC9_CFCE2           EQU (AT91C_PIO_PC9) ;-  
AT91C_PC9_RTS2            EQU (AT91C_PIO_PC9) ;-  
AT91C_PIO_PD0             EQU (1 <<  0) ;- Pin Controlled by PD0
AT91C_PD0_TK0             EQU (AT91C_PIO_PD0) ;-  
AT91C_PD0_PWM3            EQU (AT91C_PIO_PD0) ;-  
AT91C_PIO_PD1             EQU (1 <<  1) ;- Pin Controlled by PD1
AT91C_PD1_TF0             EQU (AT91C_PIO_PD1) ;-  
AT91C_PIO_PD10            EQU (1 << 10) ;- Pin Controlled by PD10
AT91C_PD10_TD1            EQU (AT91C_PIO_PD10) ;-  
AT91C_PIO_PD11            EQU (1 << 11) ;- Pin Controlled by PD11
AT91C_PD11_RD1            EQU (AT91C_PIO_PD11) ;-  
AT91C_PIO_PD12            EQU (1 << 12) ;- Pin Controlled by PD12
AT91C_PD12_TK1            EQU (AT91C_PIO_PD12) ;-  
AT91C_PD12_PCK0           EQU (AT91C_PIO_PD12) ;-  
AT91C_PIO_PD13            EQU (1 << 13) ;- Pin Controlled by PD13
AT91C_PD13_RK1            EQU (AT91C_PIO_PD13) ;-  
AT91C_PIO_PD14            EQU (1 << 14) ;- Pin Controlled by PD14
AT91C_PD14_TF1            EQU (AT91C_PIO_PD14) ;-  
AT91C_PIO_PD15            EQU (1 << 15) ;- Pin Controlled by PD15
AT91C_PD15_RF1            EQU (AT91C_PIO_PD15) ;-  
AT91C_PIO_PD16            EQU (1 << 16) ;- Pin Controlled by PD16
AT91C_PD16_RTS1           EQU (AT91C_PIO_PD16) ;-  
AT91C_PIO_PD17            EQU (1 << 17) ;- Pin Controlled by PD17
AT91C_PD17_CTS1           EQU (AT91C_PIO_PD17) ;-  
AT91C_PIO_PD18            EQU (1 << 18) ;- Pin Controlled by PD18
AT91C_PD18_SPI1_NPCS2     EQU (AT91C_PIO_PD18) ;-  
AT91C_PD18_IRQ            EQU (AT91C_PIO_PD18) ;-  
AT91C_PIO_PD19            EQU (1 << 19) ;- Pin Controlled by PD19
AT91C_PD19_SPI1_NPCS3     EQU (AT91C_PIO_PD19) ;-  
AT91C_PD19_FIQ            EQU (AT91C_PIO_PD19) ;-  
AT91C_PIO_PD2             EQU (1 <<  2) ;- Pin Controlled by PD2
AT91C_PD2_TD0             EQU (AT91C_PIO_PD2) ;-  
AT91C_PIO_PD20            EQU (1 << 20) ;- Pin Controlled by PD20
AT91C_PD20_TIOA0          EQU (AT91C_PIO_PD20) ;-  
AT91C_PIO_PD21            EQU (1 << 21) ;- Pin Controlled by PD21
AT91C_PD21_TIOA1          EQU (AT91C_PIO_PD21) ;-  
AT91C_PIO_PD22            EQU (1 << 22) ;- Pin Controlled by PD22
AT91C_PD22_TIOA2          EQU (AT91C_PIO_PD22) ;-  
AT91C_PIO_PD23            EQU (1 << 23) ;- Pin Controlled by PD23
AT91C_PD23_TCLK0          EQU (AT91C_PIO_PD23) ;-  
AT91C_PIO_PD24            EQU (1 << 24) ;- Pin Controlled by PD24
AT91C_PD24_SPI0_NPCS1     EQU (AT91C_PIO_PD24) ;-  
AT91C_PD24_PWM0           EQU (AT91C_PIO_PD24) ;-  
AT91C_PIO_PD25            EQU (1 << 25) ;- Pin Controlled by PD25
AT91C_PD25_SPI0_NPCS2     EQU (AT91C_PIO_PD25) ;-  
AT91C_PD25_PWM1           EQU (AT91C_PIO_PD25) ;-  
AT91C_PIO_PD26            EQU (1 << 26) ;- Pin Controlled by PD26
AT91C_PD26_PCK0           EQU (AT91C_PIO_PD26) ;-  
AT91C_PD26_PWM2           EQU (AT91C_PIO_PD26) ;-  
AT91C_PIO_PD27            EQU (1 << 27) ;- Pin Controlled by PD27
AT91C_PD27_PCK1           EQU (AT91C_PIO_PD27) ;-  
AT91C_PD27_SPI0_NPCS3     EQU (AT91C_PIO_PD27) ;-  
AT91C_PIO_PD28            EQU (1 << 28) ;- Pin Controlled by PD28
AT91C_PD28_TSADTRG        EQU (AT91C_PIO_PD28) ;-  
AT91C_PD28_SPI1_NPCS1     EQU (AT91C_PIO_PD28) ;-  
AT91C_PIO_PD29            EQU (1 << 29) ;- Pin Controlled by PD29
AT91C_PD29_TCLK1          EQU (AT91C_PIO_PD29) ;-  
AT91C_PD29_SCK1           EQU (AT91C_PIO_PD29) ;-  
AT91C_PIO_PD3             EQU (1 <<  3) ;- Pin Controlled by PD3
AT91C_PD3_RD0             EQU (AT91C_PIO_PD3) ;-  
AT91C_PIO_PD30            EQU (1 << 30) ;- Pin Controlled by PD30
AT91C_PD30_TIOB0          EQU (AT91C_PIO_PD30) ;-  
AT91C_PD30_SCK2           EQU (AT91C_PIO_PD30) ;-  
AT91C_PIO_PD31            EQU (1 << 31) ;- Pin Controlled by PD31
AT91C_PD31_TIOB1          EQU (AT91C_PIO_PD31) ;-  
AT91C_PD31_PWM1           EQU (AT91C_PIO_PD31) ;-  
AT91C_PIO_PD4             EQU (1 <<  4) ;- Pin Controlled by PD4
AT91C_PD4_RK0             EQU (AT91C_PIO_PD4) ;-  
AT91C_PIO_PD5             EQU (1 <<  5) ;- Pin Controlled by PD5
AT91C_PD5_RF0             EQU (AT91C_PIO_PD5) ;-  
AT91C_PIO_PD6             EQU (1 <<  6) ;- Pin Controlled by PD6
AT91C_PD6_AC97RX          EQU (AT91C_PIO_PD6) ;-  
AT91C_PIO_PD7             EQU (1 <<  7) ;- Pin Controlled by PD7
AT91C_PD7_AC97TX          EQU (AT91C_PIO_PD7) ;-  
AT91C_PD7_TIOA5           EQU (AT91C_PIO_PD7) ;-  
AT91C_PIO_PD8             EQU (1 <<  8) ;- Pin Controlled by PD8
AT91C_PD8_AC97FS          EQU (AT91C_PIO_PD8) ;-  
AT91C_PD8_TIOB5           EQU (AT91C_PIO_PD8) ;-  
AT91C_PIO_PD9             EQU (1 <<  9) ;- Pin Controlled by PD9
AT91C_PD9_AC97CK          EQU (AT91C_PIO_PD9) ;-  
AT91C_PD9_TCLK5           EQU (AT91C_PIO_PD9) ;-  
AT91C_PIO_PE0             EQU (1 <<  0) ;- Pin Controlled by PE0
AT91C_PE0_LCDPWR          EQU (AT91C_PIO_PE0) ;-  
AT91C_PE0_PCK0            EQU (AT91C_PIO_PE0) ;-  
AT91C_PIO_PE1             EQU (1 <<  1) ;- Pin Controlled by PE1
AT91C_PE1_LCDMOD          EQU (AT91C_PIO_PE1) ;-  
AT91C_PIO_PE10            EQU (1 << 10) ;- Pin Controlled by PE10
AT91C_PE10_LCDD3          EQU (AT91C_PIO_PE10) ;-  
AT91C_PE10_LCDD5          EQU (AT91C_PIO_PE10) ;-  
AT91C_PIO_PE11            EQU (1 << 11) ;- Pin Controlled by PE11
AT91C_PE11_LCDD4          EQU (AT91C_PIO_PE11) ;-  
AT91C_PE11_LCDD6          EQU (AT91C_PIO_PE11) ;-  
AT91C_PIO_PE12            EQU (1 << 12) ;- Pin Controlled by PE12
AT91C_PE12_LCDD5          EQU (AT91C_PIO_PE12) ;-  
AT91C_PE12_LCDD7          EQU (AT91C_PIO_PE12) ;-  
AT91C_PIO_PE13            EQU (1 << 13) ;- Pin Controlled by PE13
AT91C_PE13_LCDD6          EQU (AT91C_PIO_PE13) ;-  
AT91C_PE13_LCDD10         EQU (AT91C_PIO_PE13) ;-  
AT91C_PIO_PE14            EQU (1 << 14) ;- Pin Controlled by PE14
AT91C_PE14_LCDD7          EQU (AT91C_PIO_PE14) ;-  
AT91C_PE14_LCDD11         EQU (AT91C_PIO_PE14) ;-  
AT91C_PIO_PE15            EQU (1 << 15) ;- Pin Controlled by PE15
AT91C_PE15_LCDD8          EQU (AT91C_PIO_PE15) ;-  
AT91C_PE15_LCDD12         EQU (AT91C_PIO_PE15) ;-  
AT91C_PIO_PE16            EQU (1 << 16) ;- Pin Controlled by PE16
AT91C_PE16_LCDD9          EQU (AT91C_PIO_PE16) ;-  
AT91C_PE16_LCDD13         EQU (AT91C_PIO_PE16) ;-  
AT91C_PIO_PE17            EQU (1 << 17) ;- Pin Controlled by PE17
AT91C_PE17_LCDD10         EQU (AT91C_PIO_PE17) ;-  
AT91C_PE17_LCDD14         EQU (AT91C_PIO_PE17) ;-  
AT91C_PIO_PE18            EQU (1 << 18) ;- Pin Controlled by PE18
AT91C_PE18_LCDD11         EQU (AT91C_PIO_PE18) ;-  
AT91C_PE18_LCDD15         EQU (AT91C_PIO_PE18) ;-  
AT91C_PIO_PE19            EQU (1 << 19) ;- Pin Controlled by PE19
AT91C_PE19_LCDD12         EQU (AT91C_PIO_PE19) ;-  
AT91C_PE19_LCDD18         EQU (AT91C_PIO_PE19) ;-  
AT91C_PIO_PE2             EQU (1 <<  2) ;- Pin Controlled by PE2
AT91C_PE2_LCDCC           EQU (AT91C_PIO_PE2) ;-  
AT91C_PIO_PE20            EQU (1 << 20) ;- Pin Controlled by PE20
AT91C_PE20_LCDD13         EQU (AT91C_PIO_PE20) ;-  
AT91C_PE20_LCDD19         EQU (AT91C_PIO_PE20) ;-  
AT91C_PIO_PE21            EQU (1 << 21) ;- Pin Controlled by PE21
AT91C_PE21_LCDD14         EQU (AT91C_PIO_PE21) ;-  
AT91C_PE21_LCDD20         EQU (AT91C_PIO_PE21) ;-  
AT91C_PIO_PE22            EQU (1 << 22) ;- Pin Controlled by PE22
AT91C_PE22_LCDD15         EQU (AT91C_PIO_PE22) ;-  
AT91C_PE22_LCDD21         EQU (AT91C_PIO_PE22) ;-  
AT91C_PIO_PE23            EQU (1 << 23) ;- Pin Controlled by PE23
AT91C_PE23_LCDD16         EQU (AT91C_PIO_PE23) ;-  
AT91C_PE23_LCDD22         EQU (AT91C_PIO_PE23) ;-  
AT91C_PIO_PE24            EQU (1 << 24) ;- Pin Controlled by PE24
AT91C_PE24_LCDD17         EQU (AT91C_PIO_PE24) ;-  
AT91C_PE24_LCDD23         EQU (AT91C_PIO_PE24) ;-  
AT91C_PIO_PE25            EQU (1 << 25) ;- Pin Controlled by PE25
AT91C_PE25_LCDD18         EQU (AT91C_PIO_PE25) ;-  
AT91C_PIO_PE26            EQU (1 << 26) ;- Pin Controlled by PE26
AT91C_PE26_LCDD19         EQU (AT91C_PIO_PE26) ;-  
AT91C_PIO_PE27            EQU (1 << 27) ;- Pin Controlled by PE27
AT91C_PE27_LCDD20         EQU (AT91C_PIO_PE27) ;-  
AT91C_PIO_PE28            EQU (1 << 28) ;- Pin Controlled by PE28
AT91C_PE28_LCDD21         EQU (AT91C_PIO_PE28) ;-  
AT91C_PIO_PE29            EQU (1 << 29) ;- Pin Controlled by PE29
AT91C_PE29_LCDD22         EQU (AT91C_PIO_PE29) ;-  
AT91C_PIO_PE3             EQU (1 <<  3) ;- Pin Controlled by PE3
AT91C_PE3_LCDVSYNC        EQU (AT91C_PIO_PE3) ;-  
AT91C_PIO_PE30            EQU (1 << 30) ;- Pin Controlled by PE30
AT91C_PE30_LCDD23         EQU (AT91C_PIO_PE30) ;-  
AT91C_PIO_PE31            EQU (1 << 31) ;- Pin Controlled by PE31
AT91C_PE31_PWM2           EQU (AT91C_PIO_PE31) ;-  
AT91C_PE31_PCK1           EQU (AT91C_PIO_PE31) ;-  
AT91C_PIO_PE4             EQU (1 <<  4) ;- Pin Controlled by PE4
AT91C_PE4_LCDHSYNC        EQU (AT91C_PIO_PE4) ;-  
AT91C_PIO_PE5             EQU (1 <<  5) ;- Pin Controlled by PE5
AT91C_PE5_LCDDOTCK        EQU (AT91C_PIO_PE5) ;-  
AT91C_PIO_PE6             EQU (1 <<  6) ;- Pin Controlled by PE6
AT91C_PE6_LCDDEN          EQU (AT91C_PIO_PE6) ;-  
AT91C_PIO_PE7             EQU (1 <<  7) ;- Pin Controlled by PE7
AT91C_PE7_LCDD0           EQU (AT91C_PIO_PE7) ;-  
AT91C_PE7_LCDD2           EQU (AT91C_PIO_PE7) ;-  
AT91C_PIO_PE8             EQU (1 <<  8) ;- Pin Controlled by PE8
AT91C_PE8_LCDD1           EQU (AT91C_PIO_PE8) ;-  
AT91C_PE8_LCDD3           EQU (AT91C_PIO_PE8) ;-  
AT91C_PIO_PE9             EQU (1 <<  9) ;- Pin Controlled by PE9
AT91C_PE9_LCDD2           EQU (AT91C_PIO_PE9) ;-  
AT91C_PE9_LCDD4           EQU (AT91C_PIO_PE9) ;-  

// - *****************************************************************************
// -               PERIPHERAL ID DEFINITIONS FOR AT91SAM9G45
// - *****************************************************************************
AT91C_ID_FIQ              EQU ( 0) ;- Advanced Interrupt Controller (FIQ)
AT91C_ID_SYS              EQU ( 1) ;- System Controller
AT91C_ID_PIOA             EQU ( 2) ;- Parallel IO Controller A
AT91C_ID_PIOB             EQU ( 3) ;- Parallel IO Controller B
AT91C_ID_PIOC             EQU ( 4) ;- Parallel IO Controller C
AT91C_ID_PIOD_E           EQU ( 5) ;- Parallel IO Controller D and E
AT91C_ID_TRNG             EQU ( 6) ;- True Random Number Generator
AT91C_ID_US0              EQU ( 7) ;- USART 0
AT91C_ID_US1              EQU ( 8) ;- USART 1
AT91C_ID_US2              EQU ( 9) ;- USART 2
AT91C_ID_US3              EQU (10) ;- USART 2
AT91C_ID_MCI0             EQU (11) ;- Multimedia Card Interface 0
AT91C_ID_TWI0             EQU (12) ;- TWI 0
AT91C_ID_TWI1             EQU (13) ;- TWI 1
AT91C_ID_SPI0             EQU (14) ;- Serial Peripheral Interface
AT91C_ID_SPI1             EQU (15) ;- Serial Peripheral Interface
AT91C_ID_SSC0             EQU (16) ;- Serial Synchronous Controller 0
AT91C_ID_SSC1             EQU (17) ;- Serial Synchronous Controller 1
AT91C_ID_TC               EQU (18) ;- Timer Counter 0, 1, 2, 3, 4, 5
AT91C_ID_PWMC             EQU (19) ;- Pulse Width Modulation Controller
AT91C_ID_TSADC            EQU (20) ;- Touch Screen Controller
AT91C_ID_HDMA             EQU (21) ;- HDMA
AT91C_ID_UHPHS            EQU (22) ;- USB Host High Speed
AT91C_ID_LCDC             EQU (23) ;- LCD Controller
AT91C_ID_AC97C            EQU (24) ;- AC97 Controller
AT91C_ID_EMAC             EQU (25) ;- Ethernet MAC
AT91C_ID_ISI              EQU (26) ;- Image Sensor Interface
AT91C_ID_UDPHS            EQU (27) ;- USB Device HS
AT91C_ID_MCI1             EQU (29) ;- Multimedia Card Interface 1
AT91C_ID_VDEC             EQU (30) ;- Video Decoder
AT91C_ID_IRQ0             EQU (31) ;- Advanced Interrupt Controller (IRQ0)
AT91C_ALL_INT             EQU (0xEFFFFFFF) ;- ALL VALID INTERRUPTS

// - *****************************************************************************
// -               BASE ADDRESS DEFINITIONS FOR AT91SAM9G45
// - *****************************************************************************
AT91C_BASE_SFR            EQU (0xFFF74000) ;- (SFR) Base Address
AT91C_BASE_SYS            EQU (0xFFFFC000) ;- (SYS) Base Address
AT91C_BASE_EBI            EQU (0xFFFFE200) ;- (EBI) Base Address
AT91C_BASE_DDR2CP1        EQU (0xFFFFE400) ;- (DDR2CP1) Base Address
AT91C_BASE_DDR2C          EQU (0xFFFFE600) ;- (DDR2C) Base Address
AT91C_BASE_SMC            EQU (0xFFFFE800) ;- (SMC) Base Address
AT91C_BASE_MATRIX         EQU (0xFFFFEA00) ;- (MATRIX) Base Address
AT91C_BASE_AIC            EQU (0xFFFFF000) ;- (AIC) Base Address
AT91C_BASE_PDC_DBGU       EQU (0xFFFFEF00) ;- (PDC_DBGU) Base Address
AT91C_BASE_DBGU           EQU (0xFFFFEE00) ;- (DBGU) Base Address
AT91C_BASE_PIOA           EQU (0xFFFFF200) ;- (PIOA) Base Address
AT91C_BASE_PIOB           EQU (0xFFFFF400) ;- (PIOB) Base Address
AT91C_BASE_PIOC           EQU (0xFFFFF600) ;- (PIOC) Base Address
AT91C_BASE_PIOD           EQU (0xFFFFF800) ;- (PIOD) Base Address
AT91C_BASE_PIOE           EQU (0xFFFFFA00) ;- (PIOE) Base Address
AT91C_BASE_PMC            EQU (0xFFFFFC00) ;- (PMC) Base Address
AT91C_BASE_CKGR           EQU (0xFFFFFC1C) ;- (CKGR) Base Address
AT91C_BASE_RSTC           EQU (0xFFFFFD00) ;- (RSTC) Base Address
AT91C_BASE_SHDWC          EQU (0xFFFFFD10) ;- (SHDWC) Base Address
AT91C_BASE_RTTC           EQU (0xFFFFFD20) ;- (RTTC) Base Address
AT91C_BASE_PITC           EQU (0xFFFFFD30) ;- (PITC) Base Address
AT91C_BASE_WDTC           EQU (0xFFFFFD40) ;- (WDTC) Base Address
AT91C_BASE_RTC            EQU (0xFFFFFDB0) ;- (RTC) Base Address
AT91C_BASE_TC0            EQU (0xFFF7C000) ;- (TC0) Base Address
AT91C_BASE_TC1            EQU (0xFFF7C040) ;- (TC1) Base Address
AT91C_BASE_TC2            EQU (0xFFF7C080) ;- (TC2) Base Address
AT91C_BASE_TC3            EQU (0xFFFD4000) ;- (TC3) Base Address
AT91C_BASE_TC4            EQU (0xFFFD4040) ;- (TC4) Base Address
AT91C_BASE_TC5            EQU (0xFFFD4080) ;- (TC5) Base Address
AT91C_BASE_TCB0           EQU (0xFFF7C000) ;- (TCB0) Base Address
AT91C_BASE_TCB1           EQU (0xFFFD4000) ;- (TCB1) Base Address
AT91C_BASE_MCI0           EQU (0xFFF80000) ;- (MCI0) Base Address
AT91C_BASE_MCI1           EQU (0xFFFD0000) ;- (MCI1) Base Address
AT91C_BASE_TWI0           EQU (0xFFF84000) ;- (TWI0) Base Address
AT91C_BASE_TWI1           EQU (0xFFF88000) ;- (TWI1) Base Address
AT91C_BASE_PDC_US0        EQU (0xFFF8C100) ;- (PDC_US0) Base Address
AT91C_BASE_US0            EQU (0xFFF8C000) ;- (US0) Base Address
AT91C_BASE_PDC_US1        EQU (0xFFF90100) ;- (PDC_US1) Base Address
AT91C_BASE_US1            EQU (0xFFF90000) ;- (US1) Base Address
AT91C_BASE_PDC_US2        EQU (0xFFF94100) ;- (PDC_US2) Base Address
AT91C_BASE_US2            EQU (0xFFF94000) ;- (US2) Base Address
AT91C_BASE_PDC_US3        EQU (0xFFF98100) ;- (PDC_US3) Base Address
AT91C_BASE_US3            EQU (0xFFF98000) ;- (US3) Base Address
AT91C_BASE_PDC_SSC0       EQU (0xFFF9C100) ;- (PDC_SSC0) Base Address
AT91C_BASE_SSC0           EQU (0xFFF9C000) ;- (SSC0) Base Address
AT91C_BASE_PDC_SSC1       EQU (0xFFFA0100) ;- (PDC_SSC1) Base Address
AT91C_BASE_SSC1           EQU (0xFFFA0000) ;- (SSC1) Base Address
AT91C_BASE_PWMC_CH0       EQU (0xFFFB8200) ;- (PWMC_CH0) Base Address
AT91C_BASE_PWMC_CH1       EQU (0xFFFB8220) ;- (PWMC_CH1) Base Address
AT91C_BASE_PWMC_CH2       EQU (0xFFFB8240) ;- (PWMC_CH2) Base Address
AT91C_BASE_PWMC_CH3       EQU (0xFFFB8260) ;- (PWMC_CH3) Base Address
AT91C_BASE_PWMC           EQU (0xFFFB8000) ;- (PWMC) Base Address
AT91C_BASE_PDC_SPI0       EQU (0xFFFA4100) ;- (PDC_SPI0) Base Address
AT91C_BASE_PDC_SPI1       EQU (0xFFFA8100) ;- (PDC_SPI1) Base Address
AT91C_BASE_SPI0           EQU (0xFFFA4000) ;- (SPI0) Base Address
AT91C_BASE_SPI1           EQU (0xFFFA8000) ;- (SPI1) Base Address
AT91C_BASE_PDC_TSADC      EQU (0xFFFB0100) ;- (PDC_TSADC) Base Address
AT91C_BASE_TSADC          EQU (0xFFFB0000) ;- (TSADC) Base Address
AT91C_BASE_UDPHS_EPTFIFO  EQU (0x00600000) ;- (UDPHS_EPTFIFO) Base Address
AT91C_BASE_UDPHS_EPT_0    EQU (0xFFF78100) ;- (UDPHS_EPT_0) Base Address
AT91C_BASE_UDPHS_EPT_1    EQU (0xFFF78120) ;- (UDPHS_EPT_1) Base Address
AT91C_BASE_UDPHS_EPT_2    EQU (0xFFF78140) ;- (UDPHS_EPT_2) Base Address
AT91C_BASE_UDPHS_EPT_3    EQU (0xFFF78160) ;- (UDPHS_EPT_3) Base Address
AT91C_BASE_UDPHS_EPT_4    EQU (0xFFF78180) ;- (UDPHS_EPT_4) Base Address
AT91C_BASE_UDPHS_EPT_5    EQU (0xFFF781A0) ;- (UDPHS_EPT_5) Base Address
AT91C_BASE_UDPHS_EPT_6    EQU (0xFFF781C0) ;- (UDPHS_EPT_6) Base Address
AT91C_BASE_UDPHS_DMA_1    EQU (0xFFF78310) ;- (UDPHS_DMA_1) Base Address
AT91C_BASE_UDPHS_DMA_2    EQU (0xFFF78320) ;- (UDPHS_DMA_2) Base Address
AT91C_BASE_UDPHS_DMA_3    EQU (0xFFF78330) ;- (UDPHS_DMA_3) Base Address
AT91C_BASE_UDPHS_DMA_4    EQU (0xFFF78340) ;- (UDPHS_DMA_4) Base Address
AT91C_BASE_UDPHS_DMA_5    EQU (0xFFF78350) ;- (UDPHS_DMA_5) Base Address
AT91C_BASE_UDPHS_DMA_6    EQU (0xFFF78360) ;- (UDPHS_DMA_6) Base Address
AT91C_BASE_UDPHS          EQU (0xFFF78000) ;- (UDPHS) Base Address
AT91C_BASE_PDC_AC97C      EQU (0xFFFAC100) ;- (PDC_AC97C) Base Address
AT91C_BASE_AC97C          EQU (0xFFFAC000) ;- (AC97C) Base Address
AT91C_BASE_LCDC           EQU (0x00500000) ;- (LCDC) Base Address
AT91C_BASE_LCDC_16B_TFT   EQU (0x00500000) ;- (LCDC_16B_TFT) Base Address
AT91C_BASE_HDMA_CH_0      EQU (0xFFFFEC3C) ;- (HDMA_CH_0) Base Address
AT91C_BASE_HDMA_CH_1      EQU (0xFFFFEC64) ;- (HDMA_CH_1) Base Address
AT91C_BASE_HDMA_CH_2      EQU (0xFFFFEC8C) ;- (HDMA_CH_2) Base Address
AT91C_BASE_HDMA_CH_3      EQU (0xFFFFECB4) ;- (HDMA_CH_3) Base Address
AT91C_BASE_HDMA_CH_4      EQU (0xFFFFECDC) ;- (HDMA_CH_4) Base Address
AT91C_BASE_HDMA_CH_5      EQU (0xFFFFED04) ;- (HDMA_CH_5) Base Address
AT91C_BASE_HDMA_CH_6      EQU (0xFFFFED2C) ;- (HDMA_CH_6) Base Address
AT91C_BASE_HDMA_CH_7      EQU (0xFFFFED54) ;- (HDMA_CH_7) Base Address
AT91C_BASE_HDMA           EQU (0xFFFFEC00) ;- (HDMA) Base Address
AT91C_BASE_HECC           EQU (0xFFFFE200) ;- (HECC) Base Address
AT91C_BASE_EMACB          EQU (0xFFFBC000) ;- (EMACB) Base Address
AT91C_BASE_ISI            EQU (0xFFFB4000) ;- (ISI) Base Address
AT91C_BASE_UHPHS_OHCI     EQU (0x00700000) ;- (UHPHS_OHCI) Base Address
AT91C_BASE_UHPHS_EHCI     EQU (0x00800000) ;- (UHPHS_EHCI) Base Address
AT91C_BASE_TRNG           EQU (0xFFFCC000) ;- (TRNG) Base Address

// - *****************************************************************************
// -               MEMORY MAPPING DEFINITIONS FOR AT91SAM9G45
// - *****************************************************************************
// - ITCM
AT91C_ITCM                EQU (0x00100000) ;- Maximum ITCM Area base address
// - DTCM
AT91C_DTCM                EQU (0x00200000) ;- Maximum DTCM Area base address
// - IRAM
AT91C_IRAM                EQU (0x00300000) ;- Maximum Internal SRAM base address
AT91C_IRAM_SIZE           EQU (0x00010000) ;- Maximum Internal SRAM size in byte (64 Kbytes)
// - IRAM_MIN
AT91C_IRAM_MIN            EQU (0x00300000) ;- Minimum Internal RAM base address
AT91C_IRAM_MIN_SIZE       EQU (0x00004000) ;- Minimum Internal RAM size in byte (16 Kbytes)
// - IROM
AT91C_IROM                EQU (0x00400000) ;- Internal ROM base address
AT91C_IROM_SIZE           EQU (0x00010000) ;- Internal ROM size in byte (64 Kbytes)
// - EBI_CS0
AT91C_EBI_CS0             EQU (0x10000000) ;- EBI Chip Select 0 base address
AT91C_EBI_CS0_SIZE        EQU (0x10000000) ;- EBI Chip Select 0 size in byte (262144 Kbytes)
// - EBI_CS1
AT91C_EBI_CS1             EQU (0x20000000) ;- EBI Chip Select 1 base address
AT91C_EBI_CS1_SIZE        EQU (0x10000000) ;- EBI Chip Select 1 size in byte (262144 Kbytes)
// - EBI_SDRAM
AT91C_EBI_SDRAM           EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_SIZE      EQU (0x10000000) ;- SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
// - EBI_SDRAM_16BIT
AT91C_EBI_SDRAM_16BIT     EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_16BIT_SIZE EQU (0x02000000) ;- SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
// - EBI_SDRAM_32BIT
AT91C_EBI_SDRAM_32BIT     EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_32BIT_SIZE EQU (0x04000000) ;- SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
// - EBI_CS2
AT91C_EBI_CS2             EQU (0x30000000) ;- EBI Chip Select 2 base address
AT91C_EBI_CS2_SIZE        EQU (0x10000000) ;- EBI Chip Select 2 size in byte (262144 Kbytes)
// - EBI_CS3
AT91C_EBI_CS3             EQU (0x40000000) ;- EBI Chip Select 3 base address
AT91C_EBI_CS3_SIZE        EQU (0x10000000) ;- EBI Chip Select 3 size in byte (262144 Kbytes)
// - EBI_SM
AT91C_EBI_SM              EQU (0x40000000) ;- NANDFLASH on EBI Chip Select 3 base address
AT91C_EBI_SM_SIZE         EQU (0x10000000) ;- NANDFLASH on EBI Chip Select 3 size in byte (262144 Kbytes)
// - EBI_CS4
AT91C_EBI_CS4             EQU (0x50000000) ;- EBI Chip Select 4 base address
AT91C_EBI_CS4_SIZE        EQU (0x10000000) ;- EBI Chip Select 4 size in byte (262144 Kbytes)
// - EBI_CF0
AT91C_EBI_CF0             EQU (0x50000000) ;- CompactFlash 0 on EBI Chip Select 4 base address
AT91C_EBI_CF0_SIZE        EQU (0x10000000) ;- CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
// - EBI_CS5
AT91C_EBI_CS5             EQU (0x60000000) ;- EBI Chip Select 5 base address
AT91C_EBI_CS5_SIZE        EQU (0x10000000) ;- EBI Chip Select 5 size in byte (262144 Kbytes)
// - EBI_CF1
AT91C_EBI_CF1             EQU (0x60000000) ;- CompactFlash 1 on EBIChip Select 5 base address
AT91C_EBI_CF1_SIZE        EQU (0x10000000) ;- CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
// - DDR2
AT91C_DDR2                EQU (0x70000000) ;- DDR2/LPDDR space base address
AT91C_DDR2_SIZE           EQU (0x10000000) ;- DDR2/LPDDR space size in byte (262144 Kbytes)
#endif /* __IAR_SYSTEMS_ASM__ */


#endif /* AT91SAM9G45_H */
