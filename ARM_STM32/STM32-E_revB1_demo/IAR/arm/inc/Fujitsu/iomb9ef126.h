/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Fujitsu MB9EF126
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 52703 $
 **
 ***************************************************************************/

#ifndef __IOMB9EF126_H
#define __IOMB9EF126_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   MB9EF126 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/


/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 0
#error This file should only be compiled in little endian mode
#endif

/* RUN Profile Power Domain Configuration Register (SYSC_RUNPDCFGR) */
typedef struct {
  __REG16 PD2ON           : 1;
  __REG16 PD3ON           : 1;
  __REG16 PD4ON           : 1;
  __REG16 PD5ON           : 1;
  __REG16                 :12;
} __sysc_runpdcfgr_bits;

/* RUN Profile Clock Source Enable Register (SYSC_RUNCKSRER) */
typedef struct {
  __REG16 SRCOSCEN        : 1;
  __REG16 RCOSCEN         : 1;
  __REG16 MOSCEN          : 1;
  __REG16 SOSCEN          : 1;
  __REG16 MAINPLLEN       : 1;
  __REG16 SSCGPLLEN       : 1;
  __REG16 GFXPLLEN        : 1;
  __REG16                 : 9;
} __sysc_runcksrer_bits;

/* RUN Profile Clock Select Register (SYSC_RUNCKSELR) */
typedef struct {
  __REG32 SYSCSL          : 3;
  __REG32                 : 1;
  __REG32 PERI0CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI1CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI3CSL        : 3;
  __REG32                 : 9;
  __REG32 PIXCSL          : 2;
  __REG32                 : 2;
  __REG32 SPICSL          : 2;
  __REG32                 : 2;
} __sysc_runckselr_bits;

/* RUN Profile Clock Enable Register (SYSC_RUNCKER) */
typedef struct {
  __REG32 ENCFGPD1        : 1;
  __REG32                 : 3;
  __REG32 ENHPMPD2        : 1;
  __REG32 ENTRACEPD2      : 1;
  __REG32 ENDMAPD2        : 1;
  __REG32 ENPERI0PD2      : 1;
  __REG32 ENPERI1PD2      : 1;
  __REG32                 : 1;
  __REG32 ENPERI3PD2      : 1;
  __REG32 ENPERI4PD2      : 1;
  __REG32                 : 4;
  __REG32 ENSYSPD3        : 1;
  __REG32 ENTRACEPD3      : 1;
  __REG32 ENHPMPD3        : 1;
  __REG32 ENMEMEPD3       : 1;
  __REG32 ENSPIPD3        : 1;
  __REG32 ENEXTBUSPD3     : 1;
  __REG32                 : 2;
  __REG32 ENCFGPD4        : 1;
  __REG32                 : 3;
  __REG32 ENGFXPD5        : 1;
  __REG32 ENPIXPD5        : 1;
  __REG32 ENSPIPD5        : 1;
  __REG32                 : 1;
} __sysc_runcker_bits;

/* RUN Profile Clock Divider Register 0 (SYSC_RUNCKDIVR0) */
typedef struct {
  __REG32 SYSDIV          : 4;
  __REG32                 : 4;
  __REG32 TRACEDIV        : 2;
  __REG32                 : 6;
  __REG32 HPMDIV          : 2;
  __REG32                 : 6;
  __REG32 CFGDIV          : 2;
  __REG32                 : 6;
} __sysc_runckdivr0_bits;

/* RUN Profile Clock Divider Register 1 (SYSC_RUNCKDIVR1) */
typedef struct {
  __REG32 EXTBUSDIV       : 3;
  __REG32                 : 5;
  __REG32 MEMEDIV         : 2;
  __REG32                 : 6;
  __REG32 PERI4DIV        : 2;
  __REG32                 : 6;
  __REG32 GFXDIV          : 2;
  __REG32                 : 6;
} __sysc_runckdivr1_bits;

/* RUN Profile Clock Divider Register 2 (SYSC_RUNCKDIVR2) */
typedef struct {
  __REG32 PERI0DIV        : 4;
  __REG32                 : 4;
  __REG32 PERI1DIV        : 4;
  __REG32                 :12;
  __REG32 PERI3DIV        : 4;
  __REG32                 : 4;
} __sysc_runckdivr2_bits;

/* RUN Profile PLL Control Register (SYSC_RUNPLLCNTR) */
typedef struct {
  __REG32 PLLDIVL         : 2;
  __REG32                 : 6;
  __REG32 PLLDIVM         : 4;
  __REG32                 : 4;
  __REG32 PLLDIVN         : 7;
  __REG32                 : 9;
} __sysc_runpllcntr_bits;

/* RUN Profile SSCG Control Register 0 (SYSC_RUNSSCGCNTR0) */
typedef struct {
  __REG32 SSCGDIVL        : 2;
  __REG32                 : 6;
  __REG32 SSCGDIVM        : 4;
  __REG32                 : 4;
  __REG32 SSCGDIVN        : 6;
  __REG32                 : 2;
  __REG32 SSCGDIVP        : 5;
  __REG32                 : 3;
} __sysc_runsscgcntr0_bits;

/* RUN Profile SSCG Control Register 1 (SYSC_RUNSSCGCNTR1) */
typedef struct {
  __REG32 SSCGRATE        :10;
  __REG32                 : 6;
  __REG32 SSCGMODE        : 1;
  __REG32 SSCGFREQ        : 2;
  __REG32                 : 5;
  __REG32 SSCGSSEN        : 1;
  __REG32                 : 7;
} __sysc_runsscgcntr1_bits;

/* RUN Profile Graphics PLL Control Register 0 (SYSC_RUNGFXCNTR0) */
typedef struct {
  __REG32 GFXDIVL         : 2;
  __REG32                 :14;
  __REG32 GFXDIVN         : 6;
  __REG32                 : 2;
  __REG32 GFXDIVP         : 5;
  __REG32                 : 3;
} __sysc_rungfxcntr0_bits;

/* RUN Profile Graphics PLL Control Register 1 (SYSC_RUNGFXCNTR1) */
typedef struct {
  __REG32 GFXRATE         :10;
  __REG32                 : 6;
  __REG32 GFXMODE         : 1;
  __REG32 GFXFREQ         : 2;
  __REG32                 : 5;
  __REG32 GFXSSEN         : 1;
  __REG32                 : 7;
} __sysc_rungfxcntr1_bits;

/* RUN Profile Low Voltage Detector Configuration Register (SYSC_RUNLVDCFGR) */
typedef struct {
  __REG32 LVDE50          : 1;
  __REG32 LVDR50          : 1;
  __REG32 SV50            : 3;
  __REG32                 : 3;
  __REG32 LVDE33          : 1;
  __REG32 LVDR33          : 1;
  __REG32 SV33            : 3;
  __REG32                 : 3;
  __REG32 LVDE12          : 1;
  __REG32 LVDR12          : 1;
  __REG32 SV12            : 3;
  __REG32                 :11;
} __sysc_runlvdcfgr_bits;

/* RUN Profile Clock Supervisor Configuration Register (SYSC_RUNCSVCFGR) */
typedef struct {
  __REG16 MOCSVE          : 1;
  __REG16 SOCSVE          : 1;
  __REG16 MPCSVE          : 1;
  __REG16 SPCSVE          : 1;
  __REG16 GPCSVE          : 1;
  __REG16                 :11;
} __sysc_runcsvcfgr_bits;

/* Trigger RUN Control Register (SYSC_TRGRUNCNTR) */
typedef struct {
  __REG16 APPLYRUN        : 8;
  __REG16                 : 8;
} __sysc_trgruncntr_bits;

/* PSS Profile Power Domain Configuration Register (SYSC_PSSPDCFGR) */
typedef struct {
  __REG16 PD2ON           : 1;
  __REG16 PD3ON           : 1;
  __REG16 PD4ON           : 1;
  __REG16 PD5ON           : 1;
  __REG16                 :12;
} __sysc_psspdcfgr_bits;

/* PSS Profile Clock Source Enable Register (SYSC_PSSCKSRER) */
typedef struct {
  __REG16 SRCOSCEN        : 1;
  __REG16 RCOSCEN         : 1;
  __REG16 MOSCEN          : 1;
  __REG16 SOSCEN          : 1;
  __REG16 MAINPLLEN       : 1;
  __REG16 SSCGPLLEN       : 1;
  __REG16 GFXPLLEN        : 1;
  __REG16                 : 9;
} __sysc_psscksrer_bits;

/* PSS Profile Clock Select Register (SYSC_PSSCKSELR) */
typedef struct {
  __REG32 SYSCSL          : 3;
  __REG32                 : 1;
  __REG32 PERI0CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI1CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI3CSL        : 3;
  __REG32                 : 9;
  __REG32 PIXCSL          : 2;
  __REG32                 : 2;
  __REG32 SPICSL          : 2;
  __REG32                 : 2;
} __sysc_pssckselr_bits;

/* PSS Profile Clock Enable Register (SYSC_PSSCKER) */
typedef struct {
  __REG32 ENCFGPD1        : 1;
  __REG32                 : 3;
  __REG32 ENHPMPD2        : 1;
  __REG32 ENTRACEPD2      : 1;
  __REG32 ENDMAPD2        : 1;
  __REG32 ENPERI0PD2      : 1;
  __REG32 ENPERI1PD2      : 1;
  __REG32                 : 1;
  __REG32 ENPERI3PD2      : 1;
  __REG32 ENPERI4PD2      : 1;
  __REG32                 : 4;
  __REG32 ENSYSPD3        : 1;
  __REG32 ENTRACEPD3      : 1;
  __REG32 ENHPMPD3        : 1;
  __REG32 ENMEMEPD3       : 1;
  __REG32 ENSPIPD3        : 1;
  __REG32 ENEXTBUSPD3     : 1;
  __REG32                 : 2;
  __REG32 ENCFGPD4        : 1;
  __REG32                 : 3;
  __REG32 ENGFXPD5        : 1;
  __REG32 ENPIXPD5        : 1;
  __REG32 ENSPIPD5        : 1;
  __REG32                 : 1;
} __sysc_psscker_bits;

/* PSS Profile Clock Divider Register 0 (SYSC_PSSCKDIVR0) */
typedef struct {
  __REG32 SYSDIV          : 4;
  __REG32                 : 4;
  __REG32 TRACEDIV        : 2;
  __REG32                 : 6;
  __REG32 HPMDIV          : 2;
  __REG32                 : 6;
  __REG32 CFGDIV          : 2;
  __REG32                 : 6;
} __sysc_pssckdivr0_bits;

/* PSS Profile Clock Divider Register 1 (SYSC_PSSCKDIVR1) */
typedef struct {
  __REG32 EXTBUSDIV       : 3;
  __REG32                 : 5;
  __REG32 MEMEDIV         : 2;
  __REG32                 : 6;
  __REG32 PERI4DIV        : 2;
  __REG32                 : 6;
  __REG32 GFXDIV          : 2;
  __REG32                 : 6;
} __sysc_pssckdivr1_bits;

/* PSS Profile Clock Divider Register 2 (SYSC_PSSCKDIVR2) */
typedef struct {
  __REG32 PERI0DIV        : 4;
  __REG32                 : 4;
  __REG32 PERI1DIV        : 4;
  __REG32                 :12;
  __REG32 PERI3DIV        : 4;
  __REG32                 : 4;
} __sysc_pssckdivr2_bits;

/* PSS Profile PLL Control Register (SYSC_PSSPLLCNTR) */
typedef struct {
  __REG32 PLLDIVL         : 2;
  __REG32                 : 6;
  __REG32 PLLDIVM         : 4;
  __REG32                 : 4;
  __REG32 PLLDIVN         : 7;
  __REG32                 : 9;
} __sysc_psspllcntr_bits;

/* PSS Profile SSCG Control Register 0 (SYSC_PSSSSCGCNTR0) */
typedef struct {
  __REG32 SSCGDIVL        : 2;
  __REG32                 : 6;
  __REG32 SSCGDIVM        : 4;
  __REG32                 : 4;
  __REG32 SSCGDIVN        : 6;
  __REG32                 : 2;
  __REG32 SSCGDIVP        : 5;
  __REG32                 : 3;
} __sysc_psssscgcntr0_bits;

/* PSS Profile SSCG Control Register 1 (SYSC_PSSSSCGCNTR1) */
typedef struct {
  __REG32 SSCGRATE        :10;
  __REG32                 : 6;
  __REG32 SSCGMODE        : 1;
  __REG32 SSCGFREQ        : 2;
  __REG32                 : 5;
  __REG32 SSCGSSEN        : 1;
  __REG32                 : 7;
} __sysc_psssscgcntr1_bits;

/* PSS Profile Graphics PLL Control Register 0 (SYSC_PSSGFXCNTR0) */
typedef struct {
  __REG32 GFXDIVL         : 2;
  __REG32                 :14;
  __REG32 GFXDIVN         : 6;
  __REG32                 : 2;
  __REG32 GFXDIVP         : 5;
  __REG32                 : 3;
} __sysc_pssgfxcntr0_bits;

/* PSS Profile Graphics PLL Control Register 1 (SYSC_PSSGFXCNTR1) */
typedef struct {
  __REG32 GFXRATE         :10;
  __REG32                 : 6;
  __REG32 GFXMODE         : 1;
  __REG32 GFXFREQ         : 2;
  __REG32                 : 5;
  __REG32 GFXSSEN         : 1;
  __REG32                 : 7;
} __sysc_pssgfxcntr1_bits;

/* PSS Profile Low Voltage Detector Configuration Register (SYSC_PSSLVDCFGR) */
typedef struct {
  __REG32 LVDE50          : 1;
  __REG32 LVDR50          : 1;
  __REG32 SV50            : 3;
  __REG32                 : 3;
  __REG32 LVDE33          : 1;
  __REG32 LVDR33          : 1;
  __REG32 SV33            : 3;
  __REG32                 : 3;
  __REG32 LVDE12          : 1;
  __REG32 LVDR12          : 1;
  __REG32 SV12            : 3;
  __REG32                 :11;
} __sysc_psslvdcfgr_bits;

/* PSS Profile Clock Supervisor Configuration Register (SYSC_PSSCSVCFGR) */
typedef struct {
  __REG16 MOCSVE          : 1;
  __REG16 SOCSVE          : 1;
  __REG16 MPCSVE          : 1;
  __REG16 SPCSVE          : 1;
  __REG16 GPCSVE          : 1;
  __REG16                 :11;
} __sysc_psscsvcfgr_bits;

/* PSS Enable Register (SYSC_PSSENR) */
typedef struct {
  __REG16 PSSEN           : 8;
  __REG16                 : 8;
} __sysc_pssenr_bits;

/* APPLIED Profile Power Domain Configuration Register (SYSC_APPPDCFGR) */
typedef struct {
  __REG16 PD2ON           : 1;
  __REG16 PD3ON           : 1;
  __REG16 PD4ON           : 1;
  __REG16 PD5ON           : 1;
  __REG16                 :12;
} __sysc_apppdcfgr_bits;

/* APPLIED Profile Clock Source Enable Register (SYSC_APPCKSRER) */
typedef struct {
  __REG16 SRCOSCEN        : 1;
  __REG16 RCOSCEN         : 1;
  __REG16 MOSCEN          : 1;
  __REG16 SOSCEN          : 1;
  __REG16 MAINPLLEN       : 1;
  __REG16 SSCGPLLEN       : 1;
  __REG16 GFXPLLEN        : 1;
  __REG16                 : 9;
} __sysc_appcksrer_bits;

/* APPLIED Profile Clock Select Register (SYSC_APPCKSELR) */
typedef struct {
  __REG32 SYSCSL          : 3;
  __REG32                 : 1;
  __REG32 PERI0CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI1CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI3CSL        : 3;
  __REG32                 : 9;
  __REG32 PIXCSL          : 2;
  __REG32                 : 2;
  __REG32 SPICSL          : 2;
  __REG32                 : 2;
} __sysc_appckselr_bits;

/* APPLIED Profile Clock Enable Register (SYSC_APPCKER) */
typedef struct {
  __REG32 ENCFGPD1        : 1;
  __REG32                 : 3;
  __REG32 ENHPMPD2        : 1;
  __REG32 ENTRACEPD2      : 1;
  __REG32 ENDMAPD2        : 1;
  __REG32 ENPERI0PD2      : 1;
  __REG32 ENPERI1PD2      : 1;
  __REG32                 : 1;
  __REG32 ENPERI3PD2      : 1;
  __REG32 ENPERI4PD2      : 1;
  __REG32                 : 4;
  __REG32 ENSYSPD3        : 1;
  __REG32 ENTRACEPD3      : 1;
  __REG32 ENHPMPD3        : 1;
  __REG32 ENMEMEPD3       : 1;
  __REG32 ENSPIPD3        : 1;
  __REG32 ENEXTBUSPD3     : 1;
  __REG32                 : 2;
  __REG32 ENCFGPD4        : 1;
  __REG32                 : 3;
  __REG32 ENGFXPD5        : 1;
  __REG32 ENPIXPD5        : 1;
  __REG32 ENSPIPD5        : 1;
  __REG32                 : 1;
} __sysc_appcker_bits;

/* APPLIED Profile Clock Divider Register 0 (SYSC_APPCKDIVR0) */
typedef struct {
  __REG32 SYSDIV          : 4;
  __REG32                 : 4;
  __REG32 TRACEDIV        : 2;
  __REG32                 : 6;
  __REG32 HPMDIV          : 2;
  __REG32                 : 6;
  __REG32 CFGDIV          : 2;
  __REG32                 : 6;
} __sysc_appckdivr0_bits;

/* APPLIED Profile Clock Divider Register 1 (SYSC_APPCKDIVR1) */
typedef struct {
  __REG32 EXTBUSDIV       : 3;
  __REG32                 : 5;
  __REG32 MEMEDIV         : 2;
  __REG32                 : 6;
  __REG32 PERI4DIV        : 2;
  __REG32                 : 6;
  __REG32 GFXDIV          : 2;
  __REG32                 : 6;
} __sysc_appckdivr1_bits;

/* APPLIED Profile Clock Divider Register 2 (SYSC_APPCKDIVR2) */
typedef struct {
  __REG32 PERI0DIV        : 4;
  __REG32                 : 4;
  __REG32 PERI1DIV        : 4;
  __REG32                 :12;
  __REG32 PERI3DIV        : 4;
  __REG32                 : 4;
} __sysc_appckdivr2_bits;

/* APPLIED Profile PLL Control Register (SYSC_APPPLLCNTR) */
typedef struct {
  __REG32 PLLDIVL         : 2;
  __REG32                 : 6;
  __REG32 PLLDIVM         : 4;
  __REG32                 : 4;
  __REG32 PLLDIVN         : 7;
  __REG32                 : 9;
} __sysc_apppllcntr_bits;

/* APPLIED Profile SSCG Control Register 0 (SYSC_APPSSCGCNTR0) */
typedef struct {
  __REG32 SSCGDIVL        : 2;
  __REG32                 : 6;
  __REG32 SSCGDIVM        : 4;
  __REG32                 : 4;
  __REG32 SSCGDIVN        : 6;
  __REG32                 : 2;
  __REG32 SSCGDIVP        : 5;
  __REG32                 : 3;
} __sysc_appsscgcntr0_bits;

/* APPLIED Profile SSCG Control Register 1 (SYSC_APPSSCGCNTR1) */
typedef struct {
  __REG32 SSCGRATE        :10;
  __REG32                 : 6;
  __REG32 SSCGMODE        : 1;
  __REG32 SSCGFREQ        : 2;
  __REG32                 : 5;
  __REG32 SSCGSSEN        : 1;
  __REG32                 : 7;
} __sysc_appsscgcntr1_bits;

/* APPLIED Profile Graphics PLL Control Register 0 (SYSC_APPGFXCNTR0) */
typedef struct {
  __REG32 GFXDIVL         : 2;
  __REG32                 :14;
  __REG32 GFXDIVN         : 6;
  __REG32                 : 2;
  __REG32 GFXDIVP         : 5;
  __REG32                 : 3;
} __sysc_appgfxcntr0_bits;

/* APPLIED Profile Graphics PLL Control Register 1 (SYSC_APPGFXCNTR1) */
typedef struct {
  __REG32 GFXRATE         :10;
  __REG32                 : 6;
  __REG32 GFXMODE         : 1;
  __REG32 GFXFREQ         : 2;
  __REG32                 : 5;
  __REG32 GFXSSEN         : 1;
  __REG32                 : 7;
} __sysc_appgfxcntr1_bits;

/* APPLIED Profile Low Voltage Detector Configuration Register (SYSC_APPLVDCFGR) */
typedef struct {
  __REG32 LVDE50          : 1;
  __REG32 LVDR50          : 1;
  __REG32 SV50            : 3;
  __REG32                 : 3;
  __REG32 LVDE33          : 1;
  __REG32 LVDR33          : 1;
  __REG32 LV33            : 3;
  __REG32                 : 3;
  __REG32 LVDE12          : 1;
  __REG32 LVDR12          : 1;
  __REG32 SV12            : 3;
  __REG32                 :11;
} __sysc_applvdcfgr_bits;

/* APPLIED Profile Clock Supervisor Configuration Register (SYSC_APPCSVCFGR) */
typedef struct {
  __REG16 MOCSVE          : 1;
  __REG16 SOCSVE          : 1;
  __REG16 MPCSVE          : 1;
  __REG16 SPCSVE          : 1;
  __REG16 GPCSVE          : 1;
  __REG16                 :11;
} __sysc_appcsvcfgr_bits;

/* Power Domain Status Register (SYSC_PDSTSR) */
typedef struct {
  __REG16 PD2ON           : 1;
  __REG16 PD3ON           : 1;
  __REG16 PD4ON           : 1;
  __REG16 PD5ON           : 1;
  __REG16                 :12;
} __sysc_pdstsr_bits;

/* Clock Source Enable Status Register (SYSC_CKSRESTSR) */
typedef struct {
  __REG16 SRCOSCEN        : 1;
  __REG16 RCOSCEN         : 1;
  __REG16 MOSCEN          : 1;
  __REG16 SOSCEN          : 1;
  __REG16 MAINPLLEN       : 1;
  __REG16 SSCGPLLEN       : 1;
  __REG16 GFXPLLEN        : 1;
  __REG16                 : 1;
  __REG16 SRCCLKRDY       : 1;
  __REG16 RCCLKRDY        : 1;
  __REG16 MAINCLKRDY      : 1;
  __REG16 SUBCLKRDY       : 1;
  __REG16 MAINPLLRDY      : 1;
  __REG16 SSCGPLLRDY      : 1;
  __REG16 GFXPLLRDY       : 1;
  __REG16                 : 1;
} __sysc_cksrestsr_bits;

/* Clock Select Status Register (SYSC_CKSELSTSR) */
typedef struct {
  __REG32 SYSCSL          : 3;
  __REG32                 : 1;
  __REG32 PERI0CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI1CSL        : 3;
  __REG32                 : 1;
  __REG32 PERI3CSL        : 3;
  __REG32                 : 9;
  __REG32 PIXCSL          : 2;
  __REG32                 : 2;
  __REG32 SPICSL          : 2;
  __REG32                 : 2;
} __sysc_ckselstsr_bits;

/* Clock Enable Status Register (SYSC_CKESTSR) */
typedef struct {
  __REG32 ENCFGPD1        : 1;
  __REG32                 : 3;
  __REG32 ENHPMPD2        : 1;
  __REG32 ENTRACEPD2      : 1;
  __REG32 ENDMAPD2        : 1;
  __REG32 ENPERI0PD2      : 1;
  __REG32 ENPERI1PD2      : 1;
  __REG32                 : 1;
  __REG32 ENPERI3PD2      : 1;
  __REG32 ENPERI4PD2      : 1;
  __REG32                 : 4;
  __REG32 ENSYSPD3        : 1;
  __REG32 ENTRACEPD3      : 1;
  __REG32 ENHPMPD3        : 1;
  __REG32 ENMEMEPD3       : 1;
  __REG32 ENSPIPD3        : 1;
  __REG32 ENEXTBUSPD3     : 1;
  __REG32                 : 2;
  __REG32 ENCFGPD4        : 1;
  __REG32                 : 3;
  __REG32 ENGFXPD5        : 1;
  __REG32 ENPIXPD5        : 1;
  __REG32 ENSPIPD5        : 1;
  __REG32                 : 1;
} __sysc_ckestsr_bits;

/* Clock Divider Register 0 (SYSC_CKDIVSTSR0) */
typedef struct {
  __REG32 SYSDIV          : 4;
  __REG32                 : 4;
  __REG32 TRACEDIV        : 2;
  __REG32                 : 6;
  __REG32 HPMDIV          : 2;
  __REG32                 : 6;
  __REG32 CFGDIV          : 2;
  __REG32                 : 6;
} __sysc_ckdivstsr0_bits;

/* Clock Divider Register 1 (SYSC_CKDIVSTSR1) */
typedef struct {
  __REG32 EXTBUSDIV       : 3;
  __REG32                 : 5;
  __REG32 MEMEDIV         : 2;
  __REG32                 : 6;
  __REG32 PERI4DIV        : 2;
  __REG32                 : 6;
  __REG32 GFXDIV          : 2;
  __REG32                 : 6;
} __sysc_ckdivstsr1_bits;

/* Clock Divider Register 1 (SYSC_CKDIVSTSR1) */
typedef struct {
  __REG32 PERI0DIV        : 4;
  __REG32                 : 4;
  __REG32 PERI1DIV        : 4;
  __REG32                 :12;
  __REG32 PERI3DIV        : 4;
  __REG32                 : 4;
} __sysc_ckdivstsr2_bits;

/* PLL Status Register (SYSC_PLLSTSR) */
typedef struct {
  __REG32 PLLDIVL         : 2;
  __REG32                 : 6;
  __REG32 PLLDIVM         : 4;
  __REG32                 : 4;
  __REG32 PLLDIVN         : 7;
  __REG32                 : 9;
} __sysc_pllstsr_bits;

/* SSCG PLL Status Register 0 (SYSC_SSCGSTSR0) */
typedef struct {
  __REG32 SSCGDIVL        : 2;
  __REG32                 : 6;
  __REG32 SSCGDIVM        : 4;
  __REG32                 : 4;
  __REG32 SSCGDIVN        : 6;
  __REG32                 : 2;
  __REG32 SSCGDIVP        : 5;
  __REG32                 : 3;
} __sysc_sscgstsr0_bits;

/* SSCG PLL Status Register 1 (SYSC_SSCGSTSR1) */
typedef struct {
  __REG32 SSCGRATE        :10;
  __REG32                 : 6;
  __REG32 SSCGMODE        : 1;
  __REG32 SSCGFREQ        : 2;
  __REG32                 : 5;
  __REG32 SSCGSSEN        : 1;
  __REG32                 : 7;
} __sysc_sscgstsr1_bits;

/* Graphics PLL Status Register 0 (SYSC_GFXSTSR0) */
typedef struct {
  __REG32 GFXDIVL         : 2;
  __REG32                 :14;
  __REG32 GFXDIVN         : 6;
  __REG32                 : 2;
  __REG32 GFXDIVP         : 5;
  __REG32                 : 3;
} __sysc_gfxstsr0_bits;

/* Graphics PLL Status Register 1 (SYSC_GFXSTSR1) */
typedef struct {
  __REG32 GFXRATE         :10;
  __REG32                 : 6;
  __REG32 GFXMODE         : 1;
  __REG32 GFXFREQ         : 2;
  __REG32                 : 5;
  __REG32 GFXSSEN         : 1;
  __REG32                 : 7;
} __sysc_gfxstsr1_bits;

/* LVD Configuration Status Register (SYSC_LVDCFGSTSR) */
typedef struct {
  __REG32 LVDE50          : 1;
  __REG32 LVDR50          : 1;
  __REG32 SV50            : 3;
  __REG32 LVDRDY50        : 1;
  __REG32                 : 2;
  __REG32 LVDE33          : 1;
  __REG32 LVDR33          : 1;
  __REG32 SV33            : 3;
  __REG32 LVDRDY33        : 1;
  __REG32                 : 2;
  __REG32 LVDE12          : 1;
  __REG32 LVDR12          : 1;
  __REG32 SV12            : 3;
  __REG32 LVDRDY12        : 1;
  __REG32                 :10;
} __sysc_lvdcfgstsr_bits;

/* CSV Configuration Status Register (SYSC_CSVCFGSTSR) */
typedef struct {
  __REG16 MOCSVE          : 1;
  __REG16 SOCSVE          : 1;
  __REG16 MPCSVE          : 1;
  __REG16 SPCSVE          : 1;
  __REG16 GPCSVE          : 1;
  __REG16                 :11;
} __sysc_csvcfgstsr_bits;

/* System Status Register (SYSC_SYSSTSR) */
typedef struct {
  __REG32 IRPARUN         : 1;
  __REG32 IRPAPSS         : 1;
  __REG32 IPPAPSS         : 1;
  __REG32                 : 5;
  __REG32 RUNDN           : 1;
  __REG32 PSSDN           : 1;
  __REG32 DEVSTAT         : 1;
  __REG32                 : 5;
  __REG32 RUNBUSY         : 1;
  __REG32 PSSBUSY         : 1;
  __REG32                 :14;
} __sysc_sysstsr_bits;

/* System Status Register (SYSC_SYSSTSR) */
typedef struct {
  __REG32                 : 8;
  __REG32 RUNDNIE         : 1;
  __REG32                 :23;
} __sysc_sysinter_bits;

/* System Status Interrupt Clear Register (SYSC_SYSICLR) */
typedef struct {
  __REG32                 : 8;
  __REG32 RUNDNICLR       : 1;
  __REG32 PSSDNICLR       : 1;
  __REG32                 :22;
} __sysc_sysiclr_bits;

/* System Error Register (SYSC_SYSERRR) */
typedef struct {
  __REG32 RUNERRIF        : 1;
  __REG32 RUNWKERRIF      : 1;
  __REG32 PSSERRIF        : 1;
  __REG32                 : 5;
  __REG32 MOMISS          : 1;
  __REG32 SOMISS          : 1;
  __REG32 MPMISS          : 1;
  __REG32 SPMISS          : 1;
  __REG32 GPMISS          : 1;
  __REG32                 : 3;
  __REG32 TRGERRIF        : 1;
  __REG32                 : 3;
  __REG32 RUNTRGEIF       : 1;
  __REG32 PSSENEIF        : 1;
  __REG32                 : 2;
  __REG32 LVD50IF         : 1;
  __REG32 LVD33IF         : 1;
  __REG32 LVD12IF         : 1;
  __REG32                 : 5;
} __sysc_syserrr_bits;

/* System Error Interrupt Clear Register (SYSC_SYSERRICLR) */
typedef struct {
  __REG32 RUNERRICLR      : 1;
  __REG32 RUNWKERRICLR    : 1;
  __REG32 PSSERRICLR      : 1;
  __REG32                 : 5;
  __REG32 MOMISSICLR      : 1;
  __REG32 SOMISSICLR      : 1;
  __REG32 MPMISSICLR      : 1;
  __REG32 SPMISSICLR      : 1;
  __REG32 GPMISSICLR      : 1;
  __REG32                 : 3;
  __REG32 TRGERRICLR      : 1;
  __REG32                 : 3;
  __REG32 RUNTRGEICLR     : 1;
  __REG32 PSSENEICLR      : 1;
  __REG32                 : 2;
  __REG32 LVD50ICLR       : 1;
  __REG32 LVD33ICLR       : 1;
  __REG32 LVD12ICLR       : 1;
  __REG32                 : 5;
} __sysc_syserriclr_bits;

/* Clock Supervisor Configuration for Main Clock Register (SYSC_CSVMOCFGR) */
/* Clock Supervisor Configuration for Sub Clock Register (SYSC_CSVSOCFGR) */
/* Clock Supervisor Configuration for Main PLL Clock Register (SYSC_CSVMPCFGR) */
/* Clock Supervisor Configuration for SSCG-PLL Clock Register (SYSC_CSVSPCFGR) */
/* Clock Supervisor Configuration for Graphics PLL Clock Register (SYSC_CSVGPCFGR) */
typedef struct {
  __REG32 REFCLKWND       : 8;
  __REG32 LOWTHR          : 8;
  __REG32 UPTHR           : 8;
  __REG32                 : 8;
} __sysc_csvmocfgr_bits;

/* Clock Supervisor Test Register (SYSC_CSVTESTR) */
typedef struct {
  __REG32 MOCLKGATE       : 1;
  __REG32 SOCLKGATE       : 1;
  __REG32 MPCLKGATE       : 1;
  __REG32 SPCLKGATE       : 1;
  __REG32 GPCLKGATE       : 1;
  __REG32                 :27;
} __sysc_csvtestr_bits;

/* Reset Control Register (SYSC_RSTCNTR) */
typedef struct {
  __REG32 SWRST           : 8;
  __REG32                 : 8;
  __REG32 SWHRST          : 8;
  __REG32 DBGR            : 8;
} __sysc_rstcntr_bits;

/* User Reset Cause Register (SYSC_RSTCAUSEUR) */
/* BootROM Reset Cause Register (SYSC_RSTCAUSEBT) */
typedef struct {
  __REG32 PRSTX           : 1;
  __REG32 RSTX            : 1;
  __REG32 SWHRST          : 1;
  __REG32 PRFERR          : 1;
  __REG32 WDR             : 1;
  __REG32 CSVRMC          : 1;
  __REG32 CSVRSC          : 1;
  __REG32 CSVRMP          : 1;
  __REG32 CSVRSP          : 1;
  __REG32 CSVRGP          : 1;
  __REG32                 : 6;
  __REG32 SWR             : 1;
  __REG32                 : 7;
  __REG32 FAKEPDR         : 1;
  __REG32 PD2R            : 1;
  __REG32 PD3R            : 1;
  __REG32 PD4R            : 1;
  __REG32 PD5R            : 1;
  __REG32                 : 3;
} __sysc_rstcauseur_bits;

/* Slow RC SCT Trigger Register (SYSC_SRCSCTTRG) */
typedef struct {
  __REG32 CGCPT           : 1;
  __REG32                 :31;
} __sysc_srcscttrg_bits;

/* Slow RC SCT Control Register (SYSC_SRCSCTCNTR) */
typedef struct {
  __REG32 MODE            : 1;
  __REG32 DBGEN           : 1;
  __REG32                 :30;
} __sysc_srcsctcntr_bits;

/* Slow RC SCT Compare Prescaler Register (SYSC_SRCSCTCPR) */
typedef struct {
  __REG32 CMPR            :16;
  __REG32 PSCL            : 4;
  __REG32                 :12;
} __sysc_srcsctcpr_bits;

/* Slow RC SCT Status Register (SYSC_SRCSCTSTATR) */
typedef struct {
  __REG32 INTF            : 1;
  __REG32 TRSTS           : 1;
  __REG32 BUSY            : 1;
  __REG32                 :29;
} __sysc_srcsctstatr_bits;

/* Slow RC SCT Interrupt Enable Register (SYSC_SRCSCTINTER) */
typedef struct {
  __REG32 INTE            : 1;
  __REG32                 :31;
} __sysc_srcsctinter_bits;

/* Slow RC SCT Interrupt Clear Register (SYSC_SRCSCTICLR) */
typedef struct {
  __REG32 INTC            : 1;
  __REG32                 :31;
} __sysc_srcscticlr_bits;

/* RC SCT Trigger Register (SYSC_RCSCTTRG) */
typedef struct {
  __REG32 CGCPT           : 1;
  __REG32                 :31;
} __sysc_rcscttrg_bits;

/* RC SCT Control Register (SYSC_RCSCTCNTR) */
typedef struct {
  __REG32 MODE            : 1;
  __REG32 DBGEN           : 1;
  __REG32                 :30;
} __sysc_rcsctcntr_bits;

/* RC SCT Compare Prescaler Register (SYSC_RCSCTCPR) */
typedef struct {
  __REG32 CMPR            :16;
  __REG32 PSCL            : 4;
  __REG32                 :12;
} __sysc_rcsctcpr_bits;

/* RC SCT Status Register (SYSC_RCSCTSTATR) */
typedef struct {
  __REG32 INTF            : 1;
  __REG32 TRSTS           : 1;
  __REG32 BUSY            : 1;
  __REG32                 :29;
} __sysc_rcscstat_bits;

/* RC SCT Interrupt Enable Register (SYSC_RCSCTINTER) */
typedef struct {
  __REG32 INTE            : 1;
  __REG32                 :31;
} __sysc_rcsctinter_bits;

/* RC SCT Interrupt Clear Register (SYSC_RCSCTICLR) */
typedef struct {
  __REG32 INTE            : 1;
  __REG32                 :31;
} __sysc_rcscticlr_bits;

/* Main SCT Trigger Register (SYSC_MAINSCTTRG) */
typedef struct {
  __REG32 CGCPT           : 1;
  __REG32                 :31;
} __sysc_mainscttrg_bits;

/* Main SCT Control Register (SYSC_MAINSCTCNTR) */
typedef struct {
  __REG32 MODE            : 1;
  __REG32 DBGEN           : 1;
  __REG32                 :30;
} __sysc_mainsctcntr_bits;

/* Main SCT Compare Prescaler Register (SYSC_MAINSCTCPR) */
typedef struct {
  __REG32 CMPR            :16;
  __REG32 PSCL            : 4;
  __REG32                 :12;
} __sysc_mainsctcpr_bits;

/* Main SCT Status Register (SYSC_MAINSCTSTATR) */
typedef struct {
  __REG32 INTF            : 1;
  __REG32 TRSTS           : 1;
  __REG32 BUSY            : 1;
  __REG32                 :29;
} __sysc_mainsctstatr_bits;

/* Main SCT Interrupt Enable Register (SYSC_MAINSCTINTER) */
typedef struct {
  __REG32 INTE            : 1;
  __REG32                 :31;
} __sysc_mainsctinter_bits;

/* Main SCT Interrupt Clear Register (SYSC_MAINSCTICLR) */
typedef struct {
  __REG32 INTC            : 1;
  __REG32                 :31;
} __sysc_mainscticlr_bits;

/* Sub SCT Trigger Register (SYSC_SUBSCTTRG) */
typedef struct {
  __REG32 CGCPT           : 1;
  __REG32                 :31;
} __sysc_subscttrg_bits;

/* Sub SCT Control Register (SYSC_SUBSCTCNTR) */
typedef struct {
  __REG32 MODE            : 1;
  __REG32 DBGEN           : 1;
  __REG32                 :30;
} __sysc_subsctcntr_bits;

/* Sub SCT Compare Prescaler Register (SYSC_SUBSCTCPR) */
typedef struct {
  __REG32 CMPR            :16;
  __REG32 PSCL            : 4;
  __REG32                 :12;
} __sysc_subsctcpr_bits;

/* Sub SCT Status Register (SYSC_SUBSCTSTATR) */
typedef struct {
  __REG32 INTF            : 1;
  __REG32 TRSTS           : 1;
  __REG32 BUSY            : 1;
  __REG32                 :29;
} __sysc_subsctstatr_bits;

/* Sub SCT Interrupt Enable Register (SYSC_SUBSCTINTER) */
typedef struct {
  __REG32 INTE            : 1;
  __REG32                 :31;
} __sysc_subsctinter_bits;

/* Sub SCT Interrupt Clear Register (SYSC_SUBSCTICLR) */
typedef struct {
  __REG32 INTC            : 1;
  __REG32                 :31;
} __sysc_subscticlr_bits;

/* Clock Output Function Configuration Register (SYSC_CKOTCFGR) */
typedef struct {
  __REG32 CKSEL           : 3;
  __REG32                 : 5;
  __REG32 CKOUTDIV        : 3;
  __REG32                 :21;
} __sysc_ckotcfgr_bits;

/* Special Configuration Register (SYSC_SPCCFGR) */
typedef struct {
  __REG32 FASTON          : 1;
  __REG32 FAKEPWRCNT      : 1;
  __REG32                 : 6;
  __REG32 PLLSTABS        : 1;
  __REG32 SSCGSTABS       : 1;
  __REG32 GFXSTABS        : 1;
  __REG32                 : 1;
  __REG32 PLLINSEL        : 1;
  __REG32                 : 3;
  __REG32 FCIMEN          : 1;
  __REG32 FCISEN          : 1;
  __REG32                 : 2;
  __REG32 GPIL            : 2;
  __REG32                 : 1;
  __REG32 PSSPADCTRL      : 1;
  __REG32 HOLDIO          : 1;
  __REG32                 : 7;
} __sysc_spccfgr_bits;

/* RC Configuration Register (SYSC_RCCFGR) */
typedef struct {
  __REG32 RCTRM           : 8;
  __REG32 SFREQ           : 1;
  __REG32                 :23;
} __sysc_rccfgr_bits;

/* JTAG Detect Register (SYSC_JTAGDETECT) */
typedef struct {
  __REG32 DBGCON          : 1;
  __REG32                 :31;
} __sysc_jtagdetect_bits;

/* JTAG Configuration Register (SYSC_JTAGCNFG) */
typedef struct {
  __REG32 DBGDONE         : 1;
  __REG32                 :31;
} __sysc_jtagcnfg_bits;

/* JTAG Wakeup Register (SYSC_JTAGWAKEUP) */
typedef struct {
  __REG32 DBGDONE         : 1;
  __REG32                 :31;
} __sysc_jtagwakeup_bits;

/* Timer Control Register (RTC_WTCR) */
typedef struct {
  __REG32 ST              : 1;
  __REG32 OE              : 1;
  __REG32 UPDT            : 1;
  __REG32 CSM             : 1;
  __REG32 RCKSEL          : 2;
  __REG32                 : 2;
  __REG32 ACAL            : 1;
  __REG32 MTRG            : 1;
  __REG32 ENUP            : 1;
  __REG32 CCKSEL          : 1;
  __REG32 SCAL            : 3;
  __REG32 UPCAL           : 1;
  __REG32                 :16;
} __rtc_wtcr_bits;

/* Timer Status Register (RTC_WTSR) */
typedef struct {
  __REG32 RUN             : 1;
  __REG32 CSF             : 1;
  __REG32                 :30;
} __rtc_wtsr_bits;

/* Interrupt Status Register (RTC_WINS) */
typedef struct {
  __REG32 SUBSEC          : 1;
  __REG32 SEC             : 1;
  __REG32 MIN             : 1;
  __REG32 HOUR            : 1;
  __REG32 DAY             : 1;
  __REG32 CFD             : 1;
  __REG32 CALD            : 1;
  __REG32                 :25;
} __rtc_wins_bits;

/* Interrupt Enable Register (RTC_WINE) */
typedef struct {
  __REG32 SUBSECE         : 1;
  __REG32 SECE            : 1;
  __REG32 MINE            : 1;
  __REG32 HOURE           : 1;
  __REG32 DAYE            : 1;
  __REG32 CFDE            : 1;
  __REG32 CALDE           : 1;
  __REG32                 :25;
} __rtc_wine_bits;

/* Interrupt Clear Register (RTC_WINC) */
typedef struct {
  __REG32 SUBSECC         : 1;
  __REG32 SECC            : 1;
  __REG32 MINC            : 1;
  __REG32 HOURC           : 1;
  __REG32 DAYC            : 1;
  __REG32 CFDCC           : 1;
  __REG32 CALDC           : 1;
  __REG32                 :25;
} __rtc_winc_bits;

/* Sub-Second Register (RTC_WTBR) */
typedef struct {
  __REG32 WTBR            :24;
  __REG32                 : 8;
} __rtc_wtbr_bits;

/* Real Time Register (RTC_WRT) */
typedef struct {
  __REG32 WTSR            : 6;
  __REG32                 : 2;
  __REG32 WTMR            : 6;
  __REG32                 : 2;
  __REG32 WTHR            : 5;
  __REG32                 :11;
} __rtc_wrt_bits;

/* Calibration Clock Counter Register (RTC_CNTCAL) */
typedef struct {
  __REG32 CNTCAL          :24;
  __REG32                 : 8;
} __rtc_cntcal_bits;

/* Calibration Clock Period Counter Register (RTC_CNTPCAL) */
typedef struct {
  __REG32 CNTPCAL         :11;
  __REG32                 :21;
} __rtc_cntpcal_bits;

/* Calibration Duration Register (RTC_DURMW) */
typedef struct {
  __REG32 DURMW           :24;
  __REG32                 : 8;
} __rtc_durmw_bits;

/* Calibration Trigger Register (RTC_CALTRG) */
typedef struct {
  __REG32 CALTRG          :12;
  __REG32                 :20;
} __rtc_caltrg_bits;

/* Debug Register (RTC_DEBUG) */
typedef struct {
  __REG32 DBGEN           : 1;
  __REG32                 :31;
} __rtc_debug_bits;

/* Watchdog Reset Cause Register (WDG_RSTCAUSE) */
typedef struct {
  __REG32 RSTCAUSE0       : 1;
  __REG32 RSTCAUSE1       : 1;
  __REG32 RSTCAUSE2       : 1;
  __REG32 RSTCAUSE3       : 1;
  __REG32 RSTCAUSE4       : 1;
  __REG32                 :27;
} __wdg_rstcause_bits;

/* Watchdog Trigger 0 Register (WDG_TRG0) */
typedef struct {
  __REG32 WDGTRG0         : 8;
  __REG32                 :24;
} __wdg_trg0_bits;

/* Watchdog Trigger 1 Register (WDG_TRG1) */
typedef struct {
  __REG32 WDGTRG1         : 8;
  __REG32                 :24;
} __wdg_trg1_bits;

/* Watchdog Interrupt Configuration Register (WDG_INT) */
typedef struct {
  __REG32 IRQFLAG         : 1;
  __REG32 NMIFLAG         : 1;
  __REG32                 :30;
} __wdg_int_bits;

/* Watchdog Interrupt Clear Register (WDG_INTCLR) */
typedef struct {
  __REG32 IRQCLR          : 1;
  __REG32 NMICLR          : 1;
  __REG32                 :30;
} __wdg_intclr_bits;

/* Watchdog Trigger 0 Configuration Register (WDG_TRG0CFG) */
typedef struct {
  __REG32 WDGTRG0CFG      : 8;
  __REG32                 :24;
} __wdg_trg0cfg_bits;

/* Watchdog Trigger 1 Configuration Register (WDG_TRG1CFG) */
typedef struct {
  __REG32 WDGTRG1CFG      : 8;
  __REG32                 :24;
} __wdg_trg1cfg_bits;

/* Watchdog Reset Delay Counter Register (WDG_RSTDLY) */
typedef struct {
  __REG32 WDGRSTDLY       :16;
  __REG32                 :16;
} __wdg_rstdly_bits;

/* Watchdog Configuration Register (WDG_CFG) */
typedef struct {
  __REG32 WDENRUN         : 1;
  __REG32 WDENPSS         : 1;
  __REG32 ALLOWSTOPCLK    : 1;
  __REG32 DEBUGEN         : 1;
  __REG32                 : 4;
  __REG32 CLKSEL          : 2;
  __REG32                 : 6;
  __REG32 OBSSEL          : 5;
  __REG32                 : 3;
  __REG32 LOCK            : 1;
  __REG32                 : 7;
} __wdg_cfg_bits;

/* Security Control Register (SCCFG_CTRL) */
typedef struct {
  __REG32 JTAGSWEN        : 1;
  __REG32                 :31;
} __sccfg_ctrl_bits;

/* Security Status Register 0 (SCCFG_STAT0) */
typedef struct {
  __REG32 TCFPEN          : 1;
  __REG32 TCFPS           : 1;
  __REG32 TCFPSLOCK       : 1;
  __REG32                 : 5;
  __REG32 EEFPEN          : 1;
  __REG32 EEFPS           : 1;
  __REG32 EEFPSLOCK       : 1;
  __REG32                 : 5;
  __REG32 TCFLNKOK        : 2;
  __REG32                 : 6;
  __REG32 EEFLNKOK        : 1;
  __REG32                 : 7;
} __sccfg_stat0_bits;

/* Security Status Register 1 (SCCFG_STAT1) */
typedef struct {
  __REG32 JTAGEN          : 1;
  __REG32 JTAGSWEN        : 1;
  __REG32 SCMEN           : 1;
  __REG32 FPPEN           : 1;
  __REG32 TCFCEEN         : 1;
  __REG32 EEFCEEN         : 1;
  __REG32                 :18;
  __REG32 CFGLOCK         : 1;
  __REG32                 : 7;
} __sccfg_stat1_bits;

/* Security Status Register 2 (SCCFG_STAT2) */
typedef struct {
  __REG32 SECEN           : 1;
  __REG32 SECLOCK         : 1;
  __REG32 SEC             : 1;
  __REG32                 : 5;
  __REG32 DBGRDY          : 1;
  __REG32                 :15;
  __REG32 CFGLOCK         : 1;
  __REG32                 : 7;
} __sccfg_stat2_bits;

/* CRC Configuration Register (CRCn_CFG) */
typedef struct {
  __REG32 CIRQCLR         : 1;
  __REG32                 : 7;
  __REG32 ROBYT           : 1;
  __REG32 ROBIT           : 1;
  __REG32 RIBYT           : 1;
  __REG32 RIBIT           : 1;
  __REG32                 : 4;
  __REG32 LEN             : 6;
  __REG32 SZ              : 2;
  __REG32 CIRQ            : 1;
  __REG32 CIEN            : 1;
  __REG32 CDEN            : 1;
  __REG32                 : 1;
  __REG32 LOCK            : 1;
  __REG32                 : 3;
} __crcn_cfg_bits;

/* Flash ECC Control Register (TCFCFG_FECCCTRL) */
typedef struct {
  __REG32 ECCOFF          : 1;
  __REG32                 :31;
} __tcfcfg_feccctrl_bits;

/* Flash ECC Bit Error Injection Register (TCFCFG_FECCEIR) */
typedef struct {
  __REG32 FECCEIR         : 7;
  __REG32                 :25;
} __tcfcfg_fecceir_bits;

/* Flash Interrupt Control Register (TCFCFG_FICTRLn) */
typedef struct {
  __REG32 RDYIE           : 1;
  __REG32 HANGIE          : 1;
  __REG32                 : 6;
  __REG32 RDYIC           : 1;
  __REG32 HANGIC          : 1;
  __REG32 WR32FC          : 1;
  __REG32                 :21;
} __tcfcfg_fictrl_bits;

/* Flash Status Register (TCFCFG_FSTATn) */
typedef struct {
  __REG32 RDY             : 1;
  __REG32 HANG            : 1;
  __REG32                 : 2;
  __REG32 WR32F           : 1;
  __REG32                 : 3;
  __REG32 RDYINT          : 1;
  __REG32 HANGINT         : 1;
  __REG32                 :22;
} __tcfcfg_fstat_bits;

/* Flash SEC Interrupt Register (TCFCFG_FSECIR) */
typedef struct {
  __REG32 SECIE           : 1;
  __REG32                 : 7;
  __REG32 SECIC           : 1;
  __REG32                 :23;
} __tcfcfg_fsecir_bits;

/* Flash CAM Output Upper Register (TCFCFG_FCAMHRn) */
typedef struct {
  __REG32 CAMH            :13;
  __REG32                 :19;
} __tcfcfg_fcamhr_bits;

/* Configuration Register (EEFCFG_CR) */
typedef struct {
  __REG32 FAWC            : 2;
  __REG32                 : 6;
  __REG32 WE              : 1;
  __REG32                 : 7;
  __REG32 SWFRST          : 1;
  __REG32                 :15;
} __eefcfg_cr_bits;

/* ECC Control Register (EEFCFG_ECR) */
typedef struct {
  __REG32 ECCOFF          : 1;
  __REG32                 :31;
} __eefcfg_ecr_bits;

/* Write Command Sequencer Configuration Register (EEFCFG_WCR) */
typedef struct {
  __REG32 DMAEN           : 1;
  __REG32                 : 7;
  __REG32 CSEN            : 1;
  __REG32                 :23;
} __eefcfg_wcr_bits;

/* Write Command Sequencer Status Register (EEFCFG_WSR) */
typedef struct {
  __REG32 ST              : 2;
  __REG32                 :30;
} __eefcfg_wsr_bits;

/* ECC Bit Error Injection Register (EEFCFG_EEIR) */
typedef struct {
  __REG32 EEIR            : 7;
  __REG32                 :25;
} __eefcfg_eeir_bits;

/* Write Mode Enable Register (EEFCFG_WMER) */
typedef struct {
  __REG32 MME             : 1;
  __REG32                 : 7;
  __REG32 AME             : 1;
  __REG32                 :23;
} __eefcfg_wmer_bits;

/* Interrupt Control Register (EEFCFG_ICR) */
typedef struct {
  __REG32 RDYIE           : 1;
  __REG32 HANGIE          : 1;
  __REG32 ERRIE           : 1;
  __REG32                 : 5;
  __REG32 RDYIC           : 1;
  __REG32 HANGIC          : 1;
  __REG32 ERRIC           : 1;
  __REG32                 :21;
} __eefcfg_icr_bits;

/* Status Register (EEFCFG_SR) */
typedef struct {
  __REG32 RDY             : 1;
  __REG32 HANG            : 1;
  __REG32 WDCYC           : 1;
  __REG32                 : 5;
  __REG32 RDYINT          : 1;
  __REG32 HANGINT         : 1;
  __REG32 ERRINT          : 1;
  __REG32                 :21;
} __eefcfg_sr_bits;

/* SEC Interrupt Register (EEFCFG_SECIR) */
typedef struct {
  __REG32 SECIE           : 1;
  __REG32                 : 7;
  __REG32 SECIC           : 1;
  __REG32                 : 7;
  __REG32 SECINT          : 1;
  __REG32                 :15;
} __eefcfg_secir_bits;

/* Flash CAM Output Higher Register (EEFCFG_FCAMHR) */
typedef struct {
  __REG32 CAMH            :13;
  __REG32                 :19;
} __eefcfg_fcamhr_bits;

/* TCMRAM_IF Configuration Register 0 (TRCFG_TCMCFG0) */
typedef struct {
  __REG32 ERRECC          : 7;
  __REG32                 : 1;
  __REG32 LOCKSTATUS      : 1;
  __REG32                 :15;
  __REG32 DWAIT           : 2;
  __REG32                 : 6;
} __trcfg_tcmcfg0_bits;

/* SRAM_IF Configuration Register 0 (SRCFG_CFG0) */
typedef struct {
  __REG32 ERRECC          : 7;
  __REG32                 : 1;
  __REG32 LOCKSTATUS      : 1;
  __REG32                 : 7;
  __REG32 WRWAIT          : 2;
  __REG32                 : 6;
  __REG32 RDWAIT          : 2;
  __REG32                 : 6;
} __srcfg_cfg0_bits;

/* SRAM_IF Configuration Register 2 (SRCFG_CFG2) */
typedef struct {
  __REG32 BYPASSEN        : 1;
  __REG32                 :31;
} __srcfg_cfg2_bits;

/* SRAM_IF Error Flag Register (SRCFG_ERRFLG) */
typedef struct {
  __REG32 SECFLG          : 1;
  __REG32                 : 7;
  __REG32 SECCLR          : 1;
  __REG32                 :23;
} __srcfg_errflg_bits;

/* SRAM_IF Interrupt Enable Register (SRCFG_INTE) */
typedef struct {
  __REG32 SECINTEN        : 1;
  __REG32                 :31;
} __srcfg_inte_bits;

/* SRAM_IF ECC Enable Register (SRCFG_ECCE) */
typedef struct {
  __REG32 ECCEN           : 1;
  __REG32                 :31;
} __srcfg_ecce_bits;

/* BootROM Hardware Interface Configuration Register (EXCFG_CNFG) */
typedef struct {
  __REG32 LOCKSTATUS      : 1;
  __REG32                 : 7;
  __REG32 SWAPREG         : 1;
  __REG32                 :23;
} __excfg_cnfg_bits;

/* IUNIT NMI Status Register (IRQn_NMIST) */
typedef struct {
  __REG32 NMISN           : 6;
  __REG32                 : 2;
  __REG32 NMIPS           : 4;
  __REG32                 :20;
} __irqn_nmist_bits;

/* IUNIT IRQ Status Register (IRQn_IRQST) */
typedef struct {
  __REG32 IRQSN           :10;
  __REG32                 : 6;
  __REG32 IRQPS           : 5;
  __REG32                 :11;
} __irqn_irqst_bits;

/* IUNIT NMI Priority Level Register (IRQn_NMIPLn) */
typedef struct {
  __REG32 NMIPL0          : 4;
  __REG32                 : 4;
  __REG32 NMIPL1          : 4;
  __REG32                 : 4;
  __REG32 NMIPL2          : 4;
  __REG32                 : 4;
  __REG32 NMIPL3          : 4;
  __REG32                 : 4;
} __irqn_nmipl_bits;

/* IUNIT IRQ Priority Level Register (IRQn_IRQPLn) */
typedef struct {
  __REG32 IRQPL0          : 5;
  __REG32                 : 3;
  __REG32 IRQPL1          : 5;
  __REG32                 : 3;
  __REG32 IRQPL2          : 5;
  __REG32                 : 3;
  __REG32 IRQPL3          : 5;
  __REG32                 : 3;
} __irqn_irqpl_bits;

/* IUNIT NMI Set Register (IRQn_NMIS) */
typedef struct {
  __REG32 NMIS0           : 1;
  __REG32 NMIS1           : 1;
  __REG32 NMIS2           : 1;
  __REG32 NMIS3           : 1;
  __REG32 NMIS4           : 1;
  __REG32 NMIS5           : 1;
  __REG32 NMIS6           : 1;
  __REG32 NMIS7           : 1;
  __REG32 NMIS8           : 1;
  __REG32 NMIS9           : 1;
  __REG32 NMIS10          : 1;
  __REG32 NMIS11          : 1;
  __REG32 NMIS12          : 1;
  __REG32 NMIS13          : 1;
  __REG32 NMIS14          : 1;
  __REG32 NMIS15          : 1;
  __REG32 NMIS16          : 1;
  __REG32 NMIS17          : 1;
  __REG32 NMIS18          : 1;
  __REG32 NMIS19          : 1;
  __REG32 NMIS20          : 1;
  __REG32 NMIS21          : 1;
  __REG32 NMIS22          : 1;
  __REG32 NMIS23          : 1;
  __REG32 NMIS24          : 1;
  __REG32 NMIS25          : 1;
  __REG32 NMIS26          : 1;
  __REG32 NMIS27          : 1;
  __REG32 NMIS28          : 1;
  __REG32 NMIS29          : 1;
  __REG32 NMIS30          : 1;
  __REG32 NMIS31          : 1;
} __irqn_nmis_bits;

/* IUNIT NMI Reset Register (IRQn_NMIR) */
typedef struct {
  __REG32 NMIR0           : 1;
  __REG32 NMIR1           : 1;
  __REG32 NMIR2           : 1;
  __REG32 NMIR3           : 1;
  __REG32 NMIR4           : 1;
  __REG32 NMIR5           : 1;
  __REG32 NMIR6           : 1;
  __REG32 NMIR7           : 1;
  __REG32 NMIR8           : 1;
  __REG32 NMIR9           : 1;
  __REG32 NMIR10          : 1;
  __REG32 NMIR11          : 1;
  __REG32 NMIR12          : 1;
  __REG32 NMIR13          : 1;
  __REG32 NMIR14          : 1;
  __REG32 NMIR15          : 1;
  __REG32 NMIR16          : 1;
  __REG32 NMIR17          : 1;
  __REG32 NMIR18          : 1;
  __REG32 NMIR19          : 1;
  __REG32 NMIR20          : 1;
  __REG32 NMIR21          : 1;
  __REG32 NMIR22          : 1;
  __REG32 NMIR23          : 1;
  __REG32 NMIR24          : 1;
  __REG32 NMIR25          : 1;
  __REG32 NMIR26          : 1;
  __REG32 NMIR27          : 1;
  __REG32 NMIR28          : 1;
  __REG32 NMIR29          : 1;
  __REG32 NMIR30          : 1;
  __REG32 NMIR31          : 1;
} __irqn_nmir_bits;

/* IUNIT NMI Software Status Register (IRQn_NMISIS) */
typedef struct {
  __REG32 NMSIS0          : 1;
  __REG32 NMSIS1          : 1;
  __REG32 NMSIS2          : 1;
  __REG32 NMSIS3          : 1;
  __REG32 NMSIS4          : 1;
  __REG32 NMSIS5          : 1;
  __REG32 NMSIS6          : 1;
  __REG32 NMSIS7          : 1;
  __REG32 NMSIS8          : 1;
  __REG32 NMSIS9          : 1;
  __REG32 NMSIS10         : 1;
  __REG32 NMSIS11         : 1;
  __REG32 NMSIS12         : 1;
  __REG32 NMSIS13         : 1;
  __REG32 NMSIS14         : 1;
  __REG32 NMSIS15         : 1;
  __REG32 NMSIS16         : 1;
  __REG32 NMSIS17         : 1;
  __REG32 NMSIS18         : 1;
  __REG32 NMSIS19         : 1;
  __REG32 NMSIS20         : 1;
  __REG32 NMSIS21         : 1;
  __REG32 NMSIS22         : 1;
  __REG32 NMSIS23         : 1;
  __REG32 NMSIS24         : 1;
  __REG32 NMSIS25         : 1;
  __REG32 NMSIS26         : 1;
  __REG32 NMSIS27         : 1;
  __REG32 NMSIS28         : 1;
  __REG32 NMSIS29         : 1;
  __REG32 NMSIS30         : 1;
  __REG32 NMSIS31         : 1;
} __irqn_nmisis_bits;

/* IUNIT IRQ Set Register (IRQn_IRQSn) */
typedef struct {
  __REG32 IRQS0           : 1;
  __REG32 IRQS1           : 1;
  __REG32 IRQS2           : 1;
  __REG32 IRQS3           : 1;
  __REG32 IRQS4           : 1;
  __REG32 IRQS5           : 1;
  __REG32 IRQS6           : 1;
  __REG32 IRQS7           : 1;
  __REG32 IRQS8           : 1;
  __REG32 IRQS9           : 1;
  __REG32 IRQS10          : 1;
  __REG32 IRQS11          : 1;
  __REG32 IRQS12          : 1;
  __REG32 IRQS13          : 1;
  __REG32 IRQS14          : 1;
  __REG32 IRQS15          : 1;
  __REG32 IRQS16          : 1;
  __REG32 IRQS17          : 1;
  __REG32 IRQS18          : 1;
  __REG32 IRQS19          : 1;
  __REG32 IRQS20          : 1;
  __REG32 IRQS21          : 1;
  __REG32 IRQS22          : 1;
  __REG32 IRQS23          : 1;
  __REG32 IRQS24          : 1;
  __REG32 IRQS25          : 1;
  __REG32 IRQS26          : 1;
  __REG32 IRQS27          : 1;
  __REG32 IRQS28          : 1;
  __REG32 IRQS29          : 1;
  __REG32 IRQS30          : 1;
  __REG32 IRQS31          : 1;
} __irqn_irqs_bits;

/* IUNIT IRQ Reset Register (IRQn_IRQRn) */
typedef struct {
  __REG32 IRQR0           : 1;
  __REG32 IRQR1           : 1;
  __REG32 IRQR2           : 1;
  __REG32 IRQR3           : 1;
  __REG32 IRQR4           : 1;
  __REG32 IRQR5           : 1;
  __REG32 IRQR6           : 1;
  __REG32 IRQR7           : 1;
  __REG32 IRQR8           : 1;
  __REG32 IRQR9           : 1;
  __REG32 IRQR10          : 1;
  __REG32 IRQR11          : 1;
  __REG32 IRQR12          : 1;
  __REG32 IRQR13          : 1;
  __REG32 IRQR14          : 1;
  __REG32 IRQR15          : 1;
  __REG32 IRQR16          : 1;
  __REG32 IRQR17          : 1;
  __REG32 IRQR18          : 1;
  __REG32 IRQR19          : 1;
  __REG32 IRQR20          : 1;
  __REG32 IRQR21          : 1;
  __REG32 IRQR22          : 1;
  __REG32 IRQR23          : 1;
  __REG32 IRQR24          : 1;
  __REG32 IRQR25          : 1;
  __REG32 IRQR26          : 1;
  __REG32 IRQR27          : 1;
  __REG32 IRQR28          : 1;
  __REG32 IRQR29          : 1;
  __REG32 IRQR30          : 1;
  __REG32 IRQR31          : 1;
} __irqn_irqr_bits;

/* IUNIT IRQ Software Interrupt Status Register (IRQn_IRQSISn) */
typedef struct {
  __REG32 IRQSIS0           : 1;
  __REG32 IRQSIS1           : 1;
  __REG32 IRQSIS2           : 1;
  __REG32 IRQSIS3           : 1;
  __REG32 IRQSIS4           : 1;
  __REG32 IRQSIS5           : 1;
  __REG32 IRQSIS6           : 1;
  __REG32 IRQSIS7           : 1;
  __REG32 IRQSIS8           : 1;
  __REG32 IRQSIS9           : 1;
  __REG32 IRQSIS10          : 1;
  __REG32 IRQSIS11          : 1;
  __REG32 IRQSIS12          : 1;
  __REG32 IRQSIS13          : 1;
  __REG32 IRQSIS14          : 1;
  __REG32 IRQSIS15          : 1;
  __REG32 IRQSIS16          : 1;
  __REG32 IRQSIS17          : 1;
  __REG32 IRQSIS18          : 1;
  __REG32 IRQSIS19          : 1;
  __REG32 IRQSIS20          : 1;
  __REG32 IRQSIS21          : 1;
  __REG32 IRQSIS22          : 1;
  __REG32 IRQSIS23          : 1;
  __REG32 IRQSIS24          : 1;
  __REG32 IRQSIS25          : 1;
  __REG32 IRQSIS26          : 1;
  __REG32 IRQSIS27          : 1;
  __REG32 IRQSIS28          : 1;
  __REG32 IRQSIS29          : 1;
  __REG32 IRQSIS30          : 1;
  __REG32 IRQSIS31          : 1;
} __irqn_irqsis_bits;

/* IUNIT IRQ Channel Enable Set Register (IRQn_IRQCESn) */
typedef struct {
  __REG32 IRQCES0           : 1;
  __REG32 IRQCES1           : 1;
  __REG32 IRQCES2           : 1;
  __REG32 IRQCES3           : 1;
  __REG32 IRQCES4           : 1;
  __REG32 IRQCES5           : 1;
  __REG32 IRQCES6           : 1;
  __REG32 IRQCES7           : 1;
  __REG32 IRQCES8           : 1;
  __REG32 IRQCES9           : 1;
  __REG32 IRQCES10          : 1;
  __REG32 IRQCES11          : 1;
  __REG32 IRQCES12          : 1;
  __REG32 IRQCES13          : 1;
  __REG32 IRQCES14          : 1;
  __REG32 IRQCES15          : 1;
  __REG32 IRQCES16          : 1;
  __REG32 IRQCES17          : 1;
  __REG32 IRQCES18          : 1;
  __REG32 IRQCES19          : 1;
  __REG32 IRQCES20          : 1;
  __REG32 IRQCES21          : 1;
  __REG32 IRQCES22          : 1;
  __REG32 IRQCES23          : 1;
  __REG32 IRQCES24          : 1;
  __REG32 IRQCES25          : 1;
  __REG32 IRQCES26          : 1;
  __REG32 IRQCES27          : 1;
  __REG32 IRQCES28          : 1;
  __REG32 IRQCES29          : 1;
  __REG32 IRQCES30          : 1;
  __REG32 IRQCES31          : 1;
} __irqn_irqces_bits;

/* IUNIT IRQ Channel Enable Clear Register (IRQn_IRQCECn) */
typedef struct {
  __REG32 IRQCEC0           : 1;
  __REG32 IRQCEC1           : 1;
  __REG32 IRQCEC2           : 1;
  __REG32 IRQCEC3           : 1;
  __REG32 IRQCEC4           : 1;
  __REG32 IRQCEC5           : 1;
  __REG32 IRQCEC6           : 1;
  __REG32 IRQCEC7           : 1;
  __REG32 IRQCEC8           : 1;
  __REG32 IRQCEC9           : 1;
  __REG32 IRQCEC10          : 1;
  __REG32 IRQCEC11          : 1;
  __REG32 IRQCEC12          : 1;
  __REG32 IRQCEC13          : 1;
  __REG32 IRQCEC14          : 1;
  __REG32 IRQCEC15          : 1;
  __REG32 IRQCEC16          : 1;
  __REG32 IRQCEC17          : 1;
  __REG32 IRQCEC18          : 1;
  __REG32 IRQCEC19          : 1;
  __REG32 IRQCEC20          : 1;
  __REG32 IRQCEC21          : 1;
  __REG32 IRQCEC22          : 1;
  __REG32 IRQCEC23          : 1;
  __REG32 IRQCEC24          : 1;
  __REG32 IRQCEC25          : 1;
  __REG32 IRQCEC26          : 1;
  __REG32 IRQCEC27          : 1;
  __REG32 IRQCEC28          : 1;
  __REG32 IRQCEC29          : 1;
  __REG32 IRQCEC30          : 1;
  __REG32 IRQCEC31          : 1;
} __irqn_irqcec_bits;

/* IUNIT IRQ Channel Enable Register (IRQn_IRQCEn) */
typedef struct {
  __REG32 IRQCE0            : 1;
  __REG32 IRQCE1            : 1;
  __REG32 IRQCE2            : 1;
  __REG32 IRQCE3            : 1;
  __REG32 IRQCE4            : 1;
  __REG32 IRQCE5            : 1;
  __REG32 IRQCE6            : 1;
  __REG32 IRQCE7            : 1;
  __REG32 IRQCE8            : 1;
  __REG32 IRQCE9            : 1;
  __REG32 IRQCE10           : 1;
  __REG32 IRQCE11           : 1;
  __REG32 IRQCE12           : 1;
  __REG32 IRQCE13           : 1;
  __REG32 IRQCE14           : 1;
  __REG32 IRQCE15           : 1;
  __REG32 IRQCE16           : 1;
  __REG32 IRQCE17           : 1;
  __REG32 IRQCE18           : 1;
  __REG32 IRQCE19           : 1;
  __REG32 IRQCE20           : 1;
  __REG32 IRQCE21           : 1;
  __REG32 IRQCE22           : 1;
  __REG32 IRQCE23           : 1;
  __REG32 IRQCE24           : 1;
  __REG32 IRQCE25           : 1;
  __REG32 IRQCE26           : 1;
  __REG32 IRQCE27           : 1;
  __REG32 IRQCE28           : 1;
  __REG32 IRQCE29           : 1;
  __REG32 IRQCE30           : 1;
  __REG32 IRQCE31           : 1;
} __irqn_irqce_bits;

/* IUNIT NMI Hold Clear Register (IRQn_NMIHC) */
typedef struct {
  __REG32 NMIHCN            : 5;
  __REG32                   :27;
} __irqn_nmihc_bits;

/* IUNIT NMI Hold Status Register (IRQn_NMIHS) */
typedef struct {
  __REG32 NMIHS0            : 1;
  __REG32 NMIHS1            : 1;
  __REG32 NMIHS2            : 1;
  __REG32 NMIHS3            : 1;
  __REG32 NMIHS4            : 1;
  __REG32 NMIHS5            : 1;
  __REG32 NMIHS6            : 1;
  __REG32 NMIHS7            : 1;
  __REG32 NMIHS8            : 1;
  __REG32 NMIHS9            : 1;
  __REG32 NMIHS10           : 1;
  __REG32 NMIHS11           : 1;
  __REG32 NMIHS12           : 1;
  __REG32 NMIHS13           : 1;
  __REG32 NMIHS14           : 1;
  __REG32 NMIHS15           : 1;
  __REG32 NMIHS16           : 1;
  __REG32 NMIHS17           : 1;
  __REG32 NMIHS18           : 1;
  __REG32 NMIHS19           : 1;
  __REG32 NMIHS20           : 1;
  __REG32 NMIHS21           : 1;
  __REG32 NMIHS22           : 1;
  __REG32 NMIHS23           : 1;
  __REG32 NMIHS24           : 1;
  __REG32 NMIHS25           : 1;
  __REG32 NMIHS26           : 1;
  __REG32 NMIHS27           : 1;
  __REG32 NMIHS28           : 1;
  __REG32 NMIHS29           : 1;
  __REG32 NMIHS30           : 1;
  __REG32 NMIHS31           : 1;
} __irqn_nmihs_bits;

/* IUNIT IRQ Hold Clear Register (IRQn_IRQHC) */
typedef struct {
  __REG32 IRQHCN            : 9;
  __REG32                   :23;
} __irqn_irqhc_bits;

/* IUNIT IRQ Hold Status Register (IRQn_IRQHSn) */
typedef struct {
  __REG32 IRQHS0            : 1;
  __REG32 IRQHS1            : 1;
  __REG32 IRQHS2            : 1;
  __REG32 IRQHS3            : 1;
  __REG32 IRQHS4            : 1;
  __REG32 IRQHS5            : 1;
  __REG32 IRQHS6            : 1;
  __REG32 IRQHS7            : 1;
  __REG32 IRQHS8            : 1;
  __REG32 IRQHS9            : 1;
  __REG32 IRQHS10           : 1;
  __REG32 IRQHS11           : 1;
  __REG32 IRQHS12           : 1;
  __REG32 IRQHS13           : 1;
  __REG32 IRQHS14           : 1;
  __REG32 IRQHS15           : 1;
  __REG32 IRQHS16           : 1;
  __REG32 IRQHS17           : 1;
  __REG32 IRQHS18           : 1;
  __REG32 IRQHS19           : 1;
  __REG32 IRQHS20           : 1;
  __REG32 IRQHS21           : 1;
  __REG32 IRQHS22           : 1;
  __REG32 IRQHS23           : 1;
  __REG32 IRQHS24           : 1;
  __REG32 IRQHS25           : 1;
  __REG32 IRQHS26           : 1;
  __REG32 IRQHS27           : 1;
  __REG32 IRQHS28           : 1;
  __REG32 IRQHS29           : 1;
  __REG32 IRQHS30           : 1;
  __REG32 IRQHS31           : 1;
} __irqn_irqhs_bits;

/* IUNIT IRQ Priority Level Mask Register (IRQn_IRQPLM) */
typedef struct {
  __REG32 IRQPLM            : 6;
  __REG32                   :26;
} __irqn_irqplm_bits;

/* IUNIT Control and Status Register (IRQn_CSR) */
typedef struct {
  __REG32 IRQEN             : 1;
  __REG32                   :15;
  __REG32 LST               : 1;
  __REG32                   :15;
} __irqn_csr_bits;

/* IUNIT Nesting Level Status Register (IRQn_NESTL) */
typedef struct {
  __REG32 NMINL             : 5;
  __REG32                   : 3;
  __REG32 IRQNL             : 5;
  __REG32                   :19;
} __irqn_nestl_bits;

/* IUNIT NMI Raw Status Register (IRQn_NMIRS) */
typedef struct {
  __REG32 NMIRS0            : 1;
  __REG32 NMIRS1            : 1;
  __REG32 NMIRS2            : 1;
  __REG32 NMIRS3            : 1;
  __REG32 NMIRS4            : 1;
  __REG32 NMIRS5            : 1;
  __REG32 NMIRS6            : 1;
  __REG32 NMIRS7            : 1;
  __REG32 NMIRS8            : 1;
  __REG32 NMIRS9            : 1;
  __REG32 NMIRS10           : 1;
  __REG32 NMIRS11           : 1;
  __REG32 NMIRS12           : 1;
  __REG32 NMIRS13           : 1;
  __REG32 NMIRS14           : 1;
  __REG32 NMIRS15           : 1;
  __REG32 NMIRS16           : 1;
  __REG32 NMIRS17           : 1;
  __REG32 NMIRS18           : 1;
  __REG32 NMIRS19           : 1;
  __REG32 NMIRS20           : 1;
  __REG32 NMIRS21           : 1;
  __REG32 NMIRS22           : 1;
  __REG32 NMIRS23           : 1;
  __REG32 NMIRS24           : 1;
  __REG32 NMIRS25           : 1;
  __REG32 NMIRS26           : 1;
  __REG32 NMIRS27           : 1;
  __REG32 NMIRS28           : 1;
  __REG32 NMIRS29           : 1;
  __REG32 NMIRS30           : 1;
  __REG32 NMIRS31           : 1;
} __irqn_nmirs_bits;

/* IUNIT NMI Preprocessed Status Register (IRQn_NMIPS) */
typedef struct {
  __REG32 NMIPS0            : 1;
  __REG32 NMIPS1            : 1;
  __REG32 NMIPS2            : 1;
  __REG32 NMIPS3            : 1;
  __REG32 NMIPS4            : 1;
  __REG32 NMIPS5            : 1;
  __REG32 NMIPS6            : 1;
  __REG32 NMIPS7            : 1;
  __REG32 NMIPS8            : 1;
  __REG32 NMIPS9            : 1;
  __REG32 NMIPS10           : 1;
  __REG32 NMIPS11           : 1;
  __REG32 NMIPS12           : 1;
  __REG32 NMIPS13           : 1;
  __REG32 NMIPS14           : 1;
  __REG32 NMIPS15           : 1;
  __REG32 NMIPS16           : 1;
  __REG32 NMIPS17           : 1;
  __REG32 NMIPS18           : 1;
  __REG32 NMIPS19           : 1;
  __REG32 NMIPS20           : 1;
  __REG32 NMIPS21           : 1;
  __REG32 NMIPS22           : 1;
  __REG32 NMIPS23           : 1;
  __REG32 NMIPS24           : 1;
  __REG32 NMIPS25           : 1;
  __REG32 NMIPS26           : 1;
  __REG32 NMIPS27           : 1;
  __REG32 NMIPS28           : 1;
  __REG32 NMIPS29           : 1;
  __REG32 NMIPS30           : 1;
  __REG32 NMIPS31           : 1;
} __irqn_nmips_bits;

/* IUNIT IRQ Raw Status Register (IRQn_IRQRSn) */
typedef struct {
  __REG32 IRQRS0            : 1;
  __REG32 IRQRS1            : 1;
  __REG32 IRQRS2            : 1;
  __REG32 IRQRS3            : 1;
  __REG32 IRQRS4            : 1;
  __REG32 IRQRS5            : 1;
  __REG32 IRQRS6            : 1;
  __REG32 IRQRS7            : 1;
  __REG32 IRQRS8            : 1;
  __REG32 IRQRS9            : 1;
  __REG32 IRQRS10           : 1;
  __REG32 IRQRS11           : 1;
  __REG32 IRQRS12           : 1;
  __REG32 IRQRS13           : 1;
  __REG32 IRQRS14           : 1;
  __REG32 IRQRS15           : 1;
  __REG32 IRQRS16           : 1;
  __REG32 IRQRS17           : 1;
  __REG32 IRQRS18           : 1;
  __REG32 IRQRS19           : 1;
  __REG32 IRQRS20           : 1;
  __REG32 IRQRS21           : 1;
  __REG32 IRQRS22           : 1;
  __REG32 IRQRS23           : 1;
  __REG32 IRQRS24           : 1;
  __REG32 IRQRS25           : 1;
  __REG32 IRQRS26           : 1;
  __REG32 IRQRS27           : 1;
  __REG32 IRQRS28           : 1;
  __REG32 IRQRS29           : 1;
  __REG32 IRQRS30           : 1;
  __REG32 IRQRS31           : 1;
} __irqn_irqrs_bits;

/* IUNIT IRQ Preprocessed Status Register (IRQn_IRQPSn) */
typedef struct {
  __REG32 IRQPS0            : 1;
  __REG32 IRQPS1            : 1;
  __REG32 IRQPS2            : 1;
  __REG32 IRQPS3            : 1;
  __REG32 IRQPS4            : 1;
  __REG32 IRQPS5            : 1;
  __REG32 IRQPS6            : 1;
  __REG32 IRQPS7            : 1;
  __REG32 IRQPS8            : 1;
  __REG32 IRQPS9            : 1;
  __REG32 IRQPS10           : 1;
  __REG32 IRQPS11           : 1;
  __REG32 IRQPS12           : 1;
  __REG32 IRQPS13           : 1;
  __REG32 IRQPS14           : 1;
  __REG32 IRQPS15           : 1;
  __REG32 IRQPS16           : 1;
  __REG32 IRQPS17           : 1;
  __REG32 IRQPS18           : 1;
  __REG32 IRQPS19           : 1;
  __REG32 IRQPS20           : 1;
  __REG32 IRQPS21           : 1;
  __REG32 IRQPS22           : 1;
  __REG32 IRQPS23           : 1;
  __REG32 IRQPS24           : 1;
  __REG32 IRQPS25           : 1;
  __REG32 IRQPS26           : 1;
  __REG32 IRQPS27           : 1;
  __REG32 IRQPS28           : 1;
  __REG32 IRQPS29           : 1;
  __REG32 IRQPS30           : 1;
  __REG32 IRQPS31           : 1;
} __irqn_irqps_bits;

/* IUNIT Unlock Register (IRQn_UNLOCK) */
typedef struct {
  __REG32 UNLOCK0           : 1;
  __REG32 UNLOCK1           : 1;
  __REG32 UNLOCK2           : 1;
  __REG32 UNLOCK3           : 1;
  __REG32 UNLOCK4           : 1;
  __REG32 UNLOCK5           : 1;
  __REG32 UNLOCK6           : 1;
  __REG32 UNLOCK7           : 1;
  __REG32 UNLOCK8           : 1;
  __REG32 UNLOCK9           : 1;
  __REG32 UNLOCK10            : 1;
  __REG32 UNLOCK11            : 1;
  __REG32 UNLOCK12            : 1;
  __REG32 UNLOCK13            : 1;
  __REG32 UNLOCK14            : 1;
  __REG32 UNLOCK15            : 1;
  __REG32 UNLOCK16            : 1;
  __REG32 UNLOCK17            : 1;
  __REG32 UNLOCK18            : 1;
  __REG32 UNLOCK19            : 1;
  __REG32 UNLOCK20            : 1;
  __REG32 UNLOCK21            : 1;
  __REG32 UNLOCK22            : 1;
  __REG32 UNLOCK23            : 1;
  __REG32 UNLOCK24            : 1;
  __REG32 UNLOCK25            : 1;
  __REG32 UNLOCK26            : 1;
  __REG32 UNLOCK27            : 1;
  __REG32 UNLOCK28            : 1;
  __REG32 UNLOCK29            : 1;
  __REG32 UNLOCK30            : 1;
  __REG32 UNLOCK31            : 1;
} __irqn_unlock_bits;

/* IUNIT ECC Error Interrupt Register (IRQn_EEI) */
typedef struct {
  __REG32 EENC              : 1;
  __REG32                   : 7;
  __REG32 EENS              : 1;
  __REG32                   : 7;
  __REG32 EEIC              : 1;
  __REG32                   : 7;
  __REG32 EEIS              : 1;
  __REG32                   : 7;
} __irqn_eei_bits;

/* IUNIT ECC Address Number Register (IRQn_EAN) */
typedef struct {
  __REG32 EAN               : 8;
  __REG32                   :24;
} __irqn_ean_bits;

/* IUNIT ECC Test Register (IRQn_ET) */
typedef struct {
  __REG32 ET                : 1;
  __REG32                   :31;
} __irqn_et_bits;

/* IUNIT ECC Error Bits Register (IRQn_EEB2) */
typedef struct {
  __REG32 EEBE2             : 7;
  __REG32                   : 1;
  __REG32 EEBO2             : 7;
  __REG32                   :17;
} __irqn_eeb2_bits;

/* Pin Configuration Register (PPC_PCFGRn) */
typedef struct {
  __REG16 POF               : 3;
  __REG16                   : 3;
  __REG16 ODR               : 2;
  __REG16 PDE               : 1;
  __REG16 PUE               : 1;
  __REG16 PIL               : 2;
  __REG16 PIE               : 1;
  __REG16 PID               : 1;
  __REG16 POD               : 1;
  __REG16 POE               : 1;
} __ppc_pcfgrn_bits;

/* Configuration Register (EICUn_CNFGR) */
typedef struct {
  __REG32 CLKSEL            : 2;
  __REG32 PRESCALE          : 6;
  __REG32                   : 8;
  __REG32 OBSCH             : 5;
  __REG32                   : 1;
  __REG32 BUSY              : 1;
  __REG32 DATAVALID         : 1;
  __REG32 DATARESET         : 1;
  __REG32 OBSEN             : 1;
  __REG32 IRQEN             : 1;
  __REG32                   : 5;
} __eicun_cnfgr_bits;

/* External Interrupt Pin Enable Register (EICUn_IRENR) */
typedef struct {
  __REG32 IREN0             : 1;
  __REG32 IREN1             : 1;
  __REG32 IREN2             : 1;
  __REG32 IREN3             : 1;
  __REG32 IREN4             : 1;
  __REG32 IREN5             : 1;
  __REG32 IREN6             : 1;
  __REG32 IREN7             : 1;
  __REG32 IREN8             : 1;
  __REG32 IREN9             : 1;
  __REG32 IREN10            : 1;
  __REG32 IREN11            : 1;
  __REG32 IREN12            : 1;
  __REG32 IREN13            : 1;
  __REG32 IREN14            : 1;
  __REG32 IREN15            : 1;
  __REG32 IREN16            : 1;
  __REG32 IREN17            : 1;
  __REG32 IREN18            : 1;
  __REG32 IREN19            : 1;
  __REG32 IREN20            : 1;
  __REG32 IREN21            : 1;
  __REG32 IREN22            : 1;
  __REG32 IREN23            : 1;
  __REG32 IREN24            : 1;
  __REG32 IREN25            : 1;
  __REG32 IREN26            : 1;
  __REG32 IREN27            : 1;
  __REG32 IREN28            : 1;
  __REG32 IREN29            : 1;
  __REG32 IREN30            : 1;
  __REG32 IREN31            : 1;
} __eicun_irenr_bits;

/* TPU Lock Status Register (TPUn_LST) */
typedef struct {
  __REG32 LST               : 1;
  __REG32                   :31;
} __tpun_lst_bits;

/* TPU Configuration Register (TPUn_CFG) */
typedef struct {
  __REG32 INTE              : 1;
  __REG32                   :15;
  __REG32 GLBPS             : 6;
  __REG32                   : 1;
  __REG32 GLBPSE            : 1;
  __REG32 DBGE              : 1;
  __REG32                   : 7;
} __tpun_cfg_bits;

/* TPU Timer Interrupt Request Register (TPUn_TIR) */
typedef struct {
  __REG32 IR                : 8;
  __REG32                   :24;
} __tpun_tir_bits;

/* TPU Timer Status Register (TPUn_TST) */
typedef struct {
  __REG32 TS                : 8;
  __REG32                   :24;
} __tpun_tst_bits;

/* TPU Timer Interrupt Enable Register (TPUn_TIE) */
typedef struct {
  __REG32 IE                : 8;
  __REG32                   :24;
} __tpun_tie_bits;

/* TPU Timer Control Register 0 (TPUn_TCN00~07) */
typedef struct {
  __REG32 ECPL              :24;
  __REG32                   : 2;
  __REG32 IRC               : 1;
  __REG32 IEC               : 1;
  __REG32 IES               : 1;
  __REG32 CONT              : 1;
  __REG32 STOP              : 1;
  __REG32 START             : 1;
} __tpun_tcn00_bits;

/* TPU Timer Control Register 1 (TPUn_TCN10~17) */
typedef struct {
  __REG32 PS                : 2;
  __REG32 TMOD              : 1;
  __REG32 FRT               : 1;
  __REG32 PL                : 1;
  __REG32                   :27;
} __tpun_tcn10_bits;

/* TPU Timer Current Count Register (TPUn_TCCn) */
typedef struct {
  __REG32 TCC               :24;
  __REG32                   : 8;
} __tpun_tcc_bits;

/* PPU Peripheral Read Attribute Register (PPUn_PR0~15) */
typedef struct {
  __REG32 PR0               : 1;
  __REG32 PR1               : 1;
  __REG32 PR2               : 1;
  __REG32 PR3               : 1;
  __REG32 PR4               : 1;
  __REG32 PR5               : 1;
  __REG32 PR6               : 1;
  __REG32 PR7               : 1;
  __REG32 PR8               : 1;
  __REG32 PR9               : 1;
  __REG32 PR10              : 1;
  __REG32 PR11              : 1;
  __REG32 PR12              : 1;
  __REG32 PR13              : 1;
  __REG32 PR14              : 1;
  __REG32 PR15              : 1;
  __REG32 PR16              : 1;
  __REG32 PR17              : 1;
  __REG32 PR18              : 1;
  __REG32 PR19              : 1;
  __REG32 PR20              : 1;
  __REG32 PR21              : 1;
  __REG32 PR22              : 1;
  __REG32 PR23              : 1;
  __REG32 PR24              : 1;
  __REG32 PR25              : 1;
  __REG32 PR26              : 1;
  __REG32 PR27              : 1;
  __REG32 PR28              : 1;
  __REG32 PR29              : 1;
  __REG32 PR30              : 1;
  __REG32 PR31              : 1;
} __ppun_pr_bits;

/* PPU Peripheral Read Attribute Set Register (PPUn_PRS0~15) */
typedef struct {
  __REG32 PRS0              : 1;
  __REG32 PRS1              : 1;
  __REG32 PRS2              : 1;
  __REG32 PRS3              : 1;
  __REG32 PRS4              : 1;
  __REG32 PRS5              : 1;
  __REG32 PRS6              : 1;
  __REG32 PRS7              : 1;
  __REG32 PRS8              : 1;
  __REG32 PRS9              : 1;
  __REG32 PRS10             : 1;
  __REG32 PRS11             : 1;
  __REG32 PRS12             : 1;
  __REG32 PRS13             : 1;
  __REG32 PRS14             : 1;
  __REG32 PRS15             : 1;
  __REG32 PRS16             : 1;
  __REG32 PRS17             : 1;
  __REG32 PRS18             : 1;
  __REG32 PRS19             : 1;
  __REG32 PRS20             : 1;
  __REG32 PRS21             : 1;
  __REG32 PRS22             : 1;
  __REG32 PRS23             : 1;
  __REG32 PRS24             : 1;
  __REG32 PRS25             : 1;
  __REG32 PRS26             : 1;
  __REG32 PRS27             : 1;
  __REG32 PRS28             : 1;
  __REG32 PRS29             : 1;
  __REG32 PRS30             : 1;
  __REG32 PRS31             : 1;
} __ppun_prs_bits;

/* PPU Peripheral Read Attribute Clear Register (PPUn_PRC0~15) */
typedef struct {
  __REG32 PRC0              : 1;
  __REG32 PRC1              : 1;
  __REG32 PRC2              : 1;
  __REG32 PRC3              : 1;
  __REG32 PRC4              : 1;
  __REG32 PRC5              : 1;
  __REG32 PRC6              : 1;
  __REG32 PRC7              : 1;
  __REG32 PRC8              : 1;
  __REG32 PRC9              : 1;
  __REG32 PRC10             : 1;
  __REG32 PRC11             : 1;
  __REG32 PRC12             : 1;
  __REG32 PRC13             : 1;
  __REG32 PRC14             : 1;
  __REG32 PRC15             : 1;
  __REG32 PRC16             : 1;
  __REG32 PRC17             : 1;
  __REG32 PRC18             : 1;
  __REG32 PRC19             : 1;
  __REG32 PRC20             : 1;
  __REG32 PRC21             : 1;
  __REG32 PRC22             : 1;
  __REG32 PRC23             : 1;
  __REG32 PRC24             : 1;
  __REG32 PRC25             : 1;
  __REG32 PRC26             : 1;
  __REG32 PRC27             : 1;
  __REG32 PRC28             : 1;
  __REG32 PRC29             : 1;
  __REG32 PRC30             : 1;
  __REG32 PRC31             : 1;
} __ppun_prc_bits;

/* PPU Peripheral Access Attribute Register (PPUn_PA0~15) */
typedef struct {
  __REG32 PA0               : 1;
  __REG32 PA1               : 1;
  __REG32 PA2               : 1;
  __REG32 PA3               : 1;
  __REG32 PA4               : 1;
  __REG32 PA5               : 1;
  __REG32 PA6               : 1;
  __REG32 PA7               : 1;
  __REG32 PA8               : 1;
  __REG32 PA9               : 1;
  __REG32 PA10              : 1;
  __REG32 PA11              : 1;
  __REG32 PA12              : 1;
  __REG32 PA13              : 1;
  __REG32 PA14              : 1;
  __REG32 PA15              : 1;
  __REG32 PA16              : 1;
  __REG32 PA17              : 1;
  __REG32 PA18              : 1;
  __REG32 PA19              : 1;
  __REG32 PA20              : 1;
  __REG32 PA21              : 1;
  __REG32 PA22              : 1;
  __REG32 PA23              : 1;
  __REG32 PA24              : 1;
  __REG32 PA25              : 1;
  __REG32 PA26              : 1;
  __REG32 PA27              : 1;
  __REG32 PA28              : 1;
  __REG32 PA29              : 1;
  __REG32 PA30              : 1;
  __REG32 PA31              : 1;
} __ppun_pa_bits;

/* PPU Peripheral Access Attribute Set Register (PPUn_PAS0~15) */
typedef struct {
  __REG32 PAS0              : 1;
  __REG32 PAS1              : 1;
  __REG32 PAS2              : 1;
  __REG32 PAS3              : 1;
  __REG32 PAS4              : 1;
  __REG32 PAS5              : 1;
  __REG32 PAS6              : 1;
  __REG32 PAS7              : 1;
  __REG32 PAS8              : 1;
  __REG32 PAS9              : 1;
  __REG32 PAS10             : 1;
  __REG32 PAS11             : 1;
  __REG32 PAS12             : 1;
  __REG32 PAS13             : 1;
  __REG32 PAS14             : 1;
  __REG32 PAS15             : 1;
  __REG32 PAS16             : 1;
  __REG32 PAS17             : 1;
  __REG32 PAS18             : 1;
  __REG32 PAS19             : 1;
  __REG32 PAS20             : 1;
  __REG32 PAS21             : 1;
  __REG32 PAS22             : 1;
  __REG32 PAS23             : 1;
  __REG32 PAS24             : 1;
  __REG32 PAS25             : 1;
  __REG32 PAS26             : 1;
  __REG32 PAS27             : 1;
  __REG32 PAS28             : 1;
  __REG32 PAS29             : 1;
  __REG32 PAS30             : 1;
  __REG32 PAS31             : 1;
} __ppun_pas_bits;

/* PPU Peripheral Access Attribute Clear Register (PPUn_PAC0~15) */
typedef struct {
  __REG32 PAC0              : 1;
  __REG32 PAC1              : 1;
  __REG32 PAC2              : 1;
  __REG32 PAC3              : 1;
  __REG32 PAC4              : 1;
  __REG32 PAC5              : 1;
  __REG32 PAC6              : 1;
  __REG32 PAC7              : 1;
  __REG32 PAC8              : 1;
  __REG32 PAC9              : 1;
  __REG32 PAC10             : 1;
  __REG32 PAC11             : 1;
  __REG32 PAC12             : 1;
  __REG32 PAC13             : 1;
  __REG32 PAC14             : 1;
  __REG32 PAC15             : 1;
  __REG32 PAC16             : 1;
  __REG32 PAC17             : 1;
  __REG32 PAC18             : 1;
  __REG32 PAC19             : 1;
  __REG32 PAC20             : 1;
  __REG32 PAC21             : 1;
  __REG32 PAC22             : 1;
  __REG32 PAC23             : 1;
  __REG32 PAC24             : 1;
  __REG32 PAC25             : 1;
  __REG32 PAC26             : 1;
  __REG32 PAC27             : 1;
  __REG32 PAC28             : 1;
  __REG32 PAC29             : 1;
  __REG32 PAC30             : 1;
  __REG32 PAC31             : 1;
} __ppun_pac_bits;

/* PPU GPIO Access Attribute Register (PPUn_GA0~15) */
typedef struct {
  __REG32 GA0               : 1;
  __REG32 GA1               : 1;
  __REG32 GA2               : 1;
  __REG32 GA3               : 1;
  __REG32 GA4               : 1;
  __REG32 GA5               : 1;
  __REG32 GA6               : 1;
  __REG32 GA7               : 1;
  __REG32 GA8               : 1;
  __REG32 GA9               : 1;
  __REG32 GA10              : 1;
  __REG32 GA11              : 1;
  __REG32 GA12              : 1;
  __REG32 GA13              : 1;
  __REG32 GA14              : 1;
  __REG32 GA15              : 1;
  __REG32 GA16              : 1;
  __REG32 GA17              : 1;
  __REG32 GA18              : 1;
  __REG32 GA19              : 1;
  __REG32 GA20              : 1;
  __REG32 GA21              : 1;
  __REG32 GA22              : 1;
  __REG32 GA23              : 1;
  __REG32 GA24              : 1;
  __REG32 GA25              : 1;
  __REG32 GA26              : 1;
  __REG32 GA27              : 1;
  __REG32 GA28              : 1;
  __REG32 GA29              : 1;
  __REG32 GA30              : 1;
  __REG32 GA31              : 1;
} __ppun_ga_bits;

/* PPU GPIO Access Attribute Set Register (PPUn_GAS0~15) */
typedef struct {
  __REG32 GAS0              : 1;
  __REG32 GAS1              : 1;
  __REG32 GAS2              : 1;
  __REG32 GAS3              : 1;
  __REG32 GAS4              : 1;
  __REG32 GAS5              : 1;
  __REG32 GAS6              : 1;
  __REG32 GAS7              : 1;
  __REG32 GAS8              : 1;
  __REG32 GAS9              : 1;
  __REG32 GAS10             : 1;
  __REG32 GAS11             : 1;
  __REG32 GAS12             : 1;
  __REG32 GAS13             : 1;
  __REG32 GAS14             : 1;
  __REG32 GAS15             : 1;
  __REG32 GAS16             : 1;
  __REG32 GAS17             : 1;
  __REG32 GAS18             : 1;
  __REG32 GAS19             : 1;
  __REG32 GAS20             : 1;
  __REG32 GAS21             : 1;
  __REG32 GAS22             : 1;
  __REG32 GAS23             : 1;
  __REG32 GAS24             : 1;
  __REG32 GAS25             : 1;
  __REG32 GAS26             : 1;
  __REG32 GAS27             : 1;
  __REG32 GAS28             : 1;
  __REG32 GAS29             : 1;
  __REG32 GAS30             : 1;
  __REG32 GAS31             : 1;
} __ppun_gas_bits;

/* PPU GPIO Access Attribute Clear Register (PPUn_GAC0~15) */
typedef struct {
  __REG32 GAC0              : 1;
  __REG32 GAC1              : 1;
  __REG32 GAC2              : 1;
  __REG32 GAC3              : 1;
  __REG32 GAC4              : 1;
  __REG32 GAC5              : 1;
  __REG32 GAC6              : 1;
  __REG32 GAC7              : 1;
  __REG32 GAC8              : 1;
  __REG32 GAC9              : 1;
  __REG32 GAC10             : 1;
  __REG32 GAC11             : 1;
  __REG32 GAC12             : 1;
  __REG32 GAC13             : 1;
  __REG32 GAC14             : 1;
  __REG32 GAC15             : 1;
  __REG32 GAC16             : 1;
  __REG32 GAC17             : 1;
  __REG32 GAC18             : 1;
  __REG32 GAC19             : 1;
  __REG32 GAC20             : 1;
  __REG32 GAC21             : 1;
  __REG32 GAC22             : 1;
  __REG32 GAC23             : 1;
  __REG32 GAC24             : 1;
  __REG32 GAC25             : 1;
  __REG32 GAC26             : 1;
  __REG32 GAC27             : 1;
  __REG32 GAC28             : 1;
  __REG32 GAC29             : 1;
  __REG32 GAC30             : 1;
  __REG32 GAC31             : 1;
} __ppun_gac_bits;

/* PPU Status Register (PPUn_ST) */
typedef struct {
  __REG32 LST               : 1;
  __REG32                   : 7;
  __REG32 PSA               : 1;
  __REG32                   :23;
} __ppun_st_bits;

/* PPU Control Register (PPUn_CTR) */
typedef struct {
  __REG32 DMAEN             : 1;
  __REG32                   : 7;
  __REG32 PTST              : 1;
  __REG32                   : 7;
  __REG32 NEAV              : 1;
  __REG32                   :15;
} __ppun_ctr_bits;

/* BECU Control Register - L (BECUn_CTRL) */
typedef struct {
  __REG16 NMI               : 1;
  __REG16 NMICL             : 1;
  __REG16                   : 6;
  __REG16 PROT              : 1;
  __REG16                   : 7;
} __becun_ctrl_bits;

/* BECU Control Register - H (BECUn_CTRH) */
typedef struct {
  __REG16 RD                : 8;
  __REG16 WR                : 8;
} __becun_ctrh_bits;

/* BECU NMI Enable Register (BECUn_NMIEN) */
typedef struct {
  __REG16 NMIEN             : 1;
  __REG16                   :15;
} __becun_nmien_bits;

/* RETENTIONRAM Configuration and Status Register (RRCFG_CSR) */
typedef struct {
  __REG32 CEIEN             : 1;
  __REG32                   : 7;
  __REG32 CEIF              : 1;
  __REG32 LCK               : 1;
  __REG32                   : 6;
  __REG32 CEIC              : 1;
  __REG32                   : 7;
  __REG32 RAWC1             : 2;
  __REG32 WAWC1             : 2;
  __REG32                   : 4;
} __rrcfg_csr_bits;

/* RETENTIONRAM Error Mask Register 1 (RRCFG_ERRMSKR1) */
typedef struct {
  __REG32 MSK               : 7;
  __REG32                   :25;
} __rrcfg_errmskr1_bits;

/* RETENTIONRAM ECC Enable Register (RRCFG_ECCEN) */
typedef struct {
  __REG32 ECCEN             : 1;
  __REG32                   :31;
} __rrcfg_eccen_bits;

/* External Interrupt Enable Register (EICn_ENIR) */
typedef struct {
  __REG32 EN0               : 1;
  __REG32 EN1               : 1;
  __REG32 EN2               : 1;
  __REG32 EN3               : 1;
  __REG32 EN4               : 1;
  __REG32 EN5               : 1;
  __REG32 EN6               : 1;
  __REG32 EN7               : 1;
  __REG32 EN8               : 1;
  __REG32 EN9               : 1;
  __REG32 EN10              : 1;
  __REG32 EN11              : 1;
  __REG32 EN12              : 1;
  __REG32 EN13              : 1;
  __REG32 EN14              : 1;
  __REG32 EN15              : 1;
  __REG32 EN16              : 1;
  __REG32 EN17              : 1;
  __REG32 EN18              : 1;
  __REG32 EN19              : 1;
  __REG32 EN20              : 1;
  __REG32 EN21              : 1;
  __REG32 EN22              : 1;
  __REG32 EN23              : 1;
  __REG32 EN24              : 1;
  __REG32 EN25              : 1;
  __REG32 EN26              : 1;
  __REG32 EN27              : 1;
  __REG32 EN28              : 1;
  __REG32 EN29              : 1;
  __REG32 EN30              : 1;
  __REG32 EN31              : 1;
} __eicn_enir_bits;

/* External Interrupt Enable Set Register (EICn_ENISR) */
typedef struct {
  __REG32 ENS0              : 1;
  __REG32 ENS1              : 1;
  __REG32 ENS2              : 1;
  __REG32 ENS3              : 1;
  __REG32 ENS4              : 1;
  __REG32 ENS5              : 1;
  __REG32 ENS6              : 1;
  __REG32 ENS7              : 1;
  __REG32 ENS8              : 1;
  __REG32 ENS9              : 1;
  __REG32 ENS10             : 1;
  __REG32 ENS11             : 1;
  __REG32 ENS12             : 1;
  __REG32 ENS13             : 1;
  __REG32 ENS14             : 1;
  __REG32 ENS15             : 1;
  __REG32 ENS16             : 1;
  __REG32 ENS17             : 1;
  __REG32 ENS18             : 1;
  __REG32 ENS19             : 1;
  __REG32 ENS20             : 1;
  __REG32 ENS21             : 1;
  __REG32 ENS22             : 1;
  __REG32 ENS23             : 1;
  __REG32 ENS24             : 1;
  __REG32 ENS25             : 1;
  __REG32 ENS26             : 1;
  __REG32 ENS27             : 1;
  __REG32 ENS28             : 1;
  __REG32 ENS29             : 1;
  __REG32 ENS30             : 1;
  __REG32 ENS31             : 1;
} __eicn_enisr_bits;

/* External Interrupt Enable Clear Register (EICn_ENICR) */
typedef struct {
  __REG32 ENC0              : 1;
  __REG32 ENC1              : 1;
  __REG32 ENC2              : 1;
  __REG32 ENC3              : 1;
  __REG32 ENC4              : 1;
  __REG32 ENC5              : 1;
  __REG32 ENC6              : 1;
  __REG32 ENC7              : 1;
  __REG32 ENC8              : 1;
  __REG32 ENC9              : 1;
  __REG32 ENC10             : 1;
  __REG32 ENC11             : 1;
  __REG32 ENC12             : 1;
  __REG32 ENC13             : 1;
  __REG32 ENC14             : 1;
  __REG32 ENC15             : 1;
  __REG32 ENC16             : 1;
  __REG32 ENC17             : 1;
  __REG32 ENC18             : 1;
  __REG32 ENC19             : 1;
  __REG32 ENC20             : 1;
  __REG32 ENC21             : 1;
  __REG32 ENC22             : 1;
  __REG32 ENC23             : 1;
  __REG32 ENC24             : 1;
  __REG32 ENC25             : 1;
  __REG32 ENC26             : 1;
  __REG32 ENC27             : 1;
  __REG32 ENC28             : 1;
  __REG32 ENC29             : 1;
  __REG32 ENC30             : 1;
  __REG32 ENC31             : 1;
} __eicn_enicr_bits;

/* External Interrupt Request Register (EICn_EIRR) */
typedef struct {
  __REG32 ER0               : 1;
  __REG32 ER1               : 1;
  __REG32 ER2               : 1;
  __REG32 ER3               : 1;
  __REG32 ER4               : 1;
  __REG32 ER5               : 1;
  __REG32 ER6               : 1;
  __REG32 ER7               : 1;
  __REG32 ER8               : 1;
  __REG32 ER9               : 1;
  __REG32 ER10              : 1;
  __REG32 ER11              : 1;
  __REG32 ER12              : 1;
  __REG32 ER13              : 1;
  __REG32 ER14              : 1;
  __REG32 ER15              : 1;
  __REG32 ER16              : 1;
  __REG32 ER17              : 1;
  __REG32 ER18              : 1;
  __REG32 ER19              : 1;
  __REG32 ER20              : 1;
  __REG32 ER21              : 1;
  __REG32 ER22              : 1;
  __REG32 ER23              : 1;
  __REG32 ER24              : 1;
  __REG32 ER25              : 1;
  __REG32 ER26              : 1;
  __REG32 ER27              : 1;
  __REG32 ER28              : 1;
  __REG32 ER29              : 1;
  __REG32 ER30              : 1;
  __REG32 ER31              : 1;
} __eicn_eirr_bits;

/* External Interrupt Request Clear Register (EICn_EIRCR) */
typedef struct {
  __REG32 ERC0              : 1;
  __REG32 ERC1              : 1;
  __REG32 ERC2              : 1;
  __REG32 ERC3              : 1;
  __REG32 ERC4              : 1;
  __REG32 ERC5              : 1;
  __REG32 ERC6              : 1;
  __REG32 ERC7              : 1;
  __REG32 ERC8              : 1;
  __REG32 ERC9              : 1;
  __REG32 ERC10             : 1;
  __REG32 ERC11             : 1;
  __REG32 ERC12             : 1;
  __REG32 ERC13             : 1;
  __REG32 ERC14             : 1;
  __REG32 ERC15             : 1;
  __REG32 ERC16             : 1;
  __REG32 ERC17             : 1;
  __REG32 ERC18             : 1;
  __REG32 ERC19             : 1;
  __REG32 ERC20             : 1;
  __REG32 ERC21             : 1;
  __REG32 ERC22             : 1;
  __REG32 ERC23             : 1;
  __REG32 ERC24             : 1;
  __REG32 ERC25             : 1;
  __REG32 ERC26             : 1;
  __REG32 ERC27             : 1;
  __REG32 ERC28             : 1;
  __REG32 ERC29             : 1;
  __REG32 ERC30             : 1;
  __REG32 ERC31             : 1;
} __eicn_eircr_bits;

/* Noise Filter Enable Register (EICn_NFER) */
typedef struct {
  __REG32 NFE0              : 1;
  __REG32 NFE1              : 1;
  __REG32 NFE2              : 1;
  __REG32 NFE3              : 1;
  __REG32 NFE4              : 1;
  __REG32 NFE5              : 1;
  __REG32 NFE6              : 1;
  __REG32 NFE7              : 1;
  __REG32 NFE8              : 1;
  __REG32 NFE9              : 1;
  __REG32 NFE10             : 1;
  __REG32 NFE11             : 1;
  __REG32 NFE12             : 1;
  __REG32 NFE13             : 1;
  __REG32 NFE14             : 1;
  __REG32 NFE15             : 1;
  __REG32 NFE16             : 1;
  __REG32 NFE17             : 1;
  __REG32 NFE18             : 1;
  __REG32 NFE19             : 1;
  __REG32 NFE20             : 1;
  __REG32 NFE21             : 1;
  __REG32 NFE22             : 1;
  __REG32 NFE23             : 1;
  __REG32 NFE24             : 1;
  __REG32 NFE25             : 1;
  __REG32 NFE26             : 1;
  __REG32 NFE27             : 1;
  __REG32 NFE28             : 1;
  __REG32 NFE29             : 1;
  __REG32 NFE30             : 1;
  __REG32 NFE31             : 1;
} __eicn_nfer_bits;

/* Noise Filter Enable Set Register (EICn_NFESR) */
typedef struct {
  __REG32 NFES0             : 1;
  __REG32 NFES1             : 1;
  __REG32 NFES2             : 1;
  __REG32 NFES3             : 1;
  __REG32 NFES4             : 1;
  __REG32 NFES5             : 1;
  __REG32 NFES6             : 1;
  __REG32 NFES7             : 1;
  __REG32 NFES8             : 1;
  __REG32 NFES9             : 1;
  __REG32 NFES10            : 1;
  __REG32 NFES11            : 1;
  __REG32 NFES12            : 1;
  __REG32 NFES13            : 1;
  __REG32 NFES14            : 1;
  __REG32 NFES15            : 1;
  __REG32 NFES16            : 1;
  __REG32 NFES17            : 1;
  __REG32 NFES18            : 1;
  __REG32 NFES19            : 1;
  __REG32 NFES20            : 1;
  __REG32 NFES21            : 1;
  __REG32 NFES22            : 1;
  __REG32 NFES23            : 1;
  __REG32 NFES24            : 1;
  __REG32 NFES25            : 1;
  __REG32 NFES26            : 1;
  __REG32 NFES27            : 1;
  __REG32 NFES28            : 1;
  __REG32 NFES29            : 1;
  __REG32 NFES30            : 1;
  __REG32 NFES31            : 1;
} __eicn_nfesr_bits;

/* Noise Filter Enable Clear Register (EICn_NFECR) */
typedef struct {
  __REG32 NFEC0             : 1;
  __REG32 NFEC1             : 1;
  __REG32 NFEC2             : 1;
  __REG32 NFEC3             : 1;
  __REG32 NFEC4             : 1;
  __REG32 NFEC5             : 1;
  __REG32 NFEC6             : 1;
  __REG32 NFEC7             : 1;
  __REG32 NFEC8             : 1;
  __REG32 NFEC9             : 1;
  __REG32 NFEC10            : 1;
  __REG32 NFEC11            : 1;
  __REG32 NFEC12            : 1;
  __REG32 NFEC13            : 1;
  __REG32 NFEC14            : 1;
  __REG32 NFEC15            : 1;
  __REG32 NFEC16            : 1;
  __REG32 NFEC17            : 1;
  __REG32 NFEC18            : 1;
  __REG32 NFEC19            : 1;
  __REG32 NFEC20            : 1;
  __REG32 NFEC21            : 1;
  __REG32 NFEC22            : 1;
  __REG32 NFEC23            : 1;
  __REG32 NFEC24            : 1;
  __REG32 NFEC25            : 1;
  __REG32 NFEC26            : 1;
  __REG32 NFEC27            : 1;
  __REG32 NFEC28            : 1;
  __REG32 NFEC29            : 1;
  __REG32 NFEC30            : 1;
  __REG32 NFEC31            : 1;
} __eicn_nfecr_bits;

/* External Interrupt Level Register (EICn_ELVR0~3) */
typedef struct {
  __REG32 LA0               : 1;
  __REG32 LB0               : 1;
  __REG32 LC0               : 1;
  __REG32                   : 1;
  __REG32 LA1               : 1;
  __REG32 LB1               : 1;
  __REG32 LC1               : 1;
  __REG32                   : 1;
  __REG32 LA2               : 1;
  __REG32 LB2               : 1;
  __REG32 LC2               : 1;
  __REG32                   : 1;
  __REG32 LA3               : 1;
  __REG32 LB3               : 1;
  __REG32 LC3               : 1;
  __REG32                   : 1;
  __REG32 LA4               : 1;
  __REG32 LB4               : 1;
  __REG32 LC4               : 1;
  __REG32                   : 1;
  __REG32 LA5               : 1;
  __REG32 LB5               : 1;
  __REG32 LC5               : 1;
  __REG32                   : 1;
  __REG32 LA6               : 1;
  __REG32 LB6               : 1;
  __REG32 LC6               : 1;
  __REG32                   : 1;
  __REG32 LA7               : 1;
  __REG32 LB7               : 1;
  __REG32 LC7               : 1;
  __REG32                   : 1;
} __eicn_elvr_bits;

/* Non-Maskable Interrupt Register (EICn_NMIR) */
typedef struct {
  __REG32 NMIINT            : 1;
  __REG32                   : 7;
  __REG32 NMICLR            : 1;
  __REG32                   :23;
} __eicn_nmir_bits;

/* DMA Request Enable Register (EICn_DRER) */
typedef struct {
  __REG32 DRE0              : 1;
  __REG32 DRE1              : 1;
  __REG32 DRE2              : 1;
  __REG32 DRE3              : 1;
  __REG32 DRE4              : 1;
  __REG32 DRE5              : 1;
  __REG32 DRE6              : 1;
  __REG32 DRE7              : 1;
  __REG32 DRE8              : 1;
  __REG32 DRE9              : 1;
  __REG32 DRE10             : 1;
  __REG32 DRE11             : 1;
  __REG32 DRE12             : 1;
  __REG32 DRE13             : 1;
  __REG32 DRE14             : 1;
  __REG32 DRE15             : 1;
  __REG32 DRE16             : 1;
  __REG32 DRE17             : 1;
  __REG32 DRE18             : 1;
  __REG32 DRE19             : 1;
  __REG32 DRE20             : 1;
  __REG32 DRE21             : 1;
  __REG32 DRE22             : 1;
  __REG32 DRE23             : 1;
  __REG32 DRE24             : 1;
  __REG32 DRE25             : 1;
  __REG32 DRE26             : 1;
  __REG32 DRE27             : 1;
  __REG32 DRE28             : 1;
  __REG32 DRE29             : 1;
  __REG32 DRE30             : 1;
  __REG32 DRE31             : 1;
} __eicn_drer_bits;

/* DMA Request Enable Set Register (EICn_DRESR) */
typedef struct {
  __REG32 DRES0             : 1;
  __REG32 DRES1             : 1;
  __REG32 DRES2             : 1;
  __REG32 DRES3             : 1;
  __REG32 DRES4             : 1;
  __REG32 DRES5             : 1;
  __REG32 DRES6             : 1;
  __REG32 DRES7             : 1;
  __REG32 DRES8             : 1;
  __REG32 DRES9             : 1;
  __REG32 DRES10            : 1;
  __REG32 DRES11            : 1;
  __REG32 DRES12            : 1;
  __REG32 DRES13            : 1;
  __REG32 DRES14            : 1;
  __REG32 DRES15            : 1;
  __REG32 DRES16            : 1;
  __REG32 DRES17            : 1;
  __REG32 DRES18            : 1;
  __REG32 DRES19            : 1;
  __REG32 DRES20            : 1;
  __REG32 DRES21            : 1;
  __REG32 DRES22            : 1;
  __REG32 DRES23            : 1;
  __REG32 DRES24            : 1;
  __REG32 DRES25            : 1;
  __REG32 DRES26            : 1;
  __REG32 DRES27            : 1;
  __REG32 DRES28            : 1;
  __REG32 DRES29            : 1;
  __REG32 DRES30            : 1;
  __REG32 DRES31            : 1;
} __eicn_dresr_bits;

/* DMA Request Enable Clear Register (EICn_DRECR) */
typedef struct {
  __REG32 DREC0             : 1;
  __REG32 DREC1             : 1;
  __REG32 DREC2             : 1;
  __REG32 DREC3             : 1;
  __REG32 DREC4             : 1;
  __REG32 DREC5             : 1;
  __REG32 DREC6             : 1;
  __REG32 DREC7             : 1;
  __REG32 DREC8             : 1;
  __REG32 DREC9             : 1;
  __REG32 DREC10            : 1;
  __REG32 DREC11            : 1;
  __REG32 DREC12            : 1;
  __REG32 DREC13            : 1;
  __REG32 DREC14            : 1;
  __REG32 DREC15            : 1;
  __REG32 DREC16            : 1;
  __REG32 DREC17            : 1;
  __REG32 DREC18            : 1;
  __REG32 DREC19            : 1;
  __REG32 DREC20            : 1;
  __REG32 DREC21            : 1;
  __REG32 DREC22            : 1;
  __REG32 DREC23            : 1;
  __REG32 DREC24            : 1;
  __REG32 DREC25            : 1;
  __REG32 DREC26            : 1;
  __REG32 DREC27            : 1;
  __REG32 DREC28            : 1;
  __REG32 DREC29            : 1;
  __REG32 DREC30            : 1;
  __REG32 DREC31            : 1;
} __eicn_drecr_bits;

/* DMA Request Flag Register (EICn_DRFR) */
typedef struct {
  __REG32 DRF0              : 1;
  __REG32 DRF1              : 1;
  __REG32 DRF2              : 1;
  __REG32 DRF3              : 1;
  __REG32 DRF4              : 1;
  __REG32 DRF5              : 1;
  __REG32 DRF6              : 1;
  __REG32 DRF7              : 1;
  __REG32 DRF8              : 1;
  __REG32 DRF9              : 1;
  __REG32 DRF10             : 1;
  __REG32 DRF11             : 1;
  __REG32 DRF12             : 1;
  __REG32 DRF13             : 1;
  __REG32 DRF14             : 1;
  __REG32 DRF15             : 1;
  __REG32 DRF16             : 1;
  __REG32 DRF17             : 1;
  __REG32 DRF18             : 1;
  __REG32 DRF19             : 1;
  __REG32 DRF20             : 1;
  __REG32 DRF21             : 1;
  __REG32 DRF22             : 1;
  __REG32 DRF23             : 1;
  __REG32 DRF24             : 1;
  __REG32 DRF25             : 1;
  __REG32 DRF26             : 1;
  __REG32 DRF27             : 1;
  __REG32 DRF28             : 1;
  __REG32 DRF29             : 1;
  __REG32 DRF30             : 1;
  __REG32 DRF31             : 1;
} __eicn_drfr_bits;

/* Sound Generator Control Register 0 (SGn_CR0) */
typedef struct {
  __REG16 START             : 1;
  __REG16 STOP              : 1;
  __REG16 RESUME            : 1;
  __REG16 RUNNING           : 1;
  __REG16 ZAICLR            : 1;
  __REG16 TCICLR            : 1;
  __REG16 AMICLR            : 1;
  __REG16                   : 1;
  __REG16 S                 : 4;
  __REG16 SGDADS            : 1;
  __REG16 DMA               : 1;
  __REG16 FSEL              : 1;
  __REG16 DBGE              : 1;
} __sgn_cr0_bits;

/* Sound Generator Control Register 1 (SGn_CR1) */
typedef struct {
  __REG8 SGAOE             : 1;
  __REG8 SGOOE             : 1;
  __REG8 AMP               : 1;
  __REG8 TONE              : 1;
  __REG8 ZAINT             : 1;
  __REG8 TCINT             : 1;
  __REG8 AMINT             : 1;
  __REG8                   : 1;
} __sgn_cr1_bits;

/* Sound Generator Extended Control Reload Register (SGn_ECRL) */
typedef struct {
  __REG16 ZADMAE            : 1;
  __REG16 TCDMAE            : 1;
  __REG16 AMDMAE            : 1;
  __REG16                   : 1;
  __REG16 ZAIE              : 1;
  __REG16 TCIE              : 1;
  __REG16 AMIE              : 1;
  __REG16                   : 1;
  __REG16 ZARLE             : 1;
  __REG16 TCRLE             : 1;
  __REG16 AMRLE             : 1;
  __REG16                   : 1;
  __REG16 SGDADR            : 1;
  __REG16 ELS               : 1;
  __REG16 IDS               : 1;
  __REG16 AUTO              : 1;
} __sgn_ecrl_bits;

/* Sound Generator Frequency Data Reload Register (SGn_FRL) */
typedef struct {
  __REG16 SGFRL             :15;
  __REG16                   : 1;
} __sgn_frl_bits;

/* Sound Generator Frequency Data Reload Register (SGn_FRL) */
typedef struct {
  __REG16 SGAR              : 9;
  __REG16                   : 7;
} __sgn_arl_bits;

/* Sound Generator Amplitude Data Reload Register (SGn_ARL) */
typedef struct {
  __REG16 SGARL             : 9;
  __REG16                   : 7;
} __sgn_ar_bits;

/* Sound Generator Time Cycle Data Reload Register & Increment
   or Decrement Data Reload Register (SGn_TCRLIDRL) */
typedef struct {
  __REG16 SGIDRL            : 8;
  __REG16 SGTCRL            : 8;
} __sgn_tcrlidrl_bits;

/* Sound Generator Target Amplitude Data Reload Register (SGn_TARL) */
typedef struct {
  __REG16 SGTARL            : 9;
  __REG16                   : 7;
} __sgn_tarl_bits;

/* Sound Generator DMA Transfer Update Enable Register (SGn_DER) */
typedef struct {
  __REG16 CRE0              : 1;
  __REG16 CRE1              : 1;
  __REG16 FRE0              : 1;
  __REG16 FRE1              : 1;
  __REG16 ARE0              : 1;
  __REG16 ARE1              : 1;
  __REG16 TARE0             : 1;
  __REG16 TARE1             : 1;
  __REG16 IDRE              : 1;
  __REG16 TCRE              : 1;
  __REG16 NRE               : 1;
  __REG16                   : 5;
} __sgn_der_bits;

/* Control Status Register (FRTn_TCCS) */
typedef struct {
  __REG16                   : 4;
  __REG16 MODE              : 1;
  __REG16                   : 1;
  __REG16 IVFE              : 1;
  __REG16 IVF               : 1;
  __REG16 CLR               : 1;
  __REG16 IVFCLR            : 1;
  __REG16 ZFCLR             : 1;
  __REG16                   : 5;
} __frtn_tccs_bits;

/* Timer Stop/Timer Clock Configuration Register (FRTn_TSTPTCLK) */
typedef struct {
  __REG16 STOP              : 1;
  __REG16                   : 7;
  __REG16 CLK               : 4;
  __REG16                   : 2;
  __REG16 FSEL              : 1;
  __REG16 ECKE              : 1;
} __frtn_tstptclk_bits;

/* Extended Control Status Register (FRTn_ETCCS) */
typedef struct {
  __REG16 CNTMD             : 1;
  __REG16 IRQZE             : 1;
  __REG16 IRQZF             : 1;
  __REG16 BFE               : 1;
  __REG16 ICUR              : 1;
  __REG16 CNTDIR            : 1;
  __REG16 DBGE              : 1;
  __REG16                   : 1;
  __REG16 ZIMC              : 3;
  __REG16                   : 1;
  __REG16 CIMC              : 3;
  __REG16                   : 1;
} __frtn_etccs_bits;

/* Compare/Zero-Interrupt Mask Register (FRTn_CIMSZIMS) */
typedef struct {
  __REG16 CIMS              : 3;
  __REG16                   : 5;
  __REG16 ZIMS              : 3;
  __REG16                   : 5;
} __frtn_cimszims_bits;

/* DMA Configuration Register (FRTn_DMACFG) */
typedef struct {
  __REG16 EN_DMA_ZD         : 1;
  __REG16 EN_DMA_CCM        : 1;
  __REG16                   :14;
} __frtn_dmacfg_bits;

/* Control Register (OCUn_OCS01) */
typedef struct {
  __REG16 CST0              : 1;
  __REG16 CST1              : 1;
  __REG16                   : 2;
  __REG16 ICE0              : 1;
  __REG16 ICE1              : 1;
  __REG16                   : 2;
  __REG16 OTD0              : 1;
  __REG16 OTD1              : 1;
  __REG16                   : 2;
  __REG16 CMOD0             : 1;
  __REG16                   : 2;
  __REG16 CMOD1             : 1;
} __ocun_ocs01_bits;

/* Control Set Register (OCUn_OCSS01) */
typedef struct {
  __REG16 CSTS0             : 1;
  __REG16 CSTS1             : 1;
  __REG16                   : 2;
  __REG16 ICES0             : 1;
  __REG16 ICES1             : 1;
  __REG16                   : 2;
  __REG16 OTDS0             : 1;
  __REG16 OTDS1             : 1;
  __REG16                   : 6;
} __ocun_ocss01_bits;

/* Control Clear Register (OCUn_OCSC01) */
typedef struct {
  __REG16 CSTC0             : 1;
  __REG16 CSTC1             : 1;
  __REG16                   : 2;
  __REG16 ICEC0             : 1;
  __REG16 ICEC1             : 1;
  __REG16                   : 2;
  __REG16 OTDC0             : 1;
  __REG16 OTDC1             : 1;
  __REG16                   : 6;
} __ocun_ocsc01_bits;

/* Status Register (OCUn_OSR01) */
typedef struct {
  __REG16 ICP0              : 1;
  __REG16 ICP1              : 1;
  __REG16                   :14;
} __ocun_osr01_bits;

/* Status Clear Register (OCUn_OSCR01) */
typedef struct {
  __REG16 ICPC0             : 1;
  __REG16 ICPC1             : 1;
  __REG16                   :14;
} __ocun_oscr01_bits;

/* Extended Output Compare Control Status Register (OCUn_EOCS01) */
typedef struct {
  __REG16 BUF0              : 1;
  __REG16 BTS0              : 1;
  __REG16 BTS1              : 1;
  __REG16                   : 1;
  __REG16 BUF1              : 1;
  __REG16 BTS2              : 1;
  __REG16 BTS3              : 1;
  __REG16                   : 1;
  __REG16 OFM0              : 1;
  __REG16 OFD0              : 1;
  __REG16                   : 2;
  __REG16 OFM1              : 1;
  __REG16 OFD1              : 1;
  __REG16                   : 2;
} __ocun_eocs01_bits;

/* Extended Output Compare Control Status Set Register High (OCUn_EOCSSH01) */
typedef struct {
  __REG8  OFMS0             : 1;
  __REG8  OFDS0             : 1;
  __REG8                    : 2;
  __REG8  OFMS1             : 1;
  __REG8  OFDS1             : 1;
  __REG8                    : 2;
} __ocun_eocssh01_bits;

/* Extended Output Compare Control Status Clear Register High (OCUn_EOCSCH01) */
typedef struct {
  __REG8  OFMC0             : 1;
  __REG8  OFDC0             : 1;
  __REG8                    : 2;
  __REG8  OFMC1             : 1;
  __REG8  OFDC1             : 1;
  __REG8                    : 2;
} __ocun_eocsch01_bits;

/* Debug Configuration Register (OCUn_DEBUG01) */
typedef struct {
  __REG8  DBGEN0            : 1;
  __REG8  DBGEN1            : 1;
  __REG8                    : 6;
} __ocun_debug01_bits;

/* DMA Configuration Register (OCUn_DMACFG01) */
typedef struct {
  __REG8  EN_DMA_REQ0       : 1;
  __REG8  EN_DMA_REQ1       : 1;
  __REG8                    : 6;
} __ocun_dmacfg01_bits;

/* Compare Mode Control Register (OCUn_OCMCR01) */
typedef struct {
  __REG8  CMPMD0            : 1;
  __REG8  INV0              : 1;
  __REG8  FDEN0             : 1;
  __REG8                    : 1;
  __REG8  CMPMD1            : 1;
  __REG8  INV1              : 1;
  __REG8  FDEN1             : 1;
  __REG8                    : 1;
} __ocun_ocmcr01_bits;

/* Input Capture Edge Register and Control Status Register (ICUn_ICEICS01) */
typedef struct {
  __REG16 EG0               : 2;
  __REG16 EG1               : 2;
  __REG16 ICE0              : 1;
  __REG16 ICE1              : 1;
  __REG16 ICP0              : 1;
  __REG16 ICP1              : 1;
  __REG16 IEI0              : 1;
  __REG16 IEI1              : 1;
  __REG16 NFE0              : 1;
  __REG16 NFE1              : 1;
  __REG16                   : 2;
  __REG16 IDSE0             : 1;
  __REG16 IDSE1             : 1;
} __icun_iceics01_bits;

/* Input Capture Clear Register (ICUn_ICC01) */
typedef struct {
  __REG16                   : 2;
  __REG16 NFEC0             : 1;
  __REG16 NFEC1             : 1;
  __REG16 ICEC0             : 1;
  __REG16 ICEC1             : 1;
  __REG16 ICC0              : 1;
  __REG16 ICC1              : 1;
  __REG16                   : 2;
  __REG16 NFES0             : 1;
  __REG16 NFES1             : 1;
  __REG16 ICES0             : 1;
  __REG16 ICES1             : 1;
  __REG16                   : 2;
} __icun_icc01_bits;

/* DMA Configuration Register (ICUn_DMACFG01) */
typedef struct {
  __REG8  EN_DMA_REQ0       : 1;
  __REG8  EN_DMA_REQ1       : 1;
  __REG8                    : 6;
} __icun_dmacfg01_bits;

/* Debug Register (ICUn_DEBUG01) */
typedef struct {
  __REG8  DBGEN0            : 1;
  __REG8  DBGEN1            : 1;
  __REG8                    : 6;
} __icun_debug01_bits;

/* PPG Control Status Register (PPGn_PCN) */
typedef struct {
  __REG16                   : 1;
  __REG16 IRS               : 3;
  __REG16 IRQF              : 1;
  __REG16 IREN              : 1;
  __REG16 EGS               : 2;
  __REG16 MOD               : 1;
  __REG16                   : 1;
  __REG16 CKS               : 2;
  __REG16 RTRG              : 1;
  __REG16 MDSE              : 1;
  __REG16                   : 2;
} __ppgn_pcn_bits;

/* Interrupt Flag Clear Register (PPGn_IRQCLR) */
typedef struct {
  __REG8  IRQCLR            : 1;
  __REG8                    : 7;
} __ppgn_irqclr_bits;

/* Software Trigger Activation Register (PPGn_SWTRIG) */
typedef struct {
  __REG8  STGR              : 1;
  __REG8                    : 7;
} __ppgn_swtrig_bits;

/* Output Enable Register (PPGn_OE) */
typedef struct {
  __REG8  OE                : 1;
  __REG8  OE2               : 1;
  __REG8                    : 6;
} __ppgn_oe_bits;

/* Timer Enable Operation Register (PPGn_CNTEN) */
typedef struct {
  __REG8  CNTE              : 1;
  __REG8                    : 7;
} __ppgn_cnten_bits;

/* Output Mask and Polarity Selection Register (PPGn_OPTMSK) */
typedef struct {
  __REG8  OSEL              : 1;
  __REG8  OSEL2             : 1;
  __REG8  PGMS              : 1;
  __REG8                    : 5;
} __ppgn_optmsk_bits;

/* Ramp Configuration Register (PPGn_RMPCFG) */
typedef struct {
  __REG8  RAMPL             : 1;
  __REG8  RAMPH             : 1;
  __REG8  RIDL              : 1;
  __REG8  RIDH              : 1;
  __REG8                    : 4;
} __ppgn_rmpcfg_bits;

/* Start Delay Mode Register (PPGn_STRD) */
typedef struct {
  __REG8  STRD              : 1;
  __REG8                    : 7;
} __ppgn_strd_bits;

/* PPG Trigger Clear Flag Register (PPGn_TRIGCLR) */
typedef struct {
  __REG8  TRGCLR            : 1;
  __REG8                    : 7;
} __ppgn_trigclr_bits;

/* Extended PPG Control Status Register 1 (PPGn_EPCN1) */
typedef struct {
  __REG16                   : 1;
  __REG16 TPCL              : 1;
  __REG16 TPCH              : 1;
  __REG16                   : 5;
  __REG16 FRML              : 1;
  __REG16 FRMH              : 1;
  __REG16                   : 2;
  __REG16 TRIG              : 1;
  __REG16                   : 3;
} __ppgn_epcn1_bits;

/* Extended PPG Control Status Register 2 (PPGn_EPCN2) */
typedef struct {
  __REG16 PRDL              : 1;
  __REG16 PRDH              : 1;
  __REG16 DTL               : 1;
  __REG16 DTH               : 1;
  __REG16 EDML              : 1;
  __REG16 EDMH              : 1;
  __REG16 TCL               : 1;
  __REG16 TCH               : 1;
  __REG16 PRDLCLR           : 1;
  __REG16 PRDHCLR           : 1;
  __REG16 DTLCLR            : 1;
  __REG16 DTHCLR            : 1;
  __REG16 EDMLCLR           : 1;
  __REG16 EDMHCLR           : 1;
  __REG16 TCLCLR            : 1;
  __REG16 TCHCLR            : 1;
} __ppgn_epcn2_bits;

/* General Control Register 1 (PPGn_GCN1) */
typedef struct {
  __REG8  TSEL              : 4;
  __REG8                    : 4;
} __ppgn_gcn1_bits;

/* General Control Register 3 (PPGn_GCN3) */
typedef struct {
  __REG8  RTG               : 3;
  __REG8                    : 5;
} __ppgn_gcn3_bits;

/* General Control Register 4 (PPGn_GCN4) */
typedef struct {
  __REG8  RCK               : 3;
  __REG8  CKSEL             : 1;
  __REG8                    : 4;
} __ppgn_gcn4_bits;

/* General Control Register 5 (PPGn_GCN5)*/
typedef struct {
  __REG8  RSH               : 2;
  __REG8                    : 6;
} __ppgn_gcn5_bits;

/* PPG Cycle Setting Register (PPGn_PCSR) */
typedef struct {
  __REG16 PCSRL             : 8;
  __REG16 PCSRH             : 8;
} __ppgn_pcsr_bits;

/* PPG Duty Setting Register (PPGn_PDUT) */
typedef struct {
  __REG16 PDUTL             : 8;
  __REG16 PDUTH             : 8;
} __ppgn_pdut_bits;

/* PPG Timer Register (PPGn_PTMR) */
typedef struct {
  __REG16 PTMRL             : 8;
  __REG16 PTMRH             : 8;
} __ppgn_ptmr_bits;

/* PPG Start Delay Register (PPGn_PSDR) */
typedef struct {
  __REG16 PSDRL             : 8;
  __REG16 PSDRH             : 8;
} __ppgn_psdr_bits;

/* PPG Timing Point Capture Register (PPGn_PTPC) */
typedef struct {
  __REG16 PTPCL             : 8;
  __REG16 PTPCH             : 8;
} __ppgn_ptpc_bits;

/* PPG End Duty Register (PPGn_PEDR) */
typedef struct {
  __REG16 PTPCL             : 8;
  __REG16 PTPCH             : 8;
} __ppgn_pedr_bits;

/* PPG DMA Configuration Register (PPGn_DMACFG) */
typedef struct {
  __REG8  ENDMAREQ          : 1;
  __REG8                    : 7;
} __ppgn_dmacfg_bits;

/* PPG Debug Enable Register (PPGn_DEBUG) */
typedef struct {
  __REG8  DBGEN             : 1;
  __REG8                    : 7;
} __ppgn_debug_bits;

/* Group Control Register (PPGGRPp_GCTRL) */
typedef struct {
  __REG8  EN0               : 1;
  __REG8  EN1               : 1;
  __REG8  EN2               : 1;
  __REG8  EN3               : 1;
  __REG8                    : 4;
} __ppggrpp_gctrl_bits;

/* PPG General Control Register (PPGGCLg_GCNR) */
typedef struct {
  __REG8  CTG0              : 1;
  __REG8  CTG1              : 1;
  __REG8                    : 6;
} __ppgglcg_gcnr_bits;

/* DMA Configuration Register (RLTn_DMACFG) */
typedef struct {
  __REG32 ENDMAUF           : 1;
  __REG32                   :31;
} __rltn_dmacfg_bits;

/* Timer Control Status Register (RLTn_TMCSR) */
typedef struct {
  __REG32                   : 3;
  __REG32 INTE              : 1;
  __REG32 RELD              : 1;
  __REG32 OUTL              : 1;
  __REG32 OUTE              : 1;
  __REG32 DBGE              : 1;
  __REG32 NFE               : 1;
  __REG32                   : 1;
  __REG32 CSL0              : 1;
  __REG32 CSL1              : 1;
  __REG32 CSL2              : 1;
  __REG32 MOD               : 3;
  __REG32 UF                : 1;
  __REG32 UFCLR             : 1;
  __REG32 TRG               : 1;
  __REG32                   : 5;
  __REG32 CNTE              : 1;
  __REG32                   : 7;
} __rltn_tmcsr_bits;

/* PWM Control Register (SMCn_PWC) */
typedef struct {
  __REG8                    : 2;
  __REG8  SC                : 1;
  __REG8  CE                : 1;
  __REG8  P                 : 3;
  __REG8                    : 1;
} __smcn_pwc_bits;

/* PWM Control Set Register (SMCn_PWCS) */
typedef struct {
  __REG8                    : 3;
  __REG8  CES               : 1;
  __REG8                    : 4;
} __smcn_pwcs_bits;

/* PWM Control Clear Register (SMCn_PWCC) */
typedef struct {
  __REG8                    : 3;
  __REG8  CEC               : 1;
  __REG8                    : 4;
} __smcn_pwcc_bits;

/* PWM Compare Registers (SMCn_PWC1, SMCn_PWC2) */
typedef struct {
  __REG16 D                 :10;
  __REG16                   : 6;
} __smcn_pwc1_bits;

/* PWM1 Selection Register (SMCn_PWS) */
typedef struct {
  __REG16 M1                : 3;
  __REG16 P1                : 3;
  __REG16                   : 2;
  __REG16 M2                : 3;
  __REG16 P2                : 3;
  __REG16 BS                : 1;
  __REG16                   : 1;
} __smcn_pws_bits;

/* PWM Selection Set Register (SMCn_PWSS) */
typedef struct {
  __REG16                   :14;
  __REG16 BS                : 1;
  __REG16                   : 1;
} __smcn_pwss_bits;

/* SMC Debug Register (SMCn_DEBUG) */
typedef struct {
  __REG8  DBGEN             : 1;
  __REG8                    : 7;
} __smcn_debug_bits;

/* SMC Trigger Selection Register (SMCTGg_PTRGS) */
typedef struct {
  __REG16 S10               : 1;
  __REG16 S11               : 1;
  __REG16 S12               : 1;
  __REG16 S13               : 1;
  __REG16 S14               : 1;
  __REG16 S15               : 1;
  __REG16                   : 2;
  __REG16 S20               : 1;
  __REG16 S21               : 1;
  __REG16 S22               : 1;
  __REG16 S23               : 1;
  __REG16 S24               : 1;
  __REG16 S25               : 1;
  __REG16                   : 2;
} __smctgg_ptrg0_bits;

/* SMC Trigger Register (SMCTGg_PTRG) */
typedef struct {
  __REG8  TR1               : 1;
  __REG8  TR2               : 1;
  __REG8                    : 6;
} __smctgg_ptrg1_bits;

/* CAN Output Enable Register (COERn) */
typedef struct {
  __REG8  OE                : 1;
  __REG8                    : 7;
} __can_coer_bits;

/* CAN Control Register (CTRLRn) */
typedef struct {
  __REG16 INIT              : 1;
  __REG16 IE                : 1;
  __REG16 SIE               : 1;
  __REG16 EIE               : 1;
  __REG16                   : 1;
  __REG16 DAR               : 1;
  __REG16 CCE               : 1;
  __REG16 TEST              : 1;
  __REG16                   : 8;
} __can_ctrlr_bits;

/* Status Register (STATRn) */
typedef struct {
  __REG16 LEC               : 3;
  __REG16 TXOK              : 1;
  __REG16 RXOK              : 1;
  __REG16 EPASS             : 1;
  __REG16 EWARN             : 1;
  __REG16 BOFF              : 1;
  __REG16                   : 8;
} __can_statr_bits;

/* Error Counter (ERRCNTn) */
typedef struct {
  __REG16 TEC               : 8;
  __REG16 REC               : 7;
  __REG16 RP                : 1;
} __can_errcnt_bits;

/* Bit Timing Register (BTRn) */
typedef struct {
  __REG16 BRP               : 6;
  __REG16 SJW               : 2;
  __REG16 TSEG1             : 4;
  __REG16 TSEG2             : 3;
  __REG16                   : 1;
} __can_btr_bits;

/* Test Register (TESTRn) */
typedef struct {
  __REG16                   : 2;
  __REG16 BASIC             : 1;
  __REG16 SILENT            : 1;
  __REG16 LBACK             : 1;
  __REG16 TX0               : 1;
  __REG16 TX1               : 1;
  __REG16 RX                : 1;
  __REG16                   : 8;
} __can_testr_bits;

/* BRP Extension Register (BRPERn) */
typedef struct {
  __REG16 BRPE              : 4;
  __REG16                   :12;
} __can_brper_bits;

/* IFx Command Request Registers (IFxCREQn) */
typedef struct {
  __REG16 MSGN              : 8;
  __REG16                   : 7;
  __REG16 BUSY              : 1;
} __can_ifcreq_bits;

/* IFx Command Mask Register (IFxCMSKn) */
typedef struct {
  __REG16 DATAB             : 1;
  __REG16 DATAA             : 1;
  __REG16 TXREQ             : 1;
  __REG16 CIP               : 1;
  __REG16 CONTROL           : 1;
  __REG16 ARB               : 1;
  __REG16 MASK              : 1;
  __REG16 WRRD              : 1;
  __REG16                   : 8;
} __can_ifcmsk_bits;

/* IFx Mask Registers (IFxMSK1n) */
typedef struct {
  __REG16 MSK0              : 1;
  __REG16 MSK1              : 1;
  __REG16 MSK2              : 1;
  __REG16 MSK3              : 1;
  __REG16 MSK4              : 1;
  __REG16 MSK5              : 1;
  __REG16 MSK6              : 1;
  __REG16 MSK7              : 1;
  __REG16 MSK8              : 1;
  __REG16 MSK9              : 1;
  __REG16 MSK10             : 1;
  __REG16 MSK11             : 1;
  __REG16 MSK12             : 1;
  __REG16 MSK13             : 1;
  __REG16 MSK14             : 1;
  __REG16 MSK15             : 1;
} __can_ifmsk1_bits;

/* IFx Mask Registers (IFxMSK2n) */
typedef struct {
  __REG16 MSK16             : 1;
  __REG16 MSK17             : 1;
  __REG16 MSK18             : 1;
  __REG16 MSK19             : 1;
  __REG16 MSK20             : 1;
  __REG16 MSK21             : 1;
  __REG16 MSK22             : 1;
  __REG16 MSK23             : 1;
  __REG16 MSK24             : 1;
  __REG16 MSK25             : 1;
  __REG16 MSK26             : 1;
  __REG16 MSK27             : 1;
  __REG16 MSK28             : 1;
  __REG16                   : 1;
  __REG16 MDIR              : 1;
  __REG16 MXTD              : 1;
} __can_ifmsk2_bits;

/* IFx Arbitration Registers (IFxARB1n) */
typedef struct {
  __REG16 ID0               : 1;
  __REG16 ID1               : 1;
  __REG16 ID2               : 1;
  __REG16 ID3               : 1;
  __REG16 ID4               : 1;
  __REG16 ID5               : 1;
  __REG16 ID6               : 1;
  __REG16 ID7               : 1;
  __REG16 ID8               : 1;
  __REG16 ID9               : 1;
  __REG16 ID10              : 1;
  __REG16 ID11              : 1;
  __REG16 ID12              : 1;
  __REG16 ID13              : 1;
  __REG16 ID14              : 1;
  __REG16 ID15              : 1;
} __can_ifarb1_bits;

/* IFx Arbitration Registers (IFxARB2n) */
typedef struct {
  __REG16 ID16              : 1;
  __REG16 ID17              : 1;
  __REG16 ID18              : 1;
  __REG16 ID19              : 1;
  __REG16 ID20              : 1;
  __REG16 ID21              : 1;
  __REG16 ID22              : 1;
  __REG16 ID23              : 1;
  __REG16 ID24              : 1;
  __REG16 ID25              : 1;
  __REG16 ID26              : 1;
  __REG16 ID27              : 1;
  __REG16 ID28              : 1;
  __REG16 DIR               : 1;
  __REG16 XTD               : 1;
  __REG16 MSGVAL            : 1;
} __can_ifarb2_bits;

/* IFx Message Control Register (IFxMCTRn) */
typedef struct {
  __REG16 DLC               : 4;
  __REG16                   : 3;
  __REG16 EOB               : 1;
  __REG16 TXRQST            : 1;
  __REG16 RMTEN             : 1;
  __REG16 RXIE              : 1;
  __REG16 TXIE              : 1;
  __REG16 UMASK             : 1;
  __REG16 INTPND            : 1;
  __REG16 MSGLST            : 1;
  __REG16 NEWDAT            : 1;
} __can_ifmctr_bits;

/* IFx Data A (IFxDTA1) */
typedef struct {
  __REG16 DATA0             : 8;
  __REG16 DATA1             : 8;
} __can_ifdta1_bits;

/* IFx Data A (IFxDTA2) */
typedef struct {
  __REG16 DATA2             : 8;
  __REG16 DATA3             : 8;
} __can_ifdta2_bits;

/* IFx Data B (IFxDTA1) */
typedef struct {
  __REG16 DATA4             : 8;
  __REG16 DATA5             : 8;
} __can_ifdtb1_bits;

/* IFx Data B (IFxDTA2) */
typedef struct {
  __REG16 DATA6             : 8;
  __REG16 DATA7             : 8;
} __can_ifdtb2_bits;

/* Interrupt Register (INTRn) */
typedef struct {
  __REG16 INTID0            : 1;
  __REG16 INTID1            : 1;
  __REG16 INTID2            : 1;
  __REG16 INTID3            : 1;
  __REG16 INTID4            : 1;
  __REG16 INTID5            : 1;
  __REG16 INTID6            : 1;
  __REG16 INTID7            : 1;
  __REG16 INTID8            : 1;
  __REG16 INTID9            : 1;
  __REG16 INTID10           : 1;
  __REG16 INTID11           : 1;
  __REG16 INTID12           : 1;
  __REG16 INTID13           : 1;
  __REG16 INTID14           : 1;
  __REG16 INTID15           : 1;
} __can_intr_bits;

/* Transmission Request Registers (TREQR1) */
typedef struct {
  __REG16 INTID1            : 1;
  __REG16 INTID2            : 1;
  __REG16 INTID3            : 1;
  __REG16 INTID4            : 1;
  __REG16 INTID5            : 1;
  __REG16 INTID6            : 1;
  __REG16 INTID7            : 1;
  __REG16 INTID8            : 1;
  __REG16 INTID9            : 1;
  __REG16 INTID10           : 1;
  __REG16 INTID11           : 1;
  __REG16 INTID12           : 1;
  __REG16 INTID13           : 1;
  __REG16 INTID14           : 1;
  __REG16 INTID15           : 1;
  __REG16 INTID16           : 1;
} __can_treqr1_bits;

/* Transmission Request Registers (TREQR2) */
typedef struct {
  __REG16 INTID17           : 1;
  __REG16 INTID18           : 1;
  __REG16 INTID19           : 1;
  __REG16 INTID20           : 1;
  __REG16 INTID21           : 1;
  __REG16 INTID22           : 1;
  __REG16 INTID23           : 1;
  __REG16 INTID24           : 1;
  __REG16 INTID25           : 1;
  __REG16 INTID26           : 1;
  __REG16 INTID27           : 1;
  __REG16 INTID28           : 1;
  __REG16 INTID29           : 1;
  __REG16 INTID30           : 1;
  __REG16 INTID31           : 1;
  __REG16 INTID32           : 1;
} __can_treqr2_bits;

/* Transmission Request Registers (TREQR3) */
typedef struct {
  __REG16 INTID33           : 1;
  __REG16 INTID34           : 1;
  __REG16 INTID35           : 1;
  __REG16 INTID36           : 1;
  __REG16 INTID37           : 1;
  __REG16 INTID38           : 1;
  __REG16 INTID39           : 1;
  __REG16 INTID40           : 1;
  __REG16 INTID41           : 1;
  __REG16 INTID42           : 1;
  __REG16 INTID43           : 1;
  __REG16 INTID44           : 1;
  __REG16 INTID45           : 1;
  __REG16 INTID46           : 1;
  __REG16 INTID47           : 1;
  __REG16 INTID48           : 1;
} __can_treqr3_bits;

/* Transmission Request Registers (TREQR4) */
typedef struct {
  __REG16 INTID49           : 1;
  __REG16 INTID50           : 1;
  __REG16 INTID51           : 1;
  __REG16 INTID52           : 1;
  __REG16 INTID53           : 1;
  __REG16 INTID54           : 1;
  __REG16 INTID55           : 1;
  __REG16 INTID56           : 1;
  __REG16 INTID57           : 1;
  __REG16 INTID58           : 1;
  __REG16 INTID59           : 1;
  __REG16 INTID60           : 1;
  __REG16 INTID61           : 1;
  __REG16 INTID62           : 1;
  __REG16 INTID63           : 1;
  __REG16 INTID64           : 1;
} __can_treqr4_bits;

/* Transmission Request Registers (TREQR5) */
typedef struct {
  __REG16 INTID65           : 1;
  __REG16 INTID66           : 1;
  __REG16 INTID67           : 1;
  __REG16 INTID68           : 1;
  __REG16 INTID69           : 1;
  __REG16 INTID70           : 1;
  __REG16 INTID71           : 1;
  __REG16 INTID72           : 1;
  __REG16 INTID73           : 1;
  __REG16 INTID74           : 1;
  __REG16 INTID75           : 1;
  __REG16 INTID76           : 1;
  __REG16 INTID77           : 1;
  __REG16 INTID78           : 1;
  __REG16 INTID79           : 1;
  __REG16 INTID80           : 1;
} __can_treqr5_bits;

/* Transmission Request Registers (TREQR6) */
typedef struct {
  __REG16 INTID81           : 1;
  __REG16 INTID82           : 1;
  __REG16 INTID83           : 1;
  __REG16 INTID84           : 1;
  __REG16 INTID85           : 1;
  __REG16 INTID86           : 1;
  __REG16 INTID87           : 1;
  __REG16 INTID88           : 1;
  __REG16 INTID89           : 1;
  __REG16 INTID90           : 1;
  __REG16 INTID91           : 1;
  __REG16 INTID92           : 1;
  __REG16 INTID93           : 1;
  __REG16 INTID94           : 1;
  __REG16 INTID95           : 1;
  __REG16 INTID96           : 1;
} __can_treqr6_bits;

/* Transmission Request Registers (TREQR7) */
typedef struct {
  __REG16 INTID97           : 1;
  __REG16 INTID98           : 1;
  __REG16 INTID99           : 1;
  __REG16 INTID100          : 1;
  __REG16 INTID101          : 1;
  __REG16 INTID102          : 1;
  __REG16 INTID103          : 1;
  __REG16 INTID104          : 1;
  __REG16 INTID105          : 1;
  __REG16 INTID106          : 1;
  __REG16 INTID107          : 1;
  __REG16 INTID108          : 1;
  __REG16 INTID109          : 1;
  __REG16 INTID110          : 1;
  __REG16 INTID111          : 1;
  __REG16 INTID112          : 1;
} __can_treqr7_bits;

/* Transmission Request Registers (TREQR8) */
typedef struct {
  __REG16 INTID113          : 1;
  __REG16 INTID114          : 1;
  __REG16 INTID115          : 1;
  __REG16 INTID116          : 1;
  __REG16 INTID117          : 1;
  __REG16 INTID118          : 1;
  __REG16 INTID119          : 1;
  __REG16 INTID120          : 1;
  __REG16 INTID121          : 1;
  __REG16 INTID122          : 1;
  __REG16 INTID123          : 1;
  __REG16 INTID124          : 1;
  __REG16 INTID125          : 1;
  __REG16 INTID126          : 1;
  __REG16 INTID127          : 1;
  __REG16 INTID128          : 1;
} __can_treqr8_bits;

/* New Data Registers (NEWDT1) */
typedef struct {
  __REG16 NEWDATA1            : 1;
  __REG16 NEWDATA2            : 1;
  __REG16 NEWDATA3            : 1;
  __REG16 NEWDATA4            : 1;
  __REG16 NEWDATA5            : 1;
  __REG16 NEWDATA6            : 1;
  __REG16 NEWDATA7            : 1;
  __REG16 NEWDATA8            : 1;
  __REG16 NEWDATA9            : 1;
  __REG16 NEWDATA10           : 1;
  __REG16 NEWDATA11           : 1;
  __REG16 NEWDATA12           : 1;
  __REG16 NEWDATA13           : 1;
  __REG16 NEWDATA14           : 1;
  __REG16 NEWDATA15           : 1;
  __REG16 NEWDATA16           : 1;
} __can_newdt1_bits;

/* New Data Registers (NEWDT2) */
typedef struct {
  __REG16 NEWDATA17           : 1;
  __REG16 NEWDATA18           : 1;
  __REG16 NEWDATA19           : 1;
  __REG16 NEWDATA20           : 1;
  __REG16 NEWDATA21           : 1;
  __REG16 NEWDATA22           : 1;
  __REG16 NEWDATA23           : 1;
  __REG16 NEWDATA24           : 1;
  __REG16 NEWDATA25           : 1;
  __REG16 NEWDATA26           : 1;
  __REG16 NEWDATA27           : 1;
  __REG16 NEWDATA28           : 1;
  __REG16 NEWDATA29           : 1;
  __REG16 NEWDATA30           : 1;
  __REG16 NEWDATA31           : 1;
  __REG16 NEWDATA32           : 1;
} __can_newdt2_bits;

/* New Data Registers (NEWDT3) */
typedef struct {
  __REG16 NEWDATA33           : 1;
  __REG16 NEWDATA34           : 1;
  __REG16 NEWDATA35           : 1;
  __REG16 NEWDATA36           : 1;
  __REG16 NEWDATA37           : 1;
  __REG16 NEWDATA38           : 1;
  __REG16 NEWDATA39           : 1;
  __REG16 NEWDATA40           : 1;
  __REG16 NEWDATA41           : 1;
  __REG16 NEWDATA42           : 1;
  __REG16 NEWDATA43           : 1;
  __REG16 NEWDATA44           : 1;
  __REG16 NEWDATA45           : 1;
  __REG16 NEWDATA46           : 1;
  __REG16 NEWDATA47           : 1;
  __REG16 NEWDATA48           : 1;
} __can_newdt3_bits;

/* New Data Registers (NEWDT4) */
typedef struct {
  __REG16 NEWDATA49           : 1;
  __REG16 NEWDATA50           : 1;
  __REG16 NEWDATA51           : 1;
  __REG16 NEWDATA52           : 1;
  __REG16 NEWDATA53           : 1;
  __REG16 NEWDATA54           : 1;
  __REG16 NEWDATA55           : 1;
  __REG16 NEWDATA56           : 1;
  __REG16 NEWDATA57           : 1;
  __REG16 NEWDATA58           : 1;
  __REG16 NEWDATA59           : 1;
  __REG16 NEWDATA60           : 1;
  __REG16 NEWDATA61           : 1;
  __REG16 NEWDATA62           : 1;
  __REG16 NEWDATA63           : 1;
  __REG16 NEWDATA64           : 1;
} __can_newdt4_bits;

/* New Data Registers (NEWDT5) */
typedef struct {
  __REG16 NEWDATA65           : 1;
  __REG16 NEWDATA66           : 1;
  __REG16 NEWDATA67           : 1;
  __REG16 NEWDATA68           : 1;
  __REG16 NEWDATA69           : 1;
  __REG16 NEWDATA70           : 1;
  __REG16 NEWDATA71           : 1;
  __REG16 NEWDATA72           : 1;
  __REG16 NEWDATA73           : 1;
  __REG16 NEWDATA74           : 1;
  __REG16 NEWDATA75           : 1;
  __REG16 NEWDATA76           : 1;
  __REG16 NEWDATA77           : 1;
  __REG16 NEWDATA78           : 1;
  __REG16 NEWDATA79           : 1;
  __REG16 NEWDATA80           : 1;
} __can_newdt5_bits;

/* New Data Registers (NEWDT6) */
typedef struct {
  __REG16 NEWDATA81           : 1;
  __REG16 NEWDATA82           : 1;
  __REG16 NEWDATA83           : 1;
  __REG16 NEWDATA84           : 1;
  __REG16 NEWDATA85           : 1;
  __REG16 NEWDATA86           : 1;
  __REG16 NEWDATA87           : 1;
  __REG16 NEWDATA88           : 1;
  __REG16 NEWDATA89           : 1;
  __REG16 NEWDATA90           : 1;
  __REG16 NEWDATA91           : 1;
  __REG16 NEWDATA92           : 1;
  __REG16 NEWDATA93           : 1;
  __REG16 NEWDATA94           : 1;
  __REG16 NEWDATA95           : 1;
  __REG16 NEWDATA96           : 1;
} __can_newdt6_bits;

/* New Data Registers (NEWDT7) */
typedef struct {
  __REG16 NEWDATA97           : 1;
  __REG16 NEWDATA98           : 1;
  __REG16 NEWDATA99           : 1;
  __REG16 NEWDATA100          : 1;
  __REG16 NEWDATA101          : 1;
  __REG16 NEWDATA102          : 1;
  __REG16 NEWDATA103          : 1;
  __REG16 NEWDATA104          : 1;
  __REG16 NEWDATA105          : 1;
  __REG16 NEWDATA106          : 1;
  __REG16 NEWDATA107          : 1;
  __REG16 NEWDATA108          : 1;
  __REG16 NEWDATA109          : 1;
  __REG16 NEWDATA110          : 1;
  __REG16 NEWDATA111          : 1;
  __REG16 NEWDATA112          : 1;
} __can_newdt7_bits;

/* New Data Registers (NEWDT8) */
typedef struct {
  __REG16 NEWDATA113          : 1;
  __REG16 NEWDATA114          : 1;
  __REG16 NEWDATA115          : 1;
  __REG16 NEWDATA116          : 1;
  __REG16 NEWDATA117          : 1;
  __REG16 NEWDATA118          : 1;
  __REG16 NEWDATA119          : 1;
  __REG16 NEWDATA120          : 1;
  __REG16 NEWDATA121          : 1;
  __REG16 NEWDATA122          : 1;
  __REG16 NEWDATA123          : 1;
  __REG16 NEWDATA124          : 1;
  __REG16 NEWDATA125          : 1;
  __REG16 NEWDATA126          : 1;
  __REG16 NEWDATA127          : 1;
  __REG16 NEWDATA128          : 1;
} __can_newdt8_bits;

/* New Data Registers (NEWDT1) */
typedef struct {
  __REG16 INTPND1           : 1;
  __REG16 INTPND2           : 1;
  __REG16 INTPND3           : 1;
  __REG16 INTPND4           : 1;
  __REG16 INTPND5           : 1;
  __REG16 INTPND6           : 1;
  __REG16 INTPND7           : 1;
  __REG16 INTPND8           : 1;
  __REG16 INTPND9           : 1;
  __REG16 INTPND10          : 1;
  __REG16 INTPND11          : 1;
  __REG16 INTPND12          : 1;
  __REG16 INTPND13          : 1;
  __REG16 INTPND14          : 1;
  __REG16 INTPND15          : 1;
  __REG16 INTPND16          : 1;
} __can_intpnd1_bits;

/* Interrupt Pending Registers (INTPND2) */
typedef struct {
  __REG16 INTPND17            : 1;
  __REG16 INTPND18            : 1;
  __REG16 INTPND19            : 1;
  __REG16 INTPND20            : 1;
  __REG16 INTPND21            : 1;
  __REG16 INTPND22            : 1;
  __REG16 INTPND23            : 1;
  __REG16 INTPND24            : 1;
  __REG16 INTPND25            : 1;
  __REG16 INTPND26            : 1;
  __REG16 INTPND27            : 1;
  __REG16 INTPND28            : 1;
  __REG16 INTPND29            : 1;
  __REG16 INTPND30            : 1;
  __REG16 INTPND31            : 1;
  __REG16 INTPND32            : 1;
} __can_intpnd2_bits;

/* Interrupt Pending Registers (INTPND3) */
typedef struct {
  __REG16 INTPND33            : 1;
  __REG16 INTPND34            : 1;
  __REG16 INTPND35            : 1;
  __REG16 INTPND36            : 1;
  __REG16 INTPND37            : 1;
  __REG16 INTPND38            : 1;
  __REG16 INTPND39            : 1;
  __REG16 INTPND40            : 1;
  __REG16 INTPND41            : 1;
  __REG16 INTPND42            : 1;
  __REG16 INTPND43            : 1;
  __REG16 INTPND44            : 1;
  __REG16 INTPND45            : 1;
  __REG16 INTPND46            : 1;
  __REG16 INTPND47            : 1;
  __REG16 INTPND48            : 1;
} __can_intpnd3_bits;

/* Interrupt Pending Registers (INTPND4) */
typedef struct {
  __REG16 INTPND49            : 1;
  __REG16 INTPND50            : 1;
  __REG16 INTPND51            : 1;
  __REG16 INTPND52            : 1;
  __REG16 INTPND53            : 1;
  __REG16 INTPND54            : 1;
  __REG16 INTPND55            : 1;
  __REG16 INTPND56            : 1;
  __REG16 INTPND57            : 1;
  __REG16 INTPND58            : 1;
  __REG16 INTPND59            : 1;
  __REG16 INTPND60            : 1;
  __REG16 INTPND61            : 1;
  __REG16 INTPND62            : 1;
  __REG16 INTPND63            : 1;
  __REG16 INTPND64            : 1;
} __can_intpnd4_bits;

/* Interrupt Pending Registers (INTPND5) */
typedef struct {
  __REG16 INTPND65            : 1;
  __REG16 INTPND66            : 1;
  __REG16 INTPND67            : 1;
  __REG16 INTPND68            : 1;
  __REG16 INTPND69            : 1;
  __REG16 INTPND70            : 1;
  __REG16 INTPND71            : 1;
  __REG16 INTPND72            : 1;
  __REG16 INTPND73            : 1;
  __REG16 INTPND74            : 1;
  __REG16 INTPND75            : 1;
  __REG16 INTPND76            : 1;
  __REG16 INTPND77            : 1;
  __REG16 INTPND78            : 1;
  __REG16 INTPND79            : 1;
  __REG16 INTPND80            : 1;
} __can_intpnd5_bits;

/* Interrupt Pending Registers (INTPND6) */
typedef struct {
  __REG16 INTPND81            : 1;
  __REG16 INTPND82            : 1;
  __REG16 INTPND83            : 1;
  __REG16 INTPND84            : 1;
  __REG16 INTPND85            : 1;
  __REG16 INTPND86            : 1;
  __REG16 INTPND87            : 1;
  __REG16 INTPND88            : 1;
  __REG16 INTPND89            : 1;
  __REG16 INTPND90            : 1;
  __REG16 INTPND91            : 1;
  __REG16 INTPND92            : 1;
  __REG16 INTPND93            : 1;
  __REG16 INTPND94            : 1;
  __REG16 INTPND95            : 1;
  __REG16 INTPND96            : 1;
} __can_intpnd6_bits;

/* Interrupt Pending Registers (INTPND7) */
typedef struct {
  __REG16 INTPND97          : 1;
  __REG16 INTPND98          : 1;
  __REG16 INTPND99          : 1;
  __REG16 INTPND100         : 1;
  __REG16 INTPND101         : 1;
  __REG16 INTPND102         : 1;
  __REG16 INTPND103         : 1;
  __REG16 INTPND104         : 1;
  __REG16 INTPND105         : 1;
  __REG16 INTPND106         : 1;
  __REG16 INTPND107         : 1;
  __REG16 INTPND108         : 1;
  __REG16 INTPND109         : 1;
  __REG16 INTPND110         : 1;
  __REG16 INTPND111         : 1;
  __REG16 INTPND112         : 1;
} __can_intpnd7_bits;

/* Interrupt Pending Registers (INTPND8) */
typedef struct {
  __REG16 INTPND113         : 1;
  __REG16 INTPND114         : 1;
  __REG16 INTPND115         : 1;
  __REG16 INTPND116         : 1;
  __REG16 INTPND117         : 1;
  __REG16 INTPND118         : 1;
  __REG16 INTPND119         : 1;
  __REG16 INTPND120         : 1;
  __REG16 INTPND121         : 1;
  __REG16 INTPND122         : 1;
  __REG16 INTPND123         : 1;
  __REG16 INTPND124         : 1;
  __REG16 INTPND125         : 1;
  __REG16 INTPND126         : 1;
  __REG16 INTPND127         : 1;
  __REG16 INTPND128         : 1;
} __can_intpnd8_bits;

/* Message Valid Registers (MSGVAL1) */
typedef struct {
  __REG16 MSGVAL1           : 1;
  __REG16 MSGVAL2           : 1;
  __REG16 MSGVAL3           : 1;
  __REG16 MSGVAL4           : 1;
  __REG16 MSGVAL5           : 1;
  __REG16 MSGVAL6           : 1;
  __REG16 MSGVAL7           : 1;
  __REG16 MSGVAL8           : 1;
  __REG16 MSGVAL9           : 1;
  __REG16 MSGVAL10          : 1;
  __REG16 MSGVAL11          : 1;
  __REG16 MSGVAL12          : 1;
  __REG16 MSGVAL13          : 1;
  __REG16 MSGVAL14          : 1;
  __REG16 MSGVAL15          : 1;
  __REG16 MSGVAL16          : 1;
} __can_msgval1_bits;

/* Message Valid Registers (MSGVAL2) */
typedef struct {
  __REG16 MSGVAL17            : 1;
  __REG16 MSGVAL18            : 1;
  __REG16 MSGVAL19            : 1;
  __REG16 MSGVAL20            : 1;
  __REG16 MSGVAL21            : 1;
  __REG16 MSGVAL22            : 1;
  __REG16 MSGVAL23            : 1;
  __REG16 MSGVAL24            : 1;
  __REG16 MSGVAL25            : 1;
  __REG16 MSGVAL26            : 1;
  __REG16 MSGVAL27            : 1;
  __REG16 MSGVAL28            : 1;
  __REG16 MSGVAL29            : 1;
  __REG16 MSGVAL30            : 1;
  __REG16 MSGVAL31            : 1;
  __REG16 MSGVAL32            : 1;
} __can_msgval2_bits;

/* Message Valid Registers (MSGVAL3) */
typedef struct {
  __REG16 MSGVAL33            : 1;
  __REG16 MSGVAL34            : 1;
  __REG16 MSGVAL35            : 1;
  __REG16 MSGVAL36            : 1;
  __REG16 MSGVAL37            : 1;
  __REG16 MSGVAL38            : 1;
  __REG16 MSGVAL39            : 1;
  __REG16 MSGVAL40            : 1;
  __REG16 MSGVAL41            : 1;
  __REG16 MSGVAL42            : 1;
  __REG16 MSGVAL43            : 1;
  __REG16 MSGVAL44            : 1;
  __REG16 MSGVAL45            : 1;
  __REG16 MSGVAL46            : 1;
  __REG16 MSGVAL47            : 1;
  __REG16 MSGVAL48            : 1;
} __can_msgval3_bits;

/* Message Valid Registers (MSGVAL4) */
typedef struct {
  __REG16 MSGVAL49            : 1;
  __REG16 MSGVAL50            : 1;
  __REG16 MSGVAL51            : 1;
  __REG16 MSGVAL52            : 1;
  __REG16 MSGVAL53            : 1;
  __REG16 MSGVAL54            : 1;
  __REG16 MSGVAL55            : 1;
  __REG16 MSGVAL56            : 1;
  __REG16 MSGVAL57            : 1;
  __REG16 MSGVAL58            : 1;
  __REG16 MSGVAL59            : 1;
  __REG16 MSGVAL60            : 1;
  __REG16 MSGVAL61            : 1;
  __REG16 MSGVAL62            : 1;
  __REG16 MSGVAL63            : 1;
  __REG16 MSGVAL64            : 1;
} __can_msgval4_bits;

/* Message Valid Registers (MSGVAL5) */
typedef struct {
  __REG16 MSGVAL65            : 1;
  __REG16 MSGVAL66            : 1;
  __REG16 MSGVAL67            : 1;
  __REG16 MSGVAL68            : 1;
  __REG16 MSGVAL69            : 1;
  __REG16 MSGVAL70            : 1;
  __REG16 MSGVAL71            : 1;
  __REG16 MSGVAL72            : 1;
  __REG16 MSGVAL73            : 1;
  __REG16 MSGVAL74            : 1;
  __REG16 MSGVAL75            : 1;
  __REG16 MSGVAL76            : 1;
  __REG16 MSGVAL77            : 1;
  __REG16 MSGVAL78            : 1;
  __REG16 MSGVAL79            : 1;
  __REG16 MSGVAL80            : 1;
} __can_msgval5_bits;

/* Message Valid Registers (MSGVAL6) */
typedef struct {
  __REG16 MSGVAL81            : 1;
  __REG16 MSGVAL82            : 1;
  __REG16 MSGVAL83            : 1;
  __REG16 MSGVAL84            : 1;
  __REG16 MSGVAL85            : 1;
  __REG16 MSGVAL86            : 1;
  __REG16 MSGVAL87            : 1;
  __REG16 MSGVAL88            : 1;
  __REG16 MSGVAL89            : 1;
  __REG16 MSGVAL90            : 1;
  __REG16 MSGVAL91            : 1;
  __REG16 MSGVAL92            : 1;
  __REG16 MSGVAL93            : 1;
  __REG16 MSGVAL94            : 1;
  __REG16 MSGVAL95            : 1;
  __REG16 MSGVAL96            : 1;
} __can_msgval6_bits;

/* Message Valid Registers (MSGVAL7) */
typedef struct {
  __REG16 MSGVAL97          : 1;
  __REG16 MSGVAL98          : 1;
  __REG16 MSGVAL99          : 1;
  __REG16 MSGVAL100         : 1;
  __REG16 MSGVAL101         : 1;
  __REG16 MSGVAL102         : 1;
  __REG16 MSGVAL103         : 1;
  __REG16 MSGVAL104         : 1;
  __REG16 MSGVAL105         : 1;
  __REG16 MSGVAL106         : 1;
  __REG16 MSGVAL107         : 1;
  __REG16 MSGVAL108         : 1;
  __REG16 MSGVAL109         : 1;
  __REG16 MSGVAL110         : 1;
  __REG16 MSGVAL111         : 1;
  __REG16 MSGVAL112         : 1;
} __can_msgval7_bits;

/* Message Valid Registers (MSGVAL8) */
typedef struct {
  __REG16 MSGVAL113         : 1;
  __REG16 MSGVAL114         : 1;
  __REG16 MSGVAL115         : 1;
  __REG16 MSGVAL116         : 1;
  __REG16 MSGVAL117         : 1;
  __REG16 MSGVAL118         : 1;
  __REG16 MSGVAL119         : 1;
  __REG16 MSGVAL120         : 1;
  __REG16 MSGVAL121         : 1;
  __REG16 MSGVAL122         : 1;
  __REG16 MSGVAL123         : 1;
  __REG16 MSGVAL124         : 1;
  __REG16 MSGVAL125         : 1;
  __REG16 MSGVAL126         : 1;
  __REG16 MSGVAL127         : 1;
  __REG16 MSGVAL128         : 1;
} __can_msgval8_bits;

/* Debug Register (DEBUGn) */
typedef struct {
  __REG16 DBGSL             : 1;
  __REG16 DBGLB             : 1;
  __REG16                   :14;
} __can_debug_bits;

/* Serial Mode Register (USARTn_SMR) */
typedef struct {
  __REG8  NFEN              : 1;
  __REG8                    : 1;
  __REG8  UPCL              : 1;
  __REG8  REST              : 1;
  __REG8  EXT               : 1;
  __REG8  OTO               : 1;
  __REG8  MD                : 2;
} __usartn_smr_bits;

/* Serial Control Register (USARTn_SCR) */
typedef struct {
  __REG8  TXE               : 1;
  __REG8  RXE               : 1;
  __REG8  CRE               : 1;
  __REG8  AD                : 1;
  __REG8  CL                : 1;
  __REG8  SBL               : 1;
  __REG8  P                 : 1;
  __REG8  PEN               : 1;
} __usartn_scr_bits;

/* Serial Mode Set Register (USARTn_SMSR) */
typedef struct {
  __REG8                    : 2;
  __REG8  UPCLS             : 1;
  __REG8  RESTS             : 1;
  __REG8                    : 4;
} __usartn_smsr_bits;

/* Serial Control Set Register (USARTn_SCSR) */
typedef struct {
  __REG8  TXES              : 1;
  __REG8  RXES              : 1;
  __REG8  CRES              : 1;
  __REG8  ADS               : 1;
  __REG8                    : 4;
} __usartn_scsr_bits;

/* Serial Control Clear Register (USARTn_SCCR) */
typedef struct {
  __REG8  TXEC              : 1;
  __REG8  RXEC              : 1;
  __REG8                    : 1;
  __REG8  ADS               : 1;
  __REG8                    : 4;
} __usartn_sccr_bits;

/* Serial Status Register (USARTn_SSR) */
typedef struct {
  __REG8  TIE               : 1;
  __REG8  RIE               : 1;
  __REG8  BDS               : 1;
  __REG8  TDRE              : 1;
  __REG8  RDRF              : 1;
  __REG8  FRE               : 1;
  __REG8  ORE               : 1;
  __REG8  PE                : 1;
} __usartn_ssr_bits;

/* Serial Status Set Register (USARTn_SSSR) */
typedef struct {
  __REG8  TIES              : 1;
  __REG8  RIES              : 1;
  __REG8                    : 6;
} __usartn_sssr_bits;

/* Serial Status Clear Register (USARTn_SSCR) */
typedef struct {
  __REG8  TIEC              : 1;
  __REG8  RIEC              : 1;
  __REG8                    : 6;
} __usartn_sscr_bits;

/* Extended Communication Control Register (USARTn_ECCR) */
typedef struct {
  __REG8  TBI               : 1;
  __REG8  RBI               : 1;
  __REG8  BIE               : 1;
  __REG8  SSM               : 1;
  __REG8  SCDE              : 1;
  __REG8  MS                : 1;
  __REG8  LBR               : 1;
  __REG8  INV               : 1;
} __usartn_eccr_bits;

/* Extended Status/Control Register (USARTn_ESCR) */
typedef struct {
  __REG8  SCES              : 1;
  __REG8  CCO               : 1;
  __REG8  SIOP              : 1;
  __REG8  SOPE              : 1;
  __REG8  LBL               : 2;
  __REG8  LBD               : 1;
  __REG8  LBIE              : 1;
} __usartn_escr_bits;

/* Extended Communication Control Set Register (USARTn_ECCSR) */
typedef struct {
  __REG8                    : 2;
  __REG8  BIES              : 1;
  __REG8                    : 3;
  __REG8  LBRS              : 1;
  __REG8                    : 1;
} __usartn_eccsr_bits;

/* Extended Status/Control Set Register (USARTn_ESCSR) */
typedef struct {
  __REG8                    : 2;
  __REG8  SIOPS             : 1;
  __REG8                    : 4;
  __REG8  LBIES             : 1;
} __usartn_escsr_bits;

/* Extended Communication Control Clear Register (USARTn_ECCCR) */
typedef struct {
  __REG8                    : 2;
  __REG8  BIEC              : 1;
  __REG8                    : 5;
} __usartn_ecccr_bits;

/* Extended Status/Control Clear Register (USARTn_ESCCR) */
typedef struct {
  __REG8                    : 2;
  __REG8  SIOPC             : 1;
  __REG8                    : 3;
  __REG8  LBDC              : 1;
  __REG8  LBIEC             : 1;
} __usartn_esccr_bits;

/* Extended Status/Control Clear Register (USARTn_ESCCR) */
typedef struct {
  __REG8  AICD              : 1;
  __REG8  RBI               : 1;
  __REG8  RDRF              : 1;
  __REG8  TDRE              : 1;
  __REG8  FREIE             : 1;
  __REG8  OREIE             : 1;
  __REG8  PEIE              : 1;
  __REG8                    : 1;
} __usartn_esir_bits;

/* Extended Interrupt Enable Register (USARTn_EIER) */
typedef struct {
  __REG8  LBSOIE            : 1;
  __REG8  BUSERRIE          : 1;
  __REG8  PEFRDIE           : 1;
  __REG8  TXHDIE            : 1;
  __REG8  RXHDIE            : 1;
  __REG8  SYNFEIE           : 1;
  __REG8  RXFIE             : 1;
  __REG8  TXFIE             : 1;
} __usartn_eier_bits;

/* Extended Serial Interrupt Set Register (USARTn_ESISR) */
typedef struct {
  __REG8                    : 4;
  __REG8  FREIES            : 1;
  __REG8  OREIES            : 1;
  __REG8  PEIES             : 1;
  __REG8                    : 1;
} __usartn_esisr_bits;

/* Extended Interrupt Enable Set Register (USARTn_EIESR) */
typedef struct {
  __REG8  LBSOIES           : 1;
  __REG8  BUSERRIES         : 1;
  __REG8  PEFRDIES          : 1;
  __REG8  TXHDIES           : 1;
  __REG8  RXHDIES           : 1;
  __REG8  SYNFEIES          : 1;
  __REG8  RXFIES            : 1;
  __REG8  TXFIES            : 1;
} __usartn_eiesr_bits;

/* Extended Serial Interrupt Clear Register (USARTn_ESICR) */
typedef struct {
  __REG8                    : 1;
  __REG8  RBIC              : 1;
  __REG8  RDRFC             : 1;
  __REG8  TDREC             : 1;
  __REG8  FREIEC            : 1;
  __REG8  OREIEC            : 1;
  __REG8  PEIEC             : 1;
  __REG8                    : 1;
} __usartn_esicr_bits;

/* Extended Interrupt Enable Clear Register (USARTn_EIECR) */
typedef struct {
  __REG8  LBSOIEC           : 1;
  __REG8  BUSERRIEC         : 1;
  __REG8  PEFRDIEC          : 1;
  __REG8  TXHDIEC           : 1;
  __REG8  RXHDIEC           : 1;
  __REG8  SYNFEIEC          : 1;
  __REG8  RXFIEC            : 1;
  __REG8  TXFIEC            : 1;
} __usartn_eiecr_bits;

/* Extended Feature Enable Register - L (USARTn_EFERL) */
typedef struct {
  __REG8  ENTXHR            : 1;
  __REG8  ENRXHR            : 1;
  __REG8  ABRE              : 1;
  __REG8  LBEDGE            : 1;
  __REG8  RSTRFM            : 1;
  __REG8  DTSTART           : 1;
  __REG8  OSDE              : 1;
  __REG8                    : 1;
} __usartn_eferl_bits;

/* Extended Feature Enable Register - H (USARTn_EFERH) */
typedef struct {
  __REG8  LBL2              : 1;
  __REG8  FIDE              : 1;
  __REG8  DBE               : 1;
  __REG8  FIDPE             : 1;
  __REG8  BRGR              : 1;
  __REG8  INTLBEN           : 1;
  __REG8                    : 2;
} __usartn_eferh_bits;

/* Reception FIFO Control Register (USARTn_RFCR) */
typedef struct {
  __REG8  RXFLC             : 5;
  __REG8                    : 1;
  __REG8  RXFCL             : 1;
  __REG8  RXFE              : 1;
} __usartn_rfcr_bits;

/* Transmission FIFO Control Register (USARTn_TFCR) */
typedef struct {
  __REG8  TXFLC             : 5;
  __REG8                    : 1;
  __REG8  TXFCL             : 1;
  __REG8  TXFE              : 1;
} __usartn_tfcr_bits;

/* Reception FIFO Control Set Register (USARTn_RFCSR) */
typedef struct {
  __REG8                    : 7;
  __REG8  RXFES             : 1;
} __usartn_rfcsr_bits;

/* Transmission FIFO Control Set Register (USARTn_TFCSR) */
typedef struct {
  __REG8                    : 7;
  __REG8  TXFES             : 1;
} __usartn_tfcsr_bits;

/* Reception FIFO Control Clear Register (USARTn_RFCCR) */
typedef struct {
  __REG8                    : 6;
  __REG8  RXFCLC            : 1;
  __REG8  RXFEC             : 1;
} __usartn_rfccr_bits;

/* Transmission FIFO Control Clear Register (USARTn_TFCCR) */
typedef struct {
  __REG8                    : 6;
  __REG8  TXFCLC            : 1;
  __REG8  TXFEC             : 1;
} __usartn_tfccr_bits;

/* Reception FIFO Status Register (USARTn_RFSR) */
typedef struct {
  __REG8  RXFVD             : 5;
  __REG8                    : 3;
} __usartn_rfsr_bits;

/* Transmission FIFO Status Register (USARTn_TFSR) */
typedef struct {
  __REG8  TXFVD             : 5;
  __REG8                    : 3;
} __usartn_tfsr_bits;

/* Checksum Status and Control Register (USARTn_CSCR) */
typedef struct {
  __REG8  DL                : 3;
  __REG8  CRCGEN            : 1;
  __REG8  CRCCHECK          : 1;
  __REG8  CRCTYPE           : 1;
  __REG8  CRCERR            : 1;
  __REG8  CRCERRIE          : 1;
} __usartn_cscr_bits;

/* Extended Status Register (USARTn_ESR) */
typedef struct {
  __REG8  SYNFE             : 1;
  __REG8  PEFRD             : 1;
  __REG8  BUSERR            : 1;
  __REG8  LBSOF             : 1;
  __REG8  RXHRI             : 1;
  __REG8  TXHRI             : 1;
  __REG8  AD                : 1;
  __REG8                    : 1;
} __usartn_esr_bits;

/* Checksum Status and Control Set Register (USARTn_CSCSR) */
typedef struct {
  __REG8                    : 7;
  __REG8  CRCERRIES         : 1;
} __usartn_cscsr_bits;

/* Checksum Status and Control Clear Register (USARTn_CSCCR) */
typedef struct {
  __REG8                    : 6;
  __REG8  CRCERRC           : 1;
  __REG8  CRCERRIEC         : 1;
} __usartn_csccr_bits;

/* Extended Status Clear Register (USARTn_ESCLR) */
typedef struct {
  __REG8  SYNFEC            : 1;
  __REG8  PEFRDC            : 1;
  __REG8  BUSERRC           : 1;
  __REG8  LBSOFC            : 1;
  __REG8  RXHRIC            : 1;
  __REG8  TXHRIC            : 1;
  __REG8                    : 2;
} __usartn_esclr_bits;

/* Baud Rate Generation Reload Register - H (USARTn_BGRLH) */
typedef struct {
  __REG8  VALUE             : 3;
  __REG8                    : 5;
} __usartn_bgrlh_bits;

/* Baud Rate Generation Reload Register - H (USARTn_BGRLH) */
typedef struct {
  __REG8  BGR_VALUE         : 3;
  __REG8                    : 5;
} __usartn_bgrh_bits;

/* Serial Transmit DMA Configuration Register (USARTn_STXDR) */
typedef struct {
  __REG8  TXDDEN            : 1;
  __REG8  TXDRQEN           : 1;
  __REG8  TXDISDOERR        : 1;
  __REG8                    : 5;
} __usartn_stxdr_bits;

/* Serial Receive DMA Configuration Register (USARTn_SRXDR) */
typedef struct {
  __REG8  RXDDEN            : 1;
  __REG8  RXDRQEN           : 1;
  __REG8  RXDISDOERR        : 1;
  __REG8                    : 5;
} __usartn_srxdr_bits;

/* Serial Transmit DMA Configuration Set Register (USARTn_STXDSR) */
typedef struct {
  __REG8  TXDDENS           : 1;
  __REG8  TXDRQENS          : 1;
  __REG8  TXDISDOERRS       : 1;
  __REG8                    : 5;
} __usartn_stxdsr_bits;

/* Serial Receive DMA Configuration Set Register (USARTn_SRXDSR) */
typedef struct {
  __REG8  RXDDENS           : 1;
  __REG8  RXDRQENS          : 1;
  __REG8  RXDISDOERRS       : 1;
  __REG8                    : 5;
} __usartn_srxdsr_bits;

/* Serial Transmit DMA Configuration Clear Register (USARTn_STXDCR) */
typedef struct {
  __REG8  TXDDENC           : 1;
  __REG8  TXDRQENC          : 1;
  __REG8  TXDISDOERRC       : 1;
  __REG8                    : 5;
} __usartn_stxdcr_bits;

/* Serial Receive DMA Configuration Clear Register (USARTn_SRXDCR) */
typedef struct {
  __REG8  RXDDENC           : 1;
  __REG8  RXDRQENC          : 1;
  __REG8  RXDISDOERRC       : 1;
  __REG8                    : 5;
} __usartn_srxdcr_bits;

/* Sync Field Timeout Register - H (USARTn_SFTRH) */
typedef struct {
  __REG8  SFTR              : 3;
  __REG8                    : 5;
} __usartn_sftrh_bits;

/* Debug Register (USARTn_DEBUG) */
typedef struct {
  __REG8  DBGEN             : 1;
  __REG8                    : 7;
} __usartn_debug_bits;

/* Bus Control and Status Register (I2Cn_IBCSR) */
typedef struct {
  __REG16 ADT               : 1;
  __REG16 GCA               : 1;
  __REG16 AAS               : 1;
  __REG16 TRX               : 1;
  __REG16 LRB               : 1;
  __REG16 AL                : 1;
  __REG16 RSC               : 1;
  __REG16 BB                : 1;
  __REG16 INT               : 1;
  __REG16 INTE              : 1;
  __REG16 GCAA              : 1;
  __REG16 ACK               : 1;
  __REG16 MSS               : 1;
  __REG16 SCC               : 1;
  __REG16 BEIE              : 1;
  __REG16 BER               : 1;
} __i2cn_ibcsr_bits;

/* Ten Bit Slave Address Register (I2Cn_ITBA) */
typedef struct {
  __REG16 TA                :10;
  __REG16                   : 6;
} __i2cn_itba_bits;

/* Ten Bit Slave Address Register (I2Cn_ITBA) */
typedef struct {
  __REG16 TM                :10;
  __REG16                   : 4;
  __REG16 RAL               : 1;
  __REG16 ENTB              : 1;
} __i2cn_itmk_bits;

/* Seven Bit Slave Mask and Address Register (I2Cn_ISBMA) */
typedef struct {
  __REG16 SA                : 7;
  __REG16                   : 1;
  __REG16 SM                : 7;
  __REG16 ENSB              : 1;
} __i2cn_isbma_bits;

/* Clock Control Register (I2Cn_ICCR) */
typedef struct {
  __REG8  CS                : 6;
  __REG8                    : 2;
} __i2cn_iccr_bits;

/* CPU and DMA Input Data Register (ICDIDAR) */
typedef struct {
  __REG16 IDIDAR            : 8;
  __REG16 ICIDAR            : 8;
} __i2cn_icdidar_bits;

/* Interface Enable and Interrupt Clear Register (I2Cn_IEICR) */
typedef struct {
  __REG16 INTCLR            : 1;
  __REG16 BERCLR            : 1;
  __REG16 ALCLR             : 1;
  __REG16                   : 5;
  __REG16 EN                : 1;
  __REG16 NSFEN             : 1;
  __REG16 NSF               : 3;
  __REG16                   : 3;
} __i2cn_ieicr_bits;

/* Debug and DMA Configuration Register (I2Cn_DDMACFG) */
typedef struct {
  __REG16 ENDMAREQRX        : 1;
  __REG16 ENDMAREQTX        : 1;
  __REG16 DMAMODE           : 1;
  __REG16                   :13;
} __i2cn_ddmacfg_bits;

/* Error Interrupt Enable Register (I2Cn_IEIER) */
typedef struct {
  __REG8  BEREIE            : 1;
  __REG8  ALEIE             : 1;
  __REG8                    : 6;
} __i2cn_ieier_bits;

/* HSSPI Module Control Register (HSSPIn_MCTRL) */
typedef struct {
  __REG32 MEN               : 1;
  __REG32 CSEN              : 1;
  __REG32 DEN               : 1;
  __REG32 CDSS              : 1;
  __REG32 MES               : 1;
  __REG32                   :27;
} __hsspin_mctrl_bits;

/* HSSPI Peripheral Communication Configuratio Register 0~3 (HSSPIn_PCC0~3) */
typedef struct {
  __REG32 CPHA              : 1;
  __REG32 CPOL              : 1;
  __REG32 ACES              : 1;
  __REG32 RTM               : 1;
  __REG32 SSPOL             : 1;
  __REG32 SS2CD             : 2;
  __REG32 SDIR              : 1;
  __REG32                   : 1;
  __REG32 CDRS              : 7;
  __REG32 SAFESYNC          : 1;
  __REG32                   :15;
} __hsspin_pcc_bits;

/* HSSPI TX Interrupt Flag Register (HSSPIn_TXF) */
typedef struct {
  __REG32 TFFS              : 1;
  __REG32 TFES              : 1;
  __REG32 TFOS              : 1;
  __REG32 TFUS              : 1;
  __REG32 TFLETS            : 1;
  __REG32 TFMTS             : 1;
  __REG32 TSSRS             : 1;
  __REG32                   :25;
} __hsspin_txf_bits;

/* HSSPI TX Interrupt Enable Register (HSSPIn_TXE) */
typedef struct {
  __REG32 TFFE              : 1;
  __REG32 TFEE              : 1;
  __REG32 TFOE              : 1;
  __REG32 TFUE              : 1;
  __REG32 TFLETE            : 1;
  __REG32 TFMTE             : 1;
  __REG32 TSSRE             : 1;
  __REG32                   :25;
} __hsspin_txe_bits;

/* HSSPI TX Interrupt Clear Register (HSSPIn_TXC) */
typedef struct {
  __REG32 TFFC              : 1;
  __REG32 TFEC              : 1;
  __REG32 TFOC              : 1;
  __REG32 TFUC              : 1;
  __REG32 TFLETC            : 1;
  __REG32 TFMTC             : 1;
  __REG32 TSSRC             : 1;
  __REG32                   :25;
} __hsspin_txc_bits;

/* HSSPI RX Interrupt Flag Register (HSSPIn_RXF) */
typedef struct {
  __REG32 RFFS              : 1;
  __REG32 RFES              : 1;
  __REG32 RFOS              : 1;
  __REG32 RFUS              : 1;
  __REG32 RFLETS            : 1;
  __REG32 RFMTS             : 1;
  __REG32 RSSRS             : 1;
  __REG32                   :25;
} __hsspin_rxf_bits;

/* HSSPI RX Interrupt Enable Register (HSSPIn_RXE) */
typedef struct {
  __REG32 RFFE              : 1;
  __REG32 RFEE              : 1;
  __REG32 RFOE              : 1;
  __REG32 RFUE              : 1;
  __REG32 RFLETE            : 1;
  __REG32 RFMTE             : 1;
  __REG32 RSSRE             : 1;
  __REG32                   :25;
} __hsspin_rxe_bits;

/* HSSPI RX Interrupt Clear Register (HSSPIn_RXC) */
typedef struct {
  __REG32 RFFC              : 1;
  __REG32 RFEC              : 1;
  __REG32 RFOC              : 1;
  __REG32 RFUC              : 1;
  __REG32 RFLETC            : 1;
  __REG32 RFMTC             : 1;
  __REG32 RSSRC             : 1;
  __REG32                   :25;
} __hsspin_rxc_bits;

/* HSSPI Fault Interrupt Flag Register (HSSPIn_FAULTF) */
typedef struct {
  __REG32 UMAFS             : 1;
  __REG32 WAFS              : 1;
  __REG32 PVFS              : 1;
  __REG32 DWCBSFS           : 1;
  __REG32 DRCBSFS           : 1;
  __REG32                   :27;
} __hsspin_faultf_bits;

/* HSSPI Fault Interrupt Clear Register (HSSPIn_FAULTC) */
typedef struct {
  __REG32 UMAFC             : 1;
  __REG32 WAFC              : 1;
  __REG32 PVFC              : 1;
  __REG32 DWCBSFC           : 1;
  __REG32 DRCBSFC           : 1;
  __REG32                   :27;
} __hsspin_faultc_bits;

/* HSSPI Direct Mode Configuration Register (HSSPIn_DMCFG) */
typedef struct {
  __REG8  MST               : 1;
  __REG8  SSDC              : 1;
  __REG8  MSTARTEN          : 1;
  __REG8                    : 5;
} __hsspin_dmcfg_bits;

/* HSSPI Direct Mode DMA Enable Register (HSSPIn_DMDMAEN) */
typedef struct {
  __REG8  RXDMAEN           : 1;
  __REG8  TXDMAEN           : 1;
  __REG8                    : 6;
} __hsspin_dmdmaen_bits;

/* HSSPI Direct Mode Start Register (HSSPIn_DMSTART) */
typedef struct {
  __REG8  START             : 1;
  __REG8                    : 7;
} __hsspin_dmstart_bits;

/* HSSPI Direct Mode Stop Register (HSSPIn_DMSTOP) */
typedef struct {
  __REG8  STOP              : 1;
  __REG8                    : 7;
} __hsspin_dmstop_bits;

/* HSSPI Direct Mode Peripheral Slave Select Register (HSSPIn_DMPSEL) */
typedef struct {
  __REG8  PSEL              : 2;
  __REG8                    : 6;
} __hsspin_dmpsel_bits;

/* HSSPI Direct Mode Transfer Protocol Register (HSSPIn_DMTRP) */
typedef struct {
  __REG8  TRP               : 4;
  __REG8                    : 4;
} __hsspin_dmtrp_bits;

/* HSSPI Direct Mode Status Register (HSSPIn_DMSTATUS) */
typedef struct {
  __REG32 RXACTIVE          : 1;
  __REG32 TXACTIVE          : 1;
  __REG32                   : 6;
  __REG32 RXFLEVEL          : 5;
  __REG32                   : 3;
  __REG32 TXFLEVEL          : 5;
  __REG32                   :11;
} __hsspin_dmstatus_bits;

/* HSSPI Transmit Bit Count Register (HSSPIn_TXBITCNT) */
typedef struct {
  __REG8  TXBITCNT          : 6;
  __REG8                    : 2;
} __hsspin_txbitcnt_bits;

/* HSSPI Receive Bit Count Register (HSSPIn_RXBITCNT) */
typedef struct {
  __REG8  RXBITCNT          : 6;
  __REG8                    : 2;
} __hsspin_rxbitcnt_bits;

/* HSSPI FIFO Configuration Register (HSSPIn_FIFOCFG) */
typedef struct {
  __REG32 RXFTH             : 4;
  __REG32 TXFTH             : 4;
  __REG32 FWIDTH            : 2;
  __REG32 TXCTRL            : 1;
  __REG32 RXFLSH            : 1;
  __REG32 TXFLSH            : 1;
  __REG32                   :19;
} __hsspin_fifocfg_bits;

/* HSSPI Command Sequencer Configuration Register (HSSPIn_CSCFG) */
typedef struct {
  __REG32 SRAM              : 1;
  __REG32 MBM               : 2;
  __REG32                   : 5;
  __REG32 SSEL0EN           : 1;
  __REG32 SSEL1EN           : 1;
  __REG32 SSEL2EN           : 1;
  __REG32 SSEL3EN           : 1;
  __REG32                   : 4;
  __REG32 MSEL              : 4;
  __REG32                   :12;
} __hsspin_cscfg_bits;

/* HSSPI Command Sequencer Idle Time Register (HSSPIn_CSITIME) */
typedef struct {
  __REG32 ITIME             :16;
  __REG32                   :16;
} __hsspin_csitime_bits;

/* HSSPI Command Sequencer Address Extension Register (HSSPIn_CSAEXT) */
typedef struct {
  __REG32                   :13;
  __REG32 AEXT              :19;
} __hsspin_csaext_bits;

/* HSSPI Read Command Sequence Data/Control Register 0~7 (HSSPIn_RDCSDC0~7) */
typedef struct {
  __REG16 DEC               : 1;
  __REG16                   : 7;
  __REG16 RDCSDATA          : 8;
} __hsspin_rdcsdc_bits;

/* HSSPI Write Command Sequence Data/Control Register 0~7 (HSSPIn_WRCSDC0~7) */
typedef struct {
  __REG16 DEC               : 1;
  __REG16                   : 7;
  __REG16 WRCSDATA          : 8;
} __hsspin_wrcsdc_bits;

/* Control Register (I2Sn_CNTREG) */
typedef struct {
  __REG32 FSPL              : 1;
  __REG32 FSLN              : 1;
  __REG32 FSPH              : 1;
  __REG32 CPOL              : 1;
  __REG32 SMPL              : 1;
  __REG32 RXDIS             : 1;
  __REG32 TXDIS             : 1;
  __REG32 MSLB              : 1;
  __REG32 FRUN              : 1;
  __REG32 BEXT              : 1;
  __REG32 ECKM              : 1;
  __REG32 RHLL              : 1;
  __REG32 SBFN              : 1;
  __REG32 MSMD              : 1;
  __REG32 MSKB              : 1;
  __REG32                   : 1;
  __REG32 OVHD              :10;
  __REG32 CKRT              : 6;
} __i2sn_cntreg_bits;

/* Control Register (I2Sn_CNTREG) */
typedef struct {
  __REG32 S0WDL             : 5;
  __REG32 S0CHL             : 5;
  __REG32 S0CHN             : 5;
  __REG32                   : 1;
  __REG32 S1WDL             : 5;
  __REG32 S1CHL             : 5;
  __REG32 S1CHN             : 5;
  __REG32                   : 1;
} __i2sn_mcr0reg_bits;

/* Operation Control Register (I2Sn_OPRREG) */
typedef struct {
  __REG32 START             : 1;
  __REG32                   :15;
  __REG32 TXENB             : 1;
  __REG32                   : 7;
  __REG32 RXENB             : 1;
  __REG32                   : 7;
} __i2sn_oprreg_bits;

/* Software Reset Register (I2Sn_SRST) */
typedef struct {
  __REG32 SRST              : 1;
  __REG32                   :31;
} __i2sn_srst_bits;

/* Interrupt Control Register (I2Sn_INTCNT) */
typedef struct {
  __REG32 RFTH              : 4;
  __REG32 RPTMR             : 2;
  __REG32                   : 2;
  __REG32 TFTH              : 4;
  __REG32                   : 4;
  __REG32 RXFIM             : 1;
  __REG32 RXFDM             : 1;
  __REG32 EOPM              : 1;
  __REG32 RXOVM             : 1;
  __REG32 RXUDM             : 1;
  __REG32 RBERM             : 1;
  __REG32                   : 2;
  __REG32 TXFIM             : 1;
  __REG32 TXFDM             : 1;
  __REG32 TXOVM             : 1;
  __REG32 TXUD0M            : 1;
  __REG32 FERRM             : 1;
  __REG32 TBERM             : 1;
  __REG32 TXUD1M            : 1;
  __REG32                   : 1;
} __i2sn_intcnt_bits;

/* Status Register (I2Sn_STATUS) */
typedef struct {
  __REG32 RXNUM             : 8;
  __REG32 TXNUM             : 8;
  __REG32 RXFI              : 1;
  __REG32 TXFI              : 1;
  __REG32 BSY               : 1;
  __REG32 EOPI              : 1;
  __REG32                   : 4;
  __REG32 RXOVR             : 1;
  __REG32 RXUDR             : 1;
  __REG32 TXOVR             : 1;
  __REG32 TXUDR0            : 1;
  __REG32 TXUDR1            : 1;
  __REG32 FERR              : 1;
  __REG32 RBERR             : 1;
  __REG32 TBERR             : 1;
} __i2sn_status_bits;

/* DMA Activate Register (I2Sn_DMAACT) */
typedef struct {
  __REG32 RDMACT            : 1;
  __REG32                   :15;
  __REG32 TDMACT            : 1;
  __REG32                   :15;
} __i2sn_dmaact_bits;

/* Debug Register (I2Sn_DEBUG) */
typedef struct {
  __REG32 DBGE              : 1;
  __REG32                   :31;
} __i2sn_debug_bits;

/* Module ID Register (I2Sn_MIDREG) */
typedef struct {
  __REG32 DBGE              : 1;
  __REG32                   :31;
} __i2sn_midreg_bits;

/* Data Direction Register (GPIO_DDR0L) */
typedef struct {
  __REG32 DD0               : 1;
  __REG32 DD1               : 1;
  __REG32 DD2               : 1;
  __REG32 DD3               : 1;
  __REG32 DD4               : 1;
  __REG32 DD5               : 1;
  __REG32 DD6               : 1;
  __REG32 DD7               : 1;
  __REG32 DD8               : 1;
  __REG32 DD9               : 1;
  __REG32 DD10              : 1;
  __REG32 DD11              : 1;
  __REG32 DD12              : 1;
  __REG32 DD13              : 1;
  __REG32 DD14              : 1;
  __REG32 DD15              : 1;
  __REG32 DD16              : 1;
  __REG32 DD17              : 1;
  __REG32 DD18              : 1;
  __REG32 DD19              : 1;
  __REG32 DD20              : 1;
  __REG32 DD21              : 1;
  __REG32 DD22              : 1;
  __REG32 DD23              : 1;
  __REG32 DD24              : 1;
  __REG32 DD25              : 1;
  __REG32 DD26              : 1;
  __REG32 DD27              : 1;
  __REG32 DD28              : 1;
  __REG32 DD29              : 1;
  __REG32 DD30              : 1;
  __REG32 DD31              : 1;
} __gpio_ddr0l_bits;

/* Data Direction Register (GPIO_DDR0H) */
typedef struct {
  __REG32 DD32              : 1;
  __REG32 DD33              : 1;
  __REG32 DD34              : 1;
  __REG32 DD35              : 1;
  __REG32 DD36              : 1;
  __REG32 DD37              : 1;
  __REG32 DD38              : 1;
  __REG32 DD39              : 1;
  __REG32 DD40              : 1;
  __REG32 DD41              : 1;
  __REG32 DD42              : 1;
  __REG32 DD43              : 1;
  __REG32 DD44              : 1;
  __REG32 DD45              : 1;
  __REG32 DD46              : 1;
  __REG32 DD47              : 1;
  __REG32 DD48              : 1;
  __REG32 DD49              : 1;
  __REG32 DD50              : 1;
  __REG32 DD51              : 1;
  __REG32 DD52              : 1;
  __REG32 DD53              : 1;
  __REG32 DD54              : 1;
  __REG32 DD55              : 1;
  __REG32 DD56              : 1;
  __REG32 DD57              : 1;
  __REG32 DD58              : 1;
  __REG32 DD59              : 1;
  __REG32 DD60              : 1;
  __REG32 DD61              : 1;
  __REG32 DD62              : 1;
  __REG32 DD63              : 1;
} __gpio_ddr0h_bits;

/* Data Direction Register (GPIO_DDR1L) */
typedef struct {
  __REG32 DD64              : 1;
  __REG32 DD65              : 1;
  __REG32 DD66              : 1;
  __REG32 DD67              : 1;
  __REG32 DD68              : 1;
  __REG32 DD69              : 1;
  __REG32 DD70              : 1;
  __REG32 DD71              : 1;
  __REG32 DD72              : 1;
  __REG32 DD73              : 1;
  __REG32 DD74              : 1;
  __REG32 DD75              : 1;
  __REG32 DD76              : 1;
  __REG32 DD77              : 1;
  __REG32 DD78              : 1;
  __REG32 DD79              : 1;
  __REG32 DD80              : 1;
  __REG32 DD81              : 1;
  __REG32 DD82              : 1;
  __REG32 DD83              : 1;
  __REG32 DD84              : 1;
  __REG32 DD85              : 1;
  __REG32 DD86              : 1;
  __REG32 DD87              : 1;
  __REG32 DD88              : 1;
  __REG32 DD89              : 1;
  __REG32 DD90              : 1;
  __REG32 DD91              : 1;
  __REG32 DD92              : 1;
  __REG32 DD93              : 1;
  __REG32 DD94              : 1;
  __REG32 DD95              : 1;
} __gpio_ddr1l_bits;

/* Data Direction Register (GPIO_DDR1H) */
typedef struct {
  __REG32 DD96              : 1;
  __REG32 DD97              : 1;
  __REG32 DD98              : 1;
  __REG32 DD99              : 1;
  __REG32 DD100             : 1;
  __REG32 DD101             : 1;
  __REG32 DD102             : 1;
  __REG32 DD103             : 1;
  __REG32 DD104             : 1;
  __REG32 DD105             : 1;
  __REG32 DD106             : 1;
  __REG32 DD107             : 1;
  __REG32 DD108             : 1;
  __REG32 DD109             : 1;
  __REG32 DD110             : 1;
  __REG32 DD111             : 1;
  __REG32 DD112             : 1;
  __REG32 DD113             : 1;
  __REG32 DD114             : 1;
  __REG32 DD115             : 1;
  __REG32 DD116             : 1;
  __REG32 DD117             : 1;
  __REG32 DD118             : 1;
  __REG32 DD119             : 1;
  __REG32 DD120             : 1;
  __REG32 DD121             : 1;
  __REG32 DD122             : 1;
  __REG32 DD123             : 1;
  __REG32 DD124             : 1;
  __REG32 DD125             : 1;
  __REG32 DD126             : 1;
  __REG32 DD127             : 1;
} __gpio_ddr1h_bits;

/* Data Direction Register (GPIO_DDR2L) */
typedef struct {
  __REG32 DD128             : 1;
  __REG32 DD129             : 1;
  __REG32 DD130             : 1;
  __REG32 DD131             : 1;
  __REG32 DD132             : 1;
  __REG32 DD133             : 1;
  __REG32 DD134             : 1;
  __REG32 DD135             : 1;
  __REG32 DD136             : 1;
  __REG32 DD137             : 1;
  __REG32 DD138             : 1;
  __REG32 DD139             : 1;
  __REG32 DD140             : 1;
  __REG32 DD141             : 1;
  __REG32 DD142             : 1;
  __REG32 DD143             : 1;
  __REG32 DD144             : 1;
  __REG32 DD145             : 1;
  __REG32 DD146             : 1;
  __REG32 DD147             : 1;
  __REG32 DD148             : 1;
  __REG32 DD149             : 1;
  __REG32 DD150             : 1;
  __REG32 DD151             : 1;
  __REG32 DD152             : 1;
  __REG32 DD153             : 1;
  __REG32 DD154             : 1;
  __REG32 DD155             : 1;
  __REG32 DD156             : 1;
  __REG32 DD157             : 1;
  __REG32 DD158             : 1;
  __REG32 DD159             : 1;
} __gpio_ddr2l_bits;

/* Data Direction Register (GPIO_DDR2H) */
typedef struct {
  __REG32 DD160             : 1;
  __REG32 DD161             : 1;
  __REG32 DD162             : 1;
  __REG32 DD163             : 1;
  __REG32 DD164             : 1;
  __REG32 DD165             : 1;
  __REG32 DD166             : 1;
  __REG32 DD167             : 1;
  __REG32 DD168             : 1;
  __REG32 DD169             : 1;
  __REG32 DD170             : 1;
  __REG32 DD171             : 1;
  __REG32 DD172             : 1;
  __REG32 DD173             : 1;
  __REG32 DD174             : 1;
  __REG32 DD175             : 1;
  __REG32 DD176             : 1;
  __REG32 DD177             : 1;
  __REG32 DD178             : 1;
  __REG32 DD179             : 1;
  __REG32 DD180             : 1;
  __REG32 DD181             : 1;
  __REG32 DD182             : 1;
  __REG32 DD183             : 1;
  __REG32 DD184             : 1;
  __REG32 DD185             : 1;
  __REG32 DD186             : 1;
  __REG32 DD187             : 1;
  __REG32 DD188             : 1;
  __REG32 DD189             : 1;
  __REG32 DD190             : 1;
  __REG32 DD191             : 1;
} __gpio_ddr2h_bits;

/* Data Direction Register (GPIO_DDR3L) */
typedef struct {
  __REG32 DD192             : 1;
  __REG32 DD193             : 1;
  __REG32 DD194             : 1;
  __REG32 DD195             : 1;
  __REG32 DD196             : 1;
  __REG32 DD197             : 1;
  __REG32 DD198             : 1;
  __REG32 DD199             : 1;
  __REG32 DD200             : 1;
  __REG32 DD201             : 1;
  __REG32 DD202             : 1;
  __REG32 DD203             : 1;
  __REG32 DD204             : 1;
  __REG32 DD205             : 1;
  __REG32 DD206             : 1;
  __REG32 DD207             : 1;
  __REG32 DD208             : 1;
  __REG32 DD209             : 1;
  __REG32 DD210             : 1;
  __REG32 DD211             : 1;
  __REG32 DD212             : 1;
  __REG32 DD213             : 1;
  __REG32 DD214             : 1;
  __REG32 DD215             : 1;
  __REG32 DD216             : 1;
  __REG32 DD217             : 1;
  __REG32 DD218             : 1;
  __REG32 DD219             : 1;
  __REG32 DD220             : 1;
  __REG32 DD221             : 1;
  __REG32 DD222             : 1;
  __REG32 DD223             : 1;
} __gpio_ddr3l_bits;

/* Data Direction Register (GPIO_DDR3H) */
typedef struct {
  __REG32 DD224             : 1;
  __REG32 DD225             : 1;
  __REG32 DD226             : 1;
  __REG32 DD227             : 1;
  __REG32 DD228             : 1;
  __REG32 DD229             : 1;
  __REG32 DD230             : 1;
  __REG32 DD231             : 1;
  __REG32 DD232             : 1;
  __REG32 DD233             : 1;
  __REG32 DD234             : 1;
  __REG32 DD235             : 1;
  __REG32 DD236             : 1;
  __REG32 DD237             : 1;
  __REG32 DD238             : 1;
  __REG32 DD239             : 1;
  __REG32 DD240             : 1;
  __REG32 DD241             : 1;
  __REG32 DD242             : 1;
  __REG32 DD243             : 1;
  __REG32 DD244             : 1;
  __REG32 DD245             : 1;
  __REG32 DD246             : 1;
  __REG32 DD247             : 1;
  __REG32 DD248             : 1;
  __REG32 DD249             : 1;
  __REG32 DD250             : 1;
  __REG32 DD251             : 1;
  __REG32 DD252             : 1;
  __REG32 DD253             : 1;
  __REG32 DD254             : 1;
  __REG32 DD255             : 1;
} __gpio_ddr3h_bits;

/* Data Direction Register (GPIO_DDR4L) */
typedef struct {
  __REG32 DD256             : 1;
  __REG32 DD257             : 1;
  __REG32 DD258             : 1;
  __REG32 DD259             : 1;
  __REG32 DD260             : 1;
  __REG32 DD261             : 1;
  __REG32 DD262             : 1;
  __REG32 DD263             : 1;
  __REG32 DD264             : 1;
  __REG32 DD265             : 1;
  __REG32 DD266             : 1;
  __REG32 DD267             : 1;
  __REG32 DD268             : 1;
  __REG32 DD269             : 1;
  __REG32 DD270             : 1;
  __REG32 DD271             : 1;
  __REG32 DD272             : 1;
  __REG32 DD273             : 1;
  __REG32 DD274             : 1;
  __REG32 DD275             : 1;
  __REG32 DD276             : 1;
  __REG32 DD277             : 1;
  __REG32 DD278             : 1;
  __REG32 DD279             : 1;
  __REG32 DD280             : 1;
  __REG32 DD281             : 1;
  __REG32 DD282             : 1;
  __REG32 DD283             : 1;
  __REG32 DD284             : 1;
  __REG32 DD285             : 1;
  __REG32 DD286             : 1;
  __REG32 DD287             : 1;
} __gpio_ddr4l_bits;

/* Data Direction Register (GPIO_DDR4H) */
typedef struct {
  __REG32 DD288             : 1;
  __REG32 DD289             : 1;
  __REG32 DD290             : 1;
  __REG32 DD291             : 1;
  __REG32 DD292             : 1;
  __REG32 DD293             : 1;
  __REG32 DD294             : 1;
  __REG32 DD295             : 1;
  __REG32 DD296             : 1;
  __REG32 DD297             : 1;
  __REG32 DD298             : 1;
  __REG32 DD299             : 1;
  __REG32 DD300             : 1;
  __REG32 DD301             : 1;
  __REG32 DD302             : 1;
  __REG32 DD303             : 1;
  __REG32 DD304             : 1;
  __REG32 DD305             : 1;
  __REG32 DD306             : 1;
  __REG32 DD307             : 1;
  __REG32 DD308             : 1;
  __REG32 DD309             : 1;
  __REG32 DD310             : 1;
  __REG32 DD311             : 1;
  __REG32 DD312             : 1;
  __REG32 DD313             : 1;
  __REG32 DD314             : 1;
  __REG32 DD315             : 1;
  __REG32 DD316             : 1;
  __REG32 DD317             : 1;
  __REG32 DD318             : 1;
  __REG32 DD319             : 1;
} __gpio_ddr4h_bits;

/* Data Direction Register (GPIO_DDR5L) */
typedef struct {
  __REG32 DD320             : 1;
  __REG32 DD321             : 1;
  __REG32 DD322             : 1;
  __REG32 DD323             : 1;
  __REG32 DD324             : 1;
  __REG32 DD325             : 1;
  __REG32 DD326             : 1;
  __REG32 DD327             : 1;
  __REG32 DD328             : 1;
  __REG32 DD329             : 1;
  __REG32 DD330             : 1;
  __REG32 DD331             : 1;
  __REG32 DD332             : 1;
  __REG32 DD333             : 1;
  __REG32 DD334             : 1;
  __REG32 DD335             : 1;
  __REG32 DD336             : 1;
  __REG32 DD337             : 1;
  __REG32 DD338             : 1;
  __REG32 DD339             : 1;
  __REG32 DD340             : 1;
  __REG32 DD341             : 1;
  __REG32 DD342             : 1;
  __REG32 DD343             : 1;
  __REG32 DD344             : 1;
  __REG32 DD345             : 1;
  __REG32 DD346             : 1;
  __REG32 DD347             : 1;
  __REG32 DD348             : 1;
  __REG32 DD349             : 1;
  __REG32 DD350             : 1;
  __REG32 DD351             : 1;
} __gpio_ddr5l_bits;

/* Data Direction Register (GPIO_DDR5H) */
typedef struct {
  __REG32 DD352             : 1;
  __REG32 DD353             : 1;
  __REG32 DD354             : 1;
  __REG32 DD355             : 1;
  __REG32 DD356             : 1;
  __REG32 DD357             : 1;
  __REG32 DD358             : 1;
  __REG32 DD359             : 1;
  __REG32 DD360             : 1;
  __REG32 DD361             : 1;
  __REG32 DD362             : 1;
  __REG32 DD363             : 1;
  __REG32 DD364             : 1;
  __REG32 DD365             : 1;
  __REG32 DD366             : 1;
  __REG32 DD367             : 1;
  __REG32 DD368             : 1;
  __REG32 DD369             : 1;
  __REG32 DD370             : 1;
  __REG32 DD371             : 1;
  __REG32 DD372             : 1;
  __REG32 DD373             : 1;
  __REG32 DD374             : 1;
  __REG32 DD375             : 1;
  __REG32 DD376             : 1;
  __REG32 DD377             : 1;
  __REG32 DD378             : 1;
  __REG32 DD379             : 1;
  __REG32 DD380             : 1;
  __REG32 DD381             : 1;
  __REG32 DD382             : 1;
  __REG32 DD383             : 1;
} __gpio_ddr5h_bits;

/* Data Direction Register (GPIO_DDR6L) */
typedef struct {
  __REG32 DD384             : 1;
  __REG32 DD385             : 1;
  __REG32 DD386             : 1;
  __REG32 DD387             : 1;
  __REG32 DD388             : 1;
  __REG32 DD389             : 1;
  __REG32 DD390             : 1;
  __REG32 DD391             : 1;
  __REG32 DD392             : 1;
  __REG32 DD393             : 1;
  __REG32 DD394             : 1;
  __REG32 DD395             : 1;
  __REG32 DD396             : 1;
  __REG32 DD397             : 1;
  __REG32 DD398             : 1;
  __REG32 DD399             : 1;
  __REG32 DD400             : 1;
  __REG32 DD401             : 1;
  __REG32 DD402             : 1;
  __REG32 DD403             : 1;
  __REG32 DD404             : 1;
  __REG32 DD405             : 1;
  __REG32 DD406             : 1;
  __REG32 DD407             : 1;
  __REG32 DD408             : 1;
  __REG32 DD409             : 1;
  __REG32 DD410             : 1;
  __REG32 DD411             : 1;
  __REG32 DD412             : 1;
  __REG32 DD413             : 1;
  __REG32 DD414             : 1;
  __REG32 DD415             : 1;
} __gpio_ddr6l_bits;

/* Data Direction Register (GPIO_DDR6H) */
typedef struct {
  __REG32 DD416             : 1;
  __REG32 DD417             : 1;
  __REG32 DD418             : 1;
  __REG32 DD419             : 1;
  __REG32 DD420             : 1;
  __REG32 DD421             : 1;
  __REG32 DD422             : 1;
  __REG32 DD423             : 1;
  __REG32 DD424             : 1;
  __REG32 DD425             : 1;
  __REG32 DD426             : 1;
  __REG32 DD427             : 1;
  __REG32 DD428             : 1;
  __REG32 DD429             : 1;
  __REG32 DD430             : 1;
  __REG32 DD431             : 1;
  __REG32 DD432             : 1;
  __REG32 DD433             : 1;
  __REG32 DD434             : 1;
  __REG32 DD435             : 1;
  __REG32 DD436             : 1;
  __REG32 DD437             : 1;
  __REG32 DD438             : 1;
  __REG32 DD439             : 1;
  __REG32 DD440             : 1;
  __REG32 DD441             : 1;
  __REG32 DD442             : 1;
  __REG32 DD443             : 1;
  __REG32 DD444             : 1;
  __REG32 DD445             : 1;
  __REG32 DD446             : 1;
  __REG32 DD447             : 1;
} __gpio_ddr6h_bits;

/* Data Direction Register (GPIO_DDR7L) */
typedef struct {
  __REG32 DD448             : 1;
  __REG32 DD449             : 1;
  __REG32 DD450             : 1;
  __REG32 DD451             : 1;
  __REG32 DD452             : 1;
  __REG32 DD453             : 1;
  __REG32 DD454             : 1;
  __REG32 DD455             : 1;
  __REG32 DD456             : 1;
  __REG32 DD457             : 1;
  __REG32 DD458             : 1;
  __REG32 DD459             : 1;
  __REG32 DD460             : 1;
  __REG32 DD461             : 1;
  __REG32 DD462             : 1;
  __REG32 DD463             : 1;
  __REG32 DD464             : 1;
  __REG32 DD465             : 1;
  __REG32 DD466             : 1;
  __REG32 DD467             : 1;
  __REG32 DD468             : 1;
  __REG32 DD469             : 1;
  __REG32 DD470             : 1;
  __REG32 DD471             : 1;
  __REG32 DD472             : 1;
  __REG32 DD473             : 1;
  __REG32 DD474             : 1;
  __REG32 DD475             : 1;
  __REG32 DD476             : 1;
  __REG32 DD477             : 1;
  __REG32 DD478             : 1;
  __REG32 DD479             : 1;
} __gpio_ddr7l_bits;

/* Data Direction Register (GPIO_DDR7H) */
typedef struct {
  __REG32 DD480             : 1;
  __REG32 DD481             : 1;
  __REG32 DD482             : 1;
  __REG32 DD483             : 1;
  __REG32 DD484             : 1;
  __REG32 DD485             : 1;
  __REG32 DD486             : 1;
  __REG32 DD487             : 1;
  __REG32 DD488             : 1;
  __REG32 DD489             : 1;
  __REG32 DD490             : 1;
  __REG32 DD491             : 1;
  __REG32 DD492             : 1;
  __REG32 DD493             : 1;
  __REG32 DD494             : 1;
  __REG32 DD495             : 1;
  __REG32 DD496             : 1;
  __REG32 DD497             : 1;
  __REG32 DD498             : 1;
  __REG32 DD499             : 1;
  __REG32 DD500             : 1;
  __REG32 DD501             : 1;
  __REG32 DD502             : 1;
  __REG32 DD503             : 1;
  __REG32 DD504             : 1;
  __REG32 DD505             : 1;
  __REG32 DD506             : 1;
  __REG32 DD507             : 1;
  __REG32 DD508             : 1;
  __REG32 DD509             : 1;
  __REG32 DD510             : 1;
  __REG32 DD511             : 1;
} __gpio_ddr7h_bits;

/* Data Direction Set Register (GPIO_DDSR0L) */
typedef struct {
  __REG32 DDS0              : 1;
  __REG32 DDS1              : 1;
  __REG32 DDS2              : 1;
  __REG32 DDS3              : 1;
  __REG32 DDS4              : 1;
  __REG32 DDS5              : 1;
  __REG32 DDS6              : 1;
  __REG32 DDS7              : 1;
  __REG32 DDS8              : 1;
  __REG32 DDS9              : 1;
  __REG32 DDS10             : 1;
  __REG32 DDS11             : 1;
  __REG32 DDS12             : 1;
  __REG32 DDS13             : 1;
  __REG32 DDS14             : 1;
  __REG32 DDS15             : 1;
  __REG32 DDS16             : 1;
  __REG32 DDS17             : 1;
  __REG32 DDS18             : 1;
  __REG32 DDS19             : 1;
  __REG32 DDS20             : 1;
  __REG32 DDS21             : 1;
  __REG32 DDS22             : 1;
  __REG32 DDS23             : 1;
  __REG32 DDS24             : 1;
  __REG32 DDS25             : 1;
  __REG32 DDS26             : 1;
  __REG32 DDS27             : 1;
  __REG32 DDS28             : 1;
  __REG32 DDS29             : 1;
  __REG32 DDS30             : 1;
  __REG32 DDS31             : 1;
} __gpio_ddsr0l_bits;

/* Data Direction Set Register (GPIO_DDSR0H) */
typedef struct {
  __REG32 DDS32             : 1;
  __REG32 DDS33             : 1;
  __REG32 DDS34             : 1;
  __REG32 DDS35             : 1;
  __REG32 DDS36             : 1;
  __REG32 DDS37             : 1;
  __REG32 DDS38             : 1;
  __REG32 DDS39             : 1;
  __REG32 DDS40             : 1;
  __REG32 DDS41             : 1;
  __REG32 DDS42             : 1;
  __REG32 DDS43             : 1;
  __REG32 DDS44             : 1;
  __REG32 DDS45             : 1;
  __REG32 DDS46             : 1;
  __REG32 DDS47             : 1;
  __REG32 DDS48             : 1;
  __REG32 DDS49             : 1;
  __REG32 DDS50             : 1;
  __REG32 DDS51             : 1;
  __REG32 DDS52             : 1;
  __REG32 DDS53             : 1;
  __REG32 DDS54             : 1;
  __REG32 DDS55             : 1;
  __REG32 DDS56             : 1;
  __REG32 DDS57             : 1;
  __REG32 DDS58             : 1;
  __REG32 DDS59             : 1;
  __REG32 DDS60             : 1;
  __REG32 DDS61             : 1;
  __REG32 DDS62             : 1;
  __REG32 DDS63             : 1;
} __gpio_ddsr0h_bits;

/* Data Direction Set Register (GPIO_DDSR1L) */
typedef struct {
  __REG32 DDS64             : 1;
  __REG32 DDS65             : 1;
  __REG32 DDS66             : 1;
  __REG32 DDS67             : 1;
  __REG32 DDS68             : 1;
  __REG32 DDS69             : 1;
  __REG32 DDS70             : 1;
  __REG32 DDS71             : 1;
  __REG32 DDS72             : 1;
  __REG32 DDS73             : 1;
  __REG32 DDS74             : 1;
  __REG32 DDS75             : 1;
  __REG32 DDS76             : 1;
  __REG32 DDS77             : 1;
  __REG32 DDS78             : 1;
  __REG32 DDS79             : 1;
  __REG32 DDS80             : 1;
  __REG32 DDS81             : 1;
  __REG32 DDS82             : 1;
  __REG32 DDS83             : 1;
  __REG32 DDS84             : 1;
  __REG32 DDS85             : 1;
  __REG32 DDS86             : 1;
  __REG32 DDS87             : 1;
  __REG32 DDS88             : 1;
  __REG32 DDS89             : 1;
  __REG32 DDS90             : 1;
  __REG32 DDS91             : 1;
  __REG32 DDS92             : 1;
  __REG32 DDS93             : 1;
  __REG32 DDS94             : 1;
  __REG32 DDS95             : 1;
} __gpio_ddsr1l_bits;

/* Data Direction Set Register (GPIO_DDSR1H) */
typedef struct {
  __REG32 DDS96             : 1;
  __REG32 DDS97             : 1;
  __REG32 DDS98             : 1;
  __REG32 DDS99             : 1;
  __REG32 DDS100            : 1;
  __REG32 DDS101            : 1;
  __REG32 DDS102            : 1;
  __REG32 DDS103            : 1;
  __REG32 DDS104            : 1;
  __REG32 DDS105            : 1;
  __REG32 DDS106            : 1;
  __REG32 DDS107            : 1;
  __REG32 DDS108            : 1;
  __REG32 DDS109            : 1;
  __REG32 DDS110            : 1;
  __REG32 DDS111            : 1;
  __REG32 DDS112            : 1;
  __REG32 DDS113            : 1;
  __REG32 DDS114            : 1;
  __REG32 DDS115            : 1;
  __REG32 DDS116            : 1;
  __REG32 DDS117            : 1;
  __REG32 DDS118            : 1;
  __REG32 DDS119            : 1;
  __REG32 DDS120            : 1;
  __REG32 DDS121            : 1;
  __REG32 DDS122            : 1;
  __REG32 DDS123            : 1;
  __REG32 DDS124            : 1;
  __REG32 DDS125            : 1;
  __REG32 DDS126            : 1;
  __REG32 DDS127            : 1;
} __gpio_ddsr1h_bits;

/* Data Direction Set Register (GPIO_DDSR2L) */
typedef struct {
  __REG32 DDS128            : 1;
  __REG32 DDS129            : 1;
  __REG32 DDS130            : 1;
  __REG32 DDS131            : 1;
  __REG32 DDS132            : 1;
  __REG32 DDS133            : 1;
  __REG32 DDS134            : 1;
  __REG32 DDS135            : 1;
  __REG32 DDS136            : 1;
  __REG32 DDS137            : 1;
  __REG32 DDS138            : 1;
  __REG32 DDS139            : 1;
  __REG32 DDS140            : 1;
  __REG32 DDS141            : 1;
  __REG32 DDS142            : 1;
  __REG32 DDS143            : 1;
  __REG32 DDS144            : 1;
  __REG32 DDS145            : 1;
  __REG32 DDS146            : 1;
  __REG32 DDS147            : 1;
  __REG32 DDS148            : 1;
  __REG32 DDS149            : 1;
  __REG32 DDS150            : 1;
  __REG32 DDS151            : 1;
  __REG32 DDS152            : 1;
  __REG32 DDS153            : 1;
  __REG32 DDS154            : 1;
  __REG32 DDS155            : 1;
  __REG32 DDS156            : 1;
  __REG32 DDS157            : 1;
  __REG32 DDS158            : 1;
  __REG32 DDS159            : 1;
} __gpio_ddsr2l_bits;

/* Data Direction Set Register (GPIO_DDSR2H) */
typedef struct {
  __REG32 DDS160            : 1;
  __REG32 DDS161            : 1;
  __REG32 DDS162            : 1;
  __REG32 DDS163            : 1;
  __REG32 DDS164            : 1;
  __REG32 DDS165            : 1;
  __REG32 DDS166            : 1;
  __REG32 DDS167            : 1;
  __REG32 DDS168            : 1;
  __REG32 DDS169            : 1;
  __REG32 DDS170            : 1;
  __REG32 DDS171            : 1;
  __REG32 DDS172            : 1;
  __REG32 DDS173            : 1;
  __REG32 DDS174            : 1;
  __REG32 DDS175            : 1;
  __REG32 DDS176            : 1;
  __REG32 DDS177            : 1;
  __REG32 DDS178            : 1;
  __REG32 DDS179            : 1;
  __REG32 DDS180            : 1;
  __REG32 DDS181            : 1;
  __REG32 DDS182            : 1;
  __REG32 DDS183            : 1;
  __REG32 DDS184            : 1;
  __REG32 DDS185            : 1;
  __REG32 DDS186            : 1;
  __REG32 DDS187            : 1;
  __REG32 DDS188            : 1;
  __REG32 DDS189            : 1;
  __REG32 DDS190            : 1;
  __REG32 DDS191            : 1;
} __gpio_ddsr2h_bits;

/* Data Direction Set Register (GPIO_DDSR3L) */
typedef struct {
  __REG32 DDS192            : 1;
  __REG32 DDS193            : 1;
  __REG32 DDS194            : 1;
  __REG32 DDS195            : 1;
  __REG32 DDS196            : 1;
  __REG32 DDS197            : 1;
  __REG32 DDS198            : 1;
  __REG32 DDS199            : 1;
  __REG32 DDS200            : 1;
  __REG32 DDS201            : 1;
  __REG32 DDS202            : 1;
  __REG32 DDS203            : 1;
  __REG32 DDS204            : 1;
  __REG32 DDS205            : 1;
  __REG32 DDS206            : 1;
  __REG32 DDS207            : 1;
  __REG32 DDS208            : 1;
  __REG32 DDS209            : 1;
  __REG32 DDS210            : 1;
  __REG32 DDS211            : 1;
  __REG32 DDS212            : 1;
  __REG32 DDS213            : 1;
  __REG32 DDS214            : 1;
  __REG32 DDS215            : 1;
  __REG32 DDS216            : 1;
  __REG32 DDS217            : 1;
  __REG32 DDS218            : 1;
  __REG32 DDS219            : 1;
  __REG32 DDS220            : 1;
  __REG32 DDS221            : 1;
  __REG32 DDS222            : 1;
  __REG32 DDS223            : 1;
} __gpio_ddsr3l_bits;

/* Data Direction Set Register (GPIO_DDSR3H) */
typedef struct {
  __REG32 DDS224            : 1;
  __REG32 DDS225            : 1;
  __REG32 DDS226            : 1;
  __REG32 DDS227            : 1;
  __REG32 DDS228            : 1;
  __REG32 DDS229            : 1;
  __REG32 DDS230            : 1;
  __REG32 DDS231            : 1;
  __REG32 DDS232            : 1;
  __REG32 DDS233            : 1;
  __REG32 DDS234            : 1;
  __REG32 DDS235            : 1;
  __REG32 DDS236            : 1;
  __REG32 DDS237            : 1;
  __REG32 DDS238            : 1;
  __REG32 DDS239            : 1;
  __REG32 DDS240            : 1;
  __REG32 DDS241            : 1;
  __REG32 DDS242            : 1;
  __REG32 DDS243            : 1;
  __REG32 DDS244            : 1;
  __REG32 DDS245            : 1;
  __REG32 DDS246            : 1;
  __REG32 DDS247            : 1;
  __REG32 DDS248            : 1;
  __REG32 DDS249            : 1;
  __REG32 DDS250            : 1;
  __REG32 DDS251            : 1;
  __REG32 DDS252            : 1;
  __REG32 DDS253            : 1;
  __REG32 DDS254            : 1;
  __REG32 DDS255            : 1;
} __gpio_ddsr3h_bits;

/* Data Direction Set Register (GPIO_DDSR4L) */
typedef struct {
  __REG32 DDS256            : 1;
  __REG32 DDS257            : 1;
  __REG32 DDS258            : 1;
  __REG32 DDS259            : 1;
  __REG32 DDS260            : 1;
  __REG32 DDS261            : 1;
  __REG32 DDS262            : 1;
  __REG32 DDS263            : 1;
  __REG32 DDS264            : 1;
  __REG32 DDS265            : 1;
  __REG32 DDS266            : 1;
  __REG32 DDS267            : 1;
  __REG32 DDS268            : 1;
  __REG32 DDS269            : 1;
  __REG32 DDS270            : 1;
  __REG32 DDS271            : 1;
  __REG32 DDS272            : 1;
  __REG32 DDS273            : 1;
  __REG32 DDS274            : 1;
  __REG32 DDS275            : 1;
  __REG32 DDS276            : 1;
  __REG32 DDS277            : 1;
  __REG32 DDS278            : 1;
  __REG32 DDS279            : 1;
  __REG32 DDS280            : 1;
  __REG32 DDS281            : 1;
  __REG32 DDS282            : 1;
  __REG32 DDS283            : 1;
  __REG32 DDS284            : 1;
  __REG32 DDS285            : 1;
  __REG32 DDS286            : 1;
  __REG32 DDS287            : 1;
} __gpio_ddsr4l_bits;

/* Data Direction Set Register (GPIO_DDSR4H) */
typedef struct {
  __REG32 DDS288            : 1;
  __REG32 DDS289            : 1;
  __REG32 DDS290            : 1;
  __REG32 DDS291            : 1;
  __REG32 DDS292            : 1;
  __REG32 DDS293            : 1;
  __REG32 DDS294            : 1;
  __REG32 DDS295            : 1;
  __REG32 DDS296            : 1;
  __REG32 DDS297            : 1;
  __REG32 DDS298            : 1;
  __REG32 DDS299            : 1;
  __REG32 DDS300            : 1;
  __REG32 DDS301            : 1;
  __REG32 DDS302            : 1;
  __REG32 DDS303            : 1;
  __REG32 DDS304            : 1;
  __REG32 DDS305            : 1;
  __REG32 DDS306            : 1;
  __REG32 DDS307            : 1;
  __REG32 DDS308            : 1;
  __REG32 DDS309            : 1;
  __REG32 DDS310            : 1;
  __REG32 DDS311            : 1;
  __REG32 DDS312            : 1;
  __REG32 DDS313            : 1;
  __REG32 DDS314            : 1;
  __REG32 DDS315            : 1;
  __REG32 DDS316            : 1;
  __REG32 DDS317            : 1;
  __REG32 DDS318            : 1;
  __REG32 DDS319            : 1;
} __gpio_ddsr4h_bits;

/* Data Direction Set Register (GPIO_DDSR5L) */
typedef struct {
  __REG32 DDS320            : 1;
  __REG32 DDS321            : 1;
  __REG32 DDS322            : 1;
  __REG32 DDS323            : 1;
  __REG32 DDS324            : 1;
  __REG32 DDS325            : 1;
  __REG32 DDS326            : 1;
  __REG32 DDS327            : 1;
  __REG32 DDS328            : 1;
  __REG32 DDS329            : 1;
  __REG32 DDS330            : 1;
  __REG32 DDS331            : 1;
  __REG32 DDS332            : 1;
  __REG32 DDS333            : 1;
  __REG32 DDS334            : 1;
  __REG32 DDS335            : 1;
  __REG32 DDS336            : 1;
  __REG32 DDS337            : 1;
  __REG32 DDS338            : 1;
  __REG32 DDS339            : 1;
  __REG32 DDS340            : 1;
  __REG32 DDS341            : 1;
  __REG32 DDS342            : 1;
  __REG32 DDS343            : 1;
  __REG32 DDS344            : 1;
  __REG32 DDS345            : 1;
  __REG32 DDS346            : 1;
  __REG32 DDS347            : 1;
  __REG32 DDS348            : 1;
  __REG32 DDS349            : 1;
  __REG32 DDS350            : 1;
  __REG32 DDS351            : 1;
} __gpio_ddsr5l_bits;

/* Data Direction Set Register (GPIO_DDSR5H) */
typedef struct {
  __REG32 DDS352            : 1;
  __REG32 DDS353            : 1;
  __REG32 DDS354            : 1;
  __REG32 DDS355            : 1;
  __REG32 DDS356            : 1;
  __REG32 DDS357            : 1;
  __REG32 DDS358            : 1;
  __REG32 DDS359            : 1;
  __REG32 DDS360            : 1;
  __REG32 DDS361            : 1;
  __REG32 DDS362            : 1;
  __REG32 DDS363            : 1;
  __REG32 DDS364            : 1;
  __REG32 DDS365            : 1;
  __REG32 DDS366            : 1;
  __REG32 DDS367            : 1;
  __REG32 DDS368            : 1;
  __REG32 DDS369            : 1;
  __REG32 DDS370            : 1;
  __REG32 DDS371            : 1;
  __REG32 DDS372            : 1;
  __REG32 DDS373            : 1;
  __REG32 DDS374            : 1;
  __REG32 DDS375            : 1;
  __REG32 DDS376            : 1;
  __REG32 DDS377            : 1;
  __REG32 DDS378            : 1;
  __REG32 DDS379            : 1;
  __REG32 DDS380            : 1;
  __REG32 DDS381            : 1;
  __REG32 DDS382            : 1;
  __REG32 DDS383            : 1;
} __gpio_ddsr5h_bits;

/* Data Direction Set Register (GPIO_DDSR6L) */
typedef struct {
  __REG32 DDS384            : 1;
  __REG32 DDS385            : 1;
  __REG32 DDS386            : 1;
  __REG32 DDS387            : 1;
  __REG32 DDS388            : 1;
  __REG32 DDS389            : 1;
  __REG32 DDS390            : 1;
  __REG32 DDS391            : 1;
  __REG32 DDS392            : 1;
  __REG32 DDS393            : 1;
  __REG32 DDS394            : 1;
  __REG32 DDS395            : 1;
  __REG32 DDS396            : 1;
  __REG32 DDS397            : 1;
  __REG32 DDS398            : 1;
  __REG32 DDS399            : 1;
  __REG32 DDS400            : 1;
  __REG32 DDS401            : 1;
  __REG32 DDS402            : 1;
  __REG32 DDS403            : 1;
  __REG32 DDS404            : 1;
  __REG32 DDS405            : 1;
  __REG32 DDS406            : 1;
  __REG32 DDS407            : 1;
  __REG32 DDS408            : 1;
  __REG32 DDS409            : 1;
  __REG32 DDS410            : 1;
  __REG32 DDS411            : 1;
  __REG32 DDS412            : 1;
  __REG32 DDS413            : 1;
  __REG32 DDS414            : 1;
  __REG32 DDS415            : 1;
} __gpio_ddsr6l_bits;

/* Data Direction Set Register (GPIO_DDSR6H) */
typedef struct {
  __REG32 DDS416            : 1;
  __REG32 DDS417            : 1;
  __REG32 DDS418            : 1;
  __REG32 DDS419            : 1;
  __REG32 DDS420            : 1;
  __REG32 DDS421            : 1;
  __REG32 DDS422            : 1;
  __REG32 DDS423            : 1;
  __REG32 DDS424            : 1;
  __REG32 DDS425            : 1;
  __REG32 DDS426            : 1;
  __REG32 DDS427            : 1;
  __REG32 DDS428            : 1;
  __REG32 DDS429            : 1;
  __REG32 DDS430            : 1;
  __REG32 DDS431            : 1;
  __REG32 DDS432            : 1;
  __REG32 DDS433            : 1;
  __REG32 DDS434            : 1;
  __REG32 DDS435            : 1;
  __REG32 DDS436            : 1;
  __REG32 DDS437            : 1;
  __REG32 DDS438            : 1;
  __REG32 DDS439            : 1;
  __REG32 DDS440            : 1;
  __REG32 DDS441            : 1;
  __REG32 DDS442            : 1;
  __REG32 DDS443            : 1;
  __REG32 DDS444            : 1;
  __REG32 DDS445            : 1;
  __REG32 DDS446            : 1;
  __REG32 DDS447            : 1;
} __gpio_ddsr6h_bits;

/* Data Direction Set Register (GPIO_DDSR7L) */
typedef struct {
  __REG32 DDS448            : 1;
  __REG32 DDS449            : 1;
  __REG32 DDS450            : 1;
  __REG32 DDS451            : 1;
  __REG32 DDS452            : 1;
  __REG32 DDS453            : 1;
  __REG32 DDS454            : 1;
  __REG32 DDS455            : 1;
  __REG32 DDS456            : 1;
  __REG32 DDS457            : 1;
  __REG32 DDS458            : 1;
  __REG32 DDS459            : 1;
  __REG32 DDS460            : 1;
  __REG32 DDS461            : 1;
  __REG32 DDS462            : 1;
  __REG32 DDS463            : 1;
  __REG32 DDS464            : 1;
  __REG32 DDS465            : 1;
  __REG32 DDS466            : 1;
  __REG32 DDS467            : 1;
  __REG32 DDS468            : 1;
  __REG32 DDS469            : 1;
  __REG32 DDS470            : 1;
  __REG32 DDS471            : 1;
  __REG32 DDS472            : 1;
  __REG32 DDS473            : 1;
  __REG32 DDS474            : 1;
  __REG32 DDS475            : 1;
  __REG32 DDS476            : 1;
  __REG32 DDS477            : 1;
  __REG32 DDS478            : 1;
  __REG32 DDS479            : 1;
} __gpio_ddsr7l_bits;

/* Data Direction Set Register (GPIO_DDSR7H) */
typedef struct {
  __REG32 DDS480            : 1;
  __REG32 DDS481            : 1;
  __REG32 DDS482            : 1;
  __REG32 DDS483            : 1;
  __REG32 DDS484            : 1;
  __REG32 DDS485            : 1;
  __REG32 DDS486            : 1;
  __REG32 DDS487            : 1;
  __REG32 DDS488            : 1;
  __REG32 DDS489            : 1;
  __REG32 DDS490            : 1;
  __REG32 DDS491            : 1;
  __REG32 DDS492            : 1;
  __REG32 DDS493            : 1;
  __REG32 DDS494            : 1;
  __REG32 DDS495            : 1;
  __REG32 DDS496            : 1;
  __REG32 DDS497            : 1;
  __REG32 DDS498            : 1;
  __REG32 DDS499            : 1;
  __REG32 DDS500            : 1;
  __REG32 DDS501            : 1;
  __REG32 DDS502            : 1;
  __REG32 DDS503            : 1;
  __REG32 DDS504            : 1;
  __REG32 DDS505            : 1;
  __REG32 DDS506            : 1;
  __REG32 DDS507            : 1;
  __REG32 DDS508            : 1;
  __REG32 DDS509            : 1;
  __REG32 DDS510            : 1;
  __REG32 DDS511            : 1;
} __gpio_ddsr7h_bits;

/* Data Direction Clear Register (GPIO_DDCR0L) */
typedef struct {
  __REG32 DDC0              : 1;
  __REG32 DDC1              : 1;
  __REG32 DDC2              : 1;
  __REG32 DDC3              : 1;
  __REG32 DDC4              : 1;
  __REG32 DDC5              : 1;
  __REG32 DDC6              : 1;
  __REG32 DDC7              : 1;
  __REG32 DDC8              : 1;
  __REG32 DDC9              : 1;
  __REG32 DDC10             : 1;
  __REG32 DDC11             : 1;
  __REG32 DDC12             : 1;
  __REG32 DDC13             : 1;
  __REG32 DDC14             : 1;
  __REG32 DDC15             : 1;
  __REG32 DDC16             : 1;
  __REG32 DDC17             : 1;
  __REG32 DDC18             : 1;
  __REG32 DDC19             : 1;
  __REG32 DDC20             : 1;
  __REG32 DDC21             : 1;
  __REG32 DDC22             : 1;
  __REG32 DDC23             : 1;
  __REG32 DDC24             : 1;
  __REG32 DDC25             : 1;
  __REG32 DDC26             : 1;
  __REG32 DDC27             : 1;
  __REG32 DDC28             : 1;
  __REG32 DDC29             : 1;
  __REG32 DDC30             : 1;
  __REG32 DDC31             : 1;
} __gpio_ddcr0l_bits;

/* Data Direction Clear Register (GPIO_DDCR0H) */
typedef struct {
  __REG32 DDC32             : 1;
  __REG32 DDC33             : 1;
  __REG32 DDC34             : 1;
  __REG32 DDC35             : 1;
  __REG32 DDC36             : 1;
  __REG32 DDC37             : 1;
  __REG32 DDC38             : 1;
  __REG32 DDC39             : 1;
  __REG32 DDC40             : 1;
  __REG32 DDC41             : 1;
  __REG32 DDC42             : 1;
  __REG32 DDC43             : 1;
  __REG32 DDC44             : 1;
  __REG32 DDC45             : 1;
  __REG32 DDC46             : 1;
  __REG32 DDC47             : 1;
  __REG32 DDC48             : 1;
  __REG32 DDC49             : 1;
  __REG32 DDC50             : 1;
  __REG32 DDC51             : 1;
  __REG32 DDC52             : 1;
  __REG32 DDC53             : 1;
  __REG32 DDC54             : 1;
  __REG32 DDC55             : 1;
  __REG32 DDC56             : 1;
  __REG32 DDC57             : 1;
  __REG32 DDC58             : 1;
  __REG32 DDC59             : 1;
  __REG32 DDC60             : 1;
  __REG32 DDC61             : 1;
  __REG32 DDC62             : 1;
  __REG32 DDC63             : 1;
} __gpio_ddcr0h_bits;

/* Data Direction Clear Register (GPIO_DDCR1L) */
typedef struct {
  __REG32 DDC64             : 1;
  __REG32 DDC65             : 1;
  __REG32 DDC66             : 1;
  __REG32 DDC67             : 1;
  __REG32 DDC68             : 1;
  __REG32 DDC69             : 1;
  __REG32 DDC70             : 1;
  __REG32 DDC71             : 1;
  __REG32 DDC72             : 1;
  __REG32 DDC73             : 1;
  __REG32 DDC74             : 1;
  __REG32 DDC75             : 1;
  __REG32 DDC76             : 1;
  __REG32 DDC77             : 1;
  __REG32 DDC78             : 1;
  __REG32 DDC79             : 1;
  __REG32 DDC80             : 1;
  __REG32 DDC81             : 1;
  __REG32 DDC82             : 1;
  __REG32 DDC83             : 1;
  __REG32 DDC84             : 1;
  __REG32 DDC85             : 1;
  __REG32 DDC86             : 1;
  __REG32 DDC87             : 1;
  __REG32 DDC88             : 1;
  __REG32 DDC89             : 1;
  __REG32 DDC90             : 1;
  __REG32 DDC91             : 1;
  __REG32 DDC92             : 1;
  __REG32 DDC93             : 1;
  __REG32 DDC94             : 1;
  __REG32 DDC95             : 1;
} __gpio_ddcr1l_bits;

/* Data Direction Clear Register (GPIO_DDCR1H) */
typedef struct {
  __REG32 DDC96             : 1;
  __REG32 DDC97             : 1;
  __REG32 DDC98             : 1;
  __REG32 DDC99             : 1;
  __REG32 DDC100            : 1;
  __REG32 DDC101            : 1;
  __REG32 DDC102            : 1;
  __REG32 DDC103            : 1;
  __REG32 DDC104            : 1;
  __REG32 DDC105            : 1;
  __REG32 DDC106            : 1;
  __REG32 DDC107            : 1;
  __REG32 DDC108            : 1;
  __REG32 DDC109            : 1;
  __REG32 DDC110            : 1;
  __REG32 DDC111            : 1;
  __REG32 DDC112            : 1;
  __REG32 DDC113            : 1;
  __REG32 DDC114            : 1;
  __REG32 DDC115            : 1;
  __REG32 DDC116            : 1;
  __REG32 DDC117            : 1;
  __REG32 DDC118            : 1;
  __REG32 DDC119            : 1;
  __REG32 DDC120            : 1;
  __REG32 DDC121            : 1;
  __REG32 DDC122            : 1;
  __REG32 DDC123            : 1;
  __REG32 DDC124            : 1;
  __REG32 DDC125            : 1;
  __REG32 DDC126            : 1;
  __REG32 DDC127            : 1;
} __gpio_ddcr1h_bits;

/* Data Direction Clear Register (GPIO_DDCR2L) */
typedef struct {
  __REG32 DDC128            : 1;
  __REG32 DDC129            : 1;
  __REG32 DDC130            : 1;
  __REG32 DDC131            : 1;
  __REG32 DDC132            : 1;
  __REG32 DDC133            : 1;
  __REG32 DDC134            : 1;
  __REG32 DDC135            : 1;
  __REG32 DDC136            : 1;
  __REG32 DDC137            : 1;
  __REG32 DDC138            : 1;
  __REG32 DDC139            : 1;
  __REG32 DDC140            : 1;
  __REG32 DDC141            : 1;
  __REG32 DDC142            : 1;
  __REG32 DDC143            : 1;
  __REG32 DDC144            : 1;
  __REG32 DDC145            : 1;
  __REG32 DDC146            : 1;
  __REG32 DDC147            : 1;
  __REG32 DDC148            : 1;
  __REG32 DDC149            : 1;
  __REG32 DDC150            : 1;
  __REG32 DDC151            : 1;
  __REG32 DDC152            : 1;
  __REG32 DDC153            : 1;
  __REG32 DDC154            : 1;
  __REG32 DDC155            : 1;
  __REG32 DDC156            : 1;
  __REG32 DDC157            : 1;
  __REG32 DDC158            : 1;
  __REG32 DDC159            : 1;
} __gpio_ddcr2l_bits;

/* Data Direction Clear Register (GPIO_DDCR2H) */
typedef struct {
  __REG32 DDC160            : 1;
  __REG32 DDC161            : 1;
  __REG32 DDC162            : 1;
  __REG32 DDC163            : 1;
  __REG32 DDC164            : 1;
  __REG32 DDC165            : 1;
  __REG32 DDC166            : 1;
  __REG32 DDC167            : 1;
  __REG32 DDC168            : 1;
  __REG32 DDC169            : 1;
  __REG32 DDC170            : 1;
  __REG32 DDC171            : 1;
  __REG32 DDC172            : 1;
  __REG32 DDC173            : 1;
  __REG32 DDC174            : 1;
  __REG32 DDC175            : 1;
  __REG32 DDC176            : 1;
  __REG32 DDC177            : 1;
  __REG32 DDC178            : 1;
  __REG32 DDC179            : 1;
  __REG32 DDC180            : 1;
  __REG32 DDC181            : 1;
  __REG32 DDC182            : 1;
  __REG32 DDC183            : 1;
  __REG32 DDC184            : 1;
  __REG32 DDC185            : 1;
  __REG32 DDC186            : 1;
  __REG32 DDC187            : 1;
  __REG32 DDC188            : 1;
  __REG32 DDC189            : 1;
  __REG32 DDC190            : 1;
  __REG32 DDC191            : 1;
} __gpio_ddcr2h_bits;

/* Data Direction Clear Register (GPIO_DDCR3L) */
typedef struct {
  __REG32 DDC192            : 1;
  __REG32 DDC193            : 1;
  __REG32 DDC194            : 1;
  __REG32 DDC195            : 1;
  __REG32 DDC196            : 1;
  __REG32 DDC197            : 1;
  __REG32 DDC198            : 1;
  __REG32 DDC199            : 1;
  __REG32 DDC200            : 1;
  __REG32 DDC201            : 1;
  __REG32 DDC202            : 1;
  __REG32 DDC203            : 1;
  __REG32 DDC204            : 1;
  __REG32 DDC205            : 1;
  __REG32 DDC206            : 1;
  __REG32 DDC207            : 1;
  __REG32 DDC208            : 1;
  __REG32 DDC209            : 1;
  __REG32 DDC210            : 1;
  __REG32 DDC211            : 1;
  __REG32 DDC212            : 1;
  __REG32 DDC213            : 1;
  __REG32 DDC214            : 1;
  __REG32 DDC215            : 1;
  __REG32 DDC216            : 1;
  __REG32 DDC217            : 1;
  __REG32 DDC218            : 1;
  __REG32 DDC219            : 1;
  __REG32 DDC220            : 1;
  __REG32 DDC221            : 1;
  __REG32 DDC222            : 1;
  __REG32 DDC223            : 1;
} __gpio_ddcr3l_bits;

/* Data Direction Clear Register (GPIO_DDCR3H) */
typedef struct {
  __REG32 DDC224            : 1;
  __REG32 DDC225            : 1;
  __REG32 DDC226            : 1;
  __REG32 DDC227            : 1;
  __REG32 DDC228            : 1;
  __REG32 DDC229            : 1;
  __REG32 DDC230            : 1;
  __REG32 DDC231            : 1;
  __REG32 DDC232            : 1;
  __REG32 DDC233            : 1;
  __REG32 DDC234            : 1;
  __REG32 DDC235            : 1;
  __REG32 DDC236            : 1;
  __REG32 DDC237            : 1;
  __REG32 DDC238            : 1;
  __REG32 DDC239            : 1;
  __REG32 DDC240            : 1;
  __REG32 DDC241            : 1;
  __REG32 DDC242            : 1;
  __REG32 DDC243            : 1;
  __REG32 DDC244            : 1;
  __REG32 DDC245            : 1;
  __REG32 DDC246            : 1;
  __REG32 DDC247            : 1;
  __REG32 DDC248            : 1;
  __REG32 DDC249            : 1;
  __REG32 DDC250            : 1;
  __REG32 DDC251            : 1;
  __REG32 DDC252            : 1;
  __REG32 DDC253            : 1;
  __REG32 DDC254            : 1;
  __REG32 DDC255            : 1;
} __gpio_ddcr3h_bits;

/* Data Direction Clear Register (GPIO_DDCR4L) */
typedef struct {
  __REG32 DDC256            : 1;
  __REG32 DDC257            : 1;
  __REG32 DDC258            : 1;
  __REG32 DDC259            : 1;
  __REG32 DDC260            : 1;
  __REG32 DDC261            : 1;
  __REG32 DDC262            : 1;
  __REG32 DDC263            : 1;
  __REG32 DDC264            : 1;
  __REG32 DDC265            : 1;
  __REG32 DDC266            : 1;
  __REG32 DDC267            : 1;
  __REG32 DDC268            : 1;
  __REG32 DDC269            : 1;
  __REG32 DDC270            : 1;
  __REG32 DDC271            : 1;
  __REG32 DDC272            : 1;
  __REG32 DDC273            : 1;
  __REG32 DDC274            : 1;
  __REG32 DDC275            : 1;
  __REG32 DDC276            : 1;
  __REG32 DDC277            : 1;
  __REG32 DDC278            : 1;
  __REG32 DDC279            : 1;
  __REG32 DDC280            : 1;
  __REG32 DDC281            : 1;
  __REG32 DDC282            : 1;
  __REG32 DDC283            : 1;
  __REG32 DDC284            : 1;
  __REG32 DDC285            : 1;
  __REG32 DDC286            : 1;
  __REG32 DDC287            : 1;
} __gpio_ddcr4l_bits;

/* Data Direction Clear Register (GPIO_DDCR4H) */
typedef struct {
  __REG32 DDC288            : 1;
  __REG32 DDC289            : 1;
  __REG32 DDC290            : 1;
  __REG32 DDC291            : 1;
  __REG32 DDC292            : 1;
  __REG32 DDC293            : 1;
  __REG32 DDC294            : 1;
  __REG32 DDC295            : 1;
  __REG32 DDC296            : 1;
  __REG32 DDC297            : 1;
  __REG32 DDC298            : 1;
  __REG32 DDC299            : 1;
  __REG32 DDC300            : 1;
  __REG32 DDC301            : 1;
  __REG32 DDC302            : 1;
  __REG32 DDC303            : 1;
  __REG32 DDC304            : 1;
  __REG32 DDC305            : 1;
  __REG32 DDC306            : 1;
  __REG32 DDC307            : 1;
  __REG32 DDC308            : 1;
  __REG32 DDC309            : 1;
  __REG32 DDC310            : 1;
  __REG32 DDC311            : 1;
  __REG32 DDC312            : 1;
  __REG32 DDC313            : 1;
  __REG32 DDC314            : 1;
  __REG32 DDC315            : 1;
  __REG32 DDC316            : 1;
  __REG32 DDC317            : 1;
  __REG32 DDC318            : 1;
  __REG32 DDC319            : 1;
} __gpio_ddcr4h_bits;

/* Data Direction Clear Register (GPIO_DDCR5L) */
typedef struct {
  __REG32 DDC320            : 1;
  __REG32 DDC321            : 1;
  __REG32 DDC322            : 1;
  __REG32 DDC323            : 1;
  __REG32 DDC324            : 1;
  __REG32 DDC325            : 1;
  __REG32 DDC326            : 1;
  __REG32 DDC327            : 1;
  __REG32 DDC328            : 1;
  __REG32 DDC329            : 1;
  __REG32 DDC330            : 1;
  __REG32 DDC331            : 1;
  __REG32 DDC332            : 1;
  __REG32 DDC333            : 1;
  __REG32 DDC334            : 1;
  __REG32 DDC335            : 1;
  __REG32 DDC336            : 1;
  __REG32 DDC337            : 1;
  __REG32 DDC338            : 1;
  __REG32 DDC339            : 1;
  __REG32 DDC340            : 1;
  __REG32 DDC341            : 1;
  __REG32 DDC342            : 1;
  __REG32 DDC343            : 1;
  __REG32 DDC344            : 1;
  __REG32 DDC345            : 1;
  __REG32 DDC346            : 1;
  __REG32 DDC347            : 1;
  __REG32 DDC348            : 1;
  __REG32 DDC349            : 1;
  __REG32 DDC350            : 1;
  __REG32 DDC351            : 1;
} __gpio_ddcr5l_bits;

/* Data Direction Clear Register (GPIO_DDCR5H) */
typedef struct {
  __REG32 DDC352            : 1;
  __REG32 DDC353            : 1;
  __REG32 DDC354            : 1;
  __REG32 DDC355            : 1;
  __REG32 DDC356            : 1;
  __REG32 DDC357            : 1;
  __REG32 DDC358            : 1;
  __REG32 DDC359            : 1;
  __REG32 DDC360            : 1;
  __REG32 DDC361            : 1;
  __REG32 DDC362            : 1;
  __REG32 DDC363            : 1;
  __REG32 DDC364            : 1;
  __REG32 DDC365            : 1;
  __REG32 DDC366            : 1;
  __REG32 DDC367            : 1;
  __REG32 DDC368            : 1;
  __REG32 DDC369            : 1;
  __REG32 DDC370            : 1;
  __REG32 DDC371            : 1;
  __REG32 DDC372            : 1;
  __REG32 DDC373            : 1;
  __REG32 DDC374            : 1;
  __REG32 DDC375            : 1;
  __REG32 DDC376            : 1;
  __REG32 DDC377            : 1;
  __REG32 DDC378            : 1;
  __REG32 DDC379            : 1;
  __REG32 DDC380            : 1;
  __REG32 DDC381            : 1;
  __REG32 DDC382            : 1;
  __REG32 DDC383            : 1;
} __gpio_ddcr5h_bits;

/* Data Direction Clear Register (GPIO_DDCR6L) */
typedef struct {
  __REG32 DDC384            : 1;
  __REG32 DDC385            : 1;
  __REG32 DDC386            : 1;
  __REG32 DDC387            : 1;
  __REG32 DDC388            : 1;
  __REG32 DDC389            : 1;
  __REG32 DDC390            : 1;
  __REG32 DDC391            : 1;
  __REG32 DDC392            : 1;
  __REG32 DDC393            : 1;
  __REG32 DDC394            : 1;
  __REG32 DDC395            : 1;
  __REG32 DDC396            : 1;
  __REG32 DDC397            : 1;
  __REG32 DDC398            : 1;
  __REG32 DDC399            : 1;
  __REG32 DDC400            : 1;
  __REG32 DDC401            : 1;
  __REG32 DDC402            : 1;
  __REG32 DDC403            : 1;
  __REG32 DDC404            : 1;
  __REG32 DDC405            : 1;
  __REG32 DDC406            : 1;
  __REG32 DDC407            : 1;
  __REG32 DDC408            : 1;
  __REG32 DDC409            : 1;
  __REG32 DDC410            : 1;
  __REG32 DDC411            : 1;
  __REG32 DDC412            : 1;
  __REG32 DDC413            : 1;
  __REG32 DDC414            : 1;
  __REG32 DDC415            : 1;
} __gpio_ddcr6l_bits;

/* Data Direction Clear Register (GPIO_DDCR6H) */
typedef struct {
  __REG32 DDC416            : 1;
  __REG32 DDC417            : 1;
  __REG32 DDC418            : 1;
  __REG32 DDC419            : 1;
  __REG32 DDC420            : 1;
  __REG32 DDC421            : 1;
  __REG32 DDC422            : 1;
  __REG32 DDC423            : 1;
  __REG32 DDC424            : 1;
  __REG32 DDC425            : 1;
  __REG32 DDC426            : 1;
  __REG32 DDC427            : 1;
  __REG32 DDC428            : 1;
  __REG32 DDC429            : 1;
  __REG32 DDC430            : 1;
  __REG32 DDC431            : 1;
  __REG32 DDC432            : 1;
  __REG32 DDC433            : 1;
  __REG32 DDC434            : 1;
  __REG32 DDC435            : 1;
  __REG32 DDC436            : 1;
  __REG32 DDC437            : 1;
  __REG32 DDC438            : 1;
  __REG32 DDC439            : 1;
  __REG32 DDC440            : 1;
  __REG32 DDC441            : 1;
  __REG32 DDC442            : 1;
  __REG32 DDC443            : 1;
  __REG32 DDC444            : 1;
  __REG32 DDC445            : 1;
  __REG32 DDC446            : 1;
  __REG32 DDC447            : 1;
} __gpio_ddcr6h_bits;

/* Data Direction Clear Register (GPIO_DDCR7L) */
typedef struct {
  __REG32 DDC448            : 1;
  __REG32 DDC449            : 1;
  __REG32 DDC450            : 1;
  __REG32 DDC451            : 1;
  __REG32 DDC452            : 1;
  __REG32 DDC453            : 1;
  __REG32 DDC454            : 1;
  __REG32 DDC455            : 1;
  __REG32 DDC456            : 1;
  __REG32 DDC457            : 1;
  __REG32 DDC458            : 1;
  __REG32 DDC459            : 1;
  __REG32 DDC460            : 1;
  __REG32 DDC461            : 1;
  __REG32 DDC462            : 1;
  __REG32 DDC463            : 1;
  __REG32 DDC464            : 1;
  __REG32 DDC465            : 1;
  __REG32 DDC466            : 1;
  __REG32 DDC467            : 1;
  __REG32 DDC468            : 1;
  __REG32 DDC469            : 1;
  __REG32 DDC470            : 1;
  __REG32 DDC471            : 1;
  __REG32 DDC472            : 1;
  __REG32 DDC473            : 1;
  __REG32 DDC474            : 1;
  __REG32 DDC475            : 1;
  __REG32 DDC476            : 1;
  __REG32 DDC477            : 1;
  __REG32 DDC478            : 1;
  __REG32 DDC479            : 1;
} __gpio_ddcr7l_bits;

/* Data Direction Clear Register (GPIO_DDCR7H) */
typedef struct {
  __REG32 DDC480            : 1;
  __REG32 DDC481            : 1;
  __REG32 DDC482            : 1;
  __REG32 DDC483            : 1;
  __REG32 DDC484            : 1;
  __REG32 DDC485            : 1;
  __REG32 DDC486            : 1;
  __REG32 DDC487            : 1;
  __REG32 DDC488            : 1;
  __REG32 DDC489            : 1;
  __REG32 DDC490            : 1;
  __REG32 DDC491            : 1;
  __REG32 DDC492            : 1;
  __REG32 DDC493            : 1;
  __REG32 DDC494            : 1;
  __REG32 DDC495            : 1;
  __REG32 DDC496            : 1;
  __REG32 DDC497            : 1;
  __REG32 DDC498            : 1;
  __REG32 DDC499            : 1;
  __REG32 DDC500            : 1;
  __REG32 DDC501            : 1;
  __REG32 DDC502            : 1;
  __REG32 DDC503            : 1;
  __REG32 DDC504            : 1;
  __REG32 DDC505            : 1;
  __REG32 DDC506            : 1;
  __REG32 DDC507            : 1;
  __REG32 DDC508            : 1;
  __REG32 DDC509            : 1;
  __REG32 DDC510            : 1;
  __REG32 DDC511            : 1;
} __gpio_ddcr7h_bits;

/* Port Output Data Register (GPIO_PODR0L) */
typedef struct {
  __REG32 POD0              : 1;
  __REG32 POD1              : 1;
  __REG32 POD2              : 1;
  __REG32 POD3              : 1;
  __REG32 POD4              : 1;
  __REG32 POD5              : 1;
  __REG32 POD6              : 1;
  __REG32 POD7              : 1;
  __REG32 POD8              : 1;
  __REG32 POD9              : 1;
  __REG32 POD10             : 1;
  __REG32 POD11             : 1;
  __REG32 POD12             : 1;
  __REG32 POD13             : 1;
  __REG32 POD14             : 1;
  __REG32 POD15             : 1;
  __REG32 POD16             : 1;
  __REG32 POD17             : 1;
  __REG32 POD18             : 1;
  __REG32 POD19             : 1;
  __REG32 POD20             : 1;
  __REG32 POD21             : 1;
  __REG32 POD22             : 1;
  __REG32 POD23             : 1;
  __REG32 POD24             : 1;
  __REG32 POD25             : 1;
  __REG32 POD26             : 1;
  __REG32 POD27             : 1;
  __REG32 POD28             : 1;
  __REG32 POD29             : 1;
  __REG32 POD30             : 1;
  __REG32 POD31             : 1;
} __gpio_podr0l_bits;

/* Port Output Data Register (GPIO_PODR0H) */
typedef struct {
  __REG32 POD32             : 1;
  __REG32 POD33             : 1;
  __REG32 POD34             : 1;
  __REG32 POD35             : 1;
  __REG32 POD36             : 1;
  __REG32 POD37             : 1;
  __REG32 POD38             : 1;
  __REG32 POD39             : 1;
  __REG32 POD40             : 1;
  __REG32 POD41             : 1;
  __REG32 POD42             : 1;
  __REG32 POD43             : 1;
  __REG32 POD44             : 1;
  __REG32 POD45             : 1;
  __REG32 POD46             : 1;
  __REG32 POD47             : 1;
  __REG32 POD48             : 1;
  __REG32 POD49             : 1;
  __REG32 POD50             : 1;
  __REG32 POD51             : 1;
  __REG32 POD52             : 1;
  __REG32 POD53             : 1;
  __REG32 POD54             : 1;
  __REG32 POD55             : 1;
  __REG32 POD56             : 1;
  __REG32 POD57             : 1;
  __REG32 POD58             : 1;
  __REG32 POD59             : 1;
  __REG32 POD60             : 1;
  __REG32 POD61             : 1;
  __REG32 POD62             : 1;
  __REG32 POD63             : 1;
} __gpio_podr0h_bits;

/* Port Output Data Register (GPIO_PODR1L) */
typedef struct {
  __REG32 POD64             : 1;
  __REG32 POD65             : 1;
  __REG32 POD66             : 1;
  __REG32 POD67             : 1;
  __REG32 POD68             : 1;
  __REG32 POD69             : 1;
  __REG32 POD70             : 1;
  __REG32 POD71             : 1;
  __REG32 POD72             : 1;
  __REG32 POD73             : 1;
  __REG32 POD74             : 1;
  __REG32 POD75             : 1;
  __REG32 POD76             : 1;
  __REG32 POD77             : 1;
  __REG32 POD78             : 1;
  __REG32 POD79             : 1;
  __REG32 POD80             : 1;
  __REG32 POD81             : 1;
  __REG32 POD82             : 1;
  __REG32 POD83             : 1;
  __REG32 POD84             : 1;
  __REG32 POD85             : 1;
  __REG32 POD86             : 1;
  __REG32 POD87             : 1;
  __REG32 POD88             : 1;
  __REG32 POD89             : 1;
  __REG32 POD90             : 1;
  __REG32 POD91             : 1;
  __REG32 POD92             : 1;
  __REG32 POD93             : 1;
  __REG32 POD94             : 1;
  __REG32 POD95             : 1;
} __gpio_podr1l_bits;

/* Port Output Data Register (GPIO_PODR1H) */
typedef struct {
  __REG32 POD96             : 1;
  __REG32 POD97             : 1;
  __REG32 POD98             : 1;
  __REG32 POD99             : 1;
  __REG32 POD100            : 1;
  __REG32 POD101            : 1;
  __REG32 POD102            : 1;
  __REG32 POD103            : 1;
  __REG32 POD104            : 1;
  __REG32 POD105            : 1;
  __REG32 POD106            : 1;
  __REG32 POD107            : 1;
  __REG32 POD108            : 1;
  __REG32 POD109            : 1;
  __REG32 POD110            : 1;
  __REG32 POD111            : 1;
  __REG32 POD112            : 1;
  __REG32 POD113            : 1;
  __REG32 POD114            : 1;
  __REG32 POD115            : 1;
  __REG32 POD116            : 1;
  __REG32 POD117            : 1;
  __REG32 POD118            : 1;
  __REG32 POD119            : 1;
  __REG32 POD120            : 1;
  __REG32 POD121            : 1;
  __REG32 POD122            : 1;
  __REG32 POD123            : 1;
  __REG32 POD124            : 1;
  __REG32 POD125            : 1;
  __REG32 POD126            : 1;
  __REG32 POD127            : 1;
} __gpio_podr1h_bits;

/* Port Output Data Register (GPIO_PODR2L) */
typedef struct {
  __REG32 POD128            : 1;
  __REG32 POD129            : 1;
  __REG32 POD130            : 1;
  __REG32 POD131            : 1;
  __REG32 POD132            : 1;
  __REG32 POD133            : 1;
  __REG32 POD134            : 1;
  __REG32 POD135            : 1;
  __REG32 POD136            : 1;
  __REG32 POD137            : 1;
  __REG32 POD138            : 1;
  __REG32 POD139            : 1;
  __REG32 POD140            : 1;
  __REG32 POD141            : 1;
  __REG32 POD142            : 1;
  __REG32 POD143            : 1;
  __REG32 POD144            : 1;
  __REG32 POD145            : 1;
  __REG32 POD146            : 1;
  __REG32 POD147            : 1;
  __REG32 POD148            : 1;
  __REG32 POD149            : 1;
  __REG32 POD150            : 1;
  __REG32 POD151            : 1;
  __REG32 POD152            : 1;
  __REG32 POD153            : 1;
  __REG32 POD154            : 1;
  __REG32 POD155            : 1;
  __REG32 POD156            : 1;
  __REG32 POD157            : 1;
  __REG32 POD158            : 1;
  __REG32 POD159            : 1;
} __gpio_podr2l_bits;

/* Port Output Data Register (GPIO_PODR2H) */
typedef struct {
  __REG32 POD160            : 1;
  __REG32 POD161            : 1;
  __REG32 POD162            : 1;
  __REG32 POD163            : 1;
  __REG32 POD164            : 1;
  __REG32 POD165            : 1;
  __REG32 POD166            : 1;
  __REG32 POD167            : 1;
  __REG32 POD168            : 1;
  __REG32 POD169            : 1;
  __REG32 POD170            : 1;
  __REG32 POD171            : 1;
  __REG32 POD172            : 1;
  __REG32 POD173            : 1;
  __REG32 POD174            : 1;
  __REG32 POD175            : 1;
  __REG32 POD176            : 1;
  __REG32 POD177            : 1;
  __REG32 POD178            : 1;
  __REG32 POD179            : 1;
  __REG32 POD180            : 1;
  __REG32 POD181            : 1;
  __REG32 POD182            : 1;
  __REG32 POD183            : 1;
  __REG32 POD184            : 1;
  __REG32 POD185            : 1;
  __REG32 POD186            : 1;
  __REG32 POD187            : 1;
  __REG32 POD188            : 1;
  __REG32 POD189            : 1;
  __REG32 POD190            : 1;
  __REG32 POD191            : 1;
} __gpio_podr2h_bits;

/* Port Output Data Register (GPIO_PODR3L) */
typedef struct {
  __REG32 POD192            : 1;
  __REG32 POD193            : 1;
  __REG32 POD194            : 1;
  __REG32 POD195            : 1;
  __REG32 POD196            : 1;
  __REG32 POD197            : 1;
  __REG32 POD198            : 1;
  __REG32 POD199            : 1;
  __REG32 POD200            : 1;
  __REG32 POD201            : 1;
  __REG32 POD202            : 1;
  __REG32 POD203            : 1;
  __REG32 POD204            : 1;
  __REG32 POD205            : 1;
  __REG32 POD206            : 1;
  __REG32 POD207            : 1;
  __REG32 POD208            : 1;
  __REG32 POD209            : 1;
  __REG32 POD210            : 1;
  __REG32 POD211            : 1;
  __REG32 POD212            : 1;
  __REG32 POD213            : 1;
  __REG32 POD214            : 1;
  __REG32 POD215            : 1;
  __REG32 POD216            : 1;
  __REG32 POD217            : 1;
  __REG32 POD218            : 1;
  __REG32 POD219            : 1;
  __REG32 POD220            : 1;
  __REG32 POD221            : 1;
  __REG32 POD222            : 1;
  __REG32 POD223            : 1;
} __gpio_podr3l_bits;

/* Port Output Data Register (GPIO_PODR3H) */
typedef struct {
  __REG32 POD224            : 1;
  __REG32 POD225            : 1;
  __REG32 POD226            : 1;
  __REG32 POD227            : 1;
  __REG32 POD228            : 1;
  __REG32 POD229            : 1;
  __REG32 POD230            : 1;
  __REG32 POD231            : 1;
  __REG32 POD232            : 1;
  __REG32 POD233            : 1;
  __REG32 POD234            : 1;
  __REG32 POD235            : 1;
  __REG32 POD236            : 1;
  __REG32 POD237            : 1;
  __REG32 POD238            : 1;
  __REG32 POD239            : 1;
  __REG32 POD240            : 1;
  __REG32 POD241            : 1;
  __REG32 POD242            : 1;
  __REG32 POD243            : 1;
  __REG32 POD244            : 1;
  __REG32 POD245            : 1;
  __REG32 POD246            : 1;
  __REG32 POD247            : 1;
  __REG32 POD248            : 1;
  __REG32 POD249            : 1;
  __REG32 POD250            : 1;
  __REG32 POD251            : 1;
  __REG32 POD252            : 1;
  __REG32 POD253            : 1;
  __REG32 POD254            : 1;
  __REG32 POD255            : 1;
} __gpio_podr3h_bits;

/* Port Output Data Register (GPIO_PODR4L) */
typedef struct {
  __REG32 POD256            : 1;
  __REG32 POD257            : 1;
  __REG32 POD258            : 1;
  __REG32 POD259            : 1;
  __REG32 POD260            : 1;
  __REG32 POD261            : 1;
  __REG32 POD262            : 1;
  __REG32 POD263            : 1;
  __REG32 POD264            : 1;
  __REG32 POD265            : 1;
  __REG32 POD266            : 1;
  __REG32 POD267            : 1;
  __REG32 POD268            : 1;
  __REG32 POD269            : 1;
  __REG32 POD270            : 1;
  __REG32 POD271            : 1;
  __REG32 POD272            : 1;
  __REG32 POD273            : 1;
  __REG32 POD274            : 1;
  __REG32 POD275            : 1;
  __REG32 POD276            : 1;
  __REG32 POD277            : 1;
  __REG32 POD278            : 1;
  __REG32 POD279            : 1;
  __REG32 POD280            : 1;
  __REG32 POD281            : 1;
  __REG32 POD282            : 1;
  __REG32 POD283            : 1;
  __REG32 POD284            : 1;
  __REG32 POD285            : 1;
  __REG32 POD286            : 1;
  __REG32 POD287            : 1;
} __gpio_podr4l_bits;

/* Port Output Data Register (GPIO_PODR4H) */
typedef struct {
  __REG32 POD288            : 1;
  __REG32 POD289            : 1;
  __REG32 POD290            : 1;
  __REG32 POD291            : 1;
  __REG32 POD292            : 1;
  __REG32 POD293            : 1;
  __REG32 POD294            : 1;
  __REG32 POD295            : 1;
  __REG32 POD296            : 1;
  __REG32 POD297            : 1;
  __REG32 POD298            : 1;
  __REG32 POD299            : 1;
  __REG32 POD300            : 1;
  __REG32 POD301            : 1;
  __REG32 POD302            : 1;
  __REG32 POD303            : 1;
  __REG32 POD304            : 1;
  __REG32 POD305            : 1;
  __REG32 POD306            : 1;
  __REG32 POD307            : 1;
  __REG32 POD308            : 1;
  __REG32 POD309            : 1;
  __REG32 POD310            : 1;
  __REG32 POD311            : 1;
  __REG32 POD312            : 1;
  __REG32 POD313            : 1;
  __REG32 POD314            : 1;
  __REG32 POD315            : 1;
  __REG32 POD316            : 1;
  __REG32 POD317            : 1;
  __REG32 POD318            : 1;
  __REG32 POD319            : 1;
} __gpio_podr4h_bits;

/* Port Output Data Register (GPIO_PODR5L) */
typedef struct {
  __REG32 POD320            : 1;
  __REG32 POD321            : 1;
  __REG32 POD322            : 1;
  __REG32 POD323            : 1;
  __REG32 POD324            : 1;
  __REG32 POD325            : 1;
  __REG32 POD326            : 1;
  __REG32 POD327            : 1;
  __REG32 POD328            : 1;
  __REG32 POD329            : 1;
  __REG32 POD330            : 1;
  __REG32 POD331            : 1;
  __REG32 POD332            : 1;
  __REG32 POD333            : 1;
  __REG32 POD334            : 1;
  __REG32 POD335            : 1;
  __REG32 POD336            : 1;
  __REG32 POD337            : 1;
  __REG32 POD338            : 1;
  __REG32 POD339            : 1;
  __REG32 POD340            : 1;
  __REG32 POD341            : 1;
  __REG32 POD342            : 1;
  __REG32 POD343            : 1;
  __REG32 POD344            : 1;
  __REG32 POD345            : 1;
  __REG32 POD346            : 1;
  __REG32 POD347            : 1;
  __REG32 POD348            : 1;
  __REG32 POD349            : 1;
  __REG32 POD350            : 1;
  __REG32 POD351            : 1;
} __gpio_podr5l_bits;

/* Port Output Data Register (GPIO_PODR5H) */
typedef struct {
  __REG32 POD352            : 1;
  __REG32 POD353            : 1;
  __REG32 POD354            : 1;
  __REG32 POD355            : 1;
  __REG32 POD356            : 1;
  __REG32 POD357            : 1;
  __REG32 POD358            : 1;
  __REG32 POD359            : 1;
  __REG32 POD360            : 1;
  __REG32 POD361            : 1;
  __REG32 POD362            : 1;
  __REG32 POD363            : 1;
  __REG32 POD364            : 1;
  __REG32 POD365            : 1;
  __REG32 POD366            : 1;
  __REG32 POD367            : 1;
  __REG32 POD368            : 1;
  __REG32 POD369            : 1;
  __REG32 POD370            : 1;
  __REG32 POD371            : 1;
  __REG32 POD372            : 1;
  __REG32 POD373            : 1;
  __REG32 POD374            : 1;
  __REG32 POD375            : 1;
  __REG32 POD376            : 1;
  __REG32 POD377            : 1;
  __REG32 POD378            : 1;
  __REG32 POD379            : 1;
  __REG32 POD380            : 1;
  __REG32 POD381            : 1;
  __REG32 POD382            : 1;
  __REG32 POD383            : 1;
} __gpio_podr5h_bits;

/* Port Output Data Register (GPIO_PODR6L) */
typedef struct {
  __REG32 POD384            : 1;
  __REG32 POD385            : 1;
  __REG32 POD386            : 1;
  __REG32 POD387            : 1;
  __REG32 POD388            : 1;
  __REG32 POD389            : 1;
  __REG32 POD390            : 1;
  __REG32 POD391            : 1;
  __REG32 POD392            : 1;
  __REG32 POD393            : 1;
  __REG32 POD394            : 1;
  __REG32 POD395            : 1;
  __REG32 POD396            : 1;
  __REG32 POD397            : 1;
  __REG32 POD398            : 1;
  __REG32 POD399            : 1;
  __REG32 POD400            : 1;
  __REG32 POD401            : 1;
  __REG32 POD402            : 1;
  __REG32 POD403            : 1;
  __REG32 POD404            : 1;
  __REG32 POD405            : 1;
  __REG32 POD406            : 1;
  __REG32 POD407            : 1;
  __REG32 POD408            : 1;
  __REG32 POD409            : 1;
  __REG32 POD410            : 1;
  __REG32 POD411            : 1;
  __REG32 POD412            : 1;
  __REG32 POD413            : 1;
  __REG32 POD414            : 1;
  __REG32 POD415            : 1;
} __gpio_podr6l_bits;

/* Port Output Data Register (GPIO_PODR6H) */
typedef struct {
  __REG32 POD416            : 1;
  __REG32 POD417            : 1;
  __REG32 POD418            : 1;
  __REG32 POD419            : 1;
  __REG32 POD420            : 1;
  __REG32 POD421            : 1;
  __REG32 POD422            : 1;
  __REG32 POD423            : 1;
  __REG32 POD424            : 1;
  __REG32 POD425            : 1;
  __REG32 POD426            : 1;
  __REG32 POD427            : 1;
  __REG32 POD428            : 1;
  __REG32 POD429            : 1;
  __REG32 POD430            : 1;
  __REG32 POD431            : 1;
  __REG32 POD432            : 1;
  __REG32 POD433            : 1;
  __REG32 POD434            : 1;
  __REG32 POD435            : 1;
  __REG32 POD436            : 1;
  __REG32 POD437            : 1;
  __REG32 POD438            : 1;
  __REG32 POD439            : 1;
  __REG32 POD440            : 1;
  __REG32 POD441            : 1;
  __REG32 POD442            : 1;
  __REG32 POD443            : 1;
  __REG32 POD444            : 1;
  __REG32 POD445            : 1;
  __REG32 POD446            : 1;
  __REG32 POD447            : 1;
} __gpio_podr6h_bits;

/* Port Output Data Register (GPIO_PODR7L) */
typedef struct {
  __REG32 POD448            : 1;
  __REG32 POD449            : 1;
  __REG32 POD450            : 1;
  __REG32 POD451            : 1;
  __REG32 POD452            : 1;
  __REG32 POD453            : 1;
  __REG32 POD454            : 1;
  __REG32 POD455            : 1;
  __REG32 POD456            : 1;
  __REG32 POD457            : 1;
  __REG32 POD458            : 1;
  __REG32 POD459            : 1;
  __REG32 POD460            : 1;
  __REG32 POD461            : 1;
  __REG32 POD462            : 1;
  __REG32 POD463            : 1;
  __REG32 POD464            : 1;
  __REG32 POD465            : 1;
  __REG32 POD466            : 1;
  __REG32 POD467            : 1;
  __REG32 POD468            : 1;
  __REG32 POD469            : 1;
  __REG32 POD470            : 1;
  __REG32 POD471            : 1;
  __REG32 POD472            : 1;
  __REG32 POD473            : 1;
  __REG32 POD474            : 1;
  __REG32 POD475            : 1;
  __REG32 POD476            : 1;
  __REG32 POD477            : 1;
  __REG32 POD478            : 1;
  __REG32 POD479            : 1;
} __gpio_podr7l_bits;

/* Port Output Data Register (GPIO_PODR7H) */
typedef struct {
  __REG32 POD480            : 1;
  __REG32 POD481            : 1;
  __REG32 POD482            : 1;
  __REG32 POD483            : 1;
  __REG32 POD484            : 1;
  __REG32 POD485            : 1;
  __REG32 POD486            : 1;
  __REG32 POD487            : 1;
  __REG32 POD488            : 1;
  __REG32 POD489            : 1;
  __REG32 POD490            : 1;
  __REG32 POD491            : 1;
  __REG32 POD492            : 1;
  __REG32 POD493            : 1;
  __REG32 POD494            : 1;
  __REG32 POD495            : 1;
  __REG32 POD496            : 1;
  __REG32 POD497            : 1;
  __REG32 POD498            : 1;
  __REG32 POD499            : 1;
  __REG32 POD500            : 1;
  __REG32 POD501            : 1;
  __REG32 POD502            : 1;
  __REG32 POD503            : 1;
  __REG32 POD504            : 1;
  __REG32 POD505            : 1;
  __REG32 POD506            : 1;
  __REG32 POD507            : 1;
  __REG32 POD508            : 1;
  __REG32 POD509            : 1;
  __REG32 POD510            : 1;
  __REG32 POD511            : 1;
} __gpio_podr7h_bits;

/* Port Output Set Register (GPIO_POSR0L) */
typedef struct {
  __REG32 POS0              : 1;
  __REG32 POS1              : 1;
  __REG32 POS2              : 1;
  __REG32 POS3              : 1;
  __REG32 POS4              : 1;
  __REG32 POS5              : 1;
  __REG32 POS6              : 1;
  __REG32 POS7              : 1;
  __REG32 POS8              : 1;
  __REG32 POS9              : 1;
  __REG32 POS10             : 1;
  __REG32 POS11             : 1;
  __REG32 POS12             : 1;
  __REG32 POS13             : 1;
  __REG32 POS14             : 1;
  __REG32 POS15             : 1;
  __REG32 POS16             : 1;
  __REG32 POS17             : 1;
  __REG32 POS18             : 1;
  __REG32 POS19             : 1;
  __REG32 POS20             : 1;
  __REG32 POS21             : 1;
  __REG32 POS22             : 1;
  __REG32 POS23             : 1;
  __REG32 POS24             : 1;
  __REG32 POS25             : 1;
  __REG32 POS26             : 1;
  __REG32 POS27             : 1;
  __REG32 POS28             : 1;
  __REG32 POS29             : 1;
  __REG32 POS30             : 1;
  __REG32 POS31             : 1;
} __gpio_posr0l_bits;

/* Port Output Set Register (GPIO_POSR0H) */
typedef struct {
  __REG32 POS32             : 1;
  __REG32 POS33             : 1;
  __REG32 POS34             : 1;
  __REG32 POS35             : 1;
  __REG32 POS36             : 1;
  __REG32 POS37             : 1;
  __REG32 POS38             : 1;
  __REG32 POS39             : 1;
  __REG32 POS40             : 1;
  __REG32 POS41             : 1;
  __REG32 POS42             : 1;
  __REG32 POS43             : 1;
  __REG32 POS44             : 1;
  __REG32 POS45             : 1;
  __REG32 POS46             : 1;
  __REG32 POS47             : 1;
  __REG32 POS48             : 1;
  __REG32 POS49             : 1;
  __REG32 POS50             : 1;
  __REG32 POS51             : 1;
  __REG32 POS52             : 1;
  __REG32 POS53             : 1;
  __REG32 POS54             : 1;
  __REG32 POS55             : 1;
  __REG32 POS56             : 1;
  __REG32 POS57             : 1;
  __REG32 POS58             : 1;
  __REG32 POS59             : 1;
  __REG32 POS60             : 1;
  __REG32 POS61             : 1;
  __REG32 POS62             : 1;
  __REG32 POS63             : 1;
} __gpio_posr0h_bits;

/* Port Output Set Register (GPIO_POSR1L) */
typedef struct {
  __REG32 POS64             : 1;
  __REG32 POS65             : 1;
  __REG32 POS66             : 1;
  __REG32 POS67             : 1;
  __REG32 POS68             : 1;
  __REG32 POS69             : 1;
  __REG32 POS70             : 1;
  __REG32 POS71             : 1;
  __REG32 POS72             : 1;
  __REG32 POS73             : 1;
  __REG32 POS74             : 1;
  __REG32 POS75             : 1;
  __REG32 POS76             : 1;
  __REG32 POS77             : 1;
  __REG32 POS78             : 1;
  __REG32 POS79             : 1;
  __REG32 POS80             : 1;
  __REG32 POS81             : 1;
  __REG32 POS82             : 1;
  __REG32 POS83             : 1;
  __REG32 POS84             : 1;
  __REG32 POS85             : 1;
  __REG32 POS86             : 1;
  __REG32 POS87             : 1;
  __REG32 POS88             : 1;
  __REG32 POS89             : 1;
  __REG32 POS90             : 1;
  __REG32 POS91             : 1;
  __REG32 POS92             : 1;
  __REG32 POS93             : 1;
  __REG32 POS94             : 1;
  __REG32 POS95             : 1;
} __gpio_posr1l_bits;

/* Port Output Set Register (GPIO_POSR1H) */
typedef struct {
  __REG32 POS96             : 1;
  __REG32 POS97             : 1;
  __REG32 POS98             : 1;
  __REG32 POS99             : 1;
  __REG32 POS100            : 1;
  __REG32 POS101            : 1;
  __REG32 POS102            : 1;
  __REG32 POS103            : 1;
  __REG32 POS104            : 1;
  __REG32 POS105            : 1;
  __REG32 POS106            : 1;
  __REG32 POS107            : 1;
  __REG32 POS108            : 1;
  __REG32 POS109            : 1;
  __REG32 POS110            : 1;
  __REG32 POS111            : 1;
  __REG32 POS112            : 1;
  __REG32 POS113            : 1;
  __REG32 POS114            : 1;
  __REG32 POS115            : 1;
  __REG32 POS116            : 1;
  __REG32 POS117            : 1;
  __REG32 POS118            : 1;
  __REG32 POS119            : 1;
  __REG32 POS120            : 1;
  __REG32 POS121            : 1;
  __REG32 POS122            : 1;
  __REG32 POS123            : 1;
  __REG32 POS124            : 1;
  __REG32 POS125            : 1;
  __REG32 POS126            : 1;
  __REG32 POS127            : 1;
} __gpio_posr1h_bits;

/* Port Output Set Register (GPIO_POSR2L) */
typedef struct {
  __REG32 POS128            : 1;
  __REG32 POS129            : 1;
  __REG32 POS130            : 1;
  __REG32 POS131            : 1;
  __REG32 POS132            : 1;
  __REG32 POS133            : 1;
  __REG32 POS134            : 1;
  __REG32 POS135            : 1;
  __REG32 POS136            : 1;
  __REG32 POS137            : 1;
  __REG32 POS138            : 1;
  __REG32 POS139            : 1;
  __REG32 POS140            : 1;
  __REG32 POS141            : 1;
  __REG32 POS142            : 1;
  __REG32 POS143            : 1;
  __REG32 POS144            : 1;
  __REG32 POS145            : 1;
  __REG32 POS146            : 1;
  __REG32 POS147            : 1;
  __REG32 POS148            : 1;
  __REG32 POS149            : 1;
  __REG32 POS150            : 1;
  __REG32 POS151            : 1;
  __REG32 POS152            : 1;
  __REG32 POS153            : 1;
  __REG32 POS154            : 1;
  __REG32 POS155            : 1;
  __REG32 POS156            : 1;
  __REG32 POS157            : 1;
  __REG32 POS158            : 1;
  __REG32 POS159            : 1;
} __gpio_posr2l_bits;

/* Port Output Set Register (GPIO_POSR2H) */
typedef struct {
  __REG32 POS160            : 1;
  __REG32 POS161            : 1;
  __REG32 POS162            : 1;
  __REG32 POS163            : 1;
  __REG32 POS164            : 1;
  __REG32 POS165            : 1;
  __REG32 POS166            : 1;
  __REG32 POS167            : 1;
  __REG32 POS168            : 1;
  __REG32 POS169            : 1;
  __REG32 POS170            : 1;
  __REG32 POS171            : 1;
  __REG32 POS172            : 1;
  __REG32 POS173            : 1;
  __REG32 POS174            : 1;
  __REG32 POS175            : 1;
  __REG32 POS176            : 1;
  __REG32 POS177            : 1;
  __REG32 POS178            : 1;
  __REG32 POS179            : 1;
  __REG32 POS180            : 1;
  __REG32 POS181            : 1;
  __REG32 POS182            : 1;
  __REG32 POS183            : 1;
  __REG32 POS184            : 1;
  __REG32 POS185            : 1;
  __REG32 POS186            : 1;
  __REG32 POS187            : 1;
  __REG32 POS188            : 1;
  __REG32 POS189            : 1;
  __REG32 POS190            : 1;
  __REG32 POS191            : 1;
} __gpio_posr2h_bits;

/* Port Output Set Register (GPIO_POSR3L) */
typedef struct {
  __REG32 POS192            : 1;
  __REG32 POS193            : 1;
  __REG32 POS194            : 1;
  __REG32 POS195            : 1;
  __REG32 POS196            : 1;
  __REG32 POS197            : 1;
  __REG32 POS198            : 1;
  __REG32 POS199            : 1;
  __REG32 POS200            : 1;
  __REG32 POS201            : 1;
  __REG32 POS202            : 1;
  __REG32 POS203            : 1;
  __REG32 POS204            : 1;
  __REG32 POS205            : 1;
  __REG32 POS206            : 1;
  __REG32 POS207            : 1;
  __REG32 POS208            : 1;
  __REG32 POS209            : 1;
  __REG32 POS210            : 1;
  __REG32 POS211            : 1;
  __REG32 POS212            : 1;
  __REG32 POS213            : 1;
  __REG32 POS214            : 1;
  __REG32 POS215            : 1;
  __REG32 POS216            : 1;
  __REG32 POS217            : 1;
  __REG32 POS218            : 1;
  __REG32 POS219            : 1;
  __REG32 POS220            : 1;
  __REG32 POS221            : 1;
  __REG32 POS222            : 1;
  __REG32 POS223            : 1;
} __gpio_posr3l_bits;

/* Port Output Set Register (GPIO_POSR3H) */
typedef struct {
  __REG32 POS224            : 1;
  __REG32 POS225            : 1;
  __REG32 POS226            : 1;
  __REG32 POS227            : 1;
  __REG32 POS228            : 1;
  __REG32 POS229            : 1;
  __REG32 POS230            : 1;
  __REG32 POS231            : 1;
  __REG32 POS232            : 1;
  __REG32 POS233            : 1;
  __REG32 POS234            : 1;
  __REG32 POS235            : 1;
  __REG32 POS236            : 1;
  __REG32 POS237            : 1;
  __REG32 POS238            : 1;
  __REG32 POS239            : 1;
  __REG32 POS240            : 1;
  __REG32 POS241            : 1;
  __REG32 POS242            : 1;
  __REG32 POS243            : 1;
  __REG32 POS244            : 1;
  __REG32 POS245            : 1;
  __REG32 POS246            : 1;
  __REG32 POS247            : 1;
  __REG32 POS248            : 1;
  __REG32 POS249            : 1;
  __REG32 POS250            : 1;
  __REG32 POS251            : 1;
  __REG32 POS252            : 1;
  __REG32 POS253            : 1;
  __REG32 POS254            : 1;
  __REG32 POS255            : 1;
} __gpio_posr3h_bits;

/* Port Output Set Register (GPIO_POSR4L) */
typedef struct {
  __REG32 POS256            : 1;
  __REG32 POS257            : 1;
  __REG32 POS258            : 1;
  __REG32 POS259            : 1;
  __REG32 POS260            : 1;
  __REG32 POS261            : 1;
  __REG32 POS262            : 1;
  __REG32 POS263            : 1;
  __REG32 POS264            : 1;
  __REG32 POS265            : 1;
  __REG32 POS266            : 1;
  __REG32 POS267            : 1;
  __REG32 POS268            : 1;
  __REG32 POS269            : 1;
  __REG32 POS270            : 1;
  __REG32 POS271            : 1;
  __REG32 POS272            : 1;
  __REG32 POS273            : 1;
  __REG32 POS274            : 1;
  __REG32 POS275            : 1;
  __REG32 POS276            : 1;
  __REG32 POS277            : 1;
  __REG32 POS278            : 1;
  __REG32 POS279            : 1;
  __REG32 POS280            : 1;
  __REG32 POS281            : 1;
  __REG32 POS282            : 1;
  __REG32 POS283            : 1;
  __REG32 POS284            : 1;
  __REG32 POS285            : 1;
  __REG32 POS286            : 1;
  __REG32 POS287            : 1;
} __gpio_posr4l_bits;

/* Port Output Set Register (GPIO_POSR4H) */
typedef struct {
  __REG32 POS288            : 1;
  __REG32 POS289            : 1;
  __REG32 POS290            : 1;
  __REG32 POS291            : 1;
  __REG32 POS292            : 1;
  __REG32 POS293            : 1;
  __REG32 POS294            : 1;
  __REG32 POS295            : 1;
  __REG32 POS296            : 1;
  __REG32 POS297            : 1;
  __REG32 POS298            : 1;
  __REG32 POS299            : 1;
  __REG32 POS300            : 1;
  __REG32 POS301            : 1;
  __REG32 POS302            : 1;
  __REG32 POS303            : 1;
  __REG32 POS304            : 1;
  __REG32 POS305            : 1;
  __REG32 POS306            : 1;
  __REG32 POS307            : 1;
  __REG32 POS308            : 1;
  __REG32 POS309            : 1;
  __REG32 POS310            : 1;
  __REG32 POS311            : 1;
  __REG32 POS312            : 1;
  __REG32 POS313            : 1;
  __REG32 POS314            : 1;
  __REG32 POS315            : 1;
  __REG32 POS316            : 1;
  __REG32 POS317            : 1;
  __REG32 POS318            : 1;
  __REG32 POS319            : 1;
} __gpio_posr4h_bits;

/* Port Output Set Register (GPIO_POSR5L) */
typedef struct {
  __REG32 POS320            : 1;
  __REG32 POS321            : 1;
  __REG32 POS322            : 1;
  __REG32 POS323            : 1;
  __REG32 POS324            : 1;
  __REG32 POS325            : 1;
  __REG32 POS326            : 1;
  __REG32 POS327            : 1;
  __REG32 POS328            : 1;
  __REG32 POS329            : 1;
  __REG32 POS330            : 1;
  __REG32 POS331            : 1;
  __REG32 POS332            : 1;
  __REG32 POS333            : 1;
  __REG32 POS334            : 1;
  __REG32 POS335            : 1;
  __REG32 POS336            : 1;
  __REG32 POS337            : 1;
  __REG32 POS338            : 1;
  __REG32 POS339            : 1;
  __REG32 POS340            : 1;
  __REG32 POS341            : 1;
  __REG32 POS342            : 1;
  __REG32 POS343            : 1;
  __REG32 POS344            : 1;
  __REG32 POS345            : 1;
  __REG32 POS346            : 1;
  __REG32 POS347            : 1;
  __REG32 POS348            : 1;
  __REG32 POS349            : 1;
  __REG32 POS350            : 1;
  __REG32 POS351            : 1;
} __gpio_posr5l_bits;

/* Port Output Set Register (GPIO_POSR5H) */
typedef struct {
  __REG32 POS352            : 1;
  __REG32 POS353            : 1;
  __REG32 POS354            : 1;
  __REG32 POS355            : 1;
  __REG32 POS356            : 1;
  __REG32 POS357            : 1;
  __REG32 POS358            : 1;
  __REG32 POS359            : 1;
  __REG32 POS360            : 1;
  __REG32 POS361            : 1;
  __REG32 POS362            : 1;
  __REG32 POS363            : 1;
  __REG32 POS364            : 1;
  __REG32 POS365            : 1;
  __REG32 POS366            : 1;
  __REG32 POS367            : 1;
  __REG32 POS368            : 1;
  __REG32 POS369            : 1;
  __REG32 POS370            : 1;
  __REG32 POS371            : 1;
  __REG32 POS372            : 1;
  __REG32 POS373            : 1;
  __REG32 POS374            : 1;
  __REG32 POS375            : 1;
  __REG32 POS376            : 1;
  __REG32 POS377            : 1;
  __REG32 POS378            : 1;
  __REG32 POS379            : 1;
  __REG32 POS380            : 1;
  __REG32 POS381            : 1;
  __REG32 POS382            : 1;
  __REG32 POS383            : 1;
} __gpio_posr5h_bits;

/* Port Output Set Register (GPIO_POSR6L) */
typedef struct {
  __REG32 POS384            : 1;
  __REG32 POS385            : 1;
  __REG32 POS386            : 1;
  __REG32 POS387            : 1;
  __REG32 POS388            : 1;
  __REG32 POS389            : 1;
  __REG32 POS390            : 1;
  __REG32 POS391            : 1;
  __REG32 POS392            : 1;
  __REG32 POS393            : 1;
  __REG32 POS394            : 1;
  __REG32 POS395            : 1;
  __REG32 POS396            : 1;
  __REG32 POS397            : 1;
  __REG32 POS398            : 1;
  __REG32 POS399            : 1;
  __REG32 POS400            : 1;
  __REG32 POS401            : 1;
  __REG32 POS402            : 1;
  __REG32 POS403            : 1;
  __REG32 POS404            : 1;
  __REG32 POS405            : 1;
  __REG32 POS406            : 1;
  __REG32 POS407            : 1;
  __REG32 POS408            : 1;
  __REG32 POS409            : 1;
  __REG32 POS410            : 1;
  __REG32 POS411            : 1;
  __REG32 POS412            : 1;
  __REG32 POS413            : 1;
  __REG32 POS414            : 1;
  __REG32 POS415            : 1;
} __gpio_posr6l_bits;

/* Port Output Set Register (GPIO_POSR6H) */
typedef struct {
  __REG32 POS416            : 1;
  __REG32 POS417            : 1;
  __REG32 POS418            : 1;
  __REG32 POS419            : 1;
  __REG32 POS420            : 1;
  __REG32 POS421            : 1;
  __REG32 POS422            : 1;
  __REG32 POS423            : 1;
  __REG32 POS424            : 1;
  __REG32 POS425            : 1;
  __REG32 POS426            : 1;
  __REG32 POS427            : 1;
  __REG32 POS428            : 1;
  __REG32 POS429            : 1;
  __REG32 POS430            : 1;
  __REG32 POS431            : 1;
  __REG32 POS432            : 1;
  __REG32 POS433            : 1;
  __REG32 POS434            : 1;
  __REG32 POS435            : 1;
  __REG32 POS436            : 1;
  __REG32 POS437            : 1;
  __REG32 POS438            : 1;
  __REG32 POS439            : 1;
  __REG32 POS440            : 1;
  __REG32 POS441            : 1;
  __REG32 POS442            : 1;
  __REG32 POS443            : 1;
  __REG32 POS444            : 1;
  __REG32 POS445            : 1;
  __REG32 POS446            : 1;
  __REG32 POS447            : 1;
} __gpio_posr6h_bits;

/* Port Output Set Register (GPIO_POSR7L) */
typedef struct {
  __REG32 POS448            : 1;
  __REG32 POS449            : 1;
  __REG32 POS450            : 1;
  __REG32 POS451            : 1;
  __REG32 POS452            : 1;
  __REG32 POS453            : 1;
  __REG32 POS454            : 1;
  __REG32 POS455            : 1;
  __REG32 POS456            : 1;
  __REG32 POS457            : 1;
  __REG32 POS458            : 1;
  __REG32 POS459            : 1;
  __REG32 POS460            : 1;
  __REG32 POS461            : 1;
  __REG32 POS462            : 1;
  __REG32 POS463            : 1;
  __REG32 POS464            : 1;
  __REG32 POS465            : 1;
  __REG32 POS466            : 1;
  __REG32 POS467            : 1;
  __REG32 POS468            : 1;
  __REG32 POS469            : 1;
  __REG32 POS470            : 1;
  __REG32 POS471            : 1;
  __REG32 POS472            : 1;
  __REG32 POS473            : 1;
  __REG32 POS474            : 1;
  __REG32 POS475            : 1;
  __REG32 POS476            : 1;
  __REG32 POS477            : 1;
  __REG32 POS478            : 1;
  __REG32 POS479            : 1;
} __gpio_posr7l_bits;

/* Port Output Set Register (GPIO_POSR7H) */
typedef struct {
  __REG32 POS480            : 1;
  __REG32 POS481            : 1;
  __REG32 POS482            : 1;
  __REG32 POS483            : 1;
  __REG32 POS484            : 1;
  __REG32 POS485            : 1;
  __REG32 POS486            : 1;
  __REG32 POS487            : 1;
  __REG32 POS488            : 1;
  __REG32 POS489            : 1;
  __REG32 POS490            : 1;
  __REG32 POS491            : 1;
  __REG32 POS492            : 1;
  __REG32 POS493            : 1;
  __REG32 POS494            : 1;
  __REG32 POS495            : 1;
  __REG32 POS496            : 1;
  __REG32 POS497            : 1;
  __REG32 POS498            : 1;
  __REG32 POS499            : 1;
  __REG32 POS500            : 1;
  __REG32 POS501            : 1;
  __REG32 POS502            : 1;
  __REG32 POS503            : 1;
  __REG32 POS504            : 1;
  __REG32 POS505            : 1;
  __REG32 POS506            : 1;
  __REG32 POS507            : 1;
  __REG32 POS508            : 1;
  __REG32 POS509            : 1;
  __REG32 POS510            : 1;
  __REG32 POS511            : 1;
} __gpio_posr7h_bits;

/* Port Output Clear Register (GPIO_POCR0L) */
typedef struct {
  __REG32 POC0              : 1;
  __REG32 POC1              : 1;
  __REG32 POC2              : 1;
  __REG32 POC3              : 1;
  __REG32 POC4              : 1;
  __REG32 POC5              : 1;
  __REG32 POC6              : 1;
  __REG32 POC7              : 1;
  __REG32 POC8              : 1;
  __REG32 POC9              : 1;
  __REG32 POC10             : 1;
  __REG32 POC11             : 1;
  __REG32 POC12             : 1;
  __REG32 POC13             : 1;
  __REG32 POC14             : 1;
  __REG32 POC15             : 1;
  __REG32 POC16             : 1;
  __REG32 POC17             : 1;
  __REG32 POC18             : 1;
  __REG32 POC19             : 1;
  __REG32 POC20             : 1;
  __REG32 POC21             : 1;
  __REG32 POC22             : 1;
  __REG32 POC23             : 1;
  __REG32 POC24             : 1;
  __REG32 POC25             : 1;
  __REG32 POC26             : 1;
  __REG32 POC27             : 1;
  __REG32 POC28             : 1;
  __REG32 POC29             : 1;
  __REG32 POC30             : 1;
  __REG32 POC31             : 1;
} __gpio_pocr0l_bits;

/* Port Output Clear Register (GPIO_POCR0H) */
typedef struct {
  __REG32 POC32             : 1;
  __REG32 POC33             : 1;
  __REG32 POC34             : 1;
  __REG32 POC35             : 1;
  __REG32 POC36             : 1;
  __REG32 POC37             : 1;
  __REG32 POC38             : 1;
  __REG32 POC39             : 1;
  __REG32 POC40             : 1;
  __REG32 POC41             : 1;
  __REG32 POC42             : 1;
  __REG32 POC43             : 1;
  __REG32 POC44             : 1;
  __REG32 POC45             : 1;
  __REG32 POC46             : 1;
  __REG32 POC47             : 1;
  __REG32 POC48             : 1;
  __REG32 POC49             : 1;
  __REG32 POC50             : 1;
  __REG32 POC51             : 1;
  __REG32 POC52             : 1;
  __REG32 POC53             : 1;
  __REG32 POC54             : 1;
  __REG32 POC55             : 1;
  __REG32 POC56             : 1;
  __REG32 POC57             : 1;
  __REG32 POC58             : 1;
  __REG32 POC59             : 1;
  __REG32 POC60             : 1;
  __REG32 POC61             : 1;
  __REG32 POC62             : 1;
  __REG32 POC63             : 1;
} __gpio_pocr0h_bits;

/* Port Output Clear Register (GPIO_POCR1L) */
typedef struct {
  __REG32 POC64             : 1;
  __REG32 POC65             : 1;
  __REG32 POC66             : 1;
  __REG32 POC67             : 1;
  __REG32 POC68             : 1;
  __REG32 POC69             : 1;
  __REG32 POC70             : 1;
  __REG32 POC71             : 1;
  __REG32 POC72             : 1;
  __REG32 POC73             : 1;
  __REG32 POC74             : 1;
  __REG32 POC75             : 1;
  __REG32 POC76             : 1;
  __REG32 POC77             : 1;
  __REG32 POC78             : 1;
  __REG32 POC79             : 1;
  __REG32 POC80             : 1;
  __REG32 POC81             : 1;
  __REG32 POC82             : 1;
  __REG32 POC83             : 1;
  __REG32 POC84             : 1;
  __REG32 POC85             : 1;
  __REG32 POC86             : 1;
  __REG32 POC87             : 1;
  __REG32 POC88             : 1;
  __REG32 POC89             : 1;
  __REG32 POC90             : 1;
  __REG32 POC91             : 1;
  __REG32 POC92             : 1;
  __REG32 POC93             : 1;
  __REG32 POC94             : 1;
  __REG32 POC95             : 1;
} __gpio_pocr1l_bits;

/* Port Output Clear Register (GPIO_POCR1H) */
typedef struct {
  __REG32 POC96             : 1;
  __REG32 POC97             : 1;
  __REG32 POC98             : 1;
  __REG32 POC99             : 1;
  __REG32 POC100            : 1;
  __REG32 POC101            : 1;
  __REG32 POC102            : 1;
  __REG32 POC103            : 1;
  __REG32 POC104            : 1;
  __REG32 POC105            : 1;
  __REG32 POC106            : 1;
  __REG32 POC107            : 1;
  __REG32 POC108            : 1;
  __REG32 POC109            : 1;
  __REG32 POC110            : 1;
  __REG32 POC111            : 1;
  __REG32 POC112            : 1;
  __REG32 POC113            : 1;
  __REG32 POC114            : 1;
  __REG32 POC115            : 1;
  __REG32 POC116            : 1;
  __REG32 POC117            : 1;
  __REG32 POC118            : 1;
  __REG32 POC119            : 1;
  __REG32 POC120            : 1;
  __REG32 POC121            : 1;
  __REG32 POC122            : 1;
  __REG32 POC123            : 1;
  __REG32 POC124            : 1;
  __REG32 POC125            : 1;
  __REG32 POC126            : 1;
  __REG32 POC127            : 1;
} __gpio_pocr1h_bits;

/* Port Output Clear Register (GPIO_POCR2L) */
typedef struct {
  __REG32 POC128            : 1;
  __REG32 POC129            : 1;
  __REG32 POC130            : 1;
  __REG32 POC131            : 1;
  __REG32 POC132            : 1;
  __REG32 POC133            : 1;
  __REG32 POC134            : 1;
  __REG32 POC135            : 1;
  __REG32 POC136            : 1;
  __REG32 POC137            : 1;
  __REG32 POC138            : 1;
  __REG32 POC139            : 1;
  __REG32 POC140            : 1;
  __REG32 POC141            : 1;
  __REG32 POC142            : 1;
  __REG32 POC143            : 1;
  __REG32 POC144            : 1;
  __REG32 POC145            : 1;
  __REG32 POC146            : 1;
  __REG32 POC147            : 1;
  __REG32 POC148            : 1;
  __REG32 POC149            : 1;
  __REG32 POC150            : 1;
  __REG32 POC151            : 1;
  __REG32 POC152            : 1;
  __REG32 POC153            : 1;
  __REG32 POC154            : 1;
  __REG32 POC155            : 1;
  __REG32 POC156            : 1;
  __REG32 POC157            : 1;
  __REG32 POC158            : 1;
  __REG32 POC159            : 1;
} __gpio_pocr2l_bits;

/* Port Output Clear Register (GPIO_POCR2H) */
typedef struct {
  __REG32 POC160            : 1;
  __REG32 POC161            : 1;
  __REG32 POC162            : 1;
  __REG32 POC163            : 1;
  __REG32 POC164            : 1;
  __REG32 POC165            : 1;
  __REG32 POC166            : 1;
  __REG32 POC167            : 1;
  __REG32 POC168            : 1;
  __REG32 POC169            : 1;
  __REG32 POC170            : 1;
  __REG32 POC171            : 1;
  __REG32 POC172            : 1;
  __REG32 POC173            : 1;
  __REG32 POC174            : 1;
  __REG32 POC175            : 1;
  __REG32 POC176            : 1;
  __REG32 POC177            : 1;
  __REG32 POC178            : 1;
  __REG32 POC179            : 1;
  __REG32 POC180            : 1;
  __REG32 POC181            : 1;
  __REG32 POC182            : 1;
  __REG32 POC183            : 1;
  __REG32 POC184            : 1;
  __REG32 POC185            : 1;
  __REG32 POC186            : 1;
  __REG32 POC187            : 1;
  __REG32 POC188            : 1;
  __REG32 POC189            : 1;
  __REG32 POC190            : 1;
  __REG32 POC191            : 1;
} __gpio_pocr2h_bits;

/* Port Output Clear Register (GPIO_POCR3L) */
typedef struct {
  __REG32 POC192            : 1;
  __REG32 POC193            : 1;
  __REG32 POC194            : 1;
  __REG32 POC195            : 1;
  __REG32 POC196            : 1;
  __REG32 POC197            : 1;
  __REG32 POC198            : 1;
  __REG32 POC199            : 1;
  __REG32 POC200            : 1;
  __REG32 POC201            : 1;
  __REG32 POC202            : 1;
  __REG32 POC203            : 1;
  __REG32 POC204            : 1;
  __REG32 POC205            : 1;
  __REG32 POC206            : 1;
  __REG32 POC207            : 1;
  __REG32 POC208            : 1;
  __REG32 POC209            : 1;
  __REG32 POC210            : 1;
  __REG32 POC211            : 1;
  __REG32 POC212            : 1;
  __REG32 POC213            : 1;
  __REG32 POC214            : 1;
  __REG32 POC215            : 1;
  __REG32 POC216            : 1;
  __REG32 POC217            : 1;
  __REG32 POC218            : 1;
  __REG32 POC219            : 1;
  __REG32 POC220            : 1;
  __REG32 POC221            : 1;
  __REG32 POC222            : 1;
  __REG32 POC223            : 1;
} __gpio_pocr3l_bits;

/* Port Output Clear Register (GPIO_POCR3H) */
typedef struct {
  __REG32 POC224            : 1;
  __REG32 POC225            : 1;
  __REG32 POC226            : 1;
  __REG32 POC227            : 1;
  __REG32 POC228            : 1;
  __REG32 POC229            : 1;
  __REG32 POC230            : 1;
  __REG32 POC231            : 1;
  __REG32 POC232            : 1;
  __REG32 POC233            : 1;
  __REG32 POC234            : 1;
  __REG32 POC235            : 1;
  __REG32 POC236            : 1;
  __REG32 POC237            : 1;
  __REG32 POC238            : 1;
  __REG32 POC239            : 1;
  __REG32 POC240            : 1;
  __REG32 POC241            : 1;
  __REG32 POC242            : 1;
  __REG32 POC243            : 1;
  __REG32 POC244            : 1;
  __REG32 POC245            : 1;
  __REG32 POC246            : 1;
  __REG32 POC247            : 1;
  __REG32 POC248            : 1;
  __REG32 POC249            : 1;
  __REG32 POC250            : 1;
  __REG32 POC251            : 1;
  __REG32 POC252            : 1;
  __REG32 POC253            : 1;
  __REG32 POC254            : 1;
  __REG32 POC255            : 1;
} __gpio_pocr3h_bits;

/* Port Output Clear Register (GPIO_POCR4L) */
typedef struct {
  __REG32 POC256            : 1;
  __REG32 POC257            : 1;
  __REG32 POC258            : 1;
  __REG32 POC259            : 1;
  __REG32 POC260            : 1;
  __REG32 POC261            : 1;
  __REG32 POC262            : 1;
  __REG32 POC263            : 1;
  __REG32 POC264            : 1;
  __REG32 POC265            : 1;
  __REG32 POC266            : 1;
  __REG32 POC267            : 1;
  __REG32 POC268            : 1;
  __REG32 POC269            : 1;
  __REG32 POC270            : 1;
  __REG32 POC271            : 1;
  __REG32 POC272            : 1;
  __REG32 POC273            : 1;
  __REG32 POC274            : 1;
  __REG32 POC275            : 1;
  __REG32 POC276            : 1;
  __REG32 POC277            : 1;
  __REG32 POC278            : 1;
  __REG32 POC279            : 1;
  __REG32 POC280            : 1;
  __REG32 POC281            : 1;
  __REG32 POC282            : 1;
  __REG32 POC283            : 1;
  __REG32 POC284            : 1;
  __REG32 POC285            : 1;
  __REG32 POC286            : 1;
  __REG32 POC287            : 1;
} __gpio_pocr4l_bits;

/* Port Output Clear Register (GPIO_POCR4H) */
typedef struct {
  __REG32 POC288            : 1;
  __REG32 POC289            : 1;
  __REG32 POC290            : 1;
  __REG32 POC291            : 1;
  __REG32 POC292            : 1;
  __REG32 POC293            : 1;
  __REG32 POC294            : 1;
  __REG32 POC295            : 1;
  __REG32 POC296            : 1;
  __REG32 POC297            : 1;
  __REG32 POC298            : 1;
  __REG32 POC299            : 1;
  __REG32 POC300            : 1;
  __REG32 POC301            : 1;
  __REG32 POC302            : 1;
  __REG32 POC303            : 1;
  __REG32 POC304            : 1;
  __REG32 POC305            : 1;
  __REG32 POC306            : 1;
  __REG32 POC307            : 1;
  __REG32 POC308            : 1;
  __REG32 POC309            : 1;
  __REG32 POC310            : 1;
  __REG32 POC311            : 1;
  __REG32 POC312            : 1;
  __REG32 POC313            : 1;
  __REG32 POC314            : 1;
  __REG32 POC315            : 1;
  __REG32 POC316            : 1;
  __REG32 POC317            : 1;
  __REG32 POC318            : 1;
  __REG32 POC319            : 1;
} __gpio_pocr4h_bits;

/* Port Output Clear Register (GPIO_POCR5L) */
typedef struct {
  __REG32 POC320            : 1;
  __REG32 POC321            : 1;
  __REG32 POC322            : 1;
  __REG32 POC323            : 1;
  __REG32 POC324            : 1;
  __REG32 POC325            : 1;
  __REG32 POC326            : 1;
  __REG32 POC327            : 1;
  __REG32 POC328            : 1;
  __REG32 POC329            : 1;
  __REG32 POC330            : 1;
  __REG32 POC331            : 1;
  __REG32 POC332            : 1;
  __REG32 POC333            : 1;
  __REG32 POC334            : 1;
  __REG32 POC335            : 1;
  __REG32 POC336            : 1;
  __REG32 POC337            : 1;
  __REG32 POC338            : 1;
  __REG32 POC339            : 1;
  __REG32 POC340            : 1;
  __REG32 POC341            : 1;
  __REG32 POC342            : 1;
  __REG32 POC343            : 1;
  __REG32 POC344            : 1;
  __REG32 POC345            : 1;
  __REG32 POC346            : 1;
  __REG32 POC347            : 1;
  __REG32 POC348            : 1;
  __REG32 POC349            : 1;
  __REG32 POC350            : 1;
  __REG32 POC351            : 1;
} __gpio_pocr5l_bits;

/* Port Output Clear Register (GPIO_POCR5H) */
typedef struct {
  __REG32 POC352            : 1;
  __REG32 POC353            : 1;
  __REG32 POC354            : 1;
  __REG32 POC355            : 1;
  __REG32 POC356            : 1;
  __REG32 POC357            : 1;
  __REG32 POC358            : 1;
  __REG32 POC359            : 1;
  __REG32 POC360            : 1;
  __REG32 POC361            : 1;
  __REG32 POC362            : 1;
  __REG32 POC363            : 1;
  __REG32 POC364            : 1;
  __REG32 POC365            : 1;
  __REG32 POC366            : 1;
  __REG32 POC367            : 1;
  __REG32 POC368            : 1;
  __REG32 POC369            : 1;
  __REG32 POC370            : 1;
  __REG32 POC371            : 1;
  __REG32 POC372            : 1;
  __REG32 POC373            : 1;
  __REG32 POC374            : 1;
  __REG32 POC375            : 1;
  __REG32 POC376            : 1;
  __REG32 POC377            : 1;
  __REG32 POC378            : 1;
  __REG32 POC379            : 1;
  __REG32 POC380            : 1;
  __REG32 POC381            : 1;
  __REG32 POC382            : 1;
  __REG32 POC383            : 1;
} __gpio_pocr5h_bits;

/* Port Output Clear Register (GPIO_POCR6L) */
typedef struct {
  __REG32 POC384            : 1;
  __REG32 POC385            : 1;
  __REG32 POC386            : 1;
  __REG32 POC387            : 1;
  __REG32 POC388            : 1;
  __REG32 POC389            : 1;
  __REG32 POC390            : 1;
  __REG32 POC391            : 1;
  __REG32 POC392            : 1;
  __REG32 POC393            : 1;
  __REG32 POC394            : 1;
  __REG32 POC395            : 1;
  __REG32 POC396            : 1;
  __REG32 POC397            : 1;
  __REG32 POC398            : 1;
  __REG32 POC399            : 1;
  __REG32 POC400            : 1;
  __REG32 POC401            : 1;
  __REG32 POC402            : 1;
  __REG32 POC403            : 1;
  __REG32 POC404            : 1;
  __REG32 POC405            : 1;
  __REG32 POC406            : 1;
  __REG32 POC407            : 1;
  __REG32 POC408            : 1;
  __REG32 POC409            : 1;
  __REG32 POC410            : 1;
  __REG32 POC411            : 1;
  __REG32 POC412            : 1;
  __REG32 POC413            : 1;
  __REG32 POC414            : 1;
  __REG32 POC415            : 1;
} __gpio_pocr6l_bits;

/* Port Output Clear Register (GPIO_POCR6H) */
typedef struct {
  __REG32 POC416            : 1;
  __REG32 POC417            : 1;
  __REG32 POC418            : 1;
  __REG32 POC419            : 1;
  __REG32 POC420            : 1;
  __REG32 POC421            : 1;
  __REG32 POC422            : 1;
  __REG32 POC423            : 1;
  __REG32 POC424            : 1;
  __REG32 POC425            : 1;
  __REG32 POC426            : 1;
  __REG32 POC427            : 1;
  __REG32 POC428            : 1;
  __REG32 POC429            : 1;
  __REG32 POC430            : 1;
  __REG32 POC431            : 1;
  __REG32 POC432            : 1;
  __REG32 POC433            : 1;
  __REG32 POC434            : 1;
  __REG32 POC435            : 1;
  __REG32 POC436            : 1;
  __REG32 POC437            : 1;
  __REG32 POC438            : 1;
  __REG32 POC439            : 1;
  __REG32 POC440            : 1;
  __REG32 POC441            : 1;
  __REG32 POC442            : 1;
  __REG32 POC443            : 1;
  __REG32 POC444            : 1;
  __REG32 POC445            : 1;
  __REG32 POC446            : 1;
  __REG32 POC447            : 1;
} __gpio_pocr6h_bits;

/* Port Output Clear Register (GPIO_POCR7L) */
typedef struct {
  __REG32 POC448            : 1;
  __REG32 POC449            : 1;
  __REG32 POC450            : 1;
  __REG32 POC451            : 1;
  __REG32 POC452            : 1;
  __REG32 POC453            : 1;
  __REG32 POC454            : 1;
  __REG32 POC455            : 1;
  __REG32 POC456            : 1;
  __REG32 POC457            : 1;
  __REG32 POC458            : 1;
  __REG32 POC459            : 1;
  __REG32 POC460            : 1;
  __REG32 POC461            : 1;
  __REG32 POC462            : 1;
  __REG32 POC463            : 1;
  __REG32 POC464            : 1;
  __REG32 POC465            : 1;
  __REG32 POC466            : 1;
  __REG32 POC467            : 1;
  __REG32 POC468            : 1;
  __REG32 POC469            : 1;
  __REG32 POC470            : 1;
  __REG32 POC471            : 1;
  __REG32 POC472            : 1;
  __REG32 POC473            : 1;
  __REG32 POC474            : 1;
  __REG32 POC475            : 1;
  __REG32 POC476            : 1;
  __REG32 POC477            : 1;
  __REG32 POC478            : 1;
  __REG32 POC479            : 1;
} __gpio_pocr7l_bits;

/* Port Output Clear Register (GPIO_POCR7H) */
typedef struct {
  __REG32 POC480            : 1;
  __REG32 POC481            : 1;
  __REG32 POC482            : 1;
  __REG32 POC483            : 1;
  __REG32 POC484            : 1;
  __REG32 POC485            : 1;
  __REG32 POC486            : 1;
  __REG32 POC487            : 1;
  __REG32 POC488            : 1;
  __REG32 POC489            : 1;
  __REG32 POC490            : 1;
  __REG32 POC491            : 1;
  __REG32 POC492            : 1;
  __REG32 POC493            : 1;
  __REG32 POC494            : 1;
  __REG32 POC495            : 1;
  __REG32 POC496            : 1;
  __REG32 POC497            : 1;
  __REG32 POC498            : 1;
  __REG32 POC499            : 1;
  __REG32 POC500            : 1;
  __REG32 POC501            : 1;
  __REG32 POC502            : 1;
  __REG32 POC503            : 1;
  __REG32 POC504            : 1;
  __REG32 POC505            : 1;
  __REG32 POC506            : 1;
  __REG32 POC507            : 1;
  __REG32 POC508            : 1;
  __REG32 POC509            : 1;
  __REG32 POC510            : 1;
  __REG32 POC511            : 1;
} __gpio_pocr7h_bits;

/* Port Input Data Register (GPIO_PIDR0L) */
typedef struct {
  __REG32 PID0              : 1;
  __REG32 PID1              : 1;
  __REG32 PID2              : 1;
  __REG32 PID3              : 1;
  __REG32 PID4              : 1;
  __REG32 PID5              : 1;
  __REG32 PID6              : 1;
  __REG32 PID7              : 1;
  __REG32 PID8              : 1;
  __REG32 PID9              : 1;
  __REG32 PID10             : 1;
  __REG32 PID11             : 1;
  __REG32 PID12             : 1;
  __REG32 PID13             : 1;
  __REG32 PID14             : 1;
  __REG32 PID15             : 1;
  __REG32 PID16             : 1;
  __REG32 PID17             : 1;
  __REG32 PID18             : 1;
  __REG32 PID19             : 1;
  __REG32 PID20             : 1;
  __REG32 PID21             : 1;
  __REG32 PID22             : 1;
  __REG32 PID23             : 1;
  __REG32 PID24             : 1;
  __REG32 PID25             : 1;
  __REG32 PID26             : 1;
  __REG32 PID27             : 1;
  __REG32 PID28             : 1;
  __REG32 PID29             : 1;
  __REG32 PID30             : 1;
  __REG32 PID31             : 1;
} __gpio_pidr0l_bits;

/* Port Input Data Register (GPIO_PIDR0H) */
typedef struct {
  __REG32 PID32             : 1;
  __REG32 PID33             : 1;
  __REG32 PID34             : 1;
  __REG32 PID35             : 1;
  __REG32 PID36             : 1;
  __REG32 PID37             : 1;
  __REG32 PID38             : 1;
  __REG32 PID39             : 1;
  __REG32 PID40             : 1;
  __REG32 PID41             : 1;
  __REG32 PID42             : 1;
  __REG32 PID43             : 1;
  __REG32 PID44             : 1;
  __REG32 PID45             : 1;
  __REG32 PID46             : 1;
  __REG32 PID47             : 1;
  __REG32 PID48             : 1;
  __REG32 PID49             : 1;
  __REG32 PID50             : 1;
  __REG32 PID51             : 1;
  __REG32 PID52             : 1;
  __REG32 PID53             : 1;
  __REG32 PID54             : 1;
  __REG32 PID55             : 1;
  __REG32 PID56             : 1;
  __REG32 PID57             : 1;
  __REG32 PID58             : 1;
  __REG32 PID59             : 1;
  __REG32 PID60             : 1;
  __REG32 PID61             : 1;
  __REG32 PID62             : 1;
  __REG32 PID63             : 1;
} __gpio_pidr0h_bits;

/* Port Input Data Register (GPIO_PIDR1L) */
typedef struct {
  __REG32 PID64             : 1;
  __REG32 PID65             : 1;
  __REG32 PID66             : 1;
  __REG32 PID67             : 1;
  __REG32 PID68             : 1;
  __REG32 PID69             : 1;
  __REG32 PID70             : 1;
  __REG32 PID71             : 1;
  __REG32 PID72             : 1;
  __REG32 PID73             : 1;
  __REG32 PID74             : 1;
  __REG32 PID75             : 1;
  __REG32 PID76             : 1;
  __REG32 PID77             : 1;
  __REG32 PID78             : 1;
  __REG32 PID79             : 1;
  __REG32 PID80             : 1;
  __REG32 PID81             : 1;
  __REG32 PID82             : 1;
  __REG32 PID83             : 1;
  __REG32 PID84             : 1;
  __REG32 PID85             : 1;
  __REG32 PID86             : 1;
  __REG32 PID87             : 1;
  __REG32 PID88             : 1;
  __REG32 PID89             : 1;
  __REG32 PID90             : 1;
  __REG32 PID91             : 1;
  __REG32 PID92             : 1;
  __REG32 PID93             : 1;
  __REG32 PID94             : 1;
  __REG32 PID95             : 1;
} __gpio_pidr1l_bits;

/* Port Input Data Register (GPIO_PIDR1H) */
typedef struct {
  __REG32 PID96             : 1;
  __REG32 PID97             : 1;
  __REG32 PID98             : 1;
  __REG32 PID99             : 1;
  __REG32 PID100            : 1;
  __REG32 PID101            : 1;
  __REG32 PID102            : 1;
  __REG32 PID103            : 1;
  __REG32 PID104            : 1;
  __REG32 PID105            : 1;
  __REG32 PID106            : 1;
  __REG32 PID107            : 1;
  __REG32 PID108            : 1;
  __REG32 PID109            : 1;
  __REG32 PID110            : 1;
  __REG32 PID111            : 1;
  __REG32 PID112            : 1;
  __REG32 PID113            : 1;
  __REG32 PID114            : 1;
  __REG32 PID115            : 1;
  __REG32 PID116            : 1;
  __REG32 PID117            : 1;
  __REG32 PID118            : 1;
  __REG32 PID119            : 1;
  __REG32 PID120            : 1;
  __REG32 PID121            : 1;
  __REG32 PID122            : 1;
  __REG32 PID123            : 1;
  __REG32 PID124            : 1;
  __REG32 PID125            : 1;
  __REG32 PID126            : 1;
  __REG32 PID127            : 1;
} __gpio_pidr1h_bits;

/* Port Input Data Register (GPIO_PIDR2L) */
typedef struct {
  __REG32 PID128            : 1;
  __REG32 PID129            : 1;
  __REG32 PID130            : 1;
  __REG32 PID131            : 1;
  __REG32 PID132            : 1;
  __REG32 PID133            : 1;
  __REG32 PID134            : 1;
  __REG32 PID135            : 1;
  __REG32 PID136            : 1;
  __REG32 PID137            : 1;
  __REG32 PID138            : 1;
  __REG32 PID139            : 1;
  __REG32 PID140            : 1;
  __REG32 PID141            : 1;
  __REG32 PID142            : 1;
  __REG32 PID143            : 1;
  __REG32 PID144            : 1;
  __REG32 PID145            : 1;
  __REG32 PID146            : 1;
  __REG32 PID147            : 1;
  __REG32 PID148            : 1;
  __REG32 PID149            : 1;
  __REG32 PID150            : 1;
  __REG32 PID151            : 1;
  __REG32 PID152            : 1;
  __REG32 PID153            : 1;
  __REG32 PID154            : 1;
  __REG32 PID155            : 1;
  __REG32 PID156            : 1;
  __REG32 PID157            : 1;
  __REG32 PID158            : 1;
  __REG32 PID159            : 1;
} __gpio_pidr2l_bits;

/* Port Input Data Register (GPIO_PIDR2H) */
typedef struct {
  __REG32 PID160            : 1;
  __REG32 PID161            : 1;
  __REG32 PID162            : 1;
  __REG32 PID163            : 1;
  __REG32 PID164            : 1;
  __REG32 PID165            : 1;
  __REG32 PID166            : 1;
  __REG32 PID167            : 1;
  __REG32 PID168            : 1;
  __REG32 PID169            : 1;
  __REG32 PID170            : 1;
  __REG32 PID171            : 1;
  __REG32 PID172            : 1;
  __REG32 PID173            : 1;
  __REG32 PID174            : 1;
  __REG32 PID175            : 1;
  __REG32 PID176            : 1;
  __REG32 PID177            : 1;
  __REG32 PID178            : 1;
  __REG32 PID179            : 1;
  __REG32 PID180            : 1;
  __REG32 PID181            : 1;
  __REG32 PID182            : 1;
  __REG32 PID183            : 1;
  __REG32 PID184            : 1;
  __REG32 PID185            : 1;
  __REG32 PID186            : 1;
  __REG32 PID187            : 1;
  __REG32 PID188            : 1;
  __REG32 PID189            : 1;
  __REG32 PID190            : 1;
  __REG32 PID191            : 1;
} __gpio_pidr2h_bits;

/* Port Input Data Register (GPIO_PIDR3L) */
typedef struct {
  __REG32 PID192            : 1;
  __REG32 PID193            : 1;
  __REG32 PID194            : 1;
  __REG32 PID195            : 1;
  __REG32 PID196            : 1;
  __REG32 PID197            : 1;
  __REG32 PID198            : 1;
  __REG32 PID199            : 1;
  __REG32 PID200            : 1;
  __REG32 PID201            : 1;
  __REG32 PID202            : 1;
  __REG32 PID203            : 1;
  __REG32 PID204            : 1;
  __REG32 PID205            : 1;
  __REG32 PID206            : 1;
  __REG32 PID207            : 1;
  __REG32 PID208            : 1;
  __REG32 PID209            : 1;
  __REG32 PID210            : 1;
  __REG32 PID211            : 1;
  __REG32 PID212            : 1;
  __REG32 PID213            : 1;
  __REG32 PID214            : 1;
  __REG32 PID215            : 1;
  __REG32 PID216            : 1;
  __REG32 PID217            : 1;
  __REG32 PID218            : 1;
  __REG32 PID219            : 1;
  __REG32 PID220            : 1;
  __REG32 PID221            : 1;
  __REG32 PID222            : 1;
  __REG32 PID223            : 1;
} __gpio_pidr3l_bits;

/* Port Input Data Register (GPIO_PIDR3H) */
typedef struct {
  __REG32 PID224            : 1;
  __REG32 PID225            : 1;
  __REG32 PID226            : 1;
  __REG32 PID227            : 1;
  __REG32 PID228            : 1;
  __REG32 PID229            : 1;
  __REG32 PID230            : 1;
  __REG32 PID231            : 1;
  __REG32 PID232            : 1;
  __REG32 PID233            : 1;
  __REG32 PID234            : 1;
  __REG32 PID235            : 1;
  __REG32 PID236            : 1;
  __REG32 PID237            : 1;
  __REG32 PID238            : 1;
  __REG32 PID239            : 1;
  __REG32 PID240            : 1;
  __REG32 PID241            : 1;
  __REG32 PID242            : 1;
  __REG32 PID243            : 1;
  __REG32 PID244            : 1;
  __REG32 PID245            : 1;
  __REG32 PID246            : 1;
  __REG32 PID247            : 1;
  __REG32 PID248            : 1;
  __REG32 PID249            : 1;
  __REG32 PID250            : 1;
  __REG32 PID251            : 1;
  __REG32 PID252            : 1;
  __REG32 PID253            : 1;
  __REG32 PID254            : 1;
  __REG32 PID255            : 1;
} __gpio_pidr3h_bits;

/* Port Input Data Register (GPIO_PIDR4L) */
typedef struct {
  __REG32 PID256            : 1;
  __REG32 PID257            : 1;
  __REG32 PID258            : 1;
  __REG32 PID259            : 1;
  __REG32 PID260            : 1;
  __REG32 PID261            : 1;
  __REG32 PID262            : 1;
  __REG32 PID263            : 1;
  __REG32 PID264            : 1;
  __REG32 PID265            : 1;
  __REG32 PID266            : 1;
  __REG32 PID267            : 1;
  __REG32 PID268            : 1;
  __REG32 PID269            : 1;
  __REG32 PID270            : 1;
  __REG32 PID271            : 1;
  __REG32 PID272            : 1;
  __REG32 PID273            : 1;
  __REG32 PID274            : 1;
  __REG32 PID275            : 1;
  __REG32 PID276            : 1;
  __REG32 PID277            : 1;
  __REG32 PID278            : 1;
  __REG32 PID279            : 1;
  __REG32 PID280            : 1;
  __REG32 PID281            : 1;
  __REG32 PID282            : 1;
  __REG32 PID283            : 1;
  __REG32 PID284            : 1;
  __REG32 PID285            : 1;
  __REG32 PID286            : 1;
  __REG32 PID287            : 1;
} __gpio_pidr4l_bits;

/* Port Input Data Register (GPIO_PIDR4H) */
typedef struct {
  __REG32 PID288            : 1;
  __REG32 PID289            : 1;
  __REG32 PID290            : 1;
  __REG32 PID291            : 1;
  __REG32 PID292            : 1;
  __REG32 PID293            : 1;
  __REG32 PID294            : 1;
  __REG32 PID295            : 1;
  __REG32 PID296            : 1;
  __REG32 PID297            : 1;
  __REG32 PID298            : 1;
  __REG32 PID299            : 1;
  __REG32 PID300            : 1;
  __REG32 PID301            : 1;
  __REG32 PID302            : 1;
  __REG32 PID303            : 1;
  __REG32 PID304            : 1;
  __REG32 PID305            : 1;
  __REG32 PID306            : 1;
  __REG32 PID307            : 1;
  __REG32 PID308            : 1;
  __REG32 PID309            : 1;
  __REG32 PID310            : 1;
  __REG32 PID311            : 1;
  __REG32 PID312            : 1;
  __REG32 PID313            : 1;
  __REG32 PID314            : 1;
  __REG32 PID315            : 1;
  __REG32 PID316            : 1;
  __REG32 PID317            : 1;
  __REG32 PID318            : 1;
  __REG32 PID319            : 1;
} __gpio_pidr4h_bits;

/* Port Input Data Register (GPIO_PIDR5L) */
typedef struct {
  __REG32 PID320            : 1;
  __REG32 PID321            : 1;
  __REG32 PID322            : 1;
  __REG32 PID323            : 1;
  __REG32 PID324            : 1;
  __REG32 PID325            : 1;
  __REG32 PID326            : 1;
  __REG32 PID327            : 1;
  __REG32 PID328            : 1;
  __REG32 PID329            : 1;
  __REG32 PID330            : 1;
  __REG32 PID331            : 1;
  __REG32 PID332            : 1;
  __REG32 PID333            : 1;
  __REG32 PID334            : 1;
  __REG32 PID335            : 1;
  __REG32 PID336            : 1;
  __REG32 PID337            : 1;
  __REG32 PID338            : 1;
  __REG32 PID339            : 1;
  __REG32 PID340            : 1;
  __REG32 PID341            : 1;
  __REG32 PID342            : 1;
  __REG32 PID343            : 1;
  __REG32 PID344            : 1;
  __REG32 PID345            : 1;
  __REG32 PID346            : 1;
  __REG32 PID347            : 1;
  __REG32 PID348            : 1;
  __REG32 PID349            : 1;
  __REG32 PID350            : 1;
  __REG32 PID351            : 1;
} __gpio_pidr5l_bits;

/* Port Input Data Register (GPIO_PIDR5H) */
typedef struct {
  __REG32 PID352            : 1;
  __REG32 PID353            : 1;
  __REG32 PID354            : 1;
  __REG32 PID355            : 1;
  __REG32 PID356            : 1;
  __REG32 PID357            : 1;
  __REG32 PID358            : 1;
  __REG32 PID359            : 1;
  __REG32 PID360            : 1;
  __REG32 PID361            : 1;
  __REG32 PID362            : 1;
  __REG32 PID363            : 1;
  __REG32 PID364            : 1;
  __REG32 PID365            : 1;
  __REG32 PID366            : 1;
  __REG32 PID367            : 1;
  __REG32 PID368            : 1;
  __REG32 PID369            : 1;
  __REG32 PID370            : 1;
  __REG32 PID371            : 1;
  __REG32 PID372            : 1;
  __REG32 PID373            : 1;
  __REG32 PID374            : 1;
  __REG32 PID375            : 1;
  __REG32 PID376            : 1;
  __REG32 PID377            : 1;
  __REG32 PID378            : 1;
  __REG32 PID379            : 1;
  __REG32 PID380            : 1;
  __REG32 PID381            : 1;
  __REG32 PID382            : 1;
  __REG32 PID383            : 1;
} __gpio_pidr5h_bits;

/* Port Input Data Register (GPIO_PIDR6L) */
typedef struct {
  __REG32 PID384            : 1;
  __REG32 PID385            : 1;
  __REG32 PID386            : 1;
  __REG32 PID387            : 1;
  __REG32 PID388            : 1;
  __REG32 PID389            : 1;
  __REG32 PID390            : 1;
  __REG32 PID391            : 1;
  __REG32 PID392            : 1;
  __REG32 PID393            : 1;
  __REG32 PID394            : 1;
  __REG32 PID395            : 1;
  __REG32 PID396            : 1;
  __REG32 PID397            : 1;
  __REG32 PID398            : 1;
  __REG32 PID399            : 1;
  __REG32 PID400            : 1;
  __REG32 PID401            : 1;
  __REG32 PID402            : 1;
  __REG32 PID403            : 1;
  __REG32 PID404            : 1;
  __REG32 PID405            : 1;
  __REG32 PID406            : 1;
  __REG32 PID407            : 1;
  __REG32 PID408            : 1;
  __REG32 PID409            : 1;
  __REG32 PID410            : 1;
  __REG32 PID411            : 1;
  __REG32 PID412            : 1;
  __REG32 PID413            : 1;
  __REG32 PID414            : 1;
  __REG32 PID415            : 1;
} __gpio_pidr6l_bits;

/* Port Input Data Register (GPIO_PIDR6H) */
typedef struct {
  __REG32 PID416            : 1;
  __REG32 PID417            : 1;
  __REG32 PID418            : 1;
  __REG32 PID419            : 1;
  __REG32 PID420            : 1;
  __REG32 PID421            : 1;
  __REG32 PID422            : 1;
  __REG32 PID423            : 1;
  __REG32 PID424            : 1;
  __REG32 PID425            : 1;
  __REG32 PID426            : 1;
  __REG32 PID427            : 1;
  __REG32 PID428            : 1;
  __REG32 PID429            : 1;
  __REG32 PID430            : 1;
  __REG32 PID431            : 1;
  __REG32 PID432            : 1;
  __REG32 PID433            : 1;
  __REG32 PID434            : 1;
  __REG32 PID435            : 1;
  __REG32 PID436            : 1;
  __REG32 PID437            : 1;
  __REG32 PID438            : 1;
  __REG32 PID439            : 1;
  __REG32 PID440            : 1;
  __REG32 PID441            : 1;
  __REG32 PID442            : 1;
  __REG32 PID443            : 1;
  __REG32 PID444            : 1;
  __REG32 PID445            : 1;
  __REG32 PID446            : 1;
  __REG32 PID447            : 1;
} __gpio_pidr6h_bits;

/* Port Input Data Register (GPIO_PIDR7L) */
typedef struct {
  __REG32 PID448            : 1;
  __REG32 PID449            : 1;
  __REG32 PID450            : 1;
  __REG32 PID451            : 1;
  __REG32 PID452            : 1;
  __REG32 PID453            : 1;
  __REG32 PID454            : 1;
  __REG32 PID455            : 1;
  __REG32 PID456            : 1;
  __REG32 PID457            : 1;
  __REG32 PID458            : 1;
  __REG32 PID459            : 1;
  __REG32 PID460            : 1;
  __REG32 PID461            : 1;
  __REG32 PID462            : 1;
  __REG32 PID463            : 1;
  __REG32 PID464            : 1;
  __REG32 PID465            : 1;
  __REG32 PID466            : 1;
  __REG32 PID467            : 1;
  __REG32 PID468            : 1;
  __REG32 PID469            : 1;
  __REG32 PID470            : 1;
  __REG32 PID471            : 1;
  __REG32 PID472            : 1;
  __REG32 PID473            : 1;
  __REG32 PID474            : 1;
  __REG32 PID475            : 1;
  __REG32 PID476            : 1;
  __REG32 PID477            : 1;
  __REG32 PID478            : 1;
  __REG32 PID479            : 1;
} __gpio_pidr7l_bits;

/* Port Input Data Register (GPIO_PIDR7H) */
typedef struct {
  __REG32 PID480            : 1;
  __REG32 PID481            : 1;
  __REG32 PID482            : 1;
  __REG32 PID483            : 1;
  __REG32 PID484            : 1;
  __REG32 PID485            : 1;
  __REG32 PID486            : 1;
  __REG32 PID487            : 1;
  __REG32 PID488            : 1;
  __REG32 PID489            : 1;
  __REG32 PID490            : 1;
  __REG32 PID491            : 1;
  __REG32 PID492            : 1;
  __REG32 PID493            : 1;
  __REG32 PID494            : 1;
  __REG32 PID495            : 1;
  __REG32 PID496            : 1;
  __REG32 PID497            : 1;
  __REG32 PID498            : 1;
  __REG32 PID499            : 1;
  __REG32 PID500            : 1;
  __REG32 PID501            : 1;
  __REG32 PID502            : 1;
  __REG32 PID503            : 1;
  __REG32 PID504            : 1;
  __REG32 PID505            : 1;
  __REG32 PID506            : 1;
  __REG32 PID507            : 1;
  __REG32 PID508            : 1;
  __REG32 PID509            : 1;
  __REG32 PID510            : 1;
  __REG32 PID511            : 1;
} __gpio_pidr7h_bits;

/* Port PPU Enable Register (GPIO_PPER0L) */
typedef struct {
  __REG32 PPE0              : 1;
  __REG32 PPE1              : 1;
  __REG32 PPE2              : 1;
  __REG32 PPE3              : 1;
  __REG32 PPE4              : 1;
  __REG32 PPE5              : 1;
  __REG32 PPE6              : 1;
  __REG32 PPE7              : 1;
  __REG32 PPE8              : 1;
  __REG32 PPE9              : 1;
  __REG32 PPE10             : 1;
  __REG32 PPE11             : 1;
  __REG32 PPE12             : 1;
  __REG32 PPE13             : 1;
  __REG32 PPE14             : 1;
  __REG32 PPE15             : 1;
  __REG32 PPE16             : 1;
  __REG32 PPE17             : 1;
  __REG32 PPE18             : 1;
  __REG32 PPE19             : 1;
  __REG32 PPE20             : 1;
  __REG32 PPE21             : 1;
  __REG32 PPE22             : 1;
  __REG32 PPE23             : 1;
  __REG32 PPE24             : 1;
  __REG32 PPE25             : 1;
  __REG32 PPE26             : 1;
  __REG32 PPE27             : 1;
  __REG32 PPE28             : 1;
  __REG32 PPE29             : 1;
  __REG32 PPE30             : 1;
  __REG32 PPE31             : 1;
} __gpio_pper0l_bits;

/* Port PPU Enable Register (GPIO_PPER0H) */
typedef struct {
  __REG32 PPE32             : 1;
  __REG32 PPE33             : 1;
  __REG32 PPE34             : 1;
  __REG32 PPE35             : 1;
  __REG32 PPE36             : 1;
  __REG32 PPE37             : 1;
  __REG32 PPE38             : 1;
  __REG32 PPE39             : 1;
  __REG32 PPE40             : 1;
  __REG32 PPE41             : 1;
  __REG32 PPE42             : 1;
  __REG32 PPE43             : 1;
  __REG32 PPE44             : 1;
  __REG32 PPE45             : 1;
  __REG32 PPE46             : 1;
  __REG32 PPE47             : 1;
  __REG32 PPE48             : 1;
  __REG32 PPE49             : 1;
  __REG32 PPE50             : 1;
  __REG32 PPE51             : 1;
  __REG32 PPE52             : 1;
  __REG32 PPE53             : 1;
  __REG32 PPE54             : 1;
  __REG32 PPE55             : 1;
  __REG32 PPE56             : 1;
  __REG32 PPE57             : 1;
  __REG32 PPE58             : 1;
  __REG32 PPE59             : 1;
  __REG32 PPE60             : 1;
  __REG32 PPE61             : 1;
  __REG32 PPE62             : 1;
  __REG32 PPE63             : 1;
} __gpio_pper0h_bits;

/* Port PPU Enable Register (GPIO_PPER1L) */
typedef struct {
  __REG32 PPE64             : 1;
  __REG32 PPE65             : 1;
  __REG32 PPE66             : 1;
  __REG32 PPE67             : 1;
  __REG32 PPE68             : 1;
  __REG32 PPE69             : 1;
  __REG32 PPE70             : 1;
  __REG32 PPE71             : 1;
  __REG32 PPE72             : 1;
  __REG32 PPE73             : 1;
  __REG32 PPE74             : 1;
  __REG32 PPE75             : 1;
  __REG32 PPE76             : 1;
  __REG32 PPE77             : 1;
  __REG32 PPE78             : 1;
  __REG32 PPE79             : 1;
  __REG32 PPE80             : 1;
  __REG32 PPE81             : 1;
  __REG32 PPE82             : 1;
  __REG32 PPE83             : 1;
  __REG32 PPE84             : 1;
  __REG32 PPE85             : 1;
  __REG32 PPE86             : 1;
  __REG32 PPE87             : 1;
  __REG32 PPE88             : 1;
  __REG32 PPE89             : 1;
  __REG32 PPE90             : 1;
  __REG32 PPE91             : 1;
  __REG32 PPE92             : 1;
  __REG32 PPE93             : 1;
  __REG32 PPE94             : 1;
  __REG32 PPE95             : 1;
} __gpio_pper1l_bits;

/* Port PPU Enable Register (GPIO_PPER1H) */
typedef struct {
  __REG32 PPE96             : 1;
  __REG32 PPE97             : 1;
  __REG32 PPE98             : 1;
  __REG32 PPE99             : 1;
  __REG32 PPE100            : 1;
  __REG32 PPE101            : 1;
  __REG32 PPE102            : 1;
  __REG32 PPE103            : 1;
  __REG32 PPE104            : 1;
  __REG32 PPE105            : 1;
  __REG32 PPE106            : 1;
  __REG32 PPE107            : 1;
  __REG32 PPE108            : 1;
  __REG32 PPE109            : 1;
  __REG32 PPE110            : 1;
  __REG32 PPE111            : 1;
  __REG32 PPE112            : 1;
  __REG32 PPE113            : 1;
  __REG32 PPE114            : 1;
  __REG32 PPE115            : 1;
  __REG32 PPE116            : 1;
  __REG32 PPE117            : 1;
  __REG32 PPE118            : 1;
  __REG32 PPE119            : 1;
  __REG32 PPE120            : 1;
  __REG32 PPE121            : 1;
  __REG32 PPE122            : 1;
  __REG32 PPE123            : 1;
  __REG32 PPE124            : 1;
  __REG32 PPE125            : 1;
  __REG32 PPE126            : 1;
  __REG32 PPE127            : 1;
} __gpio_pper1h_bits;

/* Port PPU Enable Register (GPIO_PPER2L) */
typedef struct {
  __REG32 PPE128            : 1;
  __REG32 PPE129            : 1;
  __REG32 PPE130            : 1;
  __REG32 PPE131            : 1;
  __REG32 PPE132            : 1;
  __REG32 PPE133            : 1;
  __REG32 PPE134            : 1;
  __REG32 PPE135            : 1;
  __REG32 PPE136            : 1;
  __REG32 PPE137            : 1;
  __REG32 PPE138            : 1;
  __REG32 PPE139            : 1;
  __REG32 PPE140            : 1;
  __REG32 PPE141            : 1;
  __REG32 PPE142            : 1;
  __REG32 PPE143            : 1;
  __REG32 PPE144            : 1;
  __REG32 PPE145            : 1;
  __REG32 PPE146            : 1;
  __REG32 PPE147            : 1;
  __REG32 PPE148            : 1;
  __REG32 PPE149            : 1;
  __REG32 PPE150            : 1;
  __REG32 PPE151            : 1;
  __REG32 PPE152            : 1;
  __REG32 PPE153            : 1;
  __REG32 PPE154            : 1;
  __REG32 PPE155            : 1;
  __REG32 PPE156            : 1;
  __REG32 PPE157            : 1;
  __REG32 PPE158            : 1;
  __REG32 PPE159            : 1;
} __gpio_pper2l_bits;

/* Port PPU Enable Register (GPIO_PPER2H) */
typedef struct {
  __REG32 PPE160            : 1;
  __REG32 PPE161            : 1;
  __REG32 PPE162            : 1;
  __REG32 PPE163            : 1;
  __REG32 PPE164            : 1;
  __REG32 PPE165            : 1;
  __REG32 PPE166            : 1;
  __REG32 PPE167            : 1;
  __REG32 PPE168            : 1;
  __REG32 PPE169            : 1;
  __REG32 PPE170            : 1;
  __REG32 PPE171            : 1;
  __REG32 PPE172            : 1;
  __REG32 PPE173            : 1;
  __REG32 PPE174            : 1;
  __REG32 PPE175            : 1;
  __REG32 PPE176            : 1;
  __REG32 PPE177            : 1;
  __REG32 PPE178            : 1;
  __REG32 PPE179            : 1;
  __REG32 PPE180            : 1;
  __REG32 PPE181            : 1;
  __REG32 PPE182            : 1;
  __REG32 PPE183            : 1;
  __REG32 PPE184            : 1;
  __REG32 PPE185            : 1;
  __REG32 PPE186            : 1;
  __REG32 PPE187            : 1;
  __REG32 PPE188            : 1;
  __REG32 PPE189            : 1;
  __REG32 PPE190            : 1;
  __REG32 PPE191            : 1;
} __gpio_pper2h_bits;

/* Port PPU Enable Register (GPIO_PPER3L) */
typedef struct {
  __REG32 PPE192            : 1;
  __REG32 PPE193            : 1;
  __REG32 PPE194            : 1;
  __REG32 PPE195            : 1;
  __REG32 PPE196            : 1;
  __REG32 PPE197            : 1;
  __REG32 PPE198            : 1;
  __REG32 PPE199            : 1;
  __REG32 PPE200            : 1;
  __REG32 PPE201            : 1;
  __REG32 PPE202            : 1;
  __REG32 PPE203            : 1;
  __REG32 PPE204            : 1;
  __REG32 PPE205            : 1;
  __REG32 PPE206            : 1;
  __REG32 PPE207            : 1;
  __REG32 PPE208            : 1;
  __REG32 PPE209            : 1;
  __REG32 PPE210            : 1;
  __REG32 PPE211            : 1;
  __REG32 PPE212            : 1;
  __REG32 PPE213            : 1;
  __REG32 PPE214            : 1;
  __REG32 PPE215            : 1;
  __REG32 PPE216            : 1;
  __REG32 PPE217            : 1;
  __REG32 PPE218            : 1;
  __REG32 PPE219            : 1;
  __REG32 PPE220            : 1;
  __REG32 PPE221            : 1;
  __REG32 PPE222            : 1;
  __REG32 PPE223            : 1;
} __gpio_pper3l_bits;

/* Port PPU Enable Register (GPIO_PPER3H) */
typedef struct {
  __REG32 PPE224            : 1;
  __REG32 PPE225            : 1;
  __REG32 PPE226            : 1;
  __REG32 PPE227            : 1;
  __REG32 PPE228            : 1;
  __REG32 PPE229            : 1;
  __REG32 PPE230            : 1;
  __REG32 PPE231            : 1;
  __REG32 PPE232            : 1;
  __REG32 PPE233            : 1;
  __REG32 PPE234            : 1;
  __REG32 PPE235            : 1;
  __REG32 PPE236            : 1;
  __REG32 PPE237            : 1;
  __REG32 PPE238            : 1;
  __REG32 PPE239            : 1;
  __REG32 PPE240            : 1;
  __REG32 PPE241            : 1;
  __REG32 PPE242            : 1;
  __REG32 PPE243            : 1;
  __REG32 PPE244            : 1;
  __REG32 PPE245            : 1;
  __REG32 PPE246            : 1;
  __REG32 PPE247            : 1;
  __REG32 PPE248            : 1;
  __REG32 PPE249            : 1;
  __REG32 PPE250            : 1;
  __REG32 PPE251            : 1;
  __REG32 PPE252            : 1;
  __REG32 PPE253            : 1;
  __REG32 PPE254            : 1;
  __REG32 PPE255            : 1;
} __gpio_pper3h_bits;

/* Port PPU Enable Register (GPIO_PPER4L) */
typedef struct {
  __REG32 PPE256            : 1;
  __REG32 PPE257            : 1;
  __REG32 PPE258            : 1;
  __REG32 PPE259            : 1;
  __REG32 PPE260            : 1;
  __REG32 PPE261            : 1;
  __REG32 PPE262            : 1;
  __REG32 PPE263            : 1;
  __REG32 PPE264            : 1;
  __REG32 PPE265            : 1;
  __REG32 PPE266            : 1;
  __REG32 PPE267            : 1;
  __REG32 PPE268            : 1;
  __REG32 PPE269            : 1;
  __REG32 PPE270            : 1;
  __REG32 PPE271            : 1;
  __REG32 PPE272            : 1;
  __REG32 PPE273            : 1;
  __REG32 PPE274            : 1;
  __REG32 PPE275            : 1;
  __REG32 PPE276            : 1;
  __REG32 PPE277            : 1;
  __REG32 PPE278            : 1;
  __REG32 PPE279            : 1;
  __REG32 PPE280            : 1;
  __REG32 PPE281            : 1;
  __REG32 PPE282            : 1;
  __REG32 PPE283            : 1;
  __REG32 PPE284            : 1;
  __REG32 PPE285            : 1;
  __REG32 PPE286            : 1;
  __REG32 PPE287            : 1;
} __gpio_pper4l_bits;

/* Port PPU Enable Register (GPIO_PPER4H) */
typedef struct {
  __REG32 PPE288            : 1;
  __REG32 PPE289            : 1;
  __REG32 PPE290            : 1;
  __REG32 PPE291            : 1;
  __REG32 PPE292            : 1;
  __REG32 PPE293            : 1;
  __REG32 PPE294            : 1;
  __REG32 PPE295            : 1;
  __REG32 PPE296            : 1;
  __REG32 PPE297            : 1;
  __REG32 PPE298            : 1;
  __REG32 PPE299            : 1;
  __REG32 PPE300            : 1;
  __REG32 PPE301            : 1;
  __REG32 PPE302            : 1;
  __REG32 PPE303            : 1;
  __REG32 PPE304            : 1;
  __REG32 PPE305            : 1;
  __REG32 PPE306            : 1;
  __REG32 PPE307            : 1;
  __REG32 PPE308            : 1;
  __REG32 PPE309            : 1;
  __REG32 PPE310            : 1;
  __REG32 PPE311            : 1;
  __REG32 PPE312            : 1;
  __REG32 PPE313            : 1;
  __REG32 PPE314            : 1;
  __REG32 PPE315            : 1;
  __REG32 PPE316            : 1;
  __REG32 PPE317            : 1;
  __REG32 PPE318            : 1;
  __REG32 PPE319            : 1;
} __gpio_pper4h_bits;

/* Port PPU Enable Register (GPIO_PPER5L) */
typedef struct {
  __REG32 PPE320            : 1;
  __REG32 PPE321            : 1;
  __REG32 PPE322            : 1;
  __REG32 PPE323            : 1;
  __REG32 PPE324            : 1;
  __REG32 PPE325            : 1;
  __REG32 PPE326            : 1;
  __REG32 PPE327            : 1;
  __REG32 PPE328            : 1;
  __REG32 PPE329            : 1;
  __REG32 PPE330            : 1;
  __REG32 PPE331            : 1;
  __REG32 PPE332            : 1;
  __REG32 PPE333            : 1;
  __REG32 PPE334            : 1;
  __REG32 PPE335            : 1;
  __REG32 PPE336            : 1;
  __REG32 PPE337            : 1;
  __REG32 PPE338            : 1;
  __REG32 PPE339            : 1;
  __REG32 PPE340            : 1;
  __REG32 PPE341            : 1;
  __REG32 PPE342            : 1;
  __REG32 PPE343            : 1;
  __REG32 PPE344            : 1;
  __REG32 PPE345            : 1;
  __REG32 PPE346            : 1;
  __REG32 PPE347            : 1;
  __REG32 PPE348            : 1;
  __REG32 PPE349            : 1;
  __REG32 PPE350            : 1;
  __REG32 PPE351            : 1;
} __gpio_pper5l_bits;

/* Port PPU Enable Register (GPIO_PPER5H) */
typedef struct {
  __REG32 PPE352            : 1;
  __REG32 PPE353            : 1;
  __REG32 PPE354            : 1;
  __REG32 PPE355            : 1;
  __REG32 PPE356            : 1;
  __REG32 PPE357            : 1;
  __REG32 PPE358            : 1;
  __REG32 PPE359            : 1;
  __REG32 PPE360            : 1;
  __REG32 PPE361            : 1;
  __REG32 PPE362            : 1;
  __REG32 PPE363            : 1;
  __REG32 PPE364            : 1;
  __REG32 PPE365            : 1;
  __REG32 PPE366            : 1;
  __REG32 PPE367            : 1;
  __REG32 PPE368            : 1;
  __REG32 PPE369            : 1;
  __REG32 PPE370            : 1;
  __REG32 PPE371            : 1;
  __REG32 PPE372            : 1;
  __REG32 PPE373            : 1;
  __REG32 PPE374            : 1;
  __REG32 PPE375            : 1;
  __REG32 PPE376            : 1;
  __REG32 PPE377            : 1;
  __REG32 PPE378            : 1;
  __REG32 PPE379            : 1;
  __REG32 PPE380            : 1;
  __REG32 PPE381            : 1;
  __REG32 PPE382            : 1;
  __REG32 PPE383            : 1;
} __gpio_pper5h_bits;

/* Port PPU Enable Register (GPIO_PPER6L) */
typedef struct {
  __REG32 PPE384            : 1;
  __REG32 PPE385            : 1;
  __REG32 PPE386            : 1;
  __REG32 PPE387            : 1;
  __REG32 PPE388            : 1;
  __REG32 PPE389            : 1;
  __REG32 PPE390            : 1;
  __REG32 PPE391            : 1;
  __REG32 PPE392            : 1;
  __REG32 PPE393            : 1;
  __REG32 PPE394            : 1;
  __REG32 PPE395            : 1;
  __REG32 PPE396            : 1;
  __REG32 PPE397            : 1;
  __REG32 PPE398            : 1;
  __REG32 PPE399            : 1;
  __REG32 PPE400            : 1;
  __REG32 PPE401            : 1;
  __REG32 PPE402            : 1;
  __REG32 PPE403            : 1;
  __REG32 PPE404            : 1;
  __REG32 PPE405            : 1;
  __REG32 PPE406            : 1;
  __REG32 PPE407            : 1;
  __REG32 PPE408            : 1;
  __REG32 PPE409            : 1;
  __REG32 PPE410            : 1;
  __REG32 PPE411            : 1;
  __REG32 PPE412            : 1;
  __REG32 PPE413            : 1;
  __REG32 PPE414            : 1;
  __REG32 PPE415            : 1;
} __gpio_pper6l_bits;

/* Port PPU Enable Register (GPIO_PPER6H) */
typedef struct {
  __REG32 PPE416            : 1;
  __REG32 PPE417            : 1;
  __REG32 PPE418            : 1;
  __REG32 PPE419            : 1;
  __REG32 PPE420            : 1;
  __REG32 PPE421            : 1;
  __REG32 PPE422            : 1;
  __REG32 PPE423            : 1;
  __REG32 PPE424            : 1;
  __REG32 PPE425            : 1;
  __REG32 PPE426            : 1;
  __REG32 PPE427            : 1;
  __REG32 PPE428            : 1;
  __REG32 PPE429            : 1;
  __REG32 PPE430            : 1;
  __REG32 PPE431            : 1;
  __REG32 PPE432            : 1;
  __REG32 PPE433            : 1;
  __REG32 PPE434            : 1;
  __REG32 PPE435            : 1;
  __REG32 PPE436            : 1;
  __REG32 PPE437            : 1;
  __REG32 PPE438            : 1;
  __REG32 PPE439            : 1;
  __REG32 PPE440            : 1;
  __REG32 PPE441            : 1;
  __REG32 PPE442            : 1;
  __REG32 PPE443            : 1;
  __REG32 PPE444            : 1;
  __REG32 PPE445            : 1;
  __REG32 PPE446            : 1;
  __REG32 PPE447            : 1;
} __gpio_pper6h_bits;

/* Port PPU Enable Register (GPIO_PPER7L) */
typedef struct {
  __REG32 PPE448            : 1;
  __REG32 PPE449            : 1;
  __REG32 PPE450            : 1;
  __REG32 PPE451            : 1;
  __REG32 PPE452            : 1;
  __REG32 PPE453            : 1;
  __REG32 PPE454            : 1;
  __REG32 PPE455            : 1;
  __REG32 PPE456            : 1;
  __REG32 PPE457            : 1;
  __REG32 PPE458            : 1;
  __REG32 PPE459            : 1;
  __REG32 PPE460            : 1;
  __REG32 PPE461            : 1;
  __REG32 PPE462            : 1;
  __REG32 PPE463            : 1;
  __REG32 PPE464            : 1;
  __REG32 PPE465            : 1;
  __REG32 PPE466            : 1;
  __REG32 PPE467            : 1;
  __REG32 PPE468            : 1;
  __REG32 PPE469            : 1;
  __REG32 PPE470            : 1;
  __REG32 PPE471            : 1;
  __REG32 PPE472            : 1;
  __REG32 PPE473            : 1;
  __REG32 PPE474            : 1;
  __REG32 PPE475            : 1;
  __REG32 PPE476            : 1;
  __REG32 PPE477            : 1;
  __REG32 PPE478            : 1;
  __REG32 PPE479            : 1;
} __gpio_pper7l_bits;

/* Port PPU Enable Register (GPIO_PPER7H) */
typedef struct {
  __REG32 PPE480            : 1;
  __REG32 PPE481            : 1;
  __REG32 PPE482            : 1;
  __REG32 PPE483            : 1;
  __REG32 PPE484            : 1;
  __REG32 PPE485            : 1;
  __REG32 PPE486            : 1;
  __REG32 PPE487            : 1;
  __REG32 PPE488            : 1;
  __REG32 PPE489            : 1;
  __REG32 PPE490            : 1;
  __REG32 PPE491            : 1;
  __REG32 PPE492            : 1;
  __REG32 PPE493            : 1;
  __REG32 PPE494            : 1;
  __REG32 PPE495            : 1;
  __REG32 PPE496            : 1;
  __REG32 PPE497            : 1;
  __REG32 PPE498            : 1;
  __REG32 PPE499            : 1;
  __REG32 PPE500            : 1;
  __REG32 PPE501            : 1;
  __REG32 PPE502            : 1;
  __REG32 PPE503            : 1;
  __REG32 PPE504            : 1;
  __REG32 PPE505            : 1;
  __REG32 PPE506            : 1;
  __REG32 PPE507            : 1;
  __REG32 PPE508            : 1;
  __REG32 PPE509            : 1;
  __REG32 PPE510            : 1;
  __REG32 PPE511            : 1;
} __gpio_pper7h_bits;

/* A/D Input Enable Registers (ADCn_ER32) */
typedef struct {
  __REG16 ADE15             : 1;
  __REG16 ADE14             : 1;
  __REG16 ADE13             : 1;
  __REG16 ADE12             : 1;
  __REG16 ADE11             : 1;
  __REG16 ADE10             : 1;
  __REG16 ADE9              : 1;
  __REG16 ADE8              : 1;
  __REG16 ADE7              : 1;
  __REG16 ADE6              : 1;
  __REG16 ADE5              : 1;
  __REG16 ADE4              : 1;
  __REG16 ADE3              : 1;
  __REG16 ADE2              : 1;
  __REG16 ADE1              : 1;
  __REG16 ADE0              : 1;
} __adcn_er32_bits;

/* A/D Input Enable Registers (ADCn_ER10) */
typedef struct {
  __REG16 ADE31             : 1;
  __REG16 ADE30             : 1;
  __REG16 ADE29             : 1;
  __REG16 ADE28             : 1;
  __REG16 ADE27             : 1;
  __REG16 ADE26             : 1;
  __REG16 ADE25             : 1;
  __REG16 ADE24             : 1;
  __REG16 ADE23             : 1;
  __REG16 ADE22             : 1;
  __REG16 ADE21             : 1;
  __REG16 ADE20             : 1;
  __REG16 ADE19             : 1;
  __REG16 ADE18             : 1;
  __REG16 ADE17             : 1;
  __REG16 ADE16             : 1;
} __adcn_er10_bits;

/* A/D Control Status Register 3 (ADCn_CS3) */
typedef struct {
  __REG8  INTE2             : 1;
  __REG8  INT2              : 1;
  __REG8                    : 2;
  __REG8  PAUS              : 1;
  __REG8  INTE              : 1;
  __REG8  INT               : 1;
  __REG8  BUSY              : 1;
} __adcn_cs3_bits;

/* A/D Control Status Register 2 (ADCn_CS2) */
typedef struct {
  __REG8  DBGE              : 1;
  __REG8  DPDIS             : 1;
  __REG8                    : 6;
} __adcn_cs2_bits;

/* A/D Control Status Register 1 (ADCn_CS1) */
typedef struct {
  __REG8  ACHMD             : 1;
  __REG8  STRT              : 1;
  __REG8  STS               : 2;
  __REG8  PAUS              : 1;
  __REG8  INTE              : 1;
  __REG8  INT               : 1;
  __REG8  BUSY              : 1;
} __adcn_cs1_bits;

/* A/D Control Status Register 0 (ADCn_CS0) */
typedef struct {
  __REG8  ACH               : 5;
  __REG8  S10               : 1;
  __REG8  MD                : 2;
} __adcn_cs0_bits;

/* A/D Control Status Register 1 Set Register (ADCn_CSS1) */
typedef struct {
  __REG8                    : 1;
  __REG8  STRTS             : 1;
  __REG8                    : 3;
  __REG8  INTES             : 1;
  __REG8                    : 2;
} __adcn_css1_bits;

/* A/D Control Status Register 1 Clear Register (ADCn_CSC1) */
typedef struct {
  __REG8                    : 4;
  __REG8  PAUSC             : 1;
  __REG8  INTEC             : 1;
  __REG8  INTC              : 1;
  __REG8  BUSYC             : 1;
} __adcn_csc1_bits;

/* A/D Control Status Register 3 Set Register (ADCn_CSS3) */
typedef struct {
  __REG8  INTE2S            : 1;
  __REG8                    : 7;
} __adcn_css3_bits;

/* A/D Control Status Register 3 Clear Register (ADCn_CSC3) */
typedef struct {
  __REG8  INTE2C            : 1;
  __REG8  INT2C             : 1;
  __REG8                    : 6;
} __adcn_csc3_bits;

/* Common Data Register (ADCn_CR) */
typedef struct {
  __REG16 D                 :10;
  __REG16                   : 6;
} __adcn_cd_bits;

/* ADC Conversion Time Setting Register (ADCn_CT) */
typedef struct {
  __REG16 ST                :10;
  __REG16 CT                : 6;
} __adcn_ct_bits;

/* A/D Start Channel Setting Register (ADCn_SCH) */
typedef struct {
  __REG8  ANS               : 5;
  __REG8                    : 3;
} __adcn_sch_bits;

/* A/D End Channel Setting Register (ADCn_ECH) */
typedef struct {
  __REG8  ANE               : 5;
  __REG8                    : 3;
} __adcn_ech_bits;

/* A/D Converter DMA Configuration Register (ADCn_MAR) */
typedef struct {
  __REG8  DRQEN             : 1;
  __REG8  DRQEN2            : 1;
  __REG8                    : 6;
} __adcn_mar_bits;

/* A/D Converter DMA Configuration Set Register (ADCn_MASR) */
/* A/D Converter DMA Configuration Clear Register (ADCn_MACR) */
typedef struct {
  __REG8  DRQENS            : 1;
  __REG8  DRQENS2           : 1;
  __REG8                    : 6;
} __adcn_macr_bits;

/* A/D Converter Channel Control Registers (ADCn_CC15~0) */
typedef struct {
  __REG8  RCOS0             : 2;
  __REG8  RCOE0             : 1;
  __REG8  RCOIE0            : 1;
  __REG8  RCOS1             : 2;
  __REG8  RCOE1             : 1;
  __REG8  RCOIE1            : 1;
} __adcn_cc_bits;

/* Inverted Range Selection Registers (ADCn_RCOIRS32) */
typedef struct {
  __REG16 RCOIRS15          : 1;
  __REG16 RCOIRS14          : 1;
  __REG16 RCOIRS13          : 1;
  __REG16 RCOIRS12          : 1;
  __REG16 RCOIRS11          : 1;
  __REG16 RCOIRS10          : 1;
  __REG16 RCOIRS9           : 1;
  __REG16 RCOIRS8           : 1;
  __REG16 RCOIRS7           : 1;
  __REG16 RCOIRS6           : 1;
  __REG16 RCOIRS5           : 1;
  __REG16 RCOIRS4           : 1;
  __REG16 RCOIRS3           : 1;
  __REG16 RCOIRS2           : 1;
  __REG16 RCOIRS1           : 1;
  __REG16 RCOIRS0           : 1;
} __adcn_rcoirs32_bits;

/* Inverted Range Selection Register (ADCn_RCOIRS10) */
typedef struct {
  __REG16 RCOIRS31          : 1;
  __REG16 RCOIRS30          : 1;
  __REG16 RCOIRS29          : 1;
  __REG16 RCOIRS28          : 1;
  __REG16 RCOIRS27          : 1;
  __REG16 RCOIRS26          : 1;
  __REG16 RCOIRS25          : 1;
  __REG16 RCOIRS24          : 1;
  __REG16 RCOIRS23          : 1;
  __REG16 RCOIRS22          : 1;
  __REG16 RCOIRS21          : 1;
  __REG16 RCOIRS20          : 1;
  __REG16 RCOIRS19          : 1;
  __REG16 RCOIRS18          : 1;
  __REG16 RCOIRS17          : 1;
  __REG16 RCOIRS16          : 1;
} __adcn_rcoirs10_bits;

/* Range Comparator Interrupt Flag (ADCn_RCOINT32) */
typedef struct {
  __REG16 RCOINT15          : 1;
  __REG16 RCOINT14          : 1;
  __REG16 RCOINT13          : 1;
  __REG16 RCOINT12          : 1;
  __REG16 RCOINT11          : 1;
  __REG16 RCOINT10          : 1;
  __REG16 RCOINT9           : 1;
  __REG16 RCOIRS8           : 1;
  __REG16 RCOIRS7           : 1;
  __REG16 RCOIRS6           : 1;
  __REG16 RCOIRS5           : 1;
  __REG16 RCOIRS4           : 1;
  __REG16 RCOIRS3           : 1;
  __REG16 RCOIRS2           : 1;
  __REG16 RCOIRS1           : 1;
  __REG16 RCOIRS0           : 1;
} __adcn_rcoint32_bits;

/* Range Comparator Interrupt Flag (ADCn_RCOINT10) */
typedef struct {
  __REG16 RCOINT31          : 1;
  __REG16 RCOINT30          : 1;
  __REG16 RCOINT29          : 1;
  __REG16 RCOINT28          : 1;
  __REG16 RCOINT27          : 1;
  __REG16 RCOINT26          : 1;
  __REG16 RCOINT25          : 1;
  __REG16 RCOIRS24          : 1;
  __REG16 RCOIRS23          : 1;
  __REG16 RCOIRS22          : 1;
  __REG16 RCOIRS21          : 1;
  __REG16 RCOIRS20          : 1;
  __REG16 RCOIRS19          : 1;
  __REG16 RCOIRS18          : 1;
  __REG16 RCOIRS17          : 1;
  __REG16 RCOIRS16          : 1;
} __adcn_rcoint10_bits;

/* Range Comparator Over Threshold Flag (ADCn_RCOOF32) */
typedef struct {
  __REG16 RCOOF15           : 1;
  __REG16 RCOOF14           : 1;
  __REG16 RCOOF13           : 1;
  __REG16 RCOOF12           : 1;
  __REG16 RCOOF11           : 1;
  __REG16 RCOOF10           : 1;
  __REG16 RCOOF9            : 1;
  __REG16 RCOOF8            : 1;
  __REG16 RCOOF7            : 1;
  __REG16 RCOOF6            : 1;
  __REG16 RCOOF5            : 1;
  __REG16 RCOOF4            : 1;
  __REG16 RCOOF3            : 1;
  __REG16 RCOOF2            : 1;
  __REG16 RCOOF1            : 1;
  __REG16 RCOOF0            : 1;
} __adcn_rcoof32_bits;

/* Range Comparator Over Threshold Flag (ADCn_RCOOF10) */
typedef struct {
  __REG16 RCOOF31           : 1;
  __REG16 RCOOF30           : 1;
  __REG16 RCOOF29           : 1;
  __REG16 RCOOF28           : 1;
  __REG16 RCOOF27           : 1;
  __REG16 RCOOF26           : 1;
  __REG16 RCOOF25           : 1;
  __REG16 RCOOF24           : 1;
  __REG16 RCOOF23           : 1;
  __REG16 RCOOF22           : 1;
  __REG16 RCOOF21           : 1;
  __REG16 RCOOF20           : 1;
  __REG16 RCOOF19           : 1;
  __REG16 RCOOF18           : 1;
  __REG16 RCOOF17           : 1;
  __REG16 RCOOF16           : 1;
} __adcn_rcoof10_bits;

/* Range Comparator Interrupt Clear Register (ADCn_RCOINTC32) */
typedef struct {
  __REG16 RCOINTC15         : 1;
  __REG16 RCOINTC14         : 1;
  __REG16 RCOINTC13         : 1;
  __REG16 RCOINTC12         : 1;
  __REG16 RCOINTC11         : 1;
  __REG16 RCOINTC10         : 1;
  __REG16 RCOINTC9          : 1;
  __REG16 RCOINTC8          : 1;
  __REG16 RCOINTC7          : 1;
  __REG16 RCOINTC6          : 1;
  __REG16 RCOINTC5          : 1;
  __REG16 RCOINTC4          : 1;
  __REG16 RCOINTC3          : 1;
  __REG16 RCOINTC2          : 1;
  __REG16 RCOINTC1          : 1;
  __REG16 RCOINTC0          : 1;
} __adcn_rcointc32_bits;

/* Range Comparator Interrupt Clear Register (ADCn_RCOINTC10) */
typedef struct {
  __REG16 RCOINTC31         : 1;
  __REG16 RCOINTC30         : 1;
  __REG16 RCOINTC29         : 1;
  __REG16 RCOINTC28         : 1;
  __REG16 RCOINTC27         : 1;
  __REG16 RCOINTC26         : 1;
  __REG16 RCOINTC25         : 1;
  __REG16 RCOINTC24         : 1;
  __REG16 RCOINTC23         : 1;
  __REG16 RCOINTC22         : 1;
  __REG16 RCOINTC21         : 1;
  __REG16 RCOINTC20         : 1;
  __REG16 RCOINTC19         : 1;
  __REG16 RCOINTC18         : 1;
  __REG16 RCOINTC17         : 1;
  __REG16 RCOINTC16         : 1;
} __adcn_rcointc10_bits;

/* ADC Pulse Negative Counter Reload Register n (ADCn_PCTNRLn) */
typedef struct {
  __REG8  D                 : 5;
  __REG8                    : 3;
} __adcn_pctnrl_bits;

/* ADC Pulse Negative Counter n (ADCn_PCTNCTn) */
typedef struct {
  __REG8  D                 : 5;
  __REG8                    : 3;
} __adcn_pctnct_bits;

/* ADC Pulse Counter Zero Flag Register (ADCn_PCZF32) */
typedef struct {
  __REG16 CTPZF15           : 1;
  __REG16 CTPZF14           : 1;
  __REG16 CTPZF13           : 1;
  __REG16 CTPZF12           : 1;
  __REG16 CTPZF11           : 1;
  __REG16 CTPZF10           : 1;
  __REG16 CTPZF9            : 1;
  __REG16 CTPZF8            : 1;
  __REG16 CTPZF7            : 1;
  __REG16 CTPZF6            : 1;
  __REG16 CTPZF5            : 1;
  __REG16 CTPZF4            : 1;
  __REG16 CTPZF3            : 1;
  __REG16 CTPZF2            : 1;
  __REG16 CTPZF1            : 1;
  __REG16 CTPZF0            : 1;
} __adcn_pczf32_bits;

/* ADC Pulse Counter Zero Flag Register (ADCn_PCZF10) */
typedef struct {
  __REG16 CTPZF31           : 1;
  __REG16 CTPZF30           : 1;
  __REG16 CTPZF29           : 1;
  __REG16 CTPZF28           : 1;
  __REG16 CTPZF27           : 1;
  __REG16 CTPZF26           : 1;
  __REG16 CTPZF25           : 1;
  __REG16 CTPZF24           : 1;
  __REG16 CTPZF23           : 1;
  __REG16 CTPZF22           : 1;
  __REG16 CTPZF21           : 1;
  __REG16 CTPZF20           : 1;
  __REG16 CTPZF19           : 1;
  __REG16 CTPZF18           : 1;
  __REG16 CTPZF17           : 1;
  __REG16 CTPZF16           : 1;
} __adcn_pczf10_bits;

/* ADC Pulse Counter Zero Flag Clear Register (ADCn_PCZFC32) */
typedef struct {
  __REG16 CTPZFC15          : 1;
  __REG16 CTPZFC14          : 1;
  __REG16 CTPZFC13          : 1;
  __REG16 CTPZFC12          : 1;
  __REG16 CTPZFC11          : 1;
  __REG16 CTPZFC10          : 1;
  __REG16 CTPZFC9           : 1;
  __REG16 CTPZFC8           : 1;
  __REG16 CTPZFC7           : 1;
  __REG16 CTPZFC6           : 1;
  __REG16 CTPZFC5           : 1;
  __REG16 CTPZFC4           : 1;
  __REG16 CTPZFC3           : 1;
  __REG16 CTPZFC2           : 1;
  __REG16 CTPZFC1           : 1;
  __REG16 CTPZFC0           : 1;
} __adcn_pczfc32_bits;

/* ADC Pulse Counter Zero Flag Clear Register (ADCn_PCZFC10) */
typedef struct {
  __REG16 CTPZFC31          : 1;
  __REG16 CTPZFC30          : 1;
  __REG16 CTPZFC29          : 1;
  __REG16 CTPZFC28          : 1;
  __REG16 CTPZFC27          : 1;
  __REG16 CTPZFC26          : 1;
  __REG16 CTPZFC25          : 1;
  __REG16 CTPZFC24          : 1;
  __REG16 CTPZFC23          : 1;
  __REG16 CTPZFC22          : 1;
  __REG16 CTPZFC21          : 1;
  __REG16 CTPZFC20          : 1;
  __REG16 CTPZFC19          : 1;
  __REG16 CTPZFC18          : 1;
  __REG16 CTPZFC17          : 1;
  __REG16 CTPZFC16          : 1;
} __adcn_pczfc10_bits;

/* ADC Pulse Counter Interrupt Enable Register (ADCn_PCIE32) */
typedef struct {
  __REG16 CTPIE15           : 1;
  __REG16 CTPIE14           : 1;
  __REG16 CTPIE13           : 1;
  __REG16 CTPIE12           : 1;
  __REG16 CTPIE11           : 1;
  __REG16 CTPIE10           : 1;
  __REG16 CTPIE9            : 1;
  __REG16 CTPIE8            : 1;
  __REG16 CTPIE7            : 1;
  __REG16 CTPIE6            : 1;
  __REG16 CTPIE5            : 1;
  __REG16 CTPIE4            : 1;
  __REG16 CTPIE3            : 1;
  __REG16 CTPIE2            : 1;
  __REG16 CTPIE1            : 1;
  __REG16 CTPIE0            : 1;
} __adcn_pcie32_bits;

/* ADC Pulse Counter Interrupt Enable Register (ADCn_PCIE10) */
typedef struct {
  __REG16 CTPIE31           : 1;
  __REG16 CTPIE30           : 1;
  __REG16 CTPIE29           : 1;
  __REG16 CTPIE28           : 1;
  __REG16 CTPIE27           : 1;
  __REG16 CTPIE26           : 1;
  __REG16 CTPIE25           : 1;
  __REG16 CTPIE24           : 1;
  __REG16 CTPIE23           : 1;
  __REG16 CTPIE22           : 1;
  __REG16 CTPIE21           : 1;
  __REG16 CTPIE20           : 1;
  __REG16 CTPIE19           : 1;
  __REG16 CTPIE18           : 1;
  __REG16 CTPIE17           : 1;
  __REG16 CTPIE16           : 1;
} __adcn_pcie10_bits;

/* ADC Pulse Counter Interrupt Enable Set Register (ADCn_PCIES32) */
typedef struct {
  __REG16 CTPIES15          : 1;
  __REG16 CTPIES14          : 1;
  __REG16 CTPIES13          : 1;
  __REG16 CTPIES12          : 1;
  __REG16 CTPIES11          : 1;
  __REG16 CTPIES10          : 1;
  __REG16 CTPIES9           : 1;
  __REG16 CTPIES8           : 1;
  __REG16 CTPIES7           : 1;
  __REG16 CTPIES6           : 1;
  __REG16 CTPIES5           : 1;
  __REG16 CTPIES4           : 1;
  __REG16 CTPIES3           : 1;
  __REG16 CTPIES2           : 1;
  __REG16 CTPIES1           : 1;
  __REG16 CTPIES0           : 1;
} __adcn_pcies32_bits;

/* ADC Pulse Counter Interrupt Enable Set Register (ADCn_PCIES10) */
typedef struct {
  __REG16 CTPIES31          : 1;
  __REG16 CTPIES30          : 1;
  __REG16 CTPIES29          : 1;
  __REG16 CTPIES28          : 1;
  __REG16 CTPIES27          : 1;
  __REG16 CTPIES26          : 1;
  __REG16 CTPIES25          : 1;
  __REG16 CTPIES24          : 1;
  __REG16 CTPIES23          : 1;
  __REG16 CTPIES22          : 1;
  __REG16 CTPIES21          : 1;
  __REG16 CTPIES20          : 1;
  __REG16 CTPIES19          : 1;
  __REG16 CTPIES18          : 1;
  __REG16 CTPIES17          : 1;
  __REG16 CTPIES16          : 1;
} __adcn_pcies10_bits;

/* ADC Pulse Counter Interrupt Enable Clear Register (ADCn_PCIEC32) */
typedef struct {
  __REG16 CTPIEC15          : 1;
  __REG16 CTPIEC14          : 1;
  __REG16 CTPIEC13          : 1;
  __REG16 CTPIEC12          : 1;
  __REG16 CTPIEC11          : 1;
  __REG16 CTPIEC10          : 1;
  __REG16 CTPIEC9           : 1;
  __REG16 CTPIEC8           : 1;
  __REG16 CTPIEC7           : 1;
  __REG16 CTPIEC6           : 1;
  __REG16 CTPIEC5           : 1;
  __REG16 CTPIEC4           : 1;
  __REG16 CTPIEC3           : 1;
  __REG16 CTPIEC2           : 1;
  __REG16 CTPIEC1           : 1;
  __REG16 CTPIEC0           : 1;
} __adcn_pciec32_bits;

/* ADC Pulse Counter Interrupt Enable Clear Register (ADCn_PCIEC10) */
typedef struct {
  __REG16 CTPIEC31          : 1;
  __REG16 CTPIEC30          : 1;
  __REG16 CTPIEC29          : 1;
  __REG16 CTPIEC28          : 1;
  __REG16 CTPIEC27          : 1;
  __REG16 CTPIEC26          : 1;
  __REG16 CTPIEC25          : 1;
  __REG16 CTPIEC24          : 1;
  __REG16 CTPIEC23          : 1;
  __REG16 CTPIEC22          : 1;
  __REG16 CTPIEC21          : 1;
  __REG16 CTPIEC20          : 1;
  __REG16 CTPIEC19          : 1;
  __REG16 CTPIEC18          : 1;
  __REG16 CTPIEC17          : 1;
  __REG16 CTPIEC16          : 1;
} __adcn_pciec10_bits;

/* Peripheral Enable 0 Low Register (BSU0_PEN0L) */
typedef struct {
  __REG16 ENADC0            : 1;
  __REG16 ENADC1            : 1;
  __REG16                   :14;
} __bsu0_pen0l_bits;

/* Peripheral Enable 1 Low Register (BSU0_PEN1L) */
typedef struct {
  __REG16 ENFRT0            : 1;
  __REG16 ENFRT1            : 1;
  __REG16 ENFRT2            : 1;
  __REG16 ENFRT3            : 1;
  __REG16 ENFRT4            : 1;
  __REG16 ENFRT5            : 1;
  __REG16 ENFRT6            : 1;
  __REG16 ENFRT7            : 1;
  __REG16 ENFRT8            : 1;
  __REG16 ENFRT9            : 1;
  __REG16 ENFRT10           : 1;
  __REG16 ENFRT11           : 1;
  __REG16 ENFRT12           : 1;
  __REG16 ENFRT13           : 1;
  __REG16 ENFRT14           : 1;
  __REG16 ENFRT15           : 1;
} __bsu0_pen1l_bits;

/* Peripheral Enable 2 Low Register (BSU0_PEN2L) */
typedef struct {
  __REG16 ENICU0            : 1;
  __REG16 ENICU1            : 1;
  __REG16 ENICU2            : 1;
  __REG16 ENICU3            : 1;
  __REG16 ENICU4            : 1;
  __REG16 ENICU5            : 1;
  __REG16 ENICU6            : 1;
  __REG16 ENICU7            : 1;
  __REG16 ENICU8            : 1;
  __REG16 ENICU9            : 1;
  __REG16 ENICU10           : 1;
  __REG16 ENICU11           : 1;
  __REG16 ENICU12           : 1;
  __REG16 ENICU13           : 1;
  __REG16 ENICU14           : 1;
  __REG16 ENICU15           : 1;
} __bsu0_pen2l_bits;

/* Peripheral Enable 3 Low Register (BSU0_PEN3L) */
typedef struct {
  __REG16 ENOCU0            : 1;
  __REG16 ENOCU1            : 1;
  __REG16 ENOCU2            : 1;
  __REG16 ENOCU3            : 1;
  __REG16 ENOCU4            : 1;
  __REG16 ENOCU5            : 1;
  __REG16 ENOCU6            : 1;
  __REG16 ENOCU7            : 1;
  __REG16 ENOCU8            : 1;
  __REG16 ENOCU9            : 1;
  __REG16 ENOCU10           : 1;
  __REG16 ENOCU11           : 1;
  __REG16 ENOCU12           : 1;
  __REG16 ENOCU13           : 1;
  __REG16 ENOCU14           : 1;
  __REG16 ENOCU15           : 1;
} __bsu0_pen3l_bits;

/* Peripheral Enable 4 Low Register (BSU0_PEN4L) */
typedef struct {
  __REG16 ENI2C0            : 1;
  __REG16 ENI2C1            : 1;
  __REG16 ENI2C2            : 1;
  __REG16                   :13;
} __bsu0_pen4l_bits;

/* Peripheral Enable 5 Low Register (BSU0_PEN5L) */
typedef struct {
  __REG16 ENUSART0          : 1;
  __REG16 ENUSART1          : 1;
  __REG16 ENUSART2          : 1;
  __REG16 ENUSART3          : 1;
  __REG16 ENUSART4          : 1;
  __REG16 ENUSART5          : 1;
  __REG16                   :10;
} __bsu0_pen5l_bits;

/* Peripheral Enable 6 Low Register (BSU0_PEN6L) */
typedef struct {
  __REG16 ENSMC0            : 1;
  __REG16 ENSMC1            : 1;
  __REG16 ENSMC2            : 1;
  __REG16 ENSMC3            : 1;
  __REG16 ENSMC4            : 1;
  __REG16 ENSMC5            : 1;
  __REG16 ENSMCTG0          : 1;
  __REG16                   : 9;
} __bsu0_pen6l_bits;

/* Peripheral Enable 7 Low Register (BSU0_PEN7L) */
typedef struct {
  __REG16 ENPPG0            : 1;
  __REG16 ENPPG1            : 1;
  __REG16 ENPPG2            : 1;
  __REG16 ENPPG3            : 1;
  __REG16 ENPPG4            : 1;
  __REG16 ENPPG5            : 1;
  __REG16 ENPPG6            : 1;
  __REG16 ENPPG7            : 1;
  __REG16 ENPPG8            : 1;
  __REG16 ENPPG9            : 1;
  __REG16 ENPPG10           : 1;
  __REG16 ENPPG11           : 1;
  __REG16 ENPPG12           : 1;
  __REG16 ENPPG13           : 1;
  __REG16 ENPPG14           : 1;
  __REG16 ENPPG15           : 1;
} __bsu0_pen7l_bits;

/* Peripheral Enable 7 High Register (BSU0_PEN7H) */
typedef struct {
  __REG16 ENPPG16           : 1;
  __REG16 ENPPG17           : 1;
  __REG16 ENPPG18           : 1;
  __REG16 ENPPG19           : 1;
  __REG16 ENPPG20           : 1;
  __REG16 ENPPG21           : 1;
  __REG16 ENPPG22           : 1;
  __REG16 ENPPG23           : 1;
  __REG16 ENPPG24           : 1;
  __REG16 ENPPG25           : 1;
  __REG16 ENPPG26           : 1;
  __REG16 ENPPG27           : 1;
  __REG16 ENPPG28           : 1;
  __REG16 ENPPG29           : 1;
  __REG16 ENPPG30           : 1;
  __REG16 ENPPG31           : 1;
} __bsu0_pen7h_bits;

/* Peripheral Enable 8 Low Register (BSU0_PEN8L) */
typedef struct {
  __REG16 ENPPG32           : 1;
  __REG16 ENPPG33           : 1;
  __REG16 ENPPG34           : 1;
  __REG16 ENPPG35           : 1;
  __REG16 ENPPG36           : 1;
  __REG16 ENPPG37           : 1;
  __REG16 ENPPG38           : 1;
  __REG16 ENPPG39           : 1;
  __REG16 ENPPG40           : 1;
  __REG16 ENPPG41           : 1;
  __REG16 ENPPG42           : 1;
  __REG16 ENPPG43           : 1;
  __REG16 ENPPG44           : 1;
  __REG16 ENPPG45           : 1;
  __REG16 ENPPG46           : 1;
  __REG16 ENPPG47           : 1;
} __bsu0_pen8l_bits;

/* Peripheral Enable 8 High Register (BSU0_PEN8H) */
typedef struct {
  __REG16 ENPPG48           : 1;
  __REG16 ENPPG49           : 1;
  __REG16 ENPPG50           : 1;
  __REG16 ENPPG51           : 1;
  __REG16 ENPPG52           : 1;
  __REG16 ENPPG53           : 1;
  __REG16 ENPPG54           : 1;
  __REG16 ENPPG55           : 1;
  __REG16 ENPPG56           : 1;
  __REG16 ENPPG57           : 1;
  __REG16 ENPPG58           : 1;
  __REG16 ENPPG59           : 1;
  __REG16 ENPPG60           : 1;
  __REG16 ENPPG61           : 1;
  __REG16 ENPPG62           : 1;
  __REG16 ENPPG63           : 1;
} __bsu0_pen8h_bits;

/* Peripheral Enable 9 Low Register (BSU0_PEN9L) */
typedef struct {
  __REG16 ENPPGGRP0           : 1;
  __REG16 ENPPGGRP1           : 1;
  __REG16 ENPPGGRP2           : 1;
  __REG16 ENPPGGRP3           : 1;
  __REG16 ENPPGGRP4           : 1;
  __REG16 ENPPGGRP5           : 1;
  __REG16 ENPPGGRP6           : 1;
  __REG16 ENPPGGRP7           : 1;
  __REG16 ENPPGGRP8           : 1;
  __REG16 ENPPGGRP9           : 1;
  __REG16 ENPPGGRP10          : 1;
  __REG16 ENPPGGRP11          : 1;
  __REG16 ENPPGGRP12          : 1;
  __REG16 ENPPGGRP13          : 1;
  __REG16 ENPPGGRP14          : 1;
  __REG16 ENPPGGRP15          : 1;
} __bsu0_pen9l_bits;

/* Peripheral Enable 9 High Register (BSU0_PEN9H) */
typedef struct {
  __REG16 ENPPGGLC0         : 1;
  __REG16                   :15;
} __bsu0_pen9h_bits;

/* Peripheral Enable 11 Low Register (BSU0_PEN11L) */
typedef struct {
  __REG16 ENBMC0            : 1;
  __REG16 ENBMC1            : 1;
  __REG16 ENBMC2            : 1;
  __REG16 ENBMC3            : 1;
  __REG16                   :12;
} __bsu0_pen11l_bits;

/* Peripheral Enable 0 Low Register (BSU1_PEN0L) */
typedef struct {
  __REG16 ENSG0             : 1;
  __REG16 ENSG1             : 1;
  __REG16 ENSG2             : 1;
  __REG16 ENSG3             : 1;
  __REG16                   :12;
} __bsu1_pen0l_bits;

/* Peripheral Enable 1 Low Register (BSU1_PEN1L) */
typedef struct {
  __REG16 ENCAN0            : 1;
  __REG16 ENCAN1            : 1;
  __REG16 ENCAN2            : 1;
  __REG16 ENCAN3            : 1;
  __REG16 ENCAN4            : 1;
  __REG16 ENCAN5            : 1;
  __REG16 ENCAN6            : 1;
  __REG16 ENCAN7            : 1;
  __REG16                   : 8;
} __bsu1_pen1l_bits;

/* Peripheral Enable 2 Low Register (BSU1_PEN2L) */
typedef struct {
  __REG16 ENLCD             : 1;
  __REG16                   :15;
} __bsu1_pen2l_bits;

/* Peripheral Enable 3 Low Register (BSU1_PEN3L) */
typedef struct {
  __REG16 ENFRT16           : 1;
  __REG16 ENFRT17           : 1;
  __REG16 ENFRT18           : 1;
  __REG16 ENFRT19           : 1;
  __REG16 ENFRT20           : 1;
  __REG16 ENFRT21           : 1;
  __REG16 ENFRT22           : 1;
  __REG16 ENFRT23           : 1;
  __REG16 ENFRT24           : 1;
  __REG16 ENFRT25           : 1;
  __REG16 ENFRT26           : 1;
  __REG16 ENFRT27           : 1;
  __REG16 ENFRT28           : 1;
  __REG16 ENFRT29           : 1;
  __REG16 ENFRT30           : 1;
  __REG16 ENFRT31           : 1;
} __bsu1_pen3l_bits;

/* Peripheral Enable 4 Low Register (BSU1_PEN4L) */
typedef struct {
  __REG16 ENICU16           : 1;
  __REG16 ENICU17           : 1;
  __REG16 ENICU18           : 1;
  __REG16 ENICU19           : 1;
  __REG16 ENICU20           : 1;
  __REG16 ENICU21           : 1;
  __REG16 ENICU22           : 1;
  __REG16 ENICU23           : 1;
  __REG16 ENICU24           : 1;
  __REG16 ENICU25           : 1;
  __REG16 ENICU26           : 1;
  __REG16 ENICU27           : 1;
  __REG16 ENICU28           : 1;
  __REG16 ENICU29           : 1;
  __REG16 ENICU30           : 1;
  __REG16 ENICU31           : 1;
} __bsu1_pen4l_bits;

/* Peripheral Enable 5 Low Register (BSU1_PEN5L) */
typedef struct {
  __REG16 ENOCU16           : 1;
  __REG16 ENOCU17           : 1;
  __REG16 ENOCU18           : 1;
  __REG16 ENOCU19           : 1;
  __REG16 ENOCU20           : 1;
  __REG16 ENOCU21           : 1;
  __REG16 ENOCU22           : 1;
  __REG16 ENOCU23           : 1;
  __REG16 ENOCU24           : 1;
  __REG16 ENOCU25           : 1;
  __REG16 ENOCU26           : 1;
  __REG16 ENOCU27           : 1;
  __REG16 ENOCU28           : 1;
  __REG16 ENOCU29           : 1;
  __REG16 ENOCU30           : 1;
  __REG16 ENOCU31           : 1;
} __bsu1_pen5l_bits;

/* Peripheral Enable 6 Low Register (BSU1_PEN6L) */
typedef struct {
  __REG16 ENI2C3            : 1;
  __REG16 ENI2C4            : 1;
  __REG16 ENI2C5            : 1;
  __REG16                   :13;
} __bsu1_pen6l_bits;

/* Peripheral Enable 7 Low Register (BSU1_PEN7L) */
typedef struct {
  __REG16 ENUSART6          : 1;
  __REG16 ENUSART7          : 1;
  __REG16 ENUSART8          : 1;
  __REG16 ENUSART9          : 1;
  __REG16 ENUSART10         : 1;
  __REG16 ENUSART11         : 1;
  __REG16                   :10;
} __bsu1_pen7l_bits;

/* Peripheral Enable 8 Low Register (BSU1_PEN8L) */
typedef struct {
  __REG16 ENSMC6            : 1;
  __REG16 ENSMC7            : 1;
  __REG16 ENSMC8            : 1;
  __REG16 ENSMC9            : 1;
  __REG16 ENSMC10           : 1;
  __REG16 ENSMC11           : 1;
  __REG16 ENSMCTG1          : 1;
  __REG16                   : 9;
} __bsu1_pen8l_bits;

/* Peripheral Enable 9 Low Register (BSU1_PEN9L) */
typedef struct {
  __REG16 ENPPG64           : 1;
  __REG16 ENPPG65           : 1;
  __REG16 ENPPG66           : 1;
  __REG16 ENPPG67           : 1;
  __REG16 ENPPG68           : 1;
  __REG16 ENPPG69           : 1;
  __REG16 ENPPG70           : 1;
  __REG16 ENPPG71           : 1;
  __REG16 ENPPG72           : 1;
  __REG16 ENPPG73           : 1;
  __REG16 ENPPG74           : 1;
  __REG16 ENPPG75           : 1;
  __REG16 ENPPG76           : 1;
  __REG16 ENPPG77           : 1;
  __REG16 ENPPG78           : 1;
  __REG16 ENPPG79           : 1;
} __bsu1_pen9l_bits;

/* Peripheral Enable 9 High Register (BSU1_PEN9H) */
typedef struct {
  __REG16 ENPPG80           : 1;
  __REG16 ENPPG81           : 1;
  __REG16 ENPPG82           : 1;
  __REG16 ENPPG83           : 1;
  __REG16 ENPPG84           : 1;
  __REG16 ENPPG85           : 1;
  __REG16 ENPPG86           : 1;
  __REG16 ENPPG87           : 1;
  __REG16 ENPPG88           : 1;
  __REG16 ENPPG89           : 1;
  __REG16 ENPPG90           : 1;
  __REG16 ENPPG91           : 1;
  __REG16 ENPPG92           : 1;
  __REG16 ENPPG93           : 1;
  __REG16 ENPPG94           : 1;
  __REG16 ENPPG95           : 1;
} __bsu1_pen9h_bits;

/* Peripheral Enable 10 Low Register (BSU1_PEN10L) */
typedef struct {
  __REG16 ENPPG96           : 1;
  __REG16 ENPPG97           : 1;
  __REG16 ENPPG98           : 1;
  __REG16 ENPPG99           : 1;
  __REG16 ENPPG100          : 1;
  __REG16 ENPPG101          : 1;
  __REG16 ENPPG102          : 1;
  __REG16 ENPPG103          : 1;
  __REG16 ENPPG104          : 1;
  __REG16 ENPPG105          : 1;
  __REG16 ENPPG106          : 1;
  __REG16 ENPPG107          : 1;
  __REG16 ENPPG108          : 1;
  __REG16 ENPPG109          : 1;
  __REG16 ENPPG110          : 1;
  __REG16 ENPPG111          : 1;
} __bsu1_pen10l_bits;

/* Peripheral Enable 10 High Register (BSU1_PEN10H) */
typedef struct {
  __REG16 ENPPG112          : 1;
  __REG16 ENPPG113          : 1;
  __REG16 ENPPG114          : 1;
  __REG16 ENPPG115          : 1;
  __REG16 ENPPG116          : 1;
  __REG16 ENPPG117          : 1;
  __REG16 ENPPG118          : 1;
  __REG16 ENPPG119          : 1;
  __REG16 ENPPG120          : 1;
  __REG16 ENPPG121          : 1;
  __REG16 ENPPG122          : 1;
  __REG16 ENPPG123          : 1;
  __REG16 ENPPG124          : 1;
  __REG16 ENPPG125          : 1;
  __REG16 ENPPG126          : 1;
  __REG16 ENPPG127          : 1;
} __bsu1_pen10h_bits;

/* Peripheral Enable 11 Low Register (BSU1_PEN11L) */
typedef struct {
  __REG16 ENPPGGRP16        : 1;
  __REG16 ENPPGGRP17        : 1;
  __REG16 ENPPGGRP18        : 1;
  __REG16 ENPPGGRP19        : 1;
  __REG16 ENPPGGRP20        : 1;
  __REG16 ENPPGGRP21        : 1;
  __REG16 ENPPGGRP22        : 1;
  __REG16 ENPPGGRP23        : 1;
  __REG16 ENPPGGRP24        : 1;
  __REG16 ENPPGGRP25        : 1;
  __REG16 ENPPGGRP26        : 1;
  __REG16 ENPPGGRP27        : 1;
  __REG16 ENPPGGRP28        : 1;
  __REG16 ENPPGGRP29        : 1;
  __REG16 ENPPGGRP30        : 1;
  __REG16 ENPPGGRP31        : 1;
} __bsu1_pen11l_bits;

/* Peripheral Enable 11 High Register (BSU1_PEN11H) */
typedef struct {
  __REG16 ENPPGGLC1         : 1;
  __REG16                   :15;
} __bsu1_pen11h_bits;

/* Peripheral Enable 2 Register (BSU3_PEN2) */
typedef struct {
  __REG32 ENRLT0            : 1;
  __REG32 ENRLT1            : 1;
  __REG32 ENRLT2            : 1;
  __REG32 ENRLT3            : 1;
  __REG32 ENRLT4            : 1;
  __REG32 ENRLT5            : 1;
  __REG32 ENRLT6            : 1;
  __REG32 ENRLT7            : 1;
  __REG32 ENRLT8            : 1;
  __REG32 ENRLT9            : 1;
  __REG32 ENRLT10           : 1;
  __REG32 ENRLT11           : 1;
  __REG32 ENRLT12           : 1;
  __REG32 ENRLT13           : 1;
  __REG32 ENRLT14           : 1;
  __REG32 ENRLT15           : 1;
  __REG32 ENRLT16           : 1;
  __REG32 ENRLT17           : 1;
  __REG32 ENRLT18           : 1;
  __REG32 ENRLT19           : 1;
  __REG32 ENRLT20           : 1;
  __REG32 ENRLT21           : 1;
  __REG32 ENRLT22           : 1;
  __REG32 ENRLT23           : 1;
  __REG32 ENRLT24           : 1;
  __REG32 ENRLT25           : 1;
  __REG32 ENRLT26           : 1;
  __REG32 ENRLT27           : 1;
  __REG32 ENRLT28           : 1;
  __REG32 ENRLT29           : 1;
  __REG32 ENRLT30           : 1;
  __REG32 ENRLT31           : 1;
} __bsu3_pen2_bits;

/* Peripheral Enable 4 Register (BSU3_PEN4) */
typedef struct {
  __REG32 ENUDC0            : 1;
  __REG32 ENUDC1            : 1;
  __REG32 ENUDC2            : 1;
  __REG32 ENUDC3            : 1;
  __REG32                   :28;
} __bsu3_pen4_bits;

/* Peripheral Enable 1 Register (BSU4_PEN1) */
typedef struct {
  __REG32 ENETH0            : 1;
  __REG32                   :31;
} __bsu4_pen1_bits;

/* Peripheral Enable 2 Register (BSU4_PEN2) */
typedef struct {
  __REG32 ENMLB0            : 1;
  __REG32                   :31;
} __bsu4_pen2_bits;

/* Peripheral Enable 3 Register (BSU4_PEN3) */
typedef struct {
  __REG32 ENUSB0            : 1;
  __REG32 ENUSB1            : 1;
  __REG32                   :30;
} __bsu4_pen3_bits;

/* Peripheral Enable 4 Register (BSU4_PEN4) */
typedef struct {
  __REG32 ENI2S0            : 1;
  __REG32 ENI2S1            : 1;
  __REG32 ENI2S2            : 1;
  __REG32 ENI2S3            : 1;
  __REG32 ENI2S4            : 1;
  __REG32 ENI2S5            : 1;
  __REG32 ENI2S6            : 1;
  __REG32 ENI2S7            : 1;
  __REG32                   :24;
} __bsu4_pen4_bits;

/* Peripheral Enable 5 Register (BSU4_PEN5) */
typedef struct {
  __REG32 ENFR0             : 1;
  __REG32 ENFR1             : 1;
  __REG32 ENFR2             : 1;
  __REG32 ENFR3             : 1;
  __REG32                   :28;
} __bsu4_pen5_bits;

/* Peripheral Enable 6 Register (BSU4_PEN6) */
typedef struct {
  __REG32 ENCRC0            : 1;
  __REG32 ENCRC1            : 1;
  __REG32                   :30;
} __bsu4_pen6_bits;

/* Peripheral Enable 7 Register (BSU4_PEN7) */
typedef struct {
  __REG32 ENSPI0            : 1;
  __REG32 ENSPI1            : 1;
  __REG32 ENSPI2            : 1;
  __REG32 ENSPI3            : 1;
  __REG32 ENSPI4            : 1;
  __REG32 ENSPI5            : 1;
  __REG32 ENSPI6            : 1;
  __REG32 ENSPI7            : 1;
  __REG32 ENSPI8            : 1;
  __REG32 ENSPI9            : 1;
  __REG32 ENSPI10           : 1;
  __REG32 ENSPI11           : 1;
  __REG32                   :20;
} __bsu4_pen7_bits;

/* Peripheral Enable 8 Register (BSU4_PEN8) */
typedef struct {
  __REG32 ENARH             : 1;
  __REG32                   :31;
} __bsu4_pen8_bits;

/* Peripheral Enable 2 Register (BSU6_PEN2) */
typedef struct {
  __REG32                   : 8;
  __REG32 ENEEFCFG          : 1;
  __REG32                   :23;
} __bsu6_pen2_bits;

/* Peripheral Enable 4 Register (BSU6_PEN4) */
typedef struct {
  __REG32 ENEEFLASHMIR0     : 1;
  __REG32                   :31;
} __bsu6_pen4_bits;

/* Peripheral Enable 12 Register (BSU6_PEN12) */
typedef struct {
  __REG32 ENEEFLASHMIR1     : 1;
  __REG32                   :31;
} __bsu6_pen12_bits;

/* Peripheral Enable 20 Register (BSU6_PEN20) */
typedef struct {
  __REG32 ENEEFLASHMIR2     : 1;
  __REG32                   :31;
} __bsu6_pen20_bits;

/* Peripheral Enable 3 Register (BSU7_PEN3) */
typedef struct {
  __REG32 ENRTC             : 1;
  __REG32                   :31;
} __bsu7_pen3_bits;

/* Peripheral Enable 5 Register (BSU7_PEN5) */
typedef struct {
  __REG32 ENEICU0           : 1;
  __REG32                   :31;
} __bsu7_pen5_bits;

/* Peripheral Enable 7 Register (BSU7_PEN7) */
typedef struct {
  __REG32 ENRETRAMBANK0     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK1     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK2     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK3     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK4     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK5     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK6     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK7     : 1;
  __REG32                   : 3;
} __bsu7_pen7_bits;

/* Peripheral Enable 8 Register (BSU7_PEN8) */
typedef struct {
  __REG32 ENRETRAMBANK8     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK9     : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK10    : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK11    : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK12    : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK13    : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK14    : 1;
  __REG32                   : 3;
  __REG32 ENRETRAMBANK15    : 1;
  __REG32                   : 3;
} __bsu7_pen8_bits;

/* Peripheral Enable 0 Register (BSU8_PEN0) */
typedef struct {
  __REG32 ENHSSPI0          : 1;
  __REG32                   :31;
} __bsu8_pen0_bits;

/* FastMACn Output Enable Register (ETHn_OEN) */
typedef struct {
  __REG8  MDIOOE            : 1;
  __REG8  MIIOE             : 1;
  __REG8                    : 6;
} __ethn_oen_bits;

/* FastMACn Wake On LAN Register (ETHn_WOL) */
typedef struct {
  __REG8  MPM               : 1;
  __REG8                    : 7;
} __ethn_wol_bits;

/* FastMACn Mode Register (ETHn_EMODE) */
typedef struct {
  __REG32 SRST              : 1;
  __REG32 TXGD              : 1;
  __REG32 RXGD              : 1;
  __REG32 FAST              : 1;
  __REG32 DUPLEX            : 1;
  __REG32 PRO               : 1;
  __REG32 UDAF              : 1;
  __REG32 MDAF              : 1;
  __REG32 BDAF              : 1;
  __REG32 PADTX             : 1;
  __REG32 APT               : 1;
  __REG32 PD                : 1;
  __REG32 SWPREQ            : 1;
  __REG32 RTRY              : 1;
  __REG32 TXBRDY            : 1;
  __REG32 MPSUP             : 1;
  __REG32 JUMBO             : 1;
  __REG32                   :15;
} __ethn_emode_bits;

/* FastMACn PTP Mode Register (ETHn_PMODE) */
typedef struct {
  __REG16 PTPEN             : 1;
  __REG16 PTPRDA            : 1;
  __REG16 PTPTXBRDY         : 1;
  __REG16                   : 5;
  __REG16 PTPDOMAINNUMBER   : 8;
} __ethn_pmode_bits;

/* FastMACn Flexi Filter Mode Register (ETHn_FMODE) */
typedef struct {
  __REG16 FFEN              : 1;
  __REG16 FFIE              : 1;
  __REG16                   : 5;
  __REG16 UACCEN            : 1;
  __REG16 CE0               : 1;
  __REG16 CE1               : 1;
  __REG16 CE2               : 1;
  __REG16 CE3               : 1;
  __REG16 CE4               : 1;
  __REG16 CE5               : 1;
  __REG16 CE6               : 1;
  __REG16 CE7               : 1;
} __ethn_fmode_bits;

/* FastMACn Interrupt Enable Register (ETHn_EIE) */
typedef struct {
  __REG32 TXDE              : 1;
  __REG32 RXDE              : 1;
  __REG32 TXBHOE            : 1;
  __REG32 RXBHOE            : 1;
  __REG32 SWPACKE           : 1;
  __REG32 RXPE              : 1;
  __REG32 TXSE              : 1;
  __REG32 RXSE              : 1;
  __REG32 MDEE              : 1;
  __REG32 LFE               : 1;
  __REG32 LUPE              : 1;
  __REG32 MDONEE            : 1;
  __REG32 PHYE              : 1;
  __REG32 LNGE              : 1;
  __REG32 ALNE              : 1;
  __REG32 FCSE              : 1;
  __REG32 LERRE             : 1;
  __REG32 RTRYFE            : 1;
  __REG32 EARLYE            : 1;
  __REG32                   :13;
} __ethn_eie_bits;

/* FastMAC PTP Interrupt Enable Register (ETHn_PIE) */
typedef struct {
  __REG32 PTPTXACKE         : 1;
  __REG32                   :15;
  __REG32 SYNCRXE           : 1;
  __REG32 DRQRXE            : 1;
  __REG32 PDRQRXE           : 1;
  __REG32 PDRPRXE           : 1;
  __REG32                   : 4;
  __REG32 FUPRXE            : 1;
  __REG32 DRPRXE            : 1;
  __REG32 PDRPFUPRXE        : 1;
  __REG32 ANNRXE            : 1;
  __REG32 SIGNRXE           : 1;
  __REG32 MGMTRXE           : 1;
  __REG32                   : 2;
} __ethn_pie_bits;

/* FastMACn Flexi Filter Interrupt on Match Enable Register (ETHn_FMIE) */
/* FastMACn Flexi Filter Interrupt On Reply Enable Register (ETHn_ARIE) */
typedef struct {
  __REG8  IE0               : 1;
  __REG8  IE1               : 1;
  __REG8  IE2               : 1;
  __REG8  IE3               : 1;
  __REG8  IE4               : 1;
  __REG8  IE5               : 1;
  __REG8  IE6               : 1;
  __REG8  IE7               : 1;
} __ethn_fmie_bits;

/* FastMACn Interrupt Request Register (ETHn_EIR) */
typedef struct {
  __REG32 TXD               : 1;
  __REG32 RXD               : 1;
  __REG32 TXBHO             : 1;
  __REG32 RXBHO             : 1;
  __REG32 SWPACK            : 1;
  __REG32 RXP               : 1;
  __REG32 TXS               : 1;
  __REG32 RXS               : 1;
  __REG32 MDE               : 1;
  __REG32 LF                : 1;
  __REG32 LUP               : 1;
  __REG32 MDONE             : 1;
  __REG32 PHY               : 1;
  __REG32 LNG               : 1;
  __REG32 ALN               : 1;
  __REG32 FCS               : 1;
  __REG32 LERR              : 1;
  __REG32 RTRYF             : 1;
  __REG32 EARLY             : 1;
  __REG32                   :13;
} __ethn_eir_bits;

/* FastMACn PTP Interrupt Request Register (ETHn_PIR) */
typedef struct {
  __REG32 PTPTXACK          : 1;
  __REG32                   : 7;
  __REG32 SYNCRX            : 1;
  __REG32 DRQRX             : 1;
  __REG32 PDRQRX            : 1;
  __REG32 PDRPRX            : 1;
  __REG32                   :12;
  __REG32 FUPRX             : 1;
  __REG32 DRPRX             : 1;
  __REG32 PDRPFUPRX         : 1;
  __REG32 ANNRX             : 1;
  __REG32 SIGNRX            : 1;
  __REG32 MGMTRX            : 1;
  __REG32                   : 2;
} __ethn_pir_bits;

/* FastMACn Flexi Filter Interrupt on Match Request Register (ETHn_FMIR) */
/* FastMACn Flexi Filter Interrupt on Reply Request Register (ETHn_ARIR) */
typedef struct {
  __REG8  IR0               : 1;
  __REG8  IR1               : 1;
  __REG8  IR2               : 1;
  __REG8  IR3               : 1;
  __REG8  IR4               : 1;
  __REG8  IR5               : 1;
  __REG8  IR6               : 1;
  __REG8  IR7               : 1;
} __ethn_fmir_bits;

/* FastMACn Flexi Filter Match Interrupt Overrun Register (ETHn_FMIO) */
/* FastMACn Flexi Filter Reply Interrupt Overrun Register (ETHn_ARIO) */
typedef struct {
  __REG8  IO0               : 1;
  __REG8  IO1               : 1;
  __REG8  IO2               : 1;
  __REG8  IO3               : 1;
  __REG8  IO4               : 1;
  __REG8  IO5               : 1;
  __REG8  IO6               : 1;
  __REG8  IO7               : 1;
} __ethn_fmio_bits;

/* FastMACn Interrupt Clear Register (ETHn_EIC) */
typedef struct {
  __REG32 TXDC              : 1;
  __REG32 RXDC              : 1;
  __REG32 TXBHOC            : 1;
  __REG32 RXBHOC            : 1;
  __REG32 SWPACKC           : 1;
  __REG32 RXPC              : 1;
  __REG32 TXSC              : 1;
  __REG32 RXSC              : 1;
  __REG32 MDEC              : 1;
  __REG32 LFC               : 1;
  __REG32 LUPC              : 1;
  __REG32 MDONEC            : 1;
  __REG32 PHYC              : 1;
  __REG32 LNGC              : 1;
  __REG32 ALNC              : 1;
  __REG32 FCSC              : 1;
  __REG32 LERRC             : 1;
  __REG32 RTRYFC            : 1;
  __REG32 EARLYC            : 1;
  __REG32                   :13;
} __ethn_eic_bits;

/* FastMACn PTP Interrupt Clear Register (ETHn_PIC) */
typedef struct {
  __REG32 PTPTXACKC         : 1;
  __REG32                   :15;
  __REG32 SYNCRX            : 1;
  __REG32 DRQRX             : 1;
  __REG32 PDRQRX            : 1;
  __REG32 PDRPRX            : 1;
  __REG32                   : 4;
  __REG32 FUPRX             : 1;
  __REG32 DRPRX             : 1;
  __REG32 PDRPFUPRX         : 1;
  __REG32 ANNRX             : 1;
  __REG32 SIGNRX            : 1;
  __REG32 MGMTRX            : 1;
  __REG32                   : 2;
} __ethn_pic_bits;

/* FastMACn Flexi Filter Interrupt on Match Clear Register (ETHn_FMICn) */
/* FastMACn Flexi Filter Interrupt On Reply Clear Register (ETHn_ARICn) */
typedef struct {
  __REG8  IC0               : 1;
  __REG8  IC1               : 1;
  __REG8  IC2               : 1;
  __REG8  IC3               : 1;
  __REG8  IC4               : 1;
  __REG8  IC5               : 1;
  __REG8  IC6               : 1;
  __REG8  IC7               : 1;
} __ethn_fmic_bits;

/* FastMACn Flexi Filter Match Interrupt Overrun Clear Register (ETHn_FMOCn) */
/* FastMACn Flexi Filter Reply Interrupt Overrun Clear Register (ETHn_AROCn) */
typedef struct {
  __REG8  OC0               : 1;
  __REG8  OC1               : 1;
  __REG8  OC2               : 1;
  __REG8  OC3               : 1;
  __REG8  OC4               : 1;
  __REG8  OC5               : 1;
  __REG8  OC6               : 1;
  __REG8  OC7               : 1;
} __ethn_fmoc_bits;

/* FastMAC High Priority Frame Interrupt Enable Register (ETHn_HIE) */
typedef struct {
  __REG8  HPBHOE            : 1;
  __REG8                    : 7;
} __ethn_hie_bits;

/* FastMAC High Priority Frame Interrupt Request Register (ETHn_HIR) */
typedef struct {
  __REG8  HPBHO             : 1;
  __REG8                    : 7;
} __ethn_hir_bits;

/* FastMACn High Priority Frame Interrupt Clear Register (ETHn_HIC) */
typedef struct {
  __REG8  HPBHOC            : 1;
  __REG8                    : 7;
} __ethn_hic_bits;

/* FastMACn InterFrameSpacingPart2 Register (ETHn_IFS) */
typedef struct {
  __REG8  FS                : 3;
  __REG8                    : 5;
} __ethn_ifs_bits;

/* FastMACn MDC Clock Division Factor Register (ETHn_MDCCKDIV) */
typedef struct {
  __REG16 MDCKDIV           : 8;
  __REG16                   : 8;
} __ethn_mdcckdiv_bits;

/* FastMACn MDIO Control Register (ETHn_MDCTRL) */
typedef struct {
  __REG16 POLLEN            : 1;
  __REG16 RDNWR             : 1;
  __REG16 TRIG              : 1;
  __REG16 PHYAD             : 5;
  __REG16 REGAD             : 5;
  __REG16                   : 3;
} __ethn_mdctrl_bits;

/* FastMACn Flexi Filter Channel 0~7 String Length Register (ETHn_FFSLEN0~7) */
typedef struct {
  __REG16 FMFFSLEN0         :11;
  __REG16                   : 5;
} __ethn_ffslen_bits;

/* FastMACn Flexi Filter Channel 0~7 Configuration Register (ETHn_FCCR0~7) */
typedef struct {
  __REG8  CFG               : 3;
  __REG8  MASKEN            : 1;
  __REG8  MASKPTR           : 3;
  __REG8                    : 1;
} __ethn_fccr_bits;

/* FastMACn AHB Error Control Register (ETHn_AHBERCTR) */
typedef struct {
  __REG8  WR                : 1;
  __REG8                    : 7;
} __ethn_ahberctr_bits;

/* FastMACn NMI Register (ETHn_NMI) */
typedef struct {
  __REG32 AHBERRF           : 1;
  __REG32 WAKEUP            : 1;
  __REG32                   : 6;
  __REG32 AHBERRC           : 1;
  __REG32 WAKEUPC           : 1;
  __REG32                   :22;
} __ethn_nmi_bits;

/* ETHERNETRAM Configuration and Status Register (ERCFGn_CSR) */
typedef struct {
  __REG32 CEIEN             : 1;
  __REG32                   : 7;
  __REG32 CEIF              : 1;
  __REG32 LCK               : 1;
  __REG32                   : 6;
  __REG32 CEIC              : 1;
  __REG32                   : 7;
  __REG32 RAWC1             : 2;
  __REG32 WAWC1             : 2;
  __REG32 RAWC2             : 2;
  __REG32 WAWC2             : 2;
} __ercfgn_csr_bits;

/* ETHERNETRAM Configuration and Status Register (ERCFGn_CSR) */
typedef struct {
  __REG32 MSK               : 7;
  __REG32                   :25;
} __ercfgn_errmskr1_bits;

/* ETHERNETRAM ECC Enable Register (ERCFGn_ECCEN) */
typedef struct {
  __REG32 ECCEN             : 1;
  __REG32                   :31;
} __ercfgn_eccen_bits;

/* Count Control Register 0 (UDCn_CC0) */
typedef struct {
  __REG16 CGE               : 2;
  __REG16 CGSC              : 1;
  __REG16                   : 1;
  __REG16 RLDE              : 1;
  __REG16 UCRE              : 1;
  __REG16                   : 2;
  __REG16 CES               : 2;
  __REG16 CMS               : 2;
  __REG16 CLKS              : 1;
  __REG16 CFIE              : 1;
  __REG16 CDCF              : 1;
  __REG16 M32E              : 1;
} __udcn_cc0_bits;

/* Count Control Register 1 (UDCn_CC1) */
typedef struct {
  __REG16 CGE               : 2;
  __REG16 CGSC              : 1;
  __REG16                   : 1;
  __REG16 RLDE              : 1;
  __REG16 UCRE              : 1;
  __REG16                   : 2;
  __REG16 CES               : 2;
  __REG16 CMS               : 2;
  __REG16 CLKS              : 1;
  __REG16 CFIE              : 1;
  __REG16 CDCF              : 1;
  __REG16                   : 1;
} __udcn_cc1_bits;

/* Extended Count Control Register n (UDCn_ECCn) */
typedef struct {
  __REG16 CDCFCLR           : 1;
  __REG16 UDFFCLR           : 1;
  __REG16 OVFFCLR           : 1;
  __REG16 CMPFCLR           : 1;
  __REG16 UDCLR             : 1;
  __REG16 CTUT              : 1;
  __REG16                   :10;
} __udcn_ecc_bits;

/* Count Status Register n (UDCn_CSn) */
typedef struct {
  __REG16 UDF               : 2;
  __REG16 UDFF              : 1;
  __REG16 OVFF              : 1;
  __REG16 CMPF              : 1;
  __REG16 UDIE              : 1;
  __REG16 CITE              : 1;
  __REG16                   : 1;
  __REG16 CSTR              : 1;
  __REG16                   : 7;
} __udcn_cs_bits;

/* Up/Down Counter Register (UDCn_CR) */
typedef struct {
  __REG32 UDCRL             :16;
  __REG32 UDCRH             :16;
} __udcn_cr_bits;

/* Up/Down Reload/Compare Register (UDCn_RC) */
typedef struct {
  __REG32 UDRCL             :16;
  __REG32 UDRCH             :16;
} __udcn_rc_bits;

/* Count Toggle Register n (UDCn_TGLn) */
typedef struct {
  __REG16 OUTL              : 1;
  __REG16 OUTE              : 1;
  __REG16                   : 6;
  __REG16 CDTE              : 1;
  __REG16 UDTE              : 1;
  __REG16 CMTE              : 1;
  __REG16                   : 5;
} __udcn_tgl_bits;

/* Count Debug Register (UDCn_DBG) */
typedef struct {
  __REG8  DBGEN             : 1;
  __REG8                    : 7;
} __udcn_dbg_bits;

/* RICFG8G0_HSSPI0MSTART */
typedef struct {
  __REG16 RESSEL             : 3;
  __REG16                    :13;
} __ricfg8g0_hsspi0mstart_bits;

/* GFXGCTR_LOCKSTATUS */
typedef struct {
  __REG32 LOCKSTATUS         : 1;
  __REG32                    :31;
} __gfxgctr_lockstatus_bits;

/* GFXGCTR_INTSTATUS0 */
typedef struct {
  __REG32 INTSTATUS0         :31;
  __REG32                    : 1;
} __gfxgctr_intstatus0_bits;

/* GFXGCTR_INTSTATUS1 */
typedef struct {
  __REG32 INTSTATUS1         : 7;
  __REG32                    :25;
} __gfxgctr_intstatus1_bits;

/* GFXGCTR_INTENABLE0 */
typedef struct {
  __REG32 INTENABLE0         :31;
  __REG32                    : 1;
} __gfxgctr_intenable0_bits;

/* GFXGCTR_INTENABLE1 */
typedef struct {
  __REG32 INTENABLE1         : 7;
  __REG32                    :25;
} __gfxgctr_intenable1_bits;

/* GFXGCTR_INTCLEAR0 */
typedef struct {
  __REG32 INTCLEAR0          :28;
  __REG32                    : 4;
} __gfxgctr_intclear0_bits;

/* GFXGCTR_INTCLEAR1 */
typedef struct {
  __REG32 INTCLEAR1          : 7;
  __REG32                    :25;
} __gfxgctr_intclear1_bits;

/* GFXGCTR_INTPRESET0 */
typedef struct {
  __REG32 INTPRESET0         :28;
  __REG32                    : 4;
} __gfxgctr_intpreset0_bits;

/* GFXGCTR_INTPRESET1 */
typedef struct {
  __REG32 INTPRESET1         : 7;
  __REG32                    :25;
} __gfxgctr_intpreset1_bits;

/* GFXGCTR_INTMAP0 */
typedef struct {
  __REG32 INTMAP0	           :31;
  __REG32                    : 1;
} __gfxgctr_intmap0_bits;

/* GFXGCTR_INTMAP1 */
typedef struct {
  __REG32 INTMAP1	           : 7;
  __REG32                    :25;
} __gfxgctr_intmap1_bits;

/* GFXGCTR_NMISTATUS */
typedef struct {
  __REG32 NMISTATUS          : 1;
  __REG32                    :31;
} __gfxgctr_nmistatus_bits;

/* GFXGCTR_NMICLEAR */
typedef struct {
  __REG32 NMICLEAR           : 1;
  __REG32                    :31;
} __gfxgctr_nmiclear_bits;

/* GFXGCTR_NMIPRESET */
typedef struct {
  __REG32 NMIPRESET          : 1;
  __REG32                    :31;
} __gfxgctr_nmipreset_bits;

/* GFXGCTR_CSINTSTATUS0 */
typedef struct {
  __REG32 CSINTSTATUS0       :31;
  __REG32                    : 1;
} __gfxgctr_csintstatus0_bits;

/* GFXGCTR_CSINTSTATUS1 */
typedef struct {
  __REG32 CSINTSTATUS1       : 7;
  __REG32                    :25;
} __gfxgctr_csintstatus1_bits;

/* GFXGCTR_CSINTSTATUS0 */
typedef struct {
  __REG32 CSINTENABLE0       :31;
  __REG32                    : 1;
} __gfxgctr_csintenable0_bits;

/* GFXGCTR_CSINTSTATUS1 */
typedef struct {
  __REG32 CSINTENABLE1       : 7;
  __REG32                    :25;
} __gfxgctr_csintenable1_bits;

/* GFXGCTR_CSINTCLEAR0 */
typedef struct {
  __REG32 CSINTCLEAR0        :28;
  __REG32                    : 4;
} __gfxgctr_csintclear0_bits;

/* GFXGCTR_CSINTCLEAR1 */
typedef struct {
  __REG32 CSINTCLEAR1        : 7;
  __REG32                    :25;
} __gfxgctr_csintclear1_bits;

/* GFXGCTR_CSINTPRESET0 */
typedef struct {
  __REG32 CSINTPRESET0       :28;
  __REG32                    : 4;
} __gfxgctr_csintpreset0_bits;

/* GFXGCTR_CSINTPRESET1 */
typedef struct {
  __REG32 CSINTPRESET1       : 7;
  __REG32                    :25;
} __gfxgctr_csintpreset1_bits;

/* GFXGCTR_SWRESET */
typedef struct {
  __REG32 VRAM_RSTN		       : 1;
  __REG32 PENG_RSTN		       : 1;
  __REG32 CSEQU_RSTN	       : 1;
  __REG32 SIG_RSTN		       : 1;
  __REG32 DISP_RSTN		       : 1;
  __REG32                    :27;
} __gfxgctr_swreset_bits;

/* GFXGCTR_CLOCKADJUST */
typedef struct {
  __REG32 DIV_PIX  		       : 8;
  __REG32 			  		       : 4;
  __REG32 BYPASS_CLK	       : 2;
  __REG32 BYPASS_X2_CLK      : 1;
  __REG32 			  		       : 1;
  __REG32 SHIFT_PIX		       : 8;
  __REG32                    : 7;
  __REG32 INV_CLK			       : 1;
} __gfxgctr_clockadjust_bits;

/* GFXCMD_STATUS */
typedef struct {
  __REG32 FIFOSPACE		       :17;
  __REG32 			  		       : 7;
  __REG32 FIFOEMPTY 	       : 1;
  __REG32 FIFOFULL		       : 1;
  __REG32 FIFOWMSTATE	       : 1;
  __REG32 			  		       : 2;
  __REG32 WATCHDOG		       : 1;
  __REG32 IDLE				       : 1;
  __REG32 ERROR				       : 1;
} __gfxcmd_status_bits;

/* GFXCMD_CONTROL */
typedef struct {
  __REG32 			  		       :31;
  __REG32 CLEAR				       : 1;
} __gfxcmd_control_bits;

/* GFXCMD_BUFFERADDRESS */
typedef struct {
  __REG32 LOCAL				       : 1;
  __REG32 			  		       : 1;
  __REG32 ADDR				       :30;
} __gfxcmd_bufferaddress_bits;

/* GFXCMD_BUFFERSIZE */
typedef struct {
  __REG32 SIZE				       :16;
  __REG32 			  		       :16;
} __gfxcmd_buffersize_bits;

/* GFXCMD_WATERMARKCONTROL */
typedef struct {
  __REG32 LOWWM				       :16;
  __REG32 HIGHWM  		       :16;
} __gfxcmd_watermarkcontrol_bits;

/* GFXTCON_DIR_SSQCNTS */
typedef struct {
  __REG32 SSQCNTS_SEQY			 :15;
  __REG32 									 : 1;
  __REG32 SSQCNTS_SEQX			 :15;
  __REG32 SSQCNTS_OUT				 : 1;
} __gfxtcon_dir_ssqcnts_bits;

/* GFXTCON_DIR_SWRESET */
typedef struct {
  __REG32 SWRESET						 : 1;
  __REG32 									 :31;
} __gfxtcon_dir_swreset_bits;

/* GFXTCON_DIR_SPG0POSON */
typedef struct {
  __REG32 SPGPSON_Y0					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X0					:15;
  __REG32 SPGPSON_TOGGLE0			: 1;
} __gfxtcon_dir_spg0poson_bits;

/* GFXTCON_DIR_SPG0MASKON */
typedef struct {
  __REG32 SPGMKON0						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg0maskon_bits;

/* GFXTCON_DIR_SPG0POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y0					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X0					:15;
  __REG32 SPGPSOFF_TOGGLE0		: 1;
} __gfxtcon_dir_spg0posoff_bits;

/* GFXTCON_DIR_SPG0MASKOFF */
typedef struct {
  __REG32 SPGMKON0						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg0maskoff_bits;

/* GFXTCON_DIR_SPG1POSON */
typedef struct {
  __REG32 SPGPSON_Y1					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X1					:15;
  __REG32 SPGPSON_TOGGLE1			: 1;
} __gfxtcon_dir_spg1poson_bits;

/* GFXTCON_DIR_SPG1MASKON */
typedef struct {
  __REG32 SPGMKON1						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg1maskon_bits;

/* GFXTCON_DIR_SPG1POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y1					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X1					:15;
  __REG32 SPGPSOFF_TOGGLE1		: 1;
} __gfxtcon_dir_spg1posoff_bits;

/* GFXTCON_DIR_SPG1MASKOFF */
typedef struct {
  __REG32 SPGMKON1						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg1maskoff_bits;

/* GFXTCON_DIR_SPG2POSON */
typedef struct {
  __REG32 SPGPSON_Y2					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X2					:15;
  __REG32 SPGPSON_TOGGLE2			: 1;
} __gfxtcon_dir_spg2poson_bits;

/* GFXTCON_DIR_SPG2MASKON */
typedef struct {
  __REG32 SPGMKON2						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg2maskon_bits;

/* GFXTCON_DIR_SPG2POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y2					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X2					:15;
  __REG32 SPGPSOFF_TOGGLE2		: 1;
} __gfxtcon_dir_spg2posoff_bits;

/* GFXTCON_DIR_SPG2MASKOFF */
typedef struct {
  __REG32 SPGMKON2						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg2maskoff_bits;

/* GFXTCON_DIR_SPG3POSON */
typedef struct {
  __REG32 SPGPSON_Y3					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X3					:15;
  __REG32 SPGPSON_TOGGLE3			: 1;
} __gfxtcon_dir_spg3poson_bits;

/* GFXTCON_DIR_SPG3MASKON */
typedef struct {
  __REG32 SPGMKON3						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg3maskon_bits;

/* GFXTCON_DIR_SPG3POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y3					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X3					:15;
  __REG32 SPGPSOFF_TOGGLE3		: 1;
} __gfxtcon_dir_spg3posoff_bits;

/* GFXTCON_DIR_SPG3MASKOFF */
typedef struct {
  __REG32 SPGMKON3						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg3maskoff_bits;

/* GFXTCON_DIR_SPG4POSON */
typedef struct {
  __REG32 SPGPSON_Y4					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X4					:15;
  __REG32 SPGPSON_TOGGLE4			: 1;
} __gfxtcon_dir_spg4poson_bits;

/* GFXTCON_DIR_SPG4MASKON */
typedef struct {
  __REG32 SPGMKON4						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg4maskon_bits;

/* GFXTCON_DIR_SPG4POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y4					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X4					:15;
  __REG32 SPGPSOFF_TOGGLE4		: 1;
} __gfxtcon_dir_spg4posoff_bits;

/* GFXTCON_DIR_SPG4MASKOFF */
typedef struct {
  __REG32 SPGMKON4						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg4maskoff_bits;

/* GFXTCON_DIR_SPG5POSON */
typedef struct {
  __REG32 SPGPSON_Y5					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X5					:15;
  __REG32 SPGPSON_TOGGLE5			: 1;
} __gfxtcon_dir_spg5poson_bits;

/* GFXTCON_DIR_SPG5MASKON */
typedef struct {
  __REG32 SPGMKON5						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg5maskon_bits;

/* GFXTCON_DIR_SPG5POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y5					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X5					:15;
  __REG32 SPGPSOFF_TOGGLE5		: 1;
} __gfxtcon_dir_spg5posoff_bits;

/* GFXTCON_DIR_SPG5MASKOFF */
typedef struct {
  __REG32 SPGMKON5						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg5maskoff_bits;

/* GFXTCON_DIR_SPG6POSON */
typedef struct {
  __REG32 SPGPSON_Y6					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X6					:15;
  __REG32 SPGPSON_TOGGLE6			: 1;
} __gfxtcon_dir_spg6poson_bits;

/* GFXTCON_DIR_SPG6MASKON */
typedef struct {
  __REG32 SPGMKON6						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg6maskon_bits;

/* GFXTCON_DIR_SPG6POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y6					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X6					:15;
  __REG32 SPGPSOFF_TOGGLE6		: 1;
} __gfxtcon_dir_spg6posoff_bits;

/* GFXTCON_DIR_SPG6MASKOFF */
typedef struct {
  __REG32 SPGMKON6						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg6maskoff_bits;

/* GFXTCON_DIR_SPG7POSON */
typedef struct {
  __REG32 SPGPSON_Y7					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X7					:15;
  __REG32 SPGPSON_TOGGLE7			: 1;
} __gfxtcon_dir_spg7poson_bits;

/* GFXTCON_DIR_SPG7MASKON */
typedef struct {
  __REG32 SPGMKON7						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg7maskon_bits;

/* GFXTCON_DIR_SPG7POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y7					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X7					:15;
  __REG32 SPGPSOFF_TOGGLE7		: 1;
} __gfxtcon_dir_spg7posoff_bits;

/* GFXTCON_DIR_SPG7MASKOFF */
typedef struct {
  __REG32 SPGMKON7						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg7maskoff_bits;

/* GFXTCON_DIR_SPG8POSON */
typedef struct {
  __REG32 SPGPSON_Y8					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X8					:15;
  __REG32 SPGPSON_TOGGLE8			: 1;
} __gfxtcon_dir_spg8poson_bits;

/* GFXTCON_DIR_SPG8MASKON */
typedef struct {
  __REG32 SPGMKON8						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg8maskon_bits;

/* GFXTCON_DIR_SPG8POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y8					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X8					:15;
  __REG32 SPGPSOFF_TOGGLE8		: 1;
} __gfxtcon_dir_spg8posoff_bits;

/* GFXTCON_DIR_SPG8MASKOFF */
typedef struct {
  __REG32 SPGMKON8						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg8maskoff_bits;

/* GFXTCON_DIR_SPG9POSON */
typedef struct {
  __REG32 SPGPSON_Y9					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X9					:15;
  __REG32 SPGPSON_TOGGLE9			: 1;
} __gfxtcon_dir_spg9poson_bits;

/* GFXTCON_DIR_SPG9MASKON */
typedef struct {
  __REG32 SPGMKON9						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg9maskon_bits;

/* GFXTCON_DIR_SPG9POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y9					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X9					:15;
  __REG32 SPGPSOFF_TOGGLE9		: 1;
} __gfxtcon_dir_spg9posoff_bits;

/* GFXTCON_DIR_SPG9MASKOFF */
typedef struct {
  __REG32 SPGMKON9						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg9maskoff_bits;

/* GFXTCON_DIR_SPG10POSON */
typedef struct {
  __REG32 SPGPSON_Y10					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X10					:15;
  __REG32 SPGPSON_TOGGLE10		: 1;
} __gfxtcon_dir_spg10poson_bits;

/* GFXTCON_DIR_SPG10MASKON */
typedef struct {
  __REG32 SPGMKON10						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg10maskon_bits;

/* GFXTCON_DIR_SPG10POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y10				:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X10				:15;
  __REG32 SPGPSOFF_TOGGLE10		: 1;
} __gfxtcon_dir_spg10posoff_bits;

/* GFXTCON_DIR_SPG10MASKOFF */
typedef struct {
  __REG32 SPGMKON10						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg10maskoff_bits;

/* GFXTCON_DIR_SPG11POSON */
typedef struct {
  __REG32 SPGPSON_Y11					:15;
  __REG32 									 	: 1;
  __REG32 SPGPSON_X11					:15;
  __REG32 SPGPSON_TOGGLE11		: 1;
} __gfxtcon_dir_spg11poson_bits;

/* GFXTCON_DIR_SPG11MASKON */
typedef struct {
  __REG32 SPGMKON11						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg11maskon_bits;

/* GFXTCON_DIR_SPG11POSOFF */
typedef struct {
  __REG32 SPGPSOFF_Y11				:15;
  __REG32 									 	: 1;
  __REG32 SPGPSOFF_X11				:15;
  __REG32 SPGPSOFF_TOGGLE11		: 1;
} __gfxtcon_dir_spg11posoff_bits;

/* GFXTCON_DIR_SPG11MASKOFF */
typedef struct {
  __REG32 SPGMKON11						:31;
  __REG32 									 	: 1;
} __gfxtcon_dir_spg11maskoff_bits;

/* GFXTCON_DIR_SSQCYCLE */
typedef struct {
  __REG32 SSQCYCLE						: 6;
  __REG32 									 	:26;
} __gfxtcon_dir_ssqcycle_bits;

/* GFXTCON_DIR_SMXxSIGS */
typedef struct {
  __REG32 SMX0SIGS_S0					: 3;
  __REG32 SMX0SIGS_S1					: 3;
  __REG32 SMX0SIGS_S2					: 3;
  __REG32 SMX0SIGS_S3					: 3;
  __REG32 SMX0SIGS_S4					: 3;
  __REG32 									 	:17;
} __gfxtcon_dir_smxsigs_bits;

/* GFXTCON_DIR_SSWITCH */
typedef struct {
  __REG32 SSWITCH							:13;
  __REG32 INVCTREN						: 1;
  __REG32 										: 2;
  __REG32 ENOPTCLK						:13;
  __REG32 									 	: 3;
} __gfxtcon_dir_sswitch_bits;

/* GFXTCON_RBM_CTRL */
typedef struct {
  __REG32 BYPASS							: 1;
  __REG32 IFCTYPE							: 2;
  __REG32 BITPERCOL						: 1;
  __REG32 SWAPODDEVENBIT			: 1;
  __REG32 BITORDER						: 1;
  __REG32 									 	: 2;
  __REG32 COLORDER						: 3;
  __REG32 									 	:21;
} __gfxtcon_rbm_ctrl_bits;

/* GFXTCON_DIR_PIN0_CTRL */
typedef struct {
  __REG32 BOOST0							: 2;
  __REG32 		  							: 2;
  __REG32 MODE0								: 1;
  __REG32 POLARITY0						: 1;
  __REG32 NPOLARITY0					: 1;
  __REG32 INOUT0							: 1;
  __REG32 										: 5;
  __REG32 DELAY0							: 1;
  __REG32 NDELAY0							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL0						: 2;
  __REG32 NCHANSEL0						: 2;
  __REG32 OPTCLKEN0						: 1;
  __REG32 NOPTCLKEN0					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin0_ctrl_bits;

/* GFXTCON_DIR_PIN1_CTRL */
typedef struct {
  __REG32 BOOST1							: 2;
  __REG32 		  							: 2;
  __REG32 MODE1								: 1;
  __REG32 POLARITY1						: 1;
  __REG32 NPOLARITY1					: 1;
  __REG32 INOUT1							: 1;
  __REG32 										: 5;
  __REG32 DELAY1							: 1;
  __REG32 NDELAY1							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL1						: 2;
  __REG32 NCHANSEL1						: 2;
  __REG32 OPTCLKEN1						: 1;
  __REG32 NOPTCLKEN1					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin1_ctrl_bits;

/* GFXTCON_DIR_PIN2_CTRL */
typedef struct {
  __REG32 BOOST2							: 2;
  __REG32 		  							: 2;
  __REG32 MODE2								: 1;
  __REG32 POLARITY2						: 1;
  __REG32 NPOLARITY2					: 1;
  __REG32 INOUT2							: 1;
  __REG32 										: 5;
  __REG32 DELAY2							: 1;
  __REG32 NDELAY2							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL2						: 2;
  __REG32 NCHANSEL2						: 2;
  __REG32 OPTCLKEN2						: 1;
  __REG32 NOPTCLKEN2					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin2_ctrl_bits;

/* GFXTCON_DIR_PIN3_CTRL */
typedef struct {
  __REG32 BOOST3							: 2;
  __REG32 		  							: 2;
  __REG32 MODE3								: 1;
  __REG32 POLARITY3						: 1;
  __REG32 NPOLARITY3					: 1;
  __REG32 INOUT3							: 1;
  __REG32 										: 5;
  __REG32 DELAY3							: 1;
  __REG32 NDELAY3							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL3						: 2;
  __REG32 NCHANSEL3						: 2;
  __REG32 OPTCLKEN3						: 1;
  __REG32 NOPTCLKEN3					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin3_ctrl_bits;

/* GFXTCON_DIR_PIN4_CTRL */
typedef struct {
  __REG32 BOOST4							: 2;
  __REG32 		  							: 2;
  __REG32 MODE4								: 1;
  __REG32 POLARITY4						: 1;
  __REG32 NPOLARITY4					: 1;
  __REG32 INOUT4							: 1;
  __REG32 										: 5;
  __REG32 DELAY4							: 1;
  __REG32 NDELAY4							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL4						: 2;
  __REG32 NCHANSEL4						: 2;
  __REG32 OPTCLKEN4						: 1;
  __REG32 NOPTCLKEN4					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin4_ctrl_bits;

/* GFXTCON_DIR_PIN5_CTRL */
typedef struct {
  __REG32 BOOST5							: 2;
  __REG32 		  							: 2;
  __REG32 MODE5								: 1;
  __REG32 POLARITY5						: 1;
  __REG32 NPOLARITY5					: 1;
  __REG32 INOUT5							: 1;
  __REG32 										: 5;
  __REG32 DELAY5							: 1;
  __REG32 NDELAY5							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL5						: 2;
  __REG32 NCHANSEL5						: 2;
  __REG32 OPTCLKEN5						: 1;
  __REG32 NOPTCLKEN5					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin5_ctrl_bits;

/* GFXTCON_DIR_PIN6_CTRL */
typedef struct {
  __REG32 BOOST6							: 2;
  __REG32 		  							: 2;
  __REG32 MODE6								: 1;
  __REG32 POLARITY6						: 1;
  __REG32 NPOLARITY6					: 1;
  __REG32 INOUT6							: 1;
  __REG32 										: 5;
  __REG32 DELAY6							: 1;
  __REG32 NDELAY6							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL6						: 2;
  __REG32 NCHANSEL6						: 2;
  __REG32 OPTCLKEN6						: 1;
  __REG32 NOPTCLKEN6					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin6_ctrl_bits;

/* GFXTCON_DIR_PIN7_CTRL */
typedef struct {
  __REG32 BOOST7							: 2;
  __REG32 		  							: 2;
  __REG32 MODE7								: 1;
  __REG32 POLARITY7						: 1;
  __REG32 NPOLARITY7					: 1;
  __REG32 INOUT7							: 1;
  __REG32 										: 5;
  __REG32 DELAY7							: 1;
  __REG32 NDELAY7							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL7						: 2;
  __REG32 NCHANSEL7						: 2;
  __REG32 OPTCLKEN7						: 1;
  __REG32 NOPTCLKEN7					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin7_ctrl_bits;

/* GFXTCON_DIR_PIN8_CTRL */
typedef struct {
  __REG32 BOOST8							: 2;
  __REG32 		  							: 2;
  __REG32 MODE8								: 1;
  __REG32 POLARITY8						: 1;
  __REG32 NPOLARITY8					: 1;
  __REG32 INOUT8							: 1;
  __REG32 										: 5;
  __REG32 DELAY8							: 1;
  __REG32 NDELAY8							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL8						: 2;
  __REG32 NCHANSEL8						: 2;
  __REG32 OPTCLKEN8						: 1;
  __REG32 NOPTCLKEN8					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin8_ctrl_bits;

/* GFXTCON_DIR_PIN9_CTRL */
typedef struct {
  __REG32 BOOST9							: 2;
  __REG32 		  							: 2;
  __REG32 MODE9								: 1;
  __REG32 POLARITY9						: 1;
  __REG32 NPOLARITY9					: 1;
  __REG32 INOUT9							: 1;
  __REG32 										: 5;
  __REG32 DELAY9							: 1;
  __REG32 NDELAY9							: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL9						: 2;
  __REG32 NCHANSEL9						: 2;
  __REG32 OPTCLKEN9						: 1;
  __REG32 NOPTCLKEN9					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin9_ctrl_bits;

/* GFXTCON_DIR_PIN10_CTRL */
typedef struct {
  __REG32 BOOST10							: 2;
  __REG32 		  							: 2;
  __REG32 MODE10							: 1;
  __REG32 POLARITY10					: 1;
  __REG32 NPOLARITY10					: 1;
  __REG32 INOUT10							: 1;
  __REG32 										: 5;
  __REG32 DELAY10							: 1;
  __REG32 NDELAY10						: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL10						: 2;
  __REG32 NCHANSEL10					: 2;
  __REG32 OPTCLKEN10					: 1;
  __REG32 NOPTCLKEN10					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin10_ctrl_bits;

/* GFXTCON_DIR_PIN11_CTRL */
typedef struct {
  __REG32 BOOST11							: 2;
  __REG32 		  							: 2;
  __REG32 MODE11							: 1;
  __REG32 POLARITY11					: 1;
  __REG32 NPOLARITY11					: 1;
  __REG32 INOUT11							: 1;
  __REG32 										: 5;
  __REG32 DELAY11							: 1;
  __REG32 NDELAY11						: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL11 					: 2;
  __REG32 NCHANSEL11					: 2;
  __REG32 OPTCLKEN11					: 1;
  __REG32 NOPTCLKEN11					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin11_ctrl_bits;

/* GFXTCON_DIR_PIN12_CTRL */
typedef struct {
  __REG32 BOOST12							: 2;
  __REG32 		  							: 2;
  __REG32 MODE12							: 1;
  __REG32 POLARITY12					: 1;
  __REG32 NPOLARITY12					: 1;
  __REG32 INOUT12							: 1;
  __REG32 										: 5;
  __REG32 DELAY12 						: 1;
  __REG32 NDELAY12						: 1;
  __REG32 									 	: 2;
  __REG32 CHANSEL12 					: 2;
  __REG32 NCHANSEL12					: 2;
  __REG32 OPTCLKEN12					: 1;
  __REG32 NOPTCLKEN12					: 1;
  __REG32 									 	: 9;
} __gfxtcon_dir_pin12_ctrl_bits;

/* GFXDISP_DISPLAYENABLE */
typedef struct {
  __REG32 DEN									: 1;
  __REG32 		  							:31;
} __gfxdisp_displayenable_bits;

/* GFXDISP_DISPLAYRESOLUTION */
typedef struct {
  __REG32 HTP									:12;
  __REG32 		  							: 4;
  __REG32 VTR									:12;
  __REG32 		  							: 4;
} __gfxdisp_displayresolution_bits;

/* GFXDISP_DISPLAYACTIVEAREA */
typedef struct {
  __REG32 HDP									:12;
  __REG32 		  							: 4;
  __REG32 VDR									:12;
  __REG32 		  							: 4;
} __gfxdisp_displayactivearea_bits;

/* GFXDISP_HORIZONTALSYNCHTIMINGCONF */
typedef struct {
  __REG32 HSP									:12;
  __REG32 		  							: 4;
  __REG32 HSW									:12;
  __REG32 		  							: 4;
} __gfxdisp_horizontalsynchtimingconf_bits;

/* GFXDISP_VERTICALSYNCHTIMINGCONF */
typedef struct {
  __REG32 VSP									:12;
  __REG32 		  							: 4;
  __REG32 VSW									:12;
  __REG32 		  							: 4;
} __gfxdisp_verticalsynchtimingconf_bits;

/* GFXDISP_DISPLAYCONF */
typedef struct {
  __REG32 POLSYNC							: 1;
  __REG32 		  							:31;
} __gfxdisp_displayconf_bits;

/* GFXDISP_PIXENGTRIG */
typedef struct {
  __REG32 PESCOL							:12;
  __REG32 		  							: 4;
  __REG32 PESROW 							:12;
  __REG32 		  							: 4;
} __gfxdisp_pixengtrig_bits;

/* GFXDISP_DITHERCONTROL */
typedef struct {
  __REG32 DITHER_BYPASS				: 1;
  __REG32 DITHER_MODE 				: 1;
  __REG32 										: 2;
  __REG32 DITHER_FORMAT				: 2;
  __REG32 		  							: 2;
  __REG32 DITHER_ALIGN				: 1;
  __REG32 		  							:23;
} __gfxdisp_dithercontrol_bits;

/* GFXDISP_INT0TRIGGER */
typedef struct {
  __REG32 INT0COL							:12;
  __REG32 										: 4;
  __REG32 INT0ROW							:12;
  __REG32 										: 3;
  __REG32 INT0EN							: 1;
} __gfxdisp_int0trigger_bits;

/* GFXDISP_INT1TRIGGER */
typedef struct {
  __REG32 INT1COL							:12;
  __REG32 										: 4;
  __REG32 INT1ROW							:12;
  __REG32 										: 3;
  __REG32 INT1EN							: 1;
} __gfxdisp_int1trigger_bits;

/* GFXDISP_INT2TRIGGER */
typedef struct {
  __REG32 INT2COL							:12;
  __REG32 										: 4;
  __REG32 INT2ROW							:12;
  __REG32 										: 3;
  __REG32 INT2EN							: 1;
} __gfxdisp_int2trigger_bits;

/* GFXDISP_DEBUG */
typedef struct {
  __REG32 SHOWDISTFRAMES			: 1;
  __REG32 										:31;
} __gfxdisp_debug_bits;

/* GFXSIG_SIGLOCKSTATUS */
typedef struct {
  __REG32 LOCKSTATUS					: 1;
  __REG32 										:31;
} __gfxsig_siglockstatus_bits;

/* GFXSIG_SIGSWRESET */
typedef struct {
  __REG32 SWRES								: 1;
  __REG32 										:31;
} __gfxsig_sigswreset_bits;

/* GFXSIG_SIGCTRL */
typedef struct {
  __REG32 HMASK_MODE					: 2;
  __REG32 										: 6;
  __REG32 VMASK_MODE					: 2;
  __REG32 										: 6;
  __REG32 SRCSEL							: 2;
  __REG32 										:14;
} __gfxsig_sigctrl_bits;

/* GFXSIG_MASKHORIZONTALUPPERLEFT */
typedef struct {
  __REG32 MASKHORIZONTALUPPERLEFT	:12;
  __REG32 												:20;
} __gfxsig_maskhorizontalupperleft_bits;

/* GFXSIG_MASKHORIZONTALLOWERRIGHT */
typedef struct {
  __REG32 MASKHORIZONTALLOWERRIGHT	:12;
  __REG32 													:20;
} __gfxsig_maskhorizontallowerright_bits;

/* GFXSIG_MASKVERTICALUPPERLEFT */
typedef struct {
  __REG32 MASKVERTICALUPPERLEFT			:12;
  __REG32 													:20;
} __gfxsig_maskverticalupperleft_bits;

/* GFXSIG_MASKVERTICALLOWERRIGHT */
typedef struct {
  __REG32 MASKVERTICALLOWERRIGHT		:12;
  __REG32 													:20;
} __gfxsig_maskverticallowerright_bits;

/* GFXSIG_HORIZONTALUPPERLEFTW0 */
typedef struct {
  __REG32 HORIZONTALUPPERLEFTW0			:12;
  __REG32 													:20;
} __gfxsig_horizontalupperleftw0_bits;

/* GFXSIG_HORIZONTALLOWERRIGHTW0 */
typedef struct {
  __REG32 HORIZONTALLOWERRIGHTW0		:12;
  __REG32 													:20;
} __gfxsig_horizontallowerrightw0_bits;

/* GFXSIG_VERTICALUPPERLEFTW0 */
typedef struct {
  __REG32 VERTICALUPPERLEFTW0				:12;
  __REG32 													:20;
} __gfxsig_verticalupperleftw0_bits;

/* GFXSIG_VERTICALLOWERRIGHTW0 */
typedef struct {
  __REG32 VERTICALLOWERRIGHTW0			:12;
  __REG32 													:20;
} __gfxsig_verticallowerrightw0_bits;

/* GFXSIG_ERRORTHRESHOLD */
typedef struct {
  __REG32 ERRTHRES									: 8;
  __REG32 													: 8;
  __REG32 ERRTHRESRESET							: 8;
  __REG32 													: 8;
} __gfxsig_errorthreshold_bits;

/* GFXSIG_CTRLCFGW0 */
typedef struct {
  __REG32 ENSIGNA										: 1;
  __REG32 													: 7;
  __REG32 ENSIGNB										: 1;
  __REG32 													: 7;
  __REG32 ENCOORDW0 								: 1;
  __REG32 													:14;
  __REG32 INT_TYPE	 								: 1;
} __gfxsig_ctrlcfgw0_bits;

/* GFXSIG_TRIGGERW0 */
typedef struct {
  __REG32 TRIGGER										: 1;
  __REG32 													: 7;
  __REG32 TRIGMODE									: 2;
  __REG32 													:22;
} __gfxsig_triggerw0_bits;

/* GFXSIG_IENW0 */
typedef struct {
  __REG32 IENDIFF 									: 1;
  __REG32 IENCFGCOP									: 1;
  __REG32 IENRESVAL									: 1;
  __REG32 													:29;
} __gfxsig_ienw0_bits;

/* GFXSIG_INTERRUPTSTATUSW0 */
typedef struct {
  __REG32 ISTSDIFF 									: 1;
  __REG32 ISTSCFGCOP								: 1;
  __REG32 ISTSRESVAL								: 1;
  __REG32 													:29;
} __gfxsig_interruptstatusw0_bits;

/* GFXSIG_STATUSW0 */
typedef struct {
  __REG32 PENDING 									: 1;
  __REG32 ACTIVE										: 1;
  __REG32 													: 6;
  __REG32 DIFF_A_R									: 1;
  __REG32 DIFF_A_G									: 1;
  __REG32 DIFF_A_B									: 1;
  __REG32 													: 5;
  __REG32 DIFF_B_R									: 1;
  __REG32 DIFF_B_G									: 1;
  __REG32 DIFF_B_B									: 1;
  __REG32 													:13;
} __gfxsig_statusw0_bits;

/* GFXSIG_SIGNATURE_ERROR */
typedef struct {
  __REG32 SIG_ERROR_COUNT						:12;
  __REG32 													:20;
} __gfxsig_signature_error_bits;

/* GFXAIC_STATUS */
typedef struct {
  __REG32 TYPE											: 2;
  __REG32 													: 6;
  __REG32 ID												: 8;
  __REG32 													:16;
} __gfxaic_status_bits;

/* GFXAIC_STATUS */
typedef struct {
  __REG32 CLEAR											: 1;
  __REG32 													:31;
} __gfxaic_control_bits;

/* GFXAIC_MONITORDISABLE */
typedef struct {
  __REG32 MSM												: 1;
  __REG32 PIXF0											: 1;
  __REG32 PIXF1											: 1;
  __REG32 PIXF2											: 1;
  __REG32 PIXF3											: 1;
  __REG32 PIXF4											: 1;
  __REG32 PIXF5											: 1;
  __REG32 PIXF6											: 1;
  __REG32 PIXF7											: 1;
  __REG32 PIXW											: 1;
  __REG32 CMDR											: 1;
  __REG32 CMDW											: 1;
  __REG32 													:20;
} __gfxaic_monitordisable_bits;

/* GFXAIC_SLAVEDISABLE */
typedef struct {
  __REG32 MSS												: 1;
  __REG32 VRAM0											: 1;
  __REG32 VRAM1											: 1;
  __REG32 SPIMEM										: 1;
  __REG32 SPICSR										: 1;
  __REG32 CMDSEQ										: 1;
  __REG32 PIXENG										: 1;
  __REG32 TCON											: 1;
  __REG32 SIGNATURE									: 1;
  __REG32 DISPCTRL									: 1;
  __REG32 GLOBALCTRL								: 1;
  __REG32 HPM												: 1;
  __REG32 													:20;
} __gfxaic_slavedisable_bits;

/* GFXPIX_FETCH0_STATUS */
typedef struct {
  __REG32 FETCH0_STATUSBUSY					: 1;
  __REG32 													: 3;
  __REG32 FETCH0_STATUSBUFFERSIDLE	: 1;
  __REG32 FETCH0_STATUSREQUEST			: 1;
  __REG32 FETCH0_STATUSCOMPLETE			: 1;
  __REG32 													:25;
} __gfxpix_fetch_status_bits;

/* GFXPIX_FETCH0_BURSTBUFFERMANAGEMENT */
typedef struct {
  __REG32 FETCH0_MANAGEDBURSTBUFFERS			: 8;
  __REG32 FETCH0_BURSTLENGTHFORMAXBUFFERS	: 5;
  __REG32 																: 3;
  __REG32 FETCH0_SETNUMBUFFERS						: 8;
  __REG32 FETCH0_SETBURSTLENGTH						: 5;
  __REG32 																: 3;
} __gfxpix_fetch_burstbuffermanagement_bits;

/* GFXPIX_FETCH0_SOURCEBUFFERSTRIDE */
typedef struct {
  __REG32 FETCH0_STRIDE										:12;
  __REG32 																:20;
} __gfxpix_fetch_sourcebufferstride_bits;

/* GFXPIX_FETCH0_SOURCEBUFFERATTRIBUTES */
typedef struct {
  __REG32 FETCH0_LINEWIDTH								:10;
  __REG32 																: 6;
  __REG32 FETCH0_LINECOUNT								:10;
  __REG32 																: 6;
} __gfxpix_fetch_sourcebufferattributes_bits;

/* GFXPIX_FETCH0_SOURCEBUFFERLENGTH */
typedef struct {
  __REG32 FETCH0_RLEWORDS									:21;
  __REG32 																:11;
} __gfxpix_fetch_sourcebufferlength_bits;

/* GFXPIX_FETCH0_FRAMEXOFFSET */
typedef struct {
  __REG32 																:16;
  __REG32 FETCH0_FRAMEXOFFSET							:12;
  __REG32 																: 3;
  __REG32 FETCH0_FRAMEXDIRECTION					: 1;
} __gfxpix_fetch_framexoffset_bits;

/* GFXPIX_FETCH0_FRAMEYOFFSET */
typedef struct {
  __REG32 																:16;
  __REG32 FETCH0_FRAMEYOFFSET							:12;
  __REG32 																: 3;
  __REG32 FETCH0_FRAMEYDIRECTION					: 1;
} __gfxpix_fetch_frameyoffset_bits;

/* GFXPIX_FETCH0_FRAMEDIMENSIONS */
typedef struct {
  __REG32 FETCH0_FRAMEWIDTH								:10;
  __REG32 																: 6;
  __REG32 FETCH0_FRAMEHEIGHT							:10;
  __REG32 																: 5;
  __REG32 FETCH0_FRAMESWAPDIRECTIONS			: 1;
} __gfxpix_fetch_framedimensions_bits;

/* GFXPIX_FETCH0_SKIPWINDOWOFFSET */
typedef struct {
  __REG32 FETCH0_SKIPWINDOWXOFFSET 				:10;
  __REG32 																: 6;
  __REG32 FETCH0_SKIPWINDOWYOFFSET				:10;
  __REG32 																: 6;
} __gfxpix_fetch_skipwindowoffset_bits;

/* GFXPIX_FETCH0_SKIPWINDOWDIMENSIONS */
typedef struct {
  __REG32 FETCH0_SKIPWINDOWWIDTH	 				:10;
  __REG32 																: 6;
  __REG32 FETCH0_SKIPWINDOWHEIGHT					:10;
  __REG32 																: 6;
} __gfxpix_fetch_skipwindowdimensions_bits;

/* GFXPIX_FETCH0_COLORCOMPONENTBITS */
typedef struct {
  __REG32 FETCH0_COMPONENTBITSALPHA 			: 4;
  __REG32 																: 4;
  __REG32 FETCH0_COMPONENTBITSBLUE	 			: 4;
  __REG32 																: 4;
  __REG32 FETCH0_COMPONENTBITSGREEN	 			: 4;
  __REG32 																: 4;
  __REG32 FETCH0_COMPONENTBITSRED		 			: 4;
  __REG32 																: 4;
} __gfxpix_fetch_colorcomponentbits_bits;

/* GFXPIX_FETCH0_COLORCOMPONENTSHIFT */
typedef struct {
  __REG32 FETCH0_COMPONENTSHIFTALPHA			: 5;
  __REG32 																: 3;
  __REG32 FETCH0_COMPONENTSHIFTBLUE	 			: 5;
  __REG32 																: 3;
  __REG32 FETCH0_COMPONENTSHIFTGREEN			: 5;
  __REG32 																: 3;
  __REG32 FETCH0_COMPONENTSHIFTRED	 			: 5;
  __REG32 																: 3;
} __gfxpix_fetch_colorcomponentshift_bits;

/* GFXPIX_FETCH0_CONSTANTCOLOR */
typedef struct {
  __REG32 FETCH0_CONSTANTCOLORALPHA				: 8;
  __REG32 FETCH0_CONSTANTCOLORBLUE	 			: 8;
  __REG32 FETCH0_CONSTANTCOLORGREEN				: 8;
  __REG32 FETCH0_CONSTANTCOLORRED		 			: 8;
} __gfxpix_fetch_constantcolor_bits;

/* GFXPIX_FETCH0_CONTROL */
typedef struct {
  __REG32 FETCH0_SHADOWLOAD								: 1;
  __REG32 FETCH0_START							 			: 1;
  __REG32 FETCH0_SWRESET									: 1;
  __REG32 FETCH0_CLOCKDISABLE				 			: 1;
  __REG32 FETCH0_BITSPERPIXEL				 			: 6;
  __REG32 													 			: 2;
  __REG32 FETCH0_ALPHAMULTIPLY			 			: 1;
  __REG32 FETCH0_COLORMULTIPLYENABLE 			: 1;
  __REG32 FETCH0_COLORMULTIPLYSELECT 			: 1;
  __REG32 													 			: 1;
  __REG32 FETCH0_SETFIELD						 			: 1;
  __REG32 FETCH0_TOGGLEFIELD				 			: 1;
  __REG32 													 			: 2;
  __REG32 FETCH0_DUMMYSKIPSELECT		 			: 1;
  __REG32 FETCH0_TILEMODE						 			: 2;
  __REG32 													 			: 1;
  __REG32 FETCH0_RLDENABLE					 			: 1;
  __REG32 													 			: 3;
  __REG32 FETCH0_FETCHTYPE					 			: 2;
  __REG32 FETCH0_HASMULTIPLY				 			: 1;
  __REG32 FETCH0_SHD_UPD						 			: 1;
} __gfxpix_fetch_control_bits;

/* GFXPIX_STORE0_STATUS */
typedef struct {
  __REG32 STORE0_STATUSBUSY								: 1;
  __REG32 													 			: 3;
  __REG32 STORE0_STATUSBUFFERSIDLE				: 1;
  __REG32 STORE0_STATUSREQUEST			 			: 1;
  __REG32 STORE0_STATUSCOMPLETE			 			: 1;
  __REG32 													 			:25;
} __gfxpix_store_status_bits;

/* GFXPIX_STORE0_BURSTBUFFERMANAGEMENT */
typedef struct {
  __REG32 STORE0_MANAGEDBURSTBUFFERS			: 8;
  __REG32 STORE0_MAXBURSTLENGTH						: 5;
  __REG32 													 			:11;
  __REG32 STORE0_SETBURSTLENGTH						: 5;
  __REG32 													 			: 3;
} __gfxpix_store_burstbuffermanagement_bits;

/* GFXPIX_STORE0_DESTINATIONBUFFERSTRIDE */
typedef struct {
  __REG32 STORE0_STRIDE										:12;
  __REG32 													 			:20;
} __gfxpix_store_destinationbufferstride_bits;

/* GFXPIX_STORE0_FRAMEXOFFSET */
typedef struct {
  __REG32 													 			:16;
  __REG32 STORE0_FRAMEXOFFSET							:10;
  __REG32 													 			: 6;
} __gfxpix_store_framexoffset_bits;

/* GFXPIX_STORE0_FRAMEYOFFSET */
typedef struct {
  __REG32 													 			:16;
  __REG32 STORE0_FRAMEYOFFSET							:10;
  __REG32 													 			: 6;
} __gfxpix_store_frameyoffset_bits;

/* GFXPIX_STORE0_COLORCOMPONENTBITS */
typedef struct {
  __REG32 STORE0_COMPONENTBITSALPHA  			: 4;
  __REG32 													 			: 4;
  __REG32 STORE0_COMPONENTBITSBLUE  			: 4;
  __REG32 													 			: 4;
  __REG32 STORE0_COMPONENTBITSGREEN  			: 4;
  __REG32 													 			: 4;
  __REG32 STORE0_COMPONENTBITSRED	  			: 4;
  __REG32 													 			: 4;
} __gfxpix_store_colorcomponentbits_bits;

/* GFXPIX_STORE0_COLORCOMPONENTSHIFT */
typedef struct {
  __REG32 STORE0_COMPONENTSHIFTALPHA  		: 5;
  __REG32 													 			: 3;
  __REG32 STORE0_COMPONENTSHIFTBLUE  			: 5;
  __REG32 													 			: 3;
  __REG32 STORE0_COMPONENTSHIFTGREEN			: 5;
  __REG32 													 			: 3;
  __REG32 STORE0_COMPONENTSHIFTRED  			: 5;
  __REG32 													 			: 3;
} __gfxpix_store_colorcomponentshift_bits;

/* GFXPIX_STORE0_CONTROLE */
typedef struct {
  __REG32 STORE0_START							  		: 1;
  __REG32 													 			: 1;
  __REG32 STORE0_SWRESET					  			: 1;
  __REG32 STORE0_CLOCKDISABLE				 			: 1;
  __REG32 STORE0_BITSPERPIXEL							: 7;
  __REG32 													 			: 1;
  __REG32 STORE0_COLORDITHERENABLE  			: 1;
  __REG32 STORE0_ALPHADITHERENABLE  			: 1;
  __REG32 													 			:17;
  __REG32 STORE0_SHD_UPD					  			: 1;
} __gfxpix_store_control_bits;

/* GFXPIX_HSCALER0_CONTROL */
typedef struct {
  __REG32 HSCALER0_MODE							  		: 1;
  __REG32 													 			: 3;
  __REG32 HSCALER0_SCALE_MODE			  			: 1;
  __REG32 													 			: 3;
  __REG32 HSCALER0_FILTER_MODE			 			: 1;
  __REG32 													 			: 7;
  __REG32 HSCALER0_OUTPUT_SIZE						:10;
  __REG32 													 			: 5;
  __REG32 HSCALER0_SHD_UPD		 					  : 1;
} __gfxpix_hscaler_control_bits;

/* GFXPIX_HSCALER0_SETUP1 */
typedef struct {
  __REG32 HSCALER0_SCALE_FACTOR			  		:16;
  __REG32 													 			:16;
} __gfxpix_hscaler_setup1_bits;

/* GFXPIX_HSCALER0_SETUP2 */
typedef struct {
  __REG32 HSCALER0_PHASE_OFFSET			  		:17;
  __REG32 													 			:15;
} __gfxpix_hscaler_setup2_bits;

/* GFXPIX_VSCALER0_CONTROL */
typedef struct {
  __REG32 VSCALER0_MODE							  		: 1;
  __REG32 													 			: 3;
  __REG32 VSCALER0_SCALE_MODE				  		: 1;
  __REG32 													 			: 3;
  __REG32 VSCALER0_FILTER_MODE			  		: 1;
  __REG32 													 			: 7;
  __REG32 VSCALER0_OUTPUT_SIZE			  		:10;
  __REG32 													 			: 5;
  __REG32 VSCALER0_SHD_UPD					  		: 1;
} __gfxpix_vscaler_control_bits;

/* GFXPIX_VSCALER0_SETUP1 */
typedef struct {
  __REG32 VSCALER0_SCALE_FACTOR 		  		:16;
  __REG32 													 			:16;
} __gfxpix_vscaler_setup1_bits;

/* GFXPIX_VSCALER0_SETUP2 */
typedef struct {
  __REG32 VSCALER0_PHASE_OFFSET 		  		:17;
  __REG32 													 			:15;
} __gfxpix_vscaler_setup2_bits;

/* GFXPIX_ROP0_CONTROL */
typedef struct {
  __REG32 ROP0_MODE							 		  		: 1;
  __REG32 													 			:30;
  __REG32 ROP0_SHD_UPD					 		  		: 1;
} __gfxpix_rop_control_bits;

/* GFXPIX_ROP0_RASTEROPERATIONINDICES */
typedef struct {
  __REG32 ROP0_OPINDEXALPHA			 		  		: 8;
  __REG32 ROP0_OPINDEXBLUE			 		  		: 8;
  __REG32 ROP0_OPINDEXGREEN			 		  		: 8;
  __REG32 ROP0_OPINDEXRED				 		  		: 8;
} __gfxpix_rop_rasteroperationindices_bits;

/* GFXPIX_BLITBLEND0_CONTROL */
typedef struct {
  __REG32 BLITBLEND0_MODE				 		  		: 1;
  __REG32 											 		  		:30;
  __REG32 BLITBLEND0_SHD_UPD		 		  		: 1;
} __gfxpix_blitblend_control_bits;

/* GFXPIX_BLITBLEND0_CONSTANTCOLOR */
typedef struct {
  __REG32 BLITBLEND0_CONSTANTCOLORALPHA		: 8;
  __REG32 BLITBLEND0_CONSTANTCOLORBLUE		: 8;
  __REG32 BLITBLEND0_CONSTANTCOLORGREEN		: 8;
  __REG32 BLITBLEND0_CONSTANTCOLORRED			: 8;
} __gfxpix_blitblend_constantcolor_bits;

/* GFXPIX_BLITBLEND0_COLORREDBLENDFUNCTION */
typedef struct {
  __REG32 BLITBLEND0_BLENDFUNCCOLORREDSRC	:16;
  __REG32 BLITBLEND0_BLENDFUNCCOLORREDDST	:16;
} __gfxpix_blitblend_colorredblendfunction_bits;

/* GFXPIX_BLITBLEND0_COLORREDBLENDFUNCTION */
typedef struct {
  __REG32 BLITBLEND0_BLENDFUNCCOLORGREENSRC		:16;
  __REG32 BLITBLEND0_BLENDFUNCCOLORGREENDST		:16;
} __gfxpix_blitblend_colorblueblendfunction_bits;

/* GFXPIX_BLITBLEND0_COLORGREENBLENDFUNCTION */
typedef struct {
  __REG32 BLITBLEND0_BLENDFUNCCOLORBLUESRC		:16;
  __REG32 BLITBLEND0_BLENDFUNCCOLORBLUEDST		:16;
} __gfxpix_blitblend_colorgreenblendfunction_bits;

/* GFXPIX_BLITBLEND0_ALPHABLENDFUNCTION */
typedef struct {
  __REG32 BLITBLEND0_BLENDFUNCALPHASRC				:16;
  __REG32 BLITBLEND0_BLENDFUNCALPHADST				:16;
} __gfxpix_blitblend_alphablendfunction_bits;

/* GFXPIX_BLITBLEND0_BLENDMODE1 */
typedef struct {
  __REG32 BLITBLEND0_BLENDMODECOLORRED				:16;
  __REG32 BLITBLEND0_BLENDMODECOLORGREEN			:16;
} __gfxpix_blitblend_blendmode1_bits;

/* GFXPIX_BLITBLEND0_BLENDMODE2 */
typedef struct {
  __REG32 BLITBLEND0_BLENDMODECOLORBLUE				:16;
  __REG32 BLITBLEND0_BLENDMODEALPHA						:16;
} __gfxpix_blitblend_blendmode2_bits;

/* GFXPIX_BLITBLEND0_DEBUG */
typedef struct {
  __REG32 BLITBLEND0_COLORDEBUG								:10;
  __REG32 BLITBLEND0_ALPHADEBUG								:10;
  __REG32 																		:12;
} __gfxpix_blitblend_debug_bits;

/* GFXPIX_LAYERBLEND0_CONTROL */
typedef struct {
  __REG32 LAYERBLEND0_MODE										: 2;
  __REG32 LAYERBLEND0_PRIM_C_BLD_FUNC					: 3;
  __REG32 LAYERBLEND0_SEC_C_BLD_FUNC					: 3;
  __REG32 LAYERBLEND0_PRIM_A_BLD_FUNC					: 3;
  __REG32 LAYERBLEND0_SEC_A_BLD_FUNC					: 3;
  __REG32 																		: 2;
  __REG32 LAYERBLEND0_ALPHA										: 8;
  __REG32 																		: 7;
  __REG32 LAYERBLEND0_SHD_UPD									: 1;
} __gfxpix_layerblend_control_bits;

/* GFXPIX_LAYERBLEND0_POSITION */
typedef struct {
  __REG32 LAYERBLEND0_XPOS										:12;
  __REG32 																		: 4;
  __REG32 LAYERBLEND0_YPOS										:12;
  __REG32 																		: 4;
} __gfxpix_layerblend_position_bits;

/* GFXPIX_LAYERBLEND0_TRANS_COL */
typedef struct {
  __REG32 LAYERBLEND0_BLUE										: 8;
  __REG32 LAYERBLEND0_GREEN										: 8;
  __REG32 LAYERBLEND0_RED											: 8;
  __REG32 																		: 8;
} __gfxpix_layerblend_trans_col_bits;

/* GFXPIX_LUT0_CONTROL */
typedef struct {
  __REG32 LUT0_MODE														: 2;
  __REG32 LUT0_COL_8BIT												: 1;
  __REG32 LUT0_B_EN														: 1;
  __REG32 LUT0_G_EN														: 1;
  __REG32 LUT0_R_EN														: 1;
  __REG32 LUT0_IDX_BITS												: 4;
  __REG32 																		:14;
  __REG32 LUT0_WRITE_TIMEOUT									: 1;
  __REG32 LUT0_READ_TIMEOUT										: 1;
  __REG32 																		: 5;
  __REG32 LUT0_SHD_UPD												: 1;
} __gfxpix_lut0_control_bits;

/* GFXPIX_LUT0_LUT */
typedef struct {
  __REG32 LUT0_BLUE														:10;
  __REG32 LUT0_GREEN													:10;
  __REG32 LUT0_RED 														:10;
  __REG32 																		: 2;
} __gfxpix_lut0_lut_bits;

/* GFXPIX_LUT1_CONTROL */
typedef struct {
  __REG32 LUT1_MODE														: 2;
  __REG32 LUT1_COL_8BIT												: 1;
  __REG32 LUT1_B_EN														: 1;
  __REG32 LUT1_G_EN														: 1;
  __REG32 LUT1_R_EN														: 1;
  __REG32 LUT1_IDX_BITS												: 4;
  __REG32 																		:14;
  __REG32 LUT1_WRITE_TIMEOUT									: 1;
  __REG32 LUT1_READ_TIMEOUT										: 1;
  __REG32 																		: 5;
  __REG32 LUT1_SHD_UPD												: 1;
} __gfxpix_lut1_control_bits;

/* GFXPIX_LUT1_LUT */
typedef struct {
  __REG32 LUT1_BLUE														:10;
  __REG32 LUT1_GREEN													:10;
  __REG32 LUT1_RED 														:10;
  __REG32 																		: 2;
} __gfxpix_lut1_lut_bits;

/* GFXPIX_MATRIX0_CONTROL */
typedef struct {
  __REG32 MATRIX0_MODE												: 2;
  __REG32 																		:30;
} __gfxpix_matrix_control_bits;

/* GFXPIX_MATRIX0_RED0 */
typedef struct {
  __REG32 MATRIX0_A11													:11;
  __REG32 																		: 5;
  __REG32 MATRIX0_A12													:11;
  __REG32 																		: 5;
} __gfxpix_matrix_red0_bits;

/* GFXPIX_MATRIX0_RED1 */
typedef struct {
  __REG32 MATRIX0_A13													:11;
  __REG32 																		: 5;
  __REG32 MATRIX0_C1													:11;
  __REG32 																		: 5;
} __gfxpix_matrix_red1_bits;

/* GFXPIX_MATRIX0_GREEN0 */
typedef struct {
  __REG32 MATRIX0_A21													:11;
  __REG32 																		: 5;
  __REG32 MATRIX0_A22													:11;
  __REG32 																		: 5;
} __gfxpix_matrix_green0_bits;

/* GFXPIX_MATRIX0_GREEN1 */
typedef struct {
  __REG32 MATRIX0_A23													:11;
  __REG32 																		: 5;
  __REG32 MATRIX0_C2													:11;
  __REG32 																		: 5;
} __gfxpix_matrix_green1_bits;

/* GFXPIX_MATRIX0_BLUE0 */
typedef struct {
  __REG32 MATRIX0_A31													:11;
  __REG32 																		: 5;
  __REG32 MATRIX0_A32													:11;
  __REG32 																		: 5;
} __gfxpix_matrix_blue0_bits;

/* GFXPIX_MATRIX0_BLUE1 */
typedef struct {
  __REG32 MATRIX0_A33													:11;
  __REG32 																		: 5;
  __REG32 MATRIX0_C3													:11;
  __REG32 																		: 5;
} __gfxpix_matrix_blue1_bits;

/* GFXPIX_EXTDST0_CONTROL */
typedef struct {
  __REG32 EXTDST0_KICK_MODE										: 2;
  __REG32 EXTDST0_KICK												: 1;
  __REG32 EXTDST0_ALPHA_MODE									: 3;
  __REG32 																		:24;
  __REG32 EXTDST0_KICK_RES										: 1;
  __REG32 EXTDST0_SHD_UPD 										: 1;
} __gfxpix_extdst0_control_bits;

/* GFXPIX_EXTDST0_STATUS */
typedef struct {
  __REG32 EXTDST0_KICK_CNT										: 2;
  __REG32 EXTDST0_EMPTY												: 1;
  __REG32 																		:13;
  __REG32 EXTDST0_E_KERR_STS									: 1;
  __REG32 EXTDST0_SW_KERR_STS 								: 1;
  __REG32 EXTDST0_CNT_ERR_STS									: 1;
  __REG32 								 										:13;
} __gfxpix_extdst0_status_bits;

/* GFXPIX_EXTDST0_CUR_PIXEL_CNT */
typedef struct {
  __REG32 EXTDST0_C_XVAL											:16;
  __REG32 EXTDST0_C_YVAL											:16;
} __gfxpix_extdst0_cur_pixel_cnt_bits;

/* GFXPIX_EXTDST0_LAST_PIXEL_CNT */
typedef struct {
  __REG32 EXTDST0_L_XVAL											:16;
  __REG32 EXTDST0_L_YVAL											:16;
} __gfxpix_extdst0_last_pixel_cnt_bits;

/* GFXPIX_PIXELBUS_FETCH0_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH0_SHDW								: 2;
  __REG32 PIXELBUS_FETCH0_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch0_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH1_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH1_SHDW								: 2;
  __REG32 PIXELBUS_FETCH1_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch1_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH2_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH2_SHDW								: 2;
  __REG32 PIXELBUS_FETCH2_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch2_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH3_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH3_SHDW								: 2;
  __REG32 PIXELBUS_FETCH3_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch3_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH4_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH4_SHDW								: 2;
  __REG32 PIXELBUS_FETCH4_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch4_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH5_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH5_SHDW								: 2;
  __REG32 PIXELBUS_FETCH5_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch5_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH6_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH6_SHDW								: 2;
  __REG32 PIXELBUS_FETCH6_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch6_cfg_bits;

/* GFXPIX_PIXELBUS_FETCH7_CFG */
typedef struct {
  __REG32 																		:26;
  __REG32 PIXELBUS_FETCH7_SHDW								: 2;
  __REG32 PIXELBUS_FETCH7_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_fetch7_cfg_bits;

/* GFXPIX_PIXELBUS_STORE0_CFG */
typedef struct {
  __REG32 PIXELBUS_STORE0_SRC_SEL							: 5;
  __REG32 																		:21;
  __REG32 PIXELBUS_STORE0_SHDW								: 2;
  __REG32 PIXELBUS_STORE0_SEL									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_store0_cfg_bits;

/* GFXPIX_PIXELBUS_HSCALER0_CFG */
typedef struct {
  __REG32 PIXELBUS_HSCALER0_PIXIN_SEL					: 5;
  __REG32 																		:19;
  __REG32 PIXELBUS_HSCALER0_CLKEN							: 2;
  __REG32 PIXELBUS_HSCALER0_SHDW							: 2;
  __REG32 PIXELBUS_HSCALER0_SEL								: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_hscaler0_cfg_bits;

/* GFXPIX_PIXELBUS_VSCALER0_CFG */
typedef struct {
  __REG32 PIXELBUS_VSCALER0_PIXIN_SEL					: 5;
  __REG32 																		:19;
  __REG32 PIXELBUS_VSCALER0_CLKEN							: 2;
  __REG32 PIXELBUS_VSCALER0_SHDW							: 2;
  __REG32 PIXELBUS_VSCALER0_SEL								: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_vscaler0_cfg_bits;

/* GFXPIX_PIXELBUS_ROP0_CFG */
typedef struct {
  __REG32 PIXELBUS_ROP0_PRIM_SEL							: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_ROP0_SEC_SEL								: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_ROP0_AUX_SEL								: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_ROP0_CLKEN									: 2;
  __REG32 PIXELBUS_ROP0_SHDW									: 2;
  __REG32 PIXELBUS_ROP0_SEL 									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_rop0_cfg_bits;

/* GFXPIX_PIXELBUS_ROP1_CFG */
typedef struct {
  __REG32 PIXELBUS_ROP1_PRIM_SEL							: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_ROP1_SEC_SEL								: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_ROP1_AUX_SEL								: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_ROP1_CLKEN									: 2;
  __REG32 PIXELBUS_ROP1_SHDW									: 2;
  __REG32 PIXELBUS_ROP1_SEL 									: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_rop1_cfg_bits;

/* GFXPIX_PIXELBUS_BLITBLEND0_CFG */
typedef struct {
  __REG32 PIXELBUS_BLITBLEND0_PRIM_SEL				: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_BLITBLEND0_SEC_SEL					: 5;
  __REG32 																		:11;
  __REG32 PIXELBUS_BLITBLEND0_CLKEN 					: 2;
  __REG32 PIXELBUS_BLITBLEND0_SHDW 						: 2;
  __REG32 PIXELBUS_BLITBLEND0_SEL	 						: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_blitblend0_cfg_bits;

/* GFXPIX_PIXELBUS_LAYERBLEND0_CFG */
typedef struct {
  __REG32 PIXELBUS_LAYERBLEND0_PRIM_SEL				: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_LAYERBLEND0_SEC_SEL				: 5;
  __REG32 																		:11;
  __REG32 PIXELBUS_LAYERBLEND0_CLKEN 					: 2;
  __REG32 PIXELBUS_LAYERBLEND0_SHDW 					: 2;
  __REG32 PIXELBUS_LAYERBLEND0_SEL	 					: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_layerblend0_cfg_bits;

/* GFXPIX_PIXELBUS_LAYERBLEND1_CFG */
typedef struct {
  __REG32 PIXELBUS_LAYERBLEND1_PRIM_SEL				: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_LAYERBLEND1_SEC_SEL				: 5;
  __REG32 																		:11;
  __REG32 PIXELBUS_LAYERBLEND1_CLKEN 					: 2;
  __REG32 PIXELBUS_LAYERBLEND1_SHDW 					: 2;
  __REG32 PIXELBUS_LAYERBLEND1_SEL	 					: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_layerblend1_cfg_bits;

/* GFXPIX_PIXELBUS_LAYERBLEND2_CFG */
typedef struct {
  __REG32 PIXELBUS_LAYERBLEND2_PRIM_SEL				: 5;
  __REG32 																		: 3;
  __REG32 PIXELBUS_LAYERBLEND2_SEC_SEL				: 5;
  __REG32 																		:11;
  __REG32 PIXELBUS_LAYERBLEND2_CLKEN 					: 2;
  __REG32 PIXELBUS_LAYERBLEND2_SHDW 					: 2;
  __REG32 PIXELBUS_LAYERBLEND2_SEL	 					: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_layerblend2_cfg_bits;

/* GFXPIX_PIXELBUS_LUT0_CFG */
typedef struct {
  __REG32 PIXELBUS_LUT0_SRC_SEL								: 5;
  __REG32 																		:19;
  __REG32 PIXELBUS_LUT0_CLKEN									: 2;
  __REG32 PIXELBUS_LUT0_SHDW									: 2;
  __REG32 PIXELBUS_LUT0_SEL					 					: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_lut0_cfg_bits;

/* GFXPIX_PIXELBUS_LUT1_CFG */
typedef struct {
  __REG32 PIXELBUS_LUT1_SRC_SEL								: 5;
  __REG32 																		:19;
  __REG32 PIXELBUS_LUT1_CLKEN									: 2;
  __REG32 PIXELBUS_LUT1_SHDW									: 2;
  __REG32 PIXELBUS_LUT1_SEL					 					: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_lut1_cfg_bits;

/* GFXPIX_PIXELBUS_MATRIX0_CFG */
typedef struct {
  __REG32 PIXELBUS_MATRIX0_SRC_SEL						: 5;
  __REG32 																		:19;
  __REG32 PIXELBUS_MATRIX0_CLKEN							: 2;
  __REG32 PIXELBUS_MATRIX0_SHDW								: 2;
  __REG32 PIXELBUS_MATRIX0_SEL			 					: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_matrix0_cfg_bits;

/* GFXPIX_PIXELBUS_EXTDST0_CFG */
typedef struct {
  __REG32 PIXELBUS_EXTDST0_SRC_SEL						: 5;
  __REG32 																		:21;
  __REG32 PIXELBUS_EXTDST0_SHDW								: 2;
  __REG32 PIXELBUS_EXTDST0_SEL								: 2;
  __REG32 																		: 2;
} __gfxpix_pixelbus_extdst0_cfg_bits;

/* GFXPIX_PIXELBUS_STORE0_SYNC */
typedef struct {
  __REG32 PIXELBUS_STORE0_START								: 1;
  __REG32 PIXELBUS_STORE0_SYNC_RESET_N				: 1;
  __REG32 PIXELBUS_STORE0_SLV_EN							: 2;
  __REG32 PIXELBUS_STORE0_MST_EN							: 2;
  __REG32 																		:26;
} __gfxpix_pixelbus_store0_sync_bits;

/* GFXPIX_PIXELBUS_STORE0_SYNC_STAT */
typedef struct {
  __REG32 PIXELBUS_STORE0_SYNC_BUSY						: 1;
  __REG32 PIXELBUS_STORE0_KICK_CNT_STORE0			: 2;
  __REG32 PIXELBUS_STORE0_KICK_CNT_EXTDST0		: 2;
  __REG32 																		:27;
} __gfxpix_pixelbus_store0_sync_stat_bits;

/* GFXPIX_PIXELBUS_EXTDST0_SYNC */
typedef struct {
  __REG32 PIXELBUS_EXTDST0_START							: 1;
  __REG32 PIXELBUS_EXTDST0_SYNC_RESET_N				: 1;
  __REG32 PIXELBUS_EXTDST0_SLV_EN 						: 2;
  __REG32 PIXELBUS_EXTDST0_MST_EN							: 2;
  __REG32 																		:26;
} __gfxpix_pixelbus_extdst0_sync_bits;

/* GFXPIX_PIXELBUS_EXTDST0_SYNC_STAT */
typedef struct {
  __REG32 PIXELBUS_EXTDST0_SYNC_BUSY					: 1;
  __REG32 PIXELBUS_EXTDST0_KICK_CNT_STORE0		: 2;
  __REG32 PIXELBUS_EXTDST0_KICK_CNT_EXTDST0		: 2;
  __REG32 																		:27;
} __gfxpix_pixelbus_extdst0_sync_stat_bits;


/* GFXPIX_PIXELBUS_STORE0_CLK */
typedef struct {
  __REG32 PIXELBUS_STORE0_DIV									: 8;
  __REG32 																		:24;
} __gfxpix_pixelbus_store0_clk_bits;

/* GFXPIX_PIXELBUS_EXTDST0_CLK */
typedef struct {
  __REG32 PIXELBUS_EXTDST0_DIV  							: 8;
  __REG32 																		:24;
} __gfxpix_pixelbus_extdst0_clk_bits;

/* GFXPIX_MSS_AR_ARBITRATION */
typedef struct {
  __REG32 MSS_RMASTE				  							: 8;
  __REG32 MSS_RPRIORITY			  							: 8;
  __REG32 																	: 8;
  __REG32 MSS_RSLOT					  							: 8;
} __gfxpix_mss_ar_arbitration_bits;

/* GFXPIX_MSS_AW_ARBITRATION */
typedef struct {
  __REG32 MSS_WMASTER				  							: 8;
  __REG32 MSS_WPRIORITY			  							: 8;
  __REG32 																	: 8;
  __REG32 MSS_WSLOT					  							: 8;
} __gfxpix_mss_aw_arbitration_bits;

/* GFXPIX_VRAM0_AR_ARBITRATION */
typedef struct {
  __REG32 VRAM0_RMASTER			  							: 8;
  __REG32 VRAM0_RPRIORITY		  							: 8;
  __REG32 																	: 8;
  __REG32 VRAM0_RSLOT				  							: 8;
} __gfxpix_vram0_ar_arbitration_bits;

/* GFXPIX_VRAM0_AW_ARBITRATION */
typedef struct {
  __REG32 VRAM0_WMASTER			  							: 8;
  __REG32 VRAM0_WPRIORITY		  							: 8;
  __REG32 																	: 8;
  __REG32 VRAM0_WSLOT				  							: 8;
} __gfxpix_vram0_aw_arbitration_bits;

/* GFXPIX_VRAM1_AR_ARBITRATION */
typedef struct {
  __REG32 VRAM1_RMASTER			  							: 8;
  __REG32 VRAM1_RPRIORITY		  							: 8;
  __REG32 																	: 8;
  __REG32 VRAM1_RSLOT				  							: 8;
} __gfxpix_vram1_ar_arbitration_bits;

/* GFXPIX_VRAM1_AW_ARBITRATION */
typedef struct {
  __REG32 VRAM1_WMASTER			  							: 8;
  __REG32 VRAM1_WPRIORITY		  							: 8;
  __REG32 																	: 8;
  __REG32 VRAM1_WSLOT				  							: 8;
} __gfxpix_vram1_aw_arbitration_bits;

/* GFXPIX_AHB_AR_ARBITRATION */
typedef struct {
  __REG32 AHB_RMASTER				  							: 8;
  __REG32 AHB_RPRIORITY		  								: 8;
  __REG32 																	: 8;
  __REG32 AHB_RSLOT					  							: 8;
} __gfxpix_ahb_ar_arbitration_bits;

/* GFXPIX_AHB_AW_ARBITRATION */
typedef struct {
  __REG32 AHB_WMASTER				  							: 8;
  __REG32 AHB_WPRIORITY		  								: 8;
  __REG32 																	: 8;
  __REG32 AHB_WSLOT 				  							: 8;
} __gfxpix_ahb_aw_arbitration_bits;

/* GFXPIX_HPM_AR_ARBITRATION */
typedef struct {
  __REG32 HPM_RMASTER				  							: 8;
  __REG32 HPM_RPRIORITY		  								: 8;
  __REG32 																	: 8;
  __REG32 HPM_RSLOT 				  							: 8;
} __gfxpix_hpm_ar_arbitration_bits;

/* GFXPIX_HPM_AW_ARBITRATION */
typedef struct {
  __REG32 HPM_WMASTER				  							: 8;
  __REG32 HPM_WPRIORITY		  								: 8;
  __REG32 																	: 8;
  __REG32 HPM_WSLOT 				  							: 8;
} __gfxpix_hpm_aw_arbitration_bits;

/* MCFG_DTAR */
typedef struct {
  __REG32 BSREQ							  							: 1;
  __REG32 																	:24;
  __REG32 FPPREQ		 				  							: 1;
  __REG32 																	: 6;
} __mcfg_dtar_bits;

/* MCFG_TSR */
typedef struct {
  __REG32 MD								  							: 6;
  __REG32 																	: 2;
  __REG32 SMD								  							: 5;
  __REG32 																	: 3;
  __REG32 PCMREQ		 				  							: 1;
  __REG32 RAMEXECREQ 				  							: 1;
  __REG32 JTAGREQ		 				  							: 1;
  __REG32 TAPREQ 		 				  							: 1;
  __REG32 																	:11;
  __REG32 DONE	 		 				  							: 1;
} __mcfg_tsr_bits;

/* RICFG7G0_GFX0DCLKI */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL						  							: 2;
  __REG16 																	: 6;
} __ricfg7g0_gfx0dclki_bits;

/* RICFG7G4_EIC0INTxx */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL						  							: 2;
  __REG16 																	: 6;
} __ricfg7g4_eic0int_bits;

/* RICFG7G4_EIC0INTxx */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL						  							: 3;
  __REG16 																	: 5;
} __ricfg7g4_eic0int08_bits;

/* RICFG7G4_EIC0NMI */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL						  							: 1;
  __REG16 																	: 7;
} __ricfg7g4_eic0nmi_bits;

/* RICFG0G0_ADC0ANxx */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL						  							: 3;
  __REG16 																	: 5;
} __ricfg0g0_adc0an_bits;

/* RICFG0G0_ADC0EDGI */
typedef struct {
  __REG16 PORTPIN														: 1;
  __REG16 OCU								  							: 1;
  __REG16 																	: 6;
  __REG16 PORTSEL						  							: 2;
  __REG16 																	: 6;
} __ricfg0g0_adc0edgi_bits;

/* RICFG0G0_ADC0EDGIOCU0 */
typedef struct {
  __REG16 OCU00															: 1;
  __REG16 OCU01							  							: 1;
  __REG16 OCU10							  							: 1;
  __REG16 OCU11							  							: 1;
  __REG16 																	:12;
} __ricfg0g0_adc0edgiocu0_bits;

/* RICFG0G0_ADC0EDGIOCU4 */
typedef struct {
  __REG16 OCU160														: 1;
  __REG16 OCU161						  							: 1;
  __REG16 OCU170						  							: 1;
  __REG16 OCU171						  							: 1;
  __REG16 																	:12;
} __ricfg0g0_adc0edgiocu4_bits;

/* RICFG0G0_ADC0TIMI */
typedef struct {
  __REG16 RLT																: 1;
  __REG16 PPGL							  							: 1;
  __REG16 PPGH							  							: 1;
  __REG16 																	:13;
} __ricfg0g0_adc0timi_bits;

/* RICFG0G0_ADC0TIMIRLT */
typedef struct {
  __REG16 RLT																: 4;
  __REG16 																	:12;
} __ricfg0g0_adc0timirlt_bits;

/* RICFG0G0_ADC0ZPDEN */
typedef struct {
  __REG16 ZPDEN															: 1;
  __REG16 																	:15;
} __ricfg0g0_adc0zpden_bits;

/* RICFG0G1_FRT0TEXT */
typedef struct {
  __REG16 RESSEL														: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL														: 3;
  __REG16 																	: 5;
} __ricfg0g1_frt0text_bits;

/* RICFG0G1_FRT1TEXT */
typedef struct {
  __REG16 RESSEL														: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL														: 2;
  __REG16 																	: 6;
} __ricfg0g1_frt1text_bits;

/* RICFG0G1_FRT2TEXT */
typedef struct {
  __REG16 RESSEL														: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL														: 3;
  __REG16 																	: 5;
} __ricfg0g1_frt2text_bits;

/* RICFG0G1_FRT3TEXT */
typedef struct {
  __REG16 RESSEL														: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL														: 2;
  __REG16 																	: 6;
} __ricfg0g1_frt3text_bits;

/* RICFG0G2_CUxINx */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL														: 3;
  __REG16 																	: 5;
} __ricfg0g2_cuin_bits;

/* RICFG0G2_ICUxFRTSEL */
typedef struct {
  __REG16 RESSEL 														: 1;
  __REG16 																	:15;
} __ricfg0g2_icufrtsel_bits;

/* RICFG0G3_OCU0OTDxGATE */
typedef struct {
  __REG16 RESSEL 														: 3;
  __REG16 																	:13;
} __ricfg0g3_ocu0otdgate_bits;

/* RICFG0G3_OCU0OTDxGM */
typedef struct {
  __REG16 RESSEL 														: 1;
  __REG16 																	:15;
} __ricfg0g3_ocu0otdgm_bits;

/* RICFG0G3_OCU1CMP0EXT */
typedef struct {
  __REG16 RESSEL 														: 3;
  __REG16 																	:13;
} __ricfg0g3_ocu1cmp0ext_bits;

/* RICFG0G5_USART0SCKI */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL														: 3;
  __REG16 																	: 5;
} __ricfg0g5_usart0scki_bits;

/* RICFG0G7_PPGGRPxETRGx */
typedef struct {
  __REG16 RESSEL 														: 3;
  __REG16 																	:13;
} __ricfg0g7_ppgppgagate_bits;

/* RICFG0G7_PPGxPPGxGM */
typedef struct {
  __REG16 RESSEL 														: 1;
  __REG16 																	:15;
} __ricfg0g7_ppgppgagm_bits;

/* RICFG0G9_PPGGRPxETRGx */
typedef struct {
  __REG16 RESSEL 														: 3;
  __REG16 																	:13;
} __ricfg0g9_ppggrpetrg_bits;

/* RICFG1G1_CANxRX */
typedef struct {
  __REG16 																	: 8;
  __REG16 RESSEL 														: 3;
  __REG16 																	: 5;
} __ricfg1g1_canrx_bits;

/* RICFG1G3_FRTxxTEXT */
typedef struct {
  __REG16 RESSEL 														: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL														: 3;
  __REG16 																	: 5;
} __ricfg1g3_frttext_bits;

/* RICFG1G4_ICUxxINx */
typedef struct {
  __REG16 																	: 8;
  __REG16 RESSEL 														: 3;
  __REG16 																	: 5;
} __ricfg1g4_icuin_bits;

/* RICFG1G4_ICUxxFRTSEL */
typedef struct {
  __REG16 RESSEL 														: 1;
  __REG16 																	:15;
} __ricfg1g4_icufrtsel_bits;

/* RICFG1G5_OCUxxOTDxGATE */
typedef struct {
  __REG16 RESSEL 														: 3;
  __REG16 																	:13;
} __ricfg1g5_ocuotdgate_bits;

/* RICFG1G5_OCUxxOTDxGM */
typedef struct {
  __REG16 RESSEL  													: 1;
  __REG16 																	:15;
} __ricfg1g5_ocuotdgm_bits;

/* RICFG1G7_USART6SCKI */
typedef struct {
  __REG16 																	: 8;
  __REG16 PORTSEL  													: 3;
  __REG16 																	: 5;
} __ricfg1g7_usart6scki_bits;

/* RICFG1G9_PPGxxPPGAGATE */
typedef struct {
  __REG16 RESSEL  													: 3;
  __REG16 																	:13;
} __ricfg1g9_ppgppgagate_bits;

/* RICFG1G9_PPGxxPPGAGM */
typedef struct {
  __REG16 RESSEL  													: 1;
  __REG16 																	:15;
} __ricfg1g9_ppgppgagm_bits;

/* RICFG1G11_PPGGRPxxETRGx */
typedef struct {
  __REG16 RESSEL  													: 3;
  __REG16 																	:13;
} __ricfg1g11_ppggrpetrg_bits;

/* RICFG3G2_RLT0TIN */
typedef struct {
  __REG16 RESSEL  													: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL  													: 2;
  __REG16 																	: 6;
} __ricfg3g2_rlttin_bits;

/* RICFG3G4_UDC0xINx */
typedef struct {
  __REG16 RESSEL  													: 2;
  __REG16 																	: 6;
  __REG16 PORTSEL  													: 2;
  __REG16 																	: 6;
} __ricfg3g4_udc0in_bits;

/* RICFG3G4_UDC0ZINx */
typedef struct {
  __REG16 RESSEL  													: 3;
  __REG16 																	: 5;
  __REG16 PORTSEL  													: 2;
  __REG16 																	: 6;
} __ricfg3g4_udc0zin_bits;

/* MPUXGFX_CTRL0 */
typedef struct {
  __REG32 NMI		  													: 1;
  __REG32 NMICL	  													: 1;
  __REG32 			  													: 6;
  __REG32 LST		  													: 1;
  __REG32 MPUSTOP  													: 1;
  __REG32 MPUSTOPEN													: 1;
  __REG32 POEN 	  													: 1;
  __REG32 PROT 	  													: 1;
  __REG32 																	: 3;
  __REG32 MPUEN	  													: 1;
  __REG32 MPUENC  													: 1;
  __REG32 																	: 6;
  __REG32 AP      													: 3;
  __REG32 																	: 5;
} __mpuxgfx_ctrl0_bits;

/* MPUXGFX_NMIEN */
typedef struct {
  __REG32 NMIEN	  													: 1;
  __REG32 			  													:31;
} __mpuxgfx_nmien_bits;

/* MPUXGFX_WERRC */
typedef struct {
  __REG32 AWMPV   													: 1;
  __REG32 AWPROTPRIV 												: 1;
  __REG32 AWLEN   													: 4;
  __REG32 AWBURS   													: 2;
  __REG32 AWSIZE   													: 3;
  __REG32 			  													:21;
} __mpuxgfx_werrc_bits;

/* MPUXGFX_CTRL1 */
typedef struct {
  __REG32 MPUEN   													: 1;
  __REG32 MPUENC		 												: 1;
  __REG32 			  													: 6;
  __REG32 AP		   													: 3;
  __REG32 			  													:21;
} __mpuxgfx_ctrl1_bits;

/* ERCFG0_CSR */
typedef struct {
  __REG32 CEIEN   													: 1;
  __REG32 			  													: 7;
  __REG32 CEIF			 												: 1;
  __REG32 LCK				 												: 1;
  __REG32 			  													: 6;
  __REG32 CEIC			 												: 1;
  __REG32 			  													: 7;
  __REG32 RAWC1			 												: 2;
  __REG32 WAWC1			 												: 2;
  __REG32 RAWC2			 												: 2;
  __REG32 WAWC2			 												: 2;
} __ercfg0_csr_bits;

/* ERCFG0_ERRMSKR1 */
typedef struct {
  __REG32 MSK	   														: 7;
  __REG32 			  													:25;
} __ercfg0_errmskr1_bits;

/* ERCFG0_ECCEN */
typedef struct {
  __REG32 ECCEN	   													: 1;
  __REG32 			  													:31;
} __ercfg0_eccen_bits;

/* ARH0_RHCTRL */
typedef struct {
  __REG32 LST		   													: 1;
  __REG32 			  													: 7;
  __REG32 EV		   													: 1;
  __REG32 OFL		   													: 1;
  __REG32 LV		   													: 1;
  __REG32 			  													: 1;
  __REG32 FAT0	   													: 1;
  __REG32 FAT1	   													: 1;
  __REG32 WDG0	   													: 1;
  __REG32 WDG1	   													: 1;
  __REG32 			  													: 8;
  __REG32 TBNO	   													: 4;
  __REG32 			  													: 2;
  __REG32 CANCEL  													: 1;
  __REG32 UNLOCK  													: 1;
} __arh0_rhctrl_bits;

/* ARH0_CHCTRL0 */
typedef struct {
  __REG32 REMOTERSTCL												: 1;
  __REG32 READYCL														: 1;
  __REG32 PERRORCL 													: 1;
  __REG32 CRCTOUTCL													: 1;
  __REG32 CRCERRCL 													: 1;
  __REG32 FATIRQCL													: 1;
  __REG32 DNVALIDST													: 1;
  __REG32 UPVALIDCL													: 1;
  __REG32 REMOTERST													: 1;
  __REG32 READY   													: 1;
  __REG32 PERROR   													: 1;
  __REG32 CRCTOUT  													: 1;
  __REG32 CRCERR   													: 1;
  __REG32 FATIRQ	 													: 1;
  __REG32 DNVALID  													: 1;
  __REG32 UPVALID  													: 1;
  __REG32 CONNECTED													: 1;
  __REG32 UPRDY															: 1;
  __REG32 FATAL															: 1;
  __REG32 DNHSK															: 1;
  __REG32 UPHSK															: 1;
  __REG32 PLLGOOD														: 1;
  __REG32 																	: 2;
  __REG32 INITRH														: 1;
  __REG32 RSTRTA														: 1;
  __REG32 TXCFG 														: 1;
  __REG32 																	: 3;
  __REG32 FATEIN 														: 1;
  __REG32 BYPASS 														: 1;
} __arh0_chctrl0_bits;

/* ARH0_CHSTAT0 */
typedef struct {
  __REG32 PLLBAD   													: 8;
  __REG32 UPSYNC   													: 8;
  __REG32 			  													: 8;
  __REG32 UPCRC   													: 8;
} __arh0_chstat0_bits;

/* ARH0_CHWDGCTL0 */
typedef struct {
  __REG32 WDRXIRQ0CL												: 1;
  __REG32 WDRXIRQ1CL												: 1;
  __REG32 WDRXIRQ2CL												: 1;
  __REG32 WDRXIRQ3CL												: 1;
  __REG32 WDTXIRQ0CL												: 1;
  __REG32 WDTXIRQ1CL												: 1;
  __REG32 WDTXIRQ2CL												: 1;
  __REG32 WDTXIRQ3CL												: 1;
  __REG32 WDRXIRQ0													: 1;
  __REG32 WDRXIRQ1													: 1;
  __REG32 WDRXIRQ2													: 1;
  __REG32 WDRXIRQ3													: 1;
  __REG32 WDTXIRQ0													: 1;
  __REG32 WDTXIRQ1													: 1;
  __REG32 WDTXIRQ2													: 1;
  __REG32 WDTXIRQ3													: 1;
  __REG32 			  													: 8;
  __REG32 WTRX	   													: 2;
  __REG32 WTTX	   													: 2;
  __REG32 			  													: 2;
  __REG32 WDRXIEN  													: 1;
  __REG32 WDTXIEN  													: 1;
} __arh0_chwdgctl0_bits;

/* ARH0_CHWDGCNT0 */
typedef struct {
  __REG32 CNT																:16;
  __REG32 			  													:16;
} __arh0_chwdgcnt0_bits;

/* ARH0_CHCTRL1 */
typedef struct {
  __REG32 REMOTERSTCL												: 1;
  __REG32 READYCL														: 1;
  __REG32 PERRORCL													: 1;
  __REG32 CRCTOUTCL													: 1;
  __REG32 CRCERRCL													: 1;
  __REG32 FATIRQCL													: 1;
  __REG32 DNVALIDST													: 1;
  __REG32 UPVALIDCL													: 1;
  __REG32 REMOTERST													: 1;
  __REG32 READY															: 1;
  __REG32 PERROR														: 1;
  __REG32 CRCTOUT														: 1;
  __REG32 CRCERR														: 1;
  __REG32 FATIRQ														: 1;
  __REG32 DNVALID														: 1;
  __REG32 UPVALID														: 1;
  __REG32 CONNECTED													: 1;
  __REG32 UPRDY															: 1;
  __REG32 FATAL															: 1;
  __REG32 DNHSK															: 1;
  __REG32 UPHSK															: 1;
  __REG32 PLLGOOD														: 1;
  __REG32 																	: 2;
  __REG32 INITRH														: 1;
  __REG32 RSTRTA														: 1;
  __REG32 TXCFG															: 1;
  __REG32 			  													: 3;
  __REG32 FATEIN														: 1;
  __REG32 BYPASS														: 1;
} __arh0_chctrl1_bits;

/* ARH0_CHSTAT1 */
typedef struct {
  __REG32 PLLBAD  													: 8;
  __REG32 UPSYNC  													: 8;
  __REG32 			  													: 8;
  __REG32 UPCRC	  													: 8;
} __arh0_chstat1_bits;

/* ARH0_CHWDGCTL1 */
typedef struct {
  __REG32 WDRXIRQ0CL 												: 1;
  __REG32 WDRXIRQ1CL												: 1;
  __REG32 WDRXIRQ2CL												: 1;
  __REG32 WDRXIRQ3CL												: 1;
  __REG32 WDTXIRQ0CL												: 1;
  __REG32 WDTXIRQ1CL												: 1;
  __REG32 WDTXIRQ2CL												: 1;
  __REG32 WDTXIRQ3CL												: 1;
  __REG32 WDRXIRQ0 													: 1;
  __REG32 WDRXIRQ1 													: 1;
  __REG32 WDRXIRQ2 													: 1;
  __REG32 WDRXIRQ3 													: 1;
  __REG32 WDTXIRQ0 													: 1;
  __REG32 WDTXIRQ1 													: 1;
  __REG32 WDTXIRQ2 													: 1;
  __REG32 WDTXIRQ3 													: 1;
  __REG32 			  													: 8;
  __REG32 WTRX	  													: 2;
  __REG32 WTTX	  													: 2;
  __REG32 			  													: 2;
  __REG32 WDRXIEN  													: 1;
  __REG32 WDTXIEN  													: 1;
} __arh0_chwdgctl1_bits;

/* ARH0_CHWDGCNT1 */
typedef struct {
  __REG32 CNT				 												:16;
  __REG32 			  													:16;
} __arh0_chwdgcnt1_bits;

/* ARH0_TBCTRLx */
typedef struct {
  __REG32 TBIRQCL		 												: 1;
  __REG32 CANCELEDCL 												: 1;
  __REG32 UNLOCKEDCL												: 1;
  __REG32 			  													: 5;
  __REG32 TBIRQ			 												: 1;
  __REG32 PENDING		 												: 1;
  __REG32 WAITING		 												: 1;
  __REG32 CANCELED	 												: 1;
  __REG32 UNLOCKED	 												: 1;
  __REG32 ACTIVE		 												: 1;
  __REG32 			  													: 2;
  __REG32 TBIEN			 												: 1;
  __REG32 TBDEN			 												: 1;
  __REG32 			  													:10;
  __REG32 TBIMD			 												: 1;
  __REG32 TBACT			 												: 1;
  __REG32 TBAINC		 												: 1;
  __REG32 TBCH			 												: 1;
} __arh0_tbctrl_bits;

/* ARH0_TBIDXx */
typedef struct {
  __REG8  TBIDX			 												: 5;
  __REG8  					 												: 3;
} __arh0_tbidx_bits;

/* ARH0_TFCTRLx */
typedef struct {
  __REG8  RW				 												: 1;
  __REG8  OAEN			 												: 1;
  __REG8  SZ				 												: 2;
  __REG8  ERROR 		 												: 1;
  __REG8  					 												: 1;
  __REG8  TFAINV		 												: 1;
  __REG8  TFDASWP		 												: 1;
} __arh0_tfctrl_bits;

/* ARH0_TFADDRx */
typedef struct {
  __REG32 ADDR			 												:20;
  __REG32 					 												:12;
} __arh0_tfaddr_bits;

/* ARH0_EVCTRL */
typedef struct {
  __REG32 LEVEL			 												: 8;
  __REG32 STATUS		 												: 8;
  __REG32 					 												:14;
  __REG32 FRST			 												: 1;
  __REG32 MODE			 												: 1;
} __arh0_evctrl_bits;

/* ARH0_EVIRQC */
typedef struct {
  __REG32 					 												:13;
  __REG32 EVIRQCL 	 												: 1;
  __REG32 OFLIRQCL	 												: 1;
  __REG32 LVIRQCL		 												: 1;
  __REG32 					 												: 5;
  __REG32 EVIRQ			 												: 1;
  __REG32 OFLIRQ 		 												: 1;
  __REG32 LVIRQ			 												: 1;
  __REG32 EVIEN			 												: 1;
  __REG32 					 												: 5;
  __REG32 OFLIEN 		 												: 1;
  __REG32 LVIEN 		 												: 1;
} __arh0_evirqc_bits;

/* ARH0_EVBUF0 */
typedef struct {
  __REG32 					 												:16;
  __REG32 EVIDX	 	 													: 8;
  __REG32 EVCH			 												: 1;
  __REG32 					 												: 7;
} __arh0_evbuf0_bits;

/* ARH0_APCFG00 */
typedef struct {
  __REG32 CONFIGBYTE4												: 8;
  __REG32 CONFIGBYTE3												: 8;
  __REG32 CONFIGBYTE2												: 8;
  __REG32 CONFIGBYTE1												: 8;
} __arh0_apcfg00_bits;

/* ARH0_APCFG01 */
typedef struct {
  __REG32 CONFIGBYTE8												: 8;
  __REG32 CONFIGBYTE7												: 8;
  __REG32 CONFIGBYTE6												: 8;
  __REG32 CONFIGBYTE5												: 8;
} __arh0_apcfg01_bits;

/* ARH0_APCFG02 */
typedef struct {
  __REG32 SCPREEN 													: 1;
  __REG32 				 													: 7;
  __REG32 CONFIGBYTE11											: 8;
  __REG32 CONFIGBYTE10											: 8;
  __REG32 CONFIGBYTE9												: 8;
} __arh0_apcfg02_bits;

/* ARH0_APCFG03 */
typedef struct {
  __REG32 CONFIGBYTESHELL4 									: 8;
  __REG32 CONFIGBYTESHELL3 									: 8;
  __REG32 CONFIGBYTESHELL2 									: 8;
  __REG32 CONFIGBYTESHELL1 									: 8;
} __arh0_apcfg03_bits;

/* ARH0_APCFG10 */
typedef struct {
  __REG32 CONFIGBYTE4												: 8;
  __REG32 CONFIGBYTE3												: 8;
  __REG32 CONFIGBYTE2												: 8;
  __REG32 CONFIGBYTE1												: 8;
} __arh0_apcfg10_bits;

/* ARH0_APCFG11 */
typedef struct {
  __REG32 CONFIGBYTE8												: 8;
  __REG32 CONFIGBYTE7												: 8;
  __REG32 CONFIGBYTE6												: 8;
  __REG32 CONFIGBYTE5												: 8;
} __arh0_apcfg11_bits;

/* ARH0_APCFG12 */
typedef struct {
  __REG32 				 													: 8;
  __REG32 CONFIGBYTE11											: 8;
  __REG32 CONFIGBYTE10											: 8;
  __REG32 CONFIGBYTE9												: 8;
} __arh0_apcfg12_bits;

/* ARH0_APCFG13 */
typedef struct {
  __REG32 CONFIGBYTESHELL4 									: 8;
  __REG32 CONFIGBYTESHELL3 									: 8;
  __REG32 CONFIGBYTESHELL2 									: 8;
  __REG32 CONFIGBYTESHELL1 									: 8;
} __arh0_apcfg13_bits;

/* ARH0_TST */
typedef struct {
  __REG32 ADDR						 									: 1;
  __REG32 								 									: 7;
  __REG32 TM							 									: 1;
  __REG32 RW							 									: 1;
  __REG32 								 									:22;
} __arh0_tst_bits;

/* RICFG4G1_ETH0COL */
typedef struct {
  __REG16 								 									: 8;
  __REG16 PORTSEL					 									: 1;
  __REG16 								 									: 7;
} __ricfg4g1_eth0col_bits;

/* RICFG4G4_I2SxECLK */
typedef struct {
  __REG16 RESSEL					 									: 1;
  __REG16 								 									: 7;
  __REG16 PORTSEL					 									: 2;
  __REG16 								 									: 6;
} __ricfg4g4_i2seclk_bits;

/* RICFG4G4_I2SxSCKI */
typedef struct {
  __REG16 								 									: 8;
  __REG16 PORTSEL					 									: 2;
  __REG16 								 									: 6;
} __ricfg4g4_i2sscki_bits;

/* RICFG4G7_SPIxCLKI */
typedef struct {
  __REG16 								 									: 8;
  __REG16 PORTSEL					 									: 2;
  __REG16 								 									: 6;
} __ricfg4g7_spiclki_bits;

/* RICFG4G7_SPI0MSTART */
typedef struct {
  __REG16 RESSEL					 									: 3;
  __REG16 								 									:13;
} __ricfg4g7_spimstart_bits;

/* RICFG4G7_SPI2DATA2I */
typedef struct {
  __REG16 								 									: 8;
  __REG16 PORTSEL					 									: 1;
  __REG16 								 									: 7;
} __ricfg4g7_spidata2i_bits;

/* RICFG4G8_ARH0AIC0RCK */
typedef struct {
  __REG16 								 									: 8;
  __REG16 PORTSEL					 									: 1;
  __REG16 								 									: 7;
} __ricfg4g8_arh0aicrck_bits;

/* DMA0_Ax */
typedef struct {
  __REG32 TC							 									:16;
  __REG32 TO							 									: 4;
  __REG32 BC							 									: 4;
  __REG32 BL							 									: 2;
  __REG32 AL							 									: 1;
  __REG32 IS							 									: 2;
  __REG32 ST							 									: 1;
  __REG32 PB							 									: 1;
  __REG32 EB							 									: 1;
} __dma_a_bits;

/* DMA0_Bx */
typedef struct {
  __REG32 PN							 									: 7;
  __REG32 								 									: 1;
  __REG32 DP							 									: 4;
  __REG32 SP							 									: 4;
  __REG32 SS							 									: 3;
  __REG32 CI							 									: 1;
  __REG32 EI							 									: 1;
  __REG32 								 									: 4;
  __REG32 SR							 									: 1;
  __REG32 TW							 									: 2;
  __REG32 MS							 									: 2;
  __REG32 EQ							 									: 1;
  __REG32 DQ							 									: 1;
} __dma_b_bits;

/* DMA0_Cx */
typedef struct {
  __REG32 CD							 									: 1;
  __REG32 								 									: 7;
  __REG32 CE							 									: 1;
  __REG32 								 									:23;
} __dma_c_bits;

/* DMA0_Dx */
typedef struct {
  __REG32 								 									:12;
  __REG32 FBD							 									: 1;
  __REG32 UD							 									: 1;
  __REG32 DED							 									: 1;
  __REG32 FD							 									: 1;
  __REG32 								 									:12;
  __REG32 FBS							 									: 1;
  __REG32 US							 									: 1;
  __REG32 DES							 									: 1;
  __REG32 FS							 									: 1;
} __dma_d_bits;

/* DMA0_R */
typedef struct {
  __REG32 DSHS						 									: 1;
  __REG32 								 									:23;
  __REG32 DB							 									: 2;
  __REG32 DH							 									: 1;
  __REG32 PR							 									: 2;
  __REG32 DBE							 									: 1;
  __REG32 DSHR						 									: 1;
  __REG32 DE							 									: 1;
} __dma_r_bits;

/* DMA0_CMECICx */
typedef struct {
  __REG32 ENCI						 									: 1;
  __REG32 ENSTP						 									: 1;
  __REG32 ENEOP						 									: 1;
  __REG32 								 									:13;
  __REG32 LVLREQ					 									: 1;
 	__REG32 LVLREQACK				 									: 1;
  __REG32 LVLSTP					 									: 1;
  __REG32 LVLSTPACK				 									: 1;
  __REG32 LVLEOP					 									: 1;
  __REG32 LVLEOPACK				 									: 1;
  __REG32 								 									: 3;
  __REG32 BEHREQACK				 									: 1;
  __REG32 								 									: 1;
  __REG32 BEHSTPACK				 									: 1;
  __REG32 								 									: 4;
} __dma_cmecic_bits;

/* DMA0_CMICICx */
typedef struct {
  __REG32 								 									:25;
  __REG32 DMA0_CMICIC0		 									: 1;
  __REG32 								 									: 1;
  __REG32 BEHSTPACK				 									: 1;
  __REG32 								 									: 4;
} __dma_cmicic_bits;

/* DMA0_CMCHICx */
typedef struct {
  __REG32 CI							 									: 9;
  __REG32 								 									:23;
} __dma_cmchic_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/
/***************************************************************************
 **
 ** SYSC
 **
 ***************************************************************************/
__IO_REG32(    SYSC_PROTKEYR,     0xB0600000,__READ_WRITE );
__IO_REG16_BIT(SYSC_RUNPDCFGR,    0xB0600080,__READ_WRITE ,__sysc_runpdcfgr_bits);
__IO_REG16_BIT(SYSC_RUNCKSRER,    0xB0600082,__READ_WRITE ,__sysc_runcksrer_bits);
__IO_REG32_BIT(SYSC_RUNCKSELR,    0xB0600084,__READ_WRITE ,__sysc_runckselr_bits);
__IO_REG32_BIT(SYSC_RUNCKER,      0xB0600088,__READ_WRITE ,__sysc_runcker_bits);
__IO_REG32_BIT(SYSC_RUNCKDIVR0,   0xB060008C,__READ_WRITE ,__sysc_runckdivr0_bits);
__IO_REG32_BIT(SYSC_RUNCKDIVR1,   0xB0600090,__READ_WRITE ,__sysc_runckdivr1_bits);
__IO_REG32_BIT(SYSC_RUNCKDIVR2,   0xB0600094,__READ_WRITE ,__sysc_runckdivr2_bits);
__IO_REG32_BIT(SYSC_RUNPLLCNTR,   0xB0600098,__READ_WRITE ,__sysc_runpllcntr_bits);
__IO_REG32_BIT(SYSC_RUNSSCGCNTR0, 0xB060009C,__READ_WRITE ,__sysc_runsscgcntr0_bits);
__IO_REG32_BIT(SYSC_RUNSSCGCNTR1, 0xB06000A0,__READ_WRITE ,__sysc_runsscgcntr1_bits);
__IO_REG32_BIT(SYSC_RUNGFXCNTR0,  0xB06000A4,__READ_WRITE ,__sysc_rungfxcntr0_bits);
__IO_REG32_BIT(SYSC_RUNGFXCNTR1,  0xB06000A8,__READ_WRITE ,__sysc_rungfxcntr1_bits);
__IO_REG32_BIT(SYSC_RUNLVDCFGR,   0xB06000AC,__READ_WRITE ,__sysc_runlvdcfgr_bits);
__IO_REG16_BIT(SYSC_RUNCSVCFGR,   0xB06000B0,__READ_WRITE ,__sysc_runcsvcfgr_bits);
__IO_REG16_BIT(SYSC_TRGRUNCNTR,   0xB06000B2,__READ_WRITE ,__sysc_trgruncntr_bits);
__IO_REG16_BIT(SYSC_PSSPDCFGR,    0xB0600100,__READ_WRITE ,__sysc_psspdcfgr_bits);
__IO_REG16_BIT(SYSC_PSSCKSRER,    0xB0600102,__READ_WRITE ,__sysc_psscksrer_bits);
__IO_REG32_BIT(SYSC_PSSCKSELR,    0xB0600104,__READ_WRITE ,__sysc_pssckselr_bits);
__IO_REG32_BIT(SYSC_PSSCKER,      0xB0600108,__READ_WRITE ,__sysc_psscker_bits);
__IO_REG32_BIT(SYSC_PSSCKDIVR0,   0xB060010C,__READ_WRITE ,__sysc_pssckdivr0_bits);
__IO_REG32_BIT(SYSC_PSSCKDIVR1,   0xB0600110,__READ_WRITE ,__sysc_pssckdivr1_bits);
__IO_REG32_BIT(SYSC_PSSCKDIVR2,   0xB0600114,__READ_WRITE ,__sysc_pssckdivr2_bits);
__IO_REG32_BIT(SYSC_PSSPLLCNTR,   0xB0600118,__READ_WRITE ,__sysc_psspllcntr_bits);
__IO_REG32_BIT(SYSC_PSSSSCGCNTR0, 0xB060011C,__READ_WRITE ,__sysc_psssscgcntr0_bits);
__IO_REG32_BIT(SYSC_PSSSSCGCNTR1, 0xB0600120,__READ_WRITE ,__sysc_psssscgcntr1_bits);
__IO_REG32_BIT(SYSC_PSSGFXCNTR0,  0xB0600124,__READ_WRITE ,__sysc_pssgfxcntr0_bits);
__IO_REG32_BIT(SYSC_PSSGFXCNTR1,  0xB0600128,__READ_WRITE ,__sysc_pssgfxcntr1_bits);
__IO_REG32_BIT(SYSC_PSSLVDCFGR,   0xB060012C,__READ_WRITE ,__sysc_psslvdcfgr_bits);
__IO_REG16_BIT(SYSC_PSSCSVCFGR,   0xB0600130,__READ_WRITE ,__sysc_psscsvcfgr_bits);
__IO_REG16_BIT(SYSC_PSSENR,       0xB0600132,__READ_WRITE ,__sysc_pssenr_bits);
__IO_REG16_BIT(SYSC_APPPDCFGR,    0xB0600180,__READ       ,__sysc_apppdcfgr_bits);
__IO_REG16_BIT(SYSC_APPCKSRER,    0xB0600182,__READ       ,__sysc_appcksrer_bits);
__IO_REG32_BIT(SYSC_APPCKSELR,    0xB0600184,__READ       ,__sysc_appckselr_bits);
__IO_REG32_BIT(SYSC_APPCKER,      0xB0600188,__READ       ,__sysc_appcker_bits);
__IO_REG32_BIT(SYSC_APPCKDIVR0,   0xB060018C,__READ       ,__sysc_appckdivr0_bits);
__IO_REG32_BIT(SYSC_APPCKDIVR1,   0xB0600190,__READ       ,__sysc_appckdivr1_bits);
__IO_REG32_BIT(SYSC_APPCKDIVR2,   0xB0600194,__READ       ,__sysc_appckdivr2_bits);
__IO_REG32_BIT(SYSC_APPPLLCNTR,   0xB0600198,__READ       ,__sysc_apppllcntr_bits);
__IO_REG32_BIT(SYSC_APPSSCGCNTR0, 0xB060019C,__READ       ,__sysc_appsscgcntr0_bits);
__IO_REG32_BIT(SYSC_APPSSCGCNTR1, 0xB06001A0,__READ       ,__sysc_appsscgcntr1_bits);
__IO_REG32_BIT(SYSC_APPGFXCNTR0,  0xB06001A4,__READ       ,__sysc_appgfxcntr0_bits);
__IO_REG32_BIT(SYSC_APPGFXCNTR1,  0xB06001A8,__READ       ,__sysc_appgfxcntr1_bits);
__IO_REG32_BIT(SYSC_APPLVDCFGR,   0xB06001AC,__READ       ,__sysc_applvdcfgr_bits);
__IO_REG16_BIT(SYSC_APPCSVCFGR,   0xB06001B0,__READ       ,__sysc_appcsvcfgr_bits);
__IO_REG16_BIT(SYSC_PDSTSR,       0xB0600200,__READ       ,__sysc_pdstsr_bits);
__IO_REG16_BIT(SYSC_CKSRESTSR,    0xB0600202,__READ       ,__sysc_cksrestsr_bits);
__IO_REG32_BIT(SYSC_CKSELSTSR,    0xB0600204,__READ       ,__sysc_ckselstsr_bits);
__IO_REG32_BIT(SYSC_CKESTSR,      0xB0600208,__READ       ,__sysc_ckestsr_bits);
__IO_REG32_BIT(SYSC_CKDIVSTSR0,   0xB060020C,__READ       ,__sysc_ckdivstsr0_bits);
__IO_REG32_BIT(SYSC_CKDIVSTSR1,   0xB0600210,__READ       ,__sysc_ckdivstsr1_bits);
__IO_REG32_BIT(SYSC_CKDIVSTSR2,   0xB0600214,__READ       ,__sysc_ckdivstsr2_bits);
__IO_REG32_BIT(SYSC_PLLSTSR,      0xB0600218,__READ       ,__sysc_pllstsr_bits);
__IO_REG32_BIT(SYSC_SSCGSTSR0,    0xB060021C,__READ       ,__sysc_sscgstsr0_bits);
__IO_REG32_BIT(SYSC_SSCGSTSR1,    0xB0600220,__READ       ,__sysc_sscgstsr1_bits);
__IO_REG32_BIT(SYSC_GFXSTSR0,     0xB0600224,__READ       ,__sysc_gfxstsr0_bits);
__IO_REG32_BIT(SYSC_GFXSTSR1,     0xB0600228,__READ       ,__sysc_gfxstsr1_bits);
__IO_REG32_BIT(SYSC_LVDCFGSTSR,   0xB060022C,__READ       ,__sysc_lvdcfgstsr_bits);
__IO_REG16_BIT(SYSC_CSVCFGSTSR,   0xB0600230,__READ       ,__sysc_csvcfgstsr_bits);
__IO_REG32(    SYSC_SYSIDR,       0xB0600280,__READ       );
__IO_REG32_BIT(SYSC_SYSSTSR,      0xB0600284,__READ       ,__sysc_sysstsr_bits);
__IO_REG32_BIT(SYSC_SYSINTER,     0xB0600288,__READ_WRITE ,__sysc_sysinter_bits);
__IO_REG32_BIT(SYSC_SYSICLR,      0xB060028C,__READ_WRITE ,__sysc_sysiclr_bits);
__IO_REG32_BIT(SYSC_SYSERRR,      0xB0600290,__READ_WRITE ,__sysc_syserrr_bits);
__IO_REG32_BIT(SYSC_SYSERRICLR,   0xB0600294,__READ_WRITE ,__sysc_syserriclr_bits);
__IO_REG32_BIT(SYSC_CSVMOCFGR,    0xB0600300,__READ_WRITE ,__sysc_csvmocfgr_bits);
__IO_REG32_BIT(SYSC_CSVSOCFGR,    0xB0600304,__READ_WRITE ,__sysc_csvmocfgr_bits);
__IO_REG32_BIT(SYSC_CSVMPCFGR,    0xB0600308,__READ_WRITE ,__sysc_csvmocfgr_bits);
__IO_REG32_BIT(SYSC_CSVSPCFGR,    0xB060030C,__READ_WRITE ,__sysc_csvmocfgr_bits);
__IO_REG32_BIT(SYSC_CSVGPCFGR,    0xB0600310,__READ_WRITE ,__sysc_csvmocfgr_bits);
__IO_REG32_BIT(SYSC_CSVTESTR,     0xB0600314,__READ_WRITE ,__sysc_csvtestr_bits);
__IO_REG32_BIT(SYSC_RSTCNTR,      0xB0600380,__READ_WRITE ,__sysc_rstcntr_bits);
__IO_REG32_BIT(SYSC_RSTCAUSEUR,   0xB0600384,__READ_WRITE ,__sysc_rstcauseur_bits);
__IO_REG32_BIT(SYSC_RSTCAUSEBT,   0xB0600388,__READ_WRITE ,__sysc_rstcauseur_bits);
__IO_REG32_BIT(SYSC_SRCSCTTRG,    0xB0600400,__READ_WRITE ,__sysc_srcscttrg_bits);
__IO_REG32_BIT(SYSC_SRCSCTCNTR,   0xB0600404,__READ_WRITE ,__sysc_srcsctcntr_bits);
__IO_REG32_BIT(SYSC_SRCSCTCPR,    0xB0600408,__READ_WRITE ,__sysc_srcsctcpr_bits);
__IO_REG32_BIT(SYSC_SRCSCTSTATR,  0xB060040C,__READ_WRITE ,__sysc_srcsctstatr_bits);
__IO_REG32_BIT(SYSC_SRCSCTINTER,  0xB0600410,__READ_WRITE ,__sysc_srcsctinter_bits);
__IO_REG32_BIT(SYSC_SRCSCTICLR,   0xB0600414,__READ_WRITE ,__sysc_srcscticlr_bits);
__IO_REG32_BIT(SYSC_RCSCTTRG,     0xB0600480,__READ_WRITE ,__sysc_rcscttrg_bits);
__IO_REG32_BIT(SYSC_RCSCTCNTR,    0xB0600484,__READ_WRITE ,__sysc_rcsctcntr_bits);
__IO_REG32_BIT(SYSC_RCSCTCPR,     0xB0600488,__READ_WRITE ,__sysc_rcsctcpr_bits);
__IO_REG32_BIT(SYSC_RCSCSTAT,     0xB060048C,__READ_WRITE ,__sysc_rcscstat_bits);
__IO_REG32_BIT(SYSC_RCSCTINTER,   0xB0600490,__READ_WRITE ,__sysc_rcsctinter_bits);
__IO_REG32_BIT(SYSC_RCSCTICLR,    0xB0600494,__READ_WRITE ,__sysc_rcscticlr_bits);
__IO_REG32_BIT(SYSC_MAINSCTTRG,   0xB0600500,__READ_WRITE ,__sysc_mainscttrg_bits);
__IO_REG32_BIT(SYSC_MAINSCTCNTR,  0xB0600504,__READ_WRITE ,__sysc_mainsctcntr_bits);
__IO_REG32_BIT(SYSC_MAINSCTCPR,   0xB0600508,__READ_WRITE ,__sysc_mainsctcpr_bits);
__IO_REG32_BIT(SYSC_MAINSCTSTATR, 0xB060050C,__READ_WRITE ,__sysc_mainsctstatr_bits);
__IO_REG32_BIT(SYSC_MAINSCTINTER, 0xB0600510,__READ_WRITE ,__sysc_mainsctinter_bits);
__IO_REG32_BIT(SYSC_MAINSCTICLR,  0xB0600514,__READ_WRITE ,__sysc_mainscticlr_bits);
__IO_REG32_BIT(SYSC_SUBSCTTRG,    0xB0600580,__READ_WRITE ,__sysc_subscttrg_bits);
__IO_REG32_BIT(SYSC_SUBSCTCNTR,   0xB0600584,__READ_WRITE ,__sysc_subsctcntr_bits);
__IO_REG32_BIT(SYSC_SUBSCTCPR,    0xB0600588,__READ_WRITE ,__sysc_subsctcpr_bits);
__IO_REG32_BIT(SYSC_SUBSCTSTATR,  0xB060058C,__READ_WRITE ,__sysc_subsctstatr_bits);
__IO_REG32_BIT(SYSC_SUBSCTINTER,  0xB0600590,__READ_WRITE ,__sysc_subsctinter_bits);
__IO_REG32_BIT(SYSC_SUBSCTICLR,   0xB0600594,__READ_WRITE ,__sysc_subscticlr_bits);
__IO_REG32_BIT(SYSC_CKOTCFGR,     0xB0600600,__READ_WRITE ,__sysc_ckotcfgr_bits);
__IO_REG32_BIT(SYSC_SPCCFGR,      0xB0600680,__READ_WRITE ,__sysc_spccfgr_bits);
__IO_REG32_BIT(SYSC_RCCFGR,       0xB0600684,__READ_WRITE ,__sysc_rccfgr_bits);
__IO_REG32(    SYSC_TESTR0,       0xB0600688,__READ_WRITE );
__IO_REG32(    SYSC_TESTR1,       0xB060068C,__READ_WRITE );
__IO_REG32(    SYSC_TESTR2,       0xB0600690,__READ_WRITE );
__IO_REG32_BIT(SYSC_JTAGDETECT,   0xB0600700,__READ       ,__sysc_jtagdetect_bits);
__IO_REG32_BIT(SYSC_JTAGCNFG,     0xB0600704,__READ_WRITE ,__sysc_jtagcnfg_bits);
__IO_REG32_BIT(SYSC_JTAGWAKEUP,   0xB0600708,__READ_WRITE ,__sysc_jtagwakeup_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_WTCR,          0xB0618000,__READ_WRITE ,__rtc_wtcr_bits);
__IO_REG32_BIT(RTC_WTSR,          0xB0618004,__READ_WRITE ,__rtc_wtsr_bits);
__IO_REG32_BIT(RTC_WINS,          0xB0618008,__READ_WRITE ,__rtc_wins_bits);
__IO_REG32_BIT(RTC_WINE,          0xB061800C,__READ_WRITE ,__rtc_wine_bits);
__IO_REG32_BIT(RTC_WINC,          0xB0618010,__READ_WRITE ,__rtc_winc_bits);
__IO_REG32_BIT(RTC_WTBR,          0xB0618014,__READ_WRITE ,__rtc_wtbr_bits);
__IO_REG32_BIT(RTC_WRT,           0xB0618018,__READ_WRITE ,__rtc_wrt_bits);
__IO_REG32_BIT(RTC_CNTCAL,        0xB061801C,__READ_WRITE ,__rtc_cntcal_bits);
__IO_REG32_BIT(RTC_CNTPCAL,       0xB0618020,__READ_WRITE ,__rtc_cntpcal_bits);
__IO_REG32_BIT(RTC_DURMW,         0xB0618024,__READ_WRITE ,__rtc_durmw_bits);
__IO_REG32_BIT(RTC_CALTRG,        0xB0618028,__READ_WRITE ,__rtc_caltrg_bits);
__IO_REG32_BIT(RTC_DEBUG,         0xB061802C,__READ_WRITE ,__rtc_debug_bits);

/***************************************************************************
 **
 ** WDG
 **
 ***************************************************************************/
__IO_REG32(    WDG_PROT,          0xB0608000,__READ_WRITE );
__IO_REG32(    WDG_CNT,           0xB0608008,__READ       );
__IO_REG32_BIT(WDG_RSTCAUSE,      0xB060800C,__READ_WRITE ,__wdg_rstcause_bits);
__IO_REG32_BIT(WDG_TRG0,          0xB0608010,__READ_WRITE ,__wdg_trg0_bits);
__IO_REG32_BIT(WDG_TRG1,          0xB0608018,__READ_WRITE ,__wdg_trg1_bits);
__IO_REG32_BIT(WDG_INT,           0xB0608020,__READ       ,__wdg_int_bits);
__IO_REG32_BIT(WDG_INTCLR,        0xB0608024,__READ_WRITE ,__wdg_intclr_bits);
__IO_REG32_BIT(WDG_TRG0CFG,       0xB060802C,__READ_WRITE ,__wdg_trg0cfg_bits);
__IO_REG32_BIT(WDG_TRG1CFG,       0xB0608030,__READ_WRITE ,__wdg_trg1cfg_bits);
__IO_REG32(    WDG_RUNLL,         0xB0608034,__READ_WRITE );
__IO_REG32(    WDG_RUNUL,         0xB0608038,__READ_WRITE );
__IO_REG32(    WDG_PSSLL,         0xB060803C,__READ_WRITE );
__IO_REG32(    WDG_PSSUL,         0xB0608040,__READ_WRITE );
__IO_REG32_BIT(WDG_RSTDLY,        0xB0608044,__READ_WRITE ,__wdg_rstdly_bits);
__IO_REG32_BIT(WDG_CFG,           0xB0608048,__READ_WRITE ,__wdg_cfg_bits);

/***************************************************************************
 **
 ** SC
 **
 ***************************************************************************/
__IO_REG32(    SCCFG_TCFPUSRKEY0, 0xB050F120,__WRITE      );
__IO_REG32(    SCCFG_TCFPUSRKEY1, 0xB050F124,__WRITE      );
__IO_REG32(    SCCFG_TCFPUSRKEY2, 0xB050F128,__WRITE      );
__IO_REG32(    SCCFG_TCFPUSRKEY3, 0xB050F12C,__WRITE      );
__IO_REG32(    SCCFG_EEFPUSRKEY0, 0xB050F130,__WRITE      );
__IO_REG32(    SCCFG_EEFPUSRKEY1, 0xB050F134,__WRITE      );
__IO_REG32(    SCCFG_EEFPUSRKEY2, 0xB050F138,__WRITE      );
__IO_REG32(    SCCFG_EEFPUSRKEY3, 0xB050F13C,__WRITE      );
__IO_REG32_BIT(SCCFG_CTRL,        0xB050F170,__READ_WRITE ,__sccfg_ctrl_bits);
__IO_REG32_BIT(SCCFG_STAT0,       0xB050F178,__READ_WRITE ,__sccfg_stat0_bits);
__IO_REG32_BIT(SCCFG_STAT1,       0xB050F17C,__READ_WRITE ,__sccfg_stat1_bits);
__IO_REG32_BIT(SCCFG_STAT2,       0xB050F180,__READ_WRITE ,__sccfg_stat2_bits);
__IO_REG32(    SCCFG_SECKEY0,     0xB050F190,__WRITE      );
__IO_REG32(    SCCFG_SECKEY1,     0xB050F194,__WRITE      );
__IO_REG32(    SCCFG_SECKEY2,     0xB050F198,__WRITE      );
__IO_REG32(    SCCFG_SECKEY3,     0xB050F19C,__WRITE      );
__IO_REG32(    SCCFG_MODID,       0xB050F1A0,__READ       );
__IO_REG32(    SCCFG_UNLCK,       0xB050F1A4,__WRITE      );
__IO_REG32(    SCCFG_GPREG0,      0xB050F1A8,__READ_WRITE );
__IO_REG32(    SCCFG_GPREG1,      0xB050F1AC,__READ_WRITE );

/***************************************************************************
 **
 ** CRC0
 **
 ***************************************************************************/
__IO_REG32(    CRC0_POLY,         0xB0B30000,__READ_WRITE );
__IO_REG32(    CRC0_SEED,         0xB0B30004,__READ_WRITE );
__IO_REG32(    CRC0_FXOR,         0xB0B30008,__READ_WRITE );
__IO_REG32_BIT(CRC0_CFG,          0xB0B3000C,__READ_WRITE ,__crcn_cfg_bits);
__IO_REG32(    CRC0_WR,           0xB0B30010,__READ_WRITE );
__IO_REG32(    CRC0_RD,           0xB0B30014,__READ_WRITE );

/***************************************************************************
 **
 ** TCFCFG
 **
 ***************************************************************************/
__IO_REG32(    TCFCFG_FCPROTKEY,  0xB0411000,__READ_WRITE );
__IO_REG32_BIT(TCFCFG_FECCCTRL,   0xB0411010,__READ_WRITE ,__tcfcfg_feccctrl_bits);
__IO_REG32(    TCFCFG_FDATEIR,    0xB0411018,__READ_WRITE );
__IO_REG32_BIT(TCFCFG_FECCEIR,    0xB041101C,__READ_WRITE ,__tcfcfg_fecceir_bits);
__IO_REG32_BIT(TCFCFG_FICTRL0,    0xB0411020,__READ_WRITE ,__tcfcfg_fictrl_bits);
__IO_REG32_BIT(TCFCFG_FICTRL1,    0xB0411024,__READ_WRITE ,__tcfcfg_fictrl_bits);
__IO_REG32_BIT(TCFCFG_FICTRL2,    0xB0411028,__READ_WRITE ,__tcfcfg_fictrl_bits);
__IO_REG32_BIT(TCFCFG_FICTRL3,    0xB041102C,__READ_WRITE ,__tcfcfg_fictrl_bits);
__IO_REG32_BIT(TCFCFG_FSTAT0,     0xB0411038,__READ       ,__tcfcfg_fstat_bits);
__IO_REG32_BIT(TCFCFG_FSTAT1,     0xB041103C,__READ       ,__tcfcfg_fstat_bits);
__IO_REG32_BIT(TCFCFG_FSTAT2,     0xB0411040,__READ       ,__tcfcfg_fstat_bits);
__IO_REG32_BIT(TCFCFG_FSTAT3,     0xB0411044,__READ       ,__tcfcfg_fstat_bits);
__IO_REG32_BIT(TCFCFG_FSECIR,     0xB0411050,__READ_WRITE ,__tcfcfg_fsecir_bits);
__IO_REG32(    TCFCFG_FECCEAR,    0xB0411054,__READ       );
__IO_REG32(    TCFCFG_FMIDR,      0xB0411058,__READ       );
__IO_REG32(    TCFCFG_FCAMLR0,    0xB0411060,__READ       );
__IO_REG32_BIT(TCFCFG_FCAMHR0,    0xB0411064,__READ       ,__tcfcfg_fcamhr_bits);
__IO_REG32(    TCFCFG_FCAMLR1,    0xB0411068,__READ       );
__IO_REG32_BIT(TCFCFG_FCAMHR1,    0xB041106C,__READ       ,__tcfcfg_fcamhr_bits);
__IO_REG32(    TCFCFG_FCAMLR2,    0xB0411070,__READ       );
__IO_REG32_BIT(TCFCFG_FCAMHR2,    0xB0411074,__READ       ,__tcfcfg_fcamhr_bits);
__IO_REG32(    TCFCFG_FCAMLR3,    0xB0411078,__READ       );
__IO_REG32_BIT(TCFCFG_FCAMHR3,    0xB041107C,__READ       ,__tcfcfg_fcamhr_bits);

/***************************************************************************
 **
 ** EEFCFG
 **
 ***************************************************************************/
__IO_REG32(    EEFCFG_CPR,        0xB0412000,__READ_WRITE );
__IO_REG32_BIT(EEFCFG_CR,         0xB0412008,__READ_WRITE ,__eefcfg_cr_bits);
__IO_REG32_BIT(EEFCFG_ECR,        0xB041200C,__READ_WRITE ,__eefcfg_ecr_bits);
__IO_REG32_BIT(EEFCFG_WCR,        0xB0412010,__READ_WRITE ,__eefcfg_wcr_bits);
__IO_REG32_BIT(EEFCFG_WSR,        0xB0412014,__READ       ,__eefcfg_wsr_bits);
__IO_REG32(    EEFCFG_DBEIR,      0xB0412018,__READ_WRITE );
__IO_REG32_BIT(EEFCFG_EEIR,       0xB041201C,__READ_WRITE ,__eefcfg_eeir_bits);
__IO_REG32_BIT(EEFCFG_WMER,       0xB0412020,__READ_WRITE ,__eefcfg_wmer_bits);
__IO_REG32_BIT(EEFCFG_ICR,        0xB0412024,__READ_WRITE ,__eefcfg_icr_bits);
__IO_REG32_BIT(EEFCFG_SR,         0xB0412028,__READ       ,__eefcfg_sr_bits);
__IO_REG32_BIT(EEFCFG_SECIR,      0xB041202C,__READ_WRITE ,__eefcfg_secir_bits);
__IO_REG32(    EEFCFG_EEAR,       0xB0412030,__READ       );
__IO_REG32(    EEFCFG_MIR,        0xB0412034,__READ       );
__IO_REG32(    EEFCFG_FCAMLR,     0xB0412048,__READ       );
__IO_REG32_BIT(EEFCFG_FCAMHR,     0xB041204C,__READ_WRITE ,__eefcfg_fcamhr_bits);

/***************************************************************************
 **
 ** TRCFG
 **
 ***************************************************************************/
__IO_REG32_BIT(TRCFG_TCMCFG0,     0xB0410000,__READ_WRITE ,__trcfg_tcmcfg0_bits);
__IO_REG32(    TRCFG_TCMCFG1,     0xB0410004,__READ_WRITE );
__IO_REG32(    TRCFG_TCMUNLOCK,   0xB0410008,__READ_WRITE );

/***************************************************************************
 **
 ** SRCFG
 **
 ***************************************************************************/
__IO_REG32_BIT(SRCFG_CFG0,        0xB0D00000,__READ_WRITE ,__srcfg_cfg0_bits);
__IO_REG32(    SRCFG_CFG1,        0xB0D00004,__READ_WRITE );
__IO_REG32_BIT(SRCFG_CFG2,        0xB0D00008,__READ_WRITE ,__srcfg_cfg2_bits);
__IO_REG32(    SRCFG_KEY,         0xB0D0000C,__READ_WRITE );
__IO_REG32_BIT(SRCFG_ERRFLG,      0xB0D00010,__READ_WRITE ,__srcfg_errflg_bits);
__IO_REG32_BIT(SRCFG_INTE,        0xB0D00014,__READ_WRITE ,__srcfg_inte_bits);
__IO_REG32_BIT(SRCFG_ECCE,        0xB0D00018,__READ_WRITE ,__srcfg_ecce_bits);
__IO_REG32(    SRCFG_ERRADR,      0xB0D00020,__READ       );
__IO_REG32(    SRCFG_MID,         0xB0D00024,__READ       );

/***************************************************************************
 **
 ** EXCFG
 **
 ***************************************************************************/
__IO_REG32(    EXCFG_UNLOCK,      0xFFFEFF58,__READ_WRITE );
__IO_REG32_BIT(EXCFG_CNFG,        0xFFFEFF60,__READ_WRITE ,__excfg_cnfg_bits);
__IO_REG32(    EXCFG_UNDEFINACT,  0xFFFEFF84,__READ_WRITE );
__IO_REG32(    EXCFG_SVCINACT,    0xFFFEFF88,__READ_WRITE );
__IO_REG32(    EXCFG_PABORTINACT, 0xFFFEFF8C,__READ_WRITE );
__IO_REG32(    EXCFG_DABORTINACT, 0xFFFEFF90,__READ_WRITE );
__IO_REG32(    EXCFG_IRQINACT,    0xFFFEFF98,__READ_WRITE );
__IO_REG32(    EXCFG_NMIINACT,    0xFFFEFF9C,__READ_WRITE );
__IO_REG32(    EXCFG_UNDEFACT,    0xFFFEFFC4,__READ_WRITE );
__IO_REG32(    EXCFG_SVCACT,      0xFFFEFFC8,__READ_WRITE );
__IO_REG32(    EXCFG_PABORTACT,   0xFFFEFFCC,__READ_WRITE );
__IO_REG32(    EXCFG_DABORTACT,   0xFFFEFFD0,__READ_WRITE );
__IO_REG32(    EXCFG_IRQACT,      0xFFFEFFD8,__READ_WRITE );
__IO_REG32(    EXCFG_NMIACT,      0xFFFEFFDC,__READ_WRITE );

/***************************************************************************
 **
 ** IRQ0
 **
 ***************************************************************************/
__IO_REG32(    IRQ0_NMIVAS,       0xB0400000,__READ       );
__IO_REG32_BIT(IRQ0_NMIST,        0xB0400004,__READ       ,__irqn_nmist_bits);
__IO_REG32(    IRQ0_IRQVAS,       0xB0400008,__READ       );
__IO_REG32_BIT(IRQ0_IRQST,        0xB040000C,__READ       ,__irqn_irqst_bits);
__IO_REG32(    IRQ0_NMIVA0,       0xB0400010,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA1,       0xB0400014,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA2,       0xB0400018,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA3,       0xB040001C,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA4,       0xB0400020,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA5,       0xB0400024,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA6,       0xB0400028,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA7,       0xB040002C,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA9,       0xB0400034,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA11,      0xB040003C,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA12,      0xB0400040,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA13,      0xB0400044,__READ_WRITE );
__IO_REG32(    IRQ0_NMIVA14,      0xB0400048,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA0,       0xB0400090,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA1,       0xB0400094,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA13,      0xB04000C4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA14,      0xB04000C8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA18,      0xB04000D8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA19,      0xB04000DC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA20,      0xB04000E0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA21,      0xB04000E4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA22,      0xB04000E8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA30,      0xB0400108,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA31,      0xB040010C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA32,      0xB0400110,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA33,      0xB0400114,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA34,      0xB0400118,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA35,      0xB040011C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA36,      0xB0400120,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA37,      0xB0400124,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA38,      0xB0400128,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA39,      0xB040012C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA40,      0xB0400130,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA41,      0xB0400134,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA42,      0xB0400138,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA43,      0xB040013C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA44,      0xB0400140,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA49,      0xB0400154,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA50,      0xB0400158,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA52,      0xB0400160,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA53,      0xB0400164,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA55,      0xB040016C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA56,      0xB0400170,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA61,      0xB0400184,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA62,      0xB0400188,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA63,      0xB040018C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA69,      0xB04001A4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA70,      0xB04001A8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA71,      0xB04001AC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA72,      0xB04001B0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA73,      0xB04001B4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA74,      0xB04001B8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA75,      0xB04001BC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA76,      0xB04001C0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA77,      0xB04001C4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA78,      0xB04001C8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA79,      0xB04001CC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA80,      0xB04001D0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA81,      0xB04001D4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA82,      0xB04001D8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA83,      0xB04001DC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA84,      0xB04001E0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA85,      0xB04001E4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA86,      0xB04001E8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA87,      0xB04001EC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA88,      0xB04001F0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA89,      0xB04001F4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA90,      0xB04001F8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA91,      0xB04001FC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA92,      0xB0400200,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA93,      0xB0400204,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA94,      0xB0400208,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA95,      0xB040020C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA96,      0xB0400210,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA97,      0xB0400214,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA98,      0xB0400218,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA99,      0xB040021C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA100,     0xB0400220,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA101,     0xB0400224,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA102,     0xB0400228,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA104,     0xB0400230,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA105,     0xB0400234,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA106,     0xB0400238,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA107,     0xB040023C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA112,     0xB0400250,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA113,     0xB0400254,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA114,     0xB0400258,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA115,     0xB040025C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA124,     0xB0400280,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA125,     0xB0400284,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA126,     0xB0400288,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA127,     0xB040028C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA132,     0xB04002A0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA133,     0xB04002A4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA134,     0xB04002A8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA135,     0xB04002AC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA136,     0xB04002B0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA137,     0xB04002B4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA138,     0xB04002B8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA139,     0xB04002BC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA144,     0xB04002D0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA145,     0xB04002D4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA146,     0xB04002D8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA147,     0xB04002DC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA152,     0xB04002F0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA153,     0xB04002F4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA154,     0xB04002F8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA158,     0xB0400308,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA159,     0xB040030C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA160,     0xB0400310,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA164,     0xB0400320,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA165,     0xB0400324,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA166,     0xB0400328,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA167,     0xB040032C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA168,     0xB0400330,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA169,     0xB0400334,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA170,     0xB0400338,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA171,     0xB040033C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA172,     0xB0400340,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA173,     0xB0400344,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA174,     0xB0400348,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA175,     0xB040034C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA176,     0xB0400350,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA177,     0xB0400354,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA178,     0xB0400358,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA179,     0xB040035C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA180,     0xB0400360,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA181,     0xB0400364,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA182,     0xB0400368,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA183,     0xB040036C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA184,     0xB0400370,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA185,     0xB0400374,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA186,     0xB0400378,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA187,     0xB040037C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA194,     0xB0400398,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA195,     0xB040039C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA198,     0xB04003A8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA199,     0xB04003AC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA202,     0xB04003B8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA203,     0xB04003BC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA206,     0xB04003C8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA208,     0xB04003D0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA209,     0xB04003D4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA210,     0xB04003D8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA211,     0xB04003DC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA212,     0xB04003E0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA213,     0xB04003E4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA214,     0xB04003E8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA215,     0xB04003EC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA216,     0xB04003F0,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA217,     0xB04003F4,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA218,     0xB04003F8,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA219,     0xB04003FC,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA220,     0xB0400400,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA221,     0xB0400404,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA222,     0xB0400408,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA223,     0xB040040C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA232,     0xB0400430,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA233,     0xB0400434,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA234,     0xB0400438,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA235,     0xB040043C,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA236,     0xB0400440,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA237,     0xB0400444,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA238,     0xB0400448,__READ_WRITE );
__IO_REG32(    IRQ0_IRQVA239,     0xB040044C,__READ_WRITE );
__IO_REG32_BIT(IRQ0_NMIPL0,       0xB0400890,__READ_WRITE ,__irqn_nmipl_bits);
__IO_REG32_BIT(IRQ0_NMIPL1,       0xB0400894,__READ_WRITE ,__irqn_nmipl_bits);
__IO_REG32_BIT(IRQ0_NMIPL2,       0xB0400898,__READ_WRITE ,__irqn_nmipl_bits);
__IO_REG32_BIT(IRQ0_NMIPL3,       0xB040089C,__READ_WRITE ,__irqn_nmipl_bits);
__IO_REG32_BIT(IRQ0_IRQPL0,       0xB04008B0,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL3,       0xB04008BC,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL4,       0xB04008C0,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL5,       0xB04008C4,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL7,       0xB04008CC,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL8,       0xB04008D0,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL9,       0xB04008D4,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL10,      0xB04008D8,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL11,      0xB04008DC,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL12,      0xB04008E0,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL13,      0xB04008E4,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL14,      0xB04008E8,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL15,      0xB04008EC,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL17,      0xB04008F4,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL18,      0xB04008F8,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL19,      0xB04008FC,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL20,      0xB0400900,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL21,      0xB0400904,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL22,      0xB0400908,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL23,      0xB040090C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL24,      0xB0400910,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL25,      0xB0400914,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL26,      0xB0400918,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL28,      0xB0400920,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL31,      0xB040092C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL33,      0xB0400934,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL34,      0xB0400938,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL36,      0xB0400940,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL38,      0xB0400948,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL39,      0xB040094C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL40,      0xB0400950,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL41,      0xB0400954,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL42,      0xB0400958,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL43,      0xB040095C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL44,      0xB0400960,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL45,      0xB0400964,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL46,      0xB0400968,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL48,      0xB0400970,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL49,      0xB0400974,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL50,      0xB0400978,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL51,      0xB040097C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL52,      0xB0400980,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL53,      0xB0400984,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL54,      0xB0400988,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL55,      0xB040098C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL58,      0xB0400998,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_IRQPL59,      0xB040099C,__READ_WRITE ,__irqn_irqpl_bits);
__IO_REG32_BIT(IRQ0_NMIS,         0xB0400AB0,__READ_WRITE ,__irqn_nmis_bits);
__IO_REG32_BIT(IRQ0_NMIR,         0xB0400AB4,__READ_WRITE ,__irqn_nmir_bits);
__IO_REG32_BIT(IRQ0_NMISIS,       0xB0400AB8,__READ_WRITE ,__irqn_nmisis_bits);
__IO_REG32_BIT(IRQ0_IRQS0,        0xB0400AC0,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS1,        0xB0400AC4,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS2,        0xB0400AC8,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS3,        0xB0400ACC,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS4,        0xB0400AD0,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS5,        0xB0400AD4,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS6,        0xB0400AD8,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQS7,        0xB0400ADC,__READ_WRITE ,__irqn_irqs_bits);
__IO_REG32_BIT(IRQ0_IRQR0,        0xB0400B00,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR1,        0xB0400B04,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR2,        0xB0400B08,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR3,        0xB0400B0C,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR4,        0xB0400B10,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR5,        0xB0400B14,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR6,        0xB0400B18,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQR7,        0xB0400B1C,__READ_WRITE ,__irqn_irqr_bits);
__IO_REG32_BIT(IRQ0_IRQSIS0,      0xB0400B40,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS1,      0xB0400B44,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS2,      0xB0400B48,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS3,      0xB0400B4C,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS4,      0xB0400B50,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS5,      0xB0400B54,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS6,      0xB0400B58,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQSIS7,      0xB0400B5C,__READ       ,__irqn_irqsis_bits);
__IO_REG32_BIT(IRQ0_IRQCES0,      0xB0400B80,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES1,      0xB0400B84,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES2,      0xB0400B88,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES3,      0xB0400B8C,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES4,      0xB0400B90,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES5,      0xB0400B94,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES6,      0xB0400B98,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCES7,      0xB0400B9C,__READ_WRITE ,__irqn_irqces_bits);
__IO_REG32_BIT(IRQ0_IRQCEC0,      0xB0400BC0,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC1,      0xB0400BC4,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC2,      0xB0400BC8,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC3,      0xB0400BCC,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC4,      0xB0400BD0,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC5,      0xB0400BD4,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC6,      0xB0400BD8,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCEC7,      0xB0400BDC,__READ_WRITE ,__irqn_irqcec_bits);
__IO_REG32_BIT(IRQ0_IRQCE0,       0xB0400C00,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE1,       0xB0400C04,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE2,       0xB0400C08,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE3,       0xB0400C0C,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE4,       0xB0400C10,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE5,       0xB0400C14,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE6,       0xB0400C18,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_IRQCE7,       0xB0400C1C,__READ_WRITE ,__irqn_irqce_bits);
__IO_REG32_BIT(IRQ0_NMIHC,        0xB0400C40,__READ_WRITE ,__irqn_nmihc_bits);
__IO_REG32_BIT(IRQ0_NMIHS,        0xB0400C44,__READ_WRITE ,__irqn_nmihs_bits);
__IO_REG32_BIT(IRQ0_IRQHC,        0xB0400C48,__READ_WRITE ,__irqn_irqhc_bits);
__IO_REG32_BIT(IRQ0_IRQHS0,       0xB0400C50,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS1,       0xB0400C54,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS2,       0xB0400C58,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS3,       0xB0400C5C,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS4,       0xB0400C60,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS5,       0xB0400C64,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS6,       0xB0400C68,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQHS7,       0xB0400C6C,__READ_WRITE ,__irqn_irqhs_bits);
__IO_REG32_BIT(IRQ0_IRQPLM,       0xB0400C90,__READ_WRITE ,__irqn_irqplm_bits);
__IO_REG32_BIT(IRQ0_CSR,          0xB0400C98,__READ_WRITE ,__irqn_csr_bits);
__IO_REG32_BIT(IRQ0_NESTL,        0xB0400CA0,__READ_WRITE ,__irqn_nestl_bits);
__IO_REG32_BIT(IRQ0_NMIRS,        0xB0400CA8,__READ       ,__irqn_nmirs_bits);
__IO_REG32_BIT(IRQ0_NMIPS,        0xB0400CAC,__READ       ,__irqn_nmips_bits);
__IO_REG32_BIT(IRQ0_NMIRS0,       0xB0400CB0,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS1,       0xB0400CB4,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS2,       0xB0400CB8,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS3,       0xB0400CBC,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS4,       0xB0400CC0,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS5,       0xB0400CC4,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS6,       0xB0400CC8,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_NMIRS7,       0xB0400CCC,__READ       ,__irqn_irqrs_bits);
__IO_REG32_BIT(IRQ0_IRQPS0,       0xB0400CF0,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS1,       0xB0400CF4,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS2,       0xB0400CF8,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS3,       0xB0400CFC,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS4,       0xB0400D00,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS5,       0xB0400D04,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS6,       0xB0400D08,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_IRQPS7,       0xB0400D0C,__READ       ,__irqn_irqps_bits);
__IO_REG32_BIT(IRQ0_UNLOCK,       0xB0400D30,__READ_WRITE ,__irqn_unlock_bits);
__IO_REG32(    IRQ0_MID,          0xB0400D38,__READ_WRITE );
__IO_REG32_BIT(IRQ0_EEI,          0xB0400D40,__READ_WRITE ,__irqn_eei_bits);
__IO_REG32_BIT(IRQ0_EAN,          0xB0400D44,__READ_WRITE ,__irqn_ean_bits);
__IO_REG32_BIT(IRQ0_ET,           0xB0400D48,__READ_WRITE ,__irqn_et_bits);
__IO_REG32(    IRQ0_EEB0,         0xB0400D4C,__READ_WRITE );
__IO_REG32(    IRQ0_EEB1,         0xB0400D50,__READ_WRITE );
__IO_REG32_BIT(IRQ0_EEB2,         0xB0400D54,__READ_WRITE ,__irqn_eeb2_bits);

/***************************************************************************
 **
 ** PPC
 **
 ***************************************************************************/
__IO_REG16_BIT(PPC_PCFGR000,      0xB07E8000,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR001,      0xB07E8002,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR002,      0xB07E8004,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR003,      0xB07E8006,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR004,      0xB07E8008,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR005,      0xB07E800A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR006,      0xB07E800C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR007,      0xB07E800E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR008,      0xB07E8010,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR009,      0xB07E8012,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR010,      0xB07E8014,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR011,      0xB07E8016,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR012,      0xB07E8018,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR013,      0xB07E801A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR014,      0xB07E801C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR015,      0xB07E801E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR016,      0xB07E8020,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR017,      0xB07E8022,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR018,      0xB07E8024,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR019,      0xB07E8026,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR020,      0xB07E8028,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR021,      0xB07E802A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR022,      0xB07E802C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR023,      0xB07E802E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR024,      0xB07E8030,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR025,      0xB07E8032,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR026,      0xB07E8034,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR027,      0xB07E8036,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR028,      0xB07E8038,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR029,      0xB07E803A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR030,      0xB07E803C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR031,      0xB07E803E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR032,      0xB07E8040,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR033,      0xB07E8042,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR034,      0xB07E8044,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR035,      0xB07E8046,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR036,      0xB07E8048,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR037,      0xB07E804A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR038,      0xB07E804C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR039,      0xB07E804E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR040,      0xB07E8050,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR041,      0xB07E8052,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR042,      0xB07E8054,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR043,      0xB07E8056,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR044,      0xB07E8058,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR045,      0xB07E805A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR046,      0xB07E805C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR047,      0xB07E805E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR048,      0xB07E8060,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR049,      0xB07E8062,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR050,      0xB07E8064,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR051,      0xB07E8066,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR052,      0xB07E8068,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR062,      0xB07E807C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR063,      0xB07E807E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR100,      0xB07E8080,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR101,      0xB07E8082,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR102,      0xB07E8084,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR103,      0xB07E8086,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR104,      0xB07E8088,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR105,      0xB07E808A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR106,      0xB07E808C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR107,      0xB07E808E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR108,      0xB07E8090,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR109,      0xB07E8092,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR110,      0xB07E8094,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR111,      0xB07E8096,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR112,      0xB07E8098,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR113,      0xB07E809A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR114,      0xB07E809C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR115,      0xB07E809E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR116,      0xB07E80A0,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR117,      0xB07E80A2,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR118,      0xB07E80A4,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR119,      0xB07E80A6,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR120,      0xB07E80A8,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR121,      0xB07E80AA,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR122,      0xB07E80AC,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR123,      0xB07E80AE,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR124,      0xB07E80B0,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR125,      0xB07E80B2,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR126,      0xB07E80B4,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR127,      0xB07E80B6,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR128,      0xB07E81B8,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR129,      0xB07E81BA,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR130,      0xB07E81BC,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR131,      0xB07E81BE,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR132,      0xB07E81C0,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR133,      0xB07E81C2,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR134,      0xB07E81C4,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR135,      0xB07E81C6,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR136,      0xB07E81C8,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR137,      0xB07E81CA,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR138,      0xB07E81CC,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR139,      0xB07E81CE,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR140,      0xB07E81D0,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR141,      0xB07E81D2,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR142,      0xB07E81D4,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR143,      0xB07E81D6,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR144,      0xB07E81D8,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR145,      0xB07E81DA,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR146,      0xB07E81DC,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR147,      0xB07E81DE,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR148,      0xB07E81E0,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR149,      0xB07E81E2,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR150,      0xB07E81E4,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR151,      0xB07E81E6,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR152,      0xB07E81E8,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR153,      0xB07E81EA,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR154,      0xB07E81EC,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR155,      0xB07E81EE,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR156,      0xB07E81F0,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR157,      0xB07E81F2,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR158,      0xB07E81F4,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR159,      0xB07E80F6,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR200,      0xB07E8100,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR201,      0xB07E8102,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR202,      0xB07E8104,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR203,      0xB07E8106,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR204,      0xB07E8108,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR205,      0xB07E810A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR206,      0xB07E810C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR207,      0xB07E810E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR208,      0xB07E8110,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR209,      0xB07E8112,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR210,      0xB07E8114,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR211,      0xB07E8116,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR212,      0xB07E8118,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR213,      0xB07E811A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR214,      0xB07E811C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR215,      0xB07E811E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR216,      0xB07E8120,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR217,      0xB07E8122,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR218,      0xB07E8124,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR219,      0xB07E8126,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR220,      0xB07E8128,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR221,      0xB07E812A,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR222,      0xB07E812C,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR223,      0xB07E812E,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR224,      0xB07E8130,__READ_WRITE ,__ppc_pcfgrn_bits);
__IO_REG16_BIT(PPC_PCFGR225,      0xB07E8132,__READ_WRITE ,__ppc_pcfgrn_bits);

/***************************************************************************
 **
 ** EICU0
 **
 ***************************************************************************/
__IO_REG32_BIT(EICU0_CNFGR,       0xB0628000,__READ_WRITE ,__eicun_cnfgr_bits);
__IO_REG32_BIT(EICU0_IRENR,       0xB0628004,__READ_WRITE ,__eicun_irenr_bits);
__IO_REG32(    EICU0_SPLR0,       0xB0628008,__READ       );
__IO_REG32(    EICU0_SPLR1,       0xB062800C,__READ       );
__IO_REG32(    EICU0_SPLR2,       0xB0628010,__READ       );
__IO_REG32(    EICU0_SPLR3,       0xB0628014,__READ       );
__IO_REG32(    EICU0_SPLR4,       0xB0628018,__READ       );
__IO_REG32(    EICU0_SPLR5,       0xB062801C,__READ       );
__IO_REG32(    EICU0_SPLR6,       0xB0628020,__READ       );
__IO_REG32(    EICU0_SPLR7,       0xB0628024,__READ       );

/***************************************************************************
 **
 ** TPU0
 **
 ***************************************************************************/
__IO_REG32(    TPU0_UNLOCK,       0xB0408000,__READ_WRITE );
__IO_REG32_BIT(TPU0_LST,          0xB0408004,__READ       ,__tpun_lst_bits);
__IO_REG32_BIT(TPU0_CFG,          0xB0408008,__READ_WRITE ,__tpun_cfg_bits);
__IO_REG32_BIT(TPU0_TIR,          0xB040800C,__READ       ,__tpun_tir_bits);
__IO_REG32_BIT(TPU0_TST,          0xB0408010,__READ       ,__tpun_tst_bits);
__IO_REG32_BIT(TPU0_TIE,          0xB0408014,__READ_WRITE ,__tpun_tie_bits);
__IO_REG32(    TPU0_MID,          0xB0408018,__READ       );
__IO_REG32_BIT(TPU0_TCN00,        0xB0408030,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN01,        0xB0408034,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN02,        0xB0408038,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN03,        0xB040803C,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN04,        0xB0408040,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN05,        0xB0408044,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN06,        0xB0408048,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN07,        0xB040804C,__READ_WRITE ,__tpun_tcn00_bits);
__IO_REG32_BIT(TPU0_TCN10,        0xB0408050,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN11,        0xB0408054,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN12,        0xB0408058,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN13,        0xB040805C,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN14,        0xB0408060,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN15,        0xB0408064,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN16,        0xB0408068,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCN17,        0xB040806C,__READ_WRITE ,__tpun_tcn10_bits);
__IO_REG32_BIT(TPU0_TCC0,         0xB0408070,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC1,         0xB0408074,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC2,         0xB0408078,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC3,         0xB040807C,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC4,         0xB0408080,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC5,         0xB0408084,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC6,         0xB0408088,__READ_WRITE ,__tpun_tcc_bits);
__IO_REG32_BIT(TPU0_TCC7,         0xB040808C,__READ_WRITE ,__tpun_tcc_bits);

/***************************************************************************
 **
 ** PPU0
 **
 ***************************************************************************/
__IO_REG32_BIT(PPU0_PRS2,         0xB0A00008,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS3,         0xB0A0000C,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS4,         0xB0A00010,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS5,         0xB0A00014,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS6,         0xB0A00018,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS7,         0xB0A0001C,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS8,         0xB0A00020,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS9,         0xB0A00024,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS10,        0xB0A00028,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS11,        0xB0A0002C,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS12,        0xB0A00030,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PRS13,        0xB0A00034,__READ_WRITE ,__ppun_prs_bits);
__IO_REG32_BIT(PPU0_PAS2,         0xB0A00048,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS3,         0xB0A0004C,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS4,         0xB0A00050,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS5,         0xB0A00054,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS6,         0xB0A00058,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS7,         0xB0A0005C,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS8,         0xB0A00060,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS9,         0xB0A00064,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS10,        0xB0A00068,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS11,        0xB0A0006C,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS12,        0xB0A00070,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_PAS13,        0xB0A00074,__READ_WRITE ,__ppun_pas_bits);
__IO_REG32_BIT(PPU0_GAS0,         0xB0A00080,__READ_WRITE ,__ppun_gas_bits);
__IO_REG32_BIT(PPU0_GAS1,         0xB0A00084,__READ_WRITE ,__ppun_gas_bits);
__IO_REG32_BIT(PPU0_GAS2,         0xB0A00088,__READ_WRITE ,__ppun_gas_bits);
__IO_REG32_BIT(PPU0_GAS3,         0xB0A0008C,__READ_WRITE ,__ppun_gas_bits);
__IO_REG32_BIT(PPU0_GAS4,         0xB0A00090,__READ_WRITE ,__ppun_gas_bits);
__IO_REG32_BIT(PPU0_PRC2,         0xB0A000C8,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC3,         0xB0A000CC,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC4,         0xB0A000D0,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC5,         0xB0A000D4,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC6,         0xB0A000D8,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC7,         0xB0A000DC,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC8,         0xB0A000E0,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC9,         0xB0A000E4,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC10,        0xB0A000E8,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC11,        0xB0A000EC,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC12,        0xB0A000F0,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PRC13,        0xB0A000F4,__READ_WRITE ,__ppun_prc_bits);
__IO_REG32_BIT(PPU0_PAC2,         0xB0A00108,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC3,         0xB0A0010C,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC4,         0xB0A00110,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC5,         0xB0A00114,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC6,         0xB0A00118,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC7,         0xB0A0011C,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC8,         0xB0A00120,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC9,         0xB0A00124,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC10,        0xB0A00128,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC11,        0xB0A0012C,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC12,        0xB0A00130,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_PAC13,        0xB0A00134,__READ_WRITE ,__ppun_pac_bits);
__IO_REG32_BIT(PPU0_GAC0,         0xB0A00140,__READ_WRITE ,__ppun_gac_bits);
__IO_REG32_BIT(PPU0_GAC1,         0xB0A00144,__READ_WRITE ,__ppun_gac_bits);
__IO_REG32_BIT(PPU0_GAC2,         0xB0A00148,__READ_WRITE ,__ppun_gac_bits);
__IO_REG32_BIT(PPU0_GAC3,         0xB0A0014C,__READ_WRITE ,__ppun_gac_bits);
__IO_REG32_BIT(PPU0_GAC4,         0xB0A00150,__READ_WRITE ,__ppun_gac_bits);
__IO_REG32_BIT(PPU0_PR2,          0xB0A00188,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR3,          0xB0A0018C,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR4,          0xB0A00190,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR5,          0xB0A00194,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR6,          0xB0A00198,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR7,          0xB0A0019C,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR8,          0xB0A001A0,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR9,          0xB0A001A4,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR10,         0xB0A001A8,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR11,         0xB0A001AC,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR12,         0xB0A001B0,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PR13,         0xB0A001B4,__READ_WRITE ,__ppun_pr_bits);
__IO_REG32_BIT(PPU0_PA2,          0xB0A001C8,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA3,          0xB0A001CC,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA4,          0xB0A001D0,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA5,          0xB0A001D4,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA6,          0xB0A001D8,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA7,          0xB0A001DC,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA8,          0xB0A001E0,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA9,          0xB0A001E4,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA10,         0xB0A001E8,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA11,         0xB0A001EC,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA12,         0xB0A001F0,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_PA13,         0xB0A001F4,__READ_WRITE ,__ppun_pa_bits);
__IO_REG32_BIT(PPU0_GA0,          0xB0A00200,__READ_WRITE ,__ppun_ga_bits);
__IO_REG32_BIT(PPU0_GA1,          0xB0A00204,__READ_WRITE ,__ppun_ga_bits);
__IO_REG32_BIT(PPU0_GA2,          0xB0A00208,__READ_WRITE ,__ppun_ga_bits);
__IO_REG32_BIT(PPU0_GA3,          0xB0A0020C,__READ_WRITE ,__ppun_ga_bits);
__IO_REG32_BIT(PPU0_GA4,          0xB0A00210,__READ_WRITE ,__ppun_ga_bits);
__IO_REG32(    PPU0_UNLOCK,       0xB0A00240,__READ_WRITE );
__IO_REG32_BIT(PPU0_ST,           0xB0A00248,__READ       ,__ppun_st_bits);
__IO_REG32_BIT(PPU0_CTR,          0xB0A0024C,__READ_WRITE ,__ppun_ctr_bits);

/***************************************************************************
 **
 ** BECU0
 **
 ***************************************************************************/
__IO_REG16_BIT(BECU0_CTRL,        0xB07F0000,__READ_WRITE ,__becun_ctrl_bits);
__IO_REG16_BIT(BECU0_CTRH,        0xB07F0002,__READ       ,__becun_ctrh_bits);
__IO_REG16(    BECU0_ADDRL,       0xB07F0004,__READ       );
__IO_REG16(    BECU0_ADDRH,       0xB07F0006,__READ       );
__IO_REG16(    BECU0_DATALL,      0xB07F0008,__READ       );
__IO_REG16(    BECU0_DATALH,      0xB07F000A,__READ       );
__IO_REG16(    BECU0_DATAHL,      0xB07F000C,__READ       );
__IO_REG16(    BECU0_DATAHH,      0xB07F000E,__READ       );
__IO_REG16(    BECU0_MASTERID,    0xB07F0010,__READ       );
__IO_REG16(    BECU0_MIDL,        0xB07F0012,__READ       );
__IO_REG16(    BECU0_MIDH,        0xB07F0014,__READ       );
__IO_REG16_BIT(BECU0_NMIEN,       0xB07F0018,__READ_WRITE ,__becun_nmien_bits);

/***************************************************************************
 **
 ** BECU1
 **
 ***************************************************************************/
__IO_REG16_BIT(BECU1_CTRL,        0xB08F0000,__READ_WRITE ,__becun_ctrl_bits);
__IO_REG16_BIT(BECU1_CTRH,        0xB08F0002,__READ       ,__becun_ctrh_bits);
__IO_REG16(    BECU1_ADDRL,       0xB08F0004,__READ       );
__IO_REG16(    BECU1_ADDRH,       0xB08F0006,__READ       );
__IO_REG16(    BECU1_DATALL,      0xB08F0008,__READ       );
__IO_REG16(    BECU1_DATALH,      0xB08F000A,__READ       );
__IO_REG16(    BECU1_DATAHL,      0xB08F000C,__READ       );
__IO_REG16(    BECU1_DATAHH,      0xB08F000E,__READ       );
__IO_REG16(    BECU1_MASTERID,    0xB08F0010,__READ       );
__IO_REG16(    BECU1_MIDL,        0xB08F0012,__READ       );
__IO_REG16(    BECU1_MIDH,        0xB08F0014,__READ       );
__IO_REG16_BIT(BECU1_NMIEN,       0xB08F0018,__READ_WRITE ,__becun_nmien_bits);

/***************************************************************************
 **
 ** BECU3
 **
 ***************************************************************************/
__IO_REG16_BIT(BECU3_CTRL,        0xB0AF0000,__READ_WRITE ,__becun_ctrl_bits);
__IO_REG16_BIT(BECU3_CTRH,        0xB0AF0002,__READ       ,__becun_ctrh_bits);
__IO_REG16(    BECU3_ADDRL,       0xB0AF0004,__READ       );
__IO_REG16(    BECU3_ADDRH,       0xB0AF0006,__READ       );
__IO_REG16(    BECU3_DATALL,      0xB0AF0008,__READ       );
__IO_REG16(    BECU3_DATALH,      0xB0AF000A,__READ       );
__IO_REG16(    BECU3_DATAHL,      0xB0AF000C,__READ       );
__IO_REG16(    BECU3_DATAHH,      0xB0AF000E,__READ       );
__IO_REG16(    BECU3_MASTERID,    0xB0AF0010,__READ       );
__IO_REG16(    BECU3_MIDL,        0xB0AF0012,__READ       );
__IO_REG16(    BECU3_MIDH,        0xB0AF0014,__READ       );
__IO_REG16_BIT(BECU3_NMIEN,       0xB0AF0018,__READ_WRITE ,__becun_nmien_bits);

/***************************************************************************
 **
 ** RETENTIONRAM
 **
 ***************************************************************************/
__IO_REG32(    RRCFG_UNLOCKR,     0xB0610000,__READ_WRITE );
__IO_REG32_BIT(RRCFG_CSR,         0xB0610004,__READ_WRITE ,__rrcfg_csr_bits);
__IO_REG32(    RRCFG_EAN,         0xB0610008,__READ       );
__IO_REG32(    RRCFG_ERRMSKR0,    0xB061000C,__READ_WRITE );
__IO_REG32_BIT(RRCFG_ERRMSKR1,    0xB0610010,__READ_WRITE ,__rrcfg_errmskr1_bits);
__IO_REG32_BIT(RRCFG_ECCEN,       0xB0610014,__READ_WRITE ,__rrcfg_eccen_bits);

/***************************************************************************
 **
 ** EIC0
 **
 ***************************************************************************/
__IO_REG32_BIT(EIC0_ENIR,         0xB0620000,__READ_WRITE ,__eicn_enir_bits);
__IO_REG32_BIT(EIC0_ENISR,        0xB0620004,__READ_WRITE ,__eicn_enisr_bits);
__IO_REG32_BIT(EIC0_ENICR,        0xB0620008,__READ_WRITE ,__eicn_enicr_bits);
__IO_REG32_BIT(EIC0_EIRR,         0xB062000C,__READ_WRITE ,__eicn_eirr_bits);
__IO_REG32_BIT(EIC0_EIRCR,        0xB0620010,__READ_WRITE ,__eicn_eircr_bits);
__IO_REG32_BIT(EIC0_NFER,         0xB0620014,__READ_WRITE ,__eicn_nfer_bits);
__IO_REG32_BIT(EIC0_NFESR,        0xB0620018,__READ_WRITE ,__eicn_nfesr_bits);
__IO_REG32_BIT(EIC0_NFECR,        0xB062001C,__READ_WRITE ,__eicn_nfecr_bits);
__IO_REG32_BIT(EIC0_ELVR0,        0xB0620020,__READ_WRITE ,__eicn_elvr_bits);
__IO_REG32_BIT(EIC0_ELVR1,        0xB0620024,__READ_WRITE ,__eicn_elvr_bits);
__IO_REG32_BIT(EIC0_ELVR2,        0xB0620028,__READ_WRITE ,__eicn_elvr_bits);
__IO_REG32_BIT(EIC0_ELVR3,        0xB062002C,__READ_WRITE ,__eicn_elvr_bits);
__IO_REG32_BIT(EIC0_NMIR,         0xB0620030,__READ_WRITE ,__eicn_nmir_bits);
__IO_REG32_BIT(EIC0_DRER,         0xB0620034,__READ_WRITE ,__eicn_drer_bits);
__IO_REG32_BIT(EIC0_DRESR,        0xB0620038,__READ_WRITE ,__eicn_dresr_bits);
__IO_REG32_BIT(EIC0_DRECR,        0xB062003C,__READ_WRITE ,__eicn_drecr_bits);
__IO_REG32_BIT(EIC0_DRFR,         0xB0620040,__READ       ,__eicn_drfr_bits);

/***************************************************************************
 **
 ** SG0
 **
 ***************************************************************************/
__IO_REG16_BIT(SG0_CR0,           0xB0800000,__READ_WRITE ,__sgn_cr0_bits);
__IO_REG8_BIT( SG0_CR1,           0xB0800002,__READ_WRITE ,__sgn_cr1_bits);
__IO_REG16_BIT(SG0_ECRL,          0xB0800004,__READ_WRITE ,__sgn_ecrl_bits);
__IO_REG16_BIT(SG0_FRL,           0xB0800006,__READ_WRITE ,__sgn_frl_bits);
__IO_REG16_BIT(SG0_ARL,           0xB0800008,__READ       ,__sgn_arl_bits);
__IO_REG16_BIT(SG0_AR,            0xB080000A,__READ_WRITE ,__sgn_ar_bits);
__IO_REG16_BIT(SG0_TARL,          0xB080000C,__READ_WRITE ,__sgn_tarl_bits);
__IO_REG16_BIT(SG0_TCRLIDRL,      0xB080000E,__READ_WRITE ,__sgn_tcrlidrl_bits);
__IO_REG8(     SG0_NRL,           0xB0800010,__READ_WRITE );
__IO_REG16_BIT(SG0_DER,           0xB0800012,__READ_WRITE ,__sgn_der_bits);
__IO_REG16(    SG0_DMAR,          0xB0800014,__READ_WRITE );

/***************************************************************************
 **
 ** FRT0
 **
 ***************************************************************************/
__IO_REG16(    FRT0_TCDT,         0xB0708000,__READ_WRITE );
__IO_REG16(    FRT0_CPCLRB,       0xB0708002,__READ_WRITE );
__IO_REG16(    FRT0_CPCLR,        0xB0708004,__READ_WRITE );
__IO_REG16_BIT(FRT0_TCCS,         0xB0708006,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT0_TSTPTCLK,     0xB0708008,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT0_ETCCS,        0xB070800A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT0_CIMSZIMS,     0xB070800C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT0_DMACFG,       0xB070800E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT1
 **
 ***************************************************************************/
__IO_REG16(    FRT1_TCDT,         0xB0708400,__READ_WRITE );
__IO_REG16(    FRT1_CPCLRB,       0xB0708402,__READ_WRITE );
__IO_REG16(    FRT1_CPCLR,        0xB0708404,__READ_WRITE );
__IO_REG16_BIT(FRT1_TCCS,         0xB0708406,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT1_TSTPTCLK,     0xB0708408,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT1_ETCCS,        0xB070840A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT1_CIMSZIMS,     0xB070840C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT1_DMACFG,       0xB070840E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT2
 **
 ***************************************************************************/
__IO_REG16(    FRT2_TCDT,         0xB0708800,__READ_WRITE );
__IO_REG16(    FRT2_CPCLRB,       0xB0708802,__READ_WRITE );
__IO_REG16(    FRT2_CPCLR,        0xB0708804,__READ_WRITE );
__IO_REG16_BIT(FRT2_TCCS,         0xB0708806,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT2_TSTPTCLK,     0xB0708808,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT2_ETCCS,        0xB070880A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT2_CIMSZIMS,     0xB070880C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT2_DMACFG,       0xB070880E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT3
 **
 ***************************************************************************/
__IO_REG16(    FRT3_TCDT,         0xB0708C00,__READ_WRITE );
__IO_REG16(    FRT3_CPCLRB,       0xB0708C02,__READ_WRITE );
__IO_REG16(    FRT3_CPCLR,        0xB0708C04,__READ_WRITE );
__IO_REG16_BIT(FRT3_TCCS,         0xB0708C06,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT3_TSTPTCLK,     0xB0708C08,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT3_ETCCS,        0xB0708C0A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT3_CIMSZIMS,     0xB0708C0C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT3_DMACFG,       0xB0708C0E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT16
 **
 ***************************************************************************/
__IO_REG16(    FRT16_TCDT,        0xB0818000,__READ_WRITE );
__IO_REG16(    FRT16_CPCLRB,      0xB0818002,__READ_WRITE );
__IO_REG16(    FRT16_CPCLR,       0xB0818004,__READ_WRITE );
__IO_REG16_BIT(FRT16_TCCS,        0xB0818006,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT16_TSTPTCLK,    0xB0818008,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT16_ETCCS,       0xB081800A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT16_CIMSZIMS,    0xB081800C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT16_DMACFG,      0xB081800E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT17
 **
 ***************************************************************************/
__IO_REG16(    FRT17_TCDT,        0xB0818400,__READ_WRITE );
__IO_REG16(    FRT17_CPCLRB,      0xB0818402,__READ_WRITE );
__IO_REG16(    FRT17_CPCLR,       0xB0818404,__READ_WRITE );
__IO_REG16_BIT(FRT17_TCCS,        0xB0818406,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT17_TSTPTCLK,    0xB0818408,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT17_ETCCS,       0xB081840A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT17_CIMSZIMS,    0xB081840C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT17_DMACFG,      0xB081840E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT18
 **
 ***************************************************************************/
__IO_REG16(    FRT18_TCDT,        0xB0818800,__READ_WRITE );
__IO_REG16(    FRT18_CPCLRB,      0xB0818802,__READ_WRITE );
__IO_REG16(    FRT18_CPCLR,       0xB0818804,__READ_WRITE );
__IO_REG16_BIT(FRT18_TCCS,        0xB0818806,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT18_TSTPTCLK,    0xB0818808,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT18_ETCCS,       0xB081880A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT18_CIMSZIMS,    0xB081880C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT18_DMACFG,      0xB081880E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** FRT19
 **
 ***************************************************************************/
__IO_REG16(    FRT19_TCDT,        0xB0818C00,__READ_WRITE );
__IO_REG16(    FRT19_CPCLRB,      0xB0818C02,__READ_WRITE );
__IO_REG16(    FRT19_CPCLR,       0xB0818C04,__READ_WRITE );
__IO_REG16_BIT(FRT19_TCCS,        0xB0818C06,__READ_WRITE ,__frtn_tccs_bits);
__IO_REG16_BIT(FRT19_TSTPTCLK,    0xB0818C08,__READ_WRITE ,__frtn_tstptclk_bits);
__IO_REG16_BIT(FRT19_ETCCS,       0xB0818C0A,__READ_WRITE ,__frtn_etccs_bits);
__IO_REG16_BIT(FRT19_CIMSZIMS,    0xB0818C0C,__READ_WRITE ,__frtn_cimszims_bits);
__IO_REG16_BIT(FRT19_DMACFG,      0xB0818C0E,__READ_WRITE ,__frtn_dmacfg_bits);

/***************************************************************************
 **
 ** OCU0
 **
 ***************************************************************************/
__IO_REG16(    OCU0_OCCP0,        0xB0718000,__READ       );
__IO_REG16(    OCU0_OCCP1,        0xB0718002,__READ       );
__IO_REG16(    OCU0_OCCPB0,       0xB0718004,__READ_WRITE );
__IO_REG16(    OCU0_OCCPB1,       0xB0718006,__READ_WRITE );
__IO_REG16(    OCU0_OCCPBD0,      0xB0718008,__READ_WRITE );
__IO_REG16(    OCU0_OCCPBD1,      0xB071800A,__READ_WRITE );
__IO_REG16_BIT(OCU0_OCS01,        0xB071800C,__READ_WRITE ,__ocun_ocs01_bits);
__IO_REG16_BIT(OCU0_OCSC01,       0xB071800E,__READ_WRITE ,__ocun_ocsc01_bits);
__IO_REG16_BIT(OCU0_OCSS01,       0xB0718010,__READ_WRITE ,__ocun_ocss01_bits);
__IO_REG16_BIT(OCU0_OSR01,        0xB0718012,__READ       ,__ocun_osr01_bits);
__IO_REG16_BIT(OCU0_OSCR01,       0xB0718014,__READ_WRITE ,__ocun_oscr01_bits);
__IO_REG16_BIT(OCU0_EOCS01,       0xB0718016,__READ_WRITE ,__ocun_eocs01_bits);
__IO_REG8_BIT( OCU0_EOCSSH01,     0xB0718019,__READ_WRITE ,__ocun_eocssh01_bits);
__IO_REG8_BIT( OCU0_EOCSCH01,     0xB071801B,__READ_WRITE ,__ocun_eocsch01_bits);
__IO_REG8_BIT( OCU0_DMACFG01,     0xB071801C,__READ_WRITE ,__ocun_dmacfg01_bits);
__IO_REG8_BIT( OCU0_DEBUG01,      0xB071801D,__READ_WRITE ,__ocun_debug01_bits);
__IO_REG8_BIT( OCU0_OCMCR01,      0xB071801F,__READ_WRITE ,__ocun_ocmcr01_bits);

/***************************************************************************
 **
 ** OCU1
 **
 ***************************************************************************/
__IO_REG16(    OCU1_OCCP0,        0xB0718400,__READ       );
__IO_REG16(    OCU1_OCCP1,        0xB0718402,__READ       );
__IO_REG16(    OCU1_OCCPB0,       0xB0718404,__READ_WRITE );
__IO_REG16(    OCU1_OCCPB1,       0xB0718406,__READ_WRITE );
__IO_REG16(    OCU1_OCCPBD0,      0xB0718408,__READ_WRITE );
__IO_REG16(    OCU1_OCCPBD1,      0xB071840A,__READ_WRITE );
__IO_REG16_BIT(OCU1_OCS01,        0xB071840C,__READ_WRITE ,__ocun_ocs01_bits);
__IO_REG16_BIT(OCU1_OCSC01,       0xB071840E,__READ_WRITE ,__ocun_ocsc01_bits);
__IO_REG16_BIT(OCU1_OCSS01,       0xB0718410,__READ_WRITE ,__ocun_ocss01_bits);
__IO_REG16_BIT(OCU1_OSR01,        0xB0718412,__READ       ,__ocun_osr01_bits);
__IO_REG16_BIT(OCU1_OSCR01,       0xB0718414,__READ_WRITE ,__ocun_oscr01_bits);
__IO_REG16_BIT(OCU1_EOCS01,       0xB0718416,__READ_WRITE ,__ocun_eocs01_bits);
__IO_REG8_BIT( OCU1_EOCSSH01,     0xB0718419,__READ_WRITE ,__ocun_eocssh01_bits);
__IO_REG8_BIT( OCU1_EOCSCH01,     0xB071841B,__READ_WRITE ,__ocun_eocsch01_bits);
__IO_REG8_BIT( OCU1_DMACFG01,     0xB071841C,__READ_WRITE ,__ocun_dmacfg01_bits);
__IO_REG8_BIT( OCU1_DEBUG01,      0xB071841D,__READ_WRITE ,__ocun_debug01_bits);
__IO_REG8_BIT( OCU1_OCMCR01,      0xB071841F,__READ_WRITE ,__ocun_ocmcr01_bits);

/***************************************************************************
 **
 ** OCU16
 **
 ***************************************************************************/
__IO_REG16(    OCU16_OCCP0,       0xB0828000,__READ       );
__IO_REG16(    OCU16_OCCP1,       0xB0828002,__READ       );
__IO_REG16(    OCU16_OCCPB0,      0xB0828004,__READ_WRITE );
__IO_REG16(    OCU16_OCCPB1,      0xB0828006,__READ_WRITE );
__IO_REG16(    OCU16_OCCPBD0,     0xB0828008,__READ_WRITE );
__IO_REG16(    OCU16_OCCPBD1,     0xB082800A,__READ_WRITE );
__IO_REG16_BIT(OCU16_OCS01,       0xB082800C,__READ_WRITE ,__ocun_ocs01_bits);
__IO_REG16_BIT(OCU16_OCSC01,      0xB082800E,__READ_WRITE ,__ocun_ocsc01_bits);
__IO_REG16_BIT(OCU16_OCSS01,      0xB0828010,__READ_WRITE ,__ocun_ocss01_bits);
__IO_REG16_BIT(OCU16_OSR01,       0xB0828012,__READ       ,__ocun_osr01_bits);
__IO_REG16_BIT(OCU16_OSCR01,      0xB0828014,__READ_WRITE ,__ocun_oscr01_bits);
__IO_REG16_BIT(OCU16_EOCS01,      0xB0828016,__READ_WRITE ,__ocun_eocs01_bits);
__IO_REG8_BIT( OCU16_EOCSSH01,    0xB0828019,__READ_WRITE ,__ocun_eocssh01_bits);
__IO_REG8_BIT( OCU16_EOCSCH01,    0xB082801B,__READ_WRITE ,__ocun_eocsch01_bits);
__IO_REG8_BIT( OCU16_DMACFG01,    0xB082801C,__READ_WRITE ,__ocun_dmacfg01_bits);
__IO_REG8_BIT( OCU16_DEBUG01,     0xB082801D,__READ_WRITE ,__ocun_debug01_bits);
__IO_REG8_BIT( OCU16_OCMCR01,     0xB082801F,__READ_WRITE ,__ocun_ocmcr01_bits);

/***************************************************************************
 **
 ** OCU17
 **
 ***************************************************************************/
__IO_REG16(    OCU17_OCCP0,       0xB0828400,__READ       );
__IO_REG16(    OCU17_OCCP1,       0xB0828402,__READ       );
__IO_REG16(    OCU17_OCCPB0,      0xB0828404,__READ_WRITE );
__IO_REG16(    OCU17_OCCPB1,      0xB0828406,__READ_WRITE );
__IO_REG16(    OCU17_OCCPBD0,     0xB0828408,__READ_WRITE );
__IO_REG16(    OCU17_OCCPBD1,     0xB082840A,__READ_WRITE );
__IO_REG16_BIT(OCU17_OCS01,       0xB082840C,__READ_WRITE ,__ocun_ocs01_bits);
__IO_REG16_BIT(OCU17_OCSC01,      0xB082840E,__READ_WRITE ,__ocun_ocsc01_bits);
__IO_REG16_BIT(OCU17_OCSS01,      0xB0828410,__READ_WRITE ,__ocun_ocss01_bits);
__IO_REG16_BIT(OCU17_OSR01,       0xB0828412,__READ       ,__ocun_osr01_bits);
__IO_REG16_BIT(OCU17_OSCR01,      0xB0828414,__READ_WRITE ,__ocun_oscr01_bits);
__IO_REG16_BIT(OCU17_EOCS01,      0xB0828416,__READ_WRITE ,__ocun_eocs01_bits);
__IO_REG8_BIT( OCU17_EOCSSH01,    0xB0828419,__READ_WRITE ,__ocun_eocssh01_bits);
__IO_REG8_BIT( OCU17_EOCSCH01,    0xB082841B,__READ_WRITE ,__ocun_eocsch01_bits);
__IO_REG8_BIT( OCU17_DMACFG01,    0xB082841C,__READ_WRITE ,__ocun_dmacfg01_bits);
__IO_REG8_BIT( OCU17_DEBUG01,     0xB082841D,__READ_WRITE ,__ocun_debug01_bits);
__IO_REG8_BIT( OCU17_OCMCR01,     0xB082841F,__READ_WRITE ,__ocun_ocmcr01_bits);

/***************************************************************************
 **
 ** ICU2
 **
 ***************************************************************************/
__IO_REG16(    ICU2_IPC0,         0xB0710800,__READ       );
__IO_REG16(    ICU2_IPC1,         0xB0710802,__READ       );
__IO_REG16_BIT(ICU2_ICC01,        0xB0710804,__READ_WRITE ,__icun_icc01_bits);
__IO_REG16_BIT(ICU2_ICEICS01,     0xB0710806,__READ_WRITE ,__icun_iceics01_bits);
__IO_REG8_BIT( ICU2_DMACFG01,     0xB0710808,__READ_WRITE ,__icun_dmacfg01_bits);
__IO_REG8_BIT( ICU2_DEBUG01,      0xB0710809,__READ_WRITE ,__icun_debug01_bits);

/***************************************************************************
 **
 ** ICU3
 **
 ***************************************************************************/
__IO_REG16(    ICU3_IPC0,         0xB0710C00,__READ       );
__IO_REG16(    ICU3_IPC1,         0xB0710C02,__READ       );
__IO_REG16_BIT(ICU3_ICC01,        0xB0710C04,__READ_WRITE ,__icun_icc01_bits);
__IO_REG16_BIT(ICU3_ICEICS01,     0xB0710C06,__READ_WRITE ,__icun_iceics01_bits);
__IO_REG8_BIT( ICU3_DMACFG01,     0xB0710C08,__READ_WRITE ,__icun_dmacfg01_bits);
__IO_REG8_BIT( ICU3_DEBUG01,      0xB0710C09,__READ_WRITE ,__icun_debug01_bits);

/***************************************************************************
 **
 ** ICU18
 **
 ***************************************************************************/
__IO_REG16(    ICU18_IPC0,        0xB0820800,__READ       );
__IO_REG16(    ICU18_IPC1,        0xB0820802,__READ       );
__IO_REG16_BIT(ICU18_ICC01,       0xB0820804,__READ_WRITE ,__icun_icc01_bits);
__IO_REG16_BIT(ICU18_ICEICS01,    0xB0820806,__READ_WRITE ,__icun_iceics01_bits);
__IO_REG8_BIT( ICU18_DMACFG01,    0xB0820808,__READ_WRITE ,__icun_dmacfg01_bits);
__IO_REG8_BIT( ICU18_DEBUG01,     0xB0820809,__READ_WRITE ,__icun_debug01_bits);

/***************************************************************************
 **
 ** ICU19
 **
 ***************************************************************************/
__IO_REG16(    ICU19_IPC0,        0xB0820C00,__READ       );
__IO_REG16(    ICU19_IPC1,        0xB0820C02,__READ       );
__IO_REG16_BIT(ICU19_ICC01,       0xB0820C04,__READ_WRITE ,__icun_icc01_bits);
__IO_REG16_BIT(ICU19_ICEICS01,    0xB0820C06,__READ_WRITE ,__icun_iceics01_bits);
__IO_REG8_BIT( ICU19_DMACFG01,    0xB0820C08,__READ_WRITE ,__icun_dmacfg01_bits);
__IO_REG8_BIT( ICU19_DEBUG01,     0xB0820C09,__READ_WRITE ,__icun_debug01_bits);

/***************************************************************************
 **
 ** PPGGLC
 **
 ***************************************************************************/
__IO_REG8_BIT( PPGGLC0_GCNR,      0xB074C000,__READ_WRITE ,__ppgglcg_gcnr_bits);
__IO_REG8_BIT( PPGGLC1_GCNR,      0xB085C000,__READ_WRITE ,__ppgglcg_gcnr_bits);

/***************************************************************************
 **
 ** PPGCRP
 **
 ***************************************************************************/
__IO_REG8_BIT( PPGGRP0_GCTRL,     0xB0748000,__READ_WRITE ,__ppggrpp_gctrl_bits);
__IO_REG8_BIT( PPGGRP1_GCTRL,     0xB0748400,__READ_WRITE ,__ppggrpp_gctrl_bits);
__IO_REG8_BIT( PPGGRP2_GCTRL,     0xB0748800,__READ_WRITE ,__ppggrpp_gctrl_bits);
__IO_REG8_BIT( PPGGRP3_GCTRL,     0xB0748C00,__READ_WRITE ,__ppggrpp_gctrl_bits);
__IO_REG8_BIT( PPGGRP16_GCTRL,    0xB0858000,__READ_WRITE ,__ppggrpp_gctrl_bits);
__IO_REG8_BIT( PPGGRP17_GCTRL,    0xB0858400,__READ_WRITE ,__ppggrpp_gctrl_bits);

/***************************************************************************
 **
 ** PPG0
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG0_PCN,          0xB0738000,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG0_IRQCLR,       0xB0738002,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG0_SWTRIG,       0xB0738003,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG0_OE,           0xB0738004,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG0_CNTEN,        0xB0738005,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG0_OPTMSK,       0xB0738006,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG0_RMPCFG,       0xB0738007,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG0_STRD,         0xB0738008,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG0_TRIGCLR,      0xB0738009,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG0_EPCN1,        0xB073800A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG0_EPCN2,        0xB073800C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG0_GCN1,         0xB073800E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG0_GCN3,         0xB073800F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG0_GCN4,         0xB0738010,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG0_GCN5,         0xB0738011,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG0_PCSR,         0xB0738012,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG0_PDUT,         0xB0738014,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG0_PTMR,         0xB0738016,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG0_PSDR,         0xB0738018,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG0_PTPC,         0xB073801A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG0_PEDR,         0xB073801C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG0_DMACFG,       0xB073801E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG0_DEBUG,        0xB073801F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG1
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG1_PCN,          0xB0738400,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG1_IRQCLR,       0xB0738402,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG1_SWTRIG,       0xB0738403,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG1_OE,           0xB0738404,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG1_CNTEN,        0xB0738405,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG1_OPTMSK,       0xB0738406,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG1_RMPCFG,       0xB0738407,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG1_STRD,         0xB0738408,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG1_TRIGCLR,      0xB0738409,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG1_EPCN1,        0xB073840A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG1_EPCN2,        0xB073840C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG1_GCN1,         0xB073840E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG1_GCN3,         0xB073840F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG1_GCN4,         0xB0738410,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG1_GCN5,         0xB0738411,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG1_PCSR,         0xB0738412,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG1_PDUT,         0xB0738414,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG1_PTMR,         0xB0738416,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG1_PSDR,         0xB0738418,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG1_PTPC,         0xB073841A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG1_PEDR,         0xB073841C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG1_DMACFG,       0xB073841E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG1_DEBUG,        0xB073841F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG2
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG2_PCN,          0xB0738800,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG2_IRQCLR,       0xB0738802,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG2_SWTRIG,       0xB0738803,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG2_OE,           0xB0738804,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG2_CNTEN,        0xB0738805,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG2_OPTMSK,       0xB0738806,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG2_RMPCFG,       0xB0738807,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG2_STRD,         0xB0738808,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG2_TRIGCLR,      0xB0738809,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG2_EPCN1,        0xB073880A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG2_EPCN2,        0xB073880C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG2_GCN1,         0xB073880E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG2_GCN3,         0xB073880F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG2_GCN4,         0xB0738810,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG2_GCN5,         0xB0738811,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG2_PCSR,         0xB0738812,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG2_PDUT,         0xB0738814,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG2_PTMR,         0xB0738816,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG2_PSDR,         0xB0738818,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG2_PTPC,         0xB073881A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG2_PEDR,         0xB073881C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG2_DMACFG,       0xB073881E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG2_DEBUG,        0xB073881F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG3
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG3_PCN,          0xB0738C00,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG3_IRQCLR,       0xB0738C02,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG3_SWTRIG,       0xB0738C03,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG3_OE,           0xB0738C04,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG3_CNTEN,        0xB0738C05,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG3_OPTMSK,       0xB0738C06,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG3_RMPCFG,       0xB0738C07,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG3_STRD,         0xB0738C08,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG3_TRIGCLR,      0xB0738C09,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG3_EPCN1,        0xB0738C0A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG3_EPCN2,        0xB0738C0C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG3_GCN1,         0xB0738C0E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG3_GCN3,         0xB0738C0F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG3_GCN4,         0xB0738C10,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG3_GCN5,         0xB0738C11,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG3_PCSR,         0xB0738C12,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG3_PDUT,         0xB0738C14,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG3_PTMR,         0xB0738C16,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG3_PSDR,         0xB0738C18,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG3_PTPC,         0xB0738C1A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG3_PEDR,         0xB0738C1C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG3_DMACFG,       0xB0738C1E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG3_DEBUG,        0xB0738C1F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG4
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG4_PCN,          0xB0739000,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG4_IRQCLR,       0xB0739002,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG4_SWTRIG,       0xB0739003,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG4_OE,           0xB0739004,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG4_CNTEN,        0xB0739005,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG4_OPTMSK,       0xB0739006,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG4_RMPCFG,       0xB0739007,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG4_STRD,         0xB0739008,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG4_TRIGCLR,      0xB0739009,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG4_EPCN1,        0xB073900A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG4_EPCN2,        0xB073900C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG4_GCN1,         0xB073900E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG4_GCN3,         0xB073900F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG4_GCN4,         0xB0739010,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG4_GCN5,         0xB0739011,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG4_PCSR,         0xB0739012,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG4_PDUT,         0xB0739014,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG4_PTMR,         0xB0739016,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG4_PSDR,         0xB0739018,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG4_PTPC,         0xB073901A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG4_PEDR,         0xB073901C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG4_DMACFG,       0xB073901E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG4_DEBUG,        0xB073901F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG5
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG5_PCN,          0xB0739400,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG5_IRQCLR,       0xB0739402,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG5_SWTRIG,       0xB0739403,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG5_OE,           0xB0739404,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG5_CNTEN,        0xB0739405,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG5_OPTMSK,       0xB0739406,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG5_RMPCFG,       0xB0739407,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG5_STRD,         0xB0739408,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG5_TRIGCLR,      0xB0739409,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG5_EPCN1,        0xB073940A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG5_EPCN2,        0xB073940C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG5_GCN1,         0xB073940E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG5_GCN3,         0xB073940F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG5_GCN4,         0xB0739410,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG5_GCN5,         0xB0739411,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG5_PCSR,         0xB0739412,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG5_PDUT,         0xB0739414,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG5_PTMR,         0xB0739416,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG5_PSDR,         0xB0739418,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG5_PTPC,         0xB073941A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG5_PEDR,         0xB073941C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG5_DMACFG,       0xB073941E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG5_DEBUG,        0xB073941F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG6
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG6_PCN,          0xB0739800,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG6_IRQCLR,       0xB0739802,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG6_SWTRIG,       0xB0739803,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG6_OE,           0xB0739804,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG6_CNTEN,        0xB0739805,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG6_OPTMSK,       0xB0739806,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG6_RMPCFG,       0xB0739807,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG6_STRD,         0xB0739808,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG6_TRIGCLR,      0xB0739809,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG6_EPCN1,        0xB073980A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG6_EPCN2,        0xB073980C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG6_GCN1,         0xB073980E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG6_GCN3,         0xB073980F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG6_GCN4,         0xB0739810,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG6_GCN5,         0xB0739811,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG6_PCSR,         0xB0739812,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG6_PDUT,         0xB0739814,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG6_PTMR,         0xB0739816,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG6_PSDR,         0xB0739818,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG6_PTPC,         0xB073981A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG6_PEDR,         0xB073981C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG6_DMACFG,       0xB073981E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG6_DEBUG,        0xB073981F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG7
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG7_PCN,          0xB0739C00,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG7_IRQCLR,       0xB0739C02,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG7_SWTRIG,       0xB0739C03,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG7_OE,           0xB0739C04,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG7_CNTEN,        0xB0739C05,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG7_OPTMSK,       0xB0739C06,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG7_RMPCFG,       0xB0739C07,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG7_STRD,         0xB0739C08,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG7_TRIGCLR,      0xB0739C09,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG7_EPCN1,        0xB0739C0A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG7_EPCN2,        0xB0739C0C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG7_GCN1,         0xB0739C0E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG7_GCN3,         0xB0739C0F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG7_GCN4,         0xB0739C10,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG7_GCN5,         0xB0739C11,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG7_PCSR,         0xB0739C12,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG7_PDUT,         0xB0739C14,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG7_PTMR,         0xB0739C16,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG7_PSDR,         0xB0739C18,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG7_PTPC,         0xB0739C1A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG7_PEDR,         0xB0739C1C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG7_DMACFG,       0xB0739C1E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG7_DEBUG,        0xB0739C1F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG8
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG8_PCN,          0xB073A000,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG8_IRQCLR,       0xB073A002,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG8_SWTRIG,       0xB073A003,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG8_OE,           0xB073A004,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG8_CNTEN,        0xB073A005,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG8_OPTMSK,       0xB073A006,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG8_RMPCFG,       0xB073A007,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG8_STRD,         0xB073A008,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG8_TRIGCLR,      0xB073A009,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG8_EPCN1,        0xB073A00A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG8_EPCN2,        0xB073A00C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG8_GCN1,         0xB073A00E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG8_GCN3,         0xB073A00F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG8_GCN4,         0xB073A010,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG8_GCN5,         0xB073A011,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG8_PCSR,         0xB073A012,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG8_PDUT,         0xB073A014,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG8_PTMR,         0xB073A016,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG8_PSDR,         0xB073A018,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG8_PTPC,         0xB073A01A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG8_PEDR,         0xB073A01C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG8_DMACFG,       0xB073A01E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG8_DEBUG,        0xB073A01F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG9
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG9_PCN,          0xB073A400,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG9_IRQCLR,       0xB073A402,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG9_SWTRIG,       0xB073A403,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG9_OE,           0xB073A404,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG9_CNTEN,        0xB073A405,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG9_OPTMSK,       0xB073A406,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG9_RMPCFG,       0xB073A407,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG9_STRD,         0xB073A408,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG9_TRIGCLR,      0xB073A409,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG9_EPCN1,        0xB073A40A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG9_EPCN2,        0xB073A40C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG9_GCN1,         0xB073A40E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG9_GCN3,         0xB073A40F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG9_GCN4,         0xB073A410,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG9_GCN5,         0xB073A411,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG9_PCSR,         0xB073A412,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG9_PDUT,         0xB073A414,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG9_PTMR,         0xB073A416,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG9_PSDR,         0xB073A418,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG9_PTPC,         0xB073A41A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG9_PEDR,         0xB073A41C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG9_DMACFG,       0xB073A41E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG9_DEBUG,        0xB073A41F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG10
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG10_PCN,         0xB073A800,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG10_IRQCLR,      0xB073A802,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG10_SWTRIG,      0xB073A803,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG10_OE,          0xB073A804,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG10_CNTEN,       0xB073A805,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG10_OPTMSK,      0xB073A806,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG10_RMPCFG,      0xB073A807,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG10_STRD,        0xB073A808,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG10_TRIGCLR,     0xB073A809,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG10_EPCN1,       0xB073A80A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG10_EPCN2,       0xB073A80C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG10_GCN1,        0xB073A80E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG10_GCN3,        0xB073A80F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG10_GCN4,        0xB073A810,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG10_GCN5,        0xB073A811,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG10_PCSR,        0xB073A812,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG10_PDUT,        0xB073A814,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG10_PTMR,        0xB073A816,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG10_PSDR,        0xB073A818,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG10_PTPC,        0xB073A81A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG10_PEDR,        0xB073A81C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG10_DMACFG,      0xB073A81E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG10_DEBUG,       0xB073A81F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG11
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG11_PCN,         0xB073AC00,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG11_IRQCLR,      0xB073AC02,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG11_SWTRIG,      0xB073AC03,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG11_OE,          0xB073AC04,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG11_CNTEN,       0xB073AC05,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG11_OPTMSK,      0xB073AC06,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG11_RMPCFG,      0xB073AC07,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG11_STRD,        0xB073AC08,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG11_TRIGCLR,     0xB073AC09,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG11_EPCN1,       0xB073AC0A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG11_EPCN2,       0xB073AC0C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG11_GCN1,        0xB073AC0E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG11_GCN3,        0xB073AC0F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG11_GCN4,        0xB073AC10,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG11_GCN5,        0xB073AC11,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG11_PCSR,        0xB073AC12,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG11_PDUT,        0xB073AC14,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG11_PTMR,        0xB073AC16,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG11_PSDR,        0xB073AC18,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG11_PTPC,        0xB073AC1A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG11_PEDR,        0xB073AC1C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG11_DMACFG,      0xB073AC1E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG11_DEBUG,       0xB073AC1F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG12
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG12_PCN,         0xB073B000,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG12_IRQCLR,      0xB073B002,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG12_SWTRIG,      0xB073B003,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG12_OE,          0xB073B004,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG12_CNTEN,       0xB073B005,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG12_OPTMSK,      0xB073B006,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG12_RMPCFG,      0xB073B007,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG12_STRD,        0xB073B008,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG12_TRIGCLR,     0xB073B009,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG12_EPCN1,       0xB073B00A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG12_EPCN2,       0xB073B00C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG12_GCN1,        0xB073B00E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG12_GCN3,        0xB073B00F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG12_GCN4,        0xB073B010,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG12_GCN5,        0xB073B011,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG12_PCSR,        0xB073B012,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG12_PDUT,        0xB073B014,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG12_PTMR,        0xB073B016,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG12_PSDR,        0xB073B018,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG12_PTPC,        0xB073B01A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG12_PEDR,        0xB073B01C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG12_DMACFG,      0xB073B01E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG12_DEBUG,       0xB073B01F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG13
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG13_PCN,         0xB073B400,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG13_IRQCLR,      0xB073B402,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG13_SWTRIG,      0xB073B403,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG13_OE,          0xB073B404,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG13_CNTEN,       0xB073B405,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG13_OPTMSK,      0xB073B406,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG13_RMPCFG,      0xB073B407,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG13_STRD,        0xB073B408,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG13_TRIGCLR,     0xB073B409,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG13_EPCN1,       0xB073B40A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG13_EPCN2,       0xB073B40C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG13_GCN1,        0xB073B40E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG13_GCN3,        0xB073B40F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG13_GCN4,        0xB073B410,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG13_GCN5,        0xB073B411,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG13_PCSR,        0xB073B412,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG13_PDUT,        0xB073B414,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG13_PTMR,        0xB073B416,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG13_PSDR,        0xB073B418,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG13_PTPC,        0xB073B41A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG13_PEDR,        0xB073B41C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG13_DMACFG,      0xB073B41E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG13_DEBUG,       0xB073B41F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG14
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG14_PCN,         0xB073B800,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG14_IRQCLR,      0xB073B802,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG14_SWTRIG,      0xB073B803,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG14_OE,          0xB073B804,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG14_CNTEN,       0xB073B805,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG14_OPTMSK,      0xB073B806,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG14_RMPCFG,      0xB073B807,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG14_STRD,        0xB073B808,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG14_TRIGCLR,     0xB073B809,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG14_EPCN1,       0xB073B80A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG14_EPCN2,       0xB073B80C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG14_GCN1,        0xB073B80E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG14_GCN3,        0xB073B80F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG14_GCN4,        0xB073B810,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG14_GCN5,        0xB073B811,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG14_PCSR,        0xB073B812,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG14_PDUT,        0xB073B814,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG14_PTMR,        0xB073B816,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG14_PSDR,        0xB073B818,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG14_PTPC,        0xB073B81A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG14_PEDR,        0xB073B81C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG14_DMACFG,      0xB073B81E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG14_DEBUG,       0xB073B81F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG15
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG15_PCN,         0xB073BC00,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG15_IRQCLR,      0xB073BC02,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG15_SWTRIG,      0xB073BC03,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG15_OE,          0xB073BC04,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG15_CNTEN,       0xB073BC05,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG15_OPTMSK,      0xB073BC06,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG15_RMPCFG,      0xB073BC07,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG15_STRD,        0xB073BC08,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG15_TRIGCLR,     0xB073BC09,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG15_EPCN1,       0xB073BC0A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG15_EPCN2,       0xB073BC0C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG15_GCN1,        0xB073BC0E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG15_GCN3,        0xB073BC0F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG15_GCN4,        0xB073BC10,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG15_GCN5,        0xB073BC11,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG15_PCSR,        0xB073BC12,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG15_PDUT,        0xB073BC14,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG15_PTMR,        0xB073BC16,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG15_PSDR,        0xB073BC18,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG15_PTPC,        0xB073BC1A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG15_PEDR,        0xB073BC1C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG15_DMACFG,      0xB073BC1E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG15_DEBUG,       0xB073BC1F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG64
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG64_PCN,         0xB0848000,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG64_IRQCLR,      0xB0848002,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG64_SWTRIG,      0xB0848003,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG64_OE,          0xB0848004,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG64_CNTEN,       0xB0848005,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG64_OPTMSK,      0xB0848006,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG64_RMPCFG,      0xB0848007,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG64_STRD,        0xB0848008,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG64_TRIGCLR,     0xB0848009,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG64_EPCN1,       0xB084800A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG64_EPCN2,       0xB084800C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG64_GCN1,        0xB084800E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG64_GCN3,        0xB084800F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG64_GCN4,        0xB0848010,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG64_GCN5,        0xB0848011,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG64_PCSR,        0xB0848012,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG64_PDUT,        0xB0848014,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG64_PTMR,        0xB0848016,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG64_PSDR,        0xB0848018,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG64_PTPC,        0xB084801A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG64_PEDR,        0xB084801C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG64_DMACFG,      0xB084801E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG64_DEBUG,       0xB084801F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG65
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG65_PCN,         0xB0848400,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG65_IRQCLR,      0xB0848402,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG65_SWTRIG,      0xB0848403,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG65_OE,          0xB0848404,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG65_CNTEN,       0xB0848405,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG65_OPTMSK,      0xB0848406,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG65_RMPCFG,      0xB0848407,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG65_STRD,        0xB0848408,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG65_TRIGCLR,     0xB0848409,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG65_EPCN1,       0xB084840A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG65_EPCN2,       0xB084840C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG65_GCN1,        0xB084840E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG65_GCN3,        0xB084840F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG65_GCN4,        0xB0848410,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG65_GCN5,        0xB0848411,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG65_PCSR,        0xB0848412,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG65_PDUT,        0xB0848414,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG65_PTMR,        0xB0848416,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG65_PSDR,        0xB0848418,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG65_PTPC,        0xB084841A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG65_PEDR,        0xB084841C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG65_DMACFG,      0xB084841E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG65_DEBUG,       0xB084841F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG66
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG66_PCN,         0xB0848800,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG66_IRQCLR,      0xB0848802,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG66_SWTRIG,      0xB0848803,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG66_OE,          0xB0848804,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG66_CNTEN,       0xB0848805,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG66_OPTMSK,      0xB0848806,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG66_RMPCFG,      0xB0848807,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG66_STRD,        0xB0848808,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG66_TRIGCLR,     0xB0848809,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG66_EPCN1,       0xB084880A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG66_EPCN2,       0xB084880C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG66_GCN1,        0xB084880E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG66_GCN3,        0xB084880F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG66_GCN4,        0xB0848810,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG66_GCN5,        0xB0848811,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG66_PCSR,        0xB0848812,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG66_PDUT,        0xB0848814,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG66_PTMR,        0xB0848816,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG66_PSDR,        0xB0848818,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG66_PTPC,        0xB084881A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG66_PEDR,        0xB084881C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG66_DMACFG,      0xB084881E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG66_DEBUG,       0xB084881F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG67
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG67_PCN,         0xB0848C00,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG67_IRQCLR,      0xB0848C02,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG67_SWTRIG,      0xB0848C03,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG67_OE,          0xB0848C04,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG67_CNTEN,       0xB0848C05,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG67_OPTMSK,      0xB0848C06,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG67_RMPCFG,      0xB0848C07,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG67_STRD,        0xB0848C08,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG67_TRIGCLR,     0xB0848C09,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG67_EPCN1,       0xB0848C0A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG67_EPCN2,       0xB0848C0C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG67_GCN1,        0xB0848C0E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG67_GCN3,        0xB0848C0F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG67_GCN4,        0xB0848C10,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG67_GCN5,        0xB0848C11,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG67_PCSR,        0xB0848C12,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG67_PDUT,        0xB0848C14,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG67_PTMR,        0xB0848C16,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG67_PSDR,        0xB0848C18,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG67_PTPC,        0xB0848C1A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG67_PEDR,        0xB0848C1C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG67_DMACFG,      0xB0848C1E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG67_DEBUG,       0xB0848C1F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG68
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG68_PCN,         0xB0849000,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG68_IRQCLR,      0xB0849002,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG68_SWTRIG,      0xB0849003,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG68_OE,          0xB0849004,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG68_CNTEN,       0xB0849005,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG68_OPTMSK,      0xB0849006,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG68_RMPCFG,      0xB0849007,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG68_STRD,        0xB0849008,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG68_TRIGCLR,     0xB0849009,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG68_EPCN1,       0xB084900A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG68_EPCN2,       0xB084900C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG68_GCN1,        0xB084900E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG68_GCN3,        0xB084900F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG68_GCN4,        0xB0849010,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG68_GCN5,        0xB0849011,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG68_PCSR,        0xB0849012,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG68_PDUT,        0xB0849014,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG68_PTMR,        0xB0849016,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG68_PSDR,        0xB0849018,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG68_PTPC,        0xB084901A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG68_PEDR,        0xB084901C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG68_DMACFG,      0xB084901E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG68_DEBUG,       0xB084901F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG69
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG69_PCN,         0xB0849400,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG69_IRQCLR,      0xB0849402,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG69_SWTRIG,      0xB0849403,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG69_OE,          0xB0849404,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG69_CNTEN,       0xB0849405,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG69_OPTMSK,      0xB0849406,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG69_RMPCFG,      0xB0849407,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG69_STRD,        0xB0849408,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG69_TRIGCLR,     0xB0849409,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG69_EPCN1,       0xB084940A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG69_EPCN2,       0xB084940C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG69_GCN1,        0xB084940E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG69_GCN3,        0xB084940F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG69_GCN4,        0xB0849410,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG69_GCN5,        0xB0849411,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG69_PCSR,        0xB0849412,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG69_PDUT,        0xB0849414,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG69_PTMR,        0xB0849416,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG69_PSDR,        0xB0849418,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG69_PTPC,        0xB084941A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG69_PEDR,        0xB084941C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG69_DMACFG,      0xB084941E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG69_DEBUG,       0xB084941F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG70
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG70_PCN,         0xB0849800,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG70_IRQCLR,      0xB0849802,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG70_SWTRIG,      0xB0849803,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG70_OE,          0xB0849804,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG70_CNTEN,       0xB0849805,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG70_OPTMSK,      0xB0849806,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG70_RMPCFG,      0xB0849807,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG70_STRD,        0xB0849808,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG70_TRIGCLR,     0xB0849809,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG70_EPCN1,       0xB084980A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG70_EPCN2,       0xB084980C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG70_GCN1,        0xB084980E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG70_GCN3,        0xB084980F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG70_GCN4,        0xB0849810,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG70_GCN5,        0xB0849811,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG70_PCSR,        0xB0849812,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG70_PDUT,        0xB0849814,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG70_PTMR,        0xB0849816,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG70_PSDR,        0xB0849818,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG70_PTPC,        0xB084981A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG70_PEDR,        0xB084981C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG70_DMACFG,      0xB084981E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG70_DEBUG,       0xB084981F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** PPG71
 **
 ***************************************************************************/
__IO_REG16_BIT(PPG71_PCN,         0xB0849C00,__READ_WRITE ,__ppgn_pcn_bits);
__IO_REG8_BIT( PPG71_IRQCLR,      0xB0849C02,__READ_WRITE ,__ppgn_irqclr_bits);
__IO_REG8_BIT( PPG71_SWTRIG,      0xB0849C03,__READ_WRITE ,__ppgn_swtrig_bits);
__IO_REG8_BIT( PPG71_OE,          0xB0849C04,__READ_WRITE ,__ppgn_oe_bits);
__IO_REG8_BIT( PPG71_CNTEN,       0xB0849C05,__READ_WRITE ,__ppgn_cnten_bits);
__IO_REG8_BIT( PPG71_OPTMSK,      0xB0849C06,__READ_WRITE ,__ppgn_optmsk_bits);
__IO_REG8_BIT( PPG71_RMPCFG,      0xB0849C07,__READ_WRITE ,__ppgn_rmpcfg_bits);
__IO_REG8_BIT( PPG71_STRD,        0xB0849C08,__READ_WRITE ,__ppgn_strd_bits);
__IO_REG8_BIT( PPG71_TRIGCLR,     0xB0849C09,__READ_WRITE ,__ppgn_trigclr_bits);
__IO_REG16_BIT(PPG71_EPCN1,       0xB0849C0A,__READ_WRITE ,__ppgn_epcn1_bits);
__IO_REG16_BIT(PPG71_EPCN2,       0xB0849C0C,__READ_WRITE ,__ppgn_epcn2_bits);
__IO_REG8_BIT( PPG71_GCN1,        0xB0849C0E,__READ_WRITE ,__ppgn_gcn1_bits);
__IO_REG8_BIT( PPG71_GCN3,        0xB0849C0F,__READ_WRITE ,__ppgn_gcn3_bits);
__IO_REG8_BIT( PPG71_GCN4,        0xB0849C10,__READ_WRITE ,__ppgn_gcn4_bits);
__IO_REG8_BIT( PPG71_GCN5,        0xB0849C11,__READ_WRITE ,__ppgn_gcn5_bits);
__IO_REG16_BIT(PPG71_PCSR,        0xB0849C12,__READ_WRITE ,__ppgn_pcsr_bits);
__IO_REG16_BIT(PPG71_PDUT,        0xB0849C14,__READ_WRITE ,__ppgn_pdut_bits);
__IO_REG16_BIT(PPG71_PTMR,        0xB0849C16,__READ       ,__ppgn_ptmr_bits);
__IO_REG16_BIT(PPG71_PSDR,        0xB0849C18,__READ_WRITE ,__ppgn_psdr_bits);
__IO_REG16_BIT(PPG71_PTPC,        0xB0849C1A,__READ_WRITE ,__ppgn_ptpc_bits);
__IO_REG16_BIT(PPG71_PEDR,        0xB0849C1C,__READ_WRITE ,__ppgn_pedr_bits);
__IO_REG8_BIT( PPG71_DMACFG,      0xB0849C1E,__READ_WRITE ,__ppgn_dmacfg_bits);
__IO_REG8_BIT( PPG71_DEBUG,       0xB0849C1F,__READ_WRITE ,__ppgn_debug_bits);

/***************************************************************************
 **
 ** RLT0
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT0_DMACFG,       0xB0A10000,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT0_TMCSR,        0xB0A10008,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT0_TMRLR,        0xB0A10010,__READ_WRITE );
__IO_REG32(    RLT0_TMR,          0xB0A10014,__READ       );

/***************************************************************************
 **
 ** RLT1
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT1_DMACFG,       0xB0A10400,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT1_TMCSR,        0xB0A10408,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT1_TMRLR,        0xB0A10410,__READ_WRITE );
__IO_REG32(    RLT1_TMR,          0xB0A10414,__READ       );

/***************************************************************************
 **
 ** RLT2
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT2_DMACFG,       0xB0A10800,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT2_TMCSR,        0xB0A10808,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT2_TMRLR,        0xB0A10810,__READ_WRITE );
__IO_REG32(    RLT2_TMR,          0xB0A10814,__READ       );

/***************************************************************************
 **
 ** RLT3
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT3_DMACFG,       0xB0A10C00,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT3_TMCSR,        0xB0A10C08,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT3_TMRLR,        0xB0A10C10,__READ_WRITE );
__IO_REG32(    RLT3_TMR,          0xB0A10C14,__READ       );

/***************************************************************************
 **
 ** RLT4
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT4_DMACFG,       0xB0A11000,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT4_TMCSR,        0xB0A11008,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT4_TMRLR,        0xB0A11010,__READ_WRITE );
__IO_REG32(    RLT4_TMR,          0xB0A11014,__READ       );

/***************************************************************************
 **
 ** RLT5
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT5_DMACFG,       0xB0A11400,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT5_TMCSR,        0xB0A11408,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT5_TMRLR,        0xB0A11410,__READ_WRITE );
__IO_REG32(    RLT5_TMR,          0xB0A11414,__READ       );

/***************************************************************************
 **
 ** RLT6
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT6_DMACFG,       0xB0A11800,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT6_TMCSR,        0xB0A11808,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT6_TMRLR,        0xB0A11810,__READ_WRITE );
__IO_REG32(    RLT6_TMR,          0xB0A11814,__READ       );

/***************************************************************************
 **
 ** RLT7
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT7_DMACFG,       0xB0A11C00,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT7_TMCSR,        0xB0A11C08,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT7_TMRLR,        0xB0A11C10,__READ_WRITE );
__IO_REG32(    RLT7_TMR,          0xB0A11C14,__READ       );

/***************************************************************************
 **
 ** RLT8
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT8_DMACFG,       0xB0A12000,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT8_TMCSR,        0xB0A12008,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT8_TMRLR,        0xB0A12010,__READ_WRITE );
__IO_REG32(    RLT8_TMR,          0xB0A12014,__READ       );

/***************************************************************************
 **
 ** RLT9
 **
 ***************************************************************************/
__IO_REG32_BIT(RLT9_DMACFG,       0xB0A12400,__READ_WRITE ,__rltn_dmacfg_bits);
__IO_REG32_BIT(RLT9_TMCSR,        0xB0A12408,__READ_WRITE ,__rltn_tmcsr_bits);
__IO_REG32(    RLT9_TMRLR,        0xB0A12410,__READ_WRITE );
__IO_REG32(    RLT9_TMR,          0xB0A12414,__READ       );

/***************************************************************************
 **
 ** SMC0
 **
 ***************************************************************************/
__IO_REG8_BIT( SMC0_PWC,          0xB0730000,__READ_WRITE ,__smcn_pwc_bits);
__IO_REG8_BIT( SMC0_PWCS,         0xB0730002,__READ_WRITE ,__smcn_pwcs_bits);
__IO_REG8_BIT( SMC0_PWCC,         0xB0730004,__READ_WRITE ,__smcn_pwcc_bits);
__IO_REG16_BIT(SMC0_PWC1,         0xB0730006,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC0_PWC2,         0xB0730008,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC0_PWS,          0xB073000A,__READ_WRITE ,__smcn_pws_bits);
__IO_REG16_BIT(SMC0_PWSS,         0xB073000C,__READ_WRITE ,__smcn_pwss_bits);
__IO_REG8(     SMC0_PTRGDL,       0xB073000E,__READ_WRITE );
__IO_REG8_BIT( SMC0_DEBUG,        0xB0730010,__READ_WRITE ,__smcn_debug_bits);

/***************************************************************************
 **
 ** SMC1
 **
 ***************************************************************************/
__IO_REG8_BIT( SMC1_PWC,          0xB0730400,__READ_WRITE ,__smcn_pwc_bits);
__IO_REG8_BIT( SMC1_PWCS,         0xB0730402,__READ_WRITE ,__smcn_pwcs_bits);
__IO_REG8_BIT( SMC1_PWCC,         0xB0730404,__READ_WRITE ,__smcn_pwcc_bits);
__IO_REG16_BIT(SMC1_PWC1,         0xB0730406,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC1_PWC2,         0xB0730408,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC1_PWS,          0xB073040A,__READ_WRITE ,__smcn_pws_bits);
__IO_REG16_BIT(SMC1_PWSS,         0xB073040C,__READ_WRITE ,__smcn_pwss_bits);
__IO_REG8(     SMC1_PTRGDL,       0xB073040E,__READ_WRITE );
__IO_REG8_BIT( SMC1_DEBUG,        0xB0730410,__READ_WRITE ,__smcn_debug_bits);

/***************************************************************************
 **
 ** SMC2
 **
 ***************************************************************************/
__IO_REG8_BIT( SMC2_PWC,          0xB0730800,__READ_WRITE ,__smcn_pwc_bits);
__IO_REG8_BIT( SMC2_PWCS,         0xB0730802,__READ_WRITE ,__smcn_pwcs_bits);
__IO_REG8_BIT( SMC2_PWCC,         0xB0730804,__READ_WRITE ,__smcn_pwcc_bits);
__IO_REG16_BIT(SMC2_PWC1,         0xB0730806,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC2_PWC2,         0xB0730808,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC2_PWS,          0xB073080A,__READ_WRITE ,__smcn_pws_bits);
__IO_REG16_BIT(SMC2_PWSS,         0xB073080C,__READ_WRITE ,__smcn_pwss_bits);
__IO_REG8(     SMC2_PTRGDL,       0xB073080E,__READ_WRITE );
__IO_REG8_BIT( SMC2_DEBUG,        0xB0730810,__READ_WRITE ,__smcn_debug_bits);

/***************************************************************************
 **
 ** SMC3
 **
 ***************************************************************************/
__IO_REG8_BIT( SMC3_PWC,          0xB0730C00,__READ_WRITE ,__smcn_pwc_bits);
__IO_REG8_BIT( SMC3_PWCS,         0xB0730C02,__READ_WRITE ,__smcn_pwcs_bits);
__IO_REG8_BIT( SMC3_PWCC,         0xB0730C04,__READ_WRITE ,__smcn_pwcc_bits);
__IO_REG16_BIT(SMC3_PWC1,         0xB0730C06,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC3_PWC2,         0xB0730C08,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC3_PWS,          0xB0730C0A,__READ_WRITE ,__smcn_pws_bits);
__IO_REG16_BIT(SMC3_PWSS,         0xB0730C0C,__READ_WRITE ,__smcn_pwss_bits);
__IO_REG8(     SMC3_PTRGDL,       0xB0730C0E,__READ_WRITE );
__IO_REG8_BIT( SMC3_DEBUG,        0xB0730C10,__READ_WRITE ,__smcn_debug_bits);

/***************************************************************************
 **
 ** SMC4
 **
 ***************************************************************************/
__IO_REG8_BIT( SMC4_PWC,          0xB0731000,__READ_WRITE ,__smcn_pwc_bits);
__IO_REG8_BIT( SMC4_PWCS,         0xB0731002,__READ_WRITE ,__smcn_pwcs_bits);
__IO_REG8_BIT( SMC4_PWCC,         0xB0731004,__READ_WRITE ,__smcn_pwcc_bits);
__IO_REG16_BIT(SMC4_PWC1,         0xB0731006,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC4_PWC2,         0xB0731008,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC4_PWS,          0xB073100A,__READ_WRITE ,__smcn_pws_bits);
__IO_REG16_BIT(SMC4_PWSS,         0xB073100C,__READ_WRITE ,__smcn_pwss_bits);
__IO_REG8(     SMC4_PTRGDL,       0xB073100E,__READ_WRITE );
__IO_REG8_BIT( SMC4_DEBUG,        0xB0731010,__READ_WRITE ,__smcn_debug_bits);

/***************************************************************************
 **
 ** SMC5
 **
 ***************************************************************************/
__IO_REG8_BIT( SMC5_PWC,          0xB0731400,__READ_WRITE ,__smcn_pwc_bits);
__IO_REG8_BIT( SMC5_PWCS,         0xB0731402,__READ_WRITE ,__smcn_pwcs_bits);
__IO_REG8_BIT( SMC5_PWCC,         0xB0731404,__READ_WRITE ,__smcn_pwcc_bits);
__IO_REG16_BIT(SMC5_PWC1,         0xB0731406,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC5_PWC2,         0xB0731408,__READ_WRITE ,__smcn_pwc1_bits);
__IO_REG16_BIT(SMC5_PWS,          0xB073140A,__READ_WRITE ,__smcn_pws_bits);
__IO_REG16_BIT(SMC5_PWSS,         0xB073140C,__READ_WRITE ,__smcn_pwss_bits);
__IO_REG8(     SMC5_PTRGDL,       0xB073140E,__READ_WRITE );
__IO_REG8_BIT( SMC5_DEBUG,        0xB0731410,__READ_WRITE ,__smcn_debug_bits);

/***************************************************************************
 **
 ** SMCTG0
 **
 ***************************************************************************/
__IO_REG16_BIT(SMCTG0_PTRG0,      0xB0731800,__READ_WRITE ,__smctgg_ptrg0_bits);
__IO_REG8_BIT( SMCTG0_PTRG1,      0xB0731802,__READ_WRITE ,__smctgg_ptrg1_bits);

/***************************************************************************
 **
 ** CAN0
 **
 ***************************************************************************/
__IO_REG16_BIT(CAN0_CTRLR,        0xB0808000,__READ_WRITE ,__can_ctrlr_bits);
__IO_REG16_BIT(CAN0_STATR,        0xB0808002,__READ_WRITE ,__can_statr_bits);
__IO_REG16_BIT(CAN0_ERRCNT,       0xB0808004,__READ       ,__can_errcnt_bits);
__IO_REG16_BIT(CAN0_BTR,          0xB0808006,__READ_WRITE ,__can_btr_bits);
__IO_REG16_BIT(CAN0_INTR,         0xB0808008,__READ       ,__can_intr_bits);
__IO_REG16_BIT(CAN0_TESTR,        0xB080800A,__READ_WRITE ,__can_testr_bits);
__IO_REG16_BIT(CAN0_BRPER,        0xB080800C,__READ_WRITE ,__can_brper_bits);
__IO_REG16_BIT(CAN0_IF1CREQ,      0xB0808010,__READ_WRITE ,__can_ifcreq_bits);
__IO_REG16_BIT(CAN0_IF1CMSK,      0xB0808012,__READ_WRITE ,__can_ifcmsk_bits);
__IO_REG16_BIT(CAN0_IF1MSK1,      0xB0808014,__READ_WRITE ,__can_ifmsk1_bits);
__IO_REG16_BIT(CAN0_IF1MSK2,      0xB0808016,__READ_WRITE ,__can_ifmsk2_bits);
__IO_REG16_BIT(CAN0_IF1ARB1,      0xB0808018,__READ_WRITE ,__can_ifarb1_bits);
__IO_REG16_BIT(CAN0_IF1ARB2,      0xB080801A,__READ_WRITE ,__can_ifarb2_bits);
__IO_REG16_BIT(CAN0_IF1MCTR,      0xB080801C,__READ_WRITE ,__can_ifmctr_bits);
__IO_REG16_BIT(CAN0_IF1DTA1,      0xB0808020,__READ_WRITE ,__can_ifdta1_bits);
__IO_REG16_BIT(CAN0_IF1DTA2,      0xB0808022,__READ_WRITE ,__can_ifdta2_bits);
__IO_REG16_BIT(CAN0_IF1DTB1,      0xB0808024,__READ_WRITE ,__can_ifdtb1_bits);
__IO_REG16_BIT(CAN0_IF1DTB2,      0xB0808026,__READ_WRITE ,__can_ifdtb2_bits);
__IO_REG16_BIT(CAN0_IF2CREQ,      0xB0808040,__READ_WRITE ,__can_ifcreq_bits);
__IO_REG16_BIT(CAN0_IF2CMSK,      0xB0808042,__READ_WRITE ,__can_ifcmsk_bits);
__IO_REG16_BIT(CAN0_IF2MSK1,      0xB0808044,__READ_WRITE ,__can_ifmsk1_bits);
__IO_REG16_BIT(CAN0_IF2MSK2,      0xB0808046,__READ_WRITE ,__can_ifmsk2_bits);
__IO_REG16_BIT(CAN0_IF2ARB1,      0xB0808048,__READ_WRITE ,__can_ifarb1_bits);
__IO_REG16_BIT(CAN0_IF2ARB2,      0xB080804A,__READ_WRITE ,__can_ifarb2_bits);
__IO_REG16_BIT(CAN0_IF2MCTR,      0xB080804C,__READ_WRITE ,__can_ifmctr_bits);
__IO_REG16_BIT(CAN0_IF2DTA1,      0xB0808050,__READ_WRITE ,__can_ifdta1_bits);
__IO_REG16_BIT(CAN0_IF2DTA2,      0xB0808052,__READ_WRITE ,__can_ifdta2_bits);
__IO_REG16_BIT(CAN0_IF2DTB1,      0xB0808054,__READ_WRITE ,__can_ifdtb1_bits);
__IO_REG16_BIT(CAN0_IF2DTB2,      0xB0808056,__READ_WRITE ,__can_ifdtb2_bits);
__IO_REG16_BIT(CAN0_TREQR1,       0xB0808080,__READ       ,__can_treqr1_bits);
__IO_REG16_BIT(CAN0_TREQR2,       0xB0808082,__READ       ,__can_treqr2_bits);
__IO_REG16_BIT(CAN0_TREQR3,       0xB0808084,__READ       ,__can_treqr3_bits);
__IO_REG16_BIT(CAN0_TREQR4,       0xB0808086,__READ       ,__can_treqr4_bits);
__IO_REG16_BIT(CAN0_TREQR5,       0xB0808088,__READ       ,__can_treqr5_bits);
__IO_REG16_BIT(CAN0_TREQR6,       0xB080808A,__READ       ,__can_treqr6_bits);
__IO_REG16_BIT(CAN0_TREQR7,       0xB080808C,__READ       ,__can_treqr7_bits);
__IO_REG16_BIT(CAN0_TREQR8,       0xB080808E,__READ       ,__can_treqr8_bits);
__IO_REG16_BIT(CAN0_NEWDT1,       0xB0808090,__READ       ,__can_newdt1_bits);
__IO_REG16_BIT(CAN0_NEWDT2,       0xB0808092,__READ       ,__can_newdt2_bits);
__IO_REG16_BIT(CAN0_NEWDT3,       0xB0808094,__READ       ,__can_newdt3_bits);
__IO_REG16_BIT(CAN0_NEWDT4,       0xB0808096,__READ       ,__can_newdt4_bits);
__IO_REG16_BIT(CAN0_NEWDT5,       0xB0808098,__READ       ,__can_newdt5_bits);
__IO_REG16_BIT(CAN0_NEWDT6,       0xB080809A,__READ       ,__can_newdt6_bits);
__IO_REG16_BIT(CAN0_NEWDT7,       0xB080809C,__READ       ,__can_newdt7_bits);
__IO_REG16_BIT(CAN0_NEWDT8,       0xB080809E,__READ       ,__can_newdt8_bits);
__IO_REG16_BIT(CAN0_INTPND1,      0xB08080A0,__READ       ,__can_intpnd1_bits);
__IO_REG16_BIT(CAN0_INTPND2,      0xB08080A2,__READ       ,__can_intpnd2_bits);
__IO_REG16_BIT(CAN0_INTPND3,      0xB08080A4,__READ       ,__can_intpnd3_bits);
__IO_REG16_BIT(CAN0_INTPND4,      0xB08080A6,__READ       ,__can_intpnd4_bits);
__IO_REG16_BIT(CAN0_INTPND5,      0xB08080A8,__READ       ,__can_intpnd5_bits);
__IO_REG16_BIT(CAN0_INTPND6,      0xB08080AA,__READ       ,__can_intpnd6_bits);
__IO_REG16_BIT(CAN0_INTPND7,      0xB08080AC,__READ       ,__can_intpnd7_bits);
__IO_REG16_BIT(CAN0_INTPND8,      0xB08080AE,__READ       ,__can_intpnd8_bits);
__IO_REG16_BIT(CAN0_MSGVAL1,      0xB08080B0,__READ       ,__can_msgval1_bits);
__IO_REG16_BIT(CAN0_MSGVAL2,      0xB08080B2,__READ       ,__can_msgval2_bits);
__IO_REG16_BIT(CAN0_MSGVAL3,      0xB08080B4,__READ       ,__can_msgval3_bits);
__IO_REG16_BIT(CAN0_MSGVAL4,      0xB08080B6,__READ       ,__can_msgval4_bits);
__IO_REG16_BIT(CAN0_MSGVAL5,      0xB08080B8,__READ       ,__can_msgval5_bits);
__IO_REG16_BIT(CAN0_MSGVAL6,      0xB08080BA,__READ       ,__can_msgval6_bits);
__IO_REG16_BIT(CAN0_MSGVAL7,      0xB08080BC,__READ       ,__can_msgval7_bits);
__IO_REG16_BIT(CAN0_MSGVAL8,      0xB08080BE,__READ       ,__can_msgval8_bits);
__IO_REG8_BIT( CAN0_COER,         0xB08080CE,__READ_WRITE ,__can_coer_bits);
__IO_REG16_BIT(CAN0_DEBUG,        0xB08080D0,__READ_WRITE ,__can_debug_bits);

/***************************************************************************
 **
 ** CAN1
 **
 ***************************************************************************/
__IO_REG16_BIT(CAN1_CTRLR,        0xB0808400,__READ_WRITE ,__can_ctrlr_bits);
__IO_REG16_BIT(CAN1_STATR,        0xB0808402,__READ_WRITE ,__can_statr_bits);
__IO_REG16_BIT(CAN1_ERRCNT,       0xB0808404,__READ       ,__can_errcnt_bits);
__IO_REG16_BIT(CAN1_BTR,          0xB0808406,__READ_WRITE ,__can_btr_bits);
__IO_REG16_BIT(CAN1_INTR,         0xB0808408,__READ       ,__can_intr_bits);
__IO_REG16_BIT(CAN1_TESTR,        0xB080840A,__READ_WRITE ,__can_testr_bits);
__IO_REG16_BIT(CAN1_BRPER,        0xB080840C,__READ_WRITE ,__can_brper_bits);
__IO_REG16_BIT(CAN1_IF1CREQ,      0xB0808410,__READ_WRITE ,__can_ifcreq_bits);
__IO_REG16_BIT(CAN1_IF1CMSK,      0xB0808412,__READ_WRITE ,__can_ifcmsk_bits);
__IO_REG16_BIT(CAN1_IF1MSK1,      0xB0808414,__READ_WRITE ,__can_ifmsk1_bits);
__IO_REG16_BIT(CAN1_IF1MSK2,      0xB0808416,__READ_WRITE ,__can_ifmsk2_bits);
__IO_REG16_BIT(CAN1_IF1ARB1,      0xB0808418,__READ_WRITE ,__can_ifarb1_bits);
__IO_REG16_BIT(CAN1_IF1ARB2,      0xB080841A,__READ_WRITE ,__can_ifarb2_bits);
__IO_REG16_BIT(CAN1_IF1MCTR,      0xB080841C,__READ_WRITE ,__can_ifmctr_bits);
__IO_REG16_BIT(CAN1_IF1DTA1,      0xB0808420,__READ_WRITE ,__can_ifdta1_bits);
__IO_REG16_BIT(CAN1_IF1DTA2,      0xB0808422,__READ_WRITE ,__can_ifdta2_bits);
__IO_REG16_BIT(CAN1_IF1DTB1,      0xB0808424,__READ_WRITE ,__can_ifdtb1_bits);
__IO_REG16_BIT(CAN1_IF1DTB2,      0xB0808426,__READ_WRITE ,__can_ifdtb2_bits);
__IO_REG16_BIT(CAN1_IF2CREQ,      0xB0808440,__READ_WRITE ,__can_ifcreq_bits);
__IO_REG16_BIT(CAN1_IF2CMSK,      0xB0808442,__READ_WRITE ,__can_ifcmsk_bits);
__IO_REG16_BIT(CAN1_IF2MSK1,      0xB0808444,__READ_WRITE ,__can_ifmsk1_bits);
__IO_REG16_BIT(CAN1_IF2MSK2,      0xB0808446,__READ_WRITE ,__can_ifmsk2_bits);
__IO_REG16_BIT(CAN1_IF2ARB1,      0xB0808448,__READ_WRITE ,__can_ifarb1_bits);
__IO_REG16_BIT(CAN1_IF2ARB2,      0xB080844A,__READ_WRITE ,__can_ifarb2_bits);
__IO_REG16_BIT(CAN1_IF2MCTR,      0xB080844C,__READ_WRITE ,__can_ifmctr_bits);
__IO_REG16_BIT(CAN1_IF2DTA1,      0xB0808450,__READ_WRITE ,__can_ifdta1_bits);
__IO_REG16_BIT(CAN1_IF2DTA2,      0xB0808452,__READ_WRITE ,__can_ifdta2_bits);
__IO_REG16_BIT(CAN1_IF2DTB1,      0xB0808454,__READ_WRITE ,__can_ifdtb1_bits);
__IO_REG16_BIT(CAN1_IF2DTB2,      0xB0808456,__READ_WRITE ,__can_ifdtb2_bits);
__IO_REG16_BIT(CAN1_TREQR1,       0xB0808480,__READ       ,__can_treqr1_bits);
__IO_REG16_BIT(CAN1_TREQR2,       0xB0808482,__READ       ,__can_treqr2_bits);
__IO_REG16_BIT(CAN1_TREQR3,       0xB0808484,__READ       ,__can_treqr3_bits);
__IO_REG16_BIT(CAN1_TREQR4,       0xB0808486,__READ       ,__can_treqr4_bits);
__IO_REG16_BIT(CAN1_TREQR5,       0xB0808488,__READ       ,__can_treqr5_bits);
__IO_REG16_BIT(CAN1_TREQR6,       0xB080848A,__READ       ,__can_treqr6_bits);
__IO_REG16_BIT(CAN1_TREQR7,       0xB080848C,__READ       ,__can_treqr7_bits);
__IO_REG16_BIT(CAN1_TREQR8,       0xB080848E,__READ       ,__can_treqr8_bits);
__IO_REG16_BIT(CAN1_NEWDT1,       0xB0808490,__READ       ,__can_newdt1_bits);
__IO_REG16_BIT(CAN1_NEWDT2,       0xB0808492,__READ       ,__can_newdt2_bits);
__IO_REG16_BIT(CAN1_NEWDT3,       0xB0808494,__READ       ,__can_newdt3_bits);
__IO_REG16_BIT(CAN1_NEWDT4,       0xB0808496,__READ       ,__can_newdt4_bits);
__IO_REG16_BIT(CAN1_NEWDT5,       0xB0808498,__READ       ,__can_newdt5_bits);
__IO_REG16_BIT(CAN1_NEWDT6,       0xB080849A,__READ       ,__can_newdt6_bits);
__IO_REG16_BIT(CAN1_NEWDT7,       0xB080849C,__READ       ,__can_newdt7_bits);
__IO_REG16_BIT(CAN1_NEWDT8,       0xB080849E,__READ       ,__can_newdt8_bits);
__IO_REG16_BIT(CAN1_INTPND1,      0xB08084A0,__READ       ,__can_intpnd1_bits);
__IO_REG16_BIT(CAN1_INTPND2,      0xB08084A2,__READ       ,__can_intpnd2_bits);
__IO_REG16_BIT(CAN1_INTPND3,      0xB08084A4,__READ       ,__can_intpnd3_bits);
__IO_REG16_BIT(CAN1_INTPND4,      0xB08084A6,__READ       ,__can_intpnd4_bits);
__IO_REG16_BIT(CAN1_INTPND5,      0xB08084A8,__READ       ,__can_intpnd5_bits);
__IO_REG16_BIT(CAN1_INTPND6,      0xB08084AA,__READ       ,__can_intpnd6_bits);
__IO_REG16_BIT(CAN1_INTPND7,      0xB08084AC,__READ       ,__can_intpnd7_bits);
__IO_REG16_BIT(CAN1_INTPND8,      0xB08084AE,__READ       ,__can_intpnd8_bits);
__IO_REG16_BIT(CAN1_MSGVAL1,      0xB08084B0,__READ       ,__can_msgval1_bits);
__IO_REG16_BIT(CAN1_MSGVAL2,      0xB08084B2,__READ       ,__can_msgval2_bits);
__IO_REG16_BIT(CAN1_MSGVAL3,      0xB08084B4,__READ       ,__can_msgval3_bits);
__IO_REG16_BIT(CAN1_MSGVAL4,      0xB08084B6,__READ       ,__can_msgval4_bits);
__IO_REG16_BIT(CAN1_MSGVAL5,      0xB08084B8,__READ       ,__can_msgval5_bits);
__IO_REG16_BIT(CAN1_MSGVAL6,      0xB08084BA,__READ       ,__can_msgval6_bits);
__IO_REG16_BIT(CAN1_MSGVAL7,      0xB08084BC,__READ       ,__can_msgval7_bits);
__IO_REG16_BIT(CAN1_MSGVAL8,      0xB08084BE,__READ       ,__can_msgval8_bits);
__IO_REG8_BIT( CAN1_COER,         0xB08084CE,__READ_WRITE ,__can_coer_bits);
__IO_REG16_BIT(CAN1_DEBUG,        0xB08084D0,__READ_WRITE ,__can_debug_bits);

/***************************************************************************
 **
 ** CAN2
 **
 ***************************************************************************/
__IO_REG16_BIT(CAN2_CTRLR,        0xB0808800,__READ_WRITE ,__can_ctrlr_bits);
__IO_REG16_BIT(CAN2_STATR,        0xB0808802,__READ_WRITE ,__can_statr_bits);
__IO_REG16_BIT(CAN2_ERRCNT,       0xB0808804,__READ       ,__can_errcnt_bits);
__IO_REG16_BIT(CAN2_BTR,          0xB0808806,__READ_WRITE ,__can_btr_bits);
__IO_REG16_BIT(CAN2_INTR,         0xB0808808,__READ       ,__can_intr_bits);
__IO_REG16_BIT(CAN2_TESTR,        0xB080880A,__READ_WRITE ,__can_testr_bits);
__IO_REG16_BIT(CAN2_BRPER,        0xB080880C,__READ_WRITE ,__can_brper_bits);
__IO_REG16_BIT(CAN2_IF1CREQ,      0xB0808810,__READ_WRITE ,__can_ifcreq_bits);
__IO_REG16_BIT(CAN2_IF1CMSK,      0xB0808812,__READ_WRITE ,__can_ifcmsk_bits);
__IO_REG16_BIT(CAN2_IF1MSK1,      0xB0808814,__READ_WRITE ,__can_ifmsk1_bits);
__IO_REG16_BIT(CAN2_IF1MSK2,      0xB0808816,__READ_WRITE ,__can_ifmsk2_bits);
__IO_REG16_BIT(CAN2_IF1ARB1,      0xB0808818,__READ_WRITE ,__can_ifarb1_bits);
__IO_REG16_BIT(CAN2_IF1ARB2,      0xB080881A,__READ_WRITE ,__can_ifarb2_bits);
__IO_REG16_BIT(CAN2_IF1MCTR,      0xB080881C,__READ_WRITE ,__can_ifmctr_bits);
__IO_REG16_BIT(CAN2_IF1DTA1,      0xB0808820,__READ_WRITE ,__can_ifdta1_bits);
__IO_REG16_BIT(CAN2_IF1DTA2,      0xB0808822,__READ_WRITE ,__can_ifdta2_bits);
__IO_REG16_BIT(CAN2_IF1DTB1,      0xB0808824,__READ_WRITE ,__can_ifdtb1_bits);
__IO_REG16_BIT(CAN2_IF1DTB2,      0xB0808826,__READ_WRITE ,__can_ifdtb2_bits);
__IO_REG16_BIT(CAN2_IF2CREQ,      0xB0808840,__READ_WRITE ,__can_ifcreq_bits);
__IO_REG16_BIT(CAN2_IF2CMSK,      0xB0808842,__READ_WRITE ,__can_ifcmsk_bits);
__IO_REG16_BIT(CAN2_IF2MSK1,      0xB0808844,__READ_WRITE ,__can_ifmsk1_bits);
__IO_REG16_BIT(CAN2_IF2MSK2,      0xB0808846,__READ_WRITE ,__can_ifmsk2_bits);
__IO_REG16_BIT(CAN2_IF2ARB1,      0xB0808848,__READ_WRITE ,__can_ifarb1_bits);
__IO_REG16_BIT(CAN2_IF2ARB2,      0xB080884A,__READ_WRITE ,__can_ifarb2_bits);
__IO_REG16_BIT(CAN2_IF2MCTR,      0xB080884C,__READ_WRITE ,__can_ifmctr_bits);
__IO_REG16_BIT(CAN2_IF2DTA1,      0xB0808850,__READ_WRITE ,__can_ifdta1_bits);
__IO_REG16_BIT(CAN2_IF2DTA2,      0xB0808852,__READ_WRITE ,__can_ifdta2_bits);
__IO_REG16_BIT(CAN2_IF2DTB1,      0xB0808854,__READ_WRITE ,__can_ifdtb1_bits);
__IO_REG16_BIT(CAN2_IF2DTB2,      0xB0808856,__READ_WRITE ,__can_ifdtb2_bits);
__IO_REG16_BIT(CAN2_TREQR1,       0xB0808880,__READ       ,__can_treqr1_bits);
__IO_REG16_BIT(CAN2_TREQR2,       0xB0808882,__READ       ,__can_treqr2_bits);
__IO_REG16_BIT(CAN2_TREQR3,       0xB0808884,__READ       ,__can_treqr3_bits);
__IO_REG16_BIT(CAN2_TREQR4,       0xB0808886,__READ       ,__can_treqr4_bits);
__IO_REG16_BIT(CAN2_TREQR5,       0xB0808888,__READ       ,__can_treqr5_bits);
__IO_REG16_BIT(CAN2_TREQR6,       0xB080888A,__READ       ,__can_treqr6_bits);
__IO_REG16_BIT(CAN2_TREQR7,       0xB080888C,__READ       ,__can_treqr7_bits);
__IO_REG16_BIT(CAN2_TREQR8,       0xB080888E,__READ       ,__can_treqr8_bits);
__IO_REG16_BIT(CAN2_NEWDT1,       0xB0808890,__READ       ,__can_newdt1_bits);
__IO_REG16_BIT(CAN2_NEWDT2,       0xB0808892,__READ       ,__can_newdt2_bits);
__IO_REG16_BIT(CAN2_NEWDT3,       0xB0808894,__READ       ,__can_newdt3_bits);
__IO_REG16_BIT(CAN2_NEWDT4,       0xB0808896,__READ       ,__can_newdt4_bits);
__IO_REG16_BIT(CAN2_NEWDT5,       0xB0808898,__READ       ,__can_newdt5_bits);
__IO_REG16_BIT(CAN2_NEWDT6,       0xB080889A,__READ       ,__can_newdt6_bits);
__IO_REG16_BIT(CAN2_NEWDT7,       0xB080889C,__READ       ,__can_newdt7_bits);
__IO_REG16_BIT(CAN2_NEWDT8,       0xB080889E,__READ       ,__can_newdt8_bits);
__IO_REG16_BIT(CAN2_INTPND1,      0xB08088A0,__READ       ,__can_intpnd1_bits);
__IO_REG16_BIT(CAN2_INTPND2,      0xB08088A2,__READ       ,__can_intpnd2_bits);
__IO_REG16_BIT(CAN2_INTPND3,      0xB08088A4,__READ       ,__can_intpnd3_bits);
__IO_REG16_BIT(CAN2_INTPND4,      0xB08088A6,__READ       ,__can_intpnd4_bits);
__IO_REG16_BIT(CAN2_INTPND5,      0xB08088A8,__READ       ,__can_intpnd5_bits);
__IO_REG16_BIT(CAN2_INTPND6,      0xB08088AA,__READ       ,__can_intpnd6_bits);
__IO_REG16_BIT(CAN2_INTPND7,      0xB08088AC,__READ       ,__can_intpnd7_bits);
__IO_REG16_BIT(CAN2_INTPND8,      0xB08088AE,__READ       ,__can_intpnd8_bits);
__IO_REG16_BIT(CAN2_MSGVAL1,      0xB08088B0,__READ       ,__can_msgval1_bits);
__IO_REG16_BIT(CAN2_MSGVAL2,      0xB08088B2,__READ       ,__can_msgval2_bits);
__IO_REG16_BIT(CAN2_MSGVAL3,      0xB08088B4,__READ       ,__can_msgval3_bits);
__IO_REG16_BIT(CAN2_MSGVAL4,      0xB08088B6,__READ       ,__can_msgval4_bits);
__IO_REG16_BIT(CAN2_MSGVAL5,      0xB08088B8,__READ       ,__can_msgval5_bits);
__IO_REG16_BIT(CAN2_MSGVAL6,      0xB08088BA,__READ       ,__can_msgval6_bits);
__IO_REG16_BIT(CAN2_MSGVAL7,      0xB08088BC,__READ       ,__can_msgval7_bits);
__IO_REG16_BIT(CAN2_MSGVAL8,      0xB08088BE,__READ       ,__can_msgval8_bits);
__IO_REG8_BIT( CAN2_COER,         0xB08088CE,__READ_WRITE ,__can_coer_bits);
__IO_REG16_BIT(CAN2_DEBUG,        0xB08088D0,__READ_WRITE ,__can_debug_bits);

/***************************************************************************
 **
 ** USART0
 **
 ***************************************************************************/
__IO_REG8_BIT( USART0_SMR,        0xB0728000,__READ_WRITE ,__usartn_smr_bits);
__IO_REG8_BIT( USART0_SCR,        0xB0728001,__READ_WRITE ,__usartn_scr_bits);
__IO_REG8_BIT( USART0_SMSR,       0xB0728002,__READ_WRITE ,__usartn_smsr_bits);
__IO_REG8_BIT( USART0_SCSR,       0xB0728003,__READ_WRITE ,__usartn_scsr_bits);
__IO_REG8_BIT( USART0_SCCR,       0xB0728005,__READ_WRITE ,__usartn_sccr_bits);
__IO_REG8(     USART0_TDR,        0xB0728006,__READ_WRITE );
__IO_REG8_BIT( USART0_SSR,        0xB0728007,__READ_WRITE ,__usartn_ssr_bits);
__IO_REG8(     USART0_RDR,        0xB0728008,__READ       );
__IO_REG8_BIT( USART0_SSSR,       0xB0728009,__READ_WRITE ,__usartn_sssr_bits);
__IO_REG8_BIT( USART0_SSCR,       0xB072800B,__READ_WRITE ,__usartn_sscr_bits);
__IO_REG8_BIT( USART0_ECCR,       0xB072800C,__READ_WRITE ,__usartn_eccr_bits);
__IO_REG8_BIT( USART0_ESCR,       0xB072800D,__READ_WRITE ,__usartn_escr_bits);
__IO_REG8_BIT( USART0_ECCSR,      0xB072800E,__READ_WRITE ,__usartn_eccsr_bits);
__IO_REG8_BIT( USART0_ESCSR,      0xB072800F,__READ_WRITE ,__usartn_escsr_bits);
__IO_REG8_BIT( USART0_ECCCR,      0xB0728010,__READ_WRITE ,__usartn_ecccr_bits);
__IO_REG8_BIT( USART0_ESCCR,      0xB0728011,__READ_WRITE ,__usartn_esccr_bits);
__IO_REG8_BIT( USART0_ESIR,       0xB0728012,__READ_WRITE ,__usartn_esir_bits);
__IO_REG8_BIT( USART0_EIER,       0xB0728013,__READ_WRITE ,__usartn_eier_bits);
__IO_REG8_BIT( USART0_ESISR,      0xB0728014,__READ_WRITE ,__usartn_esisr_bits);
__IO_REG8_BIT( USART0_EIESR,      0xB0728015,__READ_WRITE ,__usartn_eiesr_bits);
__IO_REG8_BIT( USART0_ESICR,      0xB0728016,__READ_WRITE ,__usartn_esicr_bits);
__IO_REG8_BIT( USART0_EIECR,      0xB0728017,__READ_WRITE ,__usartn_eiecr_bits);
__IO_REG8_BIT( USART0_EFERL,      0xB0728018,__READ_WRITE ,__usartn_eferl_bits);
__IO_REG8_BIT( USART0_EFERH,      0xB0728019,__READ_WRITE ,__usartn_eferh_bits);
__IO_REG8_BIT( USART0_RFCR,       0xB072801A,__READ_WRITE ,__usartn_rfcr_bits);
__IO_REG8_BIT( USART0_TFCR,       0xB072801B,__READ_WRITE ,__usartn_tfcr_bits);
__IO_REG8_BIT( USART0_RFCSR,      0xB072801C,__READ_WRITE ,__usartn_rfcsr_bits);
__IO_REG8_BIT( USART0_TFCSR,      0xB072801D,__READ_WRITE ,__usartn_tfcsr_bits);
__IO_REG8_BIT( USART0_RFCCR,      0xB072801E,__READ_WRITE ,__usartn_rfccr_bits);
__IO_REG8_BIT( USART0_TFCCR,      0xB072801F,__READ_WRITE ,__usartn_tfccr_bits);
__IO_REG8_BIT( USART0_RFSR,       0xB0728020,__READ       ,__usartn_rfsr_bits);
__IO_REG8_BIT( USART0_TFSR,       0xB0728021,__READ       ,__usartn_tfsr_bits);
__IO_REG8_BIT( USART0_CSCR,       0xB0728022,__READ_WRITE ,__usartn_cscr_bits);
__IO_REG8_BIT( USART0_ESR,        0xB0728023,__READ_WRITE ,__usartn_esr_bits);
__IO_REG8_BIT( USART0_CSCSR,      0xB0728024,__READ_WRITE ,__usartn_cscsr_bits);
__IO_REG8_BIT( USART0_CSCCR,      0xB0728026,__READ_WRITE ,__usartn_csccr_bits);
__IO_REG8_BIT( USART0_ESCLR,      0xB0728027,__READ_WRITE ,__usartn_esclr_bits);
__IO_REG8(     USART0_BGRLL,      0xB0728028,__WRITE      );
__IO_REG8(     USART0_BGRLM,      0xB0728029,__WRITE      );
__IO_REG8_BIT( USART0_BGRLH,      0xB072802A,__WRITE      ,__usartn_bgrlh_bits);
__IO_REG8(     USART0_BGRL,       0xB072802C,__READ       );
__IO_REG8(     USART0_BGRM,       0xB072802D,__READ       );
__IO_REG8_BIT( USART0_BGRH,       0xB072802E,__READ       ,__usartn_bgrh_bits);
__IO_REG8_BIT( USART0_STXDR,      0xB0728030,__READ_WRITE ,__usartn_stxdr_bits);
__IO_REG8_BIT( USART0_SRXDR,      0xB0728031,__READ_WRITE ,__usartn_srxdr_bits);
__IO_REG8_BIT( USART0_STXDSR,     0xB0728032,__READ_WRITE ,__usartn_stxdsr_bits);
__IO_REG8_BIT( USART0_SRXDSR,     0xB0728033,__READ_WRITE ,__usartn_srxdsr_bits);
__IO_REG8_BIT( USART0_STXDCR,     0xB0728034,__READ_WRITE ,__usartn_stxdcr_bits);
__IO_REG8_BIT( USART0_SRXDCR,     0xB0728035,__READ_WRITE ,__usartn_srxdcr_bits);
__IO_REG8(     USART0_SFTRL,      0xB0728036,__READ_WRITE );
__IO_REG8(     USART0_SFTRM,      0xB0728037,__READ_WRITE );
__IO_REG8_BIT( USART0_SFTRH,      0xB0728038,__READ_WRITE ,__usartn_sftrh_bits);
__IO_REG8(     USART0_FIDR,       0xB072803A,__READ_WRITE );
__IO_REG8_BIT( USART0_DEBUG,      0xB072803C,__READ_WRITE ,__usartn_debug_bits);

/***************************************************************************
 **
 ** USART6
 **
 ***************************************************************************/
__IO_REG8_BIT( USART6_SMR,        0xB0838000,__READ_WRITE ,__usartn_smr_bits);
__IO_REG8_BIT( USART6_SCR,        0xB0838001,__READ_WRITE ,__usartn_scr_bits);
__IO_REG8_BIT( USART6_SMSR,       0xB0838002,__READ_WRITE ,__usartn_smsr_bits);
__IO_REG8_BIT( USART6_SCSR,       0xB0838003,__READ_WRITE ,__usartn_scsr_bits);
__IO_REG8_BIT( USART6_SCCR,       0xB0838005,__READ_WRITE ,__usartn_sccr_bits);
__IO_REG8(     USART6_TDR,        0xB0838006,__READ_WRITE );
__IO_REG8_BIT( USART6_SSR,        0xB0838007,__READ_WRITE ,__usartn_ssr_bits);
__IO_REG8(     USART6_RDR,        0xB0838008,__READ       );
__IO_REG8_BIT( USART6_SSSR,       0xB0838009,__READ_WRITE ,__usartn_sssr_bits);
__IO_REG8_BIT( USART6_SSCR,       0xB083800B,__READ_WRITE ,__usartn_sscr_bits);
__IO_REG8_BIT( USART6_ECCR,       0xB083800C,__READ_WRITE ,__usartn_eccr_bits);
__IO_REG8_BIT( USART6_ESCR,       0xB083800D,__READ_WRITE ,__usartn_escr_bits);
__IO_REG8_BIT( USART6_ECCSR,      0xB083800E,__READ_WRITE ,__usartn_eccsr_bits);
__IO_REG8_BIT( USART6_ESCSR,      0xB083800F,__READ_WRITE ,__usartn_escsr_bits);
__IO_REG8_BIT( USART6_ECCCR,      0xB0838010,__READ_WRITE ,__usartn_ecccr_bits);
__IO_REG8_BIT( USART6_ESCCR,      0xB0838011,__READ_WRITE ,__usartn_esccr_bits);
__IO_REG8_BIT( USART6_ESIR,       0xB0838012,__READ_WRITE ,__usartn_esir_bits);
__IO_REG8_BIT( USART6_EIER,       0xB0838013,__READ_WRITE ,__usartn_eier_bits);
__IO_REG8_BIT( USART6_ESISR,      0xB0838014,__READ_WRITE ,__usartn_esisr_bits);
__IO_REG8_BIT( USART6_EIESR,      0xB0838015,__READ_WRITE ,__usartn_eiesr_bits);
__IO_REG8_BIT( USART6_ESICR,      0xB0838016,__READ_WRITE ,__usartn_esicr_bits);
__IO_REG8_BIT( USART6_EIECR,      0xB0838017,__READ_WRITE ,__usartn_eiecr_bits);
__IO_REG8_BIT( USART6_EFERL,      0xB0838018,__READ_WRITE ,__usartn_eferl_bits);
__IO_REG8_BIT( USART6_EFERH,      0xB0838019,__READ_WRITE ,__usartn_eferh_bits);
__IO_REG8_BIT( USART6_RFCR,       0xB083801A,__READ_WRITE ,__usartn_rfcr_bits);
__IO_REG8_BIT( USART6_TFCR,       0xB083801B,__READ_WRITE ,__usartn_tfcr_bits);
__IO_REG8_BIT( USART6_RFCSR,      0xB083801C,__READ_WRITE ,__usartn_rfcsr_bits);
__IO_REG8_BIT( USART6_TFCSR,      0xB083801D,__READ_WRITE ,__usartn_tfcsr_bits);
__IO_REG8_BIT( USART6_RFCCR,      0xB083801E,__READ_WRITE ,__usartn_rfccr_bits);
__IO_REG8_BIT( USART6_TFCCR,      0xB083801F,__READ_WRITE ,__usartn_tfccr_bits);
__IO_REG8_BIT( USART6_RFSR,       0xB0838020,__READ       ,__usartn_rfsr_bits);
__IO_REG8_BIT( USART6_TFSR,       0xB0838021,__READ       ,__usartn_tfsr_bits);
__IO_REG8_BIT( USART6_CSCR,       0xB0838022,__READ_WRITE ,__usartn_cscr_bits);
__IO_REG8_BIT( USART6_ESR,        0xB0838023,__READ_WRITE ,__usartn_esr_bits);
__IO_REG8_BIT( USART6_CSCSR,      0xB0838024,__READ_WRITE ,__usartn_cscsr_bits);
__IO_REG8_BIT( USART6_CSCCR,      0xB0838026,__READ_WRITE ,__usartn_csccr_bits);
__IO_REG8_BIT( USART6_ESCLR,      0xB0838027,__READ_WRITE ,__usartn_esclr_bits);
__IO_REG8(     USART6_BGRLL,      0xB0838028,__WRITE      );
__IO_REG8(     USART6_BGRLM,      0xB0838029,__WRITE      );
__IO_REG8_BIT( USART6_BGRLH,      0xB083802A,__WRITE      ,__usartn_bgrlh_bits);
__IO_REG8(     USART6_BGRL,       0xB083802C,__READ       );
__IO_REG8(     USART6_BGRM,       0xB083802D,__READ       );
__IO_REG8_BIT( USART6_BGRH,       0xB083802E,__READ       ,__usartn_bgrh_bits);
__IO_REG8_BIT( USART6_STXDR,      0xB0838030,__READ_WRITE ,__usartn_stxdr_bits);
__IO_REG8_BIT( USART6_SRXDR,      0xB0838031,__READ_WRITE ,__usartn_srxdr_bits);
__IO_REG8_BIT( USART6_STXDSR,     0xB0838032,__READ_WRITE ,__usartn_stxdsr_bits);
__IO_REG8_BIT( USART6_SRXDSR,     0xB0838033,__READ_WRITE ,__usartn_srxdsr_bits);
__IO_REG8_BIT( USART6_STXDCR,     0xB0838034,__READ_WRITE ,__usartn_stxdcr_bits);
__IO_REG8_BIT( USART6_SRXDCR,     0xB0838035,__READ_WRITE ,__usartn_srxdcr_bits);
__IO_REG8(     USART6_SFTRL,      0xB0838036,__READ_WRITE );
__IO_REG8(     USART6_SFTRM,      0xB0838037,__READ_WRITE );
__IO_REG8_BIT( USART6_SFTRH,      0xB0838038,__READ_WRITE ,__usartn_sftrh_bits);
__IO_REG8(     USART6_FIDR,       0xB083803A,__READ_WRITE );
__IO_REG8_BIT( USART6_DEBUG,      0xB083803C,__READ_WRITE ,__usartn_debug_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C0_IBCSR,        0xB0720000,__READ_WRITE ,__i2cn_ibcsr_bits);
__IO_REG16_BIT(I2C0_ITBA,         0xB0720002,__READ_WRITE ,__i2cn_itba_bits);
__IO_REG16_BIT(I2C0_ITMK,         0xB0720004,__READ_WRITE ,__i2cn_itmk_bits);
__IO_REG16_BIT(I2C0_ISBMA,        0xB0720006,__READ_WRITE ,__i2cn_isbma_bits);
__IO_REG8(     I2C0_IODAR,        0xB0720008,__READ_WRITE );
__IO_REG8_BIT( I2C0_ICCR,         0xB072000A,__READ_WRITE ,__i2cn_iccr_bits);
__IO_REG16_BIT(I2C0_ICDIDAR,      0xB072000C,__READ       ,__i2cn_icdidar_bits);
__IO_REG16_BIT(I2C0_IEICR,        0xB072000E,__READ_WRITE ,__i2cn_ieicr_bits);
__IO_REG16_BIT(I2C0_DDMACFG,      0xB0720010,__READ_WRITE ,__i2cn_ddmacfg_bits);
__IO_REG8_BIT( I2C0_IEIER,        0xB0720012,__READ_WRITE ,__i2cn_ieier_bits);

/***************************************************************************
 **
 ** HSSPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(HSSPI0_MCTRL,      0xB0000000,__READ_WRITE ,__hsspin_mctrl_bits);
__IO_REG32_BIT(HSSPI0_PCC0,       0xB0000004,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(HSSPI0_PCC1,       0xB0000008,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(HSSPI0_PCC2,       0xB000000C,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(HSSPI0_PCC3,       0xB0000010,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(HSSPI0_TXF,        0xB0000014,__READ_WRITE ,__hsspin_txf_bits);
__IO_REG32_BIT(HSSPI0_TXE,        0xB0000018,__READ_WRITE ,__hsspin_txe_bits);
__IO_REG32_BIT(HSSPI0_TXC,        0xB000001C,__READ_WRITE ,__hsspin_txc_bits);
__IO_REG32_BIT(HSSPI0_RXF,        0xB0000020,__READ       ,__hsspin_rxf_bits);
__IO_REG32_BIT(HSSPI0_RXE,        0xB0000024,__READ_WRITE ,__hsspin_rxe_bits);
__IO_REG32_BIT(HSSPI0_RXC,        0xB0000028,__READ_WRITE ,__hsspin_rxc_bits);
__IO_REG32_BIT(HSSPI0_FAULTF,     0xB000002C,__READ       ,__hsspin_faultf_bits);
__IO_REG32_BIT(HSSPI0_FAULTC,     0xB0000030,__READ_WRITE ,__hsspin_faultc_bits);
__IO_REG8_BIT( HSSPI0_DMCFG,      0xB0000034,__READ_WRITE ,__hsspin_dmcfg_bits);
__IO_REG8_BIT( HSSPI0_DMDMAEN,    0xB0000035,__READ_WRITE ,__hsspin_dmdmaen_bits);
__IO_REG8_BIT( HSSPI0_DMSTART,    0xB0000038,__READ_WRITE ,__hsspin_dmstart_bits);
__IO_REG8_BIT( HSSPI0_DMSTOP,     0xB0000039,__READ_WRITE ,__hsspin_dmstop_bits);
__IO_REG8_BIT( HSSPI0_DMPSEL,     0xB000003A,__READ_WRITE ,__hsspin_dmpsel_bits);
__IO_REG8_BIT( HSSPI0_DMTRP,      0xB000003B,__READ_WRITE ,__hsspin_dmtrp_bits);
__IO_REG16(    HSSPI0_DMBCC,      0xB000003C,__READ_WRITE );
__IO_REG16(    HSSPI0_DMBCS,      0xB000003E,__READ       );
__IO_REG32_BIT(HSSPI0_DMSTATUS,   0xB0000040,__READ       ,__hsspin_dmstatus_bits);
__IO_REG8_BIT( HSSPI0_TXBITCNT,   0xB0000044,__READ       ,__hsspin_txbitcnt_bits);
__IO_REG8_BIT( HSSPI0_RXBITCNT,   0xB0000045,__READ       ,__hsspin_rxbitcnt_bits);
__IO_REG32(    HSSPI0_RXSHIFT,    0xB0000048,__READ       );
__IO_REG32_BIT(HSSPI0_FIFOCFG,    0xB000004C,__READ_WRITE ,__hsspin_fifocfg_bits);
__IO_REG32(    HSSPI0_TXFIFO0,    0xB0000050,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO1,    0xB0000054,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO2,    0xB0000058,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO3,    0xB000005C,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO4,    0xB0000060,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO5,    0xB0000064,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO6,    0xB0000068,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO7,    0xB000006C,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO8,    0xB0000070,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO9,    0xB0000074,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO10,   0xB0000078,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO11,   0xB000007C,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO12,   0xB0000080,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO13,   0xB0000084,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO14,   0xB0000088,__READ_WRITE );
__IO_REG32(    HSSPI0_TXFIFO15,   0xB000008C,__READ_WRITE );
__IO_REG32(    HSSPI0_RXFIFO0,    0xB0000090,__READ       );
__IO_REG32(    HSSPI0_RXFIFO1,    0xB0000094,__READ       );
__IO_REG32(    HSSPI0_RXFIFO2,    0xB0000098,__READ       );
__IO_REG32(    HSSPI0_RXFIFO3,    0xB000009C,__READ       );
__IO_REG32(    HSSPI0_RXFIFO4,    0xB00000A0,__READ       );
__IO_REG32(    HSSPI0_RXFIFO5,    0xB00000A4,__READ       );
__IO_REG32(    HSSPI0_RXFIFO6,    0xB00000A8,__READ       );
__IO_REG32(    HSSPI0_RXFIFO7,    0xB00000AC,__READ       );
__IO_REG32(    HSSPI0_RXFIFO8,    0xB00000B0,__READ       );
__IO_REG32(    HSSPI0_RXFIFO9,    0xB00000B4,__READ       );
__IO_REG32(    HSSPI0_RXFIFO10,   0xB00000B8,__READ       );
__IO_REG32(    HSSPI0_RXFIFO11,   0xB00000BC,__READ       );
__IO_REG32(    HSSPI0_RXFIFO12,   0xB00000C0,__READ       );
__IO_REG32(    HSSPI0_RXFIFO13,   0xB00000C4,__READ       );
__IO_REG32(    HSSPI0_RXFIFO14,   0xB00000C8,__READ       );
__IO_REG32(    HSSPI0_RXFIFO15,   0xB00000CC,__READ       );
__IO_REG32_BIT(HSSPI0_CSCFG,      0xB00000D0,__READ_WRITE ,__hsspin_cscfg_bits);
__IO_REG32_BIT(HSSPI0_CSITIME,    0xB00000D4,__READ_WRITE ,__hsspin_csitime_bits);
__IO_REG32_BIT(HSSPI0_CSAEXT,     0xB00000D8,__READ_WRITE ,__hsspin_csaext_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC0,    0xB00000DC,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC1,    0xB00000DE,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC2,    0xB00000E0,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC3,    0xB00000E2,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC4,    0xB00000E4,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC5,    0xB00000E6,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC6,    0xB00000E8,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_RDCSDC7,    0xB00000EA,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC0,    0xB00000EC,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC1,    0xB00000EE,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC2,    0xB00000F0,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC3,    0xB00000F2,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC4,    0xB00000F4,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC5,    0xB00000F6,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC6,    0xB00000F8,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(HSSPI0_WRCSDC7,    0xB00000FA,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG32(    HSSPI0_MID,        0xB00000FC,__READ       );

/***************************************************************************
 **
 ** I2S0
 **
 ***************************************************************************/
__IO_REG32(    I2S0_RXFDAT0,      0xB0B20000,__READ       );
__IO_REG32(    I2S0_RXFDAT1,      0xB0B20004,__READ       );
__IO_REG32(    I2S0_RXFDAT2,      0xB0B20008,__READ       );
__IO_REG32(    I2S0_RXFDAT3,      0xB0B2000C,__READ       );
__IO_REG32(    I2S0_RXFDAT4,      0xB0B20010,__READ       );
__IO_REG32(    I2S0_RXFDAT5,      0xB0B20014,__READ       );
__IO_REG32(    I2S0_RXFDAT6,      0xB0B20018,__READ       );
__IO_REG32(    I2S0_RXFDAT7,      0xB0B2001C,__READ       );
__IO_REG32(    I2S0_RXFDAT8,      0xB0B20020,__READ       );
__IO_REG32(    I2S0_RXFDAT9,      0xB0B20024,__READ       );
__IO_REG32(    I2S0_RXFDAT10,     0xB0B20028,__READ       );
__IO_REG32(    I2S0_RXFDAT11,     0xB0B2002C,__READ       );
__IO_REG32(    I2S0_RXFDAT12,     0xB0B20030,__READ       );
__IO_REG32(    I2S0_RXFDAT13,     0xB0B20034,__READ       );
__IO_REG32(    I2S0_RXFDAT14,     0xB0B20038,__READ       );
__IO_REG32(    I2S0_RXFDAT15,     0xB0B2003C,__READ       );
__IO_REG32(    I2S0_TXFDAT0,      0xB0B20040,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT1,      0xB0B20044,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT2,      0xB0B20048,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT3,      0xB0B2004C,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT4,      0xB0B20050,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT5,      0xB0B20054,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT6,      0xB0B20058,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT7,      0xB0B2005C,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT8,      0xB0B20060,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT9,      0xB0B20064,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT10,     0xB0B20068,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT11,     0xB0B2006C,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT12,     0xB0B20070,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT13,     0xB0B20074,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT14,     0xB0B20078,__READ_WRITE );
__IO_REG32(    I2S0_TXFDAT15,     0xB0B2007C,__READ_WRITE );
__IO_REG32_BIT(I2S0_CNTREG,       0xB0B20080,__READ_WRITE ,__i2sn_cntreg_bits);
__IO_REG32_BIT(I2S0_MCR0REG,      0xB0B20084,__READ_WRITE ,__i2sn_mcr0reg_bits);
__IO_REG32(    I2S0_MCR1REG,      0xB0B20088,__READ_WRITE );
__IO_REG32(    I2S0_MCR2REG,      0xB0B2008C,__READ_WRITE );
__IO_REG32_BIT(I2S0_OPRREG,       0xB0B20090,__READ_WRITE ,__i2sn_oprreg_bits);
__IO_REG32_BIT(I2S0_SRST,         0xB0B20094,__READ_WRITE ,__i2sn_srst_bits);
__IO_REG32_BIT(I2S0_INTCNT,       0xB0B20098,__READ_WRITE ,__i2sn_intcnt_bits);
__IO_REG32_BIT(I2S0_STATUS,       0xB0B2009C,__READ_WRITE ,__i2sn_status_bits);
__IO_REG32_BIT(I2S0_DMAACT,       0xB0B200A0,__READ_WRITE ,__i2sn_dmaact_bits);
__IO_REG32_BIT(I2S0_DEBUG,        0xB0B200A4,__READ_WRITE ,__i2sn_debug_bits);
__IO_REG32(    I2S0_MIDREG,       0xB0B200A8,__READ       );

/***************************************************************************
 **
 ** I2S1
 **
 ***************************************************************************/
__IO_REG32(    I2S1_RXFDAT0,      0xB0B20400,__READ       );
__IO_REG32(    I2S1_RXFDAT1,      0xB0B20404,__READ       );
__IO_REG32(    I2S1_RXFDAT2,      0xB0B20408,__READ       );
__IO_REG32(    I2S1_RXFDAT3,      0xB0B2040C,__READ       );
__IO_REG32(    I2S1_RXFDAT4,      0xB0B20410,__READ       );
__IO_REG32(    I2S1_RXFDAT5,      0xB0B20414,__READ       );
__IO_REG32(    I2S1_RXFDAT6,      0xB0B20418,__READ       );
__IO_REG32(    I2S1_RXFDAT7,      0xB0B2041C,__READ       );
__IO_REG32(    I2S1_RXFDAT8,      0xB0B20420,__READ       );
__IO_REG32(    I2S1_RXFDAT9,      0xB0B20424,__READ       );
__IO_REG32(    I2S1_RXFDAT10,     0xB0B20428,__READ       );
__IO_REG32(    I2S1_RXFDAT11,     0xB0B2042C,__READ       );
__IO_REG32(    I2S1_RXFDAT12,     0xB0B20430,__READ       );
__IO_REG32(    I2S1_RXFDAT13,     0xB0B20434,__READ       );
__IO_REG32(    I2S1_RXFDAT14,     0xB0B20438,__READ       );
__IO_REG32(    I2S1_RXFDAT15,     0xB0B2043C,__READ       );
__IO_REG32(    I2S1_TXFDAT0,      0xB0B20440,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT1,      0xB0B20444,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT2,      0xB0B20448,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT3,      0xB0B2044C,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT4,      0xB0B20450,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT5,      0xB0B20454,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT6,      0xB0B20458,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT7,      0xB0B2045C,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT8,      0xB0B20460,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT9,      0xB0B20464,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT10,     0xB0B20468,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT11,     0xB0B2046C,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT12,     0xB0B20470,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT13,     0xB0B20474,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT14,     0xB0B20478,__READ_WRITE );
__IO_REG32(    I2S1_TXFDAT15,     0xB0B2047C,__READ_WRITE );
__IO_REG32_BIT(I2S1_CNTREG,       0xB0B20480,__READ_WRITE ,__i2sn_cntreg_bits);
__IO_REG32_BIT(I2S1_MCR0REG,      0xB0B20484,__READ_WRITE ,__i2sn_mcr0reg_bits);
__IO_REG32(    I2S1_MCR1REG,      0xB0B20488,__READ_WRITE );
__IO_REG32(    I2S1_MCR2REG,      0xB0B2048C,__READ_WRITE );
__IO_REG32_BIT(I2S1_OPRREG,       0xB0B20490,__READ_WRITE ,__i2sn_oprreg_bits);
__IO_REG32_BIT(I2S1_SRST,         0xB0B20494,__READ_WRITE ,__i2sn_srst_bits);
__IO_REG32_BIT(I2S1_INTCNT,       0xB0B20498,__READ_WRITE ,__i2sn_intcnt_bits);
__IO_REG32_BIT(I2S1_STATUS,       0xB0B2049C,__READ_WRITE ,__i2sn_status_bits);
__IO_REG32_BIT(I2S1_DMAACT,       0xB0B204A0,__READ_WRITE ,__i2sn_dmaact_bits);
__IO_REG32_BIT(I2S1_DEBUG,        0xB0B204A4,__READ_WRITE ,__i2sn_debug_bits);
__IO_REG32(    I2S1_MIDREG,       0xB0B204A8,__READ       );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO_POSR0L,       0xB0A08000,__READ_WRITE ,__gpio_posr0l_bits);
__IO_REG32_BIT(GPIO_POSR0H,       0xB0A08004,__READ_WRITE ,__gpio_posr0h_bits);
__IO_REG32_BIT(GPIO_POCR0L,       0xB0A08008,__READ_WRITE ,__gpio_pocr0l_bits);
__IO_REG32_BIT(GPIO_POCR0H,       0xB0A0800C,__READ_WRITE ,__gpio_pocr0h_bits);
__IO_REG32_BIT(GPIO_DDSR0L,       0xB0A08010,__READ_WRITE ,__gpio_ddsr0l_bits);
__IO_REG32_BIT(GPIO_DDSR0H,       0xB0A08014,__READ_WRITE ,__gpio_ddsr0h_bits);
__IO_REG32_BIT(GPIO_DDCR0L,       0xB0A08018,__READ_WRITE ,__gpio_ddcr0l_bits);
__IO_REG32_BIT(GPIO_DDCR0H,       0xB0A0801C,__READ_WRITE ,__gpio_ddcr0h_bits);
__IO_REG32_BIT(GPIO_POSR1L,       0xB0A08020,__READ_WRITE ,__gpio_posr1l_bits);
__IO_REG32_BIT(GPIO_POSR1H,       0xB0A08024,__READ_WRITE ,__gpio_posr1h_bits);
__IO_REG32_BIT(GPIO_POCR1L,       0xB0A08028,__READ_WRITE ,__gpio_pocr1l_bits);
__IO_REG32_BIT(GPIO_POCR1H,       0xB0A0802C,__READ_WRITE ,__gpio_pocr1h_bits);
__IO_REG32_BIT(GPIO_DDSR1L,       0xB0A08030,__READ_WRITE ,__gpio_ddsr1l_bits);
__IO_REG32_BIT(GPIO_DDSR1H,       0xB0A08034,__READ_WRITE ,__gpio_ddsr1h_bits);
__IO_REG32_BIT(GPIO_DDCR1L,       0xB0A08038,__READ_WRITE ,__gpio_ddcr1l_bits);
__IO_REG32_BIT(GPIO_DDCR1H,       0xB0A0803C,__READ_WRITE ,__gpio_ddcr1h_bits);
__IO_REG32_BIT(GPIO_POSR2L,       0xB0A08040,__READ_WRITE ,__gpio_posr2l_bits);
__IO_REG32_BIT(GPIO_POSR2H,       0xB0A08044,__READ_WRITE ,__gpio_posr2h_bits);
__IO_REG32_BIT(GPIO_POCR2L,       0xB0A08048,__READ_WRITE ,__gpio_pocr2l_bits);
__IO_REG32_BIT(GPIO_POCR2H,       0xB0A0804C,__READ_WRITE ,__gpio_pocr2h_bits);
__IO_REG32_BIT(GPIO_DDSR2L,       0xB0A08050,__READ_WRITE ,__gpio_ddsr2l_bits);
__IO_REG32_BIT(GPIO_DDSR2H,       0xB0A08054,__READ_WRITE ,__gpio_ddsr2h_bits);
__IO_REG32_BIT(GPIO_DDCR2L,       0xB0A08058,__READ_WRITE ,__gpio_ddcr2l_bits);
__IO_REG32_BIT(GPIO_DDCR2H,       0xB0A0805C,__READ_WRITE ,__gpio_ddcr2h_bits);
__IO_REG32_BIT(GPIO_POSR3L,       0xB0A08060,__READ_WRITE ,__gpio_posr3l_bits);
__IO_REG32_BIT(GPIO_POSR3H,       0xB0A08064,__READ_WRITE ,__gpio_posr3h_bits);
__IO_REG32_BIT(GPIO_POCR3L,       0xB0A08068,__READ_WRITE ,__gpio_pocr3l_bits);
__IO_REG32_BIT(GPIO_POCR3H,       0xB0A0806C,__READ_WRITE ,__gpio_pocr3h_bits);
__IO_REG32_BIT(GPIO_DDSR3L,       0xB0A08070,__READ_WRITE ,__gpio_ddsr3l_bits);
__IO_REG32_BIT(GPIO_DDSR3H,       0xB0A08074,__READ_WRITE ,__gpio_ddsr3h_bits);
__IO_REG32_BIT(GPIO_DDCR3L,       0xB0A08078,__READ_WRITE ,__gpio_ddcr3l_bits);
__IO_REG32_BIT(GPIO_DDCR3H,       0xB0A0807C,__READ_WRITE ,__gpio_ddcr3h_bits);
__IO_REG32_BIT(GPIO_POSR4L,       0xB0A08080,__READ_WRITE ,__gpio_posr4l_bits);
__IO_REG32_BIT(GPIO_POSR4H,       0xB0A08084,__READ_WRITE ,__gpio_posr4h_bits);
__IO_REG32_BIT(GPIO_POCR4L,       0xB0A08088,__READ_WRITE ,__gpio_pocr4l_bits);
__IO_REG32_BIT(GPIO_POCR4H,       0xB0A0808C,__READ_WRITE ,__gpio_pocr4h_bits);
__IO_REG32_BIT(GPIO_DDSR4L,       0xB0A08090,__READ_WRITE ,__gpio_ddsr4l_bits);
__IO_REG32_BIT(GPIO_DDSR4H,       0xB0A08094,__READ_WRITE ,__gpio_ddsr4h_bits);
__IO_REG32_BIT(GPIO_DDCR4L,       0xB0A08098,__READ_WRITE ,__gpio_ddcr4l_bits);
__IO_REG32_BIT(GPIO_DDCR4H,       0xB0A0809C,__READ_WRITE ,__gpio_ddcr4h_bits);
__IO_REG32_BIT(GPIO_POSR5L,       0xB0A080A0,__READ_WRITE ,__gpio_posr5l_bits);
__IO_REG32_BIT(GPIO_POSR5H,       0xB0A080A4,__READ_WRITE ,__gpio_posr5h_bits);
__IO_REG32_BIT(GPIO_POCR5L,       0xB0A080A8,__READ_WRITE ,__gpio_pocr5l_bits);
__IO_REG32_BIT(GPIO_POCR5H,       0xB0A080AC,__READ_WRITE ,__gpio_pocr5h_bits);
__IO_REG32_BIT(GPIO_DDSR5L,       0xB0A080B0,__READ_WRITE ,__gpio_ddsr5l_bits);
__IO_REG32_BIT(GPIO_DDSR5H,       0xB0A080B4,__READ_WRITE ,__gpio_ddsr5h_bits);
__IO_REG32_BIT(GPIO_DDCR5L,       0xB0A080B8,__READ_WRITE ,__gpio_ddcr5l_bits);
__IO_REG32_BIT(GPIO_DDCR5H,       0xB0A080BC,__READ_WRITE ,__gpio_ddcr5h_bits);
__IO_REG32_BIT(GPIO_POSR6L,       0xB0A080C0,__READ_WRITE ,__gpio_posr6l_bits);
__IO_REG32_BIT(GPIO_POSR6H,       0xB0A080C4,__READ_WRITE ,__gpio_posr6h_bits);
__IO_REG32_BIT(GPIO_POCR6L,       0xB0A080C8,__READ_WRITE ,__gpio_pocr6l_bits);
__IO_REG32_BIT(GPIO_POCR6H,       0xB0A080CC,__READ_WRITE ,__gpio_pocr6h_bits);
__IO_REG32_BIT(GPIO_DDSR6L,       0xB0A080D0,__READ_WRITE ,__gpio_ddsr6l_bits);
__IO_REG32_BIT(GPIO_DDSR6H,       0xB0A080D4,__READ_WRITE ,__gpio_ddsr6h_bits);
__IO_REG32_BIT(GPIO_DDCR6L,       0xB0A080D8,__READ_WRITE ,__gpio_ddcr6l_bits);
__IO_REG32_BIT(GPIO_DDCR6H,       0xB0A080DC,__READ_WRITE ,__gpio_ddcr6h_bits);
__IO_REG32_BIT(GPIO_POSR7L,       0xB0A080E0,__READ_WRITE ,__gpio_posr7l_bits);
__IO_REG32_BIT(GPIO_POSR7H,       0xB0A080E4,__READ_WRITE ,__gpio_posr7h_bits);
__IO_REG32_BIT(GPIO_POCR7L,       0xB0A080E8,__READ_WRITE ,__gpio_pocr7l_bits);
__IO_REG32_BIT(GPIO_POCR7H,       0xB0A080EC,__READ_WRITE ,__gpio_pocr7h_bits);
__IO_REG32_BIT(GPIO_DDSR7L,       0xB0A080F0,__READ_WRITE ,__gpio_ddsr7l_bits);
__IO_REG32_BIT(GPIO_DDSR7H,       0xB0A080F4,__READ_WRITE ,__gpio_ddsr7h_bits);
__IO_REG32_BIT(GPIO_DDCR7L,       0xB0A080F8,__READ_WRITE ,__gpio_ddcr7l_bits);
__IO_REG32_BIT(GPIO_DDCR7H,       0xB0A080FC,__READ_WRITE ,__gpio_ddcr7h_bits);
__IO_REG32_BIT(GPIO_PODR0L,       0xB0A08200,__READ_WRITE ,__gpio_podr0l_bits);
__IO_REG32_BIT(GPIO_PODR0H,       0xB0A08204,__READ_WRITE ,__gpio_podr0h_bits);
__IO_REG32_BIT(GPIO_DDR0L,        0xB0A08208,__READ_WRITE ,__gpio_ddr0l_bits);
__IO_REG32_BIT(GPIO_DDR0H,        0xB0A0820C,__READ_WRITE ,__gpio_ddr0h_bits);
__IO_REG32_BIT(GPIO_PODR1L,       0xB0A08210,__READ_WRITE ,__gpio_podr1l_bits);
__IO_REG32_BIT(GPIO_PODR1H,       0xB0A08214,__READ_WRITE ,__gpio_podr1h_bits);
__IO_REG32_BIT(GPIO_DDR1L,        0xB0A08218,__READ_WRITE ,__gpio_ddr1l_bits);
__IO_REG32_BIT(GPIO_DDR1H,        0xB0A0821C,__READ_WRITE ,__gpio_ddr1h_bits);
__IO_REG32_BIT(GPIO_PODR2L,       0xB0A08220,__READ_WRITE ,__gpio_podr2l_bits);
__IO_REG32_BIT(GPIO_PODR2H,       0xB0A08224,__READ_WRITE ,__gpio_podr2h_bits);
__IO_REG32_BIT(GPIO_DDR2L,        0xB0A08228,__READ_WRITE ,__gpio_ddr2l_bits);
__IO_REG32_BIT(GPIO_DDR2H,        0xB0A0822C,__READ_WRITE ,__gpio_ddr2h_bits);
__IO_REG32_BIT(GPIO_PODR3L,       0xB0A08230,__READ_WRITE ,__gpio_podr3l_bits);
__IO_REG32_BIT(GPIO_PODR3H,       0xB0A08234,__READ_WRITE ,__gpio_podr3h_bits);
__IO_REG32_BIT(GPIO_DDR3L,        0xB0A08238,__READ_WRITE ,__gpio_ddr3l_bits);
__IO_REG32_BIT(GPIO_DDR3H,        0xB0A0823C,__READ_WRITE ,__gpio_ddr3h_bits);
__IO_REG32_BIT(GPIO_PODR4L,       0xB0A08240,__READ_WRITE ,__gpio_podr4l_bits);
__IO_REG32_BIT(GPIO_PODR4H,       0xB0A08244,__READ_WRITE ,__gpio_podr4h_bits);
__IO_REG32_BIT(GPIO_DDR4L,        0xB0A08248,__READ_WRITE ,__gpio_ddr4l_bits);
__IO_REG32_BIT(GPIO_DDR4H,        0xB0A0824C,__READ_WRITE ,__gpio_ddr4h_bits);
__IO_REG32_BIT(GPIO_PODR5L,       0xB0A08250,__READ_WRITE ,__gpio_podr5l_bits);
__IO_REG32_BIT(GPIO_PODR5H,       0xB0A08254,__READ_WRITE ,__gpio_podr5h_bits);
__IO_REG32_BIT(GPIO_DDR5L,        0xB0A08258,__READ_WRITE ,__gpio_ddr5l_bits);
__IO_REG32_BIT(GPIO_DDR5H,        0xB0A0825C,__READ_WRITE ,__gpio_ddr5h_bits);
__IO_REG32_BIT(GPIO_PODR6L,       0xB0A08260,__READ_WRITE ,__gpio_podr6l_bits);
__IO_REG32_BIT(GPIO_PODR6H,       0xB0A08264,__READ_WRITE ,__gpio_podr6h_bits);
__IO_REG32_BIT(GPIO_DDR6L,        0xB0A08268,__READ_WRITE ,__gpio_ddr6l_bits);
__IO_REG32_BIT(GPIO_DDR6H,        0xB0A0826C,__READ_WRITE ,__gpio_ddr6h_bits);
__IO_REG32_BIT(GPIO_PODR7L,       0xB0A08270,__READ_WRITE ,__gpio_podr7l_bits);
__IO_REG32_BIT(GPIO_PODR7H,       0xB0A08274,__READ_WRITE ,__gpio_podr7h_bits);
__IO_REG32_BIT(GPIO_DDR7L,        0xB0A08278,__READ_WRITE ,__gpio_ddr7l_bits);
__IO_REG32_BIT(GPIO_DDR7H,        0xB0A0827C,__READ_WRITE ,__gpio_ddr7h_bits);
__IO_REG32_BIT(GPIO_PIDR0L,       0xB0A08300,__READ       ,__gpio_pidr0l_bits);
__IO_REG32_BIT(GPIO_PIDR0H,       0xB0A08304,__READ       ,__gpio_pidr0h_bits);
__IO_REG32_BIT(GPIO_PIDR1L,       0xB0A08308,__READ       ,__gpio_pidr1l_bits);
__IO_REG32_BIT(GPIO_PIDR1H,       0xB0A0830C,__READ       ,__gpio_pidr1h_bits);
__IO_REG32_BIT(GPIO_PIDR2L,       0xB0A08310,__READ       ,__gpio_pidr2l_bits);
__IO_REG32_BIT(GPIO_PIDR2H,       0xB0A08314,__READ       ,__gpio_pidr2h_bits);
__IO_REG32_BIT(GPIO_PIDR3L,       0xB0A08318,__READ       ,__gpio_pidr3l_bits);
__IO_REG32_BIT(GPIO_PIDR3H,       0xB0A0831C,__READ       ,__gpio_pidr3h_bits);
__IO_REG32_BIT(GPIO_PIDR4L,       0xB0A08320,__READ       ,__gpio_pidr4l_bits);
__IO_REG32_BIT(GPIO_PIDR4H,       0xB0A08324,__READ       ,__gpio_pidr4h_bits);
__IO_REG32_BIT(GPIO_PIDR5L,       0xB0A08328,__READ       ,__gpio_pidr5l_bits);
__IO_REG32_BIT(GPIO_PIDR5H,       0xB0A0832C,__READ       ,__gpio_pidr5h_bits);
__IO_REG32_BIT(GPIO_PIDR6L,       0xB0A08330,__READ       ,__gpio_pidr6l_bits);
__IO_REG32_BIT(GPIO_PIDR6H,       0xB0A08334,__READ       ,__gpio_pidr6h_bits);
__IO_REG32_BIT(GPIO_PIDR7L,       0xB0A08338,__READ       ,__gpio_pidr7l_bits);
__IO_REG32_BIT(GPIO_PIDR7H,       0xB0A0833C,__READ       ,__gpio_pidr7h_bits);
__IO_REG32_BIT(GPIO_PPER0L,       0xB0A08380,__READ_WRITE ,__gpio_pper0l_bits);
__IO_REG32_BIT(GPIO_PPER0H,       0xB0A08384,__READ_WRITE ,__gpio_pper0h_bits);
__IO_REG32_BIT(GPIO_PPER1L,       0xB0A08388,__READ_WRITE ,__gpio_pper1l_bits);
__IO_REG32_BIT(GPIO_PPER1H,       0xB0A0838C,__READ_WRITE ,__gpio_pper1h_bits);
__IO_REG32_BIT(GPIO_PPER2L,       0xB0A08390,__READ_WRITE ,__gpio_pper2l_bits);
__IO_REG32_BIT(GPIO_PPER2H,       0xB0A08394,__READ_WRITE ,__gpio_pper2h_bits);
__IO_REG32_BIT(GPIO_PPER3L,       0xB0A08398,__READ_WRITE ,__gpio_pper3l_bits);
__IO_REG32_BIT(GPIO_PPER3H,       0xB0A0839C,__READ_WRITE ,__gpio_pper3h_bits);
__IO_REG32_BIT(GPIO_PPER4L,       0xB0A083A0,__READ_WRITE ,__gpio_pper4l_bits);
__IO_REG32_BIT(GPIO_PPER4H,       0xB0A083A4,__READ_WRITE ,__gpio_pper4h_bits);
__IO_REG32_BIT(GPIO_PPER5L,       0xB0A083A8,__READ_WRITE ,__gpio_pper5l_bits);
__IO_REG32_BIT(GPIO_PPER5H,       0xB0A083AC,__READ_WRITE ,__gpio_pper5h_bits);
__IO_REG32_BIT(GPIO_PPER6L,       0xB0A083B0,__READ_WRITE ,__gpio_pper6l_bits);
__IO_REG32_BIT(GPIO_PPER6H,       0xB0A083B4,__READ_WRITE ,__gpio_pper6h_bits);
__IO_REG32_BIT(GPIO_PPER7L,       0xB0A083B8,__READ_WRITE ,__gpio_pper7l_bits);
__IO_REG32_BIT(GPIO_PPER7H,       0xB0A083BC,__READ_WRITE ,__gpio_pper7h_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG16_BIT(ADC0_ER32,         0xB0700000,__READ_WRITE ,__adcn_er32_bits);
__IO_REG16_BIT(ADC0_ER10,         0xB0700002,__READ_WRITE ,__adcn_er10_bits);
__IO_REG8_BIT( ADC0_CS0,          0xB0700004,__READ_WRITE ,__adcn_cs0_bits);
__IO_REG8_BIT( ADC0_CS1,          0xB0700005,__READ_WRITE ,__adcn_cs1_bits);
__IO_REG8_BIT( ADC0_CS2,          0xB0700006,__READ_WRITE ,__adcn_cs2_bits);
__IO_REG8_BIT( ADC0_CS3,          0xB0700007,__READ_WRITE ,__adcn_cs3_bits);
__IO_REG8_BIT( ADC0_CSS1,         0xB0700009,__READ_WRITE ,__adcn_css1_bits);
__IO_REG8_BIT( ADC0_CSC1,         0xB070000B,__READ_WRITE ,__adcn_csc1_bits);
__IO_REG8_BIT( ADC0_CSS3,         0xB070000D,__READ_WRITE ,__adcn_css3_bits);
__IO_REG8_BIT( ADC0_CSC3,         0xB070000F,__READ_WRITE ,__adcn_csc3_bits);
__IO_REG16_BIT(ADC0_CR,           0xB0700010,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD0,          0xB0700018,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD1,          0xB070001A,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD2,          0xB070001C,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD3,          0xB070001E,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD4,          0xB0700020,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD5,          0xB0700022,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD6,          0xB0700024,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD7,          0xB0700026,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD8,          0xB0700028,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD9,          0xB070002A,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD10,         0xB070002C,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD11,         0xB070002E,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD12,         0xB0700030,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD13,         0xB0700032,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD14,         0xB0700034,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD15,         0xB0700036,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD16,         0xB0700038,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD17,         0xB070003A,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD18,         0xB070003C,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD19,         0xB070003E,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD20,         0xB0700040,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD21,         0xB0700042,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD22,         0xB0700044,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD23,         0xB0700046,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD24,         0xB0700048,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD25,         0xB070004A,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD26,         0xB070004C,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD27,         0xB070004E,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD28,         0xB0700050,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD29,         0xB0700052,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD30,         0xB0700054,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CD31,         0xB0700056,__READ       ,__adcn_cd_bits);
__IO_REG16_BIT(ADC0_CT,           0xB070005E,__READ_WRITE ,__adcn_ct_bits);
__IO_REG8_BIT( ADC0_SCH,          0xB0700060,__READ_WRITE ,__adcn_sch_bits);
__IO_REG8_BIT( ADC0_ECH,          0xB0700061,__READ_WRITE ,__adcn_ech_bits);
__IO_REG8_BIT( ADC0_MAR,          0xB0700062,__READ_WRITE ,__adcn_mar_bits);
__IO_REG8_BIT( ADC0_MACR,         0xB0700064,__READ_WRITE ,__adcn_macr_bits);
__IO_REG8_BIT( ADC0_MASR,         0xB0700066,__READ_WRITE ,__adcn_macr_bits);
__IO_REG8(     ADC0_RCOL0,        0xB0700068,__READ_WRITE );
__IO_REG8(     ADC0_RCOH0,        0xB0700069,__READ_WRITE );
__IO_REG8(     ADC0_RCOL1,        0xB070006A,__READ_WRITE );
__IO_REG8(     ADC0_RCOH1,        0xB070006B,__READ_WRITE );
__IO_REG8(     ADC0_RCOL2,        0xB070006C,__READ_WRITE );
__IO_REG8(     ADC0_RCOH2,        0xB070006D,__READ_WRITE );
__IO_REG8(     ADC0_RCOL3,        0xB070006E,__READ_WRITE );
__IO_REG8(     ADC0_RCOH3,        0xB070006F,__READ_WRITE );
__IO_REG8_BIT( ADC0_CC0,          0xB0700070,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC1,          0xB0700071,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC2,          0xB0700072,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC3,          0xB0700073,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC4,          0xB0700074,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC5,          0xB0700075,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC6,          0xB0700076,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC7,          0xB0700077,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC8,          0xB0700078,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC9,          0xB0700079,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC10,         0xB070007A,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC11,         0xB070007B,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC12,         0xB070007C,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC13,         0xB070007D,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC14,         0xB070007E,__READ_WRITE ,__adcn_cc_bits);
__IO_REG8_BIT( ADC0_CC15,         0xB070007F,__READ_WRITE ,__adcn_cc_bits);
__IO_REG16_BIT(ADC0_RCOIRS32,     0xB0700080,__READ_WRITE ,__adcn_rcoirs32_bits);
__IO_REG16_BIT(ADC0_RCOIRS10,     0xB0700082,__READ_WRITE ,__adcn_rcoirs10_bits);
__IO_REG16_BIT(ADC0_RCOOF32,      0xB0700084,__READ       ,__adcn_rcoof32_bits);
__IO_REG16_BIT(ADC0_RCOOF10,      0xB0700086,__READ       ,__adcn_rcoof10_bits);
__IO_REG16_BIT(ADC0_RCOINT32,     0xB0700088,__READ       ,__adcn_rcoint32_bits);
__IO_REG16_BIT(ADC0_RCOINT10,     0xB070008A,__READ       ,__adcn_rcoint10_bits);
__IO_REG16_BIT(ADC0_RCOINTC32,    0xB070008C,__READ_WRITE ,__adcn_rcointc32_bits);
__IO_REG16_BIT(ADC0_RCOINTC10,    0xB070008E,__READ_WRITE ,__adcn_rcointc10_bits);
__IO_REG8(     ADC0_PCTPRL0,      0xB0700090,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL0,      0xB0700091,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT0,      0xB0700092,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT0,      0xB0700093,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL1,      0xB0700094,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL1,      0xB0700095,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT1,      0xB0700096,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT1,      0xB0700097,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL2,      0xB0700098,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL2,      0xB0700099,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT2,      0xB070009A,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT2,      0xB070009B,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL3,      0xB070009C,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL3,      0xB070009D,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT3,      0xB070009E,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT3,      0xB070009F,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL4,      0xB07000A0,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL4,      0xB07000A1,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT4,      0xB07000A2,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT4,      0xB07000A3,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL5,      0xB07000A4,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL5,      0xB07000A5,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT5,      0xB07000A6,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT5,      0xB07000A7,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL6,      0xB07000A8,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL6,      0xB07000A9,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT6,      0xB07000AA,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT6,      0xB07000AB,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL7,      0xB07000AC,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL7,      0xB07000AD,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT7,      0xB07000AE,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT7,      0xB07000AF,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL8,      0xB07000B0,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL8,      0xB07000B1,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT8,      0xB07000B2,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT8,      0xB07000B3,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL9,      0xB07000B4,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL9,      0xB07000B5,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT9,      0xB07000B6,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT9,      0xB07000B7,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL10,     0xB07000B8,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL10,     0xB07000B9,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT10,     0xB07000BA,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT10,     0xB07000BB,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL11,     0xB07000BC,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL11,     0xB07000BD,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT11,     0xB07000BE,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT11,     0xB07000BF,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL12,     0xB07000C0,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL12,     0xB07000C1,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT12,     0xB07000C2,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT12,     0xB07000C3,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL13,     0xB07000C4,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL13,     0xB07000C5,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT13,     0xB07000C6,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT13,     0xB07000C7,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL14,     0xB07000C8,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL14,     0xB07000C9,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT14,     0xB07000CA,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT14,     0xB07000CB,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL15,     0xB07000CC,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL15,     0xB07000CD,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT15,     0xB07000CE,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT15,     0xB07000CF,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL16,     0xB07000D0,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL16,     0xB07000D1,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT16,     0xB07000D2,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT16,     0xB07000D3,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL17,     0xB07000D4,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL17,     0xB07000D5,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT17,     0xB07000D6,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT17,     0xB07000D7,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL18,     0xB07000D8,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL18,     0xB07000D9,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT18,     0xB07000DA,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT18,     0xB07000DB,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL19,     0xB07000DC,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL19,     0xB07000DD,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT19,     0xB07000DE,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT19,     0xB07000DF,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL20,     0xB07000E0,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL20,     0xB07000E1,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT20,     0xB07000E2,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT20,     0xB07000E3,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL21,     0xB07000E4,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL21,     0xB07000E5,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT21,     0xB07000E6,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT21,     0xB07000E7,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL22,     0xB07000E8,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL22,     0xB07000E9,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT22,     0xB07000EA,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT22,     0xB07000EB,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL23,     0xB07000EC,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL23,     0xB07000ED,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT23,     0xB07000EE,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT23,     0xB07000EF,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL24,     0xB07000F0,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL24,     0xB07000F1,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT24,     0xB07000F2,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT24,     0xB07000F3,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL25,     0xB07000F4,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL25,     0xB07000F5,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT25,     0xB07000F6,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT25,     0xB07000F7,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL26,     0xB07000F8,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL26,     0xB07000F9,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT26,     0xB07000FA,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT26,     0xB07000FB,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL27,     0xB07000FC,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL27,     0xB07000FD,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT27,     0xB07000FE,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT27,     0xB07000FF,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL28,     0xB0700100,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL28,     0xB0700101,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT28,     0xB0700102,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT28,     0xB0700103,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL29,     0xB0700104,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL29,     0xB0700105,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT29,     0xB0700106,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT29,     0xB0700107,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL30,     0xB0700108,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL30,     0xB0700109,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT30,     0xB070010A,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT30,     0xB070010B,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG8(     ADC0_PCTPRL31,     0xB070010C,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNRL31,     0xB070010D,__READ_WRITE ,__adcn_pctnrl_bits);
__IO_REG8(     ADC0_PCTPCT31,     0xB070010E,__READ_WRITE );
__IO_REG8_BIT( ADC0_PCTNCT31,     0xB070010F,__READ_WRITE ,__adcn_pctnct_bits);
__IO_REG16_BIT(ADC0_PCZF10,       0xB0700110,__READ       ,__adcn_pczf10_bits);
__IO_REG16_BIT(ADC0_PCZF32,       0xB0700112,__READ       ,__adcn_pczf32_bits);
__IO_REG16_BIT(ADC0_PCZFC10,      0xB0700114,__READ_WRITE ,__adcn_pczfc10_bits);
__IO_REG16_BIT(ADC0_PCZFC32,      0xB0700116,__READ_WRITE ,__adcn_pczfc32_bits);
__IO_REG16_BIT(ADC0_PCIE10,       0xB0700118,__READ_WRITE ,__adcn_pcie10_bits);
__IO_REG16_BIT(ADC0_PCIE32,       0xB070011A,__READ_WRITE ,__adcn_pcie32_bits);
__IO_REG16_BIT(ADC0_PCIES10,      0xB070011C,__READ_WRITE ,__adcn_pcies10_bits);
__IO_REG16_BIT(ADC0_PCIES32,      0xB070011E,__READ_WRITE ,__adcn_pcies32_bits);
__IO_REG16_BIT(ADC0_PCIEC10,      0xB0700120,__READ_WRITE ,__adcn_pciec10_bits);
__IO_REG16_BIT(ADC0_PCIEC32,      0xB0700122,__READ_WRITE ,__adcn_pciec32_bits);

/***************************************************************************
 **
 ** BSU0
 **
 ***************************************************************************/
__IO_REG16(    BSU0_BTSTL,        0xB07FFC04,__READ_WRITE );
__IO_REG16(    BSU0_BTSTH,        0xB07FFC06,__READ_WRITE );
__IO_REG16_BIT(BSU0_PEN0L,        0xB07FFC10,__READ_WRITE ,__bsu0_pen0l_bits);
__IO_REG16_BIT(BSU0_PEN1L,        0xB07FFC14,__READ_WRITE ,__bsu0_pen1l_bits);
__IO_REG16_BIT(BSU0_PEN2L,        0xB07FFC18,__READ_WRITE ,__bsu0_pen2l_bits);
__IO_REG16_BIT(BSU0_PEN3L,        0xB07FFC1C,__READ_WRITE ,__bsu0_pen3l_bits);
__IO_REG16_BIT(BSU0_PEN4L,        0xB07FFC20,__READ_WRITE ,__bsu0_pen4l_bits);
__IO_REG16_BIT(BSU0_PEN5L,        0xB07FFC24,__READ_WRITE ,__bsu0_pen5l_bits);
__IO_REG16_BIT(BSU0_PEN6L,        0xB07FFC28,__READ_WRITE ,__bsu0_pen6l_bits);
__IO_REG16_BIT(BSU0_PEN7L,        0xB07FFC2C,__READ_WRITE ,__bsu0_pen7l_bits);
__IO_REG16_BIT(BSU0_PEN7H,        0xB07FFC2E,__READ_WRITE ,__bsu0_pen7h_bits);
__IO_REG16_BIT(BSU0_PEN8L,        0xB07FFC30,__READ_WRITE ,__bsu0_pen8l_bits);
__IO_REG16_BIT(BSU0_PEN8H,        0xB07FFC32,__READ_WRITE ,__bsu0_pen8h_bits);
__IO_REG16_BIT(BSU0_PEN9L,        0xB07FFC34,__READ_WRITE ,__bsu0_pen9l_bits);
__IO_REG16_BIT(BSU0_PEN9H,        0xB07FFC36,__READ_WRITE ,__bsu0_pen9h_bits);
__IO_REG16_BIT(BSU0_PEN11L,       0xB07FFC3C,__READ_WRITE ,__bsu0_pen11l_bits);

/***************************************************************************
 **
 ** BSU1
 **
 ***************************************************************************/
__IO_REG16(    BSU1_BTSTL,        0xB08FFC04,__READ_WRITE );
__IO_REG16(    BSU1_BTSTH,        0xB08FFC06,__READ_WRITE );
__IO_REG16_BIT(BSU1_PEN0L,        0xB08FFC10,__READ_WRITE ,__bsu1_pen0l_bits);
__IO_REG16_BIT(BSU1_PEN1L,        0xB08FFC14,__READ_WRITE ,__bsu1_pen1l_bits);
__IO_REG16_BIT(BSU1_PEN2L,        0xB08FFC18,__READ_WRITE ,__bsu1_pen2l_bits);
__IO_REG16_BIT(BSU1_PEN3L,        0xB08FFC1C,__READ_WRITE ,__bsu1_pen3l_bits);
__IO_REG16_BIT(BSU1_PEN4L,        0xB08FFC20,__READ_WRITE ,__bsu1_pen4l_bits);
__IO_REG16_BIT(BSU1_PEN5L,        0xB08FFC24,__READ_WRITE ,__bsu1_pen5l_bits);
__IO_REG16_BIT(BSU1_PEN6L,        0xB08FFC28,__READ_WRITE ,__bsu1_pen6l_bits);
__IO_REG16_BIT(BSU1_PEN7L,        0xB08FFC2C,__READ_WRITE ,__bsu1_pen7l_bits);
__IO_REG16_BIT(BSU1_PEN8L,        0xB08FFC30,__READ_WRITE ,__bsu1_pen8l_bits);
__IO_REG16_BIT(BSU1_PEN9L,        0xB08FFC34,__READ_WRITE ,__bsu1_pen9l_bits);
__IO_REG16_BIT(BSU1_PEN9H,        0xB08FFC36,__READ_WRITE ,__bsu1_pen9h_bits);
__IO_REG16_BIT(BSU1_PEN10L,       0xB08FFC38,__READ_WRITE ,__bsu1_pen10l_bits);
__IO_REG16_BIT(BSU1_PEN10H,       0xB08FFC3A,__READ_WRITE ,__bsu1_pen10h_bits);
__IO_REG16_BIT(BSU1_PEN11L,       0xB08FFC3C,__READ_WRITE ,__bsu1_pen11l_bits);
__IO_REG16_BIT(BSU1_PEN11H,       0xB08FFC3E,__READ_WRITE ,__bsu1_pen11h_bits);

/***************************************************************************
 **
 ** BSU3
 **
 ***************************************************************************/
__IO_REG32(    BSU3_BTST,         0xB0AFFC04,__READ_WRITE );
__IO_REG32_BIT(BSU3_PEN2,         0xB0AFFC18,__READ_WRITE ,__bsu3_pen2_bits);
__IO_REG32_BIT(BSU3_PEN4,         0xB0AFFC20,__READ_WRITE ,__bsu3_pen4_bits);

/***************************************************************************
 **
 ** BSU4
 **
 ***************************************************************************/
__IO_REG32(    BSU4_BTST,         0xB0BFFC04,__READ_WRITE );
__IO_REG32_BIT(BSU4_PEN1,         0xB0BFFC14,__READ_WRITE ,__bsu4_pen1_bits);
__IO_REG32_BIT(BSU4_PEN2,         0xB0BFFC18,__READ_WRITE ,__bsu4_pen2_bits);
__IO_REG32_BIT(BSU4_PEN3,         0xB0BFFC1C,__READ_WRITE ,__bsu4_pen3_bits);
__IO_REG32_BIT(BSU4_PEN4,         0xB0BFFC20,__READ_WRITE ,__bsu4_pen4_bits);
__IO_REG32_BIT(BSU4_PEN5,         0xB0BFFC24,__READ_WRITE ,__bsu4_pen5_bits);
__IO_REG32_BIT(BSU4_PEN6,         0xB0BFFC28,__READ_WRITE ,__bsu4_pen6_bits);
__IO_REG32_BIT(BSU4_PEN7,         0xB0BFFC2C,__READ_WRITE ,__bsu4_pen7_bits);
__IO_REG32_BIT(BSU4_PEN8,         0xB0BFFC30,__READ_WRITE ,__bsu4_pen8_bits);

/***************************************************************************
 **
 ** BSU5
 **
 ***************************************************************************/
__IO_REG32(    BSU5_BTST,         0xB0CFFC04,__READ_WRITE );

/***************************************************************************
 **
 ** BSU6
 **
 ***************************************************************************/
__IO_REG32(    BSU6_READ0,        0xB0418000,__READ_WRITE );
__IO_REG32(    BSU6_BTST,         0xB0418004,__READ_WRITE );
__IO_REG32_BIT(BSU6_PEN2,         0xB0418318,__READ_WRITE ,__bsu6_pen2_bits);
__IO_REG32(    BSU6_READ1,        0xB041801C,__READ_WRITE );
__IO_REG32_BIT(BSU6_PEN4,         0xB0418320,__READ_WRITE ,__bsu6_pen4_bits);
__IO_REG32(    BSU6_READ2,        0xB0418024,__READ_WRITE );
__IO_REG32_BIT(BSU6_PEN12,        0xB0418040,__READ_WRITE ,__bsu6_pen12_bits);
__IO_REG32(    BSU6_READ3,        0xB0418044,__READ_WRITE );
__IO_REG32_BIT(BSU6_PEN20,        0xB0418060,__READ_WRITE ,__bsu6_pen20_bits);
__IO_REG32(    BSU6_READ4,        0xB0418064,__READ_WRITE );

/***************************************************************************
 **
 ** BSU7
 **
 ***************************************************************************/
__IO_REG32(    BSU7_BTST,         0xB06FFC04,__READ_WRITE );
__IO_REG32_BIT(BSU7_PEN3,         0xB06FFC1C,__READ_WRITE ,__bsu7_pen3_bits);
__IO_REG32_BIT(BSU7_PEN5,         0xB06FFC24,__READ_WRITE ,__bsu7_pen5_bits);
__IO_REG32_BIT(BSU7_PEN7,         0xB06FFC2C,__READ_WRITE ,__bsu7_pen7_bits);
__IO_REG32_BIT(BSU7_PEN8,         0xB06FFC30,__READ_WRITE ,__bsu7_pen8_bits);

/***************************************************************************
 **
 ** BSU8
 **
 ***************************************************************************/
__IO_REG32(    BSU8_BTST,         0xB007FC04,__READ_WRITE );
__IO_REG32_BIT(BSU8_PEN0,         0xB007FC10,__READ_WRITE ,__bsu8_pen0_bits);

/***************************************************************************
 **
 ** ETH0
 **
 ***************************************************************************/
__IO_REG8_BIT( ETH0_OEN,          0xB0B08000,__READ_WRITE ,__ethn_oen_bits);
__IO_REG8_BIT( ETH0_WOL,          0xB0B08001,__READ_WRITE ,__ethn_wol_bits);
__IO_REG32_BIT(ETH0_EMODE,        0xB0B08004,__READ_WRITE ,__ethn_emode_bits);
__IO_REG16_BIT(ETH0_PMODE,        0xB0B08008,__READ_WRITE ,__ethn_pmode_bits);
__IO_REG16_BIT(ETH0_FMODE,        0xB0B0800C,__READ_WRITE ,__ethn_fmode_bits);
__IO_REG32_BIT(ETH0_EIE,          0xB0B08010,__READ_WRITE ,__ethn_eie_bits);
__IO_REG32_BIT(ETH0_PIE,          0xB0B08014,__READ_WRITE ,__ethn_pie_bits);
__IO_REG8_BIT( ETH0_FMIE,         0xB0B08018,__READ_WRITE ,__ethn_fmie_bits);
__IO_REG8_BIT( ETH0_ARIE,         0xB0B08019,__READ_WRITE ,__ethn_fmie_bits);
__IO_REG32_BIT(ETH0_EIR,          0xB0B0801C,__READ_WRITE ,__ethn_eir_bits);
__IO_REG32_BIT(ETH0_PIR,          0xB0B08020,__READ       ,__ethn_pir_bits);
__IO_REG8_BIT( ETH0_FMIR,         0xB0B08024,__READ       ,__ethn_fmir_bits);
__IO_REG8_BIT( ETH0_FMIO,         0xB0B08025,__READ       ,__ethn_fmio_bits);
__IO_REG8_BIT( ETH0_ARIR,         0xB0B08026,__READ       ,__ethn_fmir_bits);
__IO_REG8_BIT( ETH0_ARIO,         0xB0B08027,__READ       ,__ethn_fmio_bits);
__IO_REG32_BIT(ETH0_EIC,          0xB0B08028,__READ_WRITE ,__ethn_eic_bits);
__IO_REG32_BIT(ETH0_PIC,          0xB0B0802C,__READ_WRITE ,__ethn_pic_bits);
__IO_REG8_BIT( ETH0_FMIC,         0xB0B08030,__READ_WRITE ,__ethn_fmic_bits);
__IO_REG8_BIT( ETH0_FMOC,         0xB0B08031,__READ_WRITE ,__ethn_fmoc_bits);
__IO_REG8_BIT( ETH0_ARIC,         0xB0B08032,__READ_WRITE ,__ethn_fmic_bits);
__IO_REG8_BIT( ETH0_AROC,         0xB0B08033,__READ_WRITE ,__ethn_fmoc_bits);
__IO_REG8_BIT( ETH0_HIE,          0xB0B08034,__READ_WRITE ,__ethn_hie_bits);
__IO_REG8_BIT( ETH0_HIR,          0xB0B08035,__READ_WRITE ,__ethn_hir_bits);
__IO_REG8_BIT( ETH0_HIC,          0xB0B08036,__READ_WRITE ,__ethn_hic_bits);
__IO_REG16(    ETH0_TXPTIM,       0xB0B08038,__READ_WRITE );
__IO_REG16(    ETH0_RXPTIM,       0xB0B0803A,__READ       );
__IO_REG32(    ETH0_MACAD0,       0xB0B0803C,__READ_WRITE );
__IO_REG16(    ETH0_MACAD1,       0xB0B08040,__READ_WRITE );
__IO_REG8_BIT( ETH0_IFS,          0xB0B08042,__READ_WRITE ,__ethn_ifs_bits);
__IO_REG16_BIT(ETH0_MDCCKDIV,     0xB0B08044,__READ_WRITE ,__ethn_mdcckdiv_bits);
__IO_REG16(    ETH0_MDLPTIM,      0xB0B08046,__READ_WRITE );
__IO_REG16_BIT(ETH0_MDCTRL,       0xB0B08048,__READ_WRITE ,__ethn_mdctrl_bits);
__IO_REG16(    ETH0_MDDAT,        0xB0B0804C,__READ_WRITE );
__IO_REG32(    ETH0_TXBDTBA,      0xB0B08050,__READ_WRITE );
__IO_REG32(    ETH0_RXBDTBA,      0xB0B08054,__READ_WRITE );
__IO_REG32(    ETH0_PTPTXBDTBA,   0xB0B08058,__READ_WRITE );
__IO_REG32(    ETH0_PTPRXBDTBA,   0xB0B0805C,__READ_WRITE );
__IO_REG32(    ETH0_ARBDTBA,      0xB0B08060,__READ_WRITE );
__IO_REG32(    ETH0_HPBDTBA,      0xB0B08064,__READ_WRITE );
__IO_REG32(    ETH0_TXBDTPTR,     0xB0B08068,__READ       );
__IO_REG32(    ETH0_RXBDTPTR,     0xB0B0806C,__READ       );
__IO_REG16(    ETH0_BDTPOLINT,    0xB0B08070,__READ_WRITE );
__IO_REG32(    ETH0_MACID,        0xB0B08074,__READ       );
__IO_REG32(    ETH0_TXOKCNT,      0xB0B08078,__READ       );
__IO_REG32(    ETH0_SCOLCNT,      0xB0B0807C,__READ       );
__IO_REG32(    ETH0_MCOLCNT,      0xB0B08080,__READ       );
__IO_REG32(    ETH0_RXOKCNT,      0xB0B08084,__READ       );
__IO_REG32(    ETH0_FCSERCNT,     0xB0B08088,__READ       );
__IO_REG32(    ETH0_ALNERCNT,     0xB0B0808C,__READ       );
__IO_REG32(    ETH0_FFSPTR0,      0xB0B08090,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR1,      0xB0B08094,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR2,      0xB0B08098,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR3,      0xB0B0809C,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR4,      0xB0B080A0,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR5,      0xB0B080A4,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR6,      0xB0B080A8,__READ_WRITE );
__IO_REG32(    ETH0_FFSPTR7,      0xB0B080AC,__READ_WRITE );
__IO_REG16_BIT(ETH0_FFSLEN0,      0xB0B080B0,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN1,      0xB0B080B2,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN2,      0xB0B080B4,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN3,      0xB0B080B6,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN4,      0xB0B080B8,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN5,      0xB0B080BA,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN6,      0xB0B080BC,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG16_BIT(ETH0_FFSLEN7,      0xB0B080BE,__READ_WRITE ,__ethn_ffslen_bits);
__IO_REG8_BIT( ETH0_FCCR0,        0xB0B080C0,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR1,        0xB0B080C1,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR2,        0xB0B080C2,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR3,        0xB0B080C3,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR4,        0xB0B080C4,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR5,        0xB0B080C5,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR6,        0xB0B080C6,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG8_BIT( ETH0_FCCR7,        0xB0B080C7,__READ_WRITE ,__ethn_fccr_bits);
__IO_REG32(    ETH0_AHBERAD,      0xB0B080C8,__READ       );
__IO_REG32(    ETH0_AHBERDAT,     0xB0B080CC,__READ       );
__IO_REG8_BIT( ETH0_AHBERCTR,     0xB0B080D0,__READ_WRITE ,__ethn_ahberctr_bits);
__IO_REG32_BIT(ETH0_NMI,          0xB0B080D4,__READ_WRITE ,__ethn_nmi_bits);

/***************************************************************************
 **
 ** ETHERNETRAM0
 **
 ***************************************************************************/
__IO_REG32(    ERCFGn_UNLOCKR,    0xB0B0C000,__READ_WRITE );
__IO_REG32_BIT(ERCFGn_CSR,        0xB0B0C004,__READ_WRITE ,__ercfgn_csr_bits);
__IO_REG32(    ERCFGn_EAN,        0xB0B0C008,__READ       );
__IO_REG32(    ERCFGn_ERRMSKR0,   0xB0B0C00C,__READ_WRITE );
__IO_REG32_BIT(ERCFGn_ERRMSKR1,   0xB0B0C010,__READ_WRITE ,__ercfgn_errmskr1_bits);
__IO_REG32_BIT(ERCFGn_ECCEN,      0xB0B0C014,__READ_WRITE ,__ercfgn_eccen_bits);

/***************************************************************************
 **
 ** UDC0
 **
 ***************************************************************************/
__IO_REG16_BIT(UDC0_CC0,          0xB0A20000,__READ_WRITE ,__udcn_cc0_bits);
__IO_REG16_BIT(UDC0_ECC0,         0xB0A20002,__READ_WRITE ,__udcn_ecc_bits);
__IO_REG16_BIT(UDC0_CC1,          0xB0A20004,__READ_WRITE ,__udcn_cc1_bits);
__IO_REG16_BIT(UDC0_ECC1,         0xB0A20006,__READ_WRITE ,__udcn_ecc_bits);
__IO_REG16_BIT(UDC0_CS0,          0xB0A20008,__READ_WRITE ,__udcn_cs_bits);
__IO_REG16_BIT(UDC0_CS1,          0xB0A2000A,__READ_WRITE ,__udcn_cs_bits);
__IO_REG16_BIT(UDC0_TGL0,         0xB0A2000C,__READ_WRITE ,__udcn_tgl_bits);
__IO_REG16_BIT(UDC0_TGL1,         0xB0A2000E,__READ_WRITE ,__udcn_tgl_bits);
__IO_REG32_BIT(UDC0_CR,           0xB0A20010,__READ       ,__udcn_cr_bits);
__IO_REG32_BIT(UDC0_RC,           0xB0A20014,__READ_WRITE ,__udcn_rc_bits);
__IO_REG8_BIT( UDC0_DBG,          0xB0A20018,__READ_WRITE ,__udcn_dbg_bits);

/***************************************************************************
 **
 ** RICFG8G0
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG8G0_HSSPI0MSTART,	0xB0078000,__READ_WRITE ,__ricfg8g0_hsspi0mstart_bits);

/***************************************************************************
 **
 ** GFXGCTR
 **
 ***************************************************************************/
__IO_REG32(		 GFXGCTR_LOCKUNLOCK,		0xB0100000,__READ_WRITE );
__IO_REG32_BIT(GFXGCTR_LOCKSTATUS,		0xB0100004,__READ_WRITE ,__gfxgctr_lockstatus_bits);
__IO_REG32_BIT(GFXGCTR_INTSTATUS0,		0xB0100008,__READ_WRITE ,__gfxgctr_intstatus0_bits);
__IO_REG32_BIT(GFXGCTR_INTSTATUS1,		0xB010000C,__READ_WRITE ,__gfxgctr_intstatus1_bits);
__IO_REG32_BIT(GFXGCTR_INTENABLE0,		0xB0100010,__READ_WRITE ,__gfxgctr_intenable0_bits);
__IO_REG32_BIT(GFXGCTR_INTENABLE1,		0xB0100014,__READ_WRITE ,__gfxgctr_intenable1_bits);
__IO_REG32_BIT(GFXGCTR_INTCLEAR0,			0xB0100018,__READ_WRITE ,__gfxgctr_intclear0_bits);
__IO_REG32_BIT(GFXGCTR_INTCLEAR1,			0xB010001C,__READ_WRITE ,__gfxgctr_intclear1_bits);
__IO_REG32_BIT(GFXGCTR_INTPRESET0,		0xB0100020,__READ_WRITE ,__gfxgctr_intpreset0_bits);
__IO_REG32_BIT(GFXGCTR_INTPRESET1,		0xB0100024,__READ_WRITE ,__gfxgctr_intpreset1_bits);
__IO_REG32_BIT(GFXGCTR_INTMAP0,				0xB0100028,__READ_WRITE ,__gfxgctr_intmap0_bits);
__IO_REG32_BIT(GFXGCTR_INTMAP1,				0xB010002C,__READ_WRITE ,__gfxgctr_intmap1_bits);
__IO_REG32_BIT(GFXGCTR_NMISTATUS,			0xB0100030,__READ_WRITE ,__gfxgctr_nmistatus_bits);
__IO_REG32_BIT(GFXGCTR_NMICLEAR,			0xB0100034,__READ_WRITE ,__gfxgctr_nmiclear_bits);
__IO_REG32_BIT(GFXGCTR_NMIPRESET,			0xB0100038,__READ_WRITE ,__gfxgctr_nmipreset_bits);
__IO_REG32_BIT(GFXGCTR_CSINTSTATUS0,	0xB010003C,__READ_WRITE ,__gfxgctr_csintstatus0_bits);
__IO_REG32_BIT(GFXGCTR_CSINTSTATUS1,	0xB0100040,__READ_WRITE ,__gfxgctr_csintstatus1_bits);
__IO_REG32_BIT(GFXGCTR_CSINTENABLE0,	0xB0100044,__READ_WRITE ,__gfxgctr_csintenable0_bits);
__IO_REG32_BIT(GFXGCTR_CSINTENABLE1,	0xB0100048,__READ_WRITE ,__gfxgctr_csintenable1_bits);
__IO_REG32_BIT(GFXGCTR_CSINTCLEAR0,		0xB010004C,__READ_WRITE ,__gfxgctr_csintclear0_bits);
__IO_REG32_BIT(GFXGCTR_CSINTCLEAR1,		0xB0100050,__READ_WRITE ,__gfxgctr_csintclear1_bits);
__IO_REG32_BIT(GFXGCTR_CSINTPRESET0,	0xB0100054,__READ_WRITE ,__gfxgctr_csintpreset0_bits);
__IO_REG32_BIT(GFXGCTR_CSINTPRESET1,	0xB0100058,__READ_WRITE ,__gfxgctr_csintpreset1_bits);
__IO_REG32_BIT(GFXGCTR_SWRESET,				0xB010005C,__READ_WRITE ,__gfxgctr_swreset_bits);
__IO_REG32_BIT(GFXGCTR_CLOCKADJUST,		0xB0100060,__READ_WRITE ,__gfxgctr_clockadjust_bits);

/***************************************************************************
 **
 ** GFXSPI
 **
 ***************************************************************************/
__IO_REG32_BIT(GFXSPI_MCTRL,      0xB0101000,__READ_WRITE ,__hsspin_mctrl_bits);
__IO_REG32_BIT(GFXSPI_PCC0,       0xB0101004,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(GFXSPI_PCC1,       0xB0101008,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(GFXSPI_PCC2,       0xB010100C,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(GFXSPI_PCC3,       0xB0101010,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(GFXSPI_TXF,        0xB0101014,__READ_WRITE ,__hsspin_txf_bits);
__IO_REG32_BIT(GFXSPI_TXE,        0xB0101018,__READ_WRITE ,__hsspin_txe_bits);
__IO_REG32_BIT(GFXSPI_TXC,        0xB010101C,__READ_WRITE ,__hsspin_txc_bits);
__IO_REG32_BIT(GFXSPI_RXF,        0xB0101020,__READ       ,__hsspin_rxf_bits);
__IO_REG32_BIT(GFXSPI_RXE,        0xB0101024,__READ_WRITE ,__hsspin_rxe_bits);
__IO_REG32_BIT(GFXSPI_RXC,        0xB0101028,__READ_WRITE ,__hsspin_rxc_bits);
__IO_REG32_BIT(GFXSPI_FAULTF,     0xB010102C,__READ       ,__hsspin_faultf_bits);
__IO_REG32_BIT(GFXSPI_FAULTC,     0xB0101030,__READ_WRITE ,__hsspin_faultc_bits);
__IO_REG8_BIT( GFXSPI_DMCFG,      0xB0101034,__READ_WRITE ,__hsspin_dmcfg_bits);
__IO_REG8_BIT( GFXSPI_DMDMAEN,    0xB0101035,__READ_WRITE ,__hsspin_dmdmaen_bits);
__IO_REG8_BIT( GFXSPI_DMSTART,    0xB0101038,__READ_WRITE ,__hsspin_dmstart_bits);
__IO_REG8_BIT( GFXSPI_DMSTOP,     0xB0101039,__READ_WRITE ,__hsspin_dmstop_bits);
__IO_REG8_BIT( GFXSPI_DMPSEL,     0xB010103A,__READ_WRITE ,__hsspin_dmpsel_bits);
__IO_REG8_BIT( GFXSPI_DMTRP,      0xB010103B,__READ_WRITE ,__hsspin_dmtrp_bits);
__IO_REG16(    GFXSPI_DMBCC,      0xB010103C,__READ_WRITE );
__IO_REG16(    GFXSPI_DMBCS,      0xB010103E,__READ       );
__IO_REG32_BIT(GFXSPI_DMSTATUS,   0xB0101040,__READ       ,__hsspin_dmstatus_bits);
__IO_REG8_BIT( GFXSPI_TXBITCNT,   0xB0101044,__READ       ,__hsspin_txbitcnt_bits);
__IO_REG8_BIT( GFXSPI_RXBITCNT,   0xB0101045,__READ       ,__hsspin_rxbitcnt_bits);
__IO_REG32(    GFXSPI_RXSHIFT,    0xB0101048,__READ       );
__IO_REG32_BIT(GFXSPI_FIFOCFG,    0xB010104C,__READ_WRITE ,__hsspin_fifocfg_bits);
__IO_REG32(    GFXSPI_TXFIFO0,    0xB0101050,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO1,    0xB0101054,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO2,    0xB0101058,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO3,    0xB010105C,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO4,    0xB0101060,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO5,    0xB0101064,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO6,    0xB0101068,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO7,    0xB010106C,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO8,    0xB0101070,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO9,    0xB0101074,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO10,   0xB0101078,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO11,   0xB010107C,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO12,   0xB0101080,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO13,   0xB0101084,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO14,   0xB0101088,__READ_WRITE );
__IO_REG32(    GFXSPI_TXFIFO15,   0xB010108C,__READ_WRITE );
__IO_REG32(    GFXSPI_RXFIFO0,    0xB0101090,__READ       );
__IO_REG32(    GFXSPI_RXFIFO1,    0xB0101094,__READ       );
__IO_REG32(    GFXSPI_RXFIFO2,    0xB0101098,__READ       );
__IO_REG32(    GFXSPI_RXFIFO3,    0xB010109C,__READ       );
__IO_REG32(    GFXSPI_RXFIFO4,    0xB01010A0,__READ       );
__IO_REG32(    GFXSPI_RXFIFO5,    0xB01010A4,__READ       );
__IO_REG32(    GFXSPI_RXFIFO6,    0xB01010A8,__READ       );
__IO_REG32(    GFXSPI_RXFIFO7,    0xB01010AC,__READ       );
__IO_REG32(    GFXSPI_RXFIFO8,    0xB01010B0,__READ       );
__IO_REG32(    GFXSPI_RXFIFO9,    0xB01010B4,__READ       );
__IO_REG32(    GFXSPI_RXFIFO10,   0xB01010B8,__READ       );
__IO_REG32(    GFXSPI_RXFIFO11,   0xB01010BC,__READ       );
__IO_REG32(    GFXSPI_RXFIFO12,   0xB01010C0,__READ       );
__IO_REG32(    GFXSPI_RXFIFO13,   0xB01010C4,__READ       );
__IO_REG32(    GFXSPI_RXFIFO14,   0xB01010C8,__READ       );
__IO_REG32(    GFXSPI_RXFIFO15,   0xB01010CC,__READ       );
__IO_REG32_BIT(GFXSPI_CSCFG,      0xB01010D0,__READ_WRITE ,__hsspin_cscfg_bits);
__IO_REG32_BIT(GFXSPI_CSITIME,    0xB01010D4,__READ_WRITE ,__hsspin_csitime_bits);
__IO_REG32_BIT(GFXSPI_CSAEXT,     0xB01010D8,__READ_WRITE ,__hsspin_csaext_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC0,    0xB01010DC,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC1,    0xB01010DE,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC2,    0xB01010E0,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC3,    0xB01010E2,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC4,    0xB01010E4,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC5,    0xB01010E6,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC6,    0xB01010E8,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_RDCSDC7,    0xB01010EA,__READ_WRITE ,__hsspin_rdcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC0,    0xB01010EC,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC1,    0xB01010EE,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC2,    0xB01010F0,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC3,    0xB01010F2,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC4,    0xB01010F4,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC5,    0xB01010F6,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC6,    0xB01010F8,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG16_BIT(GFXSPI_WRCSDC7,    0xB01010FA,__READ_WRITE ,__hsspin_wrcsdc_bits);
__IO_REG32(    GFXSPI_MID,        0xB01010FC,__READ       );

/***************************************************************************
 **
 ** GFXCMD
 **
 ***************************************************************************/
__IO_REG32(		 GFXCMD_HIF,      				0xB0102000,__READ_WRITE );
__IO_REG32_BIT(GFXCMD_STATUS,     			0xB0102100,__READ_WRITE ,__gfxcmd_status_bits);
__IO_REG32_BIT(GFXCMD_CONTROL,    			0xB0102104,__READ_WRITE ,__gfxcmd_control_bits);
__IO_REG32_BIT(GFXCMD_BUFFERADDRESS,		0xB0102108,__READ_WRITE ,__gfxcmd_bufferaddress_bits);
__IO_REG32_BIT(GFXCMD_BUFFERSIZE,				0xB010210C,__READ_WRITE ,__gfxcmd_buffersize_bits);
__IO_REG32_BIT(GFXCMD_WATERMARKCONTROL,	0xB0102110,__READ_WRITE ,__gfxcmd_watermarkcontrol_bits);

/***************************************************************************
 **
 ** GFXTCON
 **
 ***************************************************************************/
__IO_REG32_BIT(GFXTCON_DIR_SSQCNTS,     		0xB0103000,__READ_WRITE ,__gfxtcon_dir_ssqcnts_bits);
__IO_REG32_BIT(GFXTCON_DIR_SWRESET,     		0xB0103400,__READ_WRITE ,__gfxtcon_dir_swreset_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG0POSON,   		0xB0103404,__READ_WRITE ,__gfxtcon_dir_spg0poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG0MASKON,  		0xB0103408,__READ_WRITE ,__gfxtcon_dir_spg0maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG0POSOFF,  		0xB010340C,__READ_WRITE ,__gfxtcon_dir_spg0posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG0MASKOFF, 		0xB0103410,__READ_WRITE ,__gfxtcon_dir_spg0maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG1POSON,   		0xB0103414,__READ_WRITE ,__gfxtcon_dir_spg1poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG1MASKON,  		0xB0103418,__READ_WRITE ,__gfxtcon_dir_spg1maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG1POSOFF,  		0xB010341C,__READ_WRITE ,__gfxtcon_dir_spg1posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG1MASKOFF, 		0xB0103420,__READ_WRITE ,__gfxtcon_dir_spg1maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG2POSON,   		0xB0103424,__READ_WRITE ,__gfxtcon_dir_spg2poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG2MASKON,  		0xB0103428,__READ_WRITE ,__gfxtcon_dir_spg2maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG2POSOFF,  		0xB010342C,__READ_WRITE ,__gfxtcon_dir_spg2posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG2MASKOFF, 		0xB0103430,__READ_WRITE ,__gfxtcon_dir_spg2maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG3POSON,   		0xB0103434,__READ_WRITE ,__gfxtcon_dir_spg3poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG3MASKON,  		0xB0103438,__READ_WRITE ,__gfxtcon_dir_spg3maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG3POSOFF,  		0xB010343C,__READ_WRITE ,__gfxtcon_dir_spg3posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG3MASKOFF, 		0xB0103440,__READ_WRITE ,__gfxtcon_dir_spg3maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG4POSON,   		0xB0103444,__READ_WRITE ,__gfxtcon_dir_spg4poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG4MASKON,  		0xB0103448,__READ_WRITE ,__gfxtcon_dir_spg4maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG4POSOFF,  		0xB010344C,__READ_WRITE ,__gfxtcon_dir_spg4posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG4MASKOFF, 		0xB0103450,__READ_WRITE ,__gfxtcon_dir_spg4maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG5POSON,   		0xB0103454,__READ_WRITE ,__gfxtcon_dir_spg5poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG5MASKON,  		0xB0103458,__READ_WRITE ,__gfxtcon_dir_spg5maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG5POSOFF,  		0xB010345C,__READ_WRITE ,__gfxtcon_dir_spg5posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG5MASKOFF, 		0xB0103460,__READ_WRITE ,__gfxtcon_dir_spg5maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG6POSON,   		0xB0103464,__READ_WRITE ,__gfxtcon_dir_spg6poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG6MASKON,  		0xB0103468,__READ_WRITE ,__gfxtcon_dir_spg6maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG6POSOFF,  		0xB010346C,__READ_WRITE ,__gfxtcon_dir_spg6posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG6MASKOFF, 		0xB0103470,__READ_WRITE ,__gfxtcon_dir_spg6maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG7POSON,   		0xB0103474,__READ_WRITE ,__gfxtcon_dir_spg7poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG7MASKON,  		0xB0103478,__READ_WRITE ,__gfxtcon_dir_spg7maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG7POSOFF,  		0xB010347C,__READ_WRITE ,__gfxtcon_dir_spg7posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG7MASKOFF, 		0xB0103480,__READ_WRITE ,__gfxtcon_dir_spg7maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG8POSON,   		0xB0103484,__READ_WRITE ,__gfxtcon_dir_spg8poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG8MASKON,  		0xB0103488,__READ_WRITE ,__gfxtcon_dir_spg8maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG8POSOFF,  		0xB010348C,__READ_WRITE ,__gfxtcon_dir_spg8posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG8MASKOFF, 		0xB0103490,__READ_WRITE ,__gfxtcon_dir_spg8maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG9POSON,   		0xB0103494,__READ_WRITE ,__gfxtcon_dir_spg9poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG9MASKON,  		0xB0103498,__READ_WRITE ,__gfxtcon_dir_spg9maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG9POSOFF,  		0xB010349C,__READ_WRITE ,__gfxtcon_dir_spg9posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG9MASKOFF, 		0xB01034A0,__READ_WRITE ,__gfxtcon_dir_spg9maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG10POSON,  		0xB01034A4,__READ_WRITE ,__gfxtcon_dir_spg10poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG10MASKON, 		0xB01034A8,__READ_WRITE ,__gfxtcon_dir_spg10maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG10POSOFF, 		0xB01034AC,__READ_WRITE ,__gfxtcon_dir_spg10posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG10MASKOFF,		0xB01034B0,__READ_WRITE ,__gfxtcon_dir_spg10maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG11POSON,  		0xB01034B4,__READ_WRITE ,__gfxtcon_dir_spg11poson_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG11MASKON, 		0xB01034B8,__READ_WRITE ,__gfxtcon_dir_spg11maskon_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG11POSOFF, 		0xB01034BC,__READ_WRITE ,__gfxtcon_dir_spg11posoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SPG11MASKOFF,		0xB01034C0,__READ_WRITE ,__gfxtcon_dir_spg11maskoff_bits);
__IO_REG32_BIT(GFXTCON_DIR_SSQCYCLE,				0xB01034C4,__READ_WRITE ,__gfxtcon_dir_ssqcycle_bits);
__IO_REG32_BIT(GFXTCON_DIR_SMX0SIGS,				0xB01034C8,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX0FCTTABLE,		0xB01034CC,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX1SIGS,				0xB01034D0,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX1FCTTABLE,		0xB01034D4,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX2SIGS,				0xB01034D8,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX2FCTTABLE,		0xB01034DC,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX3SIGS,				0xB01034E0,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX3FCTTABLE,		0xB01034E4,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX4SIGS,				0xB01034E8,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX4FCTTABLE,		0xB01034EC,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX5SIGS,				0xB01034F0,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX5FCTTABLE,		0xB01034F4,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX6SIGS,				0xB01034F8,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX6FCTTABLE,		0xB01034FC,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX7SIGS,				0xB0103500,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX7FCTTABLE,		0xB0103504,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX8SIGS,				0xB0103508,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX8FCTTABLE,		0xB010350C,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX9SIGS,				0xB0103510,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX9FCTTABLE,		0xB0103514,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX10SIGS,				0xB0103518,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX10FCTTABLE,		0xB010351C,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SMX11SIGS,				0xB0103520,__READ_WRITE ,__gfxtcon_dir_smxsigs_bits);
__IO_REG32(		 GFXTCON_DIR_SMX11FCTTABLE,		0xB0103524,__READ_WRITE );
__IO_REG32_BIT(GFXTCON_DIR_SSWITCH,					0xB0103528,__READ_WRITE ,__gfxtcon_dir_sswitch_bits);
__IO_REG32_BIT(GFXTCON_RBM_CTRL,						0xB010352C,__READ_WRITE ,__gfxtcon_rbm_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN0_CTRL,				0xB0103534,__READ_WRITE ,__gfxtcon_dir_pin0_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN1_CTRL,				0xB0103538,__READ_WRITE ,__gfxtcon_dir_pin1_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN2_CTRL,				0xB010353C,__READ_WRITE ,__gfxtcon_dir_pin2_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN3_CTRL,				0xB0103540,__READ_WRITE ,__gfxtcon_dir_pin3_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN4_CTRL,				0xB0103544,__READ_WRITE ,__gfxtcon_dir_pin4_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN5_CTRL,				0xB0103548,__READ_WRITE ,__gfxtcon_dir_pin5_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN6_CTRL,				0xB010354C,__READ_WRITE ,__gfxtcon_dir_pin6_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN7_CTRL,				0xB0103550,__READ_WRITE ,__gfxtcon_dir_pin7_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN8_CTRL,				0xB0103554,__READ_WRITE ,__gfxtcon_dir_pin8_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN9_CTRL,				0xB0103558,__READ_WRITE ,__gfxtcon_dir_pin9_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN10_CTRL,			0xB010355C,__READ_WRITE ,__gfxtcon_dir_pin10_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN11_CTRL,			0xB0103560,__READ_WRITE ,__gfxtcon_dir_pin11_ctrl_bits);
__IO_REG32_BIT(GFXTCON_DIR_PIN12_CTRL,			0xB0103564,__READ_WRITE ,__gfxtcon_dir_pin12_ctrl_bits);

/***************************************************************************
 **
 ** GFXDISP
 **
 ***************************************************************************/
__IO_REG32_BIT(GFXDISP_DISPLAYENABLE,     				0xB0104000,__READ_WRITE ,__gfxdisp_displayenable_bits);
__IO_REG32_BIT(GFXDISP_DISPLAYRESOLUTION,   			0xB0104004,__READ_WRITE ,__gfxdisp_displayresolution_bits);
__IO_REG32_BIT(GFXDISP_DISPLAYACTIVEAREA,   			0xB0104008,__READ_WRITE ,__gfxdisp_displayactivearea_bits);
__IO_REG32_BIT(GFXDISP_HORIZONTALSYNCHTIMINGCONF,	0xB010400C,__READ_WRITE ,__gfxdisp_horizontalsynchtimingconf_bits);
__IO_REG32_BIT(GFXDISP_VERTICALSYNCHTIMINGCONF,   0xB0104010,__READ_WRITE ,__gfxdisp_verticalsynchtimingconf_bits);
__IO_REG32_BIT(GFXDISP_DISPLAYCONF,     					0xB0104014,__READ_WRITE ,__gfxdisp_displayconf_bits);
__IO_REG32_BIT(GFXDISP_PIXENGTRIG,     						0xB0104018,__READ_WRITE ,__gfxdisp_pixengtrig_bits);
__IO_REG32_BIT(GFXDISP_DITHERCONTROL,     				0xB010401C,__READ_WRITE ,__gfxdisp_dithercontrol_bits);
__IO_REG32_BIT(GFXDISP_INT0TRIGGER,     					0xB0104020,__READ_WRITE ,__gfxdisp_int0trigger_bits);
__IO_REG32_BIT(GFXDISP_INT1TRIGGER,     					0xB0104024,__READ_WRITE ,__gfxdisp_int1trigger_bits);
__IO_REG32_BIT(GFXDISP_INT2TRIGGER,     					0xB0104028,__READ_WRITE ,__gfxdisp_int2trigger_bits);
__IO_REG32_BIT(GFXDISP_DEBUG,     								0xB010402C,__READ_WRITE ,__gfxdisp_debug_bits);

/***************************************************************************
 **
 ** GFXDISP
 **
 ***************************************************************************/
__IO_REG32(		 GFXSIG_SIGLOCKUNLOCK,     					0xB0105000,__READ_WRITE );
__IO_REG32_BIT(GFXSIG_SIGLOCKSTATUS,     					0xB0105004,__READ_WRITE ,__gfxsig_siglockstatus_bits);
__IO_REG32_BIT(GFXSIG_SIGSWRESET,     						0xB0105008,__READ_WRITE ,__gfxsig_sigswreset_bits);
__IO_REG32_BIT(GFXSIG_SIGCTRL,     								0xB010500C,__READ_WRITE ,__gfxsig_sigctrl_bits);
__IO_REG32_BIT(GFXSIG_MASKHORIZONTALUPPERLEFT,    0xB0105010,__READ_WRITE ,__gfxsig_maskhorizontalupperleft_bits);
__IO_REG32_BIT(GFXSIG_MASKHORIZONTALLOWERRIGHT,   0xB0105014,__READ_WRITE ,__gfxsig_maskhorizontallowerright_bits);
__IO_REG32_BIT(GFXSIG_MASKVERTICALUPPERLEFT,     	0xB0105018,__READ_WRITE ,__gfxsig_maskverticalupperleft_bits);
__IO_REG32_BIT(GFXSIG_MASKVERTICALLOWERRIGHT,     0xB010501C,__READ_WRITE ,__gfxsig_maskverticallowerright_bits);
__IO_REG32_BIT(GFXSIG_HORIZONTALUPPERLEFTW0,     	0xB0105020,__READ_WRITE ,__gfxsig_horizontalupperleftw0_bits);
__IO_REG32_BIT(GFXSIG_HORIZONTALLOWERRIGHTW0,     0xB0105024,__READ_WRITE ,__gfxsig_horizontallowerrightw0_bits);
__IO_REG32_BIT(GFXSIG_VERTICALUPPERLEFTW0,     		0xB0105028,__READ_WRITE ,__gfxsig_verticalupperleftw0_bits);
__IO_REG32_BIT(GFXSIG_VERTICALLOWERRIGHTW0,     	0xB010502C,__READ_WRITE ,__gfxsig_verticallowerrightw0_bits);
__IO_REG32(		 GFXSIG_SIGNAREFERENCERW0,     			0xB0105030,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNAREFERENCEGW0,     			0xB0105034,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNAREFERENCEBW0,     			0xB0105038,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNBREFERENCERW0,     			0xB010503C,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNBREFERENCEGW0,     			0xB0105040,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNBREFERENCEBW0,     			0xB0105044,__READ_WRITE );
__IO_REG32(		 GFXSIG_THRBRW0,     								0xB0105048,__READ_WRITE );
__IO_REG32(		 GFXSIG_THRBGW0,     								0xB010504C,__READ_WRITE );
__IO_REG32(		 GFXSIG_THRBBW0,     								0xB0105050,__READ_WRITE );
__IO_REG32_BIT(GFXSIG_ERRORTHRESHOLD,     				0xB0105054,__READ_WRITE ,__gfxsig_errorthreshold_bits);
__IO_REG32_BIT(GFXSIG_CTRLCFGW0,     							0xB0105058,__READ_WRITE ,__gfxsig_ctrlcfgw0_bits);
__IO_REG32_BIT(GFXSIG_TRIGGERW0,     							0xB010505C,__READ_WRITE ,__gfxsig_triggerw0_bits);
__IO_REG32_BIT(GFXSIG_IENW0,     									0xB0105060,__READ_WRITE ,__gfxsig_ienw0_bits);
__IO_REG32_BIT(GFXSIG_INTERRUPTSTATUSW0,     			0xB0105064,__READ_WRITE ,__gfxsig_interruptstatusw0_bits);
__IO_REG32_BIT(GFXSIG_STATUSW0,     							0xB0105068,__READ_WRITE ,__gfxsig_statusw0_bits);
__IO_REG32_BIT(GFXSIG_SIGNATURE_ERROR,     				0xB010506C,__READ_WRITE ,__gfxsig_signature_error_bits);
__IO_REG32(		 GFXSIG_SIGNATUREARW0,     					0xB0105070,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNATUREAGW0,     					0xB0105074,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNATUREABW0,     					0xB0105078,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNBTUREARW0,     					0xB010507C,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNBTUREAGW0,     					0xB0105080,__READ_WRITE );
__IO_REG32(		 GFXSIG_SIGNBTUREABW0,     					0xB0105084,__READ_WRITE );

/***************************************************************************
 **
 ** GFXAIC
 **
 ***************************************************************************/
__IO_REG32_BIT(GFXAIC_STATUS,     								0xB0106000,__READ_WRITE ,__gfxaic_status_bits);
__IO_REG32_BIT(GFXAIC_CONTROL,     								0xB0106004,__READ_WRITE ,__gfxaic_control_bits);
__IO_REG32_BIT(GFXAIC_MONITORDISABLE,     				0xB0106008,__READ_WRITE ,__gfxaic_monitordisable_bits);
__IO_REG32_BIT(GFXAIC_SLAVEDISABLE,     					0xB010600C,__READ_WRITE ,__gfxaic_slavedisable_bits);

/***************************************************************************
 **
 ** GFXPIX
 **
 ***************************************************************************/
__IO_REG32_BIT(GFXPIX_FETCH0_STATUS,     									0xB0108000,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_BURSTBUFFERMANAGEMENT,				0xB0108008,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH0_BASEADDRESS,									0xB010800C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH0_SOURCEBUFFERSTRIDE,					0xB0108010,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_SOURCEBUFFERATTRIBUTES,			0xB0108014,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_SOURCEBUFFERLENGTH,					0xB0108018,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_FRAMEXOFFSET,								0xB010801C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_FRAMEYOFFSET,								0xB0108020,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_FRAMEDIMENSIONS,							0xB0108024,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_SKIPWINDOWOFFSET,						0xB0108038,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_SKIPWINDOWDIMENSIONS,				0xB010803C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_COLORCOMPONENTBITS,					0xB0108040,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_COLORCOMPONENTSHIFT,					0xB0108044,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_CONSTANTCOLOR,								0xB0108048,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH0_CONTROL,											0xB010804C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_STATUS,     									0xB0108400,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_BURSTBUFFERMANAGEMENT,				0xB0108408,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH1_BASEADDRESS,									0xB010840C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH1_SOURCEBUFFERSTRIDE,					0xB0108410,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_SOURCEBUFFERATTRIBUTES,			0xB0108414,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_SOURCEBUFFERLENGTH,					0xB0108418,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_FRAMEXOFFSET,								0xB010841C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_FRAMEYOFFSET,								0xB0108420,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_FRAMEDIMENSIONS,							0xB0108424,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_SKIPWINDOWOFFSET,						0xB0108438,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_SKIPWINDOWDIMENSIONS,				0xB010843C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_COLORCOMPONENTSHIFT,					0xB0108444,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_CONSTANTCOLOR,								0xB0108448,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_CONTROL,											0xB010844C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH1_COLORCOMPONENTBITS,					0xB0108440,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_STATUS,     									0xB0108800,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_BURSTBUFFERMANAGEMENT,				0xB0108808,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH2_BASEADDRESS,									0xB010880C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH2_SOURCEBUFFERSTRIDE,					0xB0108810,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_SOURCEBUFFERATTRIBUTES,			0xB0108814,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_SOURCEBUFFERLENGTH,					0xB0108818,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_FRAMEXOFFSET,								0xB010881C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_FRAMEYOFFSET,								0xB0108820,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_FRAMEDIMENSIONS,							0xB0108824,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_SKIPWINDOWOFFSET,						0xB0108838,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_SKIPWINDOWDIMENSIONS,				0xB010883C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_COLORCOMPONENTBITS,					0xB0108840,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_COLORCOMPONENTSHIFT,					0xB0108844,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_CONSTANTCOLOR,								0xB0108848,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH2_CONTROL,											0xB010884C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_STATUS,     									0xB0108C00,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_BURSTBUFFERMANAGEMENT,				0xB0108C08,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH3_BASEADDRESS,									0xB0108C0C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH3_SOURCEBUFFERSTRIDE,					0xB0108C10,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_SOURCEBUFFERATTRIBUTES,			0xB0108C14,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_SOURCEBUFFERLENGTH,					0xB0108C18,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_FRAMEXOFFSET,								0xB0108C1C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_FRAMEYOFFSET,								0xB0108C20,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_FRAMEDIMENSIONS,							0xB0108C24,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_SKIPWINDOWOFFSET,						0xB0108C38,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_SKIPWINDOWDIMENSIONS,				0xB0108C3C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_COLORCOMPONENTBITS,					0xB0108C40,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_COLORCOMPONENTSHIFT,					0xB0108C44,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_CONSTANTCOLOR,								0xB0108C48,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH3_CONTROL,											0xB0108C4C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_STATUS,     									0xB0109000,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_BURSTBUFFERMANAGEMENT,				0xB0109008,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH4_BASEADDRESS,									0xB010900C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH4_SOURCEBUFFERSTRIDE,					0xB0109010,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_SOURCEBUFFERATTRIBUTES,			0xB0109014,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_SOURCEBUFFERLENGTH,					0xB0109018,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_FRAMEXOFFSET,								0xB010901C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_FRAMEYOFFSET,								0xB0109020,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_FRAMEDIMENSIONS,							0xB0109024,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_SKIPWINDOWOFFSET,						0xB0109038,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_SKIPWINDOWDIMENSIONS,				0xB010903C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_COLORCOMPONENTBITS,					0xB0109040,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_COLORCOMPONENTSHIFT,					0xB0109044,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_CONSTANTCOLOR,								0xB0109048,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH4_CONTROL,											0xB010904C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_STATUS,     									0xB0109400,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_BURSTBUFFERMANAGEMENT,				0xB0109408,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH5_BASEADDRESS,									0xB010940C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH5_SOURCEBUFFERSTRIDE,					0xB0109410,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_SOURCEBUFFERATTRIBUTES,			0xB0109414,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_SOURCEBUFFERLENGTH,					0xB0109418,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_FRAMEXOFFSET,								0xB010941C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_FRAMEYOFFSET,								0xB0109420,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_FRAMEDIMENSIONS,							0xB0109424,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_SKIPWINDOWOFFSET,						0xB0109438,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_SKIPWINDOWDIMENSIONS,				0xB010943C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_COLORCOMPONENTBITS,					0xB0109440,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_COLORCOMPONENTSHIFT,					0xB0109444,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_CONSTANTCOLOR,								0xB0109448,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH5_CONTROL,											0xB010944C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_STATUS,     									0xB0109800,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_BURSTBUFFERMANAGEMENT,				0xB0109808,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH6_BASEADDRESS,									0xB010980C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH6_SOURCEBUFFERSTRIDE,					0xB0109810,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_SOURCEBUFFERATTRIBUTES,			0xB0109814,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_SOURCEBUFFERLENGTH,					0xB0109818,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_FRAMEXOFFSET,								0xB010981C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_FRAMEYOFFSET,								0xB0109820,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_FRAMEDIMENSIONS,							0xB0109824,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_SKIPWINDOWOFFSET,						0xB0109838,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_SKIPWINDOWDIMENSIONS,				0xB010983C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_COLORCOMPONENTBITS,					0xB0109840,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_COLORCOMPONENTSHIFT,					0xB0109844,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_CONSTANTCOLOR,								0xB0109848,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH6_CONTROL,											0xB010984C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_STATUS,     									0xB0109C00,__READ_WRITE ,__gfxpix_fetch_status_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_BURSTBUFFERMANAGEMENT,				0xB0109C08,__READ_WRITE ,__gfxpix_fetch_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_FETCH7_BASEADDRESS,									0xB0109C0C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_FETCH7_SOURCEBUFFERSTRIDE,					0xB0109C10,__READ_WRITE ,__gfxpix_fetch_sourcebufferstride_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_SOURCEBUFFERATTRIBUTES,			0xB0109C14,__READ_WRITE ,__gfxpix_fetch_sourcebufferattributes_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_SOURCEBUFFERLENGTH,					0xB0109C18,__READ_WRITE ,__gfxpix_fetch_sourcebufferlength_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_FRAMEXOFFSET,								0xB0109C1C,__READ_WRITE ,__gfxpix_fetch_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_FRAMEYOFFSET,								0xB0109C20,__READ_WRITE ,__gfxpix_fetch_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_FRAMEDIMENSIONS,							0xB0109C24,__READ_WRITE ,__gfxpix_fetch_framedimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_SKIPWINDOWOFFSET,						0xB0109C38,__READ_WRITE ,__gfxpix_fetch_skipwindowoffset_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_SKIPWINDOWDIMENSIONS,				0xB0109C3C,__READ_WRITE ,__gfxpix_fetch_skipwindowdimensions_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_COLORCOMPONENTBITS,					0xB0109C40,__READ_WRITE ,__gfxpix_fetch_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_COLORCOMPONENTSHIFT,					0xB0109C44,__READ_WRITE ,__gfxpix_fetch_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_CONSTANTCOLOR,								0xB0109C48,__READ_WRITE ,__gfxpix_fetch_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_FETCH7_CONTROL,											0xB0109C4C,__READ_WRITE ,__gfxpix_fetch_control_bits);
__IO_REG32_BIT(GFXPIX_STORE0_STATUS,											0xB010A000,__READ_WRITE ,__gfxpix_store_status_bits);
__IO_REG32(		 GFXPIX_STORE0_LAST_CONTROL_WORD,						0xB010A004,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_STORE0_BURSTBUFFERMANAGEMENT,				0xB010A008,__READ_WRITE ,__gfxpix_store_burstbuffermanagement_bits);
__IO_REG32(		 GFXPIX_STORE0_BASEADDRESS,									0xB010A00C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_STORE0_DESTINATIONBUFFERSTRIDE,			0xB010A010,__READ_WRITE ,__gfxpix_store_destinationbufferstride_bits);
__IO_REG32_BIT(GFXPIX_STORE0_FRAMEXOFFSET,								0xB010A01C,__READ_WRITE ,__gfxpix_store_framexoffset_bits);
__IO_REG32_BIT(GFXPIX_STORE0_FRAMEYOFFSET,								0xB010A020,__READ_WRITE ,__gfxpix_store_frameyoffset_bits);
__IO_REG32_BIT(GFXPIX_STORE0_COLORCOMPONENTBITS,					0xB010A040,__READ_WRITE ,__gfxpix_store_colorcomponentbits_bits);
__IO_REG32_BIT(GFXPIX_STORE0_COLORCOMPONENTSHIFT,					0xB010A044,__READ_WRITE ,__gfxpix_store_colorcomponentshift_bits);
__IO_REG32_BIT(GFXPIX_STORE0_CONTROLE,										0xB010A04C,__READ_WRITE ,__gfxpix_store_control_bits);
__IO_REG32(		 GFXPIX_STORE0_PERFCOUNTER,									0xB010A050,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_HSCALER0_CONTROL,										0xB010A400,__READ_WRITE ,__gfxpix_hscaler_control_bits);
__IO_REG32_BIT(GFXPIX_HSCALER0_SETUP1,										0xB010A404,__READ_WRITE ,__gfxpix_hscaler_setup1_bits);
__IO_REG32_BIT(GFXPIX_HSCALER0_SETUP2,										0xB010A408,__READ_WRITE ,__gfxpix_hscaler_setup2_bits);
__IO_REG32_BIT(GFXPIX_VSCALER0_CONTROL,										0xB010A800,__READ_WRITE ,__gfxpix_vscaler_control_bits);
__IO_REG32_BIT(GFXPIX_VSCALER0_SETUP1,										0xB010A804,__READ_WRITE ,__gfxpix_vscaler_setup1_bits);
__IO_REG32_BIT(GFXPIX_VSCALER0_SETUP2,										0xB010A808,__READ_WRITE ,__gfxpix_vscaler_setup2_bits);
__IO_REG32_BIT(GFXPIX_ROP0_CONTROL,												0xB010AC00,__READ_WRITE ,__gfxpix_rop_control_bits);
__IO_REG32_BIT(GFXPIX_ROP0_RASTEROPERATIONINDICES,				0xB010AC04,__READ_WRITE ,__gfxpix_rop_rasteroperationindices_bits);
__IO_REG32(		 GFXPIX_ROP0_PRIM_CONTROL_WORD,							0xB010AC08,__READ_WRITE );
__IO_REG32(		 GFXPIX_ROP0_SEC_CONTROL_WORD,							0xB010AC0C,__READ_WRITE );
__IO_REG32(		 GFXPIX_ROP0_TERT_CONTROL_WORD,							0xB010AC10,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_ROP1_CONTROL,												0xB010B000,__READ_WRITE ,__gfxpix_rop_control_bits);
__IO_REG32_BIT(GFXPIX_ROP1_RASTEROPERATIONINDICES,				0xB010B004,__READ_WRITE ,__gfxpix_rop_rasteroperationindices_bits);
__IO_REG32(		 GFXPIX_ROP1_PRIM_CONTROL_WORD,							0xB010B008,__READ_WRITE );
__IO_REG32(		 GFXPIX_ROP1_SEC_CONTROL_WORD,							0xB010B00C,__READ_WRITE );
__IO_REG32(		 GFXPIX_ROP1_TERT_CONTROL_WORD,							0xB010B010,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_BLITBLEND0_CONTROL,									0xB010B400,__READ_WRITE ,__gfxpix_blitblend_control_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_CONSTANTCOLOR,						0xB010B404,__READ_WRITE ,__gfxpix_blitblend_constantcolor_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_COLORREDBLENDFUNCTION,		0xB010B408,__READ_WRITE ,__gfxpix_blitblend_colorredblendfunction_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_COLORGREENBLENDFUNCTION,	0xB010B40C,__READ_WRITE ,__gfxpix_blitblend_colorgreenblendfunction_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_COLORBLUEBLENDFUNCTION,	0xB010B410,__READ_WRITE ,__gfxpix_blitblend_colorblueblendfunction_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_ALPHABLENDFUNCTION,			0xB010B414,__READ_WRITE ,__gfxpix_blitblend_alphablendfunction_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_BLENDMODE1,							0xB010B418,__READ_WRITE ,__gfxpix_blitblend_blendmode1_bits);
__IO_REG32_BIT(GFXPIX_BLITBLEND0_BLENDMODE2,							0xB010B41C,__READ_WRITE ,__gfxpix_blitblend_blendmode2_bits);
__IO_REG32(		 GFXPIX_BLITBLEND0_PRIM_CONTROL_WORD,				0xB010B420,__READ_WRITE );
__IO_REG32(		 GFXPIX_BLITBLEND0_SEC_CONTROL_WORD,				0xB010B424,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_BLITBLEND0_DEBUG,										0xB010B428,__READ_WRITE ,__gfxpix_blitblend_debug_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND0_CONTROL,								0xB010B800,__READ_WRITE ,__gfxpix_layerblend_control_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND0_POSITION,								0xB010B804,__READ_WRITE ,__gfxpix_layerblend_position_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND0_TRANS_COL,							0xB010B808,__READ_WRITE ,__gfxpix_layerblend_trans_col_bits);
__IO_REG32(		 GFXPIX_LAYERBLEND0_PRIM_CONTROL_WORD,			0xB010B80C,__READ_WRITE );
__IO_REG32(		 GFXPIX_LAYERBLEND0_SEC_CONTROL_WORD,				0xB010B810,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_LAYERBLEND1_CONTROL,								0xB010BC00,__READ_WRITE ,__gfxpix_layerblend_control_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND1_POSITION,								0xB010BC04,__READ_WRITE ,__gfxpix_layerblend_position_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND1_TRANS_COL,							0xB010BC08,__READ_WRITE ,__gfxpix_layerblend_trans_col_bits);
__IO_REG32(		 GFXPIX_LAYERBLEND1_PRIM_CONTROL_WORD,			0xB010BC0C,__READ_WRITE );
__IO_REG32(		 GFXPIX_LAYERBLEND1_SEC_CONTROL_WORD,				0xB010BC10,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_LAYERBLEND2_CONTROL,								0xB010C000,__READ_WRITE ,__gfxpix_layerblend_control_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND2_POSITION,								0xB010C004,__READ_WRITE ,__gfxpix_layerblend_position_bits);
__IO_REG32_BIT(GFXPIX_LAYERBLEND2_TRANS_COL,							0xB010C008,__READ_WRITE ,__gfxpix_layerblend_trans_col_bits);
__IO_REG32(		 GFXPIX_LAYERBLEND2_PRIM_CONTROL_WORD,			0xB010C00C,__READ_WRITE );
__IO_REG32(		 GFXPIX_LAYERBLEND2_SEC_CONTROL_WORD,				0xB010C010,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_LUT0_CONTROL,												0xB010C400,__READ_WRITE ,__gfxpix_lut0_control_bits);
__IO_REG32(		 GFXPIX_LUT0_LAST_CONTROL_WORD,							0xB010C404,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_LUT0_LUT,														0xB010C800,__READ_WRITE ,__gfxpix_lut0_lut_bits);
__IO_REG32_BIT(GFXPIX_LUT1_CONTROL,												0xB010CC00,__READ_WRITE ,__gfxpix_lut1_control_bits);
__IO_REG32(		 GFXPIX_LUT1_LAST_CONTROL_WORD,							0xB010CC04,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_LUT1_LUT,														0xB010D000,__READ_WRITE ,__gfxpix_lut1_lut_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_CONTROL,										0xB010D400,__READ_WRITE ,__gfxpix_matrix_control_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_RED0,												0xB010D404,__READ_WRITE ,__gfxpix_matrix_red0_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_RED1,												0xB010D408,__READ_WRITE ,__gfxpix_matrix_red1_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_GREEN0,											0xB010D40C,__READ_WRITE ,__gfxpix_matrix_green0_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_GREEN1,											0xB010D410,__READ_WRITE ,__gfxpix_matrix_green1_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_BLUE0,											0xB010D414,__READ_WRITE ,__gfxpix_matrix_blue0_bits);
__IO_REG32_BIT(GFXPIX_MATRIX0_BLUE1,											0xB010D418,__READ_WRITE ,__gfxpix_matrix_blue1_bits);
__IO_REG32(		 GFXPIX_MATRIX0_LAST_CONTROL_WORD,					0xB010D41C,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_EXTDST0_CONTROL,										0xB010D800,__READ_WRITE ,__gfxpix_extdst0_control_bits);
__IO_REG32_BIT(GFXPIX_EXTDST0_STATUS,											0xB010D804,__READ_WRITE ,__gfxpix_extdst0_status_bits);
__IO_REG32(		 GFXPIX_EXTDST0_CONTROL_WORD,								0xB010D808,__READ_WRITE );
__IO_REG32_BIT(GFXPIX_EXTDST0_CUR_PIXEL_CNT,							0xB010D80C,__READ_WRITE ,__gfxpix_extdst0_cur_pixel_cnt_bits);
__IO_REG32_BIT(GFXPIX_EXTDST0_LAST_PIXEL_CNT,							0xB010D810,__READ_WRITE ,__gfxpix_extdst0_last_pixel_cnt_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH0_CFG,								0xB010DC00,__READ_WRITE ,__gfxpix_pixelbus_fetch0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH1_CFG,								0xB010DC04,__READ_WRITE ,__gfxpix_pixelbus_fetch1_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH2_CFG,								0xB010DC08,__READ_WRITE ,__gfxpix_pixelbus_fetch2_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH3_CFG,								0xB010DC0C,__READ_WRITE ,__gfxpix_pixelbus_fetch3_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH4_CFG,								0xB010DC10,__READ_WRITE ,__gfxpix_pixelbus_fetch4_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH5_CFG,								0xB010DC14,__READ_WRITE ,__gfxpix_pixelbus_fetch5_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH6_CFG,								0xB010DC18,__READ_WRITE ,__gfxpix_pixelbus_fetch6_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_FETCH7_CFG,								0xB010DC1C,__READ_WRITE ,__gfxpix_pixelbus_fetch7_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_STORE0_CFG,								0xB010DC20,__READ_WRITE ,__gfxpix_pixelbus_store0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_HSCALER0_CFG,							0xB010DC24,__READ_WRITE ,__gfxpix_pixelbus_hscaler0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_VSCALER0_CFG,							0xB010DC28,__READ_WRITE ,__gfxpix_pixelbus_vscaler0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_ROP0_CFG,									0xB010DC2C,__READ_WRITE ,__gfxpix_pixelbus_rop0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_ROP1_CFG,									0xB010DC30,__READ_WRITE ,__gfxpix_pixelbus_rop1_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_BLITBLEND0_CFG,						0xB010DC34,__READ_WRITE ,__gfxpix_pixelbus_blitblend0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_LAYERBLEND0_CFG,						0xB010DC38,__READ_WRITE ,__gfxpix_pixelbus_layerblend0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_LAYERBLEND1_CFG,						0xB010DC3C,__READ_WRITE ,__gfxpix_pixelbus_layerblend1_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_LAYERBLEND2_CFG,						0xB010DC40,__READ_WRITE ,__gfxpix_pixelbus_layerblend2_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_LUT0_CFG,									0xB010DC44,__READ_WRITE ,__gfxpix_pixelbus_lut0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_LUT1_CFG,									0xB010DC48,__READ_WRITE ,__gfxpix_pixelbus_lut1_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_MATRIX0_CFG,								0xB010DC4C,__READ_WRITE ,__gfxpix_pixelbus_matrix0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_EXTDST0_CFG,								0xB010DC50,__READ_WRITE ,__gfxpix_pixelbus_extdst0_cfg_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_STORE0_SYNC,								0xB010DC54,__READ_WRITE ,__gfxpix_pixelbus_store0_sync_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_STORE0_SYNC_STAT,					0xB010DC58,__READ_WRITE ,__gfxpix_pixelbus_store0_sync_stat_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_EXTDST0_SYNC,							0xB010DC5C,__READ_WRITE ,__gfxpix_pixelbus_extdst0_sync_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_EXTDST0_SYNC_STAT,					0xB010DC60,__READ_WRITE ,__gfxpix_pixelbus_extdst0_sync_stat_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_STORE0_CLK,								0xB010DC64,__READ_WRITE ,__gfxpix_pixelbus_store0_clk_bits);
__IO_REG32_BIT(GFXPIX_PIXELBUS_EXTDST0_CLK,								0xB010DC68,__READ_WRITE ,__gfxpix_pixelbus_extdst0_clk_bits);

/***************************************************************************
 **
 ** GFXPIX
 **
 ***************************************************************************/
__IO_REG32_BIT(GFXPIX_MSS_AR_ARBITRATION,     						0xB0110408,__READ_WRITE ,__gfxpix_mss_ar_arbitration_bits);
__IO_REG32_BIT(GFXPIX_MSS_AW_ARBITRATION,									0xB011040C,__READ_WRITE ,__gfxpix_mss_aw_arbitration_bits);
__IO_REG32_BIT(GFXPIX_VRAM0_AR_ARBITRATION,     					0xB0110428,__READ_WRITE ,__gfxpix_vram0_ar_arbitration_bits);
__IO_REG32_BIT(GFXPIX_VRAM0_AW_ARBITRATION,								0xB011042C,__READ_WRITE ,__gfxpix_vram0_aw_arbitration_bits);
__IO_REG32_BIT(GFXPIX_VRAM1_AR_ARBITRATION,     					0xB0110448,__READ_WRITE ,__gfxpix_vram1_ar_arbitration_bits);
__IO_REG32_BIT(GFXPIX_VRAM1_AW_ARBITRATION,								0xB011044C,__READ_WRITE ,__gfxpix_vram1_aw_arbitration_bits);
__IO_REG32_BIT(GFXPIX_AHB_AR_ARBITRATION,     						0xB0110468,__READ_WRITE ,__gfxpix_ahb_ar_arbitration_bits);
__IO_REG32_BIT(GFXPIX_AHB_AW_ARBITRATION,									0xB011046C,__READ_WRITE ,__gfxpix_ahb_aw_arbitration_bits);
__IO_REG32_BIT(GFXPIX_HPM_AR_ARBITRATION,     						0xB0110488,__READ_WRITE ,__gfxpix_hpm_ar_arbitration_bits);
__IO_REG32_BIT(GFXPIX_HPM_AW_ARBITRATION,									0xB011048C,__READ_WRITE ,__gfxpix_hpm_aw_arbitration_bits);

/***************************************************************************
 **
 ** MCFG
 **
 ***************************************************************************/
__IO_REG32_BIT(MCFG_DTAR,     														0xB050E000,__READ_WRITE ,__mcfg_dtar_bits);
__IO_REG32_BIT(MCFG_TSR,     															0xB050E004,__READ_WRITE ,__mcfg_tsr_bits);

/***************************************************************************
 **
 ** RICFG7G0
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG7G0_GFX0DCLKI,     										0xB06F8000,__READ_WRITE ,__ricfg7g0_gfx0dclki_bits);

/***************************************************************************
 **
 ** RICFG7G0
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG7G4_EIC0INT00,     										0xB06F9000,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT01,     										0xB06F9004,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT02,     										0xB06F9008,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT03,     										0xB06F900C,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT04,     										0xB06F9010,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT05,     										0xB06F9014,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT06,     										0xB06F9018,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT07,     										0xB06F901C,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT08,     										0xB06F9020,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT09,     										0xB06F9024,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT10,     										0xB06F9028,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT11,     										0xB06F902C,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT12,     										0xB06F9030,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT13,     										0xB06F9034,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT14,     										0xB06F9038,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT15,     										0xB06F903C,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT16,     										0xB06F9040,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT17,     										0xB06F9044,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT18,     										0xB06F9048,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT19,     										0xB06F904C,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT20,     										0xB06F9050,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT21,     										0xB06F9054,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT22,     										0xB06F9058,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT23,     										0xB06F905C,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT24,     										0xB06F9060,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT25,     										0xB06F9064,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT26,     										0xB06F9068,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT27,     										0xB06F906C,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT28,     										0xB06F9070,__READ_WRITE ,__ricfg7g4_eic0int08_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT29,     										0xB06F9074,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT30,     										0xB06F9078,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0INT31,     										0xB06F907C,__READ_WRITE ,__ricfg7g4_eic0int_bits);
__IO_REG16_BIT(RICFG7G4_EIC0NMI,     											0xB06F9080,__READ_WRITE ,__ricfg7g4_eic0nmi_bits);

/***************************************************************************
 **
 ** RICFG0G0
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G0_ADC0AN26,     										0xB07F8000,__READ_WRITE ,__ricfg0g0_adc0an_bits);
__IO_REG16_BIT(RICFG0G0_ADC0AN27,     										0xB07F8002,__READ_WRITE ,__ricfg0g0_adc0an_bits);
__IO_REG16_BIT(RICFG0G0_ADC0AN28,     										0xB07F8004,__READ_WRITE ,__ricfg0g0_adc0an_bits);
__IO_REG16_BIT(RICFG0G0_ADC0AN29,     										0xB07F8006,__READ_WRITE ,__ricfg0g0_adc0an_bits);
__IO_REG16_BIT(RICFG0G0_ADC0AN30,     										0xB07F8008,__READ_WRITE ,__ricfg0g0_adc0an_bits);
__IO_REG16_BIT(RICFG0G0_ADC0AN31,     										0xB07F800A,__READ_WRITE ,__ricfg0g0_adc0an_bits);
__IO_REG16_BIT(RICFG0G0_ADC0EDGI,     										0xB07F800C,__READ_WRITE ,__ricfg0g0_adc0edgi_bits);
__IO_REG16_BIT(RICFG0G0_ADC0EDGIOCU0,     								0xB07F800E,__READ_WRITE ,__ricfg0g0_adc0edgiocu0_bits);
__IO_REG16_BIT(RICFG0G0_ADC0EDGIOCU4,  										0xB07F8016,__READ_WRITE ,__ricfg0g0_adc0edgiocu4_bits);
__IO_REG16_BIT(RICFG0G0_ADC0TIMI,  												0xB07F801E,__READ_WRITE ,__ricfg0g0_adc0timi_bits);
__IO_REG16_BIT(RICFG0G0_ADC0TIMIRLT,  										0xB07F8020,__READ_WRITE ,__ricfg0g0_adc0timirlt_bits);
__IO_REG16_BIT(RICFG0G0_ADC0ZPDEN,  											0xB07F803E,__READ_WRITE ,__ricfg0g0_adc0zpden_bits);

/***************************************************************************
 **
 ** RICFG0G1
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G1_FRT0TEXT,     										0xB07F8400,__READ_WRITE ,__ricfg0g1_frt0text_bits);
__IO_REG16_BIT(RICFG0G1_FRT1TEXT,     										0xB07F8420,__READ_WRITE ,__ricfg0g1_frt1text_bits);
__IO_REG16_BIT(RICFG0G1_FRT2TEXT,     										0xB07F8440,__READ_WRITE ,__ricfg0g1_frt2text_bits);
__IO_REG16_BIT(RICFG0G1_FRT3TEXT,     										0xB07F8460,__READ_WRITE ,__ricfg0g1_frt3text_bits);

/***************************************************************************
 **
 ** RICFG0G2
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G2_CU2IN0,     											0xB07F8840,__READ_WRITE ,__ricfg0g2_cuin_bits);
__IO_REG16_BIT(RICFG0G2_CU2IN1,     											0xB07F8842,__READ_WRITE ,__ricfg0g2_cuin_bits);
__IO_REG16_BIT(RICFG0G2_ICU2FRTSEL,     									0xB07F8844,__READ_WRITE ,__ricfg0g2_icufrtsel_bits);
__IO_REG16_BIT(RICFG0G2_CU3IN0,     											0xB07F8860,__READ_WRITE ,__ricfg0g2_cuin_bits);
__IO_REG16_BIT(RICFG0G2_CU3IN1,     											0xB07F8862,__READ_WRITE ,__ricfg0g2_cuin_bits);
__IO_REG16_BIT(RICFG0G2_ICU3FRTSEL,     									0xB07F8864,__READ_WRITE ,__ricfg0g2_icufrtsel_bits);

/***************************************************************************
 **
 ** RICFG0G3
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G3_OCU0OTD0GATE,     								0xB07F8C00,__READ_WRITE ,__ricfg0g3_ocu0otdgate_bits);
__IO_REG16_BIT(RICFG0G3_OCU0OTD0GM,     									0xB07F8C02,__READ_WRITE ,__ricfg0g3_ocu0otdgm_bits);
__IO_REG16_BIT(RICFG0G3_OCU0OTD1GATE,     								0xB07F8C04,__READ_WRITE ,__ricfg0g3_ocu0otdgate_bits);
__IO_REG16_BIT(RICFG0G3_OCU0OTD1GM,     									0xB07F8C06,__READ_WRITE ,__ricfg0g3_ocu0otdgm_bits);
__IO_REG16_BIT(RICFG0G3_OCU1CMP0EXT,     									0xB07F8C20,__READ_WRITE ,__ricfg0g3_ocu0otdgm_bits);
__IO_REG16_BIT(RICFG0G3_OCU1FRTSEL,     									0xB07F8C22,__READ_WRITE ,__ricfg0g3_ocu0otdgm_bits);
__IO_REG16_BIT(RICFG0G3_OCU1OTD0GATE,     								0xB07F8C24,__READ_WRITE ,__ricfg0g3_ocu0otdgate_bits);
__IO_REG16_BIT(RICFG0G3_OCU1OTD0GM,     									0xB07F8C26,__READ_WRITE ,__ricfg0g3_ocu0otdgm_bits);
__IO_REG16_BIT(RICFG0G3_OCU1OTD1GATE,     								0xB07F8C28,__READ_WRITE ,__ricfg0g3_ocu0otdgate_bits);
__IO_REG16_BIT(RICFG0G3_OCU1OTD1GM,     									0xB07F8C2A,__READ_WRITE ,__ricfg0g3_ocu0otdgm_bits);

/***************************************************************************
 **
 ** RICFG0G5
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G5_USART0SCKI,     									0xB07F9400,__READ_WRITE ,__ricfg0g5_usart0scki_bits);
__IO_REG16_BIT(RICFG0G5_USART0SIN,     										0xB07F9402,__READ_WRITE ,__ricfg0g5_usart0scki_bits);

/***************************************************************************
 **
 ** RICFG0G7
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G7_PPG0PPGAGATE,     								0xB07F9C00,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG0PPGAGM,     									0xB07F9C02,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG0PPGBGATE,     								0xB07F9C04,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG0PPGBGM,     									0xB07F9C06,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG1PPGAGATE,     								0xB07F9C20,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG1PPGAGM,     									0xB07F9C22,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG1PPGBGATE,     								0xB07F9C24,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG1PPGBGM,     									0xB07F9C26,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG2PPGAGATE,     								0xB07F9C40,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG2PPGAGM,     									0xB07F9C42,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG2PPGBGATE,     								0xB07F9C44,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG2PPGBGM,     									0xB07F9C46,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG3PPGAGATE,     								0xB07F9C60,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG3PPGAGM,     									0xB07F9C62,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG3PPGBGATE,     								0xB07F9C64,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG3PPGBGM,     									0xB07F9C66,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG4PPGAGATE,     								0xB07F9C80,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG4PPGAGM,     									0xB07F9C82,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG4PPGBGATE,     								0xB07F9C84,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG4PPGBGM,     									0xB07F9C86,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG5PPGAGATE,     								0xB07F9CA0,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG5PPGAGM,     									0xB07F9CA2,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG5PPGBGATE,     								0xB07F9CA4,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG5PPGBGM,     									0xB07F9CA6,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG6PPGAGATE,     								0xB07F9CC0,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG6PPGAGM,     									0xB07F9CC2,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG6PPGBGATE,     								0xB07F9CC4,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG6PPGBGM,     									0xB07F9CC6,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG7PPGAGATE,     								0xB07F9CE0,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG7PPGAGM,     									0xB07F9CE2,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG7PPGBGATE,     								0xB07F9CE4,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG7PPGBGM,     									0xB07F9CE6,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG8PPGAGATE,     								0xB07F9D00,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG8PPGAGM,     									0xB07F9D02,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG8PPGBGATE,     								0xB07F9D04,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG8PPGBGM,     									0xB07F9D06,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG9PPGAGATE,     								0xB07F9D20,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG9PPGAGM,     									0xB07F9D22,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG9PPGBGATE,     								0xB07F9D24,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG9PPGBGM,     									0xB07F9D26,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG10PPGAGATE,     								0xB07F9D40,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG10PPGAGM,     									0xB07F9D42,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG10PPGBGATE,     								0xB07F9D44,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG10PPGBGM,     									0xB07F9D46,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG11PPGAGATE,     								0xB07F9D60,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG11PPGAGM,     									0xB07F9D62,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG11PPGBGATE,     								0xB07F9D64,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG11PPGBGM,     									0xB07F9D66,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG12PPGAGATE,     								0xB07F9D80,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG12PPGAGM,     									0xB07F9D82,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG12PPGBGATE,     								0xB07F9D84,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG12PPGBGM,     									0xB07F9D86,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG13PPGAGATE,     								0xB07F9DA0,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG13PPGAGM,     									0xB07F9DA2,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG13PPGBGATE,     								0xB07F9DA4,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG13PPGBGM,     									0xB07F9DA6,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG14PPGAGATE,     								0xB07F9DC0,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG14PPGAGM,     									0xB07F9DC2,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG14PPGBGATE,     								0xB07F9DC4,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG14PPGBGM,     									0xB07F9DC6,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG15PPGAGATE,     								0xB07F9DE0,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG15PPGAGM,     									0xB07F9DE2,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);
__IO_REG16_BIT(RICFG0G7_PPG15PPGBGATE,     								0xB07F9DE4,__READ_WRITE ,__ricfg0g7_ppgppgagate_bits);
__IO_REG16_BIT(RICFG0G7_PPG15PPGBGM,     									0xB07F9DE6,__READ_WRITE ,__ricfg0g7_ppgppgagm_bits);

/***************************************************************************
 **
 ** RICFG0G9
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG0G9_PPGGRP0ETRG0,     								0xB07FA400,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP0ETRG1,     								0xB07FA402,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP0ETRG2,     								0xB07FA404,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP0ETRG3,     								0xB07FA406,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP0RLTTRG1,     							0xB07FA40A,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP1ETRG0,     								0xB07FA420,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP1ETRG1,     								0xB07FA422,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP1ETRG2,     								0xB07FA424,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP1ETRG3,     								0xB07FA426,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP1RLTTRG1,     							0xB07FA42A,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP2ETRG0,     								0xB07FA440,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP2ETRG1,     								0xB07FA442,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP2ETRG2,     								0xB07FA444,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP2ETRG3,     								0xB07FA446,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP2RLTTRG1,     							0xB07FA44A,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP3ETRG0,     								0xB07FA460,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP3ETRG1,     								0xB07FA462,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP3ETRG2,     								0xB07FA464,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP3ETRG3,     								0xB07FA466,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG0G9_PPGGRP3RLTTRG1,     							0xB07FA46A,__READ_WRITE ,__ricfg0g9_ppggrpetrg_bits);

/***************************************************************************
 **
 ** RICFG1G1
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G1_CAN0RX,     											0xB08F8400,__READ_WRITE ,__ricfg1g1_canrx_bits);
__IO_REG16_BIT(RICFG1G1_CAN1RX,     											0xB08F8420,__READ_WRITE ,__ricfg1g1_canrx_bits);
__IO_REG16_BIT(RICFG1G1_CAN2RX,     											0xB08F8440,__READ_WRITE ,__ricfg1g1_canrx_bits);

/***************************************************************************
 **
 ** RICFG1G3
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G3_FRT16TEXT,     										0xB08F8C00,__READ_WRITE ,__ricfg1g3_frttext_bits);
__IO_REG16_BIT(RICFG1G3_FRT17TEXT,     										0xB08F8C20,__READ_WRITE ,__ricfg1g3_frttext_bits);
__IO_REG16_BIT(RICFG1G3_FRT18TEXT,     										0xB08F8C40,__READ_WRITE ,__ricfg1g3_frttext_bits);
__IO_REG16_BIT(RICFG1G3_FRT19TEXT,     										0xB08F8C60,__READ_WRITE ,__ricfg1g3_frttext_bits);

/***************************************************************************
 **
 ** RICFG1G4
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G4_ICU18IN0,     										0xB08F9040,__READ_WRITE ,__ricfg1g4_icuin_bits);
__IO_REG16_BIT(RICFG1G4_ICU18IN1,     										0xB08F9042,__READ_WRITE ,__ricfg1g4_icuin_bits);
__IO_REG16_BIT(RICFG1G4_ICU18FRTSEL,     									0xB08F9044,__READ_WRITE ,__ricfg1g4_icufrtsel_bits);
__IO_REG16_BIT(RICFG1G4_ICU19IN0,     										0xB08F9060,__READ_WRITE ,__ricfg1g4_icuin_bits);
__IO_REG16_BIT(RICFG1G4_ICU19IN1,     										0xB08F9062,__READ_WRITE ,__ricfg1g4_icuin_bits);
__IO_REG16_BIT(RICFG1G4_ICU19FRTSEL,     									0xB08F9064,__READ_WRITE ,__ricfg1g4_icufrtsel_bits);

/***************************************************************************
 **
 ** RICFG1G5
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G5_OCU16OTD0GATE,     								0xB08F9400,__READ_WRITE ,__ricfg1g5_ocuotdgate_bits);
__IO_REG16_BIT(RICFG1G5_OCU16OTD0GM,     									0xB08F9402,__READ_WRITE ,__ricfg1g5_ocuotdgm_bits);
__IO_REG16_BIT(RICFG1G5_OCU16OTD1GATE,     								0xB08F9404,__READ_WRITE ,__ricfg1g5_ocuotdgate_bits);
__IO_REG16_BIT(RICFG1G5_OCU16OTD1GM,     									0xB08F9406,__READ_WRITE ,__ricfg1g5_ocuotdgm_bits);
__IO_REG16_BIT(RICFG1G5_OCU17CMP0EXT,     								0xB08F9420,__READ_WRITE ,__ricfg1g5_ocuotdgm_bits);
__IO_REG16_BIT(RICFG1G5_OCU17FRTSEL,     									0xB08F9422,__READ_WRITE ,__ricfg1g5_ocuotdgm_bits);
__IO_REG16_BIT(RICFG1G5_OCU17OTD0GATE,     								0xB08F9424,__READ_WRITE ,__ricfg1g5_ocuotdgate_bits);
__IO_REG16_BIT(RICFG1G5_OCU17OTD0GM,     									0xB08F9426,__READ_WRITE ,__ricfg1g5_ocuotdgm_bits);
__IO_REG16_BIT(RICFG1G5_OCU17OTD1GATE,     								0xB08F9428,__READ_WRITE ,__ricfg1g5_ocuotdgate_bits);
__IO_REG16_BIT(RICFG1G5_OCU17OTD1GM,     									0xB08F942A,__READ_WRITE ,__ricfg1g5_ocuotdgm_bits);

/***************************************************************************
 **
 ** RICFG1G7
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G7_USART6SCKI,     									0xB08F9C00,__READ_WRITE ,__ricfg1g7_usart6scki_bits);
__IO_REG16_BIT(RICFG1G7_USART6SIN,     										0xB08F9C02,__READ_WRITE ,__ricfg1g7_usart6scki_bits);

/***************************************************************************
 **
 ** RICFG1G9
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G9_PPG64PPGAGATE,     								0xB08FA400,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG64PPGAGM,     									0xB08FA402,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG64PPGBGATE,     								0xB08FA404,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG64PPGBGM,     									0xB08FA406,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG65PPGAGATE,     								0xB08FA420,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG65PPGAGM,     									0xB08FA422,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG65PPGBGATE,     								0xB08FA424,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG65PPGBGM,     									0xB08FA426,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG66PPGAGATE,     								0xB08FA440,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG66PPGAGM,     									0xB08FA442,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG66PPGBGATE,     								0xB08FA444,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG66PPGBGM,     									0xB08FA446,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG67PPGAGATE,     								0xB08FA460,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG67PPGAGM,     									0xB08FA462,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG67PPGBGATE,     								0xB08FA464,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG67PPGBGM,     									0xB08FA466,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG68PPGAGATE,     								0xB08FA480,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG68PPGAGM,     									0xB08FA482,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG68PPGBGATE,     								0xB08FA484,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG68PPGBGM,     									0xB08FA486,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG69PPGAGATE,     								0xB08FA4A0,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG69PPGAGM,     									0xB08FA4A2,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG69PPGBGATE,     								0xB08FA4A4,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG69PPGBGM,     									0xB08FA4A6,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG70PPGAGATE,     								0xB08FA4C0,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG70PPGAGM,     									0xB08FA4C2,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG70PPGBGATE,     								0xB08FA4C4,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG70PPGBGM,     									0xB08FA4C6,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG71PPGAGATE,     								0xB08FA4E0,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG71PPGAGM,     									0xB08FA4E2,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);
__IO_REG16_BIT(RICFG1G9_PPG71PPGBGATE,     								0xB08FA4E4,__READ_WRITE ,__ricfg1g9_ppgppgagate_bits);
__IO_REG16_BIT(RICFG1G9_PPG71PPGBGM,     									0xB08FA4E6,__READ_WRITE ,__ricfg1g9_ppgppgagm_bits);

/***************************************************************************
 **
 ** RICFG1G11
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG1G11_PPGGRP16ETRG0,     							0xB08FAC00,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP16ETRG1,     							0xB08FAC02,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP16ETRG2,     							0xB08FAC04,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP16ETRG3,     							0xB08FAC06,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP16RLTTRG1,     						0xB08FAC08,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP17ETRG0,     							0xB08FAC20,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP17ETRG1,     							0xB08FAC22,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP17ETRG2,     							0xB08FAC24,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP17ETRG3,     							0xB08FAC26,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);
__IO_REG16_BIT(RICFG1G11_PPGGRP17RLTTRG1,     						0xB08FAC28,__READ_WRITE ,__ricfg1g11_ppggrpetrg_bits);

/***************************************************************************
 **
 ** RICFG3G2
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG3G2_RLT0TIN,     											0xB0AF8800,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT1TIN,     											0xB0AF8820,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT2TIN,     											0xB0AF8840,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT3TIN,     											0xB0AF8860,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT4TIN,     											0xB0AF8880,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT5TIN,     											0xB0AF88A0,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT6TIN,     											0xB0AF88C0,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT7TIN,     											0xB0AF88E0,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT8TIN,     											0xB0AF8900,__READ_WRITE ,__ricfg3g2_rlttin_bits);
__IO_REG16_BIT(RICFG3G2_RLT9TIN,     											0xB0AF8920,__READ_WRITE ,__ricfg3g2_rlttin_bits);

/***************************************************************************
 **
 ** RICFG3G4
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG3G4_UDC0AIN0,     										0xB0AF9000,__READ_WRITE ,__ricfg3g4_udc0in_bits);
__IO_REG16_BIT(RICFG3G4_UDC0AIN1,     										0xB0AF9004,__READ_WRITE ,__ricfg3g4_udc0in_bits);
__IO_REG16_BIT(RICFG3G4_UDC0BIN0,     										0xB0AF9008,__READ_WRITE ,__ricfg3g4_udc0in_bits);
__IO_REG16_BIT(RICFG3G4_UDC0BIN1,     										0xB0AF900C,__READ_WRITE ,__ricfg3g4_udc0in_bits);
__IO_REG16_BIT(RICFG3G4_UDC0ZIN0,     										0xB0AF9010,__READ_WRITE ,__ricfg3g4_udc0zin_bits);
__IO_REG16_BIT(RICFG3G4_UDC0ZIN1,     										0xB0AF9014,__READ_WRITE ,__ricfg3g4_udc0zin_bits);

/***************************************************************************
 **
 ** MPUXGFX
 **
 ***************************************************************************/
__IO_REG32_BIT(MPUXGFX_CTRL0,     												0xB0B00000,__READ_WRITE ,__mpuxgfx_ctrl0_bits);
__IO_REG32_BIT(MPUXGFX_NMIEN,     												0xB0B00004,__READ_WRITE ,__mpuxgfx_nmien_bits);
__IO_REG32_BIT(MPUXGFX_WERRC,     												0xB0B00008,__READ_WRITE ,__mpuxgfx_werrc_bits);
__IO_REG32(		 MPUXGFX_WERRA,     												0xB0B0000C,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_RERRC,     												0xB0B00010,__READ_WRITE ,__mpuxgfx_werrc_bits);
__IO_REG32(		 MPUXGFX_RERRA,     												0xB0B00014,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL1,     												0xB0B00018,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR1,     												0xB0B0001C,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR1,     												0xB0B00020,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL2,     												0xB0B00024,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR2,     												0xB0B00028,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR2,     												0xB0B0002C,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL3,     												0xB0B00030,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR3,     												0xB0B00034,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR3,     												0xB0B00038,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL4,     												0xB0B0003C,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR4,     												0xB0B00040,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR4,     												0xB0B00044,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL5,     												0xB0B00048,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR5,     												0xB0B0004C,__READ_WRITE );
__IO_REG32( 	 MPUXGFX_EADDR5,     												0xB0B00050,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL6,     												0xB0B00054,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR6,     												0xB0B00058,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR6,     												0xB0B0005C,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL7,     												0xB0B00060,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR7,     												0xB0B00064,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR7,     												0xB0B00068,__READ_WRITE );
__IO_REG32_BIT(MPUXGFX_CTRL8,     												0xB0B0006C,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXGFX_SADDR8,     												0xB0B00070,__READ_WRITE );
__IO_REG32(		 MPUXGFX_EADDR8,     												0xB0B00074,__READ_WRITE );
__IO_REG32(		 MPUXGFX_UNLOCK,     												0xB0B00078,__READ_WRITE );
__IO_REG32(		 MPUXGFX_MID,     													0xB0B0007C,__READ_WRITE );

/***************************************************************************
 **
 ** ERCFG0
 **
 ***************************************************************************/
__IO_REG32(		 ERCFG0_UNLOCKR,     												0xB0B08400,__READ_WRITE );
__IO_REG32_BIT(ERCFG0_CSR,     														0xB0B08404,__READ_WRITE ,__ercfg0_csr_bits);
__IO_REG32(		 ERCFG0_EAN,     														0xB0B08408,__READ_WRITE );
__IO_REG32(		 ERCFG0_ERRMSKR0,     											0xB0B0840C,__READ_WRITE );
__IO_REG32_BIT(ERCFG0_ERRMSKR1,     											0xB0B08410,__READ_WRITE ,__ercfg0_errmskr1_bits);
__IO_REG32_BIT(ERCFG0_ECCEN,     													0xB0B08414,__READ_WRITE ,__ercfg0_eccen_bits);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_MCTRL,      0xB0B38000,__READ_WRITE ,__hsspin_mctrl_bits);
__IO_REG32_BIT(SPI0_PCC0,       0xB0B38004,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI0_PCC1,       0xB0B38008,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI0_PCC2,       0xB0B3800C,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI0_PCC3,       0xB0B38010,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI0_TXF,        0xB0B38014,__READ_WRITE ,__hsspin_txf_bits);
__IO_REG32_BIT(SPI0_TXE,        0xB0B38018,__READ_WRITE ,__hsspin_txe_bits);
__IO_REG32_BIT(SPI0_TXC,        0xB0B3801C,__READ_WRITE ,__hsspin_txc_bits);
__IO_REG32_BIT(SPI0_RXF,        0xB0B38020,__READ       ,__hsspin_rxf_bits);
__IO_REG32_BIT(SPI0_RXE,        0xB0B38024,__READ_WRITE ,__hsspin_rxe_bits);
__IO_REG32_BIT(SPI0_RXC,        0xB0B38028,__READ_WRITE ,__hsspin_rxc_bits);
__IO_REG32_BIT(SPI0_FAULTF,     0xB0B3802C,__READ       ,__hsspin_faultf_bits);
__IO_REG32_BIT(SPI0_FAULTC,     0xB0B38030,__READ_WRITE ,__hsspin_faultc_bits);
__IO_REG8_BIT( SPI0_DMCFG,      0xB0B38034,__READ_WRITE ,__hsspin_dmcfg_bits);
__IO_REG8_BIT( SPI0_DMDMAEN,    0xB0B38035,__READ_WRITE ,__hsspin_dmdmaen_bits);
__IO_REG8_BIT( SPI0_DMSTART,    0xB0B38038,__READ_WRITE ,__hsspin_dmstart_bits);
__IO_REG8_BIT( SPI0_DMSTOP,     0xB0B38039,__READ_WRITE ,__hsspin_dmstop_bits);
__IO_REG8_BIT( SPI0_DMPSEL,     0xB0B3803A,__READ_WRITE ,__hsspin_dmpsel_bits);
__IO_REG8_BIT( SPI0_DMTRP,      0xB0B3803B,__READ_WRITE ,__hsspin_dmtrp_bits);
__IO_REG16(    SPI0_DMBCC,      0xB0B3803C,__READ_WRITE );
__IO_REG16(    SPI0_DMBCS,      0xB0B3803E,__READ       );
__IO_REG32_BIT(SPI0_DMSTATUS,   0xB0B38040,__READ       ,__hsspin_dmstatus_bits);
__IO_REG8_BIT( SPI0_TXBITCNT,   0xB0B38044,__READ       ,__hsspin_txbitcnt_bits);
__IO_REG8_BIT( SPI0_RXBITCNT,   0xB0B38045,__READ       ,__hsspin_rxbitcnt_bits);
__IO_REG32(    SPI0_RXSHIFT,    0xB0B38048,__READ       );
__IO_REG32_BIT(SPI0_FIFOCFG,    0xB0B3804C,__READ_WRITE ,__hsspin_fifocfg_bits);
__IO_REG32(    SPI0_TXFIFO0,    0xB0B38050,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO1,    0xB0B38054,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO2,    0xB0B38058,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO3,    0xB0B3805C,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO4,    0xB0B38060,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO5,    0xB0B38064,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO6,    0xB0B38068,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO7,    0xB0B3806C,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO8,    0xB0B38070,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO9,    0xB0B38074,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO10,   0xB0B38078,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO11,   0xB0B3807C,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO12,   0xB0B38080,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO13,   0xB0B38084,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO14,   0xB0B38088,__READ_WRITE );
__IO_REG32(    SPI0_TXFIFO15,   0xB0B3808C,__READ_WRITE );
__IO_REG32(    SPI0_RXFIFO0,    0xB0B38090,__READ       );
__IO_REG32(    SPI0_RXFIFO1,    0xB0B38094,__READ       );
__IO_REG32(    SPI0_RXFIFO2,    0xB0B38098,__READ       );
__IO_REG32(    SPI0_RXFIFO3,    0xB0B3809C,__READ       );
__IO_REG32(    SPI0_RXFIFO4,    0xB0B380A0,__READ       );
__IO_REG32(    SPI0_RXFIFO5,    0xB0B380A4,__READ       );
__IO_REG32(    SPI0_RXFIFO6,    0xB0B380A8,__READ       );
__IO_REG32(    SPI0_RXFIFO7,    0xB0B380AC,__READ       );
__IO_REG32(    SPI0_RXFIFO8,    0xB0B380B0,__READ       );
__IO_REG32(    SPI0_RXFIFO9,    0xB0B380B4,__READ       );
__IO_REG32(    SPI0_RXFIFO10,   0xB0B380B8,__READ       );
__IO_REG32(    SPI0_RXFIFO11,   0xB0B380BC,__READ       );
__IO_REG32(    SPI0_RXFIFO12,   0xB0B380C0,__READ       );
__IO_REG32(    SPI0_RXFIFO13,   0xB0B380C4,__READ       );
__IO_REG32(    SPI0_RXFIFO14,   0xB0B380C8,__READ       );
__IO_REG32(    SPI0_RXFIFO15,   0xB0B380CC,__READ       );
__IO_REG32(    SPI0_MID,        0xB0B380FC,__READ       );

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_MCTRL,      0xB0B38400,__READ_WRITE ,__hsspin_mctrl_bits);
__IO_REG32_BIT(SPI1_PCC0,       0xB0B38404,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI1_PCC1,       0xB0B38408,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI1_PCC2,       0xB0B3840C,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI1_PCC3,       0xB0B38410,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI1_TXF,        0xB0B38414,__READ_WRITE ,__hsspin_txf_bits);
__IO_REG32_BIT(SPI1_TXE,        0xB0B38418,__READ_WRITE ,__hsspin_txe_bits);
__IO_REG32_BIT(SPI1_TXC,        0xB0B3841C,__READ_WRITE ,__hsspin_txc_bits);
__IO_REG32_BIT(SPI1_RXF,        0xB0B38420,__READ       ,__hsspin_rxf_bits);
__IO_REG32_BIT(SPI1_RXE,        0xB0B38424,__READ_WRITE ,__hsspin_rxe_bits);
__IO_REG32_BIT(SPI1_RXC,        0xB0B38428,__READ_WRITE ,__hsspin_rxc_bits);
__IO_REG32_BIT(SPI1_FAULTF,     0xB0B3842C,__READ       ,__hsspin_faultf_bits);
__IO_REG32_BIT(SPI1_FAULTC,     0xB0B38430,__READ_WRITE ,__hsspin_faultc_bits);
__IO_REG8_BIT( SPI1_DMCFG,      0xB0B38434,__READ_WRITE ,__hsspin_dmcfg_bits);
__IO_REG8_BIT( SPI1_DMDMAEN,    0xB0B38435,__READ_WRITE ,__hsspin_dmdmaen_bits);
__IO_REG8_BIT( SPI1_DMSTART,    0xB0B38438,__READ_WRITE ,__hsspin_dmstart_bits);
__IO_REG8_BIT( SPI1_DMSTOP,     0xB0B38439,__READ_WRITE ,__hsspin_dmstop_bits);
__IO_REG8_BIT( SPI1_DMPSEL,     0xB0B3843A,__READ_WRITE ,__hsspin_dmpsel_bits);
__IO_REG8_BIT( SPI1_DMTRP,      0xB0B3843B,__READ_WRITE ,__hsspin_dmtrp_bits);
__IO_REG16(    SPI1_DMBCC,      0xB0B3843C,__READ_WRITE );
__IO_REG16(    SPI1_DMBCS,      0xB0B3843E,__READ       );
__IO_REG32_BIT(SPI1_DMSTATUS,   0xB0B38440,__READ       ,__hsspin_dmstatus_bits);
__IO_REG8_BIT( SPI1_TXBITCNT,   0xB0B38444,__READ       ,__hsspin_txbitcnt_bits);
__IO_REG8_BIT( SPI1_RXBITCNT,   0xB0B38445,__READ       ,__hsspin_rxbitcnt_bits);
__IO_REG32(    SPI1_RXSHIFT,    0xB0B38448,__READ       );
__IO_REG32_BIT(SPI1_FIFOCFG,    0xB0B3844C,__READ_WRITE ,__hsspin_fifocfg_bits);
__IO_REG32(    SPI1_TXFIFO0,    0xB0B38450,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO1,    0xB0B38454,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO2,    0xB0B38458,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO3,    0xB0B3845C,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO4,    0xB0B38460,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO5,    0xB0B38464,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO6,    0xB0B38468,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO7,    0xB0B3846C,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO8,    0xB0B38470,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO9,    0xB0B38474,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO10,   0xB0B38478,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO11,   0xB0B3847C,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO12,   0xB0B38480,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO13,   0xB0B38484,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO14,   0xB0B38488,__READ_WRITE );
__IO_REG32(    SPI1_TXFIFO15,   0xB0B3848C,__READ_WRITE );
__IO_REG32(    SPI1_RXFIFO0,    0xB0B38490,__READ       );
__IO_REG32(    SPI1_RXFIFO1,    0xB0B38494,__READ       );
__IO_REG32(    SPI1_RXFIFO2,    0xB0B38498,__READ       );
__IO_REG32(    SPI1_RXFIFO3,    0xB0B3849C,__READ       );
__IO_REG32(    SPI1_RXFIFO4,    0xB0B384A0,__READ       );
__IO_REG32(    SPI1_RXFIFO5,    0xB0B384A4,__READ       );
__IO_REG32(    SPI1_RXFIFO6,    0xB0B384A8,__READ       );
__IO_REG32(    SPI1_RXFIFO7,    0xB0B384AC,__READ       );
__IO_REG32(    SPI1_RXFIFO8,    0xB0B384B0,__READ       );
__IO_REG32(    SPI1_RXFIFO9,    0xB0B384B4,__READ       );
__IO_REG32(    SPI1_RXFIFO10,   0xB0B384B8,__READ       );
__IO_REG32(    SPI1_RXFIFO11,   0xB0B384BC,__READ       );
__IO_REG32(    SPI1_RXFIFO12,   0xB0B384C0,__READ       );
__IO_REG32(    SPI1_RXFIFO13,   0xB0B384C4,__READ       );
__IO_REG32(    SPI1_RXFIFO14,   0xB0B384C8,__READ       );
__IO_REG32(    SPI1_RXFIFO15,   0xB0B384CC,__READ       );
__IO_REG32(    SPI1_MID,        0xB0B384FC,__READ       );

/***************************************************************************
 **
 ** SPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI2_MCTRL,      0xB0B38800,__READ_WRITE ,__hsspin_mctrl_bits);
__IO_REG32_BIT(SPI2_PCC0,       0xB0B38804,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI2_PCC1,       0xB0B38808,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI2_PCC2,       0xB0B3880C,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI2_PCC3,       0xB0B38810,__READ_WRITE ,__hsspin_pcc_bits);
__IO_REG32_BIT(SPI2_TXF,        0xB0B38814,__READ_WRITE ,__hsspin_txf_bits);
__IO_REG32_BIT(SPI2_TXE,        0xB0B38818,__READ_WRITE ,__hsspin_txe_bits);
__IO_REG32_BIT(SPI2_TXC,        0xB0B3881C,__READ_WRITE ,__hsspin_txc_bits);
__IO_REG32_BIT(SPI2_RXF,        0xB0B38820,__READ       ,__hsspin_rxf_bits);
__IO_REG32_BIT(SPI2_RXE,        0xB0B38824,__READ_WRITE ,__hsspin_rxe_bits);
__IO_REG32_BIT(SPI2_RXC,        0xB0B38828,__READ_WRITE ,__hsspin_rxc_bits);
__IO_REG32_BIT(SPI2_FAULTF,     0xB0B3882C,__READ       ,__hsspin_faultf_bits);
__IO_REG32_BIT(SPI2_FAULTC,     0xB0B38830,__READ_WRITE ,__hsspin_faultc_bits);
__IO_REG8_BIT( SPI2_DMCFG,      0xB0B38834,__READ_WRITE ,__hsspin_dmcfg_bits);
__IO_REG8_BIT( SPI2_DMDMAEN,    0xB0B38835,__READ_WRITE ,__hsspin_dmdmaen_bits);
__IO_REG8_BIT( SPI2_DMSTART,    0xB0B38838,__READ_WRITE ,__hsspin_dmstart_bits);
__IO_REG8_BIT( SPI2_DMSTOP,     0xB0B38839,__READ_WRITE ,__hsspin_dmstop_bits);
__IO_REG8_BIT( SPI2_DMPSEL,     0xB0B3883A,__READ_WRITE ,__hsspin_dmpsel_bits);
__IO_REG8_BIT( SPI2_DMTRP,      0xB0B3883B,__READ_WRITE ,__hsspin_dmtrp_bits);
__IO_REG16(    SPI2_DMBCC,      0xB0B3883C,__READ_WRITE );
__IO_REG16(    SPI2_DMBCS,      0xB0B3883E,__READ       );
__IO_REG32_BIT(SPI2_DMSTATUS,   0xB0B38840,__READ       ,__hsspin_dmstatus_bits);
__IO_REG8_BIT( SPI2_TXBITCNT,   0xB0B38844,__READ       ,__hsspin_txbitcnt_bits);
__IO_REG8_BIT( SPI2_RXBITCNT,   0xB0B38845,__READ       ,__hsspin_rxbitcnt_bits);
__IO_REG32(    SPI2_RXSHIFT,    0xB0B38848,__READ       );
__IO_REG32_BIT(SPI2_FIFOCFG,    0xB0B3884C,__READ_WRITE ,__hsspin_fifocfg_bits);
__IO_REG32(    SPI2_TXFIFO0,    0xB0B38850,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO1,    0xB0B38854,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO2,    0xB0B38858,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO3,    0xB0B3885C,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO4,    0xB0B38860,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO5,    0xB0B38864,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO6,    0xB0B38868,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO7,    0xB0B3886C,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO8,    0xB0B38870,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO9,    0xB0B38874,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO10,   0xB0B38878,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO11,   0xB0B3887C,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO12,   0xB0B38880,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO13,   0xB0B38884,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO14,   0xB0B38888,__READ_WRITE );
__IO_REG32(    SPI2_TXFIFO15,   0xB0B3888C,__READ_WRITE );
__IO_REG32(    SPI2_RXFIFO0,    0xB0B38890,__READ       );
__IO_REG32(    SPI2_RXFIFO1,    0xB0B38894,__READ       );
__IO_REG32(    SPI2_RXFIFO2,    0xB0B38898,__READ       );
__IO_REG32(    SPI2_RXFIFO3,    0xB0B3889C,__READ       );
__IO_REG32(    SPI2_RXFIFO4,    0xB0B388A0,__READ       );
__IO_REG32(    SPI2_RXFIFO5,    0xB0B388A4,__READ       );
__IO_REG32(    SPI2_RXFIFO6,    0xB0B388A8,__READ       );
__IO_REG32(    SPI2_RXFIFO7,    0xB0B388AC,__READ       );
__IO_REG32(    SPI2_RXFIFO8,    0xB0B388B0,__READ       );
__IO_REG32(    SPI2_RXFIFO9,    0xB0B388B4,__READ       );
__IO_REG32(    SPI2_RXFIFO10,   0xB0B388B8,__READ       );
__IO_REG32(    SPI2_RXFIFO11,   0xB0B388BC,__READ       );
__IO_REG32(    SPI2_RXFIFO12,   0xB0B388C0,__READ       );
__IO_REG32(    SPI2_RXFIFO13,   0xB0B388C4,__READ       );
__IO_REG32(    SPI2_RXFIFO14,   0xB0B388C8,__READ       );
__IO_REG32(    SPI2_RXFIFO15,   0xB0B388CC,__READ       );
__IO_REG32(    SPI2_MID,        0xB0B388FC,__READ       );

/***************************************************************************
 **
 ** ARH0
 **
 ***************************************************************************/
__IO_REG32_BIT(ARH0_RHCTRL,     0xB0B40000,__READ_WRITE ,__arh0_rhctrl_bits);
__IO_REG32_BIT(ARH0_CHCTRL0,    0xB0B40004,__READ_WRITE ,__arh0_chctrl0_bits);
__IO_REG32_BIT(ARH0_CHSTAT0,    0xB0B40008,__READ_WRITE ,__arh0_chstat0_bits);
__IO_REG32_BIT(ARH0_CHWDGCTL0,  0xB0B4000C,__READ_WRITE ,__arh0_chwdgctl0_bits);
__IO_REG32_BIT(ARH0_CHWDGCNT0,  0xB0B40010,__READ_WRITE ,__arh0_chwdgcnt0_bits);
__IO_REG32_BIT(ARH0_CHCTRL1,    0xB0B40014,__READ_WRITE ,__arh0_chctrl1_bits);
__IO_REG32_BIT(ARH0_CHSTAT1,    0xB0B40018,__READ_WRITE ,__arh0_chstat1_bits);
__IO_REG32_BIT(ARH0_CHWDGCTL1,  0xB0B4001C,__READ_WRITE ,__arh0_chwdgctl1_bits);
__IO_REG32_BIT(ARH0_CHWDGCNT1,  0xB0B40020,__READ_WRITE ,__arh0_chwdgcnt1_bits);
__IO_REG32_BIT(ARH0_TBCTRL0,  	0xB0B40024,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL1,  	0xB0B40028,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL2,  	0xB0B4002C,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL3,  	0xB0B40030,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL4,  	0xB0B40034,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL5,  	0xB0B40038,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL6,  	0xB0B4003C,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL7,  	0xB0B40040,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL8,  	0xB0B40044,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL9,  	0xB0B40048,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL10,  	0xB0B4004C,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL11,  	0xB0B40050,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL12,  	0xB0B40054,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL13,  	0xB0B40058,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL14,  	0xB0B4005C,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG32_BIT(ARH0_TBCTRL15,  	0xB0B40060,__READ_WRITE ,__arh0_tbctrl_bits);
__IO_REG8_BIT( ARH0_TBIDX1,  		0xB0B40064,__READ_WRITE ,__arh0_tbidx_bits);
__IO_REG8_BIT( ARH0_TBIDX0,  		0xB0B40065,__READ_WRITE ,__arh0_tbidx_bits);
__IO_REG16(		 ARH0_TBIRQ,  		0xB0B40066,__READ_WRITE );
__IO_REG8(		 ARH0_TFIDX1,  		0xB0B40068,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL1,  	0xB0B40069,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX0,  		0xB0B4006A,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL0,  	0xB0B4006B,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX3,  		0xB0B4006C,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL3,  	0xB0B4006D,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX2,  		0xB0B4006E,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL2,  	0xB0B4006F,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX5,  		0xB0B40070,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL5,  	0xB0B40071,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX4,  		0xB0B40072,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL4,  	0xB0B40073,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX7,  		0xB0B40074,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL7,  	0xB0B40075,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX6,  		0xB0B40076,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL6,  	0xB0B40077,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX9,  		0xB0B40078,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL9,  	0xB0B40079,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX8,  		0xB0B4007A,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL8,  	0xB0B4007B,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX11, 		0xB0B4007C,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL11,  	0xB0B4007D,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX10, 		0xB0B4007E,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL10,  	0xB0B4007F,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX13, 		0xB0B40080,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL13,  	0xB0B40081,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX12, 		0xB0B40082,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL12,  	0xB0B40083,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX15, 		0xB0B40084,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL15,  	0xB0B40085,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG8(		 ARH0_TFIDX14, 		0xB0B40086,__READ_WRITE );
__IO_REG8_BIT( ARH0_TFCTRL14,  	0xB0B40087,__READ_WRITE ,__arh0_tfctrl_bits);
__IO_REG32_BIT(ARH0_TFADDR0,  	0xB0B40088,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR1,  	0xB0B4008C,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR2,  	0xB0B40090,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR3,  	0xB0B40094,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR4,  	0xB0B40098,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR5,  	0xB0B4009C,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR6,  	0xB0B400A0,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR7,  	0xB0B400A4,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR8,  	0xB0B400A8,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR9,  	0xB0B400AC,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR10,  	0xB0B400B0,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR11,  	0xB0B400B4,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR12,  	0xB0B400B8,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR13,  	0xB0B400BC,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR14,  	0xB0B400C0,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32_BIT(ARH0_TFADDR15,  	0xB0B400C4,__READ_WRITE ,__arh0_tfaddr_bits);
__IO_REG32(		 ARH0_TFDATA0,  	0xB0B400C8,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA1,  	0xB0B400CC,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA2,  	0xB0B400D0,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA3,  	0xB0B400D4,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA4,  	0xB0B400D8,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA5,  	0xB0B400DC,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA6,  	0xB0B400E0,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA7,  	0xB0B400E4,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA8,  	0xB0B400E8,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA9,  	0xB0B400EC,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA10,  	0xB0B400F0,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA11,  	0xB0B400F4,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA12,  	0xB0B400F8,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA13,  	0xB0B400FC,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA14,  	0xB0B40100,__READ_WRITE );
__IO_REG32(		 ARH0_TFDATA15,  	0xB0B40104,__READ_WRITE );
__IO_REG32_BIT(ARH0_EVCTRL,  		0xB0B40108,__READ_WRITE ,__arh0_evctrl_bits);
__IO_REG32_BIT(ARH0_EVIRQC,  		0xB0B4010C,__READ_WRITE ,__arh0_evirqc_bits);
__IO_REG32_BIT(ARH0_EVBUF0,  		0xB0B40110,__READ_WRITE ,__arh0_evbuf0_bits);
__IO_REG32(		 ARH0_EVBUF1,  		0xB0B40114,__READ_WRITE );
__IO_REG32_BIT(ARH0_APCFG00,  	0xB0B40118,__READ_WRITE ,__arh0_apcfg00_bits);
__IO_REG32_BIT(ARH0_APCFG01,  	0xB0B4011C,__READ_WRITE ,__arh0_apcfg01_bits);
__IO_REG32_BIT(ARH0_APCFG02,  	0xB0B40120,__READ_WRITE ,__arh0_apcfg02_bits);
__IO_REG32_BIT(ARH0_APCFG03,  	0xB0B40124,__READ_WRITE ,__arh0_apcfg03_bits);
__IO_REG32_BIT(ARH0_APCFG10,  	0xB0B40128,__READ_WRITE ,__arh0_apcfg10_bits);
__IO_REG32_BIT(ARH0_APCFG11,  	0xB0B4012C,__READ_WRITE ,__arh0_apcfg11_bits);
__IO_REG32_BIT(ARH0_APCFG12,  	0xB0B40130,__READ_WRITE ,__arh0_apcfg12_bits);
__IO_REG32_BIT(ARH0_APCFG13,  	0xB0B40134,__READ_WRITE ,__arh0_apcfg13_bits);
__IO_REG32_BIT(ARH0_TST,  			0xB0B40138,__READ_WRITE ,__arh0_tst_bits);
__IO_REG32(		 ARH0_UNLOCK,  		0xB0B4013C,__READ_WRITE );
__IO_REG32(		 ARH0_MID,  			0xB0B40140,__READ_WRITE );

/***************************************************************************
 **
 ** RICFG4G1
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG4G1_ETH0COL, 		0xB0BF8400,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0CRS, 		0xB0BF8402,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0MDIOI, 	0xB0BF8404,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RDX0, 	0xB0BF8406,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RDX1, 	0xB0BF8408,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RDX2, 	0xB0BF840A,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RDX3, 	0xB0BF840C,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RXCLK, 	0xB0BF840E,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RXDV, 	0xB0BF8410,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0RXER, 	0xB0BF8412,__READ_WRITE ,__ricfg4g1_eth0col_bits);
__IO_REG16_BIT(RICFG4G1_ETH0TXCLK, 	0xB0BF8414,__READ_WRITE ,__ricfg4g1_eth0col_bits);

/***************************************************************************
 **
 ** RICFG4G4
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG4G4_I2S0ECLK, 	0xB0BF9000,__READ_WRITE ,__ricfg4g4_i2seclk_bits);
__IO_REG16_BIT(RICFG4G4_I2S0SCKI, 	0xB0BF9004,__READ_WRITE ,__ricfg4g4_i2sscki_bits);
__IO_REG16_BIT(RICFG4G4_I2S0SDI, 		0xB0BF9008,__READ_WRITE ,__ricfg4g4_i2sscki_bits);
__IO_REG16_BIT(RICFG4G4_I2S0WSI, 		0xB0BF900C,__READ_WRITE ,__ricfg4g4_i2sscki_bits);
__IO_REG16_BIT(RICFG4G4_I2S1ECLK, 	0xB0BF9020,__READ_WRITE ,__ricfg4g4_i2seclk_bits);
__IO_REG16_BIT(RICFG4G4_I2S1SCKI, 	0xB0BF9024,__READ_WRITE ,__ricfg4g4_i2sscki_bits);
__IO_REG16_BIT(RICFG4G4_I2S1SDI, 		0xB0BF9028,__READ_WRITE ,__ricfg4g4_i2sscki_bits);
__IO_REG16_BIT(RICFG4G4_I2S1WSI, 		0xB0BF902C,__READ_WRITE ,__ricfg4g4_i2sscki_bits);

/***************************************************************************
 **
 ** RICFG4G7
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG4G7_SPI0CLKI, 	0xB0BF9C00,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI0DATA0I, 0xB0BF9C04,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI0DATA1I, 0xB0BF9C08,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI0DATA2I, 0xB0BF9C0C,__READ_WRITE ,__ricfg4g7_spidata2i_bits);
__IO_REG16_BIT(RICFG4G7_SPI0DATA3I, 0xB0BF9C10,__READ_WRITE ,__ricfg4g7_spidata2i_bits);
__IO_REG16_BIT(RICFG4G7_SPI0MSTART, 0xB0BF9C14,__READ_WRITE ,__ricfg4g7_spimstart_bits);
__IO_REG16_BIT(RICFG4G7_SPI0SSI, 		0xB0BF9C18,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI1CLKI, 	0xB0BF9C20,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI1DATA0I, 0xB0BF9C24,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI1DATA1I, 0xB0BF9C28,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI1DATA2I, 0xB0BF9C2C,__READ_WRITE ,__ricfg4g7_spidata2i_bits);
__IO_REG16_BIT(RICFG4G7_SPI1DATA3I, 0xB0BF9C30,__READ_WRITE ,__ricfg4g7_spidata2i_bits);
__IO_REG16_BIT(RICFG4G7_SPI1MSTART, 0xB0BF9C34,__READ_WRITE ,__ricfg4g7_spimstart_bits);
__IO_REG16_BIT(RICFG4G7_SPI1SSI, 		0xB0BF9C38,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI2CLKI, 	0xB0BF9C40,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI2DATA0I, 0xB0BF9C44,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI2DATA1I, 0xB0BF9C48,__READ_WRITE ,__ricfg4g7_spiclki_bits);
__IO_REG16_BIT(RICFG4G7_SPI2DATA2I, 0xB0BF9C4C,__READ_WRITE ,__ricfg4g7_spidata2i_bits);
__IO_REG16_BIT(RICFG4G7_SPI2DATA3I, 0xB0BF9C50,__READ_WRITE ,__ricfg4g7_spidata2i_bits);
__IO_REG16_BIT(RICFG4G7_SPI2MSTART, 0xB0BF9C54,__READ_WRITE ,__ricfg4g7_spimstart_bits);
__IO_REG16_BIT(RICFG4G7_SPI2SSI, 		0xB0BF9C58,__READ_WRITE ,__ricfg4g7_spiclki_bits);

/***************************************************************************
 **
 ** RICFG4G8
 **
 ***************************************************************************/
__IO_REG16_BIT(RICFG4G8_ARH0AIC0RCK,	0xB0BFA000,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC0RDA0, 0xB0BFA004,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC0RDA1, 0xB0BFA008,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC0TCKI, 0xB0BFA00C,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC1RCK, 	0xB0BFA010,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC1RDA0, 0xB0BFA014,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC1RDA1, 0xB0BFA018,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);
__IO_REG16_BIT(RICFG4G8_ARH0AIC1TCKI, 0xB0BFA01C,__READ_WRITE ,__ricfg4g8_arh0aicrck_bits);

/***************************************************************************
 **
 ** DMA0
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA0_A0,								0xB0C00000,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B0,								0xB0C00004,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA0,							0xB0C00008,__READ_WRITE );
__IO_REG32(		 DMA0_DA0,							0xB0C0000C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C0,								0xB0C00010,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D0,								0xB0C00014,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW0,					0xB0C00018,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW0,					0xB0C0001C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A1,								0xB0C00040,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B1,								0xB0C00044,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA1,							0xB0C00048,__READ_WRITE );
__IO_REG32(		 DMA0_DA1,							0xB0C0004C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C1,								0xB0C00050,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D1,								0xB0C00054,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW1,					0xB0C00058,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW1,					0xB0C0005C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A2,								0xB0C00080,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B2,								0xB0C00084,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA2,							0xB0C00088,__READ_WRITE );
__IO_REG32(		 DMA0_DA2,							0xB0C0008C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C2,								0xB0C00090,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D2,								0xB0C00094,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW2,					0xB0C00098,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW2,					0xB0C0009C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A3,								0xB0C000C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B3,								0xB0C000C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA3,							0xB0C000C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA3,							0xB0C000CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C3,								0xB0C000D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D3,								0xB0C000D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW3,					0xB0C000D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW3,					0xB0C000DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A4,								0xB0C00100,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B4,								0xB0C00104,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA4,							0xB0C00108,__READ_WRITE );
__IO_REG32(		 DMA0_DA4,							0xB0C0010C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C4,								0xB0C00110,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D4,								0xB0C00114,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW4,					0xB0C00118,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW4,					0xB0C0011C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A5,								0xB0C00140,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B5,								0xB0C00144,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA5,							0xB0C00148,__READ_WRITE );
__IO_REG32(		 DMA0_DA5,							0xB0C0014C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C5,								0xB0C00150,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D5,								0xB0C00154,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW5,					0xB0C00158,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW5,					0xB0C0015C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A6,								0xB0C00180,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B6,								0xB0C00184,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA6,							0xB0C00188,__READ_WRITE );
__IO_REG32(		 DMA0_DA6,							0xB0C0018C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C6,								0xB0C00190,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D6,								0xB0C00194,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW6,					0xB0C00198,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW6,					0xB0C0019C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A7,								0xB0C001C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B7,								0xB0C001C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA7,							0xB0C001C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA7,							0xB0C001CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C7,								0xB0C001D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D7,								0xB0C001D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW7,					0xB0C001D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW7,					0xB0C001DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A8,								0xB0C00200,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B8,								0xB0C00204,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA8,							0xB0C00208,__READ_WRITE );
__IO_REG32(		 DMA0_DA8,							0xB0C0020C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C8,								0xB0C00210,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D8,								0xB0C00214,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW8,					0xB0C00218,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW8,					0xB0C0021C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A9,								0xB0C00240,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B9,								0xB0C00244,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA9,							0xB0C00248,__READ_WRITE );
__IO_REG32(		 DMA0_DA9,							0xB0C0024C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C9,								0xB0C00250,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D9,								0xB0C00254,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW9,					0xB0C00258,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW9,					0xB0C0025C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A10,							0xB0C00280,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B10,							0xB0C00284,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA10,	 						0xB0C00288,__READ_WRITE );
__IO_REG32(		 DMA0_DA10, 						0xB0C0028C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C10,							0xB0C00290,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D10,							0xB0C00294,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW10,					0xB0C00298,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW10,					0xB0C0029C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A11,							0xB0C002C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B11,							0xB0C002C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA11,							0xB0C002C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA11,							0xB0C002CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C11,							0xB0C002D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D11,							0xB0C002D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW11,					0xB0C002D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW11,					0xB0C002DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A12,							0xB0C00300,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B12,							0xB0C00304,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA12,							0xB0C00308,__READ_WRITE );
__IO_REG32(		 DMA0_DA12,							0xB0C0030C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C12,							0xB0C00310,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D12,							0xB0C00314,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW12,					0xB0C00318,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW12,					0xB0C0031C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A13,							0xB0C00340,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B13,							0xB0C00344,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA13,							0xB0C00348,__READ_WRITE );
__IO_REG32(		 DMA0_DA13,							0xB0C0034C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C13,							0xB0C00350,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D13,							0xB0C00354,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW13,					0xB0C00358,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW13,					0xB0C0035C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A14,							0xB0C00380,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B14,							0xB0C00384,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA14,	 						0xB0C00388,__READ_WRITE );
__IO_REG32(		 DMA0_DA14, 						0xB0C0038C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C14,							0xB0C00390,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D14,							0xB0C00394,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW14,					0xB0C00398,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW14,					0xB0C0039C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A15,							0xB0C003C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B15,							0xB0C003C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA15,							0xB0C003C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA15,							0xB0C003CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C15,							0xB0C003D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D15,							0xB0C003D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW15,					0xB0C003D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW15,					0xB0C003DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A16,							0xB0C00400,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B16,							0xB0C00404,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA16,							0xB0C00408,__READ_WRITE );
__IO_REG32(		 DMA0_DA16,							0xB0C0040C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C16,							0xB0C00410,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D16,							0xB0C00414,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW16,					0xB0C00418,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW16,					0xB0C0041C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A17,							0xB0C00440,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B17,							0xB0C00444,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA17,							0xB0C00448,__READ_WRITE );
__IO_REG32(		 DMA0_DA17,							0xB0C0044C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C17,							0xB0C00450,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D17,							0xB0C00454,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW17,					0xB0C00458,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW17,					0xB0C0045C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A18,							0xB0C00480,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B18,							0xB0C00484,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA18,	 						0xB0C00488,__READ_WRITE );
__IO_REG32(		 DMA0_DA18, 						0xB0C0048C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C18,							0xB0C00490,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D18,							0xB0C00494,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW18,					0xB0C00498,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW18,					0xB0C0049C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A19,							0xB0C004C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B19,							0xB0C004C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA19,							0xB0C004C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA19,							0xB0C004CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C19,							0xB0C004D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D19,							0xB0C004D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW19,					0xB0C004D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW19,					0xB0C004DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A20,							0xB0C00500,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B20,							0xB0C00504,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA20,							0xB0C00508,__READ_WRITE );
__IO_REG32(		 DMA0_DA20,							0xB0C0050C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C20,							0xB0C00510,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D20,							0xB0C00514,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW20,					0xB0C00518,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW20,					0xB0C0051C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A21,							0xB0C00540,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B21,							0xB0C00544,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA21,							0xB0C00548,__READ_WRITE );
__IO_REG32(		 DMA0_DA21,							0xB0C0054C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C21,							0xB0C00550,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D21,							0xB0C00554,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW21,					0xB0C00558,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW21,					0xB0C0055C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A22,							0xB0C00580,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B22,							0xB0C00584,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA22,	 						0xB0C00588,__READ_WRITE );
__IO_REG32(		 DMA0_DA22, 						0xB0C0058C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C22,							0xB0C00590,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D22,							0xB0C00594,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW22,					0xB0C00598,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW22,					0xB0C0059C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A23,							0xB0C005C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B23,							0xB0C005C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA23,							0xB0C005C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA23,							0xB0C005CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C23,							0xB0C005D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D23,							0xB0C005D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW23,					0xB0C005D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW23,					0xB0C005DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A24,							0xB0C00600,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B24,							0xB0C00604,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA24,							0xB0C00608,__READ_WRITE );
__IO_REG32(		 DMA0_DA24,							0xB0C0060C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C24,							0xB0C00610,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D24,							0xB0C00614,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW24,					0xB0C00618,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW24,					0xB0C0061C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A25,							0xB0C00640,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B25,							0xB0C00644,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA25,							0xB0C00648,__READ_WRITE );
__IO_REG32(		 DMA0_DA25,							0xB0C0064C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C25,							0xB0C00650,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D25,							0xB0C00654,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW25,					0xB0C00658,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW25,					0xB0C0065C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A26,							0xB0C00680,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B26,							0xB0C00684,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA26,	 						0xB0C00688,__READ_WRITE );
__IO_REG32(		 DMA0_DA26, 						0xB0C0068C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C26,							0xB0C00690,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D26,							0xB0C00694,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW26,					0xB0C00698,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW26,					0xB0C0069C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A27,							0xB0C006C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B27,							0xB0C006C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA27,							0xB0C006C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA27,							0xB0C006CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C27,							0xB0C006D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D27,							0xB0C006D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW27,					0xB0C006D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW27,					0xB0C006DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A28,							0xB0C00700,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B28,							0xB0C00704,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA28,							0xB0C00708,__READ_WRITE );
__IO_REG32(		 DMA0_DA28,							0xB0C0070C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C28,							0xB0C00710,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D28,							0xB0C00714,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW28,					0xB0C00718,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW28,					0xB0C0071C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A29,							0xB0C00740,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B29,							0xB0C00744,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA29,							0xB0C00748,__READ_WRITE );
__IO_REG32(		 DMA0_DA29,							0xB0C0074C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C29,							0xB0C00750,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D29,							0xB0C00754,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW29,					0xB0C00758,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW29,					0xB0C0075C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A30,							0xB0C00780,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B30,							0xB0C00784,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA30,	 						0xB0C00788,__READ_WRITE );
__IO_REG32(		 DMA0_DA30, 						0xB0C0078C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C30,							0xB0C00790,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D30,							0xB0C00794,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW30,					0xB0C00798,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW30,					0xB0C0079C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A31,							0xB0C007C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B31,							0xB0C007C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA31,							0xB0C007C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA31,							0xB0C007CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C31,							0xB0C007D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D31,							0xB0C007D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW31,					0xB0C007D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW31,					0xB0C007DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A32,							0xB0C00800,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B32,							0xB0C00804,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA32,							0xB0C00808,__READ_WRITE );
__IO_REG32(		 DMA0_DA32,							0xB0C0080C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C32,							0xB0C00810,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D32,							0xB0C00814,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW32,					0xB0C00818,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW32,					0xB0C0081C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A33,							0xB0C00840,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B33,							0xB0C00844,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA33,							0xB0C00848,__READ_WRITE );
__IO_REG32(		 DMA0_DA33,							0xB0C0084C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C33,							0xB0C00850,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D33,							0xB0C00854,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW33,					0xB0C00858,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW33,					0xB0C0085C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A34,							0xB0C00880,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B34,							0xB0C00884,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA34,	 						0xB0C00888,__READ_WRITE );
__IO_REG32(		 DMA0_DA34, 						0xB0C0088C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C34,							0xB0C00890,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D34,							0xB0C00894,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW34,					0xB0C00898,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW34,					0xB0C0089C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A35,							0xB0C008C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B35,							0xB0C008C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA35,							0xB0C008C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA35,							0xB0C008CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C35,							0xB0C008D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D35,							0xB0C008D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW35,					0xB0C008D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW35,					0xB0C008DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A36,							0xB0C00900,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B36,							0xB0C00904,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA36,							0xB0C00908,__READ_WRITE );
__IO_REG32(		 DMA0_DA36,							0xB0C0090C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C36,							0xB0C00910,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D36,							0xB0C00914,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW36,					0xB0C00918,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW36,					0xB0C0091C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A37,							0xB0C00940,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B37,							0xB0C00944,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA37,							0xB0C00948,__READ_WRITE );
__IO_REG32(		 DMA0_DA37,							0xB0C0094C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C37,							0xB0C00950,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D37,							0xB0C00954,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW37,					0xB0C00958,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW37,					0xB0C0095C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A38,							0xB0C00980,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B38,							0xB0C00984,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA38,	 						0xB0C00988,__READ_WRITE );
__IO_REG32(		 DMA0_DA38, 						0xB0C0098C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C38,							0xB0C00990,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D38,							0xB0C00994,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW38,					0xB0C00998,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW38,					0xB0C0099C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A39,							0xB0C009C0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B39,							0xB0C009C4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA39,							0xB0C009C8,__READ_WRITE );
__IO_REG32(		 DMA0_DA39,							0xB0C009CC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C39,							0xB0C009D0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D39,							0xB0C009D4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW39,					0xB0C009D8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW39,					0xB0C009DC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A40,							0xB0C00A00,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B40,							0xB0C00A04,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA40,							0xB0C00A08,__READ_WRITE );
__IO_REG32(		 DMA0_DA40,							0xB0C00A0C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C40,							0xB0C00A10,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D40,							0xB0C00A14,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW40,					0xB0C00A18,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW40,					0xB0C00A1C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A41,							0xB0C00A40,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B41,							0xB0C00A44,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA41,							0xB0C00A48,__READ_WRITE );
__IO_REG32(		 DMA0_DA41,							0xB0C00A4C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C41,							0xB0C00A50,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D41,							0xB0C00A54,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW41,					0xB0C00A58,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW41,					0xB0C00A5C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A42,							0xB0C00A80,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B42,							0xB0C00A84,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA42,	 						0xB0C00A88,__READ_WRITE );
__IO_REG32(		 DMA0_DA42, 						0xB0C00A8C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C42,							0xB0C00A90,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D42,							0xB0C00A94,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW42,					0xB0C00A98,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW42,					0xB0C00A9C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A43,							0xB0C00AC0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B43,							0xB0C00AC4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA43,							0xB0C00AC8,__READ_WRITE );
__IO_REG32(		 DMA0_DA43,							0xB0C00ACC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C43,							0xB0C00AD0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D43,							0xB0C00AD4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW43,					0xB0C00AD8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW43,					0xB0C00ADC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A44,							0xB0C00B00,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B44,							0xB0C00B04,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA44,							0xB0C00B08,__READ_WRITE );
__IO_REG32(		 DMA0_DA44,							0xB0C00B0C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C44,							0xB0C00B10,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D44,							0xB0C00B14,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW44,					0xB0C00B18,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW44,					0xB0C00B1C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A45,							0xB0C00B40,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B45,							0xB0C00B44,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA45,							0xB0C00B48,__READ_WRITE );
__IO_REG32(		 DMA0_DA45,							0xB0C00B4C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C45,							0xB0C00B50,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D45,							0xB0C00B54,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW45,					0xB0C00B58,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW45,					0xB0C00B5C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A46,							0xB0C00B80,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B46,							0xB0C00B84,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA46,	 						0xB0C00B88,__READ_WRITE );
__IO_REG32(		 DMA0_DA46, 						0xB0C00B8C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C46,							0xB0C00B90,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D46,							0xB0C00B94,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW46,					0xB0C00B98,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW46,					0xB0C00B9C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A47,							0xB0C00BC0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B47,							0xB0C00BC4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA47,							0xB0C00BC8,__READ_WRITE );
__IO_REG32(		 DMA0_DA47,							0xB0C00BCC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C47,							0xB0C00BD0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D47,							0xB0C00BD4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW47,					0xB0C00BD8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW47,					0xB0C00BDC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A48,							0xB0C00C00,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B48,							0xB0C00C04,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA48,							0xB0C00C08,__READ_WRITE );
__IO_REG32(		 DMA0_DA48,							0xB0C00C0C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C48,							0xB0C00C10,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D48,							0xB0C00C14,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW48,					0xB0C00C18,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW48,					0xB0C00C1C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A49,							0xB0C00C40,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B49,							0xB0C00C44,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA49,							0xB0C00C48,__READ_WRITE );
__IO_REG32(		 DMA0_DA49,							0xB0C00C4C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C49,							0xB0C00C50,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D49,							0xB0C00C54,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW49,					0xB0C00C58,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW49,					0xB0C00C5C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A50,							0xB0C00C80,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B50,							0xB0C00C84,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA50,	 						0xB0C00C88,__READ_WRITE );
__IO_REG32(		 DMA0_DA50, 						0xB0C00C8C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C50,							0xB0C00C90,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D50,							0xB0C00C94,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW50,					0xB0C00C98,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW50,					0xB0C00C9C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A51,							0xB0C00CC0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B51,							0xB0C00CC4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA51,							0xB0C00CC8,__READ_WRITE );
__IO_REG32(		 DMA0_DA51,							0xB0C00CCC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C51,							0xB0C00CD0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D51,							0xB0C00CD4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW51,					0xB0C00CD8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW51,					0xB0C00CDC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A52,							0xB0C00D00,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B52,							0xB0C00D04,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA52,							0xB0C00D08,__READ_WRITE );
__IO_REG32(		 DMA0_DA52,							0xB0C00D0C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C52,							0xB0C00D10,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D52,							0xB0C00D14,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW52,					0xB0C00D18,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW52,					0xB0C00D1C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A53,							0xB0C00D40,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B53,							0xB0C00D44,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA53,							0xB0C00D48,__READ_WRITE );
__IO_REG32(		 DMA0_DA53,							0xB0C00D4C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C53,							0xB0C00D50,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D53,							0xB0C00D54,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW53,					0xB0C00D58,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW53,					0xB0C00D5C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A54,							0xB0C00D80,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B54,							0xB0C00D84,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA54,	 						0xB0C00D88,__READ_WRITE );
__IO_REG32(		 DMA0_DA54, 						0xB0C00D8C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C54,							0xB0C00D90,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D54,							0xB0C00D94,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW54,					0xB0C00D98,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW54,					0xB0C00D9C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A55,							0xB0C00DC0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B55,							0xB0C00DC4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA55,							0xB0C00DC8,__READ_WRITE );
__IO_REG32(		 DMA0_DA55,							0xB0C00DCC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C55,							0xB0C00DD0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D55,							0xB0C00DD4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW55,					0xB0C00DD8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW55,					0xB0C00DDC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A56,							0xB0C00E00,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B56,							0xB0C00E04,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA56,							0xB0C00E08,__READ_WRITE );
__IO_REG32(		 DMA0_DA56,							0xB0C00E0C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C56,							0xB0C00E10,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D56,							0xB0C00E14,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW56,					0xB0C00E18,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW56,					0xB0C00E1C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A57,							0xB0C00E40,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B57,							0xB0C00E44,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA57,							0xB0C00E48,__READ_WRITE );
__IO_REG32(		 DMA0_DA57,							0xB0C00E4C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C57,							0xB0C00E50,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D57,							0xB0C00E54,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW57,					0xB0C00E58,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW57,					0xB0C00E5C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A58,							0xB0C00E80,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B58,							0xB0C00E84,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA58,	 						0xB0C00E88,__READ_WRITE );
__IO_REG32(		 DMA0_DA58, 						0xB0C00E8C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C58,							0xB0C00E90,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D58,							0xB0C00E94,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW58,					0xB0C00E98,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW58,					0xB0C00E9C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A59,							0xB0C00EC0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B59,							0xB0C00EC4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA59,							0xB0C00EC8,__READ_WRITE );
__IO_REG32(		 DMA0_DA59,							0xB0C00ECC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C59,							0xB0C00ED0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D59,							0xB0C00ED4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW59,					0xB0C00ED8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW59,					0xB0C00EDC,__READ_WRITE );
__IO_REG32_BIT(DMA0_A60,							0xB0C00F00,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B60,							0xB0C00F04,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA60,							0xB0C00F08,__READ_WRITE );
__IO_REG32(		 DMA0_DA60,							0xB0C00F0C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C60,							0xB0C00F10,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D60,							0xB0C00F14,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW60,					0xB0C00F18,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW60,					0xB0C00F1C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A61,							0xB0C00F40,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B61,							0xB0C00F44,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA61,							0xB0C00F48,__READ_WRITE );
__IO_REG32(		 DMA0_DA61,							0xB0C00F4C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C61,							0xB0C00F50,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D61,							0xB0C00F54,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW61,					0xB0C00F58,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW61,					0xB0C00F5C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A62,							0xB0C00F80,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B62,							0xB0C00F84,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA62,	 						0xB0C00F88,__READ_WRITE );
__IO_REG32(		 DMA0_DA62, 						0xB0C00F8C,__READ_WRITE );
__IO_REG32_BIT(DMA0_C62,							0xB0C00F90,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D62,							0xB0C00F94,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW62,					0xB0C00F98,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW62,					0xB0C00F9C,__READ_WRITE );
__IO_REG32_BIT(DMA0_A63,							0xB0C00FC0,__READ_WRITE ,__dma_a_bits);
__IO_REG32_BIT(DMA0_B63,							0xB0C00FC4,__READ_WRITE ,__dma_b_bits);
__IO_REG32(		 DMA0_SA63,							0xB0C00FC8,__READ_WRITE );
__IO_REG32(		 DMA0_DA63,							0xB0C00FCC,__READ_WRITE );
__IO_REG32_BIT(DMA0_C63,							0xB0C00FD0,__READ_WRITE ,__dma_c_bits);
__IO_REG32_BIT(DMA0_D63,							0xB0C00FD4,__READ_WRITE ,__dma_d_bits);
__IO_REG32(		 DMA0_SASHDW63,					0xB0C00FD8,__READ_WRITE );
__IO_REG32(		 DMA0_DASHDW63,					0xB0C00FDC,__READ_WRITE );
__IO_REG32_BIT(DMA0_R,								0xB0C01000,__READ_WRITE ,__dma_r_bits);
__IO_REG32(		 DMA0_DIRQ1,						0xB0C01004,__READ_WRITE );
__IO_REG32(		 DMA0_DIRQ2,						0xB0C01008,__READ_WRITE );
__IO_REG32(		 DMA0_EDIRQ1,						0xB0C0100C,__READ_WRITE );
__IO_REG32(		 DMA0_EDIRQ2,						0xB0C01010,__READ_WRITE );
__IO_REG32(		 DMA0_ID,								0xB0C01014,__READ_WRITE );
__IO_REG32_BIT(DMA0_CMECIC0,					0xB0C02000,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC1,					0xB0C02004,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC2,					0xB0C02008,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC3,					0xB0C0200C,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC4,					0xB0C02010,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC5,					0xB0C02014,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC6,					0xB0C02018,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMECIC7,					0xB0C0201C,__READ_WRITE ,__dma_cmecic_bits);
__IO_REG32_BIT(DMA0_CMICIC0,					0xB0C02020,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC1,					0xB0C02024,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC2,					0xB0C02028,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC3,					0xB0C0202C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC4,					0xB0C02030,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC5,					0xB0C02034,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC6,					0xB0C02038,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC7,					0xB0C0203C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC8,					0xB0C02040,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC9,					0xB0C02044,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC10,					0xB0C02048,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC11,					0xB0C0204C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC12,					0xB0C02050,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC13,					0xB0C02054,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC14,					0xB0C02058,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC15,					0xB0C0205C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC16,					0xB0C02060,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC17,					0xB0C02064,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC18,					0xB0C02068,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC19,					0xB0C0206C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC20,					0xB0C02070,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC21,					0xB0C02074,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC22,					0xB0C02078,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC23,					0xB0C0207C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC24,					0xB0C02080,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC25,					0xB0C02084,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC26,					0xB0C02088,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC27,					0xB0C0208C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC28,					0xB0C02090,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC29,					0xB0C02094,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC30,					0xB0C02098,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC31,					0xB0C0209C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC32,					0xB0C020A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC33,					0xB0C020A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC34,					0xB0C020A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC35,					0xB0C020AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC36,					0xB0C020B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC37,					0xB0C020B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC38,					0xB0C020B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC39,					0xB0C020BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC40,					0xB0C020C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC41,					0xB0C020C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC42,					0xB0C020C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC43,					0xB0C020CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC44,					0xB0C020D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC45,					0xB0C020D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC46,					0xB0C020D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC47,					0xB0C020DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC48,					0xB0C020E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC49,					0xB0C020E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC50,					0xB0C020E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC51,					0xB0C020EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC52,					0xB0C020F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC53,					0xB0C020F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC54,					0xB0C020F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC55,					0xB0C020FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC56,					0xB0C02100,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC57,					0xB0C02104,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC58,					0xB0C02108,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC59,					0xB0C0210C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC60,					0xB0C02110,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC61,					0xB0C02114,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC62,					0xB0C02118,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC63,					0xB0C0211C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC64,					0xB0C02120,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC65,					0xB0C02124,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC66,					0xB0C02128,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC67,					0xB0C0212C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC68,					0xB0C02130,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC69,					0xB0C02134,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC70,					0xB0C02138,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC71,					0xB0C0213C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC72,					0xB0C02140,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC73,					0xB0C02144,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC74,					0xB0C02148,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC75,					0xB0C0214C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC76,					0xB0C02150,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC77,					0xB0C02154,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC78,					0xB0C02158,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC79,					0xB0C0215C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC80,					0xB0C02160,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC81,					0xB0C02164,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC82,					0xB0C02168,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC83,					0xB0C0216C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC84,					0xB0C02170,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC85,					0xB0C02174,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC86,					0xB0C02178,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC87,					0xB0C0217C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC88,					0xB0C02180,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC89,					0xB0C02184,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC90,					0xB0C02188,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC91,					0xB0C0218C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC92,					0xB0C02190,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC93,					0xB0C02194,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC94,					0xB0C02198,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC95,					0xB0C0219C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC96,					0xB0C021A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC97,					0xB0C021A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC98,					0xB0C021A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC99,					0xB0C021AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC100,				0xB0C021B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC101,				0xB0C021B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC102,				0xB0C021B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC103,				0xB0C021BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC104,				0xB0C021C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC105,				0xB0C021C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC106,				0xB0C021C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC107,				0xB0C021CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC108,				0xB0C021D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC109,				0xB0C021D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC110,				0xB0C021D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC111,				0xB0C021DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC112,				0xB0C021E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC113,				0xB0C021E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC114,				0xB0C021E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC115,				0xB0C021EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC116,				0xB0C021F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC117,				0xB0C021F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC118,				0xB0C021F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC119,				0xB0C021FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC120,				0xB0C02200,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC121,				0xB0C02204,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC122,				0xB0C02208,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC123,				0xB0C0220C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC124,				0xB0C02210,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC125,				0xB0C02214,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC126,				0xB0C02218,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC127,				0xB0C0221C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC128,				0xB0C02220,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC129,				0xB0C02224,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC130,				0xB0C02228,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC131,				0xB0C0222C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC132,				0xB0C02230,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC133,				0xB0C02234,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC134,				0xB0C02238,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC135,				0xB0C0223C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC136,				0xB0C02240,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC137,				0xB0C02244,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC138,				0xB0C02248,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC139,				0xB0C0224C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC140,				0xB0C02250,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC141,				0xB0C02254,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC142,				0xB0C02258,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC143,				0xB0C0225C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC144,				0xB0C02260,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC145,				0xB0C02264,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC146,				0xB0C02268,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC147,				0xB0C0226C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC148,				0xB0C02270,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC149,				0xB0C02274,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC150,				0xB0C02278,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC151,				0xB0C0227C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC152,				0xB0C02280,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC153,				0xB0C02284,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC154,				0xB0C02288,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC155,				0xB0C0228C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC156,				0xB0C02290,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC157,				0xB0C02294,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC158,				0xB0C02298,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC159,				0xB0C0229C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC160,				0xB0C022A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC161,				0xB0C022A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC162,				0xB0C022A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC163,				0xB0C022AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC164,				0xB0C022B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC165,				0xB0C022B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC166,				0xB0C022B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC167,				0xB0C022BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC168,				0xB0C022C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC169,				0xB0C022C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC170,				0xB0C022C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC171,				0xB0C022CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC172,				0xB0C022D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC173,				0xB0C022D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC174,				0xB0C022D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC175,				0xB0C022DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC176,				0xB0C022E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC177,				0xB0C022E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC178,				0xB0C022E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC179,				0xB0C022EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC180,				0xB0C022F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC181,				0xB0C022F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC182,				0xB0C022F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC183,				0xB0C022FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC184,				0xB0C02300,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC185,				0xB0C02304,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC186,				0xB0C02308,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC187,				0xB0C0230C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC188,				0xB0C02310,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC189,				0xB0C02314,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC190,				0xB0C02318,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC191,				0xB0C0231C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC192,				0xB0C02320,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC193,				0xB0C02324,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC194,				0xB0C02328,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC195,				0xB0C0232C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC196,				0xB0C02330,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC197,				0xB0C02334,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC198,				0xB0C02338,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC199,				0xB0C0233C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC200,				0xB0C02340,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC201,				0xB0C02344,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC202,				0xB0C02348,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC203,				0xB0C0234C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC204,				0xB0C02350,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC205,				0xB0C02354,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC206,				0xB0C02358,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC207,				0xB0C0235C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC208,				0xB0C02360,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC209,				0xB0C02364,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC210,				0xB0C02368,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC211,				0xB0C0236C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC212,				0xB0C02370,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC213,				0xB0C02374,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC214,				0xB0C02378,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC215,				0xB0C0237C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC216,				0xB0C02380,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC217,				0xB0C02384,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC218,				0xB0C02388,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC219,				0xB0C0238C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC220,				0xB0C02390,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC221,				0xB0C02394,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC222,				0xB0C02398,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC223,				0xB0C0239C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC224,				0xB0C023A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC225,				0xB0C023A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC226,				0xB0C023A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC227,				0xB0C023AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC228,				0xB0C023B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC229,				0xB0C023B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC230,				0xB0C023B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC231,				0xB0C023BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC232,				0xB0C023C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC233,				0xB0C023C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC234,				0xB0C023C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC235,				0xB0C023CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC236,				0xB0C023D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC237,				0xB0C023D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC238,				0xB0C023D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC239,				0xB0C023DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC240,				0xB0C023E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC241,				0xB0C023E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC242,				0xB0C023E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC243,				0xB0C023EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC244,				0xB0C023F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC245,				0xB0C023F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC246,				0xB0C023F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC247,				0xB0C023FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC248,				0xB0C02400,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC249,				0xB0C02404,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC250,				0xB0C02408,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC251,				0xB0C0240C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC252,				0xB0C02410,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC253,				0xB0C02414,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC254,				0xB0C02418,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC255,				0xB0C0241C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC256,				0xB0C02420,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC257,				0xB0C02424,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC258,				0xB0C02428,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC259,				0xB0C0242C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC260,				0xB0C02430,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC261,				0xB0C02434,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC262,				0xB0C02438,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC263,				0xB0C0243C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC264,				0xB0C02440,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC265,				0xB0C02444,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC266,				0xB0C02448,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC267,				0xB0C0244C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC268,				0xB0C02450,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC269,				0xB0C02454,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC270,				0xB0C02458,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC271,				0xB0C0245C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC272,				0xB0C02460,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC273,				0xB0C02464,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC274,				0xB0C02468,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC275,				0xB0C0246C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC276,				0xB0C02470,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC277,				0xB0C02474,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC278,				0xB0C02478,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC279,				0xB0C0247C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC280,				0xB0C02480,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC281,				0xB0C02484,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC282,				0xB0C02488,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC283,				0xB0C0248C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC284,				0xB0C02490,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC285,				0xB0C02494,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC286,				0xB0C02498,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC287,				0xB0C0249C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC288,				0xB0C024A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC289,				0xB0C024A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC290,				0xB0C024A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC291,				0xB0C024AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC292,				0xB0C024B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC293,				0xB0C024B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC294,				0xB0C024B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC295,				0xB0C024BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC296,				0xB0C024C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC297,				0xB0C024C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC298,				0xB0C024C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC299,				0xB0C024CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC300,				0xB0C024D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC301,				0xB0C024D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC302,				0xB0C024D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC303,				0xB0C024DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC304,				0xB0C024E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC305,				0xB0C024E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC306,				0xB0C024E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC307,				0xB0C024EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC308,				0xB0C024F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC309,				0xB0C024F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC310,				0xB0C024F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC311,				0xB0C024FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC312,				0xB0C02500,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC313,				0xB0C02504,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC314,				0xB0C02508,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC315,				0xB0C0250C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC316,				0xB0C02510,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC317,				0xB0C02514,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC318,				0xB0C02518,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC319,				0xB0C0251C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC320,				0xB0C02520,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC321,				0xB0C02524,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC322,				0xB0C02528,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC323,				0xB0C0252C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC324,				0xB0C02530,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC325,				0xB0C02534,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC326,				0xB0C02538,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC327,				0xB0C0253C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC328,				0xB0C02540,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC329,				0xB0C02544,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC330,				0xB0C02548,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC331,				0xB0C0254C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC332,				0xB0C02550,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC333,				0xB0C02554,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC334,				0xB0C02558,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC335,				0xB0C0255C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC336,				0xB0C02560,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC337,				0xB0C02564,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC338,				0xB0C02568,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC339,				0xB0C0256C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC340,				0xB0C02570,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC341,				0xB0C02574,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC342,				0xB0C02578,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC343,				0xB0C0257C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC344,				0xB0C02580,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC345,				0xB0C02584,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC346,				0xB0C02588,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC347,				0xB0C0258C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC348,				0xB0C02590,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC349,				0xB0C02594,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC350,				0xB0C02598,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC351,				0xB0C0259C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC352,				0xB0C025A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC353,				0xB0C025A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC354,				0xB0C025A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC355,				0xB0C025AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC356,				0xB0C025B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC357,				0xB0C025B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC358,				0xB0C025B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC359,				0xB0C025BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC360,				0xB0C025C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC361,				0xB0C025C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC362,				0xB0C025C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC363,				0xB0C025CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC364,				0xB0C025D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC365,				0xB0C025D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC366,				0xB0C025D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC367,				0xB0C025DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC368,				0xB0C025E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC369,				0xB0C025E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC370,				0xB0C025E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC371,				0xB0C025EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC372,				0xB0C025F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC373,				0xB0C025F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC374,				0xB0C025F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC375,				0xB0C025FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC376,				0xB0C02600,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC377,				0xB0C02604,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC378,				0xB0C02608,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC379,				0xB0C0260C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC380,				0xB0C02610,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC381,				0xB0C02614,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC382,				0xB0C02618,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC383,				0xB0C0261C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC384,				0xB0C02620,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC385,				0xB0C02624,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC386,				0xB0C02628,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC387,				0xB0C0262C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC388,				0xB0C02630,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC389,				0xB0C02634,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC390,				0xB0C02638,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC391,				0xB0C0263C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC392,				0xB0C02640,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC393,				0xB0C02644,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC394,				0xB0C02648,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC395,				0xB0C0264C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC396,				0xB0C02650,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC397,				0xB0C02654,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC398,				0xB0C02658,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC399,				0xB0C0265C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC400,				0xB0C02660,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC401,				0xB0C02664,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC402,				0xB0C02668,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC403,				0xB0C0266C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC404,				0xB0C02670,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC405,				0xB0C02674,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC406,				0xB0C02678,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC407,				0xB0C0267C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC408,				0xB0C02680,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC409,				0xB0C02684,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC410,				0xB0C02688,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC411,				0xB0C0268C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC412,				0xB0C02690,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC413,				0xB0C02694,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC414,				0xB0C02698,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC415,				0xB0C0269C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC416,				0xB0C026A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC417,				0xB0C026A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC418,				0xB0C026A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC419,				0xB0C026AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC420,				0xB0C026B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC421,				0xB0C026B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC422,				0xB0C026B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC423,				0xB0C026BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC424,				0xB0C026C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC425,				0xB0C026C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC426,				0xB0C026C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC427,				0xB0C026CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC428,				0xB0C026D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC429,				0xB0C026D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC430,				0xB0C026D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC431,				0xB0C026DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC432,				0xB0C026E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC433,				0xB0C026E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC434,				0xB0C026E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC435,				0xB0C026EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC436,				0xB0C026F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC437,				0xB0C026F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC438,				0xB0C026F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC439,				0xB0C026FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC440,				0xB0C02700,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC441,				0xB0C02704,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC442,				0xB0C02708,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC443,				0xB0C0270C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC444,				0xB0C02710,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC445,				0xB0C02714,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC446,				0xB0C02718,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC447,				0xB0C0271C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC448,				0xB0C02720,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC449,				0xB0C02724,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC450,				0xB0C02728,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC451,				0xB0C0272C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC452,				0xB0C02730,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC453,				0xB0C02734,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC454,				0xB0C02738,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC455,				0xB0C0273C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC456,				0xB0C02740,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC457,				0xB0C02744,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC458,				0xB0C02748,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC459,				0xB0C0274C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC460,				0xB0C02750,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC461,				0xB0C02754,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC462,				0xB0C02758,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC463,				0xB0C0275C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC464,				0xB0C02760,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC465,				0xB0C02764,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC466,				0xB0C02768,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC467,				0xB0C0276C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC468,				0xB0C02770,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC469,				0xB0C02774,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC470,				0xB0C02778,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC471,				0xB0C0277C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC472,				0xB0C02780,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC473,				0xB0C02784,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC474,				0xB0C02788,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC475,				0xB0C0278C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC476,				0xB0C02790,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC477,				0xB0C02794,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC478,				0xB0C02798,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC479,				0xB0C0279C,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC480,				0xB0C027A0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC481,				0xB0C027A4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC482,				0xB0C027A8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC483,				0xB0C027AC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC484,				0xB0C027B0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC485,				0xB0C027B4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC486,				0xB0C027B8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC487,				0xB0C027BC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC488,				0xB0C027C0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC489,				0xB0C027C4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC490,				0xB0C027C8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC491,				0xB0C027CC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC492,				0xB0C027D0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC493,				0xB0C027D4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC494,				0xB0C027D8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC495,				0xB0C027DC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC496,				0xB0C027E0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC497,				0xB0C027E4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC498,				0xB0C027E8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC499,				0xB0C027EC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC500,				0xB0C027F0,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC501,				0xB0C027F4,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC502,				0xB0C027F8,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMICIC503,				0xB0C027FC,__READ_WRITE ,__dma_cmicic_bits);
__IO_REG32_BIT(DMA0_CMCHIC0,					0xB0C02800,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC1,					0xB0C02804,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC2,					0xB0C02808,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC3,					0xB0C0280C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC4,					0xB0C02810,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC5,					0xB0C02814,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC6,					0xB0C02818,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC7,					0xB0C0281C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC8,					0xB0C02820,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC9,					0xB0C02824,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC10,					0xB0C02828,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC11,					0xB0C0282C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC12,					0xB0C02830,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC13,					0xB0C02834,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC14,					0xB0C02838,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC15,					0xB0C0283C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC16,					0xB0C02840,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC17,					0xB0C02844,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC18,					0xB0C02848,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC19,					0xB0C0284C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC20,					0xB0C02850,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC21,					0xB0C02854,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC22,					0xB0C02858,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC23,					0xB0C0285C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC24,					0xB0C02860,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC25,					0xB0C02864,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC26,					0xB0C02868,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC27,					0xB0C0286C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC28,					0xB0C02870,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC29,					0xB0C02874,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC30,					0xB0C02878,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC31,					0xB0C0287C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC32,					0xB0C02880,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC33,					0xB0C02884,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC34,					0xB0C02888,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC35,					0xB0C0288C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC36,					0xB0C02890,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC37,					0xB0C02894,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC38,					0xB0C02898,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC39,					0xB0C0289C,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC40,					0xB0C028A0,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC41,					0xB0C028A4,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC42,					0xB0C028A8,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC43,					0xB0C028AC,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC44,					0xB0C028B0,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC45,					0xB0C028B4,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC46,					0xB0C028B8,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC47,					0xB0C028BC,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC48,					0xB0C028C0,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC49,					0xB0C028C4,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC50,					0xB0C028C8,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC51,					0xB0C028CC,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC52,					0xB0C028D0,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC53,					0xB0C028D4,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC54,					0xB0C028D8,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC55,					0xB0C028DC,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC56,					0xB0C028E0,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC57,					0xB0C028E4,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC58,					0xB0C028E8,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC59,					0xB0C028EC,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC60,					0xB0C028F0,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC61,					0xB0C028F4,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC62,					0xB0C028F8,__READ_WRITE ,__dma_cmchic_bits);
__IO_REG32_BIT(DMA0_CMCHIC63,					0xB0C028FC,__READ_WRITE ,__dma_cmchic_bits);

/***************************************************************************
 **
 ** MPUXDMA0
 **
 ***************************************************************************/
__IO_REG32_BIT(MPUXDMA0_CTRL0,     												0xB0C08000,__READ_WRITE ,__mpuxgfx_ctrl0_bits);
__IO_REG32_BIT(MPUXDMA0_NMIEN,     												0xB0C08004,__READ_WRITE ,__mpuxgfx_nmien_bits);
__IO_REG32_BIT(MPUXDMA0_WERRC,     												0xB0C08008,__READ_WRITE ,__mpuxgfx_werrc_bits);
__IO_REG32(		 MPUXDMA0_WERRA,     												0xB0C0800C,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_RERRC,     												0xB0C08010,__READ_WRITE ,__mpuxgfx_werrc_bits);
__IO_REG32(		 MPUXDMA0_RERRA,     												0xB0C08014,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL1,     												0xB0C08018,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR1,     											0xB0C0801C,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR1,     											0xB0C08020,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL2,     												0xB0C08024,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR2,     											0xB0C08028,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR2,     											0xB0C0802C,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL3,     												0xB0C08030,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR3,     											0xB0C08034,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR3,     											0xB0C08038,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL4,     												0xB0C0803C,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR4,     											0xB0C08040,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR4,     											0xB0C08044,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL5,     												0xB0C08048,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR5,     											0xB0C0804C,__READ_WRITE );
__IO_REG32( 	 MPUXDMA0_EADDR5,     											0xB0C08050,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL6,     												0xB0C08054,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR6,     											0xB0C08058,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR6,     											0xB0C0805C,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL7,     												0xB0C08060,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR7,     											0xB0C08064,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR7,     											0xB0C08068,__READ_WRITE );
__IO_REG32_BIT(MPUXDMA0_CTRL8,     												0xB0C0806C,__READ_WRITE ,__mpuxgfx_ctrl1_bits);
__IO_REG32(		 MPUXDMA0_SADDR8,     											0xB0C08070,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_EADDR8,     											0xB0C08074,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_UNLOCK,     											0xB0C08078,__READ_WRITE );
__IO_REG32(		 MPUXDMA0_MID,     													0xB0C0807C,__READ_WRITE );


/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */
/***************************************************************************
 **
 **  MB9EF126 DMA channels
 **
 ***************************************************************************/
#define DMA_EXTDMA0             0         /* External DMA Request 0 */
#define DMA_EXTDMA0             1         /* External DMA Request 1 */
#define DMA_EIC0DMA0            8         /* External Interrupt 1  DMA Request */
#define DMA_EIC0DMA1            9         /* External Interrupt 2  DMA Request */
#define DMA_EIC0DMA2           10         /* External Interrupt 3  DMA Request */
#define DMA_EIC0DMA3           11         /* External Interrupt 4  DMA Request */
#define DMA_EIC0DMA4           12         /* External Interrupt 5  DMA Request */
#define DMA_EIC0DMA5           13         /* External Interrupt 6  DMA Request */
#define DMA_EIC0DMA6           14         /* External Interrupt 7  DMA Request */
#define DMA_EIC0DMA7           15         /* External Interrupt 8  DMA Request */
#define DMA_EIC0DMA8           16         /* External Interrupt 9  DMA Request */
#define DMA_EIC0DMA9           17         /* External Interrupt 10 DMA Request */
#define DMA_EIC0DMA10          18         /* External Interrupt 11 DMA Request */
#define DMA_EIC0DMA11          19         /* External Interrupt 12 DMA Request */
#define DMA_EIC0DMA12          20         /* External Interrupt 13 DMA Request */
#define DMA_EIC0DMA13          21         /* External Interrupt 14 DMA Request */
#define DMA_EIC0DMA14          22         /* External Interrupt 15 DMA Request */
#define DMA_EIC0DMA15          23         /* External Interrupt 16 DMA Request */
#define DMA_EIC0DMA16          24         /* External Interrupt 17 DMA Request */
#define DMA_EIC0DMA17          25         /* External Interrupt 18 DMA Request */
#define DMA_EIC0DMA18          26         /* External Interrupt 19 DMA Request */
#define DMA_EIC0DMA19          27         /* External Interrupt 20 DMA Request */
#define DMA_EIC0DMA20          28         /* External Interrupt 21 DMA Request */
#define DMA_EIC0DMA21          29         /* External Interrupt 22 DMA Request */
#define DMA_EIC0DMA22          30         /* External Interrupt 23 DMA Request */
#define DMA_EIC0DMA23          31         /* External Interrupt 24 DMA Request */
#define DMA_EIC0DMA24          32         /* External Interrupt 25 DMA Request */
#define DMA_EIC0DMA25          33         /* External Interrupt 26 DMA Request */
#define DMA_EIC0DMA26          34         /* External Interrupt 27 DMA Request */
#define DMA_EIC0DMA27          35         /* External Interrupt 28 DMA Request */
#define DMA_EIC0DMA28          36         /* External Interrupt 29 DMA Request */
#define DMA_EIC0DMA29          37         /* External Interrupt 30 DMA Request */
#define DMA_EIC0DMA30          38         /* External Interrupt 31 DMA Request */
#define DMA_EIC0DMA31          39         /* External Interrupt  DMA Request */
#define DMA_SG0DMA             40         /* Sound Generator 0 DMA Request */
#define DMA_HSSPI0DMARX        44         /* HSSPI0 Receive DMA Request */
#define DMA_HSSPI0DMAT         45         /* HSSPI0 Transmit DMA Request */
#define DMA_FRT0DMA            48         /* Free Running Timer 0 DMA Request */
#define DMA_FRT1DMA            49         /* Free Running Timer 1 DMA Request */
#define DMA_FRT2DMA            50         /* Free Running Timer 2 DMA Request */
#define DMA_FRT3DMA            51         /* Free Running Timer 3 DMA Request */
#define DMA_FRT16DMA           64         /* Free Running Timer 16 DMA Request */
#define DMA_FRT17DMA           65         /* Free Running Timer 17 DMA Request */
#define DMA_FRT18DMA           66         /* Free Running Timer 18 DMA Request */
#define DMA_FRT19DMA           67         /* Free Running Timer 19 DMA Request */
#define DMA_ICU2DMA0           84         /* Input Capture Unit 2 ch0 DMA Request */
#define DMA_ICU2DMA1           85         /* Input Capture Unit 2 ch1 DMA Request */
#define DMA_ICU3DMA0           86         /* Input Capture Unit 3 ch0 DMA Request */
#define DMA_ICU3DMA1           87         /* Input Capture Unit 3 ch1 DMA Request */
#define DMA_ICU18DMA0         116         /* Input Capture Unit 18 ch0 DMA Request */
#define DMA_ICU18DMA1         117         /* Input Capture Unit 18 ch1 DMA Request */
#define DMA_ICU19DMA0         118         /* Input Capture Unit 19 ch0 DMA Request */
#define DMA_ICU19DMA1         119         /* Input Capture Unit 19 ch1 DMA Request */
#define DMA_OCU0DMA0          144         /* Output Compare Unit 0 ch0 DMA Request */
#define DMA_OCU0DMA1          145         /* Output Compare Unit 0 ch1 DMA Request */
#define DMA_OCU1DMA0          146         /* Output Compare Unit 1 ch0 DMA Request */
#define DMA_OCU1DMA1          147         /* Output Compare Unit 1 ch1 DMA Request */
#define DMA_OCU16DMA0         176         /* Output Compare Unit 16 ch0 DMA Request */
#define DMA_OCU16DMA1         177         /* Output Compare Unit 16 ch1 DMA Request */
#define DMA_OCU17DMA0         178         /* Output Compare Unit 17 ch0 DMA Request */
#define DMA_OCU17DMA1         179         /* Output Compare Unit 17 ch1 DMA Request */
#define DMA_USART0DMARX       208         /* LIN USART 0 Receive DMA Request */
#define DMA_USART0DMATX       209         /* LIN USART 0 Transmit DMA Request */
#define DMA_USART6DMARX       220         /* LIN USART 6 Receive DMA Request */
#define DMA_USART6DMATX       221         /* LIN USART 6 Transmit DMA Request */
#define DMA_I2C0DMARX         232         /* I2C0 Receive DMA Request */
#define DMA_I2C0DMATX         233         /* I2C0 Transmit DMA Request */
#define DMA_PPG0DMA           244         /* Programmable Pulse Generator 0 DMA Request */
#define DMA_PPG1DMA           245         /* Programmable Pulse Generator 1 DMA Request */
#define DMA_PPG2DMA           246         /* Programmable Pulse Generator 2 DMA Request */
#define DMA_PPG3DMA           247         /* Programmable Pulse Generator 3 DMA Request */
#define DMA_PPG4DMA           248         /* Programmable Pulse Generator 4 DMA Request */
#define DMA_PPG5DMA           249         /* Programmable Pulse Generator 5 DMA Request */
#define DMA_PPG6DMA           250         /* Programmable Pulse Generator 6 DMA Request */
#define DMA_PPG7DMA           251         /* Programmable Pulse Generator 7 DMA Request */
#define DMA_PPG8DMA           252         /* Programmable Pulse Generator 8 DMA Request */
#define DMA_PPG9DMA           253         /* Programmable Pulse Generator 9 DMA Request */
#define DMA_PPG10DMA          254         /* Programmable Pulse Generator 10 DMA Request */
#define DMA_PPG11DMA          255         /* Programmable Pulse Generator 11 DMA Request */
#define DMA_PPG12DMA          256         /* Programmable Pulse Generator 12 DMA Request */
#define DMA_PPG13DMA          257         /* Programmable Pulse Generator 13 DMA Request */
#define DMA_PPG14DMA          258         /* Programmable Pulse Generator 14 DMA Request */
#define DMA_PPG15DMA          259         /* Programmable Pulse Generator 15 DMA Request */
#define DMA_PPG64DMA          308         /* Programmable Pulse Generator 64 DMA Request */
#define DMA_PPG65DMA          309         /* Programmable Pulse Generator 65 DMA Request */
#define DMA_PPG66DMA          310         /* Programmable Pulse Generator 66 DMA Request */
#define DMA_PPG67DMA          311         /* Programmable Pulse Generator 67 DMA Request */
#define DMA_PPG68DMA          312         /* Programmable Pulse Generator 68 DMA Request */
#define DMA_PPG69DMA          313         /* Programmable Pulse Generator 69 DMA Request */
#define DMA_PPG70DMA          314         /* Programmable Pulse Generator 70 DMA Request */
#define DMA_PPG71DMA          315         /* Programmable Pulse Generator 71 DMA Request */
#define DMA_ADC0DMA           372         /* ADC0 Conversion End DMA Request */
#define DMA_ADC0DMA2          373         /* ADC0 Scan End DMA Request */
#define DMA_RLT0DMA           376         /* Reload Timer 0 DMA Request */
#define DMA_RLT1DMA           377         /* Reload Timer 1 DMA Request */
#define DMA_RLT2DMA           378         /* Reload Timer 2 DMA Request */
#define DMA_RLT3DMA           379         /* Reload Timer 3 DMA Request */
#define DMA_RLT4DMA           380         /* Reload Timer 4 DMA Request */
#define DMA_RLT5DMA           381         /* Reload Timer 5 DMA Request */
#define DMA_RLT6DMA           382         /* Reload Timer 6 DMA Request */
#define DMA_RLT7DMA           383         /* Reload Timer 7 DMA Request */
#define DMA_RLT8DMA           384         /* Reload Timer 8 DMA Request */
#define DMA_RLT9DMA           385         /* Reload Timer 9 DMA Request */
#define DMA_I2S0DMARX         408         /* I2S0 Receive DMA Request */
#define DMA_I2S0DMATX         409         /* I2S0 Transmit DMA Request */
#define DMA_I2S1DMARX         410         /* I2S1 Receive DMA Request */
#define DMA_I2S1DMATX         411         /* I2S1 Transmit DMA Request */
#define DMA_CRC0DMA           424         /* CRC0 DMA Request */
#define DMA_SPI0DMARX         426         /* SPI0 Receive DMA Request */
#define DMA_SPI0DMATX         427         /* SPI0 Transmit DMA Request */
#define DMA_SPI1DMARX         428         /* SPI1 Receive DMA Request */
#define DMA_SPI1DMATX         429         /* SPI1 Transmit DMA Request */
#define DMA_SPI2DMARX         430         /* SPI2 Receive DMA Request */
#define DMA_SPI2DMATX         431         /* SPI2 Transmit DMA Request */
#define DMA_EEFLASHDMA        450         /* EE Flash DMA Request */
#define DMA_PPUDMA            467         /* PPU DMA Request */

/***************************************************************************
 **
 **  MB9EF126 Interrupt Lines
 **
 ***************************************************************************/
#define INT_SYSCIRQ             0         /* System Controller Status Interrupt */
#define INT_WDGIRQ              1         /* Watchdog pre-warning Interrupt */
#define INT_ETH0IRQE           18         /* FastMAC Ethernet frame Interrupt */
#define INT_ETH0IRQF           19         /* FastMAC Flexi-filter Interrupt */
#define INT_ETH0IRQH           20         /* FastMAC High-priority Frame Interrupt */
#define INT_ETH0IRQP           21         /* FastMAC PTP Frame Interrupt */
#define INT_GFXIRQ0            22         /* GFX Interrupt 0 */
#define INT_GFXIRQ1            23         /* GFX Interrupt 1 */
#define INT_ADC0IRQ            30         /* ADC0 Conversion End Interrupt */
#define INT_ADC0IRQ2           31         /* ADC0 Scan End Interrupt */
#define INT_ADC0IRQR           32         /* ADC0 Range Comp Interrupt */
#define INT_ADC0IRQP           33         /* ADC0 Pulse Detect Interrupt */
#define INT_RRCFGIRQERR        34         /* Retention RAM Single Bit Error */
#define INT_SRCFGIRQERR        35         /* System RAM Single Bit Error */
#define INT_TCFCFGIRQERR       36         /* Instruction Flash Single Bit Error */
#define INT_EECFGIRQERR        37         /* Data Flash Single Bit Error */
#define INT_IRQ0IRQERR         38         /* IUNIT Vector RAM Single Bit Error */
#define INT_ETH0IRQERR         39         /* Ethernet RAM Single Bit Error */
#define INT_EECFGIRQ           41         /* Data Flash Write Completion Interrupt */
#define INT_EICU0IRQ           42         /* External Interrupt Capture Unit 0 Interrupt */
#define INT_HSSPI0IRQRX        43         /* HSSPI0 Receive Interrupt */
#define INT_HSSPI0IRQTX        44         /* HSSPI0 Transmit Interrupt */
#define INT_HSSPI0IRQERR       45         /* HSSPI0 Error Interrupt */
#define INT_SPI0IRQRX          49         /* SPI0 Receive Interrupt */
#define INT_SPI0IRQTX          50         /* SPI0 Transmit Interrupt */
#define INT_SPI0IRQRERR        51         /* SPI0 Error Interrupt */
#define INT_SPI1IRQRX          52         /* SPI1 Receive Interrupt */
#define INT_SPI1IRQTX          53         /* SPI1 Transmit Interrupt */
#define INT_SPI1IRQERR         54         /* SPI1 Error Interrupt */
#define INT_SPI2IRQRX          55         /* SPI2 Receive Interrupt */
#define INT_SPI2IRQTX          56         /* SPI2 Transmit Interrupt */
#define INT_SPI2IRQERR         57         /* SPI2 Error Interrupt */
#define INT_CAN0IRQ            61         /* CAN0 Interrupt */
#define INT_CAN1IRQ            62         /* CAN1 Interrupt */
#define INT_CAN2IRQ            63         /* CAN2 Interrupt */
#define INT_EIC0IRQ0           69         /* External Interrupt 0 */
#define INT_EIC0IRQ1           70         /* External Interrupt 1 */
#define INT_EIC0IRQ2           71         /* External Interrupt 2 */
#define INT_EIC0IRQ3           72         /* External Interrupt 3 */
#define INT_EIC0IRQ4           73         /* External Interrupt 4 */
#define INT_EIC0IRQ5           74         /* External Interrupt 5 */
#define INT_EIC0IRQ6           75         /* External Interrupt 6 */
#define INT_EIC0IRQ7           76         /* External Interrupt 7 */
#define INT_EIC0IRQ8           77         /* External Interrupt 8 */
#define INT_EIC0IRQ9           78         /* External Interrupt 9 */
#define INT_EIC0IRQ10          79         /* External Interrupt 10 */
#define INT_EIC0IRQ11          80         /* External Interrupt 11 */
#define INT_EIC0IRQ12          81         /* External Interrupt 12 */
#define INT_EIC0IRQ13          82         /* External Interrupt 13 */
#define INT_EIC0IRQ14          83         /* External Interrupt 14 */
#define INT_EIC0IRQ15          84         /* External Interrupt 15 */
#define INT_EIC0IRQ16          85         /* External Interrupt 16 */
#define INT_EIC0IRQ17          86         /* External Interrupt 17 */
#define INT_EIC0IRQ18          87         /* External Interrupt 18 */
#define INT_EIC0IRQ19          88         /* External Interrupt 19 */
#define INT_EIC0IRQ20          89         /* External Interrupt 20 */
#define INT_EIC0IRQ21          90         /* External Interrupt 21 */
#define INT_EIC0IRQ22          91         /* External Interrupt 22 */
#define INT_EIC0IRQ23          92         /* External Interrupt 23 */
#define INT_EIC0IRQ24          93         /* External Interrupt 24 */
#define INT_EIC0IRQ25          94         /* External Interrupt 25 */
#define INT_EIC0IRQ26          95         /* External Interrupt 26 */
#define INT_EIC0IRQ27          96         /* External Interrupt 27 */
#define INT_EIC0IRQ28          97         /* External Interrupt 28 */
#define INT_EIC0IRQ29          98         /* External Interrupt 29 */
#define INT_EIC0IRQ30          99         /* External Interrupt 30 */
#define INT_EIC0IRQ31         100         /* External Interrupt 31 */
#define INT_RTCIRQ            101         /* Real Time Clock Interrupt */
#define INT_SG0IRQ            102         /* Sound Generator 0 Interrupt */
#define INT_FRT0IRQ           104         /* Free Running Timer 0 Interrupt */
#define INT_FRT1IRQ           105         /* Free Running Timer 1 Interrupt */
#define INT_FRT2IRQ           106         /* Free Running Timer 2 Interrupt */
#define INT_FRT3IRQ           107         /* Free Running Timer 3 Interrupt */
#define INT_FRT16IRQ          112         /* Free Running Timer 16 Interrupt */
#define INT_FRT17IRQ          113         /* Free Running Timer 17 Interrupt */
#define INT_FRT18IRQ          114         /* Free Running Timer 18 Interrupt */
#define INT_FRT19IRQ          115         /* Free Running Timer 19 Interrupt */
#define INT_ICU2IRQ0          124         /* Input Capture Unit 2 ch0 Interrupt */
#define INT_ICU2IRQ1          125         /* Input Capture Unit 2 ch1 Interrupt */
#define INT_ICU3IRQ0          126         /* Input Capture Unit 3 ch0 Interrupt */
#define INT_ICU3IRQ1          127         /* Input Capture Unit 3 ch1 Interrupt */
#define INT_ICU18IRQ0         132         /* Input Capture Unit 18 ch0 Interrupt */
#define INT_ICU18IRQ1         133         /* Input Capture Unit 18 ch1 Interrupt */
#define INT_ICU19IRQ0         134         /* Input Capture Unit 19 ch0 Interrupt */
#define INT_ICU19IRQ1         135         /* Input Capture Unit 19 ch1 Interrupt */
#define INT_OCU0IRQ0          136         /* Output Compare Unit 0 ch0 Interrupt */
#define INT_OCU0IRQ1          137         /* Output Compare Unit 0 ch1 Interrupt */
#define INT_OCU1IRQ0          138         /* Output Compare Unit 1 ch0 Interrupt */
#define INT_OCU1IRQ1          139         /* Output Compare Unit 1 ch1 Interrupt */
#define INT_OCU16IRQ0         144         /* Output Compare Unit 16 ch0 Interrupt */
#define INT_OCU16IRQ1         145         /* Output Compare Unit 16 ch1 Interrupt */
#define INT_OCU17IRQ0         146         /* Output Compare Unit 17 ch0 Interrupt */
#define INT_OCU17IRQ1         147         /* Output Compare Unit 17 ch1 Interrupt */
#define INT_USART0IRQRX       152         /* LIN USART 0 Receive Interrupt */
#define INT_USART0IRQTX       153         /* LIN USART 0 Transmit Interrupt */
#define INT_USART0IRQERR      154         /* LIN USART 0 Error Interrupt */
#define INT_USART6IRQRX       158         /* LIN USART 6 Receive Interrupt */
#define INT_USART6IRQTX       159         /* LIN USART 6 Transmit Interrupt */
#define INT_USART6IRQERR      160         /* LIN USART 6 Error Interrupt */
#define INT_DMA0IRQD0         164         /* DMA0 Completion Interrupt for channels 0 + 8*n */
#define INT_DMA0IRQD1         165         /* DMA0 Completion Interrupt for channels 1 + 8*n */
#define INT_DMA0IRQD2         166         /* DMA0 Completion Interrupt for channels 2 + 8*n */
#define INT_DMA0IRQD3         167         /* DMA0 Completion Interrupt for channels 3 + 8*n */
#define INT_DMA0IRQD4         168         /* DMA0 Completion Interrupt for channels 4 + 8*n */
#define INT_DMA0IRQD5         169         /* DMA0 Completion Interrupt for channels 5 + 8*n */
#define INT_DMA0IRQD6         170         /* DMA0 Completion Interrupt for channels 6 + 8*n */
#define INT_DMA0IRQD7         171         /* DMA0 Completion Interrupt for channels 7 + 8*n */
#define INT_DMA0IRQERR        172         /* DMA0 Error Interrupt */
#define INT_MSCTIRQ           173         /* Main Source Clock Timer Interrupt */
#define INT_SSCTIRQ           174         /* Sub Source Clock Timer Interrupt */
#define INT_RCSCTIRQ          175         /* RC Source Clock Timer Interrupt */
#define INT_SRCSCTIRQ         176         /* Slow RC Source Clock Timer Interrupt */
#define INT_CORE0IRQ          177         /* CORTEX R4 Performance Monitor Interrupt */
#define INT_RLT0IRQ           178         /* Reload Timer 0 Interrupt */
#define INT_RLT1IRQ           179         /* Reload Timer 1 Interrupt */
#define INT_RLT2IRQ           180         /* Reload Timer 2 Interrupt */
#define INT_RLT3IRQ           181         /* Reload Timer 3 Interrupt */
#define INT_RLT4IRQ           182         /* Reload Timer 4 Interrupt */
#define INT_RLT5IRQ           183         /* Reload Timer 5 Interrupt */
#define INT_RLT6IRQ           184         /* Reload Timer 6 Interrupt */
#define INT_RLT7IRQ           185         /* Reload Timer 7 Interrupt */
#define INT_RLT8IRQ           186         /* Reload Timer 8 Interrupt */
#define INT_RLT9IRQ           187         /* Reload Timer 9 Interrupt */
#define INT_UDC0IRQ0          194         /* Up/Down Counter 0 channel 0 Interrupt */
#define INT_UDC0IRQ1          195         /* Up/Down Counter 0 channel 1 Interrupt */
#define INT_I2S0IRQ           198         /* I2S0 Interrupt */
#define INT_I2S1IRQ           199         /* I2S1 Interrupt */
#define INT_I2C0IRQ           202         /* I2C0 Interrupt */
#define INT_I2C0IRQERR        203         /* I2C0 Error Interrupt */
#define INT_CRC0IRQ           206         /* CRC0 Interrupt */
#define INT_PPG0IRQ           208         /* Programmable Pulse Generator 0 Interrupt */
#define INT_PPG1IRQ           209         /* Programmable Pulse Generator 1 Interrupt */
#define INT_PPG2IRQ           210         /* Programmable Pulse Generator 2 Interrupt */
#define INT_PPG3IRQ           211         /* Programmable Pulse Generator 3 Interrupt */
#define INT_PPG4IRQ           212         /* Programmable Pulse Generator 4 Interrupt */
#define INT_PPG5IRQ           213         /* Programmable Pulse Generator 5 Interrupt */
#define INT_PPG6IRQ           214         /* Programmable Pulse Generator 6 Interrupt */
#define INT_PPG7IRQ           215         /* Programmable Pulse Generator 7 Interrupt */
#define INT_PPG8IRQ           216         /* Programmable Pulse Generator 8 Interrupt */
#define INT_PPG9IRQ           217         /* Programmable Pulse Generator 9 Interrupt */
#define INT_PPG10IRQ          218         /* Programmable Pulse Generator 10 Interrupt */
#define INT_PPG11IRQ          219         /* Programmable Pulse Generator 11 Interrupt */
#define INT_PPG12IRQ          220         /* Programmable Pulse Generator 12 Interrupt */
#define INT_PPG13IRQ          221         /* Programmable Pulse Generator 13 Interrupt */
#define INT_PPG14IRQ          222         /* Programmable Pulse Generator 14 Interrupt */
#define INT_PPG15IRQ          223         /* Programmable Pulse Generator 15 Interrupt */
#define INT_PPG64IRQ          232         /* Programmable Pulse Generator 64 Interrupt */
#define INT_PPG65IRQ          233         /* Programmable Pulse Generator 65 Interrupt */
#define INT_PPG66IRQ          234         /* Programmable Pulse Generator 66 Interrupt */
#define INT_PPG67IRQ          235         /* Programmable Pulse Generator 67 Interrupt */
#define INT_PPG68IRQ          236         /* Programmable Pulse Generator 68 Interrupt */
#define INT_PPG69IRQ          237         /* Programmable Pulse Generator 69 Interrupt */
#define INT_PPG70IRQ          238         /* Programmable Pulse Generator 70 Interrupt */
#define INT_PPG71IRQ          239         /* Programmable Pulse Generator 71 Interrupt */

#define NMI_EIC0NMI             0         /* External Pin NMI */
#define NMI_SYSCNMILVD          1         /* Low Voltage Detect NMI */
#define NMI_SYSCNMIERR          2         /* System Controller Error NMI */
#define NMI_WDGNMI              3         /* Watchdog NMI */
#define NMI_TPU0NMI             4         /* Timing Protection Unit NMI */
#define NMI_MPUXDMA0NMI         5         /* MPU DMA0 Access Violation NMI */
#define NMI_ETH0NMI             6         /* FastMAC 0 NMI */
#define NMI_MPUXGFXNMI          7         /* MPU IRIS Access Violation NMI */
#define NMI_IRQ0NMIERR          9         /* IRQ Double Error NMI */
#define NMI_BECU0NMI           11         /* BECU0 Access Violation NMI */
#define NMI_BECU1NMI           12         /* BECU1 Access Violation NMI */
#define NMI_BECU3NMI           13         /* BECU3 Access Violation NMI */
#define NMI_GFXNMI             14         /* GFX Signature Unit NMI */

#endif    /* __IOMB9EF126_H */
