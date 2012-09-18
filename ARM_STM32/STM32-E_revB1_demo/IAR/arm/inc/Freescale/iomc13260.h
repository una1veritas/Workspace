/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Freescale MC13260
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MC13260_H
#define __MC13260_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MC13260 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C specific declarations  ************************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
#pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 0
#error This file should only be compiled in little endian mode
#endif

/* TSM RX Observability Register (MODEM.TSM_RX_OBS) */
typedef struct {

__REG32 tsm_gpo1                    : 1;
__REG32 tsm_gpo2                    : 1;
__REG32 tsm_gpo3                    : 1;
__REG32 tsm_gpo4                    : 1;
__REG32 tsm_syn_en                  : 1;
__REG32 tsm_mpll_vco_en             : 1;
__REG32 tsm_syn_cal                 : 1;
__REG32 tsm_synth_fifo_en           : 1;
__REG32 tsm_gp_irq                  : 1;
__REG32 tsm_gpadc_trig              : 1;
__REG32 tsm_start_rx                : 1;
__REG32 tsm_rx_en                   : 1;
__REG32 tsm_rx_adc_nmos_filt_stby   : 1;
__REG32 tsm_rx_ana_top              : 1;
__REG32 tsm_rx_adc_en               : 1;
__REG32 tsm_mpll_vco_buf_rx_en      : 1;
__REG32 tsm_rx_adc_dem_reset        : 1;
__REG32 tsm_rx_adc_irst             : 1;
__REG32 tsm_rx_adc_pre              : 1;
__REG32 tsm_rx_bbf_det_en           : 1;
__REG32 tsm_rx_bbf_en               : 1;
__REG32 tsm_rx_dcoc_dac_iq_en       : 1;
__REG32 tsm_rx_qgen_en              : 1;
__REG32 tsm_rx_fifo_en              : 1;
__REG32                             : 8;

} __modem_tsm_rx_obs_bits;

/* TSM TX Observability Register (MODEM.TSM_TX_OBS) */
typedef struct {

__REG32 tsm_gpo1                    : 1;
__REG32 tsm_gpo2                    : 1;
__REG32 tsm_gpo3                    : 1;
__REG32 tsm_gpo4                    : 1;
__REG32 tsm_syn_en                  : 1;
__REG32 tsm_mpll_vco_en             : 1;
__REG32 tsm_syn_cal                 : 1;
__REG32 tsm_synth_fifo_en           : 1;
__REG32 tsm_gp_irq                  : 1;
__REG32 tsm_gpadc_trig              : 1;
__REG32 tsm_start_tx                : 1;
__REG32 tsm_tx_iq_plc_dac_clk_en    : 1;
__REG32 tsm_mpll_vco_buf_tx_en      : 1;
__REG32 tsm_tx_ramp1_trig           : 1;
__REG32 tsm_tx_ramp2_trig           : 1;
__REG32 tsm_tx_ramp3_trig           : 1;
__REG32 tsm_tx_plc_en               : 1;
__REG32 tsm_tx_output_buffer_enable : 1;
__REG32 tsm_tx_out8                 : 1;
__REG32 tsm_tx_out9                 : 1;
__REG32 tsm_tx_out10                : 1;
__REG32 tsm_tx_out11                : 1;
__REG32 tsm_tx_out12                : 1;
__REG32 tsm_tx_out13                : 1;
__REG32                             : 8;

} __modem_tsm_tx_obs_bits;

/* RX Status Register (MODEM.RX_STATUS) */
typedef struct {

__REG32 rx_bbf_det_hi_out           : 1;
__REG32 rx_bbf_det_lo_out           : 1;
__REG32                             :30;

} __modem_rx_status_bits;

/* RFDI Status Register (MODEM.RFDI_STATUS) */
typedef struct {

__REG32 rx_fifo_ov                  : 1;
__REG32 rx_fifo_un                  : 1;
__REG32 tx_fifo_ov                  : 1;
__REG32 tx_fifo_un                  : 1;
__REG32 rfdi_fifo_int               : 1;
__REG32                             :27;

} __modem_rfdi_status_bits;

/* DIGVOCOD Status Register (MODEM.DIGVOCOD_STATUS) */
typedef struct {

__REG32 mic_fifo_empty              : 1;
__REG32 mic_fifo_overflow           : 1;
__REG32 mic_fifo_underflow          : 1;
__REG32                             : 1;
__REG32 spkr_fifo_full              : 1;
__REG32 spkr_fifo_overflow          : 1;
__REG32 spkr_fifo_underflow         : 1;
__REG32                             : 1;
__REG32 int_b                       : 1;
__REG32                             :23;

} __modem_digvocod_status_bits; 

/* Synth Status Register (MODEM.SYNTH_STATUS) */
typedef struct {

__REG32 dig_synth_ld_irq            : 1;
__REG32 dig_synth_fifo_ov_irq       : 1;
__REG32 tune_abort                  : 1;
__REG32 tune_ack                    : 1;
__REG32 tune_start                  : 1;
__REG32                             :11;
__REG32 crstune_vco_divclk_cntr_tst :14;
__REG32                             : 2;

} __modem_synth_status_bits; 

/* RFDI Configuration Register (MODEM.RFDI_CFG) */
typedef struct {

__REG32 filter_mode                 : 5;
__REG32 rx_adc_i_inv                : 1;
__REG32 rx_adc_q_inv                : 1;
__REG32 rx_adc_iqswap               : 1;
__REG32 interp_scale                : 4;
__REG32 adc_5bit_mode               : 1;
__REG32 txiq_en                     : 1;
__REG32 fifo_status_clr             : 1;
__REG32 test_en                     : 1;
__REG32                             :16;

} __modem_rfdi_cfg_bits; 
 
/* RFDI Synthesizer Numerator Adjust Register (MODEM.RFDI_SYNTH_NUM_ADJ) */
typedef struct {

__REG32 num_adj                     :27;
__REG32                             : 5;

} __modem_rfdi_synth_num_adj_bits; 
 
/* Synthesizer Denominator Register (MODEM.SYNTH_DENOM) */
typedef struct {

__REG32 denom                       :27;
__REG32                             : 5;

} __modem_synth_denom_bits; 

/* Synthesizer Integer Register (MODEM.SYNTH_INTEGER) */
typedef struct {

__REG32 intg                        : 7;
__REG32 intg_byp                    : 1;
__REG32 intg_mode_sel               : 1;
__REG32                             :23;

} __modem_synth_integer_bits; 

/* Synthesizer Tune Register (MODEM.SYNTH_TUNE) */
typedef struct {

__REG32 ntune_msb                   :14;
__REG32 store_vco_divclk_cntr       : 1;
__REG32                             :17;

} __modem_synth_tune_bits; 

/* Synthesizer Tune Cycle Config Register (MODEM.SYNTH_TUNE_CYC) */
typedef struct {

__REG32 coarse_state_cycle          : 9;
__REG32 vco_settle_cycle            : 5;
__REG32                             : 2;
__REG32 fine_state_cycle            : 9;
__REG32                             : 7;

} __modem_synth_tune_cyc_bits; 

/* Synthesizer Lock Monitor Register (MODEM.SYNTH_LOCKMON) */
typedef struct {

__REG32 flag_count                  : 2;
__REG32 flag_timeout                : 2;
__REG32 flag_window                 : 2;
__REG32 ld_en                       : 1;
__REG32 ld_int_clr                  : 1;
__REG32 ld_int_en                   : 1;
__REG32 lock_reset                  : 1;
__REG32                             :22;

} __modem_synth_lockmon_bits; 
 
/* Synthesizer Configuration and Test Register (MODEM.SYNTH_CFG_TEST) */
typedef struct {

__REG32 fifo_byp                    : 1;
__REG32 fifo_ov_int_clr             : 1;
__REG32 cntr_en_test                : 1;
__REG32 cntr_reset_test             : 1;
__REG32 ct_test_mode_en             : 1;
__REG32 synvco_test_en              : 1;
__REG32 synvco_kv_test_en           : 1;
__REG32                             : 1;
__REG32 synvco_kv_test              : 6;
__REG32                             : 2;
__REG32 synvco_coarse_test          : 6;
__REG32                             : 2;
__REG32 synvco_fine_test            : 4;
__REG32                             : 4;

} __modem_synth_cfg_test_bits; 

/* Main PLL Configuration Register (MODEM.MPLL_CFG) */
typedef struct {

__REG32 mpll_lfilt_res_spi          : 6;
__REG32 mpll_lfilt_autotune_en      : 1;
__REG32 mpll_lfilt_res_val_select   : 1;
__REG32 mpll_pdet_enable            : 1;
__REG32 mpll_sfilt_en               : 1;
__REG32 mpll_sfilt_fast_charge      : 1;
__REG32 mpll_vco_autotune_en        : 1;
__REG32 mpll_vco_div_count          : 2;
__REG32                             :18;

} __modem_mpll_cfg_bits; 
 
/* Reference PLL Configuration Register 1 (MODEM.RPLL_CFG1) */
typedef struct {

__REG32 rpll_loop_div_count         :14;
__REG32 rpll_loop_div_en            : 1;
__REG32 rpll_adc_dac_div_en         : 1;
__REG32 rpll_adc_dac_div_count      : 7;
__REG32 rpll_ref_div_en             : 1;
__REG32 rpll_ref_div_count          : 8;

} __modem_rpll_cfg1_bits; 
 
/* Reference PLL Configuration Register 2 (MODEM.RPLL_CFG2) */
typedef struct {

__REG32 rpll_reg_sel                : 3;
__REG32 rpll_reg_en                 : 1;
__REG32 rpll_cpump_en               : 1;
__REG32 rpll_sfilt_en               : 1;
__REG32 rpll_sfilt_fast_charge      : 1;
__REG32 rpll_vco_buf_en             : 1;
__REG32 rpll_vco_div_en             : 1;
__REG32                             :23;

} __modem_rpll_cfg2_bits; 
  
/* RX ADC Configuration Register (MODEM.RX_ADC_CFG) */
typedef struct {

__REG32 rx_adc_reg_rx_adc_nmos_filt_en : 1;
__REG32 rx_adc_dem_en                  : 1;
__REG32 rx_adc_quantizer_dith_en       : 1;
__REG32 rx_adc_chop_en                 : 1;
__REG32 rx_adc_dem_dith_en             : 4;
__REG32 rx_adc_iamp1                   : 2;
__REG32 rx_adc_iamp2                   : 2;
__REG32 rx_adc_idac1                   : 2;
__REG32 rx_adc_idac2                   : 2;
__REG32 rx_adc_iflsh                   : 2;
__REG32 rx_adc_itrim                   : 2;
__REG32 rx_adc_clk_chopfreq            : 7;
__REG32 rx_adc_chopfreq_pwr2mode       : 1;
__REG32 rx_adc_i_channel_en            : 1;
__REG32 rx_adc_q_channel_en            : 1;
__REG32                                : 2;

} __modem_rx_adc_cfg_bits; 

/* RX BBF Configuration Register (MODEM.RX_BBF_CFG) */
typedef struct {

__REG32 rx_bbf_chop_rate               : 7;
__REG32 rx_bbf_chop_en                 : 1;
__REG32 rx_bbf_chop_sel                : 2;
__REG32 rx_bbf_chop_pwr2mode           : 1;
__REG32 rx_bbf_fastcharge              : 1;
__REG32 rx_bbf_dcoc_en                 : 1;
__REG32 rx_bbf_dcoc_cal                : 1;
__REG32 rx_bbf_i_en                    : 1;
__REG32 rx_bbf_q_en                    : 1;
__REG32 rx_bbf_det_lvl_lo              : 3;
__REG32 rx_bbf_det_lo_rst              : 1;
__REG32 rx_bbf_det_lvl_hi              : 3;
__REG32 rx_bbf_det_hi_rst              : 1;
__REG32 rx_bbf_in2_agc                 : 3;
__REG32 rx_bbf_ext_bb                  : 1;
__REG32 rx_bbf_reg_en                  : 1;
__REG32                                : 3;

} __modem_rx_bbf_cfg_bits; 
 
/* RX DCOC DAC Data Register (MODEM.RX_DCOC_DAC) */
typedef struct {

__REG32 rx_dcoc_dac_i                  : 7;
__REG32                                : 1;
__REG32 rx_dcoc_dac_q                  : 7;
__REG32                                :17;

} __modem_rx_dcoc_dac_bits; 
 
/* IIP2 DAC Configuration Register (MODEM.IIP2_DAC_CFG) */
typedef struct {

__REG32 iip2_dac_i_sw                  : 8;
__REG32 iip2_dac_q_sw                  : 8;
__REG32 iip2_dac_en                    : 1;
__REG32 ruby_spare_11                  : 1;
__REG32                                :14;

} __modem_iip2_dac_cfg_bits; 

/* RX General Configuration Register (MODEM.RX_GEN_CFG) */
typedef struct {

__REG32 rx_dcoc_dac_i_en               : 1;
__REG32 rx_dcoc_dac_q_en               : 1;
__REG32 rx_tca_curr                    : 2;
__REG32 rx_band                        : 2;
__REG32 rx_agc                         : 5;
__REG32 rx_one_step_agc                : 1;
__REG32 rx_bba_agc                     : 3;
__REG32 rx_mixer_reg_en                : 1;
__REG32 rx_qgen_reg_en                 : 1;
__REG32                                : 7;
__REG32 rx_adc_gp                      : 8;

} __modem_rx_gen_cfg_bits; 
 
/* TX and Spare Configuration Register (MODEM.TX_SPARE_CFG) */
typedef struct {

__REG32 tx_current                     : 2;
__REG32 tx_powerout_adj                : 2;
__REG32                                :15;
__REG32 ruby_spare_4                   : 1;
__REG32 ruby_spare_5                   : 1;
__REG32 ruby_spare_6                   : 1;
__REG32 ruby_spare_7                   : 1;
__REG32 ruby_spare_8                   : 1;
__REG32 ruby_spare_9                   : 1;
__REG32 ruby_spare_10                  : 1;
__REG32                                : 6;

} __modem_tx_spare_cfg_bits; 

/* Analog Codec Configuration Register (MODEM.AVOCOD_CFG) */
typedef struct {

__REG32 codec_pga_audig                : 5;
__REG32 codec_pga_en_vocod_rxpga       : 1;
__REG32                                : 2;
__REG32 codec_pga_audog                : 5;
__REG32 codec_pga_en_vocod_txpga       : 1;
__REG32                                :18;

} __modem_avocod_cfg_bits; 
 
/* DIGVOCOD Configuration Register (MODEM.DIGVOCOD_CFG) */
typedef struct {

__REG32 mic_en                         : 1;
__REG32 mic_fifo_en                    : 1;
__REG32 spkr_en                        : 1;
__REG32 spkr_fifo_en                   : 1;
__REG32 clear_flags                    : 1;
__REG32 codec_loopback_en              : 1;
__REG32 dither_dly                     : 2;
__REG32 mode                           : 5;
__REG32 txiq_mode_en                   : 1;
__REG32 vcdith_dis                     : 1;
__REG32                                :17;

} __modem_digvocod_cfg_bits; 

/* General Purpose Output Register (MODEM.GPO) */
typedef struct {

__REG32 ruby_gpo1                      : 1;
__REG32 ruby_gpo2                      : 1;
__REG32 ruby_gpo3                      : 1;
__REG32 ruby_gpo4                      : 1;
__REG32                                :28;

} __modem_gpo_bits; 

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_0_3) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_0_3  */
  /* IOMUX_SW_MUX_CTL */
  
  struct {

  __REG32 mux_in_0                       : 4;
  __REG32 mux_out_0                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_1                       : 4;
  __REG32 mux_out_1                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_2                       : 4;
  __REG32 mux_out_2                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_3                       : 4;
  __REG32 mux_out_3                      : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_0                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_1                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_2                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_3                   : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
       /* IOMUX_SW_MUX_CTL_0 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_0_byte_bit;
      __REG8 __sw_mux_ctl_0_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_1 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_1_byte_bit;
      __REG8 __sw_mux_ctl_1_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_2 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_2_byte_bit;
      __REG8 __sw_mux_ctl_2_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_3 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_3_byte_bit;
      __REG8 __sw_mux_ctl_3_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_0_3_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_4_7) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_4_7 */
  
  struct {

  __REG32 mux_in_4                       : 4;
  __REG32 mux_out_4                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_5                       : 4;
  __REG32 mux_out_5                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_6                       : 4;
  __REG32 mux_out_6                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_7                       : 4;
  __REG32 mux_out_7                      : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_4                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_5                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_6                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_7                   : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_4 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_4_byte_bit;
      __REG8 __sw_mux_ctl_4_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_5 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_5_byte_bit;
      __REG8 __sw_mux_ctl_5_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_6 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_6_byte_bit;
      __REG8 __sw_mux_ctl_6_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_7 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_7_byte_bit;
      __REG8 __sw_mux_ctl_7_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_4_7_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_8_11) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_8_11 */
  
  struct {

  __REG32 mux_in_8                       : 4;
  __REG32 mux_out_8                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_9                       : 4;
  __REG32 mux_out_9                      : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_10                      : 4;
  __REG32 mux_out_10                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_11                      : 4;
  __REG32 mux_out_11                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_8                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_9                   : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_10                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_11                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_8 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_8_byte_bit;
      __REG8 __sw_mux_ctl_8_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_9 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_9_byte_bit;
      __REG8 __sw_mux_ctl_9_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_10 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_10_byte_bit;
      __REG8 __sw_mux_ctl_10_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_11 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_11_byte_bit;
      __REG8 __sw_mux_ctl_11_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_8_11_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_12_15) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_12_15 */
  
  struct {

  __REG32 mux_in_12                      : 4;
  __REG32 mux_out_12                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_13                      : 4;
  __REG32 mux_out_13                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_14                      : 4;
  __REG32 mux_out_14                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_15                      : 4;
  __REG32 mux_out_15                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_12                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_13                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_14                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_15                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_12 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_12_byte_bit;
      __REG8 __sw_mux_ctl_12_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_13 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_13_byte_bit;
      __REG8 __sw_mux_ctl_13_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_14 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_14_byte_bit;
      __REG8 __sw_mux_ctl_14_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_15 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_15_byte_bit;
      __REG8 __sw_mux_ctl_15_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_12_15_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_16_19) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_16_19 */
  
  struct {

  __REG32 mux_in_16                      : 4;
  __REG32 mux_out_16                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_17                      : 4;
  __REG32 mux_out_17                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_18                      : 4;
  __REG32 mux_out_18                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_19                      : 4;
  __REG32 mux_out_19                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_16                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_17                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_18                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_19                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_16 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_16_byte_bit;
      __REG8 __sw_mux_ctl_16_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_17 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_17_byte_bit;
      __REG8 __sw_mux_ctl_17_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_18 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_18_byte_bit;
      __REG8 __sw_mux_ctl_18_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_19 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_19_byte_bit;
      __REG8 __sw_mux_ctl_19_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_16_19_bits;
  
/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_20_23) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_20_23 */
  
  struct {

  __REG32 mux_in_20                      : 4;
  __REG32 mux_out_20                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_21                      : 4;
  __REG32 mux_out_21                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_22                      : 4;
  __REG32 mux_out_22                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_23                      : 4;
  __REG32 mux_out_23                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_20                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_21                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_22                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_23                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_20 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_20_byte_bit;
      __REG8 __sw_mux_ctl_20_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_21 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_21_byte_bit;
      __REG8 __sw_mux_ctl_21_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_22 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_22_byte_bit;
      __REG8 __sw_mux_ctl_22_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_23 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_23_byte_bit;
      __REG8 __sw_mux_ctl_23_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_20_23_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_24_27) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_24_27 */
  
  struct {

  __REG32 mux_in_24                      : 4;
  __REG32 mux_out_24                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_25                      : 4;
  __REG32 mux_out_25                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_26                      : 4;
  __REG32 mux_out_26                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_27                      : 4;
  __REG32 mux_out_27                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_24                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_25                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_26                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_27                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_24 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_24_byte_bit;
      __REG8 __sw_mux_ctl_24_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_25 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_25_byte_bit;
      __REG8 __sw_mux_ctl_25_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_26 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_26_byte_bit;
      __REG8 __sw_mux_ctl_26_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_27 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_27_byte_bit;
      __REG8 __sw_mux_ctl_27_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_24_27_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_28_31) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_28_31 */
  
  struct {

  __REG32 mux_in_28                      : 4;
  __REG32 mux_out_28                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_29                      : 4;
  __REG32 mux_out_29                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_30                      : 4;
  __REG32 mux_out_30                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_31                      : 4;
  __REG32 mux_out_31                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_28                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_29                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_30                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_31                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_28 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_28_byte_bit;
      __REG8 __sw_mux_ctl_28_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_29 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_29_byte_bit;
      __REG8 __sw_mux_ctl_29_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_30 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_30_byte_bit;
      __REG8 __sw_mux_ctl_30_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_31 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_31_byte_bit;
      __REG8 __sw_mux_ctl_31_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_28_31_bits;

/* 32-bit Software Multiplexer Control Register (IOMUX.SW_MUX_CTL_32_35) */
/* consists of 4 bytes */
typedef union
{

  /* IOMUX_SW_MUX_CTL_32_35 */
  
  struct {

  __REG32 mux_in_32                      : 4;
  __REG32 mux_out_32                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_33                      : 4;
  __REG32 mux_out_33                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_34                      : 4;
  __REG32 mux_out_34                     : 3; 
  __REG32                                : 1;       
  __REG32 mux_in_35                      : 4;
  __REG32 mux_out_35                     : 3; 
  __REG32                                : 1;       
  
  }; 
    
  struct {

  __REG32 SW_MUX_CTL_32                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_33                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_34                  : 7;
  __REG32                                : 1;       
  __REG32 SW_MUX_CTL_35                  : 7;
  __REG32                                : 1;

  }; 


  struct {
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_32 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_32_byte_bit;
      __REG8 __sw_mux_ctl_32_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_33 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_33_byte_bit;
      __REG8 __sw_mux_ctl_33_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_34 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_34_byte_bit;
      __REG8 __sw_mux_ctl_34_byte;   
    };
    
    union {
      union 
      {
        struct 
        {

        __REG8 sw_mux_ctl  : 7;
        __REG8             : 1;       

        };
        /* IOMUX_SW_MUX_CTL_35 */
        struct 
        {
    
        __REG8 mux_in      : 4;  
        __REG8 mux_out     : 3;
        __REG8             : 1;

        };
      } __sw_mux_ctl_35_byte_bit;
      __REG8 __sw_mux_ctl_35_byte;   
    }; 
  };
} __iomux_sw_mux_ctl_32_35_bits;

 
  
/* Software PAD Control Register (IOMUX.SW_PAD_CTL_0 - IOMUX.SW_PAD_CTL_31) */
typedef struct
{
  __REG16 sre                            : 1;
  __REG16 dse                            : 2;       
  __REG16                                : 2;
  __REG16 pus                            : 2;
  __REG16 pke                            : 1;
  __REG16 hys                            : 1;
  __REG16 ode                            : 1;
  __REG16 pue                            : 1;
  __REG16                                : 5;  
} __iomux_sw_pad_ctl_bits; 

/* Interrupt Observability Control Register 0-3 (IOMUX.INT_OBS_0_3) */
typedef struct {

__REG32 int_obs_ap_0                   : 7;
__REG32                                : 1;       
__REG32 int_obs_ap_1                   : 7;
__REG32                                : 1;
__REG32 int_obs_ap_2                   : 7;
__REG32                                : 1;
__REG32 int_obs_ap_3                   : 7;
__REG32                                : 1;

} __iomux_int_obs_0_3_bits; 

/* Interrupt Observability Control Register 4-7 (IOMUX.INT_OBS_4_7) */
typedef struct {

__REG32 int_obs_ap_4                   : 7;
__REG32                                : 1;       
__REG32 int_obs_ap_5                   : 7;
__REG32                                : 1;
__REG32 int_obs_ap_6                   : 7;
__REG32                                : 1;
__REG32 int_obs_ap_7                   : 7;
__REG32                                : 1;

} __iomux_int_obs_4_7_bits; 

/* Interrupt Observability Control Register 8-9 (IOMUX.INT_OBS_8_9) */
typedef struct {

__REG32 int_obs_ap_8                   : 7;
__REG32                                : 1;       
__REG32 int_obs_ap_9                   : 7;
__REG32                                :17;

} __iomux_int_obs_8_9_bits; 
 
/* General Purpose Register (IOMUX.GPR) */
typedef struct {

__REG32 gpio7_ode                      : 1;
__REG32 gpio14_ode                     : 1;       
__REG32 gpio15_ode                     : 1;
__REG32                                :29;

} __iomux_gpr_bits; 
  
/* Abort Control Register (AAPE.ABCNTL) */
typedef struct {

__REG32                                : 8;
__REG32 DAGEN                          : 1;       
__REG32 DAOEN                          : 1;
__REG32                                :15;
__REG32 IAOEN                          : 1;
__REG32                                : 6;

} __aape_abcntl_bits; 
  
/* Abort Status Register (AAPE.ABSR) */
typedef struct {

__REG32 DAB                            : 1;
__REG32                                :15;       
__REG32 IAB                            : 1;
__REG32                                :15;

} __aape_absr_bits; 
 
/* Abort Address Register (AAPE.ABADR) */
typedef struct {

__REG32 DABORTADDR                     :32;

} __aape_abdadr_bits; 

/* Master Privilege Register 1 (AIPS.MPR_1) */
typedef struct {

__REG32 MPL7                           : 1;
__REG32 MTW7                           : 1;       
__REG32 MTR7                           : 1;
__REG32 MBW7                           : 1;
__REG32 MPL6                           : 1;
__REG32 MTW6                           : 1;       
__REG32 MTR6                           : 1;
__REG32 MBW6                           : 1;
__REG32 MPL5                           : 1;
__REG32 MTW5                           : 1;       
__REG32 MTR5                           : 1;
__REG32 MBW5                           : 1;
__REG32 MPL4                           : 1;
__REG32 MTW4                           : 1;       
__REG32 MTR4                           : 1;
__REG32 MBW4                           : 1;
__REG32 MPL3                           : 1;
__REG32 MTW3                           : 1;       
__REG32 MTR3                           : 1;
__REG32 MBW3                           : 1;
__REG32 MPL2                           : 1;
__REG32 MTW2                           : 1;       
__REG32 MTR2                           : 1;
__REG32 MBW2                           : 1;
__REG32 MPL1                           : 1;
__REG32 MTW1                           : 1;       
__REG32 MTR1                           : 1;
__REG32 MBW1                           : 1;
__REG32 MPL0                           : 1;
__REG32 MTW0                           : 1;       
__REG32 MTR0                           : 1;
__REG32 MBW0                           : 1;

} __aips_mpr_1_bits; 

/* Master Privilege Register 2 (AIPS.MPR_2) */
typedef struct {

__REG32 MPL15                          : 1;
__REG32 MTW15                          : 1;       
__REG32 MTR15                          : 1;
__REG32 MBW15                          : 1;
__REG32 MPL14                          : 1;
__REG32 MTW14                          : 1;       
__REG32 MTR14                          : 1;
__REG32 MBW14                          : 1;
__REG32 MPL13                          : 1;
__REG32 MTW13                          : 1;       
__REG32 MTR13                          : 1;
__REG32 MBW13                          : 1;
__REG32 MPL12                          : 1;
__REG32 MTW12                          : 1;       
__REG32 MTR12                          : 1;
__REG32 MBW12                          : 1;
__REG32 MPL11                          : 1;
__REG32 MTW11                          : 1;       
__REG32 MTR11                          : 1;
__REG32 MBW11                          : 1;
__REG32 MPL10                          : 1;
__REG32 MTW10                          : 1;       
__REG32 MTR10                          : 1;
__REG32 MBW10                          : 1;
__REG32 MPL9                           : 1;
__REG32 MTW9                           : 1;       
__REG32 MTR9                           : 1;
__REG32 MBW9                           : 1;
__REG32 MPL8                           : 1;
__REG32 MTW8                           : 1;       
__REG32 MTR8                           : 1;
__REG32 MBW8                           : 1;

} __aips_mpr_2_bits; 

/* (Off-Platform) Peripheral Access Control Register 1 (AIPS.PACR_1, AIPS.OPACR_1) */
typedef struct {

__REG32 TP7                            : 1;
__REG32 WP7                            : 1;
__REG32 SP7                            : 1;
__REG32 BW7                            : 1;
__REG32 TP6                            : 1;
__REG32 WP6                            : 1;
__REG32 SP6                            : 1;
__REG32 BW6                            : 1;
__REG32 TP5                            : 1;
__REG32 WP5                            : 1;
__REG32 SP5                            : 1;
__REG32 BW5                            : 1;
__REG32 TP4                            : 1;
__REG32 WP4                            : 1;
__REG32 SP4                            : 1;
__REG32 BW4                            : 1;
__REG32 TP3                            : 1;
__REG32 WP3                            : 1;
__REG32 SP3                            : 1;
__REG32 BW3                            : 1;
__REG32 TP2                            : 1;
__REG32 WP2                            : 1;
__REG32 SP2                            : 1;
__REG32 BW2                            : 1;
__REG32 TP1                            : 1;
__REG32 WP1                            : 1;
__REG32 SP1                            : 1;
__REG32 BW1                            : 1;
__REG32 TP0                            : 1;
__REG32 WP0                            : 1;
__REG32 SP0                            : 1;
__REG32 BW0                            : 1;

} __aips_pacr_opacr_1_bits; 

/* (Off-Platform) Peripheral Access Control Register 2 (AIPS.PACR_2, AIPS.OPACR_2) */
typedef struct {

__REG32 TP15                           : 1;
__REG32 WP15                           : 1;
__REG32 SP15                           : 1;
__REG32 BW15                           : 1;
__REG32 TP14                           : 1;
__REG32 WP14                           : 1;
__REG32 SP14                           : 1;
__REG32 BW14                           : 1;
__REG32 TP13                           : 1;
__REG32 WP13                           : 1;
__REG32 SP13                           : 1;
__REG32 BW13                           : 1;
__REG32 TP12                           : 1;
__REG32 WP12                           : 1;
__REG32 SP12                           : 1;
__REG32 BW12                           : 1;
__REG32 TP11                           : 1;
__REG32 WP11                           : 1;
__REG32 SP11                           : 1;
__REG32 BW11                           : 1;
__REG32 TP10                           : 1;
__REG32 WP10                           : 1;
__REG32 SP10                           : 1;
__REG32 BW10                           : 1;
__REG32 TP9                            : 1;
__REG32 WP9                            : 1;
__REG32 SP9                            : 1;
__REG32 BW9                            : 1;
__REG32 TP8                            : 1;
__REG32 WP8                            : 1;
__REG32 SP8                            : 1;
__REG32 BW8                            : 1;

} __aips_pacr_opacr_2_bits; 

/* (Off-Platform) Peripheral Access Control Register 3 (AIPS.PACR_3, AIPS.OPACR_3) */
typedef struct {

__REG32 TP23                           : 1;
__REG32 WP23                           : 1;
__REG32 SP23                           : 1;
__REG32 BW23                           : 1;
__REG32 TP22                           : 1;
__REG32 WP22                           : 1;
__REG32 SP22                           : 1;
__REG32 BW22                           : 1;
__REG32 TP21                           : 1;
__REG32 WP21                           : 1;
__REG32 SP21                           : 1;
__REG32 BW21                           : 1;
__REG32 TP20                           : 1;
__REG32 WP20                           : 1;
__REG32 SP20                           : 1;
__REG32 BW20                           : 1;
__REG32 TP19                           : 1;
__REG32 WP19                           : 1;
__REG32 SP19                           : 1;
__REG32 BW19                           : 1;
__REG32 TP18                           : 1;
__REG32 WP18                           : 1;
__REG32 SP18                           : 1;
__REG32 BW18                           : 1;
__REG32 TP17                           : 1;
__REG32 WP17                           : 1;
__REG32 SP17                           : 1;
__REG32 BW17                           : 1;
__REG32 TP16                           : 1;
__REG32 WP16                           : 1;
__REG32 SP16                           : 1;
__REG32 BW16                           : 1;

} __aips_pacr_opacr_3_bits; 

/* (Off-Platform) Peripheral Access Control Register 4 (AIPS.PACR_4, AIPS.OPACR_4) */
typedef struct {

__REG32 TP31                           : 1;
__REG32 WP31                           : 1;
__REG32 SP31                           : 1;
__REG32 BW31                           : 1;
__REG32 TP30                           : 1;
__REG32 WP30                           : 1;
__REG32 SP30                           : 1;
__REG32 BW30                           : 1;
__REG32 TP29                           : 1;
__REG32 WP29                           : 1;
__REG32 SP29                           : 1;
__REG32 BW29                           : 1;
__REG32 TP28                           : 1;
__REG32 WP28                           : 1;
__REG32 SP28                           : 1;
__REG32 BW28                           : 1;
__REG32 TP27                           : 1;
__REG32 WP27                           : 1;
__REG32 SP27                           : 1;
__REG32 BW27                           : 1;
__REG32 TP26                           : 1;
__REG32 WP26                           : 1;
__REG32 SP26                           : 1;
__REG32 BW26                           : 1;
__REG32 TP25                           : 1;
__REG32 WP25                           : 1;
__REG32 SP25                           : 1;
__REG32 BW25                           : 1;
__REG32 TP24                           : 1;
__REG32 WP24                           : 1;
__REG32 SP24                           : 1;
__REG32 BW24                           : 1;

} __aips_pacr_opacr_4_bits; 
 

/* Off-Platform Peripheral Access Control Register 5 (AIPS.OPACR_5) */
typedef struct {

__REG32                                :24;
__REG32 TP33                           : 1;
__REG32 WP33                           : 1;
__REG32 SP33                           : 1;
__REG32 BW33                           : 1;
__REG32 TP32                           : 1;
__REG32 WP32                           : 1;
__REG32 SP32                           : 1;
__REG32 BW32                           : 1; 

} __aips_opacr_5_bits; 

/* Interrupt Control Register (ASIC.INTCNTL) */
typedef struct {

__REG32                                :21;
__REG32 FIDIS                          : 1;
__REG32 NIDIS                          : 1;
__REG32                                : 9;

} __asic_intcntl_bits; 

/* Normal Interrupt Mask Register (ASIC.NIMASK) */
typedef struct {

__REG32 NIMASK                         : 5;
__REG32                                :27;

} __asic_nimask_bits; 

/* Interrupt Enable Number Register (ASIC.INTENNUM) */
typedef struct {

__REG32 ENNUM                          : 6;
__REG32                                :26;

} __asic_intennum_bits; 

/* Interrupt Disable Number Register (ASIC.INTDISNUM) */
typedef struct {

__REG32 DISNUM                          : 6;
__REG32                                 :26;

} __asic_intdisnum_bits; 
 
/* Interrupt Enable Registers (ASIC.INTENABLEH, ASIC.INTENABLEL) */
typedef struct {

__REG32 INTENABLE                       :32; 

} __asic_intenable_bits;
 
/* Interrupt Type Registers (ASIC.INTTYPEH, ASIC.INTTYPEL) */
typedef struct {

__REG32 INTTYPE                         :32; 

} __asic_inttype_bits;

/* Normal Interrupt Prioriry Level Register 7 (ASIC.NIPRIORITY7) */
typedef struct {

__REG32 NIPR56                          : 4; 
__REG32 NIPR57                          : 4;
__REG32 NIPR58                          : 4;
__REG32 NIPR59                          : 4;   
__REG32 NIPR60                          : 4; 
__REG32 NIPR61                          : 4;
__REG32 NIPR62                          : 4;
__REG32 NIPR63                          : 4;   

} __asic_nipriority7_bits; 

/* Normal Interrupt Prioriry Level Register 6 (ASIC.NIPRIORITY6) */
typedef struct {

__REG32 NIPR48                          : 4; 
__REG32 NIPR49                          : 4;
__REG32 NIPR50                          : 4;
__REG32 NIPR51                          : 4;   
__REG32 NIPR52                          : 4; 
__REG32 NIPR53                          : 4;
__REG32 NIPR54                          : 4;
__REG32 NIPR55                          : 4;   

} __asic_nipriority6_bits;  

/* Normal Interrupt Prioriry Level Register 5 (ASIC.NIPRIORITY5) */
typedef struct {

__REG32 NIPR40                          : 4; 
__REG32 NIPR41                          : 4;
__REG32 NIPR42                          : 4;
__REG32 NIPR43                          : 4;   
__REG32 NIPR44                          : 4; 
__REG32 NIPR45                          : 4;
__REG32 NIPR46                          : 4;
__REG32 NIPR47                          : 4;

} __asic_nipriority5_bits; 
 
/* Normal Interrupt Prioriry Level Register 4 (ASIC.NIPRIORITY4) */
typedef struct {

__REG32 NIPR32                          : 4; 
__REG32 NIPR33                          : 4;
__REG32 NIPR34                          : 4;
__REG32 NIPR35                          : 4;   
__REG32 NIPR36                          : 4; 
__REG32 NIPR37                          : 4;
__REG32 NIPR38                          : 4;
__REG32 NIPR39                          : 4;    

} __asic_nipriority4_bits;  

/* Normal Interrupt Prioriry Level Register 3 (ASIC.NIPRIORITY3) */
typedef struct {

__REG32 NIPR24                          : 4; 
__REG32 NIPR25                          : 4;
__REG32 NIPR26                          : 4;
__REG32 NIPR27                          : 4;   
__REG32 NIPR28                          : 4; 
__REG32 NIPR29                          : 4;
__REG32 NIPR30                          : 4;
__REG32 NIPR31                          : 4;

} __asic_nipriority3_bits;  

/* Normal Interrupt Prioriry Level Register 2 (ASIC.NIPRIORITY2) */
typedef struct {

__REG32 NIPR16                          : 4; 
__REG32 NIPR17                          : 4;
__REG32 NIPR18                          : 4;
__REG32 NIPR19                          : 4;   
__REG32 NIPR20                          : 4; 
__REG32 NIPR21                          : 4;
__REG32 NIPR22                          : 4;
__REG32 NIPR23                          : 4;

} __asic_nipriority2_bits; 

/* Normal Interrupt Prioriry Level Register 1 (ASIC.NIPRIORITY1) */
typedef struct {

__REG32 NIPR8                           : 4; 
__REG32 NIPR9                           : 4;
__REG32 NIPR10                          : 4;
__REG32 NIPR11                          : 4;   
__REG32 NIPR12                          : 4; 
__REG32 NIPR13                          : 4;
__REG32 NIPR14                          : 4;
__REG32 NIPR15                          : 4;

} __asic_nipriority1_bits; 

/* Normal Interrupt Prioriry Level Register 0 (ASIC.NIPRIORITY0) */
typedef struct {

__REG32 NIPR0                           : 4; 
__REG32 NIPR1                           : 4;
__REG32 NIPR2                           : 4;
__REG32 NIPR3                           : 4;   
__REG32 NIPR4                           : 4; 
__REG32 NIPR5                           : 4;
__REG32 NIPR6                           : 4;
__REG32 NIPR7                           : 4;

} __asic_nipriority0_bits;

/* Normal Interrupt Vector and Status Register (ASIC.NIVECSR) */
typedef struct {

__REG32 NIPRILVL                        :16;
__REG32 NIVECTOR                        :16;

} __asic_nivecsr_bits;

/* Fast Interrupt Vector and Status Register (ASIC.FIVECSR) */
typedef struct {

__REG32 FIVECTOR                        :32;

} __asic_fivecsr_bits; 

/* Interrupt Source Register High (ASIC.INTSRCH) */
typedef struct
{

  __REG32 INTIN32                         : 1;
  __REG32 INTIN33                         : 1;
  __REG32 INTIN34                         : 1;
  __REG32 INTIN35                         : 1;
  __REG32 INTIN36                         : 1;
  __REG32 INTIN37                         : 1;
  __REG32 INTIN38                         : 1;
  __REG32 INTIN39                         : 1;
  __REG32 INTIN40                         : 1;
  __REG32 INTIN41                         : 1;
  __REG32 INTIN42                         : 1;
  __REG32 INTIN43                         : 1;
  __REG32 INTIN44                         : 1;
  __REG32 INTIN45                         : 1;
  __REG32 INTIN46                         : 1;
  __REG32 INTIN47                         : 1;
  __REG32 INTIN48                         : 1;
  __REG32 INTIN49                         : 1;
  __REG32 INTIN50                         : 1;
  __REG32 INTIN51                         : 1;
  __REG32 INTIN52                         : 1;
  __REG32 INTIN53                         : 1;
  __REG32 INTIN54                         : 1;
  __REG32 INTIN55                         : 1;
  __REG32 INTIN56                         : 1;
  __REG32 INTIN57                         : 1;
  __REG32 INTIN58                         : 1;
  __REG32 INTIN59                         : 1;
  __REG32 INTIN60                         : 1;
  __REG32 INTIN61                         : 1;
  __REG32 INTIN62                         : 1;
  __REG32 INTIN63                         : 1;

} __asic_intsrch_bits;  

/* Interrupt Source Register Low (ASIC.INTSRCL) */
typedef struct
{

  __REG32 INTIN0                          : 1;
  __REG32 INTIN1                          : 1;
  __REG32 INTIN2                          : 1;
  __REG32 INTIN3                          : 1;
  __REG32 INTIN4                          : 1;
  __REG32 INTIN5                          : 1;
  __REG32 INTIN6                          : 1;
  __REG32 INTIN7                          : 1;
  __REG32 INTIN8                          : 1;
  __REG32 INTIN9                          : 1;
  __REG32 INTIN10                         : 1;
  __REG32 INTIN11                         : 1;
  __REG32 INTIN12                         : 1;
  __REG32 INTIN13                         : 1;
  __REG32 INTIN14                         : 1;
  __REG32 INTIN15                         : 1;
  __REG32 INTIN16                         : 1;
  __REG32 INTIN17                         : 1;
  __REG32 INTIN18                         : 1;
  __REG32 INTIN19                         : 1;
  __REG32 INTIN20                         : 1;
  __REG32 INTIN21                         : 1;
  __REG32 INTIN22                         : 1;
  __REG32 INTIN23                         : 1;
  __REG32 INTIN24                         : 1;
  __REG32 INTIN25                         : 1;
  __REG32 INTIN26                         : 1;
  __REG32 INTIN27                         : 1;
  __REG32 INTIN28                         : 1;
  __REG32 INTIN29                         : 1;
  __REG32 INTIN30                         : 1;
  __REG32 INTIN31                         : 1;
 
} __asic_intsrcl_bits;  

/* Interrupt Force Register High (ASIC.INTFRCH) */
typedef struct
{

  __REG32 FORCE32                         : 1;
  __REG32 FORCE33                         : 1;
  __REG32 FORCE34                         : 1;
  __REG32 FORCE35                         : 1;
  __REG32 FORCE36                         : 1;
  __REG32 FORCE37                         : 1;
  __REG32 FORCE38                         : 1;
  __REG32 FORCE39                         : 1;
  __REG32 FORCE40                         : 1;
  __REG32 FORCE41                         : 1;
  __REG32 FORCE42                         : 1;
  __REG32 FORCE43                         : 1;
  __REG32 FORCE44                         : 1;
  __REG32 FORCE45                         : 1;
  __REG32 FORCE46                         : 1;
  __REG32 FORCE47                         : 1;
  __REG32 FORCE48                         : 1;
  __REG32 FORCE49                         : 1;
  __REG32 FORCE50                         : 1;
  __REG32 FORCE51                         : 1;
  __REG32 FORCE52                         : 1;
  __REG32 FORCE53                         : 1;
  __REG32 FORCE54                         : 1;
  __REG32 FORCE55                         : 1;
  __REG32 FORCE56                         : 1;
  __REG32 FORCE57                         : 1;
  __REG32 FORCE58                         : 1;
  __REG32 FORCE59                         : 1;
  __REG32 FORCE60                         : 1;
  __REG32 FORCE61                         : 1;
  __REG32 FORCE62                         : 1;
  __REG32 FORCE63                         : 1;
  
} __asic_intfrch_bits;  

/* Interrupt Force Register Low (ASIC.INTFRCL) */
typedef struct
{

  __REG32 FORCE0                          : 1;
  __REG32 FORCE1                          : 1;
  __REG32 FORCE2                          : 1;
  __REG32 FORCE3                          : 1;
  __REG32 FORCE4                          : 1;
  __REG32 FORCE5                          : 1;
  __REG32 FORCE6                          : 1;
  __REG32 FORCE7                          : 1;
  __REG32 FORCE8                          : 1;
  __REG32 FORCE9                          : 1;
  __REG32 FORCE10                         : 1;
  __REG32 FORCE11                         : 1;
  __REG32 FORCE12                         : 1;
  __REG32 FORCE13                         : 1;
  __REG32 FORCE14                         : 1;
  __REG32 FORCE15                         : 1;
  __REG32 FORCE16                         : 1;
  __REG32 FORCE17                         : 1;
  __REG32 FORCE18                         : 1;
  __REG32 FORCE19                         : 1;
  __REG32 FORCE20                         : 1;
  __REG32 FORCE21                         : 1;
  __REG32 FORCE22                         : 1;
  __REG32 FORCE23                         : 1;
  __REG32 FORCE24                         : 1;
  __REG32 FORCE25                         : 1;
  __REG32 FORCE26                         : 1;
  __REG32 FORCE27                         : 1;
  __REG32 FORCE28                         : 1;
  __REG32 FORCE29                         : 1;
  __REG32 FORCE30                         : 1;
  __REG32 FORCE31                         : 1;

} __asic_intfrcl_bits;  


/* Normal Interrupt Pending Register High (ASIC.NIPNDH) */
typedef struct
{
 
  __REG32 NIPEND32                         : 1;
  __REG32 NIPEND33                         : 1;
  __REG32 NIPEND34                         : 1;
  __REG32 NIPEND35                         : 1;
  __REG32 NIPEND36                         : 1;
  __REG32 NIPEND37                         : 1;
  __REG32 NIPEND38                         : 1;
  __REG32 NIPEND39                         : 1;
  __REG32 NIPEND40                         : 1;
  __REG32 NIPEND41                         : 1;
  __REG32 NIPEND42                         : 1;
  __REG32 NIPEND43                         : 1;
  __REG32 NIPEND44                         : 1;
  __REG32 NIPEND45                         : 1;
  __REG32 NIPEND46                         : 1;
  __REG32 NIPEND47                         : 1;
  __REG32 NIPEND48                         : 1;
  __REG32 NIPEND49                         : 1;
  __REG32 NIPEND50                         : 1;
  __REG32 NIPEND51                         : 1;
  __REG32 NIPEND52                         : 1;
  __REG32 NIPEND53                         : 1;
  __REG32 NIPEND54                         : 1;
  __REG32 NIPEND55                         : 1;
  __REG32 NIPEND56                         : 1;
  __REG32 NIPEND57                         : 1;
  __REG32 NIPEND58                         : 1;
  __REG32 NIPEND59                         : 1;
  __REG32 NIPEND60                         : 1;
  __REG32 NIPEND61                         : 1;
  __REG32 NIPEND62                         : 1;
  __REG32 NIPEND63                         : 1;
   
} __asic_nipndh_bits;  

/* Normal Interrupt Pending Register Low (ASIC.NIPNDL) */
typedef struct
{

  __REG32 NIPEND0                          : 1;
  __REG32 NIPEND1                          : 1;
  __REG32 NIPEND2                          : 1;
  __REG32 NIPEND3                          : 1;
  __REG32 NIPEND4                          : 1;
  __REG32 NIPEND5                          : 1;
  __REG32 NIPEND6                          : 1;
  __REG32 NIPEND7                          : 1;
  __REG32 NIPEND8                          : 1;
  __REG32 NIPEND9                          : 1;
  __REG32 NIPEND10                         : 1;
  __REG32 NIPEND11                         : 1;
  __REG32 NIPEND12                         : 1;
  __REG32 NIPEND13                         : 1;
  __REG32 NIPEND14                         : 1;
  __REG32 NIPEND15                         : 1;
  __REG32 NIPEND16                         : 1;
  __REG32 NIPEND17                         : 1;
  __REG32 NIPEND18                         : 1;
  __REG32 NIPEND19                         : 1;
  __REG32 NIPEND20                         : 1;
  __REG32 NIPEND21                         : 1;
  __REG32 NIPEND22                         : 1;
  __REG32 NIPEND23                         : 1;
  __REG32 NIPEND24                         : 1;
  __REG32 NIPEND25                         : 1;
  __REG32 NIPEND26                         : 1;
  __REG32 NIPEND27                         : 1;
  __REG32 NIPEND28                         : 1;
  __REG32 NIPEND29                         : 1;
  __REG32 NIPEND30                         : 1;
  __REG32 NIPEND31                         : 1;
  
} __asic_nipndl_bits;  


/* Fast Interrupt Pending Register High (ASIC.FIPNDH) */
typedef struct
{
 
  __REG32 FIPEND32                         : 1;
  __REG32 FIPEND33                         : 1;
  __REG32 FIPEND34                         : 1;
  __REG32 FIPEND35                         : 1;
  __REG32 FIPEND36                         : 1;
  __REG32 FIPEND37                         : 1;
  __REG32 FIPEND38                         : 1;
  __REG32 FIPEND39                         : 1;
  __REG32 FIPEND40                         : 1;
  __REG32 FIPEND41                         : 1;
  __REG32 FIPEND42                         : 1;
  __REG32 FIPEND43                         : 1;
  __REG32 FIPEND44                         : 1;
  __REG32 FIPEND45                         : 1;
  __REG32 FIPEND46                         : 1;
  __REG32 FIPEND47                         : 1;
  __REG32 FIPEND48                         : 1;
  __REG32 FIPEND49                         : 1;
  __REG32 FIPEND50                         : 1;
  __REG32 FIPEND51                         : 1;
  __REG32 FIPEND52                         : 1;
  __REG32 FIPEND53                         : 1;
  __REG32 FIPEND54                         : 1;
  __REG32 FIPEND55                         : 1;
  __REG32 FIPEND56                         : 1;
  __REG32 FIPEND57                         : 1;
  __REG32 FIPEND58                         : 1;
  __REG32 FIPEND59                         : 1;
  __REG32 FIPEND60                         : 1;
  __REG32 FIPEND61                         : 1;
  __REG32 FIPEND62                         : 1;
  __REG32 FIPEND63                         : 1;
   
} __asic_fipndh_bits;  

/* Fast Interrupt Pending Register Low (ASIC.FIPNDL) */
typedef struct
{

  __REG32 FIPEND0                          : 1;
  __REG32 FIPEND1                          : 1;
  __REG32 FIPEND2                          : 1;
  __REG32 FIPEND3                          : 1;
  __REG32 FIPEND4                          : 1;
  __REG32 FIPEND5                          : 1;
  __REG32 FIPEND6                          : 1;
  __REG32 FIPEND7                          : 1;
  __REG32 FIPEND8                          : 1;
  __REG32 FIPEND9                          : 1;
  __REG32 FIPEND10                         : 1;
  __REG32 FIPEND11                         : 1;
  __REG32 FIPEND12                         : 1;
  __REG32 FIPEND13                         : 1;
  __REG32 FIPEND14                         : 1;
  __REG32 FIPEND15                         : 1;
  __REG32 FIPEND16                         : 1;
  __REG32 FIPEND17                         : 1;
  __REG32 FIPEND18                         : 1;
  __REG32 FIPEND19                         : 1;
  __REG32 FIPEND20                         : 1;
  __REG32 FIPEND21                         : 1;
  __REG32 FIPEND22                         : 1;
  __REG32 FIPEND23                         : 1;
  __REG32 FIPEND24                         : 1;
  __REG32 FIPEND25                         : 1;
  __REG32 FIPEND26                         : 1;
  __REG32 FIPEND27                         : 1;
  __REG32 FIPEND28                         : 1;
  __REG32 FIPEND29                         : 1;
  __REG32 FIPEND30                         : 1;
  __REG32 FIPEND31                         : 1;
  
} __asic_fipndl_bits;  

/* ASM 128-bit Encryption Registers (ASM_KEYx) */
typedef struct {

__REG32 KEY_DATA                        :32;

} __asm_key_bits;   

/* ASM 128-bit Counter Registers (ASM_CTRx) */
typedef struct {

__REG32 CTR_DATA                        :32;

} __asm_ctr_bits;   

/* ASM 128-bit Data Registers (ASM_DATAx) */
typedef struct {

__REG32 DATA                            :32;

} __asm_data_bits;

/* ASM Control 0 Register (ASM_CONTROL0) */
typedef struct {

__REG32                                 :24;
__REG32 START                           : 1;
__REG32 CLEAR                           : 1;
__REG32 LOAD_MAC                        : 1;
__REG32                                 : 4;
__REG32 CLEAR_IRQ                       : 1;

} __asm_control0_bits;

/* ASM Control 1 Register (ASM_CONTROL1) */
typedef struct {

__REG32 ON                              : 1;
__REG32 NORMAL_MODE                     : 1;
__REG32 BYPASS                          : 1;
__REG32 TX_DMAREQ_EN                    : 1;
__REG32 AUTOSTART_EN                    : 1;
__REG32                                 :19;
__REG32 CBC                             : 1;
__REG32 CTR                             : 1;
__REG32 SELF_TEST                       : 1;
__REG32 ECB                             : 1;
__REG32 RX_DMAREQ_EN                    : 1;
__REG32                                 : 2;
__REG32 MASK_IRQ                        : 1;

} __asm_control1_bits; 
 
/* ASM Status Register (ASM_STATUS) */
typedef struct {

__REG32                                 :24;
__REG32 DONE                            : 1;
__REG32 TEST_PASS                       : 1;
__REG32                                 : 6;

} __asm_status_bits;  

/* ASM 128-Bit AES Result Registers (ASM_AESx_RESULT) */
typedef struct {

__REG32 AES_RESULT_DATA                 :32;

} __asm_aes_result_bits; 

/* ASM 128-Bit Counter Result Registers (ASM_CTRx_RESULT) */
typedef struct {

__REG32 CTR_RESULT_DATA                 :32;

} __asm_ctr_result_bits; 

/* ASM 128-Bit CBC-MAC Registers (ASM_MACx) */
typedef struct {

__REG32 MAC_START_DATA                 :32;

} __asm_mac_bits; 

/* AP Source Clock Selection Register (CCM.ASCSR) */
typedef struct {

__REG32 CKIL_SEL                       : 1;
__REG32                                : 1;
__REG32 UART1_SEL                      : 1;
__REG32                                : 1;
__REG32 UART2_SEL                      : 1;
__REG32                                : 1;
__REG32 SSI_SEL                        : 1;
__REG32                                : 1;
__REG32 VOCOD_SEL                      : 1;
__REG32                                : 1;
__REG32 USB_SEL                        : 1;
__REG32                                : 1;
__REG32 TPRC_SEL                       : 1;
__REG32                                : 1;
__REG32 TPRC2_SEL                      : 1;
__REG32                                : 1;
__REG32 TPRC3_SEL                      : 1;
__REG32                                : 1;
__REG32 MOD_SEL                        : 1;
__REG32                                : 9;
__REG32 MOD_BYP_SEL                    : 1;
__REG32 USB_BYP_SEL                    : 1;
__REG32 ACC_BYP_SEL                    : 1;
__REG32 AP_BYP_SEL                     : 1;

} __ccm_ascsr_bits;  

/* AP Clock Selection Register (CCM.ACSR) */
typedef struct {

__REG32 ACS                            : 1;
__REG32 WRS                            : 1;
__REG32 PDS                            : 1;
__REG32 SMD                            : 1;
__REG32                                :28;

} __ccm_acsr_bits;  

/* AP Clock Divider Register 1 (CCM.ACDR1) */
typedef struct {

__REG32 IP_DIV                         : 2;
__REG32                                : 2;
__REG32 AHB_DIV                        : 2;
__REG32                                : 2;
__REG32 AP_DIV                         : 3;
__REG32                                : 1;
__REG32 MODEM_DIV                      : 3;
__REG32                                : 1;
__REG32 CKIH_DIV                       : 3;
__REG32                                : 1;
__REG32 SKIL_DIV                       : 2;
__REG32                                : 1;
__REG32 TUNE_DIV                       : 1;
__REG32                                : 4;
__REG32 TPRC_DIV                       : 4;

} __ccm_acdr1_bits;   

/* AP Clock Divider Register 2 (CCM.ACDR2) */
typedef struct {

__REG32 UART1_DIV                      : 3;
__REG32                                : 1;
__REG32 UART2_DIV                      : 3;
__REG32                                : 1;
__REG32 SSI_DIV                        : 4;
__REG32                                : 1;
__REG32 USB_DIV                        : 3;
__REG32 TPRC2_DIV                      : 4;
__REG32                                : 4;
__REG32 TPRC3_DIV                      : 4;
__REG32                                : 4;

} __ccm_acdr2_bits;  

/* AP Clock Gate Control Register (CCM.ACGCR) */
typedef struct {

__REG32 MODEM_EN                       : 1;
__REG32 UART1_EN                       : 1;
__REG32 UART2_EN                       : 1;
__REG32 SSI_EN                         : 1;
__REG32 I2C_EN                         : 1;
__REG32 USB_EN                         : 1;
__REG32                                : 1;
__REG32 ARM_CE                         : 1;
__REG32                                : 1;
__REG32 TMR_EN                         : 1;
__REG32                                : 5;
__REG32 GPAD_EN                        : 1;
__REG32                                :16;

} __ccm_acgcr_bits;  

/* Low Power Clock Gate Register 0 (CCM.LPCGR0) */
typedef struct {

__REG32 LPCGR00                        : 2;
__REG32 LPCGR01                        : 2;
__REG32 LPCGR02                        : 2;
__REG32 LPCGR03                        : 2;
__REG32 LPCGR04                        : 2;
__REG32                                : 6;
__REG32 LPCGR05                        : 2;
__REG32 LPCGR06                        : 2;
__REG32 LPCGR07                        : 2;
__REG32 LPCGR08                        : 2;
__REG32 LPCGR09                        : 2;
__REG32                                : 6;

} __ccm_lpcgr0_bits;   

/* Low Power Clock Gate Register 1 (CCM.LPCGR1) */
typedef struct {

__REG32 LPCGR10                        : 2;
__REG32 LPCGR11                        : 2;
__REG32 LPCGR12                        : 2;
__REG32 LPCGR13                        : 2;
__REG32 LPCGR14                        : 2;
__REG32                                : 6;
__REG32 LPCGR15                        : 2;
__REG32 LPCGR16                        : 2;
__REG32 LPCGR17                        : 2;
__REG32 LPCGR18                        : 2;
__REG32 LPCGR19                        : 2;
__REG32                                : 6;

} __ccm_lpcgr1_bits;   

/* Low Power Clock Gate Register 2 (CCM.LPCGR2) */
typedef struct {

__REG32 LPCGR20                        : 2;
__REG32 LPCGR21                        : 2;
__REG32 LPCGR22                        : 2;
__REG32 LPCGR23                        : 2;
__REG32 LPCGR24                        : 2;
__REG32                                : 6;
__REG32 LPCGR25                        : 2;
__REG32 LPCGR26                        : 2;
__REG32 LPCGR27                        : 2;
__REG32 LPCGR28                        : 2;
__REG32 LPCGR29                        : 2;
__REG32                                : 6;

} __ccm_lpcgr2_bits;   

/* Low Power Clock Gate Register 3 (CCM.LPCGR3) */
typedef struct {

__REG32 LPCGR30                        : 2;
__REG32 LPCGR31                        : 2;
__REG32 LPCGR32                        : 2;
__REG32 LPCGR33                        : 2;
__REG32 LPCGR34                        : 2;
__REG32                                : 6;
__REG32 LPCGR35                        : 2;
__REG32 LPCGR36                        : 2;
__REG32 LPCGR37                        : 2;
__REG32 LPCGR38                        : 2;
__REG32 LPCGR39                        : 2;
__REG32                                : 6;

} __ccm_lpcgr3_bits;   

/* Low Power Clock Gate Register 4 (CCM.LPCGR4) */
typedef struct {

__REG32 LPCGR40                        : 2;
__REG32 LPCGR41                        : 2;
__REG32 LPCGR42                        : 2;
__REG32 LPCGR43                        : 2;
__REG32 LPCGR44                        : 2;
__REG32                                : 6;
__REG32 LPCGR45                        : 2;
__REG32 LPCGR46                        : 2;
__REG32 LPCGR47                        : 2;
__REG32 LPCGR48                        : 2;
__REG32 LPCGR49                        : 2;
__REG32                                : 6;

} __ccm_lpcgr4_bits;   

/* Low Power Clock Gate Register 5 (CCM.LPCGR5) */
typedef struct {

__REG32 LPCGR50                        : 2;
__REG32 LPCGR51                        : 2;
__REG32 LPCGR52                        : 2;
__REG32 LPCGR53                        : 2;
__REG32 LPCGR54                        : 2;
__REG32                                : 6;
__REG32 LPCGR55                        : 2;
__REG32 LPCGR56                        : 2;
__REG32 LPCGR57                        : 2;
__REG32 LPCGR58                        : 2;
__REG32 LPCGR59                        : 2;
__REG32                                : 6;

} __ccm_lpcgr5_bits;   

/* AP Module_en Override Register A (CCM.AMORA) */
typedef struct {

__REG32 MOA                            :32;

} __ccm_amora_bits; 
 
/* AP Module_en Override Register B (CCM.AMORB) */
typedef struct {

__REG32 MOB                            :32;

} __ccm_amorb_bits; 

/* AP Perclk Override Register (CCM.APOR) */
typedef struct {

__REG32 UART1_OVR                      : 1;
__REG32 UART2_OVR                      : 1;
__REG32 TUNE_OVR                       : 1;
__REG32 LOCK_MON_OVR                   : 1;
__REG32 TSM_OVR                        : 1;
__REG32 VOCOD_OVR                      : 1;
__REG32 RFDI_OVR                       : 1;
__REG32 TPRC_OVR                       : 1;
__REG32 TPRC2_OVR                      : 1;
__REG32 TPRC3_OVR                      : 1;
__REG32                                :22;

} __ccm_apor_bits;  

/* AP CKO(H) Register (CCM.ACR) */
typedef struct {

__REG32 CKOS                           : 5;
__REG32                                : 1;
__REG32 CKOD                           : 1;
__REG32 CKOHDIV                        : 3;
__REG32 CKOHS                          : 5;
__REG32 CKOHD                          : 1;
__REG32                                :16;

} __ccm_acr_bits;  
 
/* AP Miscellaneous Control Register (CCM.AMCR) */
typedef struct {

__REG32 HDMA_ACK_BYP                   : 1;
__REG32                                : 2;
__REG32 MAX_ACK_BYP                    : 1;
__REG32                                : 1;
__REG32 ACK_BYP_RNGB                   : 1;
__REG32                                : 5;
__REG32 PLC_MODE                       : 2;
__REG32                                : 1;
__REG32 INT_HOLD_OFF                   : 1;
__REG32 CSB                            : 1;
__REG32 lpmd_mode                      : 2;
__REG32                                : 1;
__REG32 OSC_EN                         : 1;
__REG32 OSC32_BYP                      : 1;
__REG32                                : 3;
__REG32 TSM_AB                         : 1;
__REG32 TPRC3_AB                       : 1;
__REG32 TPRC2_AB                       : 1;
__REG32 TPRC_AB                        : 1;
__REG32 TSM_ASYNC                      : 1;
__REG32 TPRC3_ASYNC                    : 1;
__REG32 TPRC2_ASYNC                    : 1;
__REG32 TPRC_ASYNC                     : 1;

} __ccm_amcr_bits;  
 
/* Analog Interface Register 1 (CCM.AIR1) */
typedef struct {

__REG32 mpll_ref_div_ref_sel              : 1;
__REG32 rx_adc_12mhz_en                   : 1;
__REG32 rx_adc_16mhz_en                   : 1;
__REG32 rx_adc_16p8mhz_en                 : 1;
__REG32 rx_adc_9p6mhz_en                  : 1;
__REG32 tcxo_codec_ana_clk_en             : 1;
__REG32 tcxo_dist_codec_clk_div_sel       : 1;
__REG32 tcxo_dist_dac1_mux_sel            : 1;
__REG32 tcxo_dist_dac2_mux_sel            : 1;
__REG32 tcxo_dist_clk_dist_atst_en        : 1;
__REG32 tcxo_dist_rxadc_div_sel           : 2;
__REG32                                   : 1;
__REG32 rx_adc_clk_en                     : 1;
__REG32 tcxo_dist_codec_dig_clk_en        : 1;
__REG32 tcxo_dist_codec_div_sel           : 1;
__REG32 tcxo_dist_codec_rxadc_dividers_en : 1;
__REG32 tcxo_dist_dac1_clk_en             : 1;
__REG32 tcxo_dist_dac2_clk_en             : 1;
__REG32                                   : 2;
__REG32 tcxo_dist_rxfe_chop_clk_div_sel   : 3;
__REG32                                   : 1;
__REG32 gpio_ana_en                       : 1;
__REG32 tcxo_dist_rxfe_clk_en             : 1;
__REG32 tcxo_dist_rxadc_dig_clk_en        : 1;
__REG32 tcxo_dist_rxadc_clk_sel           : 1;
__REG32 tcxo_dist_rxadc_clk_en            : 1;
__REG32 tcxo_dist_resetb                  : 1;
__REG32 tcxo_dist_main_ref_pll_clk_en     : 1;

} __ccm_air1_bits;  
 
/* Analog Interface Register 2 (CCM.AIR2) */
typedef struct {

__REG32 anatest_atst_atst_en              : 1;
__REG32 anatest_atst_en                   : 1;
__REG32 anatest_atst_loopback1_en         : 1;
__REG32 anatest_atst_loopback2_en         : 1;
__REG32 anatest_atst_mode_sel1            : 2;
__REG32 anatest_atst_mode_sel2            : 2;
__REG32 anatest_atst_mode_sel3            : 2;
__REG32 anatest_atst_mode_sel4            : 2;
__REG32 anatest_atst_opamp1_en            : 1;
__REG32 anatest_atst_opamp2_en            : 1;
__REG32 anatest_atst_opamp3_en            : 1;
__REG32 anatest_atst_opamp4_en            : 1;
__REG32 anatest_atst_sel                  : 5;
__REG32 avocod_anatest_voc_pac            : 9;
__REG32 bgap_atst_sel                     : 2;

} __ccm_air2_bits;   

/* Analog Interface Register 3 (CCM.AIR3) */
typedef struct {

__REG32 codec_pga_seltest_rxpga           : 1;
__REG32 codec_pga_seltest_txpga           : 1;
__REG32 mpll_loop_div_reset               : 1;
__REG32 mpll_ref_div_reset                : 1;
__REG32                                   : 2;
__REG32 mpll_tmux_select                  : 4;
__REG32                                   : 1;
__REG32 mpll_vco_div_reset                : 1;
__REG32 reg_2p4_anatest_sel_2p4           : 4;
__REG32 reg1p2_anatest_reg_1p2_en         : 1;
__REG32 reg1p2_anatest_sel                : 4;
__REG32 reg1p4_anatest_reg_1p4en          : 1;
__REG32 reg1p4_anatest_sel                : 4;
__REG32 reg1p8_anatest_regul_aux_en       : 1;
__REG32 reg1p8_anatest_sel                : 4;
__REG32 rpll_loop_div_reset               : 1;

} __ccm_air3_bits;   

/* Analog Interface Register 4 (CCM.AIR4) */
typedef struct {

__REG32 rpll_pdet_reset                   : 1;
__REG32 rpll_ref_div_reset                : 1;
__REG32 rpll_tmux_select                  : 7;
__REG32 rx_adc_test_cntl                  : 2;
__REG32 rx_adc_test_en                    : 1;
__REG32 rx_adc_tst_en                     : 1;
__REG32 rx_atst_en                        : 1;
__REG32 bgap_atst_bgap_ref_en             : 1;
__REG32                                   : 1;
__REG32 rx_atst_sel                       : 4;
__REG32 tempsens_seltest                  : 2;
__REG32 vagreg_seltestagndgen             : 2;
__REG32 avocod_anatest_voc                : 2;
__REG32 clk_dist_atst_en                  : 1;
__REG32 clk_dist_atst_sel                 : 5;

} __ccm_air4_bits;    

/* Analog Interface Register 5 (CCM.AIR5) */
typedef struct {

__REG32 spare1                            : 1;
__REG32 spare2                            : 1;
__REG32 spare3                            : 1;
__REG32 spare4                            : 1;
__REG32 spare5                            : 1;
__REG32                                   :27;

} __ccm_air5_bits;     

/* Analog Interface Register 6 (CCM.AIR6) */
typedef struct {

__REG32 air_reg6                          :32;

} __ccm_air6_bits;     

/* Analog Interface Register 7 (CCM.AIR7) */
typedef struct {

__REG32 air_reg7                          :32;

} __ccm_air7_bits;     

/* Analog Interface Register 8 (CCM.AIR8) */
typedef struct {

__REG32 pacdac1_test_mode                 : 7;
__REG32 pacdac2_test_mode                 : 7;
__REG32                                   : 2;
__REG32 pacdac3_test_mode                 : 7;
__REG32                                   : 7;
__REG32 rx_adc_bbf_bypass_testmode        : 1;
__REG32                                   : 1;

} __ccm_air8_bits;     
 
/* Analog Interface Register 9 (CCM.AIR9) */
typedef struct {

__REG32 mpll_tmux_en                          : 3;
__REG32                                       : 2;
__REG32 tcxo_dist_atst_sel                    : 4;
__REG32                                       : 1;
__REG32 tcxo_dist_main_ref_pll_drive_strength : 2;
__REG32                                       : 1;
__REG32 rpll_rpll_out_sel                     : 2;
__REG32                                       : 1;
__REG32 gpadc_seltest_rsd_adc                 : 9;
__REG32                                       : 1;
__REG32 gpadc_seltest_rsd_inpcond             : 4;
__REG32                                       : 2;

} __ccm_air9_bits;     
 
/* Analog Interface Register 10 (CCM.AIR10) */
typedef struct {

__REG32 rx_adc_cap_reset                      : 1;
__REG32 rx_adc_cap_tune_override_en           : 1;
__REG32 rx_adc_tune_cap_en                    : 1;
__REG32 rx_adc_cap_adj_aftercal               : 4;
__REG32                                       : 1;
__REG32 rx_adc_tune_cap_hys_h                 : 8;
__REG32 rx_adc_tune_cap_hys_l                 : 8;
__REG32 rx_adc_cap_aftercal_save              : 4;
__REG32 tcxo_dist_reg1p4_anatest_sel          : 4;

} __ccm_air10_bits;     
 
/* Freescale Internal Use Register 1 (CCM.FIUR1) */
typedef struct {

__REG32 FIUR1_0                               :16;
__REG32 FIUR1_1                               :16;

} __ccm_fiur1_bits;      

/* Freescale Internal Use Register 2 (CCM.FIUR2) */
typedef struct {

__REG32 FIUR2_0                               : 1;
__REG32                                       :31;

} __ccm_fiur2_bits;      

/* Dither Control Register (CCM.DCR) */
typedef struct {

__REG32 AD_EN                                 : 1;
__REG32 AD_DIV                                : 3;
__REG32                                       : 1;
__REG32 AD_CNT                                : 3;
__REG32 MD_EN                                 : 1;
__REG32 MD_DIV                                : 3;
__REG32                                       : 1;
__REG32 MD_CNT                                : 3;
__REG32                                       :16;

} __ccm_dcr_bits;      

/* AP General Purpose Register (CCM.AGPR) */
typedef struct {

__REG32 AGPR0                                 : 1;
__REG32 AGPR1                                 : 1;
__REG32 AGPR2                                 : 1;
__REG32 AGPR3                                 : 1;
__REG32 AGPR4                                 : 1;
__REG32 AGPR5                                 : 1;
__REG32                                       :26;

} __ccm_agpr_bits; 

/* Control Register (CSPI.CONREG) */
typedef struct{

__REG32 EN          : 1;
__REG32 MODE        : 1;
__REG32 XCH         : 1;
__REG32 SMC         : 1;
__REG32 POL         : 1;
__REG32 PHA         : 1;
__REG32 SSCTL       : 1;
__REG32 SSPOL       : 1;
__REG32 DRCTL       : 2;
__REG32             : 2;
__REG32 CHIP_SELECT : 2;
__REG32             : 2;
__REG32 DATA_RATE   : 3;
__REG32             : 1;
__REG32 BURST_LENGTH:12;

} __cspi_conreg_bits; 

/* Interrupt Control Register (CSPI.INTREG) */
typedef struct{

__REG32 TEEN     : 1;
__REG32 THEN     : 1;
__REG32 TFEN     : 1;
__REG32 RREN     : 1;
__REG32 RHEN     : 1;
__REG32 RFEN     : 1;
__REG32 ROEN     : 1;
__REG32 TCEN     : 1;
__REG32          :24;

} __cspi_intreg_bits;

/* DMA Control Register (CSPI.DMAREG) */
typedef struct{

__REG32 TEDEN  : 1;
__REG32 THDEN  : 1;
__REG32        : 2;
__REG32 RHDEN  : 1;
__REG32 RFDEN  : 1;
__REG32        :26;

} __cspi_dmareg_bits;

/* Status Register (CSPI.STATREG) */
typedef struct{

__REG32 TE      : 1;
__REG32 TH      : 1;
__REG32 TF      : 1;
__REG32 RR      : 1;
__REG32 RH      : 1;
__REG32 RF      : 1;
__REG32 RO      : 1;
__REG32 TC      : 1;
__REG32         :24;

} __cspi_statreg_bits;
 
/* Sample Period Control Register (CSPI.PERIODREG) */
typedef struct{

__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32               :16;

} __cspi_periodreg_bits;

/* Test Control Register (CSPI.TESTREG) */
typedef struct{

__REG32 TXCNT     : 4;
__REG32 RXCNT     : 4;
__REG32           : 6;
__REG32 LBC       : 1;
__REG32 SWAP      : 1;
__REG32           :16;

} __cspi_testreg_bits; 

/* DPLL Control Register(DPLL.DP_CTL) */
typedef struct{

__REG32 LRF         : 1;
__REG32 BRMO        : 1;
__REG32 PLM         : 1;
__REG32 RCP         : 1;
__REG32 RST         : 1;
__REG32 UPEN        : 1;
__REG32 PRE         : 1;
__REG32 HFSM        : 1;
__REG32 ref_clk_sel : 2;
__REG32 ref_clk_div : 1;
__REG32 ADE         : 1;
__REG32 dpdck0_2_en : 1;
__REG32 mul_ctrl    : 1;
__REG32             :18;

} __dpll_dp_ctl_bits;

/* DPLL Config Register(DPLL.DP_CONFIG) */
typedef struct{

__REG32 LDREQ       : 1;
__REG32 AREN        : 1;
__REG32 SJC_CE      : 1;
__REG32 BIST_CE     : 1;
__REG32             :28;

} __dpll_dp_config_bits;

/* DPLL Operation Register(DPLL.DP_OP) */
typedef struct{

__REG32 PDF         : 4;
__REG32 MFI         : 4;
__REG32             :24;

} __dpll_dp_op_bits;

/* DPLL Multiplication Factor Denominator Register(DPLL.DP_MFD) */
typedef struct{

__REG32 MFD         :27;
__REG32             : 5;

} __dpll_dp_mfd_bits;

/* DPLL MFNxxx Register (DPLL.DP_MFN) */
typedef struct{

__REG32 MFN         :27;
__REG32             : 5;

} __dpll_dp_mfn_bits;

/* DPLL MFNxxx Register (DPLL.DP_MFNMINUS) */
typedef struct{

__REG32 MFNMINUS    :27;
__REG32             : 5;

} __dpll_dp_mfnminus_bits;

/* DPLL MFNxxx Register (DPLL.DP_MFNPLUS) */
typedef struct{

__REG32 MFNPLUS     :27;
__REG32             : 5;

} __dpll_dp_mfnplus_bits;

/* DPLL High Frequency Support, Operation Register(DPLL.DP_HFS_OP) */
typedef struct{

__REG32 HFS_PDF     : 4;
__REG32 HFS_MFI     : 4;
__REG32             :24;

} __dpll_dp_hfs_op_bits;

/* DPLL HFS MFD Register (DPLL.DP_HFS_MFD) */
typedef struct{

__REG32 HFS_MFD     :27;
__REG32             : 5;

} __dpll_dp_hfs_mfd_bits;

/* DPLL HFS Multiplication Factor Numerator Register (DPLL.DP_HFS_MFN) */
typedef struct{

__REG32 HFS_MFN     :27;
__REG32             : 5;

} __dpll_dp_hfs_mfn_bits;

/* DPLL Multiplication Factor Numerator Toggle Control Register (DPLL.DP_MFN_TOGC) */
typedef struct{

__REG32 TOG_MFN_CNT :16;
__REG32 TOG_EN      : 1;
__REG32 TOG_DIS     : 1;
__REG32             :14;

} __dpll_dp_mfn_togc_bits;

/* Desense Status Register(DPLL.DP_DESTAT) */
typedef struct{

__REG32 TOG_MFN     :27;
__REG32             : 4;
__REG32 TOG_SEL     : 1;

} __dpll_dp_destat_bits;

/* Timebase Counter Register (DSM.COUNT32) */
typedef struct{

__REG32 COUNT32     :21;
__REG32             :11;

} __dsm_count32_bits;

/* Reference Counter Register (DSM.REFCOUNT) */
typedef struct{

__REG32 REFCOUNT    :27;
__REG32             : 5;

} __dsm_refcount_bits;
 
/* Measurement Time Register (DSM.MEASTIME) */
typedef struct{

__REG32 MEASTIME    :21;
__REG32             :11;

} __dsm_meastime_bits;
  
/* Sleep Time Register (DSM.SLEEPTIME) */
typedef struct{

__REG32 SLEEPTIME   :21;
__REG32             :11;

} __dsm_sleeptime_bits;
 
/* Restart Time Register (DSM.RESTART_TIME) */
typedef struct{

__REG32 RESTART_TIME :21;
__REG32              :11;

} __dsm_restart_time_bits;

/* Wake-Up Time Register (DSM.WAKETIME) */
typedef struct{

__REG32 WAKETIME    :21;
__REG32             :11;

} __dsm_waketime_bits;
 
/* Warm-Up Time Register (DSM.WARMTIME) */
typedef struct{

__REG32 WARMTIME    :21;
__REG32             :11;

} __dsm_warmtime_bits;
 
/* Lock Time Register (DSM.LOCKTIME) */
typedef struct{

__REG32 LOCKTIME    :21;
__REG32             :11;

} __dsm_locktime_bits;

/* Control 0 Register (DSM.CONTROL0) */
typedef struct{

__REG32 MSTR_EN            : 1;
__REG32 XIRESP             : 2;
__REG32 EN_POS             : 1;
__REG32 RESTART            : 1;
__REG32 REF_SEL            : 1;
__REG32 STBY_INV           : 1;
__REG32 DEBUG              : 1;
__REG32 STBY_EN            : 1;
__REG32 STBY_COMMIT_EN     : 1;
__REG32 L1T_HSHK_EN        : 1;
__REG32 CONT_CNT32         : 1;
__REG32 WAKEUP_DISABLE     : 1;
__REG32                    :19;

} __dsm_control0_bits;

/* Control 1 Register (DSM.CONTROL1) */
typedef struct{

__REG32 SRST               : 1;
__REG32 CB_RST             : 1;
__REG32 SM_RST             : 1;
__REG32 RST_CNT32          : 1;
__REG32 RST_REFCNT         : 1;
__REG32 RST_POS            : 1;
__REG32 OFF                : 1;
__REG32 MEAS               : 1;
__REG32 SLEEP              : 1;
__REG32 STPL1T             : 1;
__REG32 SWAKE              : 1;
__REG32 RST_CNT32_EN       : 1;
__REG32 LTCH_CLR_CNT32     : 1;
__REG32 SHORT_WARM         : 1;
__REG32 WAKEUP_COUNTER_DIS : 1;
__REG32 VCO_EN_STATE       : 1;
__REG32 RX_EN_STATE        : 1;
__REG32                    :15;

} __dsm_control1_bits;


/* Counter Enable Register (DSM.CTREN) */
typedef struct{

__REG32 EN_CNT32           : 1;
__REG32 EN_REFCNT          : 1;
__REG32                    :30;

} __dsm_ctren_bits;
 
/* Status Register (DSM.STATUS) */
typedef struct{

__REG32 STBY                : 1;
__REG32 CLK_SEL             : 1;
__REG32 DIS_LIT             : 1;
__REG32 DS_REQ              : 1;
__REG32 DSM_INT_HOLDOFF_APM : 1;
__REG32 DS_ACK              : 1;
__REG32 WAKEUP              : 1;
__REG32 MUX_CLK             : 1;
__REG32 STBY_COMMIT_BEFORE  : 1;
__REG32                     :23;

} __dsm_status_bits;

/* State Register (DSM.STATE) */
typedef struct{

__REG32 STATE              : 4;
__REG32                    :28;

} __dsm_state_bits; 

/* DSM Interrupt Status Register (DSM.INT_STATUS) */
typedef struct{

__REG32 MDONE_INT           : 1;
__REG32 WTIME_INT           : 1;
__REG32 LOCK_INT            : 1;
__REG32 RSTRT_INT           : 1;
__REG32 STPL1T_INT          : 1;
__REG32 PGK_INT             : 1;
__REG32 NGK_INT             : 1;
__REG32 RWK                 : 1;
__REG32 SLEEP_INT           : 1;
__REG32 WKTIME_INT          : 1;
__REG32                     :22;

} __dsm_int_status_bits;
 
/* Mask Register (DSM.MASK) */
typedef struct{

__REG32 MDONE_INT_M         : 1;
__REG32 WTIME_INT_M         : 1;
__REG32 LOCK_INT_M          : 1;
__REG32 RSTRT_INT_M         : 1;
__REG32 STPL1T_INT_M        : 1;
__REG32 PGK_INT_M           : 1;
__REG32 NGK_INT_M           : 1;
__REG32                     : 1;
__REG32 SLEEP_INT_M         : 1;
__REG32 WKTIME_INT_M        : 1;
__REG32                     :22;

} __dsm_mask_bits;
 
/* COUNT32 Capture Register (DSM.COUNT32_CAP) */
typedef struct{

__REG32 COUNT32_CAP         :21;
__REG32                     :11;

} __dsm_count32_cap_bits; 

/* Warm-Up Period Register (DSM.WARMPER) */
typedef struct{

__REG32 WARMPER             :16;
__REG32                     :16;

} __dsm_warmper_bits;

/* Lock Period Register (DSM.LOCKPER) */
typedef struct{

__REG32 LOCKPER             :16;
__REG32                     :16;

} __dsm_lockper_bits;

/* Position Counter Register (DSM.POSCOUNT) */
typedef struct{

__REG32 POSCOUNT            :16;
__REG32                     :16;

} __dsm_poscount_bits;

/* Magic Period Register (DSM.MGPER) */
typedef struct{

__REG32 MGPER               :10;
__REG32                     :22;

} __dsm_mgper_bits;

/* CRM Control Register (DSM.CRM_CONTROL) */
typedef struct{

__REG32                     : 6;
__REG32 SW_AP               : 1;
__REG32                     :24;
__REG32 EN_MGFX             : 1;

} __dsm_crm_control_bits;

/*EPIT Control Register (EPITCR)*/
typedef struct {

__REG32 EN                : 1;
__REG32 ENMOD             : 1;
__REG32 OCIEN             : 1;
__REG32 RLD               : 1;
__REG32 PRESCALAR         :12;
__REG32 SWR               : 1;
__REG32 IOVW              : 1;
__REG32 DBGEN             : 1;
__REG32 WAITEN            : 1;
__REG32 RES               : 1;
__REG32 STOPEN            : 1;
__REG32 OM                : 2;
__REG32 CLKSRC            : 2;
__REG32                   : 6;

} __epitcr_bits; 

/* EPIT Status Register (EPITSR)*/
typedef struct {

__REG32 OCIF              : 1;
__REG32                   :31;

} __epitsr_bits;

/* EPIT Load Register (EPITLR)*/
typedef struct {

__REG32 LOAD              :32;

} __epitlr_bits;

/* EPIT Compare Register (EPITCMPR)*/
typedef struct {

__REG32 COMPARE           :32;

} __epitcmpr_bits;

/* EPIT Counter Register (EPITCNR)*/
typedef struct {

__REG32 COUNT              :32;

} __epitcnr_bits;

/* GPADC Control Register (GPADCIF.gpadc_ctrl)*/
typedef struct {

__REG16 APWD               : 1;
__REG16 PWD                : 1;
__REG16 STRT               : 1;
__REG16 IRQ_EN             : 1;
__REG16                    :12;

} __gpadcif_gpadc_ctrl_bits;
 
/* GPADC Interrupt Register (GPADCIF.INTERRUPT_STATUS_REG)*/
typedef struct {

__REG16 INTERRUPT_STATUS_REG : 1;
__REG16                      :15;

} __gpadcif_interrupt_status_reg_bits;
 
/* ADC1 Sampled Data (GPADCIF.ADC1)*/
typedef struct {

__REG16 ADC1                :10;
__REG16                     : 6;

} __gpadcif_adc1_bits;
 
/* ADC2 Sampled Data (GPADCIF.ADC2)*/
typedef struct {

__REG16 ADC2                :10;
__REG16                     : 6;

} __gpadcif_adc2_bits;

/* ADC3 Sampled Data (GPADCIF.ADC3)*/
typedef struct {

__REG16 ADC3                :10;
__REG16                     : 6;

} __gpadcif_adc3_bits;

/* Crystal Temperature Data Register 0 (GPADCIF.xtal_temp_0)*/
typedef struct {

__REG16 xtal_temp_0         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_0_bits;

/* Crystal Temperature Data Register 1 (GPADCIF.xtal_temp_1)*/
typedef struct {

__REG16 xtal_temp_1         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_1_bits;

/* Crystal Temperature Data Register 2 (GPADCIF.xtal_temp_2)*/
typedef struct {

__REG16 xtal_temp_2         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_2_bits;

/* Crystal Temperature Data Register 3 (GPADCIF.xtal_temp_3)*/
typedef struct {

__REG16 xtal_temp_3         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_3_bits;

/* Crystal Temperature Data Register 4 (GPADCIF.xtal_temp_4)*/
typedef struct {

__REG16 xtal_temp_4         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_4_bits;

/* Crystal Temperature Data Register 5 (GPADCIF.xtal_temp_5)*/
typedef struct {

__REG16 xtal_temp_5         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_5_bits;

/* Crystal Temperature Data Register 6 (GPADCIF.xtal_temp_6)*/
typedef struct {

__REG16 xtal_temp_6         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_6_bits;

/* Crystal Temperature Data Register 7 (GPADCIF.xtal_temp_7)*/
typedef struct {

__REG16 xtal_temp_7         :10;
__REG16                     : 6;

} __gpadcif_xtal_temp_7_bits;

/* GPIO Data Register (GPIO.DR) */
typedef struct {

__REG32 DR0                : 1;
__REG32 DR1                : 1;
__REG32 DR2                : 1;
__REG32 DR3                : 1;
__REG32 DR4                : 1;
__REG32 DR5                : 1;
__REG32 DR6                : 1;
__REG32 DR7                : 1;
__REG32 DR8                : 1;
__REG32 DR9                : 1;
__REG32 DR10               : 1;
__REG32 DR11               : 1;
__REG32 DR12               : 1;
__REG32 DR13               : 1;
__REG32 DR14               : 1;
__REG32 DR15               : 1;
__REG32 DR16               : 1;
__REG32 DR17               : 1;
__REG32 DR18               : 1;
__REG32 DR19               : 1;
__REG32 DR20               : 1;
__REG32 DR21               : 1;
__REG32 DR22               : 1;
__REG32 DR23               : 1;
__REG32 DR24               : 1;
__REG32 DR25               : 1;
__REG32 DR26               : 1;
__REG32 DR27               : 1;
__REG32 DR28               : 1;
__REG32 DR29               : 1;
__REG32 DR30               : 1;
__REG32 DR31               : 1;

} __gpio_dr_bits;

/* GPIO Edge Select Register (GPIO.EDGE_SEL) */
typedef struct {

__REG32 EDGE_SEL0                : 1;
__REG32 EDGE_SEL1                : 1;
__REG32 EDGE_SEL2                : 1;
__REG32 EDGE_SEL3                : 1;
__REG32 EDGE_SEL4                : 1;
__REG32 EDGE_SEL5                : 1;
__REG32 EDGE_SEL6                : 1;
__REG32 EDGE_SEL7                : 1;
__REG32 EDGE_SEL8                : 1;
__REG32 EDGE_SEL9                : 1;
__REG32 EDGE_SEL10               : 1;
__REG32 EDGE_SEL11               : 1;
__REG32 EDGE_SEL12               : 1;
__REG32 EDGE_SEL13               : 1;
__REG32 EDGE_SEL14               : 1;
__REG32 EDGE_SEL15               : 1;
__REG32 EDGE_SEL16               : 1;
__REG32 EDGE_SEL17               : 1;
__REG32 EDGE_SEL18               : 1;
__REG32 EDGE_SEL19               : 1;
__REG32 EDGE_SEL20               : 1;
__REG32 EDGE_SEL21               : 1;
__REG32 EDGE_SEL22               : 1;
__REG32 EDGE_SEL23               : 1;
__REG32 EDGE_SEL24               : 1;
__REG32 EDGE_SEL25               : 1;
__REG32 EDGE_SEL26               : 1;
__REG32 EDGE_SEL27               : 1;
__REG32 EDGE_SEL28               : 1;
__REG32 EDGE_SEL29               : 1;
__REG32 EDGE_SEL30               : 1;
__REG32 EDGE_SEL31               : 1;

} __gpio_edge_sel_bits;

/* GPIO Direction Register (GPIO.GDIR) */
typedef struct {

__REG32 GDIR0                : 1;
__REG32 GDIR1                : 1;
__REG32 GDIR2                : 1;
__REG32 GDIR3                : 1;
__REG32 GDIR4                : 1;
__REG32 GDIR5                : 1;
__REG32 GDIR6                : 1;
__REG32 GDIR7                : 1;
__REG32 GDIR8                : 1;
__REG32 GDIR9                : 1;
__REG32 GDIR10               : 1;
__REG32 GDIR11               : 1;
__REG32 GDIR12               : 1;
__REG32 GDIR13               : 1;
__REG32 GDIR14               : 1;
__REG32 GDIR15               : 1;
__REG32 GDIR16               : 1;
__REG32 GDIR17               : 1;
__REG32 GDIR18               : 1;
__REG32 GDIR19               : 1;
__REG32 GDIR20               : 1;
__REG32 GDIR21               : 1;
__REG32 GDIR22               : 1;
__REG32 GDIR23               : 1;
__REG32 GDIR24               : 1;
__REG32 GDIR25               : 1;
__REG32 GDIR26               : 1;
__REG32 GDIR27               : 1;
__REG32 GDIR28               : 1;
__REG32 GDIR29               : 1;
__REG32 GDIR30               : 1;
__REG32 GDIR31               : 1;

} __gpio_gdir_bits;

/* GPIO Interrupt Configuration Register 1 (GPIO.ICR1) */
typedef struct {

__REG32 ICR0                : 2;
__REG32 ICR1                : 2;
__REG32 ICR2                : 2;
__REG32 ICR3                : 2;
__REG32 ICR4                : 2;
__REG32 ICR5                : 2;
__REG32 ICR6                : 2;
__REG32 ICR7                : 2;
__REG32 ICR8                : 2;
__REG32 ICR9                : 2;
__REG32 ICR10               : 2;
__REG32 ICR11               : 2;
__REG32 ICR12               : 2;
__REG32 ICR13               : 2;
__REG32 ICR14               : 2;
__REG32 ICR15               : 2;

} __gpio_icr1_bits; 

/* GPIO Interrupt Configuration Register 2 (GPIO.ICR2) */
typedef struct {

__REG32 ICR16               : 2;
__REG32 ICR17               : 2;
__REG32 ICR18               : 2;
__REG32 ICR19               : 2;
__REG32 ICR20               : 2;
__REG32 ICR21               : 2;
__REG32 ICR22               : 2;
__REG32 ICR23               : 2;
__REG32 ICR24               : 2;
__REG32 ICR25               : 2;
__REG32 ICR26               : 2;
__REG32 ICR27               : 2;
__REG32 ICR28               : 2;
__REG32 ICR29               : 2;
__REG32 ICR30               : 2;
__REG32 ICR31               : 2;

} __gpio_icr2_bits; 

/* GPIO Interrupt Mask Register (GPIO.IMR) */
typedef struct {

__REG32 IMR0                : 1;
__REG32 IMR1                : 1;
__REG32 IMR2                : 1;
__REG32 IMR3                : 1;
__REG32 IMR4                : 1;
__REG32 IMR5                : 1;
__REG32 IMR6                : 1;
__REG32 IMR7                : 1;
__REG32 IMR8                : 1;
__REG32 IMR9                : 1;
__REG32 IMR10               : 1;
__REG32 IMR11               : 1;
__REG32 IMR12               : 1;
__REG32 IMR13               : 1;
__REG32 IMR14               : 1;
__REG32 IMR15               : 1;
__REG32 IMR16               : 1;
__REG32 IMR17               : 1;
__REG32 IMR18               : 1;
__REG32 IMR19               : 1;
__REG32 IMR20               : 1;
__REG32 IMR21               : 1;
__REG32 IMR22               : 1;
__REG32 IMR23               : 1;
__REG32 IMR24               : 1;
__REG32 IMR25               : 1;
__REG32 IMR26               : 1;
__REG32 IMR27               : 1;
__REG32 IMR28               : 1;
__REG32 IMR29               : 1;
__REG32 IMR30               : 1;
__REG32 IMR31               : 1;

} __gpio_imr_bits;

/* GPIO Interrupt Status Register (GPIO.ISR) */
typedef struct {

__REG32 ISR0                : 1;
__REG32 ISR1                : 1;
__REG32 ISR2                : 1;
__REG32 ISR3                : 1;
__REG32 ISR4                : 1;
__REG32 ISR5                : 1;
__REG32 ISR6                : 1;
__REG32 ISR7                : 1;
__REG32 ISR8                : 1;
__REG32 ISR9                : 1;
__REG32 ISR10               : 1;
__REG32 ISR11               : 1;
__REG32 ISR12               : 1;
__REG32 ISR13               : 1;
__REG32 ISR14               : 1;
__REG32 ISR15               : 1;
__REG32 ISR16               : 1;
__REG32 ISR17               : 1;
__REG32 ISR18               : 1;
__REG32 ISR19               : 1;
__REG32 ISR20               : 1;
__REG32 ISR21               : 1;
__REG32 ISR22               : 1;
__REG32 ISR23               : 1;
__REG32 ISR24               : 1;
__REG32 ISR25               : 1;
__REG32 ISR26               : 1;
__REG32 ISR27               : 1;
__REG32 ISR28               : 1;
__REG32 ISR29               : 1;
__REG32 ISR30               : 1;
__REG32 ISR31               : 1;

} __gpio_isr_bits; 

/* GPIO Pad Status Register (GPIO.PSR) */
typedef struct {

__REG32 PSR0                : 1;
__REG32 PSR1                : 1;
__REG32 PSR2                : 1;
__REG32 PSR3                : 1;
__REG32 PSR4                : 1;
__REG32 PSR5                : 1;
__REG32 PSR6                : 1;
__REG32 PSR7                : 1;
__REG32 PSR8                : 1;
__REG32 PSR9                : 1;
__REG32 PSR10               : 1;
__REG32 PSR11               : 1;
__REG32 PSR12               : 1;
__REG32 PSR13               : 1;
__REG32 PSR14               : 1;
__REG32 PSR15               : 1;
__REG32 PSR16               : 1;
__REG32 PSR17               : 1;
__REG32 PSR18               : 1;
__REG32 PSR19               : 1;
__REG32 PSR20               : 1;
__REG32 PSR21               : 1;
__REG32 PSR22               : 1;
__REG32 PSR23               : 1;
__REG32 PSR24               : 1;
__REG32 PSR25               : 1;
__REG32 PSR26               : 1;
__REG32 PSR27               : 1;
__REG32 PSR28               : 1;
__REG32 PSR29               : 1;
__REG32 PSR30               : 1;
__REG32 PSR31               : 1;

} __gpio_psr_bits; 

/* GPT Control Register (GPTCR) */
typedef struct{

__REG32 EN              : 1;
__REG32 ENMOD           : 1;
__REG32 DBGEN           : 1;
__REG32 WAITEN          : 1;
__REG32                 : 1;
__REG32 STOPEN          : 1;
__REG32 CLKSRC          : 3;
__REG32 FRR             : 1;
__REG32                 : 5;
__REG32 SWR             : 1;
__REG32 IM1             : 2;
__REG32 IM2             : 2;
__REG32 OM1             : 3;
__REG32 OM2             : 3;
__REG32 OM3             : 3;
__REG32 FO1             : 1;
__REG32 FO2             : 1;
__REG32 FO3             : 1;

} __gptcr_bits;
 
/* GPT Prescaler Register (GPTPR) */
typedef struct{

__REG32 PRESCALER       :12;
__REG32                 :20;

} __gptpr_bits;
 
/* GPT Status Register (GPTSR) */
typedef struct{

__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32 OF3             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32 ROV             : 1;
__REG32                 :26;

} __gptsr_bits;
 
/* GPT Interrupt Register (GPTIR) */
typedef struct{

__REG32 OF1IE           : 1;
__REG32 OF2IE           : 1;
__REG32 OF3IE           : 1;
__REG32 IF1IE           : 1;
__REG32 IF2IE           : 1;
__REG32 ROVIE           : 1;
__REG32                 :26;

} __gptir_bits;
 
/* GPT Output Compare Register (GPTOCR1,GPTOCR2,GPTOCR3) */
typedef struct{

__REG32 COMP            :32;

} __gptocr_bits;

/* GPT Input Capture Register (GPTICR1,GPTICR2) */
typedef struct{

__REG32 CAPT            :32;

} __gpticr_bits;
 
/* GPT Counter Register (GPTCNT) */
typedef struct{

__REG32 COUNT           :32;

} __gptcnt_bits;

/* Source Modulo Addressing Register (HDMA.SMAR) */
typedef struct{

__REG32 SMARCN           :16;
__REG32 SMARSZ           :16;

} __hdma_smar_bits;

/* Destination Modulo Addressing Register (HDMA.DMAR) */
typedef struct{

__REG32 DMARCN           :16;
__REG32 DMARSZ           :16;

} __hdma_dmar_bits;

/* Buffer Size and Counter Register (HDMA.BFCN) */
typedef struct{

__REG32 BFCN             :16;
__REG32 BFSZ             :16;

} __hdma_bfcn_bits;

/* Channel Link List Register (HDMA.DCLL) */
typedef struct{

__REG32 IBCIEN           : 1;
__REG32 DCLL             :31;

} __hdma_dcll_bits;

/* Transfer Complete Status Register (HDMA.TCSR) */
typedef struct{

__REG32 TCSR             : 8;
__REG32                  :24;

} __hdma_tcsr_bits; 

/* Buffer Complete Status Register (HDMA.BCSR) */
typedef struct{

__REG32 BCSR             : 8;
__REG32                  :24;

} __hdma_bcsr_bits; 

/* Intermediate Buffer Complete Status Register (HDMA.IBCSR) */
typedef struct{

__REG32 IBCSR            : 8;
__REG32                  :24;

} __hdma_ibcsr_bits;   

/* Burst Time-out Status Register (HDMA.BTSR) */
typedef struct{

__REG32 BTSR             : 8;
__REG32                  :24;

} __hdma_btsr_bits; 

/* Burst Time-Out Control Register (HDMA.BTCR) */
typedef struct{

__REG32 BTCR0            :15;
__REG32 BTCNEN0          : 1;
__REG32                  :16;

} __hdma_btcr_bits;

/* Transfer Error Status Register (HDMA.TESR) */
typedef struct{

__REG32 TESR             : 8;
__REG32                  :24;

} __hdma_tesr_bits;

/* Channel Request Time-Out Pre-Scalar Register (HDMA.CRTPR) */
typedef struct{

__REG32 CRTPR            :16;
__REG32                  :16;

} __hdma_crtpr_bits;

/* Channel Request Time-out Status Register (HDMA.CRTSR) */
typedef struct{

__REG32 CRTSR            : 8;
__REG32                  :24;

} __hdma_crtsr_bits;

/* Buffer Not Ready Status Register (HDMA.BNRSR) */
typedef struct{

__REG32 BNRSR            : 8;
__REG32                  :24;

} __hdma_bnrsr_bits;
 
/* Bandwidth Complete Status Register (HDMA.BWCSR) */
typedef struct{

__REG32 BWCSR            : 8;
__REG32                  :24;

} __hdma_bwcsr_bits;

/* DMA Global Control Register (HDMA.DGCR) */
typedef struct{

__REG32 DMAEN            : 1;
__REG32 SFTRST           : 1;
__REG32 QOS              : 1;
__REG32 DGCR_1           : 1;
__REG32 ENGDIS           : 1;
__REG32 BUF_UPDT         : 1;
__REG32 REM_XFR_EN       : 1;
__REG32 TESR_INT_EN      : 1;
__REG32 DGCR_2           :24;

} __hdma_dgcr_bits;

/* DMA Control Register (HDMA.D0_DCR – HDMA.D7_DCR) */
typedef struct{

__REG32 CE               : 1;
__REG32 DCHAINEN         : 1;
__REG32 CHPRIO           : 3;
__REG32 BBWD             : 4;
__REG32 DSIZE            : 2;
__REG32 SMEN             : 1;
__REG32 DMEN             : 1;
__REG32 SECACC           : 1;
__REG32 TE               : 1;
__REG32 TCIE             : 1;
__REG32 BCIE             : 1;
__REG32 CRTCEN           : 1;
__REG32 CRTLSB           : 8;
__REG32 MBLEN            : 4;
__REG32 BNRIEN           : 1;
__REG32 HWT_BTBEN        : 1;

} __hdma_dxdcr_bits;

/* BYTE SWAP Register (HDMA.D0_SWAP) */
typedef struct{

__REG32 DACN             :16;
__REG32                  :16;

} __hdma_d0swap_bits; 

/* Channel Request Time-Out Counter (HDMA.D0_CRTCN - HDMA.D7_CRTCN) */
typedef struct{

__REG32 CRTCN            :24;
__REG32                  : 8;

} __hdma_dxcrtcn_bits;  

/* Burst Time-Out Counter Register (HDMA.D0_BTCN – HDMA.D1_BTCN) */
typedef struct{

__REG16 BTCN             :15;
__REG16                  : 1;

} __hdma_dxbtcn_bits; 

/* activech_reg Debug Register (HDMA.activech_reg) */
typedef struct{

__REG32 activech_reg     : 8;
__REG32                  :24;

} __hdma_activech_reg_bits; 
 
/* dum_rem_xfr_reg Debug Register (HDMA.dum_rem_xfr_reg) */
typedef struct{

__REG32 dum_rem_xfr_reg  : 8;
__REG32                  :24;

} __hdma_dum_rem_xfr_reg_bits; 
 
/* dum_ch_abort Debug Register (HDMA.dum_ch_abort) */
typedef struct{

__REG32 dum_ch_abort     : 8;
__REG32                  :24;

} __hdma_dum_ch_abort_bits; 

/* dum_crtsr_reg Debug Register (HDMA.dum_crtsr_reg) */
typedef struct{

__REG32 dum_crtsr_reg    : 8;
__REG32                  :24;

} __hdma_dum_crtsr_reg_bits; 

/* rem_xfr_all_reg Debug Register (HDMA.rem_xfr_all_reg) */
typedef struct{

__REG32 rem_xfr_all_reg  :10;
__REG32                  :22;

} __hdma_rem_xfr_all_reg_bits; 

/* dma_act_ch0 Debug Register (HDMA.dma_act_ch0) */
typedef struct{

__REG32 dma_act_ch0      : 8;
__REG32                  :24;

} __hdma_dma_act_ch0_bits; 

/* dma_act_ch0_next Debug Register (HDMA.dma_act_ch0_next) */
typedef struct{

__REG32 dma_act_ch0_next : 8;
__REG32                  :24;

} __hdma_dma_act_ch0_next_bits;  

/* I2C Address Register (I2C.IADR) */
typedef struct {

__REG16      : 1;
__REG16 ADR  : 7;
__REG16      : 8;

} __i2c_iadr_bits;

/* I2C Frequency Register (I2C.IFDR) */
typedef struct {

__REG16 IC  : 6;
__REG16     :10;

} __i2c_ifdr_bits;

/* I2C Control Register (I2C.I2CR) */
typedef struct {

__REG16       : 2;
__REG16 RSTA  : 1;
__REG16 TXAK  : 1;
__REG16 MTX   : 1;
__REG16 MSTA  : 1;
__REG16 IIEN  : 1;
__REG16 IEN   : 1;
__REG16       : 8;

} __i2c_i2cr_bits;

/* I2C Status Register (I2C.I2SR) */
typedef struct {

__REG16 RXAK  : 1;
__REG16 IIF   : 1;
__REG16 SRW   : 1;
__REG16       : 1;
__REG16 IAL   : 1;
__REG16 IBB   : 1;
__REG16 IAAS  : 1;
__REG16 ICF   : 1;
__REG16       : 8;

} __i2c_i2sr_bits;

/* I2C Data Register (I2C.I2DR) */
typedef struct {

__REG16 DATA  : 8;
__REG16       : 8;

} __i2c_i2dr_bits;

/* Error IRQ Mask Register (IIM.EMASK) */
typedef struct{

__REG8            : 1;
__REG8  PARITYE_M : 1;
__REG8  SNSE_M    : 1;
__REG8  WLRE_M    : 1;
__REG8  RPE_M     : 1;
__REG8  OPE_M     : 1;
__REG8  WPE_M     : 1;
__REG8  PRGE_M    : 1;

} __iim_emask_bits;
 
/* Module Errors Register (IIM.ERR) */
typedef struct{

__REG8            : 1;
__REG8  PARITYE   : 1;
__REG8  SNSE      : 1;
__REG8  WLRE      : 1;
__REG8  RPE       : 1;
__REG8  OPE       : 1;
__REG8  WPE       : 1;
__REG8  PRGE      : 1;

} __iim_err_bits;

/* Fuse Control Register (IIM.FCTL) */
typedef struct{

__REG8  PRG       : 1;
__REG8  ESNS_1    : 1;
__REG8  ESNS_0    : 1;
__REG8  ESNS_N    : 1;
__REG8  PRG_LENGTH: 3;
__REG8  DPC       : 1;

} __iim_fctl_bits;

/* Lower Address (IIM.LA) */
typedef struct{

__REG8  A         : 8;

} __iim_la_bits;

/* Upper Address (IIM.UA) */
typedef struct{

__REG8  A         : 6;
__REG8            : 2;

} __iim_ua_bits;

/* Product Revision (IIM.PREV) */
typedef struct{

__REG8  PROD_VT   : 3;
__REG8  PROD_REV  : 5;

} __iim_prev_bits;
 
/* Program Protection Register (IIM.PRG_P) */
typedef struct{

__REG8  PROTECTION_REG : 8;

} __iim_prg_p_bits;

/* Software-Controllable Signals Register 0 (IIM.SCS0) */
typedef struct{

__REG8  SCS       : 6;
__REG8  HAB_JDE   : 1;
__REG8  LOCK      : 1;

} __iim_scs0_bits;

/* Software-Controllable Signals Register 1 (IIM.SCS1) */
typedef struct{

__REG8  SCS       : 7;
__REG8  LOCK      : 1;

} __iim_scs1_bits;

/* Software-Controllable Signals Register 2 (IIM.SCS2) */
typedef struct{

__REG8  FBRL0     : 1;
__REG8  FBRL1     : 1;
__REG8            : 5;
__REG8  LOCK      : 1;

} __iim_scs2_bits;

/* Software-Controllable Signals Register 3 (IIM.SCS3) */
typedef struct{

__REG8  FBWL0     : 1;
__REG8  FBWL1     : 1;
__REG8            : 5;
__REG8  LOCK      : 1;

} __iim_scs3_bits;

/* Explicit Sense Data Register (IIM.SDAT) */
typedef struct{

__REG8  D         : 8;

} __iim_sdat_bits;

/* Silicon Revision Register (IIM.SREV) */
typedef struct{

__REG8  SILICON_REV : 8;

} __iim_srev_bits;

/* Status Register (IIM.STAT) */
typedef struct{

__REG8  SNSD      : 1;
__REG8  PRGD      : 1;
__REG8            : 5;
__REG8  BUSY      : 1;

} __iim_stat_bits;

/* Status IRQ Mask (IIM.STATM) */
typedef struct{

__REG8  SNSD_M    : 1;
__REG8  PRGD_M    : 1;
__REG8            : 6;

} __iim_statm_bits;

/* Keypad Control Register (KPP.KPCR) */
typedef struct{

__REG16 KRE  : 8;
__REG16 KCO  : 8;

} __kpp_kpcr_bits;

/* Keypad Status Register (KPP.KPSR) */
typedef struct{

__REG16 KPKD    : 1;
__REG16 KPKR    : 1;
__REG16 KDSC    : 1;
__REG16 KRSS    : 1;
__REG16         : 4;
__REG16 KDIE    : 1;
__REG16 KRIE    : 1;
__REG16         : 6;

} __kpp_kpsr_bits;

/* Keypad Data Direction Register (KPP.KDDR) */
typedef struct{

__REG16 KRDD  : 8;
__REG16 KCDD  : 8;

} __kpp_kddr_bits;

/* Keypad Data Register (KPP.KPDR) */
typedef struct{

__REG16 KRD  : 8;
__REG16 KCD  : 8;

} __kpp_kpdr_bits;
 
/* Master Priority Register (MAX.MPR0-7,MAX.AMPR0-7) */
typedef struct {

__REG32 MSTR_0    : 3;
__REG32           : 1;
__REG32 MSTR_1    : 3;
__REG32           : 1;
__REG32 MSTR_2    : 3;
__REG32           : 1;
__REG32 MSTR_3    : 3;
__REG32           : 1;
__REG32 MSTR_4    : 3;
__REG32           : 1;
__REG32 MSTR_5    : 3;
__REG32           : 1;
__REG32 MSTR_6    : 3;
__REG32           : 5;

} __max_mprx_amprx_bits;
 
/* Slave General Purpose Control Register (MAX.SGPCR0-7,MAX.ASGPCR0-7) */
typedef struct {

__REG32 PARK      : 3;
__REG32           : 1;
__REG32 PCTL      : 2;
__REG32           : 2;
__REG32 ARB       : 2;
__REG32           : 6;
__REG32 HPE       : 8;
__REG32           : 6;
__REG32 HLP       : 1;
__REG32 RO        : 1;

} __max_sgpcr_asgpcr_bits;

/* Master General Purpose Control Register (MAX.MGPCR0-7) */
typedef struct {

__REG32 AULB      : 3;
__REG32           :29;

} __max_mgpcr_bits;
  
/* PWM Control Register (PWMCR) */
typedef struct{

__REG32 EN         : 1;
__REG32 REPEAT     : 2;
__REG32 SWR        : 1;
__REG32 PRESCALER  :12;
__REG32 CLKSRC     : 2;
__REG32 POUTC      : 2;
__REG32 HCTR       : 1;
__REG32 BCTR       : 1;
__REG32 DBGEN      : 1;
__REG32 WAITEN     : 1;
__REG32 DOZEN      : 1;
__REG32 STOPEN     : 1;
__REG32 FWM        : 2;
__REG32            : 4;

} __pwmcr_bits;

/* PWM Status Register (PWMSR) */
typedef struct{

__REG32 FIFOAV  : 3;
__REG32 FE      : 1;
__REG32 ROV     : 1;
__REG32 CMP     : 1;
__REG32 FWE     : 1;
__REG32         :25;

} __pwmsr_bits;

/* PWM Interrupt Register (PWMIR) */
typedef struct{

__REG32 FIE     : 1;
__REG32 RIE     : 1;
__REG32 CIE     : 1;
__REG32         :29;

} __pwmir_bits;

/* PWM Sample Register (PWMSAR) */
typedef struct{

__REG32 SAMPLE :16;
__REG32        :16;

} __pwmsar_bits;

/* PWM Period Register (PWMPR) */
typedef struct{

__REG32 PERIOD :16;
__REG32        :16;

} __pwmpr_bits;

/* PWM Counter Register (PWMCNR) */
typedef struct{

__REG32 COUNT  :16;
__REG32        :16;

} __pwmcnr_bits;
 
/* TMR Control Registers (TMRx.CTRL) */
typedef struct {

__REG16 Output_Mode            : 3;
__REG16 Co_Init                : 1;
__REG16 DIR                    : 1;
__REG16 LENGTH                 : 1;
__REG16 ONCE                   : 1;
__REG16 Secondary_Count_Source : 2;
__REG16 Primary_Count_Source   : 4;
__REG16 Count_Mode             : 3;

} __tmrx_ctrl_bits;

/* TMR Status and Control Registers (TMRx.SCR) */
typedef struct {

__REG16 OEN                    : 1;
__REG16 OPS                    : 1;
__REG16 FORCE                  : 1;
__REG16 VAL                    : 1;
__REG16 EEOF                   : 1;
__REG16 MSTR                   : 1;
__REG16 Capture_Mode           : 2;
__REG16 INPUT                  : 1;
__REG16 IPS                    : 1;
__REG16 IEFIE                  : 1;
__REG16 IEF                    : 1;
__REG16 TOFIE                  : 1;
__REG16 TOF                    : 1;
__REG16 TCFIE                  : 1;
__REG16 TCF                    : 1;

} __tmrx_scr_bits; 

/* TMR Comparator Status and Control Registers (TMRx.COMSCR) */
typedef struct {

__REG16 CL1                    : 2;
__REG16 CL2                    : 2;
__REG16 TCF1                   : 1;
__REG16 TCF2                   : 1;
__REG16 TCF1EN                 : 1;
__REG16 TCF2EN                 : 1;
__REG16                        : 8;

} __tmrx_comscr_bits;  

/* RTC Hours and Minutes Counter Register (RTC.HOURMIN) */
/* RTC Hours and Minutes Alarm Register (RTC.ALRM_HM) */
typedef struct{

__REG32 MINUTES  : 6;
__REG32          : 2;
__REG32 HOURS    : 5;
__REG32          :19;

} __rtc_hourmin_bits;

/* RTC Seconds Counter Register (RTC.SECONDS) */
/* RTC Seconds Alarm Register (RTC.ALRM_SEC) */
typedef struct{

__REG32 SECONDS  : 6;
__REG32          :26;

} __rtc_seconds_bits;

/* RTC Control Register (RTC.RTCCTL) */
typedef struct{

__REG32 SWR  : 1;
__REG32 GEN  : 1;
__REG32      : 3;
__REG32 XTL  : 2;
__REG32 EN   : 1;
__REG32      :24;

} __rtc_rtcctl_bits;

/* RTC Interrupt Status Register (RTC.RTCISR) */
/* RTC Interrupt Enable Register (RTC.RTCIENR) */
typedef struct{

__REG32 SW    : 1;
__REG32 MIN   : 1;
__REG32 ALM   : 1;
__REG32 DAY   : 1;
__REG32 _1HZ  : 1;
__REG32 HR    : 1;
__REG32       : 1;
__REG32 _2HZ  : 1;
__REG32 SAM0  : 1;
__REG32 SAM1  : 1;
__REG32 SAM2  : 1;
__REG32 SAM3  : 1;
__REG32 SAM4  : 1;
__REG32 SAM5  : 1;
__REG32 SAM6  : 1;
__REG32 SAM7  : 1;
__REG32       :16;

} __rtc_rtcisr_bits;

/* RTC Stopwatch Minutes Register (RTC.STPWCH) */
typedef struct{

__REG32 CNT  : 6;
__REG32      :26;

} __rtc_stpwch_bits;

/* RTC Days Counter Register (RTC.DAYR) */
typedef struct{

__REG32 DAYS  :16;
__REG32       :16;

} __rtc_dayr_bits;

/* RTC Day Alarm Register (RTC.DAYALARM) */
typedef struct{

__REG32 DAYSAL  :16;
__REG32         :16;

} __rtc_dayalarm_bits;
 
/* RNGB Version ID Register (RNGB.VER_ID) */
typedef struct{

__REG32 VERSION_MINOR  : 8;
__REG32 VERSION_MAJOR  : 8;
__REG32                :12;
__REG32 RNG_TYPE       : 4;

} __rngb_ver_id_bits; 

/* RNGB Command Register (RNGB.COMMAND) */
typedef struct{

__REG32 SELF_TEST      : 1;
__REG32 SEED           : 1;
__REG32                : 2;
__REG32 CLR_INT        : 1;
__REG32 CLR_ERR        : 1;
__REG32 CLR_RST        : 1;
__REG32                :25;

} __rngb_command_bits;
  
/* RNGB Control Register (RNGB.CONTROL) */
typedef struct{

__REG32 FIFO_UFLOW_RESPONSE : 2;
__REG32                     : 2;
__REG32 AUTO_SEED           : 1;
__REG32 MASK_DONE           : 1;
__REG32 MASK_ERR            : 1;
__REG32                     : 1;
__REG32 VERIF_MODE          : 1;
__REG32 CTL_ACC             : 1;
__REG32                     :22;

} __rngb_control_bits;

/* RNGB Status Register (RNGB.STATUS) */
typedef struct{

__REG32 SEC_STATE      : 1;
__REG32 BUSY           : 1;
__REG32 SLEEP          : 1;
__REG32 RE_SEED        : 1;
__REG32 ST_DONE        : 1;
__REG32 SEED_DONE      : 1;
__REG32 NEW_SEED_DONE  : 1;
__REG32                : 1;
__REG32 FIFO_LVL       : 3;
__REG32                : 1;
__REG32 FIFO_SIZE      : 4;
__REG32 ERROR          : 1;
__REG32                : 5;
__REG32 ST_PF          : 2;
__REG32 STAT_TEST_PF   : 8;

} __rngb_status_bits;  

/* RNGB Error Status Register (RNGB.ERROR_STATUS) */
typedef struct{

__REG32 LFSR_ERR       : 1;
__REG32 OSC_ERR        : 1;
__REG32 ST_ERR         : 1;
__REG32 STAT_ERR       : 1;
__REG32 FIFO_ERR       : 1;
__REG32                :27;

} __rngb_error_status_bits;   

/* RNGB Verification Control Register (RNGB.VERIF_CTL) */
typedef struct{

__REG32 SH_CLK_OFF     : 1;
__REG32 FRC_SYS_CLK    : 1;
__REG32 OSC_TEST       : 1;
__REG32 FAKE_SEED      : 1;
__REG32                : 4; 
__REG32 RST_SHREG      : 1;
__REG32 RST_XKEY       : 1;
__REG32                :22;

} __rngb_verif_ctl_bits;    

/* RNGB Oscillator Counter Control Register (RNGB.OSC_CNT_CTL) */
typedef struct{

__REG32 CYCLES         :18;
__REG32                :14;

} __rngb_osc_cnt_ctl_bits;     

/* RNGB Oscillator Counter Register (RNGB.OSC_CNT) */
typedef struct{

__REG32 CLOCK_PULSES   :20;
__REG32                :12;

} __rngb_osc_cnt_bits;  

/* RNGB Oscillator Counter Status (RNGB.OSC_CNT_STAT) */
typedef struct{

__REG32 OSC_STAT       : 1;
__REG32                :31;

} __rngb_osc_cnt_stat_bits;  

/* ROMPATCH Control Register (ROMPATCHCNTL) */
typedef struct{

__REG32 DATAFIX        : 8;
__REG32                :21;
__REG32 DIS            : 1;
__REG32                : 2;

} __rompatchcntl_bits;  

/* ROMPATCH Enable Register (ROMPATCHENL) */
typedef struct{

__REG32 ENABLE0   : 1;
__REG32 ENABLE1   : 1;
__REG32 ENABLE2   : 1;
__REG32 ENABLE3   : 1;
__REG32 ENABLE4   : 1;
__REG32 ENABLE5   : 1;
__REG32 ENABLE6   : 1;
__REG32 ENABLE7   : 1;
__REG32 ENABLE8   : 1;
__REG32 ENABLE9   : 1;
__REG32 ENABLE10  : 1;
__REG32 ENABLE11  : 1;
__REG32 ENABLE12  : 1;
__REG32 ENABLE13  : 1;
__REG32 ENABLE14  : 1;
__REG32 ENABLE15  : 1;
__REG32           :16;

} __rompatchenl_bits;   

/* ROMPATCH Address Registers (ROMPATCHA) */
typedef struct{

__REG32 THUMBX    : 1;
__REG32 ADDRX     :21;
__REG32           :10;

} __rompatcha_bits;   

/* ROMPATCH Status Register (ROMPATCHSR) */
typedef struct{

__REG32 SOURCE    : 6;
__REG32           :11;
__REG32 SW        : 1;
__REG32           :14;

} __rompatchsr_bits;
 
/* Ruby Software Version (RUBY.SWVERSION) */
typedef struct{

__REG32 prom_ucode_version :16;
__REG32 pram_ucode_version :16;

} __ruby_swversion_bits; 

/* Ruby System Control Register (RUBY.CONTROL) */
typedef struct{

__REG32 go                 : 1;
__REG32 rt_mode            : 1;
__REG32                    :30;

} __ruby_control_bits; 

/* Ruby Interrupt Enable Register (RUBY.IRQEN) */
typedef struct{

__REG32 irqen_done         : 1;
__REG32                    : 1;
__REG32 irqen_rx           : 1;
__REG32 irqen_tx           : 1;
__REG32 irqen_dma_cmp      : 1;
__REG32 irqen_dma_error    : 1;
__REG32                    :26;

} __ruby_irqen_bits; 
 
/* Ruby Source 1 Info (RUBY.STATUS) */
typedef struct{

__REG32 done               : 1;
__REG32 rt_status          : 1;
__REG32 irq_pend_rx        : 1;
__REG32 irq_pend_tx        : 1;
__REG32 irq_pend_dma_comp  : 1;
__REG32 irq_pend_dma_error : 1;
__REG32                    : 2;
__REG32 busy               : 1;
__REG32                    :23;

} __ruby_status_bits;  

/* Rx Status (RUBY.RXSTATUS) */
typedef struct{

__REG32 rx_status_0        : 1;
__REG32 rx_status_1        : 1;
__REG32 rx_status_2        : 1;
__REG32 rx_status_3        : 1;
__REG32 rx_status_4        : 1;
__REG32 rx_status_5        : 1;
__REG32 rx_status_6        : 1;
__REG32 rx_status_7        : 1;
__REG32 rx_status_8        : 1;
__REG32 rx_status_9        : 1;
__REG32 rx_status_10       : 1;
__REG32 rx_status_11       : 1; 
__REG32 rx_status_12       : 1;
__REG32 rx_status_13       : 1;
__REG32 rx_status_14       : 1;
__REG32 rx_status_15       : 1;
__REG32                    :16;

} __ruby_rxstatus_bits;   

/* Tx Status (RUBY.TXSTATUS) */
typedef struct{

__REG32 tx_status_0        : 1;
__REG32 tx_status_1        : 1;
__REG32 tx_status_2        : 1;
__REG32 tx_status_3        : 1;
__REG32 tx_status_4        : 1;
__REG32 tx_status_5        : 1;
__REG32 tx_status_6        : 1;
__REG32 tx_status_7        : 1;
__REG32 tx_status_8        : 1;
__REG32 tx_status_9        : 1;
__REG32 tx_status_10       : 1;
__REG32 tx_status_11       : 1; 
__REG32 tx_status_12       : 1;
__REG32 tx_status_13       : 1;
__REG32 tx_status_14       : 1;
__REG32 tx_status_15       : 1;
__REG32                    :16;

} __ruby_txstatus_bits;   

/* Ruby Parameters (RUBY.PARAM) */
typedef struct{

__REG32 GP_IN              : 8;
__REG32 GP_OUT             : 8;
__REG32 DMA_CNT            : 8;
__REG32 AXI_ID             : 4;
__REG32 AXI_DATA           : 3;
__REG32 RSSE               : 1;

} __ruby_param_bits;   

/* Ruby Parameters (RUBY.MACRO) */
typedef struct{

__REG32 NAU                : 8;
__REG32 IW_SIZE            : 8;
__REG32                    :16;

} __ruby_macro_bits;    

/* Breakpoint Control (RUBY.BPCTL) */
typedef struct{

__REG32 GBL_EN             : 1;
__REG32 BKN                : 1;
__REG32 RSM                : 1;
__REG32 SS                 : 1;
__REG32 BK0_EN             : 1;
__REG32 BK1_EN             : 1;
__REG32 BK2_EN             : 1;
__REG32 BK3_EN             : 1;
__REG32 XEB                : 1;
__REG32 BK_INT_EN          : 1;
__REG32                    : 2;
__REG32 RAS_SEL            : 3;
__REG32                    : 1;
__REG32 RV_SEL             : 5;
__REG32                    : 3;
__REG32 AS_REG_SEL         : 4;
__REG32                    : 4;

} __ruby_bpctl_bits;   

/* Breakpoint Status (RUBY.BPSTAT) */
typedef struct{

__REG32 GBL_ST             : 1;
__REG32 BKN_ST             : 1;
__REG32 BK_INT_ST          : 1;
__REG32                    : 1;
__REG32 BK0_ST             : 1;
__REG32 BK1_ST             : 1;
__REG32 BK2_ST             : 1;
__REG32 BK3_ST             : 1;
__REG32 XEB_ST             : 1;
__REG32                    : 7;
__REG32 CURR_PC            :16;

} __ruby_bpstat_bits;    

/* Breakpoint Address Match 0 (RUBY.BPAM0) */
typedef struct{

__REG32 BP_ADDR0           :16;
__REG32 BP_ADDR1           :16;

} __ruby_bpam0_bits;     

/* Breakpoint Address Match 1 (RUBY.BPAM1) */
typedef struct{

__REG32 BP_ADDR2           :16;
__REG32 BP_ADDR3           :16;

} __ruby_bpam1_bits;     
 
/* DMA Control (RUBY.DMA_CONTROL,RUBY.DMA_CONTROL_SET/CLEAR) */
typedef struct{

__REG32 channel_enable_0   : 1;
__REG32 channel_enable_1   : 1;
__REG32 channel_enable_2   : 1;
__REG32 channel_enable_3   : 1;
__REG32 channel_enable_4   : 1;
__REG32 channel_enable_5   : 1;
__REG32 channel_enable_6   : 1;
__REG32 channel_enable_7   : 1;
__REG32 channel_enable_8   : 1;
__REG32 channel_enable_9   : 1;
__REG32 channel_enable_10  : 1;
__REG32 channel_enable_11  : 1;
__REG32 channel_enable_12  : 1;
__REG32 channel_enable_13  : 1;
__REG32 channel_enable_14  : 1;
__REG32 channel_enable_15  : 1;
__REG32                    :16;

} __ruby_dma_control_set_clear_bits;  

/* DMA Complete IRQ Enable (RUBY.DMA_COMP_IRQEN) */
typedef struct{

__REG32 irqen_chan_0       : 1;
__REG32 irqen_chan_1       : 1;
__REG32 irqen_chan_2       : 1;
__REG32 irqen_chan_3       : 1;
__REG32 irqen_chan_4       : 1;
__REG32 irqen_chan_5       : 1;
__REG32 irqen_chan_6       : 1;
__REG32 irqen_chan_7       : 1;
__REG32 irqen_chan_8       : 1;
__REG32 irqen_chan_9       : 1;
__REG32 irqen_chan_10      : 1;
__REG32 irqen_chan_11      : 1;
__REG32 irqen_chan_12      : 1;
__REG32 irqen_chan_13      : 1;
__REG32 irqen_chan_14      : 1;
__REG32 irqen_chan_15      : 1;
__REG32                    :16;

} __ruby_dma_comp_irqen_bits;  
 
/* DMA Complete Status (RUBY.DMA_COMP_STAT) */
typedef struct{

__REG32 dma_comp_chan_0    : 1;
__REG32 dma_comp_chan_1    : 1;
__REG32 dma_comp_chan_2    : 1;
__REG32 dma_comp_chan_3    : 1;
__REG32 dma_comp_chan_4    : 1;
__REG32 dma_comp_chan_5    : 1;
__REG32 dma_comp_chan_6    : 1;
__REG32 dma_comp_chan_7    : 1;
__REG32 dma_comp_chan_8    : 1;
__REG32 dma_comp_chan_9    : 1;
__REG32 dma_comp_chan_10   : 1;
__REG32 dma_comp_chan_11   : 1;
__REG32 dma_comp_chan_12   : 1;
__REG32 dma_comp_chan_13   : 1;
__REG32 dma_comp_chan_14   : 1;
__REG32 dma_comp_chan_15   : 1;
__REG32                    :16;

} __ruby_dma_comp_stat_bits; 

/* DMA Transfer Error Status (RUBY.DMA_XFRERR_STAT) */
typedef struct{

__REG32 xfr_error_chan_0   : 1;
__REG32 xfr_error_chan_1   : 1;
__REG32 xfr_error_chan_2   : 1;
__REG32 xfr_error_chan_3   : 1;
__REG32 xfr_error_chan_4   : 1;
__REG32 xfr_error_chan_5   : 1;
__REG32 xfr_error_chan_6   : 1;
__REG32 xfr_error_chan_7   : 1;
__REG32 xfr_error_chan_8   : 1;
__REG32 xfr_error_chan_9   : 1;
__REG32 xfr_error_chan_10  : 1;
__REG32 xfr_error_chan_11  : 1;
__REG32 xfr_error_chan_12  : 1;
__REG32 xfr_error_chan_13  : 1;
__REG32 xfr_error_chan_14  : 1;
__REG32 xfr_error_chan_15  : 1;
__REG32                    :16;

} __ruby_dma_xfrerr_stat_bits; 

/* DMA Configuration Error Status (RUBY.DMA_CFGERR_STAT) */
typedef struct{

__REG32 cfg_error_chan_0   : 1;
__REG32 cfg_error_chan_1   : 1;
__REG32 cfg_error_chan_2   : 1;
__REG32 cfg_error_chan_3   : 1;
__REG32 cfg_error_chan_4   : 1;
__REG32 cfg_error_chan_5   : 1;
__REG32 cfg_error_chan_6   : 1;
__REG32 cfg_error_chan_7   : 1;
__REG32 cfg_error_chan_8   : 1;
__REG32 cfg_error_chan_9   : 1;
__REG32 cfg_error_chan_10  : 1;
__REG32 cfg_error_chan_11  : 1;
__REG32 cfg_error_chan_12  : 1;
__REG32 cfg_error_chan_13  : 1;
__REG32 cfg_error_chan_14  : 1;
__REG32 cfg_error_chan_15  : 1;
__REG32                    :16;

} __ruby_dma_cfgerr_stat_bits;  

/* DMA Transfer Running Status (RUBY.DMA_XRUN_STAT) */
typedef struct{

__REG32 xfr_run_chan_0   : 1;
__REG32 xfr_run_chan_1   : 1;
__REG32 xfr_run_chan_2   : 1;
__REG32 xfr_run_chan_3   : 1;
__REG32 xfr_run_chan_4   : 1;
__REG32 xfr_run_chan_5   : 1;
__REG32 xfr_run_chan_6   : 1;
__REG32 xfr_run_chan_7   : 1;
__REG32 xfr_run_chan_8   : 1;
__REG32 xfr_run_chan_9   : 1;
__REG32 xfr_run_chan_10  : 1;
__REG32 xfr_run_chan_11  : 1;
__REG32 xfr_run_chan_12  : 1;
__REG32 xfr_run_chan_13  : 1;
__REG32 xfr_run_chan_14  : 1;
__REG32 xfr_run_chan_15  : 1;
__REG32                  :16;

} __ruby_dma_xrun_stat_bits;  
 
/* DMA Channel Control Register (16 channels) (RUBY.DMA_CHAN_CTRL) */
typedef struct{

__REG32 trans_mode_select   : 3;
__REG32                     : 5;
__REG32 AXI_cache_state     : 4;
__REG32 AXI_prot_state      : 3;
__REG32 burst_type          : 1;
__REG32                     :16;

} __ruby_dma_chan_ctrl_bits; 

/* DMEM/PRAM Address (N channels) (RUBY.DMA_DMEM_PRAM_ADDR) */
typedef struct{

__REG32 starting_address    :20;
__REG32                     :12;

} __ruby_dma_dmem_pram_addr_bits; 

/* AXI Transfer Count Register (N channels) (RUBY.DMA_AXI_TRANS_CNT) */
typedef struct{

__REG32 Count               :16;
__REG32                     :16;

} __ruby_dma_axi_trans_cnt_bits; 

/* SLCDC Data Buffer Base Address Register (SLCDC.DATABASEADR)*/
typedef struct{

__REG32               : 2;
__REG32 DATABASEADDR  :30;

} __slcdc_databaseadr_bits;

/* SLCDC Data Buffer Size Register (SLCDC.DATABUFSIZE) */
typedef struct{

__REG32 DATABUFSIZE  :17;
__REG32              :15;

} __slcdc_databufsize_bits;

/* SLCDC Command Buffer Base Address Register (SLCDC.COMBASEADR)*/
typedef struct{

__REG32               : 2;
__REG32 COMBASEADDR   :30;

} __slcdc_combaseaddr_bits;

/* SLCDC Command Buffer Size Register (SLCDC.COMBUFSIZE) */
typedef struct{

__REG32 COMBUFSIZE  :17;
__REG32             :15;

} __slcdc_combufsize_bits;

/* SLCDC Command String Size Register (SLCDC.COMSTRINGSIZE) */
typedef struct{

__REG32 COMSTRINGSIZ  : 8;
__REG32               :24;

} __slcdc_comstringsize_bits;

/* SLCDC FIFO Configuration Register (SLCDC.FIFOCONFIG)*/
typedef struct{

__REG32 BURST  : 3;
__REG32        :29;

} __slcdc_fifoconfig_bits;

/* SLCDC LCD Controller Configuration Register (SLCDC.LCDCONFIG) */
typedef struct{

__REG32 WORDPPAGE  :13;
__REG32            :19;

} __slcdc_lcdconfig_bits;

/* SLCDC LCD Transfer Configuration Register (SLCDC.LCDTRANSCONFIG)*/
typedef struct{

__REG32 SKCPOL        : 1;
__REG32 CSPOL         : 1;
__REG32 XFRMODE       : 1;
__REG32 WORDDEFCOM    : 1;
__REG32 WORDDEFDAT    : 1;
__REG32 WORDDEFWRITE  : 1;
__REG32               :10;
__REG32 IMGEND        : 2;
__REG32               :14;

} __slcdc_lcdtransconfig_bits;

/* SLCDC Control/Status Register (SLCDC.SLCDCCTL_STATUS) */
typedef struct{

__REG32 GO        : 1;
__REG32 ABORT     : 1;
__REG32 BUSY      : 1;
__REG32           : 1;
__REG32 TEA       : 1;
__REG32 UNDRFLOW  : 1;
__REG32 IRQ       : 1;
__REG32 IRQEN     : 1;
__REG32 PROT1     : 1;
__REG32           : 2;
__REG32 AUTOMODE  : 2;
__REG32           :19;

} __slcdc_slcdcctl_status_bits;

/* SLCDC LCD Clock Configuration Register (SLCDC.LCDCLOCKCONFIG) */
typedef struct{

__REG32 DIVIDE  : 6;
__REG32         :26;

} __slcdc_lcdclockconfig_bits;

/* SLCDC LCD Write Data Register (SLCDC.LCDWRITEDATA) */
typedef struct{

__REG32 LCDDAT  :16;
__REG32 RS      : 1;
__REG32         :15;

} __slcdc_lcdwritedata_bits; 

/* SRC System Control Register (SRC.SSCR) */
typedef struct{

__REG32 PDN_CLKMON_CKIH       : 1;
__REG32 PDN_CLKMON_CKIL_AP    : 1;
__REG32                       : 1;
__REG32 BY_AMP_CKIH           : 1;
__REG32 PDN_AMP_CKIH          : 1;
__REG32 LCCKIL                : 1;
__REG32 MODEM_RST             : 1;
__REG32                       : 1;
__REG32 DPLL_TRIM             : 4;
__REG32                       : 4;
__REG32 PSEUDO_SLEEP          : 1;
__REG32 REGUL_DPLL_EN         : 1;
__REG32 DPLL_OUT_CNTL         : 3;
__REG32                       :11;

} __src_sscr_bits;

/* SRC Boot Mode Register (SRC.SBMR) */
typedef struct{

__REG32 FIU                   : 6;
__REG32                       : 4;
__REG32 BOOT_SEL              : 2;
__REG32                       : 2;
__REG32 OSC_FREQ              : 2;
__REG32 RBL_DIS               : 1;
__REG32                       :11;
__REG32 BOOT_CFG              : 2;
__REG32                       : 2;

} __src_sbmr_bits;

/* SRC Reset Status Register (SRC.SRSR) */
typedef struct{

__REG32 PORIN_B               : 1;
__REG32 CLKMON_CKIH           : 1;
__REG32 CLKMON_CKIL           : 1;
__REG32 WDOG                  : 1;
__REG32 SW                    : 1;
__REG32 JRST                  : 1;
__REG32 COLD_USER             : 1;
__REG32 MODEM                 : 1;
__REG32 JTAG_SW               : 1;
__REG32                       : 7;
__REG32 MODEM_RST_STAT        : 1;
__REG32                       :15;

} __src_srsr_bits;

/* SRC Cold Reset Count Register (SRC.SCRCR) */

typedef struct{
__REG32 CNT                   : 2;
__REG32                       :30;

} __src_scrcr_bits;

/* SRC Reset Bypass Register (SRC.SRBR) */
typedef struct{

__REG32                       : 1;
__REG32 CLKMON_CKIH_BYP       : 1;
__REG32 CLKMON_CKIL_AP_BYP    : 1;
__REG32 WDOG_AP_BYP           : 1;
__REG32 SW_AP_BYP             : 1;
__REG32 HIGHZ_JRST_BYP        : 1;
__REG32 RSTIN_BYP             : 1;
__REG32 JTAG_SW               : 1;
__REG32                       :24;

} __src_srbr_bits;

/* SRC General Purpose Register (SRC.SGPR) */
typedef struct{

__REG32 PFD_DA_CTRL               : 2;
__REG32 PFD_BO_CTRL               : 2;
__REG32                           : 4;
__REG32 CRVCS                     : 2;
__REG32                           :22;

} __src_sgpr_bits;
 
/* Analog RF Interface Register1 (SRC.AIR1) */
typedef struct{

__REG32 ref_osc_agcctrl           : 6;
__REG32                           : 2;
__REG32 ref_osc_atst_sel          : 4;
__REG32                           : 1;
__REG32 ref_osc_regul_atst_en     : 1;
__REG32 ref_osc_crosc_atst_en     : 1;
__REG32                           : 1;
__REG32 ref_osc_sel               : 1;
__REG32                           : 1;
__REG32 mpll_1p4_reg_en           : 1;
__REG32 mpll_1p4_reg_vout         : 2;
__REG32 vagreg_enagnd_gen         : 1;
__REG32 vagreg_enresbias          : 1;
__REG32 vagreg_enresgnd_gen       : 1;
__REG32 vagreg_envag1v25buf       : 1;
__REG32 reg_1p2_sel               : 3;
__REG32                           : 1;
__REG32 reg_1p4_sel               : 3;

} __src_air1_bits;
 
/* Analog RF Interface Register2 (SRC.AIR2) */
typedef struct{

__REG32 reg_1p2_en                : 1;
__REG32 reg_1p4_en                : 1;
__REG32 reg_1p8_en                : 1;
__REG32 reg_2p4_en                : 1;
__REG32 spare6                    : 1;
__REG32 spare7                    : 1;
__REG32 spare8                    : 1;
__REG32 spare9                    : 1;
__REG32 spare10                   : 1;
__REG32                           : 6;
__REG32 RRES                      : 1;
__REG32 ref_osc_crosscore_peakdet : 2;
__REG32 RF_ISO_BYP                : 1;
__REG32 RF_ISO_EN                 : 1;
__REG32                           :12;

} __src_air2_bits; 

/* SSI Control Register (SSI.SCR) */
typedef struct{

__REG32 SSIEN               : 1;
__REG32 TE                  : 1;
__REG32 RE                  : 1;
__REG32 NET                 : 1;
__REG32 SYN                 : 1;
__REG32 I2S_MODE            : 2;
__REG32 SYS_CLK_EN          : 1;
__REG32 TCH_EN              : 1;
__REG32 CLK_IST             : 1;
__REG32 TFR_CLK_DIS         : 1;
__REG32 RFR_CLK_DIS         : 1;
__REG32                     :20;

} __ssi_scr_bits;
 
/* SSI Interrupt Status Register (SSI.SISR) */
typedef struct{

__REG32 TFE0                : 1;
__REG32 TFE1                : 1;
__REG32 RFF0                : 1;
__REG32 RFF1                : 1;
__REG32 RLS                 : 1;
__REG32 TLS                 : 1;
__REG32 RFS                 : 1;
__REG32 TFS                 : 1;
__REG32 TUE0                : 1;
__REG32 TUE1                : 1;
__REG32 ROE0                : 1;
__REG32 ROE1                : 1;
__REG32 TDE0                : 1;
__REG32 TDE1                : 1;
__REG32 RDR0                : 1;
__REG32 RDR1                : 1;
__REG32 RXT                 : 1;
__REG32 CMDDU               : 1;
__REG32 CMDAU               : 1;
__REG32                     : 4;
__REG32 TRFC                : 1;
__REG32 RFRC                : 1;
__REG32                     : 7;

} __ssi_sisr_bits;
 
/* SSI Interrupt Enable Register (SSI.SIER) */
typedef struct{

__REG32 TFE0_EN             : 1;
__REG32 TFE1_EN             : 1;
__REG32 RFF0_EN             : 1;
__REG32 RFF1_EN             : 1;
__REG32 RLS_EN              : 1;
__REG32 TLS_EN              : 1;
__REG32 RFS_EN              : 1;
__REG32 TFS_EN              : 1;
__REG32 TUE0_EN             : 1;
__REG32 TUE1_EN             : 1;
__REG32 ROE0_EN             : 1;
__REG32 ROE1_EN             : 1;
__REG32 TDE0_EN             : 1;
__REG32 TDE1_EN             : 1;
__REG32 RDR0_EN             : 1;
__REG32 RDR1_EN             : 1;
__REG32 RXT_EN              : 1;
__REG32 CMDDU_EN            : 1;
__REG32 CMDAU_EN            : 1;
__REG32 TIE                 : 1;
__REG32 TDMAE               : 1;
__REG32 RIE                 : 1;
__REG32 RDMAE               : 1;
__REG32 TFRC_EN             : 1;
__REG32 RFRC_EN             : 1;
__REG32                     : 7;

} __ssi_sier_bits;
 
/* SSI Transmit Configuration Register (SSI.STCR) */
typedef struct{

__REG32 TEFS                : 1;
__REG32 TFSL                : 1;
__REG32 TFSI                : 1;
__REG32 TSCKP               : 1;
__REG32 TSHFD               : 1;
__REG32 TXDIR               : 1;
__REG32 TFDIR               : 1;
__REG32 TFEN0               : 1;
__REG32 TFEN1               : 1;
__REG32 TXBIT0              : 1;
__REG32                     :22;

} __ssi_stcr_bits;
 
/* SSI Receive Configuration Register (SSI.SRCR) */
typedef struct{

__REG32 REFS                : 1;
__REG32 RFSL                : 1;
__REG32 RFSI                : 1;
__REG32 RSCKP               : 1;
__REG32 RSHFD               : 1;
__REG32 RXDIR               : 1;
__REG32 RFDIR               : 1;
__REG32 RFEN0               : 1;
__REG32 RFEN1               : 1;
__REG32 RXBIT0              : 1;
__REG32 RXEXT               : 1;
__REG32                     :21;

} __ssi_srcr_bits;
 
/* SSI Transmit and Receive Clock Control Register (SSI.STCCR and SSI.SRCCR) */
typedef struct{

__REG32 PM0                 : 1;
__REG32 PM1                 : 1;
__REG32 PM2                 : 1;
__REG32 PM3                 : 1;
__REG32 PM4                 : 1;
__REG32 PM5                 : 1;
__REG32 PM6                 : 1;
__REG32 PM7                 : 1;
__REG32 DC0                 : 1;
__REG32 DC1                 : 1;
__REG32 DC2                 : 1;
__REG32 DC3                 : 1;
__REG32 DC4                 : 1;
__REG32 WL0                 : 1;
__REG32 WL1                 : 1;
__REG32 WL2                 : 1;
__REG32 WL3                 : 1;
__REG32 PSR                 : 1;
__REG32 DIV2                : 1;
__REG32                     :13;

} __ssi_strccr_bits;
 
/* SSI FIFO Control/Status Register (SSI.SFCSR) */
typedef struct{

__REG32 TFWM0               : 4;
__REG32 RFWM0               : 4;
__REG32 TFCNT0              : 4;
__REG32 RFCNT0              : 4;
__REG32 TFWM1               : 4;
__REG32 RFWM1               : 4;
__REG32 TFCNT1              : 4;
__REG32 RFCNT1              : 4;

} __ssi_sfcsr_bits;
 
/* SSI AC97 Control Register (SSI.SACNT) */
typedef struct{

__REG32 AC97EN              : 1;
__REG32 FV                  : 1;
__REG32 TIF                 : 1;
__REG32 RD                  : 1;
__REG32 WR                  : 1;
__REG32 FRDIV               : 6;
__REG32                     :21;

} __ssi_sacnt_bits; 

/* SSI AC97 Command Address Register (SSI.SACADD) */
typedef struct{

__REG32 SACADD              :19;
__REG32                     :13;

} __ssi_sacadd_bits;

/* SSI AC97 Command Data Register (SSI.SACDAT) */
typedef struct{

__REG32 SACADD              :20;
__REG32                     :12;

} __ssi_sacdat_bits;
 
/* SSI AC97 Tag Register (SSI.SATAG) */
typedef struct{

__REG32 SATAG               :16;
__REG32                     :16;

} __ssi_satag_bits;

/* SSI Transmit Time Slot Mask Register (SSI.STMSK) */
typedef struct{

__REG32 STMSK0              : 1;
__REG32 STMSK1              : 1;
__REG32 STMSK2              : 1;
__REG32 STMSK3              : 1;
__REG32 STMSK4              : 1;
__REG32 STMSK5              : 1;
__REG32 STMSK6              : 1;
__REG32 STMSK7              : 1;
__REG32 STMSK8              : 1;
__REG32 STMSK9              : 1;
__REG32 STMSK10             : 1;
__REG32 STMSK11             : 1;
__REG32 STMSK12             : 1;
__REG32 STMSK13             : 1;
__REG32 STMSK14             : 1;
__REG32 STMSK15             : 1;
__REG32 STMSK16             : 1;
__REG32 STMSK17             : 1;
__REG32 STMSK18             : 1;
__REG32 STMSK19             : 1;
__REG32 STMSK20             : 1;
__REG32 STMSK21             : 1;
__REG32 STMSK22             : 1;
__REG32 STMSK23             : 1;
__REG32 STMSK24             : 1;
__REG32 STMSK25             : 1;
__REG32 STMSK26             : 1;
__REG32 STMSK27             : 1;
__REG32 STMSK28             : 1;
__REG32 STMSK29             : 1;
__REG32 STMSK30             : 1;
__REG32 STMSK31             : 1;

} __ssi_stmsk_bits;

/* SSI Receive Time Slot Mask Register (SSI.SRMSK) */
typedef struct{

__REG32 SRMSK0              : 1;
__REG32 SRMSK1              : 1;
__REG32 SRMSK2              : 1;
__REG32 SRMSK3              : 1;
__REG32 SRMSK4              : 1;
__REG32 SRMSK5              : 1;
__REG32 SRMSK6              : 1;
__REG32 SRMSK7              : 1;
__REG32 SRMSK8              : 1;
__REG32 SRMSK9              : 1;
__REG32 SRMSK10             : 1;
__REG32 SRMSK11             : 1;
__REG32 SRMSK12             : 1;
__REG32 SRMSK13             : 1;
__REG32 SRMSK14             : 1;
__REG32 SRMSK15             : 1;
__REG32 SRMSK16             : 1;
__REG32 SRMSK17             : 1;
__REG32 SRMSK18             : 1;
__REG32 SRMSK19             : 1;
__REG32 SRMSK20             : 1;
__REG32 SRMSK21             : 1;
__REG32 SRMSK22             : 1;
__REG32 SRMSK23             : 1;
__REG32 SRMSK24             : 1;
__REG32 SRMSK25             : 1;
__REG32 SRMSK26             : 1;
__REG32 SRMSK27             : 1;
__REG32 SRMSK28             : 1;
__REG32 SRMSK29             : 1;
__REG32 SRMSK30             : 1;
__REG32 SRMSK31             : 1;

} __ssi_srmsk_bits;
 
/* SSI AC97 Channel Status Register (SSI.SACCST) */
typedef struct{

__REG32 SACCST0             : 1;
__REG32 SACCST1             : 1;
__REG32 SACCST2             : 1;
__REG32 SACCST3             : 1;
__REG32 SACCST4             : 1;
__REG32 SACCST5             : 1;
__REG32 SACCST6             : 1;
__REG32 SACCST7             : 1;
__REG32 SACCST8             : 1;
__REG32 SACCST9             : 1;
__REG32                     :22;

} __ssi_saccst_bits;
   
/* SSI AC97 Channel Enable Register (SSI.SACCEN) */
typedef struct{

__REG32 SACCEN0             : 1;
__REG32 SACCEN1             : 1;
__REG32 SACCEN2             : 1;
__REG32 SACCEN3             : 1;
__REG32 SACCEN4             : 1;
__REG32 SACCEN5             : 1;
__REG32 SACCEN6             : 1;
__REG32 SACCEN7             : 1;
__REG32 SACCEN8             : 1;
__REG32 SACCEN9             : 1;
__REG32                     :22;

} __ssi_saccen_bits;

/* SSI AC97 Channel Disable Register (SSI.SACCDIS) */
typedef struct{

__REG32 SACCDIS0            : 1;
__REG32 SACCDIS1            : 1;
__REG32 SACCDIS2            : 1;
__REG32 SACCDIS3            : 1;
__REG32 SACCDIS4            : 1;
__REG32 SACCDIS5            : 1;
__REG32 SACCDIS6            : 1;
__REG32 SACCDIS7            : 1;
__REG32 SACCDIS8            : 1;
__REG32 SACCDIS9            : 1;
__REG32                     :22;

} __ssi_saccdis_bits;
  
/* TPRC lutvals_per_ramp (TPRC.lutvals_per_ramp) */
typedef struct{

__REG16 data                : 5;
__REG16                     :11;

} __tprc_16_5bit_bits;

/* TPRC clks_betw_lut (TPRC.clks_betw_lut) */
typedef struct{

__REG16 data                : 3;
__REG16                     :13;

} __tprc_16_3bit_bits;

/* TPRC ramp_end_value (TPRC.ramp_end_value) */
/* TPRC dig_tprc_dac_out (TPRC.dig_tprc_dac_out) */
typedef struct{

__REG16 data                :14;
__REG16                     : 2;

} __tprc_16_14bit_bits;

/* TPRC mcu_ramp_trigger (TPRC.mcu_ramp_trigger) */
/* TPRC mcu_ramp_dac_reset (TPRC.mcu_ramp_dac_reset) */
/* TPRC tprc_ramp_bypass (TPRC.tprc_ramp_bypass) */
/* TPRC tprc_static (TPRC.tprc_static) */
/* TPRC tprc_dacbuf_en (TPRC.tprc_dacbuf_en) */
/* TPRC tprc_dac_en (TPRC.tprc_dac_en) */
/* TPRC tprc_dac_daz (TPRC.tprc_dac_daz) */
/* TPRC tprc_clk_en (TPRC.tprc_clk_en) */
typedef struct{

__REG16 data                : 1;
__REG16                     :15;

} __tprc_16_1bit_bits;

/* TPRC LUT (TPRC.LUT_0-LUT_F) */
typedef struct{

__REG16 data                : 8;
__REG16                     : 8;

} __tprc_16_8bit_bits;
 
/* TSM Control Register (TSM.IPR_TSM_CTRL) */
typedef struct{

__REG32 TMR_EN              : 1;
__REG32 TSM_START           : 1;
__REG32 TMR_SEL             : 2;
__REG32 TSM_WARMDOWN        : 1;
__REG32 TSM_MODE            : 1;
__REG32 MUX_IN_EN           : 1;
__REG32 RESET_PTR           : 1;
__REG32 SUSPEND_WD          : 1;
__REG32                     :23;

} __tsm_ipr_tsm_ctrl_bits; 

/* TSM RX Steps Register (TSM.IPR_TSM_RX_STEPS) */
typedef struct{

__REG32 LAST_RX_WU_STEP     : 5;
__REG32 LAST_RX_WD_STEP     : 5;
__REG32                     :22;

} __tsm_ipr_tsm_rx_steps_bits;  

/* TSM TX Steps Register (TSM.IPR_TSM_TX_STEPS) */
typedef struct{

__REG32 LAST_TX_WU_STEP     : 5;
__REG32 LAST_TX_WD_STEP     : 5;
__REG32                     :22;

} __tsm_ipr_tsm_tx_steps_bits;  

/* TSM RX Table Register (TSM.IPR_TSM_RX_TBL) */
typedef struct{

__REG32 COM_OUT             :10;
__REG32 RX_OUT              :14;
__REG32 RX_DURATIONS        : 8;

} __tsm_ipr_tsm_rx_tbl_bits; 
 
/* TSM TX Table Register (TSM.IPR_TSM_TX_TBL) */
typedef struct{

__REG32 COM_OUT             :10;
__REG32 TX_OUT              :14;
__REG32 TX_DURATIONS        : 8;

} __tsm_ipr_tsm_tx_tbl_bits;  

/* TSM Warmdown Time Register (TSM.IPR_TSM_WARMDOWN) */
typedef struct{

__REG32 WARMDOWN_TIMER      :16;
__REG32                     :16;

} __tsm_ipr_tsm_warmdown_bits;  

/* TSM Multiplex RX Input Register (TSM.IPR_TSM_MUX_IN_RX) */
typedef struct{

__REG32 MUX_IN_RX           :14;
__REG32                     :18;

} __tsm_ipr_tsm_mux_in_rx_bits;   

/* TSM Interrupt Mask Enable Register (TSM.IPR_TSM_IMR) */
typedef struct{

__REG32 EQD_EN              : 1;
__REG32                     : 1;
__REG32 GP_IRQ_EN           : 1;
__REG32                     :29;

} __tsm_ipr_tsm_imr_bits;   

/* TSM Interrupt Status Register (TSM.IPR_TSM_ISR) */
typedef struct{

__REG32 RX_EWD_IRQ          : 1;
__REG32 TX_EWD_IRQ          : 1;
__REG32 GP_IRQ              : 1;
__REG32                     :29;

} __tsm_ipr_tsm_isr_bits;  

/* TSM Multiplex RX Output Register (TSM.IPR_TSM_MUX_OUT_RX) */
typedef struct{

__REG32 MUX_OUT_RX          :14;
__REG32                     :18;

} __tsm_ipr_tsm_mux_out_rx_bits;

/* TSM CCM Control Register (TSM.IPR_TSM_CCM_CTRL) */
typedef struct{

__REG32 TSM_CLK_EN          : 1;
__REG32                     :31;

} __tsm_ipr_tsm_ccm_ctrl_bits; 

/* TSM Multiplex TX Input Register (TSM.IPR_TSM_MUX_IN_TX) */
typedef struct{

__REG32 MUX_IN_TX           :14;
__REG32                     :18;

} __tsm_ipr_tsm_mux_in_tx_bits;

/* TSM Multiplex TX Output Register (TSM.IPR_TSM_MUX_OUT_TX) */
typedef struct{

__REG32 MUX_OUT_TX          :14;
__REG32                     :18;

} __tsm_ipr_tsm_mux_out_tx_bits;

/* TSM Multiplex Common Input Register (TSM.IPR_TSM_MUX_IN_COM) */
typedef struct{

__REG32 MUX_IN_COM          :10;
__REG32                     :22;

} __tsm_ipr_tsm_mux_in_com_bits;

/* TSM Multiplex Common Output Register (TSM.IPR_TSM_MUX_OUT_COM) */
typedef struct{

__REG32 MUX_OUT_COM         :10;
__REG32                     :22;

} __tsm_ipr_tsm_mux_out_com_bits;

/* UART Receiver Register (UART.URXD) */
typedef struct{

__REG32 RX_DATA  : 8;     /* Bits 0-7             - Recieve Data*/
__REG32          : 2;     /* Bits 8-9             - Reserved*/
__REG32 PRERR    : 1;     /* Bit  10              - Receive Parity Error 1=error*/
__REG32 BRK      : 1;     /* Bit  11              - Receive break Caracter detected 1 = detected*/
__REG32 FRMERR   : 1;     /* Bit  12              - Receive Framing Error 1=error*/
__REG32 OVRRUN   : 1;     /* Bit  13              - Receive Over run Error 1=error*/
__REG32 ERR      : 1;     /* Bit  14              - Receive Error Detect (OVR,FRM,BRK,PR 0=error*/
__REG32 CHARRDY  : 1;     /* Bit  15              - Receive Character Ready 1=character valid*/
__REG32          :16;     /* Bits 31-16           - Reserved*/

} __uart_urxd_bits;

/* UART Transmitter Register (UART.UTXD) */
typedef struct{

__REG32 TX_DATA  : 8;     /* Bits 7-0             - Transmit Data*/
__REG32          :24;     /* Bits 31-16           - Reserved*/

} __uart_utxd_bits;

/* UART Control Register 1 (UART.UCR1) */
typedef struct{

__REG32 UARTEN    : 1;     /* Bit  0       - UART Enable 1 = Enable the UART*/
__REG32 DOZE      : 1;     /* Bit  1       - DOZE 1 = The UART is disabled when in DOZE state*/
__REG32 ATDMAEN   : 1;     /* Bit  2*/
__REG32 TXDMAEN   : 1;     /* Bit  3       - Transmitter Ready DMA Enable 1 = enable*/
__REG32 SNDBRK    : 1;     /* Bit  4       - Send BREAK 1 = send break char continuous*/
__REG32 RTSDEN    : 1;     /* Bit  5       - RTS Delta Interrupt Enable 1 = enable*/
__REG32 TXMPTYEN  : 1;     /* Bit  6       - Transmitter Empty Interrupt Enable 1 = enable*/
__REG32 IREN      : 1;     /* Bit  7       - Infrared Interface Enable 1 = enable*/
__REG32 RXDMAEN   : 1;     /* Bit  8       - Receive Ready DMA Enable 1 = enable*/
__REG32 RRDYEN    : 1;     /* Bit  9       - Receiver Ready Interrupt Enable 1 = Enable*/
__REG32 ICD       : 2;     /* Bit  10-11   - Idle Condition Detect*/
                                /*              - 00 = Idle for more than 4 frames*/
                                /*              - 01 = Idle for more than 8 frames*/
                                /*              - 10 = Idle for more than 16 frames*/
                                /*              - 11 = Idle for more than 32 frames*/
__REG32 IDEN      : 1;     /* Bit  12      - Idle Condition Detected Interrupt en 1=en*/
__REG32 TRDYEN    : 1;     /* Bit  13      - Transmitter Ready Interrupt Enable 1=en*/
__REG32 ADBR      : 1;     /* Bit  14      - AutoBaud Rate Detection enable 1=en*/
__REG32 ADEN      : 1;     /* Bit  15      - AutoBaud Rate Detection Interrupt en 1=en*/
__REG32           :16;

} __uart_ucr1_bits;
 
/* UART Control Register 2 (UART.UCR2) */
typedef struct{

__REG32 SRST   : 1;     /* Bit  0       -Software Reset 0 = Reset the tx and rx state machines*/
__REG32 RXEN   : 1;     /* Bit  1       -Receiver Enable 1 = Enable*/
__REG32 TXEN   : 1;     /* Bit  2       -Transmitter Enable 1= enable*/
__REG32 ATEN   : 1;     /* Bit  3       -Aging Timer Enable—This bit is used to mask the aging timer interrupt (triggered with AGTIM)*/
__REG32 RTSEN  : 1;     /* Bit  4       -Request to Send Interrupt Enable 1=enable*/
__REG32 WS     : 1;     /* Bit  5       -Word Size 0 = 7bit, 1= 8 bit*/
__REG32 STPB   : 1;     /* Bit  6       -Stop 0= 1 stop bits, 1= 2 stop bits*/
__REG32 PROE   : 1;     /* Bit  7       -Parity Odd/Even 1=Odd*/
__REG32 PREN   : 1;     /* Bit  8       -Parity Enable 1=enable parity generator*/
__REG32 RTEC   : 2;     /* Bits 9-10    -Request to Send Edge Control*/
                                /*              - 00 = Trigger interrupt on a rising edge*/
                                /*              - 01 = Trigger interrupt on a falling edge*/
                                /*              - 1X = Trigger interrupt on any edge*/
__REG32 ESCEN  : 1;     /* Bit  11      -Escape Enable 1 = Enable escape sequence detection*/
__REG32 CTS    : 1;     /* Bit  12      -Clear to Send 1 = The UARTx_CTS pin is low (active)*/
__REG32 CTSC   : 1;     /* Bit  13      -UARTx_CTS Pin controlled by 1= receiver 0= CTS bit*/
__REG32 IRTS   : 1;     /* Bit  14      -Ignore UARTx_RTS Pin 1=ignore*/
__REG32 ESCI   : 1;     /* Bit  15      -Escape Sequence Interrupt En 1=enable*/
__REG32        :16;

} __uart_ucr2_bits;
 
/* UART Control Register 3 (UART.UCR3) */
typedef struct{

__REG32 ACIEN      : 1;
__REG32 INVT       : 1;
__REG32 RXDMUXSEL  : 1;
__REG32 DTRDEN     : 1;
__REG32 AWAKEN     : 1;
__REG32 AIRINTEN   : 1;
__REG32 RXDSEN     : 1;
__REG32 ADNIMP     : 1;
__REG32 RI         : 1;
__REG32 DCD        : 1;
__REG32 DSR        : 1;
__REG32 FRAERREN   : 1;
__REG32 PARERREN   : 1;
__REG32 DTREN      : 1;
__REG32 DPEC       : 2;
__REG32            :16;

} __uart_ucr3_bits;
 
/* UART Control Register 4 (UART.UCR4) */
typedef struct{

__REG32 DREN   : 1;     /* Bit  0       -Receive Data Ready Interrupt Enable 1= enable*/
__REG32 OREN   : 1;     /* Bit  1       -Receiver Overrun Interrupt Enable 1= enable*/
__REG32 BKEN   : 1;     /* Bit  2       -BREAK Condition Detected Interrupt en 1= enable*/
__REG32 TCEN   : 1;     /* Bit  3       -Transmit Complete Interrupt Enable1 = Enable*/
__REG32 LPBYP  : 1;     /* Bit  4       -Low Power Bypass—Allows to bypass the low power new features in UART for i.MX31. To use during debug phase.*/
__REG32 IRSC   : 1;     /* Bit  5       -IR Special Case vote logic uses 1= uart ref clk*/
__REG32 IDDMAEN: 1;     /* Bit  6       -*/
__REG32 WKEN   : 1;     /* Bit  7       -WAKE Interrupt Enable 1= enable*/
__REG32 ENIRI  : 1;     /* Bit  8       -Serial Infrared Interrupt Enable 1= enable*/
__REG32 INVR   : 1;     /* Bit  9       -Inverted Infrared Reception 1= active high*/
__REG32 CTSTL  : 6;     /* Bits 10-15   -CTS Trigger Level*/
                                /*              000000 = 0 characters received*/
                                /*              000001 = 1 characters in the RxFIFO*/
                                /*              ...*/
                                /*              100000 = 32 characters in the RxFIFO (maximum)*/
                                /*              All Other Settings Reserved*/
__REG32        :16;

} __uart_ucr4_bits;

/* UART FIFO Control Register (UART.UFCR) */
typedef struct{
__REG32 RXTL    : 6;     /* Bits 0-5     -Receiver Trigger Level*/
                                /*              000000 = 0 characters received*/
                                /*              000001 = RxFIFO has 1 character*/
                                /*              ...*/
                                /*              011111 = RxFIFO has 31 characters*/
                                /*              100000 = RxFIFO has 32 characters (maximum)*/
                                /*              All Other Settings Reserved*/
__REG32 DCEDTE  : 1;     /* Bit  6       -*/
__REG32 RFDIV   : 3;     /* Bits 7-9     -Reference Frequency Divider*/
                                /*              000 = Divide input clock by 6*/
                                /*              001 = Divide input clock by 5*/
                                /*              010 = Divide input clock by 4*/
                                /*              011 = Divide input clock by 3*/
                                /*              100 = Divide input clock by 2*/
                                /*              101 = Divide input clock by 1*/
                                /*              110 = Divide input clock by 7*/
__REG32 TXTL    : 6;     /* Bits 10-15   -Transmitter Trigger Level*/
                                /*              000000 = Reserved*/
                                /*              000001 = Reserved*/
                                /*              000010 = TxFIFO has 2 or fewer characters*/
                                /*              ...*/
                                /*              011111 = TxFIFO has 31 or fewer characters*/
                                /*              100000 = TxFIFO has 32 characters (maximum)*/
                                /*              All Other Settings Reserved*/
__REG32         :16;
} __uart_ufcr_bits;
    
/* UART Status Register 1 (UART.USR1) */
typedef struct{

__REG32            : 4;
__REG32 AWAKE      : 1;
__REG32 AIRINT     : 1;
__REG32 RXDS       : 1;
__REG32 DTRD       : 1;
__REG32 AGTIM      : 1;
__REG32 RRDY       : 1;
__REG32 FRAMER     : 1;
__REG32 ESCF       : 1;
__REG32 RTSD       : 1;
__REG32 TRDY       : 1;
__REG32 RTSS       : 1;
__REG32 PARITYERR  : 1;
__REG32            :16;

} __uart_usr1_bits;
 
/* UART Status Register 2 (UART.USR2) */
typedef struct{

__REG32 RDR      : 1;
__REG32 ORE      : 1;
__REG32 BRCD     : 1;
__REG32 TXDC     : 1;
__REG32 RTSF     : 1;
__REG32 DCDIN    : 1;
__REG32 DCDDELT  : 1;
__REG32 WAKE     : 1;
__REG32 IRINT    : 1;
__REG32 RIIN     : 1;
__REG32 RIDELT   : 1;
__REG32 ACST     : 1;
__REG32 IDLE     : 1;
__REG32 DTRF     : 1;
__REG32 TXFE     : 1;
__REG32 ADET     : 1;
__REG32          :16;

} __uart_usr2_bits;

/* UART Escape Character Register (UART.UESC) */
typedef struct{

__REG32 ESC_CHAR  : 8;     /* Bits 0-7     -UART Escape Character*/
__REG32           :24;

} __uart_uesc_bits;

/* UART Escape Timer Register (UART.UTIM) */
typedef struct{

__REG32 TIM  :12;     /* Bits 0-11    -UART Escape Timer*/
__REG32      :20;

} __uart_utim_bits;

/* UART BRM Incremental Register (UART.UBIR) */
typedef struct{

__REG32 INC  :16;     /* Bits 0-15    -Incremental Numerator*/
__REG32      :16;

} __uart_ubir_bits;

/* UART BRM Modulator Register (UART.UBMR) */
typedef struct{

__REG32 MOD  :16;     /* Bits 0-15    -Modulator Denominator*/
__REG32      :16;

} __uart_ubmr_bits;

/* UART Baud Rate Count register (UART.UBRC) */
typedef struct{

__REG32 BCNT  :16;     /* Bits 0-15    -Baud Rate Count Register*/
__REG32       :16;

} __uart_ubrc_bits;

/* UART One Millisecond Register (UART.ONEMS) */
typedef struct{

__REG32 ONEMS :24;
__REG32       : 8;

} __uart_onems_bits;
 
/* UART Test Register 1 (UART.UTS) */
typedef struct{

__REG32 SOFTRST  : 1;
__REG32          : 2;
__REG32 RXFULL   : 1;
__REG32 TXFULL   : 1;
__REG32 RXEMPTY  : 1;
__REG32 TXEMPTY  : 1;
__REG32          : 2;
__REG32 RXDBG    : 1;
__REG32 LOOPIR   : 1;
__REG32 DBGEN    : 1;
__REG32 LOOP     : 1;
__REG32 FRCPERR  : 1;
__REG32          :18;

} __uart_uts_bits;
 
/* Peripheral ID Register (USB_FS.PER_ID) */
typedef struct{

__REG8  ID      : 6;
__REG8          : 2;

} __usb_fs_per_id_bits;
  
/* Peripheral ID Complement Register (USB_FS.ID_COMP) */
typedef struct{

__REG8  NID     : 6;
__REG8          : 2;

} __usb_fs_id_comp_bits;  
 
/* Peripheral Additional Info Register (USB_FS.ADD_INFO) */
typedef struct{

__REG8  HOST    : 1;
__REG8          : 2;
__REG8  IRQ_NUM : 5;

} __usb_fs_add_info_bits;   

/* OTG Control Register (USB_FS.OTG_CTRL) */
typedef struct{

__REG8          : 6;
__REG8  DM_HIGH : 1;
__REG8  DP_HIGH : 1;

} __usb_fs_otg_ctrl_bits; 

/* Interrupt Status Register (USB_FS.INT_STAT) */
/* Interrupt Enable Register (USB_FS.INT_ENB) */
typedef struct{

__REG8  USB_RST : 1;
__REG8  ERROR   : 1;
__REG8  SOF_TOK : 1;
__REG8  TOKDNE  : 1;
__REG8  SLEEP   : 1;
__REG8  RESUME  : 1;
__REG8          : 1;
__REG8  STALL   : 1;

} __usb_fs_int_stat_bits; 

/* Error Interrupt Status Register (USB_FS.ERR_STAT) */
/* Error Interrupt Enable Register (USB_FS.ERR_ENB) */
typedef struct{

__REG8  PID_ERR : 1;
__REG8  CRC5    : 1;
__REG8  CRC16   : 1;
__REG8  DFN8    : 1;
__REG8  BTO_ERR : 1;
__REG8  DMA_ERR : 1;
__REG8          : 1;
__REG8  BTS_ERR : 1;

} __usb_fs_err_stat_bits; 
 
/* Status Register (USB_FS.STAT) */
typedef struct{

__REG8          : 2;
__REG8  ODD     : 1;
__REG8  TX      : 1;
__REG8  ENDP    : 4;

} __usb_fs_stat_bits; 
 
/* Control Register (USB_FS.CTL) */
typedef struct{

__REG8  USB_EN     : 1;
__REG8  ODD_RST    : 1;
__REG8  RESUME     : 1;
__REG8             : 2;
__REG8  TxdSuspend : 1;
__REG8             : 2;

} __usb_fs_ctl_bits;  

/* Address Register (USB_FS.ADDR) */
typedef union 
{

  struct{

  __REG8  ADDR0      : 1;
  __REG8  ADDR1      : 1;
  __REG8  ADDR2      : 1;
  __REG8  ADDR3      : 1;
  __REG8  ADDR4      : 1;
  __REG8  ADDR5      : 1;
  __REG8  ADDR6      : 1;
  __REG8             : 1;

  }; 
  
  /* USB_FS_ADDR */
  struct{

  __REG8  ADDR       : 7;
  __REG8             : 1;

  };

} __usb_fs_addr_bits;  

/* Frame Number Register (USB_FS.FRM_NUML) */
typedef struct{

__REG8  FRM        : 8;

} __usb_fs_frm_numl_bits;  

/* Frame Number Register (USB_FS.FRM_NUMH) */
typedef struct{

__REG8  FRM        : 3;
__REG8             : 5;

} __usb_fs_frm_numh_bits;  

/* BDT Page Register 1 (USB_FS.BDT_PAGE_01) */
typedef struct{

__REG8  SELECT_EMI : 1;
__REG8  BDT_BA     : 7;

} __usb_fs_bdt_page_01_bits;  

/* BDT Page Register 2 (USB_FS.BDT_PAGE_02) */
typedef struct{

__REG8  BDT_BA     : 8;

} __usb_fs_bdt_page_02_bits;


/* BDT Page Register 3 (USB_FS.BDT_PAGE_03) */
typedef struct{

__REG8  BDT_BA     : 8;

} __usb_fs_bdt_page_03_bits;

/* Endpoint Control Registers (USB_FS.ENDPT_CTL) */
typedef struct{

__REG8  EP_HSHK     : 1;
__REG8  EP_STALL    : 1;
__REG8  EP_EN_TX    : 1;
__REG8  EP_EN_RX    : 1;
__REG8  EP_CTL_DIS  : 1;
__REG8              : 3;

} __usb_fs_endpt_ctl_bits;

/* Top Resume Register (USB_FS.TOP_CTRL) */
typedef struct{

__REG8  RESUME       : 1;
__REG8  RESUME_INTEN : 1;
__REG8  USB_SUSPND   : 1;
__REG8  USBPHY_BIDIR : 1;
__REG8  USBPHY_SE    : 1;
__REG8               : 3;

} __usb_fs_top_ctrl_bits;
 
/* Watchdog Control Register (WDOG.WCR) */
typedef struct{

__REG16  WDZST  : 1;
__REG16  WDBG   : 1;
__REG16  WDE    : 1;
__REG16  WDT    : 1;
__REG16  SRS    : 1;
__REG16  WDA    : 1;
__REG16         : 1;
__REG16  WDW    : 1;
__REG16  WT     : 8;

} __wdog_wcr_bits;

/* Watchdog Reset Status Register (WDOG.WRSR) */
typedef struct{

__REG16  SFTW   : 1;
__REG16  TOUT   : 1;
__REG16         :14;

} __wdog_wrsr_bits; 

/* Watchdog Interrupt Control Register (WDOG.WICR) */
typedef struct{

__REG16  WICT   : 8;
__REG16         : 6;
__REG16  WTIS   : 1;
__REG16  WIE    : 1;

} __wdog_wicr_bits; 
 
/* Watchdog Miscellaneous Control Register (WDOG.WMCR) */
typedef struct{

__REG16  PDE    : 1;
__REG16         :15;

} __wdog_wmcr_bits; 
 
#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/

/***************************************************************************
 **
 **  MODEM
 **
 ***************************************************************************/
__IO_REG32_BIT(MODEM_TSM_RX_OBS,         0x01080500,__READ      ,__modem_tsm_rx_obs_bits);
__IO_REG32_BIT(MODEM_TSM_TX_OBS,         0x01080504,__READ      ,__modem_tsm_tx_obs_bits);
__IO_REG32_BIT(MODEM_RX_STATUS,          0x01080508,__READ      ,__modem_rx_status_bits);
__IO_REG32_BIT(MODEM_RFDI_STATUS,        0x0108050C,__READ      ,__modem_rfdi_status_bits);
__IO_REG32_BIT(MODEM_DIGVOCOD_STATUS,    0x01080510,__READ      ,__modem_digvocod_status_bits);
__IO_REG32_BIT(MODEM_SYNTH_STATUS,       0x01080514,__READ      ,__modem_synth_status_bits);
__IO_REG32(MODEM_RESERVED1,          0x01080518,__READ      );
__IO_REG32(MODEM_RESERVED2,          0x0108051C,__READ      );
__IO_REG32(MODEM_RESERVED3,          0x01080520,__READ      );
__IO_REG32(MODEM_RESERVED4,          0x01080524,__READ      );
__IO_REG32(MODEM_RESERVED5,          0x01080528,__READ      );
__IO_REG32(MODEM_RESERVED6,          0x0108052C,__READ      );
__IO_REG32(MODEM_RESERVED7,          0x01080530,__READ      );
__IO_REG32(MODEM_RESERVED8,          0x01080534,__READ      );
__IO_REG32(MODEM_RESERVED9,          0x01080538,__READ      );
__IO_REG32(MODEM_RESERVED10,         0x0108053C,__READ      );
__IO_REG32(MODEM_RESERVED11,         0x01080540,__READ      );
__IO_REG32(MODEM_RESERVED12,         0x01080544,__READ      );
__IO_REG32(MODEM_RESERVED13,         0x01080548,__READ      );
__IO_REG32(MODEM_RESERVED14,         0x0108054C,__READ      );
__IO_REG32(MODEM_RESERVED15,         0x01080550,__READ      );
__IO_REG32(MODEM_RESERVED16,         0x01080554,__READ      );
__IO_REG32(MODEM_RESERVED17,         0x01080558,__READ      );
__IO_REG32(MODEM_RESERVED18,         0x0108055C,__READ      );
__IO_REG32(MODEM_RESERVED19,         0x01080560,__READ      );
__IO_REG32(MODEM_RESERVED20,         0x01080564,__READ      );
__IO_REG32(MODEM_RESERVED21,         0x01080568,__READ      );
__IO_REG32(MODEM_RESERVED22,         0x0108056C,__READ      );
__IO_REG32(MODEM_RESERVED23,         0x01080570,__READ      );
__IO_REG32(MODEM_RESERVED24,         0x01080574,__READ      );
__IO_REG32(MODEM_RESERVED25,         0x01080578,__READ      );
__IO_REG32(MODEM_RESERVED26,         0x0108057C,__READ      );
__IO_REG32_BIT(MODEM_RFDI_CFG,           0x01080580,__READ_WRITE,__modem_rfdi_cfg_bits);
__IO_REG32_BIT(MODEM_RFDI_SYNTH_NUM_ADJ, 0x01080584,__READ_WRITE,__modem_rfdi_synth_num_adj_bits);
__IO_REG32_BIT(MODEM_SYNTH_DENOM,        0x01080588,__READ_WRITE,__modem_synth_denom_bits);
__IO_REG32_BIT(MODEM_SYNTH_INTEGER,      0x0108058C,__READ_WRITE,__modem_synth_integer_bits);
__IO_REG32_BIT(MODEM_SYNTH_TUNE,         0x01080590,__READ_WRITE,__modem_synth_tune_bits);
__IO_REG32_BIT(MODEM_SYNTH_TUNE_CYC,     0x01080594,__READ_WRITE,__modem_synth_tune_cyc_bits);
__IO_REG32_BIT(MODEM_SYNTH_LOCKMON,      0x01080598,__READ_WRITE,__modem_synth_lockmon_bits);
__IO_REG32_BIT(MODEM_SYNTH_CFG_TEST,     0x0108059C,__READ_WRITE,__modem_synth_cfg_test_bits);
__IO_REG32_BIT(MODEM_MPLL_CFG,           0x010805A0,__READ_WRITE,__modem_mpll_cfg_bits);
__IO_REG32_BIT(MODEM_RPLL_CFG1,          0x010805A4,__READ_WRITE,__modem_rpll_cfg1_bits);
__IO_REG32_BIT(MODEM_RPLL_CFG2,          0x010805A8,__READ_WRITE,__modem_rpll_cfg2_bits);
__IO_REG32_BIT(MODEM_RX_ADC_CFG,         0x010805AC,__READ_WRITE,__modem_rx_adc_cfg_bits);
__IO_REG32_BIT(MODEM_RX_BBF_CFG,         0x010805B0,__READ_WRITE,__modem_rx_bbf_cfg_bits);
__IO_REG32_BIT(MODEM_RX_DCOC_DAC,        0x010805B4,__READ_WRITE,__modem_rx_dcoc_dac_bits);
__IO_REG32_BIT(MODEM_IIP2_DCOC_CFG,      0x010805B8,__READ_WRITE,__modem_iip2_dac_cfg_bits);
__IO_REG32_BIT(MODEM_RX_GEN_CFG,         0x010805BC,__READ_WRITE,__modem_rx_gen_cfg_bits);
__IO_REG32_BIT(MODEM_TX_SPARE_CFG,       0x010805C0,__READ_WRITE,__modem_tx_spare_cfg_bits);
__IO_REG32_BIT(MODEM_AVOCOD_CFG,         0x010805C4,__READ_WRITE,__modem_avocod_cfg_bits);
__IO_REG32_BIT(MODEM_DIGVOCOD_CFG,       0x010805C8,__READ_WRITE,__modem_digvocod_cfg_bits);
__IO_REG32_BIT(MODEM_GPO,                0x010805CC,__READ_WRITE,__modem_gpo_bits);
__IO_REG32(MODEM_SW1,                0x010805D0,__READ_WRITE);
__IO_REG32(MODEM_SW2,                0x010805D4,__READ_WRITE);
__IO_REG32(MODEM_SW3,                0x010805D8,__READ_WRITE);
__IO_REG32(MODEM_SW4,                0x010805DC,__READ_WRITE);
__IO_REG32(MODEM_SW5,                0x010805E0,__READ_WRITE);
__IO_REG32(MODEM_SW6,                0x010805E4,__READ_WRITE);
__IO_REG32(MODEM_SW7,                0x010805E8,__READ_WRITE);
__IO_REG32(MODEM_SW8,                0x010805EC,__READ_WRITE);
__IO_REG32(MODEM_SW9,                0x010805F0,__READ_WRITE);
__IO_REG32(MODEM_SW10,               0x010805F4,__READ_WRITE);
__IO_REG32(MODEM_SW11,               0x010805F8,__READ_WRITE);
__IO_REG32(MODEM_SW12,               0x010805Fc,__READ_WRITE);

/***************************************************************************
 **
 **  IOMUX
 **
 ***************************************************************************/
 
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_0_3,       0x010CC000,__READ_WRITE,__iomux_sw_mux_ctl_0_3_bits);
#define IOMUX_SW_MUX_CTL           IOMUX_SW_MUX_CTL_0_3          
#define IOMUX_SW_MUX_CTL_bit       IOMUX_SW_MUX_CTL_0_3_bit
#define IOMUX_SW_MUX_CTL_0         IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_0_byte
#define IOMUX_SW_MUX_CTL_0_bit     IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_0_byte_bit
#define IOMUX_SW_MUX_CTL_1         IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_1_byte
#define IOMUX_SW_MUX_CTL_1_bit     IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_1_byte_bit
#define IOMUX_SW_MUX_CTL_2         IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_2_byte
#define IOMUX_SW_MUX_CTL_2_bit     IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_2_byte_bit
#define IOMUX_SW_MUX_CTL_3         IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_3_byte
#define IOMUX_SW_MUX_CTL_3_bit     IOMUX_SW_MUX_CTL_0_3_bit.__sw_mux_ctl_3_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_4_7,       0x010CC004,__READ_WRITE,__iomux_sw_mux_ctl_4_7_bits);
#define IOMUX_SW_MUX_CTL_4         IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_4_byte
#define IOMUX_SW_MUX_CTL_4_bit     IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_4_byte_bit
#define IOMUX_SW_MUX_CTL_5         IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_5_byte
#define IOMUX_SW_MUX_CTL_5_bit     IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_5_byte_bit
#define IOMUX_SW_MUX_CTL_6         IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_6_byte
#define IOMUX_SW_MUX_CTL_6_bit     IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_6_byte_bit
#define IOMUX_SW_MUX_CTL_7         IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_7_byte
#define IOMUX_SW_MUX_CTL_7_bit     IOMUX_SW_MUX_CTL_4_7_bit.__sw_mux_ctl_7_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_8_11,       0x010CC008,__READ_WRITE,__iomux_sw_mux_ctl_8_11_bits);
#define IOMUX_SW_MUX_CTL_8         IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_8_byte
#define IOMUX_SW_MUX_CTL_8_bit     IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_8_byte_bit
#define IOMUX_SW_MUX_CTL_9         IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_9_byte
#define IOMUX_SW_MUX_CTL_9_bit     IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_9_byte_bit
#define IOMUX_SW_MUX_CTL_10        IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_10_byte
#define IOMUX_SW_MUX_CTL_10_bit    IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_10_byte_bit
#define IOMUX_SW_MUX_CTL_11        IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_11_byte
#define IOMUX_SW_MUX_CTL_11_bit    IOMUX_SW_MUX_CTL_8_11_bit.__sw_mux_ctl_11_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_12_15,       0x010CC00C,__READ_WRITE,__iomux_sw_mux_ctl_12_15_bits);
#define IOMUX_SW_MUX_CTL_12        IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_12_byte
#define IOMUX_SW_MUX_CTL_12_bit    IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_12_byte_bit
#define IOMUX_SW_MUX_CTL_13        IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_13_byte
#define IOMUX_SW_MUX_CTL_13_bit    IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_13_byte_bit
#define IOMUX_SW_MUX_CTL_14        IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_14_byte
#define IOMUX_SW_MUX_CTL_14_bit    IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_14_byte_bit
#define IOMUX_SW_MUX_CTL_15        IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_15_byte
#define IOMUX_SW_MUX_CTL_15_bit    IOMUX_SW_MUX_CTL_12_15_bit.__sw_mux_ctl_15_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_16_19,       0x010CC010,__READ_WRITE,__iomux_sw_mux_ctl_16_19_bits);
#define IOMUX_SW_MUX_CTL_16        IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_16_byte
#define IOMUX_SW_MUX_CTL_16_bit    IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_16_byte_bit
#define IOMUX_SW_MUX_CTL_17        IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_17_byte
#define IOMUX_SW_MUX_CTL_17_bit    IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_17_byte_bit
#define IOMUX_SW_MUX_CTL_18        IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_18_byte
#define IOMUX_SW_MUX_CTL_18_bit    IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_18_byte_bit
#define IOMUX_SW_MUX_CTL_19        IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_19_byte
#define IOMUX_SW_MUX_CTL_19_bit    IOMUX_SW_MUX_CTL_16_19_bit.__sw_mux_ctl_19_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_20_23,       0x010CC014,__READ_WRITE,__iomux_sw_mux_ctl_20_23_bits);
#define IOMUX_SW_MUX_CTL_20        IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_20_byte
#define IOMUX_SW_MUX_CTL_20_bit    IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_20_byte_bit
#define IOMUX_SW_MUX_CTL_21        IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_21_byte
#define IOMUX_SW_MUX_CTL_21_bit    IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_21_byte_bit
#define IOMUX_SW_MUX_CTL_22        IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_22_byte
#define IOMUX_SW_MUX_CTL_22_bit    IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_22_byte_bit
#define IOMUX_SW_MUX_CTL_23        IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_23_byte
#define IOMUX_SW_MUX_CTL_23_bit    IOMUX_SW_MUX_CTL_20_23_bit.__sw_mux_ctl_23_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_24_27,       0x010CC018,__READ_WRITE,__iomux_sw_mux_ctl_24_27_bits);
#define IOMUX_SW_MUX_CTL_24        IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_24_byte
#define IOMUX_SW_MUX_CTL_24_bit    IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_24_byte_bit
#define IOMUX_SW_MUX_CTL_25        IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_25_byte
#define IOMUX_SW_MUX_CTL_25_bit    IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_25_byte_bit
#define IOMUX_SW_MUX_CTL_26        IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_26_byte
#define IOMUX_SW_MUX_CTL_26_bit    IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_26_byte_bit
#define IOMUX_SW_MUX_CTL_27        IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_27_byte
#define IOMUX_SW_MUX_CTL_27_bit    IOMUX_SW_MUX_CTL_24_27_bit.__sw_mux_ctl_27_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_28_31,       0x010CC01C,__READ_WRITE,__iomux_sw_mux_ctl_28_31_bits);
#define IOMUX_SW_MUX_CTL_28        IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_28_byte
#define IOMUX_SW_MUX_CTL_28_bit    IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_28_byte_bit
#define IOMUX_SW_MUX_CTL_29        IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_29_byte
#define IOMUX_SW_MUX_CTL_29_bit    IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_29_byte_bit
#define IOMUX_SW_MUX_CTL_30        IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_30_byte
#define IOMUX_SW_MUX_CTL_30_bit    IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_30_byte_bit
#define IOMUX_SW_MUX_CTL_31        IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_31_byte
#define IOMUX_SW_MUX_CTL_31_bit    IOMUX_SW_MUX_CTL_28_31_bit.__sw_mux_ctl_31_byte_bit
__IO_REG32_BIT(IOMUX_SW_MUX_CTL_32_35,       0x010CC020,__READ_WRITE,__iomux_sw_mux_ctl_32_35_bits);
#define IOMUX_SW_MUX_CTL_32        IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_32_byte
#define IOMUX_SW_MUX_CTL_32_bit    IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_32_byte_bit
#define IOMUX_SW_MUX_CTL_33        IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_33_byte
#define IOMUX_SW_MUX_CTL_33_bit    IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_33_byte_bit
#define IOMUX_SW_MUX_CTL_34        IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_34_byte
#define IOMUX_SW_MUX_CTL_34_bit    IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_34_byte_bit
#define IOMUX_SW_MUX_CTL_35        IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_35_byte
#define IOMUX_SW_MUX_CTL_35_bit    IOMUX_SW_MUX_CTL_32_35_bit.__sw_mux_ctl_35_byte_bit
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_0,       0x010CC200,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_1,       0x010CC202,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_2,       0x010CC204,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_3,       0x010CC206,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_4,       0x010CC208,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_5,       0x010CC20A,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_6,       0x010CC20C,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_7,       0x010CC20E,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_8,       0x010CC210,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_9,       0x010CC212,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_10,      0x010CC214,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_11,      0x010CC216,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_12,      0x010CC218,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_13,      0x010CC21A,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_14,      0x010CC21C,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_15,      0x010CC21E,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_16,      0x010CC220,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_17,      0x010CC222,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_18,      0x010CC224,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_19,      0x010CC226,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_20,      0x010CC228,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_21,      0x010CC22A,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_22,      0x010CC22C,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_23,      0x010CC22E,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_24,      0x010CC230,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_25,      0x010CC232,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_26,      0x010CC234,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_27,      0x010CC236,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_28,      0x010CC238,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_29,      0x010CC23A,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_30,      0x010CC23C,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG16_BIT(IOMUX_SW_PAD_CTL_31,      0x010CC23E,__READ_WRITE,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_INT_OBS_0_3,        0x010CC600,__READ_WRITE,__iomux_int_obs_0_3_bits);
__IO_REG32_BIT(IOMUX_INT_OBS_4_7,        0x010CC604,__READ_WRITE,__iomux_int_obs_4_7_bits); 
__IO_REG32_BIT(IOMUX_INT_OBS_8_9,        0x010CC608,__READ_WRITE,__iomux_int_obs_8_9_bits);
__IO_REG32_BIT(IOMUX_GPR,                0x010CC60C,__READ_WRITE,__iomux_gpr_bits);
 
/***************************************************************************
 **
 **  AAPE 
 **
 ***************************************************************************/

__IO_REG32_BIT(AAPE_ABCNTL,              0x01018000,__READ_WRITE,__aape_abcntl_bits);
__IO_REG32_BIT(AAPE_ABSR,                0x01018004,__READ_WRITE,__aape_absr_bits);
__IO_REG32_BIT(AAPE_ABDADR,              0x01018008,__READ      ,__aape_abdadr_bits);

/***************************************************************************
 **
 **  AIPS
 **
 ***************************************************************************/

__IO_REG32_BIT(AIPS_MPR_1,              0x01000000,__READ_WRITE,__aips_mpr_1_bits);
__IO_REG32_BIT(AIPS_MPR_2,              0x01000004,__READ_WRITE,__aips_mpr_2_bits);
__IO_REG32_BIT(AIPS_PACR_1,             0x01000020,__READ_WRITE,__aips_pacr_opacr_1_bits);
__IO_REG32_BIT(AIPS_PACR_2,             0x01000024,__READ_WRITE,__aips_pacr_opacr_2_bits);
__IO_REG32_BIT(AIPS_PACR_3,             0x01000028,__READ_WRITE,__aips_pacr_opacr_3_bits);
__IO_REG32_BIT(AIPS_PACR_4,             0x0100002C,__READ_WRITE,__aips_pacr_opacr_4_bits);
__IO_REG32_BIT(AIPS_OPACR_1,            0x01000040,__READ_WRITE,__aips_pacr_opacr_1_bits);
__IO_REG32_BIT(AIPS_OPACR_2,            0x01000044,__READ_WRITE,__aips_pacr_opacr_2_bits);
__IO_REG32_BIT(AIPS_OPACR_3,            0x01000048,__READ_WRITE,__aips_pacr_opacr_3_bits);
__IO_REG32_BIT(AIPS_OPACR_4,            0x0100004C,__READ_WRITE,__aips_pacr_opacr_4_bits);
__IO_REG32_BIT(AIPS_OPACR_5,            0x01000050,__READ_WRITE,__aips_opacr_5_bits);

/***************************************************************************
 **
 **  ASIC
 **
 ***************************************************************************/
 
__IO_REG32_BIT(ASIC_INTCNTL,            0x01800000,__READ_WRITE,__asic_intcntl_bits); 
__IO_REG32_BIT(ASIC_NIMASK,             0x01800004,__READ_WRITE,__asic_nimask_bits); 
__IO_REG32_BIT(ASIC_INTENNUM,           0x01800008,__WRITE,     __asic_intennum_bits); 
__IO_REG32_BIT(ASIC_INTDISNUM,          0x0180000C,__WRITE,     __asic_intdisnum_bits); 
__IO_REG32_BIT(ASIC_INTENABLEH,         0x01800010,__READ_WRITE,__asic_intenable_bits); 
__IO_REG32_BIT(ASIC_INTENABLEL,         0x01800014,__READ_WRITE,__asic_intenable_bits); 
__IO_REG32_BIT(ASIC_INTTYPEH,           0x01800018,__READ_WRITE,__asic_inttype_bits); 
__IO_REG32_BIT(ASIC_INTTYPEL,           0x0180001C,__READ_WRITE,__asic_inttype_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY7,        0x01800020,__READ_WRITE,__asic_nipriority7_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY6,        0x01800024,__READ_WRITE,__asic_nipriority6_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY5,        0x01800028,__READ_WRITE,__asic_nipriority5_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY4,        0x0180002C,__READ_WRITE,__asic_nipriority4_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY3,        0x01800030,__READ_WRITE,__asic_nipriority3_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY2,        0x01800034,__READ_WRITE,__asic_nipriority2_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY1,        0x01800038,__READ_WRITE,__asic_nipriority1_bits); 
__IO_REG32_BIT(ASIC_NIPRIORITY0,        0x0180003C,__READ_WRITE,__asic_nipriority0_bits); 
__IO_REG32_BIT(ASIC_NIVECSR,            0x01800040,__READ      ,__asic_nivecsr_bits); 
__IO_REG32_BIT(ASIC_FIVECSR,            0x01800044,__READ      ,__asic_fivecsr_bits); 
__IO_REG32_BIT(ASIC_INTSRCH,            0x01800048,__READ      ,__asic_intsrch_bits); 
__IO_REG32_BIT(ASIC_INTSRCL,            0x0180004C,__READ      ,__asic_intsrcl_bits);
__IO_REG32_BIT(ASIC_INTFRCH,            0x01800050,__READ_WRITE,__asic_intfrch_bits); 
__IO_REG32_BIT(ASIC_INTFRCL,            0x01800054,__READ_WRITE,__asic_intfrcl_bits); 
__IO_REG32_BIT(ASIC_NIPNDH,             0x01800058,__READ      ,__asic_nipndh_bits); 
__IO_REG32_BIT(ASIC_NIPNDL,             0x0180005C,__READ      ,__asic_nipndl_bits);
__IO_REG32_BIT(ASIC_FIPNDH,             0x01800060,__READ      ,__asic_fipndh_bits); 
__IO_REG32_BIT(ASIC_FIPNDL,             0x01800064,__READ      ,__asic_fipndl_bits);

/***************************************************************************
 **
 **  ASM
 **
 ***************************************************************************/

__IO_REG32_BIT(ASM_KEY0,                0x01098000,__READ_WRITE,__asm_key_bits);
__IO_REG32_BIT(ASM_KEY1,                0x01098004,__READ_WRITE,__asm_key_bits);
__IO_REG32_BIT(ASM_KEY2,                0x01098008,__READ_WRITE,__asm_key_bits);
__IO_REG32_BIT(ASM_KEY3,                0x0109800C,__READ_WRITE,__asm_key_bits);
__IO_REG32_BIT(ASM_CTR0,                0x01098010,__READ_WRITE,__asm_ctr_bits);
__IO_REG32_BIT(ASM_CTR1,                0x01098014,__READ_WRITE,__asm_ctr_bits);
__IO_REG32_BIT(ASM_CTR2,                0x01098018,__READ_WRITE,__asm_ctr_bits);
__IO_REG32_BIT(ASM_CTR3,                0x0109801C,__READ_WRITE,__asm_ctr_bits);
__IO_REG32_BIT(ASM_DATA0,               0x01098020,__READ_WRITE,__asm_data_bits);
__IO_REG32_BIT(ASM_DATA1,               0x01098024,__READ_WRITE,__asm_data_bits);
__IO_REG32_BIT(ASM_DATA2,               0x01098028,__READ_WRITE,__asm_data_bits);
__IO_REG32_BIT(ASM_DATA3,               0x0109802C,__READ_WRITE,__asm_data_bits);
__IO_REG32_BIT(ASM_CONTROL0,            0x01098030,__WRITE     ,__asm_control0_bits);
__IO_REG32_BIT(ASM_CONTROL1,            0x01098034,__READ_WRITE,__asm_control1_bits);
__IO_REG32_BIT(ASM_STATUS,              0x01098038,__READ      ,__asm_status_bits);
__IO_REG32_BIT(ASM_AES0_RESULT,         0x01098040,__READ_WRITE,__asm_aes_result_bits);
__IO_REG32_BIT(ASM_AES1_RESULT,         0x01098044,__READ_WRITE,__asm_aes_result_bits);
__IO_REG32_BIT(ASM_AES2_RESULT,         0x01098048,__READ_WRITE,__asm_aes_result_bits);
__IO_REG32_BIT(ASM_AES3_RESULT,         0x0109804C,__READ_WRITE,__asm_aes_result_bits);
__IO_REG32_BIT(ASM_CTR0_RESULT,         0x01098050,__READ_WRITE,__asm_ctr_result_bits);
__IO_REG32_BIT(ASM_CTR1_RESULT,         0x01098054,__READ_WRITE,__asm_ctr_result_bits);
__IO_REG32_BIT(ASM_CTR2_RESULT,         0x01098058,__READ_WRITE,__asm_ctr_result_bits);
__IO_REG32_BIT(ASM_CTR3_RESULT,         0x0109805C,__READ_WRITE,__asm_ctr_result_bits);
__IO_REG32_BIT(ASM_MAC0,                0x01098060,__READ_WRITE,__asm_mac_bits);
__IO_REG32_BIT(ASM_MAC1,                0x01098064,__READ_WRITE,__asm_mac_bits);
__IO_REG32_BIT(ASM_MAC2,                0x01098068,__READ_WRITE,__asm_mac_bits);
__IO_REG32_BIT(ASM_MAC3,                0x0109806C,__READ_WRITE,__asm_mac_bits);

/***************************************************************************
 **
 **  CCM
 **
 ***************************************************************************/

__IO_REG32_BIT(CCM_ASCSR,               0x01094000,__READ_WRITE,__ccm_ascsr_bits);
__IO_REG32_BIT(CCM_ACSR,                0x01094004,__READ_WRITE,__ccm_acsr_bits);
__IO_REG32_BIT(CCM_ACDR1,               0x01094008,__READ_WRITE,__ccm_acdr1_bits);
__IO_REG32_BIT(CCM_ACDR2,               0x0109400C,__READ_WRITE,__ccm_acdr2_bits);
__IO_REG32_BIT(CCM_ACGCR,               0x01094010,__READ_WRITE,__ccm_acgcr_bits);
__IO_REG32_BIT(CCM_LPCGR0,              0x01094014,__READ_WRITE,__ccm_lpcgr0_bits);
__IO_REG32_BIT(CCM_LPCGR1,              0x01094018,__READ_WRITE,__ccm_lpcgr1_bits);
__IO_REG32_BIT(CCM_LPCGR2,              0x0109401C,__READ_WRITE,__ccm_lpcgr2_bits);
__IO_REG32_BIT(CCM_LPCGR3,              0x01094020,__READ_WRITE,__ccm_lpcgr3_bits);
__IO_REG32_BIT(CCM_LPCGR4,              0x01094024,__READ_WRITE,__ccm_lpcgr4_bits);
__IO_REG32_BIT(CCM_LPCGR5,              0x01094028,__READ_WRITE,__ccm_lpcgr5_bits);
__IO_REG32_BIT(CCM_AMORA,               0x01094030,__READ_WRITE,__ccm_amora_bits);
__IO_REG32_BIT(CCM_AMORB,               0x01094034,__READ_WRITE,__ccm_amorb_bits);
__IO_REG32_BIT(CCM_APOR,                0x01094038,__READ_WRITE,__ccm_apor_bits);
__IO_REG32_BIT(CCM_ACR,                 0x0109403C,__READ_WRITE,__ccm_acr_bits);
__IO_REG32_BIT(CCM_AMCR,                0x01094040,__READ_WRITE,__ccm_amcr_bits);
__IO_REG32_BIT(CCM_AIR1,                0x01094044,__READ_WRITE,__ccm_air1_bits);
__IO_REG32_BIT(CCM_AIR2,                0x01094048,__READ_WRITE,__ccm_air2_bits);
__IO_REG32_BIT(CCM_AIR3,                0x0109404C,__READ_WRITE,__ccm_air3_bits);
__IO_REG32_BIT(CCM_AIR4,                0x01094050,__READ_WRITE,__ccm_air4_bits);
__IO_REG32_BIT(CCM_AIR5,                0x01094054,__READ_WRITE,__ccm_air5_bits);
__IO_REG32_BIT(CCM_AIR6,                0x01094058,__READ_WRITE,__ccm_air6_bits);
__IO_REG32_BIT(CCM_AIR7,                0x0109405C,__READ_WRITE,__ccm_air7_bits);
__IO_REG32_BIT(CCM_AIR8,                0x01094060,__READ_WRITE,__ccm_air8_bits);
__IO_REG32_BIT(CCM_AIR9,                0x01094064,__READ_WRITE,__ccm_air9_bits);
__IO_REG32_BIT(CCM_AIR10,               0x01094068,__READ_WRITE,__ccm_air10_bits);
__IO_REG32_BIT(CCM_FIUR1,               0x0109406C,__READ      ,__ccm_fiur1_bits);
__IO_REG32_BIT(CCM_FIUR2,               0x01094070,__READ      ,__ccm_fiur2_bits);
__IO_REG32_BIT(CCM_DCR,                 0x01094074,__READ_WRITE,__ccm_dcr_bits);
__IO_REG32_BIT(CCM_AGPR,                0x01094078,__READ_WRITE,__ccm_agpr_bits);

/***************************************************************************
 **
 **  CSPI
 **
 ***************************************************************************/

__IO_REG32(CSPI_RXDATA,             0x010B0000,__READ      );
__IO_REG32(CSPI_TXDATA,             0x010B0004,__WRITE     );
__IO_REG32_BIT(CSPI_CONREG,             0x010B0008,__READ_WRITE,__cspi_conreg_bits);
__IO_REG32_BIT(CSPI_INTREG,             0x010B000C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI_DMAREG,             0x010B0010,__READ_WRITE,__cspi_dmareg_bits);
__IO_REG32_BIT(CSPI_STATREG,            0x010B0014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI_PERIODREG,          0x010B0018,__READ_WRITE,__cspi_periodreg_bits);
__IO_REG32_BIT(CSPI_TESTREG,            0x010B001C,__READ_WRITE,__cspi_testreg_bits);

/***************************************************************************
 **
 **  DPLL
 **
 ***************************************************************************/

__IO_REG32_BIT(DPLL_DP_CTL,             0x0109C000,__READ_WRITE,__dpll_dp_ctl_bits);
__IO_REG32_BIT(DPLL_DP_CONFIG,          0x0109C004,__READ_WRITE,__dpll_dp_config_bits);
__IO_REG32_BIT(DPLL_DP_OP,              0x0109C008,__READ_WRITE,__dpll_dp_op_bits);
__IO_REG32_BIT(DPLL_DP_MFD,             0x0109C00C,__READ_WRITE,__dpll_dp_mfd_bits);
__IO_REG32_BIT(DPLL_DP_MFN,             0x0109C010,__READ_WRITE,__dpll_dp_mfn_bits);
__IO_REG32_BIT(DPLL_DP_MFNMINUS,        0x0109C014,__READ_WRITE,__dpll_dp_mfnminus_bits);
__IO_REG32_BIT(DPLL_DP_MFNPLUS,         0x0109C018,__READ_WRITE,__dpll_dp_mfnplus_bits);
__IO_REG32_BIT(DPLL_DP_HFS_OP,          0x0109C01C,__READ_WRITE,__dpll_dp_hfs_op_bits);
__IO_REG32_BIT(DPLL_DP_HFS_MFD,         0x0109C020,__READ_WRITE,__dpll_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLL_DP_HFS_MFN,         0x0109C024,__READ_WRITE,__dpll_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLL_DP_MFN_TOGC,        0x0109C028,__READ_WRITE,__dpll_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLL_DP_DESTAT,          0x0109C02C,__READ      ,__dpll_dp_destat_bits);

/***************************************************************************
 **
 **  DSM
 **
 ***************************************************************************/
 
__IO_REG32_BIT(DSM_COUNT32,             0x010A8000,__READ_WRITE,__dsm_count32_bits);
__IO_REG32_BIT(DSM_REFCOUNT,            0x010A8004,__READ_WRITE,__dsm_refcount_bits); 
__IO_REG32_BIT(DSM_MEASTIME,            0x010A8008,__READ_WRITE,__dsm_meastime_bits); 
__IO_REG32_BIT(DSM_SLEEPTIME,           0x010A800C,__READ_WRITE,__dsm_sleeptime_bits); 
__IO_REG32_BIT(DSM_RESTART_TIME,        0x010A8010,__READ_WRITE,__dsm_restart_time_bits); 
__IO_REG32_BIT(DSM_WAKETIME,            0x010A8014,__READ_WRITE,__dsm_waketime_bits); 
__IO_REG32_BIT(DSM_WARMTIME,            0x010A8018,__READ_WRITE,__dsm_warmtime_bits); 
__IO_REG32_BIT(DSM_LOCKTIME,            0x010A801C,__READ_WRITE,__dsm_locktime_bits); 
__IO_REG32_BIT(DSM_CONTROL0,            0x010A8020,__READ_WRITE,__dsm_control0_bits); 
__IO_REG32_BIT(DSM_CONTROL1,            0x010A8024,__READ_WRITE,__dsm_control1_bits); 
__IO_REG32_BIT(DSM_CTREN,               0x010A8028,__READ_WRITE,__dsm_ctren_bits); 
__IO_REG32_BIT(DSM_STATUS,              0x010A802C,__READ_WRITE,__dsm_status_bits); 
__IO_REG32_BIT(DSM_STATE,               0x010A8030,__READ      ,__dsm_state_bits); 
__IO_REG32_BIT(DSM_INT_STATUS,          0x010A8034,__READ_WRITE,__dsm_int_status_bits); 
__IO_REG32_BIT(DSM_MASK,                0x010A8038,__READ_WRITE,__dsm_mask_bits); 
__IO_REG32_BIT(DSM_COUNT32_CAP,         0x010A803C,__READ      ,__dsm_count32_cap_bits); 
__IO_REG32_BIT(DSM_WARMPER,             0x010A8040,__READ_WRITE,__dsm_warmper_bits); 
__IO_REG32_BIT(DSM_LOCKPER,             0x010A8044,__READ_WRITE,__dsm_lockper_bits); 
__IO_REG32_BIT(DSM_POSCOUNT,            0x010A8048,__READ      ,__dsm_poscount_bits); 
__IO_REG32_BIT(DSM_MGPER,               0x010A804C,__READ_WRITE,__dsm_mgper_bits); 
__IO_REG32_BIT(DSM_CRM_CONTROL,         0x010A8050,__READ_WRITE,__dsm_crm_control_bits); 

/***************************************************************************
 **
 **  EPIT
 **
 ***************************************************************************/

__IO_REG32_BIT(EPITCR,                  0x010D8000,__READ_WRITE,__epitcr_bits); 
__IO_REG32_BIT(EPITSR,                  0x010D8004,__READ_WRITE,__epitsr_bits); 
__IO_REG32_BIT(EPITLR,                  0x010D8008,__READ_WRITE,__epitlr_bits); 
__IO_REG32_BIT(EPITCMPR,                0x010D800C,__READ_WRITE,__epitcmpr_bits); 
__IO_REG32_BIT(EPITCNR,                 0x010D8010,__READ      ,__epitcnr_bits); 

/***************************************************************************
 **
 **  GPADCIF
 **
 ***************************************************************************/

__IO_REG16_BIT(GPADCIF_gpadc_ctrl,           0x01084000,__READ_WRITE,__gpadcif_gpadc_ctrl_bits); 
__IO_REG16_BIT(GPADCIF_INTERRUPT_STATUS_REG, 0x01084002,__READ_WRITE,__gpadcif_interrupt_status_reg_bits); 
__IO_REG16_BIT(GPADCIF_ADC1,                 0x01084004,__READ      ,__gpadcif_adc1_bits); 
__IO_REG16_BIT(GPADCIF_ADC2,                 0x01084006,__READ      ,__gpadcif_adc2_bits); 
__IO_REG16_BIT(GPADCIF_ADC3,                 0x01084008,__READ      ,__gpadcif_adc3_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_0,          0x0108400A,__READ      ,__gpadcif_xtal_temp_0_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_1,          0x0108400C,__READ      ,__gpadcif_xtal_temp_1_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_2,          0x0108400E,__READ      ,__gpadcif_xtal_temp_2_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_3,          0x01084010,__READ      ,__gpadcif_xtal_temp_3_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_4,          0x01084012,__READ      ,__gpadcif_xtal_temp_4_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_5,          0x01084014,__READ      ,__gpadcif_xtal_temp_5_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_6,          0x01084016,__READ      ,__gpadcif_xtal_temp_6_bits); 
__IO_REG16_BIT(GPADCIF_xtal_temp_7,          0x01084018,__READ      ,__gpadcif_xtal_temp_7_bits); 

/***************************************************************************
 **
 **  GPIO
 **
 ***************************************************************************/

__IO_REG32_BIT(GPIO_DR,                 0x010DC000,__READ_WRITE,__gpio_dr_bits);  
__IO_REG32_BIT(GPIO_GDIR,               0x010DC004,__READ_WRITE,__gpio_gdir_bits);  
__IO_REG32_BIT(GPIO_PSR,                0x010DC008,__READ      ,__gpio_psr_bits);  
__IO_REG32_BIT(GPIO_ICR1,               0x010DC00C,__READ_WRITE,__gpio_icr1_bits);  
__IO_REG32_BIT(GPIO_ICR2,               0x010DC010,__READ_WRITE,__gpio_icr2_bits);  
__IO_REG32_BIT(GPIO_IMR,                0x010DC014,__READ_WRITE,__gpio_imr_bits);  
__IO_REG32_BIT(GPIO_ISR,                0x010DC018,__READ_WRITE,__gpio_isr_bits);  
__IO_REG32_BIT(GPIO_EDGE_SEL,           0x010DC01C,__READ_WRITE,__gpio_edge_sel_bits); 

/***************************************************************************
 **
 **  GPT
 **
 ***************************************************************************/
 
__IO_REG32_BIT(GPTCR,                   0x010D0000,__READ_WRITE,__gptcr_bits);    
__IO_REG32_BIT(GPTPR,                   0x010D0004,__READ_WRITE,__gptpr_bits);   
__IO_REG32_BIT(GPTSR,                   0x010D0008,__READ_WRITE,__gptsr_bits);   
__IO_REG32_BIT(GPTIR,                   0x010D000C,__READ_WRITE,__gptir_bits);   
__IO_REG32_BIT(GPTOCR1,                 0x010D0010,__READ_WRITE,__gptocr_bits);   
__IO_REG32_BIT(GPTOCR2,                 0x010D0014,__READ_WRITE,__gptocr_bits);   
__IO_REG32_BIT(GPTOCR3,                 0x010D0018,__READ_WRITE,__gptocr_bits);   
__IO_REG32_BIT(GPTICR1,                 0x010D001C,__READ      ,__gpticr_bits);   
__IO_REG32_BIT(GPTICR2,                 0x010D0020,__READ      ,__gpticr_bits);    
__IO_REG32_BIT(GPTCNT,                  0x010D0024,__READ      ,__gptcnt_bits);   

/***************************************************************************
 **
 **  HDMA
 **
 ***************************************************************************/

__IO_REG32(HDMA_D0_SBAR,            0x01090000,__READ_WRITE);
__IO_REG32(HDMA_D1_SBAR,            0x01090100,__READ_WRITE);
__IO_REG32(HDMA_D2_SBAR,            0x01090200,__READ_WRITE);
__IO_REG32(HDMA_D3_SBAR,            0x01090300,__READ_WRITE);
__IO_REG32(HDMA_D4_SBAR,            0x01090400,__READ_WRITE);
__IO_REG32(HDMA_D5_SBAR,            0x01090500,__READ_WRITE);
__IO_REG32(HDMA_D6_SBAR,            0x01090600,__READ_WRITE);
__IO_REG32(HDMA_D7_SBAR,            0x01090700,__READ_WRITE);
__IO_REG32(HDMA_D0_DBAR,            0x01090004,__READ_WRITE);
__IO_REG32(HDMA_D1_DBAR,            0x01090104,__READ_WRITE);
__IO_REG32(HDMA_D2_DBAR,            0x01090204,__READ_WRITE);
__IO_REG32(HDMA_D3_DBAR,            0x01090304,__READ_WRITE);
__IO_REG32(HDMA_D4_DBAR,            0x01090404,__READ_WRITE);
__IO_REG32(HDMA_D5_DBAR,            0x01090504,__READ_WRITE);
__IO_REG32(HDMA_D6_DBAR,            0x01090604,__READ_WRITE);
__IO_REG32(HDMA_D7_DBAR,            0x01090704,__READ_WRITE);
__IO_REG32_BIT(HDMA_D0_SMAR,            0x01090008,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D1_SMAR,            0x01090108,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D2_SMAR,            0x01090208,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D3_SMAR,            0x01090308,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D4_SMAR,            0x01090408,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D5_SMAR,            0x01090508,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D6_SMAR,            0x01090608,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D7_SMAR,            0x01090708,__READ_WRITE,__hdma_smar_bits);
__IO_REG32_BIT(HDMA_D0_DMAR,            0x0109000C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D1_DMAR,            0x0109010C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D2_DMAR,            0x0109020C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D3_DMAR,            0x0109030C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D4_DMAR,            0x0109040C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D5_DMAR,            0x0109050C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D6_DMAR,            0x0109060C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D7_DMAR,            0x0109070C,__READ_WRITE,__hdma_dmar_bits);
__IO_REG32_BIT(HDMA_D0_BFCN,            0x01090010,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D1_BFCN,            0x01090110,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D2_BFCN,            0x01090210,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D3_BFCN,            0x01090310,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D4_BFCN,            0x01090410,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D5_BFCN,            0x01090510,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D6_BFCN,            0x01090610,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D7_BFCN,            0x01090710,__READ_WRITE,__hdma_bfcn_bits);
__IO_REG32_BIT(HDMA_D0_DCLL,            0x01090014,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D1_DCLL,            0x01090114,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D2_DCLL,            0x01090214,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D3_DCLL,            0x01090314,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D4_DCLL,            0x01090414,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D5_DCLL,            0x01090514,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D6_DCLL,            0x01090614,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32_BIT(HDMA_D7_DCLL,            0x01090714,__READ_WRITE,__hdma_dcll_bits);
__IO_REG32(HDMA_D0_SACN,            0x01090018,__READ      );
__IO_REG32(HDMA_D1_SACN,            0x01090118,__READ      );
__IO_REG32(HDMA_D2_SACN,            0x01090218,__READ      );
__IO_REG32(HDMA_D3_SACN,            0x01090318,__READ      );
__IO_REG32(HDMA_D4_SACN,            0x01090418,__READ      );
__IO_REG32(HDMA_D5_SACN,            0x01090518,__READ      );
__IO_REG32(HDMA_D6_SACN,            0x01090618,__READ      );
__IO_REG32(HDMA_D7_SACN,            0x01090718,__READ      );  
__IO_REG32(HDMA_D0_DACN,            0x0109001C,__READ      );
__IO_REG32(HDMA_D1_DACN,            0x0109011C,__READ      );
__IO_REG32(HDMA_D2_DACN,            0x0109021C,__READ      );
__IO_REG32(HDMA_D3_DACN,            0x0109031C,__READ      );
__IO_REG32(HDMA_D4_DACN,            0x0109041C,__READ      );
__IO_REG32(HDMA_D5_DACN,            0x0109051C,__READ      );
__IO_REG32(HDMA_D6_DACN,            0x0109061C,__READ      );
__IO_REG32(HDMA_D7_DACN,            0x0109071C,__READ      );
__IO_REG32_BIT(HDMA_TCSR,               0x01093000,__READ_WRITE,__hdma_tcsr_bits);
__IO_REG32_BIT(HDMA_BCSR,               0x01093004,__READ_WRITE,__hdma_bcsr_bits);
__IO_REG32_BIT(HDMA_IBCSR,              0x01093008,__READ_WRITE,__hdma_ibcsr_bits);
__IO_REG32_BIT(HDMA_BTCR,               0x0109300C,__READ_WRITE,__hdma_btcr_bits); 
__IO_REG32_BIT(HDMA_BTSR,               0x01093010,__READ_WRITE,__hdma_btsr_bits); 
__IO_REG32_BIT(HDMA_TESR,               0x01093014,__READ_WRITE,__hdma_tesr_bits); 
__IO_REG32_BIT(HDMA_CRTPR,              0x01093018,__READ_WRITE,__hdma_crtpr_bits); 
__IO_REG32_BIT(HDMA_CRTSR,              0x0109301C,__READ_WRITE,__hdma_crtsr_bits); 
__IO_REG32_BIT(HDMA_BNRSR,              0x01093020,__READ_WRITE,__hdma_bnrsr_bits); 
__IO_REG32_BIT(HDMA_BWCSR,              0x01093024,__READ_WRITE,__hdma_bwcsr_bits); 
__IO_REG32_BIT(HDMA_DGCR,               0x01093028,__READ_WRITE,__hdma_dgcr_bits); 
__IO_REG32_BIT(HDMA_D0_DCR,             0x01093030,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D1_DCR,             0x01093034,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D2_DCR,             0x01093038,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D3_DCR,             0x0109303C,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D4_DCR,             0x01093040,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D5_DCR,             0x01093044,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D6_DCR,             0x01093048,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D7_DCR,             0x0109304C,__READ_WRITE,__hdma_dxdcr_bits);
__IO_REG32_BIT(HDMA_D0_SWAP,            0x010930B0,__READ_WRITE,__hdma_d0swap_bits); 
__IO_REG32_BIT(HDMA_D0_CRTCN,           0x01093100,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D1_CRTCN,           0x01093104,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D2_CRTCN,           0x01093108,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D3_CRTCN,           0x0109310C,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D4_CRTCN,           0x01093110,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D5_CRTCN,           0x01093114,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D6_CRTCN,           0x01093118,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG32_BIT(HDMA_D7_CRTCN,           0x0109311C,__READ_WRITE,__hdma_dxcrtcn_bits);
__IO_REG16_BIT(HDMA_D0_BTCN,            0x01093190,__READ      ,__hdma_dxbtcn_bits);
__IO_REG16_BIT(HDMA_D1_BTCN,            0x01093194,__READ      ,__hdma_dxbtcn_bits);
__IO_REG16_BIT(HDMA_D2_BTCN,            0x01093198,__READ      ,__hdma_dxbtcn_bits);
__IO_REG16_BIT(HDMA_D3_BTCN,            0x0109319C,__READ      ,__hdma_dxbtcn_bits);
__IO_REG32_BIT(HDMA_activech_reg,       0x01093400,__READ      ,__hdma_activech_reg_bits);
__IO_REG32_BIT(HDMA_dum_rem_xfr_reg,    0x01093404,__READ      ,__hdma_dum_rem_xfr_reg_bits);
__IO_REG32_BIT(HDMA_dum_ch_abort,       0x01093408,__READ      ,__hdma_dum_ch_abort_bits);
__IO_REG32_BIT(HDMA_dum_crtsr_reg,      0x0109340C,__READ      ,__hdma_dum_crtsr_reg_bits);
__IO_REG32_BIT(HDMA_D0_rem_xfr_all_reg, 0x01093410,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D1_rem_xfr_all_reg, 0x01093414,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D2_rem_xfr_all_reg, 0x01093418,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D3_rem_xfr_all_reg, 0x0109341C,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D4_rem_xfr_all_reg, 0x01093420,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D5_rem_xfr_all_reg, 0x01093424,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D6_rem_xfr_all_reg, 0x01093428,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_D7_rem_xfr_all_reg, 0x0109342C,__READ      ,__hdma_rem_xfr_all_reg_bits);
__IO_REG32_BIT(HDMA_dma_act_ch0,        0x01093490,__READ      ,__hdma_dma_act_ch0_bits);
__IO_REG32(HDMA_0x3494_reserved,    0x01093494,__READ      );
__IO_REG32_BIT(HDMA_dma_act_ch0_next,   0x01093498,__READ      ,__hdma_dma_act_ch0_next_bits);
__IO_REG32(HDMA_0x349C_reserved,    0x0109349C,__READ      );
__IO_REG32(HDMA_D0_current_dcll_ptr_reg, 0x01093500,__READ      );
__IO_REG32(HDMA_D1_current_dcll_ptr_reg, 0x01093504,__READ      );
__IO_REG32(HDMA_D2_current_dcll_ptr_reg, 0x01093508,__READ      );
__IO_REG32(HDMA_D3_current_dcll_ptr_reg, 0x0109350C,__READ      );
__IO_REG32(HDMA_D4_current_dcll_ptr_reg, 0x01093510,__READ      );
__IO_REG32(HDMA_D5_current_dcll_ptr_reg, 0x01093514,__READ      );
__IO_REG32(HDMA_D6_current_dcll_ptr_reg, 0x01093518,__READ      );
__IO_REG32(HDMA_D7_current_dcll_ptr_reg, 0x0109351C,__READ      );
  
/***************************************************************************
 **
 **  I2C
 **
 ***************************************************************************/

__IO_REG16_BIT(I2C_IADR,                 0x010C8000,__READ_WRITE,__i2c_iadr_bits);
__IO_REG16_BIT(I2C_IFDR,                 0x010C8004,__READ_WRITE,__i2c_ifdr_bits);
__IO_REG16_BIT(I2C_I2CR,                 0x010C8008,__READ_WRITE,__i2c_i2cr_bits);
__IO_REG16_BIT(I2C_I2SR,                 0x010C800C,__READ_WRITE,__i2c_i2sr_bits); 
__IO_REG16_BIT(I2C_I2DR,                 0x010C8010,__READ_WRITE,__i2c_i2dr_bits); 

/***************************************************************************
 **
 **  IIM
 **
 ***************************************************************************/

__IO_REG8_BIT(IIM_STAT,                 0x010F8000,__READ_WRITE,__iim_stat_bits);
__IO_REG8_BIT(IIM_STATM,                0x010F8004,__READ_WRITE,__iim_statm_bits);
__IO_REG8_BIT(IIM_ERR,                  0x010F8008,__READ_WRITE,__iim_err_bits);
__IO_REG8_BIT(IIM_EMASK,                0x010F800C,__READ_WRITE,__iim_emask_bits);
__IO_REG8_BIT(IIM_FCTL,                 0x010F8010,__READ_WRITE,__iim_fctl_bits);
__IO_REG8_BIT(IIM_UA,                   0x010F8014,__READ_WRITE,__iim_ua_bits);
__IO_REG8_BIT(IIM_LA,                   0x010F8018,__READ_WRITE,__iim_la_bits);
__IO_REG8_BIT(IIM_SDAT,                 0x010F801C,__READ      ,__iim_sdat_bits);
__IO_REG8_BIT(IIM_PREV,                 0x010F8020,__READ      ,__iim_prev_bits);
__IO_REG8_BIT(IIM_SREV,                 0x010F8024,__READ      ,__iim_srev_bits);
__IO_REG8_BIT(IIM_PRG_P,                0x010F8028,__READ      ,__iim_prg_p_bits);
__IO_REG8_BIT(IIM_SCS0,                 0x010F802C,__READ_WRITE,__iim_scs0_bits);
__IO_REG8_BIT(IIM_SCS1,                 0x010F8030,__READ_WRITE,__iim_scs1_bits);
__IO_REG8_BIT(IIM_SCS2,                 0x010F8034,__READ_WRITE,__iim_scs2_bits);
__IO_REG8_BIT(IIM_SCS3,                 0x010F8038,__READ_WRITE,__iim_scs3_bits);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/

__IO_REG16_BIT(IIM_KPCR,                 0x010AC000,__READ_WRITE,__kpp_kpcr_bits);
__IO_REG16_BIT(IIM_KPSR,                 0x010AC002,__READ_WRITE,__kpp_kpsr_bits);
__IO_REG16_BIT(IIM_KDDR,                 0x010AC004,__READ_WRITE,__kpp_kddr_bits);
__IO_REG16_BIT(IIM_KPDR,                 0x010AC006,__READ_WRITE,__kpp_kpdr_bits);

/***************************************************************************
 **
 **  MAX
 **
 ***************************************************************************/
 
__IO_REG32_BIT(MAX_MPR0,                 0x01004000,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR1,                 0x01004100,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR2,                 0x01004200,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR3,                 0x01004300,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR4,                 0x01004400,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR5,                 0x01004500,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR6,                 0x01004600,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_MPR7,                 0x01004700,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR0,                0x01004004,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR1,                0x01004104,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR2,                0x01004204,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR3,                0x01004304,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR4,                0x01004404,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR5,                0x01004504,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR6,                0x01004604,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_AMPR7,                0x01004704,__READ_WRITE,__max_mprx_amprx_bits);
__IO_REG32_BIT(MAX_SGPCR0,               0x01004010,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR1,               0x01004110,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR2,               0x01004210,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR3,               0x01004310,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR4,               0x01004410,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR5,               0x01004510,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR6,               0x01004610,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_SGPCR7,               0x01004710,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR0,              0x01004014,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR1,              0x01004114,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR2,              0x01004214,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR3,              0x01004314,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR4,              0x01004414,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR5,              0x01004514,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR6,              0x01004614,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_ASGPCR7,              0x01004714,__READ_WRITE,__max_sgpcr_asgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR0,               0x01004800,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR1,               0x01004900,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR2,               0x01004A00,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR3,               0x01004B00,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR4,               0x01004C00,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR5,               0x01004D00,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR6,               0x01004E00,__READ_WRITE,__max_mgpcr_bits);
__IO_REG32_BIT(MAX_MGPCR7,               0x01004F00,__READ_WRITE,__max_mgpcr_bits);

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/

__IO_REG32_BIT(PWM1_PWMCR,               0x010E8000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWM1_PWMSR,               0x010E8004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWM1_PWMIR,               0x010E8008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWM1_PWMSAR,              0x010E800C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWM1_PWMPR,               0x010E8010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWM1_PWMCNR,              0x010E8014,__READ      ,__pwmcnr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/

__IO_REG32_BIT(PWM2_PWMCR,               0x010EC000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWM2_PWMSR,               0x010EC004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWM2_PWMIR,               0x010EC008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWM2_PWMSAR,              0x010EC00C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWM2_PWMPR,               0x010EC010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWM2_PWMCNR,              0x010EC014,__READ      ,__pwmcnr_bits);
 
/***************************************************************************
 **
 **  TMR
 **
 ***************************************************************************/

__IO_REG16(TMR0_CMP1,                0x010D4000,__READ_WRITE);
__IO_REG16(TMR0_CMP2,                0x010D4002,__READ_WRITE);
__IO_REG16(TMR0_CAP,                 0x010D4004,__READ_WRITE);
__IO_REG16(TMR0_LOAD,                0x010D4006,__READ_WRITE);
__IO_REG16(TMR0_HOLD,                0x010D4008,__READ_WRITE);
__IO_REG16(TMR0_CNTR,                0x010D400A,__READ_WRITE);  
__IO_REG16_BIT(TMR0_CTRL,                0x010D400C,__READ_WRITE,__tmrx_ctrl_bits);    
__IO_REG16_BIT(TMR0_SCR,                 0x010D400E,__READ_WRITE,__tmrx_scr_bits);    
__IO_REG16(TMR0_CMPLD1,                0x010D4010,__READ_WRITE);
__IO_REG16(TMR0_CMPLD2,                0x010D4012,__READ_WRITE);  
__IO_REG16_BIT(TMR0_COMSCR,              0x010D4014,__READ_WRITE,__tmrx_comscr_bits);    
 __IO_REG16(TMR1_CMP1,                0x010D4020,__READ_WRITE);
__IO_REG16(TMR1_CMP2,                0x010D4022,__READ_WRITE);
__IO_REG16(TMR1_CAP,                 0x010D4024,__READ_WRITE);
__IO_REG16(TMR1_LOAD,                0x010D4026,__READ_WRITE);
__IO_REG16(TMR1_HOLD,                0x010D4028,__READ_WRITE);
__IO_REG16(TMR1_CNTR,                0x010D402A,__READ_WRITE);  
__IO_REG16_BIT(TMR1_CTRL,                0x010D402C,__READ_WRITE,__tmrx_ctrl_bits);    
__IO_REG16_BIT(TMR1_SCR,                 0x010D402E,__READ_WRITE,__tmrx_scr_bits);    
__IO_REG16(TMR1_CMPLD1,                0x010D4030,__READ_WRITE);
__IO_REG16(TMR1_CMPLD2,                0x010D4032,__READ_WRITE);  
__IO_REG16_BIT(TMR1_COMSCR,              0x010D4034,__READ_WRITE,__tmrx_comscr_bits);    
__IO_REG16(TMR2_CMP1,                0x010D4040,__READ_WRITE);
__IO_REG16(TMR2_CMP2,                0x010D4042,__READ_WRITE);
__IO_REG16(TMR2_CAP,                 0x010D4044,__READ_WRITE);
__IO_REG16(TMR2_LOAD,                0x010D4046,__READ_WRITE);
__IO_REG16(TMR2_HOLD,                0x010D4048,__READ_WRITE);
__IO_REG16(TMR2_CNTR,                0x010D404A,__READ_WRITE);  
__IO_REG16_BIT(TMR2_CTRL,                0x010D404C,__READ_WRITE,__tmrx_ctrl_bits);    
__IO_REG16_BIT(TMR2_SCR,                 0x010D404E,__READ_WRITE,__tmrx_scr_bits);    
__IO_REG16(TMR2_CMPLD1,                0x010D4050,__READ_WRITE);
__IO_REG16(TMR2_CMPLD2,                0x010D4052,__READ_WRITE);  
__IO_REG16_BIT(TMR2_COMSCR,              0x010D4054,__READ_WRITE,__tmrx_comscr_bits);    
__IO_REG16(TMR3_CMP1,                0x010D4060,__READ_WRITE);
__IO_REG16(TMR3_CMP2,                0x010D4062,__READ_WRITE);
__IO_REG16(TMR3_CAP,                 0x010D4064,__READ_WRITE);
__IO_REG16(TMR3_LOAD,                0x010D4066,__READ_WRITE);
__IO_REG16(TMR3_HOLD,                0x010D4068,__READ_WRITE);
__IO_REG16(TMR3_CNTR,                0x010D406A,__READ_WRITE);  
__IO_REG16_BIT(TMR3_CTRL,                0x010D406C,__READ_WRITE,__tmrx_ctrl_bits);    
__IO_REG16_BIT(TMR3_SCR,                 0x010D406E,__READ_WRITE,__tmrx_scr_bits);    
__IO_REG16(TMR3_CMPLD1,                0x010D4070,__READ_WRITE);
__IO_REG16(TMR3_CMPLD2,                0x010D4072,__READ_WRITE);  
__IO_REG16_BIT(TMR3_COMSCR,              0x010D4074,__READ_WRITE,__tmrx_comscr_bits);    
  
/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/

__IO_REG32_BIT(RTC_HOURMIN,             0x010A4000,__READ_WRITE,__rtc_hourmin_bits);
__IO_REG32_BIT(RTC_SECONDS,             0x010A4004,__READ_WRITE,__rtc_seconds_bits);
__IO_REG32_BIT(RTC_ALRM_HM,             0x010A4008,__READ_WRITE,__rtc_hourmin_bits);
__IO_REG32_BIT(RTC_ALRM_SEC,            0x010A400C,__READ_WRITE,__rtc_seconds_bits);
__IO_REG32_BIT(RTC_RTCCTL,              0x010A4010,__READ_WRITE,__rtc_rtcctl_bits);
__IO_REG32_BIT(RTC_RTCISR,              0x010A4014,__READ_WRITE,__rtc_rtcisr_bits);
__IO_REG32_BIT(RTC_RTCIENR,             0x010A4018,__READ_WRITE,__rtc_rtcisr_bits);
__IO_REG32_BIT(RTC_STPWCH,              0x010A401C,__READ_WRITE,__rtc_stpwch_bits);
__IO_REG32_BIT(RTC_DAYR,                0x010A4020,__READ_WRITE,__rtc_dayr_bits); 
__IO_REG32_BIT(RTC_DAYALARM,            0x010A4024,__READ_WRITE,__rtc_dayalarm_bits); 

/***************************************************************************
 **
 **  RNGB
 **
 ***************************************************************************/
 
__IO_REG32_BIT(RNGB_VER_ID,             0x010E0000,__READ      ,__rngb_ver_id_bits);  
__IO_REG32_BIT(RNGB_COMMAND,            0x010E0004,__READ_WRITE,__rngb_command_bits); 
__IO_REG32_BIT(RNGB_CONTROL,            0x010E0008,__READ_WRITE,__rngb_control_bits); 
__IO_REG32_BIT(RNGB_STATUS,             0x010E000C,__READ      ,__rngb_status_bits); 
__IO_REG32_BIT(RNGB_ERROR_STATUS,       0x010E0010,__READ      ,__rngb_error_status_bits); 
__IO_REG32(RNGB_FIFO,               0x010E0014,__READ      ); 
__IO_REG32(RNGB_ENTROPY,            0x010E0018,__WRITE     ); 
__IO_REG32_BIT(RNGB_VERIF_CTL,          0x010E0020,__READ_WRITE,__rngb_verif_ctl_bits); 
__IO_REG32(RNGB_XKEY,               0x010E0024,__READ      ); 
__IO_REG32_BIT(RNGB_OSC_CNT_CTL,        0x010E0028,__READ_WRITE,__rngb_osc_cnt_ctl_bits); 
__IO_REG32_BIT(RNGB_OSC_CNT,            0x010E002C,__READ      ,__rngb_osc_cnt_bits); 
__IO_REG32_BIT(RNGB_OSC_CNT_STAT,       0x010E0030,__READ      ,__rngb_osc_cnt_stat_bits); 

/***************************************************************************
 **
 **  ROMCP
 **
 ***************************************************************************/

__IO_REG32(ROMCP_ROMPATCHD7,        0x010E40D4,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD6,        0x010E40D8,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD5,        0x010E40DC,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD4,        0x010E40E0,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD3,        0x010E40E4,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD2,        0x010E40E8,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD1,        0x010E40EC,__READ_WRITE); 
__IO_REG32(ROMCP_ROMPATCHD0,        0x010E40F0,__READ_WRITE); 
__IO_REG32_BIT(ROMCP_ROMPATCHCNTL,      0x010E40F4,__READ_WRITE,__rompatchcntl_bits);  
__IO_REG32(ROMCP_ROMPATCHENH,       0x010E40F8,__READ_WRITE); 
__IO_REG32_BIT(ROMCP_ROMPATCHENL,       0x010E40FC,__READ_WRITE,__rompatchenl_bits);   
__IO_REG32_BIT(ROMCP_ROMPATCHA0,    0x010E4100,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA1,    0x010E4104,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA2,    0x010E4108,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA3,    0x010E410C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA4,    0x010E4110,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA5,    0x010E4114,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA6,    0x010E4118,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA7,    0x010E411C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA8,    0x010E4120,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA9,    0x010E4124,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA10,   0x010E4128,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA11,   0x010E412C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA12,   0x010E4130,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA13,   0x010E4134,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA14,   0x010E4138,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA15,   0x010E413C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHSR,    0x010E4208,__READ_WRITE,__rompatchsr_bits);    

/***************************************************************************
 **
 **  Ruby
 **
 ***************************************************************************/

__IO_REG32(RUBY_HWVERSION,      0x01080000,__READ);    
__IO_REG32_BIT(RUBY_SWVERSION,           0x01080004,__READ_WRITE,__ruby_swversion_bits);    
__IO_REG32_BIT(RUBY_CONTROL,             0x01080008,__READ_WRITE,__ruby_control_bits);    
__IO_REG32_BIT(RUBY_IRQEN,               0x0108000C,__READ_WRITE,__ruby_irqen_bits);    
__IO_REG32_BIT(RUBY_STATUS,              0x01080010,__READ_WRITE,__ruby_status_bits);    
__IO_REG32_BIT(RUBY_RXSTATUS,            0x01080014,__READ_WRITE,__ruby_rxstatus_bits);    
__IO_REG32_BIT(RUBY_TXSTATUS,            0x01080018,__READ_WRITE,__ruby_txstatus_bits);    
__IO_REG32_BIT(RUBY_PARAM,               0x01080020,__READ      ,__ruby_param_bits);    
__IO_REG32_BIT(RUBY_MACRO,               0x01080024,__READ      ,__ruby_macro_bits);    
__IO_REG32_BIT(RUBY_BPCTL,               0x01080080,__READ_WRITE,__ruby_bpctl_bits);    
__IO_REG32_BIT(RUBY_BPSTAT,              0x01080084,__READ_WRITE,__ruby_bpstat_bits);    
__IO_REG32_BIT(RUBY_BPAM0,               0x01080088,__READ_WRITE,__ruby_bpam0_bits);    
__IO_REG32_BIT(RUBY_BPAM1,               0x0108008C,__READ_WRITE,__ruby_bpam1_bits);    
__IO_REG32(RUBY_RV_REG,          0x01080090,__READ);    
__IO_REG32(RUBY_RAS_REG,         0x01080094,__READ);    
__IO_REG32(RUBY_AS_REG,          0x01080098,__READ);    
__IO_REG32_BIT(RUBY_DMA_CONTROL,         0x010800C0,__READ_WRITE,__ruby_dma_control_set_clear_bits);    
__IO_REG32_BIT(RUBY_DMA_CONTROL_SET,     0x010800C4,__READ_WRITE,__ruby_dma_control_set_clear_bits);     
__IO_REG32_BIT(RUBY_DMA_CONTROL_CLEAR,   0x010800C8,__READ_WRITE,__ruby_dma_control_set_clear_bits);     
__IO_REG32_BIT(RUBY_DMA_COMP_IRQEN,      0x010800CC,__READ_WRITE,__ruby_dma_comp_irqen_bits);    
__IO_REG32_BIT(RUBY_DMA_COMP_STAT,       0x010800D0,__READ_WRITE,__ruby_dma_comp_stat_bits);      
__IO_REG32_BIT(RUBY_DMA_XFRERR_STAT,     0x010800D4,__READ_WRITE,__ruby_dma_xfrerr_stat_bits);      
__IO_REG32_BIT(RUBY_DMA_CFGERR_STAT,     0x010800D8,__READ_WRITE,__ruby_dma_cfgerr_stat_bits);         
__IO_REG32_BIT(RUBY_DMA_XRUN_STAT,       0x010800DC,__READ      ,__ruby_dma_xrun_stat_bits);         
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL0,      0x01080100,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
#define RUBY_DMA_CHAN_CTRL     RUBY_DMA_CHAN_CTRL0
#define RUBY_DMA_CHAN_CTRL_bit RUBY_DMA_CHAN_CTRL0_bit 
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL1,      0x01080120,__READ_WRITE,__ruby_dma_chan_ctrl_bits);           
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL2,      0x01080140,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL3,      0x01080160,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL4,      0x01080180,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL5,      0x010801A0,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL6,      0x010801C0,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL7,      0x010801E0,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL8,      0x01080200,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL9,      0x01080220,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL10,     0x01080240,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL11,     0x01080260,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL12,     0x01080280,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL13,     0x010802A0,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL14,     0x010802C0,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_CHAN_CTRL15,     0x010802E0,__READ_WRITE,__ruby_dma_chan_ctrl_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR0, 0x01080104,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
#define RUBY_DMA_DMEM_PRAM_ADDR     RUBY_DMA_DMEM_PRAM_ADDR0 
#define RUBY_DMA_DMEM_PRAM_ADDR_bit RUBY_DMA_DMEM_PRAM_ADDR0_bit
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR1, 0x01080124,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR2, 0x01080144,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR3, 0x01080164,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR4, 0x01080184,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR5, 0x010801A4,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR6, 0x010801C4,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR7, 0x010801E4,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR8, 0x01080204,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR9, 0x01080224,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR10,0x01080244,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR11,0x01080264,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR12,0x01080284,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR13,0x010802A4,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR14,0x010802C4,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32_BIT(RUBY_DMA_DMEM_PRAM_ADDR15,0x010802E4,__READ_WRITE,__ruby_dma_dmem_pram_addr_bits);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS0,    0x01080108,__READ_WRITE);          
#define RUBY_DMA_AXI_ADDRESS     RUBY_DMA_AXI_ADDRESS0 
__IO_REG32(RUBY_DMA_AXI_ADDRESS1, 0x01080128,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS2, 0x01080148,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS3, 0x01080168,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS4, 0x01080188,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS5, 0x010801A8,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS6, 0x010801C8,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS7, 0x010801E8,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS8, 0x01080208,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS9, 0x01080228,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS10,0x01080248,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS11,0x01080268,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS12,0x01080288,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS13,0x010802A8,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS14,0x010802C8,__READ_WRITE);          
__IO_REG32(RUBY_DMA_AXI_ADDRESS15,0x010802E8,__READ_WRITE);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT0, 0x0108010C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
#define RUBY_DMA_AXI_TRANS_CNT     RUBY_DMA_AXI_TRANS_CNT0 
#define RUBY_DMA_AXI_TRANS_CNT_bit RUBY_DMA_AXI_TRANS_CNT0_bit
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT1, 0x0108012C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT2, 0x0108014C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT3, 0x0108016C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT4, 0x0108018C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT5, 0x010801AC,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT6, 0x010801CC,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT7, 0x010801EC,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT8, 0x0108020C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT9, 0x0108022C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT10,0x0108024C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT11,0x0108026C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT12,0x0108028C,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT13,0x010802AC,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT14,0x010802CC,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);          
__IO_REG32_BIT(RUBY_DMA_AXI_TRANS_CNT15,0x010802EC,__READ_WRITE,__ruby_dma_axi_trans_cnt_bits);           
//__IO_REG32(RUBY_GP_IN,            0x01080500,__READ      );
//__IO_REG32(RUBY_GP_OUT,           0x01080580,__READ_WRITE);
#define RUBY_GP_IN    MODEM_TSM_RX_OBS
#define RUBY_GP_OUT   MODEM_RFDI_CFG

/***************************************************************************
 **
 **  SLCDC
 **
 ***************************************************************************/

__IO_REG32_BIT(SLCDC_DATABASEADR,        0x01088000,__READ_WRITE,__slcdc_databaseadr_bits);           
__IO_REG32_BIT(SLCDC_DATABUFSIZE,        0x01088004,__READ_WRITE,__slcdc_databufsize_bits);           
__IO_REG32_BIT(SLCDC_COMBASEADR,         0x01088008,__READ_WRITE,__slcdc_combaseaddr_bits);           
__IO_REG32_BIT(SLCDC_COMBUFSIZE,         0x0108800C,__READ_WRITE,__slcdc_combufsize_bits);            
__IO_REG32_BIT(SLCDC_COMSTRINGSIZE,      0x01088010,__READ_WRITE,__slcdc_comstringsize_bits);            
__IO_REG32_BIT(SLCDC_FIFOCONFIG,         0x01088014,__READ_WRITE,__slcdc_fifoconfig_bits);            
__IO_REG32_BIT(SLCDC_LCDCONFIG,          0x01088018,__READ_WRITE,__slcdc_lcdconfig_bits);            
__IO_REG32_BIT(SLCDC_LCDTRANSCONFIG,     0x0108801C,__READ_WRITE,__slcdc_lcdtransconfig_bits);            
__IO_REG32_BIT(SLCDC_SLCDCCTL_STATUS,    0x01088020,__READ_WRITE,__slcdc_slcdcctl_status_bits);            
__IO_REG32_BIT(SLCDC_LCDCLOCKCONFIG,     0x01088024,__READ_WRITE,__slcdc_lcdclockconfig_bits);            
__IO_REG32_BIT(SLCDC_LCDWRITEDATA,       0x01088028,__READ_WRITE,__slcdc_lcdwritedata_bits);            
 
/***************************************************************************
 **
 **  SRC
 **
 ***************************************************************************/

__IO_REG32_BIT(SRC_SBMR,                 0x0108C000,__READ_WRITE,__src_sbmr_bits);            
__IO_REG32_BIT(SRC_SRSR,                 0x0108C004,__READ_WRITE,__src_srsr_bits);            
__IO_REG32_BIT(SRC_SSCR,                 0x0108C008,__READ_WRITE,__src_sscr_bits);            
__IO_REG32_BIT(SRC_SCRCR,                0x0108C00C,__READ_WRITE,__src_scrcr_bits);            
__IO_REG32_BIT(SRC_SRBR,                 0x0108C010,__READ_WRITE,__src_srbr_bits);             
__IO_REG32_BIT(SRC_SGPR,                 0x0108C014,__READ_WRITE,__src_sgpr_bits);             
__IO_REG32_BIT(SRC_AIR1,                 0x0108C018,__READ_WRITE,__src_air1_bits);             
__IO_REG32_BIT(SRC_AIR2,                 0x0108C01C,__READ_WRITE,__src_air2_bits);             

/***************************************************************************
 **
 **  SSI
 **
 ***************************************************************************/

__IO_REG32(SSI_STX0,                 0x010B4000,__READ_WRITE);             
__IO_REG32(SSI_STX1,                 0x010B4004,__READ_WRITE);             
__IO_REG32(SSI_SRX0,                 0x010B4008,__READ      );             
__IO_REG32(SSI_SRX1,                 0x010B400C,__READ      );             
__IO_REG32_BIT(SSI_SISR,                 0x010B4014,__READ      ,__ssi_sisr_bits);             
__IO_REG32_BIT(SSI_SIER,                 0x010B4018,__READ_WRITE,__ssi_sier_bits);               
__IO_REG32_BIT(SSI_STCR,                 0x010B401C,__READ_WRITE,__ssi_stcr_bits);                
__IO_REG32_BIT(SSI_SRCR,                 0x010B4020,__READ_WRITE,__ssi_srcr_bits);                
__IO_REG32_BIT(SSI_STCCR,                0x010B4024,__READ_WRITE,__ssi_strccr_bits);                
__IO_REG32_BIT(SSI_SRCCR,                0x010B4028,__READ_WRITE,__ssi_strccr_bits);                  
__IO_REG32_BIT(SSI_SFCSR,                0x010B402C,__READ_WRITE,__ssi_sfcsr_bits);                
__IO_REG32_BIT(SSI_SACNT,                0x010B4038,__READ_WRITE,__ssi_sacnt_bits);                
__IO_REG32_BIT(SSI_SACADD,               0x010B403C,__READ_WRITE,__ssi_sacadd_bits);                
__IO_REG32_BIT(SSI_SACDAT,               0x010B4040,__READ_WRITE,__ssi_sacdat_bits);                
__IO_REG32_BIT(SSI_SATAG,                0x010B4044,__READ_WRITE,__ssi_satag_bits);                
__IO_REG32_BIT(SSI_STMSK,                0x010B4048,__READ_WRITE,__ssi_stmsk_bits);                
__IO_REG32_BIT(SSI_SRMSK,                0x010B404C,__READ_WRITE,__ssi_srmsk_bits);                
__IO_REG32_BIT(SSI_SACCST,               0x010B4050,__READ      ,__ssi_saccst_bits);                
__IO_REG32_BIT(SSI_SACCEN,               0x010B4054,__WRITE     ,__ssi_saccen_bits);                
__IO_REG32_BIT(SSI_SACCDIS,              0x010B4058,__WRITE     ,__ssi_saccdis_bits);                

/***************************************************************************
 **
 **  TPRC1
 **
 ***************************************************************************/

__IO_REG16_BIT(TPRC1_lutvals_per_ramp,   0x010F0000,__READ_WRITE,__tprc_16_5bit_bits);                
__IO_REG16_BIT(TPRC1_clks_betw_lut,      0x010F0002,__READ_WRITE,__tprc_16_3bit_bits);                
__IO_REG16_BIT(TPRC1_ramp_end_value,     0x010F0004,__READ_WRITE,__tprc_16_14bit_bits);                
__IO_REG16_BIT(TPRC1_mcu_ramp_trigger,   0x010F0006,__WRITE     ,__tprc_16_1bit_bits);                  
__IO_REG16_BIT(TPRC1_mcu_ramp_dac_reset, 0x010F0008,__READ_WRITE,__tprc_16_1bit_bits);                   
__IO_REG16_BIT(TPRC1_tprc_ramp_bypass,   0x010F000A,__READ_WRITE,__tprc_16_1bit_bits);                    
__IO_REG16_BIT(TPRC1_tprc_static,        0x010F000C,__READ_WRITE,__tprc_16_1bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_0,              0x010F000E,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC1_LUT_1,              0x010F0010,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_2,              0x010F0012,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_3,              0x010F0014,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_4,              0x010F0016,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_5,              0x010F0018,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_6,              0x010F001A,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_7,              0x010F001C,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_8,              0x010F001E,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_9,              0x010F0020,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_A,              0x010F0022,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_B,              0x010F0024,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_C,              0x010F0026,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC1_LUT_D,              0x010F0028,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC1_LUT_E,              0x010F002A,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC1_LUT_F,              0x010F002C,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC1_tprc_dacbuf_en,     0x010F002E,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC1_tprc_dac_en,        0x010F0030,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC1_tprc_dac_daz,       0x010F0032,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC1_tprc_clk_en,        0x010F0034,__READ_WRITE,__tprc_16_1bit_bits);                       
  
/***************************************************************************
 **
 **  TPRC2
 **
 ***************************************************************************/
   
__IO_REG16_BIT(TPRC2_lutvals_per_ramp,   0x010FC000,__READ_WRITE,__tprc_16_5bit_bits);                
__IO_REG16_BIT(TPRC2_clks_betw_lut,      0x010FC002,__READ_WRITE,__tprc_16_3bit_bits);                
__IO_REG16_BIT(TPRC2_ramp_end_value,     0x010FC004,__READ_WRITE,__tprc_16_14bit_bits);                
__IO_REG16_BIT(TPRC2_mcu_ramp_trigger,   0x010FC006,__WRITE     ,__tprc_16_1bit_bits);                  
__IO_REG16_BIT(TPRC2_mcu_ramp_dac_reset, 0x010FC008,__READ_WRITE,__tprc_16_1bit_bits);                   
__IO_REG16_BIT(TPRC2_tprc_ramp_bypass,   0x010FC00A,__READ_WRITE,__tprc_16_1bit_bits);                    
__IO_REG16_BIT(TPRC2_tprc_static,        0x010FC00C,__READ_WRITE,__tprc_16_1bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_0,              0x010FC00E,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC2_LUT_1,              0x010FC010,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_2,              0x010FC012,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_3,              0x010FC014,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_4,              0x010FC016,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_5,              0x010FC018,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_6,              0x010FC01A,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_7,              0x010FC01C,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_8,              0x010FC01E,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_9,              0x010FC020,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_A,              0x010FC022,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_B,              0x010FC024,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_C,              0x010FC026,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC2_LUT_D,              0x010FC028,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC2_LUT_E,              0x010FC02A,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC2_LUT_F,              0x010FC02C,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC2_tprc_dacbuf_en,     0x010FC02E,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC2_tprc_dac_en,        0x010FC030,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC2_tprc_dac_daz,       0x010FC032,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC2_tprc_clk_en,        0x010FC034,__READ_WRITE,__tprc_16_1bit_bits);                       
 
/***************************************************************************
 **
 **  TPRC3
 **
 ***************************************************************************/

__IO_REG16_BIT(TPRC3_lutvals_per_ramp,   0x01100000,__READ_WRITE,__tprc_16_5bit_bits);                
__IO_REG16_BIT(TPRC3_clks_betw_lut,      0x01100002,__READ_WRITE,__tprc_16_3bit_bits);                
__IO_REG16_BIT(TPRC3_ramp_end_value,     0x01100004,__READ_WRITE,__tprc_16_14bit_bits);                
__IO_REG16_BIT(TPRC3_mcu_ramp_trigger,   0x01100006,__WRITE     ,__tprc_16_1bit_bits);                  
__IO_REG16_BIT(TPRC3_mcu_ramp_dac_reset, 0x01100008,__READ_WRITE,__tprc_16_1bit_bits);                   
__IO_REG16_BIT(TPRC3_tprc_ramp_bypass,   0x0110000A,__READ_WRITE,__tprc_16_1bit_bits);                    
__IO_REG16_BIT(TPRC3_tprc_static,        0x0110000C,__READ_WRITE,__tprc_16_1bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_0,              0x0110000E,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC3_LUT_1,              0x01100010,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_2,              0x01100012,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_3,              0x01100014,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_4,              0x01100016,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_5,              0x01100018,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_6,              0x0110001A,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_7,              0x0110001C,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_8,              0x0110001E,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_9,              0x01100020,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_A,              0x01100022,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_B,              0x01100024,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_C,              0x01100026,__READ_WRITE,__tprc_16_8bit_bits);                    
__IO_REG16_BIT(TPRC3_LUT_D,              0x01100028,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC3_LUT_E,              0x0110002A,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC3_LUT_F,              0x0110002C,__READ_WRITE,__tprc_16_8bit_bits);                     
__IO_REG16_BIT(TPRC3_tprc_dacbuf_en,     0x0110002E,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC3_tprc_dac_en,        0x01100030,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC3_tprc_dac_daz,       0x01100032,__READ_WRITE,__tprc_16_1bit_bits);                      
__IO_REG16_BIT(TPRC3_tprc_clk_en,        0x01100034,__READ_WRITE,__tprc_16_1bit_bits);                       
  
/***************************************************************************
 **
 **  TSM
 **
 ***************************************************************************/

__IO_REG32_BIT(TSM_IPR_TSM_CTRL,        0x010F4000,__READ_WRITE,__tsm_ipr_tsm_ctrl_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_RX_STEPS,    0x010F4004,__READ_WRITE,__tsm_ipr_tsm_rx_steps_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_TX_STEPS,    0x010F4008,__READ_WRITE,__tsm_ipr_tsm_tx_steps_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_RX_TBL,      0x010F400C,__READ_WRITE,__tsm_ipr_tsm_rx_tbl_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_TX_TBL,      0x010F4010,__READ_WRITE,__tsm_ipr_tsm_tx_tbl_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_WARMDOWN,    0x010F4014,__READ_WRITE,__tsm_ipr_tsm_warmdown_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_MUX_IN_RX,   0x010F4018,__READ_WRITE,__tsm_ipr_tsm_mux_in_rx_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_IMR,         0x010F401C,__READ_WRITE,__tsm_ipr_tsm_imr_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_ISR,         0x010F4020,__READ_WRITE,__tsm_ipr_tsm_isr_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_MUX_OUT_RX,  0x010F4024,__READ      ,__tsm_ipr_tsm_mux_out_rx_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_CCM_CTRL,    0x010F4028,__READ_WRITE,__tsm_ipr_tsm_ccm_ctrl_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_MUX_IN_TX,   0x010F402C,__READ_WRITE,__tsm_ipr_tsm_mux_in_tx_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_MUX_OUT_TX,  0x010F4030,__READ      ,__tsm_ipr_tsm_mux_out_tx_bits);                       
__IO_REG32_BIT(TSM_IPR_TSM_MUX_IN_COM,  0x010F4034,__READ_WRITE,__tsm_ipr_tsm_mux_in_com_bits);                         
__IO_REG32_BIT(TSM_IPR_TSM_MUX_OUT_COM, 0x010F4038,__READ      ,__tsm_ipr_tsm_mux_out_com_bits);                         

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
 
__IO_REG32_BIT(UART1_URXD,              0x010B8000,__READ      ,__uart_urxd_bits);                         
__IO_REG32_BIT(UART1_TRXD,              0x010B8040,__WRITE     ,__uart_utxd_bits);                         
__IO_REG32_BIT(UART1_UCR1,              0x010B8080,__READ_WRITE,__uart_ucr1_bits);                         
__IO_REG32_BIT(UART1_UCR2,              0x010B8084,__READ_WRITE,__uart_ucr2_bits);                         
__IO_REG32_BIT(UART1_UCR3,              0x010B8088,__READ_WRITE,__uart_ucr3_bits);                         
__IO_REG32_BIT(UART1_UCR4,              0x010B808C,__READ_WRITE,__uart_ucr4_bits);                           
__IO_REG32_BIT(UART1_UFCR,              0x010B8090,__READ_WRITE,__uart_ufcr_bits);                           
__IO_REG32_BIT(UART1_USR1,              0x010B8094,__READ_WRITE,__uart_usr1_bits);                           
__IO_REG32_BIT(UART1_USR2,              0x010B8098,__READ_WRITE,__uart_usr2_bits);                             
__IO_REG32_BIT(UART1_UESC,              0x010B809C,__READ_WRITE,__uart_uesc_bits);                             
__IO_REG32_BIT(UART1_UTIM,              0x010B80A0,__READ_WRITE,__uart_utim_bits);                             
__IO_REG32_BIT(UART1_UBIR,              0x010B80A4,__READ_WRITE,__uart_ubir_bits);                             
__IO_REG32_BIT(UART1_UBMR,              0x010B80A8,__READ_WRITE,__uart_ubmr_bits);                              
__IO_REG32_BIT(UART1_UBRC,              0x010B80AC,__READ_WRITE,__uart_ubrc_bits);                              
__IO_REG32_BIT(UART1_ONEMS,             0x010B80B0,__READ_WRITE,__uart_onems_bits);                              
__IO_REG32_BIT(UART1_UTS,               0x010B80B4,__READ_WRITE,__uart_uts_bits);                              
 
/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
 
__IO_REG32_BIT(UART2_URXD,              0x010BC000,__READ      ,__uart_urxd_bits);                         
__IO_REG32_BIT(UART2_TRXD,              0x010BC040,__WRITE     ,__uart_utxd_bits);                         
__IO_REG32_BIT(UART2_UCR1,              0x010BC080,__READ_WRITE,__uart_ucr1_bits);                         
__IO_REG32_BIT(UART2_UCR2,              0x010BC084,__READ_WRITE,__uart_ucr2_bits);                         
__IO_REG32_BIT(UART2_UCR3,              0x010BC088,__READ_WRITE,__uart_ucr3_bits);                         
__IO_REG32_BIT(UART2_UCR4,              0x010BC08C,__READ_WRITE,__uart_ucr4_bits);                           
__IO_REG32_BIT(UART2_UFCR,              0x010BC090,__READ_WRITE,__uart_ufcr_bits);                           
__IO_REG32_BIT(UART2_USR1,              0x010BC094,__READ_WRITE,__uart_usr1_bits);                           
__IO_REG32_BIT(UART2_USR2,              0x010BC098,__READ_WRITE,__uart_usr2_bits);                             
__IO_REG32_BIT(UART2_UESC,              0x010BC09C,__READ_WRITE,__uart_uesc_bits);                             
__IO_REG32_BIT(UART2_UTIM,              0x010BC0A0,__READ_WRITE,__uart_utim_bits);                             
__IO_REG32_BIT(UART2_UBIR,              0x010BC0A4,__READ_WRITE,__uart_ubir_bits);                             
__IO_REG32_BIT(UART2_UBMR,              0x010BC0A8,__READ_WRITE,__uart_ubmr_bits);                              
__IO_REG32_BIT(UART2_UBRC,              0x010BC0AC,__READ_WRITE,__uart_ubrc_bits);                              
__IO_REG32_BIT(UART2_ONEMS,             0x010BC0B0,__READ_WRITE,__uart_onems_bits);                              
__IO_REG32_BIT(UART2_UTS,               0x010BC0B4,__READ_WRITE,__uart_uts_bits);                              
  
/***************************************************************************
 **
 **  USB_IF
 **
 ***************************************************************************/

__IO_REG8_BIT(USB_FS_PER_ID,                0x010C0000,__READ      ,__usb_fs_per_id_bits);                              
__IO_REG8_BIT(USB_FS_ID_COMP,               0x010C0004,__READ      ,__usb_fs_id_comp_bits);                              
__IO_REG8(USB_FS_REV,                   0x010C0008,__READ      );                              
__IO_REG8_BIT(USB_FS_ADD_INFO,                 0x010C000C,__READ      ,__usb_fs_add_info_bits);                              
__IO_REG8_BIT(USB_FS_OTG_CTRL,                 0x010C001C,__READ_WRITE,__usb_fs_otg_ctrl_bits);                              
__IO_REG8_BIT(USB_FS_INT_STAT,                 0x010C0080,__READ_WRITE,__usb_fs_int_stat_bits);                              
__IO_REG8_BIT(USB_FS_INT_ENB,                  0x010C0084,__READ_WRITE,__usb_fs_int_stat_bits);                              
__IO_REG8_BIT(USB_FS_ERR_STAT,                 0x010C0088,__READ_WRITE,__usb_fs_err_stat_bits);                              
__IO_REG8_BIT(USB_FS_ERR_ENB,                  0x010C008C,__READ_WRITE,__usb_fs_err_stat_bits);                              
__IO_REG8_BIT(USB_FS_STAT,                     0x010C0090,__READ      ,__usb_fs_stat_bits);                              
__IO_REG8_BIT(USB_FS_CTL,                      0x010C0094,__READ_WRITE,__usb_fs_ctl_bits);                              
__IO_REG8_BIT(USB_FS_ADDR,                     0x010C0098,__READ_WRITE,__usb_fs_addr_bits);                              
__IO_REG8_BIT(USB_FS_FRM_NUML,                 0x010C00A0,__READ      ,__usb_fs_frm_numl_bits);                              
__IO_REG8_BIT(USB_FS_FRM_NUMH,                 0x010C00A4,__READ      ,__usb_fs_frm_numh_bits);                              
__IO_REG8_BIT(USB_FS_BDT_PAGE_01,              0x010C009C,__READ_WRITE,__usb_fs_bdt_page_01_bits);                              
__IO_REG8_BIT(USB_FS_BDT_PAGE_02,              0x010C00B0,__READ_WRITE,__usb_fs_bdt_page_02_bits);                              
__IO_REG8_BIT(USB_FS_BDT_PAGE_03,              0x010C00B4,__READ_WRITE,__usb_fs_bdt_page_03_bits);                              
__IO_REG8_BIT(USB_FS_ENDPT_CTL0,               0x010C00C0,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL1,               0x010C00C4,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL2,               0x010C00C8,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL3,               0x010C00CC,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL4,               0x010C00D0,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL5,               0x010C00D4,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL6,               0x010C00D8,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_ENDPT_CTL7,               0x010C00DC,__READ_WRITE,__usb_fs_endpt_ctl_bits);                               
__IO_REG8_BIT(USB_FS_TOP_CTRL,                 0x010C0100,__READ_WRITE,__usb_fs_top_ctrl_bits);                               
 
/***************************************************************************
 **
 **  WDOG
 **
 ***************************************************************************/

__IO_REG16_BIT(WDOG_WCR,                       0x010A0000,__READ_WRITE,__wdog_wcr_bits);                               
__IO_REG16(WDOG_WSR,                       0x010A0002,__READ_WRITE);                               
__IO_REG16_BIT(WDOG_WRSR,                      0x010A0004,__READ      ,__wdog_wrsr_bits);                               
__IO_REG16_BIT(WDOG_WICR,                      0x010A0006,__READ_WRITE,__wdog_wicr_bits);                               
__IO_REG16_BIT(WDOG_WMCR,                      0x010A0008,__READ_WRITE,__wdog_wmcr_bits);                               
    
/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV        0x00  /* Reset                                       */
#define UNDEFV        0x04  /* Undefined instruction                       */
#define SWIV          0x08  /* Software interrupt                          */
#define PABORTV       0x0c  /* Prefetch abort                              */
#define DABORTV       0x10  /* Data abort                                  */
#define IRQV          0x18  /* Normal interrupt                            */
#define FIQV          0x1c  /* Fast interrupt                              */

/***************************************************************************
 **
 **   MC13260 interrupt sources
 **
 ***************************************************************************/
 
#define INT_MODEM_BRKPNT        0             /* Modem Breakpoint interrupt */
#define INT_MODEM_DMA           1             /* Modem DMA complete interrupt */
#define INT_MODEM_TX            2             /* Modem TX Event */
#define INT_MODEM_RX            3             /* Modem RX Event */
#define INT_MODEM_DMA_ERR       4             /* Modem DMA Error */
#define INT_MODEM_DONE          5             /* Modem Done */
#define INT_GPADC               7             /* GPADC Interrupt */
#define INT_I2C                10             /* I2C Interrupt */
#define INT_SSI                12             /* SSI Interrupt */
#define INT_CSPI               14             /* CSPI Interrupt */
#define INT_ASM                15             /* ASM Interrupt */
#define INT_RNGB               16             /* RNGB Interrupt */
#define INT_EPIT               17             /* EPIT Interrupt */
#define INT_RTC                18             /* RTC Interrupt */
#define INT_DSM                23             /* Active-Low Deep Sleep Module  interrupt line */
#define INT_KPP                24             /* KPP Interrupt */
#define INT_TMR0               25             /* TMR Interrupt from Counter 0 */
#define INT_TMR1               26             /* TMR Interrupt from Counter 1 */
#define INT_TMR2               27             /* TMR Interrupt from Counter 2 */
#define INT_TMR3               28             /* TMR Interrupt from Counter 3 */
#define INT_GPT                29             /* "OR" of GPT Rollover interrupt line */
#define INT_UART2              32             /* UART2 Rx, UART Tx, UART Mint */
#define INT_HDMA_ERR           33             /* HDMA Transfer Error interrupt */
#define INT_HDMA_COMPLETE      34             /* HDMA combined Transfer Complete and Buffer Not Ready interrupts */
#define INT_USB_FS             35             /* USB-FS MCU Interrupt */
#define INT_SLCDC              42             /* SLCDC Interrupt */
#define INT_PWM1               43             /* PWM1 Interrupt */
#define INT_PWM2               44             /* PWM2 Interrupt */
#define INT_UART1              45             /* UART1 Rx, UART Tx, UART Mint*/
#define INT_IIM                46             /* Interrupt request to the processor */
#define INT_TSM_TX             48             /* TSM: TX End Warmdown */
#define INT_TSM_RX             49             /* TSM: RX End Warmdown */
#define INT_TSM_GP             50             /* TSM: General Purpose */
#define INT_GPIO               52             /* OR of GPIO(31:0) Interrupts */
#define INT_WDOG               55             /* Watchdog Interrupt, signals an imminent wdog timeout */
#define INT_CODEC              59             /* Voice CODEC FIFO overflow/underflow error*/
#define INT_RFDI               60             /* RFDI FIFO overflow/underflow error */
#define INT_SYNTH              61             /* Synth Control (lock monitor) */
#define INT_SYNTH_OVFL         62             /* Synth Control FIFO overflow */ 

#endif    /* __MC13260_H */
