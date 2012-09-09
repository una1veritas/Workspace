/* ========== Register definition for TWI1 peripheral ========== */
#define REG_TWI1_CR   REG_ACCESS(WoReg, 0x40088000U) /**< \brief (TWI1) Control Register */
#define REG_TWI1_MMR  REG_ACCESS(RwReg, 0x40088004U) /**< \brief (TWI1) Master Mode Register */
#define REG_TWI1_SMR  REG_ACCESS(RwReg, 0x40088008U) /**< \brief (TWI1) Slave Mode Register */
#define REG_TWI1_IADR REG_ACCESS(RwReg, 0x4008800CU) /**< \brief (TWI1) Internal Address Register */
#define REG_TWI1_CWGR REG_ACCESS(RwReg, 0x40088010U) /**< \brief (TWI1) Clock Waveform Generator Register */
#define REG_TWI1_SR   REG_ACCESS(RoReg, 0x40088020U) /**< \brief (TWI1) Status Register */
#define REG_TWI1_IER  REG_ACCESS(WoReg, 0x40088024U) /**< \brief (TWI1) Interrupt Enable Register */
#define REG_TWI1_IDR  REG_ACCESS(WoReg, 0x40088028U) /**< \brief (TWI1) Interrupt Disable Register */
#define REG_TWI1_IMR  REG_ACCESS(RoReg, 0x4008802CU) /**< \brief (TWI1) Interrupt Mask Register */
#define REG_TWI1_RHR  REG_ACCESS(RoReg, 0x40088030U) /**< \brief (TWI1) Receive Holding Register */
#define REG_TWI1_THR  REG_ACCESS(WoReg, 0x40088034U) /**< \brief (TWI1) Transmit Holding Register */
#define REG_TWI1_RPR  REG_ACCESS(RwReg, 0x40088100U) /**< \brief (TWI1) Receive Pointer Register */
#define REG_TWI1_RCR  REG_ACCESS(RwReg, 0x40088104U) /**< \brief (TWI1) Receive Counter Register */
#define REG_TWI1_TPR  REG_ACCESS(RwReg, 0x40088108U) /**< \brief (TWI1) Transmit Pointer Register */
#define REG_TWI1_TCR  REG_ACCESS(RwReg, 0x4008810CU) /**< \brief (TWI1) Transmit Counter Register */
#define REG_TWI1_RNPR REG_ACCESS(RwReg, 0x40088110U) /**< \brief (TWI1) Receive Next Pointer Register */
#define REG_TWI1_RNCR REG_ACCESS(RwReg, 0x40088114U) /**< \brief (TWI1) Receive Next Counter Register */
#define REG_TWI1_TNPR REG_ACCESS(RwReg, 0x40088118U) /**< \brief (TWI1) Transmit Next Pointer Register */
#define REG_TWI1_TNCR REG_ACCESS(RwReg, 0x4008811CU) /**< \brief (TWI1) Transmit Next Counter Register */
#define REG_TWI1_PTCR REG_ACCESS(WoReg, 0x40088120U) /**< \brief (TWI1) Transfer Control Register */
#define REG_TWI1_PTSR REG_ACCESS(RoReg, 0x40088124U) /**< \brief (TWI1) Transfer Status Register */
