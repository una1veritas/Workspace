/* ========== Register definition for PMC peripheral ========== */
#define REG_PMC_SCER   REG_ACCESS(WoReg, 0x400E0400U) /**< \brief (PMC) System Clock Enable Register */
#define REG_PMC_SCDR   REG_ACCESS(WoReg, 0x400E0404U) /**< \brief (PMC) System Clock Disable Register */
#define REG_PMC_SCSR   REG_ACCESS(RoReg, 0x400E0408U) /**< \brief (PMC) System Clock Status Register */
#define REG_PMC_PCER   REG_ACCESS(WoReg, 0x400E0410U) /**< \brief (PMC) Peripheral Clock Enable Register */
#define REG_PMC_PCDR   REG_ACCESS(WoReg, 0x400E0414U) /**< \brief (PMC) Peripheral Clock Disable Register */
#define REG_PMC_PCSR   REG_ACCESS(RoReg, 0x400E0418U) /**< \brief (PMC) Peripheral Clock Status Register */
#define REG_CKGR_MOR   REG_ACCESS(RwReg, 0x400E0420U) /**< \brief (PMC) Main Oscillator Register */
#define REG_CKGR_MCFR  REG_ACCESS(RoReg, 0x400E0424U) /**< \brief (PMC) Main Clock Frequency Register */
#define REG_CKGR_PLLR  REG_ACCESS(RwReg, 0x400E0428U) /**< \brief (PMC) PLL Register */
#define REG_PMC_MCKR   REG_ACCESS(RwReg, 0x400E0430U) /**< \brief (PMC) Master Clock Register */
#define REG_PMC_PCK    REG_ACCESS(RwReg, 0x400E0440U) /**< \brief (PMC) Programmable Clock 0 Register */
#define REG_PMC_IER    REG_ACCESS(WoReg, 0x400E0460U) /**< \brief (PMC) Interrupt Enable Register */
#define REG_PMC_IDR    REG_ACCESS(WoReg, 0x400E0464U) /**< \brief (PMC) Interrupt Disable Register */
#define REG_PMC_SR     REG_ACCESS(RoReg, 0x400E0468U) /**< \brief (PMC) Status Register */
#define REG_PMC_IMR    REG_ACCESS(RoReg, 0x400E046CU) /**< \brief (PMC) Interrupt Mask Register */
#define REG_PMC_FSMR   REG_ACCESS(RwReg, 0x400E0470U) /**< \brief (PMC) Fast Startup Mode Register */
#define REG_PMC_FSPR   REG_ACCESS(RwReg, 0x400E0474U) /**< \brief (PMC) Fast Startup Polarity Register */
#define REG_PMC_FOCR   REG_ACCESS(WoReg, 0x400E0478U) /**< \brief (PMC) Fault Output Clear Register */
#define REG_PMC_WPMR   REG_ACCESS(RwReg, 0x400E04E4U) /**< \brief (PMC) Write Protect Mode Register */
#define REG_PMC_WPSR   REG_ACCESS(RoReg, 0x400E04E8U) /**< \brief (PMC) Write Protect Status Register */
#define REG_PMC_OCR    REG_ACCESS(RwReg, 0x400E0510U) /**< \brief (PMC) Oscillator Calibration Register */
