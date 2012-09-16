/* ========== Register definition for DMAC peripheral ========== */
#define REG_DMAC_GCFG   REG_ACCESS(RwReg, 0x400B0000U) /**< \brief (DMAC) DMAC Global Configuration Register */
#define REG_DMAC_EN     REG_ACCESS(RwReg, 0x400B0004U) /**< \brief (DMAC) DMAC Enable Register */
#define REG_DMAC_SREQ   REG_ACCESS(RwReg, 0x400B0008U) /**< \brief (DMAC) DMAC Software Single Request Register */
#define REG_DMAC_CREQ   REG_ACCESS(RwReg, 0x400B000CU) /**< \brief (DMAC) DMAC Software Chunk Transfer Request Register */
#define REG_DMAC_LAST   REG_ACCESS(RwReg, 0x400B0010U) /**< \brief (DMAC) DMAC Software Last Transfer Flag Register */
#define REG_DMAC_EBCIER REG_ACCESS(WoReg, 0x400B0018U) /**< \brief (DMAC) DMAC Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register. */
#define REG_DMAC_EBCIDR REG_ACCESS(WoReg, 0x400B001CU) /**< \brief (DMAC) DMAC Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register. */
#define REG_DMAC_EBCIMR REG_ACCESS(RoReg, 0x400B0020U) /**< \brief (DMAC) DMAC Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register. */
#define REG_DMAC_EBCISR REG_ACCESS(RoReg, 0x400B0024U) /**< \brief (DMAC) DMAC Error, Chained Buffer transfer completed and Buffer transfer completed Status Register. */
#define REG_DMAC_CHER   REG_ACCESS(WoReg, 0x400B0028U) /**< \brief (DMAC) DMAC Channel Handler Enable Register */
#define REG_DMAC_CHDR   REG_ACCESS(WoReg, 0x400B002CU) /**< \brief (DMAC) DMAC Channel Handler Disable Register */
#define REG_DMAC_CHSR   REG_ACCESS(RoReg, 0x400B0030U) /**< \brief (DMAC) DMAC Channel Handler Status Register */
#define REG_DMAC_SADDR0 REG_ACCESS(RwReg, 0x400B003CU) /**< \brief (DMAC) DMAC Channel Source Address Register (ch_num = 0) */
#define REG_DMAC_DADDR0 REG_ACCESS(RwReg, 0x400B0040U) /**< \brief (DMAC) DMAC Channel Destination Address Register (ch_num = 0) */
#define REG_DMAC_DSCR0  REG_ACCESS(RwReg, 0x400B0044U) /**< \brief (DMAC) DMAC Channel Descriptor Address Register (ch_num = 0) */
#define REG_DMAC_CTRLA0 REG_ACCESS(RwReg, 0x400B0048U) /**< \brief (DMAC) DMAC Channel Control A Register (ch_num = 0) */
#define REG_DMAC_CTRLB0 REG_ACCESS(RwReg, 0x400B004CU) /**< \brief (DMAC) DMAC Channel Control B Register (ch_num = 0) */
#define REG_DMAC_CFG0   REG_ACCESS(RwReg, 0x400B0050U) /**< \brief (DMAC) DMAC Channel Configuration Register (ch_num = 0) */
#define REG_DMAC_SADDR1 REG_ACCESS(RwReg, 0x400B0064U) /**< \brief (DMAC) DMAC Channel Source Address Register (ch_num = 1) */
#define REG_DMAC_DADDR1 REG_ACCESS(RwReg, 0x400B0068U) /**< \brief (DMAC) DMAC Channel Destination Address Register (ch_num = 1) */
#define REG_DMAC_DSCR1  REG_ACCESS(RwReg, 0x400B006CU) /**< \brief (DMAC) DMAC Channel Descriptor Address Register (ch_num = 1) */
#define REG_DMAC_CTRLA1 REG_ACCESS(RwReg, 0x400B0070U) /**< \brief (DMAC) DMAC Channel Control A Register (ch_num = 1) */
#define REG_DMAC_CTRLB1 REG_ACCESS(RwReg, 0x400B0074U) /**< \brief (DMAC) DMAC Channel Control B Register (ch_num = 1) */
#define REG_DMAC_CFG1   REG_ACCESS(RwReg, 0x400B0078U) /**< \brief (DMAC) DMAC Channel Configuration Register (ch_num = 1) */
#define REG_DMAC_SADDR2 REG_ACCESS(RwReg, 0x400B008CU) /**< \brief (DMAC) DMAC Channel Source Address Register (ch_num = 2) */
#define REG_DMAC_DADDR2 REG_ACCESS(RwReg, 0x400B0090U) /**< \brief (DMAC) DMAC Channel Destination Address Register (ch_num = 2) */
#define REG_DMAC_DSCR2  REG_ACCESS(RwReg, 0x400B0094U) /**< \brief (DMAC) DMAC Channel Descriptor Address Register (ch_num = 2) */
#define REG_DMAC_CTRLA2 REG_ACCESS(RwReg, 0x400B0098U) /**< \brief (DMAC) DMAC Channel Control A Register (ch_num = 2) */
#define REG_DMAC_CTRLB2 REG_ACCESS(RwReg, 0x400B009CU) /**< \brief (DMAC) DMAC Channel Control B Register (ch_num = 2) */
#define REG_DMAC_CFG2   REG_ACCESS(RwReg, 0x400B00A0U) /**< \brief (DMAC) DMAC Channel Configuration Register (ch_num = 2) */
#define REG_DMAC_SADDR3 REG_ACCESS(RwReg, 0x400B00B4U) /**< \brief (DMAC) DMAC Channel Source Address Register (ch_num = 3) */
#define REG_DMAC_DADDR3 REG_ACCESS(RwReg, 0x400B00B8U) /**< \brief (DMAC) DMAC Channel Destination Address Register (ch_num = 3) */
#define REG_DMAC_DSCR3  REG_ACCESS(RwReg, 0x400B00BCU) /**< \brief (DMAC) DMAC Channel Descriptor Address Register (ch_num = 3) */
#define REG_DMAC_CTRLA3 REG_ACCESS(RwReg, 0x400B00C0U) /**< \brief (DMAC) DMAC Channel Control A Register (ch_num = 3) */
#define REG_DMAC_CTRLB3 REG_ACCESS(RwReg, 0x400B00C4U) /**< \brief (DMAC) DMAC Channel Control B Register (ch_num = 3) */
#define REG_DMAC_CFG3   REG_ACCESS(RwReg, 0x400B00C8U) /**< \brief (DMAC) DMAC Channel Configuration Register (ch_num = 3) */
