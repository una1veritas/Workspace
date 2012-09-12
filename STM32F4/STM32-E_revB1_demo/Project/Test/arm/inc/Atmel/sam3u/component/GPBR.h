/* ============================================================================= */
/**  SOFTWARE API DEFINITION FOR General Purpose Backup Register */
/* ============================================================================= */
/** \addtogroup SAM3U_GPBR General Purpose Backup Register */
/*@{*/

#ifndef __IAR_SYSTEMS_ASM__
/** \brief Gpbr hardware registers */
typedef struct {
  RwReg SYS_GPBR0; /**< \brief (Gpbr Offset: 0x0) General Purpose Backup Register 0 */
  RwReg SYS_GPBR1; /**< \brief (Gpbr Offset: 0x4) General Purpose Backup Register 1 */
  RwReg SYS_GPBR2; /**< \brief (Gpbr Offset: 0x8) General Purpose Backup Register 2 */
  RwReg SYS_GPBR3; /**< \brief (Gpbr Offset: 0xC) General Purpose Backup Register 3 */
} Gpbr;
#endif /* __IAR_SYSTEMS_ASM__ */
/* -------- SYS_GPBR0 : (GPBR Offset: 0x0) General Purpose Backup Register 0 -------- */
#define SYS_GPBR0_GPBR_VALUE0_Pos 0
#define SYS_GPBR0_GPBR_VALUE0_Msk (0xffffffffu << SYS_GPBR0_GPBR_VALUE0_Pos) /**< \brief (SYS_GPBR0) Value of GPBR x */
#define SYS_GPBR0_GPBR_VALUE0(value) ((SYS_GPBR0_GPBR_VALUE0_Msk & ((value) << SYS_GPBR0_GPBR_VALUE0_Pos)))
/* -------- SYS_GPBR1 : (GPBR Offset: 0x4) General Purpose Backup Register 1 -------- */
#define SYS_GPBR1_GPBR_VALUE1_Pos 0
#define SYS_GPBR1_GPBR_VALUE1_Msk (0xffffffffu << SYS_GPBR1_GPBR_VALUE1_Pos) /**< \brief (SYS_GPBR1) Value of GPBR x */
#define SYS_GPBR1_GPBR_VALUE1(value) ((SYS_GPBR1_GPBR_VALUE1_Msk & ((value) << SYS_GPBR1_GPBR_VALUE1_Pos)))
/* -------- SYS_GPBR2 : (GPBR Offset: 0x8) General Purpose Backup Register 2 -------- */
#define SYS_GPBR2_GPBR_VALUE2_Pos 0
#define SYS_GPBR2_GPBR_VALUE2_Msk (0xffffffffu << SYS_GPBR2_GPBR_VALUE2_Pos) /**< \brief (SYS_GPBR2) Value of GPBR x */
#define SYS_GPBR2_GPBR_VALUE2(value) ((SYS_GPBR2_GPBR_VALUE2_Msk & ((value) << SYS_GPBR2_GPBR_VALUE2_Pos)))
/* -------- SYS_GPBR3 : (GPBR Offset: 0xC) General Purpose Backup Register 3 -------- */
#define SYS_GPBR3_GPBR_VALUE3_Pos 0
#define SYS_GPBR3_GPBR_VALUE3_Msk (0xffffffffu << SYS_GPBR3_GPBR_VALUE3_Pos) /**< \brief (SYS_GPBR3) Value of GPBR x */
#define SYS_GPBR3_GPBR_VALUE3(value) ((SYS_GPBR3_GPBR_VALUE3_Msk & ((value) << SYS_GPBR3_GPBR_VALUE3_Pos)))

/*@}*/

