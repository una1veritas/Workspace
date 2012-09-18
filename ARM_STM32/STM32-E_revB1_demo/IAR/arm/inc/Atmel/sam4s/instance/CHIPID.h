/* %ATMEL_LICENCE% */

#ifndef _SAM4S_CHIPID_INSTANCE_
#define _SAM4S_CHIPID_INSTANCE_

/* ========== Register definition for CHIPID peripheral ========== */
#ifdef __IAR_SYSTEMS_ASM__
#define REG_CHIPID_CIDR          (0x400E0740U) /**< \brief (CHIPID) Chip ID Register */
#define REG_CHIPID_EXID          (0x400E0744U) /**< \brief (CHIPID) Chip ID Extension Register */
#else
#define REG_CHIPID_CIDR (*(RoReg*)0x400E0740U) /**< \brief (CHIPID) Chip ID Register */
#define REG_CHIPID_EXID (*(RoReg*)0x400E0744U) /**< \brief (CHIPID) Chip ID Extension Register */
#endif /* __IAR_SYSTEMS_ASM__ */

#endif /* _SAM4S_CHIPID_INSTANCE_ */
