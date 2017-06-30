#include "fatBinaryCtl.h"
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000868,0x0000004001010002,0x0000000000000628\n"
".quad 0x0000000000000000,0x0000001400010007,0x0000000000000000,0x0000000000000015\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000004b00be0002,0x0000000000000000,0x0000000000000580,0x0000000000000380\n"
".quad 0x0038004000140514,0x0001000800400003,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x2e747865742e006f,0x7078617338315a5f,0x5f36656e696c5f79,0x66696c656e72656b\n"
".quad 0x766e2e005f536650,0x5a5f2e6f666e692e,0x5f79707861733831,0x656b5f36656e696c\n"
".quad 0x665066696c656e72,0x732e766e2e005f53,0x5a5f2e6465726168,0x5f79707861733831\n"
".quad 0x656b5f36656e696c,0x665066696c656e72,0x632e766e2e005f53,0x30746e6174736e6f\n"
".quad 0x78617338315a5f2e,0x36656e696c5f7970,0x696c656e72656b5f,0x2e00005f53665066\n"
".quad 0x6261747274736873,0x6261747274732e00,0x6261746d79732e00,0x6261746d79732e00\n"
".quad 0x2e0078646e68735f,0x006f666e692e766e,0x7078617338315a5f,0x5f36656e696c5f79\n"
".quad 0x66696c656e72656b,0x65742e005f536650,0x7338315a5f2e7478,0x6e696c5f79707861\n"
".quad 0x656e72656b5f3665,0x005f53665066696c,0x6f666e692e766e2e,0x78617338315a5f2e\n"
".quad 0x36656e696c5f7970,0x696c656e72656b5f,0x6e2e005f53665066,0x6465726168732e76\n"
".quad 0x78617338315a5f2e,0x36656e696c5f7970,0x696c656e72656b5f,0x6e2e005f53665066\n"
".quad 0x6174736e6f632e76,0x38315a5f2e30746e,0x696c5f7970786173,0x6e72656b5f36656e\n"
".quad 0x5f53665066696c65,0x006d617261705f00,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x000700030000004f,0x0000000000000000,0x0000000000000000\n"
".quad 0x00060003000000c0,0x0000000000000000,0x0000000000000000,0x0007101200000032\n"
".quad 0x0000000000000000,0x0000000000000070,0x0000000300082304,0x0008120400000000\n"
".quad 0x0000000000000003,0x0000000300081104,0x00080a0400000000,0x0018002000000002\n"
".quad 0x000c170400181903,0x0010000300000000,0x000c17040021f000,0x0008000200000000\n"
".quad 0x000c17040021f000,0x0004000100000000,0x000c17040011f000,0x0000000000000000\n"
".quad 0x000000000011f000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x00005de400000000\n"
".quad 0x94001c0428004404,0x84009c042c000000,0x10015de22c000000,0x20001ca318000000\n"
".quad 0xa0009ca320044000,0xb000dce3200b8000,0xc0011ca3208a8000,0x00209c85200b8000\n"
".quad 0xd0015ce384000000,0x00401c85208a8000,0x90201c0084000000,0x00401c8530004000\n"
".quad 0x00001de794000000,0x0000000080000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000040,0x00000000000000ce,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000\n"
".quad 0x0000000000000000,0x000000000000010e,0x00000000000000f2,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000200,0x0000000000000060,0x0000000200000002\n"
".quad 0x0000000000000008,0x0000000000000018,0x7000000000000029,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000260,0x0000000000000024,0x0000000000000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x7000000000000055,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000284,0x0000000000000050,0x0000000700000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x00000001000000a3,0x0000000000000002\n"
".quad 0x0000000000000000,0x00000000000002d4,0x0000000000000038,0x0000000700000000\n"
".quad 0x0000000000000004,0x0000000000000000,0x0000000100000032,0x0000000000000006\n"
".quad 0x0000000000000000,0x000000000000030c,0x0000000000000070,0x0600000300000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x0000000500000006,0x0000000000000580\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000000a8,0x00000000000000a8\n"
".quad 0x0000000000000008,0x0000000500000001,0x00000000000002d4,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000000a8,0x00000000000000a8,0x0000000000000008\n"
".quad 0x0000000600000001,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000008,0x0000004801010001\n"
".quad 0x00000000000001b8,0x00000040000001b6,0x0000001400040003,0x0000000000000000\n"
".quad 0x0000000000002015,0x0000000000000000,0x000000000000037d,0x0000000000000000\n"
".quad 0x762e1cf200010a13,0x34206e6f69737265,0x677261742e0a332e,0x30325f6d73207465\n"
".quad 0x7365726464612e0a,0x3620657a69735f73,0x69736928ff002f34,0x746e652e20656c62\n"
".quad 0x7338315a5f207972,0x6e696c5f79707861,0x656e72656b5f3665,0x285f53665066696c\n"
".quad 0x206d617261702e0a,0x110a002a3233752e,0x322c305f3500285f,0x3116130032661f00\n"
".quad 0x00323436753f0032,0x33a21e0032321f11,0x65722e0a7b0a290a,0x353c662563009767\n"
".quad 0x0011621000113b3e,0x343600f200117218,0x3b3e383c64722520,0x03006d646c0a0a0a\n"
".quad 0xd65b202c314f0039,0x12003a3b5d261100,0x1f14003b0f005175,0x003b321f00003b32\n"
".quad 0x630a3b5d3303f413,0x672e6f742e617476,0x2100416c61626f6c,0x05001f0f00472c33\n"
".quad 0x0a3b3161001f3411,0x7225d801eb766f6d,0x6961746325202c31,0x2c326c0017782e64\n"
".quad 0x5f000016746e2520,0x2e64617100150400,0x34230018732e6f6c,0x72c3004f0100332c\n"
".quad 0x772e6c756d0a3b33,0x3564320021656469,0x610a3b348200272c,0x36260090732e6464\n"
".quad 0x03010f351100962c,0x010f0001850200b0,0x2600355d19002600,0x0000350f00ea2c37\n"
".quad 0x3b5d37a300353312,0x186e722e616d660a,0xdc0100522c342200,0x730a3b3366257801\n"
".quad 0x1b00003502003a74,0x7465720a3b34c000,0x00000a0a0a7d0a3b\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[271];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif