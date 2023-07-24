/* HoleyBytes Bytecode representation in C
 * Requires C23 compiler or better
 */

#pragma once
#include <assert.h>
#include <limits.h>
#include <stdint.h>

static_assert(CHAR_BIT == 8, "Cursed architectures are not supported");

enum hbbc_Opcode: uint8_t {
    hbbc_Op_NOP   , hbbc_Op_ADD   , hbbc_Op_SUB  , hbbc_Op_MUL   , hbbc_Op_AND  , hbbc_Op_OR   ,
    hbbc_Op_XOR   , hbbc_Op_SL    , hbbc_Op_SR   , hbbc_Op_SRS   , hbbc_Op_CMP  , hbbc_Op_CMPU ,
    hbbc_Op_DIR   , hbbc_Op_NEG   , hbbc_Op_NOT  , hbbc_Op_ADDI  , hbbc_Op_MULI , hbbc_Op_ANDI ,
    hbbc_Op_ORI   , hbbc_Op_XORI  , hbbc_Op_SLI  , hbbc_Op_SRI   , hbbc_Op_SRSI , hbbc_Op_CMPI ,
    hbbc_Op_CMPUI , hbbc_Op_CP    , hbbc_Op_SWA  , hbbc_Op_LI    , hbbc_Op_LD   , hbbc_Op_ST   ,
    hbbc_Op_BMC   , hbbc_Op_BRC   , hbbc_Op_JMP  , hbbc_Op_JEQ   , hbbc_Op_JNE  , hbbc_Op_JLT  ,
    hbbc_Op_JGT   , hbbc_Op_JLTU  , hbbc_Op_JGTU , hbbc_Op_ECALL , hbbc_Op_ADDF , hbbc_Op_SUBF ,
    hbbc_Op_MULF  , hbbc_Op_DIRF  , hbbc_Op_FMAF , hbbc_Op_NEGF  , hbbc_Op_ITF  , hbbc_Op_FTI  ,
    hbbc_Op_ADDFI , hbbc_Op_MULFI ,
} typedef hbbc_Opcode;

static_assert(sizeof(hbbc_Opcode) == 1);

#pragma pack(push, 1)
struct hbbc_ParamBBBB
    { uint8_t _0; uint8_t _1; uint8_t _2; uint8_t _3; }
    typedef hbbc_ParamBBBB;
    static_assert(sizeof(hbbc_ParamBBBB) == 32 / 8);

struct hbbc_ParamBBB
    { uint8_t _0; uint8_t _1; uint8_t _2; }
    typedef hbbc_ParamBBB;
    static_assert(sizeof(hbbc_ParamBBB) == 24 / 8);

struct hbbc_ParamBBDH
    { uint8_t _0; uint8_t _1; uint64_t _2; uint16_t _3; }
    typedef hbbc_ParamBBDH;
    static_assert(sizeof(hbbc_ParamBBDH) == 96 / 8);

struct hbbc_ParamBBD
    { uint8_t _0; uint8_t _1; uint64_t _2; }
    typedef hbbc_ParamBBD;
    static_assert(sizeof(hbbc_ParamBBD) == 80 / 8);

struct hbbc_ParamBBW
    { uint8_t _0; uint8_t _1; uint32_t _2; }
    typedef hbbc_ParamBBW;
    static_assert(sizeof(hbbc_ParamBBW) == 48 / 8);

struct hbbc_ParamBB
    { uint8_t _0; uint8_t _1; }
    typedef hbbc_ParamBB;
    static_assert(sizeof(hbbc_ParamBB) == 16 / 8);

struct hbbc_ParamBD
    { uint8_t _0; uint64_t _1; }
    typedef hbbc_ParamBD;
    static_assert(sizeof(hbbc_ParamBD) == 72 / 8);

typedef uint64_t hbbc_ParamD;
    static_assert(sizeof(hbbc_ParamD) == 64 / 8);

#pragma pack(pop)
