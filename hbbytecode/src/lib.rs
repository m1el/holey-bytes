#![no_std]

#[cfg(feature = "disasm")]
extern crate alloc;

pub use crate::instrs::*;
use core::convert::TryFrom;

mod instrs;

type OpR = u8;

type OpA = u64;
type OpO = i32;
type OpP = i16;

type OpB = u8;
type OpH = u16;
type OpW = u32;
type OpD = u64;

/// # Safety
/// Has to be valid to be decoded from bytecode.
pub unsafe trait BytecodeItem {}
unsafe impl BytecodeItem for u8 {}

impl TryFrom<u8> for Instr {
    type Error = u8;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        #[cold]
        fn failed(value: u8) -> Result<Instr, u8> {
            Err(value)
        }

        if value < COUNT {
            unsafe { Ok(core::mem::transmute::<u8, Instr>(value)) }
        } else {
            failed(value)
        }
    }
}

#[inline]
unsafe fn encode<T>(instr: T) -> (usize, [u8; instrs::MAX_SIZE]) {
    let mut buf = [0; instrs::MAX_SIZE];
    core::ptr::write(buf.as_mut_ptr() as *mut T, instr);
    (core::mem::size_of::<T>(), buf)
}

#[inline]
#[cfg(feature = "disasm")]
fn decode<T>(binary: &mut &[u8]) -> Option<T> {
    let (front, rest) = core::mem::take(binary).split_at_checked(core::mem::size_of::<T>())?;
    *binary = rest;
    unsafe { Some(core::ptr::read(front.as_ptr() as *const T)) }
}

/// Rounding mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    NearestEven = 0,
    Truncate = 1,
    Up = 2,
    Down = 3,
}

impl TryFrom<u8> for RoundingMode {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        (value <= 3).then(|| unsafe { core::mem::transmute(value) }).ok_or(())
    }
}

#[cfg(feature = "disasm")]
#[derive(Clone, Copy)]
pub enum DisasmItem {
    Func,
    Global,
}

#[cfg(feature = "disasm")]
#[derive(Debug)]
pub enum DisasmError<'a> {
    InvalidInstruction(u8),
    InstructionOutOfBounds(&'a str),
    FmtFailed(core::fmt::Error),
    HasOutOfBoundsJumps,
    HasDirectInstructionCycles,
}

#[cfg(feature = "disasm")]
impl From<core::fmt::Error> for DisasmError<'_> {
    fn from(value: core::fmt::Error) -> Self {
        Self::FmtFailed(value)
    }
}

#[cfg(feature = "disasm")]
impl core::fmt::Display for DisasmError<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            DisasmError::InvalidInstruction(b) => write!(f, "invalid instruction opcode: {b}"),
            DisasmError::InstructionOutOfBounds(name) => {
                write!(f, "instruction would go out of bounds of {name} symbol")
            }
            DisasmError::FmtFailed(error) => write!(f, "fmt failed: {error}"),
            DisasmError::HasOutOfBoundsJumps => write!(
                f,
                "the code contained jumps that dont got neither to a \
                valid symbol or local insturction"
            ),
            DisasmError::HasDirectInstructionCycles => {
                writeln!(f, "found instruction that jumps to itself")
            }
        }
    }
}

#[cfg(feature = "disasm")]
impl core::error::Error for DisasmError<'_> {}

#[cfg(feature = "disasm")]
pub fn disasm<'a>(
    binary: &mut &[u8],
    functions: &alloc::collections::BTreeMap<u32, (&'a str, u32, DisasmItem)>,
    out: &mut alloc::string::String,
    mut eca_handler: impl FnMut(&mut &[u8]),
) -> Result<(), DisasmError<'a>> {
    use {
        self::instrs::Instr,
        alloc::{
            collections::btree_map::{BTreeMap, Entry},
            vec::Vec,
        },
        core::{convert::TryInto, fmt::Write},
    };

    fn instr_from_byte(b: u8) -> Result<Instr, DisasmError<'static>> {
        b.try_into().map_err(DisasmError::InvalidInstruction)
    }

    let mut labels = BTreeMap::<u32, u32>::default();
    let mut buf = Vec::<instrs::Oper>::new();
    let mut has_cycle = false;
    let mut has_oob = false;

    '_offset_pass: for (&off, &(name, len, kind)) in functions.iter() {
        if matches!(kind, DisasmItem::Global) {
            continue;
        }

        let prev = *binary;

        *binary = &binary[off as usize..];

        let mut label_count = 0;
        while let Some(&byte) = binary.first() {
            let offset: i32 = (prev.len() - binary.len()).try_into().unwrap();
            if offset as u32 == off + len {
                break;
            }
            let Ok(inst) = instr_from_byte(byte) else { break };
            instrs::parse_args(binary, inst, &mut buf)
                .ok_or(DisasmError::InstructionOutOfBounds(name))?;

            for op in buf.drain(..) {
                let rel = match op {
                    instrs::Oper::O(rel) => rel,
                    instrs::Oper::P(rel) => rel.into(),
                    _ => continue,
                };

                has_cycle |= rel == 0;

                let global_offset: u32 = (offset + rel).try_into().unwrap();
                if functions.get(&global_offset).is_some() {
                    continue;
                }
                label_count += match labels.entry(global_offset) {
                    Entry::Occupied(_) => 0,
                    Entry::Vacant(entry) => {
                        entry.insert(label_count);
                        1
                    }
                }
            }

            if matches!(inst, Instr::ECA) {
                eca_handler(binary);
            }
        }

        *binary = prev;
    }

    let mut ordered = functions.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|(_, (name, _, _))| name);

    '_dump: for (&off, &(name, len, kind)) in ordered {
        if matches!(kind, DisasmItem::Global) {
            continue;
        }
        let prev = *binary;

        writeln!(out, "{name}:")?;

        *binary = &binary[off as usize..];
        while let Some(&byte) = binary.first() {
            let offset: i32 = (prev.len() - binary.len()).try_into().unwrap();
            if offset as u32 == off + len {
                break;
            }
            let Ok(inst) = instr_from_byte(byte) else {
                writeln!(out, "invalid instr {byte}")?;
                break;
            };
            instrs::parse_args(binary, inst, &mut buf).unwrap();

            if let Some(label) = labels.get(&offset.try_into().unwrap()) {
                write!(out, "{:>2}: ", label)?;
            } else {
                write!(out, "    ")?;
            }

            write!(out, "{inst:<8?} ")?;

            'a: for (i, op) in buf.drain(..).enumerate() {
                if i != 0 {
                    write!(out, ", ")?;
                }

                let rel = 'b: {
                    match op {
                        instrs::Oper::O(rel) => break 'b rel,
                        instrs::Oper::P(rel) => break 'b rel.into(),
                        instrs::Oper::R(r) => write!(out, "r{r}")?,
                        instrs::Oper::B(b) => write!(out, "{b}b")?,
                        instrs::Oper::H(h) => write!(out, "{h}h")?,
                        instrs::Oper::W(w) => write!(out, "{w}w")?,
                        instrs::Oper::D(d) if (d as i64) < 0 => write!(out, "{}d", d as i64)?,
                        instrs::Oper::D(d) => write!(out, "{d}d")?,
                        instrs::Oper::A(a) => write!(out, "{a}a")?,
                    }

                    continue 'a;
                };

                let global_offset: u32 = (offset + rel).try_into().unwrap();
                if let Some(&(name, ..)) = functions.get(&global_offset) {
                    if name.contains('\0') {
                        write!(out, ":{name:?}")?;
                    } else {
                        write!(out, ":{name}")?;
                    }
                } else {
                    let local_has_oob = global_offset < off
                        || global_offset > off + len
                        || prev
                            .get(global_offset as usize)
                            .map_or(true, |&b| instr_from_byte(b).is_err())
                        || prev[global_offset as usize] == 0;
                    has_oob |= local_has_oob;
                    let label = labels.get(&global_offset).unwrap();
                    if local_has_oob {
                        write!(out, "!!!!!!!!!{rel}")?;
                    } else {
                        write!(out, ":{label}")?;
                    }
                }
            }

            writeln!(out)?;

            if matches!(inst, Instr::ECA) {
                eca_handler(binary);
            }
        }

        *binary = prev;
    }

    if has_oob {
        return Err(DisasmError::HasOutOfBoundsJumps);
    }

    if has_cycle {
        return Err(DisasmError::HasDirectInstructionCycles);
    }

    Ok(())
}
