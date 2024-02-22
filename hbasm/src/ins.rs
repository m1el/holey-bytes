//! Functions for inserting instructions
//! 
//! Most of the code you see is just metaprogramming stuff.
//! This ensures that adding new instructions won't need any
//! specific changes and consistent behaviour.
//!
//! > I tried to comment stuff here, but I meanwhile forgor how it works.
//!
//! — Erin

use {
    crate::object::Object,
    rhai::{FuncRegistration, Module},
    std::{cell::RefCell, rc::Rc},
};

/// Operand types and their insertions
pub mod optypes {
    use {
        crate::{
            label::UnboundLabel,
            object::{Object, RelocKey, RelocType, SymbolRef},
        },
        rhai::{Dynamic, EvalAltResult, ImmutableString, Position},
    };

    // These types represent operand types to be inserted
    pub type R = u8;
    pub type B = i8;
    pub type H = i16;
    pub type W = i32;
    pub type D = i64;

    pub type A = Dynamic;
    pub type O = Dynamic;
    pub type P = Dynamic;

    /// Insert relocation into code
    ///
    /// - If integer, just write it to the code
    /// - Otherwise insert entry into relocation table
    ///   and fill zeroes
    pub fn insert_reloc(
        obj: &mut Object,
        ty: RelocType,
        val: &Dynamic,
    ) -> Result<(), EvalAltResult> {
        match () {
            // Direct references – insert directly to table
            _ if val.is::<SymbolRef>() => {
                obj.relocation(RelocKey::Symbol(val.clone_cast::<SymbolRef>().0), ty)
            }
            _ if val.is::<UnboundLabel>() => {
                obj.relocation(RelocKey::Symbol(val.clone_cast::<UnboundLabel>().0), ty)
            }
            _ if val.is::<DataRef>() => {
                obj.relocation(RelocKey::Symbol(val.clone_cast::<DataRef>().symbol.0), ty)
            }

            // String (indirect) reference
            _ if val.is_string() => {
                obj.relocation(RelocKey::Label(val.clone_cast::<ImmutableString>()), ty)
            }

            // Manual offset
            _ if val.is_int() => {
                let int = val.clone_cast::<i64>();
                match ty {
                    RelocType::Rel32 => obj.sections.text.extend((int as i32).to_le_bytes()),
                    RelocType::Rel16 => obj.sections.text.extend((int as i16).to_le_bytes()),
                    RelocType::Abs64 => obj.sections.text.extend(int.to_le_bytes()),
                }
            }

            _ => {
                return Err(EvalAltResult::ErrorMismatchDataType(
                    "SymbolRef, UnboundLabel, String or Int".to_owned(),
                    val.type_name().to_owned(),
                    Position::NONE,
                ))
            }
        }

        Ok(())
    }

    /// Generate macro for inserting item into the output object
    ///
    /// Pre-defines inserts for absolute address and relative offsets.
    /// These are inserted with function [`insert_reloc`]
    /// # le_bytes
    /// `gen_insert!(le_bytes: [B, …]);`
    ///
    /// Takes sequence of operand types which should be inserted
    /// by invoking `to_le_bytes` method on it.
    macro_rules! gen_insert {
        (le_bytes: [$($lety:ident),* $(,)?]) => {
            /// `insert!($thing, $obj, $type)` where
            /// - `$thing`: Value you want to insert
            /// - `$obj`: Code object
            /// - `$type`: Type of inserted value
            ///
            /// Eg. `insert!(69_u8, obj, B);`
            macro_rules! insert {
                $(($thing:expr, $obj: expr, $lety) => {
                    $obj.sections.text.extend($thing.to_le_bytes());
                };)*

                ($thing:expr, $obj:expr, A) => {
                    $crate::ins::optypes::insert_reloc(
                        $obj,
                        $crate::object::RelocType::Abs64,
                        $thing
                    )?
                };
                ($thing:expr, $obj:expr, O) => {
                    $crate::ins::optypes::insert_reloc(
                        $obj,
                        $crate::object::RelocType::Rel32,
                        $thing
                    )?
                };
                ($thing:expr, $obj:expr, P) => {
                    $crate::ins::optypes::insert_reloc(
                        $obj,
                        $crate::object::RelocType::Rel16,
                        $thing
                    )?
                };
            }
        };
    }

    gen_insert!(le_bytes: [R, B, H, W, D]);

    #[allow(clippy::single_component_path_imports)]
    pub(super) use insert;

    use crate::data::DataRef;
}

/// Rhai Types (types for function parameters as Rhai uses only 64bit signed integers)
pub mod rity {
    pub use super::optypes::{A, O, P, R};
    pub type B = i64;
    pub type H = i64;
    pub type W = i64;
    pub type D = i64;
}

/// Generic instruction (instruction of certain operands type) inserts
pub mod generic {
    use {crate::object::Object, rhai::EvalAltResult};

    pub(super) fn convert_op<A, B>(from: A) -> Result<B, EvalAltResult>
    where
        B: TryFrom<A>,
        <B as TryFrom<A>>::Error: std::error::Error + Sync + Send + 'static,
    {
        B::try_from(from).map_err(|e| {
            EvalAltResult::ErrorSystem("Data conversion error".to_owned(), Box::new(e))
        })
    }

    /// Generate opcode-generic instruction insert macro
    macro_rules! gen_ins {
        ($($($name:ident : $ty:ty),*;)*) => {
            paste::paste! {
                $(
                    /// Instruction-generic opcode insertion function
                    /// - `obj`: Code object
                    /// - `opcode`: opcode, not checked if valid for instruction type
                    /// - … for operands
                    #[inline]
                    pub fn [<$($ty:lower)*>](
                        obj: &mut Object,
                        opcode: u8,
                        $($name: $crate::ins::optypes::$ty),*,
                    ) -> Result<(), EvalAltResult> {
                        // Push opcode
                        obj.sections.text.push(opcode);

                        // Insert based on type
                        $($crate::ins::optypes::insert!(&$name, obj, $ty);)*
                        Ok(())
                    }
                )*

                /// Generate Rhai opcode-specific instruction insertion functions
                ///
                /// `gen_ins_fn!($obj, $opcode, $optype);` where:
                /// - `$obj`: Code object
                /// - `$opcode`: Opcode value
                macro_rules! gen_ins_fn {
                    $(
                        ($obj:expr, $opcode:expr, [<$($ty)*>]) => {
                            // Opcode-specific insertion function
                            // - Parameters = operands
                            move |$($name: $crate::ins::rity::$ty),*| {
                                // Invoke generic function
                                $crate::ins::generic::[<$($ty:lower)*>](
                                    &mut *$obj.borrow_mut(),
                                    $opcode,
                                    $(
                                        // Convert to desired type (from Rhai-provided values)
                                        $crate::ins::generic::convert_op::<     
                                            _,
                                            $crate::ins::optypes::$ty
                                        >($name)?
                                    ),*
                                )?;
                                Ok(())
                            }
                        };

                        // Internal-use: count args
                        (@arg_count [<$($ty)*>]) => {
                            { ["", $(stringify!($ty)),*].len() - 1 }
                        };
                    )*

                    // Specialisation for no-operand instructions
                    ($obj:expr, $opcode:expr, N) => {
                        move || {
                            $crate::ins::generic::n(&mut *$obj.borrow_mut(), $opcode);
                            Ok(())
                        }
                    };

                    // Internal-use specialisation: no-operand instructions
                    (@arg_count N) => {
                        { 0 }
                    };
                }
            }
        };
    }

    /// Specialisation for no-operand instructions – simply just push opcode
    #[inline]
    pub fn n(obj: &mut Object, opcode: u8) {
        obj.sections.text.push(opcode);
    }

    // Generate opcode-generic instruction inserters
    // (operand identifiers are arbitrary)
    //
    // New instruction types have to be added manually here
    gen_ins! {
        o0: R, o1: R;
        o0: R, o1: R, o2: R;
        o0: R, o1: R, o2: R, o3: R;
        o0: R, o1: R, o2: B;
        o0: R, o1: R, o2: H;
        o0: R, o1: R, o2: W;
        o0: R, o1: R, o2: D;
        o0: R, o1: B;
        o0: R, o1: H;
        o0: R, o1: W;
        o0: R, o1: D;
        o0: R, o1: R, o2: A;
        o0: R, o1: R, o2: A, o3: H;
        o0: R, o1: R, o2: O, o3: H;
        o0: R, o1: R, o2: P, o3: H;
        o0: R, o1: R, o2: O;
        o0: R, o1: R, o2: P;
        o0: O;
        o0: P;
    }

    #[allow(clippy::single_component_path_imports)]
    pub(super) use gen_ins_fn;
}

/// Generate instructions from instruction table
///
/// ```ignore
/// instructions!(($module, $obj) {
///     // Data from instruction table
///     $opcode, $mnemonic, $opty, $doc;
///     …
/// });
/// ```
/// - `$module`: Rhai module
/// - `$obj`: Code object
macro_rules! instructions {
    (
        ($module:expr, $obj:expr $(,)?)
        { $($opcode:expr, $mnemonic:ident, $ops:tt, $doc:literal;)* }
    ) => {{
        let (module, obj) = ($module, $obj);
        $({
            // Object is shared across all functions
            let obj = Rc::clone(&obj);

            // Register newly generated function for each instruction
            FuncRegistration::new(stringify!([<$mnemonic:lower>]))
                .with_namespace(rhai::FnNamespace::Global)
                .set_into_module::<_, { generic::gen_ins_fn!(@arg_count $ops) }, false, _, true, _>(
                    module,
                    generic::gen_ins_fn!(
                        obj,
                        $opcode,
                        $ops
                    )
                );
        })*
    }};
}

/// Setup instruction insertors
pub fn setup(module: &mut Module, obj: Rc<RefCell<Object>>) {
    // Import instructions table and use it for generation
    with_builtin_macros::with_builtin! {
        let $spec = include_from_root!("../hbbytecode/instructions.in") in {
            instructions!((module, obj) { $spec });
        }
    }
}
