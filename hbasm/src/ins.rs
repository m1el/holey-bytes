use {
    crate::object::Object,
    rhai::{FuncRegistration, Module},
    std::{cell::RefCell, rc::Rc},
};

pub mod optypes {
    use {
        crate::{
            label::UnboundLabel,
            object::{Object, RelocKey, RelocType, SymbolRef},
        },
        rhai::{Dynamic, EvalAltResult, ImmutableString, Position},
    };

    pub type R = u8;
    pub type B = i8;
    pub type H = i16;
    pub type W = i32;
    pub type D = i64;

    pub type A = Dynamic;
    pub type O = Dynamic;
    pub type P = Dynamic;

    pub fn insert_reloc(
        obj: &mut Object,
        ty: RelocType,
        val: &Dynamic,
    ) -> Result<(), EvalAltResult> {
        match () {
            _ if val.is::<SymbolRef>() => {
                obj.relocation(RelocKey::Symbol(val.clone_cast::<SymbolRef>().0), ty)
            }
            _ if val.is::<UnboundLabel>() => {
                obj.relocation(RelocKey::Symbol(val.clone_cast::<UnboundLabel>().0), ty)
            }
            _ if val.is::<DataRef>() => {
                obj.relocation(RelocKey::Symbol(val.clone_cast::<DataRef>().symbol.0), ty)
            }
            _ if val.is_string() => {
                obj.relocation(RelocKey::Label(val.clone_cast::<ImmutableString>()), ty)
            }
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
                    "SybolRef, UnboundLabel, String or Int".to_owned(),
                    val.type_name().to_owned(),
                    Position::NONE,
                ))
            }
        }

        Ok(())
    }

    macro_rules! gen_insert {
        (le_bytes: [$($lety:ident),* $(,)?]) => {
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

pub mod rity {
    pub use super::optypes::{A, O, P, R};
    pub type B = i64;
    pub type H = i64;
    pub type W = i64;
    pub type D = i64;
}

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

    macro_rules! gen_ins {
        ($($($name:ident : $ty:ty),*;)*) => {
            paste::paste! {
                $(#[inline]
                pub fn [<$($ty:lower)*>](
                    obj: &mut Object,
                    opcode: u8,
                    $($name: $crate::ins::optypes::$ty),*,
                ) -> Result<(), EvalAltResult> {
                    obj.sections.text.push(opcode);
                    $($crate::ins::optypes::insert!(&$name, obj, $ty);)*
                    Ok(())
                })*

                macro_rules! gen_ins_fn {
                    $(
                        ($obj:expr, $opcode:expr, [<$($ty)*>]) => {
                            move |$($name: $crate::ins::rity::$ty),*| {
                                $crate::ins::generic::[<$($ty:lower)*>](
                                    &mut *$obj.borrow_mut(),
                                    $opcode,
                                    $(
                                        $crate::ins::generic::convert_op::<
                                            _,
                                            $crate::ins::optypes::$ty
                                        >($name)?
                                    ),*
                                )?;
                                Ok(())
                            }
                        };

                        (@arg_count [<$($ty)*>]) => {
                            { ["", $(stringify!($ty)),*].len() - 1 }
                        };
                    )*

                    ($obj:expr, $opcode:expr, N) => {
                        move || {
                            $crate::ins::generic::n(&mut *$obj.borrow_mut(), $opcode);
                            Ok(())
                        }
                    };

                    (@arg_count N) => {
                        { 0 }
                    };
                }
            }
        };
    }

    #[inline]
    pub fn n(obj: &mut Object, opcode: u8) {
        obj.sections.text.push(opcode);
    }

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

macro_rules! instructions {
    (
        ($module:expr, $obj:expr $(,)?)
        { $($opcode:expr, $mnemonic:ident, $ops:tt, $doc:literal;)* }
    ) => {{
        let (module, obj) = ($module, $obj);
        $({
            let obj = Rc::clone(&obj);
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

pub fn setup(module: &mut Module, obj: Rc<RefCell<Object>>) {
    with_builtin_macros::with_builtin! {
        let $spec = include_from_root!("../hbbytecode/instructions.in") in {
            instructions!((module, obj) { $spec });
        }
    }
}
