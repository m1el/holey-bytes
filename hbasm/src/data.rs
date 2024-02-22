//! Data section inserts

use {
    crate::{object::SymbolRef, SharedObject},
    rhai::{CustomType, Engine, FuncRegistration, ImmutableString, Module},
};

/// Generate insertions for data types
///
/// `gen_data_instructions!($module, $obj, [$type, …]);`
/// - `$module`: Rhai module
/// - `$obj`: Code object
/// - `$type`: Type of single array item
macro_rules! gen_data_insertions {
    ($module:expr, $obj:expr, [$($ty:ident),* $(,)?] $(,)?) => {{
        let (module, obj) = ($module, $obj);
        $({
            // Clone object to each function
            let obj = ::std::rc::Rc::clone(obj);

            FuncRegistration::new(stringify!($ty))
                .with_namespace(rhai::FnNamespace::Global)
                .set_into_module::<_, 1, false, _, true, _>(module, move |arr: ::rhai::Array| {
                    let obj    = &mut *obj.borrow_mut();
                    let symbol = obj.symbol($crate::object::Section::Data);

                    // Reserve space for object so we don't resize it
                    // all the time
                    obj.sections
                        .data
                        .reserve(arr.len() * ::std::mem::size_of::<$ty>());

                    // For every item…
                    for item in arr {
                        // … try do conversions from i32 to desired type
                        //   and insert it.
                        obj.sections.data.extend(
                            match item.as_int() {
                                Ok(num) => $ty::try_from(num).map_err(|_| "i64".to_owned()),
                                Err(ty) => Err(ty.to_owned()),
                            }
                            .map_err(|err| {

                                ::rhai::EvalAltResult::ErrorMismatchDataType(
                                    stringify!($ty).to_owned(),
                                    err,
                                    ::rhai::Position::NONE,
                                )
                            })?
                            .to_le_bytes(),
                        );
                    }

                    Ok(DataRef {
                        symbol,
                        len: obj.sections.data.len() - symbol.0,
                    })
                });
        })*
    }};
}

/// Reference to entry in data section
#[derive(Clone, Copy, Debug)]
pub struct DataRef {
    pub symbol: SymbolRef,
    pub len:    usize,
}

impl CustomType for DataRef {
    fn build(mut builder: rhai::TypeBuilder<Self>) {
        builder
            .with_name("DataRef")
            .with_get("symbol", |this: &mut Self| this.symbol)
            .with_get("len", |this: &mut Self| this.len as u64 as i64);
    }
}

pub fn module(engine: &mut Engine, obj: SharedObject) -> Module {
    let mut module = Module::new();

    gen_data_insertions!(&mut module, &obj, [i8, i16, i32, i64]);

    // Specialisation for strings, they should be
    // inserted as plain UTF-8 arrays
    FuncRegistration::new("str")
        .with_namespace(rhai::FnNamespace::Global)
        .set_into_module::<_, 1, false, _, true, _>(&mut module, move |s: ImmutableString| {
            let obj = &mut *obj.borrow_mut();
            let symbol = obj.symbol(crate::object::Section::Data);

            obj.sections.data.extend(s.as_bytes());
            Ok(DataRef {
                symbol,
                len: s.len(),
            })
        });

    engine.build_type::<DataRef>();
    module
}
