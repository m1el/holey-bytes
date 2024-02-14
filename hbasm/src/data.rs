use rhai::{CustomType, Engine, FuncRegistration, ImmutableString};

use {
    crate::{object::SymbolRef, SharedObject},
    rhai::Module,
};

macro_rules! gen_data_insertions {
    ($module:expr, $obj:expr, [$($ty:ident),* $(,)?] $(,)?) => {{
        let (module, obj) = ($module, $obj);
        $({
            let obj = ::std::rc::Rc::clone(obj);

            FuncRegistration::new(stringify!($ty))
                .with_namespace(rhai::FnNamespace::Global)
                .set_into_module::<_, 1, false, _, true, _>(module, move |arr: ::rhai::Array| {
                    let obj    = &mut *obj.borrow_mut();
                    let symbol = obj.symbol($crate::object::Section::Data);

                    obj.sections
                        .data
                        .reserve(arr.len() * ::std::mem::size_of::<$ty>());

                    for item in arr {
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
