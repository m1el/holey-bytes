use {
    crate::SharedObject,
    rhai::{Engine, FuncRegistration, ImmutableString, Module},
};

macro_rules! shdm_fns {
    (
        module: $module:expr;
        shared: $shared:expr => $shname:ident;

        $(
            $vis:ident fn $name:ident($($param_name:ident: $param_ty:ty),*) $(-> $ret:ty)? $blk:block
        )*
    ) => {{
        let module = $module;
        let shared = $shared;
        $({

            let $shname = SharedObject::clone(&shared);

            FuncRegistration::new(stringify!($name))
                .with_namespace(rhai::FnNamespace::Global)
                .set_into_module::<_, { ["", $(stringify!($param_name)),*].len() - 1 }, false, _, true, _>(
                    module,
                    move |$($param_name: $param_ty),*| $(-> $ret)? {
                        let mut $shname = $shname.borrow_mut();
                        $blk
                    }
                );
        })*
    }};
}

#[derive(Clone, Copy, Debug)]
pub struct UnboundLabel(pub usize);

pub fn setup(engine: &mut Engine, module: &mut Module, object: SharedObject) {
    shdm_fns! {
        module: module;
        shared: object => obj;

        global fn label() {
            let symbol = obj.symbol(crate::object::Section::Text);
            Ok(symbol)
        }

        global fn label(label: ImmutableString) {
            let symbol = obj.symbol(crate::object::Section::Text);
            obj.labels.insert(label, symbol.0);

            Ok(symbol)
        }

        global fn declabel() {
            let index = obj.symbols.len();
            obj.symbols.push(None);

            Ok(UnboundLabel(index))
        }

        global fn declabel(label: ImmutableString) {
            let index = obj.symbols.len();
            obj.symbols.push(None);
            obj.labels.insert(label, index);

            Ok(UnboundLabel(index))
        }

        global fn here(label: UnboundLabel) {
            obj.symbols[label.0] = Some(crate::object::SymbolEntry {
                location: crate::object::Section::Text,
                offset:   obj.sections.text.len(),
            });

            Ok(())
        }
    }

    engine.register_type_with_name::<UnboundLabel>("UnboundLabel");
}
