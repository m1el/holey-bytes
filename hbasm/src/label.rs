//! Stuff related to labels

use {
    crate::SharedObject,
    rhai::{Engine, FuncRegistration, ImmutableString, Module},
};

/// Macro for creating functions for Rhai which
/// is bit more friendly
///
/// ```ignore
///     shdm_fns!{
///         module: $module;
///         shared: $shared => $shname;
///         
///         $vis fn $name($param_name: $param_ty, …) -> $ret { … }
///         …
///     }
/// ```
/// - `$module`: Rhai module
/// - `$shared`: Data to be shared across the functions
///     - `$shname`: The binding name inside functions
/// - `$vis`: Function visibility for Rhai
///     - Lowercased [`rhai::FnNamespace`] variants
/// - `$name`: Function name
/// - `$param_name`: Parameter name
/// - `$param_ty`: Rust parameter type
/// - `$ret`: Optional return type (otherwise infer)
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
        paste::paste! {
            $({

                let $shname = SharedObject::clone(&shared);
    
                FuncRegistration::new(stringify!($name))
                    .with_namespace(rhai::FnNamespace::[<$vis:camel>])
                    .set_into_module::<_, { ["", $(stringify!($param_name)),*].len() - 1 }, false, _, true, _>(
                        module,
                        move |$($param_name: $param_ty),*| $(-> $ret)? {
                            let mut $shname = $shname.borrow_mut();
                            $blk
                        }
                    );
            })*
        }
    }};
}

/// Label without any place bound
#[derive(Clone, Copy, Debug)]
pub struct UnboundLabel(pub usize);

pub fn setup(engine: &mut Engine, module: &mut Module, object: SharedObject) {
    shdm_fns! {
        module: module;
        shared: object => obj;

        // Insert unnamed label
        global fn label() {
            let symbol = obj.symbol(crate::object::Section::Text);
            Ok(symbol)
        }

        // Insert string-labeled label
        global fn label(label: ImmutableString) {
            let symbol = obj.symbol(crate::object::Section::Text);
            obj.labels.insert(label, symbol.0);

            Ok(symbol)
        }

        // Declare unbound label (to be bound later)
        global fn declabel() {
            let index = obj.symbols.len();
            obj.symbols.push(None);

            Ok(UnboundLabel(index))
        }

        // Declare unbound label (to be bound later)
        // with string label
        global fn declabel(label: ImmutableString) {
            let index = obj.symbols.len();
            obj.symbols.push(None);
            obj.labels.insert(label, index);

            Ok(UnboundLabel(index))
        }

        // Set location for unbound label
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
