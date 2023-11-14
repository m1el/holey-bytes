use {
    crate::SharedObject,
    rhai::{Engine, ImmutableString, Module},
};

#[derive(Clone, Copy, Debug)]
pub struct UnboundLabel(pub usize);

pub fn setup(engine: &mut Engine, module: &mut Module, object: SharedObject) {
    {
        let object = SharedObject::clone(&object);
        let hash = module.set_native_fn("label", move || {
            let mut obj = object.borrow_mut();
            let symbol = obj.symbol(crate::object::Section::Text);
            Ok(symbol)
        });

        module.update_fn_namespace(hash, rhai::FnNamespace::Global);
    }

    {
        let object = SharedObject::clone(&object);
        let hash = module.set_native_fn("label", move |label: ImmutableString| {
            let mut obj = object.borrow_mut();
            let symbol = obj.symbol(crate::object::Section::Text);
            obj.labels.insert(label, symbol.0);

            Ok(symbol)
        });

        module.update_fn_namespace(hash, rhai::FnNamespace::Global);
    }

    {
        let object = SharedObject::clone(&object);
        let hash = module.set_native_fn("declabel", move || {
            let mut obj = object.borrow_mut();

            let index = obj.symbols.len();
            obj.symbols.push(None);

            Ok(UnboundLabel(index))
        });

        module.update_fn_namespace(hash, rhai::FnNamespace::Global);
    }

    {
        let object = SharedObject::clone(&object);
        let hash = module.set_native_fn("declabel", move |label: ImmutableString| {
            let mut obj = object.borrow_mut();

            let index = obj.symbols.len();
            obj.symbols.push(None);
            obj.labels.insert(label, index);

            Ok(UnboundLabel(index))
        });

        module.update_fn_namespace(hash, rhai::FnNamespace::Global);
    }

    {
        module.set_native_fn("here", move |label: UnboundLabel| {
            let mut obj = object.borrow_mut();
            obj.symbols[label.0] = Some(crate::object::SymbolEntry {
                location: crate::object::Section::Text,
                offset:   obj.sections.text.len(),
            });

            Ok(())
        });
    }

    engine.register_type_with_name::<UnboundLabel>("UnboundLabel");
}
