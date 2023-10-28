mod data;
mod ins;
mod label;
mod linker;
mod object;

use {
    object::Object,
    rhai::{Engine, Module},
    std::{cell::RefCell, rc::Rc},
};

type SharedObject = Rc<RefCell<Object>>;

pub fn assembler(
    linkout: &mut impl std::io::Write,
    loader: impl FnOnce(&mut Engine) -> Result<(), Box<rhai::EvalAltResult>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = Engine::new();
    let mut module = Module::new();
    let obj = Rc::new(RefCell::new(Object::default()));
    ins::setup(&mut module, Rc::clone(&obj));
    label::setup(&mut engine, &mut module, Rc::clone(&obj));

    // Registers
    for n in 0_u8..255 {
        module.set_var(format!("r{n}"), n);
    }

    module.set_native_fn("reg", |n: i64| {
        Ok(u8::try_from(n).map_err(|_| {
            rhai::EvalAltResult::ErrorRuntime("Invalid register value".into(), rhai::Position::NONE)
        })?)
    });

    module.set_native_fn("as_i64", |n: u8| Ok(n as i64));

    let datamod = Rc::new(data::module(&mut engine, SharedObject::clone(&obj)));
    engine.register_global_module(Rc::new(module));
    engine.register_static_module("data", datamod);
    engine.register_type_with_name::<object::SymbolRef>("SymbolRef");
    loader(&mut engine)?;
    linker::link(obj, linkout)?;
    Ok(())
}
