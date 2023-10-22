mod arraylist;
mod ins;
mod label;
mod linker;

use {arraylist::ArrayList, mlua::{Result, prelude::*}, std::io::Read};

pub type Labels = ArrayList<Option<usize>>;

fn main() -> Result<()> {
    let mut code = vec![];
    std::io::stdin().read_to_end(&mut code)?;

    let lua = Lua::new();
    lua.scope(|scope| {
        // Global values
        let globals = lua.globals();
        globals.set(
            "_CODE",
            lua.create_table_from([
                ("text", ArrayList::<u8>::default()),
                ("data", ArrayList::<u8>::default()),
            ])?,
        )?;

        globals.set("_LABELS", Labels::default())?;
        globals.set("_SYM_TABLE", lua.create_table()?)?;
        globals.set("_SYM_REPLS", lua.create_table()?)?;

        // Functions
        globals.set("label", lua.create_function(label::label)?)?;
        ins::setup(&lua, scope)?;

        // Register symbols
        for n in 0..255 {
            globals.set(format!("r{n}"), n)?;
        }

        lua.load(code).exec()?;

        linker::link(
            globals.get("_SYM_REPLS")?,
            globals.get("_SYM_TABLE")?,
            &globals.get::<_, Labels>("_LABELS")?.0,
            &mut globals
                .get::<_, LuaTable>("_CODE")?
                .get::<_, LuaUserDataRefMut<ArrayList<u8>>>("text")?
                .0,
            &mut std::io::stdout(),
        )
    })?;

    Ok(())
}
