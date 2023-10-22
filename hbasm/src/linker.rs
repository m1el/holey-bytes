use mlua::prelude::*;

pub fn link(
    symrt: LuaTable,
    symtab: LuaTable,
    labels: &[Option<usize>],
    code: &mut [u8],
    out: &mut impl std::io::Write,
) -> mlua::Result<()> {
    for item in symrt.pairs::<usize, LuaTable>() {
        let (loc, val) = item?;
        let size: usize = val.get("size")?;
        let dest = labels
            .get(
                match val.get::<_, LuaValue>("label")? {
                    LuaValue::Integer(i) => i,
                    LuaValue::String(s) => symtab.get(s)?,
                    _ => {
                        return Err(mlua::Error::runtime(
                            "Invalid symbol type (int or string expected)",
                        ))
                    }
                } as usize
                    - 1,
            )
            .copied()
            .flatten()
            .ok_or_else(|| mlua::Error::runtime("Invalid label"))?;

        let loc = loc - 1;
        let dest = dest - 1;
        
        let offset = dest.wrapping_sub(loc);
        match size {
            4 => code[loc..loc + size].copy_from_slice(&(offset as u32).to_le_bytes()),
            2 => code[loc..loc + size].copy_from_slice(&(offset as u16).to_le_bytes()),
            _ => return Err(mlua::Error::runtime("Invalid symbol")),
        }
    }

    dbg!(&code);
    out.write_all(code)?;
    Ok(())
}
