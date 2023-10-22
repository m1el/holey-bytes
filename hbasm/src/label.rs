use {
    crate::{arraylist::ArrayList, Labels},
    mlua::{Result, prelude::*},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, FromLua)]
pub struct UnassignedLabel(pub usize);
impl LuaUserData for UnassignedLabel {
    fn add_fields<'lua, F: LuaUserDataFields<'lua, Self>>(fields: &mut F) {
        fields.add_field_method_get("id", |_, this| Ok(this.0));
    }

    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_method("here", |lua, this, ()| {
            match lua
                .globals()
                .get::<_, LuaUserDataRefMut<Labels>>("_LABELS")?
                .0
                .get_mut(
                    this.0
                        .checked_sub(1)
                        .ok_or_else(|| mlua::Error::runtime("Invalid label"))?,
                ) {
                Some(entry) => {
                    *entry = Some(
                        lua.globals()
                            .get::<_, LuaTable>("_CODE")?
                            .get::<_, LuaUserDataRef<ArrayList<u8>>>("text")?
                            .0
                            .len(),
                    );

                    Ok(())
                }
                None => Err(mlua::Error::runtime("Invalid label")),
            }
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, FromLua)]
pub struct Label(pub usize);
impl LuaUserData for Label {
    fn add_fields<'lua, F: LuaUserDataFields<'lua, Self>>(fields: &mut F) {
        fields.add_field_method_get("id", |_, this| Ok(this.0));
    }
}

pub fn label<'lua>(lua: &'lua Lua, val: LuaValue<'lua>) -> Result<LuaValue<'lua>> {
    let globals = lua.globals();
    let mut labels = globals.get::<_, LuaUserDataRefMut<Labels>>("_LABELS")?;

    let code_ix = globals
        .get::<_, LuaTable>("_CODE")?
        .get::<_, LuaUserDataRefMut<ArrayList<u8>>>("text")?
        .0
        .len();

    match val {
        LuaValue::Table(_) => {
            labels.0.push(None);
            Ok(LuaValue::UserData(
                lua.create_userdata(UnassignedLabel(labels.0.len()))?,
            ))
        }
        LuaValue::String(str) => {
            labels.0.push(Some(code_ix + 1));
            globals
                .get::<_, LuaTable>("_SYM_TABLE")?
                .set(str, labels.0.len())?;

            Ok(LuaValue::UserData(
                lua.create_userdata(Label(labels.0.len()))?,
            ))
        }
        LuaNil => {
            labels.0.push(Some(code_ix + 1));
            Ok(LuaValue::UserData(
                lua.create_userdata(Label(labels.0.len()))?,
            ))
        }
        _ => Err(mlua::Error::BadArgument {
            to:    Some("label".into()),
            pos:   1,
            name:  None,
            cause: std::sync::Arc::new(mlua::Error::runtime(
                "Unsupported type (nil and string are only supported)",
            )),
        }),
    }
}
