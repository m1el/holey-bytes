use mlua::prelude::*;

#[derive(Clone, Debug, Default, FromLua)]
pub struct ArrayList<T>(pub Vec<T>);
impl<T> LuaUserData for ArrayList<T>
where
    T: for<'lua> FromLua<'lua> + for<'lua> IntoLua<'lua> + Clone + std::fmt::Debug,
{
    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_meta_method(
            LuaMetaMethod::Index,
            |lua, this, index: LuaInteger| match this.0.get(
                (index as usize)
                    .checked_sub(1)
                    .ok_or_else(|| mlua::Error::runtime("Invalid index: 0"))?,
            ) {
                Some(i) => i.clone().into_lua(lua),
                None => Ok(LuaValue::Nil),
            },
        );

        methods.add_meta_method_mut(
            LuaMetaMethod::NewIndex,
            |_, this, (index, val): (LuaInteger, T)| match this.0.get_mut(
                (index as usize)
                    .checked_sub(1)
                    .ok_or_else(|| mlua::Error::runtime("Invalid index: 0"))?,
            ) {
                Some(x) => {
                    *x = val;
                    Ok(())
                }
                None => Err(mlua::Error::runtime(format!(
                    "Index out of bounds: length = {}, index = {index}",
                    this.0.len()
                ))),
            },
        );

        methods.add_meta_method(LuaMetaMethod::Len, |_, this, ()| Ok(this.0.len()));
        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, ()| {
            Ok(format!("{this:?}"))
        });
        methods.add_method_mut("push", |_, this, val: T| {
            this.0.push(val);
            Ok(())
        });
        methods.add_method_mut("pop", |lua, this, ()| match this.0.pop() {
            Some(val) => val.into_lua(lua),
            None => Ok(LuaValue::Nil),
        });
    }
}
