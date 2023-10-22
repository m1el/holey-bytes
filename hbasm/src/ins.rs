use {
    crate::arraylist::ArrayList,
    mlua::{prelude::*, Result, Scope},
};

mod opsty {
    pub type R = i8;
    pub type B = i8;
    pub type H = i16;
    pub type W = i32;
    pub type D = i64;
    pub type A = i64;
    pub type O<'lua> = super::LuaValue<'lua>;
    pub type P<'lua> = super::LuaValue<'lua>;
}

macro_rules! gen_insert {
    ($($plain:ident),* $(,)?) => {
        macro_rules! insert {
            $(
                ($label:expr, $lua:expr, $symrt:expr, $code:expr, $plain) => {
                    $code.0.extend($label.to_le_bytes());
                };
            )*

            ($label:expr, $lua:expr, $symrt:expr, $code:expr, O) => {
                insert_label_ref::<4>(
                    $label,
                    $lua,
                    &$symrt,
                    &mut $code.0,
                )?;
            };

            ($label:expr, $lua:expr, $symrt:expr, $code:expr, P) => {
                insert_label_ref::<2>(
                    $label,
                    $lua,
                    &$symrt,
                    &mut $code.0,
                )?;
            };
        }
    };
}

gen_insert!(R, B, H, W, D, A);

macro_rules! generic_ins {
    {
        ($lua:expr, $scope:expr);
        $($name:ident($($param_i:ident : $param_ty:ident),*);)*
    } => {{
        let lua   = $lua;
        let scope = $scope;

        let code = $lua.globals()
            .get::<_, LuaTable>("_CODE")?
            .get::<_, LuaAnyUserData>("text")?;

        let symrt = $lua.globals()
            .get::<_, LuaTable>("_SYM_REPLS")?;

        lua.globals().set(
            "_GENERIC",
            lua.create_table_from([$((
                    stringify!($name),
                    #[allow(unused)]
                    {
                        let code = code.clone();
                        let symrt = symrt.clone();
                        scope.create_function_mut(move |lua, (opcode, $($param_i),*): (u8, $(opsty::$param_ty),*)| {
                            let mut code = code.borrow_mut::<ArrayList<u8>>()?;
                            code.0.push(opcode);
                            $(insert!($param_i, lua, symrt, code, $param_ty);)*
                            Ok(())
                        })?
                    }
            )),*])?
        )?;
    }};
}

macro_rules! ins {
    {
        $lua:expr;
        {$($opcode:expr, $mnemonic:ident, $ty:ident, $_doc:literal;)*}
    } => {{
        use std::fmt::Write;

        let lua = $lua;
        let mut code = String::new();

        $({
            paste::paste! {
                let name = match stringify!([<$mnemonic:lower>]) {
                    "and" => "and_",
                    "or"  => "or_",
                    "not" => "not_",
                    name  => name,
                };
            }

            writeln!(
                code,
                "function {name}(...) _GENERIC.{ty}({opcode}, ...) end",
                ty     = stringify!($ty).to_lowercase(),
                opcode = $opcode,
            ).unwrap();

        })*

        lua.load(code).exec()?;
    }};
}

pub fn setup<'lua, 'scope>(lua: &'lua Lua, scope: &Scope<'lua, 'scope>) -> Result<()>
where
    'lua: 'scope,
{
    generic_ins! {
        (lua, scope);
        rr  (o0: R, o1: R);
        rrr (o0: R, o1: R, o2: R);
        rrrr(o0: R, o1: R, o2: R, o3: R);
        rrb (o0: R, o1: R, o2: B);
        rrh (o0: R, o1: R, o2: H);
        rrw (o0: R, o1: R, o2: W);
        rrd (o0: R, o1: R, o2: D);
        rb  (o0: R, o1: B);
        rh  (o0: R, o1: H);
        rw  (o0: R, o1: W);
        rd  (o0: R, o1: D);
        rrah(o0: R, o1: R, o2: A, o3: H);
        rroh(o0: R, o1: R, o2: O, o3: H);
        rrph(o0: R, o1: R, o2: P, o3: H);
        rro (o0: R, o1: R, o2: O);
        rrp (o0: R, o1: R, r2: P);
        o   (o0: O);
        p   (o0: P);
        n   ();
    }

    with_builtin_macros::with_builtin! {
        let $spec = include_from_root!("../hbbytecode/instructions.in") in {
            ins!(lua; { $spec });
        }
    }

    Ok(())
}

fn insert_label_ref<const SIZE: usize>(
    label: LuaValue,
    lua: &Lua,
    symrt: &LuaTable,
    code: &mut Vec<u8>,
) -> Result<()> {
    match label {
        LuaValue::Integer(offset) => {
            if match SIZE {
                2 => i16::try_from(offset).map(|o| code.extend(o.to_le_bytes())),
                4 => i32::try_from(offset).map(|o| code.extend(o.to_le_bytes())),
                s => {
                    return Err(mlua::Error::runtime(format!(
                        "Invalid offset size (expected 2 or 4 bytes, got {s})"
                    )));
                }
            }
            .is_err()
            {
                return Err(mlua::Error::runtime("Failed to cast offset"));
            }
            return Ok(());
        }
        LuaValue::UserData(ud) => {
            symrt.set(
                code.len() + 1,
                lua.create_table_from([("label", ud.get("id")?), ("size", SIZE)])?,
            )?;
            code.extend([0; SIZE]);
        }
        LuaValue::String(_) => {
            symrt.set(
                code.len() + 1,
                lua.create_table_from([("label", label), ("size", SIZE.into_lua(lua)?)])?,
            )?;
            code.extend([0; SIZE]);
        }
        _ => return Err(mlua::Error::runtime("Invalid label type")),
    }

    Ok(())
}
