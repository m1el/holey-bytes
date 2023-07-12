macro_rules! impl_asm_opcodes {
    (
        $generic:ident
        ($($param_i:ident: $param_ty:ty),*)
        => []
    ) => {};

    (
        $generic:ident
        ($($param_i:ident: $param_ty:ty),*)
        => [$opcode:ident, $($rest:tt)*]
    ) => {
        paste::paste! {
            #[allow(dead_code)]
            #[inline(always)]
            pub fn [<i_ $opcode:lower>](&mut self, $($param_i: $param_ty),*) {
                self.$generic(hbbytecode::opcode::$opcode, $($param_i),*)
            }
        }

        macros::asm::impl_asm_opcodes!(
            $generic($($param_i: $param_ty),*)
            => [$($rest)*]
        );
    };
}

macro_rules! impl_asm_insert {
    ($self:expr, $id:ident, I) => {
        Imm::insert(&$id, $self)
    };

    ($self:expr, $id:ident, $_:ident) => {
        $self.buf.extend($id.to_le_bytes())
    };
}

macro_rules! impl_asm {
    (
        $(
            $ityn:ident
            ($($param_i:ident: $param_ty:ident),* $(,)?)
            => [$($opcode:ident),* $(,)?],
        )*
    ) => {
        paste::paste! {
            $(
                #[allow(dead_code)]
                fn [<i_param_ $ityn>](&mut self, opcode: u8, $($param_i: macros::asm::ident_map_ty!($param_ty)),*) {
                    self.buf.push(opcode);
                    $(macros::asm::impl_asm_insert!(self, $param_i, $param_ty);)*
                }

                macros::asm::impl_asm_opcodes!(
                    [<i_param_ $ityn>]($($param_i: macros::asm::ident_map_ty!($param_ty)),*)
                    => [$($opcode,)*]
                );
            )*
        }
    };
}

#[rustfmt::skip]
macro_rules! ident_map_ty {
    (R)         => { u8 };
    (I)         => { impl Imm };
    ($id:ident) => { $id };
}

pub(crate) use {ident_map_ty, impl_asm, impl_asm_opcodes};

#[allow(clippy::single_component_path_imports)]
pub(crate) use impl_asm_insert;
