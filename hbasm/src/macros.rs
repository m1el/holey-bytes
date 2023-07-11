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

        macros::impl_asm_opcodes!(
            $generic($($param_i: $param_ty),*)
            => [$($rest)*]
        );
    };
}

macro_rules! gen_impl_asm_insert {
    ($($ty:ident),* $(,)?) => {
        macro_rules! impl_asm_insert {
            $(($self:expr, $id:ident, $ty) => {
                $self.buf.extend($id.to_le_bytes())
            };)*

            ($self:expr, $id:ident, $_:ty) => {
                Imm::insert(&$id, $self)
            };
        }
    };
}

gen_impl_asm_insert!(u8, u16, u64);

macro_rules! impl_asm {
    (
        $(
            $ityn:ident
            ($($param_i:ident: $param_ty:ty),* $(,)?)
            => [$($opcode:ident),* $(,)?],
        )*
    ) => {
        paste::paste! {
            $(
                #[allow(dead_code)]
                fn [<i_param_ $ityn>](&mut self, opcode: u8, $($param_i: $param_ty),*) {
                    self.buf.push(opcode);
                    $(macros::impl_asm_insert!(self, $param_i, $param_ty);)*
                }

                macros::impl_asm_opcodes!(
                    [<i_param_ $ityn>]($($param_i: $param_ty),*)
                    => [$($opcode,)*]
                );
            )*
        }
    };
}

pub(super) use {impl_asm, impl_asm_opcodes};

#[allow(clippy::single_component_path_imports)]
pub(super) use impl_asm_insert;
