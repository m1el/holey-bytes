//! Macros to generate [`crate::Assembler`]

/// Incremental token-tree muncher to implement specific instruction
/// functions based on generic function for instruction type
macro_rules! impl_asm_opcodes {
    ( // End case
        $generic:ident
        ($($param_i:ident: $param_ty:ty),*)
        => []
    ) => {};

    (
        $generic:ident
        ($($param_i:ident: $param_ty:ty),*)
        => [$opcode:ident, $($rest:tt)*]
    ) => {
        // Instruction-specific function
        paste::paste! {
            #[inline(always)]
            pub fn [<i_ $opcode:lower>](&mut self, $($param_i: $param_ty),*) {
                self.$generic(hbbytecode::opcode::$opcode, $($param_i),*)
            }
        }

        // And recurse!
        macros::asm::impl_asm_opcodes!(
            $generic($($param_i: $param_ty),*)
            => [$($rest)*]
        );
    };
}

/// Numeric value insert
macro_rules! impl_asm_insert {
    // Immediate - this is trait-based,
    // the insertion is delegated to its implementation
    ($self:expr, $id:ident, I) => {
        Imm::insert(&$id, $self)
    };

    // Length - cannot be more than 2048
    ($self:expr, $id:ident, L) => {{
        assert!($id <= 2048);
        $self.buf.extend($id.to_le_bytes())
    }};

    // Other numbers, just insert their bytes, little endian
    ($self:expr, $id:ident, $_:ident) => {
        $self.buf.extend($id.to_le_bytes())
    };
}

/// Implement assembler
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
                // Opcode-generic functions specific for instruction types
                pub fn [<i_param_ $ityn>](&mut self, opcode: u8, $($param_i: macros::asm::ident_map_ty!($param_ty)),*) {
                    self.buf.push(opcode);
                    $(macros::asm::impl_asm_insert!(self, $param_i, $param_ty);)*
                }

                // Generate opcode-specific functions calling the opcode-generic ones
                macros::asm::impl_asm_opcodes!(
                    [<i_param_ $ityn>]($($param_i: macros::asm::ident_map_ty!($param_ty)),*)
                    => [$($opcode,)*]
                );
            )*
        }
    };
}

/// Map operand type to Rust type
#[rustfmt::skip]
macro_rules! ident_map_ty {
    (R)         => { u8 };       // Register is just u8
    (I)         => { impl Imm }; // Immediate is anything implementing the trait
    (L)         => { u16 };      // Copy count
    ($id:ident) => { $id };      // Anything else → identity map
}

pub(crate) use {ident_map_ty, impl_asm, impl_asm_insert, impl_asm_opcodes};
