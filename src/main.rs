use holey_bytes::bytecode::ops::Operations::*;
use holey_bytes::bytecode::types::CONST_U64;
use holey_bytes::RuntimeErrors;
use holey_bytes::{bytecode::types::CONST_U8, engine::Engine};

fn main() -> Result<(), RuntimeErrors> {
    #[rustfmt::skip]
    let prog: Vec<u8> = vec![
        NOP as u8,
        ADD as u8, CONST_U8, 1, CONST_U8, 20, 0xA0,
        ADD as u8, CONST_U8, 1, CONST_U8,  0, 0xB0,
        ADD as u8, CONST_U64, 0, 0, 0, 2, CONST_U64, 0, 0, 0, 2, 0xD0,
        // SUB, CONST_U8, 4, CONST_U8,  1, 0xC8,
    ];
    let mut eng = Engine::new(prog);
    // eng.timer_callback = Some(time);
    eng.run()?;
    // eng.dump();
    Ok(())
}
