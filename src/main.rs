use holey_bytes::bytecode::ops::Operations::*;
use holey_bytes::bytecode::types::CONST_U64;
use holey_bytes::RuntimeErrors;
use holey_bytes::{bytecode::types::CONST_U8, engine::Engine};

fn main() -> Result<(), RuntimeErrors> {
    use holey_bytes::bytecode::ops::MathTypes::*;
    #[rustfmt::skip]
    let prog: Vec<u8> = vec![
        NOP as u8, NOP as u8,
        ADD as u8, EightBit as u8, 100, 20, 0xA7,
        ADD as u8, 
            EightBit as u8, 1, 0, 0xB0,
        ADD as u8,
            SixtyFourBit as u8,
                0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 2, 0xD0
               
    ];
    let mut eng = Engine::new(prog);
    // eng.timer_callback = Some(time);
    eng.run()?;
    eng.dump();
    Ok(())
}
