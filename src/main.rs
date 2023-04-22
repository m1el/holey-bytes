use holey_bytes::{
    bytecode::{
        ops::{Operations::*, RWSubTypes, RWSubTypes::*, SubTypes::*},
        types::*,
    },
    engine::Engine,
    RuntimeErrors,
};

fn main() -> Result<(), RuntimeErrors> {
    #[rustfmt::skip]
    let prog: Vec<u8> = vec![
        // NOP as u8, NOP as u8,
        ADD as u8, EightBit as u8, 100, 20, 0xA7,
        ADD as u8, 
            EightBit as u8, 1, 0, 0xB0,
        ADD as u8,
            SixtyFourBit as u8,
                0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 2, 0xD0,
        SUB as u8, EightBit as u8, 255, 0, 0xA7, 
        ADD as u8, Register8 as u8, 0xA7, 0xB0, 0xA7,
        LOAD as u8, AddrToReg as u8,
                    0, 0, 0, 0, 0, 0, 0, 2,
                    0xA0,
        JUMP as u8, 0, 0, 0, 0, 0, 0, 0, 5,
    ];
    let mut eng = Engine::new(prog);
    // eng.timer_callback = Some(time);
    eng.run()?;
    eng.dump();
    Ok(())
}
