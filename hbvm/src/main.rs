use hbvm::{
    bytecode::ops::{Operations::*, RWSubTypes::*, SubTypes::*},
    engine::Engine,
    RuntimeErrors,
};

fn main() -> Result<(), RuntimeErrors> {
    // TODO: Grab program from cmdline
    #[rustfmt::skip]
        let prog: Vec<u8> = vec![
            NOP as u8,
            255, 10,
            // ADD as u8, EightBit as u8, 100, 20, 0xA7,
            // ADD as u8,
                // EightBit as u8, 1, 0, 0xB0,
            // ADD as u8,
            //     SixtyFourBit as u8,
            //         0, 0, 0, 0, 0, 0, 0, 0,
            //         0, 0, 0, 0, 0, 0, 0, 2, 0xD0,
            SUB as u8, EightBit as u8, 255, 0, 0xA0,
            // This currently fails because of a lack of register to register addition
            // ADD as u8, Register8 as u8, 0xA0, 0xA0, 0xA1,
            // LOAD as u8, AddrToReg as u8,
                        // 0, 0, 0, 0, 0, 0, 0, 2,
                        // 0xA0,
            JUMP as u8, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

    let mut eng = Engine::new(prog);
    // eng.set_timer_callback(time);
    eng.enviroment_call_table[10] = Some(print_fn);
    eng.run()?;
    eng.dump();
    println!("{:#?}", eng.registers);

    Ok(())
}

pub fn time() -> u32 {
    9
}
pub fn print_fn(engine: &mut Engine) -> Result<&mut Engine, u64> {
    println!("hello");
    Ok(engine)
}
