use std::{result, time::Duration};

use holey_bytes::{
    bytecode::ops::{Operations::*, RWSubTypes::*, SubTypes::*},
    engine::{regs::Registers, Engine},
    RuntimeErrors,
};

fn main() -> Result<(), RuntimeErrors> {
    let mut results: Vec<Duration> = vec![];
    let iters = 1;
    for _ in 0..iters {
        // #[rustfmt::skip]
        // let prog: Vec<u8> = vec![
        //     // NOP as u8, NOP as u8,
        //     ADD as u8, EightBit as u8, 100, 20, 0xA7,
        //     ADD as u8,
        //         EightBit as u8, 1, 0, 0xB0,
        //     ADD as u8,
        //         SixtyFourBit as u8,
        //             0, 0, 0, 0, 0, 0, 0, 0,
        //             0, 0, 0, 0, 0, 0, 0, 2, 0xD0,
        //     SUB as u8, EightBit as u8, 255, 0, 0xA7,
        //     ADD as u8, Register8 as u8, 0xA7, 0xB0, 0xA7,
        //     LOAD as u8, AddrToReg as u8,
        //                 0, 0, 0, 0, 0, 0, 0, 2,
        //                 0xA0,
        //     10, 10,
        //     JUMP as u8, 0, 0, 0, 0, 0, 0, 0, 5,
        // ];
        let prog = vec![10, 10];

        use std::time::Instant;
        let now = Instant::now();

        let mut eng = Engine::new(prog);

        eng.set_timer_callback(time);
        eng.enviroment_call_table[10] = print_fn;
        eng.run()?;
        eng.dump();
        let elapsed = now.elapsed();
        // println!("Elapsed: {:.2?}", elapsed);
        results.push(elapsed);
    }
    let mut val = 0;
    for x in results {
        val = val + x.as_micros();
    }
    println!("micro seconds {}", val / iters);
    Ok(())
}

pub fn time() -> u32 {
    9
}
pub fn print_fn(reg: Registers) -> Result<(), ()> {
    println!("{:?}", reg);
    Ok(())
}
