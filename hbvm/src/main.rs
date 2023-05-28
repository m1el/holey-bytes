use hbvm::{
    bytecode::ops::{Operations::*},
    engine::Engine,
    RuntimeErrors, HaltStatus,
};

fn main() -> Result<(), RuntimeErrors> {
    // TODO: Grab program from cmdline
    #[rustfmt::skip]
        let prog: Vec<u8> = vec![
            NOP as u8,            
            JUMP as u8, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

    let mut eng = Engine::new(prog);
    // eng.set_timer_callback(time);
    eng.enviroment_call_table[10] = Some(print_fn);
    while eng.run()? != HaltStatus::Halted {}
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
