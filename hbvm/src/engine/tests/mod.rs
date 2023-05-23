use {
    super::Engine,
    crate::{HaltStatus, RuntimeErrors},
    alloc::vec,
    RuntimeErrors::*,
};

#[test]
fn empty_program() {
    let prog = vec![];
    let mut eng = Engine::new(prog);
    let ret = eng.run();
    assert_eq!(ret, Ok(HaltStatus::Halted));
}

#[test]
fn max_quantum_reached() {
    let prog = vec![0, 0, 0, 0];
    let mut eng = Engine::new(prog);
    eng.set_timer_callback(|| {
        return 1;
    });
    eng.config.quantum = 1;
    let ret = eng.run();
    assert_eq!(ret, Ok(HaltStatus::Running));
}

#[test]
fn jump_out_of_bounds() {
    use crate::engine::Operations::JUMP;
    let prog = vec![JUMP as u8, 0, 0, 0, 0, 0, 0, 1, 0];
    let mut eng = Engine::new(prog);
    let ret = eng.run();
    assert_eq!(ret, Err(InvalidJumpAddress(256)));
}

#[test]
fn invalid_system_call() {
    let prog = vec![255, 0];
    let mut eng = Engine::new(prog);
    let ret = eng.run();
    assert_eq!(ret, Err(InvalidSystemCall(0)));
}

#[test]
fn add_u8() {
    use crate::engine::{Operations::ADD, SubTypes::EightBit};

    let prog = vec![ADD as u8, EightBit as u8, 1, 1, 0xA0];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 2);
}

#[test]
fn sub_u8() {
    use crate::engine::{Operations::SUB, SubTypes::EightBit};

    let prog = vec![SUB as u8, EightBit as u8, 2, 1, 0xA0];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 1);
}
#[test]
fn mul_u8() {
    use crate::engine::{Operations::MUL, SubTypes::EightBit};

    let prog = vec![MUL as u8, EightBit as u8, 1, 1, 0xA0];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 2);
}

#[test]
fn div_u8() {
    use crate::engine::{Operations::DIV, SubTypes::EightBit};

    let prog = vec![DIV as u8, EightBit as u8, 1, 1, 0xA0];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 2);
}
