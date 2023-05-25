use {
    super::Engine,
    crate::{HaltStatus, RuntimeErrors},
    alloc::vec,
    RuntimeErrors::*,
};

#[test]
fn invalid_program() {
    let prog = vec![1, 0];
    let mut eng = Engine::new(prog);
    let ret = eng.run();
    assert_eq!(ret, Err(InvalidOpcodePair(1, 0)));
}

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
    use crate::bytecode::ops::Operations::JUMP;
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
    use crate::bytecode::ops::{MathOpSides::ConstantConstant, Operations::ADD};

    let prog = vec![ADD as u8, ConstantConstant as u8, 100, 98, 0xA0];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 2);
}

#[test]
fn sub_u8() {
    use crate::bytecode::ops::Operations::SUB;

    let prog = vec![SUB as u8];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 1);
}
#[test]
fn mul_u8() {
    use crate::bytecode::ops::{MathOpSides::ConstantConstant, Operations::MUL};

    let prog = vec![MUL as u8, ConstantConstant as u8, 1, 2, 0xA0];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 2);
}

#[test]
fn div_u8() {
    use crate::bytecode::ops::Operations::DIV;

    let prog = vec![DIV as u8];
    let mut eng = Engine::new(prog);
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 2);
}

#[test]
fn set_register() {
    let prog = alloc::vec![];
    let mut eng = Engine::new(prog);
    eng.set_register(0xA0, 1);
    assert_eq!(eng.registers.a0, 1);
}

#[test]
fn load_u8() {
    use crate::bytecode::ops::{Operations::LOAD, RWSubTypes::AddrToReg};

    let prog = vec![LOAD as u8, AddrToReg as u8, 0, 0, 0, 0, 0, 0, 1, 0, 0xA0];
    let mut eng = Engine::new(prog);
    let ret = eng.memory.set_addr8(256, 1);
    assert_eq!(ret, Ok(()));
    let _ = eng.run();
    assert_eq!(eng.registers.a0, 1);
}
#[test]
fn set_memory_8() {
    let prog = vec![];
    let mut eng = Engine::new(prog);
    let ret = eng.memory.set_addr8(256, 1);
    assert_eq!(ret, Ok(()));
}

#[test]
fn set_memory_64() {
    let prog = vec![];
    let mut eng = Engine::new(prog);
    let ret = eng.memory.set_addr64(256, 1);
    assert_eq!(ret, Ok(()));
}
