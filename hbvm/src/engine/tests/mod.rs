use {
    super::Engine,
    crate::{HaltStatus, RuntimeErrors},
    alloc::vec,
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
    assert_eq!(ret, Err(RuntimeErrors::InvalidJumpAddress(256)));
}

#[test]
fn invalid_system_call() {
    let prog = vec![255, 0];
    let mut eng = Engine::new(prog);
    let ret = eng.run();
    // assert_eq!(ret, );
}
