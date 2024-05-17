#![feature(noop_waker)]
#![feature(iter_collect_into)]
#![feature(macro_metavar_expr)]
#![feature(let_chains)]
#![feature(ptr_metadata)]
#![feature(const_mut_refs)]
#![feature(slice_ptr_get)]
#![allow(dead_code)]

use std::{
    collections::{HashSet, VecDeque},
    io,
    sync::{mpsc, Arc, Mutex},
};

#[macro_export]
macro_rules! run_tests {
    ($runner:path: $($name:ident => $input:expr;)*) => {$(
        #[test]
        fn $name() {
            $crate::tests::run_test(std::any::type_name_of_val(&$name), $input, $runner);
        }
    )*};
}

pub mod codegen;
mod ident;
mod instrs;
mod lexer;
mod log;
pub mod parser;
mod tests;
mod typechk;

#[inline]
unsafe fn encode<T>(instr: T) -> (usize, [u8; instrs::MAX_SIZE]) {
    let mut buf = [0; instrs::MAX_SIZE];
    std::ptr::write(buf.as_mut_ptr() as *mut T, instr);
    (std::mem::size_of::<T>(), buf)
}

struct TaskQueue<T> {
    inner: Mutex<TaskQueueInner<T>>,
}

impl<T> TaskQueue<T> {
    fn new(max_waiters: usize) -> Self {
        Self {
            inner: Mutex::new(TaskQueueInner::new(max_waiters)),
        }
    }

    pub fn push(&self, message: T) {
        self.extend([message]);
    }

    pub fn extend(&self, messages: impl IntoIterator<Item = T>) {
        self.inner.lock().unwrap().push(messages);
    }

    pub fn pop(&self) -> Option<T> {
        TaskQueueInner::pop(&self.inner)
    }
}

enum TaskSlot<T> {
    Waiting,
    Delivered(T),
    Closed,
}

struct TaskQueueInner<T> {
    max_waiters: usize,
    messages:    VecDeque<T>,
    parked:      VecDeque<(*mut TaskSlot<T>, std::thread::Thread)>,
}

unsafe impl<T: Send> Send for TaskQueueInner<T> {}
unsafe impl<T: Send + Sync> Sync for TaskQueueInner<T> {}

impl<T> TaskQueueInner<T> {
    fn new(max_waiters: usize) -> Self {
        Self {
            max_waiters,
            messages: Default::default(),
            parked: Default::default(),
        }
    }

    fn push(&mut self, messages: impl IntoIterator<Item = T>) {
        for msg in messages {
            if let Some((dest, thread)) = self.parked.pop_front() {
                unsafe { *dest = TaskSlot::Delivered(msg) };
                thread.unpark();
            } else {
                self.messages.push_back(msg);
            }
        }
    }

    fn pop(s: &Mutex<Self>) -> Option<T> {
        let mut res = TaskSlot::Waiting;
        {
            let mut s = s.lock().unwrap();
            if let Some(msg) = s.messages.pop_front() {
                return Some(msg);
            }

            if s.max_waiters == s.parked.len() + 1 {
                for (dest, thread) in s.parked.drain(..) {
                    unsafe { *dest = TaskSlot::Closed };
                    thread.unpark();
                }
                return None;
            }

            s.parked.push_back((&mut res, std::thread::current()));
        }

        loop {
            std::thread::park();

            let _s = s.lock().unwrap();
            match std::mem::replace(&mut res, TaskSlot::Waiting) {
                TaskSlot::Delivered(msg) => return Some(msg),
                TaskSlot::Closed => return None,
                TaskSlot::Waiting => {}
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    #[test]
    fn task_queue_sanity() {
        let queue = Arc::new(super::TaskQueue::new(1000));

        let threads = (0..10)
            .map(|_| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        queue.extend([queue.pop().unwrap()]);
                        //dbg!();
                    }
                })
            })
            .collect::<Vec<_>>();

        queue.extend(0..5);

        for t in threads {
            t.join().unwrap();
        }
    }
}
