#![allow(unused_macros)]

#[derive(PartialOrd, PartialEq, Ord, Eq, Debug)]
pub enum Level {
    Err,
    Wrn,
    Inf,
    Dbg,
}

pub const LOG_LEVEL: Level = match option_env!("LOG_LEVEL") {
    Some(val) => match val.as_bytes()[0] {
        b'e' => Level::Err,
        b'w' => Level::Wrn,
        b'i' => Level::Inf,
        b'd' => Level::Dbg,
        _ => panic!("Invalid log level."),
    },
    None => {
        if cfg!(debug_assertions) {
            Level::Dbg
        } else {
            Level::Err
        }
    }
};

macro_rules! log {
    ($level:expr, $fmt:literal $($expr:tt)*) => {
        if $level <= $crate::log::LOG_LEVEL {
            println!("{:?}: {}", $level, format_args!($fmt $($expr)*));
        }
    };

    ($level:expr, $($arg:expr),*) => {
        if $level <= $crate::log::LOG_LEVEL {
            $(println!("[{}{}{}][{:?}]: {} = {:?}", line!(), column!(), file!(), $level, stringify!($arg), $arg);)*
        }
    };
}

macro_rules! err { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Err, $($arg)*) }; }
macro_rules! wrn { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Wrn, $($arg)*) }; }
macro_rules! inf { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Inf, $($arg)*) }; }
macro_rules! dbg { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Dbg, $($arg)*) }; }

#[allow(unused_imports)]
pub(crate) use {dbg, err, inf, log, wrn};
