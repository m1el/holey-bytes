#![feature(
    let_chains,
    if_let_guard,
    macro_metavar_expr,
    anonymous_lifetime_in_impl_trait,
    core_intrinsics,
    new_uninit,
    never_type,
    unwrap_infallible,
    slice_partition_dedup,
    portable_simd,
    iter_collect_into,
    ptr_metadata,
    slice_ptr_get
)]
#![allow(internal_features, clippy::format_collect)]

use {
    parser::Ast,
    std::{
        collections::VecDeque,
        io::{self, Read},
        path::{Path, PathBuf},
        sync::Mutex,
    },
};

#[macro_export]
macro_rules! run_tests {
    ($runner:path: $($name:ident => $input:expr;)*) => {$(
        #[test]
        fn $name() {
            $crate::run_test(std::any::type_name_of_val(&$name), stringify!($name), $input, $runner);
        }
    )*};
}

pub mod codegen;
pub mod parser;

mod instrs;
mod lexer;

mod ident {
    pub type Ident = u32;

    const LEN_BITS: u32 = 6;

    pub fn len(ident: u32) -> u32 {
        ident & ((1 << LEN_BITS) - 1)
    }

    pub fn is_null(ident: u32) -> bool {
        (ident >> LEN_BITS) == 0
    }

    pub fn pos(ident: u32) -> u32 {
        (ident >> LEN_BITS).saturating_sub(1)
    }

    pub fn new(pos: u32, len: u32) -> u32 {
        debug_assert!(len < (1 << LEN_BITS));
        ((pos + 1) << LEN_BITS) | len
    }

    pub fn range(ident: u32) -> std::ops::Range<usize> {
        let (len, pos) = (len(ident) as usize, pos(ident) as usize);
        pos..pos + len
    }
}

mod log {
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
                eprintln!("{:?}: {}", $level, format_args!($fmt $($expr)*));
            }
        };

        ($level:expr, $($arg:expr),*) => {
            if $level <= $crate::log::LOG_LEVEL {
                $(eprintln!("[{}{}{}][{:?}]: {} = {:?}", line!(), column!(), file!(), $level, stringify!($arg), $arg);)*
            }
        };
    }

    macro_rules! err { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Err, $($arg)*) }; }
    macro_rules! wrn { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Wrn, $($arg)*) }; }
    macro_rules! inf { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Inf, $($arg)*) }; }
    macro_rules! dbg { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Dbg, $($arg)*) }; }

    #[allow(unused_imports)]
    pub(crate) use {dbg, err, inf, log, wrn};
}

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
        Self { inner: Mutex::new(TaskQueueInner::new(max_waiters)) }
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
    messages: VecDeque<T>,
    parked: VecDeque<(*mut TaskSlot<T>, std::thread::Thread)>,
}

unsafe impl<T: Send> Send for TaskQueueInner<T> {}
unsafe impl<T: Send + Sync> Sync for TaskQueueInner<T> {}

impl<T> TaskQueueInner<T> {
    fn new(max_waiters: usize) -> Self {
        Self { max_waiters, messages: Default::default(), parked: Default::default() }
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

pub fn parse_from_fs(extra_threads: usize, root: &str) -> io::Result<Vec<Ast>> {
    const GIT_DEPS_DIR: &str = "git-deps";

    enum Chk<'a> {
        Branch(&'a str),
        Rev(&'a str),
        Tag(&'a str),
    }

    enum ImportPath<'a> {
        Rel { path: &'a str },
        Git { link: &'a str, path: &'a str, chk: Option<Chk<'a>> },
    }

    impl<'a> TryFrom<&'a str> for ImportPath<'a> {
        type Error = ParseImportError;

        fn try_from(value: &'a str) -> Result<Self, Self::Error> {
            let (prefix, path) = value.split_once(':').unwrap_or(("", value));

            match prefix {
                "rel" | "" => Ok(Self::Rel { path }),
                "git" => {
                    let (link, path) =
                        path.split_once(':').ok_or(ParseImportError::ExpectedPath)?;
                    let (link, params) = link.split_once('?').unwrap_or((link, ""));
                    let chk = params.split('&').filter_map(|s| s.split_once('=')).find_map(
                        |(key, value)| match key {
                            "branch" => Some(Chk::Branch(value)),
                            "rev" => Some(Chk::Rev(value)),
                            "tag" => Some(Chk::Tag(value)),
                            _ => None,
                        },
                    );
                    Ok(Self::Git { link, path, chk })
                }
                _ => Err(ParseImportError::InvalidPrefix),
            }
        }
    }

    fn preprocess_git(link: &str) -> &str {
        let link = link.strip_prefix("https://").unwrap_or(link);
        link.strip_suffix(".git").unwrap_or(link)
    }

    impl<'a> ImportPath<'a> {
        fn resolve(&self, from: &str) -> Result<PathBuf, CantLoadFile> {
            let path = match self {
                Self::Rel { path } => match Path::new(from).parent() {
                    Some(parent) => PathBuf::from_iter([parent, Path::new(path)]),
                    None => PathBuf::from(path),
                },
                Self::Git { path, link, .. } => {
                    let link = preprocess_git(link);
                    PathBuf::from_iter([GIT_DEPS_DIR, link, path])
                }
            };

            path.canonicalize().map_err(|source| CantLoadFile {
                path,
                from: PathBuf::from(from),
                source,
            })
        }
    }

    #[derive(Debug)]
    enum ParseImportError {
        ExpectedPath,
        InvalidPrefix,
    }

    impl std::fmt::Display for ParseImportError {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::ExpectedPath => "expected path".fmt(f),
                Self::InvalidPrefix => "invalid prefix, expected one of rel, \
                    git or none followed by colon"
                    .fmt(f),
            }
        }
    }

    impl std::error::Error for ParseImportError {}

    impl From<ParseImportError> for io::Error {
        fn from(e: ParseImportError) -> Self {
            io::Error::new(io::ErrorKind::InvalidInput, e)
        }
    }

    #[derive(Debug)]
    struct CantLoadFile {
        path: PathBuf,
        from: PathBuf,
        source: io::Error,
    }

    impl std::fmt::Display for CantLoadFile {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "can't load file: {} (from: {})", self.path.display(), self.from.display(),)
        }
    }

    impl std::error::Error for CantLoadFile {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.source)
        }
    }

    impl From<CantLoadFile> for io::Error {
        fn from(e: CantLoadFile) -> Self {
            io::Error::new(io::ErrorKind::InvalidData, e)
        }
    }

    #[derive(Debug)]
    struct InvalidFileData(std::str::Utf8Error);

    impl std::fmt::Display for InvalidFileData {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "invalid file data")
        }
    }

    impl std::error::Error for InvalidFileData {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }

    impl From<InvalidFileData> for io::Error {
        fn from(e: InvalidFileData) -> Self {
            io::Error::new(io::ErrorKind::InvalidData, e)
        }
    }

    type Task = (u32, PathBuf, Option<std::process::Command>);

    let seen = Mutex::new(HashMap::<PathBuf, u32>::default());
    let tasks = TaskQueue::<Task>::new(extra_threads + 1);
    let ast = Mutex::new(Vec::<io::Result<Ast>>::new());

    let loader = |path: &str, from: &str| {
        let path = ImportPath::try_from(path)?;

        let physiscal_path = path.resolve(from)?;

        let id = {
            let mut seen = seen.lock().unwrap();
            let len = seen.len();
            match seen.entry(physiscal_path.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    return Ok(*entry.get());
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(len as _);
                    len as u32
                }
            }
        };

        let command = if !physiscal_path.exists() {
            let ImportPath::Git { link, chk, .. } = path else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("can't find file: {}", physiscal_path.display()),
                ));
            };

            let root = PathBuf::from_iter([GIT_DEPS_DIR, preprocess_git(link)]);

            let mut command = std::process::Command::new("git");
            command.args(["clone", "--depth", "1"]);
            if let Some(chk) = chk {
                command.args(match chk {
                    Chk::Branch(b) => ["--branch", b],
                    Chk::Tag(t) => ["--tag", t],
                    Chk::Rev(r) => ["--rev", r],
                });
            }
            command.arg(link).arg(root);
            Some(command)
        } else {
            None
        };

        tasks.push((id, physiscal_path, command));
        Ok(id)
    };

    let execute_task = |(_, path, command): Task, buffer: &mut Vec<u8>| {
        log::dbg!("{path:?}");
        if let Some(mut command) = command {
            let output = command.output()?;
            if !output.status.success() {
                let msg =
                    format!("git command failed: {}", String::from_utf8_lossy(&output.stderr));
                return Err(io::Error::new(io::ErrorKind::Other, msg));
            }
        }

        let path = path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("path contains invalid characters: {}", path.display()),
            )
        })?;
        let mut file = std::fs::File::open(path)?;
        file.read_to_end(buffer)?;
        let src = std::str::from_utf8(buffer).map_err(InvalidFileData)?;
        Ok(Ast::new(path, src, &loader))
    };

    let thread = || {
        let mut buffer = Vec::new();
        while let Some(task @ (indx, ..)) = tasks.pop() {
            dbg!();
            let res = execute_task(task, &mut buffer);
            buffer.clear();

            let mut ast = ast.lock().unwrap();
            let len = ast.len().max(indx as usize + 1);
            ast.resize_with(len, || Err(io::ErrorKind::InvalidData.into()));
            ast[indx as usize] = res;
        }
    };

    let path = Path::new(root).canonicalize()?;
    seen.lock().unwrap().insert(path.clone(), 0);
    tasks.push((0, path, None));

    if extra_threads == 0 {
        thread();
    } else {
        std::thread::scope(|s| (0..extra_threads + 1).for_each(|_| _ = s.spawn(thread)));
    }

    ast.into_inner().unwrap().into_iter().collect::<io::Result<Vec<_>>>()
}

type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FnvHasher>>;

struct FnvHasher(u64);

impl std::hash::Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0 = bytes.iter().fold(self.0, |hash, &byte| {
            let mut hash = hash;
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001B3);
            hash
        });
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self(0xCBF29CE484222325)
    }
}

#[cfg(test)]
pub fn run_test(
    name: &'static str,
    ident: &'static str,
    input: &'static str,
    test: fn(&'static str, &'static str, &mut String),
) {
    use std::{io::Write, path::PathBuf};

    let filter = std::env::var("PT_FILTER").unwrap_or_default();
    if !filter.is_empty() && !name.contains(&filter) {
        return;
    }

    let mut output = String::new();
    test(ident, input, &mut output);

    let mut root = PathBuf::from(
        std::env::var("PT_TEST_ROOT")
            .unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/tests").to_string()),
    );
    root.push(name.replace("::", "_").replace(concat!(env!("CARGO_PKG_NAME"), "_"), ""));
    root.set_extension("txt");

    let expected = std::fs::read_to_string(&root).unwrap_or_default();

    if output == expected {
        return;
    }

    if std::env::var("PT_UPDATE").is_ok() {
        std::fs::write(&root, output).unwrap();
        return;
    }

    if !root.exists() {
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::write(&root, vec![]).unwrap();
    }

    let mut proc = std::process::Command::new("diff")
        .arg("-u")
        .arg("--color")
        .arg(&root)
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::inherit())
        .spawn()
        .unwrap();

    proc.stdin.as_mut().unwrap().write_all(output.as_bytes()).unwrap();

    proc.wait().unwrap();

    panic!("test failed");
}

#[derive(Default)]
pub struct Options {
    pub fmt: bool,
    pub fmt_current: bool,
    pub extra_threads: usize,
}

pub fn run_compiler(
    root_file: &str,
    options: Options,
    out: &mut impl std::io::Write,
) -> io::Result<()> {
    let parsed = parse_from_fs(options.extra_threads, root_file)?;

    fn format_to_stdout(ast: parser::Ast) -> std::io::Result<()> {
        let source = std::fs::read_to_string(&*ast.path)?;
        parser::with_fmt_source(&source, || {
            for expr in ast.exprs() {
                use std::io::Write;
                writeln!(std::io::stdout(), "{expr}")?;
            }
            std::io::Result::Ok(())
        })
    }

    fn format_ast(ast: parser::Ast) -> std::io::Result<()> {
        let source = std::fs::read_to_string(&*ast.path)?;
        let mut output = Vec::new();
        parser::with_fmt_source(&source, || {
            for expr in ast.exprs() {
                use std::io::Write;
                writeln!(output, "{expr}")?;
            }
            std::io::Result::Ok(())
        })?;

        std::fs::write(&*ast.path, output)?;

        Ok(())
    }

    if options.fmt {
        for parsed in parsed {
            format_ast(parsed)?;
        }
    } else if options.fmt_current {
        format_to_stdout(parsed.into_iter().next().unwrap())?;
    } else {
        let mut codegen = codegen::Codegen::default();
        codegen.files = parsed;

        codegen.generate();
        codegen.dump(out)?;
    }

    Ok(())
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
