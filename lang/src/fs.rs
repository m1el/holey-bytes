use {
    crate::{
        codegen,
        parser::{self, Ast, ParserCtx},
    },
    alloc::{string::String, vec::Vec},
    core::{fmt::Write, num::NonZeroUsize},
    hashbrown::hash_map,
    std::{
        collections::VecDeque,
        eprintln,
        ffi::OsStr,
        io::{self, Write as _},
        path::{Path, PathBuf},
        string::ToString,
        sync::Mutex,
    },
};

pub struct Logger;

impl log::Log for Logger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args())
        }
    }

    fn flush(&self) {}
}

#[derive(Default)]
pub struct Options {
    pub fmt: bool,
    pub fmt_stdout: bool,
    pub dump_asm: bool,
    pub extra_threads: usize,
}

impl Options {
    pub fn from_args(args: &[&str]) -> std::io::Result<Self> {
        if args.contains(&"--help") || args.contains(&"-h") {
            log::error!("Usage: hbc [OPTIONS...] <FILE>");
            log::error!(include_str!("../command-help.txt"));
            return Err(std::io::ErrorKind::Other.into());
        }

        Ok(Options {
            fmt: args.contains(&"--fmt"),
            fmt_stdout: args.contains(&"--fmt-stdout"),
            dump_asm: args.contains(&"--dump-asm"),
            extra_threads: args
                .iter()
                .position(|&a| a == "--threads")
                .map(|i| {
                    args[i + 1].parse::<NonZeroUsize>().map_err(|e| {
                        std::io::Error::other(format!("--threads expects non zero integer: {e}"))
                    })
                })
                .transpose()?
                .map_or(1, NonZeroUsize::get)
                - 1,
        })
    }
}

pub fn run_compiler(root_file: &str, options: Options, out: &mut Vec<u8>) -> std::io::Result<()> {
    let parsed = parse_from_fs(options.extra_threads, root_file)?;

    fn format_ast(ast: parser::Ast) -> std::io::Result<()> {
        let mut output = String::new();
        write!(output, "{ast}").unwrap();
        std::fs::write(&*ast.path, output)?;
        Ok(())
    }

    if options.fmt {
        for parsed in parsed {
            format_ast(parsed)?;
        }
    } else if options.fmt_stdout {
        let ast = parsed.into_iter().next().unwrap();
        write!(out, "{ast}").unwrap();
    } else {
        let mut codegen = codegen::Codegen::default();
        codegen.files = parsed;

        codegen.generate(0);
        if options.dump_asm {
            codegen
                .disasm(unsafe { std::mem::transmute::<&mut Vec<u8>, &mut String>(out) })
                .map_err(|e| io::Error::other(e.to_string()))?;
        } else {
            codegen.assemble(out);
        }
    }

    Ok(())
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
            match core::mem::replace(&mut res, TaskSlot::Waiting) {
                TaskSlot::Delivered(msg) => return Some(msg),
                TaskSlot::Closed => return None,
                TaskSlot::Waiting => {}
            }
        }
    }
}

pub fn parse_from_fs(extra_threads: usize, root: &str) -> io::Result<Vec<Ast>> {
    fn resolve(path: &str, from: &str, tmp: &mut PathBuf) -> Result<PathBuf, CantLoadFile> {
        tmp.clear();
        match Path::new(from).parent() {
            Some(parent) => tmp.extend([parent, Path::new(path)]),
            None => tmp.push(path),
        };

        tmp.canonicalize().map_err(|source| CantLoadFile { path: std::mem::take(tmp), source })
    }

    #[derive(Debug)]
    struct CantLoadFile {
        path: PathBuf,
        source: io::Error,
    }

    impl core::fmt::Display for CantLoadFile {
        fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            write!(f, "can't load file: {}", display_rel_path(&self.path),)
        }
    }

    impl core::error::Error for CantLoadFile {
        fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
            Some(&self.source)
        }
    }

    impl From<CantLoadFile> for io::Error {
        fn from(e: CantLoadFile) -> Self {
            io::Error::new(io::ErrorKind::InvalidData, e)
        }
    }

    type Task = (u32, PathBuf);

    let seen = Mutex::new(crate::HashMap::<PathBuf, u32>::default());
    let tasks = TaskQueue::<Task>::new(extra_threads + 1);
    let ast = Mutex::new(Vec::<io::Result<Ast>>::new());

    let loader = |path: &str, from: &str, tmp: &mut _| {
        if path.starts_with("rel:") {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "`rel:` prefix was removed and is now equivalent to no prefix (remove it)"
                    .to_string(),
            ));
        }

        let mut physiscal_path = resolve(path, from, tmp)?;

        let id = {
            let mut seen = seen.lock().unwrap();
            let len = seen.len();
            match seen.entry(physiscal_path) {
                hash_map::Entry::Occupied(entry) => {
                    return Ok(*entry.get());
                }
                hash_map::Entry::Vacant(entry) => {
                    physiscal_path = entry.insert_entry(len as _).key().clone();
                    len as u32
                }
            }
        };

        if !physiscal_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("can't find file: {}", display_rel_path(&physiscal_path)),
            ));
        }

        tasks.push((id, physiscal_path));
        Ok(id)
    };

    let execute_task = |ctx: &mut _, (_, path): Task, tmp: &mut _| {
        let path = path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("path contains invalid characters: {}", display_rel_path(&path)),
            )
        })?;
        Ok(Ast::new(path, std::fs::read_to_string(path)?, ctx, &mut |path, from| {
            loader(path, from, tmp).map_err(|e| e.to_string())
        }))
    };

    let thread = || {
        let mut ctx = ParserCtx::default();
        let mut tmp = PathBuf::new();
        while let Some(task @ (indx, ..)) = tasks.pop() {
            let res = execute_task(&mut ctx, task, &mut tmp);
            let mut ast = ast.lock().unwrap();
            let len = ast.len().max(indx as usize + 1);
            ast.resize_with(len, || Err(io::ErrorKind::InvalidData.into()));
            ast[indx as usize] = res;
        }
    };

    let path = Path::new(root).canonicalize().map_err(|e| {
        io::Error::new(e.kind(), format!("can't canonicalize root file path ({root})"))
    })?;
    seen.lock().unwrap().insert(path.clone(), 0);
    tasks.push((0, path));

    if extra_threads == 0 {
        thread();
    } else {
        std::thread::scope(|s| (0..extra_threads + 1).for_each(|_| _ = s.spawn(thread)));
    }

    ast.into_inner().unwrap().into_iter().collect::<io::Result<Vec<_>>>()
}

pub fn display_rel_path(path: &(impl AsRef<OsStr> + ?Sized)) -> std::path::Display {
    static CWD: std::sync::LazyLock<PathBuf> =
        std::sync::LazyLock::new(|| std::env::current_dir().unwrap_or_default());
    std::path::Path::new(path).strip_prefix(&*CWD).unwrap_or(std::path::Path::new(path)).display()
}
