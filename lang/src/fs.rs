use {
    crate::{
        parser::{Ast, Ctx, FileKind},
        son::{self, hbvm::HbvmBackend},
        ty, FnvBuildHasher,
    },
    alloc::{string::String, vec::Vec},
    core::{fmt::Write, num::NonZeroUsize, ops::Deref},
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

type HashMap<K, V> = hashbrown::HashMap<K, V, FnvBuildHasher>;

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
    pub in_house_regalloc: bool,
    pub extra_threads: usize,
}

impl Options {
    pub fn from_args(args: &[&str], out: &mut Vec<u8>) -> std::io::Result<Self> {
        if args.contains(&"--help") || args.contains(&"-h") {
            writeln!(out, "Usage: hbc [OPTIONS...] <FILE>")?;
            writeln!(out, include_str!("../command-help.txt"))?;
            return Err(std::io::ErrorKind::Other.into());
        }

        Ok(Options {
            fmt: args.contains(&"--fmt"),
            fmt_stdout: args.contains(&"--fmt-stdout"),
            dump_asm: args.contains(&"--dump-asm"),
            in_house_regalloc: args.contains(&"--in-house-regalloc"),
            extra_threads: args
                .iter()
                .position(|&a| a == "--threads")
                .map(|i| {
                    args[i + 1].parse::<NonZeroUsize>().map_err(|e| {
                        writeln!(out, "--threads expects non zero integer: {e}")
                            .err()
                            .unwrap_or(std::io::ErrorKind::Other.into())
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

    if (options.fmt || options.fmt_stdout) && !parsed.errors.is_empty() {
        *out = parsed.errors.into_bytes();
        return Err(std::io::Error::other("fmt fialed (errors are in out)"));
    }

    if options.fmt {
        let mut output = String::new();
        for ast in parsed.ast {
            write!(output, "{ast}").unwrap();
            if ast.file.deref().trim() != output.as_str().trim() {
                std::fs::write(&*ast.path, &output)?;
            }
            output.clear();
        }
    } else if options.fmt_stdout {
        write!(out, "{}", &parsed.ast[0])?;
    } else {
        let mut backend = HbvmBackend::default();
        backend.use_in_house_regalloc = options.in_house_regalloc;

        let mut ctx = crate::son::CodegenCtx::default();
        *ctx.parser.errors.get_mut() = parsed.errors;
        let mut codegen = son::Codegen::new(&mut backend, &parsed.ast, &mut ctx);

        codegen.push_embeds(parsed.embeds);
        codegen.generate(ty::Module::MAIN);

        if !codegen.errors.borrow().is_empty() {
            drop(codegen);
            *out = ctx.parser.errors.into_inner().into_bytes();
            return Err(std::io::Error::other("compilation faoled (errors are in out)"));
        }

        codegen.assemble(out);

        if options.dump_asm {
            let mut disasm = String::new();
            codegen.disasm(&mut disasm, out).map_err(|e| io::Error::other(e.to_string()))?;
            *out = disasm.into_bytes();
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

pub struct Loaded {
    ast: Vec<Ast>,
    embeds: Vec<Vec<u8>>,
    errors: String,
}

pub fn parse_from_fs(extra_threads: usize, root: &str) -> io::Result<Loaded> {
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

    type Task = (usize, PathBuf);

    let seen_modules = Mutex::new(HashMap::<PathBuf, usize>::default());
    let seen_embeds = Mutex::new(HashMap::<PathBuf, usize>::default());
    let tasks = TaskQueue::<Task>::new(extra_threads + 1);
    let ast = Mutex::new(Vec::<io::Result<Ast>>::new());
    let embeds = Mutex::new(Vec::<Vec<u8>>::new());

    let loader = |path: &str, from: &str, kind: FileKind, tmp: &mut _| {
        let mut physiscal_path = resolve(path, from, tmp)?;

        match kind {
            FileKind::Module => {
                let id = {
                    let mut seen = seen_modules.lock().unwrap();
                    let len = seen.len();
                    match seen.entry(physiscal_path) {
                        hash_map::Entry::Occupied(entry) => {
                            return Ok(*entry.get());
                        }
                        hash_map::Entry::Vacant(entry) => {
                            physiscal_path = entry.insert_entry(len as _).key().clone();
                            len
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
            }
            FileKind::Embed => {
                let id = {
                    let mut seen = seen_embeds.lock().unwrap();
                    let len = seen.len();
                    match seen.entry(physiscal_path) {
                        hash_map::Entry::Occupied(entry) => {
                            return Ok(*entry.get());
                        }
                        hash_map::Entry::Vacant(entry) => {
                            physiscal_path = entry.insert_entry(len as _).key().clone();
                            len
                        }
                    }
                };

                let content = std::fs::read(&physiscal_path).map_err(|e| {
                    io::Error::new(
                        e.kind(),
                        format!(
                            "can't load embed file: {}: {e}",
                            display_rel_path(&physiscal_path)
                        ),
                    )
                })?;
                let mut embeds = embeds.lock().unwrap();
                if id as usize >= embeds.len() {
                    embeds.resize(id as usize + 1, Default::default());
                }
                embeds[id as usize] = content;
                Ok(id)
            }
        }
    };

    let execute_task = |ctx: &mut _, (_, path): Task, tmp: &mut _| {
        let path = path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("path contains invalid characters: {}", display_rel_path(&path)),
            )
        })?;
        Ok(Ast::new(path, std::fs::read_to_string(path)?, ctx, &mut |path, from, kind| {
            loader(path, from, kind, tmp).map_err(|e| e.to_string())
        }))
    };

    let thread = || {
        let mut ctx = Ctx::default();
        let mut tmp = PathBuf::new();
        while let Some(task @ (indx, ..)) = tasks.pop() {
            let res = execute_task(&mut ctx, task, &mut tmp);
            let mut ast = ast.lock().unwrap();
            let len = ast.len().max(indx + 1);
            ast.resize_with(len, || Err(io::ErrorKind::InvalidData.into()));
            ast[indx] = res;
        }
        ctx.errors.into_inner()
    };

    let path = Path::new(root).canonicalize().map_err(|e| {
        io::Error::new(e.kind(), format!("can't canonicalize root file path ({root})"))
    })?;
    seen_modules.lock().unwrap().insert(path.clone(), 0);
    tasks.push((0, path));

    let errors = if extra_threads == 0 {
        thread()
    } else {
        std::thread::scope(|s| {
            (0..extra_threads + 1)
                .map(|_| s.spawn(thread))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<String>()
        })
    };

    Ok(Loaded {
        ast: ast.into_inner().unwrap().into_iter().collect::<io::Result<Vec<_>>>()?,
        embeds: embeds.into_inner().unwrap(),
        errors,
    })
}

pub fn display_rel_path(path: &(impl AsRef<OsStr> + ?Sized)) -> std::path::Display {
    static CWD: std::sync::LazyLock<PathBuf> =
        std::sync::LazyLock::new(|| std::env::current_dir().unwrap_or_default());
    std::path::Path::new(path).strip_prefix(&*CWD).unwrap_or(std::path::Path::new(path)).display()
}
