#![feature(if_let_guard)]
#![feature(slice_partition_dedup)]
#![feature(noop_waker)]
#![feature(portable_simd)]
#![feature(iter_collect_into)]
#![feature(macro_metavar_expr)]
#![feature(let_chains)]
#![feature(ptr_metadata)]
#![feature(const_mut_refs)]
#![feature(slice_ptr_get)]

use std::{
    collections::VecDeque,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Mutex,
};

use parser::Ast;

use crate::parser::FileId;

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

pub fn parse_all(threads: usize) -> io::Result<Vec<Ast>> {
    const GIT_DEPS_DIR: &str = "git-deps";

    enum ImportPath<'a> {
        Root {
            path: &'a str,
        },
        Rel {
            path: &'a str,
        },
        Git {
            link:   &'a str,
            path:   &'a str,
            branch: Option<&'a str>,
            tag:    Option<&'a str>,
            rev:    Option<&'a str>,
        },
    }

    impl<'a> TryFrom<&'a str> for ImportPath<'a> {
        type Error = ParseImportError;

        fn try_from(value: &'a str) -> Result<Self, Self::Error> {
            let (prefix, path) = value.split_once(':').unwrap_or(("", value));

            match prefix {
                "" => Ok(Self::Root { path }),
                "rel" => Ok(Self::Rel { path }),
                "git" => {
                    let (link, path) =
                        path.split_once(':').ok_or(ParseImportError::ExpectedPath)?;
                    let (link, params) = link.split_once('?').unwrap_or((link, ""));
                    let [mut branch, mut tag, mut rev] = [None; 3];
                    for (key, value) in params.split('&').filter_map(|s| s.split_once('=')) {
                        match key {
                            "branch" => branch = Some(value),
                            "tag" => tag = Some(value),
                            "rev" => rev = Some(value),
                            _ => return Err(ParseImportError::UnexpectedParam),
                        }
                    }
                    Ok(Self::Git {
                        link,
                        path,
                        branch,
                        tag,
                        rev,
                    })
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
            match self {
                Self::Root { path } => Ok(Path::new(path).to_owned()),
                Self::Rel { path } => {
                    let path = PathBuf::from_iter([from, path]);
                    match path.canonicalize() {
                        Ok(path) => Ok(path),
                        Err(e) => Err(CantLoadFile(path, e)),
                    }
                }
                Self::Git { path, link, .. } => {
                    let link = preprocess_git(link);
                    Ok(PathBuf::from_iter([GIT_DEPS_DIR, link, path]))
                }
            }
        }
    }

    #[derive(Debug)]
    enum ParseImportError {
        ExpectedPath,
        InvalidPrefix,
        UnexpectedParam,
    }

    impl std::fmt::Display for ParseImportError {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::ExpectedPath => write!(f, "expected path"),
                Self::InvalidPrefix => write!(
                    f,
                    "invalid prefix, expected one of rel, \
                    git or none followed by colon"
                ),
                Self::UnexpectedParam => {
                    write!(f, "unexpected git param, expected branch, tag or rev")
                }
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
    struct CantLoadFile(PathBuf, io::Error);

    impl std::fmt::Display for CantLoadFile {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "can't load file: {}", self.0.display())
        }
    }

    impl std::error::Error for CantLoadFile {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.1)
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

    type Task = (FileId, PathBuf, Option<std::process::Command>);

    let seen = Mutex::new(HashMap::<PathBuf, FileId>::default());
    let tasks = TaskQueue::<Task>::new(threads);
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
                    len as FileId
                }
            }
        };

        let command = if !physiscal_path.exists() {
            let ImportPath::Git {
                link,
                branch,
                rev,
                tag,
                ..
            } = path
            else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("can't find file: {}", physiscal_path.display()),
                ));
            };

            let root = PathBuf::from_iter([GIT_DEPS_DIR, preprocess_git(link)]);

            let mut command = std::process::Command::new("git");
            command
                .args(["clone", "--depth", "1"])
                .args(branch.map(|b| ["--branch", b]).into_iter().flatten())
                .args(tag.map(|t| ["--tag", t]).into_iter().flatten())
                .args(rev.map(|r| ["--rev", r]).into_iter().flatten())
                .arg(link)
                .arg(root);
            Some(command)
        } else {
            None
        };

        tasks.push((id, physiscal_path, command));
        Ok(id)
    };

    let execute_task = |(_, path, command): Task, buffer: &mut Vec<u8>| {
        if let Some(mut command) = command {
            let output = command.output()?;
            if !output.status.success() {
                let msg = format!(
                    "git command failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
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
            let res = execute_task(task, &mut buffer);
            buffer.clear();

            let mut ast = ast.lock().unwrap();
            let len = ast.len().max(indx as usize + 1);
            ast.resize_with(len, || Err(io::ErrorKind::InvalidData.into()));
            ast[indx as usize] = res;
        }
    };

    std::thread::scope(|s| (0..threads).for_each(|_| _ = s.spawn(thread)));

    ast.into_inner()
        .unwrap()
        .into_iter()
        .collect::<io::Result<Vec<_>>>()
}

type HashMap<K, V> = std::collections::HashMap<K, V, FnvBuildHash>;

type FnvBuildHash = std::hash::BuildHasherDefault<FnvHasher>;

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
