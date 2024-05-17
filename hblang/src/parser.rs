use std::{
    cell::{Cell, UnsafeCell},
    collections::{HashMap, HashSet},
    io::{self, Read},
    ops::Not,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU32, AtomicUsize},
        Mutex,
    },
};

use crate::{
    codegen::bt,
    ident::{self, Ident},
    lexer::{Lexer, Token, TokenKind},
    TaskQueue,
};

pub type Pos = u32;
pub type IdentFlags = u32;
pub type Symbols = Vec<Symbol>;
pub type FileId = u32;
pub type Loader<'a> = &'a (dyn Fn(&str, &str) -> io::Result<Option<FileId>> + 'a);

pub const MUTABLE: IdentFlags = 1 << std::mem::size_of::<IdentFlags>() * 8 - 1;
pub const REFERENCED: IdentFlags = 1 << std::mem::size_of::<IdentFlags>() * 8 - 2;
const GIT_DEPS_DIR: &str = "git-deps";

pub fn parse_all(root: &str, threads: usize) -> io::Result<Vec<Ast>> {
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
        fn resolve(&self, from: &str, root: &str) -> Result<PathBuf, CantLoadFile> {
            match self {
                Self::Root { path } => Ok(PathBuf::from_iter([root, path])),
                Self::Rel { path } => {
                    let path = PathBuf::from_iter([from, path]);
                    match path.canonicalize() {
                        Ok(path) => Ok(path),
                        Err(e) => Err(CantLoadFile(path, e)),
                    }
                }
                Self::Git { path, link, .. } => {
                    let link = preprocess_git(link);
                    Ok(PathBuf::from_iter([root, GIT_DEPS_DIR, link, path]))
                }
            }
        }
    }

    #[derive(Debug)]
    enum ParseImportError {
        ExpectedPath,
        InvalidPrefix,
        ExpectedGitAlias,
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
                Self::ExpectedGitAlias => write!(f, "expected git alias as ':<alias>$'"),
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

    enum Task {
        LoadFile {
            id: FileId,
            physiscal_path: PathBuf,
        },
        FetchGit {
            id: FileId,
            physiscal_path: PathBuf,
            command: std::process::Command,
        },
    }

    let seen = Mutex::new(HashMap::<PathBuf, FileId>::new());
    let tasks = TaskQueue::<Task>::new(threads);
    let ast = Mutex::new(Vec::<io::Result<Ast>>::new());

    let loader = |path: &str, from: &str| {
        let path = ImportPath::try_from(path)?;

        let physiscal_path = path.resolve(from, root)?;

        let id = {
            let mut seen = seen.lock().unwrap();
            let len = seen.len();
            match seen.entry(physiscal_path.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    return Ok(Some(*entry.get()));
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(len as _);
                    len as FileId
                }
            }
        };

        if physiscal_path.exists() {
            tasks.push(Task::LoadFile { id, physiscal_path });
            return Ok(Some(id));
        }

        let ImportPath::Git {
            link,
            path,
            branch,
            rev,
            tag,
        } = path
        else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("can't find file: {}", physiscal_path.display()),
            ));
        };

        let root = PathBuf::from_iter([root, GIT_DEPS_DIR, preprocess_git(link)]);

        let mut command = std::process::Command::new("git");
        command
            .args(["clone", "--depth", "1"])
            .args(branch.map(|b| ["--branch", b]).into_iter().flatten())
            .args(tag.map(|t| ["--tag", t]).into_iter().flatten())
            .args(rev.map(|r| ["--rev", r]).into_iter().flatten())
            .arg(link)
            .arg(root);

        tasks.push(Task::FetchGit {
            id,
            physiscal_path,
            command,
        });

        Ok(Some(id))
    };

    let load_from_path = |path: &Path, buffer: &mut Vec<u8>| -> io::Result<Ast> {
        let path = path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "path contains invalid characters",
            )
        })?;
        let mut file = std::fs::File::open(&path)?;
        file.read_to_end(buffer)?;
        let src = std::str::from_utf8(buffer).map_err(InvalidFileData)?;
        Ok(Ast::new(&path, src, &loader))
    };

    let execute_task = |task: Task, buffer: &mut Vec<u8>| match task {
        Task::LoadFile { id, physiscal_path } => (id, load_from_path(&physiscal_path, buffer)),
        Task::FetchGit {
            id,
            physiscal_path,
            mut command,
        } => {
            let output = match command.output() {
                Ok(output) => output,
                Err(e) => return (id, Err(e)),
            };
            if !output.status.success() {
                let msg = format!(
                    "git command failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return (id, Err(io::Error::new(io::ErrorKind::Other, msg)));
            }
            (id, load_from_path(&physiscal_path, buffer))
        }
    };

    let thread = || {
        let mut buffer = Vec::new();
        while let Some(task) = tasks.pop() {
            let (indx, res) = execute_task(task, &mut buffer);

            let mut ast = ast.lock().unwrap();
            let len = ast.len().max(indx as usize + 1);
            ast.resize_with(len, || Err(io::ErrorKind::InvalidData.into()));
            ast[indx as usize] = res;

            buffer.clear();
        }
    };

    std::thread::scope(|s| (0..threads).for_each(|_| _ = s.spawn(thread)));

    ast.into_inner()
        .unwrap()
        .into_iter()
        .collect::<io::Result<Vec<_>>>()
}

pub fn ident_flag_index(flag: IdentFlags) -> u32 {
    flag & !(MUTABLE | REFERENCED)
}

pub fn no_loader(_: &str, _: &str) -> io::Result<Option<FileId>> {
    Ok(None)
}

pub struct Symbol {
    pub name:  Ident,
    pub flags: IdentFlags,
}

struct ScopeIdent {
    ident:    Ident,
    declared: bool,
    flags:    IdentFlags,
}

pub struct Parser<'a, 'b> {
    path:    &'b str,
    loader:  Loader<'b>,
    lexer:   Lexer<'b>,
    arena:   &'b Arena<'a>,
    token:   Token,
    idents:  Vec<ScopeIdent>,
    symbols: &'b mut Symbols,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(arena: &'b Arena<'a>, symbols: &'b mut Symbols, loader: Loader<'b>) -> Self {
        let mut lexer = Lexer::new("");
        Self {
            loader,
            token: lexer.next(),
            lexer,
            path: "",
            arena,
            idents: Vec::new(),
            symbols,
        }
    }

    pub fn file(&mut self, input: &'b str, path: &'b str) -> &'a [Expr<'a>] {
        self.path = path;
        self.lexer = Lexer::new(input);
        self.token = self.lexer.next();

        let f = self.collect_list(TokenKind::Semi, TokenKind::Eof, Self::expr);

        self.pop_scope(0);
        let has_undeclared = !self.idents.is_empty();
        for id in self.idents.drain(..) {
            let (line, col) = self.lexer.line_col(ident::pos(id.ident));
            eprintln!(
                "{}:{}:{} => undeclared identifier: {}",
                self.path,
                line,
                col,
                self.lexer.slice(ident::range(id.ident))
            );
        }

        if has_undeclared {
            // TODO: we need error recovery
            unreachable!();
        }

        f
    }

    fn next(&mut self) -> Token {
        std::mem::replace(&mut self.token, self.lexer.next())
    }

    fn ptr_expr(&mut self) -> &'a Expr<'a> {
        self.arena.alloc(self.expr())
    }

    fn expr(&mut self) -> Expr<'a> {
        let left = self.unit_expr();
        self.bin_expr(left, 0)
    }

    fn bin_expr(&mut self, mut fold: Expr<'a>, min_prec: u8) -> Expr<'a> {
        loop {
            let Some(prec) = self.token.kind.precedence() else {
                break;
            };

            if prec <= min_prec {
                break;
            }

            let op = self.next().kind;
            let right = self.unit_expr();
            let right = self.bin_expr(right, prec);
            let right = &*self.arena.alloc(right);
            let left = &*self.arena.alloc(fold);

            if let Some(op) = op.assign_op() {
                self.flag_idents(*left, MUTABLE);
                let right = Expr::BinOp { left, op, right };
                fold = Expr::BinOp {
                    left,
                    op: TokenKind::Assign,
                    right: self.arena.alloc(right),
                };
            } else {
                fold = Expr::BinOp { left, right, op };
                if op == TokenKind::Assign {
                    self.flag_idents(*left, MUTABLE);
                }
            }
        }

        fold
    }

    fn try_resolve_builtin(name: &str) -> Option<Ident> {
        // FIXME: we actually do this the second time in the codegen
        Some(match name {
            "int" | "i64" => bt::INT,
            "i8" => bt::I8,
            "i16" => bt::I16,
            "i32" => bt::I32,
            "u8" => bt::U8,
            "u16" => bt::U16,
            "uint" | "u32" => bt::U32,
            "bool" => bt::BOOL,
            "void" => bt::VOID,
            "never" => bt::NEVER,
            _ => return None,
        })
    }

    fn resolve_ident(&mut self, token: Token, decl: bool) -> (Ident, u32) {
        let name = self.lexer.slice(token.range());

        if let Some(builtin) = Self::try_resolve_builtin(name) {
            return (builtin, 0);
        }

        let id = match self
            .idents
            .iter_mut()
            .rfind(|elem| self.lexer.slice(ident::range(elem.ident)) == name)
        {
            Some(elem) if decl && elem.declared => {
                self.report(format_args!("redeclaration of identifier: {name}"))
            }
            Some(elem) => {
                elem.flags += 1;
                elem
            }
            None => {
                let id = ident::new(token.start, name.len() as _);
                self.idents.push(ScopeIdent {
                    ident:    id,
                    declared: false,
                    flags:    0,
                });
                self.idents.last_mut().unwrap()
            }
        };

        id.declared |= decl;

        (id.ident, ident_flag_index(id.flags))
    }

    fn move_str(&mut self, range: Token) -> &'a str {
        self.arena.alloc_str(self.lexer.slice(range.range()))
    }

    fn unit_expr(&mut self) -> Expr<'a> {
        use {Expr as E, TokenKind as T};
        let frame = self.idents.len();
        let token = self.next();
        let mut expr = match token.kind {
            T::Driective => E::Directive {
                pos:  token.start,
                name: self.move_str(token),
                args: {
                    self.expect_advance(T::LParen);
                    self.collect_list(T::Comma, T::RParen, Self::expr)
                },
            },
            T::True => E::Bool {
                pos:   token.start,
                value: true,
            },
            T::Struct => E::Struct {
                pos:    token.start,
                fields: {
                    self.expect_advance(T::LBrace);
                    self.collect_list(T::Comma, T::RBrace, |s| {
                        let name = s.expect_advance(T::Ident);
                        s.expect_advance(T::Colon);
                        let ty = s.expr();
                        (s.move_str(name), ty)
                    })
                },
            },
            T::Ident => {
                let (id, index) = self.resolve_ident(token, self.token.kind == T::Decl);
                let name = self.move_str(token);
                E::Ident { name, id, index }
            }
            T::If => E::If {
                pos:   token.start,
                cond:  self.ptr_expr(),
                then:  self.ptr_expr(),
                else_: self.advance_if(T::Else).then(|| self.ptr_expr()),
            },
            T::Loop => E::Loop {
                pos:  token.start,
                body: self.ptr_expr(),
            },
            T::Break => E::Break { pos: token.start },
            T::Continue => E::Continue { pos: token.start },
            T::Return => E::Return {
                pos: token.start,
                val: (self.token.kind != T::Semi).then(|| self.ptr_expr()),
            },
            T::Fn => E::Closure {
                pos:  token.start,
                args: {
                    self.expect_advance(T::LParen);
                    self.collect_list(T::Comma, T::RParen, |s| {
                        let name = s.expect_advance(T::Ident);
                        let (id, index) = s.resolve_ident(name, true);
                        s.expect_advance(T::Colon);
                        Arg {
                            name: s.move_str(name),
                            id,
                            index,
                            ty: s.expr(),
                        }
                    })
                },
                ret:  {
                    self.expect_advance(T::Colon);
                    self.ptr_expr()
                },
                body: self.ptr_expr(),
            },
            T::Band | T::Mul => E::UnOp {
                pos: token.start,
                op:  token.kind,
                val: {
                    let expr = self.ptr_unit_expr();
                    self.flag_idents(*expr, REFERENCED);
                    expr
                },
            },
            T::LBrace => E::Block {
                pos:   token.start,
                stmts: self.collect_list(T::Semi, T::RBrace, Self::expr),
            },
            T::Number => E::Number {
                pos:   token.start,
                value: match self.lexer.slice(token.range()).parse() {
                    Ok(value) => value,
                    Err(e) => self.report(format_args!("invalid number: {e}")),
                },
            },
            T::LParen => {
                let expr = self.expr();
                self.expect_advance(T::RParen);
                expr
            }
            tok => self.report(format_args!("unexpected token: {tok:?}")),
        };

        loop {
            let token = self.token;
            if matches!(token.kind, T::LParen | T::Ctor | T::Dot | T::Tupl) {
                self.next();
            }

            expr = match token.kind {
                T::LParen => Expr::Call {
                    func: self.arena.alloc(expr),
                    args: self.collect_list(T::Comma, T::RParen, Self::expr),
                },
                T::Ctor => E::Ctor {
                    pos:    token.start,
                    ty:     Some(self.arena.alloc(expr)),
                    fields: self.collect_list(T::Comma, T::RBrace, |s| {
                        let name = s.expect_advance(T::Ident);
                        s.expect_advance(T::Colon);
                        let val = s.expr();
                        (Some(s.move_str(name)), val)
                    }),
                },
                T::Tupl => E::Ctor {
                    pos:    token.start,
                    ty:     Some(self.arena.alloc(expr)),
                    fields: self.collect_list(T::Comma, T::RParen, |s| (None, s.expr())),
                },
                T::Dot => E::Field {
                    target: self.arena.alloc(expr),
                    field:  {
                        let token = self.expect_advance(T::Ident);
                        self.move_str(token)
                    },
                },
                _ => break,
            }
        }

        if matches!(token.kind, T::Return) {
            self.expect_advance(T::Semi);
        }

        if matches!(token.kind, T::Loop | T::LBrace | T::Fn) {
            self.pop_scope(frame);
        }

        expr
    }

    fn pop_scope(&mut self, frame: usize) {
        let mut undeclared_count = frame;
        for i in frame..self.idents.len() {
            if !self.idents[i].declared {
                self.idents.swap(i, undeclared_count);
                undeclared_count += 1;
            }
        }

        self.idents
            .drain(undeclared_count..)
            .map(|ident| Symbol {
                name:  ident.ident,
                flags: ident.flags,
            })
            .collect_into(self.symbols);
    }

    fn ptr_unit_expr(&mut self) -> &'a Expr<'a> {
        self.arena.alloc(self.unit_expr())
    }

    fn collect_list<T: Copy>(
        &mut self,
        delim: TokenKind,
        end: TokenKind,
        mut f: impl FnMut(&mut Self) -> T,
    ) -> &'a [T] {
        self.collect(|s| {
            s.advance_if(end).not().then(|| {
                let val = f(s);
                s.advance_if(delim);
                val
            })
        })
    }

    fn collect<T: Copy>(&mut self, mut f: impl FnMut(&mut Self) -> Option<T>) -> &'a [T] {
        let vec = std::iter::from_fn(|| f(self)).collect::<Vec<_>>();
        self.arena.alloc_slice(&vec)
    }

    fn advance_if(&mut self, kind: TokenKind) -> bool {
        if self.token.kind == kind {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_advance(&mut self, kind: TokenKind) -> Token {
        if self.token.kind != kind {
            self.report(format_args!(
                "expected {:?}, found {:?}",
                kind, self.token.kind
            ));
        }
        self.next()
    }

    fn report(&self, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.lexer.line_col(self.token.start);
        eprintln!("{}:{}:{} => {}", self.path, line, col, msg);
        unreachable!();
    }

    fn flag_idents(&mut self, e: Expr<'a>, flags: IdentFlags) {
        match e {
            Expr::Ident { id, .. } => find_ident(&mut self.idents, id).flags |= flags,
            Expr::Field { target, .. } => self.flag_idents(*target, flags),
            _ => {}
        }
    }
}

fn find_ident(idents: &mut [ScopeIdent], id: Ident) -> &mut ScopeIdent {
    idents
        .binary_search_by_key(&id, |si| si.ident)
        .map(|i| &mut idents[i])
        .unwrap()
}

pub fn find_symbol(symbols: &[Symbol], id: Ident) -> &Symbol {
    symbols
        .binary_search_by_key(&id, |s| s.name)
        .map(|i| &symbols[i])
        .unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arg<'a> {
    pub name:  &'a str,
    pub id:    Ident,
    pub index: u32,
    pub ty:    Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expr<'a> {
    Break {
        pos: Pos,
    },
    Continue {
        pos: Pos,
    },
    Closure {
        pos:  Pos,
        args: &'a [Arg<'a>],
        ret:  &'a Self,
        body: &'a Self,
    },
    Call {
        func: &'a Self,
        args: &'a [Self],
    },
    Return {
        pos: Pos,
        val: Option<&'a Self>,
    },
    Ident {
        name:  &'a str,
        id:    Ident,
        index: u32,
    },
    Block {
        pos:   Pos,
        stmts: &'a [Self],
    },
    Number {
        pos:   Pos,
        value: u64,
    },
    BinOp {
        left:  &'a Self,
        op:    TokenKind,
        right: &'a Self,
    },
    If {
        pos:   Pos,
        cond:  &'a Self,
        then:  &'a Self,
        else_: Option<&'a Self>,
    },
    Loop {
        pos:  Pos,
        body: &'a Self,
    },
    UnOp {
        pos: Pos,
        op:  TokenKind,
        val: &'a Self,
    },
    Struct {
        pos:    Pos,
        fields: &'a [(&'a str, Self)],
    },
    Ctor {
        pos:    Pos,
        ty:     Option<&'a Self>,
        fields: &'a [(Option<&'a str>, Self)],
    },
    Field {
        target: &'a Self,
        field:  &'a str,
    },
    Bool {
        pos:   Pos,
        value: bool,
    },
    Directive {
        pos:  u32,
        name: &'a str,
        args: &'a [Self],
    },
}

impl<'a> Expr<'a> {
    pub fn pos(&self) -> Pos {
        match self {
            Self::Call { func, .. } => func.pos(),
            Self::Ident { id, .. } => ident::pos(*id),
            Self::Break { pos }
            | Self::Directive { pos, .. }
            | Self::Continue { pos }
            | Self::Closure { pos, .. }
            | Self::Block { pos, .. }
            | Self::Number { pos, .. }
            | Self::Return { pos, .. }
            | Self::If { pos, .. }
            | Self::Loop { pos, .. }
            | Self::UnOp { pos, .. }
            | Self::Struct { pos, .. }
            | Self::Ctor { pos, .. }
            | Self::Bool { pos, .. } => *pos,
            Self::BinOp { left, .. } => left.pos(),
            Self::Field { target, .. } => target.pos(),
        }
    }
}

impl<'a> std::fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        thread_local! {
            static INDENT: Cell<usize> = Cell::new(0);
        }

        fn fmt_list<'a, T>(
            f: &mut std::fmt::Formatter,
            end: &str,
            list: &'a [T],
            fmt: impl Fn(&T, &mut std::fmt::Formatter) -> std::fmt::Result,
        ) -> std::fmt::Result {
            let first = &mut true;
            for expr in list {
                if !std::mem::take(first) {
                    write!(f, ", ")?;
                }
                fmt(expr, f)?;
            }
            write!(f, "{end}")
        }

        macro_rules! impl_parenter {
            ($($name:ident => $pat:pat,)*) => {
                $(
                    struct $name<'a>(&'a Expr<'a>);

                    impl<'a> std::fmt::Display for $name<'a> {
                        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                            if matches!(self.0, $pat) {
                                write!(f, "({})", self.0)
                            } else {
                                write!(f, "{}", self.0)
                            }
                        }
                    }
                )*
            };
        }

        impl_parenter! {
            Unary => Expr::BinOp { .. },
            Postfix => Expr::UnOp { .. } | Expr::BinOp { .. },
            Consecutive => Expr::UnOp { .. },
        }

        match *self {
            Self::Field { target, field } => {
                write!(f, "{}.{field}", Postfix(target))
            }
            Self::Directive { name, args, .. } => {
                write!(f, "@{name}(")?;
                fmt_list(f, ")", args, std::fmt::Display::fmt)
            }
            Self::Struct { fields, .. } => {
                write!(f, "struct {{")?;
                fmt_list(f, "}", fields, |(name, val), f| write!(f, "{name}: {val}",))
            }
            Self::Ctor { ty, fields, .. } => {
                let (left, rith) = if fields.iter().any(|(name, _)| name.is_some()) {
                    ('{', '}')
                } else {
                    ('(', ')')
                };

                if let Some(ty) = ty {
                    write!(f, "{}", Unary(ty))?;
                }
                write!(f, ".{left}")?;
                let first = &mut true;
                for (name, val) in fields {
                    if !std::mem::take(first) {
                        write!(f, ", ")?;
                    }
                    if let Some(name) = name {
                        write!(f, "{name}: ")?;
                    }
                    write!(f, "{val}")?;
                }
                write!(f, "{rith}")
            }
            Self::UnOp { op, val, .. } => write!(f, "{op}{}", Unary(val)),
            Self::Break { .. } => write!(f, "break;"),
            Self::Continue { .. } => write!(f, "continue;"),
            Self::If {
                cond, then, else_, ..
            } => {
                write!(f, "if {cond} {}", Consecutive(then))?;
                if let Some(else_) = else_ {
                    write!(f, " else {else_}")?;
                }
                Ok(())
            }
            Self::Loop { body, .. } => write!(f, "loop {body}"),
            Self::Closure {
                ret, body, args, ..
            } => {
                write!(f, "fn(")?;
                fmt_list(f, "", args, |arg, f| write!(f, "{}: {}", arg.name, arg.ty))?;
                write!(f, "): {ret} {body}")
            }
            Self::Call { func, args } => {
                write!(f, "{}(", Postfix(func))?;
                fmt_list(f, ")", args, std::fmt::Display::fmt)
            }
            Self::Return { val: Some(val), .. } => write!(f, "return {val};"),
            Self::Return { val: None, .. } => write!(f, "return;"),
            Self::Ident { name, .. } => write!(f, "{name}"),
            Self::Block { stmts, .. } => {
                writeln!(f, "{{")?;
                INDENT.with(|i| i.set(i.get() + 1));
                let res = (|| {
                    for stmt in stmts {
                        for _ in 0..INDENT.with(|i| i.get()) {
                            write!(f, "    ")?;
                        }
                        writeln!(f, "{stmt}")?;
                    }
                    Ok(())
                })();
                INDENT.with(|i| i.set(i.get() - 1));
                write!(f, "}}")?;
                res
            }
            Self::Number { value, .. } => write!(f, "{value}"),
            Self::Bool { value, .. } => write!(f, "{value}"),
            Self::BinOp { left, right, op } => {
                let display_branch = |f: &mut std::fmt::Formatter, expr: &Self| {
                    if let Self::BinOp { op: lop, .. } = expr
                        && op.precedence() > lop.precedence()
                    {
                        write!(f, "({expr})")
                    } else {
                        write!(f, "{expr}")
                    }
                };

                display_branch(f, left)?;
                write!(f, " {op} ")?;
                display_branch(f, right)
            }
        }
    }
}

#[repr(C)]
struct AstInner<T: ?Sized> {
    ref_count: AtomicUsize,
    mem:       ArenaChunk,
    exprs:     *const [Expr<'static>],
    path:      String,
    symbols:   T,
}

impl AstInner<[Symbol]> {
    fn layout(syms: usize) -> std::alloc::Layout {
        std::alloc::Layout::new::<AstInner<()>>()
            .extend(std::alloc::Layout::array::<Symbol>(syms).unwrap())
            .unwrap()
            .0
    }

    fn new(content: &str, path: &str, loader: Loader) -> NonNull<Self> {
        let arena = Arena::default();
        let mut syms = Vec::new();
        let mut parser = Parser::new(&arena, &mut syms, loader);
        let exprs = parser.file(content, path) as *const [Expr<'static>];

        syms.sort_unstable_by_key(|s| s.name);

        let layout = Self::layout(syms.len());

        let ptr = unsafe { std::alloc::alloc(layout) };
        let inner: *mut Self = std::ptr::from_raw_parts_mut(ptr as *mut _, syms.len());
        unsafe {
            *(inner as *mut AstInner<()>) = AstInner {
                ref_count: AtomicUsize::new(1),
                mem: ArenaChunk::default(),
                exprs,
                path: path.to_owned(),
                symbols: (),
            };
            std::ptr::addr_of_mut!((*inner).symbols)
                .as_mut_ptr()
                .copy_from_nonoverlapping(syms.as_ptr(), syms.len());
            NonNull::new_unchecked(inner)
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct Ast(NonNull<AstInner<[Symbol]>>);

impl Ast {
    pub fn new(path: &str, content: &str, loader: Loader) -> Self {
        Self(AstInner::new(content, path, loader))
    }

    pub fn exprs(&self) -> &[Expr] {
        unsafe { &*self.inner().exprs }
    }

    pub fn symbols(&self) -> &[Symbol] {
        &self.inner().symbols
    }

    pub fn path(&self) -> &str {
        &self.inner().path
    }

    fn inner(&self) -> &AstInner<[Symbol]> {
        unsafe { self.0.as_ref() }
    }
}

unsafe impl Send for Ast {}
unsafe impl Sync for Ast {}

impl Clone for Ast {
    fn clone(&self) -> Self {
        unsafe { self.0.as_ref() }
            .ref_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(self.0)
    }
}

impl Drop for Ast {
    fn drop(&mut self) {
        let inner = unsafe { self.0.as_ref() };
        if inner
            .ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
            == 1
        {
            unsafe { std::ptr::drop_in_place(self.0.as_ptr()) };

            let layout = AstInner::layout(inner.symbols.len());
            unsafe {
                std::alloc::dealloc(self.0.as_ptr() as _, layout);
            }
        }
    }
}

#[derive(Default)]
pub struct Arena<'a> {
    chunk: UnsafeCell<ArenaChunk>,
    ph:    std::marker::PhantomData<&'a ()>,
}

impl<'a> Arena<'a> {
    pub fn alloc_str(&self, token: &str) -> &'a str {
        let ptr = self.alloc_slice(token.as_bytes());
        unsafe { std::str::from_utf8_unchecked_mut(ptr) }
    }

    pub fn alloc<T>(&self, value: T) -> &'a mut T {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = self.alloc_low(layout);
        unsafe { ptr.cast::<T>().write(value) };
        unsafe { ptr.cast::<T>().as_mut() }
    }

    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &'a mut [T] {
        let layout = std::alloc::Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_low(layout);
        unsafe {
            ptr.as_ptr()
                .cast::<T>()
                .copy_from_nonoverlapping(slice.as_ptr(), slice.len())
        };
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as _, slice.len()) }
    }

    pub fn clear(&mut self) {
        let chunk = self.chunk.get_mut();
        if chunk.base.is_null() {
            return;
        }

        loop {
            let prev = ArenaChunk::prev(chunk.base);
            if prev.is_null() {
                break;
            }
            chunk.base = prev;
        }

        chunk.end = unsafe { chunk.base.add(ArenaChunk::PREV_OFFSET) };
    }

    fn alloc_low(&self, layout: std::alloc::Layout) -> NonNull<u8> {
        assert!(layout.align() <= ArenaChunk::ALIGN);
        assert!(layout.size() <= ArenaChunk::CHUNK_SIZE);

        let chunk = unsafe { &mut *self.chunk.get() };

        if let Some(ptr) = chunk.alloc(layout) {
            return ptr;
        }

        if let Some(prev) = ArenaChunk::reset(ArenaChunk::prev(chunk.base)) {
            *chunk = prev;
        } else {
            *chunk = ArenaChunk::new(chunk.base);
        }

        chunk.alloc(layout).unwrap()
    }
}

struct ArenaChunk {
    base: *mut u8,
    end:  *mut u8,
}

impl Default for ArenaChunk {
    fn default() -> Self {
        Self {
            base: std::ptr::null_mut(),
            end:  std::ptr::null_mut(),
        }
    }
}

impl ArenaChunk {
    const CHUNK_SIZE: usize = 1 << 16;
    const ALIGN: usize = std::mem::align_of::<Self>();
    const NEXT_OFFSET: usize = Self::CHUNK_SIZE - std::mem::size_of::<*mut u8>();
    const PREV_OFFSET: usize = Self::NEXT_OFFSET - std::mem::size_of::<*mut u8>();
    const LAYOUT: std::alloc::Layout =
        unsafe { std::alloc::Layout::from_size_align_unchecked(Self::CHUNK_SIZE, Self::ALIGN) };

    fn new(next: *mut u8) -> Self {
        let base = unsafe { std::alloc::alloc(Self::LAYOUT) };
        let end = unsafe { base.add(Self::PREV_OFFSET) };
        if !next.is_null() {
            Self::set_prev(next, base);
        }
        Self::set_next(base, next);
        Self::set_prev(base, std::ptr::null_mut());
        Self { base, end }
    }

    fn set_next(curr: *mut u8, next: *mut u8) {
        unsafe { std::ptr::write(curr.add(Self::NEXT_OFFSET) as *mut _, next) };
    }

    fn set_prev(curr: *mut u8, prev: *mut u8) {
        unsafe { std::ptr::write(curr.add(Self::PREV_OFFSET) as *mut _, prev) };
    }

    fn next(curr: *mut u8) -> *mut u8 {
        unsafe { std::ptr::read(curr.add(Self::NEXT_OFFSET) as *mut _) }
    }

    fn prev(curr: *mut u8) -> *mut u8 {
        if curr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { std::ptr::read(curr.add(Self::PREV_OFFSET) as *mut _) }
    }

    fn reset(prev: *mut u8) -> Option<Self> {
        if prev.is_null() {
            return None;
        }

        Some(Self {
            base: prev,
            end:  unsafe { prev.add(Self::CHUNK_SIZE) },
        })
    }

    fn alloc(&mut self, layout: std::alloc::Layout) -> Option<NonNull<u8>> {
        let padding = self.end as usize - (self.end as usize & !(layout.align() - 1));
        let size = layout.size() + padding;
        if size > self.end as usize - self.base as usize {
            return None;
        }
        unsafe { self.end = self.end.sub(size) };
        unsafe { Some(NonNull::new_unchecked(self.end)) }
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        let mut current = self.base;

        let mut prev = Self::prev(current);
        while !prev.is_null() {
            let next = Self::prev(prev);
            unsafe { std::alloc::dealloc(prev, Self::LAYOUT) };
            prev = next;
        }

        while !current.is_null() {
            let next = Self::next(current);
            unsafe { std::alloc::dealloc(current, Self::LAYOUT) };
            current = next;
        }
    }
}

#[cfg(test)]
mod tests {

    fn parse(input: &'static str, output: &mut String) {
        use std::fmt::Write;
        let mut arena = super::Arena::default();
        let mut symbols = Vec::new();
        let mut parser = super::Parser::new(&arena, &mut symbols, &super::no_loader);
        for expr in parser.file(input, "test") {
            writeln!(output, "{}", expr).unwrap();
        }
        arena.clear();
    }

    crate::run_tests! { parse:
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
    }
}
