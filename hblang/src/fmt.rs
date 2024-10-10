use {
    crate::{
        lexer::{self, TokenKind},
        parser::{self, CommentOr, CtorField, Expr, Poser, Radix, StructField},
    },
    core::fmt,
};

pub fn minify(source: &mut str) -> usize {
    fn needs_space(c: u8) -> bool {
        matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | 127..)
    }

    let mut writer = source.as_mut_ptr();
    let mut reader = &source[..];
    let mut prev_needs_whitecpace = false;
    let mut prev_needs_newline = false;
    loop {
        let mut token = lexer::Lexer::new(reader).next();
        match token.kind {
            TokenKind::Eof => break,
            TokenKind::CtIdent | TokenKind::Directive => token.start -= 1,
            _ => {}
        }

        let cpy_len = token.range().len();

        let mut prefix = 0;
        if prev_needs_whitecpace && needs_space(reader.as_bytes()[token.start as usize]) {
            prefix = b' ';
            debug_assert!(token.start != 0, "{reader}");
        }
        prev_needs_whitecpace = needs_space(reader.as_bytes()[token.end as usize - 1]);

        let inbetween_new_lines =
            reader[..token.start as usize].bytes().filter(|&b| b == b'\n').count()
                + token.kind.precedence().is_some() as usize;
        let extra_prefix_new_lines = if inbetween_new_lines > 1 {
            1 + token.kind.precedence().is_none() as usize
        } else {
            prev_needs_newline as usize
        };

        if token.kind == TokenKind::Comment && reader.as_bytes()[token.end as usize - 1] != b'/' {
            prev_needs_newline = true;
            prev_needs_whitecpace = false;
        } else {
            prev_needs_newline = false;
        }

        let sstr = reader[token.start as usize..].as_ptr();
        reader = &reader[token.end as usize..];
        unsafe {
            if extra_prefix_new_lines != 0 {
                for _ in 0..extra_prefix_new_lines {
                    writer.write(b'\n');
                    writer = writer.add(1);
                }
            } else if prefix != 0 {
                writer.write(prefix);
                writer = writer.add(1);
            }
            writer.copy_from(sstr, cpy_len);
            writer = writer.add(cpy_len);
        }
    }

    unsafe { writer.sub_ptr(source.as_mut_ptr()) }
}

pub struct Formatter<'a> {
    source: &'a str,
    depth: usize,
}

impl<'a> Formatter<'a> {
    pub fn new(source: &'a str) -> Self {
        Self { source, depth: 0 }
    }

    fn fmt_list<T: Poser, F: core::fmt::Write>(
        &mut self,
        f: &mut F,
        trailing: bool,
        end: &str,
        sep: &str,
        list: &[T],
        fmt: impl Fn(&mut Self, &T, &mut F) -> fmt::Result,
    ) -> fmt::Result {
        self.fmt_list_low(f, trailing, end, sep, list, |s, v, f| {
            fmt(s, v, f)?;
            Ok(true)
        })
    }

    fn fmt_list_low<T: Poser, F: core::fmt::Write>(
        &mut self,
        f: &mut F,
        trailing: bool,
        end: &str,
        sep: &str,
        list: &[T],
        fmt: impl Fn(&mut Self, &T, &mut F) -> Result<bool, fmt::Error>,
    ) -> fmt::Result {
        if !trailing {
            let mut first = true;
            for expr in list {
                if !core::mem::take(&mut first) {
                    write!(f, "{sep} ")?;
                }
                first = !fmt(self, expr, f)?;
            }
            return write!(f, "{end}");
        }

        writeln!(f)?;
        self.depth += 1;
        let res = (|| {
            for (i, stmt) in list.iter().enumerate() {
                for _ in 0..self.depth {
                    write!(f, "\t")?;
                }
                let add_sep = fmt(self, stmt, f)?;
                if add_sep {
                    write!(f, "{sep}")?;
                }
                if let Some(expr) = list.get(i + 1)
                    && let Some(rest) = self.source.get(expr.posi() as usize..)
                {
                    if insert_needed_semicolon(rest) {
                        write!(f, ";")?;
                    }
                    if preserve_newlines(&self.source[..expr.posi() as usize]) > 1 {
                        writeln!(f)?;
                    }
                }
                if add_sep {
                    writeln!(f)?;
                }
            }
            Ok(())
        })();
        self.depth -= 1;

        for _ in 0..self.depth {
            write!(f, "\t")?;
        }
        write!(f, "{end}")?;
        res
    }

    fn fmt_paren<F: core::fmt::Write>(
        &mut self,
        expr: &Expr,
        f: &mut F,
        cond: impl FnOnce(&Expr) -> bool,
    ) -> fmt::Result {
        if cond(expr) {
            write!(f, "(")?;
            self.fmt(expr, f)?;
            write!(f, ")")
        } else {
            self.fmt(expr, f)
        }
    }

    pub fn fmt<F: core::fmt::Write>(&mut self, expr: &Expr, f: &mut F) -> fmt::Result {
        macro_rules! impl_parenter {
            ($($name:ident => $pat:pat,)*) => {
                $(
                    let $name = |e: &Expr| matches!(e, $pat);
                )*
            };
        }

        impl_parenter! {
            unary => Expr::BinOp { .. },
            postfix => Expr::UnOp { .. } | Expr::BinOp { .. },
            consecutive => Expr::UnOp { .. },
        }

        match *expr {
            Expr::Ct { value, .. } => {
                write!(f, "$: ")?;
                self.fmt(value, f)
            }
            Expr::String { literal, .. } => write!(f, "{literal}"),
            Expr::Comment { literal, .. } => write!(f, "{literal}"),
            Expr::Mod { path, .. } => write!(f, "@use(\"{path}\")"),
            Expr::Field { target, name: field, .. } => {
                self.fmt_paren(target, f, postfix)?;
                write!(f, ".{field}")
            }
            Expr::Directive { name, args, .. } => {
                write!(f, "@{name}(")?;
                self.fmt_list(f, false, ")", ",", args, Self::fmt)
            }
            Expr::Struct { fields, trailing_comma, packed, .. } => {
                if packed {
                    write!(f, "packed ")?;
                }

                write!(f, "struct {{")?;
                self.fmt_list_low(f, trailing_comma, "}", ",", fields, |s, field, f| {
                    match field {
                        CommentOr::Or(StructField { name, ty, .. }) => {
                            write!(f, "{name}: ")?;
                            s.fmt(ty, f)?
                        }
                        CommentOr::Comment { literal, .. } => writeln!(f, "{literal}")?,
                    }
                    Ok(field.or().is_some())
                })
            }
            Expr::Ctor { ty, fields, trailing_comma, .. } => {
                if let Some(ty) = ty {
                    self.fmt_paren(ty, f, unary)?;
                }
                write!(f, ".{{")?;
                self.fmt_list(
                    f,
                    trailing_comma,
                    "}",
                    ",",
                    fields,
                    |s: &mut Self, CtorField { name, value, .. }: &_, f| {
                        if matches!(value, Expr::Ident { name: n, .. } if name == n) {
                            write!(f, "{name}")
                        } else {
                            write!(f, "{name}: ")?;
                            s.fmt(value, f)
                        }
                    },
                )
            }
            Expr::Tupl { ty, fields, trailing_comma, .. } => {
                if let Some(ty) = ty {
                    self.fmt_paren(ty, f, unary)?;
                }
                write!(f, ".(")?;
                self.fmt_list(f, trailing_comma, ")", ",", fields, Self::fmt)
            }
            Expr::Slice { item, size, .. } => {
                write!(f, "[")?;
                self.fmt(item, f)?;
                if let Some(size) = size {
                    write!(f, "; ")?;
                    self.fmt(size, f)?;
                }
                write!(f, "]")
            }
            Expr::Index { base, index } => {
                self.fmt(base, f)?;
                write!(f, "[")?;
                self.fmt(index, f)?;
                write!(f, "]")
            }
            Expr::UnOp { op, val, .. } => {
                write!(f, "{op}")?;
                self.fmt_paren(val, f, unary)
            }
            Expr::Break { .. } => write!(f, "break"),
            Expr::Continue { .. } => write!(f, "continue"),
            Expr::If { cond, then, else_, .. } => {
                write!(f, "if ")?;
                self.fmt(cond, f)?;
                write!(f, " ")?;
                self.fmt_paren(then, f, consecutive)?;
                if let Some(e) = else_ {
                    write!(f, " else ")?;
                    self.fmt(e, f)?;
                }
                Ok(())
            }
            Expr::Loop { body, .. } => {
                write!(f, "loop ")?;
                self.fmt(body, f)
            }
            Expr::Closure { ret, body, args, .. } => {
                write!(f, "fn(")?;
                self.fmt_list(f, false, "", ",", args, |s, arg, f| {
                    if arg.is_ct {
                        write!(f, "$")?;
                    }
                    write!(f, "{}: ", arg.name)?;
                    s.fmt(&arg.ty, f)
                })?;
                write!(f, "): ")?;
                self.fmt(ret, f)?;
                write!(f, " ")?;
                self.fmt_paren(body, f, consecutive)?;
                Ok(())
            }
            Expr::Call { func, args, trailing_comma } => {
                self.fmt_paren(func, f, postfix)?;
                write!(f, "(")?;
                self.fmt_list(f, trailing_comma, ")", ",", args, Self::fmt)
            }
            Expr::Return { val: Some(val), .. } => {
                write!(f, "return ")?;
                self.fmt(val, f)
            }
            Expr::Return { val: None, .. } => write!(f, "return"),
            Expr::Ident { name, is_ct: true, .. } => write!(f, "${name}"),
            Expr::Ident { name, is_ct: false, .. } => write!(f, "{name}"),
            Expr::Block { stmts, .. } => {
                write!(f, "{{")?;
                self.fmt_list(f, true, "}", "", stmts, Self::fmt)
            }
            Expr::Number { value, radix, .. } => {
                fn display_radix(radix: Radix, mut value: u64, buf: &mut [u8; 64]) -> &str {
                    fn conv_radix(d: u8) -> u8 {
                        match d {
                            0..=9 => d + b'0',
                            _ => d - 10 + b'A',
                        }
                    }

                    for (i, b) in buf.iter_mut().enumerate().rev() {
                        let d = (value % radix as u64) as u8;
                        value /= radix as u64;
                        *b = conv_radix(d);
                        if value == 0 {
                            return unsafe { core::str::from_utf8_unchecked(&buf[i..]) };
                        }
                    }

                    unreachable!()
                }
                let mut buf = [0u8; 64];
                let value = display_radix(radix, value as u64, &mut buf);
                match radix {
                    Radix::Decimal => write!(f, "{value}"),
                    Radix::Hex => write!(f, "0x{value}"),
                    Radix::Octal => write!(f, "0o{value}"),
                    Radix::Binary => write!(f, "0b{value}"),
                }
            }
            Expr::Bool { value, .. } => write!(f, "{value}"),
            Expr::Idk { .. } => write!(f, "idk"),
            Expr::BinOp {
                left,
                op: TokenKind::Assign,
                right: &Expr::BinOp { left: lleft, op, right },
            } if left.pos() == lleft.pos() => {
                self.fmt(left, f)?;
                write!(f, " {op}= ")?;
                self.fmt(right, f)
            }
            Expr::BinOp { right, op, left } => {
                let pec_miss = |e: &Expr| {
                    matches!(
                        e, Expr::BinOp { op: lop, .. } if op.precedence() > lop.precedence()
                    )
                };

                self.fmt_paren(left, f, pec_miss)?;
                if let Some(mut prev) = self.source.get(..right.pos() as usize) {
                    prev = prev.trim_end();
                    let estimate_bound =
                        prev.rfind(|c: char| c.is_ascii_whitespace()).map_or(prev.len(), |i| i + 1);
                    let exact_bound = lexer::Lexer::new(&prev[estimate_bound..]).last().start;
                    prev = &prev[..exact_bound as usize + estimate_bound];
                    if preserve_newlines(prev) > 0 {
                        writeln!(f)?;
                        for _ in 0..self.depth + 1 {
                            write!(f, "\t")?;
                        }
                        write!(f, "{op} ")?;
                    } else {
                        write!(f, " {op} ")?;
                    }
                } else {
                    write!(f, " {op} ")?;
                }
                self.fmt_paren(right, f, pec_miss)
            }
        }
    }
}

pub fn preserve_newlines(source: &str) -> usize {
    source[source.trim_end().len()..].bytes().filter(|&c| c == b'\n').count()
}

pub fn insert_needed_semicolon(source: &str) -> bool {
    let kind = lexer::Lexer::new(source).next().kind;
    kind.precedence().is_some() || matches!(kind, TokenKind::Ctor | TokenKind::Tupl)
}

impl core::fmt::Display for parser::Ast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_file(self.exprs(), &self.file, f)
    }
}

pub fn fmt_file(exprs: &[Expr], file: &str, f: &mut impl fmt::Write) -> fmt::Result {
    for (i, expr) in exprs.iter().enumerate() {
        Formatter::new(file).fmt(expr, f)?;
        if let Some(expr) = exprs.get(i + 1)
            && let Some(rest) = file.get(expr.pos() as usize..)
        {
            if insert_needed_semicolon(rest) {
                write!(f, ";")?;
            }

            if preserve_newlines(&file[..expr.pos() as usize]) > 1 {
                writeln!(f)?;
            }
        }

        if i + 1 != exprs.len() {
            writeln!(f)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub mod test {
    use {
        crate::parser::{self, ParserCtx},
        alloc::borrow::ToOwned,
        std::{fmt::Write, string::String},
    };

    pub fn format(ident: &str, input: &str) {
        let mut minned = input.to_owned();
        let len = crate::fmt::minify(&mut minned);
        minned.truncate(len);

        let ast = parser::Ast::new(ident, minned, &mut ParserCtx::default(), &|_, _| Ok(0));
        let mut output = String::new();
        write!(output, "{ast}").unwrap();

        let input_path = format!("formatter_{ident}.expected");
        let output_path = format!("formatter_{ident}.actual");
        std::fs::write(&input_path, input).unwrap();
        std::fs::write(&output_path, output).unwrap();

        let success = std::process::Command::new("diff")
            .arg("-u")
            .arg("--color")
            .arg(&input_path)
            .arg(&output_path)
            .status()
            .unwrap()
            .success();
        std::fs::remove_file(&input_path).unwrap();
        std::fs::remove_file(&output_path).unwrap();
        assert!(success, "test failed");
    }

    macro_rules! test {
        ($($name:ident => $input:expr;)*) => {$(
            #[test]
            fn $name() {
                format(stringify!($name), $input);
            }
        )*};
    }

    test! {
        comments => "// comment\n// comment\n\n// comment\n\n\
            /* comment */\n/* comment */\n\n/* comment */";
        some_ordinary_code => "loft := fn(): int return loft(1, 2, 3)";
        some_arg_per_line_code => "loft := fn(): int return loft(\
            \n\t1,\n\t2,\n\t3,\n)";
        some_ordinary_struct => "loft := fn(): int return loft.{a: 1, b: 2}";
        some_ordinary_fild_per_lin_struct => "loft := fn(): int return loft.{\
            \n\ta: 1,\n\tb: 2,\n}";
        code_block => "loft := fn(): int {\n\tloft()\n\treturn 1\n}";
    }
}
