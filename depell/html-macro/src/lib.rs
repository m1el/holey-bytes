#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]

use std::fmt::Write;
use std::str::FromStr;

macro_rules! diag {
    ($level:ident: $($tt:tt)*) => {diag!($level(proc_macro::Span::call_site()): $($tt)*)};
    ($level:ident[$span:expr]: $($tt:tt)*) => {diag!($level($span.span()): $($tt)*)};
    ($level:ident($span:expr): $($tt:tt)*) => {{
        proc_macro::Diagnostic::spanned($span, proc_macro::Level::$level, format!($($tt)*)).emit();
    }};
}

macro_rules! fatal_diag {
    ($($tt:tt)*) => {{
        diag!($($tt)*);
        return None;
    }};
}

#[proc_macro]
pub fn html(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut trees = input.into_iter().peekable();
    let mut st = State {
        final_code: "Html({ let mut ___out = String::new(); ".to_string(),
        write_target: "___out".to_string(),
        error_handler: ".unwrap()",
        ..State::default()
    };

    if st.output_fmt(&mut trees).is_none() {
        return proc_macro::TokenStream::default();
    };

    st.final_code.push_str("___out })");

    if !st.tag_stack.is_empty() {
        diag!(Error(proc_macro::Span::def_site()): "unclosed tags: {}", st.tag_stack);
    }

    proc_macro::TokenStream::from_str(&st.final_code).unwrap()
}

#[proc_macro]
pub fn write_html(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut trees = input.into_iter().peekable();
    let mut st = State {
        write_target: trees.next().expect("expected first token").to_string(),
        error_handler: ".unwrap()",
        ..State::default()
    };

    if st.output_fmt(&mut trees).is_none() {
        return proc_macro::TokenStream::default();
    };

    if !st.tag_stack.is_empty() {
        diag!(Error(proc_macro::Span::def_site()): "unclosed tags: {}", st.tag_stack);
    }

    proc_macro::TokenStream::from_str(&st.final_code).unwrap()
}

type Trees = std::iter::Peekable<proc_macro::token_stream::IntoIter>;

#[derive(Default)]
struct State {
    final_code: String,
    template: String,
    tstr: String,
    tag_stack: String,
    write_target: String,
    no_escaping: bool,
    error_handler: &'static str,
}

impl State {
    fn flush_template(&mut self) {
        if self.template.is_empty() {
            return;
        }
        write!(
            self.final_code,
            "write!({}, {:?}){};",
            self.write_target, self.template, self.error_handler
        )
        .unwrap();
        self.template.clear();
    }

    fn display_expr(&mut self, expr: impl std::fmt::Display) {
        self.flush_template();
        if std::mem::take(&mut self.no_escaping) {
            write!(
                self.final_code,
                "write!({}, \"{{}}\", {expr}){};",
                self.write_target, self.error_handler
            )
            .unwrap();
        } else {
            write!(
                self.final_code,
                "write!({}, \"{{}}\", &HtmlEscaped(&{expr})){};",
                self.write_target, self.error_handler,
            )
            .unwrap()
        }
    }

    fn expect_group(
        &mut self,
        trees: &mut Trees,
        delim: proc_macro::Delimiter,
    ) -> Option<proc_macro::TokenStream> {
        match trees.next() {
            Some(proc_macro::TokenTree::Group(g)) if g.delimiter() == delim => Some(g.stream()),
            Some(c) => fatal_diag!(Error[c]: "expected {delim:?} group"),
            None => fatal_diag!(Error: "expected {delim:?} group, got eof"),
        }
    }

    fn expect_punct(&mut self, trees: &mut Trees, c: char) -> Option<()> {
        match trees.next() {
            Some(proc_macro::TokenTree::Punct(p)) if p.as_char() == c => Some(()),
            Some(c) => fatal_diag!(Error[c]: "expected {c}"),
            None => fatal_diag!(Error: "expected {c}, got eof"),
        }
    }

    fn tag(&mut self, p: proc_macro::Punct, trees: &mut Trees) -> Option<()> {
        let temp = match trees.next() {
            Some(proc_macro::TokenTree::Ident(id)) => {
                let temp = temp_str(&id, &mut self.tstr);
                if !is_html_tag(temp) {
                    diag!(Error[p]: "expected html5 tag name");
                }
                temp
            }
            Some(proc_macro::TokenTree::Literal(lit)) => {
                let temp = temp_str(&lit, &mut self.tstr).trim_matches('"');
                if is_html_tag(temp) {
                    if !temp.contains('!') {
                        diag!(Warning[p]: "unnescessary string escaping");
                    }
                } else if !is_valid_webcomponent(temp) {
                    diag!(Error[p]: "invalid web component identifier");
                }
                temp
            }
            Some(proc_macro::TokenTree::Punct(p)) if p.as_char() == '/' => {
                let Some((_, top)) = self.tag_stack.rsplit_once(',') else {
                    fatal_diag!(Error[p]: "no tag to close");
                };

                let new_tag_stack_len = self.tag_stack.len() - top.len() - 1;

                let (temp, span) = match trees.next() {
                    Some(proc_macro::TokenTree::Ident(id)) => {
                        (temp_str(&id, &mut self.tstr), id.span())
                    }
                    Some(proc_macro::TokenTree::Literal(lit)) => {
                        (temp_str(&lit, &mut self.tstr).trim_matches('"'), lit.span())
                    }
                    Some(proc_macro::TokenTree::Punct(p)) if p.as_char() == '>' => {
                        // easter egg
                        write!(&mut self.template, "</{top}>").unwrap();
                        self.tag_stack.truncate(new_tag_stack_len);
                        return Some(());
                    }
                    Some(c) => fatal_diag!(Error[c]: "unexpected token in closing tag"),
                    None => {
                        fatal_diag!(Error[p]: "expected tag ident or string or '>'")
                    }
                };

                if temp != top {
                    diag!(Error(span): "expected closing '{top}' tag");
                }

                write!(&mut self.template, "</{top}>").unwrap();
                self.expect_punct(trees, '>')?;
                self.tag_stack.truncate(new_tag_stack_len);
                return Some(());
            }
            _ => fatal_diag!(Error[p]: "expected tag ident or string literal"),
        };

        write!(&mut self.template, "<{temp}").unwrap();
        if !is_self_closing(temp) {
            write!(self.tag_stack, ",{temp}").unwrap();
        }

        let mut has_attr = false;
        while let Some(c) = trees.next() {
            let mut has_attr_tmp = false;
            match c {
                proc_macro::TokenTree::Punct(p) if p.as_char() == '=' && has_attr => loop {
                    match trees.next() {
                        Some(proc_macro::TokenTree::Punct(p)) if p.as_char() == '!' => {
                            self.no_escaping = true;
                            continue;
                        }
                        Some(proc_macro::TokenTree::Literal(lit)) => {
                            write!(&mut self.template, "={lit}").unwrap();
                        }
                        Some(proc_macro::TokenTree::Ident(id)) => {
                            write!(&mut self.template, "=\"").unwrap();
                            self.display_expr(id);
                            write!(&mut self.template, "\"").unwrap();
                        }
                        Some(proc_macro::TokenTree::Group(g))
                            if g.delimiter() == proc_macro::Delimiter::Brace =>
                        {
                            write!(&mut self.template, "=\"").unwrap();
                            self.display_expr(g.stream());
                            write!(&mut self.template, "\"").unwrap();
                        }
                        Some(c) => {
                            diag!(Error[c]: "unexpected token in attr value")
                        }
                        None => diag!(Error[p]: "expected attribute value"),
                    }
                    break;
                },
                proc_macro::TokenTree::Punct(p) if p.as_char() == '>' => {
                    write!(&mut self.template, ">").unwrap();
                    break;
                }
                proc_macro::TokenTree::Ident(id) => {
                    write!(&mut self.template, " {id}").unwrap();
                    has_attr_tmp = true;
                }
                proc_macro::TokenTree::Literal(lit) => {
                    let temp = temp_str(&lit, &mut self.tstr).trim_matches('"');
                    if !is_valid_html_attt_name(temp) {
                        diag!(Error[p]: "invalid attribute name");
                    }
                    write!(&mut self.template, " {temp}").unwrap();
                    has_attr_tmp = true;
                }
                c => diag!(Error[c]: "unexpected token in attribute list"),
            }
            has_attr = has_attr_tmp;
        }

        Some(())
    }

    fn matches_char(t: &proc_macro::TokenTree, ch: char, spacing: proc_macro::Spacing) -> bool {
        matches!(t, proc_macro::TokenTree::Punct(p) if p.as_char() == ch && p.spacing() == spacing)
    }

    fn match_expr(&mut self, trees: &mut Trees) -> Option<()> {
        let expr = self.expect_group(trees, proc_macro::Delimiter::Parenthesis)?;
        let mut body = self
            .expect_group(trees, proc_macro::Delimiter::Brace)?
            .into_iter()
            .peekable();
        self.flush_template();

        write!(&mut self.final_code, "match {expr} {{").unwrap();

        loop {
            let mut pattern = proc_macro::TokenStream::new();
            let mut looped = false;

            while let Some(c) = body.next() {
                if !Self::matches_char(&c, '=', proc_macro::Spacing::Joint) {
                    pattern.extend([c]);
                    looped = true;
                    continue;
                }

                let nc = body.next().expect("haaaaa");
                if Self::matches_char(&nc, '>', proc_macro::Spacing::Alone) {
                    break;
                }
                pattern.extend([c, nc]);
                looped = true;
            }

            if !looped {
                break;
            }

            let body = self.expect_group(&mut body, proc_macro::Delimiter::Brace)?;

            write!(&mut self.final_code, "{pattern} => {{").unwrap();
            self.output_fmt(&mut body.into_iter().peekable())?;
            write!(&mut self.final_code, "}}").unwrap();
        }

        write!(&mut self.final_code, "}}").unwrap();

        Some(())
    }

    fn for_expr(&mut self, trees: &mut Trees) -> Option<()> {
        let mut ink = proc_macro::Span::call_site();
        let loop_var = trees
            .by_ref()
            .take_while(|t| {
                ink = t.span();
                !matches!(t, proc_macro::TokenTree::Ident(id) if temp_str(id, &mut self.tstr) == "in")
            })
            .collect::<proc_macro::TokenStream>();

        let iter = self.expect_group(trees, proc_macro::Delimiter::Parenthesis)?;
        let body = self.expect_group(trees, proc_macro::Delimiter::Brace)?;
        let else_body = match trees.peek() {
            Some(proc_macro::TokenTree::Ident(id)) if temp_str(id, &mut self.tstr) == "else" => {
                trees.next();
                self.expect_group(trees, proc_macro::Delimiter::Brace)?
            }
            _ => proc_macro::TokenStream::new(),
        };

        self.tstr.clear();

        self.flush_template();

        if else_body.is_empty() {
            write!(&mut self.final_code, "for {loop_var} in {iter} {{").unwrap();
        } else {
            write!(
                &mut self.final_code,
                "let mut looped = false;\
            for {loop_var} in {iter} {{\
            looped = true;"
            )
            .unwrap();
        }
        self.output_fmt(&mut body.into_iter().peekable())?;
        write!(&mut self.final_code, "}}").unwrap();
        if !else_body.is_empty() {
            write!(&mut self.final_code, "if !looped {{").unwrap();
            self.output_fmt(&mut else_body.into_iter().peekable())?;
            write!(&mut self.final_code, "}}").unwrap();
        }
        Some(())
    }

    fn output_fmt(&mut self, trees: &mut Trees) -> Option<()> {
        while let Some(c) = trees.next() {
            match c {
                proc_macro::TokenTree::Punct(p) if p.as_char() == '!' => self.no_escaping = true,
                proc_macro::TokenTree::Punct(p) if p.as_char() == '<' => self.tag(p, trees)?,
                proc_macro::TokenTree::Literal(lit) => self
                    .template
                    .push_str(temp_str(&lit, &mut self.tstr).trim_matches('"')),
                proc_macro::TokenTree::Ident(id) => match temp_str(&id, &mut self.tstr) {
                    "match" => self.match_expr(trees)?,
                    "for" => self.for_expr(trees)?,
                    _ => self.display_expr(id),
                },
                proc_macro::TokenTree::Group(g)
                    if g.delimiter() == proc_macro::Delimiter::Brace =>
                {
                    self.display_expr(g.stream());
                }
                c => fatal_diag!(Error[c]: "unexpected token"),
            }
        }
        self.flush_template();

        Some(())
    }
}

fn temp_str(i: impl std::fmt::Display, buf: &mut String) -> &str {
    buf.clear();
    write!(buf, "{i}").unwrap();
    buf
}

fn is_valid_html_attt_name(tag: &str) -> bool {
    tag.bytes()
        .all(|c| matches!(c, b'a'..=b'z' | b'_' | b'-' | b'A'..=b'Z' | 128..=u8::MAX))
}

fn is_valid_webcomponent(tag: &str) -> bool {
    let mut seen_dash = false;
    tag.bytes()
        .inspect(|c| seen_dash |= *c == b'-')
        .all(|c| matches!(c, b'a'..=b'z' | b'_' | b'-' | b'.' | 128..=u8::MAX))
        && seen_dash
}

fn is_html_tag(tag: &str) -> bool {
    matches!(
        tag,
        "!DOCTYPE"
            | "a"
            | "abbr"
            | "acronym"
            | "address"
            | "area"
            | "article"
            | "aside"
            | "audio"
            | "b"
            | "base"
            | "basefont"
            | "bdi"
            | "bdo"
            | "big"
            | "blockquote"
            | "body"
            | "br"
            | "button"
            | "canvas"
            | "caption"
            | "center"
            | "cite"
            | "code"
            | "col"
            | "colgroup"
            | "data"
            | "datalist"
            | "dd"
            | "del"
            | "details"
            | "dfn"
            | "dialog"
            | "div"
            | "dl"
            | "dt"
            | "em"
            | "embed"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "head"
            | "header"
            | "hr"
            | "html"
            | "i"
            | "iframe"
            | "img"
            | "input"
            | "ins"
            | "kbd"
            | "label"
            | "legend"
            | "li"
            | "link"
            | "main"
            | "map"
            | "mark"
            | "meta"
            | "meter"
            | "nav"
            | "noscript"
            | "object"
            | "ol"
            | "optgroup"
            | "option"
            | "output"
            | "p"
            | "param"
            | "picture"
            | "pre"
            | "progress"
            | "q"
            | "rp"
            | "rt"
            | "ruby"
            | "s"
            | "samp"
            | "script"
            | "section"
            | "select"
            | "small"
            | "source"
            | "span"
            | "strong"
            | "style"
            | "sub"
            | "summary"
            | "sup"
            | "svg"
            | "table"
            | "tbody"
            | "td"
            | "template"
            | "textarea"
            | "tfoot"
            | "th"
            | "thead"
            | "time"
            | "title"
            | "tr"
            | "track"
            | "u"
            | "ul"
            | "var"
            | "video"
            | "wbr"
    )
}

fn is_self_closing(tag: &str) -> bool {
    matches!(
        tag,
        "!DOCTYPE"
            | "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}
