use {
    axum::{
        body::Bytes,
        http::{header::COOKIE, request::Parts},
        response::{AppendHeaders, Html},
    },
    core::fmt,
    htmlm::{html, write_html},
    serde::{Deserialize, Serialize},
    std::{fmt::Write, net::Ipv4Addr},
};

const MAX_NAME_LENGTH: usize = 32;
const MAX_POSTNAME_LENGTH: usize = 64;
//const MAX_CODE_LENGTH: usize = 1024 * 4;
const SESSION_DURATION_SECS: u64 = 60 * 60;

type Redirect<const COUNT: usize = 1> = AppendHeaders<[(&'static str, &'static str); COUNT]>;

macro_rules! static_asset {
    ($mime:literal, $body:literal) => {
        get(|| async {
            axum::http::Response::builder()
                .header("content-type", $mime)
                .header("content-encoding", "gzip")
                .body(axum::body::Body::from(Bytes::from_static(include_bytes!(concat!(
                    $body, ".gz"
                )))))
                .unwrap()
        })
    };
}

async fn amain() {
    use axum::routing::{delete, get, post};

    let debug = cfg!(debug_assertions);

    log::set_logger(&Logger).unwrap();
    log::set_max_level(if debug { log::LevelFilter::Warn } else { log::LevelFilter::Error });

    db::init();

    let router = axum::Router::new()
        .route("/", get(Index::page))
        .route("/index.css", static_asset!("text/css", "index.css"))
        .route("/index.js", static_asset!("text/javascript", "index.js"))
        .route("/hbfmt.wasm", static_asset!("application/wasm", "hbfmt.wasm"))
        .route("/hbc.wasm", static_asset!("application/wasm", "hbc.wasm"))
        .route("/index-view", get(Index::get))
        .route("/feed", get(Index::page))
        .route("/profile", get(Profile::page))
        .route("/profile-view", get(Profile::get))
        .route("/post", get(Post::page))
        .route("/post-view", get(Post::get))
        .route("/post", post(Post::post))
        .route("/login", get(Login::page))
        .route("/login-view", get(Login::get))
        .route("/login", post(Login::post))
        .route("/login", delete(Login::delete))
        .route("/signup", get(Signup::page))
        .route("/signup-view", get(Signup::get))
        .route("/signup", post(Signup::post))
        .route(
            "/hot-reload",
            get({
                let id = std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                move || async move { id.to_string() }
            }),
        );

    let socket = tokio::net::TcpListener::bind((Ipv4Addr::UNSPECIFIED, 8080)).await.unwrap();

    axum::serve(socket, router).await.unwrap();
}

trait PublicPage: Default {
    fn render_to_buf(self, buf: &mut String);

    fn render(self) -> String {
        let mut str = String::new();
        Self::default().render_to_buf(&mut str);
        str
    }

    async fn get() -> Html<String> {
        Html(Self::default().render())
    }

    async fn page(session: Option<Session>) -> Html<String> {
        base(|s| Self::default().render_to_buf(s), session.as_ref()).await
    }
}

trait Page: Default {
    fn render_to_buf(self, session: &Session, buf: &mut String);

    fn render(self, session: &Session) -> String {
        let mut str = String::new();
        Self::default().render_to_buf(session, &mut str);
        str
    }

    async fn get(session: Session) -> Html<String> {
        Html(Self::default().render(&session))
    }

    async fn page(session: Option<Session>) -> Result<Html<String>, axum::response::Redirect> {
        match session {
            Some(session) => {
                Ok(base(|f| Self::default().render_to_buf(&session, f), Some(&session)).await)
            }
            None => Err(axum::response::Redirect::permanent("/login")),
        }
    }
}

#[derive(Default)]
struct Index;

impl PublicPage for Index {
    fn render_to_buf(self, buf: &mut String) {
        buf.push_str(include_str!("welcome-page.html"));
    }
}

#[derive(Deserialize, Default)]
struct Post {
    author: String,
    name: String,
    #[serde(skip)]
    timestamp: u64,
    #[serde(skip)]
    imports: usize,
    #[serde(skip)]
    runs: usize,
    #[serde(skip)]
    dependencies: usize,
    code: String,
    #[serde(skip)]
    error: Option<&'static str>,
}

impl Page for Post {
    fn render_to_buf(self, session: &Session, buf: &mut String) {
        let Self { name, code, error, .. } = self;
        write_html! { (buf)
            <form id="postForm" "hx-post"="/post" "hx-swap"="outherHTML">
                if let Some(e) = error { <div class="error">e</div> }
                <input name="author" type="text" value={session.name} hidden>
                <input name="name" type="text" placeholder="name" value=name
                    required maxlength=MAX_POSTNAME_LENGTH>
                <textarea id="code-edit" name="code" placeholder="code" rows=1 required>code</textarea>
                <input type="submit" value="submit">
                <pre id="compiler-output"></pre>
            </form>
            !{include_str!("post-page.html")}
        }
    }
}

impl Post {
    async fn post(
        session: Session,
        axum::Form(mut data): axum::Form<Self>,
    ) -> Result<Redirect, Html<String>> {
        db::with(|db| {
            if let Err(e) = db.create_post.insert((&data.name, &session.name, now(), &data.code)) {
                if let rusqlite::Error::SqliteFailure(e, _) = e {
                    if e.code == rusqlite::ErrorCode::ConstraintViolation {
                        data.error = Some("this name is already used");
                    }
                }
                data.error = data.error.or_else(|| {
                    log::error!("{e}");
                    Some("internal server error")
                });
            }
        });

        if data.error.is_some() {
            Err(Html(data.render(&session)))
        } else {
            Ok(redirect("/profile"))
        }
    }
}

impl fmt::Display for Post {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { author, name, timestamp, imports, runs, dependencies, code, .. } = self;
        write_html! { f <div class="preview">
            <div class="info">
                <span>author "/" name</span>
                <span apply="timestamp">timestamp</span>
            </div>
            <div class="stats">
                for (name, count) in "inps runs deps".split(' ')
                    .zip([imports, runs, dependencies])
                    .filter(|(_, &c)| c != 0)
                {
                    name ": "<span>count</span>
                }
            </div>
            <pre apply="fmt">code</pre>
            if *timestamp == 0 {
                <button "hx-get"="/post" "hx-swap"="outherHTML"
                    "hx-target"="[preview]">"edit"</button>
            }
        </div> }
        Ok(())
    }
}

#[derive(Default)]
struct Profile;

impl Page for Profile {
    fn render_to_buf(self, session: &Session, buf: &mut String) {
        db::with(|db| {
            let iter = db
                .get_user_posts
                .query_map((&session.name,), |r| {
                    Ok(Post {
                        author: r.get(0)?,
                        name: r.get(1)?,
                        timestamp: r.get(2)?,
                        code: r.get(3)?,
                        ..Default::default()
                    })
                })
                .inspect_err(|e| log::error!("{e}"))
                .into_iter()
                .flatten()
                .filter_map(|p| p.inspect_err(|e| log::error!("{e}")).ok());
            write_html! { (buf)
                for post in iter {
                    !{post}
                } else {
                    "no posts"
                }
                !{include_str!("profile-page.html")}
            }
        })
    }
}

#[derive(Serialize, Deserialize, Default)]
struct Login {
    name: String,
    password: String,
    #[serde(skip)]
    error: Option<&'static str>,
}

impl PublicPage for Login {
    fn render_to_buf(self, buf: &mut String) {
        let Login { name, password, error } = self;
        write_html! { (buf)
            <form "hx-post"="/login" "hx-swap"="outherHTML">
                if let Some(e) = error { <div class="error">e</div> }
                <input name="name" type="text" autocomplete="name" placeholder="name" value=name
                    required maxlength=MAX_NAME_LENGTH>
                <input name="password" type="password" autocomplete="current-password" placeholder="password"
                    value=password>
                <input type="submit" value="submit">
            </form>
        }
    }
}

impl Login {
    async fn post(
        axum::Form(mut data): axum::Form<Self>,
    ) -> Result<AppendHeaders<[(&'static str, String); 2]>, Html<String>> {
        // TODO: hash password
        let mut id = [0u8; 32];
        db::with(|db| match db.authenticate.query((&data.name, &data.password)) {
            Ok(mut r) => {
                if r.next().map_or(true, |v| v.is_none()) {
                    data.error = Some("invalid credentials");
                } else {
                    getrandom::getrandom(&mut id).unwrap();
                    if let Err(e) = db.login.insert((id, &data.name, now() + SESSION_DURATION_SECS))
                    {
                        log::error!("{e}");
                    }
                }
            }
            Err(e) => {
                log::error!("{e}");
                data.error = Some("internal server error");
            }
        });

        if data.error.is_some() {
            Err(Html(data.render()))
        } else {
            Ok(AppendHeaders([
                ("hx-location", "/feed".into()),
                (
                    "set-cookie",
                    format!(
                        "id={}; SameSite=Strict; Secure; Max-Age={SESSION_DURATION_SECS}",
                        to_hex(&id)
                    ),
                ),
            ]))
        }
    }

    async fn delete(session: Session) -> Redirect {
        _ = db::with(|q| q.logout.execute((session.id,)).inspect_err(|e| log::error!("{e}")));
        redirect("/login")
    }
}

#[derive(Serialize, Deserialize, Default)]
struct Signup {
    name: String,
    new_password: String,
    confirm_password: String,
    #[serde(skip)]
    error: Option<&'static str>,
}

impl PublicPage for Signup {
    fn render_to_buf(self, buf: &mut String) {
        let Signup { name, new_password, confirm_password, error } = self;
        write_html! { (buf)
            <form "hx-post"="/signup" "hx-swap"="outherHTML">
                if let Some(e) = error { <div class="error">e</div> }
                <input name="name" type="text" autocomplete="name" placeholder="name" value=name
                    maxlength=MAX_NAME_LENGTH required>
                <input name="new_password" type="password" autocomplete="new-password" placeholder="new password"
                    value=new_password>
                <input name="confirm_password" type="password" autocomplete="confirm-password"
                    placeholder="confirm password" value=confirm_password>
                <input type="submit" value="submit">
            </form>
        }
    }
}

impl Signup {
    async fn post(axum::Form(mut data): axum::Form<Self>) -> Result<Redirect, Html<String>> {
        db::with(|db| {
            // TODO: hash passwords
            match db.register.insert((&data.name, &data.new_password)) {
                Ok(_) => {}
                Err(rusqlite::Error::SqliteFailure(e, _))
                    if e.code == rusqlite::ErrorCode::ConstraintViolation =>
                {
                    data.error = Some("username already taken");
                }
                Err(e) => {
                    log::error!("{e}");
                    data.error = Some("internal server error");
                }
            };
        });

        if data.error.is_some() {
            Err(Html(data.render()))
        } else {
            Ok(redirect("/login"))
        }
    }
}

async fn base(body: impl FnOnce(&mut String), session: Option<&Session>) -> Html<String> {
    let username = session.map(|s| &s.name);

    Html(html! {
        "<!DOCTYPE html>"
        <html lang="en">
            <head>
                <link rel="stylesheet" href="/index.css">
            </head>
            <body>
                <nav>
                    <button "hx-push-url"="/" "hx-get"="/index-view" "hx-target"="main" "hx-swap"="innerHTML">"depell"</button>
                    <section>
                        if let Some(username) = username {
                            <button "hx-push-url"="/profile" "hx-get"="/profile-view" "hx-target"="main"
                                "hx-swap"="innerHTML">username</button>
                            <button "hx-push-url"="/post" "hx-get"="/post-view" "hx-target"="main"
                                "hx-swap"="innerHTML">"post"</button>
                            <button "hx-delete"="/login">"logout"</button>
                        } else {
                            <button "hx-push-url"="/login" "hx-get"="/login-view" "hx-target"="main"
                                "hx-swap"="innerHTML">"login"</button>
                            <button "hx-push-url"="/signup" "hx-get"="/signup-view" "hx-target"="main"
                                "hx-swap"="innerHTML">"signup"</button>
                        }
                    </section>
                </nav>
                <section id="post-form"></section>
                <main>|f|{body(f)}</main>
            </body>
            <script src="https://unpkg.com/htmx.org@2.0.3" integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq" crossorigin="anonymous"></script>
            <script type="module" src="/index.js"></script>
        </html>
    })
}

struct Session {
    name: String,
    id: [u8; 32],
}

#[axum::async_trait]
impl<S> axum::extract::FromRequestParts<S> for Session {
    /// If the extractor fails it'll use this "rejection" type. A rejection is
    /// a kind of error that can be converted into a response.
    type Rejection = Redirect;

    /// Perform the extraction.
    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        let err = redirect("/login");

        let value = parts
            .headers
            .get_all(COOKIE)
            .into_iter()
            .find_map(|c| c.to_str().ok()?.trim().strip_prefix("id="))
            .map(|c| c.split_once(';').unwrap_or((c, "")).0)
            .ok_or(err)?;
        let mut id = [0u8; 32];
        parse_hex(value, &mut id).ok_or(err)?;

        let (name, expiration) = db::with(|db| {
            db.get_session
                .query_row((id,), |r| Ok((r.get::<_, String>(0)?, r.get::<_, u64>(1)?)))
                .inspect_err(|e| log::error!("{e}"))
                .map_err(|_| err)
        })?;

        if expiration < now() {
            log::error!("expired");
            return Err(err);
        }

        Ok(Self { name, id })
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn parse_hex(hex: &str, dst: &mut [u8]) -> Option<()> {
    fn hex_to_nibble(b: u8) -> Option<u8> {
        Some(match b {
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            b'0'..=b'9' => b - b'0',
            _ => return None,
        })
    }

    if hex.len() != dst.len() * 2 {
        return None;
    }

    for (d, p) in dst.iter_mut().zip(hex.as_bytes().chunks_exact(2)) {
        *d = (hex_to_nibble(p[0])? << 4) | hex_to_nibble(p[1])?;
    }

    Some(())
}

fn to_hex(src: &[u8]) -> String {
    use std::fmt::Write;
    let mut buf = String::new();
    for &b in src {
        write!(buf, "{b:02x}").unwrap()
    }
    buf
}

fn main() {
    tokio::runtime::Builder::new_current_thread().enable_io().build().unwrap().block_on(amain());
}

mod db {
    use std::cell::RefCell;

    macro_rules! gen_queries {
        ($vis:vis struct $name:ident {
            $($qname:ident: $code:literal,)*
        }) => {
            #[allow(dead_code)]
            $vis struct $name<'a> {
                $($vis $qname: rusqlite::Statement<'a>,)*
            }

            impl<'a> $name<'a> {
                fn new(db: &'a rusqlite::Connection) -> Self {
                    Self {
                        $($qname: db.prepare($code).unwrap(),)*
                    }
                }
            }
        };
    }

    gen_queries! {
        pub struct Queries {
            register: "INSERT INTO user (name, password_hash) VALUES(?, ?)",
            authenticate: "SELECT name, password_hash FROM user WHERE name = ? AND password_hash = ?",
            login: "INSERT OR REPLACE INTO session (id, username, expiration) VALUES(?, ?, ?)",
            logout: "DELETE FROM session WHERE id = ?",
            get_session: "SELECT username, expiration FROM session WHERE id = ?",
            get_user_posts: "SELECT author, name, timestamp, code FROM post WHERE author = ?",
            create_post: "INSERT INTO post (name, author, timestamp, code) VALUES(?, ?, ?, ?)",
        }
    }

    struct Db {
        queries: Queries<'static>,
        _db: Box<rusqlite::Connection>,
    }

    impl Db {
        fn new() -> Self {
            let db = Box::new(rusqlite::Connection::open("db.sqlite").unwrap());
            Self {
                queries: Queries::new(unsafe {
                    std::mem::transmute::<&rusqlite::Connection, &rusqlite::Connection>(&db)
                }),
                _db: db,
            }
        }
    }

    pub fn with<T>(with: impl FnOnce(&mut Queries) -> T) -> T {
        thread_local! { static DB_CONN: RefCell<Db> = RefCell::new(Db::new()); }
        DB_CONN.with_borrow_mut(|q| with(&mut q.queries))
    }

    pub fn init() {
        let db = rusqlite::Connection::open("db.sqlite").unwrap();
        db.execute_batch(include_str!("schema.sql")).unwrap();
        Queries::new(&db);
    }
}

fn redirect(to: &'static str) -> Redirect {
    AppendHeaders([("hx-location", to)])
}

struct Logger;

impl log::Log for Logger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("{} - {}", record.module_path().unwrap_or("=="), record.args());
        }
    }

    fn flush(&self) {}
}
