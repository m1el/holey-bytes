PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user(
	name TEXT NOT NULL,
	password_hash TEXT NOT NULL,
	PRIMARY KEY (name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS session(
	id BLOB NOT NULL,
	username TEXT NOT NULL,
	expiration INTEGER NOT NULL,
	FOREIGN KEY (username) REFERENCES user (name)
	PRIMARY KEY (username)
) WITHOUT ROWID;

CREATE UNIQUE INDEX IF NOT EXISTS
	session_id ON session (id);

CREATE TABLE IF NOT EXISTS post(
	name TEXT NOT NULL,
	author TEXT,
	timestamp INTEGER,
	code TEXT NOT NULL,
	FOREIGN KEY (author) REFERENCES user(name) ON DELETE SET NULL,
	PRIMARY KEY (author, name)
);

CREATE INDEX IF NOT EXISTS
	post_timestamp ON post(timestamp DESC);

CREATE TABLE IF NOT EXISTS import(
	from_name TEXT NOT NULL,
	from_author TEXT,
	to_name TEXT NOT NULL,
	to_author TEXT,
	FOREIGN KEY (from_name, from_author) REFERENCES post(name, author),
	FOREIGN KEY (to_name, to_author) REFERENCES post(name, author)
);

CREATE INDEX IF NOT EXISTS
	dependencies ON import(from_name, from_author);

CREATE INDEX IF NOT EXISTS
	dependants ON import(to_name, to_author);

CREATE TABLE IF NOT EXISTS run(
	code_name TEXT NOT NULL,
	code_author TEXT NOT NULL,
	runner TEXT NOT NULL,
	FOREIGN KEY (code_name, code_author) REFERENCES post(name, author),
	FOREIGN KEY (runner) REFERENCES user(name),
	PRIMARY KEY (code_name, code_author, runner)
);

