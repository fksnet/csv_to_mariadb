#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime as dt
import decimal
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
import configparser

try:
    import mariadb  # pip install mariadb
except ImportError as e:
    print("Fehlende Abhängigkeit: pip install mariadb", file=sys.stderr)
    raise

try:
    from dateutil import parser as dateparser  # pip install python-dateutil
except ImportError as e:
    print("Fehlende Abhängigkeit: pip install python-dateutil", file=sys.stderr)
    raise

MAX_IDENTIFIER_LEN = 64
ENGINE_CLAUSE = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"

RESERVED = {
    "add","all","alter","analyze","and","as","asc","before","between","bigint","binary","both",
    "by","cascade","case","change","char","check","collate","column","create","cross","current_date",
    "current_time","current_timestamp","database","databases","day_microsecond","day_minute","day_second",
    "dec","decimal","declare","default","delete","desc","describe","distinct","double","drop","else",
    "end","escaped","exists","explain","false","float","for","force","foreign","from","fulltext","generated",
    "group","having","if","in","index","inner","insert","int","int1","int2","int3","int4","int8","integer",
    "interval","into","is","join","key","keys","leading","left","like","limit","lines","load","localtime",
    "localtimestamp","lock","long","longblob","longtext","match","mediumblob","mediumint","mediumtext",
    "natural","not","null","numeric","on","optimize","option","or","order","outer","primary","procedure",
    "range","read","references","regexp","rename","replace","right","rlike","schema","schemas","select",
    "set","show","smallint","spatial","sql","sql_big_result","sql_calc_found_rows","sql_small_result",
    "ssl","starting","straight_join","table","terminated","then","tinyblob","tinyint","tinytext","to",
    "trailing","true","trigger","undo","union","unique","unlock","unsigned","update","usage","use",
    "using","values","varchar","varying","virtual","when","where","with","year_month","zerofill"
}

# konfigurierbare Token (können via INI überschrieben werden)
BOOL_TRUE = {"true","1","yes","y","ja","wahr"}
BOOL_FALSE = {"false","0","no","n","nein","falsch"}
NULL_TOKENS = {"", "null", "na", "n/a", "none", "-"}

NUM_RE_INT = re.compile(r"^[+-]?\d+$")
NUM_RE_DEC_DOT = re.compile(r"^[+-]?\d+\.\d+$")
NUM_RE_DEC_COMMA = re.compile(r"^[+-]?\d+,\d+$")
DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_DE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
HAS_TIME_RE = re.compile(r"\d{1,2}:\d{2}")

def normalize_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s

def sanitize_identifier(name: str, used: set) -> str:
    s = normalize_ascii(name).lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = "_" + s
    if s in RESERVED:
        s = s + "_col"
    if len(s) > MAX_IDENTIFIER_LEN:
        s = s[:MAX_IDENTIFIER_LEN]
    base = s
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = base[:MAX_IDENTIFIER_LEN - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s

class ColStats:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.nulls = 0
        self.max_strlen = 0
        self.only_bool_values = True
        self.bool_values = set()
        self.only_int = True
        self.only_decimal = True
        self.has_decimal = False
        self.max_int_abs = 0
        self.dec_precision = 0
        self.dec_scale = 0
        self.only_date = True
        self.only_time = True
        self.has_datetime = False
        self.any_texty = False

    def add(self, raw: str):
        self.count += 1
        v = (raw or "").strip()
        v_lower = v.lower()
        if v_lower in NULL_TOKENS:
            self.nulls += 1
            return

        self.max_strlen = max(self.max_strlen, len(v))

        if v_lower in (BOOL_TRUE | BOOL_FALSE):
            self.bool_values.add(v_lower)
        else:
            self.only_bool_values = False

        if NUM_RE_INT.match(v):
            try:
                n = int(v)
                self.max_int_abs = max(self.max_int_abs, abs(n))
            except Exception:
                pass
        else:
            self.only_int = False

        dec_match = None
        if NUM_RE_DEC_DOT.match(v):
            dec_match = v
        elif NUM_RE_DEC_COMMA.match(v):
            dec_match = v.replace(",", ".")
        if dec_match:
            self.has_decimal = True
            try:
                d = decimal.Decimal(dec_match)
                s = str(d).lstrip("-")
                if "." in s:
                    intpart, frac = s.split(".", 1)
                    prec = len(intpart) + len(frac)
                    scale = len(frac)
                else:
                    prec = len(s)
                    scale = 0
                self.dec_precision = max(self.dec_precision, prec)
                self.dec_scale = max(self.dec_scale, scale)
            except Exception:
                pass
        else:
            if not NUM_RE_INT.match(v):
                self.only_decimal = False

        is_date_only = bool(DATE_ISO_RE.match(v) or DATE_DE_RE.match(v))
        is_time_only = bool(TIME_RE.match(v)) and not is_date_only
        has_time = bool(HAS_TIME_RE.search(v))

        parsed = None
        if is_date_only or is_time_only or ("-" in v or "." in v or "/" in v or "T" in v or has_time):
            try:
                parsed = dateparser.parse(v, dayfirst=True, yearfirst=False, fuzzy=False)
            except Exception:
                parsed = None

        if parsed:
            if is_time_only and not is_date_only:
                pass
            elif has_time or ("T" in v) or (parsed.hour or parsed.minute or parsed.second):
                self.has_datetime = True
                self.only_date = False
                self.only_time = False
            else:
                self.only_time = False
        else:
            self.only_date = False
            self.only_time = False

        if not (NUM_RE_INT.match(v) or NUM_RE_DEC_DOT.match(v) or NUM_RE_DEC_COMMA.match(v) or
                is_date_only or is_time_only or parsed or v_lower in (BOOL_TRUE | BOOL_FALSE)):
            self.any_texty = True

    def infer_sql(self):
        nullable = self.nulls > 0

        if self.only_bool_values and len(self.bool_values) > 0:
            return {"type": "TINYINT(1)", "nullable": nullable, "py": "bool"}

        if self.only_int and not self.has_decimal and not self.any_texty and not self.has_datetime and not self.only_time and not self.only_date:
            m = self.max_int_abs
            if m <= 2**7 - 1:
                sqlt = "TINYINT"
            elif m <= 2**15 - 1:
                sqlt = "SMALLINT"
            elif m <= 2**31 - 1:
                sqlt = "INT"
            else:
                sqlt = "BIGINT"
            return {"type": sqlt, "nullable": nullable, "py": "int"}

        if (self.has_decimal or (self.only_decimal and not self.only_int)) and not self.any_texty and not self.has_datetime:
            precision = max(1, min(65, self.dec_precision or 18))
            scale = max(0, min(30, self.dec_scale if self.dec_precision else 4))
            if scale >= precision:
                precision = min(65, scale + 4)
            return {"type": f"DECIMAL({precision},{scale})", "nullable": nullable, "py": "Decimal"}

        if self.has_datetime:
            return {"type": "DATETIME", "nullable": nullable, "py": "datetime"}
        if self.only_date and not self.only_time:
            return {"type": "DATE", "nullable": nullable, "py": "date"}
        if self.only_time and not self.only_date:
            return {"type": "TIME", "nullable": nullable, "py": "time"}

        maxlen = max(1, self.max_strlen)
        if maxlen <= 255:
            return {"type": f"VARCHAR({max(1, maxlen)})", "nullable": nullable, "py": "str"}
        elif maxlen <= 1000:
            return {"type": "VARCHAR(1000)", "nullable": nullable, "py": "str"}
        else:
            return {"type": "TEXT", "nullable": nullable, "py": "str"}

def sniff_dialect(csv_path, delimiter=None, encoding="utf-8", sample_size=65536):
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        sample = f.read(sample_size)
        if delimiter:
            class Simple(csv.Dialect):
                delimiter = delimiter
                quotechar = '"'
                escapechar = None
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            return Simple
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            dialect.lineterminator = "\n"
            return dialect
        except Exception:
            class Fallback(csv.Dialect):
                delimiter = ";"
                quotechar = '"'
                escapechar = None
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            return Fallback

def read_headers(csv_path, dialect, encoding="utf-8"):
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        headers = next(reader)
    return [h.strip() for h in headers]

def scan_csv(csv_path, dialect, headers, encoding="utf-8"):
    stats = [ColStats(h) for h in headers]
    total_rows = 0
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        _ = next(reader)
        for row in reader:
            total_rows += 1
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            for i, val in enumerate(row):
                stats[i].add(val)
    return stats, total_rows

def build_schema(headers, stats):
    used = set()
    sanitized = [sanitize_identifier(h, used) for h in headers]
    col_defs = []
    mapping = []
    for h, sname, st in zip(headers, sanitized, stats):
        info = st.infer_sql()
        col_defs.append((sname, info))
        mapping.append({
            "original_header": h,
            "column_name": sname,
            "mariadb_type": info["type"],
            "nullable": info["nullable"],
            "python_type": info["py"],
            "null_count": st.nulls,
            "row_count": st.count,
            "max_string_length": st.max_strlen
        })
    return col_defs, mapping

def ensure_connection(args):
    conn = mariadb.connect(
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.database,
        autocommit=False
    )
    return conn

def create_table(conn, table_name, col_defs, if_exists, with_id=False, engine_clause=ENGINE_CLAUSE):
    cur = conn.cursor()
    if if_exists == "replace":
        cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")

    cols_sql = []
    if with_id:
        cols_sql.append("`id` INT AUTO_INCREMENT PRIMARY KEY")
    for name, info in col_defs:
        null_sql = "NULL" if info["nullable"] else "NOT NULL"
        cols_sql.append(f"`{name}` {info['type']} {null_sql}")

    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n  " + ",\n  ".join(cols_sql) + f"\n) {engine_clause}"
    cur.execute(create_sql)
    conn.commit()
    cur.close()

def convert_value(raw, pytype):
    if raw is None:
        return None
    v = raw.strip()
    if v.lower() in NULL_TOKENS:
        return None
    try:
        if pytype == "bool":
            if v.lower() in BOOL_TRUE: return 1
            if v.lower() in BOOL_FALSE: return 0
            return None
        if pytype == "int":
            return int(v)
        if pytype == "Decimal":
            v2 = v.replace(",", ".")
            return decimal.Decimal(v2)
        if pytype == "datetime":
            return dateparser.parse(v, dayfirst=True, yearfirst=False, fuzzy=False)
        if pytype == "date":
            d = dateparser.parse(v, dayfirst=True, yearfirst=False, fuzzy=False)
            return d.date()
        if pytype == "time":
            t = dateparser.parse(v, dayfirst=True, yearfirst=False, fuzzy=False)
            return t.time()
        return v
    except Exception:
        return None

def insert_data(conn, csv_path, dialect, headers, col_defs, table_name, encoding="utf-8", batch_size=1000):
    # Wichtig: id ist NICHT in col_defs enthalten – wir inserten nur CSV-Spalten
    pytypes = [info["py"] for _, info in col_defs]
    colnames = [name for name, _ in col_defs]

    placeholders = ", ".join(["%s"] * len(colnames))
    insert_sql = f"INSERT INTO `{table_name}` (" + ", ".join([f"`{c}`" for c in colnames]) + f") VALUES ({placeholders})"

    total_inserted = 0
    with conn.cursor() as cur, open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        _ = next(reader)
        batch = []
        for row in reader:
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            conv = [convert_value(val if val is not None else "", t) for val, t in zip(row, pytypes)]
            batch.append(conv)
            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                total_inserted += len(batch)
                batch = []
        if batch:
            cur.executemany(insert_sql, batch)
            total_inserted += len(batch)
    conn.commit()
    return total_inserted

def derive_table_name(csv_path, explicit=None):
    if explicit:
        base = explicit
    else:
        base = os.path.splitext(os.path.basename(csv_path))[0]
    used = set()
    return sanitize_identifier(base, used)

def save_mapping(mapping, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

# ---------- Konfiguration (INI + CLI mit CLI-Priorität) ----------

def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1","true","yes","y","ja","wahr","on"}

def load_ini(path: str) -> dict:
    cfg = {}
    if not path:
        return cfg
    parser = configparser.ConfigParser()
    read_ok = parser.read(path, encoding="utf-8")
    if not read_ok:
        print(f"Warnung: config.ini '{path}' konnte nicht gelesen werden.", file=sys.stderr)
        return cfg

    # Wir akzeptieren entweder [csv_to_mariadb] oder [default]
    section = "csv_to_mariadb" if "csv_to_mariadb" in parser else ("default" if "default" in parser else None)
    if section:
        sec = parser[section]
        # Standard-Keys
        if "host" in sec: cfg["host"] = sec.get("host")
        if "port" in sec: cfg["port"] = sec.getint("port")
        if "user" in sec: cfg["user"] = sec.get("user")
        if "password" in sec: cfg["password"] = sec.get("password")
        if "database" in sec: cfg["database"] = sec.get("database")
        if "table" in sec: cfg["table"] = sec.get("table")
        if "encoding" in sec: cfg["encoding"] = sec.get("encoding")
        if "delimiter" in sec: cfg["delimiter"] = sec.get("delimiter")
        if "if_exists" in sec: cfg["if_exists"] = sec.get("if_exists")
        if "batch_size" in sec: cfg["batch_size"] = sec.getint("batch_size")
        if "mapping_json" in sec: cfg["mapping_json"] = sec.get("mapping_json")
        if "with_id" in sec: cfg["with_id"] = sec.getboolean("with_id")
        if "engine_clause" in sec: cfg["engine_clause"] = sec.get("engine_clause")

        # optionale Token
        if "null_tokens" in sec:
            cfg["null_tokens"] = {t.strip().lower() for t in sec.get("null_tokens").split(",")}
        if "bool_true" in sec:
            cfg["bool_true"] = {t.strip().lower() for t in sec.get("bool_true").split(",")}
        if "bool_false" in sec:
            cfg["bool_false"] = {t.strip().lower() for t in sec.get("bool_false").split(",")}
    return cfg

def build_arg_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    d = defaults or {}
    ap = argparse.ArgumentParser(description="CSV → MariaDB Import mit automatischer Typableitung.")
    ap.add_argument("csv", nargs="?", help="Pfad zur CSV-Datei")
    ap.add_argument("--config", help="Pfad zu config.ini (CLI-Argumente überschreiben INI-Werte)")
    ap.add_argument("--host", default=d.get("host","localhost"))
    ap.add_argument("--port", type=int, default=d.get("port",3306))
    ap.add_argument("--user", default=d.get("user"))
    ap.add_argument("--password", default=d.get("password"))
    ap.add_argument("--database", default=d.get("database"))
    ap.add_argument("--table", default=d.get("table"))
    ap.add_argument("--encoding", default=d.get("encoding","utf-8"))
    ap.add_argument("--delimiter", default=d.get("delimiter"))
    ap.add_argument("--if-exists", choices=["fail","replace","append"], default=d.get("if_exists","replace"))
    ap.add_argument("--batch-size", type=int, default=d.get("batch_size",1000))
    ap.add_argument("--mapping-json", default=d.get("mapping_json"))
    ap.add_argument("--with-id", action="store_true", default=d.get("with_id", False),
                    help="Primärschlüssel 'id INT AUTO_INCREMENT' als erste Spalte hinzufügen")
    ap.add_argument("--engine-clause", default=d.get("engine_clause", ENGINE_CLAUSE),
                    help="Engine/Charset/Collation Klausel für CREATE TABLE")
    # Token ggf. via CLI überschreibbar
    ap.add_argument("--null-tokens", help="Kommagetrennte Liste von NULL-Tokens")
    ap.add_argument("--bool-true", help="Kommagetrennte Liste für TRUE")
    ap.add_argument("--bool-false", help="Kommagetrennte Liste für FALSE")
    return ap

def apply_token_overrides(args):
    global NULL_TOKENS, BOOL_TRUE, BOOL_FALSE
    if args.null_tokens:
        NULL_TOKENS = {t.strip().lower() for t in args.null_tokens.split(",")}
    if args.bool_true:
        BOOL_TRUE = {t.strip().lower() for t in args.bool_true.split(",")}
    if args.bool_false:
        BOOL_FALSE = {t.strip().lower() for t in args.bool_false.split(",")}

# ---------- main ----------

def main():
    # 1) Erst nur --config parsen, dann INI lesen, dann Parser mit Defaults bauen, dann final parsen
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre.add_argument("csv", nargs="?")
    pre_args, _ = pre.parse_known_args()

    ini_defaults = load_ini(pre_args.config) if pre_args.config else {}
    ap = build_arg_parser(ini_defaults)
    args = ap.parse_args()

    if not args.csv:
        ap.error("CSV-Pfad fehlt.")

    # ggf. Token-Overrides anwenden (INI/CLI)
    if "null_tokens" in ini_defaults and not args.null_tokens:
        NULL_TOKENS.clear(); NULL_TOKENS.update(ini_defaults["null_tokens"])
    if "bool_true" in ini_defaults and not args.bool_true:
        BOOL_TRUE.clear(); BOOL_TRUE.update(ini_defaults["bool_true"])
    if "bool_false" in ini_defaults and not args.bool_false:
        BOOL_FALSE.clear(); BOOL_FALSE.update(ini_defaults["bool_false"])
    apply_token_overrides(args)

    # 2) Dialekt & Header
    dialect = sniff_dialect(args.csv, delimiter=args.delimiter, encoding=args.encoding)
    headers = read_headers(args.csv, dialect, encoding=args.encoding)

    # 3) Scan & Typableitung
    stats, total_rows = scan_csv(args.csv, dialect, headers, encoding=args.encoding)
    col_defs, mapping = build_schema(headers, stats)

    # 3a) Mapping um id ergänzen, falls gewünscht (id steht NICHT in col_defs, nur im Mapping!)
    if args.with_id:
        id_entry = {
            "original_header": "--with-id",
            "column_name": "id",
            "mariadb_type": "INT AUTO_INCREMENT",
            "nullable": False,
            "python_type": "int",
            "null_count": 0,
            "row_count": total_rows,
            "max_string_length": 0
        }
        mapping = [id_entry] + mapping

    # 4) Tabelle erzeugen
    table_name = derive_table_name(args.csv, args.table)
    conn = ensure_connection(args)
    try:
        create_table(conn, table_name, col_defs, args.if_exists, with_id=args.with_id, engine_clause=args.engine_clause)
    except mariadb.Error as e:
        print(f"Fehler beim Erstellen der Tabelle '{table_name}': {e}", file=sys.stderr)
        conn.close()
        sys.exit(2)

    # 5) Daten importieren
    try:
        inserted = insert_data(conn, args.csv, dialect, headers, col_defs, table_name, encoding=args.encoding, batch_size=args.batch_size)
    except mariadb.Error as e:
        conn.rollback()
        print(f"Fehler beim Import: {e}", file=sys.stderr)
        conn.close()
        sys.exit(3)
    finally:
        conn.close()

    # 6) Mapping speichern
    mapping_path = args.mapping_json or (args.csv + ".schema.json")
    try:
        save_mapping(mapping, mapping_path)
    except Exception as e:
        print(f"Warnung: Mapping-JSON konnte nicht gespeichert werden: {e}", file=sys.stderr)

    # 7) Zusammenfassung
    print("\n=== Zusammenfassung ===")
    print(f"CSV-Datei           : {args.csv}")
    print(f"Zeilen importiert   : {inserted} (ohne Kopfzeile)")
    print(f"Datenbank           : {args.database}")
    print(f"Tabelle             : {table_name}")
    print(f"Spalten             : {len(col_defs) + (1 if args.with_id else 0)} (inkl.{' ' if args.with_id else ' ohne '}id)")
    type_counts = Counter([info["type"] for _, info in col_defs])
    print("Typ-Verteilung      : " + ", ".join(f"{t}×{c}" for t, c in type_counts.items()))
    print(f"Mapping-JSON        : {mapping_path}")
    if args.with_id:
        print("Hinweis             : 'id INT AUTO_INCREMENT PRIMARY KEY' wurde als erste Spalte angelegt und im Mapping mit original_header='--with-id' erfasst.")

if __name__ == "__main__":
    decimal.getcontext().prec = 80
    main()
