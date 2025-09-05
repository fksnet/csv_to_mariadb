# CSV to MariaDB Importer

Ein Python-Skript, das eine CSV-Datei einliest, Spaltentypen automatisch erkennt und eine MariaDB-Tabelle erzeugt und befüllt.  
Optional kann ein `id`-Feld als `AUTO_INCREMENT PRIMARY KEY` angelegt und im Mapping-JSON mit `"original_header": "--with-id"` erfasst werden.

## Features

- automatische Erkennung von Datentypen (INT, DECIMAL, DATE, DATETIME, TIME, VARCHAR, TEXT, BOOL)
- CSV-Header → saubere MariaDB-Spaltennamen (lowercase, ASCII, `_`, max. 64, eindeutig, Keywords entschärft)
- optionales `id`-Feld (`--with-id`, Primärschlüssel)
- Mapping von CSV-Header → Spaltendefinition als JSON
- Konfiguration über **CLI** oder `config.ini` (CLI überschreibt INI)
- Zusammenfassung nach jedem Import

## Voraussetzungen

- Python 3.9+
- MariaDB-Server erreichbar
- Pakete aus `requirements.txt`

## Installation

~~~bash
git clone https://github.com/fksnet/csv_to_mariadb.git
cd csv_to_mariadb
python3 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (CMD):
# .venv\Scripts\activate
pip install -r requirements.txt
~~~

*(SSH-Alternative, falls du SSH-Keys/Deploy-Key nutzt: `git clone git@github.com:fksnet/csv_to_mariadb.git`)*

## Konfiguration

Lege eine `config.ini` auf Basis der mitgelieferten `config-sample.ini` an.  
Alle CLI-Argumente **überschreiben** gleichnamige Werte aus der INI.

Beispiel `config-sample.ini`:

~~~ini
[csv_to_mariadb]
host = 127.0.0.1
port = 3306
user = meinuser
password = geheim
database = meine_db
table = meine_tabelle
encoding = utf-8
delimiter = ;
if_exists = replace
batch_size = 2000
mapping_json = ./meine_tabelle.schema.json
with_id = true
engine_clause = ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci

# optionale Token
null_tokens = , null, n/a, -, none
bool_true   = true,1,yes,ja,wahr
bool_false  = false,0,no,nein,falsch
~~~

## Nutzung

### Direkt über CLI

~~~bash
python csv_to_mariadb.py daten.csv \
  --host 127.0.0.1 --port 3306 \
  --user meinuser --password geheim \
  --database meine_db \
  --if-exists replace \
  --with-id
~~~

### Mit `config.ini`

1) `config-sample.ini` nach `config.ini` kopieren  
2) Zugangsdaten/Optionen anpassen  
3) Starten:

~~~bash
python csv_to_mariadb.py daten.csv --config config.ini
~~~

CLI-Parameter überschreiben die INI-Werte.

## Beispiele

~~~bash
# Tabelle aus Dateinamen ableiten, id-Spalte aktivieren
python csv_to_mariadb.py kunden.csv --config config.ini --with-id

# Trennzeichen und Encoding explizit setzen
python csv_to_mariadb.py messwerte.csv --delimiter "," --encoding "latin1"
~~~

## Mapping-JSON

Beispielauszug (inkl. `id`-Feld):

~~~json
[
  {
    "original_header": "--with-id",
    "column_name": "id",
    "mariadb_type": "INT AUTO_INCREMENT",
    "nullable": false,
    "python_type": "int",
    "null_count": 0,
    "row_count": 1000,
    "max_string_length": 0
  },
  {
    "original_header": "Name",
    "column_name": "name",
    "mariadb_type": "VARCHAR(50)",
    "nullable": true,
    "python_type": "str",
    "null_count": 3,
    "row_count": 1000,
    "max_string_length": 45
  }
]
~~~

## Remote einrichten (falls du ein bestehendes lokales Projekt pushen willst)

~~~bash
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/fksnet/csv_to_mariadb.git
git push -u origin main
~~~

*(SSH-Variante: `git remote add origin git@github.com:fksnet/csv_to_mariadb.git`)*

## Lizenz

MIT

---

**Hinweis / Credits**  
Skript und Dokumentation erstellt mit Unterstützung von **ChatGPT 5**.
