CREATE TABLE search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    query TEXT,
    collected_messages TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);