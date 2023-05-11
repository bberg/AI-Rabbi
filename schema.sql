DROP TABLE IF EXISTS search_logs;
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    real_ip_address TEXT NOT NULL,
    query TEXT,
    collected_messages TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);