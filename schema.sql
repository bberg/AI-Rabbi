DROP TABLE IF EXISTS request_logs;
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    real_ip_address TEXT NOT NULL,
    query TEXT,
    collected_messages TEXT,
    user_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
DROP TABLE IF EXISTS response_logs;
CREATE TABLE IF NOT EXISTS response_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    real_ip_address TEXT NOT NULL,
    query TEXT,
    collected_messages TEXT,
    user_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);