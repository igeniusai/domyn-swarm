-- Create the DB table
CREATE TABLE IF NOT EXISTS swarm (
    deployment_name TEXT PRIMARY KEY,
    swarm TEXT,
    cfg TEXT,
    serving_handle TEXT,
    creation_dt DATETIME DEFAULT CURRENT_TIMESTAMP
);
