SELECT
    deployment_name AS name,
    swarm,
    json_extract(cfg, '$.backend.type') AS platform,
    json_extract(serving_handle, '$.url') AS endpoint
FROM
    swarm
ORDER BY
    swarm;
