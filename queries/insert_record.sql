-- Insert a new record in the swarm table
INSERT INTO
    swarm
(
    deployment_name,
    swarm,
    cfg,
    serving_handle,
    creation_dt
)
VALUES (
    :deployment_name,
    :swarm,
    :cfg,
    :serving_handle,
    :creation_dt
);
