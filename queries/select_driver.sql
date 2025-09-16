-- Select the driver
SELECT
    driver_cpus_per_task AS cpus_per_task,
    driver_mem AS mem,
    driver_threads_per_core AS threads_per_core,
    driver_wall_time AS wall_time,
    driver_enable_proxy_buffering AS enable_proxy_buffering,
    driver_nginx_timeout AS nginx_timeout
FROM
    swarm
WHERE
    jobid = :jobid;
