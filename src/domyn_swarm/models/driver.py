from pydantic import BaseModel


class DriverConfig(BaseModel):
    cpus_per_task: int = 2
    mem: str = "16GB"
    threads_per_core: int = 1
    wall_time: str = "24:00:00"
    enable_proxy_buffering: bool = True
    nginx_timeout: str | int = "60s"


class SlurmConfig(BaseModel):
    partition: str = "boost_usr_prod"
    account: str = "iGen_train"
    gpus_per_node: int = 4
    nodes: int | None = None
    replicas_per_node: int | None = None
    mem_per_cpu: str | None = None
    requires_ray: bool | None = None
    module_load: list[str] = []
    preamble: list[str] = []  # additional SLURM directives
    image: str | None = None  # path to Singularity image
