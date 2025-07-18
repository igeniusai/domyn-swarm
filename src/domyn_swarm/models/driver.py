from pydantic import BaseModel


class DriverConfig(BaseModel):
    cpus_per_task: int = 2
    mem: str = "16GB"
    threads_per_core: int = 1
    wall_time: str = "24:00:00"
    enable_proxy_buffering: bool = True
    nginx_timeout: str = "60s"
