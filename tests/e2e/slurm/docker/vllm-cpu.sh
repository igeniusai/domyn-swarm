#!/bin/sh
# The x86_64 CPU wheel needs Intel OpenMP preloaded; the arm64 wheel ships none.
iomp=$(find /opt/vllm -name 'libiomp5.so' -print -quit)
if [ -n "$iomp" ]; then
  export LD_PRELOAD="$iomp${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec /opt/vllm/bin/vllm "$@"
