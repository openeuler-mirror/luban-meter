"""Independent async Engine submissions; arrival policy lives upstream."""

import asyncio
import inspect
import threading
import time
import uuid


class AsyncEngineBackend:
    def __init__(self, parameters, factory=None):
        self.parameters = parameters
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.thread.start()
        self.engine = None
        try:
            self.call(self.initialize(factory))
        except BaseException:
            self.close()
            raise

    def call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    async def initialize(self, factory):
        if factory is None:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            names = (
                "model",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "dtype",
                "trust_remote_code",
                "enforce_eager",
            )
            settings = {name: self.parameters[name] for name in names}
            if self.parameters.get("max_model_len"):
                settings["max_model_len"] = self.parameters["max_model_len"]
            self.engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**settings, disable_log_stats=False)
            )
        else:
            self.engine = factory()
        tokenizer = self.engine.get_tokenizer()
        self.tokenizer = (
            await tokenizer if inspect.isawaitable(tokenizer) else tokenizer
        )
        config = getattr(self.engine, "model_config", None)
        if config is None and hasattr(self.engine, "get_model_config"):
            config = self.engine.get_model_config()
            if inspect.isawaitable(config):
                config = await config
        self.max_model_len = getattr(config, "max_model_len", None)
        if not self.max_model_len:
            raise RuntimeError("async Engine model context limit unavailable")

    def generate(self, prompt, settings, timeout):
        return self.call(self._generate(prompt, settings, timeout))

    async def _generate(self, prompt, settings, timeout):
        request_id = uuid.uuid4().hex
        submitted = time.perf_counter()
        output = None
        first = None
        timestamps = []
        try:
            async with asyncio.timeout(timeout):
                async for current in self.engine.generate(
                    prompt, settings, request_id
                ):
                    now = time.perf_counter()
                    output = current
                    if current.outputs and current.outputs[0].token_ids:
                        first = first or now
                        timestamps.append(now)
        except BaseException:
            aborted = self.engine.abort(request_id)
            if inspect.isawaitable(aborted):
                await aborted
            raise
        ended = time.perf_counter()
        if output is None or not output.outputs:
            raise RuntimeError("async Engine returned no output")
        metrics = getattr(output, "metrics", None)
        internal = {}
        for name in (
            "arrival_time",
            "first_scheduled_time",
            "first_token_time",
            "finished_time",
            "scheduled_ts",
            "first_token_ts",
            "last_token_ts",
        ):
            value = getattr(metrics, name, None)
            internal[name] = value if isinstance(value, (int, float)) else None
        return {
            "status": "success",
            "request_id": request_id,
            "input_tokens": len(output.prompt_token_ids),
            "output_tokens": len(output.outputs[0].token_ids),
            "submitted_at_monotonic_s": submitted,
            "completed_at_monotonic_s": ended,
            "first_output_at_monotonic_s": first,
            "output_events_monotonic_s": timestamps,
            "submission_e2el_s": ended - submitted,
            "submission_ttft_s": first - submitted if first else None,
            "engine_timestamps": internal,
            "engine_timestamps_available": any(
                value is not None for value in internal.values()
            ),
        }

    async def shutdown(self):
        if self.engine is not None:
            close = getattr(self.engine, "shutdown", None)
            if close is None:
                close = getattr(self.engine, "shutdown_background_loop", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result
        tasks = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def close(self):
        try:
            self.call(self.shutdown())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
