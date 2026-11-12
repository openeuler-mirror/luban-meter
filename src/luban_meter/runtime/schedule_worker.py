"""Execute one saved plan while holding the project's execution fence."""

import sys
import threading
import time

from luban_meter.core.planning import (
    bootstrap,
    public_plan,
    validate_plan_parameters,
)
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.core.schedule_store import ScheduleStore
from luban_meter.core.schedule_time import stamp
from luban_meter.log import (
    configure_logging,
    get_logger,
    instance_filename,
    log_context,
    resolve_log_dir,
    task_secrets,
)
from luban_meter.reporting.html import write_html
from luban_meter.utils.json_io import write_json_atomic

logger = get_logger("luban_meter.runtime.schedule_worker")


def main():
    store = ScheduleStore(sys.argv[1])
    execution_id = sys.argv[2]
    log_level = sys.argv[3]
    bootstrap()
    with store.lock("active"):
        with store.transaction() as records:
            row = dict(records.executions[execution_id])
            if (row["status"] != "starting"
                    or not records.plans[row["plan_id"]]["enabled"]):
                return 130
            saved = records.revisions[row["plan_revision"]]
            plan = saved["resolved"]
            directory = store.project / "runs" / row["run_id"]
            log_dir = resolve_log_dir(
                sys.argv[4] or None, fallback=directory / "log",
            ).resolve()
            log_name = instance_filename(row["run_id"])
            row.update(status="running", started_at=stamp(time.time()),
                       log_path=str(log_dir / log_name))
            records.save_execution(row, time.time())
        with configure_logging(
            log_dir, filename=log_name, level=log_level,
            secrets=task_secrets(plan["tasks"]),
        ), log_context(phase="schedule"):
            coordinator = RunCoordinator()
            finished = threading.Event()

            def watch_stop():
                while not finished.wait(0.1):
                    if (store.stop_requested(execution_id)
                            and coordinator.stop_file is not None):
                        logger.info("Scheduled run stop requested: %s",
                                    execution_id)
                        coordinator.interrupt(None, None)
                        return

            watcher = threading.Thread(target=watch_stop, daemon=True)
            watcher.start()
            try:
                logger.info("Scheduled run started: plan=%s execution=%s "
                            "scheduled=%s retry_of=%s",
                            row["plan_id"], execution_id, row["scheduled_at"],
                            row["retry_of"])
                validate_plan_parameters(plan, base_dir=store.project)
                result, directory = coordinator.run(
                    plan, store.project / "runs", run_id=row["run_id"],
                    log_dir=log_dir, log_level=log_level,
                )
                result["schedule"] = {
                    "plan_id": row["plan_id"], "execution_id": execution_id,
                    "scheduled_at": row["scheduled_at"],
                    "retry_of": row["retry_of"], "plan": public_plan(plan),
                }
                write_json_atomic(directory / "result.json", result)
                report = write_html(result, directory / "report.html")
                logger.info("Scheduled run completed: status=%s result=%s "
                            "report=%s", result["status"],
                            directory / "result.json", report)
                return 130 if result["status"] == "interrupted" else (
                    0 if result["status"] == "success" else 1
                )
            except Exception:
                logger.exception("Scheduled run failed")
                return 2
            finally:
                finished.set()
                watcher.join()


if __name__ == "__main__":
    raise SystemExit(main())
