import multiprocessing as mp
import queue
import time
import datetime
import os
import signal


class BackgroundWorker:
    def __init__(self, work_lim=5):
        self.work_lim = work_lim
        self.work_queue = mp.Queue()
        self.results = mp.Manager().dict()
        self.active_work = mp.Value("i", 0)
        self.lock = mp.Lock()
        self.enqueued_tasks = mp.Manager().list()
        self.running_tasks = mp.Manager().dict()
        self.current_tasks = mp.Manager().dict()  # To track current running tasks
        self.processes = []
        self.stop_event = mp.Event()
        for _ in range(work_lim):
            p = mp.Process(target=self._process_work)
            p.start()
            self.processes.append(p)

    def request_work(self, work_id, work_func, *args, **kwargs):
        with self.lock:
            self.enqueued_tasks.append(work_id)
        self.work_queue.put((work_id, work_func, args, kwargs))

    def get_queue_status(self):
        return list(self.enqueued_tasks)

    def get_results(self):
        return dict(self.results)

    def get_running_tasks(self):
        return dict(self.current_tasks)

    def remove_work(self, work_id):
        with self.lock:
            if work_id in self.enqueued_tasks:
                self.enqueued_tasks.remove(work_id)
                return True
            return False

    def force_kill_work(self, work_id):
        with self.lock:
            if work_id in self.running_tasks:
                pid = self.running_tasks[work_id]
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                del self.running_tasks[work_id]
                del self.current_tasks[work_id]
                self.active_work.value -= 1
                return True
            return False

    def _process_work(self):
        while not self.stop_event.is_set():
            try:
                work_id, work_func, args, kwargs = self.work_queue.get(timeout=0.1)
                with self.lock:
                    if work_id in self.enqueued_tasks:
                        self.enqueued_tasks.remove(work_id)
                        self.active_work.value += 1
                        self.current_tasks[work_id] = mp.current_process().name
                    else:
                        continue
                self._start_work(work_id, work_func, *args, **kwargs)
            except queue.Empty:
                continue

    def _start_work(self, work_id, work_func, *args, **kwargs):
        p = mp.Process(
            target=self._do_work, args=(work_id, work_func, *args), kwargs=kwargs
        )
        p.start()
        with self.lock:
            self.running_tasks[work_id] = p.pid
        p.join()
        with self.lock:
            if work_id in self.running_tasks:
                del self.running_tasks[work_id]
            if work_id in self.current_tasks:
                del self.current_tasks[work_id]
            self.active_work.value -= 1

    def _do_work(self, work_id, work_func, *args, **kwargs):
        start_time = datetime.datetime.now()
        try:
            result = work_func(*args, **kwargs)
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.results[work_id] = {"result": result, "duration": duration}
        finally:
            with self.lock:
                self.active_work.value -= 1

    def shutdown(self):
        self.stop_event.set()
        for p in self.processes:
            p.join()


# Example usage
if __name__ == "__main__":

    def example_work(duration):
        time.sleep(duration)
        return f"Work completed in {duration} seconds"

    worker = BackgroundWorker(work_lim=3)  # Using 24 cores

    worker.request_work("task1", example_work, 2)
    worker.request_work("task2", example_work, 3)
    worker.request_work("task3", example_work, 1)
    worker.request_work("task4", example_work, 4)
    worker.request_work("task5", example_work, 5)

    time.sleep(1)  # Give some time for tasks to start

    print("Queue status:", worker.get_queue_status())
    print("Removing task3 from queue:", worker.remove_work("task3"))
    print("Queue status after removal:", worker.get_queue_status())

    print("Running tasks:", worker.get_running_tasks())
    print("Removing 2:", worker.force_kill_work("task2"))
    print("Running tasks:", worker.get_running_tasks())

    time.sleep(10)  # Wait for all tasks to complete

    print("Results:", worker.get_results())

    worker.shutdown()
