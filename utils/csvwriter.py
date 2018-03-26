from typing import List, Any, Sequence, Callable, IO
import threading
import queue
import csv

# marking a child classes method with overrides makes sure the method overrides a parent class method
# this check is only needed during development so its no problem if this package is not installed
# to avoid errors we need to define a dummy decorator
try:
    from overrides import overrides
except ImportError:
    def overrides(method: Callable) -> Callable:
        return method


class CsvWriter:

    def __init__(self, file: IO[str], delimiter: str) -> None:
        self.__csv_file = file
        self.__csv_writer = csv.writer(file, delimiter=delimiter)

    def write_comment(self, comment: str) -> None:
        self.__csv_file.write(f"# {comment}\n")

    def write_row(self, row: Sequence[Any]) -> None:
        self.__csv_writer.writerow(row)

    def write_rows(self, rows: Sequence[Sequence[Any]]) -> None:
        self.__csv_writer.writerows(rows)

    def close(self) -> None:
        self.__csv_file.close()


class ThreadedCsvWriter(CsvWriter):
    """
    Writes content to a csv file using an extra thread
    """

    def __init__(self, file: IO[str], delimiter: str) -> None:
        super().__init__(file, delimiter)
        self.__closed: bool = False
        self.__work_queue: queue.Queue = queue.Queue()  # a thread-safe FIFO queue
        self.__work_thread = threading.Thread(target=self.__do_work)
        self.__work_thread.start()

    @overrides
    def write_comment(self, comment: str) -> None:
        self.__enqueue_work(super().write_comment, comment)

    @overrides
    def write_row(self, row: Sequence[Any]) -> None:
        self.__enqueue_work(super().write_row, row)

    @overrides
    def write_rows(self, rows: Sequence[Sequence[Any]]) -> None:
        self.__enqueue_work(super().write_rows, rows)

    def __enqueue_work(self, func: Callable, *params: Any) -> None:
        self.__work_queue.put((func, params))

    def __do_work(self) -> None:
        while not self.__closed:
            func, params = self.__work_queue.get()
            func(*params)

    def close(self) -> None:
        def stop() -> None:
            self.__closed = True
            # super without arguments does not work here inside a locally defined function
            super(ThreadedCsvWriter, self).close()
        self.__enqueue_work(stop)
        self.__work_thread.join()
