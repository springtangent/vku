import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
from . import config
from . import util
import os.path
import pathlib
import threading

class ShaderCompileEventHandler(FileSystemEventHandler):
    def compile(self, event):
        print("event handler thread_id: ", threading.get_ident())
        print("event:", event)
        if event.is_directory:
            return

        p = pathlib.Path(event.src_path)
        if p.suffix == '.spv':
            return

        shader_path = p.parent
        input_filename = p.stem + p.suffix
        output_filename = util.get_output_filename(shader_path, input_filename)
        util.compile_shader(shader_path, input_filename, output_filename)

    def on_modified(self, event):
        self.compile(event)

    def on_created(self, event):
        self.compile(event)


def main():
    print("main thread_id: ", threading.get_ident())

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = config.RESOURCE_PATH
    event_handler = LoggingEventHandler()
    compiler = ShaderCompileEventHandler()
    observer = Observer()
    observer.schedule(compiler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()