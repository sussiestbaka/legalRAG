import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion.persist import add_document, load_index

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            print(f"New PDF: {event.src_path}")
            vectordb = load_index()
            _, dup = add_document(event.src_path, vectordb)
            print("Duplicate" if dup else "Ingested")

if __name__ == "__main__":
    watch_folder = "data/incoming"
    os.makedirs(watch_folder, exist_ok=True)
    observer = Observer()
    observer.schedule(PDFHandler(), watch_folder, recursive=False)
    observer.start()
    print(f"Watching {watch_folder}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()