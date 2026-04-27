# watcher.py

import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion.persist import add_document, load_index

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(".pdf"):
            return

        print(f"New PDF detected: {event.src_path}")
        # Small delay to ensure file is fully written
        time.sleep(5)

        try:
            vectordb = load_index()
            updated_db, is_duplicate = add_document(event.src_path, vectordb)

            if is_duplicate:
                print(f"Duplicate, skipping: {event.src_path}")
                # Optionally move duplicates to a separate folder
                duplicate_folder = "data/duplicates"
                os.makedirs(duplicate_folder, exist_ok=True)
                shutil.move(event.src_path, os.path.join(duplicate_folder, os.path.basename(event.src_path)))
            else:
                print(f"Successfully ingested: {event.src_path}")
                # Move to processed folder
                processed_folder = "data/processed"
                os.makedirs(processed_folder, exist_ok=True)
                shutil.move(event.src_path, os.path.join(processed_folder, os.path.basename(event.src_path)))
        except Exception as e:
            print(f"Error ingesting {event.src_path}: {e}")

if __name__ == "__main__":
    watch_folder = "data/incoming"
    os.makedirs(watch_folder, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/duplicates", exist_ok=True)

    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=False)
    observer.start()
    print(f"Watching {watch_folder} for PDF files...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()