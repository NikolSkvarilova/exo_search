from exo_search.entities.star import iterate_stars_from_dir
from pathlib import Path
import threading
import random
import time


def main(
    stars_path: Path,
    mission: str,
    author: str,
    cadence_period: float,
    n_batches: int = None,
    threads: int = 0,
) -> None:
    """Download light curves. Filter them by mission, author, cadence period.

    Args:
        stars_path (Path): directory with stars.
        mission (str): mission.
        author (str): author.
        cadence_period (float): cadence period.
        n_batches (int, optional): how many to download. Downloads all by default.
        threads (int, optional): how many threads to create. Defaults to 0.
    """
    # Use threads
    if threads > 0:
        n_stars = len([item.name for item in stars_path.iterdir() if ".json" in item.name])
        threads = min(threads, n_stars)
        n_stars_group = n_stars // threads

        # Create threads
        threads_arr = []
        for i in range(threads):
            # Calculate which stars to pass to the thread
            start = i * n_stars_group
            end = (i + 1) * n_stars_group if i != threads - 1 else n_stars
            # Create the thread
            threads_arr.append(
                threading.Thread(
                    target=download,
                    args=(
                        stars_path,
                        start,
                        end,
                        mission,
                        author,
                        cadence_period,
                        n_batches,
                    ),
                )
            )

        # Start threads
        for thread in threads_arr:
            time.sleep(random.randrange(5, 30))
            thread.start()

        # Join threads
        for thread in threads_arr:
            thread.join()

    else:
        # Do not use threads
        download(stars_path, None, None, mission, author, cadence_period, n_batches)


def download(stars_path: Path, start: int, end: int, mission: str, author: str, cadence_period: float, n_batches: int):
    """Download light curves. Filter them by mission, author, cadence period.

    Args:
        stars_path (Path): directory with stars.
        mission (str): mission.
        author (str): author.
        cadence_period (float): cadence period.
        n_batches (int, optional): how many to download.
    """
    for star in iterate_stars_from_dir(stars_path, start, end):
        download_ok = star.download_lc(mission, author, cadence_period, n_batches)
        if not download_ok:
            continue
        star.save_to_file(stars_path / f"{star.primary_name}.json")