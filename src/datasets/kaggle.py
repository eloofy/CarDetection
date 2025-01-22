import os
import pathlib
import shutil
import requests
import zipfile
from tqdm import tqdm


def download_archive(url: str, download_path: pathlib.Path, filename: str) -> pathlib.Path:
    """
    Скачивает архив по указанному URL и сохраняет его в заданной директории.

    :param url: URL для скачивания архива.
    :param download_path: Директория, куда будет сохранен архив.
    :param filename: Имя файла для сохранения архива.
    :return: Полный путь к скачанному архиву.
    """

    download_path.mkdir(parents=True, exist_ok=True)

    if not os.access(download_path, os.W_OK):
        raise PermissionError(f"Нет прав на запись в директорию {download_path}")

    zip_filepath = download_path / filename

    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            with zip_filepath.open('wb') as f:
                with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        else:
            raise Exception(f"Ошибка при скачивании: {response.status_code}")

    print(f"Файл успешно скачан: {zip_filepath}")
    return zip_filepath


def extract_archive(zip_filepath: pathlib.Path, extract_path: pathlib.Path) -> pathlib.Path:
    """
    Распаковывает архив в указанную директорию с прогрессом.

    :param zip_filepath: Путь к архиву.
    :param extract_path: Директория для распаковки архива.
    :return: Путь к директории, куда архив был распакован.
    """

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_files = zip_ref.namelist()
        with tqdm(total=len(zip_files), desc="Распаковка архива", unit="файл") as pbar:
            for file in zip_files:
                zip_ref.extract(file, extract_path)
                pbar.update(1)

    print(f"Архив распакован в: {extract_path}")
    return extract_path


def restructure_extracted_folder(extract_path: pathlib.Path, target_folder: str) -> None:
    """
    Убирает вложенность распакованного архива и перемещает содержимое в корневую директорию.

    :param extract_path: Директория, где был распакован архив.
    :param target_folder: Имя целевой папки.
    """

    extracted_folder = None

    # Найти целевую папку
    for subdir in extract_path.rglob(target_folder):
        if subdir.is_dir() and subdir.name == target_folder:
            extracted_folder = subdir
            break

    if extracted_folder is None:
        raise FileNotFoundError(f"Папка {target_folder} не найдена после распаковки.")

    final_path = extract_path / target_folder
    if extracted_folder != final_path:
        items_to_move = list(extracted_folder.iterdir())

        # Перемещение содержимого из вложенной папки в корень
        with tqdm(total=len(items_to_move), desc="Перемещение файлов", unit="элемент") as pbar:
            for item in items_to_move:
                target_item_path = final_path / item.name
                if item.is_dir():
                    shutil.move(str(item), str(final_path))
                elif item.is_file():
                    shutil.move(str(item), str(target_item_path))
                pbar.update(1)

    # Удаляем только исходную папку, из которой перемещались данные
    if extracted_folder.exists():
        def delete_directory_recursively(directory: pathlib.Path):
            for item in directory.iterdir():
                if item.is_dir():
                    delete_directory_recursively(item)
                else:
                    item.unlink()
            directory.rmdir()

        delete_directory_recursively(extracted_folder)

    print(f"Папка {target_folder} перемещена в: {final_path}")


def process_dataset(path: pathlib.Path, url: str, target_folder: str = "DETRAC_Upload") -> None:
    """
    Выполняет скачивание, распаковку и реорганизацию структуры архива.

    :param path: Директория для сохранения и обработки архива.
    :param url: URL для скачивания архива.
    :param target_folder: Имя целевой папки.
    """

    filename = "ua-detrac-dataset.zip"
    zip_filepath = download_archive(url, path, filename)
    extract_path = extract_archive(zip_filepath, path)
    restructure_extracted_folder(extract_path, target_folder)
    zip_filepath.unlink()
    print(f"Архив удален: {zip_filepath}")


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent.parent.parent / "data"
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/dtrnngc/ua-detrac-dataset"
    process_dataset(data_path, dataset_url)