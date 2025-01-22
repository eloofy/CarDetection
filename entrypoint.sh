#!/bin/bash

echo "Обновление настроек yolo..."

.venv/bin/python -m src.settings_update.yolo_settings_update

# Проверяем, существует ли папка
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Папка $DATA_DIR не существует."
    exit 1
fi

found=0

# Перебираем все подкаталоги. Проверяем, есть ли файлы в подкаталоге
for subdir in "$DATA_DIR"/*/; do
    if [ -d "$subdir" ] && [ "$(find "$subdir" -type f | wc -l)" -gt 0 ]; then
        echo "В папке для датасетов $subdir есть файлы."
        found=1
        break
    fi
done

# Если ничего не найдено
if [ $found -eq 0 ]; then
    echo "В папке $DATA_DIR нет подкаталогов с файлами."
    echo -e "\nЗагрузка датасета..."

    .venv/bin/python -m src.datasets.kaggle
fi

echo -e "\nПроверка прав /opt/app/data/..."
#chown -R app_user:app_group /opt/app/data/
chmod -R 777 /opt/app/data/

echo -e "\nИнициализация clearml..."

.venv/bin/clearml-init

if [ $? -eq 0 ]; then
    echo "Запуск модели..."
    exec .venv/bin/python -m train  # Используем exec для запуска сервера
else
    echo "Ошибка при изменении настроек."
    exit 1
fi