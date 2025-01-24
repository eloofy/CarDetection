# YOLOv8Baseline

## Место на диске

Необходимое место может различаться на разных системах.

Рекомендуемое место на диске для хранения для запуска проекта: 40 гб
(из которых 10 гб — для датасета)

## Предварительные требования

Перед началом работы убедитесь, что на вашем компьютере установлены:
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

Также в проект добавлена интеграция с CLEARML и MLFLOW для отслеживания результатов экспериментов.
Чтобы отслеживать результаты экспериментов, необходимо зарегистрироваться на [clearml](https://app.clear.ml/dashboard)
и получить API ключ. Для этого в [настройках профиля](https://app.clear.ml/settings/workspace-configuration) нажмите
`Create new credentials`.

Полученные настройки вида:
```
api {
  web_server:https://app.clear.ml/
  api_server:https://api.clear.ml
  files_server:https://files.clear.ml
  credentials {
    "access_key"="1234567890QWERTYUIOPQWERTYUIOP"
    "secret_key"="QWERTYUIOPQWERTYUIOP!@#$%^&*()QWE1234567RTYUIOPQWERTYUIOP!@#$%^&*()"
  }
}
```

Необходимо сохранить, чтобы потом вставить в файл `configs/clearml.conf` (смотреть 2 пункт)


## Шаги по запуску

### 1. Клонирование репозитория
Склонируйте репозиторий с приложением:
```bash
git clone https://github.com/eloofy/CarDetection.git
```

### 2. Создание конфигурационных файлов

```bash
cp configs/traindataconfigs/data_CARS.yaml.example configs/traindataconfigs/data_CARS.yaml
cp configs/trainerconfigs/trainer_config_cars.yaml.example configs/trainerconfigs/trainer_config_cars.yaml
cp configs/trainmodelconfig/CARS/model_cfg_CARS_exp_1.yaml.example configs/trainmodelconfig/CARS/model_cfg_CARS_exp_1.yaml
cp configs/clearml.conf.example configs/clearml.conf
```

Вставляем в секцию api сохраненные настройки из clearml (смотреть Предварительные требования)


### 3. Скачивание датасета

Датасет можно скачать самостоятельно. 
Иначе он установится автоматически с [kaggle](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset). 

### 4. Запуск приложения

#### Запуск через docker-compose с отображением полосы прогресса

##### Для устройств без gpu

```bash
docker compose -f docker-compose.yaml build car_detection_app
```
```bash
docker compose -f docker-compose.yaml run car_detection_app
```

**Не забудьте изменить настройку `device` на `device: cuda` для запуска с gpu**

##### Для устройств с gpu

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml build car_detection_app
```
```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml run car_detection_app
```

#### Локальный запуск

Необходим установленный python версии 3.10 и выше
```bash
pip install poetry
poetry install
```

```bash
python -m train
```


Организация проекта
------------

    modeling-yolov8/
      ├── configs/
      │   ├── traindataconfigs/              <- конфигурация дынных
      │   ├── trainerconfigs/                <- конфигурация трейнера
      │   └── trainmodelconfig/              <- конфигурация модели
      │
      ├── data/                              <- данные
      │   └── DETRAC_Upload/                 <- датасет
      ├── src/                               <- код фреймворка
      │   ├── callbacks/                     <- callback функции
      │   ├── inference/                     <- предскащание с готовой моделью
      │   ├── settings_update/               <- обновление настроек yolo
      │   ├── utils/                         <- утилиты и скрипты
      │   └── trainer.py                     <- модель с YOLOv8
      │
      └── train.py                           <- онсовной скрипт запуска обучения

--------

## Пример настройки обучения
Для старта обучения необходимо настроить 3 файла:

<h3 id="subsection1">Настройка конфига данных</h3>

```yaml
path: ./DETRAC_Upload
train: 'images/train'
val: 'images/val'

nc: 4
names: ["car", "bus", "van", "other"]
```

Где:

- <u>path</u> - путь к датасету
- <u>train</u> - путь к папке с тренировочными данными
- <u>val</u> - путь к папке с валидационными данными
- <u>nc</u> - количество классов в датасете
- <u>names</u> - именна классов

<h3 id="subsection2">Настройка конфига модели</h3>

```yaml
training_params:

  # Train params

  model: yolov8n.pt # path to model file, i.e. yolov8n.pt, yolov8n.yaml
  data: ./configs/traindataconfigs/data_CARS.yaml  # path to data file, i.e. coco128.yaml
  imgsz: 640 # size of input images as integer
  epochs: 5 # number of epochs to train for
  patience: 5 # epochs to wait for no observable improvement for early stopping of training
  batch: 8 # number of images per batch (-1 for AutoBatch)
  device: cpu # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
  workers: 10 # number of worker threads for data loading (per RANK if DDP)
  optimizer: AdamW # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
  seed: 12345 # random seed for reproducibility
  cos_lr: True # use cosine learning rate scheduler
  lr0: 0.0001 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
  lrf: 0.00001 # final learning rate (lr0 * lrf)
  momentum: 0.98 # SGD momentum/Adam beta1
  weight_decay: 0.005 # optimizer weight decay 5e-4
  warmup_epochs: 1 # warmup epochs (fractions ok)
  warmup_momentum: 0.5 # warmup initial momentum
  warmup_bias_lr: 0.15 # warmup initial bias lr
  box: 10 # box loss gain
  dfl: 1.5 # dfl loss gain
  cls: 0.5 # cls loss gain (scale with pixels)
  project: CARS
  task: detect
  name: cars_exp_1 # experiment name
  close_mosaic: 0 # (int) disable mosaic augmentation for final epochs (0 to disable)
  freeze: 9 # (int or list, optional) freeze first n layers, or freeze list of layer indices during training
  mode: train # mode
  single_cls: False # train multi-class data as single-class
  amp: False # Automatic Mixed Precision (AMP) training, choices=[True, False]
  dropout: 0.09

  # Augmentation params

  hsv_h: 0.01  # image HSV-Hue augmentation (fraction)
  hsv_s: 0.5 # image HSV-Saturation augmentation (fraction)
  hsv_v: 0.1  # image HSV-Value augmentation (fraction)
  degrees : 30  # image rotation (+/- deg)
  translate: 0.1  # image translation (+/- fraction)
  scale: 0.2  # image scale (+/- gain)
  shear: 0.2  # image shear (+/- deg) from -0.5 to 0.5
  # perspective: 0.1  # image perspective (+/- fraction), range 0-0.001
#  flipud: 0.5  # image flip up-down (probability)
#  fliplr: 0.5  # image flip left-right (probability)
  mosaic: 0.1  # image mosaic (probability)
  mixup: 0.01  # image mixup (probability)
  copy_paste: 0  # segment copy-paste (probability)
  erasing: 0.2
```

Параметры обучения:

- <u>model</u> - путь к файлу модели  
- <u>data</u> - путь к файлу конфигурации набора данных  
- <u>imgsz</u> - размер входных изображений (ширина и высота)  
- <u>epochs</u> - количество эпох для обучения модели  
- <u>patience</u> - количество эпох ожидания улучшения перед ранней остановкой  
- <u>batch</u> - размер батча для обучения  
- <u>device</u> - список GPU-устройств для использования в обучении  
- <u>workers</u> - количество потоков для загрузки данных  
- <u>optimizer</u> - оптимизатор для обучения  
- <u>seed</u> - случайное зерно для воспроизводимости  
- <u>cos_lr</u> - использовать ли косинусное изменение скорости обучения (иначе линейное)  
- <u>lr0</u> - начальная скорость обучения  
- <u>lrf</u> - конечная скорость обучения  
- <u>momentum</u> - параметр момента для оптимизатора  
- <u>weight_decay</u> - параметр регуляризации (L2)  
- <u>warmup_epochs</u> - количество эпох разогрева в начале обучения  
- <u>warmup_momentum</u> - начальный момент во время разогрева  
- <u>warmup_bias_lr</u> - начальная скорость обучения для смещений во время разогрева  
- <u>box</u> - вес функции потерь для регрессии боксов  
- <u>dfl</u> 1.5 - вес функции потерь для дистанционного фокуса  
- <u>cls</u> 0.5 - вес функции потерь для классификации  
- <u>project</u> - имя проекта (clearml, mlflow) или путь для сохранения результатов  
- <u>task</u> - тип задачи  
- <u>name</u> - название эксперимента  
- <u>close_mosaic</u> - отключить ли мозаичную аугментацию  
- <u>freeze</u> 8 - количество слоев для заморозки во время обучения  
- <u>mode</u> train - режим работы (обучение, валидация)  
- <u>single_cls</u> True - рассматривать ли все классы как один класс  
- <u>amp</u> False - использовать ли автоматическую смешанную точность (AMP) для обучения  
- <u>dropout</u> - коэффициент дропаута  

Параметры аугментации:

- <u>hsv_h</u> - диапазон изменения оттенка для аугментации HSV  
- <u>hsv_s</u> - диапазон изменения насыщенности для аугментации HSV  
- <u>hsv_v</u> - диапазон изменения значения (яркости) для аугментации HSV  
- <u>degrees</u> - диапазон вращения для аугментации в градусах  
- <u>translate</u> - диапазон сдвига для аугментации  
- <u>scale</u> - диапазон масштабирования для аугментации  
- <u>shear</u> - диапазон сдвига (среза) для аугментации  
- <u>perspective</u> - диапазон перспективных искажений для аугментации  
- <u>flipud</u> - вероятность переворачивания изображения вверх ногами  
- <u>fliplr</u> - вероятность переворачивания изображения слева направо  
- <u>mosaic</u> - вероятность применения мозаичной аугментации  
- <u>mixup</u> - вероятность применения аугментации mixup  
- <u>copy_paste</u> - использовать ли аугментацию копирования-вставки  
- <u>erasing</u> - вероятность применения случайного стирания для аугментации  

> **Note:** После создания конфига модели и данных - необходимо создать кофиг трейнера.

<h3 id="subsection3">Настройка конфига трейнера</h3>

```yaml
experiment_name: FULL_exp_1
cfg_model_path: configs/trainmodelconfig/Tagil/model_cfg_tagil.yaml
cfg_data_path: configs/traindataconfigs/data_Tagil.yaml
pretrained_path: modeling_yolo/FULL_exp_195/weights/best.pt
cfg_mlflow:
  mlflow_tracking_uri: null
path_save_res_nadirs: null
debug_config:
  each_epoch: 1
  predictor_model_base_path: yolov8x-seg.pt
  conf: 0.45
  iou: 0.5
  path_with_test_images: shared_data/test_debug_images/*
  agnostic_nms: True
  max_det: 2000
  imgsz: 1024
  show_labels: False
  show_conf: False
  show_boxes: False
  line_width: False
  augment: 1
  retina_masks: True
yolo_settings_update:
  clearml: True
  mlflow: False
```
> **Note:** Пути указываются относительно корня проекта

Где:
- <u>experiment_name</u> - name of experiment
- <u>cfg_model_path</u> - путь к  [model config](#subsection1) относительно корня проекта
- <u>cfg_data_path</u> - путь к [data config](#subsection2) относительно корня проекта
- <u>pretrained_path</u> - путь к предобученным весам относительно корня проекта
- <u>cfg_mlflow</u> - конфиг mlflow

- <u>path_save_res_nadirs</u> - использование разных углов изображения
- <u>debug_config</u> - дебаг конфиг
  - <u>each_epoch</u> - запускать debug каждые n эпох
  - <u>predictor_model_base_path</u> - базовая модель
  - <u>conf</u> - порог вероятности для предсказания
  - <u>iou</u> - максимально значения пересечения боксов предсказаний
  - <u>path_with_test_images</u> - путь к тестовым изображениям
  - <u>max_det</u> - максимальное количество предсказаний на одно изображение
  - <u>imgsz</u> - размер изображения
  - <u>show_labels</u> - флаг для отображения классов
  - <u>show_conf</u> - флаг для отображения вероятностей классов
  - <u>show_boxes</u> - флаг для отображения рамок bbox
  - <u>line_width</u> - толщина линий bbox
  - <u>retina_masks</u> - использование retina mask для лучшего качества
- <u>yolo_settings_update</u> - обновление настроек YOLOv8
  - <u>clearml</u> - флаг использования
  - <u>mlflow</u> - флаг использования


> **Note:** Конфиг clearml создается в папке `~/claerml.conf`


<h3 id="subsection3">Запуск обучения</h3>

```python train.py --trainer_config <path_to_trainer_config>```

## Инференс YOLOv8

### Обзор
Скрипт predict.py предназначен для выполнения обнаружения объектов с использованием модели YOLOv8. Он принимает изображение в качестве входных данных, запускает модель YOLOv8 для обнаружения объектов и сохраняет результаты, включая сегментации, если это указано. Этот скрипт полезен для выполнения инференса на изображениях с использованием предварительно обученной модели YOLOv8.
### Пример использования
Скрипт можно запускать из командной строки с различными аргументами для указания пути к модели, пути к изображению и других параметров конфигурации YOLO.


    --best_model_path: путь к весам предварительно обученной модели yolov8.
    --path_image_predict: путь к изображению, на котором будут выполняться предсказания.
    --save_results_path: директория, в которой будут сохранены результаты.
    --data_experiment_name: название эксперимента, используется для именования директории с результатами.
    --save_segmentations_json: логическое значение, указывающее, сохранять ли сегментации в файл json.
      аргументы конфигурации yolov8:
        --yolo_conf: порог уверенности для модели yolov8.
        --yolo_iou: порог пересечения по площади (iou) для модели yolov8.
        --yolo_agnostic_nms: логическое значение, указывающее, использовать ли агностическую nms.
        --yolo_max_det: максимальное количество детекций.
        --yolo_imgsz: размер изображения для модели yolov8.
        --yolo_show_labels: логическое значение, указывающее, отображать ли метки на обнаруженных объектах.
        --yolo_show_conf: логическое значение, указывающее, отображать ли показатели уверенности на обнаруженных объектах.
        --yolo_show_boxes: логическое значение, указывающее, отображать ли рамки на обнаруженных объектах.
        --yolo_line_width: ширина линии для рамок.
        --yolo_augment: логическое значение, указывающее, использовать ли аугментацию данных во время предсказания.
        --yolo_retina_masks: логическое значение, указывающее, использовать ли маски ретины.

```python src/inference/predict.py --best_model_path path/to/model_weights.pt --path_image_predict path/to/image.jpg --save_results_path path/to/save/results --data_experiment_name my_experiment --save_segmentations_json True --yolo_conf 0.4 --yolo_iou 0.5 --yolo_agnostic_nms True --yolo_max_det 2000 --yolo_imgsz 1024 --yolo_show_labels False --yolo_show_conf False --yolo_show_boxes False --yolo_line_width 1 --yolo_augment True --yolo_retina_masks True```

## ClearML - сервер для отслеживания результатов экспериментов