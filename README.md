# Реидентификация объектов на аэрофотоснимках


Для обучения и тестирования было использовано виртуальное окружение `venv` (Python 3.10.12), все зависимости указаны в файле `requirements.txt`
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Обучение:
```console
python3 train.py --config config/config_MBR_4G_IBN.yaml
```

Тестирование обученной модели:
```console
python3 test.py --path_weights logs/Veri776/MBR_4G/IBN_256x256/best_mAP.pt
```
