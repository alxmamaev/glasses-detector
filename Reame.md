# Детекция очков

В качестве базовой модели хотел использовать MobileNetV2, но она не подошла по размерам (8mb), поэтому остановился на sqeezenet. Датасет сделал на основе CelebaFaces (выбрав лица по атрибуту очков).
Далее просто натрнировал сетку с использованием аугментаций

Запуск: `python3 inference.py /path/to/images.jpg"