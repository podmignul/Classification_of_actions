import streamlit as st
import base64
import numpy as np
import os
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
from collections import Counter
import time


st.set_page_config(page_title='Автоматизированная система распознавания действий человека', layout="wide")

# Получить текущий каталог (где находится исполняемый скрипт)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Пути
logo_path = os.path.join(current_directory, "RZD.jpg")

# Создание блока шапки с заданными цветами
header_container = st.container()
with header_container:
    st.markdown(
        f"""
        <div style="
            background-color: #E21A1A;
            padding: 10px;
            border-radius: 8px;
            display: flex;
            align-items: center;
        ">
        <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}" alt="Логотип" style="height: 40px; margin-top: 20px; margin-bottom: 20px; margin-left: 20px; margin-right: 40px;">
        <h1 style="color: white; font-size: 23pt; margin-bottom: 5px; font-family: 'RussianRail G Pro'; font-weight: 100; text-align: center;">Автоматизированная система распознавания действий человека</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


# Создание вкладок для кнопок "Прогноз", "Статистика" и "Отчет"
file, cam = st.tabs(["Файл", "Камера"])


import pathlib

# Перенаправление PosixPath на WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = os.path.join(current_directory, "models/model.pt")

# Загрузка модели
model = YOLO(model_path)

# Обработка нажатия на вкладку "Файл"
with file:
    st.markdown("<h1 style='font-family:FSRAILWAYTT Book; font-size:35px;'>Анализ файла</h1>", unsafe_allow_html=True)

    file_types = ["jpg", "jpeg", "mp4"]

    st.empty()
    # Загрузка изображения
    uploaded_file = st.file_uploader("Загрузите файл (изображение или видео)", type=file_types)
    st.empty()

    # Список классов
    classes = ['cartwheel', 'catch', 'clap', 'climb', 'dive', 'draw_sword', 'dribble', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit', 'jump', 'pick', 'pour', 'pullup', 'push', 'pushup', 'shoot_ball', 'sit', 'situp', 'swing_baseball', 'sword_exercise', 'throw']

        
    if uploaded_file is not None:
        # Создание временного места для сообщения
        message = st.empty()
        message_placeholder = st.empty()
        message.markdown("<p style='font-family:FSRAILWAYTT Book; font-size:30px;'>Файл обрабатывается...</p>", unsafe_allow_html=True)
        # Преобразование файла в формат, подходящий для модели
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            # Отображение загруженного изображения
            st.image(image_np, caption='Uploaded Image', use_column_width=True)
            # Подача изображения на вход модели
            results = model.predict(image_np)
            # Отображение результатов
            for result in results:
                action_index = result.probs.top1  # получение индекса класса с наибольшей вероятностью
                action_name = classes[action_index]  # получение названия класса
                action_prob = round(result.probs.top1conf.item(), 2)  # получение вероятности класса
                message.markdown(f"<p style='font-family:FSRAILWAYTT Book; font-size:30px;'>Распознанное на фото действие - <span style='color:red; font-weight:bold;'>{action_name}</span>, с вероятностью {action_prob}</p>", unsafe_allow_html=True)
        elif uploaded_file.type.startswith('video/'):
            # Сохранение временного файла на диск
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            # Отображение загруженного видео
            st.video(tfile.name)
            # Преобразование видео в последовательность изображений
            video = cv2.VideoCapture(tfile.name)
            actions = []
            frame_results = []  # Список для хранения результатов для каждого кадра
            fps = video.get(cv2.CAP_PROP_FPS)  # Получение количества кадров в секунду
            current_second = 1
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                # Подача каждого кадра на вход модели
                results = model.predict(frame)
                # Сбор результатов
                for result in results:
                    action_index = result.probs.top1  # получение индекса класса с наибольшей вероятностью
                    action_name = classes[action_index]  # получение названия класса
                    actions.append((current_second, action_name, result.probs.top1conf.item()))  # Сохранение результата для кадра
                if video.get(cv2.CAP_PROP_POS_FRAMES) >= current_second * fps:  # Если прошла секунда
                    # Выбор наиболее часто встречающегося действия за последнюю секунду
                    actions_last_second = [action for second, action, prob in actions if second == current_second]
                    most_common_action = Counter(actions_last_second).most_common(1)[0]
                    frame_results.append(f"На {current_second} секунде видео происходит действие - {most_common_action[0]}")
                    current_second += 1
            video.release()
            # Выбор наиболее часто встречающегося действия
            most_common_action = Counter(actions).most_common(1)[0]
            message.markdown(f"<p style='font-family:FSRAILWAYTT Book; font-size:30px;'>Самое часто встречаемое на видео действие - <span style='color:red; font-weight:bold;'>{most_common_action[0][1]}</span></p>", unsafe_allow_html=True)
            # Отображение результатов для каждой секунды
            results_str = '<br>'.join(frame_results)
            message_placeholder.markdown(results_str, unsafe_allow_html=True)




# Обработка нажатия на вкладку "Камера"
with cam:
    st.markdown("<h1 style='font-family:FSRAILWAYTT Book; font-size:35px;'>Анализ видео с камеры</h1>", unsafe_allow_html=True)

    # Создание checkbox
    checkbox_state = st.checkbox('Включить/выключить распознавание')

    # Создание пустого места для вывода результатов
    result_placeholder = st.empty()

    col1, col2, col3 = st.columns([1,1,1])

    with col2:
        video_live = st.image([])

    # Получение видеопотока с камеры
    cap = cv2.VideoCapture(0)  # 0 - это индекс камеры. Если у вас есть несколько камер, вы можете изменить этот индекс.
    
    # Добавление кнопки для включения/выключения распознавания
    if checkbox_state:
        
        # Обработка каждого кадра видеопотока
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            # Подача каждого кадра на вход модели
            results = model.predict(frame)
            
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Decrease brightness
            frame_bright = cv2.addWeighted(frame_rgb, 0.9, np.zeros(frame_rgb.shape, frame_rgb.dtype), 0, 0)

            # Display the frame
            video_live.image(frame_bright)

            # Отображение результатов каждые 5 секунды
            if time.time() - start_time >= 3:
                for result in results:
                    action_index = result.probs.top1  # получение индекса класса с наибольшей вероятностью
                    action_name = classes[action_index]  # получение названия класса

                    # Обновление пустого места новыми результатами
                    result_placeholder.markdown(f"<p style='font-family:FSRAILWAYTT Book; font-size:30px;'>Распознанное действие - <span style='color:red; font-weight:bold;'>{action_name}</span></p>", unsafe_allow_html=True)
                start_time = time.time()

        cap.release()