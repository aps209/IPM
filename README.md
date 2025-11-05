# IPM - Práctica 2 

![mediapipe_game_ipm](https://github.com/user-attachments/assets/1c51471e-8b4b-4f56-bd25-9cebfacb2af2)


## Introducción

En esta práctica veremos como realizar interfaces para la interacción persona-máquina (IPM o HCI) basadas en visión por computador. La interacción, como hemos visto en teoría no tiene porqué limitarse al diseño y desarrollo de interfaces para la manipulación de sistemas operativos, aunque sí existe también esa vertiente. Existen multitud de aplicaciones en videojuegos o juegos ‘serios’ para rehabilitación u otras finalidades. También existen aplicaciones en interacción persona-entorno, en domótica avanzada (casas inteligentes, edificios inteligentes), que no dejan de ser sistemas informáticos distribuidos con los que se interactúa. La idea es que estos sistemas puedan responder a las necesidades de las personas que los habitan y ayudar o apoyar sus tareas en el día a día. También pueden ‘pasivamente’ analizar lo que ocurre (interacción pasiva) y evitar accidentes o evaluar el estado de salud entre otros (salud electrónica, e-Health, teleasistencia, etc.).

En esta práctica se ha hecho uso de **MediaPipe**, un framework de código abierto desarrollado por Google que permite construir e implementar _pipelines_ de procesamiento multimedia (como video, audio e imágenes) en tiempo real, especialmente útiles para tareas de visión por computador y aprendizaje automático. Ofrece soluciones preentrenadas y optimizadas (como **detección facial**, **reconocimiento de gestos**, **estimación de pose**, **seguimiento de manos** o **segmentación de objetos**) que funcionan eficientemente tanto en dispositivos móviles como en un computador. Gracias a su arquitectura modular y multiplataforma, MediaPipe ha facilitado el desarrollo rápido de aplicaciones de inteligencia artificial.

Este repositorio contiene un juego sencillo usando la librería de MediaPipe que puede servir como guía para el desarrollo de la práctica. Para crear vuesto videojuego, podéis hacer uso de los modelos que ofrece MediaPipe en su página oficial:

- *```Pose Landmarker```* (usado en este repositorio), [aquí](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=es-419).
- *```Hand Landmarker```*, [aquí](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=es-419).
- *```Face Landmarker```*, [aquí](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index?hl=es-419).
- *```Holistic Landmarker```*, [aquí](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker?hl=es-419).

## Prerequisitos

Tener instalado **Conda**, [instalar aquí](https://www.anaconda.com/docs/getting-started/miniconda/install).

## Requisitos

Crear un entorno de conda:
```bash
conda create -n IPM python=3.12
conda activate IPM
```

## Instalación

Se instalan las dependencias necesarias (MediaPipe, Requests, tqdm, cv2, numpy, etc.):
```bash
pip install -r requirements.txt
```

## Descargar pesos

Script para poder descargar los pesos del modelo *Pose Landmarker*:
```bash
python download_models.py
```

Para descargar los pesos de otros modelos como _Hand Landmarker_, _Face Landmarker_ u _Holisitc Landmarker_ debes de descargarlos de los [enlaces](https://github.com/CarloHSUA/IPM/tree/main?tab=readme-ov-file#introducci%C3%B3n) de la página oficial de MediaPipe



## Ejecución
```bash
python app.py
```
