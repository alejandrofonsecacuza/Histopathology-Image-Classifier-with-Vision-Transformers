
# Histopathology Image Classifier with Vision Transformers (ViT)

Este proyecto implementa un sistema de **clasificación automática de imágenes histopatológicas** utilizando **Vision Transformers (ViT)**.  
El objetivo es explorar el potencial de los *Transformers* en el análisis de imágenes médicas, superando los enfoques tradicionales basados en CNN.

## ✨ Características principales
- Uso de **Vision Transformer (ViT)** preentrenado como backbone.
- **Transfer Learning & Fine-Tuning** para ajustar el modelo a las clases específicas.
- **Data Augmentation** estratificado para mejorar la generalización.
- **Evaluación con métricas estándar** (accuracy, precision, recall, F1-score).
- **Explainable AI (XAI)** mediante técnicas de visualización de atención.
- Implementado en **PyTorch**.

## 📊 Dataset
El conjunto de datos utilizado está disponible públicamente en [Figshare](https://figshare.com/articles/dataset/A_histopathological_image_dataset_for_endometrial_disease_diagnosis/7306361).  
Contiene **3300 imágenes histopatológicas** clasificadas en 4 categorías principales:

- `NE`: Endometrio Normal  
- `EA`: Adenocarcinoma Endometrioide  
- `EP`: Pólipo Endometrial  
- `EH`: Hiperplasia Endometrial  

> ⚠️ El dataset **no se incluye directamente en este repositorio** por temas de licencia y tamaño. Debes descargarlo manualmente.

## 🛠️ Instalación
Clona el repositorio y crea un entorno virtual:

```bash
git clone https://github.com/usuario/histopathology-vit.git
cd histopathology-vit

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
````

## ▶️ Entrenamiento del modelo

```bash
python train.py --data_path ./dataset --epochs 10 --batch_size 32 --lr 3e-5
```

## 📈 Resultados

El modelo alcanzó una **accuracy del 80.28%** en el conjunto de prueba, superando a modelos CNN tradicionales en este dataset.

Ejemplo de matriz de confusión:
![Confusion Matrix](docs/confusion_matrix.png)

Ejemplo de mapa de atención del ViT (XAI):
![Attention Map](docs/attention_map.png)


```

## 📚 Tecnologías

* [PyTorch](https://pytorch.org/)
* [Timm](https://github.com/huggingface/pytorch-image-models)
* [Scikit-learn](https://scikit-learn.org/)
* [Matplotlib / Seaborn](https://matplotlib.org/) 

## 🚀 Próximos pasos

* Ampliar el dataset con nuevas muestras.
* Optimizar hiperparámetros y explorar arquitecturas híbridas.
* Integrar interpretabilidad avanzada con Grad-CAM y Attention Rollout.


