VIDEO SUSTENTACION


[![Texto alternativo](https://img.youtube.com/vi/SWQEl9pLvOs/0.jpg)](https://www.youtube.com/watch?v=SWQEl9pLvOs)


# poker_card_totalizer
**poker_card_totalizer** es una aplicación que implementa redes neuronales convolucionales y visión artificial para identificar el valor de las cartas de poker (la cartas son de la categorías 7 a la k)

# Solucion

## Dataset
- Entrenamiento: 'dataset/train'
  - categoria 7: 'dataset/train/7' (50) imagnes
  - categoria 8: 'dataset/train/8' (50) imagnes
  - categoria 9: 'dataset/train/9' (50) imagnes
  - categoria 10: 'dataset/train/0' (50) imagnes
  - categoria j: 'dataset/train/j' (50) imagnes
  - categoria k: 'dataset/train/k' (50) imagnes
  - categoria q: 'dataset/train/q' (50) imagnes
- Prueba --> 'dataset/test'
  - categoria 7: 'dataset/test/7' (18) imagnes
  - categoria 8: 'dataset/test/8' (18) imagnes
  - categoria 9: 'dataset/test/9' (18) imagnes
  - categoria 10: 'dataset/test/0' (18) imagnes
  - categoria j: 'dataset/test/j' (18) imagnes
  - categoria k: 'dataset/test/k' (18) imagnes
  - categoria q: 'dataset/test/q' (18) imagnes

## Matriz de confusion
### model_one

| 17 |  0 |  0 |  1 |  0 |  0 |  0 |
|----|----|----|----|----|----|----|
|  0 | 18 |  0 |  0 |  0 |  0 |  0 |
|  0 |  0 | 17 |  1 |  0 |  0 |  0 |
|  0 |  0 |  6 | 12 |  0 |  0 |  0 |
|  0 |  0 |  0 |  0 | 18 |  0 |  0 |
|  0 |  0 |  0 |  0 |  1 | 15 |  2 |
|  0 |  0 |  0 |  0 |  0 |  0 | 18 |

### model_two

| 17 |  0 |  0 |  1 |  0 |  0 |  0 |
|----|----|----|----|----|----|----|
|  0 | 18 |  0 |  0 |  0 |  0 |  0 |
|  0 |  0 | 14 |  4 |  0 |  0 |  0 |
|  1 |  0 |  5 | 12 |  0 |  0 |  0 |
|  0 |  0 |  0 |  0 | 17 |  0 |  1 |
|  0 |  0 |  0 |  0 |  0 | 17 |  1 |
|  0 |  0 |  0 |  0 |  0 |  0 | 18 |

### model_three

| 18 |  0 |  0 |  0 |  0 |  0 |  0 |
|----|----|----|----|----|----|----|
|  0 | 18 |  0 |  0 |  0 |  0 |  0 |
|  0 |  0 | 18 |  0 |  0 |  0 |  0 |
|  1 |  0 |  2 | 15 |  0 |  0 |  0 |
|  0 |  0 |  0 |  0 | 18 |  0 |  0 |
|  0 |  0 |  0 |  0 |  2 | 16 |  0 |
|  0 |  0 |  0 |  0 |  0 |  0 | 18 |


## Metricas
| Nombre | Accuracy | Precision | Recall | F1 Score | Loss | Epocas de entrenamiento | Tiempo de respuesta |
|--------|----------|-----------|--------|----------|------|-------------------------|---------------------|
| model_one | 91.26 | 92.14     | 91.26  | 91.19    | 27.50| 50                      |           **        |
| model_two | 89.68 | 89.81     | 89.68  | 89.67    | 32.82| 50                      |            **       |
| model_three| 96.13| 96,95     | 96.03  | 95.99    | 23.84| 28                      |           **        |

Analisis 
El modelo 2 implementa una estructura con cuatro capas convolucionales y utiliza la función de activación "relu". Cada capa convolucional se combina con una capa de pooling para reducir la dimensionalidad espacial. Por otro lado, el modelo 3 también consta de cuatro capas convolucionales, pero utiliza la función de activación "LeakyReLU" y no incorpora capas de pooling. En cambio, el modelo 1 emplea tres capas convolucionales con diferentes configuraciones de kernel, pasos y filtros. Utiliza la función "relu" en las dos primeras capas y "tanh" en la tercera, junto con capas de pooling. Estas variaciones en la arquitectura y las funciones de activación dan lugar a diferencias en la forma en que los modelos extraen y procesan las características de los datos de entrada.
Los filtros son fundamentales para extraer características relevantes de los datos de entrada
