# poker_card_totalizer
**poker_card_totalizer** es una aplicación que implementa redes neuronales convolucionales y visión artificial para identificar el valor de las cartas de poker (la cartas son de la categorías 7 a la k)

# Solucion

## VIDEO SUSTENTACION
[![Texto alternativo](https://img.youtube.com/vi/SWQEl9pLvOs/0.jpg)](https://www.youtube.com/watch?v=SWQEl9pLvOs)

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
| Nombre | Accuracy | Precision | Recall | F1 Score | Loss | Epocas de entrenamiento | Tiempo de respuesta  |
|--------|----------|-----------|--------|----------|------|-------------------------|----------------------|
| model_one | 91.26 | 92.14     | 91.26  | 91.19    | 27.50| 50                      | 491.311 milisegundos |
| model_two | 89.68 | 89.81     | 89.68  | 89.67    | 32.82| 50                      | 604.506 milisegundos |
| model_three| 96.13| 96,95     | 96.03  | 95.99    | 23.84| 28                      | 695.891 milisegundos |

## Analisis comparativo
Para inciar se tiene un el model_one el cual implementa en su entrenamiento 50 epocas y numero de muestras de 32. Este modelo tiene 4 capas ocultas de la cuales 3 son convolucionales y una capa densa (full conected) con funcion de activacion relu. Donde dos de  las capas convolucionales utilizan una funcion de activacion relu y una la funcion de activacion tangencial, en estas capas se aimentan los filtros en 10 comenzado en 16 y terminando en 36.

Luego se tiene el model_two el cual implementa en su entrenamiento 50 epoaca y numero de muestras de 60. Este modelo cuenta con 5 capas ocultas donde 4 de ellas con convolucionales donde cada capa tiene la funcion de activacion relu y una ultima capa oculta densa con funcion de activacion relu. En la capas convolucionales se fue aumnentando los filtros sin ningun orden especifico(36, 128, 144, 256).

Y por ultimo se tiene model_three el cual cuenta para el entrenamiento 28 epocas y un batch size 45. Este modelo cuenta al igual que el anteriror modelo con 5 capas ocultas donde 4 son convolucionales y una capas densa (full conected) con funcion de activacion relu. Pero en este caso las capas convolucionales cuentan con una funcion de activacion LeakyReLU y aumnetan los filtros en cada capa de manera exponencial comenzando en 32 hasta 256.

Con lo anterior mencionado y la tabla de **metricas** podemos concluir el modelo con menos epocas de entrenamiento y batch size equilibrado obtiene mejores metricas ya que el model_three que implemento 28 epocas en su entrenamiento tuvo mejorres metircas que los dos modelos que implementaron 50 epocas. Esto tambien se debe a que el modelo con mejores metricas implemento una variante de la funcion de activacion 'relu' la cual es 'LeakyReLU' y un mayor numero de filtros en sus capas convolucionales. Por ultimo cabe acalra que modelo que se utilizara para la prediccion sera el **model_three**

### posibles escenarios de fallo
- Brindarle al modelo cartas con las cuales no fue entrenado, cartas que no esten entre las categorias de la 7 a las k.
- Imagenes donde la luz sea muy escasa o casi en la oscuridad.
- iamgenes donde la carta este muy lejana. (El aplicativo funciona no solo con dos cartas pero al tener varias cartas al momento de capturar la imagen para poder capturar todas las cartas queran muy alejadas).
