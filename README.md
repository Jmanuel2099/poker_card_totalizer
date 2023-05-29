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
- model_one
- model_two
- model_three
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
| model_two |       |           |        |          |      |                         |            **       |
| model_three| 96.13| 96,95     | 96.03  | 95.99    | 23.84| 28                      |           **        |

