# Dual Encoding for Zero-Example Video Retrieval
## Descripción detallada del problema
Descripción detallada del problema que enfrenta el artículo: deben indicar la motivación del artículo, el problema que trata de resolver, por qué el problema es relevante y definirlo como un problema de Aprendizaje de Máquina (definido como input y output).
### Motivación
La propuesta tiene el objetivo de generar un método libre de contexto que entienda la semántica para ambos, videos y oraciones.
### Problema
"Dual Encoding for Zero-Example Video Retrieval" es un problema donde se tiene una query descrita en texto de lenguaje natural y se intenta de obtener frames del video que tengan referencia/relevancia a la query, esto sin tener videos con los tags que lo represente.
También se da la inversa, tener frames de videos y obtener una oración que describa dichos frames, por consiguiente el “dual encoding”.

Proponen un encoding dual de múltiples niveles, donde la combinación de los niveles permite el manejo de ambas modalidades (texto y video).

**Relevancia**

Este problema se puede definir según un problema de Aprendizaje de Máquina de la siguiente manera. Existen dos modalidades de inputs con su respectiva modalidad de output.

Para la primera modalidad, el input es una oración en lenguaje natural, es decir, una composición de palabras. En este caso, el output es un set de frames de video, o un set de imágenes, que representan lo descrito por el input.

La segunda modalidad es el caso en que el input está dado por un set de frames de video y el output es una oración en lenguaje natural que describe la sección del video que fue entregada como input.

## Descripción detallada de las métricas
Deben describir las métricas que usan en el artículo para medir el desempeño de la solución. No solo deben enumerar los nombres de las métricas si no investigar a qué se refiere cada una y definirla formalmente (muchos artículos usan métricas que van mucho más allá de acierto o precisión). Deben además decidir qué métricas son las que usarán ustedes en su replicación. Algunos artículos usan muchas métricas distintas. En su caso, basta con que seleccionen un par de las mencionadas en el artículo.

## Descripción de los datos utilizados
Deben investigar qué datos usaron los autores para medir la eficacia de su solución, deben descargarlos para poder analizarlos, describirlos (tamaño, tipo de datos, composición, etc.) y, siempre que sea posible, dejarlos en su repositorio, o de otra forma indicar cómo se deben descargar y procesar. Como plus, deben construir un DataSet y un DataLoader para los datos (o buscar si es que existe alguno disponible en la Web).

Descripción de la arquitectura: Deben describir la arquitectura de Deep Learning usada por los autores para resolver el problema. Idealmente entender y enumerar las fórmulas usadas en la arquitectura. Si las fórmulas o la arquitectura son muy complicadas, pueden sólo incluir una descripción de alto nivel, pero procuren juntarse con su tutor para que los guíe en el entendimiento de la arquitectura.

## Repositorio de código
[Github Dual Encoding Zero-Example Video Retrieval](https://github.com/gpilleux/DualEncZeroExVidRetriev)
