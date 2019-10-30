# Dual Encoding for Zero-Example Video Retrieval
## Descripción detallada del problema

### Motivación

La motivación del artículo es poder crear un modelo que sea capaz de codificar palabras e imágenes bajo una misma representación sin la necesidad de extraer conceptos relevantes que  generan relaciones entre ambos contextos. En otras palabras, el modelo debe ser capaz de entender la semántica entre ambos dominios a partir de un método libre de contexto.

Esto resulta ser interesante al momento de extrapolar este problema, ya que entrega evidencia que se pueden relacionar dos dominios distintos sin la necesidad de extraer características relevantes de alguno de los dos, teniendo en cuenta que se considera la variable temporal entre ambos dominios.

### Problema
El problema que se aborda es el de conseguir un set de frames de video que representan un evento y éste es descrito semánticamente a partir de una *query* en forma de lenguaje natural.

La hipótesis del artículo es que dado un video o una query, en primera instancia, estos deben ser codificados en una poderosa representación por si mismo para luego aprovechar ambas representaciones con un modelo de redes neuronales [[1]].

[1]: https://arxiv.org/pdf/1809.06181.pdf

### Relevancia
El artículo tiene suma **relevancia** al cambiar el paradigma de los modelos basados en conceptos, los cuales automáticamente detectan los conceptos relevantes y los asocian a un evento en particular. Para estos tipos de modelos se tiene la dificultad de escoger los conceptos para poder entrenarlos, ya que no es tarea simple escoger conceptos que se puedan detectar y además representar en ambos dominios simultáneamente.

### Planteamiento como problema de Aprendizaje de Máquina

Este problema se puede definir según un problema de ***Aprendizaje de Máquina*** de la siguiente manera. Existen dos modalidades de inputs con sus respectivas modalidades de outputs.

La primera modalidad es el **video-to-text**, donde el input es un set de frames de video, o un conjunto de imágenes. En este caso, el output es una oración en lenguaje natural, es decir, una composición de palabras que describe semánticamente el fragmento de video que fue entregado como input.

La segunda modalidad o **text-to-video** es el caso en que el input es una oración en lenguaje natural y el output es un set de frames de video que son descrito semánticamente por el input.


## Descripción detallada de las métricas

Las métricas de evaluación que utilizaré son las mismas que utilizaron en el paper y que describiré a continuación.

1. **Recall@K (R@K, K = 1, 5, 10):** Porcentaje de obtener al menos un elemento correctamente clasificado dentro de los K elementos con mayor probabilidad de aserción. El modelo tiene un **mejor** desempeño a **mayor R@K**.
2. **Suma de Recalls:** Debido a que se evaluan ambas modalidades, se tienen más posibles respuestas correctas para la modalidad *video-to-text*, ya que hay múltiples oraciones correctas para un video, mientras sólo hay un video correcto para cada oración. Debido a esto, para tener una comparación más fiable, se considera la suma de todos los recalls (R@K, para K = 1, 5, 10) de ambas modalidades.
3. **[Median rank (Med r)]:** Es la mediana del rank. El i-ésimo rank se calcula como el error mínimo de un conjunto de errores considerando los primeros *i* elementos. A grandes rasgos, sirve para ignorar *outlayers* que puedan alterar el desempeño del modelo (ya sea para bien o para mal). El modelo tiene un **mejor** desempeño a **menor Med r**.
4. **Mean Average Precision (mAP):** Es el promedio de los Average Precision (AP) calculado para cada clase. El AP se calcula como el área bajo la curva de Precision v/s Recall. En este contexto, considerando la modalidad *video-to-text*, la clase está dada por todas las oraciones del dataset y con la modalidad *text-to-video*, la clase se representa como todos los videos.

[Median rank (Med r)]: [https://www.bmartin.cc/pubs/16aur/index.html](https://www.bmartin.cc/pubs/16aur/index.html)

## Descripción de los datos utilizados

**MSR-VTT Dataset**
Se utiliza el dataset MSR-VTT que contiene 10.000 *video clips* con 200.000 *oraciones de lenguaje natural* que describen el contenido de los videos, también denominadas *captions*. En promedio, se tienen 20 *captions* por video.

La partición que utilizaron para los datos de training, validation y testing son 6.513, 497, 2.990 videos respectivamente.

El modo de uso de cada video es mediante la agrupación de *n* frames que representan 0.5 segundos del video. Luego, se extraen las características o *features* de cada frame a través de una *red convolucional* pre-entrenada denominada **ImageNet CNN** y son estos grupos de *frames* que se utilizan para realizar distintos cálculos que vienen siendo los **encodings globales**, **encodings de la *consciencia* temporal** y **encodings de mejoras locales.**

A fin de cuentas, los datos que se tienen son los *features* de cada *frame* de los 10.000 videos en un archivo de formato binario (.bin) y los *captions* están en formato de texto (.txt).

**Vocabulario/Bag of Words (BoW)**
Para trabajar con texto se utilizaron word embeddings y debido a esto es necesario extraer el vocabulario del training set. Ejecuté el script *vocab.py* del repositorio el cual generó un vocabulario de **7.807** palabras.

**WordtoVec (W2V)**
Se trabaja con el word embedding pre-entrenado W2V que posee 1.743.364 instancias de palabras, cada una de dimensión 500. 

Los datos se pueden descargar ejecutando las celdas correspondientes del siguiente [Colab]. En este mismo ambiente se pueden observar los análisis de los datos que mencioné con anterioridad.

[Colab]: https://colab.research.google.com/drive/1JUQGNamMmAaJFmXqHoLISU4oQw3BEF4n

Además, encontré el código de los Dataloaders que utilizaron en el paper, también adjuntos en el link de Colab. 

## Descripción de la arquitectura

Como en el el paper hay dos modalidades de uso debido al *encoding dual*, existen dos arquitecturas, para las cuales haré una descripción de alto nivel seguido por la enumeración de las fórmulas.

### Primera modalidad: video-to-text

```mermaid
graph LR
FRMS[Frames] --> FEAT[Features]
FEAT --> FV1((fv1))
FEAT --> FG1[FGRU-1]
FG1 --> FGDOTS[...]
FGDOTS --> FGN[FGRU-N]

FEAT --> BGN[BGRU-N]
BGN --> BGDOTS[...]
BGDOTS --> BG1[BGRU-1]

FGN --> OutbiGRU[Output biGRU]
FGDOTS --> OutbiGRU
FG1 --> OutbiGRU

BG1 --> OutbiGRU
BGDOTS --> OutbiGRU
BGN --> OutbiGRU

OutbiGRU --> FV2((fv2))
OutbiGRU --> CNN[CNN]
CNN --> c2c3c4c5[c2,c3,c4,c5]
c2c3c4c5 --> FV3((fv3))

FV1 --> phiV((phiV))
FV2 --> phiV((phiV))
FV3 --> phiV((phiV))
```

A partir de los frames de un video, se calculan sus *features*. El video queda representado por una secuencia de vectores característicos <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/4c10d924c78b9c3efa28c35b55686ee4.svg?invert_in_darkmode" align=middle width=99.65579745pt height=24.657534pt/> donde <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/3e3c6ee78813607a4d976d92c19dd36e.svg?invert_in_darkmode" align=middle width=12.9338583pt height=14.1552444pt/> representa los *features* del t-ésimo frame.
Se entregan los *features* a una red recurrente bidireccional (biGRU), donde el output del t-ésimo elemento está dado por la concatenación de la GRU forward y GRU backward: <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/3c03e0aee10e1d311d4cf8fe67b0e238.svg?invert_in_darkmode" align=middle width=86.49222945pt height=42.0091485pt/>. Juntando todo, se obtiene un mapa característico <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/c8e77a4e9f9e6bb24b8fea77d228dfcd.svg?invert_in_darkmode" align=middle width=141.0825405pt height=24.657534pt/>.
El mapa característico <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998995pt height=22.4657235pt/> es entregado como input a una *red convolucional* (CNN) que utiliza filtros de tamaño <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/28051ccbf325120643ec3577947ed99c.svg?invert_in_darkmode" align=middle width=85.7874798pt height=22.8310566pt/>. El output final de esta red será definido como la concatenación de los outputs de la CNN para los distintos <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.07536795pt height=22.8310566pt/>.

**Level 1. Global Encoding by Mean Pooling**
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/ad3370fbe8330b49c16eed06d4be6f02.svg?invert_in_darkmode" align=middle width=105.3531138pt height=44.69878215pt/></p>

**Level 2. Temporal-Aware Encoding by biGRU**
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/620d3e550e5dc08ef6b7659ba7e3b673.svg?invert_in_darkmode" align=middle width=142.83779895pt height=25.11416325pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/4f0fc6bcea68957c598b08621e20bcf3.svg?invert_in_darkmode" align=middle width=177.88172655pt height=25.11416325pt/></p>

<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/89226535db06dd304df80b21cda3526f.svg?invert_in_darkmode" align=middle width=106.85616315pt height=44.69878215pt/></p>

**Level 3. Local-Enhanced Encoding by biGRU-CNN**
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/a8d132d8e3f4a685827e6561d947637e.svg?invert_in_darkmode" align=middle width=315.48738045pt height=17.0319402pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/dae4e2bc64a78ecfca2d8015ba5d8f35.svg?invert_in_darkmode" align=middle width=138.38667975pt height=19.5269943pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/a1331501ef473e3098d78aac19575bee.svg?invert_in_darkmode" align=middle width=159.1973493pt height=19.5269943pt/></p>

### Segunda modalidad: text-to-video

```mermaid
graph LR
SENTENCE[Frames] --> OHE[One-hot Encoding]
OHE --> WE[Word embedding]
OHE --> FS1((fs1))
WE --> FG1[FGRU-1]
FG1 --> FGDOTS[...]
FGDOTS --> FGN[FGRU-N]

WE --> BGN[BGRU-N]
BGN --> BGDOTS[...]
BGDOTS --> BG1[BGRU-1]

FGN --> OutbiGRU[Output biGRU]
FGDOTS --> OutbiGRU
FG1 --> OutbiGRU

BG1 --> OutbiGRU
BGDOTS --> OutbiGRU
BGN --> OutbiGRU

OutbiGRU --> FS2((fs2))
OutbiGRU --> CNN[CNN]
CNN --> c2c3c4c5[c2,c3,c4]
c2c3c4c5 --> FS3((fs3))

FS1 --> phiS((phiS))
FS2 --> phiS((phiS))
FS3 --> phiS((phiS))
```
La arquitectura para la segunda modalidad es muy similiar a la primera con pequeños cambios.
De partida, el input es una oración y se calculan los *vectores one-hot* de cada palabra que resulta en la secuencia <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/88df58f39df554c6294fd4ec715469f9.svg?invert_in_darkmode" align=middle width=114.59595015pt height=24.657534pt/> donde <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/dde30cc90adc3d7de889d34c65ca6f25.svg?invert_in_darkmode" align=middle width=16.7343pt height=14.1552444pt/> es el vector para la t-ésima palabra. El input del biGRU se calcula como la multiplicación del vector one-hot con una matriz del word embedding. Esta matriz es inicializada con word2vec ya mencionado. El output de la red biGRU es pasada como input a la red convolucional. Esta tiene la diferencia que se utilizan <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/91557e8db62cbf55a9dc69e2c4544517.svg?invert_in_darkmode" align=middle width=70.26238725pt height=22.8310566pt/> y el resultado final de la CNN es la concatenación de los <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/0a5ec44b76d454790dd94ab5cfe77d12.svg?invert_in_darkmode" align=middle width=14.37983415pt height=14.1552444pt/>. Las fórmulas son idénticas, cambiando el argumento.

**Level 1. Global Encoding by Mean Pooling**
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/cb438340102036e69cdf159da1e245a5.svg?invert_in_darkmode" align=middle width=109.1535555pt height=44.69878215pt/></p>

**Level 2. Temporal-Aware Encoding by biGRU**
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/c57ab8e88ade5571d232f42141fc927d.svg?invert_in_darkmode" align=middle width=146.63824065pt height=25.11416325pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/aa6ca24b6205b3d4c8c41d996b2273bd.svg?invert_in_darkmode" align=middle width=181.68216825pt height=25.11416325pt/></p>

<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/5c3012ed4a19bc8f7897767008d50bb0.svg?invert_in_darkmode" align=middle width=106.85616315pt height=44.69878215pt/></p>

**Level 3. Local-Enhanced Encoding by biGRU-CNN**
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/a8d132d8e3f4a685827e6561d947637e.svg?invert_in_darkmode" align=middle width=315.48738045pt height=17.0319402pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/8675bf236ead9451e8e9aea38f36e19d.svg?invert_in_darkmode" align=middle width=116.59253265pt height=19.5269943pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/a7fbe78c0d1bdde608506de4bb394b56.svg?invert_in_darkmode" align=middle width=158.34498735pt height=19.5269943pt/></p>

#### Common Space Learning
Una vez calculado los <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/62fde1b36cfaf9adb62a39f9800099f1.svg?invert_in_darkmode" align=middle width=31.13781825pt height=24.657534pt/> y <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/b4b27ff7613adcdb46ac7052b036e652.svg?invert_in_darkmode" align=middle width=30.2854563pt height=24.657534pt/>, se deben proyectar en un espacio común para poder ser comparados. Para esto se utilizó un algoritmo *open source* y es el que tiene el mejor desempeño en la actualidad: [VSE++].
VSE++ a grandes rasgos es una red neuronal con una Fully Connected (FC) layer que le agregaron Batch Normalization (BN).

<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/b892d31b76c72c41b63cb1808bd14607.svg?invert_in_darkmode" align=middle width=183.5864151pt height=16.438356pt/></p>
<p align="center"><img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/625a67189fc89d9d453e4b370a218498.svg?invert_in_darkmode" align=middle width=181.16626935pt height=16.438356pt/></p>
con <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/111bd6359756ad3da755b311934d08f4.svg?invert_in_darkmode" align=middle width=22.51339365pt height=22.4657235pt/> y <img src="https://rawgit.com/in	git@github.com:gpilleux/DualEncZeroExVidRetriev/master/svgs/caa83b409168bcbeb49ea01c63fe2dd8.svg?invert_in_darkmode" align=middle width=21.7295034pt height=22.4657235pt/> parametrizaciones de la capa FC.

Cabe destacar que la *Dual Encoding Network* y la *Common Space Learning Network* se entrenan en conjunto de forma end-to-end, con la excepción de la CNN que extrae los *features* de los videos, la cual está pre-entrenada.

[VSE++]: [https://github.com/fartashf/vsepp](https://github.com/fartashf/vsepp)

## Repositorio de código
[Github Dual Encoding Zero-Example Video Retrieval](https://github.com/gpilleux/DualEncZeroExVidRetriev)

