# Pytorch erabiliz, katuen eta txakurren irudiak klasifikatzeko sarea



## Training Set

Entrenatzeko eta balidazioa egiteko datuak [Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)-etik atera dira, entrenatzeko 10000 irudi erabiliko ditugu eta balidatzeko 2500. 

## Zer instalatu behar dugu?
Python behar dugu, espezifikazio hauekin egin ditut nire saiakerak:

- Enviroments: Python 3.10.6 + CUDA 12.0

Nire gomendioa da virtual environment edo anaconda erabiltzea python bertsio ezberdinak edo library ezberdinak instalatzeko
Beste aukera Docker erabiltzea izango litzateke


- Required libraries:

    - Pytorch instalatu behar dugu, nire kasuan, nire ordenagailua GPU bat daukat eta bertsio hau instalatu nuen. Pytorch-en webgunean zure bertsioa eta espezifikazioen arabera komando egokia ematen dizute
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    - Behar ditugun beste liburutegiak instalatu
    ```
    pip3 install -r requirements.txt
    ```

## Datuak prozesatu

Esan bezala entrenatzeko 10000 irudi erabiliko ditugu eta balidatzeko 2500.

$\qquad$ data/ <br>
$\qquad$ $\qquad$ train/ <br>
$\qquad$ $\qquad$ $\qquad$  | -- gato (10000 irudi) <br>
$\qquad$ $\qquad$ $\qquad$  | -- perro (10000 irudi) <br>
$\qquad$ $\qquad$ test/ <br>
$\qquad$ $\qquad$ $\qquad$  | -- gato (2500 irudi) <br>
$\qquad$ $\qquad$ $\qquad$  | -- perro (2500 irudi) <br>

## Sarea entrenatzeko

Entrenamendua egiteko virtual environment-a aktibatu (erabili baduzu) eta hau ejekutatu
```
python train.py
```

Erabilitako hiperparametroak:

- Epoch = 10
- Learning rate = 0.001

## Test egiteko

Gure kasuan, irudiak test_imgs direktorian dauzkate
```
python test.py
```



