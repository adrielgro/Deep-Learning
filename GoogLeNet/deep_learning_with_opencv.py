import numpy as np
import argparse
import time
import cv2

# constructor para recibir los argumentos de la imagen, red neuronal, modelo y diccionario
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="ruta de la imagen")
ap.add_argument("-p", "--prototxt", required=True, help="ruta al proto Caffe 'desplegable', archivo prototxt")
ap.add_argument("-m", "--model", required=True, help="ruta al modelo Caffe pre-entregado")
ap.add_argument("-l", "--labels", required=True, help="ruta a las etiquetas de ImageNet")
args = vars(ap.parse_args())

# cargamos la entrada de la imagen desde el disco
image = cv2.imread(args["image"])

# cargamos las etiquetas de clase desde el disco
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# nuestra CNN requiere dimensiones espaciales fijas para nuestra(s) imagen(es) de entrada
# por lo que debemos asegurarnos de que se redimensione a 224x224 pixeles mientras se
# realiza una resta de media (104, 117, 123) para normalizar la entrada;
# despues de ejecutar este comando nuestro "blob" ahora tiene la forma(shape):
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# cargar nuestro modelo serializado desde el disco
print("[INFO] cargando modelo...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# establecer el blob como entrada a la red y realizar un
# forward-pass para obtener nuestra clasificacion de salida
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] la classification ha tomado {:.5} segundos".format(end - start))

# clasificamos los indices de las probabilidades en orden descendente
# (las probabilidades mas alta primero) y agarramos las predicciones de las 5 mejores
idxs = np.argsort(preds[0])[::-1][:5]

# bucle sobre las predicciones de las 5 mejores y mostrarlas
for (i, idx) in enumerate(idxs):
    # dibujar la prediccion superior en la imagen de entrada
    if i == 0:
        text = "Etiqueta: {}, {:.2f}%".format(classes[idx],
                                           preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    # mostramos la etiqueta predicha + la probabilidad asociada a la consola
    print("[INFO] {}. etiqueta: {}, probabilidad: {:.5}".format(i + 1,
                                                            classes[idx], preds[0][idx]))

# mostrar la imagen de salida
cv2.imshow("Deep Learning UABC", image)
cv2.waitKey(0)