import tensorflow as tf
import numpy as np
from keras.optimizers import Adam

model = None
output_class = ['Ambroxol Tablet', 'Diapet Kapsul', 'Flamar 50 Tablet', 'Folavit Tablet', 'Laxing Kapsul',
                'Meftormin Tablet', 'Omeprazole Kapsul', 'Ranitidine Tablet', 'Sangobion Kapsul', 'Simcobal Kapsul']
data = {
    "Ambroxol Tablet":
        ["Ambroxol Tablet adalah obat untuk mengencerkan dahak.",
         ],
    "Diapet Kapsul":
        ["Diapet Kapsul merupakan obat herbal yang mengandung ekstrak daun jambu biji, kunyit, buah mojokeling dan kulit buah delima yang dikemas dalam bentuk sediaan kapsul.",
        ],
    "Flamar 50 Tablet":
        ["FLAMAR 50 TABLET mengandung zat aktif Natrium Diclofenac yang merupakan obat golongan NSAID. Obat ini digunakan untuk mengurangi nyeri",
         ],
    "Folavit Tablet":
        ["Folavit Tablet adalah suplemen yang mengandung asam folat atau vitamin B9.",
        ],
    "Laxing Kapsul":
        ["LAXING Kapsul merupakan obat tradisional yang digunakan untuk membantu melancarkan buang air besar (BAB)",
        ],
    "Meftormin Tablet":
        ["METFORMIN Tablet merupakan obat antidiabetes generik yang dapat mengontrol dan menurunkan kadar gula darah pada penderita diabetes tipe 2.",
        ],
    "Omeprazole Kapsul":
        ["OMEPRAZOLE Kapsul merupakan obat golongan proton pump inhibitor (PPI). Obat ini diindikasikan untuk tukak lambung dan tukak duodenum",
         ],
    "Ranitidine Tablet":
        ["Ranitidin Tablet adalah obat yang digunakan untuk mengobati gejala atau penyakit yang berkaitan dengan produksi asam lambung berlebih.",
         ],
    "Sangobion Kapsul":
        ["SANGOBION Kapsul adalah vitamin dan zat besi penambah darah.",
         ],
    "Simcobal Kapsul":
        ["Simcobal Kapsul adalah digunakan untuk mengatasi kekurangan vitamin B 12 pada penderita anemia dan gangguan saraf perifer (neuropati perifer).",
        ],
}


def load_artifacts():
    global model
    model = tf.keras.models.load_model("yuyumodel.h5")

def obat(image_path):
    global model, output_class
    test_image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(150, 150))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    accuracy = np.max(predicted_array) * 100
    accuracy = round(accuracy, 2)  # Mengubah akurasi menjadi dua desimal
    return predicted_value, data[predicted_value][0], accuracy


