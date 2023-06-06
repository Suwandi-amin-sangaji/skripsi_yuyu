import tensorflow as tf
import numpy as np
from keras.optimizers import Adam

model = None
output_class = ['Ambroxol Tablet', 'Diapet Kapsul', 'Flamar 50 Tablet', 'Folavit Tablet', 'Laxing Kapsul',
                'Meftormin Tablet', 'Omeprazole Kapsul', 'Ranitidine Tablet', 'Sangobion Kapsul', 'Simcobal Kapsul',
                'Ibuprofen Tablet','Cetirizine Tablet','Promag Tablet','Paracetamol Tablet','Clindamycin HCL Kapsul'
                ]
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
    "Ibuprofen Tablet":
        ["Ibuprofen Tablet adalah obat untuk meredakan nyeri dan peradangan.",
         ],
    "Cetirizine Tablet":
        ["Cetirizine Tablet adalah jenis obat yang dapat digunakan untuk mengobati alergi.",
         ],
    "Promag Tablet":
        ["Promag Tablet digunakan untuk mengurangi gejala-gejala yang berhubungan dengan kelebihan asam lambung .",
         ],
    "Paracetamol Tablet":
        ["Paracetamol Tablet merupakan obat yang dapat digunakan untuk meringankan rasa sakit pada sakit kepala, sakit gigi, dan menurunkan demam ",
         ],
    "Clindamycin HCL Kapsul":
        ["Clindamycin HCL Kapsul merupakan Obat untuk mengatasi berbagai infeksi bakteri, seperti infeksi pada paru, kulit, darah, organ reproduksi wanita, atau organ dalam ",
         ],
}


# def load_artifacts():
#     global model
#     model = tf.keras.models.load_model("yuyumodel.h5")

# def obat(image_path):
#     global model, output_class
#     test_image = tf.keras.preprocessing.image.load_img(
#         image_path, target_size=(150, 150))
#     test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
#     test_image = np.expand_dims(test_image, axis=0)
#     predicted_array = model.predict(test_image)
#     predicted_value = output_class[np.argmax(predicted_array)]
#     accuracy = np.max(predicted_array) * 100
#     accuracy = round(accuracy, 2)  # Mengubah akurasi menjadi dua desimal
#     return predicted_value, data[predicted_value][0], accuracy

def load_artifacts():
    global model
    model = tf.keras.models.load_model("yuyumodel.h5")

def obat(image_path):
    global model, output_class
    test_image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    accuracy = np.max(predicted_array) * 100
    accuracy = round(accuracy, 2)  # Mengubah akurasi menjadi dua desimal

    # Menambahkan batasan ambang batas akurasi
    accuracy_threshold = 55  # Anda dapat mengatur nilai ini sesuai kebutuhan

    if accuracy >= accuracy_threshold:
        return predicted_value, data[predicted_value][0], accuracy
    else:
        return "Error: Gambar tidak dapat diklasifikasikan karena berada di luar kelas data pelatihan.", None, accuracy
