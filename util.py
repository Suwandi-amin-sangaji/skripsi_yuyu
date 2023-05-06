import tensorflow as tf
import numpy as np
from keras.optimizers import Adam

model = None
output_class = ['Ambroxol Tablet', 'Diapet Kapsul', 'Flamar 50 Tablet', 'Folavit Tablet', 'Laxing Kapsul',
                'Meftormin Tablet', 'Omeprazole Kapsul', 'Ranitidine Tablet', 'Sangobion Kapsul', 'Simcobal Kapsul']
data = {
    "Ambroxol Tablet":
        ["Ambroxol adalah obat untuk mengencerkan dahak. Obat ini dapat digunakan pada berbagai kondisi dengan batuk berdahak, termasuk batuk pilek (common cold), bronkitis, emfisema, atau bronkiektasis Ambroxol bekerja dengan memecah serat mukopolisakarida pada dahak. Cara kerja ini akan membuat dahak menjadi lebih encer dan lebih mudah dikeluarkan saat batuk. Ambroxol dapat ditemukan dalam sediaan tablet, sirop, dan drops (obat tetes)",
         "4XOAGNzWvqY", "oKFOqMZmuA8"],
    "Diapet Kapsul":
        ["DIAPET merupakan obat herbal yang mengandung ekstrak daun jambu biji, kunyit, buah mojokeling dan kulit buah delima yang dikemas dalam bentuk sediaan kapsul. Diapet digunakan untuk membantu mengurangi frekuensi buang air besar. Tidak boleh diberikan pada anak dibawah 5 tahun dan penderita harus minum oralit.",
         "Bhi7S06pwv4", "IHPBJySIXZw"],
    "Flamar 50 Tablet":
        ["FLAMAR 50 MG 10 TABLET mengandung zat aktif Natrium Diclofenac yang merupakan obat golongan NSAID. Obat ini digunakan untuk mengurangi nyeri, gangguan inflamasi (radang), dismenore (nyeri haid), nyeri ringan sampai sedang pasca operasi khususnya ketika pasien juga mengalami peradangan. Dapat pula digunakan untuk mengurangi rasa sakit pada penderita arthritis, rheumatoid arthritis, osteoarthritis, sakit gigi, migrain akut, asam urat dan nyeri karena batu ginjal dan batu empedu. Dalam penggunaan obat ini harus SESUAI DENGAN PETUNJUK DOKTER",
         "aUwFXDLOFO0", "w0ikFMTuS9c"],
    "Folavit Tablet":
        ["Folavit Tablet adalah suplemen yang mengandung asam folat atau vitamin B9. Asam folat adalah nutrisi penting yang dibutuhkan oleh tubuh untuk membantu pembentukan sel-sel darah merah dan DNA, serta mendukung pertumbuhan dan perkembangan janin yang sehat selama kehamilan. Kekurangan asam folat dapat menyebabkan anemia, masalah janin, dan masalah kesehatan lainnya. Folavit Tablet dapat direkomendasikan oleh dokter atau profesional kesehatan untuk memenuhi kebutuhan asam folat yang meningkat selama kehamilan, menyusui, atau pada kondisi medis tertentu. Namun, sebaiknya konsultasikan dengan dokter sebelum mengonsumsi suplemen ini untuk mengetahui dosis yang tepat dan memastikan kesesuaian dengan kondisi kesehatan Anda.",
         "bYVih298o1Y", "6R8YObQbE88"],
    "Laxing Kapsul":
        ["LAXING merupakan obat tradisional yang digunakan untuk membantu melancarkan buang air besar (BAB)",
         "qAGCI0-pQ3E", "rgEEXhbar3A"],
    "Meftormin Tablet":
        ["METFORMIN 500 MG merupakan obat antidiabetes generik yang dapat mengontrol dan menurunkan kadar gula darah pada penderita diabetes tipe 2. Metformin termasuk ke dalam obat antidiabetes golongan Biguanide, yang bekerja dengan cara menghambat produksi glukosa (glukoneogenesis) di hati. Penghambatan tersebut mengakibatkan terjadinya penundaan absorbsi atau penyerapan glukosa di usus, sehingga menurunkan glukosa plasma baik basal maupun postprandial (setelah makan). Selain itu, Metformin juga bekerja dengan memperbaiki sensitivitas insulin dengan cara meningkatkan ambilan dan penggunaan glukosa di jaringan perifer. Dengan demikian, maka akan terjadi perbaikan toleransi glukosa pada pasien diabetes tipe 2. Obat ini dapat dikonsumsi secara tunggal, dikombinasikan dengan obat antidiabetes lain, atau diberikan bersama insulin. Dalam penggunaan obat ini harus SESUAI DENGAN PETUNJUK DOKTER.",
         "lHyL41grGUo", "2I8Tjb4Fy-Q"],
    "Omeprazole Kapsul":
        ["OMEPRAZOLE merupakan obat golongan proton pump inhibitor (PPI). Obat ini diindikasikan untuk tukak lambung dan tukak duodenum, tukak lambung dan duodenum yang terkait dengan AINS, lesi lambung dan duodenum, regimen eradikasi H. pylori pada tukak peptik, refluks esofagitis, Sindrom Zollinger Ellison.",
         "jAqVxsEgWIM", "xhW0RTg8kRI"],
    "Ranitidine Tablet":
        ["Ranitidin adalah obat yang digunakan untuk mengobati gejala atau penyakit yang berkaitan dengan produksi asam lambung berlebih. Beberapa kondisi yang dapat ditangani dengan ranitidin adalah tukak lambung, penyakit maag, penyakit asam lambung (GERD), dan sindrom Zollinger-Ellison. Produksi asam lambung yang berlebihan dapat memicu iritasi serta peradangan pada dinding lambung dan saluran pencernaan. Hal ini dapat menyebabkan berbagai gejala, seperti rasa panas pada ulu hati dan tenggorokan, mual, serta kembung. Ranitidin bekerja dengan cara menghambat produksi asam lambung yang berlebih, sehingga gejala tersebut dapat mereda.Businessman suffering from stomach pain.Pada tahun 2019, BPOM sempat menarik ranitidin dari peredaran karena terbukti terkontaminasi N-Nitrosodimethylamine (NDMA), yaitu zat yang dapat menimbulkan kanker jika dikonsumsi dalam jumlah berlebih atau dalam jangka panjang. Namun, setelah kajian secara paralel oleh industri farmasi dan BPOM, perlu diketahui bahwa produk ranitidin yang beredar sekarang telah dipastikan tidak mengandung NDMA melebihi batas yang diperbolehkan.",
         "rYwBL_6hB2I", "I_fUpP-hq3A"],
    "Sangobion Kapsul":
        ["SANGOBION adalah vitamin dan zat besi penambah darah dengan kandungan Ferrous gluconate, manganese sulfate, copper Sulfate, vitamin C, folic acid, vitamin B12. Kandungan pada produk ini membantu proses pembentukan hemoglobin ditubuh sehingga dapat membantu mengatasi anemia saat menstruasi, hamil, menyusui, masa pertumbuhan, dan setelah mengalami pendarahan.",
            "rYwBL_6hB2I", "I_fUpP-hq3A"],
    "Simcobal Kapsul":
        ["Simcobal adalah sediaan obat dalam bentuk kapsul yang diproduksi oleh Lapi Indonesia. Simcobal digunakan untuk mengatasi kekurangan vitamin B 12 pada penderita anemia dan gangguan saraf perifer (neuropati perifer). Setiap kapsul Simcobal mengandung Mecobalamin 500 mcg. Mecobalamin adalah vitamin B12 yang terjadi secara alami di dalam tubuh. Pada kondisi anemia, mecobalamin dapat meningkatkan produksi eritrosit dengan menambah sintesis asam nukleat di tulang belakang serta membantu proses pematangan dan pembelahan eritrosit.",
            "GbE9C2tTW2k", "PkfX4sZwrQ4"],
}


def load_artifacts():
    global model
    model = tf.keras.models.load_model("yuyumodel.h5")


def classify_waste(image_path):
    global model, output_class
    test_image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(64, 64))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    return predicted_value, data[predicted_value][0], data[predicted_value][1], data[predicted_value][2]
