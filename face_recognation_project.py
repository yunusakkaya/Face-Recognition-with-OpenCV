import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # yapay zeka algoritmalarınn bulunduğu kütüphaneler
from tensorflow.keras import models, layers 


Image_Size= 256 # resimin pixel boyutu
Batch_Size = 2 # ayrılacak grup sayısı
Channels=3 # red green blue
Epochs=100 # eğitim modelinin devir/tekrar sayısı

# eğitim verisinin yüklenmesi
dataset = tf.keras.preprocessing.image_dataset_from_directory(r"real_and_fake_face",shuffle=True,image_size = (256,256),batch_size=32)

#veri dosyasındaki klasör isimleri sınıf ismi olarak ayarlanır
class_names = dataset.class_names
class_names

# verinin eğitim için ayrılması
def splitting_dataset_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000): 
    
    ds_size=len(ds)
    
    if shuffle: # shuffle true 
        ds = ds.shuffle(shuffle_size, seed=12) # verinin karıştırılması
        
    train_size=int(train_split * ds_size) # eğitim setinin boyutu ayrlanması
    val_size= int(val_split * ds_size) # validation, doğruluk testi için kullanılan veri miktarının boyutu ayarlanması
    
    train_ds= ds.take(train_size) #eğitim setinin veri içinden alınması
    
    val_ds = ds.skip(train_size).take(val_size) # validation setinin alınması
    test_ds = ds.skip(train_size).skip(val_size) # test setinin alınması
    
    return train_ds, val_ds, test_ds # sonuçları döndür

train_ds, val_ds, test_ds = splitting_dataset_tf(dataset) # datayı bölme fonksiyonunun çağırılması

# işlem kolaylığı için görüntü boyutunu küçültme
resize_and_rescale = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(Image_Size,Image_Size), layers.experimental.preprocessing.Rescaling(1.0/255)])

# görüntüleri rastgele çevirerek eğitmek için
data_aug = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),layers.experimental.preprocessing.RandomRotation(0.2),])

# girdinin matris olarak boyutunun ayarlanması
input_shape = (Batch_Size,Image_Size, Image_Size,Channels)
n_classes = 2

#modelin basamaklarının belirlenmesi
model = models.Sequential([resize_and_rescale,data_aug,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape), # evrişiim katmanları
    layers.MaxPooling2D((2,2)), # pooling katmanları
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    
    layers.Flatten(), # bitiş bağlantı katmanları
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation= 'softmax'),
    
])

# modelin eğitilmesi 
model.build(input_shape=input_shape)

model.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy']) # optimizer ın belirlenmesi
history = model.fit(train_ds,epochs=1,batch_size=Batch_Size,verbose=1,validation_data=val_ds)

scores = model.evaluate(test_ds) #  doğruluk payının ölçümü


def pred(model, img): # tahmin fonksiyonun oluşturulması
    img_array = tf.keras.preprocessing.image.img_to_array(frame) # resmin arraya dönüştürülmesi
    img_array = tf.expand_dims(img_array, 0) 
    
    predictions = model.predict(img_array) # tahmin işlemi
    
    predicted_class = class_names[np.argmax(predictions[0])] # tahmine göre sınıfa atama
    confidence = round(100 * (np.max(predictions[0])), 2) # uygunluk yüzdesi hesabı
    return predicted_class, confidence # sonuçları döndür




import cv2  #opencv import
import face_recognition #yüz tanıma kütüphanwsi


face_cascade = cv2.CascadeClassifier(r"C:\Users\yunus\Desktop\ıvır zıvır\haarcascade_frontalface_default.xml") # viole jones yüz tanıma hazır algoritma

train_img = face_recognition.load_image_file(r"resim.jpg") #yüzü tanınacak kişinin fotoğrafı
train_img_encoding = face_recognition.face_encodings(train_img)[0] # yüzü kütüphaneye tanıtma (0 siyah beyaz için)

enc_list = [train_img_encoding] # tanııtlmış yüzü tanınan yüzler listesine eklemek
name_list = ["Yunus Akkaya"] # yüzlerin isimlerini kaydetmek

cap = cv2.VideoCapture(0) # kamerayı açıp görüntüyü kaydetmek

eyes_locations=[]
face_locations = [] #yüz kordinatları
face_encodings = [] #videodaki yüzün kaydı
face_names = [] #ekrana bastırılacak isimler
process_this_frame = True 
f_location = [(0, 0, 0, 0)]
counter=0

while True:
    # Videodan anlık kare alma
    ret, frame = cap.read()
    
    # Aldığımız kareyi 1/4 oranında küçültüyoruz ve bu daha hızlı sonuç vermeyi sağlıyor
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # BGR(opencv) türündeki resmi RGB(face_recognition) formatına çeviriyoruz
    rgb_small_frame = small_frame[:, :, ::-1]
    
    predicted_class, confidence = pred(model, frame)

    if process_this_frame:
               
            
       # Uyumlu tüm yüzlerin lokasyonlarını bulma
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        
        if f_location == face_locations: # yüz aynı yerdeyse sayıcıyı artır hareket algılama için
            
            counter=counter+1
            
        else:
            counter=0    
        
        f_location = face_locations

        for face_encoding in face_encodings:
            # Eşleşen yüzleri topla
            matches = face_recognition.compare_faces(enc_list, face_encoding)
            name = "Bilinmeyen"
            
            if counter >= 50:
                name = "this is a photo"

            # tek yüz birden fazla yüzle eşleşirse ilkini bastır
            if True in matches:
            
                    
                first_match_index = matches.index(True) # eşleşmeleri kaydetme
                name = name_list[first_match_index] # eşleşenlerin sırasına göre ismi kaydetme
                
                if counter >= 50: # görüntü ne kadar süre donuk kalırsa fotoğraf kalıcak ayarlanamsı
                    name = "this is a photo" # belli bir süre sabitlikten sonra fotoğraf sayılacak
            
            face_names.append(name) # eşleşen yüzün ismini gösterilecek isimlere ekle

    process_this_frame = not process_this_frame # aynı frame işlenmesin 
    print(len(face_names)) 
    if len(face_names) >=2 :
        print ("ekranda",str(len(face_names)),"kişi var") # ekranda birden fazla kişi varsa uyarı ver
    
    # Sonuçları göster
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Yüzü çerçeve içerisine al
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2) # çerçeve ekleme
        font = cv2.FONT_HERSHEY_DUPLEX # font belirleme
        cv2.putText(frame, name, (left + 6, bottom + 15), font, 1.0, (255, 255, 255), 1) # isim yazdırma
        cv2.putText(frame,predicted_class, (left + 12, bottom + 40), font, 1.0, (255, 255, 255), 1) # sınıf yazdırma
        
        
    # Oluşan çerçeveyi ekrana yansıt
    cv2.imshow('Video', frame)

    # Çıkış için 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()


