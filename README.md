# Car-brand-classification

1. Problemin müəyyənləşdirilməsi (Problem Definition)
Layihənin məqsədi:
Bu layihənin əsas məqsədi müxtəlif avtomobil brendlərinə aid loqoları şəkil əsasında avtomatik olaraq tanıya biləcək bir təsnifat modeli (classification model) qurmaqdır. Modelə bir avtomobil loqosu şəkli verildikdə, o bu loqonun hansı brendə aid olduğunu müəyyən etməlidir.

Problem növü:
Bu problem təsnifat (classification) problemidir, çünki giriş olaraq bir şəkil verilir və çıxış olaraq model həmin şəkili 8 avtomobil brendindən birinə aid etməlidir.

2. Məlumatların toplanması və tanınması (Data Collection & Understanding)
  
Bu layihədə istifadə olunan məlumatlar Kaggle platformasından əldə edilib:

Datasetin linki:
https://www.kaggle.com/code/volkandl/car-brand-classification/input

Bu dataset avtomobil brendlərinin loqolarını ehtiva edən şəkillərdən ibarətdir.Fayl zip formatındadır. faylı yükləyirik və unzip edirik


# Dataset qovluqları
base_dir = "/content/Car_Brand_Logos"
train_dir = os.path.join(base_dir, "Train")
test_dir  = os.path.join(base_dir, "Test")
     
Brendlərin siyahısı və şəkil sayı:
     
Train dəstində şəkil sayı per brend: {'mercedes': 341, 'hyundai': 302, 'opel': 301, 'skoda': 314, 'volkswagen': 330, 'toyota': 306, 'mazda': 317, 'lexus': 301}
Test  dəstində şəkil sayı per brend: {'mercedes': 50, 'hyundai': 50, 'opel': 50, 'skoda': 50, 'volkswagen': 50, 'toyota': 50, 'mazda': 50, 'lexus': 50}
Ümumi Train şəkil sayı: 2512
Ümumi Test  şəkil sayı: 400
İlkin vizuallaşdırma:

3. Məlumatların təmizlənməsi və işlənməsi (Data Cleaning & Preprocessing)
1.Şəkillərin ölçülərinin və formatlarının standartlaşdırılması Modelə verilən şəkillərin ölçüləri müxtəlif ola bilər. Bu səbəbdən, onları eyni ölçüyə salmaq lazımdır.

Biz bütün şəkilləri 150x150 ölçüsünə gətiririk:
img_size = (150, 150)
     
Modeldə istifadə üçün şəkillər aşağıdakı şəkildə yüklənəcək:
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizasiya
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)
     
Found 2513 images belonging to 8 classes.
Found 400 images belonging to 8 classes.
2.Data Augmentation (Veri Artırma)
Modelin overfitting etməməsi üçün train şəkilləri üzərində bəzi təsadüfi dəyişikliklər edilir:

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
     
3.Normalizasiya və bölünmə Normalizasiya şəkillərdəki pikselləri 0–255 aralığından 0–1 aralığına salır (rescale=1./255).

Bölünmə: Dataset artıq Train və Test olaraq əvvəlcədən bölünüb. Əgər əlavə olaraq validation dəsti lazımdırsa, validation_split istifadə edilə bilər.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

     
Found 2013 images belonging to 8 classes.
Found 500 images belonging to 8 classes.

4. Xüsusiyyət mühəndisliyi və Transfer Learning (Feature Engineering / Transfer Learning)
Modelin öyrənmə prosesini sürətləndirmək və daha yaxşı nəticə əldə etmək (pre-trained) modellərdən istifadə edirik. Bu texnikaya Transfer Learning deyilir.

İstifadə olunan pre-trained model: VGG16

Tanınmış, stabil və asan tətbiq olunan modeldir.

ImageNet datası üzərində əvvəlcədən öyrədilmişdir.

Ənənəvi konvolyusiya qatları çox dərin olmayan, ancaq güclü nəticə verən modeldir.

5. Modellərin qurulması və öyrədilməsi (Model Training)

6. Modellərin Qiymətləndirilməsi (Model Evaluation)
Modelin test və validation nəticələrinə əsasən nə qədər düzgün işlədiyini qiymətləndirmək. Bunun üçün aşağıdakı metriklərdən istifadə edəcəyik:

Accuracy (Dəqiqlik)

Precision (Dəqiqlik göstəricisi)

Recall (Xatırlama)

F1-score

Confusion Matrix (Çaşqınlıq matrisası)


# Confusion Matrix
![image](https://github.com/user-attachments/assets/fddefcf3-0ff5-46bf-b90e-044723d5a813)

     
Classification Report:
              precision    recall  f1-score   support

     hyundai       0.70      0.62      0.66        50
       lexus       0.77      0.48      0.59        50
       mazda       0.79      0.74      0.76        50
    mercedes       0.66      0.62      0.64        50
        opel       0.69      0.50      0.58        50
       skoda       0.58      0.60      0.59        50
      toyota       0.47      0.58      0.52        50
  volkswagen       0.42      0.68      0.52        50

    accuracy                           0.60       400
   macro avg       0.64      0.60      0.61       400
weighted avg       0.64      0.60      0.61       400

Overfitting yoxlaması Train və validation nəticələrini müqayisə etmək üçün:


# Sadə accuracy qrafiki
![image](https://github.com/user-attachments/assets/c1ea6dee-1dd6-4c9b-90ea-ba03a295c241)

# Confusion Matrix
![image](https://github.com/user-attachments/assets/c4864134-0194-42d0-88b2-55a2f354e18a)

# Əlavə statistikalar

              precision    recall  f1-score   support

     hyundai       0.55      0.58      0.56        50
       lexus       0.68      0.52      0.59        50
       mazda       0.80      0.74      0.77        50
    mercedes       0.58      0.68      0.62        50
        opel       0.58      0.66      0.62        50
       skoda       0.70      0.64      0.67        50
      toyota       0.59      0.52      0.55        50
  volkswagen       0.56      0.64      0.60        50

    accuracy                           0.62       400
   macro avg       0.63      0.62      0.62       400
weighted avg       0.63      0.62      0.62       400
