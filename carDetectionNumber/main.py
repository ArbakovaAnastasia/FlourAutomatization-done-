#pip install matplotlib pytesseract opencv-python
import matplotlib.pyplot as plt
import pytesseract as pt
import cv2
import configparser

#КОРОТКО:
# благодаря предобученной модели, находим на изображении автомобильный номер,
# затем воспользовавшись классификатором нужно определить Есть ли на изображении автомобильный номер и извлечь его
# после преобразовать из RGB в оттенки серого, уменьшив количество цветов на ихображении, который могу мешать обнаружению
# далее передается изображение в tesserract, для извлечения текста

#функция получения изображения
def open_img(img_path): #передаем путь до изображения
    carplate_img = cv2.imread(img_path) #читаем изображение
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB) #тк bgr, преобразуем в rgb
    plt.axis('off') #убрать оси
    plt.imshow(carplate_img)
    plt.savefig('carplate_img.png')  #сохранить изображение
    plt.close()

    return carplate_img   #возвращаем изображение

#функция обнаружения и извлечения координат номеров на изображении
def carplate_extract(image, carplate_haar_cascade): #передаем изображение и предобученную модель
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10,x+15:x+w-20]

    return carplate_img   #возвращает изображение извлеченного автомобильного номера

#функция увеличивает изображение извленного автомобильного номера
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    plt.axis('off')
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image #возвращает увеличенное изображение извленного автомобильного номера

def main():
    config = configparser.ConfigParser()                           #пути из конфигфайла
    config.read('path_to_config_file.cfg')
    carplate_img_rgb_path = config.get('Paths', 'carplate_img_rgb')
    carplate_haar_cascade_path = config.get('Paths', 'carplate_haar_cascade')

    carplate_img_rgb = open_img(img_path=carplate_img_rgb_path)    #получаем изображение
    carplate_haar_cascade = cv2.CascadeClassifier(carplate_haar_cascade_path)  #получаем классификатор haar_cascade, предобученная модель описывает характеристики российских автомобильных номеров
    carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade) #функция обнаружения и извлечения координат номеров на изображении
    carplate_extract_img = enlarge_img(carplate_extract_img, 150) #функция увеличивает изображение извленного автомобильного номера
    plt.imshow(carplate_extract_img)
    plt.savefig('carplate_extract_img.png')  #сохраняем изображение извлеченного автомобильного номера
    plt.close()
    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY) #из rgb в оттенки серого
    plt.axis('off')
    plt.imshow(carplate_extract_img_gray, cmap='gray')
    plt.savefig('carplate_extract_img_gray.png')  #сохраняем изображение извлеченного автомобильного номера уже в серых оттенках
    plt.close()

    #извлекаем текст из изображения с помощью tesseract
    pt.pytesseract.tesseract_cmd = 'C:\\Users\\Nastya\\AppData\\Local\\Tesseract.exe'
    print('Номер автомобиля:  ', pt.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    )

    #генерируем HTML с результатом распознавания
    with open('carplate_detection_result.html', 'w') as file:
        file.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Car Detection Result</title>
    </head>
    <body>
        <h1>Car Detection Result</h1>
        <div><p>Number of car: {carplate_number}</p></div>
        Picture:
        <div><img src="carplate_img.png" alt="Carplate Image">
        <img src="carplate_extract_img.png" alt="Extracted Carplate Image"></div>      
    </body>
    </html>
    '''.format(carplate_number=pt.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )))

if __name__ == '__main__':
    main()