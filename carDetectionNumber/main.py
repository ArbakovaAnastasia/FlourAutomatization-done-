#pip install matplotlib pytesseract opencv-python
import matplotlib.pyplot as plt
import pytesseract as pt
import cv2
import configparser

def open_img(img_path):
    carplate_img = cv2.imread(img_path)
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(carplate_img)
    #plt.show()
    plt.savefig('carplate_img.png')  # Save the image
    plt.close()

    return carplate_img

def carplate_extract(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10,x+15:x+w-20]

    return carplate_img

def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    plt.axis('off')
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image

def main():
    config = configparser.ConfigParser()
    config.read('path_to_config_file.cfg')

    carplate_img_rgb_path = config.get('Paths', 'carplate_img_rgb')
    carplate_haar_cascade_path = config.get('Paths', 'carplate_haar_cascade')

    carplate_img_rgb = open_img(img_path=carplate_img_rgb_path)
    carplate_haar_cascade = cv2.CascadeClassifier(carplate_haar_cascade_path)
    #carplate_img_rgb = open_img(img_path = 'C:\\Users\\Nastya\\Desktop\\carDetectionNumber\\cars\\3.jpg')
    #carplate_haar_cascade = cv2.CascadeClassifier('C:\\Users\\Nastya\\Desktop\\carDetectionNumber\\haar_cascades\\haarcascade_russian_plate_number.xml')
    carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
    carplate_extract_img = enlarge_img(carplate_extract_img, 150)
    plt.imshow(carplate_extract_img)
    #plt.show()
    plt.savefig('carplate_extract_img.png')  # Save the extracted carplate image
    plt.close()

    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
    plt.axis('off')
    plt.imshow(carplate_extract_img_gray, cmap='gray')
    #plt.show()
    plt.savefig('carplate_extract_img_gray.png')  # Save the grayscale carplate image
    plt.close()

    pt.pytesseract.tesseract_cmd = 'C:\\Users\\Nastya\\AppData\\Local\\Tesseract.exe'
    print('Номер автомобиля:  ', pt.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    )

    # Generate HTML file with the results
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
        <!-- <div><img src="carplate_extract_img_gray.png" alt="Grayscale Carplate Image"></div> -->
        
    </body>
    </html>
    '''.format(carplate_number=pt.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )))


if __name__ == '__main__':
    main()