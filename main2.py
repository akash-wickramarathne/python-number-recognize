import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model("handwritten.keras")


def predict_digit(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (28, 28))
    img_normalized = tf.keras.utils.normalize(img_resized, axis=1)
    img_normalized = img_normalized.reshape(1, 28, 28, 1)
    prediction = model.predict(img_normalized)
    print(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img_resized, cmap=plt.cm.binary)
    plt.show()


def run_existing_while_loop():
    image_number = 1
    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            image_number += 1
        except Exception as e:
            print("Error!", e)
        finally:
            image_number += 1


def draw_annotations():
    drawing = False
    ix, iy = -1, -1
    img = np.zeros((512, 512, 3), np.uint8)

    def draw_circle(event, x, y, flags, param):
        nonlocal ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            predict_digit(img)
        elif key == ord('c'):
            img = np.zeros((512, 512, 3), np.uint8)
        elif key == 27:  # Escape key
            break

    cv2.destroyAllWindows()


def main():
    choice = input("Enter 1 to run existing while loop, 2 to draw annotations: ")
    if choice == '1':
        run_existing_while_loop()
    elif choice == '2':
        draw_annotations()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
