import matplotlib.pyplot as plt

def show_images(first_image, last_image, images, labels, class_names):
    plt.figure(figsize=(10,10))
    for i in range(first_image, last_image):
        plt.subplot(5,5,i-first_image+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()