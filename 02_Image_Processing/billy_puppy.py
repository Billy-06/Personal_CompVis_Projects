import cv2

img = cv2.imread('Computer-Vision-with-Python_udemy/DATA/00-puppy.jpg')

while True:

    cv2.imshow('Puppy',img)
    
    # If we've waited 4 milliseconds adn we've pressed the Esc key
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()