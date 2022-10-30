import os
import cv2
import numpy as np

class BlemishRemoval():
    __window_name = "Click to remove remove blemishes"
    __input_images = []
    __image = None
    __r = 15

    def __init__(self):
        try:
            for file in os.listdir('../img'):
                self.__input_images.append(cv2.imread('../img/' + file, cv2.IMREAD_COLOR))
                print('[INFO] Loaded image(s)')
        except:
                print('[ERROR] An error occured while reading the image(s)')

        cv2.namedWindow(self.__window_name)
        cv2.setMouseCallback(self.__window_name, self.__mouseCB)

    def run(self):
        for image in self.__input_images:
            self.__image = image.copy()
            k = 0
            while k != 27:
                cv2.imshow(self.__window_name, self.__image)
                k = cv2.waitKey(20)

            cv2.imwrite("../output.png", self.__image)
        cv2.destroyAllWindows()

    def __mouseCB(self, action, x, y, flags, userdata):
        if action == cv2.EVENT_LBUTTONDOWN:
            newX, newY = self.__getClonePatch(x,y)
            patch = self.__image[newY:(newY+2*self.__r), newX:(newX+2*self.__r)]
            mask = 255 * np.ones(patch.shape, patch.dtype)
            self.__image = cv2.seamlessClone(patch, self.__image, mask, (x,y), cv2.NORMAL_CLONE)
            cv2.imshow(self.__window_name, self.__image)

        elif action == cv2.EVENT_LBUTTONUP:
            cv2.imshow(self.__window_name, self.__image)

    def __getClonePatch(self, x, y):
        fft_values = []
        directions = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]) * (2 * self.__r)
        for i in range(0,8):
            crop = cv2.cvtColor(self.__getROI((x,y) + directions[i]), cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(crop)
            fshift = np.fft.fftshift(f)
            magnitude = np.mean(20*np.log(np.abs(fshift)))
            fft_values.append(magnitude)

        min_index = np.argmin(np.asarray(fft_values))
        return (x, y) + directions[min_index]

    def __getROI(self, xy):
        x, y = xy[:]
        image_ROI = self.__image[y:(y+2*self.__r), x:(x+2*self.__r)]
        return image_ROI

if __name__ == "__main__":
    obj = BlemishRemoval()
    obj.run()