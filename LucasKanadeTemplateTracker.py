import cv2
import numpy as np
import os


# Function to warp
def warping(img, w, rectangle):
    xMin, yMin, xMax, yMax = rectangle

    # warpedImg = np.zeros((yMax - yMin, xMax - xMin), dtype=np.float64)
    wy, wx = np.indices((yMax - yMin, xMax - xMin))
    wy = np.reshape(wy, (1, -1)) + yMin
    wx = np.reshape(wx, (1, -1)) + xMin
    w1 = np.ones_like(wy)
    positions = np.squeeze(np.stack((wx, wy, w1), axis=0))
    new_positions = np.dot(w, positions)
    new_x, new_y = new_positions
    new_y = np.reshape(new_y, (yMax - yMin, xMax - xMin)).astype(np.int32)
    new_x = np.reshape(new_x, (yMax - yMin, xMax - xMin)).astype(np.int32)
    blank = np.zeros_like(new_y) - 1
    valid_y = np.where(np.bitwise_and(0 <= new_y, new_y < img.shape[0]), new_y, blank)
    valid_y = np.where(np.bitwise_and(0 <= new_x, new_x < img.shape[1]), valid_y, blank)
    valid_x = np.where(np.bitwise_and(0 <= new_y, new_y < img.shape[0]), new_x, blank)
    valid_x = np.where(np.bitwise_and(0 <= new_x, new_x < img.shape[1]), valid_x, blank)
    warpedImg = img[valid_y, valid_x]
    # min_pos = np.array([[xMin], [yMin], [1]], dtype=np.int32)
    # max_pos = np.array([[xMax], [yMax], [1]], dtype=np.int32)
    # new_min_pos = np.dot(w, min_pos)
    # new_max_pos = np.dot(w, max_pos)
    # for yIndex, y in enumerate(range(yMin, yMax)):
    #     for xIndex, x in enumerate(range(xMin, xMax)):
    #         warpedPoint = np.matmul(H, [x, y, 1])
    #         warpedX = warpedPoint[0] / warpedPoint[2]
    #         warpedY = warpedPoint[1] / warpedPoint[2]
    #         warpedImg[yIndex, xIndex] = img[int(round(warpedY)), int(round(warpedX))]
    return warpedImg


# Core function
def affineLKtracker(rectangle, image, tmp, p):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Warp matrix of size 2 x 3
    w = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    warp = warping(img.astype(np.float64), w, rectangle).astype(np.uint8)

    imgx = cv2.Sobel(np.float64(img), cv2.CV_64F, 1, 0, ksize=5)
    imgy = cv2.Sobel(np.float64(img), cv2.CV_64F, 0, 1, ksize=5)

    Ix = warping(imgx, w, rectangle)
    Iy = warping(imgy, w, rectangle)
    # cv2.imshow("a", (Ix - np.min(Ix)) / (np.max(Ix) - np.min(Ix)))
    # cv2.waitKey(330)
    # cv2.imshow("a", (Iy - np.min(Iy)) / (np.max(Iy) - np.min(Iy)))
    # cv2.waitKey(330)

    # Error calculation
    errorImage = tmp.astype(np.float64) - warp.astype(np.float64)
    errorMean = np.mean(errorImage)
    errorTest = errorImage < errorMean
    errorImage = np.where(errorTest, errorImage * 0.5, np.sign(errorImage) * errorMean)
    # for row in range(errorImage.shape[0]):
    #     for col in range(errorImage.shape[1]):
    #         if abs(errorImage[row, col]) < errorMean:
    #             errorImage[row, col] = errorImage[row, col] * 0.5
    #         else:
    #             errorImage[row, col] = np.sign(errorImage[row, col]) * errorMean

    # width = int(errorImage.shape[1] * 2)
    # height = int(errorImage.shape[0] * 2)
    # dim = (width, height)
    # resized = cv2.resize(errorImage, dim, interpolation = cv2.INTER_AREA)
    # cv2.imshow('Error image for car', resized)
    # cv2.waitKey(0)

    # Finding Steep decent
    height, width = warp.shape
    steep_images = np.array([np.zeros((height, width)) for _ in range(6)])
    for i in range(height):
        for j in range(width):
            steep = np.matmul([Ix[i, j], Iy[i, j]], [[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
            for k in range(len(steep_images)):
                steep_images[k][i][j] = steep[k]

    # Hessian matrix calculation
    hessian = np.zeros((6, 6))
    steepDescent = np.zeros((6, 1))
    for i in range(height):
        for j in range(width):
            A = np.array([[st[i][j] for st in steep_images]])
            A_Trans = np.transpose(A)
            A_multi = np.dot(A_Trans, A)
            hessian = hessian + A_multi
            steepDescent = steepDescent + (A_Trans * errorImage[i][j])
    inv_hessian = np.linalg.inv(hessian)
    dp = np.dot(steepDescent.transpose(), inv_hessian)  # + errorImage * np.sum(steep_images, axis=0)
    p += dp[0]
    # for i in range(len(steep_images)):
    #     st = steep_images[i]
    #     cv2.imshow('steep%d' % i, (st - np.min(st)) / (np.max(st) - np.min(st)))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    return p


# Increasing Brightness
def adjust_gamma(image, gamma=1.5):  # Change the value of gamma to control brightness
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# Main function
def main():
    # Reading images
    input_file_headers = [("datasets/DragonBaby/DragonBaby/img/", "DragonBaby", [(156, 72), (214, 146)]),
                          ("datasets/Car4/Car4/img/", "Car", [(65, 45), (180, 140)]),
                          ("datasets/Bolt2/Bolt2/img/", "Bolt", [(263, 81), (309, 140)])]
    for i in range(len(input_file_headers)):
        file_header, base_name, corners = input_file_headers[i]
        source_tree = list(os.walk(file_header))
        root, subdir, filenames = source_tree[0]
        image_list = []
        for j in range(len(filenames)):
            source_filename = "%s/%s" % (root, filenames[j])
            source_image = cv2.imread(source_filename)
            image_list.append(source_image)
        first_image = np.copy(image_list[0])

        # Increasing brightness
        image_list = [adjust_gamma(image_list[j]) for j in range(len(image_list))]

        # Drawing bounding box
        cv2.rectangle(first_image, corners[0], corners[1], (0, 255, 0), 2)

        # Cropping to get template, and grayscale conversion
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        template = first_image[y1:y2, x1:x2]
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.GaussianBlur(template, (3, 3), 0)

        # Bounding box dimensions
        rectangle = [x1, y1, x2, y2]

        # Calling LK Tracker
        p = np.zeros((6,))
        for j in range(1, 10):
            print("iteration", j)
            p = affineLKtracker(rectangle, image_list[i], template, p)

        # cv2.imshow(base_name, template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
