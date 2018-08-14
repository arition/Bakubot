import bisect
import cv2
import numpy as np
import pillowfight
from PIL import Image

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.01
REPLACE_MARGIN = 20
DETECT_MARGIN = 40


def align_images(im1, im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, _ = im2.shape
    im1_reg = cv2.warpPerspective(
        im1, h, (width, height), borderValue=(255, 255, 255))

    return im1_reg, h


def replace_images(im1, im2, immask1, immask2):
    # im1 = jpn
    # im2 = chs

    mg = REPLACE_MARGIN
    dg = DETECT_MARGIN
    shape = np.shape(im1)
    mask = np.zeros((shape[0], shape[1]), np.int8)
    # select not white pixels
    indexarr_temp = (immask1 != 255).nonzero()
    indexarr1 = sorted(
        zip(indexarr_temp[0], indexarr_temp[1]), key=lambda t: (t[0], t[1]))
    indexarr1_x = [a[0] for a in indexarr1]
    indexarr_temp = (immask2 != 255).nonzero()
    indexarr2 = sorted(
        zip(indexarr_temp[0], indexarr_temp[1]), key=lambda t: (t[0], t[1]))
    indexarr2_x = [a[0] for a in indexarr2]
    for (x, y) in indexarr1:
        left = bisect.bisect_left(indexarr2_x, x-dg)
        right = bisect.bisect_right(indexarr2_x, x+dg)
        for (dx, dy) in indexarr2[left:right]:
            if y-dg < dy < y+dg:
                mask[0 if x-mg < 0 else x-mg: shape[0] if x+mg > shape[0] else x+mg,
                     0 if y-mg < 0 else y-mg: shape[1] if y+mg > shape[1] else y+mg] = 1
                break

    for (x, y) in indexarr2:
        left = bisect.bisect_left(indexarr1_x, x-dg)
        right = bisect.bisect_right(indexarr1_x, x+dg)
        for (dx, dy) in indexarr1[left:right]:
            if y-dg < dy < y+dg:
                mask[0 if x-mg < 0 else x-mg: shape[0] if x+mg > shape[0] else x+mg,
                     0 if y-mg < 0 else y-mg: shape[1] if y+mg > shape[1] else y+mg] = 1
                break

    mask_text = mask
    mask_bg = (mask == 0).astype(np.int8)
    img_text = cv2.bitwise_or(im2, im2, mask=mask_text)
    img_bg = cv2.bitwise_or(im1, im1, mask=mask_bg)
    result = cv2.bitwise_or(img_text, img_bg)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result


def bound(diff, diff2):
    blank_image = np.zeros((np.shape(diff)[0], np.shape(diff)[1], 3), np.uint8)
    blank_image[:, :, :] = 255
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_TC89_L1)
    cnts_1 = cnts[1]
    thresh = cv2.threshold(diff2, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_TC89_L1)
    cnts_2 = cnts[1]

    # loop over the contours
    for c in cnts_1:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(blank_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for c in cnts_2:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(blank_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return blank_image


def main():
    for i in range(1, 198):
        # Read reference image
        ref_filename = "jpn/p_"+str(i).zfill(3)+".jpg"
        print("Reading reference image : ", ref_filename)
        im_reference = cv2.imread(ref_filename, cv2.IMREAD_COLOR)

        # Read image to be aligned
        im_filename = "chs/"+str(i).zfill(3)+".jpg"
        print("Reading image to align : ", im_filename)
        im = cv2.imread(im_filename, cv2.IMREAD_COLOR)

        print("Aligning images ...")
        # Registered image will be resotred in im_aligned.
        # The estimated homography will be stored in h.
        im_aligned, _ = align_images(im, im_reference)

        # Write aligned image to disk.
        aligned_filename = "aligned.jpg"
        print("Saving aligned image : ", aligned_filename)
        cv2.imwrite(aligned_filename, im_aligned)

        swt_filename = "mask.jpg"
        print("Applying swt filter ...")
        imp_aligned = Image.open(aligned_filename)
        impswt = pillowfight.swt(
            imp_aligned, output_type=pillowfight.SWT_OUTPUT_BW_TEXT)
        imswt = cv2.cvtColor(np.array(impswt), cv2.COLOR_RGB2BGR)
        print("Saving swt filtered image : ", swt_filename)
        cv2.imwrite(swt_filename, imswt)

        swt_filename_ref = "mask_ref.jpg"
        print("Applying swt filter to ref ...")
        imp_ref = Image.open(ref_filename)
        impswt_ref = pillowfight.swt(
            imp_ref, output_type=pillowfight.SWT_OUTPUT_BW_TEXT)
        imswt_ref = cv2.cvtColor(np.array(impswt_ref), cv2.COLOR_RGB2BGR)
        print("Saving swt filtered image : ", swt_filename_ref)
        cv2.imwrite(swt_filename_ref, imswt_ref)

        result_filename = "result/"+str(i).zfill(3)+".jpg"
        print("Replacing texts ...")
        imswt_gray = cv2.cvtColor(imswt, cv2.COLOR_BGR2GRAY)
        imswt_ref_gray = cv2.cvtColor(imswt_ref, cv2.COLOR_BGR2GRAY)
        im_result = replace_images(
            im_reference, im_aligned, imswt_gray, imswt_ref_gray)
        print("Saving result image : ", result_filename)
        cv2.imwrite(result_filename, im_result)

        im_bound = bound(imswt_gray, imswt_ref_gray)
        cv2.imwrite("bound.jpg", im_bound)


if __name__ == '__main__':
    main()
