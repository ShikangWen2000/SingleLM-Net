import numpy as np
#from skimage.metrics import peak_signal_noise_ratio
import cv2
#from skimage.metrics import structural_similarity
def mse(img1, img2):
    err = np.square(img1.astype(np.float64) - img2.astype(np.float64))
    return np.mean(err)

def psnr(img1, img2, data_rang):
    mse_value = mse(img1, img2)
    if mse_value == 0:
        return float("inf")
    return 20 * np.log10(data_rang) - 10 * np.log10(mse_value)

def ssim(img1, img2, data_rang, C1=6.5025, C2=58.5225, window_size=11, sigma=1.5):
    img1 = img1.astype(np.float64) / data_rang
    img2 = img2.astype(np.float64) / data_rang

    # 计算局部均值
    kernel = cv2.getGaussianKernel(window_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, kernel)
    mu2 = cv2.filter2D(img2, -1, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 计算局部方差和协方差
    sigma1_sq = cv2.filter2D(img1 * img1, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2

    # 根据SSIM公式计算结果
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(ssim_map)

    return ssim


def calculate_evaluation(_ldr, _hdr, max_val):
    ssim_vl = ssim(_ldr, _hdr, max_val)
    psnr_vl = psnr(_ldr, _hdr, max_val)
    mse_vl = mse(_ldr, _hdr)

    return psnr_vl, ssim_vl, mse_vl