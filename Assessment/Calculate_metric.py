import os
import subprocess
import pandas as pd
import argparse
import numpy as np
import cv2
import tempfile
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate glare information and SSIM, and PSNR values.")
    parser.add_argument("--reference_HDR", type=str, help="Path to the reference_HDR folder.")
    parser.add_argument("--Pred_HDR", type=str, help="Path to the Pred_HDR folder.")
    parser.add_argument("--resize", type=int, default=512, help="Target size of the images. Default: 512")
    parser.add_argument("--output_path", type=str, default='Test_result', help="Path to the output folder.")
    parser.add_argument("--model_name", type=str, default='Model', help="your model name")
    parser.add_argument("--threads", type=int, default=8, help="how many threads to use for processing. Default: 8")
    parser.add_argument("--flip", type=str, default="True", help="Flip the image or not.")
    parser.add_argument("--Mask", type=str, default="False", help="Masking the image or not.")
    return parser.parse_args()

def apply_circle_mask2(input_image):
    # Read the input image

    if input_image is None:
        raise ValueError("Input image not found")

    # Check if image has 3 channels (color image)
    if len(input_image.shape) != 3 or input_image.shape[2] != 3:
        raise ValueError("Input image should have 3 channels")

    # Image dimensions
    height, width, _ = input_image.shape

    # Calculate the mask size and center
    mask_size = min(height, width)
    center = (mask_size // 2, mask_size // 2)
    circle_radius = mask_size // 2

    # Create a black image and draw a white circle (mask)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, circle_radius, (255), thickness=-1)

    # Apply the mask: retain the circular area
    masked_image = cv2.bitwise_and(input_image, input_image, mask=mask)

    return masked_image

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

    # calculate the local mean
    kernel = cv2.getGaussianKernel(window_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, kernel)
    mu2 = cv2.filter2D(img2, -1, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # calculate the local variance and covariance
    sigma1_sq = cv2.filter2D(img1 * img1, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2

    # according to the formula, calculate the SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(ssim_map)

    return ssim

def calculate_evaluation(_ldr, _hdr, max_val):
    ssim_vl = ssim(_ldr, _hdr, max_val)
    psnr_vl = psnr(_ldr, _hdr, max_val)
    mse_vl = mse(_ldr, _hdr)

    return psnr_vl, ssim_vl, mse_vl

def evalglare_results(input_path):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as result_file:
        subprocess.run(['evalglare', '-d', input_path], stdout=result_file, universal_newlines=True)
        result_file.seek(0)
        lines = result_file.readlines()

    last_line = lines[-1]
    values = last_line.split()

    try:
        dgp = float(values[1])
        av_lum = float(values[2])
        E_v = float(values[3])
        ugr = float(values[7])
    except ValueError as e:
        print(f"Error processing evalglare results for file '{input_path}': {e}, dgp = {values[1]} \
              av_lum = {values[2]}, E_v = {values[3]}, ugr = {values[6]}")
        dgp = None
        av_lum = None
        E_v = None
        ugr = None

    os.unlink(result_file.name)
    return dgp, av_lum, E_v, ugr

def compute_ssim_psnr(image1, image2):
    psnr, ssim, mse = calculate_evaluation(image1, image2, 32000.0)
    return ssim, psnr

def update_hdr_file_with_view(input_path):
    # check if the file already has the view settings
    if has_view_settings(input_path):
        #print("File already has view settings.")
        return input_path

    # build the command to add the view settings to the HDR file
    output_path = input_path.replace(".hdr", "_updated.hdr")
    command = f'cd {os.path.dirname(input_path)} && getinfo -a "VIEW= -vta -vv 186 -vh 186" < {os.path.basename(input_path)} > {os.path.basename(output_path)}'

    # execute the command and wait for it to finish
    subprocess.run(command, shell=True)
    # replace the original file with the updated file
    os.remove(input_path)
    os.rename(output_path, input_path)

    # return the original path
    #_ = has_view_settings(input_path)
    return input_path

def has_view_settings(hdr_path):
    # run the getinfo command to retrieve the file header information
    process = subprocess.Popen(["getinfo", hdr_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hdr_info, _ = process.communicate()
    #print(hdr_info)
    # check if the "VIEW=" string is in the header information
    return b"VIEW=" in hdr_info
def process_input_file(file, args):
    pred_hdr_image_path = os.path.join(args.Pred_HDR, file)
    updated_pred_hdr_image_path = update_hdr_file_with_view(pred_hdr_image_path)
    dgp_pred_hdr, av_lum_pred_hdr, E_v_pred_hdr, ugr_pred_hdr = evalglare_results(updated_pred_hdr_image_path)
    reference_hdr_image_path = os.path.join(args.reference_HDR, file)
    #updated_ref_hdr_image_path = add_hdr_file_black_fisheye(reference_hdr_image_path)
    updated_ref_hdr_image_path = reference_hdr_image_path
    dgp_reference_hdr, av_lum_reference_hdr, E_v_reference_hdr, ugr_reference_hdr = evalglare_results(updated_ref_hdr_image_path)
    pred_hdr_image = io_and_resize(updated_pred_hdr_image_path, args.resize)
    #rgb2bgr
    if args.flip == "True":
        pred_hdr_image = np.flip(pred_hdr_image, -1)
    reference_hdr_image_1 = io_and_resize(updated_ref_hdr_image_path, args.resize)
    #mask
    reference_hdr_image = apply_circle_mask2(reference_hdr_image_1)
    pred_hdr_image_1 = apply_circle_mask2(pred_hdr_image)
    ssim_psnr_pred_hdr = compute_ssim_psnr(pred_hdr_image_1, reference_hdr_image)
    file_basename = os.path.splitext(file)[0]
    return {
        'Pred_HDR': (file_basename, dgp_pred_hdr, av_lum_pred_hdr, E_v_pred_hdr, ugr_pred_hdr),
        'reference_HDR': (file_basename, dgp_reference_hdr, av_lum_reference_hdr, E_v_reference_hdr, ugr_reference_hdr),
        'Pred_HDR_vs_reference_HDR': (file_basename, *ssim_psnr_pred_hdr)
    }
def io_and_resize(image_path, size):
    image_size = (int(size), int(size))
    hdr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #hdr_image = hdr_image[:, :, [2, 1, 0]]
    hdr_image = cv2.resize(hdr_image, image_size)
    return hdr_image

def check_directories_alignment(args):
    reference_hdr_files = set(os.listdir(args.reference_HDR))
    pred_hdr_files = set(os.listdir(args.Pred_HDR))
    if not reference_hdr_files == pred_hdr_files:
        missing_in_pred_hdr = sorted(reference_hdr_files.difference(pred_hdr_files))

        print("Error: The file names in the reference_HDR and Pred_HDR directories are not aligned.")
        if missing_in_pred_hdr:
            print(f"Missing in Pred_HDR: {', '.join(missing_in_pred_hdr)}")

        raise Exception("File names are not aligned in the directories.")

def main():

    args = parse_arguments()
    # Check if the file names in the directories
    check_directories_alignment(args)

    all_files = os.listdir(args.Pred_HDR)
    completed_files = set()
    num_files = len(all_files)

    results = {
        'Pred_HDR': [],
        'reference_HDR': [],
        'Pred_HDR_vs_reference_HDR': []
    }

    while len(completed_files) < num_files:
        remaining_files = [file for file in all_files if file not in completed_files]

        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_file = {executor.submit(process_input_file, file, args): file for file in remaining_files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results['Pred_HDR'].append(result['Pred_HDR'])
                    results['reference_HDR'].append(result['reference_HDR'])
                    results['Pred_HDR_vs_reference_HDR'].append(result['Pred_HDR_vs_reference_HDR'])

                    completed_files.add(file)
                    print(f"Tasks completed: {len(completed_files)}/{num_files}")

                except PermissionError:
                    print(f"Error: Permission denied for file '{file}'. Retrying in next iteration...")
                except Exception as e:
                    print(f"Error processing file '{file}': {e}. Retrying in next iteration...")

        executor.shutdown(wait=True)  # Ensure all threads are closed

    # Save results to Excel
    cwd = os.getcwd()
    pre_path = os.path.join(cwd, args.output_path)
    model_path = os.path.join(pre_path, args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_path = os.path.join(model_path, '{}.xlsx'.format(args.model_name))
    
    with pd.ExcelWriter(file_path) as writer:
        for key, data in results.items():
            if key in ['Pred_HDR_vs_reference_HDR']:
                df = pd.DataFrame(data, columns=['File Name', 'SSIM', 'PSNR'])
            else:
                df = pd.DataFrame(data, columns=['File Name', 'DGP', 'AV_LUM', 'E_V', 'UGR'])
            # Sort DataFrame by 'File Name'
            df = df.sort_values(by='File Name')
            df.to_excel(writer, sheet_name=f'{key}_evalglare', index=False)

if __name__ == "__main__":
    args = parse_arguments()
    main()