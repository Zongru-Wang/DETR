import os
import random
import numpy as np
from PIL import Image, ImageChops
import itertools

# 定义文件路径
base_dir = './SJJ_Mix'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')
output_dir = './SJJ_Mix_combined'

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取类别文件
with open(os.path.join(base_dir, 'classes.txt'), 'r') as file:
    classes = file.read().splitlines()

# 标签映射
label_mapping = {label: idx for idx, label in enumerate(classes)}

# 读取标签文件
def read_labels(label_path):
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return [list(map(float, label.strip().split())) for label in labels]

# 保存标签文件
def save_labels(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            label[0] = int(label[0])  # 确保类别是整数
            file.write(' '.join(map(str, label)) + '\n')

# 获取随机平移
def get_random_translation(max_translation):
    return random.uniform(-max_translation, max_translation)

# 平移标签
def translate_labels(labels, translation, width):
    for label in labels:
        label[1] = (label[1] + translation / width) % 1  # 确保标签坐标在 [0, 1] 范围内
    return labels

# 平移图像
def translate_image(image, translation):
    return ImageChops.offset(image, int(translation), 0)

# 随机组合不同类型的图像
def combine_images(image_paths, translations, target_shape):
    combined_image = np.zeros(target_shape + (3,), dtype=np.uint8)
    for img_path, translation in zip(image_paths, translations):
        img = Image.open(img_path).convert("RGB").resize(target_shape)
        img = translate_image(img, translation)
        img_array = np.array(img)
        combined_image = np.add(combined_image, img_array)  # 合并图像
    combined_image = np.clip(combined_image, 0, 255)  # 防止溢出
    return Image.fromarray(combined_image)

# 创建组合图像并保存
def combine_and_save_images(single_label_images, combined_image_dir, num_images_per_combination, target_shape):
    labels = list(single_label_images.keys())
    print(f"Creating combined images for labels: {labels}")
    all_combinations = []

    # 生成所有可能的标签组合（至少两个标签）
    for r in range(2, len(labels) + 1):
        combinations = list(itertools.combinations(labels, r))
        all_combinations.extend(combinations)

    for combination_labels in all_combinations:
        combination_name = '_'.join(combination_labels)
        save_dir = os.path.join(combined_image_dir, combination_name)
        os.makedirs(save_dir, exist_ok=True)
        label_save_dir = os.path.join(combined_image_dir, f"{combination_name}_labels")
        os.makedirs(label_save_dir, exist_ok=True)

        for i in range(num_images_per_combination):
            image_paths = []
            labels_list = []
            translations = []

            for label in combination_labels:
                if label not in single_label_images or len(single_label_images[label]) == 0:
                    print(f"No images found for label: {label}")
                    continue
                
                img_name = random.choice(single_label_images[label])
                img_path = os.path.join(image_dir, label, img_name)
                image_paths.append(img_path)

                # 平移图像
                translation = get_random_translation(0.3) * target_shape[0]
                translations.append(translation)
                label_file_name = img_name.replace('.jpg', '.txt')
                label_file_path = os.path.join(label_dir, label, label_file_name)
                
                if os.path.exists(label_file_path):
                    labels = read_labels(label_file_path)
                    labels_translated = translate_labels(labels, translation, target_shape[0])
                    labels_list.append((label_mapping[label], labels_translated))
                else:
                    print(f"Label file not found: {label_file_path}")
                    labels_list.append((label_mapping[label], []))  # 如果没有标签文件，添加空标签

            if len(image_paths) != len(combination_labels):
                print(f"Skipping combination {combination_name} due to missing images.")
                continue

            combined_img = combine_images(image_paths, translations, target_shape)

            # 生成组合文件名
            original_filenames = [os.path.basename(path).replace('.jpg', '') for path in image_paths]
            combined_filename = '_'.join(original_filenames) + f'_{i}.png'
            save_path = os.path.join(save_dir, combined_filename)
            combined_img.save(save_path)

            # 合并标签
            combined_labels = []
            for label_id, labels in labels_list:
                for label in labels:
                    combined_labels.append([label_id] + label[1:])

            label_filename = combined_filename.replace('.png', '.txt')
            label_save_path = os.path.join(label_save_dir, label_filename)
            save_labels(label_save_path, combined_labels)

            print(f"Combined image saved to: {save_path}")
            print(f"Combined labels saved to: {label_save_path}")

# 获取单标签图像
def get_single_label_images(image_dir):
    single_label_images = {}
    for label in classes:
        label_dir = os.path.join(image_dir, label)
        if os.path.exists(label_dir):
            single_label_images[label] = [img for img in os.listdir(label_dir) if img.endswith('.jpg')]
    return single_label_images

# 获取单标签图像
single_label_images = get_single_label_images(image_dir)

# 创建组合图像并保存
num_images_per_combination = 5  # 每种组合生成的图像数量
target_shape = (256, 256)
combine_and_save_images(single_label_images, output_dir, num_images_per_combination, target_shape)
