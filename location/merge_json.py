import json
import os

def load_json(file_path):
    """
    加载 JSON 文件并返回其内容作为字典。
    """
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在。")
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return {}

def merge_descriptions(desc1, desc2, separator=" | "):
    """
    合并两个描述字符串，使用指定的分隔符。
    避免重复合并相同的描述，并保持描述的顺序。
    """
    descriptions = []
    if desc1:
        descriptions.append(desc1.strip())
    if desc2:
        if desc2.strip() not in descriptions:
            descriptions.append(desc2.strip())
    return separator.join(descriptions)

def merge_json_files(json_file1, json_file2, output_json, field_name1, field_name2, merged_field_name=None):
    """
    合并两个 JSON 文件，对于相同的图像，将它们的描述合并。

    Args:
        json_file1 (str): 第一个 JSON 文件的路径。
        json_file2 (str): 第二个 JSON 文件的路径。
        output_json (str): 输出合并后 JSON 文件的路径。
        field_name1 (str): 第一个 JSON 文件中描述字段的名称（例如 "Beard_and_Age"）。
        field_name2 (str): 第二个 JSON 文件中描述字段的名称。
        merged_field_name (str, optional): 合并后字段的名称。如果未提供，默认使用 field_name1。
    """
    data1 = load_json(json_file1)
    data2 = load_json(json_file2)

    if merged_field_name is None:
        merged_field_name = field_name1  # 默认使用第一个字段名称

    merged_data = {}

    # 获取所有唯一的图像名称
    all_images = set(data1.keys()).union(set(data2.keys()))

    for image in all_images:
        desc1 = data1.get(image, {}).get(field_name1, "")
        desc2 = data2.get(image, {}).get(field_name2, "")

        if desc1 and desc2:
            merged_desc = merge_descriptions(desc1, desc2)
        elif desc1:
            merged_desc = desc1
        elif desc2:
            merged_desc = desc2
        else:
            merged_desc = ""

        if merged_desc:
            merged_data[image] = {
                merged_field_name: merged_desc
            }

    # 写入合并后的 JSON 文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"合并完成！合并后的 JSON 文件已保存到 {output_json}")

if __name__ == "__main__":
    # 定义第一个 JSON 文件的路径
    json_path1 = "/home/hyl/yujia/A_few_shot/data_text_private/metadata_processed.json"  # 替换为您的第一个 JSON 文件路径

    # 定义第二个 JSON 文件的路径
    json_path2 = "/home/hyl/yujia/A_few_shot/location/location.json"  # 替换为您的第二个 JSON 文件路径

    # 定义输出合并后 JSON 文件的路径
    output_json_path = "/home/hyl/yujia/A_few_shot/data_text_private/metadata_location.json"  # 替换为您希望保存的路径

    # 定义两个 JSON 文件中描述字段的名称
    # 假设两个 JSON 文件中描述字段都叫 "Beard_and_Age"
    field1 = "Beard_and_Age"
    field2 = "Beard_and_Age"

    # 如果您希望合并后的字段名称不同，可以设置 merged_field_name
    # 例如 merged_field_name = "Combined_Description"
    # 如果设置为 None，默认使用 field1 的名称
    merged_field = None  # 或者设置为 "Combined_Description"

    merge_json_files(
        json_file1=json_path1,
        json_file2=json_path2,
        output_json=output_json_path,
        field_name1=field1,
        field_name2=field2,
        merged_field_name=merged_field
    )
