import requests

def translate_text(text, source_lang='zh', target_lang='en'):
    """
    使用 DeepLX API 翻译文本。

    参数:
    text (str): 要翻译的文本
    source_lang (str): 源语言代码（默认是中文 'zh'）
    target_lang (str): 目标语言代码（默认是英文 'en'）

    返回:
    str: 翻译后的文本
    """
    # 定义API的URL
    url = 'https://deeplx.mingming.dev/translate'
    
    # 定义请求体
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    
    # 设置请求头
    headers = {
        'Content-Type': 'application/json'
    }
    
    # 发送POST请求
    response = requests.post(url, json=payload, headers=headers)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析 JSON 响应并返回翻译后的文本
        result = response.json()
        return result.get('data', '翻译失败，没有返回数据')
    else:
        # 返回错误信息
        return f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}"

if __name__ == "__main__":
    # 示例使用
    result = translate_text("你好，世界！")
    print(result)
