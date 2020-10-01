# 字典数据特征化
from sklearn.feature_extraction import DictVectorizer

# 文本数据特征化
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_dict():
    print("【字典特征抽取 One-Hot编码】")
    test_data = [
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}
    ]
    dv = DictVectorizer()
    value = dv.fit_transform(test_data)
    names = dv.get_feature_names()
    print(value)
    print(names)


def extract_text_count_alphabet():
    print("【文本特征抽取 字母频率统计】")
    test_data = ["Hello python! I love python."]
    cv = CountVectorizer()
    value = cv.fit_transform(test_data)
    names = cv.get_feature_names()
    print(value)
    print(names)


def extract_text_count_hans():
    print("【文本特征抽取 汉字频率统计】")
    test_data = ["中文对话特征抽取测试。", "中文对话", "特征抽取", "抽取测试"]
    # 分割字词
    print("Before: ", test_data)
    test_data = [' '.join(list(jieba.cut(item))) for item in test_data]
    print("After : ", test_data)
    cv = CountVectorizer()
    value = cv.fit_transform(test_data)
    names = cv.get_feature_names()
    print(value)
    print(names)


def extract_text_tf_idf():
    print("【文本特征抽取 逆文档频率】")
    test_data = ["I love china", "china is my hometown"]
    tv = TfidfVectorizer()
    value = tv.fit_transform(test_data)
    names = tv.get_feature_names()
    print(value)
    print(names)


if __name__ == '__main__':
    extract_dict()
    print("="*32)

    extract_text_count_alphabet()
    print("="*32)

    extract_text_count_hans()
    print("="*32)

    extract_text_tf_idf()
